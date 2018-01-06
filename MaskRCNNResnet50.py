import collections
import numpy as np
import chainer
from chainer import cuda
import chainer.functions as F
from chainercv.links.model.faster_rcnn.faster_rcnn import FasterRCNN
from chainercv.links.model.faster_rcnn.region_proposal_network import \
    RegionProposalNetwork
from chainercv.links.model.faster_rcnn.utils.loc2bbox import loc2bbox
from chainercv.transforms.image.resize import resize
from chainercv.utils import non_maximum_suppression
from C4Backbone import C4Backbone
from ResnetRoIMaskHead import ResnetRoIMaskHead
import cv2


class MaskRCNNResnet50(FasterRCNN):
    feat_stride = 16

    def __init__(self,
                 n_fg_class,
                 pretrained_model=None,
                 min_size=600,
                 max_size=1000,
                 ratios=[0.5, 1, 2],
                 anchor_scales=[8, 16, 32],
                 rpn_initialW=None,
                 loc_initialW=None,
                 score_initialW=None,
                 proposal_creator_params={}):
        if n_fg_class is None:
            raise ValueError(
                'The n_fg_class needs to be supplied as an argument')

        if loc_initialW is None:
            loc_initialW = chainer.initializers.Normal(0.001)
        if score_initialW is None:
            score_initialW = chainer.initializers.Normal(0.01)
        if rpn_initialW is None:
            rpn_initialW = chainer.initializers.Normal(0.01)

        extractor = C4Backbone('auto')

        rpn = RegionProposalNetwork(
            1024,
            1024,
            ratios=ratios,
            anchor_scales=anchor_scales,
            feat_stride=self.feat_stride,
            initialW=rpn_initialW,
            proposal_creator_params=proposal_creator_params,
        )

        head = ResnetRoIMaskHead(
            n_fg_class + 1,
            roi_size=7,
            spatial_scale=1. / self.feat_stride,
            loc_initialW=loc_initialW,
            score_initialW=score_initialW,
            mask_initialW=chainer.initializers.Normal(0.01))

        super().__init__(
            extractor,
            rpn,
            head,
            mean=np.array([122.7717, 115.9465, 102.9801],
                          dtype=np.float32)[:, None, None],
            min_size=min_size,
            max_size=max_size)

    def __call__(self, x, scale=1.):
        img_size = x.shape[2:]

        h = self.extractor(x)
        rpn_locs, rpn_scores, rois, roi_indices, anchor =\
            self.rpn(h, img_size, scale)
        roi_cls_locs, roi_scores, mask = self.head(h, rois, roi_indices)
        return roi_cls_locs, roi_scores, rois, roi_indices, mask

    def predict(self, imgs):
        prepared_imgs = list()
        sizes = list()
        for img in imgs:
            size = img.shape[1:]
            img = self.prepare(img.astype(np.float32))
            prepared_imgs.append(img)
            sizes.append(size)

        bboxes = list()
        labels = list()
        scores = list()
        masks = list()
        for img, size in zip(prepared_imgs, sizes):
            with chainer.using_config('train', False), \
                    chainer.function.no_backprop_mode():
                img_var = chainer.Variable(self.xp.asarray(img[None]))
                scale = img_var.shape[3] / size[1]
                roi_cls_locs, roi_scores, rois, roi_indices, mask = self.__call__(
                    img_var, scale=scale)
            # We are assuming that batch size is 1.
            roi_cls_loc = roi_cls_locs.data
            roi_score = roi_scores.data
            roi = rois / scale

            # Convert predictions to bounding boxes in image coordinates.
            # Bounding boxes are scaled to the scale of the input images.
            mean = self.xp.tile(
                self.xp.asarray(self.loc_normalize_mean), self.n_class)
            std = self.xp.tile(
                self.xp.asarray(self.loc_normalize_std), self.n_class)
            roi_cls_loc = (roi_cls_loc * std + mean).astype(np.float32)
            roi_cls_loc = roi_cls_loc.reshape((-1, self.n_class, 4))
            roi = self.xp.broadcast_to(roi[:, None], roi_cls_loc.shape)
            cls_bbox = loc2bbox(
                roi.reshape((-1, 4)), roi_cls_loc.reshape((-1, 4)))
            cls_bbox = cls_bbox.reshape((-1, self.n_class * 4))
            # clip bounding box
            cls_bbox[:, 0::2] = self.xp.clip(cls_bbox[:, 0::2], 0, size[0])
            cls_bbox[:, 1::2] = self.xp.clip(cls_bbox[:, 1::2], 0, size[1])

            prob = F.softmax(roi_score).data
            mask = F.sigmoid(mask).data

            raw_cls_bbox = cuda.to_cpu(cls_bbox)
            raw_prob = cuda.to_cpu(prob)
            mask = cuda.to_cpu(mask)
            raw_roi = cuda.to_cpu(roi)

            # roiとcls_bboxを見比べて、maskをroiのサイズにリサイズ -> cls_bboxの部分だけ使うという処理に変更する
            bbox, label, score, mask, roi = self._suppress(
                raw_cls_bbox, raw_prob, mask, raw_roi)

            # maskは修正前のboxで予測しているので、その大きさにresizeしたあと、修正後のbboxに貼り付ける
            mask_per_image = list()
            for i, (b, m, r) in enumerate(zip(bbox, mask, roi)):
                # 修正前のbox
                w = r[3] - r[1]
                h = r[2] - r[0]
                m = cv2.resize(m, (w, h)) * 255
                m = m.astype(np.uint8)
                _, m = cv2.threshold(m, 100, 255, cv2.THRESH_BINARY)

                # ここめっちゃダサいんだけど、場合分けする
                l = b - r
                l = l.astype(np.int32)
                if l[0] < 0:
                    pad = np.zeros((-l[0], m.shape[1]), dtype=np.uint8)
                    m = np.concatenate([pad, m], axis=0)
                else:
                    m = m[l[0]:, :]

                if l[1] < 0:
                    pad = np.zeros((m.shape[0], -l[1]), dtype=np.uint8)
                    m = np.concatenate([pad, m], axis=1)
                else:
                    m = m[:, l[1]:]

                if l[2] < 0:
                    m = m[:l[2], :]
                else:
                    pad = np.zeros((l[2], m.shape[1]), dtype=np.uint8)
                    m = np.concatenate([m, pad], axis=0)

                if l[3] < 0:
                    m = m[:, :l[3]]
                else:
                    pad = np.zeros((m.shape[0], l[3]), dtype=np.uint8)
                    m = np.concatenate([m, pad], axis=1)

                mask_per_image.append(m)
            bboxes.append(bbox)
            labels.append(label)
            scores.append(score)
            masks.append(mask_per_image)

        return bboxes, labels, scores, masks

    def prepare(self, img):
        _, H, W = img.shape

        scale = 1.

        scale = self.min_size / min(H, W)

        if scale * max(H, W) > self.max_size:
            scale = self.max_size / max(H, W)

        img = resize(img, (int(H * scale), int(W * scale)))

        # 元のコードは平均を引くだけ、だったんだけど、なんか[0,1]にするだけでうまくいかないかなぁ
        #        img = (img - self.mean).astype(np.float32, copy=False)
        img = img.astype(np.float32) / 255

        return img

    def _suppress(self, raw_cls_bbox, raw_prob, raw_mask, raw_roi):
        bbox = list()
        label = list()
        score = list()
        roi_mask = list()
        roi = list()
        # skip cls_id = 0 because it is the background class
        # -> maskは0から始まるから、l-1を使う
        # -> あーしまったTrainChainで最後のクラスToothBlushは範囲外になっておるわ・・
        for l in range(1, self.n_class - 1):
            cls_bbox_l = raw_cls_bbox.reshape((-1, self.n_class, 4))[:, l, :]
            prob_l = raw_prob[:, l]
            mask = prob_l > self.score_thresh
            cls_bbox_l = cls_bbox_l[mask]
            prob_l = prob_l[mask]
            keep = non_maximum_suppression(cls_bbox_l, self.nms_thresh, prob_l)
            bbox.append(cls_bbox_l[keep])
            # The labels are in [0, self.n_class - 2].
            label.append((l - 1) * np.ones((len(keep), )))
            score.append(prob_l[keep])
            mask_l = raw_mask[mask, l]
            roi_mask.append(mask_l[keep])
            raw_roi_l = raw_roi[:, l, :][mask]
            roi.append(raw_roi_l[keep])

        bbox = np.concatenate(bbox, axis=0).astype(np.float32)
        label = np.concatenate(label, axis=0).astype(np.int32)
        score = np.concatenate(score, axis=0).astype(np.float32)
        roi_mask = np.concatenate(roi_mask, axis=0)
        roi = np.concatenate(roi, axis=0)
        return bbox, label, score, roi_mask, roi
