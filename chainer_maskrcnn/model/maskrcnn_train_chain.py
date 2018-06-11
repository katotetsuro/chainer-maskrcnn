import chainer
import chainer.functions as F
from chainercv.links.model.faster_rcnn.utils.anchor_target_creator import\
    AnchorTargetCreator
from chainercv.links.model.faster_rcnn.faster_rcnn_train_chain import FasterRCNNTrainChain, _smooth_l1_loss, _fast_rcnn_loc_loss
import cv2
import numpy as np
from .maskrcnn import MaskRCNN
from chainer_maskrcnn.utils.proposal_target_creator import ProposalTargetCreator
from .extractor.feature_pyramid_network import FeaturePyramidNetwork
from .extractor.c4_backbone import C4Backbone


class MaskRCNNTrainChain(FasterRCNNTrainChain):
    def __init__(self,
                 faster_rcnn,
                 rpn_sigma=3.,
                 roi_sigma=1.,
                 anchor_target_creator=AnchorTargetCreator(),
                 proposal_target_creator=ProposalTargetCreator()):
        super(MaskRCNNTrainChain, self).__init__(
            faster_rcnn, proposal_target_creator=proposal_target_creator)

    def __call__(self, imgs, bboxes, labels, masks, scale):
        def strip(x): return x.data if isinstance(x, chainer.Variable) else x
        bboxes = strip(bboxes)
        labels = strip(labels)
        masks = strip(masks)

        scale = np.asscalar(chainer.cuda.to_cpu(scale))
        n = bboxes.shape[0]
        if n != 1:
            raise ValueError(
                'Currently only batch size 1 is supported. n={}'.format(n))

        _, _, H, W = imgs.shape
        img_size = (H, W)

        features = self.faster_rcnn.extractor(imgs)

        # Since batch size is one, convert variables to singular form
        bbox = bboxes[0]
        label = labels[0]
        mask = masks[0]

        # iterate over feature pyramids
        proposals = []
        rpn_outputs = []
        gt_data = []
        for feature in features:
            rpn_locs, rpn_scores, rois, roi_indices, anchor = self.faster_rcnn.rpn(
                feature, img_size, scale)

            # Since batch size is one, convert variables to singular form
            rpn_score = rpn_scores[0]
            rpn_loc = rpn_locs[0]
            roi = rois

            # Sample RoIs and forward
            # gt_roi_labelになった時点で [0, NUM_FOREGROUND_CLASS-1]が[1, NUM_FOREGROUND_CLASS]にシフトしている
            sample_roi, gt_roi_loc, gt_roi_label, gt_roi_mask = self.proposal_target_creator(
                roi,
                bbox,
                label,
                mask,
                self.loc_normalize_mean,
                self.loc_normalize_std,
                mask_size=self.faster_rcnn.head.mask_size)

            sample_roi_index = self.xp.zeros(
                (len(sample_roi), ), dtype=np.int32)

            proposals.append((sample_roi, sample_roi_index,
                              1 / self.faster_rcnn.feat_stride * s))
            rpn_outputs.append((rpn_loc, rpn_score, roi, anchor))
            gt_data.append((gt_roi_loc, gt_roi_label, gt_roi_mask))

        if len(features) == 1:
            sample_roi, sample_roi_index, s = proposals[0]
            roi_cls_loc, roi_score, roi_cls_mask = self.faster_rcnn.head(
                features[0], sample_roi, sample_roi_index, s)

        else:
            roi_cls_loc, roi_score, roi_cls_mask = self.faster_rcnn.head(
                features, proposals)

        # RPN losses
        rpn_loc_loss = chainer.Variable(
            self.xp.array(0, dtype=self.xp.float32))
        rpn_cls_loss = chainer.Variable(
            self.xp.array(0, dtype=self.xp.float32))
        for (p, r) in zip(proposals, rpn_outputs):
            rpn_loc, rpn_score, _, anchor = r
            gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(
                bbox, anchor, img_size)
            rpn_loc_loss += _fast_rcnn_loc_loss(rpn_loc, gt_rpn_loc,
                                                gt_rpn_label, self.rpn_sigma)
            rpn_cls_loss += F.softmax_cross_entropy(rpn_score, gt_rpn_label)

        # Losses for outputs of the head.
        n_sample = roi_cls_loc.shape[0]
        roi_cls_loc = roi_cls_loc.reshape(n_sample, -1, 4)
        # light-headのときはそのまま使う
        if roi_cls_loc.shape[1] == 1:
            roi_loc = roi_cls_loc.reshape(n_sample, 4)
        else:
            roi_loc = roi_cls_loc[self.xp.arange(n_sample), gt_roi_label]

        gt_roi_loc = self.xp.concatenate([g[0] for g in gt_data], axis=0)
        gt_roi_label = self.xp.concatenate([g[1] for g in gt_data], axis=0)
        gt_roi_mask = self.xp.concatenate([g[2] for g in gt_data], axis=0)
        roi_loc_loss = _fast_rcnn_loc_loss(roi_loc, gt_roi_loc,
                                           gt_roi_label, self.roi_sigma)
        roi_cls_loss = F.softmax_cross_entropy(roi_score, gt_roi_label)

        # mask
        # https://engineer.dena.jp/2017/12/chainercvmask-r-cnn.html
        roi_mask = roi_cls_mask[self.xp.arange(n_sample), gt_roi_label]
        mask_loss = F.sigmoid_cross_entropy(
            roi_mask[0:gt_roi_mask.shape[0]], gt_roi_mask)
        loss = rpn_loc_loss + rpn_cls_loss + roi_loc_loss + roi_cls_loss + mask_loss

        chainer.reporter.report({
            'rpn_loc_loss': rpn_loc_loss,
            'rpn_cls_loss': rpn_cls_loss,
            'roi_loc_loss': roi_loc_loss,
            'roi_cls_loss': roi_cls_loss,
            'mask_loss': mask_loss,
            'loss': loss
        }, self)

        return loss
