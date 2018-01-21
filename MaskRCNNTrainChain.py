import chainer
import chainer.functions as F
from chainer import cuda
from chainer.cuda import to_cpu
from chainercv.links.model.faster_rcnn.utils.anchor_target_creator import\
    AnchorTargetCreator
from chainercv.links.model.faster_rcnn.faster_rcnn_train_chain import FasterRCNNTrainChain, _smooth_l1_loss, _fast_rcnn_loc_loss
import cv2
import numpy as np
from MaskRCNNResnet50 import MaskRCNNResnet50
from ProposalTargetCreator import ProposalTargetCreator
from feature_pyramid_network import FeaturePyramidNetwork
from C4Backbone import C4Backbone
import time

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
        if isinstance(bboxes, chainer.Variable):
            bboxes = bboxes.data
        if isinstance(labels, chainer.Variable):
            labels = labels.data
        if isinstance(scale, chainer.Variable):
            scale = scale.data
        if isinstance(masks, chainer.Variable):
            masks = masks.data

        scale = np.asscalar(cuda.to_cpu(scale))
        n = bboxes.shape[0]
        if n != 1:
            raise ValueError(
                'Currently only batch size 1 is supported. n={}'.format(n))

        _, _, H, W = imgs.shape
        img_size = (H, W)

        start_iter = time.time()
        features = self.faster_rcnn.extractor(imgs)
        # hacky scale coefficient relative to c4 backbone's output size
        if isinstance(self.faster_rcnn.extractor, FeaturePyramidNetwork):
            scale_coef = [0.5, 1, 2, 4]
            mask_size = 28
        elif isinstance(self.faster_rcnn.extractor, C4Backbone):
            scale_coef = [1]
            mask_size = 14
        else:
            raise ValueError('unknown backbone:', self.faster_rcnn)
        # iterate over feature pyramids
        # 
        proposals = list()
        for s, feature in zip(scale_coef, features):
            rpn_locs, rpn_scores, rois, roi_indices, anchor = self.faster_rcnn.rpn(
                feature, img_size, scale)

            # Since batch size is one, convert variables to singular form
            bbox = bboxes[0]
            label = labels[0]
            rpn_score = rpn_scores[0]
            rpn_loc = rpn_locs[0]
            mask = masks[0]
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
                mask_size=mask_size)

            sample_roi_index = self.xp.zeros(
                (len(sample_roi), ), dtype=np.int32)

            proposals.append((sample_roi, sample_roi_index, 1 / self.faster_rcnn.feat_stride * s))
            
        start_head = time.time()
        if len(features) == 1:
            sample_roi, sample_roi_index, s = proposals[0]
            roi_cls_loc, roi_score, roi_cls_mask = self.faster_rcnn.head(
                features[0], sample_roi, sample_roi_index, s)

        else:
            roi_cls_loc, roi_score, roi_cls_mask = self.faster_rcnn.head(
                features, proposals)
        end_head = time.time()
        print ("elapsed_time per head:{0}".format(end_head-start_head) + "[sec]")

        print('sample_roi', sample_roi.shape)
        # RPN losses
        gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(
            bbox, anchor, img_size)
        rpn_loc_loss = _fast_rcnn_loc_loss(rpn_loc, gt_rpn_loc,
                                           gt_rpn_label, self.rpn_sigma)
        rpn_cls_loss = F.softmax_cross_entropy(rpn_score, gt_rpn_label)

        # Losses for outputs of the head.
        n_sample = roi_cls_loc.shape[0]
        roi_cls_loc = roi_cls_loc.reshape(n_sample, -1, 4)
        # light-headのときはそのまま使う
        if roi_cls_loc.shape[1] == 1:
            roi_loc = roi_cls_loc.reshape(n_sample, 4)
        else:
            roi_loc = roi_cls_loc[self.xp.arange(n_sample), gt_roi_label]

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

        end_iter = time.time()
        print ("elapsed_time per iter:{0}".format(end_iter - start_iter) + "[sec]")

        start_bw = time.time()
        loss.backward()
        end_bw = time.time()
        print('backward time:{}'.format(end_bw - start_bw))

        n = 0
        for l in self.faster_rcnn.links():
            for p in l.params():
                if hasattr(p, 'size'):
                    n += p.size

        print('num params', n)


        return loss
