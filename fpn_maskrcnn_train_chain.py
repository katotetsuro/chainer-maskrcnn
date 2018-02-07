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

measure_time = False

class FPNMaskRCNNTrainChain(FasterRCNNTrainChain):
    def __init__(self,
                 faster_rcnn,
                 rpn_sigma=3.,
                 roi_sigma=1.,
                 anchor_target_creator=AnchorTargetCreator(),
                 proposal_target_creator=ProposalTargetCreator()):
        super(FPNMaskRCNNTrainChain, self).__init__(
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

        features = self.faster_rcnn.extractor(imgs)
        # Since batch size is one, convert variables to singular form
        bbox = bboxes[0]
        label = labels[0]
        mask = masks[0]
        
        # iterate over feature pyramids
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.faster_rcnn.rpn(
            features, img_size, scale)

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
            mask_size=14)

        sample_roi_index = self.xp.zeros(
            (len(sample_roi), ), dtype=np.int32)


        # RPN losses
        gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(
            bbox, anchor, img_size)
        rpn_loc_loss = _fast_rcnn_loc_loss(rpn_loc, gt_rpn_loc,
                                       gt_rpn_label, self.rpn_sigma)
        rpn_cls_loss = F.softmax_cross_entropy(rpn_score, gt_rpn_label)


        loss = rpn_loc_loss + rpn_cls_loss

        chainer.reporter.report({
            'rpn_loc_loss': rpn_loc_loss,
            'rpn_cls_loss': rpn_cls_loss,
            'loss': loss
        }, self)

        return loss
