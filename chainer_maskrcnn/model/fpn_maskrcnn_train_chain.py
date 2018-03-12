import chainer
import chainer.functions as F
from chainercv.links.model.faster_rcnn.utils.anchor_target_creator import\
    AnchorTargetCreator
from chainercv.links.model.faster_rcnn.faster_rcnn_train_chain import FasterRCNNTrainChain, _smooth_l1_loss, _fast_rcnn_loc_loss
import cv2
import numpy as np
from .maskrcnn_resnet50 import MaskRCNNResnet50
from .extractor.c4_backbone import C4Backbone
from .extractor.feature_pyramid_network import FeaturePyramidNetwork
from chainer_maskrcnn.utils.proposal_target_creator import ProposalTargetCreator


class FPNMaskRCNNTrainChain(FasterRCNNTrainChain):
    def __init__(self,
                 faster_rcnn,
                 mask_loss_fun,
                 binary_mask=True,
                 rpn_sigma=3.,
                 roi_sigma=1.,
                 anchor_target_creator=AnchorTargetCreator()):
        # todo: clean up class dependencies
        proposal_target_creator = ProposalTargetCreator(
            faster_rcnn.extractor.anchor_sizes)
        super().__init__(
            faster_rcnn, proposal_target_creator=proposal_target_creator)
        self.mask_loss_fun = mask_loss_fun
        self.binary_mask = binary_mask

    def __call__(self, imgs, bboxes, labels, masks, scale):
        def strip(x): return x.data if isinstance(x, chainer.Variable) else x
        bboxes = strip(bboxes)
        labels = strip(labels)
        scale = strip(scale)
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

        rpn_locs, rpn_scores, rois, roi_indices, anchor, levels = self.faster_rcnn.rpn(
            features, img_size, scale)

        # Since batch size is one, convert variables to singular form
        rpn_score = rpn_scores[0]
        rpn_loc = rpn_locs[0]
        roi = rois

        # Sample RoIs and forward
        # gt_roi_labelになった時点で [0, NUM_FOREGROUND_CLASS-1]が[1, NUM_FOREGROUND_CLASS]にシフトしている
        sample_roi, sample_levels, gt_roi_loc, gt_roi_label, gt_roi_mask = self.proposal_target_creator(
            roi,
            bbox,
            label,
            mask,
            levels,
            self.loc_normalize_mean,
            self.loc_normalize_std,
            mask_size=self.faster_rcnn.head.mask_size,
            binary_mask=self.binary_mask)

        sample_roi_index = self.xp.zeros(
            (len(sample_roi), ), dtype=np.int32)

        # join roi and index of batch
        indices_and_rois = self.xp.concatenate(
            (sample_roi_index[:, None], sample_roi), axis=1).astype(self.xp.float32)

        # RPN losses
        gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(
            bbox, anchor, img_size)
        rpn_loc_loss = _fast_rcnn_loc_loss(rpn_loc, gt_rpn_loc,
                                           gt_rpn_label, self.rpn_sigma)
        rpn_cls_loss = F.softmax_cross_entropy(rpn_score, gt_rpn_label)

        # Losses for outputs of the head.
        roi_cls_loc, roi_score, roi_cls_mask = self.faster_rcnn.head(
            features, indices_and_rois, sample_levels, self.faster_rcnn.extractor.spatial_scales)

        # Losses for outputs of the head.
        n_sample = roi_cls_loc.shape[0]
        roi_cls_loc = roi_cls_loc.reshape(n_sample, -1, 4)
        # light-headのときはそのまま使う
        if roi_cls_loc.shape[1] == 1:
            roi_loc = roi_cls_loc.reshape(n_sample, 4)
        else:
            roi_loc = roi_cls_loc[self.xp.arange(n_sample), gt_roi_label]

        roi_loc_loss = _fast_rcnn_loc_loss(roi_loc, gt_roi_loc, gt_roi_label,
                                           self.roi_sigma)
        roi_cls_loss = F.softmax_cross_entropy(roi_score, gt_roi_label)

        mask_loss = self.mask_loss_fun(
            roi_cls_mask, gt_roi_mask, self.xp, gt_roi_label)
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
