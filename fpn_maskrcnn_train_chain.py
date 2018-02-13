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
import math

measure_time = False

class FPNMaskRCNNTrainChain(FasterRCNNTrainChain):
    def __init__(self,
                 faster_rcnn,
                 rpn_sigma=3.,
                 roi_sigma=1.,
                 anchor_target_creator=AnchorTargetCreator()):
        # todo: clean up class dependencies
        proposal_target_creator = ProposalTargetCreator(faster_rcnn.extractor.anchor_sizes)
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

        fw_s = time.time()
        fw_e_s = time.time()
        features = self.faster_rcnn.extractor(imgs)
        fw_e_e = time.time()
        #print(f'forward(extractor):{fw_e_e-fw_e_s}')

        # Since batch size is one, convert variables to singular form
        bbox = bboxes[0]
        label = labels[0]
        mask = masks[0]

        fw_rpn_s = time.time()
        rpn_locs, rpn_scores, rois, roi_indices, anchor, levels = self.faster_rcnn.rpn(
            features, img_size, scale)
        fw_rpn_e = time.time()
        #print(f'forward(rpn):{fw_rpn_e-fw_rpn_s}')

        # Since batch size is one, convert variables to singular form
        rpn_score = rpn_scores[0]
        rpn_loc = rpn_locs[0]
        roi = rois

        # Sample RoIs and forward
        # gt_roi_labelになった時点で [0, NUM_FOREGROUND_CLASS-1]が[1, NUM_FOREGROUND_CLASS]にシフトしている
        fw_prop_s = time.time()
        sample_roi, gt_roi_loc, gt_roi_label, gt_roi_mask, gt_mask_indices, split_index = self.proposal_target_creator(
            roi,
            bbox,
            label,
            mask,
            levels,
            self.loc_normalize_mean,
            self.loc_normalize_std,
            mask_size=28)
        fw_prop_e = time.time()
        #print(f'forward(proposal):{fw_prop_e-fw_prop_s}')


        #print('check', sample_roi.shape[0], gt_roi_loc.shape[0], gt_roi_label.shape[0], split_index)

        sample_roi_index = self.xp.zeros(
            (len(sample_roi), ), dtype=np.int32)

        # join roi and index of batch
        indices_and_rois = self.xp.concatenate(
            (sample_roi_index[:, None], sample_roi), axis=1).astype(self.xp.float32)
        split_index = chainer.cuda.to_cpu(split_index)
        #print(type(split_index), split_index)
        indices_and_rois = self.xp.split(indices_and_rois, split_index)

        # RPN losses
        gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(
            bbox, anchor, img_size)
        rpn_loc_loss = _fast_rcnn_loc_loss(rpn_loc, gt_rpn_loc,
                                       gt_rpn_label, self.rpn_sigma)
        rpn_cls_loss = F.softmax_cross_entropy(rpn_score, gt_rpn_label)

        # Losses for outputs of the head.
        fw_head_s = time.time()
        roi_cls_loc, roi_score, roi_cls_mask = self.faster_rcnn.head(
                features, indices_and_rois, self.faster_rcnn.extractor.spatial_scales)
        fw_head_e = time.time()
        #print(f'forward(head):{fw_head_e-fw_head_s}')

        # Losses for outputs of the head.
        n_sample = roi_cls_loc.shape[0]
        roi_cls_loc = roi_cls_loc.reshape(n_sample, -1, 4)
        # light-headのときはそのまま使う
        if roi_cls_loc.shape[1] == 1:
            roi_loc = roi_cls_loc.reshape(n_sample, 4)
        else:
            roi_loc = roi_cls_loc[self.xp.arange(n_sample), gt_roi_label]

        #print(roi_loc.shape, gt_roi_loc.shape, roi_score.shape,  gt_roi_label.shape)
        roi_loc_loss = _fast_rcnn_loc_loss(roi_loc, gt_roi_loc, gt_roi_label,
                                           self.roi_sigma)
        roi_cls_loss = F.softmax_cross_entropy(roi_score, gt_roi_label)

        # mask
        # https://engineer.dena.jp/2017/12/chainercvmask-r-cnn.html
        roi_mask = roi_cls_mask[self.xp.arange(n_sample), gt_roi_label-1]
        mask_loss = F.sigmoid_cross_entropy(roi_mask[gt_mask_indices],
                                            gt_roi_mask)

        loss = rpn_loc_loss + rpn_cls_loss + roi_loc_loss + roi_cls_loss + mask_loss

        fw_e = time.time()
        #print(f'forward(total):{fw_e-fw_s}')

        chainer.reporter.report({
            'rpn_loc_loss': rpn_loc_loss,
            'rpn_cls_loss': rpn_cls_loss,
            'roi_loc_loss': roi_loc_loss,
            'roi_cls_loss': roi_cls_loss,
            'mask_loss': mask_loss,
            'loss': loss
        }, self)

        #bw_s = time.time()
        #loss.backward()
        #bw_e = time.time()
        #print(f'backward(total):{bw_e-bw_s}')

        return loss
