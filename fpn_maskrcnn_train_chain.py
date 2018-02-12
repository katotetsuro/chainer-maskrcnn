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
            mask_size=28)

        #return sample_roi, sample_levels, gt_roi_label, gt_roi_mask

        sample_roi_index = self.xp.zeros(
            (len(sample_roi), ), dtype=np.int32)

        # sample_levelsが[0,...0,1,...,1,2,...]となるように並べた場合、元々の配列のindexがいくつのデータかを指す
        # cupyのargsortの実装はquicksortで、同じlevelの中でも、元の順序関係が保持されないことに注意
        level_order_box_indices = sample_levels.argsort()

        # cupyのargsortにmergesortが実装されていないので、しょうがなく一度cpuに戻してやる場合
        #sample_levels = chainer.cuda.to_cpu(samle_levels)
        #level_order_box_indices = sample_levels.argsort(kind='mergesort')
        #level_order_box_indices = chainer.cuda.to_gpu(level_order_box_indices)

        # join roi and index of batch
        indices_and_rois = self.xp.concatenate(
            (sample_roi_index[:, None], sample_roi), axis=1).astype(self.xp.float32)
        # separate (rois, roi_indices) by sample_levels
        # ↓このやり方は、level_order_box_indicesと順序が一致しない
        #indices_and_rois = [indices_and_rois[sample_levels==i] for i in range(len(features))]

        indices_and_rois = indices_and_rois[level_order_box_indices]
        # cupyにuniqueがないので、自力実装
        # levelが変わる部分のindexを抽出して、levelごとにsplitする
        sorted_levels = sample_levels[level_order_box_indices]
        sp, = np.where([sorted_levels[i]!=sorted_levels[i+1] for i in range(len(sample_levels)-1)])
        sp += 1
        indices_and_rois = self.xp.split(indices_and_rois, sp)

        gt_roi_loc = gt_roi_loc[level_order_box_indices]
        gt_roi_label = gt_roi_label[level_order_box_indices]

        # RPN losses
        gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(
            bbox, anchor, img_size)
        rpn_loc_loss = _fast_rcnn_loc_loss(rpn_loc, gt_rpn_loc,
                                       gt_rpn_label, self.rpn_sigma)
        rpn_cls_loss = F.softmax_cross_entropy(rpn_score, gt_rpn_label)

        # Losses for outputs of the head.
        roi_cls_loc, roi_score, roi_cls_mask = self.faster_rcnn.head(
                features, indices_and_rois, self.faster_rcnn.extractor.spatial_scales)

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

        # mask
        # https://engineer.dena.jp/2017/12/chainercvmask-r-cnn.html
        l = level_order_box_indices
        a, = self.xp.where(l < gt_roi_mask.shape[0])
        roi_mask = roi_cls_mask[self.xp.arange(n_sample), gt_roi_label-1]
        mask_loss = F.sigmoid_cross_entropy(roi_mask[a],
                                            gt_roi_mask[l[a]])

        loss = rpn_loc_loss + rpn_cls_loss + roi_loc_loss + roi_cls_loss + mask_loss

        chainer.reporter.report({
            'rpn_loc_loss': rpn_loc_loss,
            'rpn_cls_loss': rpn_cls_loss,
            'roi_loc_loss': roi_loc_loss,
            'roi_cls_loss': roi_cls_loss,
            'mask_loss': mask_loss,
            'loss': loss
        }, self)

        #import pdb; pdb.set_trace()

        if math.isnan(loss.data.any()):
            loss = chainer.Variable(self.xp.array([0], dtype=self.xp.float32))
        return loss
