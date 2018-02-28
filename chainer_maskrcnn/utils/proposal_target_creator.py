import numpy as np
from chainer import cuda
from chainercv.links.model.faster_rcnn.utils.bbox2loc import bbox2loc
from chainercv.utils.bbox.bbox_iou import bbox_iou
import cv2
from chainer_maskrcnn.model.rpn.multilevel_region_proposal_network import map_rois_to_fpn_levels

# GroundTruthと近いbox, label, maskだけをフィルタリングする


class ProposalTargetCreator(object):
    def __init__(self,
                 sizes=[16],
                 n_sample=256,
                 pos_ratio=0.25,
                 pos_iou_thresh=0.5,
                 neg_iou_thresh_hi=0.5,
                 neg_iou_thresh_lo=0.0):
        self.sizes = sizes
        self.n_sample = n_sample
        self.pos_ratio = pos_ratio
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh_hi = neg_iou_thresh_hi
        self.neg_iou_thresh_lo = neg_iou_thresh_lo

    def __call__(self,
                 roi,
                 bbox,
                 label,
                 mask,
                 levels,
                 loc_normalize_mean=(0., 0., 0., 0.),
                 loc_normalize_std=(0.1, 0.1, 0.2, 0.2),
                 mask_size=14,
                 binary_mask=True):
        """
        binary_mask = False -> keypoint
        """
        xp = cuda.get_array_module(roi)
        roi = cuda.to_cpu(roi)
        bbox = cuda.to_cpu(bbox)
        label = cuda.to_cpu(label)
        mask = cuda.to_cpu(mask)
        levels = cuda.to_cpu(levels)

        n_bbox, _ = bbox.shape
        n_proposal = roi.shape[0]
        roi = np.concatenate((roi, bbox), axis=0)

        # assign feature levels of ground truth boxes
        bbox_levels = map_rois_to_fpn_levels(np, bbox)
        levels = np.concatenate([levels, bbox_levels])

        pos_roi_per_image = np.round(self.n_sample * self.pos_ratio)
        iou = bbox_iou(roi, bbox)
        gt_assignment = iou.argmax(axis=1)
        max_iou = iou.max(axis=1)
        # Offset range of classes from [0, n_fg_class - 1] to [1, n_fg_class].
        # The label with value 0 is the background.
        gt_roi_label = label[gt_assignment] + 1

        # Select foreground RoIs as those with >= pos_iou_thresh IoU.
        pos_index = np.where(max_iou >= self.pos_iou_thresh)[0]
        pos_roi_per_this_image = int(min(pos_roi_per_image, pos_index.size))
        if pos_index.size > 0:
            pos_index = np.random.choice(
                pos_index, size=pos_roi_per_this_image, replace=False)

        # Select background RoIs as those within
        # [neg_iou_thresh_lo, neg_iou_thresh_hi).
        neg_index = np.where((max_iou < self.neg_iou_thresh_hi) &
                             (max_iou >= self.neg_iou_thresh_lo))[0]
        neg_roi_per_this_image = self.n_sample - pos_roi_per_this_image
        neg_roi_per_this_image = int(
            min(neg_roi_per_this_image, neg_index.size))
        if neg_index.size > 0:
            neg_index = np.random.choice(
                neg_index, size=neg_roi_per_this_image, replace=False)

        # The indices that we're selecting (both positive and negative).
        keep_index = np.append(pos_index, neg_index)
        gt_roi_label = gt_roi_label[keep_index]
        gt_roi_label[pos_roi_per_this_image:] = 0  # negative labels --> 0
        sample_roi = roi[keep_index]
        sample_levels = levels[keep_index]

        # Compute offsets and scales to match sampled RoIs to the GTs.
        gt_roi_loc = bbox2loc(sample_roi, bbox[gt_assignment[keep_index]])
        gt_roi_loc = ((gt_roi_loc - np.array(loc_normalize_mean, np.float32)) /
                      np.array(loc_normalize_std, np.float32))

        # https://engineer.dena.jp/2017/12/chainercvmask-r-cnn.html
        gt_roi_mask = []
        _, h, w = mask.shape
        if binary_mask:
            for i, idx in enumerate(gt_assignment[pos_index]):
                A = mask[idx,
                         np.max((int(sample_roi[i, 0]),
                                 0)):np.min((int(sample_roi[i, 2]), h)),
                         np.max((int(sample_roi[i, 1]),
                                 0)):np.min((int(sample_roi[i, 3]), w))]
                gt_roi_mask.append(
                    cv2.resize(A, (mask_size, mask_size)).astype(np.int32))
        else:
            for i, idx in enumerate(gt_assignment[pos_index]):
                m = np.zeros((mask_size, mask_size), dtype=np.int32)
                # remind: shape of keypoints is (N, 17, 3), N is number of bbox, 17 is number of keypoints, 3 is (x, y, v)
                # v=0: unlabeled, v=1, labeled but invisible, v=2 labeled and visible

                # bbox's (y0, x0), (y1, x1)
                y0, x0, y1, x1 = list(map(int, sample_roi[i, :4]))
                kp = mask[idx]  # shape is (17, 3)
                # convert keypoints coordinate (y, x) into mask coordinate system [0, mask_size]x[0, mask_size]
                kp[:, :2] = (kp[:, :2] - [y0, x0]) / [y1-y0, x1-x0] * mask_size
                # mask_size x mask_size 空間でどこにあるかをラベルとして扱う(あとでsoftmax cross entropyする）
                # -1でignoreされる
                keypoint_labels = np.zeros(17, dtype=np.int32)
                for j, r in enumerate(kp):
                    y, x, v = list(map(int, r))
                    if v == 2 and 0 <= y and y < mask_size and 0 <= x and x < mask_size:
                        keypoint_labels[j] = y * mask_size + x

                    else:
                        keypoint_labels[j] = -1

                gt_roi_mask.append(keypoint_labels)

        gt_roi_mask = xp.array(gt_roi_mask)

        if xp != np:
            sample_roi = cuda.to_gpu(sample_roi)
            gt_roi_loc = cuda.to_gpu(gt_roi_loc)
            gt_roi_label = cuda.to_gpu(gt_roi_label)
            gt_roi_mask = cuda.to_gpu(gt_roi_mask)
            sample_levels = cuda.to_gpu(sample_levels)
        return sample_roi, sample_levels, gt_roi_loc, gt_roi_label, gt_roi_mask
