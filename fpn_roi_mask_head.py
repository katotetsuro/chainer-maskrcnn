import chainer
import chainer.links as L
import chainer.functions as F
from chainer.links.model.vision.resnet import ResNet50Layers, BuildingBlock, _global_average_pooling_2d
import numpy as np
import copy
from roi_align_2d_yx import _roi_align_2d_yx


class FPNRoIMaskHead(chainer.Chain):
    def __init__(self,
                 n_class,
                 roi_size_box,
                 roi_size_mask,
                 loc_initialW=None,
                 score_initialW=None,
                 mask_initialW=None):
        # n_class includes the background
        super().__init__()
        with self.init_scope():
            # layers for box prediction path
            self.conv1 = L.Convolution2D(
                in_channels=None, out_channels=256, ksize=3, pad=1)
            self.fc1 = L.Linear(None, 1024)
            self.fc2 = L.Linear(None, 1024)
            self.cls_loc = L.Linear(1024, n_class * 4, initialW=loc_initialW)
            self.score = L.Linear(1024, n_class, initialW=score_initialW)

            # mask prediction path
            self.mask1 = L.Convolution2D(None, 256, ksize=3, pad=1)
            self.mask2 = L.Convolution2D(None, 256, ksize=3, pad=1)
            self.mask3 = L.Convolution2D(None, 256, ksize=3, pad=1)
            self.mask4 = L.Convolution2D(None, 256, ksize=3, pad=1)
            self.deconv1 = L.Deconvolution2D(
                in_channels=None,
                out_channels=256,
                ksize=2,
                stride=2,
                pad=0,
                initialW=mask_initialW)
            self.conv2 = L.Convolution2D(
                in_channels=None,
                out_channels=n_class - 1,
                ksize=3,
                stride=1,
                pad=1,
                initialW=mask_initialW)

        self.n_class = n_class
        self.roi_size_box = roi_size_box
        self.roi_size_mask = roi_size_mask

    def __call__(self, x, rois, roi_indices, spatial_scale):

        roi_indices = roi_indices.astype(np.float32)
        indices_and_rois = self.xp.concatenate(
            (roi_indices[:, None], rois), axis=1)

        pool_box = _roi_align_2d_yx(x, indices_and_rois, self.roi_size_box,
                                    self.roi_size_box, spatial_scale)
        pool_mask = _roi_align_2d_yx(x, indices_and_rois, self.roi_size_mask,
                                     self.roi_size_mask, spatial_scale)

        h = F.relu(self.conv1(pool_box))
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        roi_cls_locs = self.cls_loc(h)
        roi_scores = self.score(h)
        # at prediction time, we use two pass method.
        # at first path, we predict box location and class
        # at second path, we predict mask with accurate location from first path
        if chainer.config.train:
            mask = self.conv2(self.deconv1(pool_mask))
            return roi_cls_locs, roi_scores, mask
        else:
            # cache tfp for second path
            self.tfp = tfp
            return roi_cls_locs, roi_scores

    def predict_mask(self, rois, roi_indices, spatial_scale):
        roi_indices = roi_indices.astype(np.float32)
        indices_and_rois = self.xp.concatenate(
            (roi_indices[:, None], rois), axis=1)
        pool = _roi_align_2d_yx(self.tfp, indices_and_rois, self.roi_size,
                                self.roi_size, spatial_scale)

        mask = self.conv2(self.deconv1(pool))
        return mask