import chainer
import chainer.links as L
import chainer.functions as F
from chainer.links.model.vision.resnet import ResNet50Layers, BuildingBlock, _global_average_pooling_2d
import numpy as np
import copy
from chainer_maskrcnn.functions.roi_align_2d_yx import _roi_align_2d_yx


class FPNRoIKeypointHead(chainer.Chain):
    mask_size = 56

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
            self.cls_loc = L.Linear(1024, 4, initialW=loc_initialW)
            self.score = L.Linear(1024, n_class, initialW=score_initialW)

            # mask prediction path
            self.mask_convs = chainer.ChainList()
            for i in range(8):
                self.mask_convs.add_link(
                    L.Convolution2D(None, 256, ksize=3, pad=1))
            #self.mask1 = L.Convolution2D(None, 256, ksize=3, pad=1)
            #self.mask2 = L.Convolution2D(None, 256, ksize=3, pad=1)
            #self.mask3 = L.Convolution2D(None, 256, ksize=3, pad=1)
            #self.mask4 = L.Convolution2D(None, 256, ksize=3, pad=1)
            self.deconv1 = L.Deconvolution2D(
                in_channels=None,
                out_channels=256,
                ksize=2,
                stride=2,
                pad=0,
                initialW=mask_initialW)
            self.conv2 = L.Convolution2D(
                in_channels=None,
                out_channels=17,
                ksize=1,
                stride=1,
                pad=0,
                initialW=mask_initialW)

        self.n_class = n_class
        self.roi_size_box = roi_size_box
        self.roi_size_mask = roi_size_mask

    def __call__(self, x, indices_and_rois, levels, spatial_scales):

        pool_box = list()
        levels = chainer.cuda.to_cpu(levels).astype(np.int32)
        for l, i in zip(levels, indices_and_rois):
            pool_box.append(_roi_align_2d_yx(x[l], i[None], self.roi_size_box,
                                             self.roi_size_box, spatial_scales[l]))

        pool_box = F.concat(pool_box, axis=0)

        h = F.relu(self.conv1(pool_box))
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        roi_cls_locs = self.cls_loc(h)
        roi_scores = self.score(h)
        # at prediction time, we use two pass method.
        # at first path, we predict box location and class
        # at second path, we predict mask with accurate location from first path
        if chainer.config.train:
            pool_mask = list()
            for l, i in zip(levels, indices_and_rois):
                pool_mask.append(_roi_align_2d_yx(x[l], i[None], self.roi_size_mask,
                                                  self.roi_size_mask, spatial_scales[l]))
            mask = F.concat(pool_mask, axis=0)
            for l in self.mask_convs.children():
                mask = F.relu(l(mask))
            #mask = F.relu(self.mask1(pool_mask))
            #mask = F.relu(self.mask2(mask))
            #mask = F.relu(self.mask3(mask))
            #mask = F.relu(self.mask4(mask))
            mask = self.conv2(self.deconv1(mask))
            *_, h, w = mask.shape
            mask = F.resize_images(mask, output_shape=(2 * h, 2 * w))
            return roi_cls_locs, roi_scores, mask
        else:
            # cache
            self.x = x
            return roi_cls_locs, roi_scores

    def predict_mask(self, levels, indices_and_rois, spatial_scales):
        pool_mask = list()
        for l, i in zip(levels, indices_and_rois):
            pool_mask.append(_roi_align_2d_yx(self.x[l], i[None], self.roi_size_mask,
                                              self.roi_size_mask, spatial_scales[l]))
        mask = F.concat(pool_mask, axis=0)
        for l in self.mask_convs:
            mask = F.relu(l(mask))
        #mask = F.relu(self.mask1(pool_mask))
        #mask = F.relu(self.mask2(mask))
        #mask = F.relu(self.mask3(mask))
        #mask = F.relu(self.mask4(mask))
        mask = self.conv2(self.deconv1(mask))
        *_, h, w = mask.shape
        mask = F.resize_images(mask, output_shape=(2 * h, 2 * w))

        return mask
