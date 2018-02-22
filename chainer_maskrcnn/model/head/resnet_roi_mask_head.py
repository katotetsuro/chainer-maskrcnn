# ResNet50を使ったMaskHeadの実装
import chainer
import chainer.links as L
import chainer.functions as F
from chainer.links.model.vision.resnet import ResNet50Layers, BuildingBlock, _global_average_pooling_2d
import numpy as np
import copy
from chainer_maskrcnn.functions.roi_align_2d_yx import _roi_align_2d_yx


class ResnetRoIMaskHead(chainer.Chain):
    def __init__(self,
                 n_class,
                 roi_size,
                 spatial_scale,
                 loc_initialW=None,
                 score_initialW=None,
                 mask_initialW=None):
        # n_class includes the background
        super().__init__()
        with self.init_scope():
            # res5ブロックがほしいだけなのに全部読み込むのは無駄ではある
            resnet50 = ResNet50Layers()
            self.res5 = copy.deepcopy(resnet50.res5)
            # strideは1にする
            self.res5.a.conv1.stride = (1, 1)
            self.res5.a.conv4.stride = (1, 1)
            # 論文　図3の左から2つめ
            self.conv1 = L.Convolution2D(
                in_channels=None, out_channels=2048, ksize=3, stride=1, pad=1)
            # マスク推定ブランチへ
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

            self.cls_loc = L.Linear(2048, n_class * 4, initialW=loc_initialW)
            self.score = L.Linear(2048, n_class, initialW=score_initialW)

        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale

    def __call__(self, x, rois, roi_indices, spatial_scale):
        roi_indices = roi_indices.astype(np.float32)
        indices_and_rois = self.xp.concatenate(
            (roi_indices[:, None], rois), axis=1)

        pool = _roi_align_2d_yx(x, indices_and_rois, self.roi_size,
                                self.roi_size, spatial_scale)

        # h: 分岐する直前まで
        h = F.relu(self.res5(pool))
        h = F.relu(self.conv1(h))
        # global average pooling
        gap = _global_average_pooling_2d(h)
        roi_cls_locs = self.cls_loc(gap)
        roi_scores = self.score(gap)
        # mask
        mask = self.conv2(F.relu(self.deconv1(h)))
        return roi_cls_locs, roi_scores, mask
