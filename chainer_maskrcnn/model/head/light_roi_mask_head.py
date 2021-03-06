# Light-Head R-CNN: In Defense of Two-Stage Object Detector  http://arxiv.org/abs/1711.07264
import chainer
import chainer.links as L
import chainer.functions as F
from chainer.links.model.vision.resnet import ResNet50Layers, BuildingBlock, _global_average_pooling_2d
import numpy as np
import copy
from chainer_maskrcnn.functions.roi_align_2d_yx import _roi_align_2d_yx


class LightRoIMaskHead(chainer.Chain):
    mask_size = 14

    def __init__(self,
                 n_class,
                 roi_size,
                 loc_initialW=None,
                 score_initialW=None,
                 mask_initialW=None):
        # n_class includes the background
        super().__init__()
        with self.init_scope():
            # Separable Convolution Layers 変数の名前はpaperに準拠してみた
            k = 15
            C_mid = 256
            C_out = 490
            # 初期値ワカラン・・ chainer.initializers.Normal(0.001)とかの方がいいか？
            # レイヤーの名前は論文の図を見たときに up left, bottom left, up right, bottom rightの４つw
            p = int(k / 2)
            self.conv_ul = L.Convolution2D(
                in_channels=None, out_channels=C_mid, ksize=(k, 1), pad=(p, 0))
            self.conv_bl = L.Convolution2D(
                in_channels=C_mid,
                out_channels=C_out,
                ksize=(1, k),
                pad=(0, p))
            self.conv_ur = L.Convolution2D(
                in_channels=None, out_channels=C_mid, ksize=(1, k), pad=(0, p))
            self.conv_br = L.Convolution2D(
                in_channels=C_mid,
                out_channels=C_out,
                ksize=(k, 1),
                pad=(p, 0))
            self.fc = L.Linear(None, 2048)
            self.cls_loc = L.Linear(2048, 4, initialW=loc_initialW)
            self.score = L.Linear(2048, n_class, initialW=score_initialW)
            # マスク推定ブランチへ
            self.conv2 = L.Convolution2D(
                in_channels=None,
                out_channels=256,
                ksize=3,
                stride=1,
                pad=1,
                initialW=mask_initialW)
            self.conv3_ = L.Convolution2D(
                in_channels=None,
                out_channels=256,
                ksize=3,
                stride=1,
                pad=1,
                initialW=mask_initialW)
            self.conv4 = L.Convolution2D(
                in_channels=None,
                out_channels=256,
                ksize=3,
                stride=1,
                pad=1,
                initialW=mask_initialW)
            self.deconv1_ = L.Deconvolution2D(
                in_channels=None,
                out_channels=n_class - 1,
                ksize=2,
                stride=2,
                pad=0,
                initialW=mask_initialW)

        self.n_class = n_class
        self.roi_size = roi_size

    def __call__(self, x, rois, roi_indices, spatial_scale):
        roi_indices = roi_indices.astype(np.float32)
        indices_and_rois = self.xp.concatenate(
            (roi_indices[:, None], rois), axis=1)

        # roi poolingをする前に、thin feature mapに変換します
        # activationしないっぽいことが書いてあるんだよなー
        left_path = self.conv_bl(self.conv_ul(x))
        right_path = self.conv_br(self.conv_ur(x))
        tfp = left_path + right_path

        pool = _roi_align_2d_yx(tfp, indices_and_rois, self.roi_size,
                                self.roi_size, spatial_scale)

        h = F.relu(self.fc(pool))
        roi_cls_locs = self.cls_loc(h)
        roi_scores = self.score(h)
        # at prediction time, we use two pass method.
        # at first path, we predict box location and class
        # at second path, we predict mask with accurate location from first path
        if chainer.config.train:
            mask = F.relu(self.conv2(pool))
            mask = F.relu(self.conv3_(mask))
            mask = F.relu(self.conv4(mask))
            mask = self.deconv1_(pool)
            #mask = self.conv2(self.deconv1(pool))
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

        mask = F.relu(self.conv2(pool))
        mask = F.relu(self.conv3_(mask))
        mask = F.relu(self.conv4(mask))
        mask = self.deconv1_(pool)
#        mask = self.deconv1(pool)
#        mask = self.conv2_(mask)
#        mask = self.conv3(mask)
#        mask = self.conv2(self.deconv1(pool))
        return mask
