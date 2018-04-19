import chainer
import chainer.links as L
import chainer.functions as F


class ConvBatch(chainer.Chain):
    def __init__(self, out_channels, ksize, stride, pad, activation):
        super().__init__()
        with self.init_scope():
            self.c = L.Convolution2D(in_channels=None, out_channels=out_channels,
                                     ksize=ksize, stride=stride, pad=pad)
            self.bn = L.BatchNormalization(size=out_channels)
        self.activation = activation

    def __call__(self, x):
        return self.activation(self.bn(self.c(x)))


class Darknet(chainer.Chain):
    # determined by network architecture (where stride >1 occurs.)
    feat_strides = [16]
    # inverse of feat_strides. used in RoIAlign to calculate x in Image Coord to x' in feature map
    spatial_scales = list(map(lambda x: 1. / x, feat_strides))
    anchor_base = 16  # from original implementation. why?
    # from FPN paper.
    anchor_sizes = [64]
    # anchor_sizes / anchor_base anchor_base is invisible from lamba function??
    anchor_scales = list(map(lambda x: x / 16., anchor_sizes))

    def __init__(self, activation=F.relu):
        super().__init__()

        with self.init_scope():
            self.conv1 = ConvBatch(16, 3, 1, 1, activation)
            self.conv2 = ConvBatch(32, 3, 1, 1, activation)
            self.conv3 = ConvBatch(64, 3, 1, 1, activation)
            self.conv4 = ConvBatch(128, 3, 1, 1, activation)
            self.conv5 = ConvBatch(256, 3, 1, 1, activation)
            # self.conv6 = ConvBatch(512, 3, 1, 1)
            # self.conv7 = ConvBatch(1024, 3, 1, 1)
        # anchor_sizes / anchor_base anchor_base is invisible from lamba function??
        self.anchor_scales = list(
            map(lambda x: x / float(self.anchor_base), self.anchor_sizes))

    def __call__(self, x):
        h = self.conv1(x)
        h = F.max_pooling_2d(h, ksize=2, stride=2)
        h = self.conv2(h)
        h = F.max_pooling_2d(h, ksize=2, stride=2)
        h = self.conv3(h)
        h = F.max_pooling_2d(h, ksize=2, stride=2)
        h = self.conv4(h)
        h = F.max_pooling_2d(h, ksize=2, stride=2)
        h = self.conv5(h)
        # h = F.max_pooling_2d(h, ksize=2, stride=2)
        # h = self.conv6(h)
        # h = F.max_pooling_2d(h, ksize=2, stride=2)
        # h = self.conv7(h)

        return h,
