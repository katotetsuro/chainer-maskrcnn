import chainer
import chainer.links as L
import chainer.functions as F
from chainer.links.model.vision.resnet import ResNet50Layers


class FeaturePyramidNetwork(chainer.Chain):
    def __init__(self):
        super().__init__()
        with self.init_scope():
            # bottom up
            self.resnet = ResNet50Layers('auto')
            del self.resnet.fc6
            # top layer (reduce channel)
            self.toplayer = L.Convolution2D(
                in_channels=None, out_channels=256, ksize=1, stride=1, pad=0)

            # conv layer for top-down pathway
            self.conv_p4 = L.Convolution2D(None, 256, ksize=3, stride=1, pad=1)
            self.conv_p3 = L.Convolution2D(None, 256, ksize=3, stride=1, pad=1)
            self.conv_p2 = L.Convolution2D(None, 256, ksize=3, stride=1, pad=1)

            # lateral connection
            self.lat_p4 = L.Convolution2D(
                in_channels=None, out_channels=256, ksize=1, stride=1, pad=0)
            self.lat_p3 = L.Convolution2D(
                in_channels=None, out_channels=256, ksize=1, stride=1, pad=0)
            self.lat_p2 = L.Convolution2D(
                in_channels=None, out_channels=256, ksize=1, stride=1, pad=0)

    def __call__(self, x):
        # bottom-up pathway
        h = F.relu(self.resnet.bn1(self.resnet.conv1(x)))
        h = F.max_pooling_2d(h, ksize=(2, 2))
        c2 = self.resnet.res2(h)
        c3 = self.resnet.res3(c2)
        c4 = self.resnet.res4(c3)
        c5 = self.resnet.res5(c4)

        # top
        p5 = self.toplayer(c5)
        p4 = self.conv_p4(
            F.unpooling_2d(p5, ksize=2, outsize=(
                c4.shape[2:4])) + self.lat_p4(c4))
        p3 = self.conv_p3(
            F.unpooling_2d(p4, ksize=2, outsize=(
                c3.shape[2:4])) + self.lat_p3(c3))
        p2 = self.conv_p2(
            F.unpooling_2d(p3, ksize=2, outsize=(
                c2.shape[2:4])) + self.lat_p2(c2))
        # fine to coarse
        return p2, p3, p4, p5  
