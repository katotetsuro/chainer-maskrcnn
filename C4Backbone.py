from chainer.links.model.vision.resnet import ResNet50Layers
import collections
import chainer.functions as F
from chainer.links import BatchNormalization

class C4Backbone(ResNet50Layers):
    def __init__(self, pretrained_model):
        super().__init__(pretrained_model)
        del self.res5
        del self.fc6
        
        for l in self.links():
            if isinstance(l, BatchNormalization):
                l.disable_update()

    @property
    def functions(self):
        return collections.OrderedDict(
            [('conv1', [self.conv1, self.bn1, F.relu]),
             ('pool1', [lambda x: F.max_pooling_2d(x, ksize=3, stride=2)]),
             ('res2', [self.res2]), ('res3', [self.res3]), ('res4',
                                                            [self.res4])])

    def __call__(self, x, **kwargs):
        return super().__call__(x, ['res4'], **kwargs)['res4']
