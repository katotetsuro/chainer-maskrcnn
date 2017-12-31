# COCOMaskLoaderを使うとメモリリークが起こることをたしかめる
from COCODataset import COCOMaskLoader
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import reporter

class SimpleChain(chainer.Chain):
    def __init__(self):
        super().__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(3, 1)
            
    def __call__(self, x, *args):
        h = self.conv(x)
        loss = F.mean_squared_error(h, x)
        reporter.report({'loss': loss}, self)
        return loss


# In[6]:


import numpy as np
from chainer import training
from chainer.training import extensions
from os.path import isfile
import _pickle

class T():
    def __call__(self, in_data):
        x, *_ = in_data
        return x.astype(np.float32)

def train_loop():
    model = SimpleChain()
    if chainer.cuda.available:
        model.to_gpu()
    
    optimizer = chainer.optimizers.MomentumSGD(0.001)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(5e-4))
    
    if isfile('coco_data.pkl'):
        print('再利用します')
        with open('coco_data.pkl', 'rb') as f:
            train_data = _pickle.load(f)
    else:
        train_data = COCOMaskLoader(split='val')
        print('次回のために書き出します')
        with open('coco_data.pkl', 'wb') as f:
            _pickle.dump(train_data, f)
            
    train_data = chainer.datasets.TransformDataset(train_data, T())
    train_iter = chainer.iterators.SerialIterator(train_data, 1, shuffle=False)

    updater = training.StandardUpdater(train_iter, optimizer, device=chainer.cuda.get_device().id)
    trainer = training.Trainer(updater, (100, 'epoch'), out='result/leak')
    trainer.extend(extensions.LogReport(trigger=(100, 'iteration')))
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss']))

    
    trainer.run()
    


train_loop()
