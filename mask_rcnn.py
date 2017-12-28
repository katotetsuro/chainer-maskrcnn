import chainer
from chainer.datasets import TransformDataset
from chainer import training
from chainer.training import extensions
from chainer.datasets import TransformDataset
from chainercv import transforms
import cv2
import numpy as np
from MaskRCNNTrainChain import MaskRCNNTrainChain
from MaskRCNNResnet50 import MaskRCNNResnet50
from COCODataset import COCOMaskLoader

import argparse
from os.path import exists
import gc
import tracemalloc

class Transform(object):

    def __init__(self, faster_rcnn):
        self.faster_rcnn = faster_rcnn

    def __call__(self, in_data):
        img, bbox, label, label_img = in_data
#        scale = 1.0
        _, H, W = img.shape
        img = self.faster_rcnn.prepare(img)
        _, o_H, o_W = img.shape
        scale = o_H / H
        bbox = transforms.resize_bbox(bbox, (H, W), (o_H, o_W))
        for i, im in enumerate(label_img):
            label_img[i] = cv2.resize(im, (o_W, o_H), interpolation=cv2.INTER_NEAREST)
            

        return np.array(img), bbox, label, label_img, scale

def memory_profile(trainer):
    gc.collect()
    print(gc.get_stats())
    snapshot = tracemalloc.take_snapshot()
    snapshot.dump('debug/dump_{}'.format(trainer.updater.iteration))

import objgraph
def breakpoint(trainer):
#    import pdb; pdb.set_trace()
    print('memory----------------------------------------------------------')
    objgraph.show_growth()
    import pdb; pdb.set_trace()

#@profile
def main(args=None):
#    tracemalloc.start()
    parser = argparse.ArgumentParser(
        description='Mask R-CNN')
    parser.add_argument('--gpu', '-g', type=int, default=0)
    parser.add_argument('--lr', '-l', type=float, default=1e-3)
    parser.add_argument('--out', '-o', default='result',
                        help='Output directory')
    parser.add_argument('--step_size', '-ss', type=int, default=50000)
    parser.add_argument('--iteration', '-i', type=int, default=500000)
    parser.add_argument('--weight', '-w', type=str, default='')
    parser.add_argument('--label_file', '-f', type=str, default='YOLOv2/data/label_coco.txt')
    
    if args:
        args = parser.parse_args(args)
    else:
        args = parser.parse_args()
    
    print('lr:{}'.format(args.lr))
    print('output:{}'.format(args.out))
    print('weight:{}'.format(args.weight))
    print('label file:{}'.format(args.label_file))
    
    with open(args.label_file, "r") as f:
        labels = f.read().strip().split("\n")
    
    faster_rcnn = MaskRCNNResnet50(n_fg_class=len(labels))
    faster_rcnn.use_preset('evaluate')
    model = MaskRCNNTrainChain(faster_rcnn)
    if exists(args.weight):
        chainer.serializers.load_npz(args.weight , model.faster_rcnn)
    chainer.cuda.get_device_from_id(args.gpu).use()
    model.to_gpu()
        
    optimizer = chainer.optimizers.MomentumSGD(lr=args.lr, momentum=0.9)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(rate=0.0005))

    coco_train_data = COCOMaskLoader(category_filter=labels)
    train_data = TransformDataset(coco_train_data, Transform(faster_rcnn))

    train_iter = chainer.iterators.SerialIterator(
        train_data, batch_size=1, repeat=True, shuffle=False)
#     test_iter = chainer.iterators.SerialIterator(
#         coco_test_data, batch_size=1, repeat=False, shuffle=False)
    updater = chainer.training.updater.StandardUpdater(
        train_iter, optimizer, device=0)
    
    trainer = training.Trainer(
        updater, (args.iteration, 'iteration'), args.out)

    trainer.extend(
            extensions.snapshot_object(model.faster_rcnn, 'model_iter_{.updater.iteration}.npz'),
            trigger=(5000, 'iteration'))

    trainer.extend(extensions.ExponentialShift('lr', 0.1),
                   trigger=(2, 'epoch'))

    log_interval = 100, 'iteration'
    plot_interval = 3000, 'iteration'
    print_interval = 100, 'iteration'

    trainer.extend(chainer.training.extensions.observe_lr(),
                   trigger=log_interval)
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.PrintReport(
        ['iteration', 'epoch', 'elapsed_time', 'lr',
         'main/loss',
         'main/mask_loss',
         'main/roi_loc_loss',
         'main/roi_cls_loss',
         'main/rpn_loc_loss',
         'main/rpn_cls_loss',
         'validation/main/map',
         ]), trigger=print_interval)
    trainer.extend(extensions.ProgressBar(update_interval=200))
    trainer.extend(extensions.dump_graph('main/loss'))
#    trainer.extend(breakpoint, trigger=(100, 'iteration'))
 #   trainer.extend(memory_profile, trigger=(500, 'iteration'))
    
    trainer.run()


# In[ ]:

main([
        '-o', 'result/leak',
        '-w', 'result/fix_activation_2/model_iter_5000.npz'
    ])


