import chainer
from chainer.datasets import TransformDataset
from chainer.training import extensions
from chainercv import transforms
from chainerui.utils import save_args
from chainerui.extensions import CommandsExtension
import cv2
import numpy as np
from chainer_maskrcnn.model.fpn_maskrcnn_train_chain import FPNMaskRCNNTrainChain
from chainer_maskrcnn.model.maskrcnn_resnet50 import MaskRCNNResnet50
from chainer_maskrcnn.dataset.coco_dataset import COCOMaskLoader

import argparse
from os.path import exists, isfile
import time
import _pickle as pickle


class Transform(object):
    def __init__(self, faster_rcnn):
        self.faster_rcnn = faster_rcnn

    def __call__(self, in_data):
        img, bbox, label, label_img = in_data
        _, H, W = img.shape
        img = self.faster_rcnn.prepare(img)
        _, o_H, o_W = img.shape
        scale = o_H / H
        bbox = transforms.resize_bbox(bbox, (H, W), (o_H, o_W))
        for i, im in enumerate(label_img):
            label_img[i] = cv2.resize(
                im, (o_W, o_H), interpolation=cv2.INTER_NEAREST)

        return img, bbox, label, label_img, scale

# mask
# https://engineer.dena.jp/2017/12/chainercvmask-r-cnn.html


def calc_mask_loss(roi_cls_mask, gt_roi_mask, xp, gt_roi_label):
    roi_mask = roi_cls_mask[xp.arange(
        roi_cls_mask.shape[0]), gt_roi_label - 1]
    return chainer.functions.sigmoid_cross_entropy(roi_mask[:gt_roi_mask.shape[0]],
                                                   gt_roi_mask)


def main():
    parser = argparse.ArgumentParser(description='Mask R-CNN')
    parser.add_argument('--gpu', '-g', type=int, default=0)
    parser.add_argument('--lr', '-l', type=float, default=1e-3)
    parser.add_argument(
        '--out', '-o', default='result', help='Output directory')
    parser.add_argument('--iteration', '-i', type=int, default=200000)
    parser.add_argument('--weight', '-w', type=str, default='')
    parser.add_argument(
        '--label_file', '-f', type=str, default='data/label_coco.txt')
    parser.add_argument('--backbone', type=str, default='fpn')
    parser.add_argument('--head_arch', '-a', type=str, default='fpn')
    parser.add_argument('--multi_gpu', '-m', type=int, default=0)
    parser.add_argument('--batch_size', '-b', type=int, default=1)

    args = parser.parse_args()

    print('lr:{}'.format(args.lr))
    print('output:{}'.format(args.out))
    print('weight:{}'.format(args.weight))
    print('label file:{}'.format(args.label_file))
    print('iteration::{}'.format(args.iteration))
    print('backbone architecture:{}'.format(args.backbone))
    print('head architecture:{}'.format(args.head_arch))

    if args.multi_gpu:
        print('try to use chainer.training.updaters.MultiprocessParallelUpdater')
        if not chainer.training.updaters.MultiprocessParallelUpdater.available():
            print('MultiprocessParallelUpdater is not available')
            args.multi_gpu = 0

    with open(args.label_file, "r") as f:
        labels = f.read().strip().split("\n")

    faster_rcnn = MaskRCNNResnet50(
        n_fg_class=len(labels), backbone=args.backbone, head_arch=args.head_arch)
    faster_rcnn.use_preset('evaluate')
    model = FPNMaskRCNNTrainChain(faster_rcnn, mask_loss_fun=calc_mask_loss)
    if exists(args.weight):
        chainer.serializers.load_npz(
            args.weight, model.faster_rcnn, strict=False)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    optimizer = chainer.optimizers.MomentumSGD(lr=args.lr, momentum=0.9)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(rate=0.0005))

    pkl_file = 'train_data.pkl'
    if isfile(pkl_file):
        print('pklから読み込みます')
        dataload_start = time.time()
        with open(pkl_file, 'rb') as f:
            coco_train_data = pickle.load(f)
        dataload_end = time.time()
        print('pklからの読み込み {}'.format(dataload_end - dataload_start))
    else:
        dataload_start = time.time()
        coco_train_data = COCOMaskLoader(category_filter=labels)
        dataload_end = time.time()
        print('普通の読み込み {}'.format(dataload_end - dataload_start))
        print('次回のために保存します')
        with open(pkl_file, 'wb') as f:
            pickle.dump(coco_train_data, f)

    train_data = TransformDataset(coco_train_data, Transform(faster_rcnn))

    if args.multi_gpu:
        train_iters = [chainer.iterators.SerialIterator(
            train_data, 1, repeat=True, shuffle=True) for i in range(8)]
        updater = chainer.training.updater.MultiprocessParallelUpdater(
            train_iters, optimizer, device=range(8))

    else:
        train_iter = chainer.iterators.SerialIterator(
            train_data, batch_size=args.batch_size, repeat=True, shuffle=False)
        updater = chainer.training.updater.StandardUpdater(
            train_iter, optimizer, device=args.gpu)

    trainer = chainer.training.Trainer(updater, (args.iteration, 'iteration'),
                                       args.out)

    trainer.extend(
        extensions.snapshot_object(model.faster_rcnn,
                                   'model_{.updater.iteration}.npz'),
        trigger=(5000, 'iteration'))

    trainer.extend(
        extensions.ExponentialShift('lr', 0.1), trigger=(2, 'epoch'))

    log_interval = 100, 'iteration'
    trainer.extend(
        chainer.training.extensions.observe_lr(), trigger=log_interval)
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(
        extensions.PrintReport([
            'iteration',
            'epoch',
            'elapsed_time',
            'lr',
            'main/loss',
            'main/mask_loss',
            'main/roi_loc_loss',
            'main/roi_cls_loss',
            'main/rpn_loc_loss',
            'main/rpn_cls_loss',
        ]),
        trigger=(100, 'iteration'))
    trainer.extend(extensions.ProgressBar(update_interval=200))
    trainer.extend(extensions.dump_graph('main/loss'))

    save_args(args, args.out)
    trainer.extend(CommandsExtension(), trigger=(100, 'iteration'))

    trainer.run()


if __name__ == '__main__':
    main()
