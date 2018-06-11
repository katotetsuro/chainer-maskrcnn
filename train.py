import chainer
from chainer.datasets import TransformDataset
from chainer.training import extensions
from chainercv import transforms
from chainer_maskrcnn.extentions.evaluator.instance_segmentation_voc_evaluator import InstanceSegmentationVOCEvaluator
from chainerui.utils import save_args
from chainerui.extensions import CommandsExtension
import cv2
import numpy as np
from chainer_maskrcnn.model.fpn_maskrcnn_train_chain import FPNMaskRCNNTrainChain
from chainer_maskrcnn.model.maskrcnn import MaskRCNN
from chainer_maskrcnn.dataset.coco_dataset import COCOMaskLoader

import argparse
from os.path import exists, isfile
import time
import _pickle as pickle
import warnings


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
        bbox[:, 2:] = np.maximum(bbox[:, 2:], bbox[:, 2:] + 1)
        for i, im in enumerate(label_img):
            label_img[i] = cv2.resize(
                im, (o_W, o_H), interpolation=cv2.INTER_NEAREST)

        return img, bbox, label, label_img, scale


class EvaluatorTransform():
    """
    chainercvのevalに合うようにTransform
    """

    def __call__(self, in_data):
        i, b, l, m = in_data
        return i, np.stack(m), l


def calc_mask_loss(roi_cls_mask, gt_roi_mask, xp, gt_roi_label):
    """
    mask loss
    https://engineer.dena.jp/2017/12/chainercvmask-r-cnn.html
    """
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
    parser.add_argument('--head-arch', '-a', type=str, default='fpn')
    parser.add_argument('--multi-gpu', '-m', type=int, default=0)
    parser.add_argument('--batch-size', '-b', type=int, default=1)

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

    faster_rcnn = MaskRCNN(
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

    train_data = COCOMaskLoader(category_filter=labels, data_type='2017')
    train_data = TransformDataset(train_data, Transform(faster_rcnn))
    test_data = COCOMaskLoader(
        category_filter=labels, data_type='2017', split='val')
    test_data = TransformDataset(test_data, EvaluatorTransform())

    if args.multi_gpu:
        train_iters = [chainer.iterators.SerialIterator(
            train_data, 1, repeat=True, shuffle=True) for i in range(8)]
        updater = chainer.training.updater.MultiprocessParallelUpdater(
            train_iters, optimizer, device=range(8))

    else:
        train_iter = chainer.iterators.MultithreadIterator(
            train_data, batch_size=args.batch_size, repeat=True, shuffle=False)
        test_iter = chainer.iterators.SerialIterator(
            test_data, batch_size=args.batch_size, repeat=False, shuffle=False)
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
            'validation/main/map'
        ]),
        trigger=(100, 'iteration'))
    trainer.extend(extensions.ProgressBar(update_interval=200))
    trainer.extend(extensions.dump_graph('main/loss'))

    evaluator = InstanceSegmentationVOCEvaluator(
        test_iter, model.faster_rcnn, label_names=labels)
    trainer.extend(evaluator, trigger=(10000, 'iteration'))

    save_args(args, args.out)
    trainer.extend(CommandsExtension(), trigger=(100, 'iteration'))

    try:
        np.seterr(all='warn')
        trainer.run()
    except Warning:
        import pdb
        pdb.set_trace()


if __name__ == '__main__':
    main()
