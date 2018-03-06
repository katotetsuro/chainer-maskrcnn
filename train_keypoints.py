import chainer
from chainer.datasets import TransformDataset
from chainer.training import extensions
from chainercv import transforms
from chainerui.utils import save_args
from chainerui.extensions import CommandsExtension
import cv2
import numpy as np
from chainer_maskrcnn.model.fpn_keypoint_maskrcnn_train_chain import FPNKeypointMaskRCNNTrainChain
from chainer_maskrcnn.model.keypoint_maskrcnn_resnet50 import MaskRCNNResnet50
from chainer_maskrcnn.dataset.coco_dataset import COCOKeypointsLoader

import argparse
from os.path import exists, isfile
import time
import pickle


class Transform():
    def __init__(self, faster_rcnn):
        self.faster_rcnn = faster_rcnn

    def __call__(self, in_data):
        img, bbox, keypoints = in_data
        _, H, W = img.shape
        img = self.faster_rcnn.prepare(img)
        _, o_H, o_W = img.shape
        scale = o_H / H

        bbox = transforms.resize_bbox(bbox, (H, W), (o_H, o_W))
        # shape of keypoints is (N, 17, 3), N is number of bbox, 17 is number of keypoints, 3 is (x, y, v)
        # v=0: unlabeled, v=1, labeled but invisible, v=2 labeled and visible
        keypoints = keypoints.astype(np.float32)
        kp = keypoints[:, :, [1, 0]]
        kp = np.concatenate([kp * scale, keypoints[:, :, 2, None]], axis=2)

        return img, bbox, kp, scale


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
    parser.add_argument('--head_arch', '-a', type=str, default='fpn_keypoint')
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

    faster_rcnn = MaskRCNNResnet50(
        n_fg_class=1, backbone=args.backbone, head_arch=args.head_arch)
    faster_rcnn.use_preset('evaluate')
    #model = MaskRCNNTrainChain(faster_rcnn)
    model = FPNKeypointMaskRCNNTrainChain(faster_rcnn)
    if exists(args.weight):
        chainer.serializers.load_npz(
            args.weight, model.faster_rcnn, strict=False)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    optimizer = chainer.optimizers.MomentumSGD(lr=args.lr, momentum=0.9)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(rate=0.0005))

    pkl_file = 'train_data_kp.pkl'
    if isfile(pkl_file):
        print('pklから読み込みます')
        dataload_start = time.time()
        with open(pkl_file, 'rb') as f:
            coco_train_data = pickle.load(f)
        dataload_end = time.time()
        print('pklからの読み込み {}'.format(dataload_end - dataload_start))
    else:
        dataload_start = time.time()
        coco_train_data = COCOKeypointsLoader()
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
        trigger=(20000, 'iteration'))

    trainer.extend(
        extensions.ExponentialShift('lr', 0.1), trigger=(1, 'epoch'))

    log_interval = 100, 'iteration'
    print_interval = 100, 'iteration'

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
        trigger=print_interval)
    trainer.extend(extensions.ProgressBar(update_interval=200))
    trainer.extend(extensions.dump_graph('main/loss'))

    save_args(args, args.out)
    trainer.extend(CommandsExtension(), trigger=(100, 'iteration'))

    trainer.run()


if __name__ == '__main__':
    main()
