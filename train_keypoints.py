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
from chainer_maskrcnn.dataset.coco_dataset import COCOKeypointsLoader
from chainer_maskrcnn.dataset.depth_dataset import DepthDataset
from chainer_maskrcnn.utils.depth_transformer import DepthTransformer

import argparse
from os.path import exists, isfile
import time
import _pickle as pickle


def calc_mask_loss(roi_cls_mask, gt_roi_mask, xp, gt_roi_label, num_keypoints=17):
    # 出力を (n_proposals, 17, mask_size, mask_size) から (n_positive_sample *17, mask_size*mask_size) にreshapeして、softmax crossentropyを取る
    num_positives = gt_roi_mask.shape[0]
    roi_mask = roi_cls_mask[:num_positives].reshape(
        (num_positives * num_keypoints, -1))
    gt_roi_mask = gt_roi_mask.reshape((-1,))
    return chainer.functions.softmax_cross_entropy(roi_mask, gt_roi_mask)


def load_dataset(dataset, file):
    if isfile(file):
        print('pklから読み込みます')
        dataload_start = time.time()
        with open(file, 'rb') as f:
            train_data = pickle.load(f)
        dataload_end = time.time()
        print('pklからの読み込み {}'.format(dataload_end - dataload_start))
    else:
        dataload_start = time.time()
        train_data = dataset()
        dataload_end = time.time()
        print('普通の読み込み {}'.format(dataload_end - dataload_start))
        if file is not '':
            print('次回のために保存します')
            with open(file, 'wb') as f:
                pickle.dump(train_data, f)
    return train_data


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
        label = np.zeros(bbox.shape[0], dtype=np.int32)
        # shape of keypoints is (N, 17, 3), N is number of bbox, 17 is number of keypoints, 3 is (x, y, v)
        # v=0: unlabeled, v=1, labeled but invisible, v=2 labeled and visible
        keypoints = keypoints.astype(np.float32)
        kp = keypoints[:, :, [1, 0]]
        kp = np.concatenate([kp * scale, keypoints[:, :, 2, None]], axis=2)

        return img, bbox, label, kp, scale


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
    parser.add_argument('--dataset', default='coco', choices=['coco', 'depth'])
    parser.add_argument('--n_mask_convs', type=int, default=None)
    parser.add_argument('--min_size', type=int, default=600)
    parser.add_argument('--max_size', type=int, default=1000)

    args = parser.parse_args()

    print('lr:{}'.format(args.lr))
    print('output:{}'.format(args.out))
    print('weight:{}'.format(args.weight))
    print('label file:{}'.format(args.label_file))
    print('iteration::{}'.format(args.iteration))
    print('backbone architecture:{}'.format(args.backbone))
    print('head architecture:{}'.format(args.head_arch))

    if args.dataset == 'coco':
        train_data = load_dataset(COCOKeypointsLoader, 'train_data_kp.pkl')
        n_keypoints = train_data.n_keypoints
    elif args.dataset == 'depth':
        train_data = load_dataset(
            lambda: DepthDataset(path='data/rgbd/train.txt', root='data/rgbd/'), '')
        n_keypoints = train_data.n_keypoints
        train_data = chainer.datasets.TransformDataset(
            train_data, DepthTransformer())
    print(f'number of keypoints={n_keypoints}')

    if args.multi_gpu:
        print('try to use chainer.training.updaters.MultiprocessParallelUpdater')
        if not chainer.training.updaters.MultiprocessParallelUpdater.available():
            print('MultiprocessParallelUpdater is not available')
            args.multi_gpu = 0

    faster_rcnn = MaskRCNNResnet50(
        n_fg_class=1, backbone=args.backbone, head_arch=args.head_arch,
        n_keypoints=n_keypoints, n_mask_convs=args.n_mask_convs, min_size=args.min_size, max_size=args.max_size)
    faster_rcnn.use_preset('evaluate')
    model = FPNMaskRCNNTrainChain(
        faster_rcnn, mask_loss_fun=lambda x, y, z, w: calc_mask_loss(x, y, z, w, num_keypoints=n_keypoints), binary_mask=False)
    if exists(args.weight):
        chainer.serializers.load_npz(
            args.weight, model.faster_rcnn, strict=False)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    optimizer = chainer.optimizers.MomentumSGD(lr=args.lr, momentum=0.9)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(rate=0.0005))

    # TransformでFaster-RCNNのprepareを参照するので、初期化順が複雑に入り組んでしまったなー
    train_data = TransformDataset(train_data, Transform(faster_rcnn))
    if args.multi_gpu:
        train_iters = [chainer.iterators.SerialIterator(
            train_data, 1, repeat=True, shuffle=True) for i in range(8)]
        updater = chainer.training.updater.MultiprocessParallelUpdater(
            train_iters, optimizer, device=range(8))

    else:
        train_iter = chainer.iterators.SerialIterator(
            train_data, batch_size=args.batch_size, repeat=True, shuffle=True)
        updater = chainer.training.updater.StandardUpdater(
            train_iter, optimizer, device=args.gpu)

    trainer = chainer.training.Trainer(updater, (args.iteration, 'iteration'),
                                       args.out)

    trainer.extend(
        extensions.snapshot_object(model.faster_rcnn,
                                   'model_{.updater.iteration}.npz'),
        trigger=(20000, 'iteration'))

    trainer.extend(
        extensions.ExponentialShift('lr', 0.1), trigger=(3, 'epoch'))

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
