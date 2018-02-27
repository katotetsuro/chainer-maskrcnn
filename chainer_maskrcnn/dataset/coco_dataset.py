import numpy as np
import os
from os.path import join
import chainer
from chainercv.utils import read_image
from pycocotools.coco import COCO
from PIL import Image
from random import shuffle


class COCOMaskLoader(chainer.dataset.DatasetMixin):
    def __init__(self,
                 anno_dir='data/annotations',
                 img_dir='data',
                 split='train',
                 data_type='2014',
                 category_filter=None):
        if split not in ['train', 'val', 'validation']:
            raise ValueError(
                'please pick split from \'train\', \'val\',\'validation\'')

        if split == 'validation':
            split = 'val'

        ann_file = '%s/instances_%s%s.json' % (anno_dir, split, data_type)
        self.coco = COCO(ann_file)

        self.img_dir = '{}/{}{}'.format(img_dir, split, data_type)
        print('load jpg images from {}'.format(self.img_dir))
        target_cats = [] if category_filter is None else category_filter
        self.cat_ids = self.coco.getCatIds(catNms=target_cats)
        # cat_idsの中のどれかが含まれる画像、を探したい（or検索）
        # getImgIdsの引数にしていするとand検索されるので、泥草する
        img_ids = set()
        for cat_id in self.cat_ids:
            img_ids |= set(self.coco.getImgIds(catIds=[cat_id]))

        print('before filter: {}'.format(len(img_ids)))
        #        img_ids = list(filter(lambda x: self._contain_large_enough_annotation(x), img_ids))
        img_ids = list(
            filter(lambda x: self._contain_large_annotation_only(x), img_ids))
        print('after filter: {}'.format(len(img_ids)))

#        shuffle(img_ids)
        self.img_infos = [(i['file_name'], i['id'])
                          for i in self.coco.loadImgs(img_ids)]
        # len(self.imgs)を呼ぶたびにメモリ使用量が増えるという罠?
        self.length = len(self.img_infos)

    def __len__(self):
        return self.length

    # 少なくとも１つ十分大きいものがあればOKとするフィルタ
    def _contain_large_enough_annotation(self, img_id, min_w=10, min_h=10):
        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
        for ann in anns:
            x, y, w, h = [int(j) for j in ann['bbox']]
            if w <= min_w or h <= min_h:
                continue

            if ann['category_id'] in self.cat_ids:
                return True

        return False

    # 画像内の全てのアノテーションが大きくないとダメだぞというフィルタ
    def _contain_large_annotation_only(self, img_id, min_w=10, min_h=10):
        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
        for ann in anns:
            x, y, w, h = [int(j) for j in ann['bbox']]
            if ann['category_id'] in self.cat_ids and (w <= min_w
                                                       or h <= min_h):
                return False

        return True

    def get_example(self, i):
        file_name, img_id = self.img_infos[i]
        img = read_image(join(self.img_dir, file_name), color=True)
        assert img.shape[0] == 3
        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
        gt_boxes = []
        gt_masks = []
        gt_labels = []
        for ann in anns:
            x, y, w, h = [int(j) for j in ann['bbox']]
            if w <= 10:
                #                print('w<=10が検出されました {} {} このアノテーションは無視します'.format(file_name, w))
                continue
            if h <= 10:
                #                print('h<=10が検出されました {} {} このアノテーションは無視します'.format(file_name, h))
                continue

            # これめっちゃ罠で、category_idが連続してないんだよなー
            if ann['category_id'] in self.cat_ids:
                continuous_cat_id = self.cat_ids.index(ann['category_id'])
                gt_boxes.append(
                    np.array([y, x, y + h, x + w], dtype=np.float32))
                gt_masks.append(self.coco.annToMask(ann))
                gt_labels.append(continuous_cat_id)

        if len(gt_boxes) == 0:
            print(
                '小さすぎるアノテーションを削除した結果、この画像にはground_truthが一つも含まれませんでした。これで学習にエラーが出る場合、事前に小さなアノテーションしか含まれない画像はself.imgsから削除することを検討してください'
            )

        # gt_masksはlistのままにしておいて、Transformでcv::resizeしたあとにnumpy arrayにするという泥臭
        return img, np.array(gt_boxes), np.array(
            gt_labels, dtype=np.int32), gt_masks


class COCOKeypointsLoader(chainer.dataset.DatasetMixin):
    def __init__(self,
                 anno_dir='data/annotations',
                 img_dir='data',
                 split='train',
                 data_type='2014'):
        if split not in ['train', 'val', 'validation']:
            raise ValueError(
                'please pick split from \'train\', \'val\',\'validation\'')

        if split == 'validation':
            split = 'val'

        ann_file = '%s/person_keypoints_%s%s.json' % (
            anno_dir, split, data_type)
        self.coco = COCO(ann_file)

        self.img_dir = '{}/{}{}'.format(img_dir, split, data_type)
        print('load jpg images from {}'.format(self.img_dir))
        img_ids = self.coco.getImgIds(catIds=[1])  # person only
        self.img_infos = list()
        for info in img_infos:
            file_name, img_id = info
            anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
            if len(anns) > 0:
                self.img_infos.append(info)

        self.length = len(self.img_infos)
        print('number of valid data.', self.length)

    def __len__(self):
        return self.length

    def get_example(self, i):
        if i >= self.length:
            raise IndexError()

        file_name, img_id = self.img_infos[i]
        img = read_image(join(self.img_dir, file_name), color=True)
        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))

        keypoints = list()
        for ann in anns:
            kp = ann['keypoints']
            kp = np.array(kp).reshape((-1, 3))
            keypoints.append(kp)
        keypoints = np.concatenate(keypoints)

        return img, keypoints
