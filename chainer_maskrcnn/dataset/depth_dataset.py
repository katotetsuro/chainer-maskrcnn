from pathlib import Path
import chainer
import numpy as np
import cv2


class DepthDataset(chainer.dataset.DatasetMixin):
    n_keypoints = 20

    def __init__(self, path, root='.'):
        super().__init__()
        with open(path, 'r') as f:
            self.data = list(map(lambda x: x.strip(), f.readlines()))
        self.root = root

    def __len__(self):
        return len(self.data)

    def get_example(self, index):
        if index >= self.__len__():
            raise IndexError

        with np.load(Path(self.root).joinpath(self.data[index])) as f:
            img = f['depth']
            keypoints = f['keypoints']

        if keypoints.shape[1] == 2:
            visible = np.zeros((len(keypoints))).reshape((-1, 1))
            visible.fill(2)
            keypoints = np.concatenate((keypoints, visible), axis=1)
        else:
            keypoints[:, 2] = (keypoints[:, 2] > 0.2) * 2

        assert keypoints.shape[1] == 3

        if len(keypoints) > 20:
            print(f'複数人が写っています {self.data[index]}')

        # compute bounding box
        x0 = np.min(keypoints[:, :2], axis=0) - [10, 10]
        x1 = np.max(keypoints[:, :2], axis=0) + [10, 0]
        bbox = np.concatenate([x0, x1]).reshape((1, 4))

        # (number of box, numberof keypoints, (y,x,visibility))
        keypoints = keypoints[None]

        # make number of channels 3
        # なんとなく[0,255]くらいのfloatの配列にしておく
        # FasterRCNNのprepareメソッドで /255されるという複雑さ
        img = img.astype(np.float32) / 4000 * 255
        img = np.stack([img, img, img])

        return img, bbox, keypoints
