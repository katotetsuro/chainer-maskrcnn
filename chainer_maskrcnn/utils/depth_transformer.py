import numpy as np


class DepthTransformer():
    def __call__(self, in_data):
        x, bbox, keypoint = in_data

        x += (np.random.rand(1).astype(np.float32) - 0.5) * 30

        return x, bbox, keypoint
