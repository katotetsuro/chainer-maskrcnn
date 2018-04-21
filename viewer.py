import argparse
import time

import pyrealsense2 as rs
import numpy as np
import cv2

import chainer
from chainer import using_config
from chainer_maskrcnn.model.maskrcnn_resnet50 import MaskRCNNResnet50

import vis


class SimpleInfer:
    def __init__(self, weight, file=None):
        self.model = MaskRCNNResnet50(1, n_keypoints=20, n_mask_convs=2,
                                      min_size=240, backbone='darknet', head_arch='fpn_keypoint')
        chainer.serializers.load_npz(weight, self.model, strict=True)
        self.in_channels = self.model.extractor.conv1.c.W.shape[1]
        print('number of parameters:{}'.format(
            sum(p.data.size for p in self.model.params())))
        if hasattr(self.model, 'to_intel64'):
            self.model.to_intel64()

        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        if file is not None:
            print('load from {}'.format(file))
            self.config.enable_device_from_file(file)
        else:
            print('launch device')
            self.config.enable_stream(
                rs.stream.depth, 424, 240, rs.format.z16, 30)
            self.config.enable_stream(
                rs.stream.color, 640, 480, rs.format.bgr8, 30)

        self.margin = 52
        self.depth_offset = 0
        self.wname = 'depth'
        self.avg_fps = 15.0

    def run(self):
        # Start streaming
        self.pipeline.start(self.config)
        try:
            while True:
                self.main_loop()
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print('quit')
                    break

        finally:
            # Stop streaming
            self.pipeline.stop()

    def main_loop(self):
        start = time.time()
        # Wait for a coherent pair of frames: depth and color
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            return

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(
            depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # D435が16:9でしかキャプチャできないので4:3にクロップする
        cropped_depth = depth_image[:, self.margin:-self.margin]
        # depth_datasetでやってしまった謎の変換
        cropped_depth = np.clip(cropped_depth.astype(
            np.float32) - self.depth_offset, 0, 4000) / 3000 * 255

        if self.in_channels == 1:
            cropped_depth = cropped_depth[None]
        else:
            cropped_depth = np.stack(
                [cropped_depth, cropped_depth, cropped_depth])

        s = 56
        with using_config('train', False), using_config('enable_backprop', False), using_config('use_ideep', 'auto'):
            box, label, score, keypoints = self.model.predict(
                cropped_depth[None])

            if len(box[0]):
                kps = np.argmax(keypoints[0][0], axis=2)
                indices = np.argsort(keypoints[0][0], axis=2)
                kp_logits = keypoints[0][0][0,
                                            np.arange(20), indices[:, :, -1]]
                kp_probs = chainer.functions.softmax(
                    keypoints[0][0]).array[0, np.arange(20), indices[:, :, -1]]
                vis_keys = []
                for kp, b, logits, probs in zip(kps, box[0], kp_logits, kp_probs):
                    sh, sw = (b[2] - b[0]) / s, (b[3] - b[1]) / s
                    for i, (k, l, p) in enumerate(zip(kp, logits, probs)):
                        vis_keys.append([k // s * sh + b[0], k %
                                         s * sw + b[1] + self.margin, l, p])

                vis_keys = np.array(vis_keys)
                depth_colormap = vis.vis_keypoints(
                    depth_colormap.transpose((1, 0, 2)).copy(), vis_keys.transpose((1, 0)), kp_thresh=3)
                depth_colormap = depth_colormap.transpose((1, 0, 2))

        # Show images
    #        cv2.imshow('color', color_image)
        cv2.imshow(self.wname, depth_colormap)
        end = time.time()
        self.avg_fps = self.avg_fps * 0.9 + 1. / (end - start) * 0.1
        print(self.avg_fps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', help='bag file')
    parser.add_argument('--weight', help='pretrained_weight')
    args = parser.parse_args()
    SimpleInfer(weight=args.weight, file=args.file).run()
