import pyrealsense2 as rs
import numpy as np
import cv2

import chainer
from chainer import using_config
from chainer_maskrcnn.model.maskrcnn_resnet50 import MaskRCNNResnet50
import time
import vis

model = MaskRCNNResnet50(1, n_keypoints=20, n_mask_convs=2,
                         min_size=240, backbone='darknet', head_arch='fpn_keypoint')

if hasattr(model, 'to_intel64'):
    model.to_intel64()
chainer.serializers.load_npz(
    'result/depth_trained_model/fastdepth-model_40000.npz', model, strict=True)

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

config.enable_device_from_file('20180419_121047.bag')
#config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 30)
#config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

margin = 52
scale = 2
depth_offset = 0
freeze = False
wname = 'depth'

# Start streaming
pipeline.start(config)


def main_loop():
    start = time.time()
    # Wait for a coherent pair of frames: depth and color
    if not freeze:
        frames = pipeline.wait_for_frames()
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
    cropped_depth = depth_image[:, margin:-margin]
    # depth_datasetでやってしまった謎の変換
    cropped_depth = np.clip(cropped_depth.astype(
        np.float32) - depth_offset, 0, 4000) / 3000 * 255

#        cropped_depth = cropped_depth[None]
    # 1-chに直したいなー
    cropped_depth = np.stack([cropped_depth, cropped_depth, cropped_depth])

    s = 56
    with using_config('train', False), using_config('enable_backprop', False), using_config('use_ideep', 'auto'):
        box, label, score, keypoints = model.predict(cropped_depth[None])

        if len(box[0]):
            kps = np.argmax(keypoints[0][0], axis=2)
            indices = np.argsort(keypoints[0][0], axis=2)
            kp_logits = keypoints[0][0][0,
                                        np.arange(20), indices[:, :, -1]]
            kp_probs = chainer.functions.softmax(
                keypoints[0][0]).array[0, np.arange(20), indices[:, :, -1]]
            s = 56
            vis_keys = []
            for kp, b, logits, probs in zip(kps, box[0], kp_logits, kp_probs):
                sh, sw = (b[2] - b[0]) / s, (b[3] - b[1]) / s
                for i, (k, l, p) in enumerate(zip(kp, logits, probs)):
                    vis_keys.append([k // s * sh + b[0], k %
                                     s * sw + b[1] + margin, l, p])

            vis_keys = np.array(vis_keys)
            depth_colormap = vis.vis_keypoints(
                depth_colormap.transpose((1, 0, 2)).copy(), vis_keys.transpose((1, 0)), kp_thresh=3)
            depth_colormap = depth_colormap.transpose((1, 0, 2))

    # Show images
#        cv2.imshow('color', color_image)
    cv2.imshow(wname, depth_colormap)
    end = time.time()
    print(1. / (end - start))


def main():
    try:
        cv2.namedWindow(wname)
        # trackbar = cv2.createTrackbar(
        #     'depth_offset', wname, 500, 5000, on_change_offset)
        while True:
            main_loop()
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print('quit')
                break
            if key == ord('s'):
                freeze = not freeze

    finally:
        # Stop streaming
        cv2.destroyAllWindows()
        pipeline.stop()


if __name__ == '__main__':
    main()
