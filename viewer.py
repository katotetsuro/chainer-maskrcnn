import pyrealsense2 as rs
import numpy as np
import cv2

import chainer
from chainer import using_config
from chainer_maskrcnn.model.maskrcnn_resnet50 import MaskRCNNResnet50
import time

model = MaskRCNNResnet50(1, n_keypoints=20, n_mask_convs=2,
                         min_size=240, backbone='darknet', head_arch='fpn_keypoint')
chainer.serializers.load_npz(
    'result/depth_trained_model/fastdepth-model_40000.npz', model, strict=True)

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

config.enable_device_from_file('20180417_163559.bag')
# config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 30)
# config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

margin = 52
scale = 2
depth_offset = 1000
freeze = False

# Start streaming
pipeline.start(config)

try:
    def on_change_offset(v):
        print('new depth offset:{}'.format(v))
        depth_offset = v

    wname = 'depth'
    cv2.namedWindow(wname)
    # trackbar = cv2.createTrackbar(
    #     'depth_offset', wname, 500, 5000, on_change_offset)
    while True:
        start = time.time()
        # Wait for a coherent pair of frames: depth and color
        if not freeze:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(
            depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # D435が16:9でしかキャプチャできないので4:3にクロップする
        cropped_depth = depth_image[:, margin:-margin]
        # depth_datasetでやってしまった謎の変換
        cropped_depth = (cropped_depth.astype(
            np.float32) - depth_offset) / 4000 * 255
#        cropped_depth = cropped_depth[None]
        # 1-chに直したいなー
        cropped_depth = np.stack([cropped_depth, cropped_depth, cropped_depth])

        s = 56
        with using_config('train', False), using_config('enable_backprop', False), using_config('use_ideep', 'auto'):
            box, label, score, keypoints = model.predict(cropped_depth[None])
            if len(score[0]):
                indices = np.array(score[0]) > 0.9
                # import pdb
                # pdb.set_trace()
                kps = np.argmax(keypoints[0][0], axis=2)
                for b, kp in zip(box[0][indices], kps[indices]):
                    #sb = (b * scale).astype(np.int32)
                    sb = (b * 1).astype(np.int32)
                    cv2.rectangle(
                        depth_colormap, (sb[1] + margin, sb[0]), (sb[3] + margin, sb[2]), (0, 0, 255), 3)
                    sh, sw = (sb[2] - sb[0]) / s, (sb[3] - sb[1]) / s
                    for k in kp:
                        cv2.circle(
                            depth_colormap, (int(k % s * sw + sb[1] + margin), int(k // s * sh + sb[0])), 5, (255, 0, 0))

        # Show images
#        cv2.imshow('color', color_image)
        cv2.imshow(wname, depth_colormap)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print('quit')
            break
        if key == ord('s'):
            freeze = not freeze

        end = time.time()
#        print(1. / (end - start))

finally:
    # Stop streaming
    cv2.destroyAllWindows()
    pipeline.stop()
