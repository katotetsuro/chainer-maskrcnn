import numpy as np
import cv2
import matplotlib.pyplot as plt


def get_keypoints():
    keypoints = [
        'SpineBase',
        'SpineMid',
        'Neck',
        'Head',
        'ShoulderLeft',
        'ElbowLeft',
        'WristLeft',
        'HandLeft',
        'ShoulderRight',
        'ElbowRight',
        'WristRight',
        'HandRight',
        'HipLeft',
        'KneeLeft',
        'AnkleLeft',
        'FootLeft',
        'HipRight',
        'KneeRight',
        'AnkleRight',
        'FootRight'
    ]
    keypoint_flip_map = {
        'ShoulderLeft': 'ShoulderRight',
        'ElbowLeft': 'ElbowRight',
        'WristLeft': 'WristRight',
        'HipLeft': 'HipRight',
        'KneeLeft': 'KneeRight',
        'FootLeft': 'FootRight'
    }
    return keypoints, keypoint_flip_map


def kp_connections(keypoints):
    kp_lines = [
        [keypoints.index('ShoulderRight'), keypoints.index('ElbowRight')],
        [keypoints.index('ElbowRight'), keypoints.index('WristRight')],
        [keypoints.index('ShoulderLeft'), keypoints.index('ElbowLeft')],
        [keypoints.index('ElbowLeft'), keypoints.index('WristLeft')],
        [keypoints.index('HipRight'), keypoints.index('KneeRight')],
        [keypoints.index('KneeRight'), keypoints.index('AnkleRight')],
        [keypoints.index('HipLeft'), keypoints.index('KneeLeft')],
        [keypoints.index('KneeLeft'), keypoints.index('AnkleLeft')],
        [keypoints.index('ShoulderRight'), keypoints.index('Neck')],
        [keypoints.index('Neck'), keypoints.index('ShoulderLeft')],
        [keypoints.index('Neck'), keypoints.index('Head')],
        [keypoints.index('Neck'), keypoints.index('SpineBase')],
        [keypoints.index('SpineBase'), keypoints.index('HipRight')],
        [keypoints.index('SpineBase'), keypoints.index('HipLeft')]
    ]
    return kp_lines


def vis_keypoints(img, kps, kp_thresh=2, alpha=0.7):
    """Visualizes keypoints (adapted from vis_one_image).
    kps has shape (4, #keypoints) where 4 rows are (x, y, logit, prob).
    """
    dataset_keypoints, _ = get_keypoints()
    kp_lines = kp_connections(dataset_keypoints)

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kp_lines) + 2)]
    colors = [(int(c[2] * 255), int(c[1] * 255), int(c[0] * 255))
              for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw mid shoulder / mid hip first for better visualization.
    mid_shoulder = (
        kps[:2, dataset_keypoints.index('ShoulderRight')] +
        kps[:2, dataset_keypoints.index('ShoulderLeft')]) / 2.0
    sc_mid_shoulder = np.minimum(
        kps[2, dataset_keypoints.index('ShoulderRight')],
        kps[2, dataset_keypoints.index('ShoulderLeft')])
    mid_hip = (
        kps[:2, dataset_keypoints.index('HipRight')] +
        kps[:2, dataset_keypoints.index('HipLeft')]) / 2.0
    sc_mid_hip = np.minimum(
        kps[2, dataset_keypoints.index('HipRight')],
        kps[2, dataset_keypoints.index('HipLeft')])

    if sc_mid_shoulder > kp_thresh and sc_mid_hip > kp_thresh:
        cv2.line(
            kp_mask, tuple(mid_shoulder.astype(np.int32)), tuple(
                mid_hip.astype(np.int32)),
            color=colors[len(kp_lines) + 1], thickness=2, lineType=cv2.LINE_AA)

    # Draw the keypoints.
    for l in range(len(kp_lines)):
        i1 = kp_lines[l][0]
        i2 = kp_lines[l][1]
        p1 = kps[0, i1].astype(np.int32), kps[1, i1].astype(np.int32)
        p2 = kps[0, i2].astype(np.int32), kps[1, i2].astype(np.int32)
        if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
            cv2.line(
                kp_mask, p1, p2,
                color=colors[l], thickness=2, lineType=cv2.LINE_AA)
        if kps[2, i1] > kp_thresh:
            cv2.circle(
                kp_mask, p1,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
        if kps[2, i2] > kp_thresh:
            cv2.circle(
                kp_mask, p2,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)
