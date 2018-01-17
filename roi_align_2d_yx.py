from roi_align.roi_align_2d import roi_align_2d

def _roi_align_2d_yx(x, indices_and_rois, outh, outw, spatial_scale):
    xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
    pool = roi_align_2d(
        x, xy_indices_and_rois, outh, outw, spatial_scale)
    return pool
