import numpy as np

__all__ = [
    "debug_lvl_opt",
    "image_opts",
    "contour_opts",
    "mask_opts",
    "proj_opts",
    "span_opts",
    "edge_opts",
    "camera_opts",
    "param_opts",
    "output_opts",
    "OptionsMixIn",
]

debug_lvl_opt = {
    "DEBUG_LEVEL": 0,
}

debug_out_opt = {
    "DEBUG_OUTPUT": "file",
}

image_opts = {
    **debug_lvl_opt,
    "PAGE_MARGIN_X": 50,  # reduced px to ignore near L/R edge
    "PAGE_MARGIN_Y": 20,  # reduced px to ignore near T/B edge
}


contour_opts = {
    **debug_lvl_opt,
    "TEXT_MIN_WIDTH": 15,  # min reduced px width of detected text contour
    "TEXT_MIN_HEIGHT": 2,  # min reduced px height of detected text contour
    "TEXT_MIN_ASPECT": 1.5,  # filter out text contours below this w/h ratio
    "TEXT_MAX_THICKNESS": 10,  # max reduced px thickness of detected text contour
}

mask_opts = {
    "ADAPTIVE_WINSZ": 55,  # window size for adaptive threshold in reduced px
    **contour_opts,
    **debug_lvl_opt,
}

proj_opts = {
    "RVEC_IDX": slice(0, 3),  # index of rvec in params vector
    "TVEC_IDX": slice(3, 6),  # index of tvec in params vector
    "CUBIC_IDX": slice(6, 8),  # index of cubic slopes in params vector
}

span_opts = {
    **debug_lvl_opt,
    "SPAN_MIN_WIDTH": 30,  # minimum reduced px width for span
    "SPAN_PX_PER_STEP": 20,  # reduced px spacing for sampling along spans
}

edge_opts = {
    "EDGE_MAX_OVERLAP": 1.0,  # max reduced px horiz. overlap of contours in span
    "EDGE_MAX_LENGTH": 100.0,  # max reduced px length of edge connecting contours
    "EDGE_ANGLE_COST": 10.0,  # cost of angles in edges (tradeoff vs. length)
    "EDGE_MAX_ANGLE": 7.5,  # maximum change in angle allowed between contours
}

camera_opts = {"FOCAL_LENGTH": 1.2}  # normalized focal length of camera

param_opts = {
    "K": np.array(
        [
            [camera_opts["FOCAL_LENGTH"], 0, 0],
            [0, camera_opts["FOCAL_LENGTH"], 0],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )  # default intrinsic parameter matrix
}

output_opts = {
    "OUTPUT_ZOOM": 1.0,  # how much to zoom output relative to *original* image
    "OUTPUT_DPI": 300,  # just affects stated DPI of PNG, not appearance
    "REMAP_DECIMATE": 16,  # downscaling factor for remapping image
}

class OptionsMixIn:
    def opt(self, option_name):
        return self.opts[option_name]
