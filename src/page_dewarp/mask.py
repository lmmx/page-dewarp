import numpy as np
from cv2 import (
    adaptiveThreshold,
    ADAPTIVE_THRESH_MEAN_C,
    THRESH_BINARY_INV,
    cvtColor,
    COLOR_RGB2GRAY,
    dilate,
    erode,
)
from .debug_utils import debug_show
from .contours import get_contours
from .options import mask_opts, contour_opts, OptionsMixIn

__all__ = ["Mask", "make_tight_mask"]


def box(width, height):
    return np.ones((height, width), dtype=np.uint8)


class Mask(OptionsMixIn):
    opts = mask_opts
    contour_opts = contour_opts

    def __init__(self, name, small, pagemask, masktype, opts_dict):
        self.name = name
        self.small = small
        self.pagemask = pagemask
        self.masktype = masktype
        if opts_dict:
            self.opts.update(opts_dict)
        self.calculate()
    
    def calculate(self):
        sgray = cvtColor(self.small, COLOR_RGB2GRAY)
        if self.masktype == "text":
            mask = adaptiveThreshold(
                sgray,
                255,
                ADAPTIVE_THRESH_MEAN_C,
                THRESH_BINARY_INV,
                self.opt("ADAPTIVE_WINSZ"),
                25,
            )
            if self.opt("DEBUG_LEVEL") >= 3:
                debug_show(self.name, 0.1, "thresholded", mask)
            mask = dilate(mask, box(9, 1))
            if self.opt("DEBUG_LEVEL") >= 3:
                debug_show(self.name, 0.2, "dilated", mask)
            mask = erode(mask, box(1, 3))
            if self.opt("DEBUG_LEVEL") >= 3:
                debug_show(self.name, 0.3, "eroded", mask)
        else:
            mask = adaptiveThreshold(
                sgray,
                255,
                ADAPTIVE_THRESH_MEAN_C,
                THRESH_BINARY_INV,
                self.opt("ADAPTIVE_WINSZ"),
                7,
            )
            if DEBUG_LEVEL >= 3:
                debug_show(self.name, 0.4, "thresholded", mask)
            mask = erode(mask, box(3, 1), iterations=3)
            if DEBUG_LEVEL >= 3:
                debug_show(self.name, 0.5, "eroded", mask)
            mask = dilate(mask, box(8, 2))
            if DEBUG_LEVEL >= 3:
                debug_show(self.name, 0.6, "dilated", mask)
        self.value = np.minimum(mask, self.pagemask)

    def contours(self):
        return get_contours(self.name, self.small, self.value, self.contour_opts)
