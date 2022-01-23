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
from .options import cfg

__all__ = ["box", "Mask"]


def box(width, height):
    return np.ones((height, width), dtype=np.uint8)


class Mask:
    def __init__(self, name, small, pagemask, text=True):
        self.name = name
        self.small = small
        self.pagemask = pagemask
        self.text = text
        self.calculate()

    def calculate(self):
        sgray = cvtColor(self.small, COLOR_RGB2GRAY)
        mask = adaptiveThreshold(
            src=sgray,
            maxValue=255,
            adaptiveMethod=ADAPTIVE_THRESH_MEAN_C,
            thresholdType=THRESH_BINARY_INV,
            blockSize=cfg.mask_opts.ADAPTIVE_WINSZ,
            C=25 if self.text else 7,
        )
        self.log(0.1, "thresholded", mask)
        mask = (
            dilate(mask, box(9, 1))
            if self.text
            else erode(mask, box(3, 1), iterations=3)
        )
        self.log(0.2, "dilated" if self.text else "eroded", mask)
        mask = erode(mask, box(1, 3)) if self.text else dilate(mask, box(8, 2))
        self.log(0.3, "eroded" if self.text else "dilated", mask)
        self.value = np.minimum(mask, self.pagemask)

    def log(self, step, text, display):
        if cfg.debug_lvl_opt.DEBUG_LEVEL >= 3:
            if not self.text:
                step += 0.3  # text images from 0.1 to 0.3, table images from 0.4 to 0.6
            debug_show(self.name, step, text, display)

    def contours(self):
        return get_contours(self.name, self.small, self.value)
