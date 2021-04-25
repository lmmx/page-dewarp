from PIL import Image
import numpy as np
from cv2 import (
    resize,
    remap,
    INTER_CUBIC,
    INTER_AREA,
    cvtColor,
    COLOR_RGB2GRAY,
    BORDER_REPLICATE,
    adaptiveThreshold,
    ADAPTIVE_THRESH_MEAN_C,
    THRESH_BINARY,
)
from .debug_utils import debug_show
from .normalisation import norm2pix
from .options import cfg
from .projection import project_xy

__all__ = ["RemappedImage"]


def round_nearest_multiple(i, factor):
    i = int(i)
    rem = i % factor
    return i + factor - rem if rem else i


class RemappedImage:
    def __init__(self, name, img, small, page_dims, params):
        height = 0.5 * page_dims[1] * cfg.output_opts.OUTPUT_ZOOM * img.shape[0]
        height = round_nearest_multiple(height, cfg.output_opts.REMAP_DECIMATE)
        width = round_nearest_multiple(
            height * page_dims[0] / page_dims[1], cfg.output_opts.REMAP_DECIMATE
        )
        print("  output will be {}x{}".format(width, height))
        height_small, width_small = np.floor_divide(
            [height, width], cfg.output_opts.REMAP_DECIMATE
        )
        page_x_range = np.linspace(0, page_dims[0], width_small)
        page_y_range = np.linspace(0, page_dims[1], height_small)
        page_x_coords, page_y_coords = np.meshgrid(page_x_range, page_y_range)
        page_xy_coords = np.hstack(
            (
                page_x_coords.flatten().reshape((-1, 1)),
                page_y_coords.flatten().reshape((-1, 1)),
            )
        )
        page_xy_coords = page_xy_coords.astype(np.float32)
        image_points = project_xy(page_xy_coords, params)
        image_points = norm2pix(img.shape, image_points, False)
        image_x_coords = image_points[:, 0, 0].reshape(page_x_coords.shape)
        image_y_coords = image_points[:, 0, 1].reshape(page_y_coords.shape)
        image_x_coords = resize(
            image_x_coords, (width, height), interpolation=INTER_CUBIC
        )
        image_y_coords = resize(
            image_y_coords, (width, height), interpolation=INTER_CUBIC
        )
        img_gray = cvtColor(img, COLOR_RGB2GRAY)
        remapped = remap(
            img_gray,
            image_x_coords,
            image_y_coords,
            INTER_CUBIC,
            None,
            BORDER_REPLICATE,
        )
        thresh = adaptiveThreshold(
            remapped,
            255,
            ADAPTIVE_THRESH_MEAN_C,
            THRESH_BINARY,
            cfg.mask_opts.ADAPTIVE_WINSZ,
            25,
        )
        pil_image = Image.fromarray(thresh)
        pil_image = pil_image.convert("1")
        self.threshfile = name + "_thresh.png"
        pil_image.save(
            self.threshfile,
            dpi=(cfg.output_opts.OUTPUT_DPI, cfg.output_opts.OUTPUT_DPI),
        )
        if cfg.debug_lvl_opt.DEBUG_LEVEL >= 1:
            height = small.shape[0]
            width = int(round(height * float(thresh.shape[1]) / thresh.shape[0]))
            display = resize(thresh, (width, height), interpolation=INTER_AREA)
            debug_show(name, 6, "output", display)
