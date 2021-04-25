import argparse
from .options import cfg
from .parser_utils import add_default_argument
from functools import reduce


class ArgParser(argparse.ArgumentParser):
    config_map = {k: section for section in cfg for k in cfg[section]}

    @classmethod
    def set_config_param(cls, param, value):
        section = cls.config_map[param]
        config_section = getattr(cfg, section)
        setattr(config_section, param, value)

    @classmethod
    def get_config_param(cls, param):
        section = cls.config_map[param]
        return reduce(getattr, [section, param], cfg)

    add_default_argument = add_default_argument

    def prepare_arguments(self):
        self.add_argument(
            dest="input_images",
            metavar="IMAGE_FILE_OR_FILES",
            nargs="+",
            help="One or more images to process",
        )
        self.add_default_argument("--debug-level", choices=[0, 1, 2, 3])
        self.add_default_argument("--debug-output", choices=["file", "screen", "both"])
        self.add_default_argument(["-p", "--pdf"], "CONVERT_TO_PDF")
        self.add_default_argument("--x-margin", "PAGE_MARGIN_X")
        self.add_default_argument("--y-margin", "PAGE_MARGIN_Y")
        self.add_default_argument("--min-text-width", "TEXT_MIN_WIDTH")
        self.add_default_argument("--min-text-height", "TEXT_MIN_HEIGHT")
        self.add_default_argument("--min-text-aspect", "TEXT_MIN_ASPECT")
        self.add_default_argument("--max-text-thickness", "TEXT_MAX_THICKNESS")
        self.add_default_argument("--adaptive-winsz")
        self.add_default_argument("--rotation-vec-param-idx", "RVEC_IDX")
        self.add_default_argument("--translation-vec-param-idx", "TVEC_IDX")
        self.add_default_argument("--cubic-slope-param-idx", "CUBIC_IDX")
        self.add_default_argument("--min-span-width", "SPAN_MIN_WIDTH")
        self.add_default_argument("--span-spacing", "SPAN_PX_PER_STEP")
        self.add_default_argument("--max-edge-overlap", "EDGE_MAX_OVERLAP")
        self.add_default_argument("--max-edge-length", "EDGE_MAX_LENGTH")
        self.add_default_argument("--edge-angle-cost")
        self.add_default_argument("--max-edge-angle", "EDGE_MAX_ANGLE")
        self.add_default_argument("--focal-length")
        self.add_default_argument("--output-zoom")
        self.add_default_argument("--output-dpi")
        self.add_default_argument("--downscale", "REMAP_DECIMATE")

    def __init__(self):
        super().__init__()
        # The config was read in already (`cfg` in `options.py`, which was imported)
        self.prepare_arguments() # First set up the parser to read runtime parameters
        self.parsed = self.parse_args()
        self.input_images = self.parsed.input_images
        self.store_parsed_config() # Store any supplied parameters in the global config

    def store_parsed_config(self):
        for opt in self.config_map:
            if opt == "K":
                continue  # K is based entirely on FOCAL_LENGTH
            # Redundant but thorough: any unchanged defaults will be reassigned
            configured_opt = getattr(self.parsed, opt)
            if configured_opt is not None:
                self.set_config_param(opt, configured_opt)
