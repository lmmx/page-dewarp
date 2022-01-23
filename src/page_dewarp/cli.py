import argparse
from .options import cfg, Config
from .parser_utils import add_default_argument
from functools import reduce


class ArgParser(argparse.ArgumentParser):
    config_map = {k: section for section in cfg for k in cfg[section]}

    @classmethod
    def config_comments(cls):
        config_map = cls.config_map
        default_toml = Config.parse_defaults_with_comments()
        comments_dict = {}
        for k, section in config_map.items():
            value = default_toml[section][k]
            if hasattr(value, "trivia"):
                comment = value.trivia.comment.lstrip("# ")
                if comment == "":
                    comment = None
            else:
                comment = None
            comments_dict.update({k: comment})
        return comments_dict

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
        self.add_default_argument(["-d", "--debug-level"], choices=[0, 1, 2, 3])
        self.add_default_argument(
            ["-o", "--debug-output"], choices=["file", "screen", "both"]
        )
        self.add_default_argument(
            ["-p", "--pdf"], "CONVERT_TO_PDF", help="Merge dewarped images into a PDF"
        )
        self.add_default_argument(["-vw", "--max-screen-width"], "SCREEN_MAX_W")
        self.add_default_argument(["-vh", "--max-screen-height"], "SCREEN_MAX_H")
        self.add_default_argument(["-x", "--x-margin"], "PAGE_MARGIN_X")
        self.add_default_argument(["-y", "--y-margin"], "PAGE_MARGIN_Y")
        self.add_default_argument(["-tw", "--min-text-width"], "TEXT_MIN_WIDTH")
        self.add_default_argument(["-th", "--min-text-height"], "TEXT_MIN_HEIGHT")
        self.add_default_argument(["-ta", "--min-text-aspect"], "TEXT_MIN_ASPECT")
        self.add_default_argument(["-tk", "--max-text-thickness"], "TEXT_MAX_THICKNESS")
        self.add_default_argument(["-wz", "--adaptive-winsz"])
        self.add_default_argument(["-ri", "--rotation-vec-param-idx"], "RVEC_IDX")
        self.add_default_argument(["-ti", "--translation-vec-param-idx"], "TVEC_IDX")
        self.add_default_argument(["-ci", "--cubic-slope-param-idx"], "CUBIC_IDX")
        self.add_default_argument(["-sw", "--min-span-width"], "SPAN_MIN_WIDTH")
        self.add_default_argument(["-sp", "--span-spacing"], "SPAN_PX_PER_STEP")
        self.add_default_argument(["-eo", "--max-edge-overlap"], "EDGE_MAX_OVERLAP")
        self.add_default_argument(["-el", "--max-edge-length"], "EDGE_MAX_LENGTH")
        self.add_default_argument(["-ec", "--edge-angle-cost"])
        self.add_default_argument(["-ea", "--max-edge-angle"], "EDGE_MAX_ANGLE")
        self.add_default_argument(["-f", "--focal-length"])
        self.add_default_argument(["-z", "--output-zoom"])
        self.add_default_argument(["-dpi", "--output-dpi"])
        self.add_default_argument(["-s", "--shrink"], "REMAP_DECIMATE")

    def __init__(self):
        super().__init__()
        # The config was read in already (`cfg` in `options.py`, which was imported)
        self.prepare_arguments()  # First set up the parser to read runtime parameters
        self.parsed = self.parse_args()
        self.input_images = self.parsed.input_images
        self.store_parsed_config()  # Store any supplied parameters in the global config
        self.config_comments()

    def store_parsed_config(self):
        for opt in self.config_map:
            # Redundant but thorough: any unchanged defaults will be reassigned
            configured_opt = getattr(self.parsed, opt)
            if configured_opt is not None:
                self.set_config_param(opt, configured_opt)
