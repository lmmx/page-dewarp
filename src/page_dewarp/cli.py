import argparse
from typing import Annotated, get_args, get_origin, get_type_hints

import msgspec

from .options import Config, cfg
from .snoopy import snoop

__all__ = ["ArgParser"]


def stringify_hint(type_hint) -> str:
    match type_hint:
        case type():
            hint = type_hint.__name__
        case _:
            hint = str(type_hint)
    return hint


class ArgParser(argparse.ArgumentParser):
    config_map = msgspec.structs.asdict(cfg)

    def add_default_argument(
        self,
        name_or_flags,
        arg_name=None,
        action=None,
        help=None,
        const=None,
        choices=None,
        nargs=None,
        metavar=None,
        required=None,
    ):
        kwargs = {
            "action": action,
            "nargs": nargs,
            "const": const,
            "choices": choices,
            "help": help,
            "metavar": metavar,
            "required": required,
        }
        if arg_name is None:
            title = name_or_flags if isinstance(name_or_flags, str) else name_or_flags[1]
            arg_name = title.lstrip("-").upper().replace("-", "_")
        default_value = self.get_config_param(arg_name)
        default_type = type(default_value)
        if default_type is bool and action is const is None:
            kwargs["action"] = f"store_{'false' if default_value else 'true'}"
        else:
            kwargs["type"] = default_type
        if isinstance(name_or_flags, str):
            name_or_flags = [name_or_flags]  # will be star-expanded so must be sequence
        if kwargs["help"] is None:
            kwargs["help"] = self.get_description(arg_name)
            # comment in TOML as a string if present, or `None` if absent
            # kwargs["help"] = self.config_comments().get(arg_name)
        self.add_argument(
            *name_or_flags,  # one or two values
            dest=arg_name,
            default=default_value,
            **{kw: v for kw, v in kwargs.items() if v is not None},
        )

    def get_description(self, field_name: str) -> str:
        hints = get_type_hints(Config, include_extras=True)
        if field_name in Config.__struct_fields__:
            if get_origin(hints[field_name]) is Annotated:
                type_hint, meta = get_args(hints[field_name])
                hint = stringify_hint(type_hint)
                desc = meta.description + " "
            else:
                # If the type is unannotated no meta so no description
                hint = stringify_hint(hints[field_name])
                desc = ""
            return f"{desc}(type: {hint})"
        else:
            raise TypeError(f"{field_name} is not a field in Config")

    @classmethod
    def set_config_param(cls, param, value):
        cls.config_map.update({param: value})

    @classmethod
    def get_config_param(cls, param):
        return cls.config_map[param]

    add_default_argument = add_default_argument  # overwrite

    def prepare_arguments(self):
        self.add_argument(
            dest="input_images",
            metavar="IMAGE_FILE_OR_FILES",
            nargs="+",
            help="One or more images to process",
        )
        self.add_default_argument(["-d", "--debug-level"], choices=[0, 1, 2, 3])
        self.add_default_argument(
            ["-o", "--debug-output"],
            choices=["file", "screen", "both"],
        )
        self.add_default_argument(
            ["-p", "--pdf"],
            "CONVERT_TO_PDF",
            help="Merge dewarped images into a PDF",
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
        self.add_default_argument(["-nb", "--no-binary"], "NO_BINARY")
        self.add_default_argument(["-s", "--shrink"], "REMAP_DECIMATE")

    def __init__(self):
        super().__init__()
        # The config was read in already (`cfg` in `options.py`, which was imported)
        self.prepare_arguments()  # First set up the parser to read runtime parameters
        self.parsed = self.parse_args()
        self.input_images = self.parsed.input_images
        self.store_parsed_config()  # Store any supplied parameters in the global config

    @snoop()
    def store_parsed_config(self):
        for opt in self.config_map:
            # Redundant but thorough: any unchanged defaults will be reassigned
            configured_opt = getattr(self.parsed, opt)
            if configured_opt is not None:
                self.set_config_param(opt, configured_opt)
