"""CLI parser for page_dewarp.

This module defines a subclass of `argparse.ArgumentParser` to parse
CLI arguments from the user and store them in the global config.
"""

import argparse
from typing import Annotated, get_args, get_origin, get_type_hints

from .options import Config, cfg
from .snoopy import snoop


__all__ = ["ArgParser"]


def stringify_hint(type_hint) -> str:
    """Convert a type hint to a user-readable string representation."""
    match type_hint:
        case type():
            hint = type_hint.__name__
        case _:
            hint = str(type_hint)
    return hint


class ArgParser(argparse.ArgumentParser):
    """Parser for command-line arguments using a global config.

    This class extends `argparse.ArgumentParser` but hooks into the global
    `Config` object (`cfg`), automatically populating arguments from defaults
    and storing parsed values back into that config.
    """

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
        """Add an argument with defaults coming from the global config.

        Args:
            name_or_flags (list[str]): Short or long flags, e.g. `["-d", "--debug-level"]`.
            arg_name (str): Name used as the config key.
            action (str): Argparse action (`store`, `store_true`, etc.).
            help (str): Help text for argparse.
            const (Any): Constant value for actions such as `store_const`.
            choices (list | tuple): Allowed values for this argument.
            nargs (int | str): Number of arguments consumed.
            metavar (str): Display name in usage messages.
            required (bool): Whether the argument is required.

        """
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
            title = (
                name_or_flags if isinstance(name_or_flags, str) else name_or_flags[1]
            )
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
        """Return a generated help string for a Config field.

        We look up the annotated type metadata (if any) to build an
        informative help string showing the type and default value.
        """
        hints = get_type_hints(Config, include_extras=True)
        if field_name in Config.__struct_fields__:
            field_idx = Config.__struct_fields__.index(field_name)
            if get_origin(hints[field_name]) is Annotated:
                type_hint, meta = get_args(hints[field_name])
                hint = stringify_hint(type_hint)
                desc = meta.description + " "
            else:
                # If the type is unannotated no meta so no description
                hint = stringify_hint(hints[field_name])
                desc = ""
            default = Config.__struct_defaults__[field_idx]
            return f"{desc}(type: {hint}, default: {default})"
        else:
            raise TypeError(f"{field_name} is not a field in Config")

    @classmethod
    def set_config_param(cls, param, value):
        """Set a parameter in the global config map."""
        setattr(cfg, param, value)

    @classmethod
    def get_config_param(cls, param):
        """Retrieve a parameter value from the global config map."""
        return getattr(cfg, param)

    # Overwrite with the function from global of the same name
    add_default_argument = add_default_argument

    def prepare_arguments(self):
        """Define all the standard arguments available via the CLI."""
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
        self.add_default_argument(["-it", "--max-iter"], "OPT_MAX_ITER")
        self.add_default_argument(["-m", "--method"], "OPT_METHOD")
        self.add_default_argument(["-dev", "--device"], "DEVICE")
        self.add_default_argument(["-b", "--batch"], "USE_BATCH")
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
        self.add_default_argument(["-sh", "--shear-cost"], "SHEAR_COST")
        self.add_default_argument(["-mc", "--max-corrections"], "MAX_CORR")
        self.add_default_argument(["-s", "--shrink"], "REMAP_DECIMATE")

    def __init__(self):
        """Initialize the ArgParser, then parse and store CLI parameters."""
        super().__init__()
        # The config was read in already (`cfg` in `options.py`, which was imported)
        self.prepare_arguments()  # First set up the parser to read runtime parameters
        self.parsed = self.parse_args()
        self.input_images = self.parsed.input_images
        self.store_parsed_config()  # Store any supplied parameters in the global config

    @snoop()
    def store_parsed_config(self):
        """Write any parsed CLI options back into the global config."""
        for opt in Config.__struct_fields__:
            value = getattr(self.parsed, opt)
            if value is not None:
                setattr(cfg, opt, value)
