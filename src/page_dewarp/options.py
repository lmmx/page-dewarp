from pathlib import Path

import numpy as np
import toml
import tomlkit

from .attr_dict import AttrDict

__all__ = ["Config", "cfg"]


class Config(AttrDict):
    defaults_toml = Path(__file__).parent / "default_options.toml"

    @classmethod
    def from_defaults(cls):
        d = toml.load(cls.defaults_toml)
        cfg = cls.from_dict(d)
        return cfg

    @classmethod
    def parse_defaults_with_comments(cls):
        d = tomlkit.loads(cls.defaults_toml.read_text())
        return d


cfg = Config.from_defaults()


def K():
    "Default intrinsic parameter matrix, depends on FOCAL_LENGTH"
    return np.array(
        [
            [cfg.camera_opts.FOCAL_LENGTH, 0, 0],
            [0, cfg.camera_opts.FOCAL_LENGTH, 0],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )


for k, v in cfg.proj_opts.items():
    setattr(cfg.proj_opts, k, slice(*v))
