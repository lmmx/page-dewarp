"""CLI entry point for the page-dewarp package.

When invoked via `python -m page_dewarp`, this module:
- Enforces the minimum supported Python version.
- Parses command-line arguments.
- Loads configuration settings.
- Processes input images (e.g., dewarping, thresholding).
"""

import msgspec
from cv2 import namedWindow

from .cli import ArgParser
from .image import WarpedImage
from .options import Config, cfg
from .snoopy import snoop


# for some reason pylint complains about cv2 members being undefined :(
# pylint: disable=E1101


@snoop()
def main():
    """Parse CLI arguments and dewarp images."""
    parser = ArgParser()
    config = msgspec.convert(msgspec.structs.asdict(cfg), Config)

    if config.DEBUG_LEVEL > 0 and config.DEBUG_OUTPUT != "file":
        namedWindow("Dewarp")

    outfiles = []
    print(f"Parsed config: {config}")

    for imgfile in parser.input_images:
        processed_img = WarpedImage(imgfile, config=config)
        if processed_img.written:
            outfiles.append(processed_img.outfile)
            print(f"  wrote {processed_img.outfile}", end="\n\n")


if __name__ == "__main__":
    main()
