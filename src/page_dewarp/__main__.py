import sys
from cv2 import namedWindow
import numpy as np
from .cli import ArgParser
from .debug_utils import cCOLOURS, debug_show
from .image import WarpedImage
from .pdf import save_pdf


# for some reason pylint complains about cv2 members being undefined :(
# pylint: disable=E1101

WINDOW_NAME = "Dewarp"  # Window name for visualization

def main():
    parser = ArgParser()
    args = parser.parse_args()

    input_images = args.input_images
    global DEBUG_LEVEL, DEBUG_OUTPUT
    DEBUG_LEVEL = args.DEBUG_LEVEL
    DEBUG_OUTPUT = args.DEBUG_OUTPUT
    convert_to_pdf = args.to_pdf

    if DEBUG_LEVEL > 0 and DEBUG_OUTPUT != "file":
        namedWindow(WINDOW_NAME)

    outfiles = []

    for imgfile in input_images:
        processed_img = WarpedImage(imgfile)
        if processed_img.written:
            outfiles.append(processed_img.outfile)
            print(f"  wrote {processed_img.outfile}", end="\n\n")

    if convert_to_pdf:
        save_pdf(outfiles)


if __name__ == "__main__":
    main()
