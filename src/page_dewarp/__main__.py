from cv2 import namedWindow
from .cli import ArgParser
from .debug_utils import cCOLOURS, debug_show
from .image import WarpedImage
from .pdf import save_pdf
from .options import cfg


# for some reason pylint complains about cv2 members being undefined :(
# pylint: disable=E1101

WINDOW_NAME = "Dewarp"  # Window name for visualization


def main():
    parser = ArgParser()

    if cfg.debug_lvl_opt.DEBUG_LEVEL > 0 and cfg.debug_out_opt.DEBUG_OUTPUT != "file":
        namedWindow(WINDOW_NAME)

    outfiles = []

    for imgfile in parser.input_images:
        processed_img = WarpedImage(imgfile)
        if processed_img.written:
            outfiles.append(processed_img.outfile)
            print(f"  wrote {processed_img.outfile}", end="\n\n")

    if cfg.pdf_opts.CONVERT_TO_PDF:
        save_pdf(outfiles)


if __name__ == "__main__":
    main()
