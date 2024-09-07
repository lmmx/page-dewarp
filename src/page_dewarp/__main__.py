import msgspec
from cv2 import namedWindow

from .cli import ArgParser
from .image import WarpedImage
from .options import Config
from .pdf import save_pdf
from .snoopy import snoop

# for some reason pylint complains about cv2 members being undefined :(
# pylint: disable=E1101

@snoop()
def main():
    parser = ArgParser()
    config = msgspec.convert(parser.config_map, Config)
    if config.DEBUG_LEVEL > 0 and config.DEBUG_OUTPUT != "file":
        namedWindow("Dewarp")

    outfiles = []
    print(f"Parsed config: {config}")

    for imgfile in parser.input_images:
        processed_img = WarpedImage(imgfile, config=config)
        if processed_img.written:
            outfiles.append(processed_img.outfile)
            print(f"  wrote {processed_img.outfile}", end="\n\n")

    if config.CONVERT_TO_PDF:
        save_pdf(outfiles)


if __name__ == "__main__":
    main()
