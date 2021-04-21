import argparse


class ArgParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__()
        self.add_argument(
            dest="input_images",
            metavar="IMAGE_FILE_OR_FILES",
            nargs="+",
            help="One or more images to process",
        )
        self.add_argument(
            "--debug-level",
            dest="DEBUG_LEVEL",
            type=int,
            default=0,
            choices=[0, 1, 2, 3],
        )
        self.add_argument(
            "--debug-to",
            dest="DEBUG_OUTPUT",
            default="file",
            choices=["file", "screen", "both"],
        )
        self.add_argument(
            "-p",
            "--pdf",
            dest="to_pdf",
            action="store_true",
            help="Convert result to PDF",
        )
