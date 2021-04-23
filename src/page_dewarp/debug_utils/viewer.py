from cv2 import imwrite, imshow, waitKey, FONT_HERSHEY_SIMPLEX, LINE_AA, putText
from ..options import debug_out_opt

__all__ = ["debug_show"]


def debug_show(name, step, text, display):
    if debug_out_opt["DEBUG_OUTPUT"] != "screen":
        filetext = text.replace(" ", "_")
        outfile = name + "_debug_" + str(step) + "_" + filetext + ".png"
        imwrite(outfile, display)

    if debug_out_opt["DEBUG_OUTPUT"] != "file":

        image = display.copy()
        height = image.shape[0]

        putText(
            image,
            text,
            (16, height - 16),
            FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            3,
            LINE_AA,
        )

        putText(
            image,
            text,
            (16, height - 16),
            FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            1,
            LINE_AA,
        )

        imshow(WINDOW_NAME, image)

        while waitKey(5) < 0:
            pass
