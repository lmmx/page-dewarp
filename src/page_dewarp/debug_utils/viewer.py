import cv2

__all__ = ["debug_show"]

def debug_show(name, step, text, display):
    if DEBUG_OUTPUT != "screen":
        filetext = text.replace(" ", "_")
        outfile = name + "_debug_" + str(step) + "_" + filetext + ".png"
        cv2.imwrite(outfile, display)

    if DEBUG_OUTPUT != "file":

        image = display.copy()
        height = image.shape[0]

        cv2.putText(
            image,
            text,
            (16, height - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )

        cv2.putText(
            image,
            text,
            (16, height - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.imshow(WINDOW_NAME, image)

        while cv2.waitKey(5) < 0:
            pass


