# Command-Line Usage for `page_dewarp`

When you run the `page-dewarp` command, it applies a **cubic sheet model** to dewarp images and optionally threshold them. Internally, it uses:

- **ArgParser** (from `cli.py`) to parse arguments.
- **WarpedImage** (from `image.py`) to load & process each image.
- **RemappedImage** (from `dewarp.py`) to apply the warp transform.
- **Contours** logic (from `contours.py`) if text-based span detection is used.

## Basic Command

page-dewarp [options] IMAGE_FILE_OR_FILES...

### Example

page-dewarp input.jpg

Creates a `input_thresh.png` with a thresholded, dewarped version of `input.jpg`.  
Use multiple files:

page-dewarp image1.jpg image2.jpg

That produces `image1_thresh.png` and `image2_thresh.png`.

## Options

The CLI options are defined in [`cli.py`](../../cli.py) and [ArgParser](../../cli.py).  
Briefly:

- **Debug level**: `-d {0,1,2,3}`  
- **Debug output**: `-o {file,screen,both}`  
- **PDF**: `-p` merges outputs into a PDF (see [`pdf.py`](../../pdf.py)).  
- **Margins** (`-x`, `-y`): ignore edges of the image in detection.  
- …(and many more: see `page-dewarp --help`).

## PDF Output

If you use `-p` or `--pdf`, `page-dewarp` attempts to save a PDF. Currently, it just prints instructions for using ImageMagick (`convert -compress Group4 …`). Eventually [`save_pdf`](../../pdf.py) might handle it automatically.

## More Details

- **Contours**: If text contour detection is too few, the program reverts to line detection.
- **Edge constraints**: For spanning text lines horizontally.  
- **Cubic slopes**: The flattening is driven by a 2D -> 3D -> 2D transformation, solved in [`solve.py`](../../solve.py).

For debug images, see the `--debug-level` flag or check [`debug_utils/viewer.py`](../../debug_utils/viewer.py) for how images get displayed/saved.