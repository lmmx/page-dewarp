"""Utilities for converting dewarped outputs into PDFs.

Currently, we only print instructions for using ImageMagick,
as PDF merging is not yet fully implemented.
"""

__all__ = ["save_pdf"]


def save_pdf(outfiles: list[str]) -> None:
    """Print a command-line hint for merging images into a PDF.

    Args:
        outfiles: A list of file paths (e.g., PNG images) to combine.

    """
    print("To convert to PDF (requires ImageMagick):")
    print("  convert -compress Group4 " + " ".join(outfiles) + " output.pdf")
    raise NotImplementedError("TODO: handle PDF conversion programmatically.")
