def save_pdf(outfiles):
    print("To convert to PDF (requires ImageMagick):")
    print("  convert -compress Group4 " + " ".join(outfiles) + " output.pdf")
    raise NotImplementedError("TODO: handle PDF conversion")
