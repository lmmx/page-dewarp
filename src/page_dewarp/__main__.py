# src/page_dewarp/__main__.py
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


def _warmup_jax():
    """Pre-warm JAX to avoid initialization overhead during processing."""
    try:
        import jax
        import jax.numpy as jnp

        # Force JAX initialization and JIT compilation warmup
        _ = jax.devices()
        x = jnp.ones(10)
        _ = (x + x).block_until_ready()
    except ImportError:
        pass


def _should_use_batch(config: Config, num_images: int) -> bool:
    """Determine whether to use batched processing."""
    if config.USE_BATCH == "auto":
        # Use batch if multiple images
        return num_images > 1
    elif config.USE_BATCH in ["on", "1"]:
        return True
    elif config.USE_BATCH == ["off", "0"]:
        return False
    else:
        raise ValueError(f"Invalid option for USE_BATCH: {config.USE_BATCH!r}")


@snoop()
def main():
    """Parse CLI arguments and dewarp images."""
    parser = ArgParser()
    config = msgspec.convert(msgspec.structs.asdict(cfg), Config)

    if config.DEBUG_LEVEL > 0 and config.DEBUG_OUTPUT != "file":
        namedWindow("Dewarp")

    print(f"Parsed config: {config}")

    num_images = len(parser.input_images)
    use_batched = _should_use_batch(config, num_images)

    if use_batched:
        from .backends import HAS_JAX

        if HAS_JAX:
            # Warm up JAX early to overlap with any other initialization
            _warmup_jax()

            from .batch import process_images_batched

            print(f"Processing {num_images} images with batched optimization...\n")
            results = process_images_batched(parser.input_images, config)

            print("\n=== Summary ===")
            for result in results:
                if result.success:
                    print(f"  ✓ {result.output_path}")
                else:
                    print(f"  ✗ {result.input_path}: {result.error}")
            return
        else:
            pass  # JAX not available, gracefully fall back to sequential processing

    # Sequential processing
    outfiles = []
    for imgfile in parser.input_images:
        processed_img = WarpedImage(imgfile, config=config)
        if processed_img.written:
            outfiles.append(processed_img.outfile)
            print(f"  wrote {processed_img.outfile}", end="\n\n")


if __name__ == "__main__":
    main()
