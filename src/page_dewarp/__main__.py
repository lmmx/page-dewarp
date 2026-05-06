"""CLI entry point for the page-dewarp package.

When invoked via `python -m page_dewarp`, this module:
- Parses command-line arguments.
- Configures logging based on verbosity flags.
- Processes input images (dewarping, thresholding).
- Prints user-facing output (progress, results, errors).

This is the CLI boundary: all user-facing print() calls live here.
Processing code uses logging only.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import msgspec
from cv2 import namedWindow

from .cli import ArgParser
from .image import WarpedImage
from .logging_config import configure_logging, get_logger
from .options import Config, cfg


logger = get_logger()


@dataclass
class ProcessingResult:
    """Result of processing a single image."""

    input_path: str
    output_path: str | None
    success: bool
    error: str | None = None


def _warmup_jax() -> None:
    """Pre-warm JAX to avoid initialization overhead during processing."""
    try:
        import jax
        import jax.numpy as jnp

        _ = jax.devices()
        x = jnp.ones(10)
        _ = (x + x).block_until_ready()
    except ImportError:
        pass


def _should_use_batch(config: Config, num_images: int) -> bool:
    """Determine whether to use batched processing."""
    if config.USE_BATCH == "auto":
        return num_images > 1
    elif config.USE_BATCH in ("on", "1"):
        return True
    elif config.USE_BATCH in ("off", "0"):
        return False
    else:
        raise ValueError(f"Invalid option for USE_BATCH: {config.USE_BATCH!r}")


def _collect_images(input_images: list[str]) -> list[Path]:
    """Validate that there are no directory paths passed."""
    img_paths = list(map(Path, input_images))
    bad_paths = [p for p in img_paths if p.is_dir()]
    if bad_paths:
        dir_reprs = [repr(str(p.absolute())) for p in bad_paths]
        raise IsADirectoryError(
            "Paths should be to image files, not directories:\n  - "
            + "\n  - ".join(dir_reprs),
        )
    return img_paths


def _run_batched(
    image_files: list[Path],
    config: Config,
    quiet: bool,
    verbose: bool,
) -> list[ProcessingResult]:
    """Run batched processing using JAX backend."""
    from .backends import HAS_JAX
    from .batch import process_images_batched

    if not HAS_JAX:
        logger.warning("JAX not available, falling back to sequential processing")
        return _run_sequential(image_files, config, quiet, verbose)

    _warmup_jax()

    if not quiet:
        print(f"Processing {len(image_files)} images (batched)...")

    batch_results = process_images_batched(
        [str(f) for f in image_files],
        config,
    )

    results = []
    for br in batch_results:
        results.append(
            ProcessingResult(
                input_path=br.input_path,
                output_path=br.output_path,
                success=br.success,
                error=br.error,
            ),
        )
        if not quiet:
            name = Path(br.input_path).name
            if br.success:
                print(f"  ✓ {name} → {br.output_path}")
            else:
                print(f"  ✗ {name}: {br.error}", file=sys.stderr)

    return results


def _run_sequential(
    image_files: list[Path],
    config: Config,
    quiet: bool,
    verbose: bool,
) -> list[ProcessingResult]:
    """Run sequential processing."""
    results = []

    for imgfile in image_files:
        name = imgfile.name

        if not quiet:
            print(f"  {name}...", end="", flush=True)

        try:
            processed = WarpedImage(str(imgfile), config=config)

            if processed.written:
                if not quiet:
                    print(f" → {processed.outfile}")
                results.append(
                    ProcessingResult(
                        input_path=str(imgfile),
                        output_path=processed.outfile,
                        success=True,
                    ),
                )
            else:
                if not quiet:
                    print(" skipped (insufficient spans)")
                results.append(
                    ProcessingResult(
                        input_path=str(imgfile),
                        output_path=None,
                        success=False,
                        error="insufficient spans",
                    ),
                )

        except Exception as e:
            logger.exception("Processing failed", extra={"file": str(imgfile)})
            if not quiet:
                print(f" failed: {e}", file=sys.stderr)
            results.append(
                ProcessingResult(
                    input_path=str(imgfile),
                    output_path=None,
                    success=False,
                    error=str(e),
                ),
            )

    return results


def main() -> None:
    """Parse CLI arguments and dewarp images."""
    parser = ArgParser()
    config = msgspec.convert(msgspec.structs.asdict(cfg), Config)

    # Configure logging based on CLI flags
    configure_logging(
        quiet=parser.quiet,
        verbose=parser.verbose,
        debug=parser.debug_logging,
        log_file=parser.log_file,
    )

    logger.debug(
        "Configuration loaded",
        extra={
            "debug_level": config.DEBUG_LEVEL,
            "opt_method": config.OPT_METHOD,
            "device": config.DEVICE,
        },
    )

    # Set up debug window if needed
    if config.DEBUG_LEVEL > 0 and config.DEBUG_OUTPUT != "file":
        namedWindow("Dewarp")

    image_files = _collect_images(parser.input_images)
    num_images = len(image_files)

    if num_images == 0:
        print("No images to process.", file=sys.stderr)
        sys.exit(1)

    # CLI boundary: announce start
    if not parser.quiet:
        print(f"Processing {num_images} image{'s' if num_images != 1 else ''}...")

    # Choose processing mode
    use_batched = _should_use_batch(config, num_images)

    if use_batched:
        results = _run_batched(image_files, config, parser.quiet, parser.verbose)
    else:
        results = _run_sequential(image_files, config, parser.quiet, parser.verbose)

    # Summarize
    successes = sum(1 for r in results if r.success)
    failures = len(results) - successes

    if not parser.quiet:
        print(f"\nDone. {successes} succeeded, {failures} failed.")

    # In quiet mode, just print output paths (the actual results)
    if parser.quiet:
        for r in results:
            if r.success and r.output_path:
                print(r.output_path)

    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
