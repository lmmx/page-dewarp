# src/page_dewarp/batch.py
"""Batched image processing for page dewarping.

This module provides efficient processing of multiple images by:
1. Preprocessing images in parallel (CPU)
2. Running optimization in parallel (GPU via vmap)
3. Generating outputs in parallel (CPU)
"""

from __future__ import annotations

import multiprocessing as mp
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from cv2 import INTER_AREA, imread, rectangle
from cv2 import resize as cv2_resize

from .mask import Mask
from .solve import get_default_params
from .spans import assemble_spans, keypoints_from_samples, sample_spans


if TYPE_CHECKING:
    from .options import Config

__all__ = ["process_images_batched", "BatchedResult"]


@dataclass
class BatchedResult:
    """Result of processing a single image in a batch."""

    input_path: str
    output_path: str | None
    success: bool
    error: str | None = None


@dataclass
class PreparedData:
    """Serializable data from preprocessing (no cv2 images)."""

    file_path: str
    cv2_img_shape: tuple[int, ...]
    small_shape: tuple[int, ...]
    corners: np.ndarray
    rough_dims: tuple[float, float]
    span_counts: list[int]
    params: np.ndarray
    dstpoints: np.ndarray


def _resize_to_screen(
    img: np.ndarray,
    screen_max_w: int,
    screen_max_h: int,
) -> np.ndarray:
    """Downsample image to fit within screen dimensions."""
    height, width = img.shape[:2]
    scl_x = float(width) / screen_max_w
    scl_y = float(height) / screen_max_h
    scl = int(np.ceil(max(scl_x, scl_y)))
    if scl > 1.0:
        inv_scl = 1.0 / scl
        return cv2_resize(img, (0, 0), None, inv_scl, inv_scl, INTER_AREA)
    return img


def _calculate_page_extents(
    height: int,
    width: int,
    margin_x: int,
    margin_y: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create page mask and outline."""
    xmin = margin_x
    ymin = margin_y
    xmax, ymax = (width - xmin), (height - ymin)

    pagemask = np.zeros((height, width), dtype=np.uint8)
    rectangle(pagemask, (xmin, ymin), (xmax, ymax), color=255, thickness=-1)

    page_outline = np.array(
        [[xmin, ymin], [xmin, ymax], [xmax, ymax], [xmax, ymin]],
    )
    return pagemask, page_outline


def _preprocess_single(args: tuple) -> PreparedData | tuple[str, str]:
    """Preprocess a single image. Returns PreparedData or (path, error)."""
    imgfile, config_dict = args

    screen_max_w = config_dict["SCREEN_MAX_W"]
    screen_max_h = config_dict["SCREEN_MAX_H"]
    margin_x = config_dict["PAGE_MARGIN_X"]
    margin_y = config_dict["PAGE_MARGIN_Y"]

    try:
        file_path = Path(imgfile)
        cv2_img = imread(str(file_path))
        small = _resize_to_screen(cv2_img, screen_max_w, screen_max_h)
        stem = file_path.stem

        height, width = small.shape[:2]
        pagemask, page_outline = _calculate_page_extents(
            height,
            width,
            margin_x,
            margin_y,
        )

        mask = Mask(stem, small, pagemask, "text")
        contour_list = mask.contours()
        spans = assemble_spans(stem, small, pagemask, contour_list)

        if len(spans) < 3:
            mask = Mask(stem, small, pagemask, "line")
            contour_list = mask.contours()
            new_spans = assemble_spans(stem, small, pagemask, contour_list)
            if len(new_spans) > len(spans):
                spans = new_spans

        if len(spans) < 1:
            return (str(imgfile), f"Insufficient spans: {len(spans)}")

        span_points = sample_spans(small.shape, spans)
        corners, ycoords, xcoords = keypoints_from_samples(
            stem,
            small,
            pagemask,
            page_outline,
            span_points,
        )
        rough_dims, span_counts, params = get_default_params(corners, ycoords, xcoords)
        dstpoints = np.vstack((corners[0].reshape((1, 1, 2)),) + tuple(span_points))

        return PreparedData(
            file_path=str(file_path),
            cv2_img_shape=cv2_img.shape,
            small_shape=small.shape,
            corners=corners,
            rough_dims=tuple(rough_dims),
            span_counts=span_counts,
            params=params,
            dstpoints=dstpoints,
        )
    except Exception as e:
        return (str(imgfile), str(e))


def _finalize_single(args: tuple) -> tuple[str, str | None, str | None]:
    """Generate output for a single image."""
    file_path, corners, rough_dims, opt_params, config_dict = args

    try:
        import msgspec

        from .dewarp import RemappedImage
        from .image import get_page_dims
        from .options import Config

        config = msgspec.convert(config_dict, Config)
        cv2_img = imread(str(file_path))
        small = _resize_to_screen(cv2_img, config.SCREEN_MAX_W, config.SCREEN_MAX_H)
        stem = Path(file_path).stem
        page_dims = get_page_dims(corners, rough_dims, opt_params)

        if np.any(page_dims < 0):
            page_dims = np.array(rough_dims)

        remap = RemappedImage(
            stem,
            cv2_img,
            small,
            page_dims,
            opt_params,
            config=config,
        )
        return (file_path, remap.threshfile, None)
    except Exception as e:
        return (file_path, None, str(e))


def _run_parallel(func, args_list, n_workers):
    """Run function in parallel, suppressing fork warnings."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="os.fork\\(\\) was called",
            category=RuntimeWarning,
        )
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            return list(executor.map(func, args_list))


def process_images_batched(
    image_files: list[str | Path],
    config: Config,
) -> list[BatchedResult]:
    """Process multiple images with parallel optimization."""
    import msgspec

    from .optimise._jax_vmap import OptimizationProblem, optimise_params_vmap

    config_dict = msgspec.structs.asdict(config)
    n_workers = min(len(image_files), max(1, mp.cpu_count() - 1))

    # Phase 1: Parallel preprocessing
    print("=== Phase 1: Preprocessing ===")
    preprocess_args = [(str(f), config_dict) for f in image_files]

    prepared: list[PreparedData] = []
    failed: list[BatchedResult] = []
    failed_indices: set[int] = set()

    if len(image_files) <= 2:
        preprocess_results = [_preprocess_single(args) for args in preprocess_args]
    else:
        preprocess_results = _run_parallel(
            _preprocess_single,
            preprocess_args,
            n_workers,
        )

    for i, result in enumerate(preprocess_results):
        if isinstance(result, PreparedData):
            n_pts = sum(result.span_counts)
            print(
                f"  {Path(result.file_path).name}: "
                f"{len(result.span_counts)} spans, {n_pts} points",
            )
            prepared.append(result)
        else:
            path, error = result
            print(f"  {Path(path).name}: FAILED - {error}")
            failed_indices.add(i)
            failed.append(
                BatchedResult(
                    input_path=path,
                    output_path=None,
                    success=False,
                    error=error,
                ),
            )

    if not prepared:
        return failed

    # Phase 2: Parallel BFGS optimization (vmap)
    print("\n=== Phase 2: Parallel Optimization ===")
    problems = [
        OptimizationProblem(
            name=Path(p.file_path).stem,
            dstpoints=p.dstpoints,
            span_counts=p.span_counts,
            params=p.params,
            corners=p.corners,
            rough_dims=p.rough_dims,
        )
        for p in prepared
    ]

    optimized_params_list = optimise_params_vmap(problems, config.DEBUG_LEVEL)

    # Phase 3: Parallel output generation
    print("\n=== Phase 3: Generating Outputs ===")
    finalize_args = [
        (p.file_path, p.corners, p.rough_dims, opt_params, config_dict)
        for p, opt_params in zip(prepared, optimized_params_list)
    ]

    if len(finalize_args) <= 2:
        finalize_results = [_finalize_single(args) for args in finalize_args]
    else:
        finalize_results = _run_parallel(_finalize_single, finalize_args, n_workers)

    successful: list[BatchedResult] = []
    for file_path, output_path, error in finalize_results:
        if error:
            print(f"  {Path(file_path).name}: FAILED - {error}")
            successful.append(
                BatchedResult(
                    input_path=file_path,
                    output_path=None,
                    success=False,
                    error=error,
                ),
            )
        else:
            print(f"  wrote {output_path}")
            successful.append(
                BatchedResult(
                    input_path=file_path,
                    output_path=output_path,
                    success=True,
                ),
            )

    # Merge results in original order
    all_results: list[BatchedResult] = []
    prep_idx = 0
    fail_idx = 0

    for i in range(len(image_files)):
        if i in failed_indices:
            all_results.append(failed[fail_idx])
            fail_idx += 1
        else:
            all_results.append(successful[prep_idx])
            prep_idx += 1

    return all_results
