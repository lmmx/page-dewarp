"""Device detection and selection for JAX optimization."""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import jax


__all__ = ["list_backends", "get_device"]


def list_backends() -> list[str]:
    """List the device backend names.

    If the GPU backends are installed will give `['cpu', 'cuda']`.
    """
    import jax

    return list(jax._src.xla_bridge.backends())


def get_device(spec: str) -> jax.Device:
    """Get a JAX device from a specification string.

    Args:
        spec: Device specification. One of:
            - "auto": GPU if available, otherwise CPU
            - "cpu": Force CPU
            - "gpu": Default GPU
            - "gpu:N": Specific GPU by index (e.g., "gpu:0")

    Returns:
        A JAX Device object.

    Raises:
        ValueError: If the specification is invalid or device unavailable.

    """
    import jax

    spec = spec.lower().strip()

    if spec == "cpu":
        return jax.devices("cpu")[0]

    if spec == "auto":
        # JAX already defaults to GPU if available
        return jax.devices()[0]

    if spec.startswith("gpu"):
        # Don't just try to load it, avoid the warning noise
        print("checking...")
        dev_backends = list_backends()
        print("checked...")
        if "cuda" not in dev_backends:
            raise ValueError(
                f"GPU requested but unavailable. Available: {dev_backends}. "
                "Install with GPU support: pip install page-dewarp[jax-cuda-12] or `-13`",
            )

        # Only now do we check for GPU directly.
        gpu_devices = jax.devices("gpu")

        if spec == "gpu":
            return gpu_devices[0]

        # Parse "gpu:N" format
        if spec.startswith("gpu:"):
            try:
                idx = int(spec[4:])
            except ValueError:
                raise ValueError(f"Invalid GPU index in '{spec}'") from None
            if idx >= len(gpu_devices):
                available = ", ".join(str(i) for i in range(len(gpu_devices)))
                raise ValueError(f"GPU {idx} unavailable. Available: {available}")
            return gpu_devices[idx]

    raise ValueError(f"Invalid device: '{spec}'. Use 'auto', 'cpu', 'gpu', or 'gpu:N'.")
