"""Device detection and selection for JAX optimization."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .logging_config import get_logger


if TYPE_CHECKING:
    import jax


__all__ = ["list_backends", "get_device"]

logger = get_logger("device")


def list_backends() -> list[str]:
    """List the device backend names."""
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
        logger.debug("Device selected", extra={"device": "cpu"})
        return jax.devices("cpu")[0]

    if spec == "auto":
        device = jax.devices()[0]
        logger.debug("Device auto-selected", extra={"device": device.device_kind})
        return device

    if spec.startswith("gpu"):
        dev_backends = list_backends()
        logger.debug("Available backends", extra={"backends": dev_backends})

        if "cuda" not in dev_backends:
            raise ValueError(
                f"GPU requested but unavailable. Available: {dev_backends}. "
                "Install with GPU support: pip install page-dewarp[jax-cuda-12]",
            )

        gpu_devices = jax.devices("gpu")

        if spec == "gpu":
            logger.debug("Device selected", extra={"device": "gpu:0"})
            return gpu_devices[0]

        if spec.startswith("gpu:"):
            try:
                idx = int(spec[4:])
            except ValueError:
                raise ValueError(f"Invalid GPU index in '{spec}'") from None

            if idx >= len(gpu_devices):
                available = ", ".join(str(i) for i in range(len(gpu_devices)))
                raise ValueError(f"GPU {idx} unavailable. Available: {available}")

            logger.debug("Device selected", extra={"device": f"gpu:{idx}"})
            return gpu_devices[idx]

    raise ValueError(f"Invalid device: '{spec}'. Use 'auto', 'cpu', 'gpu', or 'gpu:N'.")
