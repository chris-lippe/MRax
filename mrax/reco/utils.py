from __future__ import annotations

from typing import Literal

import jax.numpy as jnp
import numpy as np

from mrax.devices import DeviceLike, device_function

from .acquisition import complete_partial_fourier
from .variational import ifftnc


def estimate_sensitivities_cartesian(
    kspace: np.ndarray | jnp.ndarray,
    *,
    frame: int = 0,
    eps: float = 1e-8,
) -> np.ndarray | None:
    """Estimate coil sensitivity maps from Cartesian k-space via RSS normalization."""

    k = jnp.asarray(kspace)
    if k.ndim == 4:
        k0 = k[frame]
    elif k.ndim == 3:
        k0 = k
    else:
        return None

    if k0.ndim != 3 or k0.shape[0] <= 1:
        return None

    coil_imgs = ifftnc(k0)
    rss = jnp.sqrt(jnp.sum(jnp.abs(coil_imgs) ** 2, axis=0, keepdims=True))
    return np.asarray(coil_imgs / (rss + eps))


def pick_sensitivity_frame(
    kspace: np.ndarray | jnp.ndarray,
    *,
    strategy: Literal["first", "max_energy"] | str = "max_energy",
) -> int:
    """Choose a reference frame for sensitivity estimation."""

    arr = np.asarray(kspace)
    if arr.ndim != 4:
        return 0
    mode = str(strategy).lower()
    if mode == "first":
        return 0
    if mode != "max_energy":
        raise ValueError("strategy must be 'first' or 'max_energy'")
    energies = np.sum(np.abs(arr) ** 2, axis=(1, 2, 3))
    return int(np.argmax(energies))


@device_function
def pocs_complete_kspace(
    kspace: np.ndarray | jnp.ndarray,
    *,
    iterations: int = 20,
    mask: np.ndarray,
    spatial_ndim: int = 2,
    device: DeviceLike = "cpu",
) -> np.ndarray:
    """Fill missing Cartesian k-space via POCS (useful for partial Fourier)."""

    return np.asarray(complete_partial_fourier(kspace, mask, spatial_ndim=spatial_ndim,
        method="pocs", iterations=iterations))



__all__ = [
    "estimate_sensitivities_cartesian",
    "pick_sensitivity_frame",
    "pocs_complete_kspace",
]
