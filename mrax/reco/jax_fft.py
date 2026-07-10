"""JAX-based centered FFT helpers (Cartesian k-space).

This module is intentionally lightweight (no Optax / NUFFT imports) so that
simple FFT/IFFT reconstructions can use JAX without pulling in the full
variational reconstruction stack.
"""

from __future__ import annotations

from typing import Sequence

import jax.numpy as jnp

Array = jnp.ndarray


def cartesian_fft_jax(array: Array, axes: Sequence[int] = (-2, -1)) -> Array:
    """Centered FFT with orthonormal scaling (JAX)."""

    arr = jnp.asarray(array)
    axes = tuple(int(ax) for ax in axes)
    return jnp.fft.fftshift(
        jnp.fft.fftn(jnp.fft.ifftshift(arr, axes=axes), axes=axes, norm="ortho"),
        axes=axes,
    )


def cartesian_ifft_jax(array: Array, axes: Sequence[int] = (-2, -1)) -> Array:
    """Centered inverse FFT with orthonormal scaling (JAX)."""

    arr = jnp.asarray(array)
    axes = tuple(int(ax) for ax in axes)
    return jnp.fft.fftshift(
        jnp.fft.ifftn(jnp.fft.ifftshift(arr, axes=axes), axes=axes, norm="ortho"),
        axes=axes,
    )


__all__ = ["Array", "cartesian_fft_jax", "cartesian_ifft_jax"]

