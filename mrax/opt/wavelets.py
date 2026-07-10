"""Small collection of wavelet transforms used for sparse regularization."""

from __future__ import annotations

import jax.numpy as jnp

Array = jnp.ndarray


def haar_forward2d(x: Array) -> Array:
    """Single-level orthonormal Haar transform for even-sized 2D arrays."""
    if x.ndim != 2:
        raise ValueError("haar_forward2d expects a 2D array")
    if x.shape[0] % 2 or x.shape[1] % 2:
        raise ValueError("haar_forward2d requires even spatial dimensions")

    low_rows = (x[0::2, :] + x[1::2, :]) / 2.0
    high_rows = (x[0::2, :] - x[1::2, :]) / 2.0

    ll = (low_rows[:, 0::2] + low_rows[:, 1::2]) / 2.0
    lh = (low_rows[:, 0::2] - low_rows[:, 1::2]) / 2.0
    hl = (high_rows[:, 0::2] + high_rows[:, 1::2]) / 2.0
    hh = (high_rows[:, 0::2] - high_rows[:, 1::2]) / 2.0
    return jnp.stack([ll, lh, hl, hh], axis=0)


def haar_inverse2d(coeffs: Array) -> Array:
    """Inverse of haar_forward2d."""
    if coeffs.shape[0] != 4:
        raise ValueError("haar_inverse2d expects coeffs with shape (4, H/2, W/2)")
    ll, lh, hl, hh = coeffs
    h, w = ll.shape

    low_rows_even = ll + lh
    low_rows_odd = ll - lh
    high_rows_even = hl + hh
    high_rows_odd = hl - hh

    row_even = jnp.zeros((h, 2 * w))
    row_even = row_even.at[:, 0::2].set(low_rows_even + high_rows_even)
    row_even = row_even.at[:, 1::2].set(low_rows_odd + high_rows_odd)

    row_odd = jnp.zeros((h, 2 * w))
    row_odd = row_odd.at[:, 0::2].set(low_rows_even - high_rows_even)
    row_odd = row_odd.at[:, 1::2].set(low_rows_odd - high_rows_odd)

    out = jnp.zeros((2 * h, 2 * w))
    out = out.at[0::2, :].set(row_even)
    out = out.at[1::2, :].set(row_odd)
    return out


__all__ = ["Array", "haar_forward2d", "haar_inverse2d"]
