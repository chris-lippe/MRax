"""Quantitative products derived from explicit reconstructed complex echoes."""
from __future__ import annotations

from itertools import product

import jax.numpy as jnp
import numpy as np

from .phase import unwrap_phase_image


def frequency_from_echoes(coil_images, echo_spacing_seconds, *, spatial_ndim,
                          unwrap=True, mask=None, mask_snr=0.,
                          corner_fraction=.2, signal_fraction=0., unwrap_method="quality_guided"):
    """Combine receiver cross-products and convert paired-echo phase to Hz.

    Consecutive frames form echo pairs. The frequency sign is the phase of
    ``conj(first_echo) * second_echo`` divided by positive ``2*pi*delta_TE``.
    Optional masking is explicit: a threshold uses the mean magnitude in the
    two darkest corners, and/or a fraction of the maximum signal. These masks
    describe phase reliability, independently of any downstream analysis ROI.
    """
    coils = jnp.asarray(coil_images)
    if not jnp.iscomplexobj(coils) or not np.isfinite(coils).all():
        raise ValueError("Frequency products require finite complex echo images")
    if spatial_ndim not in (1, 2, 3) or coils.ndim != spatial_ndim + 2 or coils.shape[0] % 2:
        raise ValueError("Frequency products require complete explicit complex echo pairs")
    if not np.isfinite(echo_spacing_seconds) or echo_spacing_seconds <= 0:
        raise ValueError("Echo spacing must be finite and positive in seconds")
    if not np.isfinite(mask_snr) or mask_snr < 0 or not 0 < corner_fraction <= .5 or not 0 <= signal_fraction < 1:
        raise ValueError("Invalid phase-reliability threshold or corner fraction")
    cross = jnp.sum(jnp.conj(coils[::2]) * coils[1::2], axis=1)
    magnitude = jnp.sqrt(jnp.abs(cross))
    spatial_shape = magnitude.shape[-spatial_ndim:]
    corners = []
    for high in product((False, True), repeat=spatial_ndim):
        slices = tuple(slice(-max(1, int(size * corner_fraction)), None) if side else
            slice(0, max(1, int(size * corner_fraction))) for size, side in zip(spatial_shape, high, strict=True))
        corners.append(jnp.mean(magnitude[(slice(None),) + slices], axis=tuple(range(1, spatial_ndim + 1))))
    noise = jnp.mean(jnp.sort(jnp.stack(corners), axis=0)[:2], axis=0)
    axes = tuple(range(1, magnitude.ndim))
    threshold = jnp.maximum(mask_snr * noise, signal_fraction * jnp.max(magnitude, axis=axes))
    reliable = (magnitude >= threshold.reshape((-1,) + (1,) * spatial_ndim)) & (magnitude > 0)
    if mask is not None:
        reliable &= jnp.broadcast_to(jnp.asarray(mask, bool), reliable.shape)
    phase = jnp.where(reliable, jnp.angle(cross), 0)
    if unwrap:
        if unwrap_method == "quality_guided":
            from .phase_quality import quality_guided_unwrap

            phase = quality_guided_unwrap(phase, spatial_ndim=spatial_ndim, mask=reliable, weights=magnitude)
        elif unwrap_method == "weighted_laplacian":
            phase = unwrap_phase_image(phase, spatial_ndim=spatial_ndim, weights=magnitude * reliable, max_iterations=200)
        else:
            raise ValueError("Unknown phase-unwrapping method")
    frequency = jnp.where(reliable, phase / (2 * jnp.pi * echo_spacing_seconds), 0)
    return frequency, reliable
