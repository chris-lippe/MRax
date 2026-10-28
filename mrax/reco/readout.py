"""Explicit Fourier inversion for delayed or incompletely sampled readouts."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from mrax.data_struct.acquisition import ReadoutModel

from .variational import fftnc


def invert_readout_profiles(samples, model: ReadoutModel):
    """Synthesize a uniform full profile using a truncated-SVD Fourier inverse.

    This linear band-limited reconstruction supports complex images.
    Both gradient polarities and acquisition delays are represented by measured
    coordinates. Singular values below the explicit relative cutoff are removed.
    The returned conditioning diagnostics describe the retained subspace.
    """
    array = jnp.asarray(samples)
    count = len(model.coordinates)
    profiles = len(model.target_trajectory) // model.target_points
    if array.shape[-1] != profiles * count or model.target_points != model.image_points:
        raise ValueError("Profile inversion requires matching complete measured and target readouts")
    position = np.arange(model.image_points) - model.image_points // 2
    matrix = np.exp(-2j * np.pi * model.coordinates[:, None] * position) / np.sqrt(model.image_points)
    left, singular, right = np.linalg.svd(matrix, full_matrices=False)
    keep = singular > singular[0] * model.relative_cutoff
    inverse = (right.conj().T[:, keep] / singular[keep]) @ left[:, keep].conj().T
    grouped = array.reshape(array.shape[:-1] + (profiles, count))
    projection = jnp.einsum("...pm,nm->...pn", grouped, jnp.asarray(inverse, dtype=array.dtype), precision=jax.lax.Precision.HIGHEST)
    targets = fftnc(projection, axes=(-1,))
    # Validate the convention instead of assigning a Fourier spectrum to an
    # arbitrary target trajectory with a different readout origin or spacing.
    trajectory = model.target_trajectory.reshape((profiles, model.target_points, -1))
    direction = trajectory[:, -1] - trajectory[:, 0]
    direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
    frequency = np.einsum("pmd,pd->pm", trajectory, direction)
    expected = (np.arange(model.target_points) - model.target_points // 2) / model.target_points
    if not np.allclose(frequency, expected, atol=2e-6):
        raise ValueError("Profile target coordinates must describe a centered uniform full readout")
    return targets.reshape(array.shape[:-1] + (-1,)), {
        "stage": "profile_inversion", "rank": int(keep.sum()),
        "image_points": model.image_points,
        "retained_condition": float(singular[0] / singular[keep][-1]),
        "relative_cutoff": model.relative_cutoff,
    }
