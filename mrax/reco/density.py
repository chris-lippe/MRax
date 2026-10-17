"""Geometric quadrature for explicitly described centre-out radial samples."""
from __future__ import annotations

from operator import index

import numpy as np


def triangulated_density(trajectory, *, spatial_shape):
    """Barycentric quadrature on a 2D/3D sampling mesh in cycles/pixel.

    Each simplex contributes its area/volume equally to its vertices. Repeated
    coordinates share their cell weight. The convex hull bounds support; no
    unmeasured frequencies are extrapolated. These initialization weights never
    replace statistical noise weights in an iterative data term.
    """
    from scipy.spatial import Delaunay, QhullError

    coords = np.asarray(trajectory, float)
    ndim = len(spatial_shape)
    if coords.ndim != 2 or coords.shape[1] != ndim or ndim not in (2, 3) or not np.isfinite(coords).all():
        raise ValueError("Density estimation requires finite 2D/3D cycles/pixel coordinates")
    unique, inverse, counts = np.unique(np.round(coords, 7), axis=0, return_inverse=True, return_counts=True)
    if len(unique) <= ndim:
        raise ValueError("Too few distinct coordinates for sampling-density estimation")
    try:
        cells = Delaunay(unique).simplices
    except QhullError as exc:
        raise ValueError("Sampling coordinates do not span the declared encoding space") from exc
    edges = unique[cells[:, 1:]] - unique[cells[:, :1]]
    volume = np.abs(np.linalg.det(edges)) / (2 if ndim == 2 else 6)
    weights = np.zeros(len(unique), float)
    np.add.at(weights, cells.ravel(), np.repeat(volume / (ndim + 1), ndim + 1))
    weights *= int(np.prod(spatial_shape))
    return (weights[inverse] / counts[inverse]).astype(np.float32)


def radial_cell_density(trajectory, *, points_per_readout, angular_weights=None, target_sum=None):
    """Integrate radial cells, including the central cell and variable spacing.

    ``trajectory`` has shape ``(sample, 2 or 3)`` and consists of consecutive,
    monotonically outward readouts. Midpoints delimit cells; the inner boundary
    is zero, and the last boundary extends half the final sample spacing.
    A cell's radial measure is ``(outer**D - inner**D) / D``. Angular weights
    are solid angles per readout, defaulting to equal angular coverage of the
    circle or sphere. Unequal direction coverage needs explicit angular weights.

    Without ``target_sum`` the result retains geometric units inherited from
    the trajectory; an explicit positive target requests a global normalization.
    These quadrature weights are separate from statistical noise weights. The
    function does not infer sequence names, scanner versions or sample roles.
    """
    points = index(points_per_readout)
    coordinates = np.asarray(trajectory, dtype=float)
    if points < 2 or coordinates.ndim != 2 or coordinates.shape[1] not in (2, 3) or not len(coordinates) or len(coordinates) % points:
        raise ValueError("Radial trajectories require complete readouts with at least two points")
    if not np.isfinite(coordinates).all():
        raise ValueError("Radial coordinates must be finite")
    ndim = coordinates.shape[1]
    radius = np.linalg.norm(coordinates, axis=-1).reshape((-1, points))
    spacing = np.diff(radius, axis=-1)
    if np.any(spacing <= 0):
        raise ValueError("Centre-out radial coordinates must have strictly increasing radii")
    readouts = len(radius)
    if angular_weights is None:
        angles = np.full(readouts, (2 * np.pi if ndim == 2 else 4 * np.pi) / readouts)
    else:
        angles = np.asarray(angular_weights, float)
        if angles.shape != (readouts,) or not np.isfinite(angles).all() or np.any(angles < 0) or not np.any(angles > 0):
            raise ValueError("Angular weights must be finite nonnegative solid angles per readout")
    edges = np.concatenate((np.zeros((readouts, 1)), (radius[:, 1:] + radius[:, :-1]) / 2,
        radius[:, -1:] + spacing[:, -1:] / 2), axis=-1)
    weights = np.diff(edges**ndim, axis=-1) / ndim * angles[:, None]
    if target_sum is not None:
        if not np.isfinite(target_sum) or target_sum <= 0:
            raise ValueError("Quadrature normalization must be finite and positive")
        weights *= target_sum / np.sum(weights)
    return weights.ravel()


def radial_profile_density(trajectory, *, points_per_readout, center_out=True, nyquist_radius=.5):
    """Radial volume increments per profile, relative to a Nyquist-radius ball.

    This convention integrates to one for a centre-out profile reaching the
    declared radius, or two for a full opposing profile. It does not include an
    angular average. Use geometric solid-angle quadrature for volume integration.
    """
    coordinates = np.asarray(trajectory, float)
    points = index(points_per_readout)
    if coordinates.ndim != 2 or coordinates.shape[1] not in (2, 3) or points < 2 or not len(coordinates) or len(coordinates) % points or not np.isfinite(coordinates).all():
        raise ValueError("Radial profile coordinates must contain complete finite readouts")
    if not np.isfinite(nyquist_radius) or nyquist_radius <= 0:
        raise ValueError("Nyquist radius must be finite and positive")
    ndim = coordinates.shape[-1]
    if center_out:
        return radial_cell_density(coordinates, points_per_readout=points,
            angular_weights=np.ones(len(coordinates) // points)) * ndim / nyquist_radius**ndim
    profiles = coordinates.reshape((-1, points, ndim))
    direction = profiles[:, -1] - profiles[:, 0]
    norm = np.linalg.norm(direction, axis=-1, keepdims=True)
    if np.any(norm == 0):
        raise ValueError("Full radial profiles must have a nonzero direction")
    direction /= norm
    position = np.einsum("pmd,pd->pm", profiles, direction)
    if np.any(np.diff(position, axis=-1) <= 0):
        raise ValueError("Full radial profiles must progress monotonically through their origin")
    edges = np.concatenate((position[:, :1] - np.diff(position, axis=-1)[:, :1] / 2,
        (position[:, 1:] + position[:, :-1]) / 2,
        position[:, -1:] + np.diff(position, axis=-1)[:, -1:] / 2), axis=-1)
    primitive = np.sign(edges) * (np.abs(edges) / nyquist_radius)**ndim
    return np.diff(primitive, axis=-1).ravel()
