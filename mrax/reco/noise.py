"""Explicit receiver-noise calibration shared by reconstruction methods."""
from __future__ import annotations

import warnings
from dataclasses import replace
from operator import index

import numpy as np


def estimate_noise_whitener(noise, *, coil_axis=1, eigenvalue_floor=1e-6, covariance_scale=1.):
    """Return W with W covariance(noise) W.H = I, using centred noise samples.

    All axes except ``coil_axis`` are independent noise observations. The default
    accepts (record, receiver, sample) arrays. Eigenvalues below the declared
    fraction of the largest are floored with a warning. Absolute amplitudes
    become noise-normalized; different noise/imaging bandwidths need an explicit
    scale correction supplied by the caller.
    """
    values = np.asarray(noise)
    if values.ndim < 2 or not np.isfinite(values).all():
        raise ValueError("Noise calibration requires finite arrays with a receiver axis")
    if not np.isfinite(eigenvalue_floor) or not 0 < eigenvalue_floor <= 1:
        raise ValueError("Noise eigenvalue_floor must be in (0, 1]")
    if not np.isfinite(covariance_scale) or covariance_scale <= 0:
        raise ValueError("Noise covariance_scale must be finite and positive")
    values = np.moveaxis(values, index(coil_axis), 0)
    values = values.reshape((values.shape[0], -1)).astype(np.complex128)
    if values.shape[1] < 2:
        raise ValueError("Noise calibration needs at least two observations")
    values -= values.mean(axis=1, keepdims=True)
    covariance = (values @ values.conj().T) * (covariance_scale / (values.shape[1] - 1))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    maximum = float(eigenvalues[-1])
    if maximum <= 0:
        raise ValueError("Noise calibration has zero covariance")
    floor = maximum * eigenvalue_floor
    if np.any(eigenvalues < floor):
        warnings.warn("Noise covariance is poorly conditioned; eigenvalues were floored",
                      RuntimeWarning, stacklevel=2)
    return (eigenvectors / np.sqrt(np.maximum(eigenvalues, floor))) @ eigenvectors.conj().T


def apply_receiver_transform(values, transform, *, coil_axis):
    """Apply a square receiver matrix to samples, images or sensitivity maps."""
    values = np.asarray(values)
    transform = np.asarray(transform)
    axis = index(coil_axis)
    moved = np.moveaxis(values, axis, 0)
    if transform.shape != (moved.shape[0], moved.shape[0]) or not np.isfinite(transform).all():
        raise ValueError("Receiver transform must be a finite square matrix matching the receivers")
    dtype = np.result_type(values.dtype, np.complex64)
    result = transform.astype(dtype) @ moved.reshape((moved.shape[0], -1))
    return np.moveaxis(result.reshape(moved.shape), 0, axis)


def prewhiten_acquisition(acquisition, *, noise=None, noise_coil_axis=1, eigenvalue_floor=1e-6,
                         covariance_scale=1., auxiliary_coil_axes=None):
    """Return (prepared acquisition, W) using explicit or attached noise records.

    Apply W to externally supplied sensitivities as well. Calibrate sensitivities
    after whitening when using attached calibration data. Receiver-dependent
    sampling masks or statistical weights cannot be mixed by this operation.
    Raw auxiliary roles use (record, receiver, sample) axes; prepared calibration
    grids and EPI navigators use receiver axes 0 and 2, respectively. Supply
    ``auxiliary_coil_axes`` for custom role layouts.
    """
    acquisition.require_prepared()
    for name in ('acquired', 'statistical_weights'):
        value = getattr(acquisition, name)
        if value is not None:
            value = np.broadcast_to(value, acquisition.samples.shape)
            if value.strides[1] and any(not np.array_equal(value[:, coil], value[:, 0])
                                       for coil in range(1, value.shape[1])):
                raise ValueError("Noise whitening requires common receiver sampling and weights")
    noise = acquisition.auxiliary_samples.get('noise') if noise is None else noise
    if noise is None:
        raise ValueError("Noise whitening requires explicit noise calibration samples")
    transform = estimate_noise_whitener(noise, coil_axis=noise_coil_axis,
                                       eigenvalue_floor=eigenvalue_floor, covariance_scale=covariance_scale)
    auxiliary = {}
    for name, values in acquisition.auxiliary_samples.items():
        # Prepared calibration grids are C,*encoding; raw roles are record,C,N.
        axis = (auxiliary_coil_axes or {}).get(name, {'calibration_grid': 0, 'epi_navigators': 2}.get(name, 1))
        auxiliary[name] = apply_receiver_transform(values, transform, coil_axis=axis)
    prepared = replace(acquisition,
        samples=apply_receiver_transform(acquisition.samples, transform, coil_axis=1),
        auxiliary_samples=auxiliary)
    return prepared, transform
