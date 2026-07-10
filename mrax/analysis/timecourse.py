from __future__ import annotations

from typing import Any, Sequence

import jax.numpy as jnp
import numpy as np

from mrax.data_struct.dimensions import reshape_frames_to_dims
from mrax.data_struct.experiment import Experiment
from mrax.data_struct.quantity import Quantity

from .roi import ROI, broadcast_mask

ArrayLike = Any


_STAT_FUNCS = {
    "mean": "nanmean",
    "median": "nanmedian",
    "std": "nanstd",
    "variance": "nanvar",
    "var": "nanvar",
    "sum": "nansum",
}


def _array_module(*values: object):
    for value in values:
        if isinstance(value, jnp.ndarray):
            return jnp
    return np


def _as_array(value: ArrayLike) -> tuple[np.ndarray, str | None]:
    if isinstance(value, Quantity):
        arr = value.values
        unit = value.si_unit
    else:
        arr = value
        unit = None
    xp = _array_module(arr)
    return xp.asarray(arr), unit


def _resolve_stat(stat: str, xp):
    if stat not in _STAT_FUNCS:
        raise ValueError(f"Unknown stat '{stat}' for timecourse extraction")
    return getattr(xp, _STAT_FUNCS[stat])


def extract_timecourse(
    array: ArrayLike,
    roi: ROI | ArrayLike,
    *,
    stat: str = "mean",
    spatial_axes: Sequence[int] | None = None,
    mask_dims: Sequence[str] | None = None,
    target_dims: Sequence[str] | None = None,
) -> ArrayLike:
    """Extract a timecourse by reducing over ROI spatial axes."""

    arr, unit = _as_array(array)
    roi_mask = roi.mask if isinstance(roi, ROI) else roi
    roi_dims = roi.dims if isinstance(roi, ROI) else mask_dims
    xp = _array_module(arr, roi_mask)
    arr = xp.asarray(arr)
    if xp is jnp:
        roi_mask = jnp.asarray(roi_mask)
    mask_ndim = xp.asarray(roi_mask).ndim

    mask = broadcast_mask(roi_mask, arr.shape, mask_dims=roi_dims, target_dims=target_dims)
    mask = xp.asarray(mask)
    reducer = _resolve_stat(stat, xp)

    if spatial_axes is None:
        if target_dims is not None and roi_dims is not None:
            spatial_axes = [target_dims.index(dim) for dim in roi_dims]
        else:
            spatial_axes = list(range(arr.ndim - mask_ndim, arr.ndim))

    masked = xp.where(mask, arr, xp.nan)
    reduced = reducer(masked, axis=tuple(spatial_axes))

    if unit:
        return Quantity(reduced, unit)
    return reduced


def extract_timecourse_from_experiment(
    data: ArrayLike,
    experiment: Experiment,
    roi: ROI,
    *,
    stat: str = "mean",
) -> dict[str, Any]:
    """Extract a timecourse with dimension metadata from an Experiment."""

    arr, unit = _as_array(data)
    arr = reshape_frames_to_dims(arr, experiment.dims)

    dim_names = list(experiment.dims.keys())
    spatial_axes = [dim_names.index(name) for name in experiment.spatial_dim_names]
    non_spatial_names = [name for name in dim_names if name not in experiment.spatial_dim_names]

    xp = _array_module(arr, roi.mask)
    arr = xp.asarray(arr)
    mask = broadcast_mask(roi.mask, arr.shape, mask_dims=roi.dims, target_dims=dim_names)
    mask = xp.asarray(mask)
    reducer = _resolve_stat(stat, xp)
    masked = xp.where(mask, arr, xp.nan)
    reduced = reducer(masked, axis=tuple(spatial_axes))

    observables: dict[str, Quantity] = {}
    dim_obs_map = getattr(experiment, "_mrax_dim_observables", None)
    dim_obs_map = dim_obs_map or {}
    for name in non_spatial_names:
        obs_key = dim_obs_map.get(name)
        if obs_key and obs_key in experiment.observables:
            observables[name] = experiment.observables[obs_key]

    if unit:
        reduced = Quantity(reduced, unit)

    return {
        "timecourse": reduced,
        "dim_names": non_spatial_names,
        "observables": observables,
    }


__all__ = [
    "extract_timecourse",
    "extract_timecourse_from_experiment",
]
