"""Coordinates and profiles for the same dimensions shown by image sliders."""

from __future__ import annotations

import numpy as np

from mrax.analysis import broadcast_mask
from mrax.data_struct.dimensions import dim_observable_map, reshape_frames_to_dims
from mrax.data_struct.geometry import SpatialGeometry

_TIME_TO_SECONDS = {"s": 1., "ms": .001, "us": 1e-6, "min": 60.}


def axis_coordinates(experiment, dimension, indices=None, *, geometry=None):
    """Return a unit-labelled coordinate vector for a declared dimension."""
    indices = indices or {}
    size = experiment.dims[dimension]
    obs = experiment.observables
    if dimension in experiment.spatial_dim_names:
        if isinstance(geometry, dict):
            geometry = SpatialGeometry.from_metadata(geometry)
        if geometry is not None:
            spacing = geometry.voxel_sizes
            axis = experiment.spatial_dim_names.index(dimension)
            if spacing is not None and len(spacing) > axis and spacing[axis] > 0:
                return f"{dimension} distance (mm)", np.arange(size) * spacing[axis]
        return f"{dimension} index", np.arange(size)
    if dimension in ("t", "frame") and experiment.exp_type not in ("dti", "dki"):
        for name in ("time", "t", "frame_times", "elapsed_time"):
            qty = obs.get(name)
            if qty is not None and qty.si_unit in _TIME_TO_SECONDS:
                values = np.asarray(qty.values).reshape(-1)
                if values.size == size:
                    return "Elapsed time (s)", values * _TIME_TO_SECONDS[qty.si_unit]
        duration = obs.get("scan_time")
        if duration is not None and duration.si_unit in _TIME_TO_SECONDS:
            values = np.asarray(duration.values).reshape(-1)
            if values.size == 1:
                return "Elapsed time (s)", np.arange(size) * float(values[0]) * _TIME_TO_SECONDS[duration.si_unit] / size
        tr = obs.get("TR")
        if tr is not None and tr.si_unit in _TIME_TO_SECONDS:
            values = np.asarray(tr.values).reshape(-1)
            if values.size in (1, size):
                durations = np.broadcast_to(values, (size,)) * _TIME_TO_SECONDS[tr.si_unit]
                return "Elapsed time (s)", np.r_[0., np.cumsum(durations[:-1])]
    mapping = getattr(experiment, "_mrax_dim_observables", None)
    if mapping is None:
        mapping = dim_observable_map(experiment.exp_type)
    key = (mapping or {}).get(dimension)
    if dimension == "frame" and experiment.exp_type in ("dti", "dki"):
        key = "bvals" if "bvals" in obs else "bval"
    qty = obs.get(key) if key else None
    if qty is not None:
        values = np.asarray(qty.values)
        if values.ndim == 1 and values.size == size:
            result = values
        elif values.size == experiment.frame_count:
            grid = values.reshape(experiment.non_spatial_shape)
            selection = tuple(slice(None) if dim == dimension else indices.get(dim, 0)
                              for dim in experiment.non_spatial_dim_names)
            result = grid[selection]
        else:
            result = None
        if result is not None:
            unit = "" if qty.si_unit == "1" else f" ({qty.si_unit})"
            return f"{key}{unit}", np.asarray(result)
    return dimension, np.arange(size)


def profile_selection(experiment, dimension, indices):
    """Indices for a one-dimensional sweep with other frame axes fixed."""
    selected = []
    for name in experiment.non_spatial_dim_names:
        index = indices.get(name, indices.get("frame", 0) if name in ("t", "frame") else 0)
        selected.append(slice(None) if name == dimension else max(0, min(int(index), experiment.dims[name] - 1)))
    grid = np.arange(experiment.frame_count).reshape(experiment.non_spatial_shape)
    frame_ids = np.atleast_1d(grid[tuple(selected)]).reshape(-1)
    if dimension in experiment.spatial_dim_names:
        frame_ids = np.repeat(frame_ids, experiment.dims[dimension])
    return tuple(selected), frame_ids


def image_profile(data, experiment, dimension, indices, *, roi=None, plane_dims=("x", "y")):
    arr = np.asarray(reshape_frames_to_dims(data, experiment.dims))
    if roi is not None:
        mask = broadcast_mask(roi.mask, arr.shape, mask_dims=roi.dims, target_dims=tuple(experiment.dims))
    selection = []
    retained = []
    for name in experiment.dims:
        if name == dimension or name in plane_dims:
            selection.append(slice(None))
            retained.append(name)
        else:
            index = indices.get(name, indices.get("frame", 0) if name in ("t", "frame") else 0)
            selection.append(max(0, min(int(index), experiment.dims[name] - 1)))
    sliced = arr[tuple(selection)]
    sliced = np.abs(sliced) if np.iscomplexobj(sliced) else sliced
    if roi is not None:
        sliced = np.where(np.asarray(mask)[tuple(selection)], sliced, np.nan)
    reduce_axes = tuple(i for i, name in enumerate(retained) if name != dimension)
    if reduce_axes:
        finite = np.isfinite(sliced)
        count = finite.sum(axis=reduce_axes)
        total = np.where(finite, sliced, 0).sum(axis=reduce_axes)
        values = np.divide(total, count, out=np.full(np.shape(total), np.nan), where=count > 0)
    else:
        values = sliced
    return np.atleast_1d(values).reshape(-1)
