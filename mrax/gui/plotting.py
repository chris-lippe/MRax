"""Coordinates and profiles for the same dimensions shown by image sliders."""

from __future__ import annotations

import numpy as np

from mrax.analysis import broadcast_mask
from mrax.data_struct.dimensions import dim_observable_map, reshape_frames_to_dims
from mrax.data_struct.geometry import SpatialGeometry

_TIME_TO_SECONDS = {"s": 1., "ms": .001, "us": 1e-6, "min": 60.}


def _spectral_coordinates(experiment):
    names = {"frequency", "frequencies", "freq", "spectral_frequency", "chemical_shift", "ppm", "freq_offsets", "offsets"}
    for name, quantity in experiment.observables.items():
        values = np.asarray(quantity.values)
        if name.casefold() not in names or values.ndim != 1 or values.size <= 1:
            continue
        dimensions = tuple(getattr(quantity, "dims", None) or ())
        matches = list(dimensions) if dimensions else [dim for dim, size in experiment.dims.items() if size == values.size]
        if len(matches) == 1 and experiment.dims.get(matches[0]) == values.size:
            yield matches[0], name, quantity


def profile_dimension(experiment, requested="auto", *, non_imaging=False):
    """Choose a varying profile axis, using declared spectral coordinates when available."""
    if requested in experiment.dims:
        return requested
    if non_imaging:
        spectral = next(_spectral_coordinates(experiment), None)
        if spectral is not None:
            return spectral[0]
        named = [name for name in experiment.dims
                 if name.casefold() in {"frequency", "frequencies", "freq", "f", "ppm", "spectral", "spectrum", "sample", "samples"}
                 and experiment.dims[name] > 1]
        if named:
            return max(named, key=lambda name: experiment.dims[name])
        spatial = [name for name in experiment.spatial_dim_names if experiment.dims[name] > 1]
        if spatial:
            return max(spatial, key=lambda name: experiment.dims[name])
    non_spatial = [name for name in experiment.non_spatial_dim_names if name != "slice_package"]
    candidates = non_spatial or list(experiment.spatial_dim_names)
    return next((name for name in candidates if experiment.dims[name] > 1), next(iter(experiment.dims)))


def axis_coordinates(experiment, dimension, indices=None, *, geometry=None, spectral=False):
    """Return a unit-labelled coordinate vector for a declared dimension."""
    indices = indices or {}
    size = experiment.dims[dimension]
    obs = experiment.observables
    if spectral:
        for dim, name, quantity in _spectral_coordinates(experiment):
            if dim == dimension:
                unit = "" if quantity.si_unit == "1" else f" ({quantity.si_unit})"
                return f"{name}{unit}", np.asarray(quantity.values)
        mapping = getattr(experiment, "_mrax_dim_observables", None) or {}
        for name, quantity in obs.items():
            declared = tuple(getattr(quantity, "dims", None) or ()) == (dimension,)
            matched = mapping.get(dimension) == name or name.casefold() == dimension.casefold()
            timed = name.casefold() in {"time", "t", "sample_times"} and sum(value == size for value in experiment.dims.values()) == 1
            if (declared or matched or timed) and np.shape(quantity.values) == (size,):
                unit = "" if quantity.si_unit == "1" else f" ({quantity.si_unit})"
                return f"{name}{unit}", np.asarray(quantity.values)
        return "Sample index" if dimension in experiment.spatial_dim_names else f"{dimension} index", np.arange(size)
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
    if key is None:
        matches = [
            name for name, value in obs.items()
            if name.casefold() == dimension.casefold()
            or tuple(getattr(value, "dims", None) or ()) == (dimension,)
        ]
        if len(matches) == 1:
            key = matches[0]
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


def scalar_component(array, component="auto"):
    array = np.asarray(array)
    if component == "phase":
        return np.angle(array)
    if component == "imag":
        return np.imag(array)
    if component == "real":
        return np.real(array)
    if component == "magnitude" or (component == "auto" and np.iscomplexobj(array)):
        return np.abs(array)
    return array


def image_profile(
    data, experiment, dimension, indices, *, roi=None, plane_dims=("x", "y"), component="auto"
):
    arr = np.asarray(reshape_frames_to_dims(data, experiment.dims))
    selection = []
    retained = []
    for name in experiment.dims:
        if name == dimension or name in plane_dims:
            selection.append(slice(None))
            retained.append(name)
        else:
            index = indices.get(name, indices.get("frame", 0) if name in ("t", "frame") else 0)
            selection.append(max(0, min(int(index), experiment.dims[name] - 1)))
    sliced = scalar_component(arr[tuple(selection)], component)
    if roi is not None:
        # Select the ROI before broadcasting: a plane/curve must not allocate a
        # mask the size of the entire multi-frame acquisition.
        source = np.asarray(roi.mask)
        dimensions = tuple(experiment.dims)
        mask_dims = (
            tuple(roi.dims)
            if roi.dims is not None
            else dimensions[len(dimensions) - source.ndim :]
        )
        if len(mask_dims) != source.ndim or set(mask_dims) - set(dimensions):
            raise ValueError("ROI dimensions do not match the experiment")
        mask_selection = [selection[dimensions.index(name)] for name in mask_dims]
        for axis, index in enumerate(mask_selection):
            if isinstance(index, int) and source.shape[axis] == 1:
                mask_selection[axis] = 0
        selected_dims = tuple(
            name for name, index in zip(mask_dims, mask_selection) if isinstance(index, slice)
        )
        ordered_dims = tuple(name for name in retained if name in selected_dims)
        selected_mask = source[tuple(mask_selection)].transpose(
            tuple(selected_dims.index(name) for name in ordered_dims)
        )
        mask = broadcast_mask(
            selected_mask, sliced.shape, mask_dims=ordered_dims, target_dims=tuple(retained)
        )
        sliced = np.where(mask, sliced, np.nan)
    reduce_axes = tuple(i for i, name in enumerate(retained) if name != dimension)
    if reduce_axes:
        finite = np.isfinite(sliced)
        count = finite.sum(axis=reduce_axes)
        total = np.where(finite, sliced, 0).sum(axis=reduce_axes)
        values = np.divide(total, count, out=np.full(np.shape(total), np.nan), where=count > 0)
    else:
        values = sliced
    return np.atleast_1d(values).reshape(-1)
