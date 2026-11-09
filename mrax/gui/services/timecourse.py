"""Read-only ROI timecourses on the same inputs used by processing and fitting."""

from __future__ import annotations

import warnings

import numpy as np

from mrax.analysis.roi import broadcast_mask
from mrax.analysis.timecourse import extract_timecourse
from mrax.gui.plotting import axis_coordinates, profile_selection
from mrax.gui.session import frame_axis_values

from .cloning import clone_workspace
from .processing import _source
from .view_data import json_value, scalar_component


def analyze_timecourse(workspace, options):
    session = workspace.session
    if not session.is_loaded:
        raise ValueError("Load an experiment before analyzing a timecourse")
    names = options["roi_names"]
    if len(names) != len(set(names)):
        raise ValueError("Choose each ROI once")
    source, component = workspace, options["component"]
    if options["input"] == "reconstruction":
        if component not in {"auto", "magnitude", "phase"}:
            raise ValueError("Reconstruction timecourses use magnitude or phase")
        if component != "auto":
            source = clone_workspace(workspace)
            source.session.active_reconstruction_component = component
        component = "auto"
    data, dims, _obs, _model, unit, _geometry, rois = _source(source, options["input"], component)
    data = np.asarray(scalar_component(data, "auto"), dtype=float)
    spatial = session.spatial_dim_names
    axes = [name for name in dims if name not in spatial]
    dimension = options["axis"] or next((name for name in axes if dims[name] > 1), "")
    if dimension not in axes or dims[dimension] < 2:
        raise ValueError("Choose an acquisition axis with at least two samples")
    if len(names) * dims[dimension] > 100_000:
        raise ValueError("Select fewer ROIs or a shorter acquisition axis")
    experiment = session.build_experiment()
    indices = {
        name: min(max(0, session.active_indices.get(name, 0)), dims[name] - 1)
        for name in axes
    }
    label, coordinates = axis_coordinates(experiment, dimension, indices)
    if label == dimension and len(axes) == 1:
        name, values = frame_axis_values(experiment)
        observable = experiment.observables.get(name)
        if observable is not None:
            unit_label = f" ({observable.si_unit})" if observable.si_unit != "1" else ""
            label, coordinates = name + unit_label, values
    selection, frame_ids = profile_selection(experiment, dimension, indices)
    included = (
        session.frame_mask.as_array()[frame_ids]
        if session.frame_mask is not None
        else np.ones(dims[dimension], dtype=bool)
    )
    spatial_axes = tuple(list(dims).index(name) for name in spatial)
    shape = tuple(dims[name] for name in spatial)
    available = {roi.name: roi for roi in rois}
    state = session.visual_state
    if options["input"] == "view" and not state.show_original and state.mask_enabled:
        mask_roi = available.get(state.mask_roi)
        if mask_roi is None:
            raise ValueError("Choose an enabled View mask for this experiment")
        if tuple(np.shape(mask_roi.mask)) != shape:
            raise ValueError("The View mask must match the experiment's native grid")
        mask = broadcast_mask(mask_roi.mask, data.shape, mask_dims=mask_roi.dims, target_dims=tuple(dims))
        data = np.where(mask, data, np.nan)
    series = []
    for name in names:
        roi = available.get(name)
        if roi is None:
            raise ValueError("Choose enabled ROIs from the current experiment")
        if tuple(np.shape(roi.mask)) != shape or tuple(roi.dims or spatial) != tuple(spatial):
            raise ValueError("Timecourse ROIs must match the experiment's native grid")
        mask = broadcast_mask(roi.mask, data.shape, mask_dims=roi.dims, target_dims=tuple(dims))
        counts = np.count_nonzero(np.isfinite(data) & mask, axis=spatial_axes)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            values = np.asarray(
                extract_timecourse(
                    data, roi, stat=options["statistic"],
                    spatial_axes=spatial_axes, target_dims=tuple(dims),
                ),
                dtype=float,
            )
        values = np.atleast_1d(values[selection]).copy()
        counts = np.atleast_1d(counts[selection]).copy()
        values[counts == 0] = np.nan
        if options["respect_exclusions"]:
            values[~included] = np.nan
            counts[~included] = 0
        series.append({"roi": name, "values": values, "counts": counts})
    if options["statistic"] == "variance" and unit and unit != "1":
        unit = f"({unit})^2"
    return json_value(
        {
            "experiment": session.selected_experiment,
            "input": options["input"],
            "component": options["component"],
            "statistic": options["statistic"],
            "axis": dimension,
            "x": np.asarray(coordinates),
            "x_name": label,
            "unit": unit or "",
            "included": included,
            "respect_exclusions": options["respect_exclusions"],
            "series": series,
        }
    )
