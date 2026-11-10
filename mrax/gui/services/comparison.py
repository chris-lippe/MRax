"""Read-only native-grid comparison planes; never apply or bypass a pipeline."""

from __future__ import annotations

import math

import numpy as np

from .cloning import clone_workspace
from .pipeline import display_session, install, snapshot
from .view_data import display_options, render_image


def default_comparison(workspace):
    source = {
        "experiment": workspace.session.selected_experiment or "",
        "component": "auto",
        "colormap": "gray",
        "frame": None,
    }
    return {
        "enabled": False,
        "mode": "split",
        "opacity": 0.5,
        "frame": None,
        "plane": None,
        "indices": None,
        "layout": "swipe",
        "position": 0.5,
        "linked_window": True,
        "left": {**source, "kind": "original"},
        "right": {**source, "kind": "processed"},
    }


def comparison_sources(workspace, catalog):
    from .scene import overlay_sources

    sources = []
    for experiment in catalog:
        name = experiment["name"]
        pipeline = workspace.pipelines.get(name)
        for kind in ("original", "processed"):
            state = pipeline.get("source" if kind == "original" else "output") if pipeline else None
            frames = experiment["frames"]
            if state:
                frames = math.prod(
                    value for key, value in state["dims"].items() if key not in {"x", "y", "z"}
                )
            sources.append(
                {
                    "experiment": name,
                    "kind": kind,
                    "frames": frames,
                    "dimensions": {
                        dim: state["dims"][dim]
                        for dim in experiment["spatial_dimensions"]
                        if dim in state["dims"]
                    }
                    if state
                    else experiment["spatial_dimensions"],
                    "label": f"{name}: {kind.capitalize()}",
                }
            )
    dimensions = {entry["name"]: entry["spatial_dimensions"] for entry in catalog}
    sources.extend(
        {**source, "dimensions": dimensions[source["experiment"]]}
        for source in overlay_sources(workspace, catalog)
        if source["kind"] != "image" and source["experiment"] in dimensions
    )
    return sources


def same_source(first, second):
    return all(
        first.get(key) == second.get(key)
        for key in ("experiment", "kind", "map_name", "registration_key")
    )


def validate_comparison(workspace, raw, catalog):
    from .scene import COLORMAPS, COMPONENTS

    defaults = default_comparison(workspace)
    if not isinstance(raw, dict) or set(raw) - set(defaults):
        raise ValueError("Invalid comparison settings")
    settings = {**defaults, **raw}
    if not isinstance(settings["enabled"], bool) or not isinstance(settings["linked_window"], bool):
        raise ValueError("Comparison switches must be true or false")
    if settings["mode"] not in {"overlay", "split"}:
        raise ValueError("Choose Overlay or Split view")
    opacity = settings["opacity"]
    if not isinstance(opacity, (int, float)) or not math.isfinite(opacity) or not 0 <= opacity <= 1:
        raise ValueError("Overlay opacity must be between zero and one")
    if settings["layout"] not in {"swipe", "side-by-side"}:
        raise ValueError("Choose Swipe or Side by side")
    if settings["frame"] is not None and (
        type(settings["frame"]) is not int or settings["frame"] < 0
    ):
        raise ValueError("Choose a non-negative comparison frame")
    if settings["plane"] is not None and settings["plane"] not in {"axial", "coronal", "sagittal"}:
        raise ValueError("Choose an available comparison plane")
    if settings["indices"] is not None and (
        not isinstance(settings["indices"], dict)
        or any(
            not isinstance(key, str) or type(value) is not int or value < 0
            for key, value in settings["indices"].items()
        )
    ):
        raise ValueError("Choose non-negative comparison indices")
    position = settings["position"]
    if (
        not isinstance(position, (int, float))
        or not math.isfinite(position)
        or not 0.05 <= position <= 0.95
    ):
        raise ValueError("The comparison divider must be between 5 and 95 percent")
    choices = comparison_sources(workspace, catalog)
    selected = []
    for side in ("left", "right"):
        source = settings[side]
        if not isinstance(source, dict) or set(source) - {
            "experiment",
            "kind",
            "map_name",
            "registration_key",
            "frame",
            "component",
            "colormap",
            "window",
        }:
            raise ValueError("Invalid comparison source")
        choice = next((item for item in choices if same_source(item, source)), None)
        # Empty studies still have a valid, inactive display configuration.
        if choice is None and (workspace.session.is_loaded or settings["enabled"]):
            raise ValueError("Choose an available comparison source")
        if choice is not None:
            selected.append(choice)
        if (
            source.get("component", "auto") not in COMPONENTS
            or source.get("colormap", "gray") not in COLORMAPS
        ):
            raise ValueError("Choose an available comparison component and colormap")
        frame = source.get("frame")
        if frame is not None and (
            type(frame) is not int or frame < 0 or (choice and frame >= choice["frames"])
        ):
            raise ValueError("Comparison frame is out of range")
        window = source.get("window")
        if window is not None and (
            not isinstance(window, dict)
            or set(window) != {"center", "width"}
            or not all(
                isinstance(value, (int, float)) and math.isfinite(value)
                for value in window.values()
            )
            or window["width"] <= 0
        ):
            raise ValueError("Choose a finite comparison window with positive width")
    if len(selected) == 2:
        following = [
            choice["frames"]
            for side, choice in zip(("left", "right"), selected)
            if settings[side].get("frame") is None and choice["frames"] > 1
        ]
        if settings["frame"] is not None and settings["frame"] >= min(following, default=1):
            raise ValueError("Choose a frame available in both images")
        dimensions = {
            dim: min(choice["dimensions"].get(dim, 1) for choice in selected)
            for dim in set().union(*(choice["dimensions"] for choice in selected))
        }
        if settings["indices"] is not None and any(
            dim not in dimensions or index >= dimensions[dim]
            for dim, index in settings["indices"].items()
        ):
            raise ValueError("Choose a slice available in both images")
        if settings["plane"] not in {None, "axial"} and any(
            "z" not in choice["dimensions"] for choice in selected
        ):
            raise ValueError("Choose Axial for two-dimensional images")
    return settings


def _plane(workspace, source, navigation):
    """Clone metadata only; render one plane from the chosen immutable buffer."""
    from .scene import render_overlay

    context = clone_workspace(workspace)
    session = context.session
    if source["experiment"] not in session.experiment_names:
        raise ValueError("This comparison experiment is unavailable")
    frame = source.get("frame")
    follow_frame = frame is None
    frame = navigation["frame"] if follow_frame else frame
    indices = navigation["indices"]
    plane = navigation["plane"]
    session.select_experiment(source["experiment"])
    kind = source["kind"]
    context.view_label = "Image"
    if kind in {"original", "processed"}:
        pipeline = context.pipelines.get(session.selected_experiment)
        state = (
            pipeline["source" if kind == "original" else "output"]
            if pipeline
            else snapshot(session, original=kind == "original")
        )
        install(context, state, update_model=False)
    elif kind == "map":
        context.view_label = "Map"
        session.active_map = source.get("map_name")
    elif kind == "reconstruction":
        context.view_label = "MRax reconstruction"
    elif kind != "registration":
        raise ValueError("Unknown comparison source")
    grid = display_session(context)
    frame_count = 1 if kind == "map" else grid.frame_count
    if follow_frame and frame_count == 1:
        frame = 0
    if not 0 <= frame < frame_count:
        raise ValueError("This source has no matching frame. Choose a fixed comparison frame.")
    if plane != "axial" and not grid.supports_image_planes():
        raise ValueError("This source has no matching anatomical plane. Choose Axial.")
    session.set_image_plane(plane)
    grid.set_image_plane(plane)
    for dimension, index in indices.items():
        if dimension == "frame" or dimension in grid.image_plane_dims:
            continue
        size = grid.experiment_dims.get(dimension)
        if size is not None and not 0 <= index < size:
            raise ValueError("This source has no matching slice. Choose a shared slice.")
        session.set_active_index(dimension, index)
        grid.set_active_index(dimension, index)
    session.set_active_frame(frame)
    grid.set_active_frame(frame)
    if kind == "registration":
        values = render_overlay(
            context, {**source, "component": source.get("component", "auto"), "frame": frame}
        )
    else:
        display_options(context)["component"] = source.get("component", "auto")
        if kind == "reconstruction":
            component = source.get("component", "auto")
            if component not in {"auto", "magnitude", "phase"}:
                raise ValueError("Reconstruction comparison supports Magnitude or Phase")
            values = grid.reconstruction_image(
                component="phase" if component == "phase" else "magnitude",
                combined=True,
                unwrap_phase=False,
            )
        else:
            values = render_image(context)[0]
    values = np.asarray(values, dtype="<f8")
    if values.ndim != 2:
        raise ValueError("Choose a scalar image component for comparison")
    finite = values[np.isfinite(values)]
    unit = (grid.current_experiment.get("extra_metadata") or {}).get("image_unit", "")
    if kind == "map":
        unit = grid.map_unit() or ""
    if source.get("component") == "phase":
        unit = "rad"
    return (
        values,
        {
            "shape": list(values.shape),
            "spacing": list(grid.image_plane_voxel_sizes),
            "plane_dims": list(grid.image_plane_dims),
            "frame": frame,
            "unit": str(unit),
            "range": [float(finite.min()), float(finite.max())] if finite.size else [0.0, 1.0],
        },
        grid,
    )


def render_comparison(workspace, *, catalog=None):
    from .scene import experiment_catalog, require_common_grid, scene_settings

    settings = scene_settings(workspace)["comparison"]
    names = {settings[side]["experiment"] for side in ("left", "right")}
    for side in ("left", "right"):
        source = settings[side]
        if source["kind"] == "registration":
            result = workspace.registration_results[source["registration_key"]]
            names.add(result["metadata"]["moving_experiment"])
    choices = comparison_sources(
        workspace, experiment_catalog(workspace, names) if catalog is None else catalog
    )
    selected = [
        next(item for item in choices if same_source(item, settings[side]))
        for side in ("left", "right")
    ]
    following = [
        choice["frames"]
        for side, choice in zip(("left", "right"), selected)
        if settings[side].get("frame") is None and choice["frames"] > 1
    ]
    frame_count = min(following, default=1)
    dimensions = {
        dim: min(choice["dimensions"].get(dim, 1) for choice in selected)
        for dim in set().union(*(choice["dimensions"] for choice in selected))
    }
    cursor = settings["indices"]
    indices = dict(workspace.session.active_indices) if cursor is None else cursor
    resolved = {dim: indices.get(dim, 0) for dim in dimensions}
    if cursor is None:
        resolved = {dim: min(index, dimensions[dim] - 1) for dim, index in resolved.items()}
    elif set(cursor) - set(dimensions) or any(
        index >= dimensions[dim] for dim, index in resolved.items()
    ):
        raise ValueError("Choose a slice available in both images")
    plane = settings["plane"]
    if plane is None:
        plane = (
            workspace.session.active_image_plane
            if all("z" in choice["dimensions"] for choice in selected)
            else "axial"
        )
    navigation = {
        "frame": min(workspace.session.active_frame, frame_count - 1)
        if settings["frame"] is None
        else settings["frame"],
        "plane": plane,
        "indices": resolved,
    }
    arrays, planes, grids = [], [], []
    for side in ("left", "right"):
        values, metadata, grid = _plane(workspace, settings[side], navigation)
        arrays.append(values)
        planes.append(metadata)
        grids.append(grid)
    reason = None
    try:
        require_common_grid(
            grids[0].spatial_dim_names,
            grids[0].spatial_shape,
            grids[0].build_dataset().spatial_geometry,
            grids[1].spatial_dim_names,
            grids[1].spatial_shape,
            grids[1].build_dataset().spatial_geometry,
            source_frame=planes[0]["frame"],
            target_frame=planes[1]["frame"],
        )
        if (
            planes[0]["plane_dims"] != planes[1]["plane_dims"]
            or planes[0]["shape"] != planes[1]["shape"]
        ):
            raise ValueError("The comparison planes use different image grids")
    except ValueError as exc:
        reason = str(exc)
    return arrays, {
        "planes": planes,
        "geometry_compatible": reason is None,
        "geometry_message": reason,
        "same_units": planes[0]["unit"] == planes[1]["unit"],
    }
