"""Shared, non-destructive display layers and independent fitting selections."""

from __future__ import annotations

import copy
import math
import re
from uuid import uuid4

import numpy as np

from mrax.analysis import ROI
from mrax.data_struct import SpatialGeometry
from mrax.data_struct.dimensions import (
    reshape_dims_to_frames,
    spatial_names,
    spatial_shape_from_dims,
)
from mrax.gui.registration import apply_transform_to_image

from .cloning import clone_workspace
from .view_data import display_options, registration_plane, render_image, scalar_component

COLORMAPS = {"gray", "viridis", "diverging", "hot"}
COMPONENTS = {"auto", "magnitude", "phase", "real", "imag"}
SPATIAL_VIEWS = {"Image", "Map", "MRax reconstruction"}


def scene_settings(workspace, experiment=None):
    from .comparison import default_comparison

    name = experiment if experiment is not None else workspace.session.selected_experiment or ""
    options = workspace.display_settings.get(name, {})
    if not isinstance(options, dict) or not isinstance(options.get("scene", {}), dict):
        raise ValueError("Invalid saved display settings")
    saved = options.get("scene", {})
    return {
        "comparison": {**default_comparison(workspace), **saved.get("comparison", {})},
        "layers": list(saved.get("layers", [])),
        "hidden_rois": list(saved.get("hidden_rois", [])),
        "roi_opacity": saved.get("roi_opacity", 0.35),
        "roi_styles": copy.deepcopy(saved.get("roi_styles", {})),
        "fit_region": saved.get("fit_region", "all"),
        "fit_rois": saved.get("fit_rois"),
    }


def experiment_catalog(workspace, names=None):
    """Metadata from each experiment's own state. Never change the active experiment."""
    from .reconstruction import reconstruction_label

    session = clone_workspace(workspace, share_sources=True).session
    result = []
    for name in session.experiment_names:
        if names is not None and name not in names:
            continue
        session.select_experiment(name)
        result.append(
            {
                "name": name,
                "exp_type": session.current_experiment.get("exp_type") or None,
                "frames": session.frame_count,
                "spatial_dimensions": {
                    dim: session.experiment_dims[dim] for dim in session.spatial_dim_names
                },
                "maps": session.map_names,
                "has_reconstruction": session.has_reconstruction,
                "reconstruction_label": reconstruction_label(workspace, name),
                "rois": [r.name for r in session.rois],
            }
        )
    return result


def overlay_sources(workspace, catalog):
    sources = []
    for entry in catalog:
        common = {"experiment": entry["name"], "frames": entry["frames"]}
        sources.append({**common, "kind": "image", "label": f"{entry['name']} · Image"})
        if entry["has_reconstruction"]:
            sources.append(
                {**common, "kind": "reconstruction", "label": f"{entry['name']} · {entry.get('reconstruction_label', 'Reconstruction')}"}
            )
        sources.extend(
            {
                **common,
                "kind": "map",
                "map_name": name,
                "frames": workspace.session.map_frame_count(name, experiment=entry["name"]),
                "label": f"{entry['name']} · {name}",
            }
            for name in entry["maps"]
        )
    for key, result in workspace.registration_results.items():
        meta = result["metadata"]
        moving = next(
            (entry for entry in catalog if entry["name"] == meta["moving_experiment"]), None
        )
        sources.append(
            {
                "experiment": meta["reference_experiment"],
                "kind": "registration",
                "registration_key": key,
                "frames": 1
                if meta.get("moving_source") == "map" or moving is None
                else moving["frames"],
                "label": (
                    f"{meta['moving_experiment']} → {meta['reference_experiment']}"
                    f" · {key[-8:]} (registered)"
                ),
            }
        )
    return sources


def _same_source(a, b):
    return all(
        a.get(key) == b.get(key) for key in ("kind", "experiment", "map_name", "registration_key")
    )


def require_common_grid(
    source_dims,
    source_shape,
    source_geometry,
    target_dims,
    target_shape,
    target_geometry,
    *,
    source_frame=0,
    target_frame=0,
):
    """Require matching native coordinates and any available physical geometry."""
    if tuple(source_dims) != tuple(target_dims) or tuple(source_shape) != tuple(target_shape):
        raise ValueError("Native grids differ. Register the experiments before combining them")
    a, b = source_geometry, target_geometry
    sa = a.voxel_sizes if a is not None else None
    sb = b.voxel_sizes if b is not None else None
    if sa and sb:
        shared = min(len(sa), len(sb), len(source_dims))
        if not np.allclose(sa[:shared], sb[:shared], atol=1e-6, rtol=1e-6):
            raise ValueError("Image grids differ in voxel spacing. Register the experiments first")
    aa = a.affine_for_frame(source_frame) if a is not None else None
    bb = b.affine_for_frame(target_frame) if b is not None else None
    if (aa is None) != (bb is None) or (
        aa is not None and not np.allclose(aa, bb, atol=1e-6, rtol=1e-6)
    ):
        raise ValueError("Image grids differ in physical geometry. Register the experiments first")


def render_overlay(workspace, layer):
    """Return scalar values on the base grid, without modifying measurement arrays."""
    from .pipeline import display_session

    target = display_session(workspace)
    if workspace.view_label not in SPATIAL_VIEWS or (
        not target.visual_state.show_original
        and target.selected_experiment not in workspace.pipelines
    ):
        raise ValueError("Choose a native image, reconstruction or map as the base for layers")
    if layer["kind"] == "registration":
        result = workspace.registration_results.get(layer.get("registration_key"))
        if result is None:
            raise ValueError("This registration is unavailable")
        meta = result["metadata"]
        reference_grid = meta.get("reference_grid")
        if reference_grid:
            reference_dims = reference_grid["dims"]
            reference_geometry = SpatialGeometry.from_metadata(reference_grid.get("geometry"))
        else:
            reference = clone_workspace(workspace, share_sources=True).session
            reference.select_experiment(meta["reference_experiment"])
            reference_dims = reference.experiment_dims
            reference_geometry = reference.build_dataset().spatial_geometry
        require_common_grid(
            spatial_names(reference_dims),
            spatial_shape_from_dims(reference_dims),
            reference_geometry,
            target.spatial_dim_names,
            target.spatial_shape,
            target.build_dataset().spatial_geometry,
            source_frame=meta.get("reference_geometry_frame", meta.get("reference_frame", 0)),
            target_frame=target.active_frame,
        )
        if meta.get("scope") != "volume":
            if meta.get("plane", "axial") != target.active_image_plane or any(
                target.active_indices.get(dim, 0) != meta.get("indices", {}).get(dim, 0)
                for dim in target.spatial_dim_names
                if dim not in target.image_plane_dims
            ):
                raise ValueError("This registration applies to its stored plane and slice")
        frame = target.active_frame if layer.get("frame") is None else layer["frame"]
        pipeline_step = next(
            (
                step
                for pipeline in workspace.pipelines.values()
                for step in pipeline["steps"]
                if step["enabled"]
                and step.get("result_key") == layer.get("registration_key")
                and step["options"].get("direction", "moving_to_reference") == "moving_to_reference"
                and step.get("result_data") is not None
            ),
            None,
        )
        if pipeline_step is not None and meta.get("moving_source") != "map":
            frames = np.asarray(
                reshape_dims_to_frames(
                    pipeline_step["result_data"], pipeline_step["output_grid"]["dims"]
                )
            )
            if not 0 <= frame < len(frames):
                raise ValueError("The registered sequence has no matching frame")
            corrected = frames[frame]
            if meta.get("scope") != "volume":
                corrected = registration_plane(corrected, target, native_2d=True)
        elif meta.get("moving_source") == "map" or frame == meta.get("moving_frame", 0):
            corrected = np.asarray(result["corrected"])
        elif result.get("pipeline_owned"):
            raise ValueError(
                "This earlier registration is available at its recorded frame. Rerun it for the current sequence"
            )
        else:
            source = clone_workspace(workspace, share_sources=True).session
            source.select_experiment(meta["moving_experiment"])
            if not 0 <= frame < source.frame_count:
                raise ValueError(
                    "The registered sequence has no matching frame. Select an overlay frame"
                )
            source.set_image_plane(meta.get("plane", "axial"))
            for dim, value in meta.get("indices", {}).items():
                if dim != "frame":
                    source.set_active_index(dim, value)
            values = (
                source.reconstruction_display_array()
                if meta.get("moving_source") == "reconstruction"
                else source.image_array
            )
            moving = np.asarray(reshape_dims_to_frames(values, source.experiment_dims))[frame]
            if meta.get("scope") != "volume":
                moving = registration_plane(moving, source, native_2d=bool(meta.get("plane_dims")))
            corrected = apply_transform_to_image(
                np.abs(moving) if np.iscomplexobj(moving) else moving,
                result["fixed"],
                result["transform"],
                interpolator=result.get("options", {}).get("interpolator", "linear"),
            )
        data = registration_plane(corrected, target)
        expected = render_image(workspace)[0].shape[:2]
        if data.shape != expected:
            raise ValueError("The registered grid does not match the base image")
        return np.asarray(scalar_component(data, layer["component"]), dtype=np.float64)

    context = clone_workspace(workspace, share_sources=True)
    context.session.select_experiment(layer["experiment"])
    source = display_session(context, fitted=True, map_name=layer.get("map_name")) if layer["kind"] == "map" else context.session

    if layer["kind"] == "map":
        source.active_map = layer.get("map_name")
        if layer.get("frame") is not None:
            source.set_map_frame(layer["frame"])
        else:
            for dim in source.map_dimensions():
                if dim in target.active_indices:
                    source.set_active_index(dim, target.active_indices[dim])
        frame = source.active_frame
    else:
        frame = target.active_frame if layer.get("frame") is None else layer["frame"]
    if layer["kind"] != "map" and not 0 <= frame < source.frame_count:
        raise ValueError("The overlay has no matching frame. Select an explicit overlay frame")
    require_common_grid(
        source.spatial_dim_names,
        source.spatial_shape,
        source.build_dataset().spatial_geometry,
        target.spatial_dim_names,
        target.spatial_shape,
        target.build_dataset().spatial_geometry,
        source_frame=frame,
        target_frame=target.active_frame,
    )
    source.set_image_plane(target.active_image_plane)
    for dim, value in target.active_indices.items():
        if dim != "frame" and dim in source.spatial_dim_names:
            source.set_active_index(dim, value)
    if layer["kind"] != "map":
        source.set_active_frame(frame)
    if layer["kind"] == "map":
        source.active_map = layer.get("map_name")
        data = source.map_image()
    elif layer["kind"] == "reconstruction":
        data = source.reconstruction_image()
    else:
        data = source.frame_image()
    if data.shape != render_image(workspace)[0].shape[:2]:
        raise ValueError("The overlay does not match the base image grid")
    return np.asarray(scalar_component(data, layer["component"]), dtype=np.float64)


def scene_info(workspace, catalog, *, render_layer=render_overlay, ranges=None):
    from .comparison import comparison_sources
    from .intensity import fixed_range, series_data

    if ranges is None:
        ranges = {}
    scene = scene_settings(workspace)
    sources = overlay_sources(workspace, catalog)
    layers = []
    for layer in scene["layers"]:
        source = next((s for s in sources if _same_source(s, layer)), None)
        reason = "This source is unavailable" if source is None else None
        scale = [0.0, 1.0]
        current = scale
        if reason is None:
            try:
                plane = render_layer(workspace, layer)
                if layer.get("per_frame"):
                    current = fixed_range({}, "frame", plane)
                entry = workspace.session.data["experiments"][layer["experiment"]]
                key = tuple(
                    layer.get(k)
                    for k in ("experiment", "kind", "map_name", "registration_key", "component")
                )
                scale = fixed_range(
                    ranges,
                    ("layer", *key),
                    plane,
                    series_data(entry, layer["kind"], layer.get("map_name")),
                    layer["component"],
                )
            except (ValueError, RuntimeError, KeyError) as exc:
                reason = str(exc)
        layers.append(
            {
                **layer,
                "label": source["label"] if source else "Unavailable layer",
                "unavailable": reason,
                "fixed_range": scale,
                "frame_range": current if layer.get("per_frame") else scale,
            }
        )
    return {
        **scene,
        "layers": layers,
        "sources": sources,
        "comparison_sources": comparison_sources(workspace, catalog),
    }


def validate_scenes(workspace):
    """Validate saved layer references and bounds before a package becomes active."""
    active = workspace.session.selected_experiment
    for name, options in workspace.display_settings.items():
        if name == "" and isinstance(options, dict) and not options.get("scene"):
            continue
        if name not in workspace.session.experiment_names or not isinstance(options, dict):
            raise ValueError("Display settings reference an unavailable experiment")
        if "scene" in options:
            workspace.session.select_experiment(name)
            update_scene(workspace, options["scene"])
    if active:
        workspace.session.select_experiment(active)


def update_scene(workspace, changes):
    from .comparison import validate_comparison

    settings = scene_settings(workspace)
    if not isinstance(changes, dict) or set(changes) - set(settings):
        raise ValueError("Unknown scene setting")
    settings.update(changes)
    catalog = experiment_catalog(workspace)
    settings["comparison"] = validate_comparison(workspace, settings["comparison"], catalog)
    names = {r.name for r in workspace.session.shared_rois}
    for key in ("hidden_rois", "fit_rois"):
        values = settings[key]
        if values is None and key == "fit_rois":
            continue
        if (
            not isinstance(values, list)
            or not all(isinstance(value, str) for value in values)
            or len(values) != len(set(values))
            or set(values) - names
        ):
            raise ValueError("Choose available ROI names once each")
    if settings["fit_region"] not in {"rois", "all"}:
        raise ValueError("Unknown fitting region")
    if not isinstance(settings["roi_opacity"], (int, float)):
        raise ValueError("ROI opacity must be a number")
    opacity = float(settings["roi_opacity"])
    if not math.isfinite(opacity) or not 0 <= opacity <= 1:
        raise ValueError("ROI opacity must be between zero and one")
    settings["roi_opacity"] = opacity
    styles = settings["roi_styles"]
    if not isinstance(styles, dict) or set(styles) - names:
        raise ValueError("Choose available ROIs for display appearance")
    for style in styles.values():
        if not isinstance(style, dict) or set(style) - {
            "color", "opacity", "label_colors", "label_opacities", "hidden_labels"
        }:
            raise ValueError("Unknown ROI appearance setting")
        color = style.get("color")
        colors = style.get("label_colors", {})
        if color is not None and (
            not isinstance(color, str) or re.fullmatch(r"#[0-9a-fA-F]{6}", color) is None
        ):
            raise ValueError("ROI colors require six hexadecimal digits")
        if not isinstance(colors, dict) or any(
            not isinstance(key, str)
            or not key.isdecimal()
            or not 1 <= int(key) <= 65535
            or key != str(int(key))
            or not isinstance(value, str)
            or re.fullmatch(r"#[0-9a-fA-F]{6}", value) is None
            for key, value in colors.items()
        ):
            raise ValueError("Label colors require positive label values and hexadecimal colors")
        alphas = style.get("label_opacities", {})
        if not isinstance(alphas, dict) or any(
            not isinstance(key, str)
            or not key.isdecimal()
            or not 1 <= int(key) <= 65535
            or key != str(int(key))
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= 1
            for key, value in alphas.items()
        ):
            raise ValueError("Label opacities require positive labels and values from zero to one")
        hidden = style.get("hidden_labels", [])
        if (
            not isinstance(hidden, list)
            or any(type(value) is not int or not 1 <= value <= 65535 for value in hidden)
            or len(hidden) != len(set(hidden))
        ):
            raise ValueError("Choose positive label values once each for hidden labels")
        alpha = style.get("opacity")
        if alpha is not None and (
            isinstance(alpha, bool)
            or not isinstance(alpha, (int, float))
            or not math.isfinite(alpha)
            or not 0 <= alpha <= 1
        ):
            raise ValueError("ROI opacity must be between zero and one")
    if not isinstance(settings["layers"], list) or len(settings["layers"]) > 8:
        raise ValueError("Use at most eight image layers")
    sources = overlay_sources(workspace, catalog)
    ids = set()
    layers = []
    allowed = {
        "id",
        "kind",
        "experiment",
        "map_name",
        "registration_key",
        "frame",
        "component",
        "colormap",
        "opacity",
        "visible",
        "window",
        "per_frame",
    }
    for raw in settings["layers"]:
        if not isinstance(raw, dict):
            raise ValueError("Invalid saved overlay layer")
        layer = {key: value for key, value in raw.items() if key in allowed}
        if (
            not isinstance(layer.get("id"), str)
            or not re.fullmatch(r"[\w-]{1,80}", layer["id"])
            or layer["id"] in ids
        ):
            raise ValueError("Each overlay needs a unique identifier")
        ids.add(layer["id"])
        source = next((s for s in sources if _same_source(s, layer)), None)
        if source is None:
            raise ValueError("Choose an available overlay source")
        if (
            not isinstance(layer.get("component"), str)
            or not isinstance(layer.get("colormap"), str)
            or layer["component"] not in COMPONENTS
            or layer["colormap"] not in COLORMAPS
        ):
            raise ValueError("Choose an available component and colormap")
        if not isinstance(layer.get("opacity", 0.5), (int, float)):
            raise ValueError("Overlay opacity must be a number")
        alpha = float(layer.get("opacity", 0.5))
        if not math.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError("Overlay opacity must be between zero and one")
        frame = layer.get("frame")
        if frame is not None and (not isinstance(frame, int) or not 0 <= frame < source["frames"]):
            raise ValueError("Overlay frame is out of range")
        window = layer.get("window")
        if type(layer.get("per_frame", False)) is not bool:
            raise ValueError("Choose whether the layer scales per frame")
        if window is not None and (
            not isinstance(window, dict)
            or set(window) != {"center", "width"}
            or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in window.values())
            or window["width"] <= 0
        ):
            raise ValueError("Choose a finite overlay window with positive width")
        layer.update(opacity=alpha, visible=bool(layer.get("visible", True)))
        layers.append(layer)
    settings["layers"] = layers
    display_options(workspace)["scene"] = settings


def fitting_roi(workspace, names):
    """Union selected masks on the current processing grid."""
    from .pipeline import display_steps, project_roi

    session = workspace.session
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise ValueError("Choose a list of ROI names for fitting")
    selected = [r for r in session.rois if r.name in names]
    if not names or len(set(names)) != len(names) or len(selected) != len(names):
        raise ValueError("Select at least one enabled ROI for fitting")
    mask = np.zeros(session.roi_shape, dtype=bool)
    dims = tuple(session.roi_dim_names)
    space = (session.current_experiment.get("extra_metadata") or {}).get(
        "registration_space", session.selected_experiment
    )
    for roi in selected:
        roi = project_roi(workspace, roi)
        roi_space = roi.metadata.get("registration_space")
        compatible = roi.metadata.get("compatible_experiments", [])
        if roi_space and roi_space != space and session.selected_experiment not in compatible:
            raise ValueError(
                "A selected ROI belongs to another registration space. Align the experiments or transfer the ROI first"
            )
        if np.asarray(roi.mask).shape != mask.shape or (roi.dims and tuple(roi.dims) != dims):
            raise ValueError("Fit ROIs must match the experiment's current grid")
        mask |= np.asarray(roi.mask) != 0
    if not np.any(mask):
        raise ValueError("Selected fit ROIs are empty")
    applied = [
        step["id"] + ":" + str(step["version"])
        for step in display_steps(workspace)
        if step["enabled"] and step["kind"] in {"register", "stored_registration"}
    ]
    return ROI(
        "Fit selection",
        mask,
        dims=dims,
        metadata={
            "source_rois": list(names),
            "registration_space": space,
            "registration_steps": applied,
        },
    )


def add_registration_layer(workspace, key):
    """Keep results with their reference scene without navigating away from current work."""
    result = workspace.registration_results[key]
    name = result["metadata"]["reference_experiment"]
    scene = scene_settings(workspace, name)
    if len(scene["layers"]) >= 8:
        return
    scene["layers"].append(
        {
            "id": uuid4().hex,
            "experiment": name,
            "kind": "registration",
            "registration_key": key,
            "frame": 0
            if result["metadata"].get("moving_source") == "map"
            else result["metadata"].get("moving_frame", 0),
            "component": "auto",
            "colormap": "hot",
            "opacity": 0.5,
            "visible": True,
            "window": None,
        }
    )
    workspace.display_settings.setdefault(name, {})["scene"] = scene


def show_registration_layer(workspace, key):
    """Show a stored alignment even when its automatic layer was removed or hidden."""
    result = workspace.registration_results.get(key)
    if result is None:
        raise ValueError("Choose a stored registration")
    metadata = result["metadata"]
    display_options(workspace)["base_view"] = workspace.view_label
    display_options(workspace)["plot_axis"] = workspace.plot_x_axis_key
    session = workspace.session
    session.select_experiment(metadata["reference_experiment"])
    session.set_image_plane(metadata.get("plane", "axial"))
    for dim, index in metadata.get("indices", {}).items():
        if dim in session.spatial_dim_names:
            session.set_active_index(dim, index)
    session.set_active_frame(metadata.get("reference_frame", 0))
    source = metadata.get("reference_source", "image")
    workspace.view_label = {"map": "Map", "reconstruction": "MRax reconstruction"}.get(
        source, "Image"
    )
    if source == "map":
        session.active_map = metadata["reference_map"]
    session.visual_state.show_original = True
    workspace.plot_x_axis_key = display_options(workspace).get("plot_axis", "auto")
    scene = scene_settings(workspace)
    layer = next((layer for layer in scene["layers"] if layer.get("registration_key") == key), None)
    if layer is not None:
        layer["visible"] = True
        if layer.get("opacity", 0.5) == 0:
            layer["opacity"] = 0.5
        display_options(workspace)["scene"] = scene
    elif len(scene["layers"]) >= 8:
        raise ValueError("Remove an image layer before showing another registration")
    else:
        add_registration_layer(workspace, key)


def rename_scene_roi(workspace, previous, new=None):
    for options in workspace.display_settings.values():
        scene = options.get("scene", {})
        styles = scene.get("roi_styles", {})
        if previous in styles:
            style = styles.pop(previous)
            if new is not None:
                styles[new] = style
        for key in ("hidden_rois", "fit_rois"):
            if scene.get(key) is not None:
                scene[key] = [
                    new if name == previous else name
                    for name in scene[key]
                    if name != previous or new is not None
                ]
