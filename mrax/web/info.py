"""Bounded browser metadata. Full volumes stay in the Python application."""

from __future__ import annotations

import numpy as np

from mrax.gui.models import model_choices
from mrax.gui.plotting import profile_dimension
from mrax.gui.services.pipeline import display_session, display_steps, pipeline_info, project_roi
from mrax.gui.services.reconstruction import reconstruction_label
from mrax.gui.services.roi_tools import roi_runs
from mrax.gui.services.scene import experiment_catalog, render_overlay, scene_info, scene_settings
from mrax.gui.services.state import _fit_settings_to_dict, _preview_result_to_json
from mrax.gui.services.view_data import (
    curve_for_workspace,
    display_options,
    displayed_roi,
    json_value,
    render_image,
)


def _cache(record):
    cache = record.info_cache
    if cache.get("revision") != record.revision:
        cache.clear()
        cache["revision"] = record.revision
    return cache


def image_for_record(record, layer=None):
    workspace, session = record.workspace, record.workspace.session
    cache = _cache(record)
    key = (
        record.view_revision,
        session.selected_experiment,
        workspace.view_label,
        session.active_image_plane,
        tuple(sorted(session.active_indices.items())),
        session.active_frame,
        session.active_z,
        session.active_map,
        session.active_reconstruction_component,
        session.active_reconstruction_coil,
        session.active_reconstruction_unwrap_phase,
        session.active_kspace_mode,
        session.active_kspace_centering,
        repr(session.visual_state),
        repr(workspace.display_settings),
    )
    if cache.get("view_key") != key:
        cache["view_key"] = key
        cache["images"] = {}
    images = cache.setdefault("images", {})
    if layer not in images:
        if layer is None:
            images[layer] = render_image(workspace)
        else:
            selected = next(
                (row for row in scene_settings(workspace)["layers"] if row["id"] == layer), None
            )
            if selected is None:
                raise ValueError("This overlay layer is unavailable")
            images[layer] = (render_overlay(workspace, selected), False)
    return images[layer]


def workspace_info(record):
    workspace, session = record.workspace, record.workspace.session
    session = display_session(workspace)
    loaded = session.is_loaded
    cache = _cache(record)
    image, editable = image_for_record(record)
    shape = list(image.shape[:2])
    settings = display_options(workspace)
    scene = scene_settings(workspace)
    if "catalog" not in cache:
        cache["catalog"] = experiment_catalog(workspace) if loaded else []
    catalog = cache["catalog"]
    rois = []
    for native_roi in workspace.session.shared_rois:
        roi = project_roi(workspace, native_roi, steps=display_steps(workspace))
        enabled = any(r.name == roi.name for r in session.rois)
        statistics = cache.setdefault("roi_statistics", {})
        identity = id(roi.mask)
        if identity not in statistics:
            statistics[identity] = (
                roi.mask,
                int(np.count_nonzero(roi.mask)),
                [int(v) for v in np.unique(roi.mask) if v != 0],
            )
        _, pixels, labels = statistics[identity]
        planes = cache.setdefault("roi_planes", {})
        dimensions = tuple(roi.dims or session.spatial_dim_names)
        plane_key = (
            session.selected_experiment,
            session.active_image_plane,
            tuple(image.shape),
            tuple((name, session.active_indices.get(name, 0)) for name in dimensions),
            editable,
            repr(session.visual_state),
            enabled,
        )
        if identity not in planes or planes[identity][0] != plane_key:
            plane = displayed_roi(workspace, native_roi, image, editable) if enabled else None
            planes[identity] = (
                plane_key,
                plane.astype(int).tolist() if plane is not None and plane.size <= 4096 else [],
                roi_runs(plane) if plane is not None else [],
            )
        _, mask, runs = planes[identity]
        rois.append(
            {
                "name": roi.name,
                "pixels": pixels,
                "mask": mask,
                "runs": runs,
                "labels": labels,
                "enabled": enabled,
                "visible": enabled and roi.name not in scene["hidden_rois"],
                "metadata": json_value(roi.metadata),
            }
        )
    preview = workspace.preview_results_by_experiment.get(session.selected_experiment or "")
    if settings.get("plot_whole_image", False):
        preview = None
    selected_roi = session.roi(session.fit_settings.selected_roi) if session.rois else None
    if (
        preview
        and preview.get("roi_name")
        and (selected_roi is None or preview["roi_name"] != selected_roi.name)
    ):
        preview = None
    axis_key = (session.selected_experiment, workspace.plot_x_axis_key)
    if loaded and cache.get("profile_axis_key") != axis_key:
        cache.update(
            profile_axis_key=axis_key,
            profile_axis=profile_dimension(session.build_experiment(), workspace.plot_x_axis_key, non_imaging=session.is_non_imaging),
        )
    axis = cache.get("profile_axis", "")
    profile_indices = dict(session.active_indices)
    profile_indices.pop(axis, None)
    if axis == "t":
        profile_indices.pop("frame", None)
    curve_key = (
        session.selected_experiment,
        session.fit_settings.selected_roi,
        settings.get("plot_whole_image", False),
        settings.get("component", "auto"),
        axis,
        tuple(sorted(profile_indices.items())),
        repr(session.visual_state),
        session.fit_settings.use_view_state_data,
    )
    if cache.get("curve_key") != curve_key:
        curve = curve_for_workspace(workspace) if loaded else None
        cache.update(curve_key=curve_key, curve=curve)
    curve = cache["curve"]
    if (
        preview
        and curve
        and (
            len(preview.get("x", [])) != len(curve["x"])
            or (
                workspace.plot_x_axis_key != "auto"
                and workspace.plot_x_axis_key in session.spatial_dim_names
            )
        )
    ):
        preview = None
    geometry = session.current_experiment.get("spatial_geometry") if loaded else None
    map_names = session.map_names if loaded else []
    mode = workspace.view_label
    unit = (
        "rad"
        if (mode in {"Image", "Map"} and settings.get("component") == "phase")
        or (mode == "MRax reconstruction" and session.active_reconstruction_component == "phase")
        or (mode == "K-space" and session.active_kspace_mode == "phase")
        else session.map_unit()
        if mode == "Map" and map_names
        else ""
    )
    controls = (
        session.dimension_controls(
            "kspace" if mode == "K-space" else "map" if mode == "Map" else "image"
        )
        if loaded
        else []
    )
    if loaded and mode == "Image":
        unit = (session.current_experiment.get("extra_metadata") or {}).get("image_unit", unit)
    from mrax.gui.services.pipeline_blocks import fit_result_for_map, reconstruction_available

    latest_fit = fit_result_for_map(workspace)
    can_reconstruct, reconstruction_unavailable = reconstruction_available(session)
    registrations = [
        {
            "key": key,
            "kind": row.get("kind"),
            "dimension": row["transform"].GetDimension() if row.get("transform") else None,
            "metadata": row.get("metadata"),
            "diagnostics": row.get("diagnostics"),
        }
        for key, row in workspace.registration_results.items()
    ]
    return json_value(
        {
            "id": workspace.key,
            "label": workspace.label,
            "file_location": record.file_location,
            "revision": record.revision,
            "view_revision": record.view_revision,
            "loaded": loaded,
            "non_imaging": session.is_non_imaging if loaded else False,
            "data_role": session.data_role if loaded else None,
            "experiments": session.experiment_names,
            "experiment_catalog": catalog,
            "scene": scene_info(
                workspace,
                catalog,
                render_layer=lambda _workspace, layer: image_for_record(record, layer["id"])[0],
            ),
            "experiment": session.selected_experiment or "",
            "frame": session.active_frame,
            "z": session.active_indices.get(session.image_slice_dim, 0) if loaded else 0,
            "frame_count": session.frame_count if loaded else 0,
            "slice_count": session.view_dimensions.get(session.image_slice_dim, 1) if loaded else 0,
            "slice_dim": session.image_slice_dim if loaded else "z",
            "shape": shape,
            "spacing": list(session.image_plane_voxel_sizes) if loaded else [1, 1],
            "spacing_unit": "mm"
            if isinstance(geometry, dict) and geometry.get("voxel_sizes")
            else "px",
            "plane": session.active_image_plane,
            "can_change_plane": loaded
            and session.supports_image_planes(
                "k_space" if mode == "K-space" else "map" if mode == "Map" else "image"
            ),
            "plane_dims": list(session.image_plane_dims) if loaded else ["x", "y"],
            "plane_axis_labels": list(
                session.image_plane_dims if mode == "K-space" else session.image_plane_axis_labels
            )
            if loaded
            else ["x", "y"],
            "indices": dict(session.active_indices),
            "plane_axis_directions": list(session.image_plane_axis_directions)
            if loaded and mode != "K-space"
            else [1, 1],
            "dimension_controls": controls,
            "dimensions": session.experiment_dims if loaded else {},
            "view": mode,
            "editable": editable,
            "display": {**settings, "plot_axis": workspace.plot_x_axis_key},
            "channels": image.shape[-1] if image.ndim == 3 else 1,
            "image_dtype": "float64-le" if image.dtype.itemsize > 4 else "float32-le",
            "unit": unit,
            "maps": [{"name": name, "unit": session.map_unit(name)} for name in map_names],
            "active_map": session.active_map,
            "reconstruction_component": session.active_reconstruction_component,
            "reconstruction_label": reconstruction_label(workspace),
            "reconstruction_coil": session.active_reconstruction_coil,
            "coil_count": session.reconstruction_coil_count if loaded else 0,
            "unwrap_phase": session.active_reconstruction_unwrap_phase,
            "kspace_mode": session.active_kspace_mode,
            "kspace_centering": session.active_kspace_centering,
            "has_reconstruction": session.has_reconstruction if loaded else False,
            "has_kspace": session.current_experiment.get("raw_kspace") is not None
            if loaded
            else False,
            "can_reconstruct": can_reconstruct,
            "reconstruction_unavailable": reconstruction_unavailable,
            "reconstruction_encoding": getattr(
                session.current_experiment.get("acquisition"), "kind", None
            )
            if loaded
            else None,
            "can_load_raw": session.can_lazy_load_raw_data(require_acquisition=True)
            if loaded
            else False,
            "rois": rois,
            "selected_roi": None if selected_roi is None else selected_roi.name,
            "roi_undo": bool(workspace.roi_history),
            "timecourse": curve,
            "preview": None if preview is None else _preview_result_to_json(preview),
            "dirty": workspace.dirty,
            "model": session.fit_settings.model_name or session.current_model_name
            if loaded
            else "identity",
            "fit_settings": _fit_settings_to_dict(session.fit_settings),
            "frame_exclusions": session.frame_mask.excluded_indices
            if session.frame_mask is not None
            else [],
            "visual_state": session.visual_state,
            "registrations": registrations,
            "drifts": [
                {"key": key, "metadata": row.get("metadata"), "summary": row.get("summary")}
                for key, row in workspace.drift_results.items()
            ],
            "fit_info": None
            if latest_fit is None
            else {k: v for k, v in latest_fit.items() if k not in {"fields", "spatial_geometry"}},
            "diagnostics_count": len(workspace.diagnostics),
            "pipeline": pipeline_info(workspace),
            "models": [{"key": m.key, "label": m.label} for m in model_choices()],
        }
    )
