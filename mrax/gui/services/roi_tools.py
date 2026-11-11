"""Native-grid ROI editing commands."""

from __future__ import annotations

import copy

import numpy as np
from scipy.ndimage import distance_transform_edt

from mrax.analysis import ROI
from mrax.gui.masks import embed_mask_slice, interpolate_mask_along_slices, mask_slice_for_display

from .geometry import (
    _auto_threshold_mask,
    _brush_mask,
    _brush_radius_from_size,
    _copy_roi_list,
    _delete_mask_slice,
    _ellipse_mask,
    _polygon_mask,
    _regular_polygon_points,
    _seeded_region_mask,
    _shift_array_zero,
    _smooth_lasso_support_points,
)
from .roi import _rect_mask


def remember_rois(workspace):
    session = workspace.session
    session._save_current_view_state()
    snapshot = {
        "shared": _copy_roi_list(session.shared_rois),
        "enabled": {
            name: [r.name for r in state.rois] for name, state in session.view_states.items()
        },
        "selections": {
            name: {
                "hidden_rois": list(
                    workspace.display_settings.get(name, {}).get("scene", {}).get("hidden_rois", [])
                ),
                "fit_rois": copy.deepcopy(
                    workspace.display_settings.get(name, {}).get("scene", {}).get("fit_rois")
                ),
                "roi_styles": copy.deepcopy(
                    workspace.display_settings.get(name, {}).get("scene", {}).get("roi_styles", {})
                ),
                "selected_roi": state.fit_settings.selected_roi,
            }
            for name, state in session.view_states.items()
        },
    }
    workspace.roi_history.append(snapshot)
    while len(workspace.roi_history) > 20 or (
        len(workspace.roi_history) > 1
        and sum(r.mask.nbytes for row in workspace.roi_history for r in row["shared"])
        > 64 * 1024 * 1024
    ):
        workspace.roi_history.pop(0)


def undo_rois(workspace):
    if not workspace.roi_history:
        raise ValueError("There is no ROI edit to undo")
    row = workspace.roi_history.pop()
    session = workspace.session
    previous_names = {r.name for r in session.shared_rois}
    session.shared_rois = row["shared"]
    lookup = {r.name: r for r in session.shared_rois}
    for name, state in session.view_states.items():
        state.rois = [lookup[key] for key in row["enabled"].get(name, []) if key in lookup]
        selection = row.get("selections", {}).get(name)
        if selection is not None:
            state.fit_settings.selected_roi = selection["selected_roi"]
            scene = workspace.display_settings.setdefault(name, {}).setdefault("scene", {})
            for key in ("hidden_rois", "fit_rois"):
                scene[key] = selection[key]
            current_styles = scene.get("roi_styles", {})
            styles = {key: value for key, value in current_styles.items() if key in lookup}
            for key, value in selection.get("roi_styles", {}).items():
                if key not in previous_names and key in lookup:
                    styles.setdefault(key, value)
            old, new = row.get("renamed", (None, None))
            if old in lookup and new in previous_names:
                if new in current_styles:
                    styles[old] = current_styles[new]
                else:
                    styles.pop(old, None)
            scene["roi_styles"] = styles
    session.rois = [
        lookup[key] for key in row["enabled"].get(session.selected_experiment, []) if key in lookup
    ]
    selected = row.get("selections", {}).get(session.selected_experiment)
    if selected is not None:
        session.fit_settings.selected_roi = selected["selected_roi"]


def _commit(session, roi, mask, metadata=None):
    replacement = ROI(
        roi.name,
        np.asarray(mask).copy(),
        dims=roi.dims,
        metadata={**roi.metadata, **(metadata or {})},
    )
    session.shared_rois = [replacement if r.name == roi.name else r for r in session.shared_rois]
    session.rois = [replacement if r.name == roi.name else r for r in session.rois]
    for state in session.view_states.values():
        state.rois = [replacement if r.name == roi.name else r for r in state.rois]
    return replacement


def _plane(session, roi):
    return mask_slice_for_display(
        roi.mask, dims=roi.dims, indices=session.active_indices, plane_dims=session.image_plane_dims
    )


def _subtract_from_others(session, name, covered):
    for other in list(session.rois):
        if other.name != name and other.mask.shape == covered.shape:
            _commit(
                session,
                other,
                np.where(covered, 0, other.mask),
                {"kind": "mask", "consolidated": True},
            )


def _compose_rois(workspace, options):
    session = workspace.session
    name = str(options.get("name", "")).strip()
    if not name or len(name) > 80 or any(r.name == name for r in session.shared_rois):
        raise ValueError("Choose a unique ROI name")
    names = options.get("sources", [])
    if (
        not isinstance(names, list)
        or len(names) < 2
        or not all(isinstance(key, str) for key in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError("Choose at least two distinct source ROIs")
    lookup = {r.name: r for r in session.shared_rois}
    if any(key not in lookup for key in names):
        raise ValueError("A source ROI no longer exists")
    sources = [lookup[key] for key in names]
    if any(
        tuple(r.mask.shape) != tuple(session.roi_shape)
        or tuple(r.dims or session.roi_dim_names) != tuple(session.roi_dim_names)
        for r in sources
    ):
        raise ValueError("Source ROIs must use the same native grid")
    mode = str(options.get("output", "binary"))
    overlap = str(options.get("overlap", "first"))
    if mode not in {"binary", "indexed"} or overlap not in {"first", "last", "error"}:
        raise ValueError("Choose a binary or indexed mask and a valid overlap rule")
    if len(sources) > 65535:
        raise ValueError("An indexed mask supports at most 65535 labels")
    mask = np.zeros(session.roi_shape, dtype=np.uint16)
    for index, source in enumerate(sources, 1):
        covered = np.asarray(source.mask) != 0
        if mode == "indexed":
            if overlap == "error" and np.any(covered & (mask != 0)):
                raise ValueError("Source ROIs overlap; choose a priority rule")
            if overlap == "first":
                covered &= mask == 0
        mask[covered] = index if mode == "indexed" else 1
    if not np.any(mask):
        raise ValueError("The source ROIs contain no labelled voxels")
    metadata = {"kind": "mask", "consolidated": True, "source_rois": names}
    if mode == "indexed":
        metadata["label_names"] = {str(i): r.name for i, r in enumerate(sources, 1)}
    remember_rois(workspace)
    session.add_roi(ROI(name, mask, dims=tuple(session.roi_dim_names), metadata=metadata))
    session.fit_settings.selected_roi = name


def _embed(session, plane):
    return embed_mask_slice(
        plane,
        spatial_shape=session.roi_shape,
        spatial_dims=session.roi_dim_names,
        indices=session.active_indices,
        plane_dims=session.image_plane_dims,
    )


def geometry_mask(session, options, image):
    shape = tuple(session.frame_image().shape)
    kind = options.get("tool", "rect")
    bounds = np.asarray(options.get("bounds", [0, 0, 1, 1]), dtype=float)
    if bounds.shape != (4,) or not np.all(np.isfinite(bounds)) or np.any(bounds[2:] <= 0):
        raise ValueError("Geometry requires finite bounds with a positive width and height")
    metadata = {
        "kind": kind,
        "bounds": bounds.tolist(),
        "angle": float(options.get("angle", 0)),
        "plane_dims": list(session.image_plane_dims),
        "indices": dict(session.active_indices),
    }
    if kind in {"rect", "ellipse"}:
        mask = (_rect_mask if kind == "rect" else _ellipse_mask)(
            shape, bounds, angle=metadata["angle"]
        )
    elif kind in {"polygon", "lasso", "regular_polygon"}:
        points = np.asarray(options.get("points", []), dtype=float)
        if kind == "regular_polygon":
            if not 3 <= int(options.get("sides", 6)) <= 64:
                raise ValueError("Choose between 3 and 64 polygon sides")
            points = _regular_polygon_points(
                (bounds[0] + bounds[2] / 2, bounds[1] + bounds[3] / 2),
                min(bounds[2:]) / 2,
                int(options.get("sides", 6)),
                angle=metadata["angle"],
            )
        if (
            points.ndim != 2
            or points.shape[1] != 2
            or not 3 <= len(points) <= 4096
            or not np.all(np.isfinite(points))
        ):
            raise ValueError("A polygon requires between 3 and 4096 finite vertices")
        if kind == "lasso":
            points = _smooth_lasso_support_points(
                points, image=image, snap=bool(options.get("snap", False))
            )
        mask = _polygon_mask(shape, points)
        metadata["points"] = points.tolist()
    elif kind == "otsu":
        mask, extra = _auto_threshold_mask(image)
        metadata.update(extra)
    elif kind == "wand":
        seed = np.asarray(options.get("seed", [0, 0]), dtype=float)
        if seed.shape != (2,) or not np.all(np.isfinite(seed)):
            raise ValueError("A finite seed is required")
        tolerance = float(options.get("tolerance", 0.08))
        if not 0 <= tolerance <= 1:
            raise ValueError("Wand tolerance must be between zero and one")
        mask, extra = _seeded_region_mask(
            image, (int(seed[1]), int(seed[0])), tolerance_fraction=tolerance
        )
        metadata.update(extra)
    else:
        raise ValueError("Unknown ROI tool")
    if not np.any(mask):
        raise ValueError("The selection contains no native voxels")
    return mask, metadata


def _native_roi_command(workspace, action, options, image=None):
    session = workspace.session
    if image is None:
        image = session.frame_image()
    name = str(options.get("name", "")).strip()
    roi = next((r for r in session.shared_rois if r.name == name), None)
    if action == "compose":
        _compose_rois(workspace, options)
        return
    if action == "empty":
        if not name or roi is not None:
            raise ValueError("Choose a unique ROI name")
        remember_rois(workspace)
        session.add_roi(
            ROI(
                name,
                np.zeros(session.roi_shape, dtype=np.uint16),
                dims=tuple(session.roi_dim_names),
                metadata={"kind": "mask", "consolidated": True},
            )
        )
        session.fit_settings.selected_roi = name
        return
    if action == "undo":
        undo_rois(workspace)
        return
    if action in {"create", "geometry", "draw"}:
        if not name or len(name) > 80:
            raise ValueError("Choose an ROI name")
        if action == "create" and roi is not None:
            raise ValueError("An ROI with this name already exists")
        if action == "geometry" and (roi is None or roi.metadata.get("consolidated")):
            raise ValueError("Only an unconsolidated geometric ROI can be reshaped")
        mode = str(options.get("mode", "add"))
        if action == "draw":
            if roi is None or tuple(roi.mask.shape) != tuple(session.roi_shape):
                raise ValueError("Select an ROI on the current native grid")
            if mode not in {"add", "subtract"}:
                raise ValueError("Choose an add or subtract drawing round")
        plane, metadata = geometry_mask(session, options, image)
        label = int(options.get("label", 1))
        if not 1 <= label <= 65535:
            raise ValueError("Labels must be between 1 and 65535; zero is background")
        remember_rois(workspace)
        covered = _embed(session, plane)
        if action == "draw":
            mask = np.asarray(roi.mask, dtype=np.uint16).copy()
            if mode == "subtract":
                if options.get("label_only"):
                    covered &= mask == label
                mask[covered] = 0
            else:
                mask[covered] = label
            metadata = {"kind": "mask", "consolidated": True}
        else:
            mask = covered.astype(np.uint16) * label
        if roi is not None:
            _commit(session, roi, mask, metadata)
        else:
            session.add_roi(ROI(name, mask, dims=tuple(session.roi_dim_names), metadata=metadata))
        session.fit_settings.selected_roi = name
        if options.get("auto_subtract") and not (action == "draw" and mode == "subtract"):
            _subtract_from_others(session, name, covered)
        return
    if action in {"enable_all", "disable_all", "apply_all"}:
        remember_rois(workspace)
        if action == "enable_all":
            for item in session.shared_rois:
                session.enable_shared_roi(item.name)
        elif action == "disable_all":
            session.clear_rois()
        else:
            session.apply_rois_to_all_experiments()
        return
    if roi is None:
        raise ValueError("Select an existing ROI")
    if action == "consolidate":
        if roi.metadata.get("consolidated"):
            return
        remember_rois(workspace)
        converted = _commit(session, roi, roi.mask, {"kind": "mask", "consolidated": True})
        for key in ("bounds", "points", "angle", "sides"):
            converted.metadata.pop(key, None)
        return
    if tuple(np.shape(roi.mask)) != tuple(session.roi_shape):
        raise ValueError("This ROI belongs to a different native grid")
    if action == "brush":
        points = np.asarray(options.get("points", []), dtype=float)
        if (
            points.ndim != 2
            or points.shape[1] != 2
            or not 1 <= len(points) <= 4096
            or not np.all(np.isfinite(points))
        ):
            raise ValueError("A brush stroke requires finite native coordinates")
        size = float(options.get("size", 5))
        label = int(options.get("label", 1))
        if not 1 <= size <= 512 or not 0 <= label <= 65535:
            raise ValueError("Invalid brush size or label")
        stamp = np.zeros(tuple(session.frame_image().shape), dtype=bool)
        for index, point in enumerate(points):
            previous = points[max(0, index - 1)]
            for x, y in np.linspace(
                previous, point, max(1, int(np.ceil(np.linalg.norm(point - previous))) + 1)
            ):
                stamp |= _brush_mask(stamp.shape, (int(y), int(x)), _brush_radius_from_size(size))
        plane = np.asarray(_plane(session, roi), dtype=np.uint16).copy()
        if options.get("erase") and options.get("label_only"):
            stamp &= plane == label
        plane[stamp] = 0 if options.get("erase") else label
        remember_rois(workspace)
        mask = np.asarray(roi.mask, dtype=np.uint16).copy()
        covered = _embed(session, np.ones(plane.shape, dtype=bool))
        mask[covered] = _embed(session, plane)[covered]
        _commit(session, roi, mask, {"kind": "mask", "consolidated": True})
        if options.get("auto_subtract") and not options.get("erase"):
            _subtract_from_others(session, name, _embed(session, stamp))
        return
    if action == "rename":
        new_name = str(options.get("new_name", "")).strip()
        if not new_name or any(r.name == new_name for r in session.shared_rois):
            raise ValueError("Choose a unique nonempty name")
        remember_rois(workspace)
        workspace.roi_history[-1]["renamed"] = (name, new_name)
        session.rename_roi(name, new_name)
        return
    if action in {"enable", "disable", "remove"}:
        remember_rois(workspace)
        if action == "enable":
            session.enable_shared_roi(name)
        elif action == "disable":
            session.remove_roi(name)
        elif action == "remove":
            session.remove_shared_roi(name)
        return
    package_roi = None
    if "slice_package" in roi.dims and action in {
        "shift",
        "delete_slice",
        "copy_slice",
        "interpolate",
        "grow",
    }:
        package_roi = roi
        package_mask = np.asarray(roi.mask).copy()
        package_index = int(session.active_indices.get("slice_package", 0))
        native = package_mask[package_index]
        native = native[
            tuple(slice(0, session.view_dimensions[dim]) for dim in session.spatial_dim_names)
        ]
        roi = ROI(roi.name, native, dims=tuple(session.spatial_dim_names), metadata=roi.metadata)
    mask = np.asarray(roi.mask).copy()
    metadata = {"kind": "mask", "consolidated": True}
    if action in {"combine", "subtract"}:
        names = options.get("others", [])
        others = [r for r in session.shared_rois if r.name in names and r.name != name]
        if not others or any(r.mask.shape != mask.shape for r in others):
            raise ValueError("Choose other ROIs on the same grid")
        for other in others:
            mask = (
                np.maximum(mask, other.mask)
                if action == "combine"
                else np.where(np.asarray(other.mask) != 0, 0, mask)
            )
    elif action == "shift":
        shifts = options.get("shifts", {})
        if set(shifts) - set(roi.dims):
            raise ValueError("Unknown spatial shift axis")
        mask = _shift_array_zero(
            mask, {tuple(roi.dims).index(dim): int(value) for dim, value in shifts.items()}
        )
    elif action == "delete_slice":
        mask = _delete_mask_slice(
            mask, dims=roi.dims, indices=session.active_indices, plane_dims=session.image_plane_dims
        )
    elif action == "copy_slice":
        axis_name = str(options.get("axis", session.image_slice_dim))
        if axis_name not in roi.dims:
            raise ValueError("Choose a spatial slice axis")
        axis = tuple(roi.dims).index(axis_name)
        source = int(options.get("source", session.active_indices.get(axis_name, 0)))
        targets = options.get("targets", list(range(mask.shape[axis])))
        if not 0 <= source < mask.shape[axis] or any(
            not 0 <= int(t) < mask.shape[axis] for t in targets
        ):
            raise ValueError("Slice index out of range")
        selection = np.take(mask, source, axis=axis).copy()
        for index in targets:
            key = [slice(None)] * mask.ndim
            key[axis] = int(index)
            mask[tuple(key)] = selection
    elif action == "interpolate":
        axis_name = str(options.get("axis", session.image_slice_dim))
        result = np.zeros_like(mask)
        for label in np.unique(mask):
            if label:
                filled = interpolate_mask_along_slices(
                    mask == label,
                    dims=roi.dims,
                    slice_dim=axis_name,
                    interpolation=str(options.get("method", "linear")),
                    extrapolation=str(options.get("extrapolation", "none")),
                )
                result[filled] = label
        result[mask != 0] = mask[mask != 0]
        mask = result
    elif action == "grow":
        radius = float(options.get("radius", 1))
        if mask.ndim != 3 or not np.any(mask):
            raise ValueError("Choose a nonempty 3D ROI")
        if not np.isfinite(radius) or not 0 < radius <= 64:
            raise ValueError("Growth must be greater than zero and at most 64 voxels")
        distance, nearest = distance_transform_edt(mask == 0, return_indices=True)
        covered = (mask == 0) & (distance <= radius)
        mask[covered] = mask[tuple(nearest[:, covered])]
    elif action == "relabel":
        old, new = int(options.get("old", 1)), int(options.get("label", 1))
        if not 1 <= old <= 65535 or not 1 <= new <= 65535 or old not in mask:
            raise ValueError("Choose an existing label and a positive replacement label")
        mask = np.asarray(mask, dtype=np.uint16)
        mask[mask == old] = new
        label_names = dict(roi.metadata.get("label_names", {}))
        if str(old) in label_names:
            previous = label_names.pop(str(old))
            label_names[str(new)] = (
                f"{label_names[str(new)]}; {previous}" if str(new) in label_names else previous
            )
            metadata["label_names"] = label_names
    elif action == "delete_label":
        label = int(options.get("label", 1))
        if not 1 <= label <= 65535 or label not in mask:
            raise ValueError("Choose an existing positive label")
        mask[mask == label] = 0
        label_names = dict(roi.metadata.get("label_names", {}))
        label_names.pop(str(label), None)
        metadata["label_names"] = label_names
    else:
        raise ValueError("Unknown ROI action")
    remember_rois(workspace)
    if package_roi is not None:
        package_mask[package_index] = np.pad(
            mask,
            tuple(
                (0, target - size)
                for target, size in zip(package_mask[package_index].shape, mask.shape)
            ),
        )
        roi, mask = package_roi, package_mask
    _commit(session, roi, mask, metadata)


def roi_command(workspace, action, options, image=None):
    """Edit the displayed grid, retaining shared masks in protected source coordinates."""
    from .cloning import clone_workspace
    from .pipeline import display_session, display_steps, project_roi
    from .view_data import render_image

    steps = display_steps(workspace)
    native_actions = {
        "undo",
        "rename",
        "remove",
        "enable",
        "disable",
        "enable_all",
        "disable_all",
        "apply_all",
        "consolidate",
    }
    transformed = any(
        s["enabled"] and s["kind"] in {"register", "stored_registration"} for s in steps
    )
    if not transformed or action in native_actions:
        return _native_roi_command(workspace, action, options, image)
    session = workspace.session
    work = clone_workspace(workspace)
    work.session = display_session(work)
    projected = {r.name: project_roi(workspace, r, steps=steps) for r in session.shared_rois}
    work.session.shared_rois = list(projected.values())
    work.session.rois = [projected[r.name] for r in session.rois]
    for roi in work.session.shared_rois:
        roi.metadata.update(kind="mask", consolidated=True)
    _native_roi_command(
        work, action, options, render_image(workspace)[0] if image is None else image
    )
    originals = {r.name: r for r in session.shared_rois}
    lookup = {}
    for roi in work.session.shared_rois:
        previous = projected.get(roi.name)
        if previous is not None and np.array_equal(roi.mask, previous.mask):
            lookup[roi.name] = originals[roi.name]
        else:
            native = project_roi(workspace, roi, steps=steps, inverse=True)
            native.metadata.update(kind="mask", consolidated=True)
            lookup[roi.name] = native
    remember_rois(workspace)
    session.shared_rois = list(lookup.values())
    session.rois = [lookup[r.name] for r in work.session.rois]
    for name, state in session.view_states.items():
        if name in work.session.view_states:
            state.rois = [
                lookup[r.name] for r in work.session.view_states[name].rois if r.name in lookup
            ]
    session.fit_settings.selected_roi = work.session.fit_settings.selected_roi
    workspace.pipeline_mask_cache.clear()
    session._save_current_view_state()


def roi_runs(mask):
    """Lossless row runs avoid one DOM node and JSON element per voxel."""
    values = np.asarray(mask)
    padded = np.pad(values, ((0, 0), (1, 1)))
    rows, boundaries = np.nonzero(padded[:, 1:] != padded[:, :-1])
    labels = padded[rows, boundaries + 1]
    keep = (labels[:-1] != 0) & (rows[:-1] == rows[1:])
    return [
        [int(y), int(start), int(end - start), int(label)]
        for y, start, end, label in zip(
            rows[:-1][keep], boundaries[:-1][keep], boundaries[1:][keep], labels[:-1][keep]
        )
    ]
