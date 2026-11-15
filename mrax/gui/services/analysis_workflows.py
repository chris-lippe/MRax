"""Application import, reconstruction, model and fitting workflows."""

from __future__ import annotations

import copy
import dataclasses
import json
import math
from pathlib import Path
from uuid import uuid4

import numpy as np

from mrax.analysis import ROI
from mrax.data_struct.dimensions import reshape_dims_to_frames
from mrax.devices import resolve_device
from mrax.gui.masks import embed_mask_slice, mask_slice_for_display
from mrax.gui.models import make_model, model_parameter_specs
from mrax.gui.registration import (
    DriftCorrectionOptions,
    RegistrationOptions,
    alignment_metrics,
    apply_transform_to_image,
    manual_affine_transform,
    physical_grid_transform,
    registration_method_factory,
    resample_mask_with_transform,
    run_drift_correction,
    run_registration,
    transform_metadata,
)
from mrax.gui.session import FitSettings, GuiSession

from .importing import (
    _apply_observable_dim_compatibility,
    _default_dim_order,
    _detect_import_vendor,
    _observables_from_json_text,
    load_dicom_experiment,
    load_experiment_array,
)
from .registration_state import _failed_linear_drift_groups, _linear_drift_result_summary
from .scene import add_registration_layer, fitting_roi
from .state import _fit_settings_from_dict, _fit_settings_to_dict
from .view_data import display_options, json_value, registration_plane
from .view_state import WorkspaceFitInputs


def set_fit_settings(workspace, values):
    from .fit_setup import fitting_dimensions

    session = workspace.session
    allowed = {field.name for field in dataclasses.fields(FitSettings)}
    if set(values) - allowed:
        raise ValueError("Unknown fit setting")
    data = {**_fit_settings_to_dict(session.fit_settings), **values}
    proposed = _fit_settings_from_dict(data)
    if not 1 <= proposed.num_steps <= 1000000 or not 0 < proposed.step_size <= 10:
        raise ValueError("Choose a positive step count and step size")
    if proposed.optimizer_name not in {
        "auto",
        "adam",
        "adamw",
        "sgd",
        "rmsprop",
        "fista",
        "split_bregman",
    }:
        raise ValueError("Unknown optimizer")
    model = proposed.model_name or session.current_model_name
    specs = model_parameter_specs(
        model,
        pools=proposed.selected_pools or None,
        param_overrides=proposed.param_overrides or None,
        pool_overrides=proposed.pool_overrides or None,
        parameter_constant_dims=proposed.parameter_constant_dims or None,
    )
    for name, override in proposed.param_overrides.items():
        if name not in specs:
            raise ValueError("Unknown model parameter")
        if override.get("bounds") is not None:
            lo, hi = override["bounds"]
            if not np.isfinite([lo, hi]).all() or lo >= hi:
                raise ValueError("Parameter lower bounds must be below upper bounds")
    dimensions = fitting_dimensions(session, model)
    dimension_names = [dimension["name"] for dimension in dimensions]
    for name, entries in proposed.regularizer_configs.items():
        if name not in specs:
            raise ValueError("A regularizer references an unknown parameter")
        for entry in entries:
            if (
                entry.get("type") not in {"smoothness", "tv", "tv2"}
                or not np.isfinite(float(entry.get("weight", 0)))
                or float(entry.get("weight", 0)) < 0
            ):
                raise ValueError("Invalid regularization type or weight")
            if entry.get("axes") is not None:
                axes = []
                for axis in entry["axes"]:
                    if isinstance(axis, str) and axis in dimension_names:
                        axis = dimension_names.index(axis)
                    elif isinstance(axis, (int, np.integer)) and not isinstance(axis, bool):
                        axis = int(axis) if axis >= 0 else len(dimensions) + int(axis)
                    else:
                        raise ValueError("Choose detected regularization dimensions")
                    if not 0 <= axis < len(dimensions):
                        raise ValueError("Choose detected regularization dimensions")
                    if axis not in axes:
                        axes.append(axis)
                if not axes and entry.get("weight", 0) > 0:
                    raise ValueError("Select at least one regularization dimension")
                entry["axes"] = axes
    session.fit_settings = proposed
    session._save_current_view_state()


def update_metadata(workspace, options):
    session = workspace.session
    observables = None
    if "observables" in options:
        observables = _observables_from_json_text(json.dumps(options["observables"]))
        observables, _notes = _apply_observable_dim_compatibility(
            observables,
            session.experiment_dims,
            exp_type=options.get("exp_type", session.current_experiment.get("exp_type")),
            model_name=options.get("model_name", session.current_model_name),
            policy="fail",
        )
    # Validate a candidate session before changing the active workspace.
    candidate = copy.copy(session)
    candidate.data = {
        **session.data,
        "experiments": {name: dict(row) for name, row in session.data["experiments"].items()},
        "summary": copy.deepcopy(session.data.get("summary", [])),
        "manifest": copy.deepcopy(session.data.get("manifest", {})),
    }
    candidate.import_metadata = copy.deepcopy(session.import_metadata)
    # fit_study() otherwise returns the cached Study with the pre-edit observables.
    candidate.study = None
    candidate.update_experiment_metadata(
        exp_type=options.get("exp_type"),
        model_name=options.get("model_name"),
        observables=observables,
        extra_metadata=options.get("metadata"),
    )
    pipeline = workspace.pipelines.get(session.selected_experiment)
    if pipeline:
        # Metadata edits are experiment edits, independent of image processing.
        for state in (pipeline["source"], pipeline["output"]):
            if observables is not None:
                state["observables"] = copy.deepcopy(observables)
            if options.get("model_name") is not None:
                state["model"] = options["model_name"]
            if options.get("exp_type") is not None:
                state["exp_type"] = options["exp_type"]
    workspace.session = candidate


def import_source(workspace, source, options, progress):
    from .source_protection import remember_source

    source = None if source is None else Path(source)
    kind = str(options.get("kind") or "auto")
    session = workspace.session
    if kind in {"auto", "study"} and source is not None:
        if (
            kind == "study"
            or source.suffix.lower() == ".mrax"
            or source.name in {"study_manifest.json", "mrax_package.json"}
            or (
                source.is_dir()
                and (
                    (source / "mrax_package.json").exists()
                    or (source / "study_manifest.json").exists()
                )
            )
            or (
                source.suffix.lower() == ".json"
                and json.loads(source.read_text()).get("format") == "mrax.gui.state"
            )
        ):
            from .workspace_io import load_workspace_path

            imported = load_workspace_path(
                source,
                key=workspace.key,
                extraction_root=Path(options["extraction_root"]),
                reference_root=options.get("reference_root"),
            )
            remember_source(imported, source)
            return imported
        kind = (
            _detect_import_vendor(source)
            if source.is_dir() or source.suffix.lower() == ".seq"
            else (
                "nifti"
                if source.name.lower().endswith((".nii", ".nii.gz"))
                else "numpy"
                if source.suffix.lower() in {".npy", ".npz"}
                else "dicom"
            )
        )
        kind = kind.lower()
    if kind in {"bruker", "pulseq", "dicom-study"} or (
        kind == "dicom" and source is not None and source.is_dir()
    ):
        if session.is_loaded and not options.get("replace"):
            raise ValueError("Import a vendor study into a new workspace")
        replacement = GuiSession()
        if kind == "bruker":
            replacement.import_bruker_study(
                source,
                field_strength_mhz=options.get("field_strength_mhz"),
                pdata_index=options.get("pdata_index") or "1",
                selected_experiments=options.get("selected_experiments"),
                load_raw_kspace=bool(options.get("load_raw_kspace")),
                type_overrides=options.get("type_overrides") or {},
                model_overrides_by_name=options.get("model_overrides_by_name") or {},
                progress_callback=progress,
            )
        elif kind == "pulseq":
            replacement.import_pulseq_study(
                source, options=options.get("vendor_options") or options, progress_callback=progress
            )
        else:
            replacement.load_dicom_study(source, progress_callback=progress)
        workspace.session = replacement
        remember_source(workspace, source)
        workspace.label = replacement.study_name
        for issue in (replacement.import_metadata or {}).get("import_issues", []):
            workspace.diagnostics.append(
                {"level": "warning", "operation": "import", "message": json_value(issue)}
            )
        return workspace
    progress(0, 2, "Reading experiment data")
    if kind == "dicom":
        imported = load_dicom_experiment(source)
        array, dims = imported["data"], imported["dims"]
        metadata, observables = imported["extra_metadata"], imported["observables"]
        exp_type = options.get("exp_type") or imported["exp_type"]
        model_name = options.get("model_name") or imported["model_name"]
    else:
        shape = tuple(int(v) for v in options.get("shape", []))
        dtype = str(options.get("dtype", "float32"))
        parsed_dtype = np.dtype(dtype)
        if parsed_dtype.hasobject or (
            shape and (min(shape) < 1 or math.prod(shape) * parsed_dtype.itemsize > 2 * 1024**3)
        ):
            raise ValueError("Invalid or oversized array shape/dtype")
        array, metadata = load_experiment_array(
            source=kind,
            path=source,
            shape=shape,
            dtype=dtype,
            endian=str(options.get("endian", "native")),
            byte_offset=int(options.get("byte_offset", 0)),
            memory_order=str(options.get("memory_order", "C")),
            npz_key=options.get("npz_key"),
        )
        order = tuple(options.get("dim_order") or _default_dim_order(array.ndim, source=kind))
        if len(order) != array.ndim or len(set(order)) != len(order):
            raise ValueError("Declare a unique dimension for each array axis")
        dims = dict(zip(order, array.shape))
        observables = _observables_from_json_text(json.dumps(options.get("observables") or {}))
        exp_type = str(options.get("exp_type") or "anatomical")
        model_name = str(options.get("model_name") or "identity")
    observables, notes = _apply_observable_dim_compatibility(
        observables,
        dims,
        exp_type=exp_type,
        model_name=model_name,
        policy=str(options.get("observable_policy", "fail")),
    )
    metadata.update(options.get("metadata") or {})
    name = str(options.get("name") or f"Experiment {len(session.experiment_names) + 1}")
    progress(1, 2, "Creating MRax experiment")
    if not np.asarray(array).dtype.isnative:
        array = np.asarray(array).astype(np.asarray(array).dtype.newbyteorder("="))
    session.add_experiment(
        name,
        array,
        dims=dims,
        exp_type=exp_type,
        model_name=model_name,
        observables=observables,
        extra_metadata=metadata,
    )
    workspace.view_label = "Image"
    remember_source(workspace, source)
    for note in notes:
        workspace.diagnostics.append({"level": "info", "operation": "import", "message": note})
    progress(2, 2, "Experiment imported")
    return workspace


def _registration_context(session, name, source, frame, map_name=None, *, workspace=None):
    if name not in session.experiment_names:
        raise ValueError("Choose an available registration experiment")
    if source not in {"image", "map", "reconstruction"}:
        raise ValueError("Unknown registration source")
    session = copy.copy(session)
    session.active_indices = dict(session.active_indices)
    session.view_states = dict(session.view_states)
    session.select_experiment(name)
    if source == "map" and workspace is not None:
        from .pipeline import display_session

        context = copy.copy(workspace)
        context.session = session
        context.view_label = "Map"
        session = display_session(context, fitted=True, map_name=map_name)
    if not 0 <= int(frame) < session.frame_count:
        raise ValueError("Registration frame is out of range")
    session.set_active_frame(int(frame))
    return session


def _registration_source(
    session,
    name,
    source,
    frame,
    map_name=None,
    *,
    volume=False,
    aggregation="frame",
    workspace=None,
):
    session = _registration_context(session, name, source, frame, map_name, workspace=workspace)
    if source == "map":
        if map_name not in session.map_names:
            raise ValueError("Choose a fitted map from the selected experiment")
        data = session.map_array(map_name)
    elif source == "reconstruction":
        data = session.reconstruction_display_array()
    else:
        data = session.image_array
    data = np.asarray(data)
    if source == "map":
        spatial = tuple(session.spatial_dim_names)
        spatial_shape = tuple(session.experiment_dims[name] for name in spatial)
        if data.shape == spatial_shape:
            pass
        elif data.shape == tuple(size for size in spatial_shape if size != 1):
            data = data.reshape(spatial_shape)
        else:
            from mrax.core.loss_builder import _field_metadata

            shape, names, _ = _field_metadata(
                session.build_experiment(),
                model=make_model(session.fit_settings.model_name or session.current_model_name),
            )
            if data.shape != shape:
                raise ValueError(
                    "Map dimensions do not match the selected experiment's parameter grid"
                )
            data = data[
                tuple(
                    slice(None) if name in spatial else session.active_indices.get(name, 0)
                    for name in names
                )
            ]
    else:
        if aggregation == "frame":
            data = np.asarray(reshape_dims_to_frames(data, session.experiment_dims))[int(frame)]
        elif aggregation in {"mean", "median"}:
            from mrax.data_struct.dimensions import reshape_frames_to_dims

            dims = session.experiment_dims
            data = np.asarray(reshape_frames_to_dims(data, dims))
            names = list(dims)
            if "slice_package" in dims:
                axis = names.index("slice_package")
                data = np.take(data, session.active_indices.get("slice_package", 0), axis=axis)
                names.remove("slice_package")
            axes = tuple(
                index for index, dim in enumerate(names) if dim not in session.spatial_dim_names
            )
            reducer = np.nanmean if aggregation == "mean" else np.nanmedian
            data = reducer(np.abs(data), axis=axes) if axes else np.abs(data)
        else:
            raise ValueError("Choose a single frame, mean image or median image")
    return data.copy() if volume else registration_plane(data, session, native_2d=True).copy()


def run_alignment(workspace, options, progress):
    session = workspace.session
    fixed_name = options.get("reference_experiment") or session.selected_experiment
    moving_name = options.get("moving_experiment") or session.selected_experiment
    volume = options.get("scope") == "volume"
    fixed = _registration_source(
        session,
        fixed_name,
        options.get("reference_source", "image"),
        int(options.get("reference_frame", 0)),
        options.get("reference_map"),
        volume=volume,
        aggregation=options.get("reference_aggregation", "frame"),
        workspace=workspace,
    )
    moving = _registration_source(
        session,
        moving_name,
        options.get("moving_source", "image"),
        int(options.get("moving_frame", session.active_frame)),
        options.get("moving_map"),
        volume=volume,
        aggregation=options.get("moving_aggregation", "frame"),
        workspace=workspace,
    )
    if np.iscomplexobj(fixed):
        fixed = np.abs(fixed)
    if np.iscomplexobj(moving):
        moving = np.abs(moving)
    fixed_mask = moving_mask = None
    for which, name, experiment, image, frame in (
        (
            "fixed",
            options.get("reference_roi"),
            fixed_name,
            fixed,
            int(options.get("reference_frame", 0)),
        ),
        (
            "moving",
            options.get("moving_roi"),
            moving_name,
            moving,
            int(options.get("moving_frame", session.active_frame)),
        ),
    ):
        source = options.get("reference_source" if which == "fixed" else "moving_source", "image")
        map_name = options.get("reference_map" if which == "fixed" else "moving_map")
        context = _registration_context(
            session, experiment, source, frame, map_name, workspace=workspace
        )
        roi = next((r for r in context.rois if r.name == name), None)
        if name and roi is None:
            raise ValueError("A registration mask is unavailable")
        if roi is not None:
            mask = np.asarray(roi.mask) != 0
            if volume and roi.dims is not None:
                from mrax.analysis import broadcast_mask

                spatial = tuple(context.spatial_dim_names)
                mask = mask[
                    tuple(
                        slice(None)
                        if dim in spatial
                        else (0 if mask.shape[axis] == 1 else context.active_indices.get(dim, 0))
                        for axis, dim in enumerate(roi.dims)
                    )
                ]
                mask = broadcast_mask(
                    mask,
                    image.shape,
                    mask_dims=tuple(dim for dim in roi.dims if dim in spatial),
                    target_dims=spatial,
                )
            elif not volume:
                mask = mask_slice_for_display(
                    mask,
                    dims=roi.dims,
                    indices=context.active_indices,
                    plane_dims=context.image_plane_dims,
                )
            if mask.shape != image.shape:
                raise ValueError("The registration mask must match its experiment's selected grid")
            if which == "fixed":
                fixed_mask = mask
            else:
                moving_mask = mask
    progress(0, 3, "Preparing registration")
    config = options.get("settings") or {}
    allowed = {field.name for field in dataclasses.fields(RegistrationOptions)}
    if set(config) - allowed:
        raise ValueError("Unknown registration setting")
    registration_options = RegistrationOptions(**config)
    if registration_options.iterations < 1 or not 0 < registration_options.sampling_percentage <= 1:
        raise ValueError("Invalid registration iterations or sampling")
    from .pipeline import snapshot

    reference_context = _registration_context(
        session,
        fixed_name,
        options.get("reference_source", "image"),
        int(options.get("reference_frame", 0)),
        options.get("reference_map"),
        workspace=workspace,
    )
    moving_context = _registration_context(
        session,
        moving_name,
        options.get("moving_source", "image"),
        int(options.get("moving_frame", session.active_frame)),
        options.get("moving_map"),
        workspace=workspace,
    )
    fixed_grid, moving_grid = snapshot(reference_context), snapshot(moving_context)
    pre_transform, spacing = physical_grid_transform(
        fixed_grid,
        moving_grid,
        fixed_axes=reference_context.spatial_dim_names
        if volume
        else reference_context.image_plane_dims,
        moving_axes=moving_context.spatial_dim_names if volume else moving_context.image_plane_dims,
        fixed_frame=reference_context.active_frame,
        moving_frame=moving_context.active_frame,
        fixed_indices=reference_context.active_indices,
        moving_indices=moving_context.active_indices,
    )
    if fixed_grid.get("geometry"):
        from mrax.data_struct import SpatialGeometry

        geometry = SpatialGeometry.from_metadata(fixed_grid["geometry"])
        fixed_grid["geometry"] = dataclasses.replace(
            geometry,
            affine=geometry.affine_for_frame(reference_context.active_frame),
            frame_affines=(),
            frame_dims=(),
            frame_shapes=(),
        ).to_metadata()
    if options.get("manual"):
        if volume:
            raise ValueError("Manual affine controls operate on a two-dimensional plane")
        manual = {k: float(v) for k, v in (options.get("manual_settings") or {}).items()}
        transform = manual_affine_transform(fixed.shape, **manual)
        if pre_transform is not None:
            import SimpleITK as sitk

            transform = sitk.CompositeTransform([pre_transform, transform])
        corrected = apply_transform_to_image(
            moving, fixed, transform, interpolator=registration_options.interpolator
        )
        diagnostics = {"transform_parameters": list(transform.GetParameters())}
    else:
        progress(1, 3, "Optimizing registration")
        result = run_registration(
            fixed,
            moving,
            registration_options,
            fixed_mask=fixed_mask,
            moving_mask=moving_mask,
            moving_pre_transform=pre_transform,
            fixed_spacing=spacing,
        )
        transform, corrected = result.transform, np.asarray(result.corrected)
        diagnostics = dataclasses.asdict(result.diagnostics)
    diagnostics.update(
        pre=alignment_metrics(fixed, moving, mask=fixed_mask)
        if fixed.shape == moving.shape
        else {},
        post=alignment_metrics(fixed, corrected, mask=fixed_mask),
        transform=transform_metadata(transform),
        reference_mask_pixels=None if fixed_mask is None else int(fixed_mask.sum()),
        moving_mask_pixels=None if moving_mask is None else int(moving_mask.sum()),
    )
    key = "registration-" + uuid4().hex[:12]
    metadata = {
        "reference_experiment": fixed_name,
        "moving_experiment": moving_name,
        "reference_source": options.get("reference_source", "image"),
        "moving_source": options.get("moving_source", "image"),
        "reference_frame": int(options.get("reference_frame", 0)),
        "moving_frame": int(options.get("moving_frame", session.active_frame)),
        "reference_aggregation": options.get("reference_aggregation", "frame"),
        "moving_aggregation": options.get("moving_aggregation", "frame"),
        "reference_grid": {
            key: value for key, value in fixed_grid.items() if key in {"dims", "geometry", "space"}
        },
        "moving_grid": {
            key: value for key, value in moving_grid.items() if key in {"dims", "geometry", "space"}
        },
        "reference_map": options.get("reference_map"),
        "moving_map": options.get("moving_map"),
        "plane": reference_context.active_image_plane,
        "plane_dims": list(reference_context.image_plane_dims),
        "moving_plane_dims": list(moving_context.image_plane_dims),
        "moving_indices": dict(moving_context.active_indices),
        "indices": dict(reference_context.active_indices),
        "scope": "volume" if volume else "plane",
    }
    workspace.registration_results[key] = {
        "key": key,
        "kind": "manual" if options.get("manual") else "simpleitk",
        "fixed": fixed,
        "moving": moving,
        "corrected": corrected,
        "transform": transform,
        "metadata": metadata,
        "options": dataclasses.asdict(registration_options),
        "diagnostics": diagnostics,
    }
    display_options(workspace)["registration_key"] = key
    add_registration_layer(workspace, key)
    progress(3, 3, "Registration stored")
    return {"key": key, "diagnostics": json_value(diagnostics)}


def transform_roi(workspace, options):
    result = workspace.registration_results.get(str(options.get("transform", "")))
    roi = next((r for r in workspace.session.shared_rois if r.name == options.get("name")), None)
    if result is None or roi is None:
        raise ValueError("Choose an ROI and stored transform")
    inverse = options.get("direction") == "reference_to_moving"
    reference = result["moving"] if inverse else result["fixed"]
    mask = np.asarray(roi.mask)
    meta = result["metadata"]
    source_dims = meta.get("plane_dims" if inverse else "moving_plane_dims")
    target_dims = meta.get("moving_plane_dims" if inverse else "plane_dims")
    source_name = meta["reference_experiment" if inverse else "moving_experiment"]
    source_grid = meta.get("reference_grid" if inverse else "moving_grid") or {}
    target_grid = meta.get("moving_grid" if inverse else "reference_grid") or {}
    if (
        roi.metadata.get("registration_space")
        and source_grid.get("space")
        and roi.metadata["registration_space"] != source_grid["space"]
    ):
        from .pipeline import project_roi

        work = copy.copy(workspace)
        work.session = _registration_context(
            workspace.session,
            source_name,
            "image",
            meta.get("reference_frame" if inverse else "moving_frame", 0),
        )
        roi = project_roi(work, roi, inverse=True)
        if roi.metadata.get("registration_space") != source_grid["space"]:
            raise ValueError(
                "The ROI is outside this transform's source space. Choose the matching direction or align it first"
            )
        mask = np.asarray(roi.mask)
    if result["transform"].GetDimension() == 2 and (mask.ndim > 2 or source_dims):
        context = copy.copy(workspace.session)
        context.view_states = dict(context.view_states)
        context.active_indices = dict(context.active_indices)
        context.select_experiment(meta["reference_experiment" if inverse else "moving_experiment"])
        context.set_active_frame(meta.get("reference_frame" if inverse else "moving_frame", 0))
        mask = mask_slice_for_display(
            mask,
            dims=roi.dims,
            indices=context.active_indices,
            plane_dims=source_dims or context.image_plane_dims,
        )
    sampled = resample_mask_with_transform(
        mask, reference, result["transform"], inverse=inverse, output_dtype=mask.dtype
    )
    target = result["metadata"]["moving_experiment" if inverse else "reference_experiment"]
    session = workspace.session
    active = session.selected_experiment
    session.select_experiment(target)
    try:
        if session.experiment_dims.get("slice_package", 1) > 1:
            raise ValueError(
                "Label transfer requires a target experiment with a single slice package"
            )
        spatial_dims = tuple(
            dim
            for dim in target_grid.get("dims", session.experiment_dims)
            if dim in {"x", "y", "z"}
        )
        spatial_shape = tuple(
            target_grid.get("dims", session.experiment_dims)[dim] for dim in spatial_dims
        )
        target_indices = meta.get(
            "moving_indices" if inverse else "indices", session.active_indices
        )
        if sampled.ndim == 2 and len(spatial_shape) == 3:
            sampled = embed_mask_slice(
                sampled,
                spatial_shape=spatial_shape,
                spatial_dims=spatial_dims,
                indices=target_indices,
                plane_dims=target_dims or session.image_plane_dims,
            )
        elif sampled.ndim == 2 and target_dims:
            sampled = sampled.transpose([target_dims.index(dim) for dim in spatial_dims])
        if sampled.shape != spatial_shape:
            raise ValueError("Transformed mask does not match the target native grid")
        name = str(options.get("new_name") or roi.name + " transformed")
        existing = {value.name for value in session.shared_rois}
        if options.get("new_name") and name in existing:
            raise ValueError("Choose a unique name for the transferred ROI")
        base, number = name, 2
        while name in existing:
            name, number = f"{base} {number}", number + 1
        session.add_roi(
            ROI(
                name,
                sampled,
                dims=spatial_dims,
                metadata={
                    **roi.metadata,
                    "kind": "mask",
                    "consolidated": True,
                    "source_roi": roi.name,
                    "registration_transform": result["key"],
                    "source_experiment": target,
                    "registration_space": target_grid.get("space", target),
                    "registration_steps": [],
                },
            )
        )
        from .scene import scene_settings

        style = scene_settings(workspace, source_name)["roi_styles"].get(roi.name)
        if style is not None:
            target_scene = scene_settings(workspace, target)
            target_scene["roi_styles"][name] = copy.deepcopy(style)
            workspace.display_settings.setdefault(target, {})["scene"] = target_scene
        session.fit_settings.selected_roi = name
        session._save_current_view_state()
        workspace.view_label = "Image"
    except Exception:
        if active:
            session.select_experiment(active)
        raise


def run_operation(workspace, operation, options, progress):
    session = workspace.session
    if operation == "workflow":
        from .reusable_workflows import apply_workflow

        return workspace, apply_workflow(
            workspace,
            options["document"],
            options["experiments"],
            options.get("bindings", {}),
            progress,
            mode=options.get("mode", "replace"),
            run=options.get("run", True),
            device=options.get("device", "cpu"),
        )
    if operation == "pipeline":
        from .pipeline import apply_step, change_pipeline

        if options.get("action") == "registration":
            return workspace, apply_step(workspace, options, progress, kind="stored_registration")
        return workspace, change_pipeline(workspace, options, progress)
    if operation == "reconstruct" and options.get("view") != "k_space":
        from .pipeline import apply_step

        return workspace, apply_step(workspace, options, progress, kind="reconstruct")
    if operation in {"register", "drift"}:
        from .pipeline import apply_step

        return workspace, apply_step(workspace, options, progress, kind=operation)
    if operation == "process":
        from .processing import run_processing

        return workspace, run_processing(workspace, options, progress)
    device = str(options.get("device", "cpu"))
    if operation in {"preview", "full_fit", "reconstruct", "_reconstruct_step"}:
        resolve_device(device)
    if operation == "import":
        return import_source(workspace, options.get("source"), options, progress), {
            "imported": True
        }
    if operation in {"preview", "full_fit"}:
        if options.get("settings") is not None:
            set_fit_settings(workspace, options["settings"])
        if "excluded_frames" in options:
            from .fit_setup import apply_fit_exclusions

            apply_fit_exclusions(session, options["excluded_frames"])
        roi_name = options.get("roi_name") or session.fit_settings.selected_roi
        selected_mask = None
        use_roi = bool(options.get("use_roi", True))
        if operation == "full_fit" and "roi_names" in options and use_roi:
            selected_mask = fitting_roi(workspace, options["roi_names"])
        kwargs = WorkspaceFitInputs(workspace)._visual_state_fit_kwargs(
            include_roi=operation == "preview" or use_roi,
            roi_override=selected_mask,
        )
        if selected_mask is not None and "input_roi" not in kwargs:
            kwargs["input_roi"] = selected_mask
        if operation == "preview":
            result = session.run_preview_fit(
                roi_name=roi_name,
                num_steps=options.get("steps"),
                device=device,
                progress_callback=progress,
                **kwargs,
            )
            observed, predicted = np.asarray(result["observed"]), np.asarray(result["predicted"])
            if np.iscomplexobj(observed):
                observed = np.abs(observed)
            if np.iscomplexobj(predicted):
                predicted = np.abs(predicted)
            valid = (
                np.isfinite(observed)
                & np.isfinite(predicted)
                & np.asarray(result["included"], dtype=bool)
            )
            if not np.any(valid):
                raise ValueError("The fit has no finite included samples")
            payload = {
                "observed": json_value(observed),
                "predicted": json_value(predicted),
                "included": np.asarray(result["included"], dtype=bool).tolist(),
                "x": json_value(np.asarray(result["x"])),
                "x_name": result["x_name"],
                "x_unit": getattr(
                    session.build_experiment().observables.get(result["x_name"]), "si_unit", None
                ),
                "model": result["model"],
                "roi_name": roi_name or session.roi().name,
                "parameters": {
                    k: float(np.asarray(getattr(v, "values", v)).reshape(-1)[0])
                    for k, v in result["fields"].items()
                },
                "relative_rmse": float(
                    np.sqrt(np.mean((predicted[valid] - observed[valid]) ** 2))
                    / max(np.max(np.abs(observed[valid])), 1e-12)
                ),
            }
            from .pipeline import signature

            payload["pipeline_signature"] = signature(workspace)
            workspace.preview_results_by_experiment[session.selected_experiment] = payload
            return workspace, payload
        result = session.run_current_fit(
            roi_name=roi_name,
            use_roi=use_roi,
            device=device,
            progress_callback=progress,
            **kwargs,
        )
        if not any(
            np.any(np.isfinite(np.asarray(getattr(value, "values", value))))
            for value in result["fields"].values()
        ):
            raise ValueError("The fit produced no finite maps")
        session.replace_fitted_maps(result["fields"])
        workspace.fit_results_by_experiment[session.selected_experiment] = result
        from .pipeline import _public_step, current_pipeline, signature, snapshot

        result["pipeline_signature"] = signature(workspace)
        result["pipeline_grid"] = {
            k: v for k, v in snapshot(session).items() if k in {"dims", "geometry", "unit", "space"}
        }
        session.current_experiment["fit_grid"] = copy.deepcopy(result["pipeline_grid"])
        session.current_experiment.setdefault("extra_metadata", {})["fit_grid"] = copy.deepcopy(
            result["pipeline_grid"]
        )
        pipeline = current_pipeline(workspace)
        result["pipeline_steps"] = (
            [_public_step(s) for s in pipeline["steps"]]
            if pipeline is not None and pipeline["enabled"]
            else []
        )
        result["roi_names"] = (
            list(options["roi_names"])
            if selected_mask is not None
            else ([roi_name] if use_roi else [])
        )
        return workspace, {
            "maps": session.map_names,
            "fit_notes": result.get("fit_notes", []),
            "optimizer_name": result.get("optimizer_name"),
            "loss_history": json_value(np.asarray(result["loss_history"])),
        }
    if operation == "_drift_step":
        fields = dict(options.get("settings") or {})
        method = fields.pop("method", "linear")
        registration_fields = fields.pop("registration", {})
        estimation_dimensions = fields.pop("estimation_dimensions", ())
        if set(fields) - {f.name for f in dataclasses.fields(DriftCorrectionOptions)}:
            raise ValueError("Unknown drift setting")
        drift_options = DriftCorrectionOptions(**fields)
        if not 1 <= drift_options.num_steps <= 1000000 or not np.isfinite(drift_options.step_size) or drift_options.step_size <= 0:
            raise ValueError("Choose a positive drift step count and step size")
        if drift_options.reference == "current":
            experiment = session.build_experiment()
            names = experiment.non_spatial_dim_names
            indices = np.unravel_index(session.active_frame, experiment.non_spatial_shape)
            chosen = names if drift_options.dimensions is None else drift_options.dimensions
            selected = tuple(name for name in names if name in chosen and name != "slice_package")
            current = dict(zip(names, indices, strict=True))
            index = np.ravel_multi_index(tuple(current[name] for name in selected),
                tuple(experiment.dims[name] for name in selected)) if selected else 0
            drift_options = dataclasses.replace(drift_options, reference=int(index))
        mask = session.roi(options.get("roi_name")) if options.get("roi_name") else None
        if options.get("roi_name") and mask is None:
            raise ValueError("Choose an available drift mask")
        progress(0, 2, "Preparing series alignment")
        if method == "linear":
            result = run_drift_correction(
                session.build_experiment(), session.build_dataset(), drift_options,
                mask=None if mask is None else mask.mask, mask_dims=None if mask is None else mask.dims)
            failed = _failed_linear_drift_groups(result)
            summary = _linear_drift_result_summary(result, {})
            result_kind = "mrax_linear_drift"
        elif method == "framewise":
            from mrax.preprocessing.motion import motion_correct_dataset

            registration_options = RegistrationOptions(**registration_fields)
            result = motion_correct_dataset(session.build_experiment(), session.build_dataset(),
                dimensions=drift_options.dimensions, reference=drift_options.reference,
                estimation_dimensions=estimation_dimensions,
                mask=None if mask is None else mask.mask, mask_dims=None if mask is None else mask.dims,
                registration_method=registration_method_factory(registration_options),
                interpolator=registration_options.interpolator,
                normalize_intensity=registration_options.normalize_intensity,
                validate_alignment=registration_options.validate_alignment,
                max_displacement_voxels=registration_options.max_displacement_voxels,
                progress_callback=lambda index, count: progress(index, max(1, count), "Aligning series frames"))
            failed = []
            result_kind = "mrax_framewise_motion"
            summary = {"kind": result_kind, "dimensions": list(result.dimensions),
                "group_dimensions": list(result.group_dimensions), "group_indices": list(result.group_indices),
                "estimation_dimensions": list(result.estimation_dimensions),
                "groups": [{"reference_index": group.reference_index,
                    "diagnostics": [{"metric_value": value.metric_value,
                        "optimizer_stop_condition": value.optimizer_stop_condition,
                        "transform_parameters": list(value.transform_parameters),
                        "iterations": len(value.iteration_metrics)} for value in group.diagnostics]}
                    for group in result.groups]}
        else:
            raise ValueError("Choose frame-wise alignment or linear drift correction")
        if failed or not np.any(np.isfinite(result.corrected)):
            raise ValueError("Drift optimization did not produce valid diagnostics")
        key = "drift-" + uuid4().hex[:12]
        metadata = {"experiment": session.selected_experiment, "dims": session.experiment_dims}
        summary["metadata"] = metadata
        workspace.drift_results[key] = {
            "key": key,
            "kind": result_kind,
            "corrected": np.asarray(result.corrected),
            "metadata": metadata,
            "summary": summary,
        }
        progress(2, 2, "Series alignment stored")
        return workspace, {"key": key, "summary": json_value(summary)}
    if operation in {"reconstruct", "_reconstruct_step"}:
        if options.get("view") != "k_space":
            from .reconstruction import run_reconstruction

            return workspace, run_reconstruction(workspace, options, progress, device=device)
        progress(0, 1, "Loading raw k-space")
        session.prepare_raw_view(
            str(options.get("view", "mrax_reconstruction")),
            progress_callback=progress,
            device=device,
        )
        workspace.view_label = "K-space"
        progress(1, 1, "Reconstruction ready")
        return workspace, {"reconstructed": True}
    raise ValueError("Unknown scientific operation")


def run_drift_step(workspace, options, progress):
    return run_operation(workspace, "_drift_step", options, progress)[1]
