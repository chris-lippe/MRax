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
from mrax.gui.models import model_parameter_specs
from mrax.gui.registration import (
    DriftCorrectionOptions,
    RegistrationOptions,
    alignment_metrics,
    apply_transform_to_image,
    manual_affine_transform,
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


def _registration_source(session, name, source, frame, map_name=None, *, volume=False):
    if name not in session.experiment_names:
        raise ValueError("Choose an available registration experiment")
    if source not in {"image", "map", "reconstruction"}:
        raise ValueError("Unknown registration source")
    active = session.selected_experiment
    session.select_experiment(name)
    try:
        if source == "map":
            if map_name not in session.map_names:
                raise ValueError("Choose a fitted map from the selected experiment")
            data = session.map_array(map_name)
        elif source == "reconstruction":
            data = session.reconstruction_display_array()
        else:
            data = session.image_array
        data = np.asarray(data)
        if source != "map":
            if not 0 <= int(frame) < session.frame_count:
                raise ValueError("Registration frame is out of range")
            data = np.asarray(reshape_dims_to_frames(data, session.experiment_dims))[int(frame)]
        return data.copy() if volume else registration_plane(data, session).copy()
    finally:
        if active:
            session.select_experiment(active)


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
    )
    moving = _registration_source(
        session,
        moving_name,
        options.get("moving_source", "image"),
        int(options.get("moving_frame", session.active_frame)),
        options.get("moving_map"),
        volume=volume,
    )
    if np.iscomplexobj(fixed):
        fixed = np.abs(fixed)
    if np.iscomplexobj(moving):
        moving = np.abs(moving)
    fixed_mask = moving_mask = None
    for which, name, experiment, image in (
        ("fixed", options.get("reference_roi"), fixed_name, fixed),
        ("moving", options.get("moving_roi"), moving_name, moving),
    ):
        context = copy.copy(session)
        context.view_states = dict(session.view_states)
        context.select_experiment(experiment)
        roi = next((r for r in context.rois if r.name == name), None)
        if name and roi is None:
            raise ValueError("A registration mask is unavailable")
        if roi is not None:
            mask = np.asarray(roi.mask) != 0
            if not volume:
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
    if options.get("manual"):
        if volume:
            raise ValueError("Manual affine controls operate on a two-dimensional plane")
        manual = {k: float(v) for k, v in (options.get("manual_settings") or {}).items()}
        transform = manual_affine_transform(fixed.shape, **manual)
        corrected = apply_transform_to_image(
            moving, fixed, transform, interpolator=registration_options.interpolator
        )
        diagnostics = {"transform_parameters": list(transform.GetParameters())}
    else:
        progress(1, 3, "Optimizing registration")
        result = run_registration(
            fixed, moving, registration_options, fixed_mask=fixed_mask, moving_mask=moving_mask
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
    reference_context = copy.copy(session)
    reference_context.view_states = dict(session.view_states)
    reference_context.select_experiment(fixed_name)
    metadata = {
        "reference_experiment": fixed_name,
        "moving_experiment": moving_name,
        "reference_source": options.get("reference_source", "image"),
        "moving_source": options.get("moving_source", "image"),
        "reference_frame": int(options.get("reference_frame", 0)),
        "moving_frame": int(options.get("moving_frame", session.active_frame)),
        "reference_map": options.get("reference_map"),
        "moving_map": options.get("moving_map"),
        "plane": reference_context.active_image_plane,
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
    if result["transform"].GetDimension() == 2 and mask.ndim > 2:
        mask = mask_slice_for_display(
            mask,
            dims=roi.dims,
            indices=workspace.session.active_indices,
            plane_dims=workspace.session.image_plane_dims,
        )
    sampled = resample_mask_with_transform(
        mask, reference, result["transform"], inverse=inverse, output_dtype=mask.dtype
    )
    target = result["metadata"]["moving_experiment" if inverse else "reference_experiment"]
    session = workspace.session
    active = session.selected_experiment
    session.select_experiment(target)
    try:
        if sampled.ndim == 2 and len(session.spatial_shape) == 3:
            sampled = embed_mask_slice(
                sampled,
                spatial_shape=session.spatial_shape,
                spatial_dims=session.spatial_dim_names,
                indices=session.active_indices,
                plane_dims=session.image_plane_dims,
            )
        if sampled.shape != session.spatial_shape:
            raise ValueError("Transformed mask does not match the target native grid")
        session.add_roi(
            ROI(
                str(options.get("new_name") or roi.name + " transformed"),
                sampled,
                dims=tuple(session.spatial_dim_names),
                metadata={
                    **roi.metadata,
                    "kind": "mask",
                    "consolidated": True,
                    "source_roi": roi.name,
                    "registration_transform": result["key"],
                    "source_experiment": target,
                },
            )
        )
    finally:
        if active:
            session.select_experiment(active)


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
            k: v for k, v in snapshot(session).items() if k in {"dims", "geometry", "unit"}
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
        fields = options.get("settings") or {}
        if set(fields) - {f.name for f in dataclasses.fields(DriftCorrectionOptions)}:
            raise ValueError("Unknown drift setting")
        drift_options = DriftCorrectionOptions(**fields)
        if drift_options.reference == "current":
            drift_options = dataclasses.replace(drift_options, reference=session.active_frame)
        mask = session.roi(options.get("roi_name")) if options.get("roi_name") else None
        progress(0, 2, "Fitting MRax linear drift model")
        result = run_drift_correction(
            session.build_experiment(),
            session.build_dataset(),
            drift_options,
            mask=None if mask is None else mask.mask,
        )
        failed = _failed_linear_drift_groups(result)
        if failed or not np.any(np.isfinite(result.corrected)):
            raise ValueError("Drift optimization did not produce valid diagnostics")
        key = "drift-" + uuid4().hex[:12]
        metadata = {"experiment": session.selected_experiment, "dims": session.experiment_dims}
        summary = _linear_drift_result_summary(result, metadata)
        workspace.drift_results[key] = {
            "key": key,
            "kind": "mrax_linear_drift",
            "corrected": np.asarray(result.corrected),
            "metadata": metadata,
            "summary": summary,
        }
        progress(2, 2, "Drift correction stored")
        return workspace, {"key": key, "summary": json_value(summary)}
    if operation in {"reconstruct", "_reconstruct_step"}:
        progress(0, 1, "Loading raw data and reconstructing")
        session.prepare_raw_view(
            str(options.get("view", "mrax_reconstruction")),
            progress_callback=progress,
            device=device,
        )
        workspace.view_label = {
            "mrax_reconstruction": "MRax reconstruction",
            "vendor_vs_mrax": "Vendor vs MRax",
            "vendor_mrax_diff": "Vendor MRax Diff",
            "k_space": "K-space",
        }.get(str(options.get("view")), "MRax reconstruction")
        progress(1, 1, "Reconstruction ready")
        return workspace, {"reconstructed": True}
    raise ValueError("Unknown scientific operation")


def run_drift_step(workspace, options, progress):
    return run_operation(workspace, "_drift_step", options, progress)[1]
