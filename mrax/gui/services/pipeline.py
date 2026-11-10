"""Transactional, non-destructive processing pipelines for individual experiments."""

from __future__ import annotations

import copy
import hashlib
import json
from uuid import uuid4

import numpy as np

from mrax.analysis import ROI
from mrax.data_struct import Experiment, Quantity
from mrax.data_struct.dimensions import reshape_dims_to_frames, reshape_frames_to_dims
from mrax.gui.models import make_model
from mrax.gui.session import FitSettings, VisualViewState, _spatial_geometry_from_metadata

from .cloning import clone_workspace


def current_pipeline(workspace):
    return workspace.pipelines.get(workspace.session.selected_experiment)


def _set_dims(session, dims):
    manifest = dict(session.data.get("manifest", {}))
    summary = dict(manifest.get("summary", {}))
    summary["experiments"] = [
        {**row, "dims": dict(dims)} if row.get("name") == session.selected_experiment else row
        for row in summary.get("experiments", [])
    ]
    manifest["summary"] = summary
    session.data["manifest"] = manifest


def display_steps(workspace, *, fitted=False, map_name=None):
    if fitted or workspace.view_label == "Map":
        from .pipeline_blocks import fit_result_for_map

        result = fit_result_for_map(workspace, map_name)
        if "pipeline_steps" in result:
            return result["pipeline_steps"]
    pipeline = current_pipeline(workspace)
    return pipeline["steps"] if pipeline and pipeline["enabled"] else []


def display_session(workspace, *, fitted=False, map_name=None):
    """A metadata-only map view on the grid recorded when those maps were fitted."""
    session = workspace.session
    pipeline = current_pipeline(workspace)
    raw_view = (
        not fitted
        and workspace.view_label
        in {
            "MRax reconstruction",
            "K-space",
            "Vendor vs MRax",
            "Vendor MRax Diff",
        }
        and pipeline is not None
    )
    if not fitted and workspace.view_label != "Map" and not raw_view:
        return session
    from .pipeline_blocks import fit_result_for_map

    result = fit_result_for_map(workspace, map_name)
    grid = pipeline["source"] if raw_view else result.get("pipeline_grid")
    if not grid:
        return session
    context = copy.copy(session)
    if map_name is not None:
        context.active_map = map_name
    context.data = {**session.data, "experiments": dict(session.data["experiments"])}
    entry = dict(session.current_experiment)
    entry["spatial_geometry"] = grid["geometry"]
    entry["mr_image"] = (
        grid["data"]
        if raw_view
        else np.broadcast_to(np.zeros((), dtype=np.float32), tuple(grid["dims"].values()))
    )
    context.data["experiments"][session.selected_experiment] = entry
    _set_dims(context, grid["dims"])
    context.active_indices = {
        key: min(value, max(0, grid["dims"].get(key, session.frame_count) - 1))
        for key, value in session.active_indices.items()
    }
    context._slice_cache = {}
    context.study = None
    steps = [] if raw_view else display_steps(workspace, fitted=True, map_name=map_name)
    if fitted:
        context.rois = [project_roi(workspace, r, steps=steps) for r in session.rois]
        context.shared_rois = [project_roi(workspace, r, steps=steps) for r in session.shared_rois]
    return context


def signature(workspace):
    pipeline = current_pipeline(workspace)
    steps = [] if not pipeline else [s for s in pipeline["steps"] if s["kind"] != "fit"]
    if not pipeline or not pipeline["enabled"] or not any(s["enabled"] for s in steps):
        return "original"
    values = [(s["id"], s["version"], s["enabled"]) for s in steps]
    return hashlib.sha256(json.dumps(values).encode()).hexdigest()


def snapshot(session, *, original=False):
    experiment = session.build_experiment()
    entry = session.current_experiment
    values = entry.get("original_mr_image") if original else entry.get("mr_image")
    if values is None:
        values = entry["mr_image"]
    if isinstance(values, np.ndarray):
        values = values.view()
        values.flags.writeable = False
    geometry = session.build_dataset().spatial_geometry
    return {
        "data": values,
        "dims": dict(experiment.dims),
        "observables": copy.deepcopy(experiment.observables),
        "model": session.current_model_name or "identity",
        "exp_type": experiment.exp_type,
        "unit": (entry.get("extra_metadata") or {}).get("image_unit", ""),
        "geometry": None if geometry is None else geometry.to_metadata(),
    }


def install(workspace, state, *, update_model=True):
    """Install a working buffer without touching the protected source or fit maps."""
    session = workspace.session
    name = session.selected_experiment
    dims = dict(state["dims"])
    values = state["data"]
    if tuple(values.shape) != tuple(dims.values()):
        raise ValueError("Pipeline data does not match its declared dimensions")
    entry = session.current_experiment
    session._ensure_original_image_snapshot(entry)
    if isinstance(values, np.ndarray):
        values = values.view()
        values.flags.writeable = False
    entry["mr_image"] = values
    entry["observables"] = copy.deepcopy(state["observables"])
    entry["spatial_geometry"] = copy.deepcopy(state["geometry"])
    entry["extra_metadata"] = {
        **(entry.get("extra_metadata") or {}),
        "image_unit": state["unit"],
        "spatial_geometry": copy.deepcopy(state["geometry"]),
    }
    pipeline = current_pipeline(workspace)
    if pipeline is not None:
        source_grid = _grid(pipeline["source"])
        entry["pipeline_source_grid"] = source_grid
        entry["extra_metadata"]["pipeline_source_grid"] = source_grid
    fit_grid = (workspace.fit_results_by_experiment.get(name) or {}).get("pipeline_grid")
    if fit_grid is not None:
        entry["fit_grid"] = copy.deepcopy(fit_grid)
        entry["extra_metadata"]["fit_grid"] = copy.deepcopy(fit_grid)
    _set_dims(session, dims)
    if update_model and state["model"] != session.current_model_name:
        selected = session.fit_settings.selected_roi
        session.fit_settings = FitSettings(model_name=state["model"], selected_roi=selected)
    entry["model"] = state["model"]
    if session.study is not None:
        index = list(session.study.names).index(name)
        session.study.experiments[index] = Experiment(
            dims=dims,
            observables=copy.deepcopy(state["observables"]),
            exp_type=state["exp_type"],
        )
        session.study.models[index] = make_model(state["model"])
        session.study.datasets[index].data = values
        session.study.datasets[index].spatial_geometry = _spatial_geometry_from_metadata(
            {"spatial_geometry": state["geometry"]}, spatial_shape=session.spatial_shape
        )
    session.visual_state = VisualViewState(show_original=False)
    session.fit_settings.use_view_state_data = False
    session.set_active_frame(min(session.active_frame, max(0, session.frame_count - 1)))
    for dim, value in list(session.active_indices.items()):
        if dim in dims:
            session.active_indices[dim] = min(value, max(0, dims[dim] - 1))
    if session.frame_mask is not None:
        session.frame_mask.resize(max(1, session.frame_count))
    session._slice_cache.clear()


def ensure_pipeline(workspace):
    session = workspace.session
    if not session.is_loaded:
        raise ValueError("Load an experiment before processing")
    name = session.selected_experiment
    if name not in workspace.pipelines:
        from .provenance import pipeline_sources

        source = snapshot(session, original=True)
        workspace.pipelines[name] = {
            "source": source,
            "output": source,
            "steps": [],
            "enabled": True,
            "generation": 0,
            "sources": pipeline_sources(workspace),
        }
    session.current_experiment["pipeline_source_grid"] = _grid(workspace.pipelines[name]["source"])
    return workspace.pipelines[name]


def _family(method):
    from .processing import FILTERS

    if method in FILTERS:
        return "denoise"
    if method in {"normalize", "dilution"}:
        return "normalize"
    return method


def _capture_inputs(workspace, options):
    from .scene import fitting_roi

    references = {}
    for key in ("reference_experiment",):
        name = options.get(key)
        if name and name != workspace.session.selected_experiment:
            if name not in workspace.session.experiment_names:
                raise ValueError("Choose an available reference experiment")
            context = clone_workspace(workspace)
            context.session.select_experiment(name)
            references[name] = snapshot(context.session)
    mask = None
    roi_names = options.get("roi_names")
    if roi_names is None and options.get("roi_name"):
        roi_names = [options["roi_name"]]
    if roi_names is not None:
        roi = fitting_roi(workspace, roi_names)
        mask = {"mask": np.asarray(roi.mask).copy(), "dims": tuple(roi.dims)}
    return references, mask


def _public_step(step):
    return {
        key: copy.deepcopy(step.get(key))
        for key in (
            "id",
            "version",
            "kind",
            "method",
            "family",
            "label",
            "enabled",
            "options",
            "result_key",
            "input_grid",
            "output_grid",
            "plane_dims",
            "noise_estimate",
            "result_maps",
        )
        if key in step
    }


def pipeline_info(workspace):
    from .provenance import pipeline_sources

    pipeline = current_pipeline(workspace)
    fit = workspace.fit_results_by_experiment.get(workspace.session.selected_experiment) or {}
    preview = (
        workspace.preview_results_by_experiment.get(workspace.session.selected_experiment) or {}
    )
    stored = [] if pipeline is None else [row["result"] for row in pipeline.get("fits", {}).values()]
    return {
        "enabled": True if pipeline is None else pipeline["enabled"],
        "generation": 0 if pipeline is None else pipeline["generation"],
        "signature": signature(workspace),
        "sources": pipeline_sources(workspace),
        "steps": [] if pipeline is None else [_public_step(s) for s in pipeline["steps"]],
        "fits_stale": any(
            result and result.get("pipeline_signature", "original") != signature(workspace)
            for result in (fit, preview, *stored)
        ),
    }


def _grid(state):
    return {"dims": dict(state["dims"]), "geometry": copy.deepcopy(state["geometry"])}


def _evaluate(workspace, progress, *, start_at=0):
    from .pipeline_progress import PipelineRun

    report = PipelineRun(
        workspace.session.selected_experiment,
        current_pipeline(workspace)["steps"],
        progress,
        start_at,
    )
    try:
        _evaluate_steps(workspace, progress, report, start_at=start_at)
        report.finish()
    except BaseException as error:
        report.fail(error)
        raise


def _evaluate_steps(workspace, progress, report, *, start_at=0):
    from .processing import apply_processing

    session = workspace.session
    name = session.selected_experiment
    pipeline = current_pipeline(workspace)
    install(workspace, pipeline["output"] if start_at else pipeline["source"], update_model=False)
    session.current_experiment["original_mr_image"] = pipeline["source"]["data"]
    enabled = [step for step in pipeline["steps"] if step["enabled"]]
    for index, step in enumerate(enabled):
        if index < start_at:
            continue
        report.start(step)
        progress(index, max(1, len(enabled)), step["label"])

        def step_progress(done, total, message, index=index):
            report.update(done, total)
            progress(index + min(1, max(0, done / max(1, total))), max(1, len(enabled)), message)

        options = copy.deepcopy(step["options"])
        work = clone_workspace(workspace, readonly_arrays=True)
        work._pipeline_evaluating = True
        if step.get("input") is not None:
            install(work, step["input"])
            options["input"] = "image"
        step["input_grid"] = _grid(snapshot(work.session))
        for reference, state in step.get("references", {}).items():
            work.session.select_experiment(reference)
            install(work, state)
        work.session.select_experiment(name)
        if step.get("mask") is not None:
            frozen = step["mask"]
            mask = project_roi(
                workspace,
                ROI("_pipeline_mask", frozen["mask"], dims=frozen["dims"]),
                steps=step.get("input_steps", enabled[:index]),
            )
            work.session.rois = [mask]
            options["roi_names"] = [mask.name]
            options["roi_name"] = mask.name
        if step["kind"] == "fit":
            from .pipeline_blocks import run_fitting_block

            # A fit sees exactly its image-producing prefix, never later operations.
            work.pipelines[name]["steps"] = copy.deepcopy(enabled[:index])
            record = run_fitting_block(work, step, step_progress)
            session.replace_fitted_maps(work.session.fitted_maps)
            session.active_map = work.session.active_map
            pipeline.setdefault("fits", {}).update(work.pipelines[name].get("fits", {}))
            pipeline.setdefault("fits", {})[step["id"]] = record
            workspace.fit_results_by_experiment[name] = record["result"]
            step["result_maps"] = record["maps"]
            step["output_grid"] = step["input_grid"]
            report.complete()
            continue
        if step["kind"] == "process":
            if step.get("noise_mask") is not None:
                frozen = step["noise_mask"]
                work._processing_noise_roi = project_roi(
                    workspace, ROI("_pipeline_noise", frozen["mask"], dims=frozen["dims"]),
                    steps=step.get("input_steps", enabled[:index]),
                )
            summary = apply_processing(work, options, step_progress)
            step.pop("noise_estimate", None)
            if summary.get("noise_estimate") is not None:
                step["noise_estimate"] = summary["noise_estimate"]
        else:
            _apply_alignment(work, step, options, step_progress)
        install(workspace, snapshot(work.session))
        if step["kind"] in {"register", "stored_registration"}:
            step["result_data"] = session.current_experiment["mr_image"]
        if step["kind"] == "reconstruct":
            entry = session.current_experiment
            for key in (
                "raw_kspace",
                "processed",
                "acquisition",
                "trajectory",
                "density",
                "sampling_mask",
                "kspace_metadata",
            ):
                if key in work.session.current_experiment:
                    entry[key] = work.session.current_experiment[key]
            if session.study is not None:
                source_dataset = work.session.build_dataset()
                target_dataset = session.study.datasets[list(session.study.names).index(name)]
                for key in (
                    "processed",
                    "acquisition",
                    "trajectory",
                    "density",
                    "sampling_mask",
                    "kspace_metadata",
                    "kspace",
                ):
                    setattr(target_dataset, key, getattr(source_dataset, key))
        workspace.registration_results.update(work.registration_results)
        workspace.drift_results.update(work.drift_results)
        workspace.display_settings = work.display_settings
        for display in workspace.display_settings.values():
            for layer in display.get("scene", {}).get("layers", []):
                if step.get("result_key") and layer.get("registration_key") == step["result_key"]:
                    layer["visible"] = False
        step["output_grid"] = _grid(snapshot(session))
        report.complete()
    pipeline["output"] = snapshot(session)
    pipeline["enabled"] = True
    pipeline["generation"] += 1
    workspace.pipeline_mask_cache.clear()
    session._save_current_view_state()
    progress(len(enabled), max(1, len(enabled)), "Pipeline applied")


def _commit(workspace, candidate):
    candidate.dirty = True
    workspace.__dict__.update(candidate.__dict__)


def _make_step(candidate, options, *, kind="process", previous=None, enabled=True):
    from .processing import processing_methods
    from .scene import fitting_roi

    method = options.get("method") if kind == "process" else kind
    family = "register" if kind == "stored_registration" else _family(method)
    if kind == "stored_registration":
        row = candidate.registration_results.get(options.get("key"))
        if row is None:
            raise ValueError("Choose a stored registration")
    clean = {
        k: v for k, v in options.items() if k not in {"step_id", "name"} and not k.startswith("_")
    }
    if kind not in {"reconstruct", "fit"}:
        clean.setdefault("input", "image")
    if kind == "reconstruct" and "workflow" not in clean and clean.get("pipeline") is None:
        saved = (candidate.session.current_experiment.get("kspace_metadata") or {}).get("reconstruction_pipeline")
        if saved:
            clean["pipeline"] = copy.deepcopy(saved)
    if kind in {"reconstruct", "fit"}:
        from .pipeline_blocks import block_options

        clean = block_options(candidate, kind, clean)
    if kind == "stored_registration":
        clean["reference_experiment"] = row["metadata"][
            "moving_experiment"
            if clean.get("direction") == "reference_to_moving"
            else "reference_experiment"
        ]
    references, mask = _capture_inputs(candidate, clean)
    label = {
        "register": "Registration",
        "stored_registration": "Registration",
        "reconstruct": "Image reconstruction",
        "drift": "Drift correction",
    }.get(kind)
    if kind == "process":
        row = next((m for m in processing_methods(candidate.session) if m["id"] == method), None)
        if row is None:
            raise ValueError("Choose a processing method")
        label = row["label"]
    elif kind in {"reconstruct", "fit"}:
        label = clean["workflow"]["name"]
    step = {
        "id": previous["id"] if previous else uuid4().hex,
        "version": uuid4().hex,
        "kind": kind,
        "method": method,
        "family": family,
        "label": label,
        "enabled": enabled,
        "options": clean,
        "references": references,
        "mask": mask,
    }
    if kind == "fit":
        used = {row.get("map_prefix") for row in current_pipeline(candidate)["steps"] if previous is None or row["id"] != previous["id"]}
        prefix, number = label, 2
        while prefix in used:
            prefix, number = f"{label} ({number})", number + 1
        step["map_prefix"] = (
            previous.get("map_prefix", prefix)
            if previous and previous["label"] == label else prefix
        )
    if kind == "process" and clean.get("noise_roi_names") is not None:
        roi = fitting_roi(candidate, clean["noise_roi_names"])
        step["noise_mask"] = {"mask": np.asarray(roi.mask).copy(), "dims": tuple(roi.dims)}
    source = clean.get("input", "image")
    if kind == "process" and source not in {"image", "view"}:
        from .processing import _source

        values, dims, observables, model, unit, geometry, _rois = _source(candidate, source, "auto")
        step["input"] = {
            "data": values,
            "dims": dims,
            "observables": observables,
            "model": model,
            "unit": unit,
            "geometry": None if geometry is None else geometry.to_metadata(),
            "exp_type": candidate.session.current_experiment.get("exp_type"),
        }
        step["input_steps"] = (
            display_steps(candidate, fitted=True, map_name=source[4:]) if source.startswith("map:") else []
        )
    return step


def apply_step(workspace, options, progress, *, kind="process", append=False, enabled=True):
    if kind == "process" and options.get("method") == "dti":
        from .processing import apply_processing

        candidate = clone_workspace(workspace)
        candidate.session = display_session(candidate, fitted=True)
        summary = apply_processing(candidate, options, progress)
        workspace.session.replace_fitted_maps(candidate.session.fitted_maps)
        workspace.session.active_map = candidate.session.active_map
        workspace.fit_results_by_experiment = candidate.fit_results_by_experiment
        workspace.view_label = "Map"
        return summary
    candidate = clone_workspace(workspace, readonly_arrays=True)
    if kind == "stored_registration":
        row = candidate.registration_results.get(options.get("key"))
        if row is None:
            raise ValueError("Choose a stored registration")
        source = row["metadata"][
            "reference_experiment"
            if options.get("direction") == "reference_to_moving"
            else "moving_experiment"
        ]
        candidate.session.select_experiment(source)
    if kind == "register" and options.get("moving_experiment"):
        candidate.session.select_experiment(options["moving_experiment"])
    pipeline = ensure_pipeline(candidate)
    if kind == "reconstruct" and not candidate.session.can_reconstruct_raw_data() and candidate.session.can_lazy_load_raw_data(require_acquisition=True):
        candidate.session.load_raw_data_for_experiment(progress_callback=progress, require_acquisition=True)
    method = options.get("method") if kind == "process" else kind
    family = "register" if kind == "stored_registration" else _family(method)
    requested = options.get("step_id")
    previous = (
        next((s for s in pipeline["steps"] if s["id"] == requested), None)
        if requested
        else None
        if append
        else next((s for s in pipeline["steps"] if s["family"] == family), None)
    )
    if requested and previous is None:
        raise ValueError("The pipeline step no longer exists")
    if previous is not None and previous["kind"] != kind and previous["family"] != family:
        raise ValueError("Choose the matching operation to edit this pipeline step")
    step = _make_step(candidate, options, kind=kind, previous=previous, enabled=enabled)
    start_at = 0
    if previous is None:
        if len(pipeline["steps"]) >= 64:
            raise ValueError("A pipeline supports up to 64 processing steps")
        if kind == "reconstruct":
            pipeline["steps"].insert(0, step)
        else:
            start_at = sum(s["enabled"] for s in pipeline["steps"])
            pipeline["steps"].append(step)
    else:
        pipeline["steps"][pipeline["steps"].index(previous)] = step
    from .pipeline_blocks import validate_chain

    validate_chain(candidate, pipeline["steps"])
    _evaluate(candidate, progress, start_at=start_at)
    candidate.view_label = "Image"
    summary = {
        "experiment": candidate.session.selected_experiment,
        "step_id": step["id"],
        "method": method,
        "pipeline": pipeline_info(candidate),
    }
    if kind in {"register", "stored_registration", "drift"}:
        summary["key"] = step["result_key"]
        if kind == "drift":
            summary["summary"] = copy.deepcopy(
                candidate.drift_results[step["result_key"]]["summary"]
            )
        else:
            summary["diagnostics"] = copy.deepcopy(
                candidate.registration_results[step["result_key"]]["diagnostics"]
            )
    _commit(workspace, candidate)
    return summary


def change_pipeline(workspace, options, progress):
    if options.get("action") == "replace":
        from .pipeline_editing import apply_pipeline_draft

        return apply_pipeline_draft(workspace, options, progress)
    candidate = clone_workspace(workspace, readonly_arrays=True)
    pipeline = ensure_pipeline(candidate)
    action = options.get("action")
    if action == "enabled":
        enabled = options.get("enabled")
        if not isinstance(enabled, bool):
            raise ValueError("Choose original or processed data")
        pipeline["enabled"] = enabled
        install(candidate, pipeline["output"] if enabled else pipeline["source"])
        pipeline["generation"] += 1
        candidate.view_label = "Image"
    elif action == "clear":
        pipeline["steps"].clear()
        _evaluate(candidate, progress)
        candidate.view_label = "Image"
    else:
        step = next((s for s in pipeline["steps"] if s["id"] == options.get("step_id")), None)
        if step is None:
            raise ValueError("The pipeline step no longer exists")
        if action == "remove":
            pipeline["steps"].remove(step)
        elif action == "toggle" and isinstance(options.get("enabled"), bool):
            step["enabled"] = options["enabled"]
        else:
            raise ValueError("Unknown pipeline change")
        _evaluate(candidate, progress)
    candidate.pipeline_mask_cache.clear()
    candidate.session._save_current_view_state()
    summary = {
        "experiment": candidate.session.selected_experiment,
        "pipeline": pipeline_info(candidate),
    }
    _commit(workspace, candidate)
    return summary


def project_roi(workspace, roi, *, steps=None, inverse=False):
    pipeline = current_pipeline(workspace)
    if pipeline is None:
        return roi
    if steps is None:
        if not pipeline["enabled"] or getattr(workspace, "_pipeline_evaluating", False):
            return roi
        steps = pipeline["steps"]
    transforms = [
        s for s in steps if s["enabled"] and s["kind"] in {"register", "stored_registration"}
    ]
    if not transforms:
        return roi
    first_grid = transforms[-1]["output_grid"] if inverse else transforms[0]["input_grid"]
    spatial_shape = tuple(
        size
        for name, size in first_grid["dims"].items()
        if name in workspace.session.spatial_dim_names
    )
    if tuple(np.shape(roi.mask)) != spatial_shape:
        # Shared ROIs can belong to another experiment with a different grid.
        return roi
    key = (id(roi.mask), tuple((s["id"], s.get("result_key")) for s in transforms), inverse)
    cache = workspace.pipeline_mask_cache
    if key not in cache:
        values = np.asarray(roi.mask)
        for step in reversed(transforms) if inverse else transforms:
            values = _resample(workspace, step, values, mask=True, inverse=inverse)
        if len(cache) >= 64:
            cache.clear()
        cache[key] = (roi.mask, values)
    return ROI(roi.name, cache[key][1], dims=roi.dims, metadata=dict(roi.metadata))


def _resample(workspace, step, values, *, mask=False, inverse=False):
    from mrax.gui.registration import (
        apply_transform_to_image,
        inverse_transform,
        resample_mask_with_transform,
    )

    result = workspace.registration_results[step["result_key"]]
    transform = result["transform"]
    if inverse != (step["options"].get("direction") == "reference_to_moving"):
        transform = inverse_transform(transform)
    grid = step["input_grid"] if inverse else step["output_grid"]
    dims = grid["dims"]
    spatial = [name for name in dims if name in workspace.session.spatial_dim_names]
    shape = tuple(dims[name] for name in spatial)
    target = np.zeros(shape, dtype=np.float32)
    interpolator = "nearest" if mask else result.get("options", {}).get("interpolator", "linear")
    if mask and values.ndim == 3 and transform.GetDimension() == 2:
        plane_dims = step["plane_dims"]
        other = next(d for d in spatial if d not in plane_dims)
        index = int(result["metadata"].get("indices", {}).get(other, 0))
        selection = [slice(None)] * 3
        selection[spatial.index(other)] = index
        order = [d for d in spatial if d != other]
        plane = values[tuple(selection)]
        if order != plane_dims:
            plane = plane.T
        corrected = resample_mask_with_transform(
            plane, np.zeros(plane.shape), transform, output_dtype=values.dtype
        )
        output = values.copy()
        output[tuple(selection)] = corrected if order == plane_dims else corrected.T
        return output
    if values.ndim != transform.GetDimension():
        raise ValueError("The registration transform does not match this mask or image grid")
    if mask:
        return resample_mask_with_transform(values, target, transform, output_dtype=values.dtype)
    if np.iscomplexobj(values):
        return apply_transform_to_image(
            values.real, target, transform, interpolator=interpolator
        ) + 1j * apply_transform_to_image(values.imag, target, transform, interpolator=interpolator)
    result = apply_transform_to_image(values, target, transform, interpolator=interpolator)
    return np.asarray(np.rint(result) if mask else result, dtype=values.dtype if mask else None)


def _apply_alignment(workspace, step, options, progress):
    from .analysis_workflows import run_alignment, run_drift_step, run_operation

    session = workspace.session
    if step["kind"] == "reconstruct":
        from .pipeline_blocks import resolved_block

        _document, options = resolved_block(workspace, "reconstruct", options)
        options["device"] = step["options"].get("device", "cpu")
        saved_plan = (session.current_experiment.get("kspace_metadata") or {}).get("reconstruction_pipeline")
        if not session.has_reconstruction or (options.get("pipeline") is not None and options["pipeline"] != saved_plan):
            result = run_operation(workspace, "_reconstruct_step", options, progress)[1]
            step["label"] = _document.name
        component = options.get("component", "magnitude")
        if component not in {"magnitude", "phase"}:
            raise ValueError("Choose magnitude or phase for reconstruction")
        session.set_reconstruction_component(component)
        data = session.reconstruction_display_array(
            component=component, combined=True, unwrap_phase=False
        )
        state = snapshot(session)
        try:
            state["data"] = reshape_frames_to_dims(data, state["dims"])
        except ValueError as error:
            raise ValueError("Reconstruction output must match the experiment grid; keep its native output geometry") from error
        if component == "phase":
            state.update(unit="rad", model="identity")
        install(workspace, state)
        return
    if step["kind"] == "drift":
        result = run_drift_step(workspace, options, progress)
        step["result_key"] = result["key"]
        session.replace_image_data(workspace.drift_results[result["key"]]["corrected"])
        return
    if step["kind"] == "stored_registration":
        key = options.get("key")
        if key not in workspace.registration_results:
            raise ValueError("Choose a stored registration")
        step["result_key"] = key
        row = workspace.registration_results[key]
        reference = row["metadata"][
            "moving_experiment"
            if options.get("direction") == "reference_to_moving"
            else "reference_experiment"
        ]
    else:
        result = run_alignment(workspace, options, progress)
        step["result_key"] = result["key"]
        row = workspace.registration_results[result["key"]]
        row["pipeline_owned"] = True
        reference = options.get("reference_experiment") or session.selected_experiment
    context = clone_workspace(workspace)
    context.session.select_experiment(reference)
    context.session.set_image_plane(row["metadata"].get("plane", session.active_image_plane))
    step["plane_dims"] = list(context.session.image_plane_dims)
    target = snapshot(context.session)
    before = snapshot(session)
    spatial = session.spatial_dim_names
    target_shape = tuple(target["dims"][name] for name in context.session.spatial_dim_names)
    output = {
        **before,
        "dims": {
            k: target_shape[spatial.index(k)] if k in spatial else v
            for k, v in before["dims"].items()
        },
        "geometry": target["geometry"],
    }
    step["output_grid"] = _grid(output)
    data = np.asarray(reshape_dims_to_frames(session.image_array, session.experiment_dims))
    if row["transform"].GetDimension() == 2 and len(spatial) == 3:
        if target_shape != tuple(before["dims"][d] for d in spatial):
            raise ValueError("Use volume registration to change a three-dimensional grid")
        from mrax.gui.masks import mask_slice_for_display

        corrected = data.copy()
        plane_dims = step["plane_dims"]
        other = next(d for d in spatial if d not in plane_dims)
        selected = int(row["metadata"].get("indices", {}).get(other, 0))
        spatial_axis = spatial.index(other)
        plane_shape = tuple(before["dims"][d] for d in plane_dims)
        from mrax.gui.registration import apply_transform_to_image

        for frame in corrected:
            plane = mask_slice_for_display(
                frame, dims=tuple(spatial), indices={other: selected}, plane_dims=plane_dims
            )
            transformed = apply_transform_to_image(
                plane.real,
                np.zeros(plane_shape),
                row["transform"],
                interpolator=row.get("options", {}).get("interpolator", "linear"),
            )
            if np.iscomplexobj(plane):
                transformed = transformed + 1j * apply_transform_to_image(
                    plane.imag,
                    np.zeros(plane_shape),
                    row["transform"],
                    interpolator=row.get("options", {}).get("interpolator", "linear"),
                )
            selection = [slice(None)] * len(spatial)
            selection[spatial_axis] = selected
            target_dims = [d for d in spatial if d != other]
            frame[tuple(selection)] = (
                transformed if target_dims == list(plane_dims) else transformed.T
            )
        output["geometry"] = before["geometry"]
    else:
        corrected = np.stack([_resample(workspace, step, frame) for frame in data])
    output["data"] = reshape_frames_to_dims(corrected, output["dims"])
    install(workspace, output)


def encode_pipelines(workspace, arrays):
    buffers = {}

    def encode(value):
        if isinstance(value, Quantity):
            return {
                "__pipeline_type": "quantity",
                "unit": value.si_unit,
                "values": encode(value.values),
            }
        if (
            isinstance(value, np.ndarray)
            or hasattr(value, "block_until_ready")
            or isinstance(value, complex)
        ):
            if id(value) in buffers:
                return {"__pipeline_type": "array", "key": buffers[id(value)]}
            key = "pipeline_" + str(len(arrays))
            arrays[key] = np.asarray(value)
            buffers[id(value)] = key
            return {"__pipeline_type": "array", "key": key}
        if isinstance(value, dict):
            return {str(k): encode(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [encode(v) for v in value]
        if isinstance(value, np.generic):
            return value.item()
        return value

    return encode(workspace.pipelines)


def decode_pipelines(payload, arrays):
    def decode(value):
        if isinstance(value, dict):
            if value.get("__pipeline_type") == "array":
                array = arrays[value["key"]]
                array.flags.writeable = False
                return array
            if value.get("__pipeline_type") == "quantity":
                return Quantity(decode(value["values"]), value["unit"])
            return {k: decode(v) for k, v in value.items()}
        if isinstance(value, list):
            return [decode(v) for v in value]
        return value

    return decode(payload)
