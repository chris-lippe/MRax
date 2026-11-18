"""Transactional, non-destructive processing pipelines for individual experiments."""

from __future__ import annotations

import copy
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
    from .pipeline_state import image_steps

    return image_steps(pipeline) if pipeline and pipeline["enabled"] else []


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
    entry["extra_metadata"] = {
        **(entry.get("extra_metadata") or {}),
        "registration_space": grid.get("space") or session.selected_experiment,
    }
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
    from .pipeline_state import image_signature, image_steps

    pipeline = current_pipeline(workspace)
    if not pipeline or not pipeline["enabled"]:
        return "original"
    return image_signature(image_steps(pipeline))


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
        "space": session.selected_experiment
        if original
        else (
            (entry.get("extra_metadata") or {}).get("registration_space")
            or session.selected_experiment
        ),
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
    entry["processed"] = dict(entry.get("processed") or {})
    space = state.get("space") or name
    if space != name:
        entry["processed"]["registration_space"] = space
    else:
        entry["processed"].pop("registration_space", None)
    entry["extra_metadata"] = {
        **(entry.get("extra_metadata") or {}),
        "image_unit": state["unit"],
        "spatial_geometry": copy.deepcopy(state["geometry"]),
        "registration_space": space,
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
        session.study.datasets[index].processed = dict(entry["processed"])
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
            "id": "main",
            "source": source,
            "output": source,
            "steps": [],
            "enabled": True,
            "generation": 0,
            "sources": pipeline_sources(workspace),
            "source_processed": dict(session.current_experiment.get("processed") or {}),
        }
    workspace.pipelines[name].setdefault("id", "main")
    workspace.pipeline_sets.setdefault(name, {"main": {"name": "Pipeline 1", "cache": None}})
    session.current_experiment["pipeline_source_grid"] = _grid(workspace.pipelines[name]["source"])
    return workspace.pipelines[name]


def _family(method):
    from .processing import FILTERS

    if method in FILTERS:
        return "denoise"
    return method


def _capture_inputs(workspace, options):
    from .scene import fitting_roi

    references = {}
    for key in ("reference_experiment",):
        name = options.get(key)
        if name and name != workspace.session.selected_experiment:
            if name not in workspace.session.experiment_names:
                raise ValueError("Choose an available reference experiment")
            context = clone_workspace(workspace, readonly_arrays=True)
            context.session.select_experiment(name)
            references[name] = snapshot(context.session)
    mask = None
    roi_names = options.get("roi_names")
    if roi_names is None and options.get("roi_name"):
        roi_names = [options["roi_name"]]
    if roi_names is not None:
        fitting_roi(workspace, roi_names)
        mask = {
            "rois": [
                {
                    "mask": np.asarray(roi.mask).copy(),
                    "dims": tuple(roi.dims or workspace.session.roi_dim_names),
                    "metadata": copy.deepcopy(roi.metadata),
                }
                for roi in workspace.session.rois
                if roi.name in roi_names
            ]
        }
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
            "pending",
        )
        if key in step
    }


def pipeline_info(workspace):
    from .pipeline_choices import active_id, choices
    from .provenance import pipeline_sources

    pipeline = current_pipeline(workspace)
    fit = workspace.fit_results_by_experiment.get(workspace.session.selected_experiment) or {}
    preview = (
        workspace.preview_results_by_experiment.get(workspace.session.selected_experiment) or {}
    )
    stored = (
        [] if pipeline is None else [row["result"] for row in pipeline.get("fits", {}).values()]
    )
    return {
        "id": active_id(workspace),
        "choices": choices(workspace),
        "enabled": True if pipeline is None else pipeline["enabled"],
        "generation": 0 if pipeline is None else pipeline["generation"],
        "signature": signature(workspace),
        "sources": pipeline_sources(workspace),
        "steps": [] if pipeline is None else [
            {**_public_step(step),
             "result_maps": pipeline.get("fits", {}).get(step["id"], {}).get("maps", step.get("result_maps", []))}
            for step in pipeline["steps"]
        ],
        "fits_stale": any(
            result and result.get("pipeline_signature", "original") != signature(workspace)
            for result in (fit, preview, *stored)
        ),
    }


def _grid(state):
    return {
        "dims": dict(state["dims"]),
        "geometry": copy.deepcopy(state["geometry"]),
        "space": state.get("space"),
    }


def _evaluate(workspace, progress, *, start_at=0, run_fits=None, execute=None, original=False):
    from .pipeline_progress import PipelineRun

    report = PipelineRun(
        workspace.session.selected_experiment,
        current_pipeline(workspace)["steps"],
        progress,
        start_at,
        pipeline_id=current_pipeline(workspace).get("id", "main"),
    )
    try:
        _evaluate_steps(
            workspace, progress, report, start_at=start_at, run_fits=run_fits, execute=execute,
            original=original,
        )
        report.finish()
    except BaseException as error:
        report.fail(error)
        raise


def _evaluate_steps(workspace, progress, report, *, start_at=0, run_fits=None, execute=None, original=False):
    from .processing import apply_processing

    session = workspace.session
    name = session.selected_experiment
    pipeline = current_pipeline(workspace)
    install(workspace, pipeline["output"] if start_at else pipeline["source"], update_model=False)
    session.current_experiment["original_mr_image"] = pipeline["source"]["data"]
    previous = {step["id"]: step for step in pipeline.get("applied_steps", [])}
    enabled = []
    for step in pipeline["steps"]:
        if not step["enabled"]:
            continue
        if original and step["kind"] != "fit":
            continue
        if step.get("pending") and execute is not None and step["id"] not in execute:
            if step["id"] in previous:
                enabled.append(copy.copy(previous[step["id"]]))
            continue
        enabled.append(step)
    for index, step in enumerate(enabled):
        if index < start_at:
            continue
        report.start(step)
        if step["kind"] == "fit" and run_fits is not None and step["id"] not in run_fits:
            from .pipeline_state import image_signature

            result = pipeline.get("fits", {}).get(step["id"], {}).get("result", {})
            if result.get("pipeline_signature") == image_signature(enabled[:index]):
                report.complete(unchanged=True)
                continue
        progress(index, max(1, len(enabled)), step["label"])

        def step_progress(done, total, message, index=index):
            report.update(done, total)
            progress(index + min(1, max(0, done / max(1, total))), max(1, len(enabled)), message)

        options = copy.deepcopy(step["options"])
        work = clone_workspace(workspace, readonly_arrays=True)
        work._pipeline_evaluating = True
        if step["kind"] == "reconstruct":
            install(work, pipeline["source"], update_model=False)
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
            mask_values = np.zeros(work.session.roi_shape, dtype=bool)
            prefix = step.get("input_steps", enabled[:index])
            input_space = (work.session.current_experiment.get("extra_metadata") or {}).get(
                "registration_space", name
            )
            for item in frozen.get("rois", [frozen]):
                projected = project_roi(
                    workspace,
                    ROI(
                        "_pipeline_mask",
                        item["mask"],
                        dims=tuple(item["dims"]),
                        metadata=dict(item.get("metadata") or {}),
                    ),
                    steps=prefix,
                )
                if projected.mask.shape != mask_values.shape:
                    raise ValueError("The selected ROI no longer matches this pipeline input grid")
                if projected.metadata.get("registration_space") not in {
                    None,
                    input_space,
                } and name not in projected.metadata.get("compatible_experiments", []):
                    raise ValueError(
                        "The selected ROI is outside this pipeline's registration space. Transfer it or choose another ROI"
                    )
                mask_values |= projected.mask != 0
            mask = ROI(
                "_pipeline_mask",
                mask_values,
                dims=tuple(work.session.roi_dim_names),
                metadata={
                    "registration_space": input_space,
                    "registration_steps": [
                        value["id"] + ":" + str(value["version"])
                        for value in prefix
                        if value["enabled"] and value["kind"] in {"register", "stored_registration"}
                    ],
                },
            )
            work.session.rois = [mask]
            options["roi_names"] = [mask.name]
            options["roi_name"] = mask.name
        if step["kind"] == "fit":
            from .pipeline_blocks import run_fitting_block

            # A fit sees exactly its image-producing prefix, never later operations.
            work.pipelines[name]["enabled"] = not original
            work.pipelines[name]["steps"] = [copy.copy(row) for row in enabled[:index]]
            work.pipelines[name]["applied_steps"] = [
                _public_step(row) for row in enabled[:index]
            ]
            record = run_fitting_block(work, step, step_progress)
            session.replace_fitted_maps(work.session.fitted_maps)
            session.active_map = work.session.active_map
            session.fit_settings = copy.deepcopy(work.session.fit_settings)
            session.frame_mask = copy.deepcopy(work.session.frame_mask)
            workspace.display_settings = work.display_settings
            pipeline.setdefault("fits", {}).update(work.pipelines[name].get("fits", {}))
            pipeline.setdefault("fits", {})[step["id"]] = record
            workspace.fit_results_by_experiment[name] = record["result"]
            step["result_maps"] = record["maps"]
            step["output_grid"] = step["input_grid"]
            step["pending"] = False
            report.complete()
            continue
        if step["kind"] == "process":
            if step.get("noise_mask") is not None:
                frozen = step["noise_mask"]
                work._processing_noise_roi = project_roi(
                    workspace,
                    ROI("_pipeline_noise", frozen["mask"], dims=frozen["dims"]),
                    steps=step.get("input_steps", enabled[:index]),
                )
            summary = apply_processing(work, options, step_progress)
            step.pop("noise_estimate", None)
            if summary.get("noise_estimate") is not None:
                step["noise_estimate"] = summary["noise_estimate"]
        else:
            _apply_alignment(work, step, options, step_progress)
        step["pending"] = False
        output = snapshot(work.session)
        step["output_grid"] = _grid(output)
        install(workspace, output)
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
    if not original:
        pipeline["output"] = snapshot(session)
        pipeline["applied_steps"] = [copy.copy(step) for step in enabled]
        pipeline["enabled"] = True
    pipeline["generation"] += 1
    workspace.pipeline_mask_cache.clear()
    session._save_current_view_state()
    progress(len(enabled), max(1, len(enabled)), "Pipeline applied")


def _commit(workspace, candidate):
    candidate.dirty = True
    workspace.__dict__.update(candidate.__dict__)


def _make_step(candidate, options, *, kind="process", previous=None, enabled=True):
    from .processing import FILTERS, processing_methods
    from .processing_scope import freeze_scope
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
        saved = (candidate.session.current_experiment.get("kspace_metadata") or {}).get(
            "reconstruction_pipeline"
        )
        if saved:
            clean["pipeline"] = copy.deepcopy(saved)
    if kind in {"reconstruct", "fit"}:
        from .pipeline_blocks import block_options

        clean = block_options(candidate, kind, clean)
        if kind == "fit" and previous and "workflow" not in options:
            saved = previous["options"]["workflow"]
            clean["workflow"].update(name=saved["name"], description=saved.get("description", ""))
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
    if kind == "drift" and (clean.get("settings") or {}).get("method") == "framewise":
        label = "Series alignment"
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
    if previous and "map_prefix" in previous:
        step["map_prefix"] = previous["map_prefix"]
    if kind == "fit":
        used = {
            row.get("map_prefix")
            for row in current_pipeline(candidate)["steps"]
            if previous is None or row["id"] != previous["id"]
        }
        prefix, number = label, 2
        while prefix in used:
            prefix, number = f"{label} ({number})", number + 1
        step["map_prefix"] = previous.get("map_prefix", prefix) if previous else prefix
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
            display_steps(candidate, fitted=True, map_name=source[4:])
            if source.startswith("map:")
            else []
        )
    if kind == "process" and method in FILTERS:
        dims = step["input"]["dims"] if step.get("input") else candidate.session.experiment_dims
        freeze_scope(candidate.session, clean, dims)
    return step


def apply_step(workspace, options, progress, *, kind="process", enabled=True):
    from .pipeline_state import primary_step
    from .processing import FILTERS
    from .processing_scope import freeze_scope

    if kind == "process" and options.get("method") in FILTERS:
        dims = workspace.session.experiment_dims
        if str(options.get("input", "image")).startswith("map:"):
            dims = {
                name: size for name, size in dims.items() if name in workspace.session.roi_dim_names
            }
        elif options.get("input") in {"original", "reconstruction"} and current_pipeline(workspace):
            dims = current_pipeline(workspace)["source"]["dims"]
        options = freeze_scope(workspace.session, copy.deepcopy(options), dims)
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
    if kind == "fit" and options.get("settings") is not None:
        from .analysis_workflows import set_fit_settings

        set_fit_settings(candidate, options["settings"])
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
    if (
        kind == "reconstruct"
        and not candidate.session.can_reconstruct_raw_data()
        and candidate.session.can_lazy_load_raw_data(require_acquisition=True)
    ):
        candidate.session.load_raw_data_for_experiment(
            progress_callback=progress, require_acquisition=True
        )
    method = options.get("method") if kind == "process" else kind
    requested = options.get("step_id")
    previous = (
        next((s for s in pipeline["steps"] if s["id"] == requested), None)
        if requested
        else primary_step(pipeline, kind, options)
    )
    if requested and previous is None:
        raise ValueError("The pipeline step no longer exists")
    if previous is not None and previous["kind"] != kind and {previous["kind"], kind} != {"register", "stored_registration"}:
        raise ValueError("Choose the matching operation to edit this pipeline step")
    step = _make_step(candidate, options, kind=kind, previous=previous, enabled=enabled)
    if kind == "fit" and previous is None:
        step["map_prefix"] = ""
    start_at = 0
    if previous is None:
        if len(pipeline["steps"]) >= 64:
            raise ValueError("A pipeline supports up to 64 processing steps")
        if kind == "reconstruct":
            pipeline["steps"].insert(0, step)
        else:
            if pipeline["enabled"] and not any(s.get("pending") for s in pipeline["steps"]):
                start_at = sum(s["enabled"] for s in pipeline["steps"])
            pipeline["steps"].append(step)
    else:
        pipeline["steps"][pipeline["steps"].index(previous)] = step
        if kind == "fit" and not requested:
            # A fit action uses the current image state, including later processing.
            pipeline["steps"] = [row for row in pipeline["steps"] if row["id"] != step["id"]] + [step]
            if pipeline["enabled"] and not any(s.get("pending") for s in pipeline["steps"]):
                start_at = sum(s["enabled"] for s in pipeline["steps"][:-1])
    from .pipeline_blocks import validate_chain

    validate_chain(candidate, pipeline["steps"])
    _evaluate(
        candidate, progress, start_at=start_at,
        run_fits={step["id"]} if kind == "fit" else set(), execute={step["id"]},
        original=kind == "fit" and not pipeline["enabled"],
    )
    candidate.view_label = "Map" if kind == "fit" else "Image"
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
    if kind == "fit":
        from .view_data import json_value

        result = pipeline["fits"][step["id"]]["result"]
        summary.update(
            maps=candidate.session.map_names,
            fit_notes=result.get("fit_notes", []),
            optimizer_name=result.get("optimizer_name"),
            loss_history=json_value(np.asarray(result["loss_history"])),
        )
    _commit(workspace, candidate)
    return summary


def change_pipeline(workspace, options, progress):
    from .pipeline_choices import active_id, change_choice

    if options.get("action") not in {"select"} and options.get("pipeline_id") is not None and options["pipeline_id"] != active_id(workspace):
        raise ValueError("Select this pipeline before editing or running it")
    if options.get("action") in {"create", "duplicate", "select", "rename", "delete"}:
        return change_choice(workspace, options)
    if options.get("action") == "configure":
        from .pipeline_blocks import configure_block

        return configure_block(workspace, options["kind"], options.get("options", {}), progress)
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
        _evaluate(candidate, progress, run_fits=set())
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
        _evaluate(candidate, progress, run_fits=set(), execute=set())
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
        from .pipeline_state import image_steps

        steps = image_steps(pipeline)
    transforms = [
        s for s in steps if s["enabled"] and s["kind"] in {"register", "stored_registration"}
    ]
    if not transforms:
        return roi
    space = roi.metadata.get("registration_space")
    applied = list(roi.metadata.get("registration_steps") or [])
    applicable = []
    for step in reversed(transforms) if inverse else transforms:
        token = step["id"] + ":" + str(step["version"])
        if not inverse and token in applied:
            continue
        source = step["output_grid"] if inverse else step["input_grid"]
        target = step["input_grid"] if inverse else step["output_grid"]
        if space and source.get("space") and space != source["space"]:
            continue
        applicable.append(step)
        space = target.get("space", space)
        if inverse:
            applied = [value for value in applied if value != token]
        else:
            applied.append(token)
    if not applicable:
        return roi
    transforms = list(reversed(applicable)) if inverse else applicable
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
    metadata = dict(roi.metadata)
    if space:
        metadata["registration_space"] = space
    metadata["registration_steps"] = applied
    return ROI(roi.name, cache[key][1], dims=roi.dims, metadata=metadata)


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
    plane_order = None
    if values.ndim == 2 and result["metadata"].get("plane_dims"):
        reverse = inverse != (step["options"].get("direction") == "reference_to_moving")
        source_axes = result["metadata"]["plane_dims" if reverse else "moving_plane_dims"]
        target_axes = result["metadata"]["moving_plane_dims" if reverse else "plane_dims"]
        source_grid = step["output_grid"] if inverse else step["input_grid"]
        source_spatial = [name for name in source_grid["dims"] if name in spatial]
        values = values.transpose([source_spatial.index(dim) for dim in source_axes])
        target = np.zeros(tuple(dims[dim] for dim in target_axes), dtype=np.float32)
        plane_order = [target_axes.index(dim) for dim in spatial]
    if mask:
        sampled = resample_mask_with_transform(values, target, transform, output_dtype=values.dtype)
    elif np.iscomplexobj(values):
        sampled = apply_transform_to_image(
            values.real, target, transform, interpolator=interpolator
        ) + 1j * apply_transform_to_image(values.imag, target, transform, interpolator=interpolator)
    else:
        sampled = apply_transform_to_image(values, target, transform, interpolator=interpolator)
    return sampled.transpose(plane_order) if plane_order is not None else sampled


def _apply_alignment(workspace, step, options, progress):
    from .analysis_workflows import run_alignment, run_drift_step, run_operation

    session = workspace.session
    if step["kind"] == "reconstruct":
        from .pipeline_blocks import resolved_block

        _document, options = resolved_block(workspace, "reconstruct", options)
        options["device"] = step["options"].get("device", "cpu")
        saved_plan = (session.current_experiment.get("kspace_metadata") or {}).get(
            "reconstruction_pipeline"
        )
        if not session.has_reconstruction or (
            options.get("pipeline") is not None and options["pipeline"] != saved_plan
        ):
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
            raise ValueError(
                "Reconstruction output must match the experiment grid. Keep its native output geometry"
            ) from error
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
    reverse = options.get("direction") == "reference_to_moving"
    recorded_grid = row["metadata"].get("moving_grid" if reverse else "reference_grid")
    if recorded_grid:
        target.update(recorded_grid)
    before = snapshot(session)
    if before["dims"].get("slice_package", 1) > 1:
        raise ValueError(
            "Cross-experiment registration requires a single slice package per moving experiment"
        )
    if options.get("moving_source") == "map":
        map_grid = row["metadata"].get("moving_grid") or {}
        if map_grid.get("space", before.get("space")) != before.get("space") or any(
            map_grid.get("dims", before["dims"]).get(dim) != before["dims"][dim]
            for dim in session.spatial_dim_names
        ):
            raise ValueError(
                "The moving map uses an earlier image grid. Use current images or rerun its fit before registration"
            )
    spatial = session.spatial_dim_names
    target_shape = tuple(target["dims"][name] for name in context.session.spatial_dim_names)
    output = {
        **before,
        "dims": {
            k: target_shape[spatial.index(k)] if k in spatial else v
            for k, v in before["dims"].items()
        },
        "geometry": target["geometry"],
        "space": target.get("space", reference),
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


def encode_pipelines(workspace, arrays, value=None):
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
            identity = (value.__array_interface__["data"][0], value.shape, value.strides, value.dtype.str) if isinstance(value, np.ndarray) else id(value)
            if identity in buffers:
                return {"__pipeline_type": "array", "key": buffers[identity]}
            key = "pipeline_" + str(len(arrays))
            arrays[key] = np.asarray(value)
            buffers[identity] = key
            return {"__pipeline_type": "array", "key": key}
        if isinstance(value, dict):
            return {str(k): encode(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [encode(v) for v in value]
        if isinstance(value, np.generic):
            return value.item()
        return value

    return encode(workspace.pipelines if value is None else value)


def decode_pipelines(payload, arrays):
    def decode(value):
        if isinstance(value, dict):
            if value.get("format") == "mrax.workflow":
                from mrax.workflows import WorkflowDocument

                return WorkflowDocument.model_validate(value).model_dump()
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
