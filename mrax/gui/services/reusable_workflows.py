"""Capture portable recipes and apply them transactionally to target experiments."""

from __future__ import annotations

import copy
import dataclasses

import numpy as np

from mrax.gui.registration import DriftCorrectionOptions, RegistrationOptions
from mrax.gui.session import FrameMask, VisualViewState
from mrax.workflows import WorkflowDocument, new_workflow

from .analysis_workflows import set_fit_settings
from .cloning import clone_workspace
from .pipeline import _commit, _evaluate, current_pipeline, ensure_pipeline
from .processing import FILTERS, _settings, processing_methods
from .state import _fit_settings_to_dict

FIT_REFERENCES = (
    "sat_trans_reference_scan",
    "ab_mt_t1_map_ref",
    "ab_mt_b1_map_ref",
    "ab_mt_b0_map_ref",
)


def _json_settings(value):
    if isinstance(value, dict):
        return {key: _json_settings(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_settings(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def resource_choices(workspace):
    """Metadata-only choices. No images, masks, transforms or acquisition values."""
    context = clone_workspace(workspace, readonly_arrays=True)
    choices = []
    for name in context.session.experiment_names:
        context.session.select_experiment(name)
        session = context.session
        values = {
            "experiment": [{"value": n, "label": n} for n in session.experiment_names],
            "roi": [
                {"value": roi.name, "label": roi.name} for roi in session.rois if np.any(roi.mask)
            ],
            "map": [{"value": name, "label": name} for name in session.map_names],
            "sat_trans_reference_scan": [
                {"value": name, "label": name} for name in session.sat_trans_reference_choices()
            ],
        }
        for key in FIT_REFERENCES[1:]:
            values[key] = [
                {"value": row["ref"], "label": row["label"]}
                for row in session.ab_mt_map_choices(key.split("_")[2])
            ]
        choices.append(
            {
                "experiment": name,
                "dims": dict(session.experiment_dims),
                "resources": values,
            }
        )
    return choices


def capture_workflow(workspace, kind, *, name=None, description="", draft=None):
    session = workspace.session
    if not session.is_loaded:
        raise ValueError("Load an experiment before capturing a workflow")
    draft = draft or {}
    bindings, suggested, seen = [], {}, {}

    def bind(kind, value, *, experiment=None):
        if value is None or value == "":
            return None
        key = (kind, value, experiment)
        if key not in seen:
            binding = "input_" + str(len(bindings) + 1)
            seen[key] = binding
            row = {
                "id": binding,
                "kind": kind,
                "label": {
                    "roi": "ROI",
                    "map": "Map",
                    "experiment": "Reference experiment",
                    "sat_trans_reference_scan": "Reference scan",
                    "ab_mt_t1_map_ref": "T1 prior map",
                    "ab_mt_b1_map_ref": "B1 prior map",
                    "ab_mt_b0_map_ref": "B0 prior map",
                }[kind]
                + " "
                + str(len(bindings) + 1),
            }
            if experiment is not None:
                row["experiment"] = experiment
            if kind == "experiment" and value == session.selected_experiment:
                row["default"] = "target"
            bindings.append(row)
            suggested[binding] = value
        return {"$binding": seen[key]}

    steps = []
    if kind == "processing":
        pipeline = current_pipeline(workspace)
        rows = [] if pipeline is None else pipeline["steps"]
        if not rows:
            raise ValueError("Apply processing before capturing a pipeline")
        for step in rows:
            operation = step["kind"]
            if operation in {"reconstruct", "fit"}:
                from .pipeline_blocks import block_document, capture_block_bindings

                child, mapping = block_document(workspace, operation, step["options"])
                assignments = capture_block_bindings(child, mapping, bind, session.selected_experiment)
                steps.append({"id": "step_" + str(len(steps) + 1), "operation": operation,
                              "version": 1, "enabled": step["enabled"],
                              "settings": {"workflow": child.model_dump(), "bindings": assignments}})
                continue
            if operation == "stored_registration":
                raise ValueError(
                    "Stored transforms belong to their images. Capture a fresh registration step instead"
                )
            options = copy.deepcopy(step["options"])
            for key in ("name", "input", "step_id", "moving_experiment"):
                if key != "input" or operation != "process":
                    options.pop(key, None)
            if operation == "process":
                if options.get("input") == "view":
                    if session.visual_state != VisualViewState():
                        raise ValueError(
                            "Capture processing on current image data, without implicit display adjustments"
                        )
                    options["input"] = "image"
                options.setdefault("input", "image")
                method = next(
                    (m for m in processing_methods(session) if m["id"] == options["method"]), None
                )
                if method is None:
                    raise ValueError("The processing method is unavailable")
                options["parameters"] = _settings(method, options.get("parameters", {}))
                if options["method"] in FILTERS:
                    options.setdefault("axes", list(session.spatial_dim_names))
                if options["input"].startswith("map:"):
                    options["input"] = bind("map", options["input"][4:])
            if operation == "register":
                metadata = workspace.registration_results.get(step.get("result_key"), {}).get(
                    "metadata", {}
                )
                options.setdefault("plane", metadata.get("plane", session.active_image_plane))
                source_indices = metadata.get("indices", session.active_indices)
                options.setdefault(
                    "indices",
                    {dim: int(source_indices.get(dim, 0)) for dim in session.spatial_dim_names},
                )
                options["settings"] = {
                    **dataclasses.asdict(RegistrationOptions()),
                    **options.get("settings", {}),
                }
            if operation == "drift":
                options["settings"] = {
                    **dataclasses.asdict(DriftCorrectionOptions()),
                    **options.get("settings", {}),
                }
                if options["settings"].get("reference") == "current":
                    options["settings"]["reference"] = session.active_frame
            reference = options.get("reference_experiment")
            experiment_binding = None
            if reference:
                options["reference_experiment"] = bind("experiment", reference)
                experiment_binding = options["reference_experiment"]["$binding"]
            for key in ("roi_name", "moving_roi", "reference_roi", "moving_map", "reference_map"):
                if options.get(key):
                    resource = "map" if key.endswith("_map") else "roi"
                    options[key] = bind(
                        resource,
                        options[key],
                        experiment=experiment_binding if key.startswith("reference_") else None,
                    )
            for key in ("roi_names", "noise_roi_names"):
                if options.get(key) is not None:
                    options[key] = [bind("roi", value) for value in options[key]]
            steps.append(
                {
                    "id": "step_" + str(len(steps) + 1),
                    "operation": operation,
                    "version": 1,
                    "enabled": step["enabled"],
                    "settings": options,
                }
            )
    elif kind == "reconstruction":
        from .reconstruction import reconstruction_settings

        pipeline = current_pipeline(workspace)
        previous = next(
            (s for s in (pipeline or {}).get("steps", []) if s["kind"] == "reconstruct"), None
        )
        previous_options = reconstruction_settings((previous or {}).get("options", {}))
        component = draft.get("component") or previous_options.get(
            "component", "magnitude"
        )
        plan = draft.get("pipeline") or previous_options.get("pipeline")
        if name is None and plan:
            name = plan["name"]
        steps.append(
            {
                "id": "reconstruction",
                "operation": "reconstruct",
                "version": 1,
                "settings": {"component": component, **({"pipeline": copy.deepcopy(plan)} if plan else {})},
            }
        )
    elif kind == "fitting":
        settings = _fit_settings_to_dict(session.fit_settings)
        settings.update(copy.deepcopy(draft.get("settings", {})))
        settings["model_name"] = settings.get("model_name") or session.current_model_name
        for key in ("output_dir", "selected_roi", "use_view_state_data"):
            settings.pop(key, None)
        for key in FIT_REFERENCES:
            if settings.get(key):
                settings[key] = bind(key, settings[key])
        scene = workspace.display_settings.get(session.selected_experiment, {}).get("scene", {})
        use_roi = draft.get("use_roi", scene.get("fit_region", "rois") == "rois")
        rois = draft.get("roi_names", scene.get("fit_rois"))
        steps.append(
            {
                "id": "fitting",
                "operation": "fit",
                "version": 1,
                "settings": {
                    "settings": settings,
                    "use_roi": use_roi,
                    "roi_names": None if rois is None else [bind("roi", value) for value in rois],
                    "excluded_frames": draft.get(
                        "excluded_frames",
                        [] if session.frame_mask is None else session.frame_mask.excluded_indices,
                    ),
                },
            }
        )
    else:
        raise ValueError("Choose a processing, reconstruction or fitting workflow")
    document = new_workflow(
        kind, _json_settings(steps), name=name, description=description, bindings=bindings
    )
    return document, suggested


def resolve_settings(document, workspace, values):
    session = workspace.session
    if not isinstance(values, dict) or set(values) - {b.id for b in document.bindings}:
        raise ValueError("Unknown workflow binding")
    resource_catalog = None
    resolved = {}

    def resolve_binding(binding_id):
        nonlocal resource_catalog
        if binding_id in resolved:
            return resolved[binding_id]
        binding = next(b for b in document.bindings if b.id == binding_id)
        value = values.get(binding.id)
        if binding.kind == "experiment" and (
            value == "$target" or (not value and binding.default == "target")
        ):
            value = session.selected_experiment
        if not isinstance(value, str) or not value:
            raise ValueError("Assign " + binding.label + " for this target")
        if binding.kind == "experiment":
            if value not in session.experiment_names:
                raise ValueError("The reference experiment is unavailable")
        else:
            if resource_catalog is None:
                resource_catalog = resource_choices(workspace)
            experiment = (
                resolve_binding(binding.experiment)
                if binding.experiment
                else session.selected_experiment
            )
            row = next(r for r in resource_catalog if r["experiment"] == experiment)
            if value not in [r["value"] for r in row["resources"][binding.kind]]:
                raise ValueError(binding.label + " is unavailable for its target experiment")
        resolved[binding.id] = value
        return value

    def replace(value, key=None):
        if key == "workflow":
            return copy.deepcopy(value)
        if isinstance(value, dict):
            if set(value) == {"$binding"}:
                binding_id = value["$binding"]
                resolved_value = resolve_binding(binding_id)
                return "map:" + resolved_value if key == "input" else resolved_value
            return {name: replace(item, name) for name, item in value.items()}
        if isinstance(value, list):
            return [replace(item, key) for item in value]
        return value

    return [(step, replace(step.settings)) for step in document.steps]


def _validate_target(workspace, document, bindings):
    session = workspace.session
    if not session.is_loaded:
        raise ValueError("The target experiment has no image data")
    resolved = resolve_settings(document, workspace, bindings)
    for step, options in resolved:
        if not step.enabled:
            continue
        if "workflow" in options:
            from .pipeline_blocks import resolved_block

            resolved_block(workspace, step.operation, options)
            continue
        if step.operation == "process":
            method = next(
                (m for m in processing_methods(session) if m["id"] == options["method"]), None
            )
            if method is None or method["unavailable"]:
                raise ValueError(method["unavailable"] if method else "Unknown processing method")
            _settings(method, options.get("parameters", {}))
            if options["method"] in FILTERS:
                axes = options.get("axes", session.spatial_dim_names)
                if (
                    not isinstance(axes, list)
                    or not 1 <= len(axes) <= 3
                    or len(set(axes)) != len(axes)
                    or any(a not in session.experiment_dims or a == "slice_package" for a in axes)
                ):
                    raise ValueError("Filter axes are incompatible with this experiment")
                if method.get("axis_count") is not None and len(axes) != method["axis_count"]:
                    raise ValueError(
                        f"{method['label']} needs {method['axis_count']} filter dimensions"
                    )
                if "scope" in options:
                    from .processing_scope import freeze_scope, region

                    context = clone_workspace(workspace, readonly_arrays=True)
                    dims = context.session.experiment_dims
                    if str(options.get("input", "image")).startswith("map:"):
                        dims = {
                            name: size
                            for name, size in dims.items()
                            if name in context.session.roi_dim_names
                        }
                    frozen = freeze_scope(context.session, copy.deepcopy(options), dims)
                    if "slice_package" in frozen["indices"]:
                        context.session.set_active_index(
                            "slice_package", frozen["indices"]["slice_package"]
                        )
                    region(context.session, frozen, dims, context.session.image_geometry())
            if options.get("axis") and options["axis"] not in session.experiment_dims:
                raise ValueError("The acquisition axis is unavailable")
            source = options.get("input", "image")
            if source not in {"image", "original", "reconstruction"} and not source.startswith(
                "map:"
            ):
                raise ValueError("Unknown processing input")
            if source == "reconstruction" and not session.has_reconstruction:
                raise ValueError("This processing workflow requires reconstructed images")
        elif step.operation == "reconstruct":
            from .pipeline_blocks import validate_reconstruction

            validate_reconstruction(workspace, options)
        elif step.operation in {"register", "drift"}:
            cls = RegistrationOptions if step.operation == "register" else DriftCorrectionOptions
            if set(options.get("settings", {})) - {f.name for f in dataclasses.fields(cls)}:
                raise ValueError("Unknown alignment setting")
            cls(**options.get("settings", {}))
            if step.operation == "register":
                if options.get("plane") is not None:
                    session.set_image_plane(options["plane"])
                indices = options.get("indices", {})
                if not isinstance(indices, dict) or any(
                    name not in session.spatial_dim_names
                    or type(value) is not int
                    or not 0 <= value < session.experiment_dims[name]
                    for name, value in indices.items()
                ):
                    raise ValueError("Registration slice indices are incompatible with this target")
                for field in ("moving_frame", "reference_frame"):
                    value = options.get(field, 0)
                    experiment = (
                        options.get("reference_experiment", session.selected_experiment)
                        if field == "reference_frame"
                        else session.selected_experiment
                    )
                    context = clone_workspace(workspace, readonly_arrays=True)
                    context.session.select_experiment(experiment)
                    if type(value) is not int or not 0 <= value < context.session.frame_count:
                        raise ValueError("Registration frame is outside the target acquisition")
        elif step.operation == "fit":
            set_fit_settings(workspace, options["settings"])
            excluded = options.get("excluded_frames", [])
            if (
                any(n >= session.frame_count for n in excluded)
                or len(set(excluded)) >= session.frame_count
            ):
                raise ValueError("Frame exclusions must leave at least one available sample")
            if options.get("use_roi", True):
                from .scene import fitting_roi

                fitting_roi(
                    workspace,
                    options.get("roi_names")
                    if options.get("roi_names") is not None
                    else [roi.name for roi in session.rois if np.any(roi.mask)],
                )
            session.build_fit_inputs()
    return resolved


def validate_workflow_targets(workspace, document, experiments, bindings, *, mode="replace"):
    document = WorkflowDocument.model_validate(
        document.model_dump() if isinstance(document, WorkflowDocument) else document
    )
    if not experiments or len(experiments) > 128 or len(set(experiments)) != len(experiments):
        raise ValueError("Choose one or more distinct target experiments")
    if set(bindings) - set(experiments):
        raise ValueError("Bindings reference an unselected target experiment")
    context = clone_workspace(workspace, readonly_arrays=True)
    results = []
    for name in experiments:
        try:
            if name not in context.session.experiment_names:
                raise ValueError("The target experiment is unavailable")
            context.session.select_experiment(name)
            if document.kind == "processing" and mode == "append":
                existing = (current_pipeline(context) or {}).get("steps", [])
                kinds = [s["kind"] for s in existing] + [s.operation for s in document.steps]
                if [i for i, kind in enumerate(kinds) if kind == "reconstruct"] not in ([], [0]):
                    raise ValueError("A reconstruction block must occur once at the start. Replace the pipeline to load it")
            _validate_target(context, document, bindings.get(name, {}))
            results.append({"experiment": name, "ready": True, "errors": []})
        except (ValueError, TypeError, KeyError, RuntimeError) as exc:
            results.append({"experiment": name, "ready": False, "errors": [str(exc)]})
    return results


def validate_application_options(document, mode, run):
    if mode not in {"replace", "append"} or type(run) is not bool:
        raise ValueError("Choose a supported workflow application mode")
    if document.kind != "processing" and mode != "replace":
        raise ValueError("Append is available only for processing workflows")
    if document.kind != "fitting" and not run:
        raise ValueError("Loading without execution is available only for fitting workflows")


def apply_workflow(
    workspace, document, experiments, bindings, progress, *, mode="replace", run=True, device="cpu"
):
    """All selected experiments in one workspace succeed together or remain untouched."""
    from .analysis_workflows import run_operation
    from .pipeline import _make_step
    from .pipeline_blocks import validate_chain

    document = WorkflowDocument.model_validate(document)
    validate_application_options(document, mode, run)
    checks = validate_workflow_targets(workspace, document, experiments, bindings, mode=mode)
    failures = [row for row in checks if not row["ready"]]
    if failures:
        raise ValueError(
            "; ".join(row["experiment"] + ": " + ", ".join(row["errors"]) for row in failures)
        )
    candidate = clone_workspace(workspace, readonly_arrays=True)
    original_experiment = candidate.session.selected_experiment
    original_view = candidate.view_label
    for index, name in enumerate(experiments):
        candidate.session.select_experiment(name)

        def report(done, total, detail, index=index, name=name):
            progress(
                index + min(1, max(0, done / max(1, total))), len(experiments), name + ": " + detail
            )

        report(0, 1, "Applying " + document.name)
        steps = resolve_settings(document, candidate, bindings.get(name, {}))
        if document.kind == "fitting":
            options = steps[0][1]
            set_fit_settings(candidate, options["settings"])
            mask = FrameMask.all(candidate.session.frame_count)
            mask.exclude(*options.get("excluded_frames", []))
            candidate.session.frame_mask = mask
            candidate.display_settings.setdefault(name, {}).setdefault("scene", {}).update(
                {
                    "fit_region": "rois" if options.get("use_roi", True) else "all",
                    "fit_rois": options.get("roi_names"),
                }
            )
            candidate.session._save_current_view_state()
            if run:
                run_operation(
                    candidate,
                    "full_fit",
                    {
                        **options,
                        "device": device,
                        "roi_names": options.get("roi_names")
                        if options.get("roi_names") is not None
                        else [roi.name for roi in candidate.session.rois if np.any(roi.mask)],
                    },
                    report,
                )
                candidate.fit_results_by_experiment[name]["workflow"] = document.model_dump()
        else:
            pipeline = ensure_pipeline(candidate)
            if document.kind == "reconstruction":
                pipeline["steps"] = [s for s in pipeline["steps"] if s["kind"] != "reconstruct"]
            elif mode == "replace":
                pipeline["steps"] = []
            for step in document.steps:
                options = next(options for item, options in steps if item.id == step.id)
                if step.operation == "fit":
                    options = {**options, "device": device}
                if step.operation == "reconstruct":
                    options = {**options, "view": "mrax_reconstruction", "device": device}
                if step.operation == "register":
                    if options.get("plane") is not None:
                        candidate.session.set_image_plane(options["plane"])
                    candidate.session.active_indices.update(options.get("indices", {}))
                block = _make_step(candidate, options, kind=step.operation, enabled=step.enabled)
                if document.kind == "reconstruction":
                    pipeline["steps"].insert(0, block)
                else:
                    pipeline["steps"].append(block)
            validate_chain(candidate, pipeline["steps"])
            _evaluate(candidate, report)
            candidate.view_label = "Image"
        candidate.session._save_current_view_state()
        report(1, 1, "Workflow applied")
    candidate.session.select_experiment(original_experiment)
    candidate.view_label = (
        "Map"
        if document.kind == "fitting" and run and original_experiment in experiments
        else "Image"
        if document.kind != "fitting" and original_experiment in experiments
        else original_view
    )
    _commit(workspace, candidate)
    return {
        "workflow": document.name,
        "kind": document.kind,
        "experiments": experiments,
        "configured_only": document.kind == "fitting" and not run,
    }
