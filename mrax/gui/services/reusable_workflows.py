"""Capture portable recipes and apply them transactionally to target experiments."""

from __future__ import annotations

import copy
import dataclasses
import math

import numpy as np

from mrax.gui.registration import DriftCorrectionOptions, RegistrationOptions
from mrax.gui.session import FrameMask, VisualViewState
from mrax.workflows import WorkflowDocument, new_workflow, single_operation

from .analysis_workflows import set_fit_settings
from .cloning import clone_workspace
from .fit_resources import capture_fit_inputs, resolve_fit_inputs
from .pipeline import _commit, _evaluate, current_pipeline, ensure_pipeline
from .processing import FILTERS, _settings, processing_methods
from .state import _fit_settings_to_dict


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
    context = clone_workspace(workspace, share_sources=True)
    choices = []
    for name in context.session.experiment_names:
        context.session.select_experiment(name)
        session = context.session
        values = {
            "experiment": [{"value": n, "label": n} for n in session.experiment_names],
            "image": [{"value": n, "label": n} for n in session.experiment_names],
            "roi": [
                {"value": roi.name, "label": roi.name} for roi in session.rois if np.any(roi.mask)
            ],
            "map": [{"value": name, "label": name} for name in session.map_names],
        }
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
    if draft.get("operation") == "register":
        moving = (draft.get("options") or {}).get("moving_experiment")
        if moving and moving != session.selected_experiment:
            workspace = clone_workspace(workspace, share_sources=True)
            workspace.session.select_experiment(moving)
            session = workspace.session
    bindings, suggested, seen, produced = [], {}, {}, {}

    def bind(kind, value, *, experiment=None, label=None):
        if value is None or value == "":
            return None
        if (
            kind == "map"
            and value in produced
            and (experiment is None or suggested.get(experiment) == session.selected_experiment)
        ):
            return copy.deepcopy(produced[value])
        key = (kind, value, experiment)
        if key not in seen:
            binding = "input_" + str(len(bindings) + 1)
            seen[key] = binding
            row = {
                "id": binding,
                "kind": kind,
                "label": label
                or (
                    {
                        "roi": "ROI",
                        "map": "Map",
                        "experiment": "Reference experiment",
                        "image": "Image",
                    }[kind]
                    + " "
                    + str(len(bindings) + 1)
                ),
            }
            if any(value["label"] == row["label"] for value in bindings):
                row["label"] += " " + str(len(bindings) + 1)
            if experiment is not None:
                row["experiment"] = experiment
            if kind == "experiment" and value == session.selected_experiment:
                row["default"] = "target"
            bindings.append(row)
            suggested[binding] = value
        return {"$binding": seen[key]}

    steps = []
    if kind in {"processing", "denoising", "registration"}:
        pipeline = current_pipeline(workspace)
        if draft.get("operation"):
            if draft["operation"] not in {"process", "register", "drift"}:
                raise ValueError("Choose a denoising or registration configuration")
            rows = [
                {"kind": draft["operation"], "options": draft.get("options", {}), "enabled": True}
            ]
        else:
            rows = [] if pipeline is None else pipeline["steps"]
            if kind != "processing":
                rows = [
                    s
                    for s in rows
                    if (
                        s["kind"] == "process"
                        if kind == "denoising"
                        else s["kind"] in {"register", "drift"}
                    )
                ][-1:]
        if not rows:
            raise ValueError("Apply processing before capturing a pipeline")
        for step in rows:
            operation = step["kind"]
            if operation in {"reconstruct", "fit"}:
                from .pipeline_blocks import block_document, capture_block_bindings

                child, mapping = block_document(workspace, operation, step["options"])
                assignments = capture_block_bindings(
                    child, mapping, bind, session.selected_experiment
                )

                def flatten(value, assignments=assignments):
                    if isinstance(value, dict):
                        if set(value) == {"$binding"}:
                            return copy.deepcopy(assignments[value["$binding"]])
                        return {key: flatten(item) for key, item in value.items()}
                    if isinstance(value, list):
                        return [flatten(item) for item in value]
                    return value

                block_id = "step_" + str(len(steps) + 1)
                steps.append(
                    {
                        "id": block_id,
                        "operation": operation,
                        "version": 1,
                        "enabled": step["enabled"],
                        "settings": flatten(child.steps[0].settings),
                    }
                )
                if operation == "fit" and step["enabled"]:
                    record = (pipeline or {}).get("fits", {}).get(step.get("id"), {})
                    prefix = record.get("prefix", "") + " / "
                    for field in record.get("maps", []):
                        produced[field] = {
                            "$output": block_id,
                            "port": "map",
                            "name": field[len(prefix) :] if field.startswith(prefix) else field,
                        }
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
                saved = options.get("settings", {})
                if saved.get("method") == "framewise":
                    options["settings"] = {
                        **saved,
                        "registration": {
                            **dataclasses.asdict(RegistrationOptions()),
                            **saved.get("registration", {}),
                        },
                    }
                else:
                    options["settings"] = {**dataclasses.asdict(DriftCorrectionOptions()), **saved}
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
        component = draft.get("component") or previous_options.get("component", "magnitude")
        plan = draft.get("pipeline") or previous_options.get("pipeline")
        if name is None and plan:
            name = plan["name"]
        steps.append(
            {
                "id": "reconstruction",
                "operation": "reconstruct",
                "version": 1,
                "settings": {
                    "component": component,
                    **({"pipeline": copy.deepcopy(plan)} if plan else {}),
                },
            }
        )
    elif kind == "fitting":
        settings = _fit_settings_to_dict(session.fit_settings)
        settings.update(copy.deepcopy(draft.get("settings", {})))
        settings["model_name"] = settings.get("model_name") or session.current_model_name
        for key in ("output_dir", "selected_roi", "use_view_state_data"):
            settings.pop(key, None)
        inputs = capture_fit_inputs(settings, bind, session.selected_experiment)
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
                    "inputs": inputs,
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


def resolve_settings(
    document, workspace, values, *, outputs=None, only=None, target=None, references=None
):
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
            value = target or session.selected_experiment
        if not isinstance(value, str) or not value:
            raise ValueError("Assign " + binding.label + " for this target")
        if binding.kind in {"experiment", "image"}:
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

    def resolve_resource(reference):
        if "$binding" in reference:
            binding = next(b for b in document.bindings if b.id == reference["$binding"])
            value = resolve_binding(binding.id)
            owner = (
                value
                if binding.kind in {"image", "experiment"}
                else resolve_binding(binding.experiment)
                if binding.experiment
                else session.selected_experiment
            )
            return {"kind": binding.kind, "experiment": owner, "name": value}
        if outputs is None:
            return copy.deepcopy(reference)
        result = outputs.get(reference["$output"])
        if result is None:
            raise ValueError("The required workflow output is unavailable")
        owner = result["experiment"]
        if reference["port"] == "image":
            if owner == session.selected_experiment:
                raise ValueError("A named reference image must belong to another experiment")
            if references is not None:
                references[owner] = result["grid"]
            return {"kind": "image", "experiment": owner, "name": owner}
        field = result["maps"].get(reference["name"])
        if field is None:
            raise ValueError("The fitting block did not produce map " + reference["name"])
        return {"kind": "map", "experiment": owner, "name": field}

    def replace(value, key=None):
        if key == "workflow":
            return copy.deepcopy(value)
        if isinstance(value, dict):
            if "$output" in value:
                resource = resolve_resource(value)
                if "$output" in resource:
                    return resource
                if resource["kind"] == "image":
                    return resource["experiment"]
                if resource["experiment"] != session.selected_experiment:
                    raise ValueError("This map input requires a result on the target experiment")
                return "map:" + resource["name"] if key == "input" else resource["name"]
            if set(value) == {"$binding"}:
                binding_id = value["$binding"]
                resolved_value = resolve_binding(binding_id)
                return "map:" + resolved_value if key == "input" else resolved_value
            return {name: replace(item, name) for name, item in value.items()}
        if isinstance(value, list):
            return [replace(item, key) for item in value]
        return value

    result = []
    for step in document.steps:
        if only is not None and step.id != only:
            continue
        options = replace({key: value for key, value in step.settings.items() if key != "inputs"})
        if step.operation == "fit":
            options["settings"] = resolve_fit_inputs(
                options["settings"], step.settings.get("inputs", {}), resolve_resource
            )
        result.append((step, options))
    return result


def _validate_target(workspace, document, bindings, *, resolved=None):
    session = workspace.session
    if not session.is_loaded:
        raise ValueError("The target experiment has no image data")
    resolved = resolve_settings(document, workspace, bindings) if resolved is None else resolved
    for step, options in resolved:
        if not step.enabled:
            continue
        if _has_output(options):
            # Output resources are checked against the actual preceding result at execution.
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
            settings = options.get("settings", {})
            if step.operation == "drift":
                from mrax.preprocessing.motion import _normalize_registration_dimensions

                available = tuple(
                    name
                    for name in session.experiment_dims
                    if name not in session.spatial_dim_names
                )
                dimensions = _normalize_registration_dimensions(
                    settings.get("dimensions"), available
                )
                if math.prod(session.experiment_dims[name] for name in dimensions) < 2:
                    raise ValueError("Series alignment requires at least two acquisition frames")
                groups = tuple(name for name in available if name not in dimensions)
                _normalize_registration_dimensions(
                    settings.get("estimation_dimensions", []), groups
                )
            if step.operation == "drift" and settings.get("method") == "framewise":
                if set(settings) - {
                    "method",
                    "dimensions",
                    "estimation_dimensions",
                    "reference",
                    "registration",
                }:
                    raise ValueError("Unknown series alignment setting")
                cls, settings = RegistrationOptions, settings.get("registration", {})
            else:
                cls = (
                    RegistrationOptions if step.operation == "register" else DriftCorrectionOptions
                )
                settings = {key: value for key, value in settings.items() if key != "method"}
            if set(settings) - {f.name for f in dataclasses.fields(cls)}:
                raise ValueError("Unknown alignment setting")
            cls(**settings)
            if step.operation == "register":
                if (
                    options.get("reference_experiment", session.selected_experiment)
                    == session.selected_experiment
                    and options.get("moving_source", "image")
                    == options.get("reference_source", "image")
                    and options.get("moving_frame", session.active_frame)
                    == options.get("reference_frame", 0)
                    and options.get("moving_map") == options.get("reference_map")
                    and options.get("moving_aggregation", "frame")
                    == options.get("reference_aggregation", "frame")
                    and not options.get("manual")
                ):
                    raise ValueError("Choose a different reference experiment or frame")
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
                    value = options.get(
                        field, session.active_frame if field == "moving_frame" else 0
                    )
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
            from mrax.core.fit_sanity import check_fit_sanity

            experiment, model, dataset, study = session.build_fit_inputs()
            report = check_fit_sanity(
                experiment, model, dataset, study_globals=study.study_globals, check_data=False
            )
            errors = (
                report.errors
                + ["Missing observable: " + name for name in report.missing_observables]
                + ["Provide one of: " + ", ".join(group) for group in report.missing_any_of]
            )
            if errors:
                raise ValueError(". ".join(errors))
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
            seen_targets = set()
            for step in document.steps:
                context.session.select_experiment(
                    _step_target(step, bindings.get(name, {}), document, name)
                )
                target_name = context.session.selected_experiment
                if target_name not in seen_targets:
                    pipeline = current_pipeline(context)
                    if pipeline and (
                        (mode == "replace" and not single_operation(document, "fit"))
                        or single_operation(document, "reconstruct")
                    ):
                        from .pipeline import install

                        install(context, pipeline["source"], update_model=False)
                    seen_targets.add(target_name)
                resolved = resolve_settings(
                    document, context, bindings.get(name, {}), only=step.id, target=name
                )
                _validate_target(context, document, bindings.get(name, {}), resolved=resolved)
            results.append({"experiment": name, "ready": True, "errors": []})
        except (ValueError, TypeError, KeyError, RuntimeError) as exc:
            results.append({"experiment": name, "ready": False, "errors": [str(exc)]})
    return results


def validate_application_options(document, mode, run):
    if mode not in {"replace", "append"} or type(run) is not bool:
        raise ValueError("Choose a supported workflow application mode")
    if not (single_operation(document, "fit") or single_operation(document, "reconstruct")) and not run:
        raise ValueError("Loading without execution requires a fitting or reconstruction workflow")


def _has_output(value):
    if isinstance(value, dict):
        return "$output" in value or any(_has_output(item) for item in value.values())
    return isinstance(value, (list, tuple)) and any(_has_output(item) for item in value)


def _output_dependencies(document):
    needed = set()

    def inspect(value):
        if isinstance(value, dict):
            if "$output" in value:
                needed.add(value["$output"])
            for item in value.values():
                inspect(item)
        elif isinstance(value, list):
            for item in value:
                inspect(item)

    for step in document.steps:
        inspect(step.settings)
    return needed


def _step_target(step, bindings, document, target):
    if step.target is None:
        return target
    binding = next(b for b in document.bindings if b.id == step.target["$binding"])
    value = bindings.get(binding.id)
    if value == "$target" or (not value and binding.default == "target"):
        return target
    if not isinstance(value, str) or not value:
        raise ValueError("Assign " + binding.label + " for this target")
    return value


def apply_workflow(
    workspace, document, experiments, bindings, progress, *, mode="replace", run=True, device="cpu"
):
    """All selected experiments in one workspace succeed together or remain untouched."""
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
    if len(document.steps) == 1:
        return _apply_single_workflow(
            workspace, document, experiments, bindings, progress, run=run, device=device
        )
    candidate = clone_workspace(workspace, readonly_arrays=True)
    original_experiment = candidate.session.selected_experiment
    original_view = candidate.view_label
    changed_targets, fitted_targets = set(), set()
    needed_outputs = _output_dependencies(document)
    for index, name in enumerate(experiments):
        candidate.session.select_experiment(name)

        def report(done, total, detail, index=index, name=name):
            progress(
                index + min(1, max(0, done / max(1, total))), len(experiments), name + ": " + detail
            )

        report(0, 1, "Applying " + document.name)
        from .pipeline import install, snapshot

        outputs, initialized, previous, consumed = {}, set(), {}, {}
        for step in document.steps:
            target = _step_target(step, bindings.get(name, {}), document, name)
            candidate.session.select_experiment(target)
            changed_targets.add(target)
            pipeline = ensure_pipeline(candidate)
            pipeline["enabled"] = True
            if target not in initialized:
                previous[target] = list(pipeline["steps"])
                consumed[target] = set()
                if mode == "replace":
                    pipeline["steps"] = []
                    pipeline["applied_steps"] = []
                    install(candidate, pipeline["source"], update_model=False)
                initialized.add(target)
            frozen = {}
            options = resolve_settings(
                document,
                candidate,
                bindings.get(name, {}),
                outputs=outputs,
                only=step.id,
                target=name,
                references=frozen,
            )[0][1]
            context = clone_workspace(candidate, readonly_arrays=True)
            for reference, state in frozen.items():
                context.session.select_experiment(reference)
                install(context, state)
            context.session.select_experiment(target)
            _validate_target(context, document, {}, resolved=[(step, options)])
            if step.operation == "fit":
                options = {**options, "device": device}
            if step.operation == "reconstruct":
                options = {**options, "view": "mrax_reconstruction", "device": device}
            if step.operation == "register":
                if options.get("plane") is not None:
                    candidate.session.set_image_plane(options["plane"])
                candidate.session.active_indices.update(options.get("indices", {}))
            retained = {s["id"] for s in pipeline["steps"]}
            prior = next(
                (
                    s
                    for s in previous[target]
                    if s.get("workflow_step_id") == step.id
                    and s.get("workflow_name") == document.name
                    and s["id"] not in consumed[target]
                ),
                None,
            )
            if prior is None:
                from .pipeline import _family

                family = _family(options.get("method") if step.operation == "process" else step.operation)
                prior = next((s for s in previous[target] if s["family"] == family and s["id"] not in consumed[target]), None)
            if prior is not None:
                consumed[target].add(prior["id"])
            block = _make_step(
                context, options, kind=step.operation, enabled=step.enabled, previous=prior
            )
            block["workflow_step_id"], block["workflow_name"] = step.id, document.name
            block["references"].update(frozen)
            if step.operation == "fit":
                block["map_prefix"] = step.id
            start_at = (
                sum(s["enabled"] for s in pipeline["steps"])
                if pipeline["enabled"] and not any(s.get("pending") for s in pipeline["steps"])
                else 0
            )
            if prior is not None and prior["id"] in retained:
                pipeline["steps"] = [block if s["id"] == prior["id"] else s for s in pipeline["steps"]]
                start_at = 0
            elif step.operation == "reconstruct":
                pipeline["steps"].insert(sum(s["kind"] == "reconstruct" for s in pipeline["steps"]), block)
                start_at = 0
            else:
                pipeline["steps"].append(block)
            _evaluate(candidate, report, start_at=start_at,
                      run_fits={block["id"]} if step.operation == "fit" else set(),
                      execute={block["id"]})
            if step.operation == "fit" and step.enabled:
                fitted_targets.add(target)
            record = pipeline.get("fits", {}).get(block["id"], {})
            if step.id in needed_outputs:
                outputs[step.id] = {
                    "experiment": target,
                    "grid": snapshot(candidate.session) if step.operation != "fit" else None,
                    "maps": {
                        key: record.get("prefix", step.id) + " / " + key
                        for key in record.get("result", {}).get("fields", {})
                    },
                }
        candidate.session.select_experiment(name)
        candidate.view_label = "Image"
        candidate.session._save_current_view_state()
        report(1, 1, "Workflow applied")
    candidate.session.select_experiment(original_experiment)
    for target in changed_targets:
        candidate.session.select_experiment(target)
        validate_chain(candidate, candidate.pipelines[target]["steps"])
    candidate.session.select_experiment(original_experiment)
    candidate.view_label = (
        "Map"
        if document.kind == "fitting" and run and original_experiment in fitted_targets
        else "Image"
        if run and original_experiment in changed_targets
        else original_view
    )
    _commit(workspace, candidate)
    return {
        "workflow": document.name,
        "kind": document.kind,
        "experiments": experiments,
        "configured_only": False,
    }


def _apply_single_workflow(workspace, document, experiments, bindings, progress, *, run, device):
    from .pipeline import apply_step
    from .pipeline_blocks import configure_block

    candidate = clone_workspace(workspace, readonly_arrays=True)
    original = candidate.session.selected_experiment
    view = candidate.view_label
    step = document.steps[0]
    for index, name in enumerate(experiments):
        target = _step_target(step, bindings.get(name, {}), document, name)
        candidate.session.select_experiment(target)
        options = resolve_settings(document, candidate, bindings.get(name, {}), target=name)[0][1]
        if step.operation == "fit":
            set_fit_settings(candidate, options["settings"])
            mask = FrameMask.all(candidate.session.frame_count)
            mask.exclude(*options.get("excluded_frames", []))
            candidate.session.frame_mask = mask
            candidate.display_settings.setdefault(target, {}).setdefault("scene", {}).update({
                "fit_region": "rois" if options.get("use_roi", True) else "all",
                "fit_rois": options.get("roi_names"),
            })
        if step.operation in {"fit", "reconstruct"}:
            options = {
                "workflow": document.model_dump(), "bindings": bindings.get(name, {}),
                "device": device,
            }

        def report(done, total, detail, index=index, target=target):
            progress(index + min(1, max(0, done / max(1, total))), len(experiments), target + ": " + detail)

        if run:
            apply_step(candidate, options, report, kind=step.operation, enabled=step.enabled)
        else:
            configure_block(candidate, step.operation, options, report)
        candidate.session._save_current_view_state()
    candidate.session.select_experiment(original)
    if not run:
        candidate.view_label = view
    _commit(workspace, candidate)
    return {
        "workflow": document.name, "kind": document.kind, "experiments": experiments,
        "configured_only": not run,
    }
