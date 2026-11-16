"""Embedded reconstruction and fitting workflows in an image processing chain."""

from __future__ import annotations

import copy

from mrax.workflow_rules import validate_pipeline_order
from mrax.workflows import WorkflowDocument


def block_document(workspace, kind, options):
    """Snapshot a leaf workflow. No live library or external file reference is kept."""
    from .reusable_workflows import capture_workflow

    expected = "reconstruction" if kind == "reconstruct" else "fitting"
    if "workflow" in options:
        raw = options["workflow"]
        if not isinstance(raw, dict):
            raise ValueError(f"Choose a {expected} workflow for this block")
        document = WorkflowDocument.model_validate(raw)
        if len(document.steps) != 1 or document.steps[0].operation != kind:
            raise ValueError(f"Choose a workflow with one {expected} block")
        bindings = options.get("bindings", {})
        if not isinstance(bindings, dict) or set(bindings) != {b.id for b in document.bindings}:
            raise ValueError("Assign every workflow input to this experiment")
        if any(not isinstance(value, str) or not value for value in bindings.values()):
            raise ValueError("Choose an available resource for every workflow input")
        target = document.steps[0].target
        if target is not None and bindings[target["$binding"]] not in {
            "$target",
            workspace.session.selected_experiment,
        }:
            raise ValueError("This block targets another experiment. Apply it from Workflows")
    else:
        draft = copy.deepcopy(options)
        if kind == "reconstruct" and not draft.get("pipeline"):
            from mrax.reco.plans import default_plan

            entry = workspace.session.current_experiment
            draft["pipeline"] = (entry.get("kspace_metadata") or {}).get(
                "reconstruction_pipeline"
            ) or default_plan(entry.get("acquisition"))
        document, bindings = capture_workflow(workspace, expected, draft=draft)
    return document, copy.deepcopy(bindings)


def block_options(workspace, kind, options):
    document, bindings = block_document(workspace, kind, options)
    return {
        "workflow": document.model_dump(),
        "bindings": bindings,
        **({"device": options["device"]} if "device" in options else {}),
    }


def configure_block(workspace, kind, options, progress):
    """Load settings into one block while retaining the last applied image state."""
    from .cloning import clone_workspace
    from .pipeline import _commit, _make_step, ensure_pipeline, pipeline_info
    from .pipeline_state import primary_step

    if kind not in {"reconstruct", "fit"}:
        raise ValueError("Choose reconstruction or fitting settings")
    candidate = clone_workspace(workspace, readonly_arrays=True)
    session = candidate.session
    if kind == "reconstruct" and not session.can_reconstruct_raw_data() and session.can_lazy_load_raw_data(require_acquisition=True):
        session.load_raw_data_for_experiment(progress_callback=progress, require_acquisition=True)
    pipeline = ensure_pipeline(candidate)
    previous = primary_step(pipeline, kind, options)
    requested = options.get("step_id")
    if requested:
        previous = next((row for row in pipeline["steps"] if row["id"] == requested and row["kind"] == kind), None)
        if previous is None:
            raise ValueError("The pipeline step no longer exists")
    step = _make_step(candidate, options, kind=kind, previous=previous)
    if kind == "fit" and previous is None:
        step["map_prefix"] = ""
    # Keep provenance of the output visible while a new configuration is pending.
    pipeline.setdefault("applied_steps", [
        copy.copy(row) for row in pipeline["steps"] if not row.get("pending")
    ])
    step["pending"] = True
    if previous is not None:
        pipeline["steps"][pipeline["steps"].index(previous)] = step
    elif len(pipeline["steps"]) >= 64:
        raise ValueError("A pipeline supports up to 64 processing steps")
    elif kind == "reconstruct":
        pipeline["steps"].insert(0, step)
    else:
        pipeline["steps"].append(step)
    validate_chain(candidate, pipeline["steps"])
    pipeline["generation"] += 1
    summary = {"step_id": step["id"], "pipeline": pipeline_info(candidate)}
    _commit(workspace, candidate)
    return summary


def resolved_block(workspace, kind, options, *, validate=True):
    from .reusable_workflows import _validate_target, resolve_settings

    document, bindings = block_document(workspace, kind, options)
    resolved = (
        _validate_target(workspace, document, bindings)
        if validate
        else resolve_settings(document, workspace, bindings)
    )
    return document, resolved[0][1]


def reconstruction_available(session):
    """Validate immutable acquired buffers once per session/acquisition identity."""
    acquisition = session.current_experiment.get("acquisition") if session.is_loaded else None
    if acquisition is None:
        return (
            False,
            "Load prepared raw k-space in Reconstruct before inserting a reconstruction block.",
        )
    cached = getattr(session, "_reconstruction_validation", None)
    if cached is not None and cached[0] is acquisition:
        return cached[1], cached[2]
    try:
        acquisition.require_prepared()
    except (ValueError, TypeError) as error:
        result = (False, str(error))
    else:
        result = (True, None)
    session._reconstruction_validation = (acquisition, *result)
    return result


def validate_reconstruction(workspace, options):
    from mrax.reco.plans import compile_plan, default_plan

    available, reason = reconstruction_available(workspace.session)
    if not available:
        raise ValueError(reason)
    acquisition = workspace.session.current_experiment["acquisition"]
    compile_plan(acquisition, options.get("pipeline") or default_plan(acquisition))


def capture_block_bindings(document, mapping, bind, target):
    assignments = {}

    def assign(binding):
        if binding.id in assignments:
            return assignments[binding.id]
        experiment = None
        if binding.experiment:
            parent = next(b for b in document.bindings if b.id == binding.experiment)
            experiment = assign(parent)["$binding"]
        value = mapping[binding.id]
        assignments[binding.id] = bind(
            binding.kind,
            target if value == "$target" else value,
            experiment=experiment,
            label=binding.label,
        )
        return assignments[binding.id]

    for binding in document.bindings:
        assign(binding)
    return assignments


def block_catalog(workspace):
    from mrax.reco.plans import reconstruction_presets

    from .reusable_workflows import resource_choices

    defaults = {}
    for kind in ("reconstruct", "fit"):
        draft = {"use_roi": False} if kind == "fit" and not workspace.session.rois else {}
        defaults[kind] = block_options(workspace, kind, draft)
    available, reason = reconstruction_available(workspace.session)
    acquisition = workspace.session.current_experiment.get("acquisition")
    presets = reconstruction_presets()
    for preset in presets:
        preset["compatible"] = available and preset["plan"]["encoding"] in {"any", acquisition.kind}
    return {
        "defaults": defaults,
        "presets": presets,
        "resources": resource_choices(workspace),
        "can_reconstruct": available,
        "reconstruction_unavailable": reason,
    }


def validate_chain(workspace, steps):
    from .cloning import clone_workspace

    if len(steps) > 64:
        raise ValueError("A pipeline supports up to 64 processing steps")
    validate_pipeline_order([step["kind"] for step in steps])
    prefixes = set()
    owners = {
        record.get("prefix"): key
        for key, record in (workspace.pipelines.get(workspace.session.selected_experiment) or {})
        .get("fits", {})
        .items()
    }
    for step in steps:
        if step["kind"] == "fit":
            prefix, number = step.get("map_prefix", step["label"]), 2
            base = prefix or step["label"]
            while prefix in prefixes or (prefix in owners and owners[prefix] != step["id"]):
                prefix, number = f"{base} ({number})", number + 1
            step["map_prefix"] = prefix
            prefixes.add(prefix)
        if step["kind"] in {"reconstruct", "fit"}:
            # Disabled blocks retain structural validation, but need no current inputs.
            block_document(workspace, step["kind"], step["options"])
            if step["enabled"] and step["kind"] == "reconstruct":
                context = clone_workspace(workspace, readonly_arrays=True)
                resolved_block(context, step["kind"], step["options"])


def fit_result_for_map(workspace, map_name=None, *, experiment=None):
    name = experiment or workspace.session.selected_experiment
    selected = workspace.session.active_map if map_name is None else map_name
    pipeline = workspace.pipelines.get(name) or {}
    for record in pipeline.get("fits", {}).values():
        if selected in record["maps"]:
            return record["result"]
    return workspace.fit_results_by_experiment.get(name) or {}


def run_fitting_block(workspace, step, progress):
    from mrax.gui.session import FrameMask

    from .analysis_workflows import run_operation, set_fit_settings

    document, options = resolved_block(workspace, "fit", step["options"])
    session = workspace.session
    original_maps = dict(session.fitted_maps)
    original_dims = dict((session.current_experiment.get("extra_metadata") or {}).get("fitted_map_dims") or {})
    pipeline = workspace.pipelines[session.selected_experiment]
    prior = workspace.fit_results_by_experiment.get(session.selected_experiment)
    if original_maps and not (prior or {}).get("block_id"):
        # Preserve the grid/provenance of maps produced before the first fit block.
        from uuid import uuid4

        result = prior or {
            "pipeline_grid": copy.deepcopy(
                session.current_experiment.get("fit_grid")
                or {
                    key: value
                    for key, value in pipeline["source"].items()
                    if key in {"dims", "geometry", "unit", "space"}
                }
            ),
            "pipeline_steps": [],
        }
        covered = {name for row in pipeline.get("fits", {}).values() for name in row["maps"]}
        remaining = [name for name in original_maps if name not in covered]
        if remaining:
            pipeline.setdefault("fits", {})["earlier-" + uuid4().hex] = {
                "maps": remaining,
                "result": result,
            }
    set_fit_settings(workspace, options["settings"])
    workspace.display_settings.setdefault(session.selected_experiment, {}).setdefault("scene", {}).update(
        fit_region="rois" if options.get("use_roi", True) else "all",
        fit_rois=options.get("roi_names"),
    )
    session.fit_settings.use_view_state_data = False
    session.fit_settings.output_dir = None
    mask = FrameMask.all(session.frame_count)
    mask.exclude(*options.get("excluded_frames", []))
    session.frame_mask = mask
    run_operation(
        workspace,
        "full_fit",
        {
            **options,
            "device": step["options"].get("device", "cpu"),
            "roi_names": options.get("roi_names")
            if options.get("roi_names") is not None
            else [r.name for r in session.rois],
        },
        progress,
    )
    result = workspace.fit_results_by_experiment[session.selected_experiment]
    result["workflow"] = document.model_dump()
    result["block_id"] = step["id"]
    pipeline = workspace.pipelines[session.selected_experiment]
    previous = pipeline.get("fits", {}).get(step["id"], {})
    for key in previous.get("maps", []):
        original_maps.pop(key, None)
    prefix = step.get("map_prefix", document.name)
    lead = prefix + " / " if prefix else ""
    maps = {lead + key: value for key, value in result["fields"].items()}
    dimensions = {**original_dims, **{lead + key: dims for key, dims in result["field_dims"].items()}}
    session.replace_fitted_maps({**original_maps, **maps}, dimensions=dimensions)
    session.active_map = next(iter(maps), session.active_map)
    return {"prefix": prefix, "maps": list(maps), "result": result}
