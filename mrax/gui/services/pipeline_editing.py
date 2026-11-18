"""Validate and apply a complete proposed chain in one transaction."""

import copy
import re
from uuid import uuid4

from .cloning import clone_workspace
from .pipeline import _commit, _evaluate, _make_step, ensure_pipeline, pipeline_info


def apply_pipeline_draft(workspace, options, progress):
    candidate = clone_workspace(workspace, readonly_arrays=True)
    pipeline = ensure_pipeline(candidate)
    if options.get("pipeline_id", pipeline["id"]) != pipeline["id"]:
        raise ValueError("Select this pipeline before applying its draft")
    generation = options.get("base_generation")
    if type(generation) is not int or generation != pipeline["generation"]:
        raise ValueError("The pipeline changed while this draft was being edited. Reload its steps")
    rows = options.get("steps")
    if not isinstance(rows, list) or len(rows) > 64:
        raise ValueError("A pipeline supports up to 64 processing steps")
    existing = {step["id"]: step for step in pipeline["steps"]}
    result = []
    ids = set()
    run_fits = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) - {
            "id",
            "source_id",
            "source_version",
            "kind",
            "options",
            "enabled",
        }:
            raise ValueError("Unknown pipeline step setting")
        client_id = row.get("id")
        if (
            not isinstance(client_id, str)
            or re.fullmatch(r"[A-Za-z0-9_-]{1,80}", client_id) is None
            or client_id in ids
        ):
            raise ValueError("Each draft step needs a unique identifier")
        ids.add(client_id)
        source_id = row.get("source_id")
        if source_id is not None and (not isinstance(source_id, str) or source_id not in existing):
            raise ValueError("A source step no longer exists")
        previous = existing.get(source_id)
        if previous is not None and row.get("source_version") != previous["version"]:
            raise ValueError("A source step changed while this draft was being edited")
        if client_id in existing and client_id != source_id:
            raise ValueError("A draft cannot replace the identity of another step")
        kind = row.get("kind")
        if kind not in {"process", "register", "stored_registration", "drift", "reconstruct", "fit"}:
            raise ValueError("Choose a supported pipeline operation")
        if previous is not None and kind != previous["kind"]:
            raise ValueError("Insert a new step to change its operation type")
        settings = row.get("options")
        enabled = row.get("enabled")
        if not isinstance(settings, dict) or type(enabled) is not bool:
            raise ValueError("Choose step settings and an enabled state")
        if kind == "process" and settings.get("method") == "dti":
            raise ValueError("Derived maps belong in Analyze after fitting")
        if (
            settings.get("moving_experiment", candidate.session.selected_experiment)
            != candidate.session.selected_experiment
        ):
            raise ValueError("Pipeline steps must target the selected experiment")
        if previous is not None and settings == previous["options"]:
            # Preserve frozen references, masks and explicit input snapshots.
            step = copy.copy(previous)
            step["options"] = copy.deepcopy(settings)
            if client_id != source_id or enabled != previous["enabled"]:
                step["version"] = uuid4().hex
        else:
            step = _make_step(
                candidate,
                settings,
                kind=kind,
                previous=previous if client_id == source_id else None,
                enabled=enabled,
            )
        if previous is None or client_id != source_id:
            step["id"] = uuid4().hex
            if kind == "fit":
                step["map_prefix"] = step["label"]
        step.update(enabled=enabled, _draft_id=client_id)
        if kind == "fit" and (
            previous is None or client_id != source_id or settings != previous["options"]
            or not previous["enabled"] or previous.get("pending")
        ):
            run_fits.add(step["id"])
        result.append(step)
    pipeline["steps"] = result
    from .pipeline_blocks import validate_chain
    validate_chain(candidate, result)
    _evaluate(candidate, progress, run_fits=run_fits)
    for step in result:
        step.pop("_draft_id", None)
    candidate.view_label = "Image"
    candidate.session._save_current_view_state()
    summary = {
        "experiment": candidate.session.selected_experiment,
        "pipeline": pipeline_info(candidate),
    }
    _commit(workspace, candidate)
    return summary
