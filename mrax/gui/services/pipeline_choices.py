"""Named experiment pipelines with shared sources and cached results."""

import copy
from uuid import uuid4

import numpy as np

from mrax.gui.session import FrameMask

from .state import _fit_settings_from_dict, _fit_settings_to_dict

NAVIGATION = (
    "active_frame", "active_z", "active_indices", "active_image_plane", "active_map",
    "active_kspace_mode", "active_kspace_centering", "active_reconstruction_component",
    "active_reconstruction_coil", "active_reconstruction_unwrap_phase",
)
FIT_METADATA = ("fit_grid", "fitted_map_dims")


def active_id(workspace, experiment=None):
    experiment = experiment or workspace.session.selected_experiment
    return workspace.pipelines.get(experiment, {}).get("id", "main")


def choices(workspace, experiment=None):
    from .pipeline_state import image_steps

    experiment = experiment or workspace.session.selected_experiment
    selected = active_id(workspace, experiment)
    entries = workspace.pipeline_sets.get(experiment, {"main": {"name": "Pipeline 1"}})
    result = []
    for identifier, entry in entries.items():
        cached = entry.get("cache")
        if identifier == selected:
            pipeline = workspace.pipelines.get(experiment, {})
            maps = (workspace.session.data or {}).get("experiments", {}).get(experiment, {}).get("fitted_maps", {})
        else:
            pipeline = cached["pipeline"]
            maps = cached["maps"]
        result.append({
            "id": identifier, "name": entry["name"], "active": identifier == selected,
            "has_image": bool(pipeline and image_steps(pipeline)), "maps": sorted(maps),
            "cached": bool(pipeline.get("applied_steps") or maps),
        })
    return result


def capture(workspace):
    session = workspace.session
    name = session.selected_experiment
    entry = session.current_experiment
    pipeline = workspace.pipelines[name]
    return {
        "pipeline": pipeline,
        "image": pipeline["output"] if pipeline["enabled"] else pipeline["source"],
        "maps": dict(session.fitted_maps),
        "map_units": dict(entry.get("fitted_map_units") or {}),
        "map_metadata": {key: entry.get("extra_metadata", {}).get(key, entry.get(key)) for key in FIT_METADATA},
        "processed": dict(entry.get("processed") or {}),
        "reconstruction": {key: value for key, value in (entry.get("kspace_metadata") or {}).items()
                           if key.startswith("reconstruction_") or key == "output_product"},
        "fit": workspace.fit_results_by_experiment.get(name),
        "preview": workspace.preview_results_by_experiment.get(name),
        "fit_settings": _fit_settings_to_dict(session.fit_settings),
        "frame_mask": None if session.frame_mask is None else session.frame_mask.as_array().copy(),
        "navigation": {key: copy.deepcopy(getattr(session, key)) for key in NAVIGATION},
        "view": workspace.view_label,
        "fit_region": {key: workspace.display_settings.get(name, {}).get("scene", {}).get(key)
                       for key in ("fit_region", "fit_rois")},
    }


def restore(workspace, cached):
    from .pipeline import install

    session = workspace.session
    name = session.selected_experiment
    workspace.pipelines = {**workspace.pipelines, name: cached["pipeline"]}
    for mapping, key in (("fit_results_by_experiment", "fit"), ("preview_results_by_experiment", "preview")):
        records = {**getattr(workspace, mapping)}
        if cached[key] is None:
            records.pop(name, None)
        else:
            records[name] = cached[key]
        setattr(workspace, mapping, records)
    entry = session.current_experiment
    entry["processed"] = dict(cached["processed"])
    entry["kspace_metadata"] = {
        **{key: value for key, value in (entry.get("kspace_metadata") or {}).items()
           if not key.startswith("reconstruction_") and key != "output_product"},
        **cached["reconstruction"],
    }
    entry["fitted_maps"] = dict(cached["maps"])
    entry["fitted_map_units"] = dict(cached["map_units"])
    metadata = {**entry.get("extra_metadata", {})}
    for key in FIT_METADATA:
        metadata.pop(key, None)
        entry.pop(key, None)
        value = cached["map_metadata"].get(key)
        if value is not None:
            metadata[key] = copy.deepcopy(value)
            if key == "fit_grid":
                entry[key] = copy.deepcopy(value)
    entry["extra_metadata"] = metadata
    install(workspace, cached["image"], update_model=False)
    session.fit_settings = _fit_settings_from_dict(cached["fit_settings"])
    mask = cached["frame_mask"]
    session.frame_mask = FrameMask.all(session.frame_count) if mask is None else FrameMask(np.asarray(mask, dtype=bool).copy())
    for key, value in cached["navigation"].items():
        setattr(session, key, copy.deepcopy(value))
    if session.active_map not in session.map_names:
        session.active_map = next(iter(session.map_names), None)
    session.set_active_frame(min(session.active_frame, session.frame_count - 1))
    workspace.view_label = cached["view"] if cached["view"] != "Map" or session.map_names else "Image"
    scene = workspace.display_settings.setdefault(name, {}).setdefault("scene", {})
    scene.update(fit_region=cached["fit_region"].get("fit_region") or "all", fit_rois=cached["fit_region"].get("fit_rois"))
    workspace.pipeline_mask_cache.clear()
    session._save_current_view_state()


def change_choice(workspace, options):
    from .cloning import clone_workspace
    from .comparison import default_comparison
    from .pipeline import _commit, ensure_pipeline, pipeline_info

    candidate = clone_workspace(workspace, readonly_arrays=True)
    pipeline = ensure_pipeline(candidate)
    experiment = candidate.session.selected_experiment
    entries = candidate.pipeline_sets[experiment]
    selected = pipeline["id"]
    action = options["action"]
    if action == "rename":
        name = str(options.get("name") or "").strip()
        if not name or len(name) > 80:
            raise ValueError("Give the pipeline a name of up to 80 characters")
        if any(row["name"].casefold() == name.casefold() for key, row in entries.items() if key != selected):
            raise ValueError("Choose a different pipeline name")
        entries[selected]["name"] = name
    else:
        cached = capture(candidate)
        if action in {"create", "duplicate"}:
            if len(entries) >= 16:
                raise ValueError("Use at most 16 pipelines per experiment")
            number = 1
            while any(row["name"] == f"Pipeline {number}" for row in entries.values()):
                number += 1
            identifier = uuid4().hex
            next_cache = cached
            if action == "create":
                source = pipeline["source"]
                next_cache = {
                    **cached,
                    "pipeline": {"id": identifier, "source": source, "output": source, "steps": [],
                                 "enabled": True, "generation": 0, "sources": pipeline.get("sources", {}),
                                 "source_processed": pipeline.get("source_processed", {})},
                    "image": source, "maps": {}, "map_units": {}, "map_metadata": {},
                    "processed": pipeline.get("source_processed", {}), "reconstruction": {}, "fit": None, "preview": None,
                    "view": "Image", "navigation": {**cached["navigation"], "active_map": None},
                }
            else:
                # Structural copies share read-only numerical buffers.
                next_cache = clone_workspace(candidate, readonly_arrays=True)
                next_cache = capture(next_cache)
                next_cache["pipeline"]["id"] = identifier
            entries[identifier] = {"name": f"Pipeline {number}", "cache": next_cache}
        elif action in {"select", "delete"}:
            identifier = options.get("pipeline_id")
            if action == "delete":
                if len(entries) == 1:
                    raise ValueError("Keep one pipeline per experiment")
                identifier = next(key for key in entries if key != selected)
            if identifier not in entries:
                raise ValueError("Choose an available pipeline for this experiment")
            if identifier == selected:
                return {"pipeline": pipeline_info(candidate)}
        else:
            raise ValueError("Unknown pipeline selection action")
        entries[selected]["cache"] = cached
        restore(candidate, entries[identifier]["cache"])
        entries[identifier]["cache"] = None
        if action == "delete":
            del entries[selected]
            for owner, display in candidate.display_settings.items():
                comparison = display.get("scene", {}).get("comparison", {})
                if any(comparison.get(side, {}).get("experiment") == experiment and comparison.get(side, {}).get("pipeline_id") == selected for side in ("left", "right")):
                    replacement = default_comparison(candidate)
                    for side in ("left", "right"):
                        replacement[side]["experiment"] = owner
                    display["scene"]["comparison"] = replacement
        candidate.display_settings.setdefault(experiment, {}).setdefault("scene", {})["comparison"] = default_comparison(candidate)
    _commit(workspace, candidate)
    return {"experiment": experiment, "pipeline": pipeline_info(candidate)}


def comparison_context(workspace, identifier):
    name = workspace.session.selected_experiment
    if identifier == active_id(workspace):
        return
    cached = workspace.pipeline_sets.get(name, {}).get(identifier, {}).get("cache")
    if cached is None:
        raise ValueError("This pipeline has no cached results")
    restore(workspace, cached)


def validate_sets(workspace):
    from mrax.workflow_rules import validate_pipeline_order

    for name, entries in workspace.pipeline_sets.items():
        if name not in workspace.session.experiment_names or not 1 <= len(entries) <= 16 or active_id(workspace, name) not in entries:
            raise ValueError("Invalid experiment pipelines")
        for identifier, entry in entries.items():
            if not isinstance(identifier, str) or not identifier or not isinstance(entry.get("name"), str) or not entry["name"].strip() or len(entry["name"]) > 80:
                raise ValueError("Invalid pipeline name or identifier")
            cached = entry.get("cache")
            if identifier == active_id(workspace, name):
                if cached is not None:
                    raise ValueError("The selected pipeline must use the active experiment state")
            elif not cached or cached["pipeline"]["id"] != identifier:
                raise ValueError("A pipeline cache is missing")
            else:
                validate_pipeline_order([step["kind"] for step in cached["pipeline"]["steps"]])
