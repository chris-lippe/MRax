"""Multipool configuration and read-only fitting readiness."""

from __future__ import annotations

import copy
import re

import numpy as np

from mrax.devices import resolve_device
from mrax.gui.models import (
    make_model,
    model_available_pools,
    model_default_pools,
    model_fit_inputs,
    model_parameter_specs,
    model_pool_definitions,
)
from mrax.gui.session import FrameMask

from .scene import fitting_roi
from .state import _fit_settings_to_dict
from .view_state import WorkspaceFitInputs

RESERVED_POOLS = {"water", "mt"}
POOL_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,63}")


def fit_catalog(workspace):
    session = workspace.session
    settings = session.fit_settings
    model = settings.model_name or session.current_model_name
    inputs = model_fit_inputs(model)
    reference_scan = inputs.reference_scan and session.current_experiment.get("exp_type") == "sat_trans"
    choices = {
        "workspace_id": workspace.key,
        "experiment": session.selected_experiment,
        "model": model,
        "inputs": {"reference_scan": reference_scan, "prior_maps": inputs.prior_maps, "derived_maps": inputs.derived_maps},
        "pools": model_available_pools(model, settings.pool_overrides),
        "default_pools": model_default_pools(model),
        "field_dimensions": fitting_dimensions(session, model),
        "reference_scans": session.sat_trans_reference_choices() if reference_scan else [],
        "prior_maps": {kind: session.ab_mt_map_choices(kind) for kind in inputs.prior_maps},
    }
    return {
        "choices": choices,
        "settings": _fit_settings_to_dict(settings),
        "parameters": model_parameter_specs(
            model, pools=settings.selected_pools or None,
            param_overrides=settings.param_overrides or None,
            pool_overrides=settings.pool_overrides or None,
            parameter_constant_dims=settings.parameter_constant_dims or None,
        ),
    }


def fitting_dimensions(session, model_name=None):
    from mrax.core.loss_builder import _field_metadata

    if not session.is_loaded:
        return []
    experiment = session.build_experiment()
    model = make_model(model_name or session.fit_settings.model_name or session.current_model_name)
    shape, names, _ = _field_metadata(experiment, model=model)
    return [
        {"name": name, "label": name.replace("_", " ").capitalize(), "size": size,
         "axis": axis, "regularizable": name != "slice_package" and size > 1}
        for axis, (name, size) in enumerate(zip(names, shape, strict=True))
    ]


def fitting_frame_page(session, offset=0, limit=32):
    from mrax.core.loss_builder import _dim_map_for_experiment

    if not session.is_loaded:
        return {"frame_count": 0, "frames": []}
    experiment = session.build_experiment()
    count = int(experiment.frame_count)
    names = list(experiment.non_spatial_dim_names)
    shape = tuple(experiment.dims[name] for name in names)
    model = make_model(session.fit_settings.model_name or session.current_model_name)
    dim_map = _dim_map_for_experiment(experiment, model)
    coordinates = []
    for key, quantity in experiment.observables.items():
        dimension = next((name for name in names if dim_map.get(name) == key), None)
        size = np.size(quantity.values)
        if np.ndim(quantity.values) != 1 or size <= 1 or not (
            size == count or (dimension and size == experiment.dims[dimension])
        ):
            continue
        values = np.asarray(quantity.values)
        if np.iscomplexobj(values) or not np.issubdtype(values.dtype, np.number):
            continue
        coordinates.append((key, values, quantity.si_unit, dimension))
    frames = []
    for index in range(min(offset, count), min(offset + limit, count)):
        indices = dict(zip(names, np.unravel_index(index, shape), strict=True)) if shape else {}
        labels = []
        represented = set()
        for key, values, unit, dimension in coordinates[:6]:
            value = values[index] if values.size == count else values[indices[dimension]]
            labels.append(f"{key}: {float(value):.6g}" + (f" {unit}" if unit else ""))
            if dimension:
                represented.add(dimension)
        labels += [f"{name.replace('_', ' ')}: {int(value) + 1}" for name, value in indices.items()
                   if name not in represented and (name != "frame" or not labels)]
        frames.append({"index": index, "label": ", ".join(labels) or "Single image"})
    return {"frame_count": count, "frames": frames}


def _owner(parameter, pools):
    return next(
        (pool for pool in sorted(pools, key=len, reverse=True) if parameter.endswith("_" + pool)),
        None,
    )


def edit_pool(workspace, options):
    from .analysis_workflows import set_fit_settings

    session = workspace.session
    values = copy.deepcopy(_fit_settings_to_dict(session.fit_settings))
    values.update(copy.deepcopy(options.get("settings") or {}))
    model = values.get("model_name") or session.current_model_name
    defaults = model_default_pools(model)
    if not defaults:
        raise ValueError("This model has no editable pools")
    definitions = model_pool_definitions(model, values.get("pool_overrides"))
    active = list(values.get("selected_pools") or defaults)
    previous = active[:]
    operation = options.get("operation")
    name = str(options.get("name") or "").strip()
    if not POOL_NAME.fullmatch(name):
        raise ValueError("Pool names must start with a letter and use letters, digits or underscores")
    overrides = values.setdefault("pool_overrides", {})
    if operation == "add":
        if name in active:
            raise ValueError("A pool with this name is already included")
        template = str(options.get("template") or "")
        if template not in definitions:
            raise ValueError("Choose an available pool template")
        if (name.lower() in RESERVED_POOLS or template in RESERVED_POOLS) and name != template:
            raise ValueError("Water and MT keep their model-defined names. Choose an exchange-pool template")
        overrides[name] = copy.deepcopy(definitions[template])
        active.append(name)
    elif operation in {"remove", "rename"}:
        if name not in active:
            raise ValueError("Choose an included pool")
        if operation == "remove":
            if len(active) == 1:
                raise ValueError("Keep at least one included pool")
            active.remove(name)
            replacement = None
        else:
            if name in RESERVED_POOLS:
                raise ValueError("Water and MT keep their model-defined names")
            replacement = str(options.get("new_name") or "").strip()
            if not POOL_NAME.fullmatch(replacement):
                raise ValueError("Pool names must start with a letter and use letters, digits or underscores")
            if replacement.lower() in RESERVED_POOLS or replacement in active:
                raise ValueError("Choose a unique name for the extra pool")
            overrides[replacement] = copy.deepcopy(definitions[name])
            active[active.index(name)] = replacement
        overrides.pop(name, None)
        for field in ("param_overrides", "regularizer_configs", "parameter_constant_dims"):
            remapped = {}
            for key, value in (values.get(field) or {}).items():
                if _owner(key, previous) == name:
                    if replacement is not None:
                        remapped[key[: -len(name)] + replacement] = value
                else:
                    remapped[key] = value
            values[field] = remapped
    else:
        raise ValueError("Choose add, remove or rename")
    values["selected_pools"] = active
    set_fit_settings(workspace, values)
    if options.get("excluded_frames") is not None:
        apply_fit_exclusions(session, options["excluded_frames"])


def apply_fit_exclusions(session, indices):
    if not isinstance(indices, list) or any(
        not isinstance(index, int) or isinstance(index, bool)
        or not 0 <= index < session.frame_count for index in indices
    ):
        raise ValueError("Excluded frame indices must refer to available frames")
    included = np.ones(session.frame_count, dtype=bool)
    included[indices] = False
    if not included.any():
        raise ValueError("Keep at least one included frame")
    session.frame_mask = FrameMask(included)


def check_fit_readiness(workspace, options):
    from .analysis_workflows import set_fit_settings

    session = workspace.session
    errors = []
    try:
        if not session.is_loaded:
            raise ValueError("Load an experiment before checking a fit")
        set_fit_settings(workspace, options.get("settings") or {})
        if options.get("excluded_frames") is not None:
            apply_fit_exclusions(session, options["excluded_frames"])
        resolve_device(options.get("device", "cpu"))
        use_roi = bool(options.get("use_roi", True))
        selected = None
        if use_roi:
            try:
                names = options.get("roi_names")
                selected = fitting_roi(workspace, names if names is not None else [])
            except ValueError as exc:
                errors.append(str(exc))
        kwargs = WorkspaceFitInputs(workspace)._visual_state_fit_kwargs(
            include_roi=selected is not None, roi_override=selected,
        )
        result = session.run_fit_sanity_check(use_roi=selected is not None, **kwargs)
        result["errors"] = errors + result["errors"]
        result["ok"] = not result["errors"]
        result["report"] = "\n".join([
            "Fit sanity check",
            "Model: " + str(result["model"]),
            *("Error: " + error for error in result["errors"]),
            *("Warning: " + warning for warning in result["warnings"]),
            *result["notes"],
            "Status: ready" if result["ok"] else "Status: needs attention",
        ])
        return result
    except Exception as exc:
        errors.append(str(exc))
        return {
            "name": session.selected_experiment or "",
            "model": session.fit_settings.model_name or session.current_model_name,
            "ok": False, "errors": errors, "warnings": [], "notes": [],
            "report": "\n".join(errors),
        }
