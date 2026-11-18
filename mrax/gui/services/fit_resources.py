"""Adapt model inputs to generic workflow resources."""

from __future__ import annotations

import copy
from dataclasses import fields

from mrax.gui.models import model_fit_inputs
from mrax.gui.session import FitSettings, _decode_map_ref, _encode_map_ref


def _resource_fields():
    return {
        item.name: item.metadata["workflow_input"]
        for item in fields(FitSettings)
        if "workflow_input" in item.metadata
    }


def _input_specs(model_name):
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("Choose a fitting model before saving or applying a workflow")
    capabilities = model_fit_inputs(model_name)
    result = {}
    for field, spec in _resource_fields().items():
        enabled = getattr(capabilities, spec["capability"])
        if enabled is True or (enabled and spec["role"] in enabled):
            result[spec["role"]] = {**spec, "field": field}
    return result


def capture_fit_inputs(settings, bind, target):
    """Remove session resource names and capture the active model's inputs."""
    specs = _input_specs(settings.get("model_name"))
    values = {name: settings.pop(name, None) for name in _resource_fields()}
    inputs = {}
    for role, spec in specs.items():
        value = values[spec["field"]]
        if not value:
            continue
        owner = None
        if spec["kind"] == "map":
            decoded = _decode_map_ref(value)
            if decoded is None:
                raise ValueError("Choose an available " + spec["label"])
            experiment, value = decoded
            if experiment != target:
                owner = bind("experiment", experiment)["$binding"]
        inputs[role] = bind(spec["kind"], value, experiment=owner, label=spec["label"])
    return inputs


def resolve_fit_inputs(settings, inputs, resolve):
    """Validate a fit payload and translate resource roles to session settings."""
    resources = _resource_fields()
    portable = (
        {item.name for item in fields(FitSettings)}
        - resources.keys()
        - {
            "output_dir",
            "selected_roi",
            "use_view_state_data",
        }
    )
    if set(settings) - portable or not isinstance(settings.get("model_name"), str):
        raise ValueError("A fitting workflow needs a model and portable fit settings")
    for key in (
        "param_overrides",
        "pool_overrides",
        "regularizer_configs",
        "parameter_constant_dims",
    ):
        if key in settings and not isinstance(settings[key], dict):
            raise ValueError(key + " must be an object")
    if "selected_pools" in settings and (
        not isinstance(settings["selected_pools"], list)
        or not all(isinstance(value, str) for value in settings["selected_pools"])
    ):
        raise ValueError("Selected pools must be a list of pool names")
    for key in ("auto_mask_nan_observables", "compute_derived_maps"):
        if settings.get(key) is not None and type(settings[key]) is not bool:
            raise ValueError(key + " must be a boolean or null")
    if "num_steps" in settings and type(settings["num_steps"]) is not int:
        raise ValueError("Fit step count must be an integer")
    if "step_size" in settings and type(settings["step_size"]) not in {int, float}:
        raise ValueError("Fit step size must be a number")
    specs = _input_specs(settings["model_name"])
    if inputs.keys() - specs.keys():
        raise ValueError(
            "The selected model does not support input " + sorted(inputs.keys() - specs.keys())[0]
        )
    result = {**copy.deepcopy(settings), **dict.fromkeys(resources)}
    for role, reference in inputs.items():
        spec = specs[role]
        resource = resolve(reference)
        if "$output" in resource:
            expected = spec["kind"]
            if resource["port"] != expected:
                raise ValueError(spec["label"] + " requires a resource of type " + expected)
            result[spec["field"]] = resource
            continue
        if resource["kind"] != spec["kind"]:
            raise ValueError(spec["label"] + " requires a resource of type " + spec["kind"])
        result[spec["field"]] = (
            _encode_map_ref(resource["experiment"], resource["name"])
            if spec["kind"] == "map"
            else resource["experiment"]
        )
    return result
