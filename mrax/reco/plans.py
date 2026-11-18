"""Versioned, data-free reconstruction plans used inside .mflow workflows.

Included recipe stages inherit acquisition settings unless an option overrides
them. Removed/disabled stages do not run. ``when=configured`` explicitly keeps
an acquisition correction only when its metadata configure that correction.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import replace
from importlib.resources import files

from mrax.data_struct.acquisition import ReconstructionRecipe

from .pipeline import _ORDER, _RECIPE_KEYS, ReconstructionPipeline, ReconstructionStage

SOLVERS = {"fft", "variational", "nufft"}
LABELS = {
    "profile_inversion": "Readout profile inversion",
    "shift": "Image shift",
    "readout_regrid": "Readout regridding",
    "epi_phase": "EPI phase correction",
    "window": "K-space window",
    "partial_fourier": "Partial Fourier completion",
    "gridding": "Kaiser–Bessel gridding",
    "fft": "Fourier transform",
    "variational": "Variational reconstruction",
    "nufft": "NUFFT adjoint",
    "format": "Output geometry",
    "combine": "Coil combination",
    "product": "Image product",
}


def _validate_options(kind, options):
    if kind == "variational" and set(options) - {
        "num_steps",
        "regularizer",
        "regularizer_weight",
        "optimizer_name",
        "step_size",
        "tv_iterations",
        "prox_tolerance",
        "nufft_eps",
        "bregman_penalty",
        "grad_clip_norm",
    }:
        raise ValueError("Unknown portable variational reconstruction option")
    if kind in _RECIPE_KEYS:
        defaults = ReconstructionRecipe()
        for key, value in options.items():
            expected = getattr(defaults, key)
            valid = (
                type(value) is bool
                if isinstance(expected, bool)
                else type(value) is int
                if isinstance(expected, int)
                else type(value) in {int, float}
                if isinstance(expected, float)
                else isinstance(value, str)
                if isinstance(expected, str)
                else isinstance(value, (list, tuple))
                if isinstance(expected, tuple)
                else value is None or type(value) in {int, float}
            )
            if not valid:
                raise ValueError(f"Invalid value for {key}")
        # Partial options inherit other fields from the target acquisition. Supply
        # only dependent placeholders here. Compile_plan validates the merged recipe.
        validation = dict(options)
        if validation.get("output_product") == "frequency_hz":
            validation.setdefault("echo_spacing_seconds", 1.0)
        if validation.get("window_parameters"):
            validation.setdefault("kspace_window", ("none",) * len(validation["window_parameters"]))
        if validation.get("output_stack_permutation"):
            validation.setdefault("stack_slices", 2)
        try:
            replace(defaults, **validation)
        except (TypeError, IndexError, ValueError) as error:
            raise ValueError(f"Invalid {LABELS[kind]} settings: {error}") from error
    else:
        for key, value in options.items():
            if value is None:
                if key not in {
                    "frame_batch_size",
                    "chunk_size",
                    "batch_size",
                    "optimizer_name",
                    "step_size",
                    "bregman_penalty",
                    "grad_clip_norm",
                }:
                    raise ValueError(f"{key} cannot be empty")
                continue
            if key == "density_compensation":
                if type(value) is not bool:
                    raise ValueError("Density compensation must be true or false")
            elif key in {"regularizer", "optimizer_name"}:
                if not isinstance(value, str):
                    raise ValueError(f"{key} must be a name")
                if key == "regularizer" and value not in {"none", "tv", "l1", "l2"}:
                    raise ValueError("Choose none, TV, L1 or L2 regularization")
            else:
                if type(value) not in {int, float} or not math.isfinite(value) or value < 0:
                    raise ValueError(f"{key} must be a finite non-negative number")
                if (
                    key
                    in {
                        "frame_batch_size",
                        "chunk_size",
                        "batch_size",
                        "num_steps",
                        "tv_iterations",
                    }
                    and type(value) is not int
                ):
                    raise ValueError(f"{key} must be an integer")
                if (
                    key
                    in {
                        "chunk_size",
                        "batch_size",
                        "tv_iterations",
                        "step_size",
                        "bregman_penalty",
                        "grad_clip_norm",
                    }
                    and value == 0
                ):
                    raise ValueError(f"{key} must be positive")
                if key in {"eps", "nufft_eps"} and not 1e-12 <= value <= 1e-2:
                    raise ValueError("NUFFT tolerance must be between 1e-12 and 1e-2")


def normalize_plan(value):
    if not isinstance(value, dict) or set(value) - {"schema_version", "name", "encoding", "stages"}:
        raise ValueError("Unknown reconstruction plan setting")
    if type(value.get("schema_version")) is not int or value["schema_version"] != 1:
        raise ValueError("Unsupported reconstruction plan version")
    name = value.get("name")
    if not isinstance(name, str) or not name.strip() or len(name) > 160:
        raise ValueError("Give the reconstruction pipeline a name")
    if not isinstance(value.get("encoding", "any"), str) or value.get("encoding", "any") not in {
        "any",
        "cartesian",
        "sampled",
    }:
        raise ValueError("Unknown reconstruction encoding")
    stages = value.get("stages")
    if not isinstance(stages, list) or not 1 <= len(stages) <= 32:
        raise ValueError("Choose one to 32 reconstruction stages")
    names, order = [], []
    for stage in stages:
        if not isinstance(stage, dict) or set(stage) - {"kind", "options", "enabled", "when"}:
            raise ValueError("Unknown reconstruction stage setting")
        kind = stage.get("kind")
        if (
            not isinstance(kind, str)
            or kind not in LABELS
            or not isinstance(stage.get("options", {}), dict)
        ):
            raise ValueError("Choose a supported reconstruction stage and its options")
        if type(stage.get("enabled", True)) is not bool:
            raise ValueError("Choose whether each reconstruction stage is enabled")
        when = stage.get("when", "always")
        if not isinstance(when, str) or when not in {
            "always",
            "configured",
            "cartesian",
            "sampled",
        }:
            raise ValueError("Unknown reconstruction stage condition")
        if kind in SOLVERS and when != "always":
            raise ValueError("The reconstruction solver must be unconditional")
        try:
            ReconstructionStage(kind, stage.get("options", {}))
            _validate_options(kind, stage.get("options", {}))
        except (TypeError, IndexError) as error:
            raise ValueError(f"Invalid {LABELS[kind]} settings: {error}") from error
        names.append(kind)
        if stage.get("enabled", True):
            order.append(_ORDER[kind])
    if len(names) != len(set(names)):
        raise ValueError("Each reconstruction stage can occur once. Replace its settings instead")
    active = [s["kind"] for s in stages if s.get("enabled", True)]
    if sum(kind in SOLVERS for kind in active) != 1:
        raise ValueError("Keep exactly one Fourier, variational or NUFFT solver")
    if not {"format", "combine"}.issubset(active):
        raise ValueError("Keep output geometry and coil combination stages")
    if order != sorted(order):
        raise ValueError(
            "K-space corrections precede the solver. Geometry and coil combination follow it"
        )
    if "nufft" in active and "gridding" in active:
        raise ValueError("NUFFT has its own interpolation. Remove the gridding stage")
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("Reconstruction settings must be finite JSON values") from error
    result = copy.deepcopy(value)
    result["name"] = name.strip()
    result.setdefault("encoding", "any")
    return result


def configured_stage(kind, acquisition, overrides=None):
    recipe = (
        replace(acquisition.recipe, **(overrides or {}))
        if kind in _RECIPE_KEYS
        else acquisition.recipe
    )
    return {
        "profile_inversion": recipe.profile_inversion and acquisition.readout_model is not None,
        "shift": bool(recipe.image_shift or recipe.frame_image_shifts),
        "readout_regrid": recipe.readout_regrid and acquisition.readout_coordinates is not None,
        "epi_phase": recipe.epi_phase_correction != "none",
        "window": bool(recipe.kspace_window),
        "partial_fourier": acquisition.kind == "cartesian" and recipe.partial_fourier != "none",
        "gridding": acquisition.kind == "sampled",
        "product": recipe.output_product != "image",
    }.get(kind, True)


def compile_plan(acquisition, value):
    plan = normalize_plan(value)
    if plan["encoding"] not in {"any", acquisition.kind}:
        raise ValueError(f"{plan['name']} requires {plan['encoding']} acquisitions")
    stages = []
    for stage in plan["stages"]:
        if not stage.get("enabled", True):
            continue
        kind, when = stage["kind"], stage.get("when", "always")
        options = copy.deepcopy(stage.get("options", {}))
        if when in {"cartesian", "sampled"} and when != acquisition.kind:
            continue
        if when == "configured" and not configured_stage(kind, acquisition, options):
            continue
        inherited = {key: getattr(acquisition.recipe, key) for key in _RECIPE_KEYS.get(kind, ())}
        merged = {**inherited, **options}
        if kind in _RECIPE_KEYS:
            replace(acquisition.recipe, **merged)
        stages.append(ReconstructionStage(kind, merged))
    result = ReconstructionPipeline(tuple(stages))
    names = {s.kind for s in stages}
    if acquisition.kind == "cartesian" and "gridding" in names:
        raise ValueError("Cartesian acquisitions do not use sampled-data gridding")
    if acquisition.kind == "sampled" and "nufft" not in names and "gridding" not in names:
        raise ValueError(
            "Sampled Fourier and variational reconstruction need a gridding initialization stage"
        )
    if "nufft" in names and acquisition.kind != "sampled":
        raise ValueError("NUFFT requires sampled data with a trajectory")
    for stage in stages:
        if stage.kind == "partial_fourier" and acquisition.kind != "cartesian":
            raise ValueError("Partial Fourier completion requires Cartesian data")
        if (
            stage.kind == "profile_inversion"
            and stage.options.get("profile_inversion")
            and acquisition.readout_model is None
        ):
            raise ValueError("Readout profile inversion requires a readout-profile model")
        if (
            stage.kind == "readout_regrid"
            and stage.options.get("readout_regrid")
            and acquisition.readout_coordinates is None
        ):
            raise ValueError("Readout regridding requires explicit readout coordinates")
        if (
            stage.kind == "epi_phase"
            and stage.options.get("epi_phase_correction") != "none"
            and (acquisition.kind != "cartesian" or acquisition.readout_polarity is None)
        ):
            raise ValueError("EPI phase correction requires Cartesian readout polarities")
    return result


def reconstruction_presets():
    from mrax.workflows import loads_workflow

    result = []
    for path in sorted(files("mrax.reco").joinpath("presets").iterdir(), key=lambda p: p.name):
        if path.name.endswith(".mflow"):
            document = loads_workflow(path.read_text(encoding="utf-8"))
            result.append(
                {
                    "id": path.name.removesuffix(".mflow"),
                    "filename": path.name,
                    "name": document.name,
                    "description": document.description,
                    "document": document.model_dump(),
                    "plan": normalize_plan(document.steps[0].settings["pipeline"]),
                }
            )
    return result


def default_plan(acquisition=None):
    name = (
        "acquisition-fourier"
        if acquisition is None
        else ("cartesian-fourier" if acquisition.kind == "cartesian" else "gridded-fourier")
    )
    return next(p["plan"] for p in reconstruction_presets() if p["id"] == name)
