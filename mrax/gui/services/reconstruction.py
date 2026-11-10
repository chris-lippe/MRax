"""Reconstruction plans, stage settings and named outputs for the workbench."""

from __future__ import annotations

import copy
import inspect

from mrax.data_struct.acquisition import ReconstructionRecipe
from mrax.reco.nufft import prepare_nufft
from mrax.reco.pipeline import _ORDER, _RECIPE_KEYS
from mrax.reco.plans import (
    LABELS,
    SOLVERS,
    compile_plan,
    configured_stage,
    default_plan,
    normalize_plan,
    reconstruction_presets,
)
from mrax.reco.variational import solve_variational

from .view_data import json_value


def reconstruction_settings(options):
    document = options.get("workflow")
    return document["steps"][0]["settings"] if document else options


def reconstruction_label(workspace, experiment=None):
    session = workspace.session
    name = experiment or session.selected_experiment
    entry = (session.data or {}).get("experiments", {}).get(name, {})
    if not entry:
        return "Reconstruction"
    pipeline = workspace.pipelines.get(name) or {}
    previous = next((s for s in pipeline.get("steps", []) if s["kind"] == "reconstruct"), None)
    plan = reconstruction_settings((previous or {}).get("options", {})).get("pipeline")
    if plan:
        return str(plan["name"])
    metadata = entry.get("kspace_metadata") or {}
    if metadata.get("reconstruction_name"):
        return str(metadata["reconstruction_name"])
    acquisition = entry.get("acquisition")
    method = metadata.get("reconstruction_method")
    if method in {"nufft", "variational"}:
        return LABELS[method]
    if acquisition is None:
        return "Reconstruction"
    return (
        "Gridded Fourier"
        if acquisition is not None and acquisition.kind == "sampled"
        else "Cartesian Fourier"
    )


def reconstruction_catalog(workspace):
    from .processing import _field

    session = workspace.session
    acquisition = session.current_experiment.get("acquisition") if session.is_loaded else None
    recipe = ReconstructionRecipe() if acquisition is None else acquisition.recipe
    dimensions = (
        len(acquisition.encoding_shape)
        if acquisition is not None
        else len(session.spatial_dim_names) or 2
    )
    defaults = {
        kind: {key: getattr(recipe, key) for key in keys} for kind, keys in _RECIPE_KEYS.items()
    }
    defaults["fft"] = {"frame_batch_size": None}
    defaults["nufft"] = {
        "eps": inspect.signature(prepare_nufft).parameters["eps"].default,
        "density_compensation": True,
    }
    defaults["variational"] = {
        key: parameter.default
        for key, parameter in inspect.signature(solve_variational).parameters.items()
        if key
        in {
            "num_steps",
            "regularizer",
            "regularizer_weight",
            "optimizer_name",
            "step_size",
            "tv_iterations",
            "prox_tolerance",
            "nufft_eps",
        }
    }

    def field(
        stage, key, label, *, kind="number", choices=(), minimum=None, maximum=None, nullable=False
    ):
        base, _, index = key.partition(".")
        value = defaults[stage].get(base)
        if index:
            value = value[int(index)] if value and len(value) > int(index) else 0.0
        return _field(
            key,
            label,
            value=value,
            kind=kind,
            choices=choices,
            minimum=minimum,
            maximum=maximum,
            nullable=nullable,
        )

    fields = {
        "profile_inversion": [],
        "shift": [
            field("shift", f"image_shift.{i}", f"Constant shift axis {i + 1} (voxels)")
            for i in range(dimensions)
        ],
        "readout_regrid": [
            field(
                "readout_regrid",
                "readout_interpolation",
                "Interpolation",
                kind="choice",
                choices=("adjoint", "linear", "lanczos"),
            ),
            field(
                "readout_regrid",
                "readout_kernel_radius",
                "Kernel radius",
                kind="integer",
                minimum=2,
                maximum=128,
            ),
        ],
        "epi_phase": [
            field(
                "epi_phase",
                "epi_phase_correction",
                "Phase correction",
                kind="choice",
                choices=("none", "entropy", "navigator", "fixed"),
            ),
            field("epi_phase", "epi_phase_prior.0", "Constant phase (rad)"),
            field("epi_phase", "epi_phase_prior.1", "Linear phase (rad)"),
        ],
        "window": [
            field(
                "window",
                "kspace_window",
                "Window (all encoding axes)",
                kind="choice",
                choices=("none", "hann", "hamming", "sine", "sine_squared"),
            ),
            field("window", "window_parameters", "Window parameter", minimum=0),
        ],
        "partial_fourier": [
            field(
                "partial_fourier",
                "partial_fourier",
                "Completion",
                kind="choice",
                choices=("none", "pocs", "homodyne"),
            ),
            field(
                "partial_fourier",
                "partial_fourier_iterations",
                "Iterations",
                kind="integer",
                minimum=1,
                maximum=1000,
            ),
            field(
                "partial_fourier",
                "phase_window",
                "Phase window",
                kind="choice",
                choices=("boxcar", "hann"),
            ),
            field("partial_fourier", "partial_fourier_tolerance", "Tolerance", minimum=0),
        ],
        "gridding": [
            field("gridding", "gridding_kernel_width", "Kernel width", minimum=1, maximum=16),
            field("gridding", "gridding_shape_parameter", "Shape parameter", minimum=0.1),
            field("gridding", "deapodize", "Deapodize", kind="boolean"),
        ],
        "fft": [
            field(
                "fft",
                "frame_batch_size",
                "Frames per batch (automatic if empty)",
                kind="integer",
                nullable=True,
                minimum=0,
            )
        ],
        "nufft": [
            field("nufft", "eps", "NUFFT tolerance", minimum=1e-12, maximum=1e-2),
            field("nufft", "density_compensation", "Density compensation", kind="boolean"),
        ],
        "variational": [
            field(
                "variational",
                "regularizer",
                "Regularizer",
                kind="choice",
                choices=("none", "tv", "l1", "l2"),
            ),
            field("variational", "regularizer_weight", "Regularization weight", minimum=0),
            field(
                "variational", "num_steps", "Iterations", kind="integer", minimum=0, maximum=100000
            ),
            field(
                "variational",
                "optimizer_name",
                "Optimizer",
                kind="choice",
                choices=("fista", "split_bregman", "adam"),
            ),
            field(
                "variational",
                "step_size",
                "Step size (automatic if empty)",
                nullable=True,
                minimum=1e-12,
            ),
        ],
        "format": [
            field("format", "output_scale", "Intensity scale", minimum=1e-12),
            field("format", "checkerboard", "Checkerboard phase correction", kind="boolean"),
            field("format", "elliptical_support", "Elliptical support", kind="boolean"),
        ],
        "combine": [
            field(
                "combine", "coil_combine", "Combine coils", kind="choice", choices=("rss", "sum")
            ),
            field(
                "combine", "average_repeats", "Magnitude-average repeats", kind="integer", minimum=1
            ),
        ],
        "product": [
            field(
                "product",
                "output_product",
                "Product",
                kind="choice",
                choices=("image", "frequency_hz"),
            ),
            field(
                "product", "echo_spacing_seconds", "Echo spacing (s)", nullable=True, minimum=1e-12
            ),
            field("product", "product_unwrap", "Unwrap product phase", kind="boolean"),
        ],
    }
    additions = {
        "profile_inversion": {"profile_inversion": True},
        "readout_regrid": {"readout_regrid": True},
        "epi_phase": {"epi_phase_correction": "entropy"},
        "window": {"kspace_window": ["hann"] * dimensions},
        "partial_fourier": {"partial_fourier": "pocs"},
    }
    stages = []
    for kind, label in LABELS.items():
        reason = None
        if acquisition is not None:
            if kind in {"gridding", "nufft", "profile_inversion"} and acquisition.kind != "sampled":
                reason = "Requires sampled k-space and a trajectory."
            elif kind == "partial_fourier" and acquisition.kind != "cartesian":
                reason = "Requires Cartesian k-space."
            elif kind == "profile_inversion" and acquisition.readout_model is None:
                reason = "No readout-profile model is available."
            elif kind == "readout_regrid" and acquisition.readout_coordinates is None:
                reason = "No nonuniform readout coordinates are available."
            elif kind == "epi_phase" and (
                acquisition.kind != "cartesian" or acquisition.readout_polarity is None
            ):
                reason = "Requires Cartesian readout polarities."
        stages.append(
            {
                "kind": kind,
                "label": label,
                "order": _ORDER[kind],
                "solver": kind in SOLVERS,
                "fields": fields[kind],
                "defaults": defaults[kind],
                "insert_options": additions.get(kind, {}),
                "available": reason is None,
                "reason": reason,
                "note": "No acquisition density weights are available; the NUFFT adjoint is unweighted."
                if kind == "nufft"
                and acquisition is not None
                and acquisition.density is None
                and acquisition.readout_model is None
                else None,
                "configured": False if acquisition is None else configured_stage(kind, acquisition),
            }
        )
    presets = reconstruction_presets()
    for preset in presets:
        encoding = preset["plan"]["encoding"]
        preset["compatible"] = acquisition is None or encoding in {"any", acquisition.kind}
    pipeline = workspace.pipelines.get(session.selected_experiment) or {}
    previous = next((s for s in pipeline.get("steps", []) if s["kind"] == "reconstruct"), None)
    saved = (
        (session.current_experiment.get("kspace_metadata") or {}).get("reconstruction_pipeline")
        if session.is_loaded
        else None
    )
    previous_settings = reconstruction_settings((previous or {}).get("options", {}))
    plan = previous_settings.get("pipeline") or saved or default_plan(acquisition)
    return json_value(
        {
            "prepared": acquisition is not None,
            "encoding": None if acquisition is None else acquisition.kind,
            "encoding_dimensions": dimensions,
            "stages": stages,
            "presets": presets,
            "plan": plan,
            "component": previous_settings.get("component", "magnitude"),
            "label": reconstruction_label(workspace),
        }
    )


def run_reconstruction(workspace, options, progress, *, device="cpu"):
    session = workspace.session
    if not session.can_reconstruct_raw_data():
        session.load_raw_data_for_experiment(progress_callback=progress, require_acquisition=True)
    acquisition = session.current_experiment.get("acquisition")
    if acquisition is None or not session.can_reconstruct_raw_data():
        raise ValueError("Load prepared raw k-space before reconstructing")
    value = normalize_plan(options.get("pipeline") or default_plan(acquisition))
    try:
        plan = compile_plan(acquisition, value)
        progress(0, 1, "Applying " + value["name"])
        result = session.reconstruct_raw_data_for_experiment(
            device=device,
            pipeline=plan,
            reconstruction_name=value["name"],
            plan_settings=value,
        )
    except (TypeError, IndexError) as error:
        raise ValueError("Invalid reconstruction stage settings: " + str(error)) from error
    progress(1, 1, value["name"] + " complete")
    return {
        "reconstructed": True,
        "name": value["name"],
        "method": result.method,
        "stages": [s.kind for s in plan.stages],
        "pipeline": copy.deepcopy(value),
    }
