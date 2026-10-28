"""Editable, vendor-independent reconstruction stage plans.

Plans declaratively describe algorithms. Importers may suggest a plan;
editing it preserves the acquired samples and their validity mask.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from operator import index
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from mrax.data_struct.acquisition import AcquisitionData

_RECIPE_KEYS = {
    "readout_regrid": {"readout_regrid", "readout_interpolation", "readout_kernel_radius"},
    "profile_inversion": {"profile_inversion"},
    "epi_phase": {"epi_phase_correction", "epi_phase_prior"},
    "shift": {"image_shift", "frame_image_shifts"},
    "partial_fourier": {"partial_fourier", "partial_fourier_iterations", "phase_window", "homodyne_weighting", "partial_fourier_tolerance"},
    "gridding": {"gridding_shape", "gridding_kernel_width", "gridding_shape_parameter", "deapodize", "deapodization_normalization"},
    "format": {"output_shape", "crop_offset", "output_permutation", "output_flips", "frame_permutations", "stack_slices", "output_stack_permutation", "output_frame_indices", "checkerboard", "elliptical_support", "output_scale", "encoding_scale"},
    "combine": {"coil_combine", "average_repeats"},
    "product": {"output_product", "echo_spacing_seconds", "product_unwrap", "product_unwrap_method", "product_mask_snr", "product_signal_fraction", "product_corner_fraction"},
    "window": {"kspace_window", "window_parameters", "window_powers"},
}
_SOLVERS = {"fft", "variational", "model_based"}
_ORDER = {"profile_inversion": -3, "shift": -2, "readout_regrid": -1, "epi_phase": 0, "window": 1, "partial_fourier": 2, "gridding": 2, "fft": 3,
          "variational": 3, "model_based": 3, "format": 4, "combine": 5, "product": 6}


@dataclass(frozen=True)
class ReconstructionStage:
    """One named algorithm and explicit options; unknown names are rejected."""

    kind: str
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.kind not in _ORDER:
            raise ValueError(f"Unknown reconstruction stage: {self.kind}")
        options = dict(self.options)
        if self.kind in _RECIPE_KEYS and set(options) - _RECIPE_KEYS[self.kind]:
            raise ValueError(f"Unknown options for reconstruction stage: {self.kind}")
        if self.kind == "fft":
            if set(options) - {"frame_batch_size"}:
                raise ValueError("Unknown FFT options")
            batch_size = options.get("frame_batch_size")
            if batch_size is not None and index(batch_size) < 0:
                raise ValueError("frame_batch_size must be nonnegative or None")
        object.__setattr__(self, "options", MappingProxyType(options))


@dataclass(frozen=True)
class ReconstructionPipeline:
    """Ordered stage plan for the shared acquisition reconstruction entry point.

    Shift and partial-Fourier/gridding stages precede exactly one solver. Output
    formatting and coil combination follow it. Partial-Fourier values initialize
    iterative solvers; the acquired support defines their data term.
    """

    stages: tuple[ReconstructionStage, ...]

    def __post_init__(self):
        stages = tuple(self.stages)
        names = [stage.kind for stage in stages]
        if len(names) != len(set(names)):
            raise ValueError("Reconstruction stages must not be repeated")
        if sum(name in _SOLVERS for name in names) != 1:
            raise ValueError("A reconstruction pipeline requires exactly one solver")
        if "format" not in names or "combine" not in names:
            raise ValueError("Output format and coil combination must be explicit stages")
        order = [_ORDER[name] for name in names]
        if order != sorted(order):
            raise ValueError("Reconstruction stages are in an invalid order")
        if "partial_fourier" in names and "gridding" in names:
            raise ValueError("Partial Fourier and sampled-data gridding require different encodings")
        object.__setattr__(self, "stages", stages)

    @classmethod
    def from_acquisition(cls, acquisition: AcquisitionData, *, method="fft", **optimizer_options):
        """Make the importer's suggested numerical operations visible/editable."""
        recipe = acquisition.recipe
        stages = []
        for kind in ("profile_inversion", "shift", "readout_regrid", "epi_phase", "window", "partial_fourier", "gridding"):
            if kind == "window" and not recipe.kspace_window:
                continue
            if kind == "readout_regrid" and acquisition.readout_coordinates is None:
                continue
            if kind == "profile_inversion" and not recipe.profile_inversion:
                continue
            if kind == "epi_phase" and recipe.epi_phase_correction == "none":
                continue
            if kind == "shift" and not (recipe.image_shift or recipe.frame_image_shifts):
                continue
            if kind == "partial_fourier" and recipe.partial_fourier == "none":
                continue
            if kind == "gridding" and acquisition.kind != "sampled":
                continue
            stages.append(ReconstructionStage(kind, {key: getattr(recipe, key) for key in _RECIPE_KEYS[kind]}))
        stages.append(ReconstructionStage(method, optimizer_options))
        for kind in ("format", "combine"):
            stages.append(ReconstructionStage(kind, {key: getattr(recipe, key) for key in _RECIPE_KEYS[kind]}))
        if recipe.output_product != "image":
            stages.append(ReconstructionStage("product", {key: getattr(recipe, key) for key in _RECIPE_KEYS["product"]}))
        return cls(tuple(stages))

    @classmethod
    def from_toml(cls, path: str | Path):
        """Read a declarative plan using allowlisted keys and values."""
        import tomllib

        with Path(path).open("rb") as handle:
            document = tomllib.load(handle)
        if set(document) != {"schema_version", "stages"} or document["schema_version"] != 1:
            raise ValueError("Unsupported reconstruction plan schema")
        stages = []
        for item in document["stages"]:
            if set(item) - {"kind", "options"} or "kind" not in item:
                raise ValueError("A stage requires kind and optional options")
            stages.append(ReconstructionStage(item["kind"], item.get("options", {})))
        return cls(tuple(stages))

    def replace_stage(self, kind, *, replacement=None, **options):
        """Return a modified plan; the original acquisition remains untouched."""
        if kind not in {stage.kind for stage in self.stages}:
            raise ValueError(f"No reconstruction stage named {kind}")
        return type(self)(tuple(ReconstructionStage(replacement or kind, {**stage.options, **options})
            if stage.kind == kind else stage for stage in self.stages))

    def run(self, acquisition: AcquisitionData, *, device="cpu", **context):
        from .acquisition import reconstruct_acquisition

        names = {stage.kind for stage in self.stages}
        if acquisition.kind == "sampled" and "gridding" not in names:
            raise ValueError("Sampled acquisitions require an explicit gridding stage")
        if acquisition.kind == "cartesian" and "gridding" in names:
            raise ValueError("Cartesian acquisitions cannot use a sampled-data gridding stage")
        recipe = replace(acquisition.recipe, kspace_window=(), window_parameters=(), output_product="image", profile_inversion=False, readout_regrid=False, epi_phase_correction="none", image_shift=(), frame_image_shifts=(), partial_fourier="none")
        solver = None
        solver_options = {}
        for stage in self.stages:
            if stage.kind in _SOLVERS:
                solver = stage.kind
                solver_options = dict(stage.options)
            else:
                options = dict(stage.options)
                # TOML arrays become immutable tuples in the numerical recipe.
                for key, value in options.items():
                    if isinstance(value, list):
                        options[key] = tuple(tuple(row) if isinstance(row, list) else row for row in value)
                recipe = replace(recipe, **options)
        if set(context) & set(solver_options):
            raise ValueError("Optimizer options must have a single explicit source")
        return reconstruct_acquisition(replace(acquisition, recipe=recipe), method=solver,
            device=device, **solver_options, **context)
