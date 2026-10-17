from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from functools import wraps
from typing import Callable, Dict, List, Mapping, Sequence, Tuple

import jax.numpy as jnp
import numpy as np

from mrax.opt.regularizers import make_regularizer_fn

from .quantity import Quantity


def _normalize_unit(unit: str) -> str:
    return unit.strip().lower()


def _is_any_unit(unit: str) -> bool:
    return _normalize_unit(unit) == "any"


@dataclass
class Model:
    """Model specification (forward, parametrization, regularization)."""

    name: str
    model_apply: Callable[[dict, jnp.ndarray, object | None], jnp.ndarray]
    parametrization_factory: Callable[[tuple[int, int]], Tuple[dict, Callable[[dict], dict]]]
    regularizer_type: str | dict[str, str] = "smoothness"
    regularizer_configs: dict[str, List[dict]] | None = None
    reg_weights: dict[str, float] | None = None
    parameter_units: Dict[str, str] | None = None
    derived_map_units: Dict[str, str] | None = None
    parameter_constant_dims: Dict[str, tuple[int | str, ...]] | None = None
    observable_units: Dict[str, str] | None = None
    required_observables: Sequence[str] | None = None
    provided_observables: Sequence[str] | None = None
    parameter_specs: dict[str, dict[str, object]] | None = None

    def __post_init__(self):
        if self.parameter_units is None or not self.parameter_units:
            raise ValueError(f"Model '{self.name}' must declare parameter_units for all parameters")
        # Ensure unit strings are present and non-empty.
        for key, unit in self.parameter_units.items():
            if not isinstance(unit, str) or not unit.strip():
                raise ValueError(f"Model '{self.name}' has an invalid SI unit for parameter '{key}'")
        if self.derived_map_units is not None:
            for key, unit in self.derived_map_units.items():
                if not isinstance(unit, str) or not unit.strip():
                    raise ValueError(
                        f"Model '{self.name}' has an invalid SI unit for derived map '{key}'"
                    )
        if self.observable_units is not None:
            if not isinstance(self.observable_units, dict):
                raise TypeError(f"Model '{self.name}' observable_units must be a dict of observable -> unit")
            normalized_units: Dict[str, str] = {}
            for key, unit in self.observable_units.items():
                if not isinstance(key, str) or not key.strip():
                    raise ValueError(f"Model '{self.name}' has an invalid observable name '{key}'")
                if not isinstance(unit, str) or not unit.strip():
                    raise ValueError(
                        f"Model '{self.name}' has an invalid unit for observable '{key}'"
                    )
                normalized_units[str(key)] = str(unit)
            self.observable_units = normalized_units
            lookup: dict[str, str] = {}
            for key, unit in normalized_units.items():
                key_norm = key.strip().lower()
                if key_norm in lookup:
                    if _normalize_unit(lookup[key_norm]) != _normalize_unit(unit):
                        raise ValueError(
                            f"Model '{self.name}' observable_units has conflicting units for '{key}'"
                        )
                    continue
                lookup[key_norm] = unit
            self._mrax_observable_units_ci = lookup
        else:
            self._mrax_observable_units_ci = {}
        if self.parameter_constant_dims is not None:
            if not isinstance(self.parameter_constant_dims, dict):
                raise TypeError(
                    f"Model '{self.name}' parameter_constant_dims must be a dict of parameter -> dims"
                )
            for key, dims in self.parameter_constant_dims.items():
                if not isinstance(dims, (list, tuple)):
                    raise TypeError(
                        f"Model '{self.name}' constant dims for '{key}' must be a list or tuple"
                    )
                for dim in dims:
                    if not isinstance(dim, (int, str)):
                        raise TypeError(
                            f"Model '{self.name}' constant dims for '{key}' must be axis indices or names"
                        )
        for field_name in ("required_observables", "provided_observables"):
            value = getattr(self, field_name)
            if value is None:
                continue
            if not isinstance(value, Sequence) or isinstance(value, str):
                raise TypeError(f"Model '{self.name}' {field_name} must be a sequence of observable names")
            setattr(self, field_name, tuple(str(v) for v in value))
        if self.required_observables:
            if not self.observable_units:
                raise ValueError(
                    f"Model '{self.name}' requires observable_units for required observables"
                )
            missing = [key for key in self.required_observables if key not in self.observable_units]
            if missing:
                missing_str = ", ".join(sorted(missing))
                raise ValueError(
                    f"Model '{self.name}' missing observable_units for required observables: {missing_str}"
                )

        # Guard direct model_apply calls so users use make_model_callable instead.
        if not getattr(self, "_mrax_guarded_model_apply", False):
            raw_apply = self.model_apply

            allowed_prefixes = (
                "mrax.core",
                "mrax.reco",
                "mrax.io",
                "mrax.preprocessing",
                "mrax.analysis",
                "tests",
            )

            def _is_internal_caller() -> bool:
                frame = sys._getframe(2)
                while frame is not None:
                    mod_name = frame.f_globals.get("__name__", "")
                    if any(mod_name.startswith(prefix) for prefix in allowed_prefixes):
                        return True
                    frame = frame.f_back
                return False

            @wraps(raw_apply)
            def guarded_apply(fields, offsets, observables=None, *args, **kwargs):
                if kwargs.pop("_allow_direct", False):
                    return raw_apply(fields, offsets, observables, *args, **kwargs)
                if os.environ.get("MRAX_ALLOW_MODEL_APPLY", "") == "1":
                    return raw_apply(fields, offsets, observables, *args, **kwargs)
                if _is_internal_caller():
                    return raw_apply(fields, offsets, observables, *args, **kwargs)
                raise RuntimeError(
                    f"Direct calls to Model.model_apply are disabled for '{self.name}'. "
                    "Use make_model_callable(experiment, model, study_globals=...) or "
                    "run_fit(..., return_model_callable=True) instead."
                )

            self._raw_model_apply = raw_apply
            self.model_apply = guarded_apply
            self._mrax_guarded_model_apply = True

    def make_regularizer_terms(
        self,
        spatial_shape: tuple[int, int],
        regularizer_configs: dict[str, List[dict]] | None = None,
        regularizer_type: str | dict[str, str] | None = None,
        reg_weights: dict[str, float] | None = None,
        smoothed_tv: bool = False,
    ):
        reg_terms: dict[str, list[tuple[float, Callable[[jnp.ndarray], jnp.ndarray]]]] = {}
        configs_to_use = regularizer_configs if regularizer_configs is not None else self.regularizer_configs
        reg_type_to_use = regularizer_type if regularizer_type is not None else self.regularizer_type
        weights_to_use = reg_weights if reg_weights is not None else self.reg_weights

        if configs_to_use is not None:
            for key, configs in configs_to_use.items():
                entries = []
                for cfg in configs:
                    weight = cfg.get("weight", 0.0)
                    if not np.isfinite(weight) or weight < 0:
                        raise ValueError("Regularizer weights must be finite and nonnegative")
                    if weight == 0.0:
                        continue
                    reg_type = cfg.get("type", "smoothness")
                    axes = cfg.get("axes")
                    fn = make_regularizer_fn(reg_type, axes, smoothed_tv=smoothed_tv)
                    if fn is None:
                        continue
                    entries.append((weight, fn))
                if entries:
                    reg_terms[key] = entries
        else:
            def resolve_type(param: str) -> str:
                if isinstance(reg_type_to_use, dict):
                    return reg_type_to_use.get(param, "smoothness")
                return reg_type_to_use

            candidate_keys = list(weights_to_use or {})
            for key in candidate_keys:
                weight = weights_to_use.get(key, 0.0) if weights_to_use else 0.0
                if not np.isfinite(weight) or weight < 0:
                    raise ValueError("Regularizer weights must be finite and nonnegative")
                if weight == 0.0:
                    continue
                reg_type = resolve_type(key)
                fn = make_regularizer_fn(reg_type, None, smoothed_tv=smoothed_tv)
                if fn is None:
                    continue
                reg_terms.setdefault(key, []).append((weight, fn))
        return reg_terms

    def validate_observable_units(
        self,
        observables: Mapping[str, object] | None,
        *,
        allow_raw: bool = False,
        context: str = "Observable",
    ) -> None:
        """Validate observable units against the model's expected units."""
        if not observables:
            return
        expected = getattr(self, "_mrax_observable_units_ci", {}) or {}
        if not expected:
            return
        for key, value in observables.items():
            if not isinstance(key, str):
                continue
            expected_unit = expected.get(key.strip().lower())
            if expected_unit is None:
                continue
            if value is None:
                continue
            if isinstance(value, Quantity):
                actual_unit = value.si_unit
                if _is_any_unit(expected_unit) or _is_any_unit(actual_unit):
                    continue
                if _normalize_unit(expected_unit) != _normalize_unit(actual_unit):
                    raise ValueError(
                        f"{context} '{key}' for model '{self.name}' must use unit "
                        f"'{expected_unit}', got '{actual_unit}'"
                    )
            else:
                if allow_raw:
                    continue
                raise TypeError(
                    f"{context} '{key}' for model '{self.name}' must be a Quantity with unit "
                    f"'{expected_unit}' (use unit 'any' to bypass strict checks)"
                )

    def observable_requirements(self) -> dict[str, tuple]:
        """Return required/optional observable hints based on model metadata and name."""
        required = list(self.required_observables or ())
        optional: list[str] = []
        any_of: list[tuple[str, ...]] = []
        notes: list[str] = []

        def add_required(*names: str):
            for name in names:
                if name and name not in required:
                    required.append(name)

        def add_optional(*names: str):
            for name in names:
                if name and name not in optional:
                    optional.append(name)

        name = (self.name or "").lower()
        if "wasabi" in name:
            add_required("freq_offsets", "sat_b1", "sat_time", "field_strength_mhz")
        if "multipool_lorentzian" in name:
            add_required("freq_offsets", "field_strength_mhz")
            add_optional("b0_map", "b0_shift", "b0_shift_ppm")
        if "zaiss_full" in name or "zaiss" in name:
            add_required("freq_offsets", "sat_b1", "sat_time", "field_strength_mhz")
            add_optional("r1_map", "t1_map", "b0_map", "b1_map", "spin_lock")
        if name.startswith("ab_mt") or "ab_mt" in name:
            add_required("freq_offsets", "sat_b1", "sat_time", "TR", "field_strength_mhz")
            any_of.append(("r1_map", "t1_map"))
            add_optional(
                "b0_map",
                "b1_map",
                "z_prev",
                "z_reference_image",
                "spin_lock",
                "mt_offset",
                "mt_line_shape",
            )
            notes.append(
                "AB-MT models need r1_map or t1_map; insert it from a prior T1 fit "
                "(Experiment.insert_observables) and consider adding b0_map/b1_map from WASABI."
            )
        if name in ("t1_ir", "t1"):
            add_required("TI")
        if name == "fair_asl":
            add_required("TI", "fair_label")
        if name == "t1_vtr":
            add_required("TR")
        if name == "t1_vfa":
            add_required("TR", "flip_angle")
        if name in ("t2_se", "t2", "mse"):
            add_required("TE")
        if name in ("t2star_se", "t2star", "mge"):
            add_required("TE")
        if name in ("dti", "dti_tensor", "diffusion", "dki"):
            add_required("bvals", "bvecs")

        return {
            "required": tuple(required),
            "optional": tuple(optional),
            "any_of": tuple(any_of),
            "notes": tuple(notes),
        }

    def help(self) -> str:
        """Return a quick-start help summary for this model."""
        lines = [f"Model help: {self.name}"]
        reqs = self.observable_requirements()
        if reqs["required"]:
            lines.append(f"required_observables={list(reqs['required'])}")
        if reqs["any_of"]:
            lines.append(f"requires_any_of={list(reqs['any_of'])}")
        if reqs["optional"]:
            lines.append(f"optional_observables={list(reqs['optional'])}")
        if self.provided_observables:
            lines.append(f"provided_observables={list(self.provided_observables)}")
        if reqs["notes"]:
            lines.extend(reqs["notes"])

        if self.parameter_units:
            params = ", ".join(sorted(self.parameter_units.keys()))
            lines.append(f"parameters={params}")
        if self.regularizer_type:
            lines.append(f"default_regularizer={self.regularizer_type}")
        lines.append("tips: use run_fit(..., mask=..., num_steps=..., step_size=...) to fit")
        return "\n".join(lines)
