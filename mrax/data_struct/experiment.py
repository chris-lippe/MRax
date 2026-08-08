from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Mapping

import numpy as np

from .dataset import DataSet
from .dimensions import dim_observable_map, frame_count_from_dims, non_spatial_names, spatial_names
from .quantity import Quantity, require_quantity_dict

if TYPE_CHECKING:
    from .model import Model


@dataclass
class Experiment:
    """Describes the acquisition setup (dims, observables such as TE/TR/offsets)."""

    dims: Dict[str, int]
    observables: Dict[str, Quantity] = field(default_factory=dict)
    exp_type: str | None = None

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.dims.values())

    @property
    def spatial_dim_names(self) -> list[str]:
        return spatial_names(self.dims)

    @property
    def non_spatial_dim_names(self) -> list[str]:
        return non_spatial_names(self.dims)

    @property
    def spatial_shape(self) -> tuple[int, ...]:
        return tuple(self.dims[name] for name in self.spatial_dim_names)

    @property
    def non_spatial_shape(self) -> tuple[int, ...]:
        return tuple(self.dims[name] for name in self.non_spatial_dim_names)

    @property
    def frame_count(self) -> int:
        return frame_count_from_dims(self.dims)

    def describe_dims(self) -> str:
        dims = ", ".join(f"{name}={size}" for name, size in self.dims.items())
        obs = ", ".join(f"{name}: {list(val.shape)}" for name, val in self.observables.items())
        obs_str = obs if obs else "none"
        return f"dims({dims}); observables({obs_str})"

    def merged_observables(self, study_globals: Mapping[str, Quantity] | None = None) -> dict[str, Quantity]:
        """Return observables merged with study-level globals (if available)."""
        globals_map = (
            study_globals
            or getattr(self, "study_globals", None)
            or {}
        )
        globals_map = require_quantity_dict(dict(globals_map), "Study global")
        merged = dict(globals_map)
        merged.update(self.observables)
        return merged

    def insert_observables(
        self,
        updates: Mapping[str, object] | None,
        *,
        units: Mapping[str, str] | None = None,
        overwrite: bool = True,
        in_place: bool = False,
    ) -> "Experiment":
        """Return an Experiment with additional observables merged in.

        ``updates`` values may be Quantity, raw arrays (with ``units``), or (values, unit) tuples.
        """
        if not updates:
            return self

        merged = dict(self.observables)
        units = units or {}
        for key, value in updates.items():
            if not overwrite and key in merged:
                continue
            if isinstance(value, Quantity):
                merged[key] = value
            elif isinstance(value, (tuple, list)) and len(value) == 2:
                vals, unit = value
                merged[key] = Quantity(vals, str(unit))
            else:
                unit = units.get(key)
                if unit is None:
                    raise ValueError(f"Observable '{key}' requires an explicit unit when inserted as raw values")
                merged[key] = Quantity(value, str(unit))

        if in_place:
            self.observables = merged
            return self

        new_exp = Experiment(dims=dict(self.dims), observables=merged, exp_type=self.exp_type)
        for attr in (
            "study_globals",
            "_mrax_dim_observables",
            "_mrax_default_identity_fit_warning",
            "_mrax_manual_type_override_warning",
        ):
            if hasattr(self, attr):
                setattr(new_exp, attr, getattr(self, attr))
        return new_exp

    def describe(
        self,
        dataset: DataSet | None = None,
        *,
        max_values: int = 6,
        max_samples: int = 100000,
    ) -> str:
        """Return a detailed, human-readable summary of this experiment."""
        lines: list[str] = []
        lines.append(f"Experiment exp_type={self.exp_type or 'unknown'}")
        lines.append(f"dims={self.dims}")
        lines.append(f"frame_count={self.frame_count}, spatial_shape={self.spatial_shape}")
        dim_map = getattr(self, "_mrax_dim_observables", None)
        if not dim_map:
            dim_map = dim_observable_map(self.exp_type, self.exp_type)
        if dim_map:
            lines.append(f"dim_map={{{', '.join(f'{k}:{v}' for k, v in dim_map.items())}}}")

        if dataset is not None:
            data = np.asarray(dataset.data)
            sample = data.ravel()
            if sample.size > max_samples:
                step = max(1, sample.size // max_samples)
                sample = sample[::step]
            stats = f"min={float(np.nanmin(sample)):.4g}, max={float(np.nanmax(sample)):.4g}, mean={float(np.nanmean(sample)):.4g}"
            lines.append(f"dataset shape={dataset.data.shape}, dtype={data.dtype}, {stats}")

        if not self.observables:
            lines.append("observables: none")
            return "\n".join(lines)

        lines.append("observables:")
        for key in sorted(self.observables.keys()):
            qty = self.observables[key]
            arr = np.asarray(qty.values)
            shape = arr.shape
            unit = qty.si_unit
            summary = ""
            if arr.ndim == 0:
                summary = f"value={float(arr):.4g}"
            elif arr.ndim == 1 and arr.size <= max_values:
                summary = f"values={np.array2string(arr, precision=4, separator=', ')}"
            else:
                flat = arr.reshape(-1)
                if flat.size > max_samples:
                    step = max(1, flat.size // max_samples)
                    flat = flat[::step]
                summary = (
                    f"min={float(np.nanmin(flat)):.4g}, max={float(np.nanmax(flat)):.4g}, "
                    f"mean={float(np.nanmean(flat)):.4g}"
                )
                if arr.ndim == 1:
                    uniq = np.unique(np.round(arr.astype(np.float64), 6)).size
                    summary += f", unique={int(uniq)}"
            lines.append(f"  - {key}: shape={shape}, unit={unit}, {summary}")
        return "\n".join(lines)

    def help(self) -> str:
        """Return a quick-start help summary for this experiment."""
        from mrax.library.default_experiments import get_default_experiment

        exp_type = self.exp_type or "unknown"
        spec = get_default_experiment(exp_type)
        default_model = None
        if spec is not None:
            try:
                default_model = spec.model_factory().name
            except Exception:
                default_model = spec.name

        lines = [
            f"Experiment help (exp_type={exp_type})",
            f"dims={self.dims}",
            f"observables={sorted(self.observables.keys()) or ['none']}",
            f"default_model={default_model or 'unknown'}",
        ]

        if exp_type in ("sat_trans",):
            lines.append("available models: wasabi, multipool_lorentzian, ab_mt_n, zaiss_full")
        if exp_type == "fair_asl":
            lines.append("available models: fair_asl (default), t1_ir")
        if exp_type in ("ir", "vtr", "vfa"):
            lines.append("available models: t1_ir, t1_vtr, t1_vfa")
        if exp_type in ("mse", "mge"):
            lines.append("available models: t2_se, t2star_se")
        if exp_type in ("dce",):
            lines.append("available models: identity (default); use mrax.analysis.dce for concentration analysis")
        if exp_type in ("dti", "dki"):
            lines.append("available models: dti_tensor (default), dki")

        lines.append("tips: use experiment.describe(dataset) for value summaries.")
        lines.append("tips: use experiment.merged_observables() to include study globals.")
        lines.append("tips: use experiment.insert_observables({...}) to add maps like r1_map/b0_map/b1_map.")
        lines.append("tips: use experiment.explain_sanity(model, dataset) to check fit readiness.")
        lines.append("tips: use recipe=... with load_bruker_study(...) for custom scan typing and model selection.")
        return "\n".join(lines)

    def sanity_check(
        self,
        model: "Model",
        dataset: DataSet,
        *,
        study_globals: Mapping[str, Quantity] | None = None,
    ):
        """Run a pre-fit sanity check for this experiment/model/dataset."""
        from mrax.core.fit_sanity import check_fit_sanity

        return check_fit_sanity(self, model, dataset, study_globals=study_globals)

    def explain_sanity(
        self,
        model: "Model",
        dataset: DataSet,
        *,
        study_globals: Mapping[str, Quantity] | None = None,
    ) -> str:
        """Return a human-readable sanity check report."""
        report = self.sanity_check(model, dataset, study_globals=study_globals)
        return report.explain()

    def validate(self, dataset: DataSet):
        if tuple(dataset.shape[: len(self.dims)]) != self.shape:
            raise ValueError(
                f"Dataset shape {dataset.shape} does not match experiment dims {self.shape}."
            )
        require_quantity_dict(self.observables, "Observable")
        geometry = dataset.spatial_geometry
        if geometry is not None and geometry.spatial_shape is not None:
            if tuple(geometry.spatial_shape) != tuple(self.spatial_shape):
                raise ValueError(
                    f"Spatial geometry shape {geometry.spatial_shape} does not match "
                    f"experiment spatial shape {self.spatial_shape}."
                )
