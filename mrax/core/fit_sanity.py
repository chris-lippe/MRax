from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import jax
import numpy as np

from mrax.data_struct import DataSet, Experiment, Model, Quantity
from mrax.data_struct.dimensions import frame_count_from_dims, non_spatial_names
from mrax.data_struct.quantity import require_quantity_dict


@dataclass
class FitSanityReport:
    """Structured report describing whether a fit is ready to run."""

    experiment_name: str
    model_name: str
    dataset_shape: tuple[int, ...]
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    missing_observables: list[str] = field(default_factory=list)
    missing_any_of: list[tuple[str, ...]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (self.errors or self.missing_observables or self.missing_any_of)

    def explain(self) -> str:
        lines: list[str] = []
        lines.append("Fit sanity check")
        lines.append(f"model={self.model_name}, dataset_shape={self.dataset_shape}")
        if self.errors:
            lines.append("errors:")
            lines.extend(f"  - {msg}" for msg in self.errors)
        if self.missing_observables:
            lines.append("missing_observables:")
            lines.append("  - " + ", ".join(self.missing_observables))
        if self.missing_any_of:
            lines.append("missing_any_of:")
            for group in self.missing_any_of:
                lines.append("  - " + " or ".join(group))
        if self.warnings:
            lines.append("warnings:")
            lines.extend(f"  - {msg}" for msg in self.warnings)
        if self.notes:
            lines.append("notes:")
            lines.extend(f"  - {msg}" for msg in self.notes)
        if self.ok:
            lines.append("status: ok")
        else:
            lines.append("status: fix required items before running run_fit")
        return "\n".join(lines)


@dataclass
class FitDebugReport:
    """Structured prototype-fit report for debugging model inputs and one optimizer step."""

    experiment_name: str
    model_name: str
    dataset_shape: tuple[int, ...]
    optimizer_name: str
    step_size: float
    sanity: FitSanityReport
    field_shape: tuple[int, ...] = field(default_factory=tuple)
    field_dim_names: tuple[str, ...] = field(default_factory=tuple)
    data_summary: str = ""
    frame_data_summary: str = ""
    merged_observable_summaries: dict[str, str] = field(default_factory=dict)
    expanded_observable_summaries: dict[str, str] = field(default_factory=dict)
    initial_param_summaries: dict[str, str] = field(default_factory=dict)
    initial_field_summaries: dict[str, str] = field(default_factory=dict)
    gradient_summaries: dict[str, str] = field(default_factory=dict)
    update_summaries: dict[str, str] = field(default_factory=dict)
    optimizer_state_init_summary: str = ""
    optimizer_state_after_step_summary: str = ""
    predicted_before_summary: str = ""
    predicted_after_summary: str = ""
    initial_loss: float | None = None
    loss_after_one_step: float | None = None
    prototype_ran: bool = False
    prototype_error: str | None = None
    notes: list[str] = field(default_factory=list)

    def explain(self) -> str:
        lines = [
            "Fit debug prototype",
            f"model={self.model_name}, dataset_shape={self.dataset_shape}",
            f"optimizer={self.optimizer_name}, step_size={self.step_size}",
            f"field_shape={self.field_shape}, field_dims={list(self.field_dim_names)}",
            f"sanity_ok={self.sanity.ok}",
        ]
        if self.data_summary:
            lines.append(f"dataset: {self.data_summary}")
        if self.frame_data_summary:
            lines.append(f"frame_data: {self.frame_data_summary}")
        if self.merged_observable_summaries:
            lines.append("merged_observables:")
            lines.extend(f"  - {key}: {value}" for key, value in sorted(self.merged_observable_summaries.items()))
        if self.expanded_observable_summaries:
            lines.append("expanded_observables:")
            lines.extend(f"  - {key}: {value}" for key, value in sorted(self.expanded_observable_summaries.items()))
        if self.initial_param_summaries:
            lines.append("initial_params:")
            lines.extend(f"  - {key}: {value}" for key, value in sorted(self.initial_param_summaries.items()))
        if self.initial_field_summaries:
            lines.append("initial_fields:")
            lines.extend(f"  - {key}: {value}" for key, value in sorted(self.initial_field_summaries.items()))
        if self.initial_loss is not None:
            lines.append(f"initial_loss={self.initial_loss:.6g}")
        if self.predicted_before_summary:
            lines.append(f"predicted_before: {self.predicted_before_summary}")
        if self.gradient_summaries:
            lines.append("gradients:")
            lines.extend(f"  - {key}: {value}" for key, value in sorted(self.gradient_summaries.items()))
        if self.update_summaries:
            lines.append("updates:")
            lines.extend(f"  - {key}: {value}" for key, value in sorted(self.update_summaries.items()))
        if self.optimizer_state_init_summary:
            lines.append(f"optimizer_state_init: {self.optimizer_state_init_summary}")
        if self.optimizer_state_after_step_summary:
            lines.append(f"optimizer_state_after_step: {self.optimizer_state_after_step_summary}")
        if self.loss_after_one_step is not None:
            lines.append(f"loss_after_one_step={self.loss_after_one_step:.6g}")
        if self.predicted_after_summary:
            lines.append(f"predicted_after: {self.predicted_after_summary}")
        if self.prototype_error:
            lines.append(f"prototype_error: {self.prototype_error}")
        if self.notes:
            lines.append("notes:")
            lines.extend(f"  - {msg}" for msg in self.notes)
        if self.prototype_ran:
            lines.append("status: prototype step completed")
        elif self.prototype_error:
            lines.append("status: prototype step failed")
        else:
            lines.append("status: prototype step skipped")
        return "\n".join(lines)


def _key_matches(name: str, candidates: set[str]) -> bool:
    if name in candidates:
        return True
    lower = name.lower()
    for cand in candidates:
        if cand.lower() == lower:
            return True
    return False


def _has_observable(name: str, keys: set[str]) -> bool:
    if _key_matches(name, keys):
        return True
    if name == "freq_offsets":
        for alias in ("offsets", "offset", "freq_offsets"):
            if _key_matches(alias, keys):
                return True
    return False


def _format_number(value: float) -> str:
    return f"{value:.6g}"


def _summarize_array(arr: Any, *, include_norm: bool = False, max_values: int = 6) -> str:
    data = np.asarray(arr)
    summary = f"shape={tuple(int(v) for v in data.shape)}, dtype={data.dtype}"
    if data.size == 0:
        return summary + ", empty"
    if np.issubdtype(data.dtype, np.number):
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            summary += ", all_non_finite"
        else:
            summary += (
                f", min={_format_number(float(np.min(finite)))}, "
                f"max={_format_number(float(np.max(finite)))}, "
                f"mean={_format_number(float(np.mean(finite)))}"
            )
            if include_norm:
                summary += f", l2={_format_number(float(np.linalg.norm(finite.reshape(-1))))}"
        if data.ndim <= 1 and data.size <= max_values:
            summary += f", values={np.array2string(data, precision=4, separator=', ')}"
        return summary
    if data.ndim <= 1 and data.size <= max_values:
        return summary + f", values={data.tolist()}"
    return summary


def _summarize_quantity_map(mapping: Mapping[str, Quantity]) -> dict[str, str]:
    return {
        key: f"unit={value.si_unit}, {_summarize_array(value.values)}"
        for key, value in mapping.items()
    }


def _summarize_array_map(mapping: Mapping[str, Any], *, include_norm: bool = False) -> dict[str, str]:
    return {
        key: _summarize_array(value, include_norm=include_norm)
        for key, value in mapping.items()
    }


def _summarize_tree_state(tree: Any) -> str:
    leaves = jax.tree_util.tree_leaves(tree)
    numeric_leaves = []
    total_values = 0
    for leaf in leaves:
        try:
            arr = np.asarray(leaf)
        except Exception:
            continue
        total_values += int(arr.size)
        if np.issubdtype(arr.dtype, np.number) and arr.size:
            numeric_leaves.append(arr.reshape(-1))
    summary = f"leaves={len(leaves)}, total_values={total_values}"
    if numeric_leaves:
        merged = np.concatenate(numeric_leaves)
        finite = merged[np.isfinite(merged)]
        if finite.size:
            summary += (
                f", min={_format_number(float(np.min(finite)))}, "
                f"max={_format_number(float(np.max(finite)))}, "
                f"mean={_format_number(float(np.mean(finite)))}"
            )
    return summary


def _validate_initial_params(default: Any, candidate: Any):
    if candidate is None:
        return default
    tree_default = jax.tree_util.tree_structure(default)
    tree_candidate = jax.tree_util.tree_structure(candidate)
    if tree_default != tree_candidate:
        raise ValueError("initial_params do not match the expected parameter tree")

    def _same_shape(a, b):
        if hasattr(a, "shape") and hasattr(b, "shape") and a.shape != b.shape:
            raise ValueError("initial_params contain arrays with mismatched shapes")
        return b

    jax.tree_util.tree_map(_same_shape, default, candidate)
    return candidate


def check_fit_sanity(
    experiment: Experiment,
    model: Model,
    dataset: DataSet,
    *,
    study_globals: Mapping[str, Quantity] | None = None,
) -> FitSanityReport:
    """Validate experiment/model/dataset consistency before running a fit."""

    report = FitSanityReport(
        experiment_name=experiment.exp_type or "experiment",
        model_name=model.name,
        dataset_shape=tuple(dataset.data.shape),
    )

    try:
        experiment.validate(dataset)
    except Exception as exc:
        report.errors.append(str(exc))

    globals_map = (
        study_globals
        or getattr(model, "_mrax_study_globals", None)
        or getattr(experiment, "study_globals", None)
        or {}
    )
    try:
        global_obs = require_quantity_dict(dict(globals_map), "Study global")
    except Exception as exc:
        report.errors.append(str(exc))
        global_obs = {}

    try:
        exp_obs = require_quantity_dict(dict(experiment.observables), "Observable")
    except Exception as exc:
        report.errors.append(str(exc))
        exp_obs = {}

    if exp_obs or global_obs:
        try:
            merged_obs = {**global_obs, **exp_obs}
            model.validate_observable_units(merged_obs, allow_raw=False, context="Observable")
        except Exception as exc:
            report.errors.append(str(exc))

    merged_keys = set(exp_obs) | set(global_obs)

    reqs = model.observable_requirements()
    required = list(reqs.get("required", ()))
    any_of = list(reqs.get("any_of", ()))

    missing = [name for name in required if not _has_observable(name, merged_keys)]
    report.missing_observables.extend(missing)
    for group in any_of:
        if not any(_has_observable(name, merged_keys) for name in group):
            report.missing_any_of.append(tuple(group))

    data = np.asarray(dataset.data)
    if not np.isfinite(data).all():
        report.warnings.append("dataset contains NaNs/Infs; consider auto_mask_nan_observables=True or a mask")

    frame_count = frame_count_from_dims(experiment.dims)
    non_spatial = non_spatial_names(experiment.dims)
    dim_sizes = {name: int(experiment.dims[name]) for name in non_spatial}
    for key, qty in exp_obs.items():
        arr = np.asarray(qty.values)
        if arr.ndim == 1:
            if arr.shape[0] not in (frame_count, 1) and arr.shape[0] not in dim_sizes.values():
                report.warnings.append(
                    f"observable '{key}' length {arr.shape[0]} does not match frame count or any non-spatial dim"
                )

    report.notes.extend(reqs.get("notes", ()))
    return report


def debug_fit_prototype(
    experiment: Experiment,
    model: Model,
    dataset: DataSet,
    *,
    optimizer_name: str = "adam",
    step_size: float = 0.02,
    regularizer_weight=0.0,
    regularizer_prox=None,
    optimizer_base_factory=None,
    box_constraints=None,
    initial_params: dict | None = None,
    regularizer_configs: dict[str, list[dict]] | None = None,
    regularizer_type: str | dict[str, str] | None = None,
    regularizer_weights: dict[str, float] | None = None,
    optimizer_factory=None,
    optimizer=None,
    mask: object | None = None,
    mask_dims: tuple[str, ...] | None = None,
    mask_fill_value: float | int | np.ndarray | None = 0.0,
    study_globals: Mapping[str, Quantity] | None = None,
    frame_mask: object | None = None,
    data_variance: object | None = None,
    variance_estimation_mask: object | None = None,
    variance_estimation_mask_dims: tuple[str, ...] | None = None,
    auto_mask_nan_observables: bool = True,
) -> FitDebugReport:
    """Run a single non-jitted prototype optimization step and summarize the fit state."""

    import optax

    from mrax.core.loss_builder import (
        _align_predicted,
        _field_metadata,
        _flatten_to_frames,
        build_loss,
        make_model_callable,
    )
    from mrax.data_struct.dimensions import expand_observables_to_frames
    from mrax.opt.optimizers import make_optimizer

    sanity = check_fit_sanity(experiment, model, dataset, study_globals=study_globals)
    field_shape, field_dim_names, _ = _field_metadata(experiment, model=model)
    report = FitDebugReport(
        experiment_name=experiment.exp_type or "experiment",
        model_name=model.name,
        dataset_shape=tuple(dataset.data.shape),
        optimizer_name=optimizer_name,
        step_size=float(step_size),
        sanity=sanity,
        field_shape=field_shape,
        field_dim_names=tuple(field_dim_names),
        data_summary=_summarize_array(dataset.data),
        frame_data_summary=_summarize_array(_flatten_to_frames(np.asarray(dataset.data), experiment)),
    )

    if not sanity.ok:
        report.notes.append("prototype step skipped because the sanity check did not pass")
        return report

    globals_map = (
        study_globals
        or getattr(model, "_mrax_study_globals", None)
        or getattr(experiment, "study_globals", None)
        or {}
    )
    global_obs = require_quantity_dict(dict(globals_map), "Study global")
    exp_obs = require_quantity_dict(dict(experiment.observables), "Observable")
    merged_observables = {**global_obs, **exp_obs}
    report.merged_observable_summaries = _summarize_quantity_map(merged_observables)

    dim_map = getattr(experiment, "_mrax_dim_observables", None)
    observable_arrays = expand_observables_to_frames(merged_observables, experiment.dims, dim_map)
    report.expanded_observable_summaries = _summarize_array_map(observable_arrays)

    try:
        loss_fn, default_params, unpack = build_loss(
            experiment,
            model,
            dataset,
            regularizer_configs=regularizer_configs,
            regularizer_type=regularizer_type,
            regularizer_weights=regularizer_weights,
            mask=mask,
            mask_dims=mask_dims,
            mask_fill_value=mask_fill_value,
            study_globals=dict(global_obs),
            frame_mask=frame_mask,
            data_variance=data_variance,
            variance_estimation_mask=variance_estimation_mask,
            variance_estimation_mask_dims=variance_estimation_mask_dims,
            auto_mask_nan_observables=auto_mask_nan_observables,
        )
        params = _validate_initial_params(default_params, initial_params)
        fields_before = unpack(params)
        report.initial_param_summaries = _summarize_array_map(params)
        report.initial_field_summaries = {
            key: f"unit={model.parameter_units.get(key, '')}, {_summarize_array(value)}"
            for key, value in fields_before.items()
        }

        offsets_array = observable_arrays.get("offsets")
        model_fn = make_model_callable(
            experiment,
            model,
            dict(global_obs),
            allow_raw_observables=True,
        )
        predicted_before = _align_predicted(
            np.asarray(model_fn(fields_before, offsets_array, observable_arrays)),
            experiment,
        )
        report.predicted_before_summary = _summarize_array(predicted_before)

        if optimizer is not None:
            opt = optimizer
        elif optimizer_factory is not None:
            opt = optimizer_factory(step_size)
        else:
            opt = make_optimizer(
                optimizer_name,
                step_size,
                regularizer_weight=regularizer_weight,
                regularizer_prox=regularizer_prox,
                base_optimizer_factory=optimizer_base_factory,
                box_constraints=box_constraints,
            )
        opt_state_init = opt.init(params)
        report.optimizer_state_init_summary = _summarize_tree_state(opt_state_init)

        loss_before, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state_after = opt.update(grads, opt_state_init, params)
        params_after = optax.apply_updates(params, updates)
        loss_after = loss_fn(params_after)
        fields_after = unpack(params_after)
        predicted_after = _align_predicted(
            np.asarray(model_fn(fields_after, offsets_array, observable_arrays)),
            experiment,
        )

        report.initial_loss = float(loss_before)
        report.loss_after_one_step = float(loss_after)
        report.gradient_summaries = _summarize_array_map(grads, include_norm=True)
        report.update_summaries = _summarize_array_map(updates, include_norm=True)
        report.optimizer_state_after_step_summary = _summarize_tree_state(opt_state_after)
        report.predicted_after_summary = _summarize_array(predicted_after)
        if not np.isfinite(report.initial_loss):
            report.notes.append("initial loss is non-finite")
        if not np.isfinite(report.loss_after_one_step):
            report.notes.append("loss after one step is non-finite")
        report.prototype_ran = True
    except Exception as exc:
        report.prototype_error = str(exc)
    return report


__all__ = [
    "FitDebugReport",
    "FitSanityReport",
    "check_fit_sanity",
    "debug_fit_prototype",
]
