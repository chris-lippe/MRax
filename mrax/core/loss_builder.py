from __future__ import annotations

import warnings
from dataclasses import replace
from typing import Callable, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from mrax.data_struct.dataset import DataSet
from mrax.data_struct.dimensions import dim_observable_map, expand_observables_to_frames
from mrax.data_struct.experiment import Experiment
from mrax.data_struct.model import Model
from mrax.data_struct.quantity import Quantity, attach_units, require_quantity_dict
from mrax.data_struct.study import Study
from mrax.devices import DeviceLike, bind_device, device_function, jit_on_device, resolve_device

from .fit_execution import FitError, FitState, execute_optimizer
from .variance import VarianceEstimate, estimate_data_variance


def warn_if_fit_requires_model_override(experiment: Experiment, model: Model) -> None:
    """Warn when an imported default model is being used as a fit model."""

    model_name = str(getattr(model, "name", "")).lower()
    identity_message = getattr(experiment, "_mrax_default_identity_fit_warning", None)
    manual_message = getattr(experiment, "_mrax_manual_type_override_warning", None)
    if identity_message and model_name == "identity":
        message = identity_message
    elif manual_message:
        message = manual_message
    else:
        message = None
    if not message:
        return
    warnings.warn(str(message), RuntimeWarning, stacklevel=3)


def _resolve_globals(experiment, model, explicit):
    if explicit is not None:
        return explicit
    current = getattr(experiment, "study_globals", None)
    return current if current is not None else getattr(model, "_mrax_study_globals", None)


def _prepare_reference_image(reference_image, experiment):
    """Validate a fixed spatial reference for signal-domain forward fitting."""
    if reference_image is None:
        return None
    shape, _ = _extract_spatial_metadata(experiment)
    value = reference_image.values if isinstance(reference_image, Quantity) else reference_image
    reference = jnp.asarray(value)
    while reference.ndim > len(shape) and reference.shape[0] == 1:
        reference = reference[0]
    if tuple(reference.shape) != tuple(shape):
        raise ValueError("reference_image must match the experiment spatial shape")
    if not np.isfinite(np.asarray(reference)).all():
        raise ValueError("reference_image must be finite")
    return reference


def _rescale_previous_signal(previous, previous_reference, reference):
    """Keep a measured previous signal fixed when changing its normalization."""
    values = jnp.asarray(previous)
    if values.ndim == 1:
        values = values.reshape((-1,) + (1,) * reference.ndim)
    denominator = reference.reshape((1,) + reference.shape)
    nonzero = jnp.abs(denominator) > 0
    safe = jnp.where(nonzero, denominator, 1)
    return jnp.where(nonzero, values * jnp.asarray(previous_reference) / safe, 0)


def _prepare_mask_arrays(
    mask: object | None,
    mask_dims: tuple[str, ...] | None,
    experiment: Experiment,
    field_shape: tuple[int, ...],
    field_dim_names: list[str],
    spatial_dim_names: list[str],
) -> tuple[jnp.ndarray | None, jnp.ndarray | None]:
    """Validate and broadcast a mask for data and field shapes."""

    if mask is None:
        return None, None

    mask_array = mask.values if isinstance(mask, Quantity) else mask
    mask_array = jnp.asarray(mask_array)
    mask_array = _sanitize_mask_array(mask_array)

    dims_to_use = tuple(mask_dims) if mask_dims is not None else tuple(spatial_dim_names)
    if any(dim not in field_dim_names for dim in dims_to_use):
        raise ValueError(f"Mask dimensions {dims_to_use} must be a subset of field dims {field_dim_names}")
    if mask_array.ndim != len(dims_to_use):
        raise ValueError(
            f"Mask has {mask_array.ndim} dims but mask_dims specifies {len(dims_to_use)} axes"
        )

    expected = tuple(experiment.dims[dim] for dim in dims_to_use)
    if tuple(mask_array.shape) != expected:
        raise ValueError(
            f"Mask shape {mask_array.shape} does not match experiment dims {expected} for {dims_to_use}"
        )

    idx_map = {dim: idx for idx, dim in enumerate(dims_to_use)}
    if len(idx_map) != len(dims_to_use):
        raise ValueError("Mask dimensions must be unique")
    data_order = [idx_map[dim] for dim in experiment.dims if dim in idx_map]
    field_order = [idx_map[dim] for dim in field_dim_names if dim in idx_map]
    data_shape = []
    for dim in experiment.dims.keys():
        if dim in idx_map:
            data_shape.append(mask_array.shape[idx_map[dim]])
        else:
            data_shape.append(1)
    mask_data = jnp.broadcast_to(jnp.reshape(jnp.transpose(mask_array, data_order), tuple(data_shape)), experiment.shape)

    field_target = []
    for dim in field_dim_names:
        if dim in idx_map:
            field_target.append(mask_array.shape[idx_map[dim]])
        else:
            field_target.append(1)
    mask_fields = jnp.broadcast_to(jnp.reshape(jnp.transpose(mask_array, field_order), tuple(field_target)), field_shape)
    return mask_data, mask_fields


def _prepare_frame_mask(mask: object | None, experiment: Experiment) -> jnp.ndarray | None:
    """Validate and flatten a per-frame mask."""
    if mask is None:
        return None
    mask_array = mask.values if isinstance(mask, Quantity) else mask
    mask_array = jnp.asarray(mask_array)
    mask_array = _sanitize_mask_array(mask_array)
    frame_count = int(experiment.frame_count)
    non_spatial = experiment.non_spatial_shape
    if mask_array.ndim == 0:
        return jnp.full((frame_count,), mask_array)
    if mask_array.shape == (frame_count,):
        return mask_array
    if non_spatial and tuple(int(v) for v in mask_array.shape) == tuple(int(v) for v in non_spatial):
        return mask_array.reshape((frame_count,))
    if int(mask_array.size) == frame_count:
        return mask_array.reshape((frame_count,))
    raise ValueError(
        f"frame_mask shape {tuple(int(v) for v in mask_array.shape)} does not match frame_count={frame_count}"
    )


def _frame_mask_from_observables(
    observables: dict[str, jnp.ndarray], experiment: Experiment
) -> jnp.ndarray | None:
    """Return a per-frame mask that drops NaN/inf entries in per-frame observables."""
    frame_count = int(experiment.frame_count)
    non_spatial = tuple(int(v) for v in experiment.non_spatial_shape)
    mask = None
    for arr in observables.values():
        arr = jnp.asarray(arr)
        if arr.ndim == 0:
            continue
        finite = None
        if arr.ndim == 1 and int(arr.shape[0]) == frame_count:
            finite = jnp.isfinite(arr)
        elif non_spatial and tuple(int(v) for v in arr.shape) == non_spatial:
            finite = jnp.isfinite(arr).reshape((frame_count,))
        elif arr.size == frame_count and (not non_spatial or arr.ndim <= len(non_spatial)):
            finite = jnp.isfinite(arr).reshape((frame_count,))
        if finite is not None:
            mask = finite if mask is None else (mask & finite)
    return mask


def _data_mask_from_observables(
    observables: dict[str, jnp.ndarray], experiment: Experiment
) -> jnp.ndarray | None:
    """Return a per-voxel data mask based on NaN/inf entries in observables."""
    mask = None
    for arr in observables.values():
        try:
            arr = jnp.asarray(arr)
            if arr.ndim == 0:
                continue
            finite = jnp.isfinite(arr)
            if finite.ndim == 1 and finite.shape[0] == experiment.frame_count:
                finite = finite.reshape((experiment.frame_count,) + (1,) * len(experiment.spatial_shape))
            finite = jnp.broadcast_to(finite, (experiment.frame_count,) + experiment.spatial_shape)
            finite = _unflatten_from_frames(finite, experiment)
        except Exception:
            continue
        mask = finite if mask is None else (mask & finite)
    return mask


def _reduce_mask_to_field(
    mask_data: jnp.ndarray | None,
    experiment: Experiment,
    field_dim_names: Sequence[str],
) -> jnp.ndarray | None:
    """Reduce a data mask to the field shape by collapsing explained dims."""
    if mask_data is None:
        return None
    dim_names = list(experiment.dims.keys())
    reduce_axes = [idx for idx, name in enumerate(dim_names) if name not in field_dim_names]
    mask_bool = jnp.asarray(mask_data) > 0
    if reduce_axes:
        mask_bool = jnp.any(mask_bool, axis=tuple(reduce_axes))
    remaining = [name for name in dim_names if name in field_dim_names]
    return jnp.transpose(mask_bool, [remaining.index(name) for name in field_dim_names])


def _sanitize_observables(observables: dict[str, jnp.ndarray]) -> dict[str, jnp.ndarray]:
    """Replace NaN/inf entries in observable arrays with zeros."""
    sanitized: dict[str, jnp.ndarray] = {}
    for key, arr in observables.items():
        try:
            arr = jnp.asarray(arr)
            if arr.ndim == 0:
                sanitized[key] = arr
                continue
            finite = jnp.isfinite(arr)
            fill = jnp.zeros((), dtype=arr.dtype)
            sanitized[key] = jnp.where(finite, arr, fill)
        except Exception:
            sanitized[key] = arr
    return sanitized


def _sanitize_mask_array(mask: jnp.ndarray | None) -> jnp.ndarray | None:
    """Replace NaN/inf entries in mask arrays with zeros."""
    if mask is None:
        return None
    arr = jnp.asarray(mask)
    if np.any(np.asarray(arr) < 0):
        raise ValueError("Masks must have nonnegative weights")
    try:
        finite = jnp.isfinite(arr)
        fill = jnp.zeros((), dtype=arr.dtype)
        return jnp.where(finite, arr, fill)
    except Exception:
        return arr


def _mask_for_regularized_field(
    mask: jnp.ndarray | None,
    field: jnp.ndarray,
) -> jnp.ndarray | None:
    """Collapse a full field mask along parameter-constant singleton axes."""
    if mask is None:
        return None
    mask_arr = jnp.asarray(mask)
    field_arr = jnp.asarray(field)
    if mask_arr.shape == field_arr.shape:
        return mask_arr
    if mask_arr.ndim != field_arr.ndim:
        return mask_arr
    for axis, (mask_size, field_size) in enumerate(zip(mask_arr.shape, field_arr.shape)):
        if mask_size != field_size and field_size == 1:
            mask_arr = jnp.max(mask_arr, axis=axis, keepdims=True)
    return mask_arr


def _masked_rms(data: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """Compute the root-mean-square magnitude over finite masked entries.

    RMS scaling remains well conditioned for constant non-zero images, including
    those with zero standard deviation.
    """
    data_arr = jnp.asarray(data)
    mask_arr = jnp.asarray(mask)
    real_dtype = jnp.real(data_arr).dtype
    weight = mask_arr.astype(real_dtype)
    safe = jnp.where(mask_arr, data_arr, jnp.zeros((), dtype=data_arr.dtype))
    denom = jnp.sum(weight) + 1e-8
    mean_square = jnp.sum(weight * jnp.abs(safe) ** 2) / denom
    return jnp.sqrt(mean_square)


def _prepare_data_variance(
    data_variance: object | None,
    experiment: Experiment,
    data_shape: tuple[int, ...],
) -> jnp.ndarray:
    """Validate observation variances and broadcast them to flattened data."""
    if data_variance is None:
        return jnp.ones(data_shape)

    variance = data_variance.values if isinstance(data_variance, Quantity) else data_variance
    variance_array = jnp.asarray(variance)
    target_shape = tuple(int(size) for size in data_shape)
    frame_count = int(experiment.frame_count)
    spatial_shape = tuple(int(size) for size in experiment.spatial_shape)
    non_spatial_shape = tuple(int(size) for size in experiment.non_spatial_shape)

    try:
        if variance_array.ndim == 0:
            prepared = jnp.broadcast_to(variance_array, target_shape)
        elif tuple(variance_array.shape) == tuple(experiment.shape):
            prepared = _flatten_to_frames(variance_array, experiment)
        elif tuple(variance_array.shape) == target_shape:
            prepared = variance_array
        elif tuple(variance_array.shape) in {(frame_count,), non_spatial_shape}:
            prepared = variance_array.reshape((frame_count,) + (1,) * len(spatial_shape))
            prepared = jnp.broadcast_to(prepared, target_shape)
        elif spatial_shape and tuple(variance_array.shape) == spatial_shape:
            prepared = variance_array.reshape((1,) + spatial_shape)
            prepared = jnp.broadcast_to(prepared, target_shape)
        else:
            prepared = jnp.broadcast_to(variance_array, tuple(experiment.shape))
            prepared = _flatten_to_frames(prepared, experiment)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "data_variance must be scalar, per-frame, spatial, or broadcastable "
            f"to data shape {tuple(experiment.shape)}"
        ) from exc

    variance_values = np.asarray(prepared)
    if np.iscomplexobj(variance_values):
        raise ValueError("data_variance must be real")
    if not np.isfinite(variance_values).all() or np.any(variance_values <= 0):
        raise ValueError("data_variance must contain only finite positive values")
    return jnp.asarray(prepared)


def _resolve_data_variance(
    data_variance: object | None,
    experiment: Experiment,
    dataset: DataSet,
    data_shape: tuple[int, ...],
    variance_estimation_mask: object | None,
    variance_estimation_mask_dims: tuple[str, ...] | None,
) -> tuple[jnp.ndarray, VarianceEstimate | None]:
    estimate = None
    variance = data_variance
    if isinstance(data_variance, str):
        if data_variance.lower() != "auto":
            raise ValueError("data_variance string value must be 'auto'")
        estimate = estimate_data_variance(
            experiment,
            dataset,
            mask=variance_estimation_mask,
            mask_dims=variance_estimation_mask_dims,
        )
        variance = estimate.variance
    return _prepare_data_variance(variance, experiment, data_shape), estimate


def _extract_spatial_metadata(experiment: Experiment) -> tuple[tuple[int, ...], list[str]]:
    """Return the spatial shape and names, skipping leading frame/time dims."""
    names = experiment.spatial_dim_names
    return experiment.spatial_shape, names


def _dim_map_for_experiment(experiment: Experiment, model: Model | None = None) -> dict[str, str | None]:
    dim_map = getattr(experiment, "_mrax_dim_observables", None)
    if dim_map is None:
        dim_map = dim_observable_map(experiment.exp_type, getattr(model, "name", None))
    return dim_map or {}


def _has_per_frame_observables(experiment: Experiment) -> bool:
    """Return True if any observable varies across frames."""
    observables = getattr(experiment, "observables", None) or {}
    if not observables:
        return False
    frame_count = int(experiment.frame_count)
    if frame_count <= 1:
        return False
    non_spatial = tuple(int(v) for v in experiment.non_spatial_shape)
    for qty in observables.values():
        try:
            arr = np.asarray(qty.values if isinstance(qty, Quantity) else qty)
        except Exception:
            continue
        if arr.ndim == 0:
            continue
        if int(arr.shape[0]) == frame_count:
            return True
        if non_spatial and tuple(int(v) for v in arr.shape[: len(non_spatial)]) == non_spatial:
            return True
        if int(arr.size) == frame_count:
            return True
    return False


def _unexplained_dim_names(experiment: Experiment, model: Model | None = None) -> list[str]:
    """Return implicit non-spatial parameter axes, such as repeats named 't'."""
    dim_map = _dim_map_for_experiment(experiment, model=model)
    has_frame_observables = _has_per_frame_observables(experiment)
    unexplained: list[str] = []
    for name in experiment.non_spatial_dim_names:
        if int(experiment.dims.get(name, 1)) <= 1:
            continue
        obs_name = dim_map.get(name)
        if obs_name is not None:
            continue
        if name == "frame" and name not in dim_map and has_frame_observables:
            continue
        # Only treat explicit None mappings or known repeat dims as unexplained by default.
        if (name in dim_map) or (name in ("t", "frame")):
            unexplained.append(name)
    return unexplained


def _field_metadata(
    experiment: Experiment, model: Model | None = None
) -> tuple[tuple[int, ...], list[str], list[str]]:
    """Return field shape/names and unexplained dim names."""
    extra_dims = _unexplained_dim_names(experiment, model=model)
    field_dim_names = list(extra_dims) + list(experiment.spatial_dim_names)
    field_shape = tuple(int(experiment.dims[name]) for name in field_dim_names)
    return field_shape, field_dim_names, extra_dims


def _expand_fields_to_frames(
    fields: dict[str, jnp.ndarray],
    experiment: Experiment,
    extra_dim_names: Sequence[str],
) -> dict[str, jnp.ndarray]:
    """Broadcast fields along explained dims to per-frame arrays for model_apply."""
    if not extra_dim_names:
        return {k: (v.values if isinstance(v, Quantity) else v) for k, v in fields.items()}

    non_spatial_names = experiment.non_spatial_dim_names
    non_spatial_shape = tuple(int(experiment.dims[name]) for name in non_spatial_names)
    spatial_shape = experiment.spatial_shape
    frame_count = int(experiment.frame_count)
    extra_set = set(extra_dim_names)
    expected_field_ndim = len(extra_dim_names) + len(spatial_shape)

    def expand_value(val):
        if isinstance(val, dict):
            return {k: expand_value(v) for k, v in val.items()}
        if isinstance(val, Quantity):
            val = val.values
        arr = jnp.asarray(val)
        if arr.ndim == 0:
            return arr
        if arr.ndim >= len(spatial_shape) + 1 and int(arr.shape[0]) == frame_count:
            if arr.ndim == len(spatial_shape) + 1:
                return arr
            lead = arr.shape[1 : arr.ndim - len(spatial_shape)]
            if all(int(s) == 1 for s in lead):
                return arr.reshape((frame_count,) + spatial_shape)
        if arr.ndim == expected_field_ndim + 1 and int(arr.shape[0]) == frame_count:
            return arr
        if arr.ndim < expected_field_ndim:
            arr = arr.reshape((1,) * (expected_field_ndim - arr.ndim) + arr.shape)
        if arr.ndim != expected_field_ndim:
            return arr
        shape = []
        extra_axis = 0
        for name in non_spatial_names:
            if name in extra_set:
                shape.append(arr.shape[extra_axis])
                extra_axis += 1
            else:
                shape.append(1)
        shape.extend(arr.shape[extra_axis:])
        arr = arr.reshape(tuple(shape))
        arr = jnp.broadcast_to(arr, non_spatial_shape + spatial_shape)
        return arr.reshape((frame_count,) + spatial_shape)

    return {k: expand_value(v) for k, v in fields.items()}


def _flatten_to_frames(arr: jnp.ndarray, experiment: Experiment) -> jnp.ndarray:
    spatial_shape = experiment.spatial_shape
    frame_count = experiment.frame_count
    if arr.shape == experiment.shape:
        dim_names = list(experiment.dims.keys())
        non_spatial = experiment.non_spatial_dim_names
        spatial = experiment.spatial_dim_names
        axis_order = [dim_names.index(name) for name in non_spatial + spatial]
        if axis_order != list(range(arr.ndim)):
            arr = jnp.transpose(arr, axis_order)
        return arr.reshape((frame_count,) + spatial_shape)
    if arr.ndim == len(spatial_shape):
        return jnp.broadcast_to(arr, (frame_count,) + spatial_shape)
    if arr.ndim == len(spatial_shape) + 1 and int(arr.shape[0]) == frame_count:
        return arr
    return arr


def _unflatten_from_frames(arr: jnp.ndarray, experiment: Experiment) -> jnp.ndarray:
    """Reshape frame-major arrays back to experiment dim order."""
    spatial_shape = experiment.spatial_shape
    frame_count = experiment.frame_count
    non_spatial = experiment.non_spatial_dim_names
    spatial = experiment.spatial_dim_names
    if arr.ndim == len(spatial_shape) + 1 and int(arr.shape[0]) == frame_count:
        target = tuple(int(experiment.dims[name]) for name in non_spatial + spatial)
        arr = arr.reshape(target)
    dim_names = list(experiment.dims.keys())
    canonical = list(non_spatial) + list(spatial)
    if canonical != dim_names and arr.ndim == len(dim_names):
        perm = [canonical.index(name) for name in dim_names]
        arr = jnp.transpose(arr, perm)
    return arr


def _align_predicted(predicted: jnp.ndarray, experiment: Experiment) -> jnp.ndarray:
    spatial_shape = experiment.spatial_shape
    frame_count = experiment.frame_count
    arr = predicted
    while arr.ndim > len(spatial_shape) + 1 and int(arr.shape[0]) == 1:
        arr = jnp.squeeze(arr, axis=0)
    if len(spatial_shape) == 0 and arr.ndim > 1 and int(arr.shape[0]) == frame_count:
        tail_shape = tuple(int(v) for v in arr.shape[1:])
        if tail_shape and all(size == 1 for size in tail_shape):
            return arr.reshape((frame_count,))
    if spatial_shape and arr.ndim > len(spatial_shape) + 1 and int(arr.shape[0]) == frame_count:
        tail_shape = tuple(int(v) for v in arr.shape[-len(spatial_shape) :])
        middle_shape = tuple(int(v) for v in arr.shape[1 : arr.ndim - len(spatial_shape)])
        if tail_shape == tuple(int(v) for v in spatial_shape) and all(size == 1 for size in middle_shape):
            return arr.reshape((frame_count,) + spatial_shape)
    if arr.shape == experiment.shape:
        return _flatten_to_frames(arr, experiment)
    if arr.shape == spatial_shape:
        return jnp.broadcast_to(arr, (frame_count,) + spatial_shape)
    if arr.ndim == len(spatial_shape) + 2 and int(arr.shape[0]) == 1 and int(arr.shape[1]) == frame_count:
        return arr.reshape((frame_count,) + spatial_shape)
    if arr.ndim == len(spatial_shape) + 1 and int(arr.shape[0]) == frame_count:
        return arr
    return arr


def _normalize_constant_dims(
    constant_dims: dict[str, tuple[int | str, ...]] | None,
    field_dim_names: list[str],
    field_shape: tuple[int, ...],
) -> dict[str, tuple[int, ...]]:
    """Convert per-parameter constant-dim specs into axis indices."""

    if not constant_dims:
        return {}

    normalized: dict[str, tuple[int, ...]] = {}
    for name, dims in constant_dims.items():
        axes: list[int] = []
        for dim in dims:
            if isinstance(dim, str):
                if dim not in field_dim_names:
                    raise ValueError(
                        f"Unknown dimension '{dim}' in parameter_constant_dims for '{name}'"
                    )
                axes.append(field_dim_names.index(dim))
            else:
                axis = int(dim)
                if axis < 0 or axis >= len(field_shape):
                    raise ValueError(
                        f"Dimension index {axis} out of range for parameter '{name}' "
                        f"(field_shape={field_shape})"
                    )
                axes.append(axis)
        if axes:
            normalized[name] = tuple(sorted(set(axes)))
    return normalized


def _base_name(name: str) -> str:
    return name[:-4] if name.endswith("_raw") else name


def _constant_axes_for_name(name: str, constant_dims: dict[str, tuple[int, ...]]):
    if not constant_dims:
        return None
    if name in constant_dims:
        return constant_dims[name]
    return constant_dims.get(_base_name(name))


def _collapse_value_along_axes(value, axes: tuple[int, ...]):
    arr = jnp.asarray(value)
    for axis in sorted(set(axes)):
        if axis >= arr.ndim:
            raise ValueError(
                f"Cannot enforce constancy along axis {axis} for parameter with shape {arr.shape}"
            )
        arr = jnp.take(arr, jnp.array([0]), axis=axis)
    return arr


def _collapse_params_for_constant_dims(
    params: dict, constant_dims: dict[str, tuple[int, ...]]
) -> dict:
    """Reduce parameter shapes for axes marked constant."""

    if not constant_dims:
        return params

    collapsed: dict = {}
    for key, value in params.items():
        if isinstance(value, dict):
            collapsed[key] = _collapse_params_for_constant_dims(value, constant_dims)
            continue
        axes = _constant_axes_for_name(key, constant_dims)
        collapsed[key] = _collapse_value_along_axes(value, axes) if axes else value
    return collapsed


def _broadcast_fields_to_field_shape(
    fields: dict[str, jnp.ndarray],
    constant_dims: dict[str, tuple[int, ...]],
    field_shape: tuple[int, ...],
) -> dict[str, jnp.ndarray]:
    """Broadcast fitted fields back to the full field shape for downstream use."""

    if not constant_dims:
        return fields

    broadcasted: dict[str, jnp.ndarray] = {}
    for key, value in fields.items():
        axes = _constant_axes_for_name(key, constant_dims)
        if axes and hasattr(value, "shape"):
            arr = jnp.asarray(value)
            if arr.ndim != len(field_shape):
                broadcasted[key] = arr
                continue
            target_shape = list(arr.shape)
            for axis in axes:
                target_shape[axis] = field_shape[axis]
            broadcasted[key] = jnp.broadcast_to(arr, tuple(target_shape))
        else:
            broadcasted[key] = value
    return broadcasted


def _mask_fields_for_output(
    fields: dict[str, jnp.ndarray],
    mask_fields: jnp.ndarray | None,
    *,
    fill_value: float | int | jnp.ndarray | None = 0.0,
):
    """Apply mask to output fields, optionally filling masked values."""

    if mask_fields is None or fill_value is None:
        return fields

    mask_bool = jnp.asarray(mask_fields) > 0
    masked: dict[str, jnp.ndarray] = {}
    for key, value in fields.items():
        try:
            arr = jnp.asarray(value)
            fill = jnp.asarray(fill_value, dtype=arr.dtype)
            masked[key] = jnp.where(mask_bool, arr, fill)
        except Exception:
            masked[key] = value
    return masked


@device_function
def build_loss(
    experiment: Experiment,
    model: Model,
    dataset: DataSet,
    regularizer_configs: dict[str, list[dict]] | None = None,
    regularizer_type: str | dict[str, str] | None = None,
    regularizer_weights: dict[str, float] | None = None,
    mask: object | None = None,
    mask_dims: tuple[str, ...] | None = None,
    mask_fill_value: float | int | jnp.ndarray | None = 0.0,
    study_globals: dict[str, Quantity] | None = None,
    frame_mask: object | None = None,
    data_variance: object | None = None,
    variance_estimation_mask: object | None = None,
    variance_estimation_mask_dims: tuple[str, ...] | None = None,
    auto_mask_nan_observables: bool = True,
    parameter_space: str = "raw",
    smoothed_tv: bool = False,
    physical_scales: dict[str, float] | None = None,
    balance_parameters: bool = True,
    data_normalization: str = "roi",
    device: DeviceLike = "cpu",
    reference_image: object | None = None,
    *,
    data_loss: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] | None = None,
) -> Tuple[Callable[[dict], jnp.ndarray], dict, Callable[[dict], dict]]:
    """Build loss(params) combining data misfit and model regularization.

    ``data_variance`` supplies known relative observation variances as a scalar,
    per-frame array, spatial array, or array broadcastable to the data. Pass
    ``"auto"`` to robustly estimate per-frame variances before constructing the
    loss. The data misfit is an inverse-variance-weighted mean of RMS-scaled
    squared residuals. The loss is invariant to a common variance multiplier;
    ``None`` selects uniform variance.

    ``reference_image`` multiplies relative model predictions by a fixed spatial
    reference. Supply measured signal intensities as the data in this mode.

    ``data_normalization="roi"`` computes RMS over observations retained by
    finite-data, observable, ROI and frame masks. ``"full"`` retains the previous
    finite-image normalization. Fractional masks weight residuals; their positive
    support determines the RMS sample set.

    ``data_loss`` optionally maps RMS-scaled residuals and their nonnegative
    weights to a scalar differentiable data misfit. It is independent of the
    optimizer; model regularizers are still added by this function.
    """
    if data_normalization not in ("roi", "full"):
        raise ValueError("data_normalization must be 'roi' or 'full'")
    if data_loss is not None and not callable(data_loss):
        raise TypeError("data_loss must be callable or None")
    experiment.validate(dataset)
    spatial_shape, spatial_dim_names = _extract_spatial_metadata(experiment)
    field_shape, field_dim_names, extra_dim_names = _field_metadata(experiment, model=model)
    constant_dims = _normalize_constant_dims(
        model.parameter_constant_dims, field_dim_names, field_shape
    )
    mask_data, mask_fields = _prepare_mask_arrays(
        mask,
        mask_dims,
        experiment,
        field_shape,
        field_dim_names,
        spatial_dim_names,
    )

    params, unpack = model.parametrization_factory(field_shape)
    params = jax.device_put(params, device)
    unpack = jit_on_device(unpack, device)
    field_keys = set(unpack(params))
    if set(constant_dims) - field_keys:
        raise ValueError("parameter_constant_dims contains unknown parameter fields")
    params = _collapse_params_for_constant_dims(params, constant_dims)
    reg_terms = model.make_regularizer_terms(
        field_shape,
        regularizer_configs=regularizer_configs,
        regularizer_type=regularizer_type,
        reg_weights=regularizer_weights,
        smoothed_tv=smoothed_tv,
    )
    configured_keys = set((regularizer_configs if regularizer_configs is not None else model.regularizer_configs) or {})
    configured_keys.update((regularizer_weights if regularizer_weights is not None else model.reg_weights) or {})
    if configured_keys - field_keys:
        raise ValueError("Regularization contains unknown parameter fields")

    data_flat = _flatten_to_frames(jnp.asarray(dataset.data), experiment)
    data_flat = jnp.asarray(data_flat)
    data_finite = None
    data_mask = None
    data_mask_flat = None
    if data_flat.ndim != 0:
        data_finite = jnp.isfinite(data_flat)
        data_mask_flat = jnp.where(data_finite, 1.0, 0.0)
        data_mask = _unflatten_from_frames(data_mask_flat, experiment)
        data_flat = jnp.where(data_finite, data_flat, jnp.zeros((), dtype=data_flat.dtype))
    observables = require_quantity_dict(experiment.observables, "Observable")
    inferred_globals = _resolve_globals(experiment, model, study_globals)
    global_obs = require_quantity_dict(inferred_globals or {}, "Study global")
    merged_observables = {**global_obs, **observables}
    dim_map = getattr(experiment, "_mrax_dim_observables", None)
    observable_arrays = expand_observables_to_frames(merged_observables, experiment.dims, dim_map)
    obs_data_mask = None
    auto_frame_mask = None
    if auto_mask_nan_observables:
        obs_data_mask = _data_mask_from_observables(observable_arrays, experiment)
        auto_frame_mask = _frame_mask_from_observables(observable_arrays, experiment)
        observable_arrays = _sanitize_observables(observable_arrays)
    offsets_array = observable_arrays.get("offsets")
    model_fn = make_model_callable(
        experiment,
        model,
        inferred_globals,
        allow_raw_observables=True,
        reference_image=reference_image,
    )

    if obs_data_mask is not None:
        mask_data = obs_data_mask if mask_data is None else (mask_data * obs_data_mask)
        obs_field_mask = _reduce_mask_to_field(obs_data_mask, experiment, field_dim_names)
        if obs_field_mask is not None:
            mask_fields = obs_field_mask if mask_fields is None else (mask_fields * obs_field_mask)

    if data_mask is not None:
        mask_data = data_mask if mask_data is None else (mask_data * data_mask)
        data_field_mask = _reduce_mask_to_field(data_mask, experiment, field_dim_names)
        if data_field_mask is not None:
            mask_fields = data_field_mask if mask_fields is None else (mask_fields * data_field_mask)

    # Attach metadata for downstream helpers (initialization/output broadcasting).
    unpack._mrax_constant_field_axes = constant_dims
    unpack._mrax_constant_param_axes = constant_dims
    unpack._mrax_full_spatial_shape = field_shape
    unpack._mrax_extra_dim_names = tuple(extra_dim_names)
    unpack._mrax_mask_data = mask_data
    unpack._mrax_mask_fields = mask_fields
    if parameter_space == "physical":
        from .physical_parameters import physical_parametrization
        params, unpack = physical_parametrization(model, params, unpack)
    elif parameter_space != "raw":
        raise ValueError("parameter_space must be 'raw' or 'physical'")
    unpack._mrax_reg_terms = reg_terms
    tv_operators = {}
    if parameter_space == "physical":
        from mrax.opt.tv import TVOperator
        initial_fields = unpack(params)
        for key in params:
            field = initial_fields[key]
            field_mask = _mask_for_regularized_field(mask_fields, field)
            operators = []
            for weight, fn in reg_terms.get(key, ()):
                if hasattr(fn, "tv_order"):
                    op = TVOperator.create(field.shape, fn.tv_axes, fn.tv_order, field_mask,
                                           weight * unpack._mrax_physical_scales[key], field.dtype)
                    if op.axes:
                        operators.append(op)
            tv_operators[key] = tuple(operators)
        unpack._mrax_tv_operators = tv_operators

    mask_flat = _flatten_to_frames(jnp.asarray(mask_data), experiment) if mask_data is not None else None
    variance_flat, variance_estimate = _resolve_data_variance(
        data_variance,
        experiment,
        dataset,
        tuple(data_flat.shape),
        variance_estimation_mask,
        variance_estimation_mask_dims,
    )
    unpack._mrax_variance_estimate = variance_estimate
    frame_weights = _prepare_frame_mask(frame_mask, experiment)
    if auto_mask_nan_observables and auto_frame_mask is not None:
        frame_weights = auto_frame_mask if frame_weights is None else (frame_weights * auto_frame_mask)

    included_flat = jnp.ones(data_flat.shape, dtype=bool)
    if frame_weights is not None:
        fw = jnp.asarray(frame_weights)
        fw = fw.reshape((int(experiment.frame_count),) + (1,) * len(spatial_shape))
        included_flat = included_flat & jnp.broadcast_to(fw > 0, data_flat.shape)
    if mask_flat is not None:
        included_flat = included_flat & jnp.broadcast_to(mask_flat > 0, data_flat.shape)
    included_host = np.asarray(included_flat)
    if not np.any(included_host):
        raise ValueError("No observations remain after data and frame masking")
    normalization_mask = included_flat if data_normalization == "roi" else data_finite
    scale = 1.0 / jnp.maximum(_masked_rms(data_flat, normalization_mask), 1e-6)
    unpack._mrax_data_normalization = data_normalization
    # Relative inverse variance avoids overflow while retaining the same weights.
    variance_host = np.asarray(variance_flat, dtype=np.float64)
    precision_flat = jnp.asarray(np.min(variance_host[included_host]) / variance_host)
    weights_flat = jnp.where(included_flat, precision_flat, 0.0)
    if frame_weights is not None:
        weights_flat = weights_flat * jnp.broadcast_to(fw, data_flat.shape)
    if mask_flat is not None:
        weights_flat = weights_flat * jnp.broadcast_to(mask_flat, data_flat.shape)

    def smooth_loss(p):
        fields = unpack(p)
        predicted = model_fn(fields, offsets_array, observable_arrays)
        predicted = _align_predicted(jnp.asarray(predicted), experiment)
        residual = (predicted - data_flat) * scale
        if data_loss is None:
            # Preserve the quadratic complex Hessian at a zero residual.
            squared_residual = jnp.real(residual * jnp.conj(residual))
            misfit = jnp.sum(weights_flat * squared_residual) / (jnp.sum(weights_flat) + 1e-8)
        else:
            misfit = jnp.asarray(data_loss(residual, weights_flat))
            if misfit.ndim != 0 or jnp.iscomplexobj(misfit):
                raise ValueError("data_loss must return a real scalar")

        reg_term = 0.0
        for key, field in fields.items():
            configs = reg_terms.get(key, ())
            field_mask = _mask_for_regularized_field(mask_fields, field)
            for weight, reg_fn in configs:
                if parameter_space == "physical" and key in p and hasattr(reg_fn, "tv_order"):
                    continue
                reg_term = reg_term + weight * reg_fn(field, field_mask)
        return misfit + reg_term

    if parameter_space == "physical":
        from .physical_parameters import balance_physical_coordinates
        params = balance_physical_coordinates(smooth_loss, params, unpack, tv_operators,
                                               stored_scales=physical_scales, balance=balance_parameters)

    def loss_fn(p):
        return smooth_loss(p) + sum(op.value(p[key]) for key, operators in tv_operators.items() for op in operators)

    unpack._mrax_smooth_loss = smooth_loss
    return loss_fn, params, unpack


def make_model_callable(
    experiment: Experiment,
    model: Model,
    study_globals: dict[str, Quantity] | None = None,
    *,
    allow_raw_observables: bool = False,
    reference_image: object | None = None,
):
    """Build predictions, optionally multiplying a relative model by a fixed reference."""
    reference = _prepare_reference_image(reference_image, experiment)
    globals_map = require_quantity_dict(
        _resolve_globals(experiment, model, study_globals) or {},
        "Study global",
    )
    default_obs = dict(globals_map)
    default_obs.update(experiment.observables)
    validate_units = getattr(model, "validate_observable_units", None)
    if validate_units is not None:
        validate_units(default_obs, allow_raw=False, context="Observable")

    # Prediction must use the same frame order as loss evaluation when
    # observables are stored along separate experimental dimensions.
    expanded_obs = expand_observables_to_frames(
        default_obs, experiment.dims, _dim_map_for_experiment(experiment, model=model)
    )
    default_obs = {key: default_obs[key].with_values(value) for key, value in expanded_obs.items()}

    base_apply = getattr(model, "model_apply", None)
    if base_apply is None:
        raise AttributeError(f"Model '{model}' does not define model_apply")
    extra_dim_names = _unexplained_dim_names(experiment, model=model)

    def model_fn(fields, offsets=None, observables=None):
        merged = dict(default_obs)
        if observables:
            if validate_units is not None:
                validate_units(observables, allow_raw=allow_raw_observables, context="Observable")
            merged.update(observables)
        offs = offsets
        if offs is None:
            offs = merged.get("offsets")
            if isinstance(offs, Quantity):
                offs = offs.values
        fields_for_model = _expand_fields_to_frames(fields, experiment, extra_dim_names)
        if reference is not None and "z_prev" in merged and "z_reference_image" in merged:
            previous, previous_reference = merged["z_prev"], merged["z_reference_image"]
            previous_values = previous.values if isinstance(previous, Quantity) else previous
            reference_values = previous_reference.values if isinstance(previous_reference, Quantity) else previous_reference
            adjusted = _rescale_previous_signal(previous_values, reference_values, reference)
            merged["z_prev"] = previous.with_values(adjusted) if isinstance(previous, Quantity) else adjusted
        # Dynamic fields retain a leading frame axis alongside the spatial axes.
        merged["_mrax_spatial_ndim"] = len(experiment.spatial_shape)
        prediction = base_apply(fields_for_model, offs, merged)
        if extra_dim_names or reference is not None:
            prediction = _align_predicted(jnp.asarray(prediction), experiment)
        if reference is not None:
            prediction = prediction * reference
        return prediction

    return model_fn


@device_function
def run_fit(
    experiment: Experiment,
    model: Model,
    dataset: DataSet,
    optimizer_name: str | None = None,
    step_size: float | Callable[[jax.Array], jax.Array] = 0.02,
    num_steps: int = 200,
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
    mask_fill_value: float | int | jnp.ndarray | None = 0.0,
    study_globals: dict[str, Quantity] | None = None,
    frame_mask: object | None = None,
    data_variance: object | None = None,
    variance_estimation_mask: object | None = None,
    variance_estimation_mask_dims: tuple[str, ...] | None = None,
    auto_mask_nan_observables: bool = True,
    return_model_callable: bool = False,
    compute_derived_maps: bool | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
    chunk_size: int = 32,
    initial_state: FitState | None = None,
    convergence_tolerance: float | None = None,
    parameter_tolerance: float = 1e-5,
    convergence_patience: int = 5,
    min_steps: int = 0,
    return_fit_info: bool = False,
    proximal_tolerance: float | None = None,
    proximal_max_steps: int | None = None,
    bregman_penalty: float = 1.0,
    bregman_inner_steps: int = 1,
    balance_physical_parameters: bool = True,
    physical_metric: str = "curvature",
    proximal_max_step_size: float | None = None,
    grad_clip_norm: float | None = None,
    data_normalization: str = "roi",
    parameter_space: str = "auto",
    fit_active_only: bool = False,
    device: DeviceLike = "cpu",
    reference_image: object | None = None,
    *,
    fista_restart: str = "gradient",
    fista_backtracking: str = "full",
    data_loss: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] | None = None,
):
    """Fit the model to data for a given experiment.

    Supports Optax optimizers plus ``split_bregman`` and ``fista`` with optional
    proximal regularizers, box constraints, explicit parameter initializations,
    and per-parameter regularizer configs/weights (override model defaults).
    Custom Optax optimizers can be injected via ``optimizer`` or ``optimizer_factory``.
    ``frame_mask`` and ``auto_mask_nan_observables`` let you ignore frames with
    missing per-frame observables (e.g., incomplete sat_time/b1 grids).
    ``data_variance`` applies inverse-variance weighting while retaining RMS loss
    scaling. Pass ``"auto"`` to estimate robust per-frame variances before
    optimization; ``None`` (the default) gives every included sample uniform
    variance.
    ``reference_image`` fits measured signal intensities by multiplying relative
    model predictions by a fixed spatial reference. Observations retain their
    measured signal scale.
    ``variance_estimation_mask`` optionally identifies a background/noise-only
    region for automatic estimation and is independent of the fitting ``mask``.
    ``mask_fill_value`` controls how masked output fields are filled (use ``None`` to skip masking).
    NaN/inf entries in data or masks are treated as masked during fitting.
    ``compute_derived_maps`` controls derived map evaluation. When ``None``, derived maps
    are automatically computed for diffusion models that provide ``derived_maps``.
    ``step_size`` also accepts an Optax-compatible learning-rate schedule.
    ``chunk_size`` bounds device-side execution between progress/finiteness checks.
    Convergence is opt-in: ``convergence_tolerance`` and ``parameter_tolerance``
    must hold for ``convergence_patience`` consecutive updates after ``min_steps``.
    ``initial_state`` resumes raw parameters and optimizer moments from a previous
    :class:`mrax.core.fit_execution.FitInfo`; it is mutually exclusive with ``initial_params``.
    ``return_fit_info`` appends :class:`mrax.core.fit_execution.FitInfo` to the normal return tuple. Its
    ``final_loss`` evaluates the returned parameters; history entries precede updates.
    Nonfinite fits raise :class:`mrax.core.fit_execution.FitError` with the last valid chunk checkpoint.
    ``data_normalization="roi"`` uses retained observations for RMS scaling;
    select ``"full"`` to reproduce the previous finite-image normalization.
    ``physical_metric`` selects a fixed curvature or initial-gradient block
    metric for true-TV fitting; coordinates, boxes and TV weights are rescaled
    together. ``balance_physical_parameters=False`` disables this balancing.
    ``proximal_max_step_size`` optionally caps line-search step growth.
    ``parameter_space="physical"`` also permits affine coordinates with smooth
    model penalties; ``"auto"`` retains the default optimizer-dependent choice.
    ``fit_active_only=True`` packs the positive mask support and native TV graph
    while retaining the original forward-model geometry and output fields.
    ``grad_clip_norm`` optionally clips gradients of built-in optimizers;
    proximal convergence checks still use the unmodified gradient.
    Split Bregman uses one safeguarded Newton/CG primal iteration by default;
    ``bregman_inner_steps`` controls additional primal work before shrinkage
    and the Bregman multiplier update. ``proximal_max_steps`` bounds each CG
    solve. Named, injected and factory-based proximal fits share defaults.
    """
    from mrax.opt.optimizers import make_optimizer

    warn_if_fit_requires_model_override(experiment, model)

    loss_kwargs = dict(
        regularizer_configs=regularizer_configs,
        regularizer_type=regularizer_type,
        regularizer_weights=regularizer_weights,
        mask=mask,
        mask_dims=mask_dims,
        mask_fill_value=mask_fill_value,
        study_globals=study_globals,
        frame_mask=frame_mask,
        data_variance=data_variance,
        variance_estimation_mask=variance_estimation_mask,
        variance_estimation_mask_dims=variance_estimation_mask_dims,
        auto_mask_nan_observables=auto_mask_nan_observables,
        data_normalization=data_normalization,
        data_loss=data_loss,
        device=device,
        reference_image=reference_image,
    )
    loss_fn, default_params, unpack = build_loss(experiment, model, dataset, **loss_kwargs)
    has_tv = any(hasattr(fn, "tv_order") for entries in unpack._mrax_reg_terms.values() for _, fn in entries)
    selected = (optimizer_name or "auto").lower()
    custom_optimizer = optimizer is not None or optimizer_factory is not None
    if optimizer is None and optimizer_factory is not None:
        optimizer = optimizer_factory(step_size)
    native_name = getattr(getattr(optimizer, "update", None), "_mrax_proximal_name", None)
    if native_name is not None:
        selected = native_name
    elif custom_optimizer and selected == "fista":
        raise ValueError("Injected FISTA must retain its native fitting interface; use make_optimizer "
                         "directly or an optimizer_factory, not an Optax chain wrapper for field TV")
    if custom_optimizer and grad_clip_norm is not None:
        raise ValueError("Configure clipping in an injected optimizer or optimizer_factory")
    if selected == "auto":
        selected = "fista" if has_tv and not custom_optimizer else "adam"
    if selected == "bregman":
        selected = "split_bregman"
    if proximal_tolerance is None:
        proximal_tolerance = 1e-8 if selected in ("fista", "split_bregman") else 1e-10
    if proximal_max_steps is None:
        proximal_max_steps = 2048 if selected in ("fista", "split_bregman") else 4096
    if parameter_space not in ("auto", "raw", "physical"):
        raise ValueError("parameter_space must be auto, raw or physical")
    true_tv = has_tv and selected in ("fista", "split_bregman") and (not custom_optimizer or native_name is not None)
    if true_tv and selected == "fista" and parameter_space == "raw":
        raise ValueError("Native field TV FISTA requires physical or auto parameter_space; "
                         "a nonlinear raw transform does not preserve its convex TV proximal problem")
    physical = true_tv or parameter_space == "physical"
    if physical and (selected not in ("fista", "split_bregman") or (custom_optimizer and native_name is None)):
        raise ValueError("Explicit physical coordinates require FISTA or split Bregman")
    parameter_space, regularization_mode = "raw", "none"
    if physical:
        if regularizer_prox is not None or any(np.any(np.asarray(w) != 0) for w in jax.tree.leaves(regularizer_weight)) or box_constraints is not None:
            raise ValueError("TV map fitting uses physical parameter_specs bounds; separate raw-parameter proximal penalties are unsupported")
        if optimizer_base_factory is not None:
            import optax
            if optimizer_base_factory is not optax.sgd:
                raise ValueError("True TV FISTA and split Bregman use objective-aware gradient steps, not an Adam base")
        loss_fn, default_params, unpack = build_loss(
            experiment, model, dataset, parameter_space="physical", balance_parameters=False,
            physical_scales=getattr(initial_state, "parameter_scales", None), **loss_kwargs)
        parameter_space, regularization_mode = "physical", "proximal_tv" if has_tv else "smooth"
    elif has_tv:
        warnings.warn(f"Optimizer '{'custom' if custom_optimizer else selected}' with TV uses smoothed TV gradients; "
                      "choose FISTA or split_bregman for true TV proximal fitting.", RuntimeWarning, stacklevel=2)
        loss_fn, default_params, unpack = build_loss(experiment, model, dataset, smoothed_tv=True, **loss_kwargs)
        regularization_mode = "smoothed_tv"
    state_optimizer = "custom" if custom_optimizer and native_name is None else selected
    constant_param_axes = getattr(unpack, "_mrax_constant_param_axes", {})
    full_spatial_shape = getattr(unpack, "_mrax_full_spatial_shape", None)
    if full_spatial_shape is None:
        full_spatial_shape, _ = _extract_spatial_metadata(experiment)

    def _validate_init(default, candidate):
        if candidate is None:
            return default
        try:
            import jax

            if constant_param_axes:
                candidate = _collapse_params_for_constant_dims(candidate, constant_param_axes)

            tree_default = jax.tree_util.tree_structure(default)
            tree_candidate = jax.tree_util.tree_structure(candidate)
            if tree_default != tree_candidate:
                raise ValueError("initial_params do not match the expected parameter tree")

            def _same_shape(a, b):
                if hasattr(a, "shape") and hasattr(b, "shape") and a.shape != b.shape:
                    raise ValueError("initial_params contain arrays with mismatched shapes")
                return b

            jax.tree_util.tree_map(_same_shape, default, candidate)
        except Exception as exc:
            raise ValueError("Invalid initial_params provided") from exc
        return candidate

    if initial_state is not None:
        if not isinstance(initial_state, FitState):
            raise TypeError("initial_state must be a FitState")
        if initial_params is not None:
            raise ValueError("Provide initial_params or initial_state, not both")
        if initial_state.parameter_space != parameter_space:
            raise ValueError("initial_state uses a different parameter space; use raw initial_params to start a new optimizer")
        if initial_state.optimizer_name is not None and initial_state.optimizer_name != state_optimizer:
            raise ValueError("initial_state belongs to a different optimizer")
        initial_params = initial_state.params
    elif initial_params is not None and physical:
        initial_params = _collapse_params_for_constant_dims(initial_params, constant_param_axes)
        initial_params = unpack._mrax_from_raw(initial_params)
    # A resumed active state is already packed. Reconstruct full coordinates
    # only to build the same objective/metric; validate its layout before use.
    if initial_state is not None and initial_state.active_layout is not None:
        if not fit_active_only:
            raise ValueError("Resuming an active state requires fit_active_only=True")
        params = default_params
    else:
        params = _validate_init(default_params, initial_params)

    if physical and initial_state is None:
        from .physical_parameters import balance_physical_coordinates
        params = balance_physical_coordinates(unpack._mrax_smooth_loss, params, unpack,
            unpack._mrax_tv_operators, balance=balance_physical_parameters and num_steps != 0, metric=physical_metric)
    coordinate_scales = dict(unpack._mrax_physical_scales) if physical else None

    active_layout = None
    if fit_active_only:
        support = getattr(unpack, "_mrax_mask_fields", None)
        if support is None:
            raise ValueError("fit_active_only requires an explicit or finite-data mask")
        if regularizer_prox is not None or box_constraints is not None or any(np.any(np.asarray(w) != 0) for w in jax.tree.leaves(regularizer_weight)):
            raise ValueError("Use model field bounds and regularizers with active packing")
        from .active_parameters import pack_problem
        loss_fn, params, unpack = pack_problem(loss_fn, params, unpack, support)
        active_layout = unpack._mrax_active_layout
        if initial_state is not None:
            if initial_state.active_layout != active_layout:
                raise ValueError("initial_state has a different active-mask layout")
            params = initial_state.params

    step_function = convergence_check = None
    if physical and native_name is not None:
        opt = optimizer.update._mrax_bind_context((loss_fn, unpack._mrax_smooth_loss,
            unpack._mrax_physical_bounds, unpack._mrax_tv_operators))
    elif physical:
        opt = make_optimizer(selected, step_size, device=device,
            proximal_context=(loss_fn, unpack._mrax_smooth_loss,
                unpack._mrax_physical_bounds, unpack._mrax_tv_operators),
            proximal_tolerance=proximal_tolerance, proximal_max_steps=proximal_max_steps,
            bregman_penalty=bregman_penalty, bregman_inner_steps=bregman_inner_steps,
            proximal_max_step_size=proximal_max_step_size, grad_clip_norm=grad_clip_norm,
            fista_restart=fista_restart, fista_backtracking=fista_backtracking)
    elif optimizer is not None:
        opt = optimizer
    elif optimizer_factory is not None:
        opt = optimizer_factory(step_size)
    else:
        opt = make_optimizer(
            selected,
            step_size,
            regularizer_weight=regularizer_weight,
            regularizer_prox=regularizer_prox,
            base_optimizer_factory=optimizer_base_factory,
            box_constraints=box_constraints,
            bregman_penalty=bregman_penalty,
            bregman_inner_steps=bregman_inner_steps,
            proximal_tolerance=proximal_tolerance,
            proximal_max_steps=proximal_max_steps,
            proximal_max_step_size=proximal_max_step_size,
            grad_clip_norm=grad_clip_norm,
            fista_restart=fista_restart,
            fista_backtracking=fista_backtracking,
            device=device,
        )
    try:
        params, losses, fit_info = execute_optimizer(
            loss_fn, params, opt, num_steps=num_steps, chunk_size=chunk_size,
            initial_state=initial_state, convergence_tolerance=convergence_tolerance,
            parameter_tolerance=parameter_tolerance, convergence_patience=convergence_patience,
            min_steps=min_steps, progress_callback=progress_callback,
            step_function=step_function, convergence_check=convergence_check,
            device=device,
        )
    except FitError as error:
        if error.last_valid_state is not None:
            error.last_valid_state = replace(error.last_valid_state, parameter_space=parameter_space,
                                             optimizer_name=state_optimizer, parameter_scales=coordinate_scales,
                                             active_layout=active_layout)
        raise
    fit_info = replace(fit_info, optimizer_name=state_optimizer,
                       parameter_space=parameter_space, regularization_mode=regularization_mode,
                       state=replace(fit_info.state, parameter_space=parameter_space,
                                     optimizer_name=state_optimizer, parameter_scales=coordinate_scales,
                                     active_layout=active_layout))

    fitted_fields_raw = unpack(params)
    fitted_fields_raw = _broadcast_fields_to_field_shape(
        fitted_fields_raw,
        getattr(unpack, "_mrax_constant_field_axes", {}),
        full_spatial_shape,
    )
    fitted_fields_raw = _mask_fields_for_output(
        fitted_fields_raw,
        getattr(unpack, "_mrax_mask_fields", None),
        fill_value=mask_fill_value,
    )
    try:
        fitted_fields = attach_units(fitted_fields_raw, model.parameter_units)
    except Exception as exc:
        raise KeyError(
            f"Parameter units missing for model '{model.name}'. "
            "Every parameter must declare an SI unit."
        ) from exc

    def _auto_derive_maps(model: Model) -> bool:
        if not hasattr(model, "derived_maps"):
            return False
        name = (model.name or "").lower()
        if name in ("dti", "dki", "dti_tensor", "diffusion"):
            return True
        required = tuple(getattr(model, "required_observables", ()) or ())
        return ("bvals" in required) or ("bvecs" in required)

    do_derived = _auto_derive_maps(model) if compute_derived_maps is None else compute_derived_maps
    if do_derived and hasattr(model, "derived_maps"):
        observables = experiment.merged_observables(study_globals)
        try:
            derived = model.derived_maps(fitted_fields, observables)  # type: ignore[attr-defined]
        except TypeError:
            derived = model.derived_maps(fitted_fields)  # type: ignore[attr-defined]
        if derived:
            derived_units = getattr(model, "derived_map_units", {}) or {}
            for key, value in derived.items():
                if key in fitted_fields:
                    continue
                if isinstance(value, Quantity):
                    fitted_fields[key] = value
                else:
                    unit = derived_units.get(key)
                    fitted_fields[key] = Quantity(value, unit) if unit else value

    if return_model_callable:
        model_fn = bind_device(make_model_callable(experiment, model, study_globals=study_globals,
                                                  reference_image=reference_image), device)
        if return_fit_info:
            return fitted_fields, losses, model_fn, fit_info
        return fitted_fields, losses, model_fn
    if return_fit_info:
        return fitted_fields, losses, fit_info
    return fitted_fields, losses


def build_study_losses(study: Study, device: DeviceLike = "cpu"):
    """Build per-experiment losses for a collection bundled as a Study."""

    device = resolve_device(device)
    losses = []
    for name, experiment, model, dataset in study.iter_entries():
        loss_fn, params, unpack = build_loss(
            experiment,
            model,
            dataset,
            device=device,
            study_globals=getattr(study, "study_globals", None),
        )
        losses.append(
            {
                "name": name,
                "loss_fn": loss_fn,
                "init_params": params,
                "unpack": unpack,
                "model": model,
                "dataset": dataset,
            }
        )
    return losses
