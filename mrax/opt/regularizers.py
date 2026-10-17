"""Penalty functions for common imaging regularizers."""

from __future__ import annotations

from typing import Mapping

import jax.numpy as jnp
import numpy as np


def _normalize_axes(field: jnp.ndarray, axes: tuple[int, ...] | None) -> tuple[int, ...]:
    if axes is None:
        return tuple(range(field.ndim))
    normalized: list[int] = []
    for axis in axes:
        axis = int(axis)
        if axis < 0:
            axis += field.ndim
        if axis < 0 or axis >= field.ndim:
            raise ValueError(f"axis {axis} is out of bounds for field with {field.ndim} dimensions")
        if axis not in normalized:
            normalized.append(axis)
    return tuple(normalized)


def _euclidean_norm_zero_at_origin(values: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
    """Euclidean norm with a zero value for zero vectors and finite gradients."""
    squared = jnp.sum(jnp.abs(values) ** 2, axis=axis)
    eps = jnp.asarray(1e-12, dtype=squared.dtype)
    return jnp.sqrt(squared + eps) - jnp.sqrt(eps)


def _weighted_mean(values: jnp.ndarray, weight: jnp.ndarray | None) -> jnp.ndarray:
    """Mean with optional non-negative weights and a safe empty-mask denominator."""
    if weight is None:
        return jnp.mean(values)
    weight = jnp.asarray(weight, dtype=values.dtype)
    return jnp.sum(weight * values) / (jnp.sum(weight) + 1e-8)


def _forward_diff_with_pad(arr: jnp.ndarray, axis: int) -> jnp.ndarray:
    diff = jnp.diff(arr, axis=axis)
    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (0, 1)
    return jnp.pad(diff, pad_width)


def _pad_weight_for_grad(weight: jnp.ndarray, axis: int) -> jnp.ndarray:
    """Pad a (N-1) diff mask so it matches the forward-diff-with-pad shape."""
    pad_width = [(0, 0)] * weight.ndim
    pad_width[axis] = (0, 1)
    return jnp.pad(weight, pad_width)


def spatial_smoothness_penalty(
    field: jnp.ndarray, axes: tuple[int, ...] | None = None, mask: jnp.ndarray | None = None
) -> jnp.ndarray:
    """Encourage smooth parameter maps via finite-difference gradients."""
    axes = _normalize_axes(field, axes)
    axes = tuple(ax for ax in axes if field.shape[ax] > 1)
    if not axes:
        return jnp.array(0.0, dtype=field.dtype)

    weight_mask = None
    if mask is not None:
        weight_mask = jnp.asarray(mask)
        weight_mask = jnp.broadcast_to(weight_mask, field.shape)

    penalty = 0.0
    for axis in axes:
        diff = jnp.diff(field, axis=axis)
        if weight_mask is not None:
            slicer0 = [slice(None)] * weight_mask.ndim
            slicer0[axis] = slice(0, -1)
            slicer1 = [slice(None)] * weight_mask.ndim
            slicer1[axis] = slice(1, None)
            w = weight_mask[tuple(slicer0)] * weight_mask[tuple(slicer1)]
            denom = jnp.sum(w) + 1e-8
            penalty = penalty + jnp.sum(w * diff**2) / denom
        else:
            penalty = penalty + jnp.mean(diff**2)
    return penalty / max(len(axes), 1)


def total_variation_penalty(
    field: jnp.ndarray, axes: tuple[int, ...] | None = None, mask: jnp.ndarray | None = None,
    *, smoothed: bool = False,
) -> jnp.ndarray:
    """Masked isotropic TV; smoothing is an explicit gradient-only option."""
    from .tv import TVOperator
    return TVOperator.create(field.shape, axes, mask=mask, dtype=jnp.real(field).dtype).value(field, smoothed)


def higher_order_total_variation_penalty(
    field: jnp.ndarray,
    order: int = 2,
    axes: tuple[int, ...] | None = None,
    mask: jnp.ndarray | None = None,
    *, smoothed: bool = False,
) -> jnp.ndarray:
    """Masked higher-order TV with the complete finite-difference stencil."""
    from .tv import TVOperator
    return TVOperator.create(field.shape, axes, order, mask, dtype=jnp.real(field).dtype).value(field, smoothed)


def make_regularizer_fn(reg_type: str, axes: tuple[int, ...] | None = None, *, smoothed_tv=False):
    reg_type = reg_type.lower()
    if reg_type.startswith("tv"):
        suffix = reg_type[2:]
        order = 1
        if suffix:
            if not suffix.isdigit():
                raise ValueError(f"Unknown regularizer type '{reg_type}'")
            order = int(suffix)
        def tv(field, mask=None):
            return higher_order_total_variation_penalty(field, order, axes, mask, smoothed=smoothed_tv)
        tv.tv_axes = axes
        tv.tv_order = order
        return tv
    if reg_type == "smoothness":
        return lambda field, mask=None: spatial_smoothness_penalty(field, axes=axes, mask=mask)
    if reg_type == "none":
        return None
    raise ValueError(f"Unknown regularizer type '{reg_type}'")


def _mask_for_scale(arr: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    if mask is None:
        return arr
    mask_arr = np.asarray(mask)
    if mask_arr.shape == arr.shape:
        return np.where(mask_arr > 0, arr, np.nan)
    if arr.ndim >= 2 and mask_arr.ndim == 2 and arr.shape[-2:] == mask_arr.shape:
        shape = (1,) * (arr.ndim - 2) + mask_arr.shape
        return np.where(mask_arr.reshape(shape) > 0, arr, np.nan)
    return arr


def estimate_field_scales(
    fields: Mapping[str, object],
    *,
    mask: np.ndarray | None = None,
    method: str = "median",
    min_scale: float = 1e-6,
) -> dict[str, float]:
    """Estimate per-field scales for regularizer normalization."""
    scales: dict[str, float] = {}
    for key, value in fields.items():
        arr = value.values if hasattr(value, "values") else value
        arr = np.asarray(arr)
        arr = _mask_for_scale(arr, mask)
        abs_arr = np.abs(arr).astype(np.float64)
        if method == "mean":
            scale = float(np.nanmean(abs_arr))
        elif method == "median":
            scale = float(np.nanmedian(abs_arr))
        else:
            raise ValueError(f"Unknown scale method '{method}'")
        if not np.isfinite(scale) or scale <= min_scale:
            scale = 1.0
        scales[key] = scale
    return scales


def estimate_param_scales(
    model,
    field_shape: tuple[int, ...],
    *,
    mask: np.ndarray | None = None,
    method: str = "median",
    min_scale: float = 1e-6,
    param_specs: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, float]:
    """Estimate parameter scales from a model's initialization or param specs."""
    if method == "bounds":
        if not param_specs:
            raise ValueError("param_specs is required when method='bounds'")
        scales: dict[str, float] = {}
        for key, spec in param_specs.items():
            bounds = spec.get("bounds") if isinstance(spec, Mapping) else None
            if bounds is None:
                scale = 1.0
            else:
                try:
                    vals = np.asarray(bounds, dtype=np.float64).reshape(-1)
                    scale = float(np.max(np.abs(vals))) if vals.size == 2 else 1.0
                except Exception:
                    scale = 1.0
            if not np.isfinite(scale) or scale <= min_scale:
                scale = 1.0
            scales[key] = scale
        return scales
    params, unpack = model.parametrization_factory(field_shape)
    fields = unpack(params)
    return estimate_field_scales(fields, mask=mask, method=method, min_scale=min_scale)


def _regularizer_exponent(reg_type: str, type_exponents: Mapping[str, float]) -> float:
    reg_type = reg_type.lower()
    if reg_type.startswith("tv"):
        return type_exponents.get("tv", 1.0)
    if reg_type in type_exponents:
        return type_exponents[reg_type]
    if reg_type in ("smoothness", "l2"):
        return type_exponents.get("smoothness", 2.0)
    return type_exponents.get("default", 1.0)


def autoscale_regularizer_configs(
    regularizer_configs: Mapping[str, list[dict]] | None,
    param_scales: Mapping[str, float],
    *,
    type_exponents: Mapping[str, float] | None = None,
    min_scale: float = 1e-6,
) -> dict[str, list[dict]] | None:
    """Scale regularizer weights by parameter magnitude."""
    if not regularizer_configs:
        return None
    exponents = {
        "smoothness": 2.0,
        "l2": 2.0,
        "tv": 1.0,
        "default": 1.0,
    }
    if type_exponents:
        exponents.update({k.lower(): float(v) for k, v in type_exponents.items()})
    scaled: dict[str, list[dict]] = {}
    for key, configs in regularizer_configs.items():
        scale = float(param_scales.get(key, 1.0))
        scale = scale if scale > min_scale else 1.0
        updated = []
        for cfg in configs:
            cfg_local = dict(cfg)
            weight = float(cfg_local.get("weight", 0.0))
            if weight != 0.0:
                reg_type = str(cfg_local.get("type", "smoothness"))
                exponent = _regularizer_exponent(reg_type, exponents)
                cfg_local["weight"] = weight / (scale**exponent)
            updated.append(cfg_local)
        scaled[key] = updated
    return scaled


__all__ = [
    "autoscale_regularizer_configs",
    "estimate_field_scales",
    "estimate_param_scales",
    "higher_order_total_variation_penalty",
    "make_regularizer_fn",
    "spatial_smoothness_penalty",
    "total_variation_penalty",
]
