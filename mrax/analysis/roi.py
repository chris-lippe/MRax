from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import jax.numpy as jnp
import numpy as np

from mrax.data_struct.quantity import Quantity

_LOG = logging.getLogger(__name__)

ArrayLike = Any


STAT_ALIASES = {
    "var": "variance",
    "stddev": "std",
    "sd": "std",
    "n": "count",
}


@dataclass(frozen=True)
class ROI:
    """Region-of-interest definition with an optional dimension label list."""

    name: str
    mask: np.ndarray
    dims: tuple[str, ...] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        mask = np.asarray(self.mask)
        if mask.ndim == 0:
            raise ValueError("ROI mask must be an array with at least 1 dimension")
        if self.dims is not None and len(self.dims) != mask.ndim:
            raise ValueError("ROI dims must match mask dimensions")
        object.__setattr__(self, "mask", mask)


def _as_array(value: ArrayLike) -> tuple[np.ndarray, str | None]:
    if isinstance(value, Quantity):
        arr = value.values
        unit = value.si_unit
    else:
        arr = value
        unit = None
    xp = _array_module(arr)
    return xp.asarray(arr), unit


def _array_module(*values: object):
    for value in values:
        if isinstance(value, jnp.ndarray):
            return jnp
    return np


def _as_mask(mask: ArrayLike) -> np.ndarray:
    xp = _array_module(mask)
    arr = xp.asarray(mask)
    if arr.dtype == bool:
        return arr
    return arr != 0


def match_spatial_mask(
    mask: ArrayLike | None,
    spatial_shape: Sequence[int],
    *,
    allow_transpose: bool = True,
) -> np.ndarray | None:
    """Validate or transpose a 2D/3D mask to match a spatial shape."""

    if mask is None:
        return None
    arr = np.asarray(mask)
    target = tuple(int(v) for v in spatial_shape)
    if tuple(arr.shape) == target:
        return arr
    if allow_transpose and tuple(arr.shape) == target[::-1]:
        return arr.T
    raise ValueError(f"Mask shape {arr.shape} does not match spatial shape {target}")


def normalize_stats(stats: Iterable[str]) -> list[str]:
    normalized: list[str] = []
    for stat in stats:
        key = STAT_ALIASES.get(stat, stat)
        normalized.append(key)
    return normalized


def broadcast_mask(
    mask: ArrayLike,
    target_shape: Sequence[int],
    *,
    mask_dims: Sequence[str] | None = None,
    target_dims: Sequence[str] | None = None,
) -> np.ndarray:
    """Broadcast an ROI mask to a target array shape."""

    mask_arr = _as_mask(mask)
    target_shape = tuple(int(v) for v in target_shape)
    xp = _array_module(mask_arr)

    if mask_arr.shape == target_shape:
        return mask_arr

    if mask_dims is not None and target_dims is not None:
        if len(mask_dims) != mask_arr.ndim:
            raise ValueError("mask_dims must match mask.ndim")
        shape: list[int] = []
        for dim in target_dims:
            if dim in mask_dims:
                shape.append(int(mask_arr.shape[mask_dims.index(dim)]))
            else:
                shape.append(1)
        try:
            reshaped = mask_arr.reshape(tuple(shape))
            return xp.broadcast_to(reshaped, target_shape).astype(bool)
        except ValueError as exc:
            raise ValueError(
                f"Mask with dims {mask_dims} cannot be aligned to target dims {target_dims}"
            ) from exc

    if mask_arr.ndim <= len(target_shape):
        lead = len(target_shape) - mask_arr.ndim
        try:
            reshaped = mask_arr.reshape((1,) * lead + mask_arr.shape)
            return xp.broadcast_to(reshaped, target_shape).astype(bool)
        except ValueError:
            _LOG.debug("Ignored optional operation failure", exc_info=True)

    try:
        return xp.broadcast_to(mask_arr, target_shape).astype(bool)
    except ValueError as exc:
        raise ValueError(
            f"Mask shape {mask_arr.shape} cannot be broadcast to target shape {target_shape}"
        ) from exc


def extract_roi_values(
    array: ArrayLike,
    roi: ROI | ArrayLike,
    *,
    mask_dims: Sequence[str] | None = None,
    target_dims: Sequence[str] | None = None,
    drop_nan: bool = True,
) -> np.ndarray:
    """Return flattened ROI values from an array."""

    arr, _ = _as_array(array)
    roi_mask = roi.mask if isinstance(roi, ROI) else roi
    roi_dims = roi.dims if isinstance(roi, ROI) else mask_dims
    xp = _array_module(arr, roi_mask)
    arr = xp.asarray(arr)
    roi_mask = xp.asarray(roi_mask) if xp is jnp else roi_mask
    mask = broadcast_mask(roi_mask, arr.shape, mask_dims=roi_dims, target_dims=target_dims)
    mask = xp.asarray(mask)
    values = arr[mask]
    if drop_nan:
        values = values[xp.isfinite(values)]
    return values


def summarize_values(values: ArrayLike, *, stats: Iterable[str], ddof: int = 1) -> dict[str, float]:
    """Summarize 1D values with common descriptive statistics."""

    requested = normalize_stats(stats)
    xp = _array_module(values)
    arr = xp.asarray(values, dtype=float).ravel()
    finite = arr[xp.isfinite(arr)]
    nan_count = int(arr.size - finite.size)
    n = int(finite.size)

    if n == 0:
        return {
            key: float("nan")
            for key in requested
            if key not in ("count", "nan_count")
        } | {"count": 0.0, "nan_count": float(nan_count)}

    mean = float(xp.mean(finite))
    var = float(xp.var(finite, ddof=ddof)) if n > ddof else float("nan")
    std = float(xp.sqrt(var)) if np.isfinite(var) else float("nan")
    median = float(xp.median(finite))
    min_val = float(xp.min(finite))
    max_val = float(xp.max(finite))
    p25 = float(xp.percentile(finite, 25))
    p75 = float(xp.percentile(finite, 75))

    catalog = {
        "mean": mean,
        "variance": var,
        "std": std,
        "median": median,
        "min": min_val,
        "max": max_val,
        "count": float(n),
        "nan_count": float(nan_count),
        "p25": p25,
        "p75": p75,
    }
    return {key: catalog[key] for key in requested if key in catalog}


def roi_stats(
    array: ArrayLike,
    roi: ROI,
    *,
    stats: Iterable[str] = ("mean", "variance"),
    ddof: int = 1,
    target_dims: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Compute summary statistics for an ROI within a map."""

    arr, unit = _as_array(array)
    values = extract_roi_values(arr, roi, mask_dims=roi.dims, target_dims=target_dims)
    summary = summarize_values(values, stats=stats, ddof=ddof)
    if unit:
        summary["unit"] = unit
    return summary


def collect_roi_stats(
    maps: Mapping[str, ArrayLike],
    rois: Sequence[ROI],
    *,
    stats: Iterable[str] = ("mean", "variance"),
    ddof: int = 1,
    target_dims: Sequence[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Collect ROI statistics across multiple parameter maps.

    Returns a long-form list of rows with keys: map, roi, stat, value, unit (optional).
    """

    rows: list[dict[str, Any]] = []
    metadata = dict(metadata) if metadata else {}
    normalized = normalize_stats(stats)

    for map_name, map_values in maps.items():
        for roi in rois:
            summary = roi_stats(
                map_values,
                roi,
                stats=normalized,
                ddof=ddof,
                target_dims=target_dims,
            )
            for stat_name in normalized:
                if stat_name not in summary:
                    continue
                row = {
                    "map": map_name,
                    "roi": roi.name,
                    "stat": stat_name,
                    "value": summary[stat_name],
                }
                if "unit" in summary:
                    row["unit"] = summary["unit"]
                row.update(metadata)
                rows.append(row)
    return rows


__all__ = [
    "ROI",
    "broadcast_mask",
    "collect_roi_stats",
    "extract_roi_values",
    "normalize_stats",
    "roi_stats",
    "summarize_values",
]
