from __future__ import annotations

from typing import Any, Sequence

import jax
import jax.numpy as jnp
import jax.scipy.signal as jsignal
import numpy as np
from scipy.interpolate import interp1d

ArrayLike = Any


def _array_module(*values: object):
    for value in values:
        if isinstance(value, jnp.ndarray):
            return jnp
    return np


def _normalize_tuple(
    value: float | Sequence[float], ndim: int, *, name: str, cast: type = float
) -> tuple[float, ...]:
    if np.isscalar(value):
        return (cast(value),) * ndim
    vals = tuple(cast(v) for v in value)
    if len(vals) != ndim:
        raise ValueError(f"{name} must have length {ndim}")
    return vals


def _gaussian_kernel1d(xp, sigma: float, truncate: float):
    sigma = float(max(sigma, 1e-8))
    radius = int(truncate * sigma + 0.5)
    x = xp.arange(-radius, radius + 1)
    kernel = xp.exp(-(x**2) / (2.0 * sigma**2))
    return kernel / xp.sum(kernel)


def _convolve1d(xp, arr, kernel, axis: int, *, mode: str):
    pad = kernel.shape[0] // 2
    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (pad, pad)
    padded = xp.pad(arr, pad_width, mode=mode)
    if xp is jnp:
        kernel_shape = [1] * arr.ndim
        kernel_shape[axis] = kernel.shape[0]
        kernel_nd = kernel.reshape(kernel_shape)
        return jsignal.convolve(padded, kernel_nd, mode="valid")
    def _conv_1d(vec):
        return np.convolve(vec, np.asarray(kernel), mode="valid")
    return np.apply_along_axis(_conv_1d, axis, np.asarray(padded))


def _gaussian_filter(xp, array, sigma, *, mode: str, truncate: float):
    arr = xp.asarray(array)
    sigmas = _normalize_tuple(sigma, arr.ndim, name="sigma", cast=float)
    out = arr
    for axis, sig in enumerate(sigmas):
        if sig <= 0:
            continue
        kernel = _gaussian_kernel1d(xp, sig, truncate)
        out = _convolve1d(xp, out, kernel, axis, mode=mode)
    return out


def _gaussian_filter1d(xp, array, sigma: float, *, axis: int, mode: str, truncate: float):
    arr = xp.asarray(array)
    if sigma <= 0:
        return arr
    kernel = _gaussian_kernel1d(xp, float(sigma), truncate)
    return _convolve1d(xp, arr, kernel, axis, mode=mode)


def _interp_linear_sorted(times, values, new_times, *, fill_value):
    # times: (T,), values: (T, N) or (T,), new_times: (M,)
    xp = jnp if isinstance(times, jnp.ndarray) else np
    times = xp.asarray(times)
    new_times = xp.asarray(new_times)
    values = xp.asarray(values)
    if times.ndim != 1:
        raise ValueError("times must be 1D")
    if values.shape[0] != times.shape[0]:
        raise ValueError("values length must match times")
    if times.shape[0] == 1:
        return xp.broadcast_to(values[0], (new_times.shape[0],) + values.shape[1:])

    idx = xp.searchsorted(times, new_times, side="left")
    idx0 = xp.clip(idx - 1, 0, times.shape[0] - 1)
    idx1 = xp.clip(idx, 0, times.shape[0] - 1)

    t0 = times[idx0]
    t1 = times[idx1]
    v0 = values[idx0]
    v1 = values[idx1]
    denom = xp.where(t1 == t0, 1.0, t1 - t0)
    frac = (new_times - t0) / denom
    interp = v0 + (v1 - v0) * frac.reshape(frac.shape + (1,) * (values.ndim - 1))

    if isinstance(fill_value, str):
        if str(fill_value).lower() == "extrapolate":
            return interp
        raise ValueError("fill_value must be a float or 'extrapolate' for JAX linear interpolation")

    left_mask = new_times < times[0]
    right_mask = new_times > times[-1]
    if xp is jnp:
        interp = xp.where(left_mask, fill_value, interp)
        interp = xp.where(right_mask, fill_value, interp)
    else:
        interp = np.where(left_mask, fill_value, interp)
        interp = np.where(right_mask, fill_value, interp)
    return interp


def nan_gaussian_filter(array: ArrayLike, sigma: float | Sequence[float], *, mode: str = "reflect") -> np.ndarray:
    """Gaussian filter that preserves NaNs by normalizing with a validity mask."""

    xp = _array_module(array)
    arr = xp.asarray(array, dtype=float)
    if not xp.isnan(arr).any():
        return _gaussian_filter(xp, arr, sigma, mode=mode, truncate=4.0)

    mask = xp.isfinite(arr).astype(arr.dtype)
    filled = xp.where(xp.isfinite(arr), arr, 0.0)
    filtered = _gaussian_filter(xp, filled, sigma, mode=mode, truncate=4.0)
    weights = _gaussian_filter(xp, mask, sigma, mode=mode, truncate=4.0)
    out = filtered / weights
    out = xp.where(weights == 0, xp.nan, out)
    return out


def interpolate_timecourses(
    image_series: ArrayLike,
    times: ArrayLike,
    new_times: ArrayLike,
    *,
    kind: str = "linear",
    fill_value: float | str = np.nan,
    axis: int = 0,
    strict_nan: bool = False,
) -> np.ndarray:
    """Interpolate a time-first stack to a new time grid."""

    xp = _array_module(image_series, times, new_times)
    series = xp.asarray(image_series)
    times_arr = xp.asarray(times, dtype=float)
    new_times_arr = xp.asarray(new_times, dtype=float)

    if series.shape[axis] != times_arr.shape[0]:
        raise ValueError("times must match the length of the time axis")

    order = xp.argsort(times_arr)
    sorted_times = times_arr[order]
    sorted_series = xp.take(series, order, axis=axis)

    if kind != "linear":
        if xp is np:
            interp = interp1d(
                np.asarray(sorted_times),
                np.asarray(sorted_series),
                axis=axis,
                kind=kind,
                bounds_error=False,
                fill_value=fill_value,
            )
            out = interp(np.asarray(new_times_arr))
            if strict_nan:
                invalid = ~np.isfinite(np.asarray(series))
                mask = np.any(invalid, axis=axis, keepdims=True)
                if np.any(mask):
                    out = np.where(mask, np.nan, out)
            return out
        raise ValueError("Only linear interpolation is supported for JAX inputs.")

    series_moved = xp.moveaxis(sorted_series, axis, 0)
    flat = series_moved.reshape((series_moved.shape[0], -1))
    if xp is jnp:
        interp_flat = jax.vmap(
            lambda y: _interp_linear_sorted(sorted_times, y, new_times_arr, fill_value=fill_value),
            in_axes=1,
            out_axes=1,
        )(flat)
    else:
        interp_flat = np.stack(
            [
                _interp_linear_sorted(sorted_times, flat[:, idx], new_times_arr, fill_value=fill_value)
                for idx in range(flat.shape[1])
            ],
            axis=1,
        )
    out = interp_flat.reshape((new_times_arr.shape[0],) + series_moved.shape[1:])
    out = xp.moveaxis(out, 0, axis)

    if strict_nan:
        invalid = ~xp.isfinite(series)
        mask = xp.any(invalid, axis=axis, keepdims=True)
        if xp.any(mask):
            out = xp.where(mask, xp.nan, out)
    return out


def apply_baseline(
    image_series: ArrayLike,
    *,
    baseline_frames: int = 50,
    mode: str = "subtract",
    axis: int = 0,
) -> np.ndarray:
    """Apply baseline correction to a time-first stack."""

    xp = _array_module(image_series)
    arr = xp.asarray(image_series, dtype=float)
    if baseline_frames <= 0 or arr.shape[axis] == 0:
        return arr

    baseline_slice = [slice(None)] * arr.ndim
    baseline_slice[axis] = slice(0, min(baseline_frames, arr.shape[axis]))
    baseline = xp.nanmean(arr[tuple(baseline_slice)], axis=axis, keepdims=True)

    if mode == "divide":
        corrected = arr / baseline
        corrected = xp.where(xp.isfinite(corrected), corrected, xp.nan)
        return corrected
    if mode == "subtract":
        return arr - baseline
    raise ValueError("mode must be 'subtract' or 'divide'")


def apply_dilution_correction(
    image_series: ArrayLike,
    dilution: ArrayLike,
    *,
    baseline_frames: int = 50,
    clip_min: float = 1e-2,
    smooth_sigma: float = 4.0,
) -> np.ndarray:
    """Apply dilution correction using a baseline-normalized dilution map."""

    xp = _array_module(image_series, dilution)
    series = xp.asarray(image_series, dtype=float)
    dil = xp.asarray(dilution, dtype=float)
    dil = apply_baseline(dil, baseline_frames=baseline_frames, mode="divide", axis=0)
    dil = xp.clip(dil, clip_min, None)
    if smooth_sigma and smooth_sigma > 0:
        dil = _gaussian_filter1d(xp, dil, sigma=smooth_sigma, axis=0, mode="reflect", truncate=4.0)
    corrected = series / dil
    corrected = xp.where(xp.isfinite(corrected), corrected, xp.nan)
    return corrected


def mean_timecourses_by_label(
    image_series: ArrayLike,
    label_mask: ArrayLike,
    *,
    labels: Sequence[int] | None = None,
    baseline_frames: int = 50,
    baseline_method: str = "mean",
    baseline_times: ArrayLike | None = None,
    baseline_reference_time: float | None = None,
    baseline_tail_frames: int = 3,
    smooth_sigma: float = 2.0,
    dilution: ArrayLike | None = None,
    dilation_baseline_frames: int = 50,
) -> tuple[np.ndarray, list[int]]:
    """Compute mean ROI timecourses for each label in the mask."""

    xp = _array_module(image_series, label_mask)
    series = xp.asarray(image_series, dtype=float)
    if smooth_sigma and smooth_sigma > 0:
        series = _gaussian_filter1d(xp, series, sigma=smooth_sigma, axis=0, mode="reflect", truncate=4.0)

    if dilution is not None:
        series = apply_dilution_correction(
            series,
            dilution,
            baseline_frames=dilation_baseline_frames,
            smooth_sigma=max(0.0, smooth_sigma * 2.0),
        )

    mask = xp.asarray(label_mask)
    if mask.ndim != series.ndim - 1:
        raise ValueError("label_mask must match the spatial dimensions of image_series")

    if labels is None:
        labels = [int(v) for v in xp.unique(mask) if int(v) != 0]
    labels = list(labels)

    timecourses = np.zeros((series.shape[0], len(labels)), dtype=float)
    for idx, label in enumerate(labels):
        roi = mask == label
        mean_tc = xp.nanmean(series[:, roi], axis=1)
        if baseline_frames > 0:
            baseline = estimate_timecourse_baseline(
                mean_tc,
                baseline_frames=baseline_frames,
                method=baseline_method,
                times=baseline_times,
                reference_time=baseline_reference_time,
                tail_frames=baseline_tail_frames,
            )
            mean_tc = mean_tc - baseline
        timecourses[:, idx] = np.asarray(mean_tc)

    return timecourses, labels


def estimate_timecourse_baseline(
    values: ArrayLike,
    *,
    baseline_frames: int,
    method: str = "mean",
    times: ArrayLike | None = None,
    reference_time: float | None = None,
    tail_frames: int = 3,
) -> float:
    """Estimate a scalar baseline for one ROI timecourse."""

    data = np.asarray(values, dtype=float)
    if data.ndim != 1:
        raise ValueError("values must be a 1D timecourse")

    frame_count = min(max(int(baseline_frames), 0), data.shape[0])
    if frame_count == 0:
        return 0.0

    baseline_values = data[:frame_count]
    method_key = str(method).strip().lower()
    if method_key == "mean":
        return float(np.nanmean(baseline_values))
    if method_key == "median":
        return float(np.nanmedian(baseline_values))
    if method_key != "tail_theil_sen":
        raise ValueError(f"Unsupported baseline method: {method!r}")
    if times is None:
        raise ValueError("times are required for tail_theil_sen baseline estimation")

    baseline_times = np.asarray(times, dtype=float)
    if baseline_times.ndim != 1 or baseline_times.shape[0] < frame_count:
        raise ValueError("times must be 1D and cover all baseline frames")

    tail_count = min(max(int(tail_frames), 1), frame_count)
    y = baseline_values[-tail_count:]
    x = baseline_times[:frame_count][-tail_count:]
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if y.size == 0:
        return float("nan")
    if y.size == 1:
        return float(y[0])

    slopes = [
        (y[j] - y[i]) / (x[j] - x[i])
        for i in range(y.size)
        for j in range(i + 1, y.size)
        if x[j] != x[i]
    ]
    if not slopes:
        return float(np.nanmedian(y))

    slope = float(np.nanmedian(slopes))
    intercept = float(np.nanmedian(y - slope * x))
    target_time = float(x[-1] if reference_time is None else reference_time)
    return intercept + slope * target_time


def stack_timecourses_by_group(
    timecourses: Sequence[np.ndarray],
    *,
    min_length: int | None = None,
) -> np.ndarray:
    """Stack per-subject timecourses into a (subjects, time, labels) array."""

    if not timecourses:
        return np.empty((0, 0, 0))
    lengths = [tc.shape[0] for tc in timecourses]
    target = min(lengths) if min_length is None else min(min_length, min(lengths))
    stacked = np.stack([tc[:target] for tc in timecourses], axis=0)
    return stacked


__all__ = [
    "apply_baseline",
    "apply_dilution_correction",
    "estimate_timecourse_baseline",
    "interpolate_timecourses",
    "mean_timecourses_by_label",
    "nan_gaussian_filter",
    "stack_timecourses_by_group",
]
