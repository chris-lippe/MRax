from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from mrax.analysis.roi import match_spatial_mask
from mrax.analysis.timecourse_processing import mean_timecourses_by_label, nan_gaussian_filter
from mrax.data_struct.quantity import Quantity

DEFAULT_GADOLINIUM_RELAXIVITY_MMM_MS = 4.1 / 1000.0
DEFAULT_DCE_TR_MS = 12.5
DEFAULT_DCE_FLIP_ANGLE_DEG = 15.0


@dataclass
class DCEAnalysisResult:
    """Processed DCE concentration series and optional ROI summaries."""

    concentration_mM: np.ndarray
    signal_ratio: np.ndarray
    reference_image: np.ndarray
    times_seconds: np.ndarray
    frame_time_seconds: float
    baseline_frames: int
    timecourses_mM: np.ndarray | None = None
    labels: list[int] | None = None


def _as_array(value: object) -> np.ndarray:
    if isinstance(value, Quantity):
        return np.asarray(value.values)
    return np.asarray(value)


def _first_finite_scalar(value: object | None) -> float | None:
    if value is None:
        return None
    arr = _as_array(value).astype(float, copy=False).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(arr[0])


def infer_dce_times_seconds(
    frame_count: int,
    *,
    total_scan_time_ms: float | None = None,
    frame_time_ms: float | None = None,
    centered: bool = True,
) -> tuple[np.ndarray, float]:
    """Infer DCE frame times in seconds from total scan time or frame spacing."""

    if frame_count <= 0:
        raise ValueError("frame_count must be positive")

    dt_seconds: float | None = None
    if total_scan_time_ms is not None and np.isfinite(total_scan_time_ms) and total_scan_time_ms > 0:
        dt_seconds = float(total_scan_time_ms) * 1e-3 / float(frame_count)
    elif frame_time_ms is not None and np.isfinite(frame_time_ms) and frame_time_ms > 0:
        dt_seconds = float(frame_time_ms) * 1e-3
    else:
        dt_seconds = 1.0

    if centered:
        times = (np.arange(frame_count, dtype=float) + 0.5) * dt_seconds
    else:
        times = np.arange(frame_count, dtype=float) * dt_seconds
    return times, float(dt_seconds)


def replace_isolated_spikes(
    image_series: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    threshold: float = 1.05,
) -> np.ndarray:
    """Replace isolated high-intensity frames using neighboring frames."""

    series = np.asarray(image_series, dtype=float)
    if series.ndim < 2 or series.shape[0] < 3 or threshold <= 1.0:
        return series.copy()

    mask_arr = None
    if mask is not None:
        mask_arr = np.asarray(mask, dtype=bool)
        if mask_arr.shape != series.shape[1:]:
            mask_arr = match_spatial_mask(mask_arr, series.shape[1:])

    reduced = series.reshape((series.shape[0], -1))
    if mask_arr is not None:
        reduced = series[:, mask_arr]
    frame_means = np.nanmean(reduced, axis=1)

    out = series.copy()
    for idx in range(1, series.shape[0] - 1):
        cur = float(frame_means[idx])
        prev = float(frame_means[idx - 1])
        nxt = float(frame_means[idx + 1])
        if not np.isfinite(cur):
            continue
        high_prev = np.isfinite(prev) and cur > prev * threshold
        high_next = np.isfinite(nxt) and cur > nxt * threshold
        if high_prev and high_next:
            out[idx] = 0.5 * (series[idx - 1] + series[idx + 1])
        elif high_prev:
            out[idx] = series[idx + 1]
        elif high_next:
            out[idx] = series[idx - 1]
    return out


def spgr_signal_ratio_to_concentration(
    signal_ratio: np.ndarray,
    t1_map_ms: np.ndarray,
    *,
    repetition_time_ms: float = DEFAULT_DCE_TR_MS,
    flip_angle_deg: float = DEFAULT_DCE_FLIP_ANGLE_DEG,
    relaxivity_mM_ms: float = DEFAULT_GADOLINIUM_RELAXIVITY_MMM_MS,
    eps: float = 1e-8,
) -> np.ndarray:
    """Convert baseline-normalized SPGR signal ratios to CA concentration in mM."""

    ratio = np.asarray(signal_ratio, dtype=float)
    t1_ms = np.asarray(t1_map_ms, dtype=float)
    if ratio.ndim != t1_ms.ndim + 1:
        raise ValueError("signal_ratio must be time-first and t1_map_ms must match its spatial shape")

    theta = np.deg2rad(float(flip_angle_deg))
    tr_ms = float(repetition_time_ms)
    r1 = float(relaxivity_mM_ms)

    valid_t1 = np.isfinite(t1_ms) & (t1_ms > eps)
    t1_safe = np.where(valid_t1, t1_ms, np.nan)

    e10 = np.exp(-tr_ms / t1_safe)
    s_star = (1.0 - e10) / (1.0 - np.cos(theta) * e10)

    numer = 1.0 - np.cos(theta) * s_star[None, ...] * ratio
    denom = 1.0 - s_star[None, ...] * ratio
    numer = np.where(np.isfinite(numer), np.clip(numer, eps, None), np.nan)
    denom = np.where(np.isfinite(denom), np.clip(denom, eps, None), np.nan)

    with np.errstate(divide="ignore", invalid="ignore"):
        r1_t = np.log(numer / denom) / tr_ms
        concentration = (r1_t - (1.0 / t1_safe)[None, ...]) / r1
    concentration[:, ~valid_t1] = np.nan
    return concentration


def analyze_dce_series(
    image_series: np.ndarray,
    t1_map_ms: np.ndarray | Quantity,
    *,
    label_mask: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    times_seconds: Sequence[float] | None = None,
    total_scan_time_ms: float | None = None,
    frame_time_ms: float | None = None,
    repetition_time_ms: float = DEFAULT_DCE_TR_MS,
    flip_angle_deg: float = DEFAULT_DCE_FLIP_ANGLE_DEG,
    relaxivity_mM_ms: float = DEFAULT_GADOLINIUM_RELAXIVITY_MMM_MS,
    infusion_time_seconds: float | None = None,
    baseline_seconds: float | None = None,
    baseline_frames: int | None = None,
    min_baseline_frames: int = 5,
    smooth_sigma_frames: float = 2.0,
    spike_threshold: float = 1.05,
) -> DCEAnalysisResult:
    """Preprocess a DCE FLASH series and compute CA concentration maps/timecourses."""

    series = np.asarray(image_series, dtype=float)
    if series.ndim < 2:
        raise ValueError("image_series must be time-first with at least one spatial dimension")

    spatial_shape = series.shape[1:]
    t1_arr = _as_array(t1_map_ms).astype(float, copy=False)
    if t1_arr.shape != spatial_shape:
        t1_arr = match_spatial_mask(t1_arr, spatial_shape)

    mask_arr = None
    if label_mask is not None:
        label_mask = match_spatial_mask(np.asarray(label_mask), spatial_shape)
        if mask is None:
            mask = np.asarray(label_mask) != 0
    if mask is not None:
        mask_arr = match_spatial_mask(np.asarray(mask, dtype=bool), spatial_shape)
    else:
        mask_arr = np.isfinite(t1_arr) & (t1_arr > 0)

    if times_seconds is None:
        times_arr, frame_time_seconds = infer_dce_times_seconds(
            series.shape[0],
            total_scan_time_ms=total_scan_time_ms,
            frame_time_ms=frame_time_ms,
        )
    else:
        times_arr = np.asarray(times_seconds, dtype=float)
        if times_arr.ndim != 1 or times_arr.shape[0] != series.shape[0]:
            raise ValueError("times_seconds must be a 1D array matching the number of frames")
        if times_arr.shape[0] >= 2:
            frame_time_seconds = float(np.nanmedian(np.diff(times_arr)))
        else:
            frame_time_seconds = float((frame_time_ms or 1000.0) * 1e-3)

    if baseline_seconds is None and infusion_time_seconds is not None:
        baseline_seconds = float(infusion_time_seconds)
    if baseline_frames is None:
        if baseline_seconds is not None and np.isfinite(baseline_seconds) and baseline_seconds > 0:
            baseline_sel = times_arr <= float(baseline_seconds)
            baseline_frames = int(np.sum(baseline_sel))
        else:
            baseline_frames = 0
    baseline_frames = max(int(baseline_frames), int(min_baseline_frames))
    baseline_frames = min(baseline_frames, int(series.shape[0]))
    baseline_slice = slice(0, baseline_frames)

    cleaned = replace_isolated_spikes(series, mask=mask_arr, threshold=spike_threshold)
    reference_image = np.nanmean(cleaned[baseline_slice], axis=0)

    baseline = np.nanmean(cleaned[baseline_slice], axis=0)
    baseline = np.where(np.isfinite(baseline) & (baseline > 0), baseline, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        signal_ratio = cleaned / baseline[None, ...]
    signal_ratio = np.where(np.isfinite(signal_ratio), signal_ratio, np.nan)

    if smooth_sigma_frames and smooth_sigma_frames > 0:
        sigma = (float(smooth_sigma_frames),) + (0.0,) * (signal_ratio.ndim - 1)
        signal_ratio = nan_gaussian_filter(signal_ratio, sigma=sigma)

    concentration = spgr_signal_ratio_to_concentration(
        signal_ratio,
        t1_arr,
        repetition_time_ms=repetition_time_ms,
        flip_angle_deg=flip_angle_deg,
        relaxivity_mM_ms=relaxivity_mM_ms,
    )
    concentration = np.where(mask_arr[None, ...], concentration, np.nan)

    roi_timecourses = None
    labels: list[int] | None = None
    if label_mask is not None:
        roi_timecourses, labels = mean_timecourses_by_label(
            concentration,
            label_mask,
            baseline_frames=0,
            smooth_sigma=0.0,
        )

    return DCEAnalysisResult(
        concentration_mM=np.asarray(concentration),
        signal_ratio=np.asarray(signal_ratio),
        reference_image=np.asarray(reference_image),
        times_seconds=np.asarray(times_arr),
        frame_time_seconds=float(frame_time_seconds),
        baseline_frames=int(baseline_frames),
        timecourses_mM=None if roi_timecourses is None else np.asarray(roi_timecourses),
        labels=labels,
    )


__all__ = [
    "DCEAnalysisResult",
    "DEFAULT_DCE_FLIP_ANGLE_DEG",
    "DEFAULT_DCE_TR_MS",
    "DEFAULT_GADOLINIUM_RELAXIVITY_MMM_MS",
    "analyze_dce_series",
    "infer_dce_times_seconds",
    "replace_isolated_spikes",
    "spgr_signal_ratio_to_concentration",
]
