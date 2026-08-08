"""Observation-noise variance estimation for quantitative fitting."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from mrax.data_struct.dataset import DataSet
from mrax.data_struct.experiment import Experiment
from mrax.data_struct.quantity import Quantity

_NORMAL_MAD = 0.6744897501960817


@dataclass(frozen=True)
class VarianceEstimate:
    """Per-frame observation-variance estimate and estimator diagnostics.

    ``variance`` uses squared data units. For complex data it estimates
    :math:`E[|n|^2]`, the sum of real- and imaginary-channel variances.
    """

    variance: np.ndarray
    sample_count: np.ndarray
    method: str
    pooled_variance: float
    warnings: tuple[str, ...] = ()

    @property
    def standard_deviation(self) -> np.ndarray:
        """Return per-frame standard deviations in data units."""
        return np.sqrt(self.variance)

    @property
    def relative_variance(self) -> np.ndarray:
        """Return variances normalized to a median of one."""
        median = float(np.median(self.variance))
        return self.variance / median


def _flatten_to_frames(array: np.ndarray, experiment: Experiment) -> np.ndarray:
    dim_names = list(experiment.dims)
    non_spatial = experiment.non_spatial_dim_names
    spatial = experiment.spatial_dim_names
    axis_order = [dim_names.index(name) for name in non_spatial + spatial]
    if axis_order != list(range(array.ndim)):
        array = np.transpose(array, axis_order)
    return array.reshape((int(experiment.frame_count),) + tuple(experiment.spatial_shape))


def _prepare_mask(
    mask: object | None,
    mask_dims: tuple[str, ...] | None,
    experiment: Experiment,
) -> np.ndarray:
    if mask is None:
        return np.ones((int(experiment.frame_count),) + tuple(experiment.spatial_shape), dtype=bool)

    values = mask.values if isinstance(mask, Quantity) else mask
    array = np.asarray(values)
    dims = tuple(mask_dims) if mask_dims is not None else tuple(experiment.spatial_dim_names)
    if array.ndim != len(dims) or any(name not in experiment.dims for name in dims):
        raise ValueError("variance-estimation mask dimensions do not match the experiment")
    expected = tuple(int(experiment.dims[name]) for name in dims)
    if tuple(array.shape) != expected:
        raise ValueError(
            f"variance-estimation mask shape {tuple(array.shape)} does not match {expected}"
        )

    axis_by_name = {name: axis for axis, name in enumerate(dims)}
    aligned_shape = tuple(array.shape[axis_by_name[name]] if name in axis_by_name else 1 for name in experiment.dims)
    full_mask = np.broadcast_to(array.reshape(aligned_shape), tuple(experiment.shape))
    return _flatten_to_frames(np.isfinite(full_mask) & (full_mask > 0), experiment)


def _high_frequency_coefficients(
    frame: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    usable_axes = [axis for axis, size in enumerate(frame.shape) if int(size) >= 2]
    if not usable_axes:
        return np.asarray([], dtype=frame.dtype)

    if len(usable_axes) == 1:
        axis = usable_axes[0]
        differences = np.diff(frame, axis=axis) / np.sqrt(2.0)
        left = [slice(None)] * frame.ndim
        right = [slice(None)] * frame.ndim
        left[axis] = slice(0, -1)
        right[axis] = slice(1, None)
        valid_pairs = valid[tuple(left)] & valid[tuple(right)]
        coefficients = differences[valid_pairs]
        return coefficients[np.isfinite(coefficients)]

    axes = sorted(usable_axes, key=lambda axis: int(frame.shape[axis]), reverse=True)[:2]
    moved = np.moveaxis(frame, axes, (-2, -1))
    moved_valid = np.moveaxis(valid, axes, (-2, -1))
    height = 2 * (moved.shape[-2] // 2)
    width = 2 * (moved.shape[-1] // 2)
    moved = moved[..., :height, :width]
    moved_valid = moved_valid[..., :height, :width]

    top_left = moved[..., 0::2, 0::2]
    top_right = moved[..., 0::2, 1::2]
    bottom_left = moved[..., 1::2, 0::2]
    bottom_right = moved[..., 1::2, 1::2]
    block_valid = (
        moved_valid[..., 0::2, 0::2]
        & moved_valid[..., 0::2, 1::2]
        & moved_valid[..., 1::2, 0::2]
        & moved_valid[..., 1::2, 1::2]
    )
    # The diagonal Haar coefficient cancels constant and linear image structure;
    # division by two preserves the variance of independent pixel noise.
    coefficients = (top_left - top_right - bottom_left + bottom_right) / 2.0
    locally_constant = (
        (top_left == top_right)
        & (top_left == bottom_left)
        & (top_left == bottom_right)
    )
    coefficients = coefficients[block_valid & ~locally_constant]
    return coefficients[np.isfinite(coefficients)]


def _mad_variance(coefficients: np.ndarray) -> float:
    if np.iscomplexobj(coefficients):
        real = np.real(coefficients)
        imag = np.imag(coefficients)
        real_sigma = np.median(np.abs(real - np.median(real))) / _NORMAL_MAD
        imag_sigma = np.median(np.abs(imag - np.median(imag))) / _NORMAL_MAD
        return float(real_sigma**2 + imag_sigma**2)
    center = np.median(coefficients)
    sigma = np.median(np.abs(coefficients - center)) / _NORMAL_MAD
    return float(sigma**2)


def _looks_like_magnitude_data(data: np.ndarray, valid: np.ndarray) -> bool:
    if np.iscomplexobj(data):
        return False
    values = data[valid]
    return bool(values.size and np.mean(values >= 0) >= 0.995)


def _find_automatic_background(
    data: np.ndarray,
    valid: np.ndarray,
    min_samples: int,
    *,
    excluded_frame: int | None = None,
) -> np.ndarray | None:
    if excluded_frame is not None and int(data.shape[0]) > 1:
        keep = np.arange(int(data.shape[0])) != int(excluded_frame)
        data = data[keep]
        valid = valid[keep]
    safe = np.where(valid, np.abs(data), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        signal_rms = np.sqrt(np.nanmean(safe.astype(np.float64) ** 2, axis=0))
    candidate = np.all(valid, axis=0) & np.isfinite(signal_rms) & (signal_rms > 0)
    count = int(np.sum(candidate))
    if count < max(2 * int(min_samples), 16):
        return None

    desired = max(2 * int(min_samples), int(np.ceil(0.01 * count)))
    fraction = min(max(desired / count, 0.01), 0.1)
    candidate_values = signal_rms[candidate]
    threshold = float(np.quantile(candidate_values, fraction))
    background = candidate & (signal_rms <= threshold)
    if int(np.sum(background)) < int(min_samples):
        return None

    background_level = float(np.median(signal_rms[background]))
    field_level = float(np.quantile(candidate_values, 0.75))
    if field_level <= 0 or background_level / field_level >= 0.25:
        return None
    return background


def _background_variance(
    samples: np.ndarray,
    *,
    magnitude_data: bool,
) -> float:
    if magnitude_data:
        # For Rayleigh background magnitude M, median(M^2) = 2 sigma^2 log(2).
        # The inferred quadrature sigma^2 is also the high-SNR magnitude variance.
        return float(np.median(np.asarray(samples, dtype=np.float64) ** 2) / (2.0 * np.log(2.0)))
    return _mad_variance(samples)


def estimate_data_variance(
    experiment: Experiment,
    dataset: DataSet,
    *,
    mask: object | None = None,
    mask_dims: tuple[str, ...] | None = None,
    method: str = "auto",
    min_samples: int = 32,
    shrinkage: float = 32.0,
) -> VarianceEstimate:
    """Estimate robust per-frame observation variances before model fitting.

    ``method="auto"`` first looks for a low-signal background region. Nonnegative
    magnitude data uses the Rayleigh-background median-square estimator; signed
    real or complex data uses a median-absolute-deviation estimate. When no clear
    background exists, automatic mode falls back to ``"wavelet_mad"``, which
    applies a normalized diagonal Haar high-pass transform over the two largest
    spatial axes. Supplying ``mask`` explicitly defines the noise-sampling region
    rather than a fitting or tissue mask.

    Estimates with limited support are shrunk toward the median frame variance.
    Magnitude MRI is treated with the usual high-SNR Gaussian approximation;
    low-SNR Rician or noncentral-chi data require an acquisition-specific noise
    model instead of automatic Gaussian variance weighting.
    """
    method_name = method.lower()
    if method_name not in {"auto", "background", "wavelet_mad"}:
        raise ValueError("method must be 'auto', 'background', or 'wavelet_mad'")
    if int(min_samples) < 2:
        raise ValueError("min_samples must be at least 2")
    if not np.isfinite(shrinkage) or float(shrinkage) < 0:
        raise ValueError("shrinkage must be finite and non-negative")

    experiment.validate(dataset)
    data = _flatten_to_frames(np.asarray(dataset.data), experiment)
    valid = _prepare_mask(mask, mask_dims, experiment) & np.isfinite(data)
    raw_variance = np.full((int(experiment.frame_count),), np.nan, dtype=np.float64)
    sample_count = np.zeros((int(experiment.frame_count),), dtype=np.int64)
    estimator_warnings: list[str] = []
    method_used = method_name

    background_masks: list[np.ndarray | None] | None = None
    if method_name in {"auto", "background"}:
        if mask is None:
            background_masks = [
                _find_automatic_background(
                    data,
                    valid,
                    int(min_samples),
                    excluded_frame=frame_index,
                )
                for frame_index in range(int(experiment.frame_count))
            ]
        else:
            background = np.any(valid, axis=0)
            background_masks = [background] * int(experiment.frame_count)
        if background_masks is not None and any(item is not None for item in background_masks):
            magnitude_data = _looks_like_magnitude_data(data, valid)
            method_used = "background_rayleigh" if magnitude_data else "background_mad"
            for frame_index, (frame, frame_valid, background) in enumerate(
                zip(data, valid, background_masks)
            ):
                if background is None:
                    continue
                frame_samples = frame[frame_valid & background]
                frame_samples = frame_samples[np.isfinite(frame_samples)]
                sample_count[frame_index] = int(frame_samples.size)
                if frame_samples.size >= int(min_samples):
                    estimate = _background_variance(
                        frame_samples,
                        magnitude_data=magnitude_data,
                    )
                    if np.isfinite(estimate) and estimate > 0:
                        raw_variance[frame_index] = estimate
        elif method_name == "background":
            raise ValueError(
                "background variance estimation could not identify sufficient low-signal support; "
                "provide an explicit noise mask or use method='auto'"
            )

    if not np.any(np.isfinite(raw_variance) & (raw_variance > 0)):
        if method_name == "background":
            raise ValueError(
                "background variance estimation found no finite positive frame estimates; "
                "check the noise mask or use method='auto'"
            )
        if method_name == "auto":
            estimator_warnings.append("no clear background found; wavelet MAD fallback used")
        method_used = "wavelet_mad"
        raw_variance.fill(np.nan)
        sample_count.fill(0)
        for frame_index, (frame, frame_valid) in enumerate(zip(data, valid)):
            coefficients = _high_frequency_coefficients(frame, frame_valid)
            sample_count[frame_index] = int(coefficients.size)
            if coefficients.size >= int(min_samples):
                estimate = _mad_variance(coefficients)
                if np.isfinite(estimate) and estimate > 0:
                    raw_variance[frame_index] = estimate

    usable = np.isfinite(raw_variance) & (raw_variance > 0)
    if not np.any(usable):
        raise ValueError(
            "automatic data-variance estimation needs at least one frame with "
            f"{int(min_samples)} valid spatial high-frequency coefficients"
        )

    pooled = float(np.median(raw_variance[usable]))
    estimate_warnings = estimator_warnings
    if not np.all(usable):
        raw_variance[~usable] = pooled
        estimate_warnings.append("insufficient frame support; pooled variance substituted")

    strength = float(shrinkage)
    if strength > 0:
        blend = sample_count.astype(np.float64) / (sample_count + strength)
        variance = blend * raw_variance + (1.0 - blend) * pooled
    else:
        variance = raw_variance
    floor = max(pooled * 1e-6, np.finfo(np.float64).tiny)
    variance = np.maximum(variance, floor)

    return VarianceEstimate(
        variance=variance,
        sample_count=sample_count,
        method=method_used,
        pooled_variance=pooled,
        warnings=tuple(estimate_warnings),
    )


__all__ = ["VarianceEstimate", "estimate_data_variance"]
