"""Local MRI noise estimates in the original image intensity units.

Signal estimates use robust scales of locally detrended patches. Noise-only magnitude ROIs
use the central chi second moment (2 * effective coils * sigma**2). Neither
method removes magnitude bias or makes BM3D/BM4D spatially adaptive.
"""

from __future__ import annotations

import numpy as np

_NORMAL_MAD = 0.6744897501960817


def _patch(values, selected):
    coordinates = np.column_stack(np.nonzero(selected)).astype(float)
    samples = values[selected]
    if len(samples) > 4096:
        indices = np.linspace(0, len(samples) - 1, 4096, dtype=int)
        coordinates, samples = coordinates[indices], samples[indices]
    if np.ptp(samples) == 0:
        return 0.0, float(samples[0]), 0.0, 0.0
    design = np.column_stack((np.ones(len(samples)), coordinates - coordinates.mean(axis=0)))
    basis, singular, _ = np.linalg.svd(design, full_matrices=False)
    rank = int(np.sum(singular > singular[0] * max(design.shape) * np.finfo(float).eps))
    basis = basis[:, :rank]
    residual = samples - basis @ (basis.T @ samples)
    # Correct the variance lost by fitting the local affine intensity trend.
    leverage = np.sum(basis * basis, axis=1)
    residual /= np.sqrt(np.maximum(1 - leverage, 0.1))
    sigma = float(np.median(np.abs(residual - np.median(residual))) / _NORMAL_MAD)
    correlations = []
    if values.size > 4096:
        return sigma, float(np.median(samples)), 0.0, sigma
    detail = np.where(selected, values, np.nan)
    normalization = 1.0
    for axis in range(values.ndim):
        if values.shape[axis] >= 3:
            detail = np.diff(detail, n=2, axis=axis)
            normalization *= np.sqrt(6.0)
    detail = detail[np.isfinite(detail)]
    detail_sigma = sigma
    if len(detail) >= 16:
        detail_sigma = float(
            np.median(np.abs(detail - np.median(detail))) / (_NORMAL_MAD * normalization)
        )
    field = np.full(values.shape, np.nan)
    field[selected] = residual
    for axis in range(values.ndim):
        a, b = [slice(None)] * values.ndim, [slice(None)] * values.ndim
        a[axis], b[axis] = slice(None, -1), slice(1, None)
        left, right = field[tuple(a)], field[tuple(b)]
        valid = np.isfinite(left) & np.isfinite(right)
        if valid.sum() >= 16 and np.std(left[valid]) > 0 and np.std(right[valid]) > 0:
            correlations.append(float(np.corrcoef(left[valid], right[valid])[0, 1]))
    return sigma, float(np.median(samples)), max(correlations, default=0.0), detail_sigma


def estimate_noise(data, *, mask=None, model="auto", coils=1):
    """Return sigma and diagnostics for a real 2D/3D image or a complex image.

    ``auto`` favours signal-bearing patches in nonnegative MRI images, avoiding
    low-noise background dominating a spatially variable estimate. ``gaussian``
    uses all selected patches, suitable for signed data or a uniform signal ROI.
    ``background`` requires a noise-only magnitude ROI. ``coils`` is its effective
    number of independent complex channels (one for a Rician distribution).
    A selected ROI estimates noise only. It does not limit denoising.
    """
    array = np.asarray(data)
    if array.ndim not in (2, 3):
        raise ValueError("Noise estimation requires a 2D image or 3D volume")
    if model not in {"auto", "gaussian", "background"}:
        raise ValueError("Choose automatic, Gaussian or magnitude-background noise estimation")
    if isinstance(coils, bool) or not isinstance(coils, (int, np.integer)) or not 1 <= coils <= 64:
        raise ValueError("Effective noise coils must be an integer from 1 to 64")
    selected = (
        np.ones(array.shape, dtype=bool) if mask is None else np.array(mask, dtype=bool, copy=True)
    )
    if selected.shape != array.shape:
        raise ValueError("The noise ROI must match the input image grid")
    if np.iscomplexobj(array):
        if model == "background":
            raise ValueError("Magnitude-background estimation requires magnitude input")
        parts = {
            key: estimate_noise(value, mask=mask, model=model, coils=coils)
            for key, value in (("real", array.real), ("imaginary", array.imag))
        }
        result = dict(parts["real"])
        result.update(
            sigma=float(np.sqrt(np.mean([p["sigma"] ** 2 for p in parts.values()]))),
            components=parts,
            method="insufficient_samples"
            if all(p["samples"] < 16 for p in parts.values())
            else "complex_components",
            samples=max(p["samples"] for p in parts.values()),
            range=[
                min(p["range"][0] for p in parts.values()),
                max(p["range"][1] for p in parts.values()),
            ],
            warnings=sorted({w for p in parts.values() for w in p["warnings"]}),
        )
        return result
    array = np.asarray(array, dtype=np.float64)
    selected &= np.isfinite(array) & (array != 0)
    count = int(selected.sum())
    if model == "background" and mask is None:
        raise ValueError("Select a noise-only magnitude ROI for background estimation")
    if count < 16:
        if mask is not None:
            raise ValueError("The noise ROI needs at least 16 finite, nonzero pixels in each block")
        return {
            "sigma": 0.0,
            "range": [0.0, 0.0],
            "samples": count,
            "patches": 0,
            "method": "insufficient_samples",
            "warnings": ["Too few nonzero samples to estimate noise."],
        }
    if model == "background":
        samples = array[selected]
        if np.any(samples < 0):
            raise ValueError("A noise-only magnitude ROI cannot contain negative values")
        scale = float(np.max(samples))
        sigma = scale * float(np.sqrt(np.mean((samples / scale) ** 2) / (2 * coils)))
        return {
            "sigma": sigma,
            "range": [sigma, sigma],
            "samples": count,
            "patches": 1,
            "method": "magnitude_background",
            "warnings": [
                "Assumes a signal-free ROI and independent effective coils. Receiver count may differ."
            ],
        }
    if np.ptp(array[selected]) == 0:
        return {
            "sigma": 0.0,
            "range": [0.0, 0.0],
            "samples": count,
            "patches": 1,
            "method": "constant",
            "warnings": [],
        }
    width = 8 if array.ndim == 2 else 5
    sizes = tuple(min(width, n) for n in array.shape)
    starts = [
        sorted({*range(0, max(1, n - size + 1), size), n - size})
        for n, size in zip(array.shape, sizes)
    ]
    total = int(np.prod([len(s) for s in starts]))
    # Sample intact patches uniformly, never decimate neighbouring pixels.
    indices = np.linspace(0, total - 1, min(total, 256), dtype=int)
    patches = []
    for index in indices:
        cell = np.unravel_index(index, tuple(len(s) for s in starts))
        start = tuple(starts[axis][value] for axis, value in enumerate(cell))
        region = tuple(slice(i, i + size) for i, size in zip(start, sizes))
        keep = selected[region]
        if keep.sum() >= 16:
            patches.append(_patch(array[region], keep))
    if not patches:
        # A small or irregular ROI can still supply enough samples for a fit.
        lower, upper = (
            np.min(np.nonzero(selected), axis=1),
            np.max(np.nonzero(selected), axis=1) + 1,
        )
        region = tuple(slice(a, b) for a, b in zip(lower, upper))
        patches = [_patch(array[region], selected[region])]
    estimates = np.asarray(patches)
    warnings = []
    if model == "auto":
        # Fine-scale separable second differences suppress ramps, edges and
        # smooth texture. Avoid promoting structured patch residuals to noise.
        # Their L2 normalization preserves sigma for independent Gaussian noise.
        estimates[:, 0] = np.minimum(estimates[:, 0], estimates[:, 3])
    if model == "auto" and np.all(array[selected] >= 0):
        # At high magnitude SNR the Gaussian approximation is appropriate.
        # Do not guess a Rician/chi correction or a receiver count from intensities.
        signal = estimates[:, 1] > 4 * estimates[:, 0]
        low, high = np.quantile(estimates[:, 1], [0.2, 0.9])
        floor = estimates[estimates[:, 1] <= low, 0]
        background_scale = float(np.median(floor))
        if low < 4.5 * background_scale and high - low > 6 * background_scale:
            signal &= estimates[:, 1] > low + 3 * background_scale
        if np.any(signal):
            estimates = estimates[signal]
        else:
            warnings.append(
                "Low magnitude SNR: use a uniform signal ROI or a noise-only background ROI with effective coils."
            )
    sigmas = estimates[:, 0]
    sigma = float(np.median(sigmas))
    limits = [float(value) for value in np.quantile(sigmas, [0.1, 0.9])]
    if limits[1] > 2 * max(limits[0], np.finfo(float).tiny):
        warnings.append(
            "Noise estimates vary across patches. One sigma cannot represent spatially varying noise."
        )
    if np.median(estimates[:, 2]) > 0.2:
        warnings.append(
            "Correlated noise or image structure may bias this estimate. Consider a uniform ROI."
        )
    if len(estimates) < 4:
        warnings.append(
            "Few local patches are available. Check the estimate against a larger uniform ROI."
        )
    return {
        "sigma": sigma,
        "range": limits,
        "samples": count,
        "patches": len(estimates),
        "method": "local_signal_patches" if model == "auto" else "local_gaussian_patches",
        "warnings": warnings,
    }


def estimate_noise_sigma(data, **kwargs):
    """Return only the standard deviation. See :func:`estimate_noise` for diagnostics."""
    return estimate_noise(data, **kwargs)["sigma"]
