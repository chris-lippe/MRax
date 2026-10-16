"""Smooth receiver sensitivities from centred Cartesian calibration samples."""
from __future__ import annotations

import numpy as np
from scipy.fft import fftshift, ifftn, ifftshift
from scipy.ndimage import gaussian_filter


def estimate_cartesian_sensitivities(acquisition, *, smoothing_sigma=2., relative_floor=1e-6):
    """Estimate RSS-normalized maps on the acquisition encoding grid.

    ``auxiliary_samples['calibration_grid']`` must contain a complete centred
    Cartesian block with receiver and spatial axes. Calibration is zero-padded;
    coil images are RSS-normalized, smoothed and normalized again. Processing
    receivers one at a time bounds temporary memory. These maps are an estimate;
    externally calibrated maps can be passed directly to reconstruction.
    """
    acquisition.require_prepared()
    if acquisition.kind != 'cartesian' or acquisition.readout_coordinates is not None:
        raise ValueError("Cartesian calibration is required for this sensitivity estimator")
    if not np.isfinite(smoothing_sigma) or smoothing_sigma < 0 or not 0 < relative_floor < 1:
        raise ValueError("Sensitivity smoothing and relative_floor must be finite and nonnegative")
    calibration = acquisition.auxiliary_samples.get('calibration_grid')
    if calibration is None:
        raise ValueError("No complete centred calibration_grid is available; supply sensitivity maps explicitly")
    shape = acquisition.encoding_shape
    calibration = np.asarray(calibration)
    if calibration.shape[0] != acquisition.coil_count or calibration.ndim != len(shape) + 1 or any(a > b for a, b in zip(calibration.shape[1:], shape)):
        raise ValueError("Calibration receivers and spatial dimensions must match the acquisition")
    dtype = np.result_type(calibration.dtype, np.complex64)
    maps = np.empty((acquisition.coil_count,) + shape, dtype=dtype)
    energy = np.zeros(shape, dtype=np.float32 if dtype == np.complex64 else np.float64)
    target = tuple(slice(n // 2 - k // 2, n // 2 - k // 2 + k) for n, k in zip(shape, calibration.shape[1:]))
    for coil in range(acquisition.coil_count):
        grid = np.zeros(shape, dtype=dtype)
        grid[target] = calibration[coil]
        image = fftshift(ifftn(ifftshift(grid), norm='ortho'))
        maps[coil] = image
        energy += np.abs(image)**2
    denominator = np.maximum(np.sqrt(energy), np.sqrt(np.max(energy)) * relative_floor)
    if not np.any(denominator > 0):
        raise ValueError("Calibration contains no signal")
    maps /= denominator
    if smoothing_sigma:
        energy.fill(0)
        for coil in range(acquisition.coil_count):
            maps[coil] = gaussian_filter(maps[coil].real, smoothing_sigma) + 1j * gaussian_filter(maps[coil].imag, smoothing_sigma)
            energy += np.abs(maps[coil])**2
        maps /= np.maximum(np.sqrt(energy), np.sqrt(np.max(energy)) * relative_floor)
    return maps
