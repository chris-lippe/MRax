"""Vendor-neutral k-space and readout-profile processing helpers.

The functions in this module operate on plain arrays and numeric parameters. Importers
can map scanner metadata to these arguments, while users can apply the same corrections
manually to data from any source.
"""

from __future__ import annotations

from typing import Literal, Sequence

import jax.numpy as jnp
import numpy as np

ReadoutTransform = Literal["none", "fft", "ifft", "fft_ifftshift", "ifft_ifftshift"]


def _normalize_axis(axis: int, ndim: int) -> int:
    axis_i = int(axis)
    if axis_i < 0:
        axis_i += int(ndim)
    if axis_i < 0 or axis_i >= int(ndim):
        raise ValueError(f"axis {axis} is out of bounds for array with {ndim} dimensions")
    return axis_i


def _reshape_vector_for_axis(values: np.ndarray, *, axis: int, ndim: int) -> np.ndarray:
    shape = [1] * int(ndim)
    shape[int(axis)] = int(values.size)
    return values.reshape(tuple(shape))


def apply_channel_scale_phase(
    data: np.ndarray,
    *,
    scale: Sequence[float] | np.ndarray | None = None,
    phase_degrees: Sequence[float] | np.ndarray | None = None,
    coil_axis: int = 1,
) -> np.ndarray:
    """Apply per-coil scale factors and phase offsets to k-space or coil images.

    Parameters
    ----------
    data:
        Array with one coil/channel dimension. Typical shapes are
        ``(frame, coil, sample)`` for non-Cartesian streams or
        ``(frame, coil, *spatial)`` for Cartesian grids and coil images.
    scale:
        Optional real multiplicative factors, one per coil.
    phase_degrees:
        Optional phase offsets in degrees, one per coil. Positive values multiply the
        corresponding coil by ``exp(1j * phase)``.
    coil_axis:
        Axis containing the coil/channel dimension.

    Returns
    -------
    np.ndarray
        Corrected copy of the input array.

    Notes
    -----
    Use this for receiver normalization from any vendor. Bruker maps
    ``RecoScaleChan`` and ``RecoPhaseChan`` to these arguments; other importers can map
    equivalent receiver gains and phase references the same way.
    """

    arr = np.asarray(data)
    axis = _normalize_axis(coil_axis, arr.ndim)
    channel_count = int(arr.shape[axis])
    out = arr

    if scale is not None:
        scale_arr = np.asarray(scale, dtype=np.float32).ravel()
        if scale_arr.size != channel_count:
            raise ValueError(f"scale has {scale_arr.size} entries but coil axis has {channel_count}")
        out = out * _reshape_vector_for_axis(scale_arr, axis=axis, ndim=out.ndim)

    if phase_degrees is not None:
        phase_arr = np.asarray(phase_degrees, dtype=np.float32).ravel()
        if phase_arr.size != channel_count:
            raise ValueError(f"phase_degrees has {phase_arr.size} entries but coil axis has {channel_count}")
        phase = np.exp(1j * np.deg2rad(phase_arr)).astype(np.complex64)
        out = out * _reshape_vector_for_axis(phase, axis=axis, ndim=out.ndim)

    return np.asarray(out)


def apply_samplewise_correction(
    data: np.ndarray,
    correction: Sequence[complex] | np.ndarray,
    *,
    sample_axis: int = -1,
) -> np.ndarray:
    """Multiply a k-space stream by a per-sample complex correction.

    The correction can be a one-dimensional vector matching ``sample_axis`` or an
    already-broadcastable array. This is useful for B0/demodulation phase corrections,
    trajectory-specific scaling, or calibration vectors derived outside MRax.
    """

    arr = np.asarray(data)
    axis = _normalize_axis(sample_axis, arr.ndim)
    corr = np.asarray(correction)
    if corr.ndim == 1:
        if int(corr.size) != int(arr.shape[axis]):
            raise ValueError(f"correction has {corr.size} samples but data axis has {arr.shape[axis]}")
        corr = _reshape_vector_for_axis(corr, axis=axis, ndim=arr.ndim)
    return np.asarray(arr * corr)


def estimate_radial_density(
    trajectory: np.ndarray,
    *,
    spatial_shape: tuple[int, ...] | None = None,
    power: float | None = None,
) -> np.ndarray:
    """Estimate radial density compensation weights from a 2D or 3D trajectory.

    Parameters
    ----------
    trajectory:
        ``(sample, D)`` array in normalized k-space coordinates, usually cycles/pixel.
    spatial_shape:
        Optional output image shape. When supplied, weights are normalized to sum to the
        number of spatial pixels/voxels. Otherwise they are normalized to mean 1.
    power:
        Radial power. Defaults to ``1`` for 2D trajectories and ``2`` for 3D trajectories,
        matching common ``rho`` and ``rho^2`` density compensation.

    Notes
    -----
    This is a simple analytic estimate. Prefer sequence-provided or iteratively
    optimized density weights when they are available and validated.
    """

    traj = np.asarray(trajectory, dtype=np.float32)
    if traj.ndim != 2 or traj.shape[1] not in (2, 3):
        raise ValueError("trajectory must have shape (num_samples, 2) or (num_samples, 3)")
    exponent = float(traj.shape[1] - 1 if power is None else power)
    density = np.linalg.norm(traj, axis=1).astype(np.float32) ** exponent
    if spatial_shape is not None:
        target = float(np.prod(spatial_shape))
        total = float(np.sum(density))
        if total > 0.0 and np.isfinite(total) and target > 0.0:
            density = density * (target / total)
    else:
        mean = float(np.mean(density)) if density.size else 0.0
        if mean > 0.0 and np.isfinite(mean):
            density = density / mean
    if not np.all(np.isfinite(density)) or not np.any(density > 0):
        density = np.ones((traj.shape[0],), dtype=np.float32)
    return density.astype(np.float32)


def readout_crop_bounds(input_size: int, output_size: int, offset: int | None = None) -> tuple[int, int]:
    """Return ``(start, stop)`` bounds for a readout crop.

    ``offset=None`` performs a centered crop. Explicit offsets are useful when a vendor
    reconstruction records an asymmetric cutoff.
    """

    in_size = int(input_size)
    out_size = int(output_size)
    if in_size <= 0 or out_size <= 0:
        raise ValueError("input_size and output_size must be positive")
    if out_size > in_size:
        raise ValueError("output_size cannot exceed input_size")
    start = (in_size - out_size) // 2 if offset is None else int(offset)
    stop = start + out_size
    if start < 0 or stop > in_size:
        raise ValueError("crop offset and output_size exceed input_size")
    return start, stop


def crop_along_axis(
    data: np.ndarray,
    output_size: int,
    *,
    axis: int = -1,
    offset: int | None = None,
) -> np.ndarray:
    """Crop one axis of an array, centered by default."""

    arr = np.asarray(data)
    axis_i = _normalize_axis(axis, arr.ndim)
    start, stop = readout_crop_bounds(int(arr.shape[axis_i]), int(output_size), offset=offset)
    slices = [slice(None)] * arr.ndim
    slices[axis_i] = slice(start, stop)
    return np.asarray(arr[tuple(slices)])


def apply_readout_phase_shift(
    data: np.ndarray,
    shift_samples: float,
    *,
    axis: int = -1,
) -> np.ndarray:
    """Apply a fractional readout shift as a linear phase ramp.

    ``shift_samples`` is expressed in samples over the selected axis. Positive values
    use ``exp(+2j*pi*shift*n/N)``. Pass a negative shift when your convention requires
    the inverse ramp.
    """

    arr = np.asarray(data)
    axis_i = _normalize_axis(axis, arr.ndim)
    n = int(arr.shape[axis_i])
    if n <= 0:
        raise ValueError("readout axis must be non-empty")
    if float(shift_samples) == 0.0:
        return np.asarray(arr)
    sample = np.arange(n, dtype=np.float32)
    ramp = np.exp(2j * np.pi * float(shift_samples) * sample / float(n)).astype(np.complex64)
    return np.asarray(arr * _reshape_vector_for_axis(ramp, axis=axis_i, ndim=arr.ndim))


def fid_profiles_to_kspace(
    profiles: np.ndarray,
    *,
    axis: int = -1,
    transform: ReadoutTransform = "fft_ifftshift",
    output_points: int | None = None,
    crop_offset: int | None = None,
    readout_shift: float | None = None,
    dtype: np.dtype | type | None = np.complex64,
) -> np.ndarray:
    """Transform time-domain readout profiles into frequency-domain k-space profiles.

    Parameters
    ----------
    profiles:
        Input readout profiles. The selected ``axis`` is interpreted as the readout
        time/frequency axis.
    transform:
        Readout transform to apply before cropping. ``"fft_ifftshift"`` matches the
        common pattern where a centered FID-like readout is shifted to FFT order and
        transformed along the readout.
    output_points, crop_offset:
        Optional crop on the transformed readout axis. ``crop_offset=None`` means
        centered crop.
    readout_shift:
        Optional fractional sample shift applied after the crop as a phase ramp.
    dtype:
        Optional output dtype. Use ``None`` to keep NumPy's transform dtype.

    Returns
    -------
    np.ndarray
        Frequency-domain readout profiles.

    Notes
    -----
    This applies to sequences that store raw FIDs or
    oversampled readouts that must be Fourier-transformed, cropped, and phase shifted
    before samples are paired with a trajectory.
    """

    arr = np.asarray(profiles)
    axis_i = _normalize_axis(axis, arr.ndim)
    mode = str(transform).lower()
    if mode == "none":
        out = arr
    elif mode == "fft":
        out = np.fft.fft(arr, axis=axis_i)
    elif mode == "ifft":
        out = np.fft.ifft(arr, axis=axis_i)
    elif mode == "fft_ifftshift":
        out = np.fft.fft(np.fft.ifftshift(arr, axes=axis_i), axis=axis_i)
    elif mode == "ifft_ifftshift":
        out = np.fft.ifft(np.fft.ifftshift(arr, axes=axis_i), axis=axis_i)
    else:
        raise ValueError("transform must be one of 'none', 'fft', 'ifft', 'fft_ifftshift', 'ifft_ifftshift'")

    if output_points is not None:
        out = crop_along_axis(out, int(output_points), axis=axis_i, offset=crop_offset)
    if readout_shift is not None and float(readout_shift) != 0.0:
        out = apply_readout_phase_shift(out, float(readout_shift), axis=axis_i)
    if dtype is not None:
        out = out.astype(dtype, copy=False)
    return np.asarray(out)


def pad_centered_kspace(array, target_shape):
    """Pad trailing Fourier axes, preserving DC at index ``size // 2``."""
    values = jnp.asarray(array)
    target = tuple(int(size) for size in target_shape)
    if not target or len(target) > values.ndim or any(size <= 0 for size in target):
        raise ValueError("Target shape must identify positive trailing encoding dimensions")
    widths = [(0, 0)] * (values.ndim - len(target))
    for source, destination in zip(values.shape[-len(target):], target, strict=True):
        if destination < source:
            raise ValueError("Fourier padding cannot discard acquired samples")
        before = destination // 2 - source // 2
        widths.append((before, destination - source - before))
    return jnp.pad(values, widths)
