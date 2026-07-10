"""Vendor-agnostic variational MRI reconstruction (Cartesian + NUFFT).

This module provides forward/adjoint encoding operators and small-footprint solvers
that reuse the project's Optax/proximal tooling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Sequence

import jax
import jax.numpy as jnp
import jax.scipy.ndimage as jnd
import numpy as np
import optax

from mrax.core.loss_builder import (
    _align_predicted,
    _broadcast_fields_to_field_shape,
    _collapse_params_for_constant_dims,
    _data_mask_from_observables,
    _extract_spatial_metadata,
    _field_metadata,
    _flatten_to_frames,
    _frame_mask_from_observables,
    _mask_fields_for_output,
    _masked_rms,
    _normalize_constant_dims,
    _prepare_frame_mask,
    _prepare_mask_arrays,
    _reduce_mask_to_field,
    _sanitize_observables,
    make_model_callable,
    warn_if_fit_requires_model_override,
)
from mrax.data_struct.dimensions import expand_observables_to_frames
from mrax.data_struct.experiment import Experiment
from mrax.data_struct.model import Model
from mrax.data_struct.quantity import Quantity, attach_units, require_quantity_dict
from mrax.opt.optimizers import make_optimizer

try:  # Runtime dependency; keep this lazy so import-time errors stay localized.
    import jax_finufft
except Exception:  # pragma: no cover - exercised via skipif tests
    jax_finufft = None  # type: ignore[assignment]

Array = jnp.ndarray


@dataclass(frozen=True)
class RegridApodizationCorrection:
    """Kaiser-Bessel deapodization settings for gridding-style reconstructions.

    The correction is applied in the image domain after optional cropping. ``full_shape``
    and ``crop_offset`` describe the grid coordinate system used to evaluate the
    separable correction when the reconstruction is cropped from an oversampled grid.
    """

    kernel_width: float
    shape_parameter: float
    full_shape: tuple[int, ...] | None = None
    crop_offset: tuple[int, ...] | None = None
    strength: float = 1.0


def fftnc(x: Array, axes: Sequence[int] = (-2, -1)) -> Array:
    """Centered FFT with orthonormal scaling (JAX)."""

    axes = tuple(int(ax) for ax in axes)
    return jnp.fft.fftshift(
        jnp.fft.fftn(jnp.fft.ifftshift(x, axes=axes), axes=axes, norm="ortho"),
        axes=axes,
    )


def ifftnc(x: Array, axes: Sequence[int] = (-2, -1)) -> Array:
    """Centered inverse FFT with orthonormal scaling (JAX)."""

    axes = tuple(int(ax) for ax in axes)
    return jnp.fft.fftshift(
        jnp.fft.ifftn(jnp.fft.ifftshift(x, axes=axes), axes=axes, norm="ortho"),
        axes=axes,
    )


def _as_mask(mask: Array | np.ndarray | None, spatial_shape: tuple[int, ...]) -> Array | None:
    if mask is None:
        return None
    mask_arr = jnp.asarray(mask)
    if mask_arr.shape == spatial_shape:
        return mask_arr
    spatial_ndim = len(spatial_shape)
    if mask_arr.ndim >= spatial_ndim and tuple(mask_arr.shape[-spatial_ndim:]) == spatial_shape:
        leading_shape = mask_arr.shape[:-spatial_ndim]
        if all(int(s) == 1 for s in leading_shape):
            return mask_arr.reshape(spatial_shape)
        lead_axes = tuple(range(mask_arr.ndim - spatial_ndim))
        return jnp.any(mask_arr != 0, axis=lead_axes)
    raise ValueError(f"mask shape {tuple(mask_arr.shape)} does not match spatial shape {spatial_shape}")


def _coerce_reference_image(reference_image: Array | np.ndarray, spatial_shape: tuple[int, ...]) -> Array:
    """Return a spatial-only reference image, squeezing leading singleton dims when possible."""
    ref = jnp.asarray(reference_image)
    if ref.shape == spatial_shape:
        return ref
    if ref.ndim == len(spatial_shape) + 1 and int(ref.shape[0]) == 1:
        ref = ref[0]
        if ref.shape == spatial_shape:
            return ref
    if ref.ndim > len(spatial_shape) and tuple(int(v) for v in ref.shape[-len(spatial_shape) :]) == spatial_shape:
        leading = ref.shape[:-len(spatial_shape)]
        if all(int(s) == 1 for s in leading):
            return ref.reshape(spatial_shape)
    return ref


def _infer_spatial_shape_from_kspace(kspace: Array | np.ndarray) -> tuple[int, int]:
    k = np.asarray(kspace)
    if k.ndim < 2:
        raise ValueError("kspace must be at least 2D for Cartesian reconstruction")
    return (int(k.shape[-2]), int(k.shape[-1]))


def _has_coils_cartesian(
    kspace: Array | np.ndarray, spatial_shape: tuple[int, ...], sensitivities: Array | np.ndarray | None
) -> bool:
    k = np.asarray(kspace)
    if sensitivities is not None:
        return True
    # Heuristic: (.., coils, *spatial) must have at least one extra dim beyond spatial.
    return k.ndim >= len(spatial_shape) + 2


def _ensure_coil_axis(
    arr: Array,
    *,
    has_coils: bool,
    spatial_ndim: int,
    coil_axis: int,
) -> tuple[Array, bool]:
    """Return (arr_with_coil_axis, added_coil_axis)."""

    if has_coils:
        return arr, False
    # Insert a singleton coil axis just before spatial dims.
    axis = coil_axis if coil_axis >= 0 else arr.ndim + coil_axis + 1
    if axis < 0 or axis > arr.ndim:
        raise ValueError("Invalid coil_axis for array without coils")
    return jnp.expand_dims(arr, axis=axis), True


def _coil_combine_sense(
    coil_images: Array,
    sensitivities: Array | None,
    *,
    coil_axis: int,
    eps: float = 1e-8,
) -> Array:
    if sensitivities is None:
        return jnp.sum(coil_images, axis=coil_axis)
    sens = jnp.asarray(sensitivities)
    coil_axis = int(coil_axis)
    target_coil_axis = -int(sens.ndim)
    if coil_axis != target_coil_axis:
        coil_images = jnp.moveaxis(coil_images, coil_axis, target_coil_axis)
    if sens.ndim < 3:
        raise ValueError("sensitivities must have shape (coils, *spatial) for SENSE-style combining")

    # Broadcast sensitivities across any leading (batch) dimensions.
    leading_ndim = coil_images.ndim - sens.ndim
    if leading_ndim < 0:
        raise ValueError("coil_images has fewer dimensions than sensitivities")
    sens_b = sens.reshape((1,) * leading_ndim + sens.shape)

    num = jnp.sum(jnp.conj(sens_b) * coil_images, axis=target_coil_axis)
    denom = jnp.sum(jnp.abs(sens_b) ** 2, axis=target_coil_axis)
    return num / (denom + eps)


def _checkerboard(shape: tuple[int, ...]) -> Array:
    spatial_shape = tuple(int(v) for v in shape)
    parity = jnp.zeros(spatial_shape, dtype=jnp.int32)
    for axis, size in enumerate(spatial_shape):
        reshape = (1,) * axis + (int(size),) + (1,) * (len(spatial_shape) - axis - 1)
        parity = parity + jnp.arange(int(size), dtype=jnp.int32).reshape(reshape)
    return jnp.where((parity % 2) == 0, 1.0, -1.0).astype(jnp.float32)


def _center_crop_lastn(
    image: Array,
    output_shape: tuple[int, ...],
    crop_offset: tuple[int, ...] | None = None,
) -> Array:
    arr = jnp.asarray(image)
    out_shape = tuple(int(v) for v in output_shape)
    spatial_ndim = len(out_shape)
    in_shape = tuple(int(v) for v in arr.shape[-spatial_ndim:])
    if any(v <= 0 for v in out_shape):
        raise ValueError("output_shape must be positive")
    if any(out > inn for out, inn in zip(out_shape, in_shape)):
        raise ValueError("output_shape cannot be larger than the gridded image shape")
    if crop_offset is None:
        offsets = tuple((inn - out) // 2 for out, inn in zip(out_shape, in_shape))
    else:
        offsets = tuple(int(v) for v in crop_offset)
    if len(offsets) != spatial_ndim:
        raise ValueError("crop_offset dimensionality must match output_shape")
    if any(off < 0 or off + out > inn for off, out, inn in zip(offsets, out_shape, in_shape)):
        raise ValueError("crop_offset and output_shape exceed the gridded image shape")
    slices = (slice(None),) * (arr.ndim - spatial_ndim) + tuple(
        slice(off, off + out) for off, out in zip(offsets, out_shape)
    )
    return arr[slices]


def _kaiser_bessel_deapodization_1d(
    output_size: int,
    *,
    full_size: int,
    offset: int,
    kernel_width: float,
    shape_parameter: float,
    strength: float,
) -> Array:
    """Approximate deapodization for a Kaiser-Bessel gridding kernel."""

    if output_size <= 0 or full_size <= 0:
        raise ValueError("deapodization sizes must be positive")
    width = float(kernel_width)
    beta = float(shape_parameter)
    if width <= 0.0 or beta <= 0.0:
        return jnp.ones((int(output_size),), dtype=jnp.float32)

    idx = jnp.arange(int(offset), int(offset) + int(output_size), dtype=jnp.float32)
    u = (idx - (float(full_size) / 2.0)) / float(full_size)
    arg = beta**2 - (np.pi * width * u) ** 2
    sqrt_pos = jnp.sqrt(jnp.maximum(arg, 0.0))
    sqrt_neg = jnp.sqrt(jnp.maximum(-arg, 0.0))
    pos = jnp.sinh(sqrt_pos) / jnp.where(sqrt_pos > 1e-6, sqrt_pos, 1.0)
    neg = jnp.sin(sqrt_neg) / jnp.where(sqrt_neg > 1e-6, sqrt_neg, 1.0)
    response = jnp.where(arg >= 0.0, pos, neg)
    response = response / jnp.maximum(jnp.max(jnp.abs(response)), 1e-6)
    correction = 1.0 / jnp.maximum(jnp.abs(response), 1e-3)
    if float(strength) != 1.0:
        correction = correction ** float(strength)
    return correction.astype(jnp.float32)


def _apply_regrid_apodization_correction(
    image: Array,
    correction: RegridApodizationCorrection,
    spatial_shape: tuple[int, ...],
) -> Array:
    arr = jnp.asarray(image)
    out_shape = tuple(int(v) for v in spatial_shape)
    spatial_ndim = len(out_shape)
    full_shape = tuple(int(v) for v in (correction.full_shape or out_shape))
    if len(full_shape) != spatial_ndim:
        raise ValueError("apodization full_shape dimensionality must match spatial_shape")
    crop_offset = tuple(
        int(v) for v in (correction.crop_offset or tuple((int(full) - int(out)) // 2 for full, out in zip(full_shape, out_shape)))
    )
    if len(crop_offset) != spatial_ndim:
        raise ValueError("apodization crop_offset dimensionality must match spatial_shape")
    corrected = arr
    for axis, (out, full, off) in enumerate(zip(out_shape, full_shape, crop_offset)):
        corr_1d = _kaiser_bessel_deapodization_1d(
            int(out),
            full_size=int(full),
            offset=int(off),
            kernel_width=float(correction.kernel_width),
            shape_parameter=float(correction.shape_parameter),
            strength=float(correction.strength),
        )
        reshape = (1,) * (arr.ndim - spatial_ndim + axis) + (int(out),) + (1,) * (spatial_ndim - axis - 1)
        corrected = corrected * corr_1d.reshape(reshape)
    return corrected


def _ellipse_mask(shape: tuple[int, ...], radius: float = 1.0) -> Array:
    spatial_shape = tuple(int(v) for v in shape)
    dist = jnp.zeros(spatial_shape, dtype=jnp.float32)
    for axis, size in enumerate(spatial_shape):
        coords = (jnp.arange(int(size), dtype=jnp.float32) + 0.5 - size / 2.0) / (size / 2.0)
        reshape = (1,) * axis + (int(size),) + (1,) * (len(spatial_shape) - axis - 1)
        dist = dist + coords.reshape(reshape) ** 2
    return (dist <= float(radius) ** 2).astype(jnp.float32)


def _apply_image_mask(image: Array, image_mask: str | Array | np.ndarray | None, spatial_shape: tuple[int, ...]) -> Array:
    if image_mask is None:
        return image
    arr = jnp.asarray(image)
    target_shape = tuple(int(v) for v in spatial_shape)
    if isinstance(image_mask, str):
        mask_name = image_mask.lower()
        if mask_name not in ("ellipse", "elliptical", "circle", "circular"):
            raise ValueError(f"Unsupported image_mask '{image_mask}'")
        mask_arr = _ellipse_mask(target_shape)
    else:
        mask_arr = jnp.asarray(image_mask, dtype=jnp.float32)
        if tuple(int(v) for v in mask_arr.shape) != target_shape:
            raise ValueError("image_mask must match the reconstructed spatial shape")
    return arr * mask_arr.reshape((1,) * (arr.ndim - mask_arr.ndim) + mask_arr.shape)


def checkerboard_phase(shape: tuple[int, ...]) -> Array:
    """Return a ``+1/-1`` checkerboard phase array for spatial image axes.

    This is useful after FFT-style reconstructions when a scanner convention or gridding
    path leaves the image on the opposite centered-grid parity. It is the public version
    of the phase step optionally used by :func:`reconstruct_regrid`.
    """

    return _checkerboard(tuple(int(v) for v in shape))


def center_crop_spatial(
    image: Array | np.ndarray,
    output_shape: tuple[int, ...],
    crop_offset: tuple[int, ...] | None = None,
) -> Array:
    """Crop the last ``len(output_shape)`` axes of an image or coil-image stack.

    Use this after reconstructing on an oversampled grid. ``crop_offset=None`` performs a
    centered crop; explicit offsets reproduce vendor cutoff windows or asymmetric grids.
    """

    return _center_crop_lastn(jnp.asarray(image), output_shape, crop_offset)


def apply_regrid_apodization_correction(
    image: Array | np.ndarray,
    correction: RegridApodizationCorrection,
    spatial_shape: tuple[int, ...],
) -> Array:
    """Apply separable Kaiser-Bessel deapodization to a gridded reconstruction.

    Apply this only when the gridding kernel matches the correction. For example, a
    vendor reconstruction using a Kaiser-Bessel convolution kernel can require this
    image-domain correction; applying it to nearest-neighbor gridding is usually only a
    diagnostic experiment.
    """

    return _apply_regrid_apodization_correction(jnp.asarray(image), correction, spatial_shape)


def elliptical_support_mask(shape: tuple[int, ...], radius: float = 1.0) -> Array:
    """Return a circular/elliptical support mask for 2D or 3D image shapes."""

    return _ellipse_mask(tuple(int(v) for v in shape), radius=radius)


def apply_image_support_mask(
    image: Array | np.ndarray,
    image_mask: str | Array | np.ndarray | None,
    spatial_shape: tuple[int, ...],
) -> Array:
    """Apply an image-domain support mask to the last spatial axes.

    ``image_mask`` may be ``"ellipse"``/``"circle"`` or an explicit mask matching
    ``spatial_shape``. Passing ``None`` returns the input unchanged.
    """

    return _apply_image_mask(jnp.asarray(image), image_mask, spatial_shape)


def combine_coil_images(
    coil_images: Array | np.ndarray,
    *,
    method: Literal["sum", "rss", "sense"] = "sense",
    sensitivities: Array | np.ndarray | None = None,
    coil_axis: int | None = None,
    eps: float = 1e-8,
) -> Array:
    """Combine coil images with sum, root-sum-of-squares, or SENSE weighting.

    Parameters
    ----------
    coil_images:
        Array containing a coil axis and trailing spatial axes.
    method:
        ``"sum"`` for direct complex summation, ``"rss"`` for magnitude RSS, or
        ``"sense"`` for sensitivity-weighted combination. ``"sense"`` falls back to
        sum when sensitivities are not supplied, matching :func:`reconstruct_regrid`.
    sensitivities:
        Optional ``(coil, *spatial)`` sensitivity map for SENSE combination.
    coil_axis:
        Axis containing coils. If omitted, MRax assumes ``(frame, coil, *spatial)`` for
        non-SENSE data and infers the axis from ``sensitivities`` for SENSE data.
    """

    arr = jnp.asarray(coil_images)
    combine = str(method).lower()
    if combine not in ("sum", "rss", "sense"):
        raise ValueError("method must be one of 'sum', 'rss', or 'sense'")

    sens = None if sensitivities is None else jnp.asarray(sensitivities)
    if coil_axis is None:
        if sens is not None:
            spatial_ndim = int(sens.ndim) - 1
            axis = -(spatial_ndim + 1)
        else:
            axis = 1 if arr.ndim >= 2 else 0
    else:
        axis = int(coil_axis)

    if combine == "rss":
        return jnp.sqrt(jnp.sum(jnp.abs(arr) ** 2, axis=axis))
    if combine == "sense":
        return _coil_combine_sense(arr, sens, coil_axis=axis, eps=eps) if sens is not None else jnp.sum(arr, axis=axis)
    return jnp.sum(arr, axis=axis)


def _forward_gradient_last2(x: Array) -> Array:
    dx = jnp.diff(x, axis=-2)
    dy = jnp.diff(x, axis=-1)
    dx = jnp.pad(dx, [(0, 0)] * (x.ndim - 2) + [(0, 1), (0, 0)])
    dy = jnp.pad(dy, [(0, 0)] * (x.ndim - 2) + [(0, 0), (0, 1)])
    return jnp.stack([dx, dy], axis=-1)


def _divergence_last2(v: Array) -> Array:
    px = v[..., 0]
    py = v[..., 1]
    pad_x = [(0, 0)] * px.ndim
    pad_x[-2] = (1, 0)
    pad_y = [(0, 0)] * py.ndim
    pad_y[-1] = (1, 0)
    px_pad = jnp.pad(px, pad_x)
    py_pad = jnp.pad(py, pad_y)
    div_x = px_pad[..., 1:, :] - px_pad[..., :-1, :]
    div_y = py_pad[..., :, 1:] - py_pad[..., :, :-1]
    return div_x + div_y


def _shift_image_last2(
    image: Array,
    shift: Array | np.ndarray,
    *,
    order: int = 1,
    mode: str = "constant",
    cval: float = 0.0,
) -> Array:
    """Shift the last two axes of an array by ``shift`` (index units) using JAX."""

    if int(order) not in (0, 1):
        raise ValueError("shift order must be 0 (nearest) or 1 (linear) for JAX shifts")

    arr = jnp.asarray(image)
    shift_arr = jnp.asarray(shift, dtype=jnp.float32).reshape(-1)
    if shift_arr.shape[0] != 2:
        raise ValueError("shift must have length 2 for 2D reconstruction")

    ny, nx = arr.shape[-2:]
    coords = jnp.meshgrid(
        jnp.arange(ny, dtype=jnp.float32),
        jnp.arange(nx, dtype=jnp.float32),
        indexing="ij",
    )
    coords_stack = jnp.stack(coords, axis=0)
    shift_b = shift_arr.reshape((2, 1, 1))
    coords_shifted = coords_stack - shift_b

    def _apply(single: Array) -> Array:
        return jnd.map_coordinates(single, coords_shifted, order=order, mode=str(mode), cval=float(cval))

    if arr.ndim == 2:
        if jnp.iscomplexobj(arr):
            real = _apply(jnp.real(arr))
            imag = _apply(jnp.imag(arr))
            return real + 1j * imag
        return _apply(arr)

    leading_shape = arr.shape[:-2]
    batch = int(np.prod(leading_shape)) if leading_shape else 1
    flat = arr.reshape((batch, ny, nx))
    if jnp.iscomplexobj(flat):
        real = jax.vmap(_apply)(jnp.real(flat))
        imag = jax.vmap(_apply)(jnp.imag(flat))
        out = real + 1j * imag
    else:
        out = jax.vmap(_apply)(flat)
    return out.reshape(leading_shape + (ny, nx))


def _default_density_from_trajectory(
    trajectory: Array | np.ndarray,
    *,
    spatial_shape: tuple[int, ...] | None = None,
) -> Array:
    """Simple radial density compensation (:math:`|k|`) with light normalization.

    If ``spatial_shape`` is provided, normalize so the weights sum to the number of
    spatial pixels. Otherwise, normalize to mean 1.
    """

    traj = jnp.asarray(trajectory, dtype=jnp.float32)
    if traj.ndim != 2 or traj.shape[1] not in (2, 3):
        raise ValueError("trajectory must have shape (num_samples, 2) or (num_samples, 3) for density compensation")
    r = jnp.linalg.norm(traj, axis=1)
    density = r if int(traj.shape[1]) == 2 else r**2
    if spatial_shape is not None:
        target = float(np.prod(spatial_shape))
        total = jnp.sum(density)
        scale = jnp.where(
            (total > 0) & jnp.isfinite(total) & (target > 0),
            target / total,
            1.0,
        )
        density = density * scale
    else:
        mean = jnp.mean(density)
        scale = jnp.where((mean > 0) & jnp.isfinite(mean), 1.0 / mean, 1.0)
        density = density * scale
    finite = jnp.all(jnp.isfinite(density))
    density = jnp.where(finite, density, jnp.ones_like(density))
    return density.astype(jnp.float32)


def prox_tv_chambolle(
    x: Array,
    weight: float,
    *,
    iterations: int = 30,
    tau: float = 0.25,
) -> Array:
    """prox_{weight*TV}(x) via Chambolle's projection algorithm (2D TV on last two axes).

    Works on batched arrays; TV is computed along the last two axes.
    """

    weight_f = jnp.asarray(weight, dtype=jnp.float32)
    x = jnp.asarray(x)
    if iterations <= 0:
        return x

    def _solve(x_in: Array) -> Array:
        p0 = jnp.zeros(x_in.shape + (2,), dtype=x_in.dtype)

        def body(_, p):
            div_p = _divergence_last2(p)
            grad = _forward_gradient_last2(div_p - x_in / (weight_f + 1e-12))
            p_new = p + tau * grad
            norm = jnp.linalg.norm(p_new, axis=-1, keepdims=True)
            denom = jnp.maximum(1.0, norm)
            return p_new / denom

        p = jax.lax.fori_loop(0, iterations, body, p0)
        return x_in - weight_f * _divergence_last2(p)

    return jax.lax.cond(weight_f <= 0.0, lambda _: x, lambda _: _solve(x), operand=None)


def regrid_nearest(
    samples: Array,
    trajectory: Array,
    grid_shape: tuple[int, ...],
    *,
    density: Array | None = None,
) -> tuple[Array, Array]:
    """Nearest-neighbor regridding for non-Cartesian samples.

    Parameters
    - ``samples``: shape ``(..., num_samples)``.
    - ``trajectory``: shape ``(num_samples, D)`` with columns in cycles/pixel,
      range ~[-0.5, 0.5).
    - ``grid_shape``: output Cartesian grid shape.
    - ``density``: optional weights shape ``(num_samples,)``.

    Returns
    - ``(grid, weights)`` where ``grid`` has shape ``(..., *grid_shape)`` and ``weights`` has shape ``grid_shape``.
    """

    samples = jnp.asarray(samples)
    traj = jnp.asarray(trajectory)
    shape = tuple(int(v) for v in grid_shape)
    if traj.ndim != 2 or traj.shape[1] != len(shape) or traj.shape[1] not in (2, 3):
        raise ValueError("trajectory must have shape (num_samples, len(grid_shape)) for 2D or 3D regridding")
    if any(v <= 0 for v in shape):
        raise ValueError("grid_shape must be positive")
    m = int(traj.shape[0])
    if samples.shape[-1] != m:
        raise ValueError("samples last dimension must match trajectory length")

    w = jnp.ones((m,), dtype=jnp.float32) if density is None else jnp.asarray(density, dtype=jnp.float32)
    if w.shape != (m,):
        raise ValueError("density must have shape (num_samples,)")

    coords = (traj + 0.5) * jnp.asarray(shape, dtype=jnp.float32)
    coords = jnp.floor(coords).astype(jnp.int32)
    flat_idx = jnp.zeros((m,), dtype=jnp.int32)
    stride = 1
    for axis in range(len(shape) - 1, -1, -1):
        idx = jnp.clip(coords[:, axis], 0, shape[axis] - 1)
        flat_idx = flat_idx + idx * int(stride)
        stride *= int(shape[axis])

    leading_shape = samples.shape[:-1]
    leading = int(np.prod(leading_shape)) if leading_shape else 1
    samples_flat = samples.reshape((leading, m))
    grid_size = int(np.prod(shape))
    grid_flat = jnp.zeros((leading, grid_size), dtype=samples.dtype)
    grid_flat = grid_flat.at[:, flat_idx].add(samples_flat * w[None, :])

    weights_flat = jnp.zeros((grid_size,), dtype=jnp.float32)
    weights_flat = weights_flat.at[flat_idx].add(w)

    grid = grid_flat.reshape(leading_shape + shape)
    weights = weights_flat.reshape(shape)
    denom = jnp.where(weights == 0, 1.0, weights)
    grid = grid / denom
    return grid, weights


def reconstruct_regrid(
    kspace_samples: Array | np.ndarray,
    trajectory: Array | np.ndarray,
    *,
    spatial_shape: tuple[int, ...],
    sensitivities: Array | np.ndarray | None = None,
    density: Array | np.ndarray | None = None,
    grid_shape: tuple[int, ...] | None = None,
    crop_offset: tuple[int, ...] | None = None,
    phase_correction: Literal["none", "checkerboard"] | None = None,
    apodization_correction: RegridApodizationCorrection | None = None,
    image_mask: str | Array | np.ndarray | None = None,
    coil_combine: Literal["sum", "rss", "sense"] = "sense",
    magnitude: bool = False,
    axes: Sequence[int] | None = None,
) -> Array:
    """Quick non-Cartesian reconstruction via regridding + IFFT (not iterative).

    If ``density`` is omitted, a simple radial density compensation (:math:`|k|`) normalized
    to sum to the number of spatial pixels is applied. Pass
    ``density=np.ones(num_samples)`` to disable.

    Optional arguments expose common vendor-style gridding post-processing without
    changing the default MRax path:

    - ``grid_shape`` can reconstruct on an oversampled grid before cropping to
      ``spatial_shape``.
    - ``phase_correction="checkerboard"`` applies a :math:`\\pi` phase alternation
      on all spatial image axes.
    - ``apodization_correction`` applies a separable Kaiser-Bessel deapodization.
    - ``image_mask="ellipse"`` masks the final image support.
    - ``coil_combine="rss"`` uses root-sum-of-squares coil combination.
    """

    y = jnp.asarray(kspace_samples)
    traj = jnp.asarray(trajectory)
    output_shape = tuple(int(v) for v in spatial_shape)
    spatial_ndim = len(output_shape)
    if spatial_ndim not in (2, 3):
        raise ValueError("reconstruct_regrid supports 2D or 3D spatial shapes")
    if traj.ndim != 2 or int(traj.shape[1]) != spatial_ndim:
        raise ValueError("trajectory dimensionality must match spatial_shape")
    sens = None if sensitivities is None else jnp.asarray(sensitivities)
    combine = str(coil_combine).lower()
    if combine not in ("sum", "rss", "sense"):
        raise ValueError("coil_combine must be one of 'sum', 'rss', or 'sense'")
    if combine == "sense" and sens is None:
        combine = "sum"
    phase = "none" if phase_correction is None else str(phase_correction).lower()
    if phase not in ("none", "checkerboard"):
        raise ValueError("phase_correction must be None, 'none', or 'checkerboard'")

    if y.ndim == 1:
        y = y[None, :]
    if y.ndim == 2 and sens is None:
        # Treat as (batch, num_samples) for single-coil.
        y = y[:, None, :]
    if y.ndim == 2 and sens is not None:
        # Treat as (coils, num_samples).
        y = y[None, :, :]

    if y.shape[-1] != traj.shape[0]:
        raise ValueError("kspace_samples length does not match trajectory length")

    dens = (
        _default_density_from_trajectory(traj, spatial_shape=spatial_shape)
        if density is None
        else jnp.asarray(density)
    )
    target_grid_shape = tuple(int(v) for v in (grid_shape or output_shape))
    transform_axes = tuple(range(-spatial_ndim, 0)) if axes is None else tuple(int(ax) for ax in axes)
    grid, weights = regrid_nearest(y, traj, target_grid_shape, density=dens)
    coil_images = ifftnc(grid, axes=transform_axes)
    if phase == "checkerboard":
        coil_images = coil_images * _checkerboard(target_grid_shape).reshape(
            (1,) * (coil_images.ndim - spatial_ndim) + target_grid_shape
        )
    if target_grid_shape != output_shape:
        coil_images = _center_crop_lastn(coil_images, output_shape, crop_offset)
    if apodization_correction is not None:
        coil_images = _apply_regrid_apodization_correction(coil_images, apodization_correction, output_shape)
    coil_images = _apply_image_mask(coil_images, image_mask, output_shape)

    if combine == "rss":
        combined = jnp.sqrt(jnp.sum(jnp.abs(coil_images) ** 2, axis=-(spatial_ndim + 1)))
    elif combine == "sense":
        combined = _coil_combine_sense(coil_images, sens, coil_axis=-(spatial_ndim + 1))
    else:
        combined = jnp.sum(coil_images, axis=-(spatial_ndim + 1))
    if magnitude:
        combined = jnp.abs(combined)
    if kspace_samples is not None and np.asarray(kspace_samples).ndim in (1, 2) and combined.shape[0] == 1:
        combined = combined[0]
    return combined


@dataclass(frozen=True)
class MRIEncodingOperator:
    """Encoding operator A for multi-coil MRI: y = M F S x (Cartesian) or NUFFT (non-Cartesian)."""

    spatial_shape: tuple[int, int]
    sensitivities: Array | None = None  # (coils, *spatial)
    mask: Array | None = None  # (*spatial,) for Cartesian
    trajectory: Array | None = None  # (num_samples, 2) with (kx, ky) in cycles/pixel for non-Cartesian
    density: Array | None = None  # (num_samples,) optional weights for non-Cartesian
    axes: tuple[int, int] = (-2, -1)
    nufft_eps: float = 1e-6

    @property
    def is_non_cartesian(self) -> bool:
        return self.trajectory is not None

    @property
    def coil_count(self) -> int:
        return 1 if self.sensitivities is None else int(self.sensitivities.shape[0])

    def _apply_mask(self, kspace: Array) -> Array:
        if self.mask is None:
            return kspace
        return kspace * self.mask

    def forward(self, image: Array) -> Array:
        """Apply A(image) -> k-space (shape: ``batch..., coils, *spatial`` or ``batch..., coils, num_samples``)."""

        x = jnp.asarray(image)
        if x.shape[-2:] != self.spatial_shape:
            raise ValueError(
                f"image spatial shape {tuple(x.shape[-2:])} does not match operator {self.spatial_shape}"
            )

        sens = None if self.sensitivities is None else jnp.asarray(self.sensitivities)
        coil_count = 1 if sens is None else int(sens.shape[0])
        input_has_coils = x.ndim >= 3 and int(x.shape[-3]) == coil_count
        x_coils, _ = _ensure_coil_axis(x, has_coils=input_has_coils, spatial_ndim=2, coil_axis=-3)
        if sens is not None and not input_has_coils:
            x_coils = x_coils * sens  # broadcast over batch dims

        if not self.is_non_cartesian:
            kspace = fftnc(x_coils, axes=self.axes)
            return self._apply_mask(kspace)

        if jax_finufft is None:
            raise ImportError(
                "Non-Cartesian reconstruction requires 'jax-finufft' (jax_finufft). "
                "Install it with `pip install 'mrax[nufft]'`."
            )

        nufft_norm = float(np.sqrt(np.prod(self.spatial_shape)))
        traj = jnp.asarray(self.trajectory)
        if traj.ndim != 2 or traj.shape[1] != 2:
            raise ValueError("trajectory must have shape (num_samples, 2) for 2D NUFFT")
        omega = (2.0 * jnp.pi) * traj.astype(jnp.float32)
        kx = omega[:, 0]
        ky = omega[:, 1]

        # Collapse batch dims for FINUFFT and keep coil axis as broadcast dimension.
        batch_shape = x_coils.shape[:-3]
        batch = int(np.prod(batch_shape)) if batch_shape else 1
        x_flat = x_coils.reshape((batch, x_coils.shape[-3]) + self.spatial_shape).astype(jnp.complex64)
        kx_b = jnp.broadcast_to(kx[None, :], (batch, kx.shape[0]))
        ky_b = jnp.broadcast_to(ky[None, :], (batch, ky.shape[0]))
        samples = jax_finufft.nufft2(x_flat, kx_b, ky_b, iflag=-1, eps=float(self.nufft_eps))
        out = samples.reshape(batch_shape + (x_coils.shape[-3], kx.shape[0]))
        if nufft_norm > 0:
            out = out / nufft_norm
        return out

    def adjoint(self, kspace: Array) -> Array:
        """Apply A^H(kspace) -> combined image (shape: ``batch..., *spatial``)."""

        y = jnp.asarray(kspace)

        if not self.is_non_cartesian:
            has_coils = _has_coils_cartesian(y, self.spatial_shape, self.sensitivities)
            y_coils, added = _ensure_coil_axis(y, has_coils=has_coils, spatial_ndim=2, coil_axis=-3)
            y_coils = self._apply_mask(y_coils)
            coil_images = ifftnc(y_coils, axes=self.axes)
            combined = _coil_combine_sense(coil_images, self.sensitivities, coil_axis=-3)
            return combined

        if jax_finufft is None:
            raise ImportError(
                "Non-Cartesian reconstruction requires 'jax-finufft' (jax_finufft). "
                "Install it with `pip install 'mrax[nufft]'`."
            )
        nufft_norm = float(np.sqrt(np.prod(self.spatial_shape)))
        traj = jnp.asarray(self.trajectory)
        if traj.ndim != 2 or traj.shape[1] != 2:
            raise ValueError("trajectory must have shape (num_samples, 2) for 2D NUFFT")

        omega = (2.0 * jnp.pi) * traj.astype(jnp.float32)
        kx = omega[:, 0]
        ky = omega[:, 1]

        # Normalize k-space shape to (..., coils, num_samples) when possible.
        original_ndim = y.ndim
        if y.ndim == 1:
            y_coils = y[None, None, :]
        elif y.ndim == 2 and self.sensitivities is None:
            # Ambiguous case: treat as (batch, num_samples) for single-coil inputs.
            y_coils = y[:, None, :]
        elif y.ndim == 2:
            # With sensitivities, treat the leading dimension as coils.
            y_coils = y[None, :, :]
        else:
            y_coils = y

        if y_coils.shape[-1] != kx.shape[0]:
            raise ValueError("kspace sample count does not match trajectory length")
        if self.density is not None:
            w = jnp.asarray(self.density).astype(jnp.float32)
            if w.shape != (kx.shape[0],):
                raise ValueError("density must have shape (num_samples,)")
            y_coils = y_coils * w

        batch_shape = y_coils.shape[:-2]
        batch = int(np.prod(batch_shape)) if batch_shape else 1
        y_flat = y_coils.reshape((batch, y_coils.shape[-2], y_coils.shape[-1])).astype(jnp.complex64)
        kx_b = jnp.broadcast_to(kx[None, :], (batch, kx.shape[0]))
        ky_b = jnp.broadcast_to(ky[None, :], (batch, ky.shape[0]))
        coil_images = jax_finufft.nufft1(
            self.spatial_shape, y_flat, kx_b, ky_b, iflag=1, eps=float(self.nufft_eps)
        )
        if nufft_norm > 0:
            coil_images = coil_images / nufft_norm
        coil_images = coil_images.reshape(batch_shape + (y_coils.shape[-2],) + self.spatial_shape)
        combined = _coil_combine_sense(coil_images, self.sensitivities, coil_axis=-3)
        if original_ndim == 1 or (original_ndim == 2 and self.sensitivities is not None):
            combined = combined[0]
        return combined


def build_operator(
    *,
    spatial_shape: tuple[int, int] | None = None,
    kspace: Array | np.ndarray | None = None,
    sensitivities: Array | np.ndarray | None = None,
    mask: Array | np.ndarray | None = None,
    trajectory: Array | np.ndarray | None = None,
    density: Array | np.ndarray | None = None,
    axes: Sequence[int] = (-2, -1),
    nufft_eps: float = 1e-6,
) -> MRIEncodingOperator:
    """Build a vendor-agnostic MRI encoding operator from generic inputs."""

    if trajectory is None:
        if spatial_shape is None:
            if kspace is None:
                raise ValueError("spatial_shape or kspace must be provided for Cartesian operators")
            spatial_shape = _infer_spatial_shape_from_kspace(kspace)
        mask_arr = _as_mask(mask, spatial_shape)
        sens_arr = None if sensitivities is None else jnp.asarray(sensitivities)
        if sens_arr is not None and tuple(sens_arr.shape[-2:]) != spatial_shape:
            raise ValueError("sensitivities must have shape (coils, *spatial_shape)")
        return MRIEncodingOperator(
            spatial_shape=spatial_shape,
            sensitivities=None if sens_arr is None else sens_arr,
            mask=mask_arr,
            trajectory=None,
            density=None,
            axes=tuple(int(ax) for ax in axes),
            nufft_eps=float(nufft_eps),
        )

    if spatial_shape is None:
        raise ValueError("spatial_shape must be provided for non-Cartesian operators")
    traj_arr = jnp.asarray(trajectory)
    dens_arr = (
        jnp.asarray(_default_density_from_trajectory(trajectory, spatial_shape=spatial_shape))
        if density is None
        else jnp.asarray(density)
    )
    sens_arr = None if sensitivities is None else jnp.asarray(sensitivities)
    if sens_arr is not None and tuple(sens_arr.shape[-2:]) != spatial_shape:
        raise ValueError("sensitivities must have shape (coils, *spatial_shape)")
    return MRIEncodingOperator(
        spatial_shape=spatial_shape,
        sensitivities=None if sens_arr is None else sens_arr,
        mask=None,
        trajectory=traj_arr,
        density=dens_arr,
        axes=tuple(int(ax) for ax in axes),
        nufft_eps=float(nufft_eps),
    )


def reconstruct_ifft(
    kspace: Array | np.ndarray,
    *,
    sensitivities: Array | np.ndarray | None = None,
    mask: Array | np.ndarray | None = None,
    axes: Sequence[int] = (-2, -1),
) -> Array:
    """Direct (zero-filled) reconstruction for Cartesian k-space via :math:`A^H y`."""

    op = build_operator(kspace=kspace, sensitivities=sensitivities, mask=mask, axes=axes)
    return op.adjoint(jnp.asarray(kspace))


def reconstruct_variational(
    kspace: Array | np.ndarray,
    *,
    spatial_shape: tuple[int, int] | None = None,
    sensitivities: Array | np.ndarray | None = None,
    mask: Array | np.ndarray | None = None,
    trajectory: Array | np.ndarray | None = None,
    density: Array | np.ndarray | None = None,
    frame_shifts: Array | np.ndarray | None = None,
    frame_axis: int = 0,
    shift_order: int = 1,
    shift_mode: str = "constant",
    shift_cval: float = 0.0,
    optimizer_name: str = "fista",
    step_size: float = 0.5,
    num_steps: int = 50,
    regularizer: Literal["none", "l2", "l1", "tv"] = "tv",
    regularizer_weight: float = 0.01,
    tv_iterations: int = 30,
    nufft_eps: float = 1e-6,
    initial_image: Array | np.ndarray | None = None,
    return_history: bool = False,
    grad_clip_norm: float | None = None,
) -> tuple[Array, list[float]] | Array:
    """Solve a variational reconstruction with Optax/proximal variants.

    Minimizes: :math:`\\tfrac12 \\|\\|A x - y\\|\\|_2^2 + R(x)`

    - Cartesian operators use centered FFT and optional sampling masks.
    - Non-Cartesian operators use FINUFFT (jax_finufft) when trajectory is provided.
    - TV regularization uses a proper proximal operator (Chambolle) when
      ``optimizer_name`` is ``"fista"`` or ``"split_bregman"``; otherwise TV is treated
      as a differentiable penalty.
    - If ``density`` is omitted for non-Cartesian data, a simple radial density
      compensation (:math:`|k|`) normalized so the weights sum to the number of spatial
      pixels is applied to the data term. Pass ``density=np.ones(num_samples)``
      to disable.
    - If ``frame_shifts`` is provided, the data term is built from per-frame shifted
      images: ``y_i = A(shift(x, shift_i))``. Shifts are specified in index units
      (row, col). In this mode, ``kspace`` must include a frame axis.
    - For proximal solvers, the effective shrinkage threshold is
      ``regularizer_weight * step_size``.
    - Optional gradient clipping helps stabilize long iterative runs.
    """

    y = jnp.asarray(kspace)
    op = build_operator(
        spatial_shape=spatial_shape,
        kspace=kspace,
        sensitivities=sensitivities,
        mask=mask,
        trajectory=trajectory,
        density=density,
        nufft_eps=nufft_eps,
    )
    shifts_arr = None
    if frame_shifts is not None:
        shifts_arr = jnp.asarray(frame_shifts, dtype=jnp.float32)
        if shifts_arr.ndim != 2 or shifts_arr.shape[1] != 2:
            raise ValueError("frame_shifts must have shape (frames, 2)")
        y = jnp.moveaxis(y, int(frame_axis), 0)
        frame_count = int(y.shape[0])
        if int(shifts_arr.shape[0]) != frame_count:
            raise ValueError("frame_shifts length must match kspace frame axis")
        y = _normalize_kspace_for_model(
            y,
            frame_count=frame_count,
            coil_count=op.coil_count,
            is_non_cartesian=op.is_non_cartesian,
            sensitivities=op.sensitivities,
        )
        if not op.is_non_cartesian:
            y = op._apply_mask(y)
    else:
        if not op.is_non_cartesian:
            y = op._apply_mask(y)

    if initial_image is None:
        if shifts_arr is None:
            x0 = op.adjoint(y)
        else:
            def _adjoint_one(y_frame, shift):
                adj = op.adjoint(y_frame)
                return _shift_image_last2(
                    adj,
                    -shift,
                    order=shift_order,
                    mode=shift_mode,
                    cval=shift_cval,
                )

            x0 = jnp.mean(jax.vmap(_adjoint_one)(y, shifts_arr), axis=0)
    else:
        x0 = jnp.asarray(initial_image)
    if x0.shape[-2:] != op.spatial_shape:
        raise ValueError("initial_image does not match spatial_shape")

    reg = regularizer.lower()
    if reg not in ("none", "l2", "l1", "tv"):
        raise ValueError(f"Unknown regularizer '{regularizer}'")
    reg_weight = float(regularizer_weight)

    def data_loss(x: Array) -> Array:
        if shifts_arr is None:
            pred = op.forward(x)
        else:
            def _forward_one(shift):
                shifted = _shift_image_last2(
                    x,
                    shift,
                    order=shift_order,
                    mode=shift_mode,
                    cval=shift_cval,
                )
                return op.forward(shifted)

            pred = jax.vmap(_forward_one)(shifts_arr)
        diff = pred - y
        if op.is_non_cartesian and op.density is not None:
            w = jnp.asarray(op.density, dtype=jnp.float32)
            diff = diff * jnp.sqrt(w)
        return 0.5 * jnp.mean(jnp.abs(diff) ** 2)

    def smooth_reg(x: Array) -> Array:
        if reg == "none":
            return jnp.array(0.0, dtype=jnp.float32)
        if reg == "l2":
            return reg_weight * jnp.mean(jnp.abs(x) ** 2)
        if reg == "tv":
            # Isotropic TV penalty as a differentiable fallback.
            grad = _forward_gradient_last2(x)
            norm = jnp.sqrt(jnp.sum(jnp.abs(grad) ** 2, axis=-1) + 1e-8)
            return reg_weight * jnp.mean(norm)
        if reg == "l1":
            return reg_weight * jnp.mean(jnp.abs(x))
        raise AssertionError("unreachable")

    opt_name = optimizer_name.lower()
    use_prox = opt_name in ("fista", "split_bregman", "bregman") and reg in ("l1", "tv") and reg_weight > 0
    if use_prox and reg == "tv":
        def prox(arr, threshold):
            return prox_tv_chambolle(arr, threshold, iterations=tv_iterations)
        loss_fn = data_loss
        opt = make_optimizer(
            opt_name,
            step_size,
            regularizer_weight=reg_weight,
            regularizer_prox=prox,
            base_optimizer_factory=lambda lr: optax.sgd(lr),
        )
    elif use_prox and reg == "l1":
        from mrax.opt.ops import soft_threshold

        loss_fn = data_loss
        opt = make_optimizer(
            opt_name,
            step_size,
            regularizer_weight=reg_weight,
            regularizer_prox=soft_threshold,
            base_optimizer_factory=lambda lr: optax.sgd(lr),
        )
    else:
        def loss_fn(x):
            return data_loss(x) + smooth_reg(x)
        opt = make_optimizer(optimizer_name, step_size)

    opt_state = opt.init(x0)
    use_grad_clip = grad_clip_norm is not None and grad_clip_norm > 0
    clip_norm = float(grad_clip_norm) if use_grad_clip else 0.0

    @jax.jit
    def step(x, state):
        loss, grads = jax.value_and_grad(loss_fn)(x)
        if use_grad_clip:
            grad_norm = optax.global_norm(grads)
            scale = jnp.minimum(1.0, clip_norm / (grad_norm + 1e-8))
            grads = jax.tree_util.tree_map(lambda g: g * scale, grads)
        updates, state = opt.update(grads, state, x)
        x = optax.apply_updates(x, updates)
        return x, state, loss

    history: list[float] = []
    x = x0
    for _ in range(int(num_steps)):
        x, opt_state, loss = step(x, opt_state)
        if return_history:
            history.append(float(loss))

    if return_history:
        return x, history
    return x


def _infer_sampling_mask_cartesian(kspace: Array) -> Array:
    """Infer a boolean sampling mask from (potentially batched) Cartesian k-space."""

    y = jnp.asarray(kspace)
    if y.ndim < 2:
        raise ValueError("kspace must be at least 2D for Cartesian reconstruction")
    spatial_axes = (-2, -1)
    reduce_axes = tuple(range(y.ndim - len(spatial_axes)))
    mag = jnp.abs(y)
    if reduce_axes:
        mag = jnp.any(mag > 0, axis=reduce_axes)
    return mag > 0


def _normalize_kspace_for_model(
    kspace: Array | np.ndarray,
    *,
    frame_count: int,
    coil_count: int,
    is_non_cartesian: bool,
    sensitivities: Array | None,
) -> Array:
    """Return k-space normalized to (frames, coils, ...)."""

    y = jnp.asarray(kspace)
    frame_count = int(frame_count)
    coil_count = int(coil_count)

    if is_non_cartesian:
        if y.ndim == 1:
            if frame_count > 1:
                raise ValueError("kspace must include a frame axis matching experiment.frame_count")
            y = y[None, None, :]
        elif y.ndim == 2:
            if frame_count > 1:
                if int(y.shape[0]) != frame_count:
                    raise ValueError("kspace frame dimension does not match experiment.frame_count")
                y = y[:, None, :]
            else:
                if sensitivities is not None and coil_count > 1 and int(y.shape[0]) == coil_count:
                    y = y[None, :, :]
                elif int(y.shape[0]) == frame_count:
                    y = y[:, None, :]
                else:
                    raise ValueError("kspace shape does not match expected frame/coil layout for non-Cartesian data")
        elif y.ndim == 3:
            if int(y.shape[0]) == frame_count and int(y.shape[1]) == coil_count:
                y = y
            elif int(y.shape[1]) == frame_count and int(y.shape[0]) == coil_count:
                y = jnp.swapaxes(y, 0, 1)
            else:
                raise ValueError("kspace shape does not match expected frame/coil layout for non-Cartesian data")
        else:
            raise ValueError(
                "non-Cartesian kspace must have shape (num_samples,), (frames, num_samples), "
                "(coils, num_samples), or (frames, coils, num_samples)"
            )
    else:
        if y.ndim == 2:
            if frame_count > 1:
                raise ValueError("kspace must include a frame axis matching experiment.frame_count")
            y = y[None, None, :, :]
        elif y.ndim == 3:
            if frame_count > 1:
                if int(y.shape[0]) != frame_count:
                    raise ValueError("kspace frame dimension does not match experiment.frame_count")
                y = y[:, None, :, :]
            else:
                if sensitivities is not None and coil_count > 1 and int(y.shape[0]) == coil_count:
                    y = y[None, :, :, :]
                elif int(y.shape[0]) == frame_count:
                    y = y[:, None, :, :]
                else:
                    raise ValueError("kspace shape does not match expected frame/coil layout for Cartesian data")
        elif y.ndim == 4:
            if int(y.shape[0]) == frame_count:
                y = y
            elif int(y.shape[1]) == frame_count and int(y.shape[0]) == coil_count:
                y = jnp.swapaxes(y, 0, 1)
            else:
                raise ValueError("kspace shape does not match expected frame/coil layout for Cartesian data")
        else:
            raise ValueError(
                "Cartesian kspace must have shape (Ny, Nx), (frames, Ny, Nx), (coils, Ny, Nx), "
                "or (frames, coils, Ny, Nx)"
            )

    if int(y.shape[0]) != frame_count:
        raise ValueError("kspace frame dimension does not match experiment.frame_count")
    if int(y.shape[1]) != coil_count:
        raise ValueError("kspace coil dimension does not match sensitivities")
    return y


def _align_kspace_global_phase_by_frame(
    y: Array,
    *,
    expected_frames: int,
    mode: str,
    return_rotation: bool = False,
) -> Array | tuple[Array, Array | None]:
    """Apply a global (scalar) phase correction to each frame to align to a reference frame.

    This is useful when fitting magnitude-only models to complex k-space that may have
    frame-dependent receiver phase flips (e.g., due to refocusing/phase cycling).
    """

    y = jnp.asarray(y)
    mode_norm = str(mode or "none").lower()
    if mode_norm in ("none", "off", "false", "0"):
        return (y, None) if return_rotation else y
    if y.ndim < 2:
        return (y, None) if return_rotation else y
    if expected_frames <= 1 or int(y.shape[0]) != int(expected_frames):
        return (y, None) if return_rotation else y
    if not jnp.issubdtype(y.dtype, jnp.complexfloating):
        return (y, None) if return_rotation else y

    if mode_norm in ("auto", "align", "global", "energy", "max_energy", "max"):
        reduce_axes = tuple(range(1, y.ndim))
        energies = jnp.sum(jnp.abs(y) ** 2, axis=reduce_axes)
        ref_idx = int(jnp.argmax(energies))
    elif mode_norm in ("first", "frame0", "ref", "reference"):
        ref_idx = 0
    else:
        raise ValueError("frame_phase_correction must be one of: none, auto/align, first")

    y_ref = y[ref_idx]
    frames = int(expected_frames)
    y_flat = y.reshape((frames, -1))
    ref_flat = y_ref.reshape((-1,))
    inner = jnp.sum(jnp.conj(ref_flat)[None, :] * y_flat, axis=1)
    phis = jnp.angle(inner)
    rot = jnp.exp(-1j * phis).reshape((frames,) + (1,) * (y.ndim - 1))
    aligned = y * rot
    return (aligned, rot) if return_rotation else aligned


def reconstruct_model_based(
    kspace: Array | np.ndarray,
    *,
    experiment: Experiment,
    model: Model,
    spatial_shape: tuple[int, int] | None = None,
    sensitivities: Array | np.ndarray | None = None,
    mask: Array | np.ndarray | None = None,
    completion_kspace: Array | np.ndarray | None = None,
    completion_weight: float = 0.0,
    trajectory: Array | np.ndarray | None = None,
    density: Array | np.ndarray | None = None,
    optimizer_name: str = "adam",
    step_size: float = 0.02,
    num_steps: int = 200,
    regularizer_weight=0.0,
    regularizer_prox=None,
    optimizer_base_factory=None,
    box_constraints=None,
    initial_params: dict | None = None,
    regularizer_configs: dict[str, list[dict]] | None = None,
    regularizer_type: str | dict[str, str] | None = None,
    regularizer_weights: dict[str, float] | None = None,
    roi_mask: object | None = None,
    roi_mask_dims: tuple[str, ...] | None = None,
    study_globals: dict[str, Quantity] | None = None,
    frame_mask: object | None = None,
    auto_mask_nan_observables: bool = True,
    nufft_eps: float = 1e-6,
    reference_image: Array | np.ndarray | Quantity | None = None,
    reference_kspace: Array | np.ndarray | None = None,
    reference_mask: Array | np.ndarray | None = None,
    reference_weight: float = 1.0,
    initial_reference_image: Array | np.ndarray | None = None,
    frame_phase_correction: str | None = "none",
    return_history: bool = False,
    return_model_callable: bool = False,
) -> (
    tuple[dict[str, Quantity], list[float], Callable[[dict, Array | None, dict[str, Any] | None], Array]]
    | tuple[dict[str, Quantity], list[float]]
    | dict[str, Quantity]
):
    """Directly reconstruct parameter maps from k-space via a chained model+encoding operator.

    Minimizes: :math:`\\tfrac12 \\|\\|A(M(\\theta)) - y\\|\\|_2^2 + R(\\theta)`

    Notes
    -----
    - The signal model comes from :class:`mrax.data_struct.model.Model` and is treated as a
      (generally non-linear) operator mapping parameter fields -> image series.
    - The encoding operator :math:`A` is the same as in :func:`reconstruct_variational` (FFT/NUFFT,
      optional sampling mask, optional coil sensitivities).
    - ``roi_mask`` weights regularizers and masks output fields; when provided, the model
      prediction is also zeroed outside the mask to prevent NaN observables from propagating
      through the k-space data term.
    - ``frame_mask`` (or ``auto_mask_nan_observables=True``) can drop frames with missing
      per-frame observables; masked frames do not contribute to the data term.
    - ``reference_image`` / ``reference_kspace`` can be used to model acquisitions where the
      signal model predicts a *relative* image series (e.g., Z-spectra). In that case the
      encoded prediction becomes :math:`A(reference\\_image \\cdot M(\\theta))`. When
      ``reference_kspace`` is provided, the reference image is reconstructed jointly by adding
      a second data term :math:`reference\\_weight \\|\\|A_{ref}(reference\\_image) - reference\\_kspace\\|\\|^2`.
    - ``completion_kspace`` / ``completion_weight`` can add an optional prior on the *missing*
      k-space samples (i.e., outside ``mask``) by penalizing the difference to a filled
      k-space estimate (useful for partial Fourier / POCS-style completion). To match the
      weighting of a full-data mean-squared-error objective, set
      ``completion_weight=(1-f)/f`` where :math:`f=\\mathrm{mean}(mask)`.
    - ``frame_phase_correction`` can apply a global phase alignment per frame (recommended
      for magnitude-only models when raw k-space contains frame-dependent phase flips).
    """

    warn_if_fit_requires_model_override(experiment, model)

    # Mirror run_fit behavior: auto-attach globals if present on the experiment.
    if (
        study_globals is None
        and getattr(model, "_mrax_study_globals", None) is None
    ):
        exp_globals = getattr(experiment, "study_globals", None)
        if exp_globals:
            model._mrax_study_globals = exp_globals
            study_globals = exp_globals

    full_spatial_shape, spatial_dim_names = _extract_spatial_metadata(experiment)
    if spatial_shape is None:
        spatial_shape = tuple(int(v) for v in full_spatial_shape[:2])  # type: ignore[assignment]
    if len(spatial_shape) != 2:
        raise ValueError("reconstruct_model_based currently supports 2D spatial_shape only")

    field_shape, field_dim_names, _extra_dim_names = _field_metadata(experiment, model=model)
    constant_dims = _normalize_constant_dims(model.parameter_constant_dims, field_dim_names, field_shape)
    mask_data, mask_fields = _prepare_mask_arrays(
        roi_mask,
        roi_mask_dims,
        experiment,
        field_shape,
        field_dim_names,
        spatial_dim_names,
    )

    params, unpack = model.parametrization_factory(field_shape)
    params = _collapse_params_for_constant_dims(params, constant_dims)

    reg_terms = model.make_regularizer_terms(
        field_shape,
        regularizer_configs=regularizer_configs,
        regularizer_type=regularizer_type,
        reg_weights=regularizer_weights,
    )

    # Observables: merge study globals (if any) with experiment observables and unwrap to arrays for JIT.
    observables = require_quantity_dict(experiment.observables, "Observable")
    inferred_globals = (
        study_globals
        or getattr(model, "_mrax_study_globals", None)
        or getattr(experiment, "study_globals", None)
    )
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

    if obs_data_mask is not None:
        mask_data = obs_data_mask if mask_data is None else (mask_data * obs_data_mask)
        obs_field_mask = _reduce_mask_to_field(obs_data_mask, experiment, field_dim_names)
        if obs_field_mask is not None:
            mask_fields = obs_field_mask if mask_fields is None else (mask_fields * obs_field_mask)
    support_mask = None
    if mask_data is not None:
        support_mask = _flatten_to_frames(jnp.asarray(mask_data), experiment) > 0

    frame_weights = _prepare_frame_mask(frame_mask, experiment)
    if auto_mask_nan_observables and auto_frame_mask is not None:
        frame_weights = auto_frame_mask if frame_weights is None else (frame_weights * auto_frame_mask)
    frame_weights = jnp.asarray(frame_weights) if frame_weights is not None else None

    y = jnp.asarray(kspace)
    if trajectory is None and mask is None:
        mask = _infer_sampling_mask_cartesian(y)

    op = build_operator(
        spatial_shape=spatial_shape,
        sensitivities=sensitivities,
        mask=mask,
        trajectory=trajectory,
        density=density,
        nufft_eps=nufft_eps,
    )
    frame_count = int(experiment.frame_count)
    y_coils = _normalize_kspace_for_model(
        y,
        frame_count=frame_count,
        coil_count=op.coil_count,
        is_non_cartesian=op.is_non_cartesian,
        sensitivities=op.sensitivities,
    )
    if op.is_non_cartesian:
        traj_len = int(np.asarray(op.trajectory).shape[0]) if op.trajectory is not None else 0
        if int(y_coils.shape[-1]) != traj_len:
            raise ValueError("kspace sample count does not match trajectory length")
    else:
        if tuple(int(v) for v in y_coils.shape[-2:]) != spatial_shape:
            raise ValueError(
                f"kspace spatial shape {tuple(int(v) for v in y_coils.shape[-2:])} does not match {spatial_shape}"
            )

    data_finite = jnp.isfinite(y_coils)
    y_coils = jnp.where(data_finite, y_coils, jnp.zeros((), dtype=y_coils.dtype))
    if not op.is_non_cartesian:
        y_coils = op._apply_mask(y_coils)

    use_completion = False
    op_full: MRIEncodingOperator | None = None
    completion_w = float(completion_weight)
    y_completion_coils: Array | None = None
    completion_finite: Array | None = None
    if completion_kspace is not None:
        y_completion = jnp.asarray(completion_kspace)
        if y_completion.shape != y.shape:
            raise ValueError(
                f"completion_kspace shape {tuple(int(v) for v in y_completion.shape)} does not match kspace shape {tuple(int(v) for v in y.shape)}"
            )
        y_completion_coils = _normalize_kspace_for_model(
            y_completion,
            frame_count=frame_count,
            coil_count=op.coil_count,
            is_non_cartesian=op.is_non_cartesian,
            sensitivities=op.sensitivities,
        )
        if y_completion_coils.shape != y_coils.shape:
            raise ValueError("completion_kspace does not match normalized kspace shape")
        completion_finite = jnp.isfinite(y_completion_coils)
        y_completion_coils = jnp.where(
            completion_finite, y_completion_coils, jnp.zeros((), dtype=y_completion_coils.dtype)
        )
        if completion_w > 0.0 and (not op.is_non_cartesian) and op.mask is not None:
            if not bool(jnp.all(op.mask)):
                op_full = build_operator(
                    spatial_shape=spatial_shape,
                    sensitivities=sensitivities,
                    mask=None,
                    trajectory=None,
                    density=None,
                    nufft_eps=nufft_eps,
                )
                use_completion = True

    expected_frames = experiment.frame_count if hasattr(experiment, "frame_count") else int(
        experiment.dims.get("frame", int(y_coils.shape[0]))
    )
    y_coils, phase_rot = _align_kspace_global_phase_by_frame(
        y_coils,
        expected_frames=int(expected_frames),
        mode=str(frame_phase_correction or "none"),
        return_rotation=True,
    )
    if y_completion_coils is not None and phase_rot is not None:
        y_completion_coils = y_completion_coils * phase_rot

    scale = 1.0 / jnp.maximum(_masked_rms(y_coils, data_finite), 1e-6)

    ref_fixed: Array | None = None
    if reference_image is not None and reference_kspace is None:
        ref_arr = reference_image.values if isinstance(reference_image, Quantity) else reference_image
        ref_fixed = _coerce_reference_image(ref_arr, spatial_shape)
        if ref_fixed.shape != spatial_shape:
            raise ValueError(f"reference_image shape {tuple(ref_fixed.shape)} does not match spatial_shape {spatial_shape}")

    op_ref: MRIEncodingOperator | None = None
    y_ref_coils: Array | None = None
    ref_finite: Array | None = None
    scale_ref: Array | None = None
    ref_weight = float(reference_weight)

    params_all: Any = params
    if reference_kspace is not None:
        y_ref = jnp.asarray(reference_kspace)
        if y_ref.ndim == 4 and int(y_ref.shape[0]) == 1:
            y_ref = y_ref[0]
        if y_ref.ndim not in (2, 3):
            raise ValueError("reference_kspace must have shape (Ny, Nx), (coils, Ny, Nx), or (1, coils, Ny, Nx)")
        if tuple(int(v) for v in y_ref.shape[-2:]) != spatial_shape:
            raise ValueError(
                f"reference_kspace spatial shape {tuple(int(v) for v in y_ref.shape[-2:])} does not match {spatial_shape}"
            )

        mask_ref = reference_mask
        if trajectory is None and mask_ref is None:
            mask_ref = _infer_sampling_mask_cartesian(y_ref)

        op_ref = build_operator(
            spatial_shape=spatial_shape,
            sensitivities=sensitivities,
            mask=mask_ref,
            trajectory=trajectory,
            density=density,
            nufft_eps=nufft_eps,
        )

        coil_count_ref = op_ref.coil_count
        y_ref_has_coils = y_ref.ndim >= 3 and int(y_ref.shape[-3]) == coil_count_ref
        y_ref_coils, _ = _ensure_coil_axis(y_ref, has_coils=y_ref_has_coils, spatial_ndim=2, coil_axis=-3)
        ref_finite = jnp.isfinite(y_ref_coils)
        y_ref_coils = jnp.where(ref_finite, y_ref_coils, jnp.zeros((), dtype=y_ref_coils.dtype))
        if not op_ref.is_non_cartesian:
            y_ref_coils = op_ref._apply_mask(y_ref_coils)
        scale_ref = 1.0 / jnp.maximum(_masked_rms(y_ref_coils, ref_finite), 1e-6)

        if initial_reference_image is not None:
            ref0 = _coerce_reference_image(initial_reference_image, spatial_shape)
        elif reference_image is not None:
            ref_arr = reference_image.values if isinstance(reference_image, Quantity) else reference_image
            ref0 = _coerce_reference_image(ref_arr, spatial_shape)
        else:
            ref0 = jnp.abs(op_ref.adjoint(y_ref_coils))

        if ref0.shape != spatial_shape:
            raise ValueError(
                f"initial reference image has shape {tuple(ref0.shape)} but expected {spatial_shape}"
            )

        params_all = {"model": params, "reference_image": ref0}

    model_fn = make_model_callable(
        experiment,
        model,
        inferred_globals,
        allow_raw_observables=True,
    )

    z_prev_base = observable_arrays.get("z_prev")
    z_ref_image = observable_arrays.get("z_reference_image")
    adjust_z_prev = z_prev_base is not None and z_ref_image is not None
    observable_arrays_fixed = observable_arrays
    if adjust_z_prev and reference_kspace is None and ref_fixed is not None:
        ref_safe = ref_fixed.reshape((1,) + ref_fixed.shape) + 1e-6
        z_prev_adj = jnp.asarray(z_prev_base) * z_ref_image.reshape((1,) + z_ref_image.shape) / ref_safe
        observable_arrays_fixed = dict(observable_arrays)
        observable_arrays_fixed["z_prev"] = z_prev_adj

    def model_apply(fields: dict, ref_for_zprev: Array | None) -> Array:
        obs = observable_arrays_fixed
        if adjust_z_prev and reference_kspace is not None and ref_for_zprev is not None:
            ref_safe = ref_for_zprev.reshape((1,) + ref_for_zprev.shape) + 1e-6
            z_prev_adj = jnp.asarray(z_prev_base) * z_ref_image.reshape((1,) + z_ref_image.shape) / ref_safe
            obs = dict(observable_arrays)
            obs["z_prev"] = z_prev_adj
        return jnp.asarray(model_fn(fields, offsets_array, obs))

    def loss_fn(p):
        if reference_kspace is None:
            p_model = p
            ref = ref_fixed
        else:
            p_model = p["model"]
            ref = p["reference_image"]

        fields = unpack(p_model)
        pred_img = model_apply(fields, ref)
        pred_img = _align_predicted(jnp.asarray(pred_img), experiment)
        if support_mask is not None:
            pred_img = jnp.where(support_mask, pred_img, jnp.zeros((), dtype=pred_img.dtype))
        if ref is not None:
            pred_img = pred_img * ref
        if support_mask is not None:
            pred_img = jnp.where(support_mask, pred_img, jnp.zeros((), dtype=pred_img.dtype))
        if use_completion:
            assert op_full is not None and y_completion_coils is not None
            pred_full = op_full.forward(pred_img)
            pred_k = op._apply_mask(pred_full)
            diff = (pred_k - y_coils) * scale
        else:
            pred_k = op.forward(pred_img)
            diff = (pred_k - y_coils) * scale

        w_frame = None
        if frame_weights is not None:
            w_frame = frame_weights.reshape((int(diff.shape[0]),) + (1,) * (diff.ndim - 1))
            diff = jnp.where(w_frame > 0, diff, 0.0)

        weights = None
        if not op.is_non_cartesian and op.mask is not None:
            weights = op.mask
        if w_frame is not None:
            weights = w_frame if weights is None else (weights * w_frame)
        weights = data_finite if weights is None else (weights * data_finite)

        if weights is not None:
            num = jnp.sum(jnp.abs(diff) ** 2 * weights)
            denom = (jnp.sum(weights) + 1e-8).astype(jnp.float32)
            misfit = 0.5 * num / denom
        else:
            misfit = 0.5 * jnp.mean(jnp.abs(diff) ** 2)

        if use_completion:
            assert op.mask is not None and op_full is not None and y_completion_coils is not None
            inv_mask = (1.0 - op.mask.astype(jnp.float32)).reshape(op.mask.shape)
            diff_missing = (pred_full - y_completion_coils) * scale
            if w_frame is not None:
                diff_missing = jnp.where(w_frame > 0, diff_missing, 0.0)
                weight_missing = inv_mask * w_frame
            else:
                weight_missing = inv_mask
            if completion_finite is not None:
                weight_missing = weight_missing * completion_finite
            num_missing = jnp.sum(jnp.abs(diff_missing) ** 2 * weight_missing)
            denom_missing = (jnp.sum(weight_missing) + 1e-8).astype(jnp.float32)
            misfit = misfit + completion_w * (0.5 * num_missing / denom_missing)

        reg_term: Array = jnp.array(0.0, dtype=jnp.float32)
        for key, field in fields.items():
            configs = reg_terms.get(key, ())
            for weight, reg_fn in configs:
                reg_term = reg_term + weight * reg_fn(field, mask_fields)

        total = misfit + reg_term
        if reference_kspace is not None:
            assert op_ref is not None and y_ref_coils is not None and scale_ref is not None
            pred_ref_k = op_ref.forward(ref)
            diff_ref = (pred_ref_k - y_ref_coils) * scale_ref
            weights_ref = None
            if not op_ref.is_non_cartesian and op_ref.mask is not None:
                weights_ref = op_ref.mask
            if ref_finite is not None:
                weights_ref = ref_finite if weights_ref is None else (weights_ref * ref_finite)
            if weights_ref is not None:
                num_ref = jnp.sum(jnp.abs(diff_ref) ** 2 * weights_ref)
                denom_ref = (jnp.sum(weights_ref) + 1e-8).astype(jnp.float32)
                misfit_ref = 0.5 * num_ref / denom_ref
            else:
                misfit_ref = 0.5 * jnp.mean(jnp.abs(diff_ref) ** 2)
            total = total + ref_weight * misfit_ref
        return total

    def _validate_init(default, candidate):
        if candidate is None:
            return default
        try:
            if constant_dims:
                candidate = _collapse_params_for_constant_dims(candidate, constant_dims)

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

    params = _validate_init(params, initial_params)
    if reference_kspace is not None:
        params_all = {"model": params, "reference_image": params_all["reference_image"]}
    else:
        params_all = params

    opt = make_optimizer(
        optimizer_name,
        step_size,
        regularizer_weight=regularizer_weight,
        regularizer_prox=regularizer_prox,
        base_optimizer_factory=optimizer_base_factory,
        box_constraints=box_constraints,
    )
    opt_state = opt.init(params_all)

    @jax.jit
    def train_step(params, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    keep_losses = return_history or return_model_callable
    losses: list[float] = []
    for _ in range(int(num_steps)):
        params_all, opt_state, loss = train_step(params_all, opt_state)
        if keep_losses:
            losses.append(float(loss))

    final_model_params = params_all if reference_kspace is None else params_all["model"]
    fitted_fields_raw = unpack(final_model_params)
    if reference_kspace is not None:
        fitted_fields_raw = dict(fitted_fields_raw)
        fitted_fields_raw["reference_image"] = params_all["reference_image"]
    fitted_fields_raw = _broadcast_fields_to_field_shape(fitted_fields_raw, constant_dims, field_shape)
    fitted_fields_raw = _mask_fields_for_output(fitted_fields_raw, mask_fields)
    units = dict(model.parameter_units or {})
    if reference_kspace is not None:
        units.setdefault("reference_image", "1")
    fitted_fields = attach_units(fitted_fields_raw, units)

    if not return_history and not return_model_callable:
        return fitted_fields

    if return_model_callable:
        model_callable = make_model_callable(experiment, model, study_globals=inferred_globals)
        return fitted_fields, losses, model_callable

    return fitted_fields, losses


__all__ = [
    "Array",
    "MRIEncodingOperator",
    "RegridApodizationCorrection",
    "build_operator",
    "fftnc",
    "ifftnc",
    "prox_tv_chambolle",
    "regrid_nearest",
    "reconstruct_regrid",
    "reconstruct_ifft",
    "reconstruct_variational",
    "reconstruct_model_based",
]
