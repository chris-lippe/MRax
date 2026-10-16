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
    _mask_for_regularized_field,
    _masked_rms,
    _normalize_constant_dims,
    _prepare_frame_mask,
    _prepare_mask_arrays,
    _reduce_mask_to_field,
    _resolve_globals,
    _sanitize_observables,
    make_model_callable,
    warn_if_fit_requires_model_override,
)
from mrax.data_struct.dimensions import expand_observables_to_frames
from mrax.data_struct.experiment import Experiment
from mrax.data_struct.model import Model
from mrax.data_struct.quantity import Quantity, attach_units, require_quantity_dict
from mrax.devices import DeviceLike, bind_device, device_function, jit_on_device
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
    normalization: str = "relative"


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
        return mask_arr
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


def _infer_spatial_shape_from_kspace(kspace: Array | np.ndarray, spatial_ndim: int = 2) -> tuple[int, ...]:
    shape = np.shape(kspace)
    if len(shape) < spatial_ndim:
        raise ValueError("K-space has fewer dimensions than the declared encoding axes")
    return tuple(int(n) for n in shape[-spatial_ndim:])


def _has_coils_cartesian(
    kspace: Array | np.ndarray, spatial_shape: tuple[int, ...], sensitivities: Array | np.ndarray | None
) -> bool:
    dimensions = np.ndim(kspace)
    if sensitivities is not None:
        return True
    # Heuristic: (.., coils, *spatial) must have at least one extra dim beyond spatial.
    return dimensions >= len(spatial_shape) + 2


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
    normalization: str = "relative",
) -> Array:
    """Approximate deapodization for a Kaiser-Bessel gridding kernel."""
    from jax.scipy.special import i0e

    if output_size <= 0 or full_size <= 0:
        raise ValueError("deapodization sizes must be positive")
    width = float(kernel_width)
    beta = float(shape_parameter)
    if width <= 0.0 or beta <= 0.0:
        return jnp.ones((int(output_size),), dtype=jnp.float32)

    idx = jnp.arange(int(offset), int(offset) + int(output_size), dtype=jnp.float32)
    u = (idx - (int(full_size) // 2)) / float(full_size)
    arg = beta**2 - (np.pi * width * u) ** 2
    sqrt_pos = jnp.sqrt(jnp.maximum(arg, 0.0))
    sqrt_neg = jnp.sqrt(jnp.maximum(-arg, 0.0))
    scaled_sinh = (jnp.exp(sqrt_pos - beta) - jnp.exp(-sqrt_pos - beta)) / (2 * jnp.maximum(sqrt_pos, 1e-6))
    pos = jnp.where(sqrt_pos > 1e-6, scaled_sinh, jnp.exp(-beta)) / i0e(beta)
    neg = jnp.sinc(sqrt_neg / np.pi) * jnp.exp(-beta) / i0e(beta)
    response = jnp.where(arg >= 0.0, pos, neg)
    dc_response = -jnp.expm1(-2 * beta) / (2 * beta * i0e(beta))
    response = response / dc_response
    correction = 1.0 / jnp.maximum(jnp.abs(response), 1e-3)
    if float(strength) != 1.0:
        correction = correction ** float(strength)
    if normalization == "inverse_fft":
        kernel_integral = width * dc_response
        correction *= (full_size / kernel_integral)**float(strength)
    elif normalization != "relative":
        raise ValueError("Deapodization normalization must be relative or inverse_fft")
    correction *= jnp.where(response < 0, -1., 1.)
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
            normalization=correction.normalization,
        )
        reshape = (1,) * (arr.ndim - spatial_ndim + axis) + (int(out),) + (1,) * (spatial_ndim - axis - 1)
        corrected = corrected * corr_1d.reshape(reshape)
    return corrected


def _ellipse_mask(shape: tuple[int, ...], radius: float = 1.0) -> Array:
    spatial_shape = tuple(int(v) for v in shape)
    dist = jnp.zeros(spatial_shape, dtype=jnp.float32)
    for axis, size in enumerate(spatial_shape):
        # Spatial support shares the centered Fourier grid's DC origin. A
        # half-pixel offset clips different voxels on opposite boundaries.
        coords = (jnp.arange(int(size), dtype=jnp.float32) - size // 2) / (size / 2.0)
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
    of the optional phase step in the acquisition recipe.
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
        sum when sensitivities are not supplied.
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
    normalize_occupancy: bool = False,
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
    coords = jnp.floor(coords + .5).astype(jnp.int32)
    flat_idx = jnp.zeros((m,), dtype=jnp.int32)
    stride = 1
    for axis in range(len(shape) - 1, -1, -1):
        idx = jnp.mod(coords[:, axis], shape[axis])
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
    if normalize_occupancy:
        grid = grid / jnp.where(weights == 0, 1.0, weights)
    return grid, weights




@dataclass(frozen=True)
class MRIEncodingOperator:
    """Explicit encoding and its mathematical adjoint on a declared spatial grid."""

    spatial_shape: tuple[int, ...]
    sensitivities: Array | None = None
    mask: Array | None = None
    trajectory: Array | None = None
    density: Array | None = None
    axes: tuple[int, ...] = (-2, -1)
    nufft_eps: float = 1e-6
    independent_coils: bool = False
    readout_coordinates: Array | None = None
    readout_polarity: Array | None = None
    readout_phase: Array | None = None

    @property
    def is_non_cartesian(self) -> bool:
        return self.trajectory is not None

    @property
    def coil_count(self) -> int:
        return 1 if self.sensitivities is None else int(self.sensitivities.shape[-len(self.spatial_shape) - 1])

    def _apply_mask(self, kspace: Array, *, adjoint: bool = False) -> Array:
        if self.mask is None:
            return kspace
        return kspace * (jnp.conj(self.mask) if adjoint else self.mask)

    def _readout_transform(self, array, *, adjoint=False):
        """A separable readout Fourier matrix and its exact conjugate transpose."""
        size = self.spatial_shape[-1]
        position = jnp.arange(size) - size // 2
        coordinates = (jnp.broadcast_to(jnp.fft.fftshift(jnp.fft.fftfreq(size)), (2, size))
                       if self.readout_coordinates is None else jnp.asarray(self.readout_coordinates))
        matrix = jnp.exp(-2j * jnp.pi * coordinates[:, :, None] * position[None, None, :]) / np.sqrt(size)
        negative = jnp.asarray(self.readout_polarity) < 0
        phase = jnp.asarray(1., dtype=array.dtype)
        if self.readout_phase is not None:
            coefficients = jnp.asarray(self.readout_phase)
            coefficients = coefficients.reshape(coefficients.shape[:-1] + (1,) * (array.ndim - coefficients.ndim) + (2,))
            phase = jnp.exp(1j * (coefficients[..., 0, None] + coefficients[..., 1, None] * position / size))
        if not adjoint:
            positive = jnp.einsum("...n,mn->...m", array, matrix[0], precision=jax.lax.Precision.HIGHEST)
            opposite = jnp.einsum("...n,mn->...m", array * phase, matrix[1], precision=jax.lax.Precision.HIGHEST)
            return jnp.where(negative, opposite, positive)
        positive = jnp.einsum("...m,mn->...n", jnp.where(negative, 0, array), jnp.conj(matrix[0]), precision=jax.lax.Precision.HIGHEST)
        opposite = jnp.einsum("...m,mn->...n", jnp.where(negative, array, 0), jnp.conj(matrix[1]), precision=jax.lax.Precision.HIGHEST)
        return positive + opposite * jnp.conj(phase)

    def forward(self, image: Array, *, image_has_coils: bool | None = None) -> Array:
        """Encode image batches; a receiver axis is always explicitly requested."""
        x = jnp.asarray(image)
        ndim = len(self.spatial_shape)
        if tuple(x.shape[-ndim:]) != self.spatial_shape:
            raise ValueError("Image shape does not match the encoding grid")
        has_coils = self.independent_coils if image_has_coils is None else image_has_coils
        if has_coils:
            if x.ndim < ndim + 1:
                raise ValueError("Explicit coil images require a receiver axis")
            if self.sensitivities is not None and x.shape[-ndim - 1] != self.coil_count:
                raise ValueError("Coil images do not match sensitivity receiver count")
            coils = x
        else:
            coils = jnp.expand_dims(x, axis=-ndim - 1)
            if self.sensitivities is not None:
                coils = coils * jnp.asarray(self.sensitivities)
        if not self.is_non_cartesian:
            if self.readout_coordinates is not None or self.readout_phase is not None:
                hybrid = fftnc(coils, axes=self.axes[:-1]) if len(self.axes) > 1 else coils
                return self._apply_mask(self._readout_transform(hybrid))
            return self._apply_mask(fftnc(coils, axes=self.axes))
        if jax_finufft is None:
            raise ImportError("Non-Cartesian encoding requires jax-finufft")
        trajectory = jnp.asarray(self.trajectory)
        if trajectory.ndim != 2 or trajectory.shape[1] != ndim:
            raise ValueError("Trajectory columns must match encoding axes")
        complex_dtype = jnp.result_type(coils.dtype, jnp.complex64)
        real_dtype = jnp.real(jnp.asarray(0, complex_dtype)).dtype
        omega = (2 * jnp.pi) * trajectory.astype(real_dtype)
        batch_shape = coils.shape[:-ndim - 1]
        batch = int(np.prod(batch_shape)) if batch_shape else 1
        receivers = coils.shape[-ndim - 1]
        source = coils.reshape((batch, receivers) + self.spatial_shape).astype(complex_dtype)
        points = tuple(jnp.broadcast_to(omega[None, :, i], (batch, len(trajectory))) for i in range(ndim))
        result = jax_finufft.nufft2(source, *points, iflag=-1, eps=float(self.nufft_eps))
        return result.reshape(batch_shape + (receivers, len(trajectory))) / np.sqrt(np.prod(self.spatial_shape))

    def adjoint(
        self, kspace: Array, *, kspace_has_coils: bool | None = None,
        density_compensation: bool = False, normalize_sensitivities: bool = False,
    ) -> Array:
        """Apply the true adjoint; compensation and SENSE normalization are opt-in."""
        y = jnp.asarray(kspace)
        ndim = len(self.spatial_shape)
        if not self.is_non_cartesian:
            has_coils = (_has_coils_cartesian(y, self.spatial_shape, self.sensitivities)
                         if kspace_has_coils is None else kspace_has_coils)
            coils, _ = _ensure_coil_axis(y, has_coils=has_coils, spatial_ndim=ndim, coil_axis=-ndim - 1)
            if self.readout_coordinates is not None or self.readout_phase is not None:
                hybrid = self._readout_transform(self._apply_mask(coils, adjoint=True), adjoint=True)
                images = ifftnc(hybrid, axes=self.axes[:-1]) if len(self.axes) > 1 else hybrid
            else:
                images = ifftnc(self._apply_mask(coils, adjoint=True), axes=self.axes)
        else:
            if jax_finufft is None:
                raise ImportError("Non-Cartesian encoding requires jax-finufft")
            trajectory = jnp.asarray(self.trajectory)
            if trajectory.ndim != 2 or trajectory.shape[1] != ndim:
                raise ValueError("Trajectory columns must match encoding axes")
            original_ndim = y.ndim
            if y.ndim == 1:
                coils = y[None, None, :]
            elif y.ndim == 2 and self.sensitivities is None and kspace_has_coils is not True:
                coils = y[:, None, :]
            elif y.ndim == 2:
                coils = y[None, :, :]
            else:
                coils = y
            if coils.shape[-1] != len(trajectory):
                raise ValueError("Samples do not match trajectory length")
            complex_dtype = jnp.result_type(coils.dtype, jnp.complex64)
            real_dtype = jnp.real(jnp.asarray(0, complex_dtype)).dtype
            if density_compensation and self.density is not None:
                coils = coils * jnp.asarray(self.density, dtype=real_dtype)
            batch_shape = coils.shape[:-2]
            batch = int(np.prod(batch_shape)) if batch_shape else 1
            source = coils.reshape((batch, coils.shape[-2], coils.shape[-1])).astype(complex_dtype)
            omega = (2 * jnp.pi) * trajectory.astype(real_dtype)
            points = tuple(jnp.broadcast_to(omega[None, :, i], (batch, len(trajectory))) for i in range(ndim))
            images = jax_finufft.nufft1(self.spatial_shape, source, *points, iflag=1, eps=float(self.nufft_eps))
            images = images.reshape(batch_shape + (coils.shape[-2],) + self.spatial_shape) / np.sqrt(np.prod(self.spatial_shape))
            if original_ndim == 1 or (original_ndim == 2 and (self.sensitivities is not None or kspace_has_coils is True)):
                images = images[0]
        if self.independent_coils:
            return images
        if self.sensitivities is None:
            return jnp.sum(images, axis=-ndim - 1)
        sensitivities = jnp.asarray(self.sensitivities)
        result = jnp.sum(images * jnp.conj(sensitivities), axis=-ndim - 1)
        if normalize_sensitivities:
            result = result / jnp.maximum(jnp.sum(jnp.abs(sensitivities) ** 2, axis=-ndim - 1), 1e-12)
        return result


def build_operator(
    *,
    spatial_shape: tuple[int, ...] | None = None,
    kspace: Array | np.ndarray | None = None,
    sensitivities: Array | np.ndarray | None = None,
    mask: Array | np.ndarray | None = None,
    trajectory: Array | np.ndarray | None = None,
    density: Array | np.ndarray | None = None,
    axes: Sequence[int] | None = None,
    nufft_eps: float = 1e-6,
    independent_coils: bool = False,
    readout_coordinates=None, readout_polarity=None, readout_phase=None,
) -> MRIEncodingOperator:
    """Build one explicit encoding operator for Cartesian or sampled inputs."""
    if spatial_shape is None:
        if kspace is None or trajectory is not None:
            raise ValueError("An explicit encoding grid is required")
        spatial_shape = _infer_spatial_shape_from_kspace(kspace, len(axes) if axes is not None else 2)
    spatial_shape = tuple(int(n) for n in spatial_shape)
    if not 1 <= len(spatial_shape) <= 3 or any(n <= 0 for n in spatial_shape):
        raise ValueError("Encoding grid must contain one to three positive dimensions")
    if axes is None:
        axes = tuple(range(-len(spatial_shape), 0))
    elif kspace is not None:
        axes = tuple(axis - np.ndim(kspace) if axis >= 0 else axis for axis in axes)
    axes = tuple(int(axis) for axis in axes)
    if set(axes) != set(range(-len(spatial_shape), 0)):
        raise ValueError("Encoding axes must identify the trailing spatial grid")
    if not np.isfinite(nufft_eps) or nufft_eps <= 0:
        raise ValueError("nufft_eps must be finite and positive")
    sens = None if sensitivities is None else jnp.asarray(sensitivities)
    if sens is not None and (sens.ndim not in (len(spatial_shape) + 1, len(spatial_shape) + 2) or tuple(sens.shape[-len(spatial_shape):]) != spatial_shape):
        raise ValueError("Sensitivities require an explicit receiver axis and matching grid")
    points = None if trajectory is None else jnp.asarray(trajectory)
    if points is not None and (points.ndim != 2 or points.shape[1] != len(spatial_shape)):
        raise ValueError("Trajectory columns must match the encoding dimensions")
    density_array = None if density is None else jnp.asarray(density)
    if readout_coordinates is not None or readout_phase is not None:
        if points is not None or readout_polarity is None:
            raise ValueError("Readout Fourier calibration requires Cartesian phase axes and explicit polarities")
        if readout_coordinates is not None:
            checked_coordinates = np.asarray(readout_coordinates)
            if checked_coordinates.shape != (2, spatial_shape[-1]) or not np.isfinite(checked_coordinates).all():
                raise ValueError("Readout coordinates must contain finite positive/negative readout vectors")
        if readout_phase is not None and (np.shape(readout_phase)[-1:] != (2,) or not np.isfinite(readout_phase).all()):
            raise ValueError("Readout phase requires finite offset/slope pairs")
        checked_polarity = np.asarray(readout_polarity)
        if not np.isin(checked_polarity, [-1, 0, 1]).all() or not np.array_equal(checked_polarity,
                np.broadcast_to(checked_polarity[..., :1], checked_polarity.shape)):
            raise ValueError("Readout polarity must be constant along each readout")
    if density_array is not None:
        if points is None or density_array.shape != (len(points),):
            raise ValueError("Density must match sampled trajectory length")
        if not isinstance(density_array, jax.core.Tracer):
            checked = np.asarray(density_array)
            if not np.isfinite(checked).all() or np.any(checked < 0) or not np.any(checked > 0):
                raise ValueError("Density must be finite, nonnegative and have positive support")
    return MRIEncodingOperator(
        spatial_shape=spatial_shape, sensitivities=sens,
        mask=_as_mask(mask, spatial_shape) if points is None else None,
        trajectory=points, density=density_array, axes=axes,
        nufft_eps=float(nufft_eps), independent_coils=independent_coils,
        readout_coordinates=readout_coordinates, readout_polarity=readout_polarity,
        readout_phase=readout_phase,
    )




@device_function
def solve_variational(
    kspace: Array | np.ndarray, *,
    spatial_shape: tuple[int, ...] | None = None,
    sensitivities: Array | np.ndarray | None = None,
    mask: Array | np.ndarray | None = None,
    trajectory: Array | np.ndarray | None = None,
    density: Array | np.ndarray | None = None,
    data_weights: Array | np.ndarray | None = None,
    independent_coils: bool = False,
    readout_coordinates=None, readout_polarity=None, readout_phase=None,
    frame_shifts: Array | np.ndarray | None = None,
    frame_axis: int = 0, shift_order: int = 1,
    shift_mode: str = "constant", shift_cval: float = 0.0,
    optimizer_name: str | None = None, step_size: float | None = None,
    num_steps: int = 50,
    regularizer: Literal["none", "l2", "l1", "tv"] = "tv",
    regularizer_weight: float = 0.01, tv_iterations: int = 2048,
    prox_tolerance: float = 1e-7, nufft_eps: float = 1e-6,
    bregman_penalty: float | None = None,
    initial_image: Array | np.ndarray | None = None,
    return_history: bool = False, grad_clip_norm: float | None = None,
    device: DeviceLike = "cpu",
) -> tuple[Array, list[float]] | Array:
    """Minimize a measured-sample mean squared error plus an image penalty.

    Sampling masks retain frame and receiver dimensions. Density compensation
    only initializes the image; statistical likelihood weights are supplied
    separately through ``data_weights``. TV defaults to objective-aware FISTA
    with a certified proximal step. Explicit Adam with TV warns and smooths TV.
    Spatial grids can have one, two or three dimensions.
    """
    import warnings
    from operator import index

    from mrax.core.fit_execution import execute_optimizer
    from mrax.core.tv_fit import make_tv_steps
    from mrax.opt.tv import L1Operator, TVOperator

    num_steps, tv_iterations = index(num_steps), index(tv_iterations)
    if num_steps < 0 or tv_iterations < 1:
        raise ValueError("num_steps must be nonnegative and tv_iterations positive")
    if not np.isfinite(regularizer_weight) or regularizer_weight < 0:
        raise ValueError("regularizer_weight must be finite and nonnegative")
    if not np.isfinite(prox_tolerance) or prox_tolerance <= 0:
        raise ValueError("prox_tolerance must be finite and positive")
    if grad_clip_norm is not None and (not np.isfinite(grad_clip_norm) or grad_clip_norm <= 0):
        raise ValueError("grad_clip_norm must be finite and positive")
    reg = regularizer.lower()
    if reg not in {"none", "l2", "l1", "tv"}:
        raise ValueError(f"Unknown regularizer '{regularizer}'")
    name = (optimizer_name or ("fista" if reg in {"tv", "l1"} and regularizer_weight else "adam")).lower()
    if name == "bregman":
        name = "split_bregman"
    proximal = name in {"fista", "split_bregman"}
    if reg in {"tv", "l1"} and regularizer_weight and not proximal:
        warnings.warn(f"{name} uses a smoothed {reg.upper()} penalty; select FISTA or split_bregman for exact proximal regularization",
                      UserWarning, stacklevel=2)
    op = build_operator(spatial_shape=spatial_shape, kspace=kspace,
        sensitivities=sensitivities, trajectory=trajectory, density=density,
        nufft_eps=nufft_eps, independent_coils=independent_coils,
        readout_coordinates=readout_coordinates, readout_polarity=readout_polarity, readout_phase=readout_phase)
    ndim = len(op.spatial_shape)
    y = jnp.asarray(kspace)
    if op.is_non_cartesian:
        has_coils = independent_coils or (sensitivities is not None)
        if y.ndim == 1:
            y = y[None, :]
        elif y.ndim == 2 and not has_coils:
            y = y[:, None, :]
    else:
        has_coils = independent_coils or _has_coils_cartesian(y, op.spatial_shape, op.sensitivities)
        if initial_image is not None and not has_coils:
            initial = jnp.asarray(initial_image)
            layout = jax.eval_shape(op.forward, jax.ShapeDtypeStruct(initial.shape, initial.dtype))
            has_coils = layout.shape == y.shape
        if not has_coils:
            y = jnp.expand_dims(y, -ndim - 1)
    if not independent_coils and y.shape[-(2 if op.is_non_cartesian else ndim + 1)] != op.coil_count:
        raise ValueError("Multiple receivers require sensitivities or independent_coils=True")

    def sample_weights(value, default):
        if value is None:
            return jnp.asarray(default, dtype=jnp.real(y).dtype)
        array = jnp.asarray(value)
        # Image-shaped frame masks acquire their missing receiver dimension.
        if array.ndim == y.ndim - 1 and tuple(array.shape) == tuple(y.shape[:-ndim - 1] + y.shape[-ndim:]) and not op.is_non_cartesian:
            array = jnp.expand_dims(array, -ndim - 1)
        if np.broadcast_shapes(array.shape, y.shape) != y.shape:
            raise ValueError("Sampling weights do not broadcast to measured samples")
        checked = np.asarray(array)
        if not np.isfinite(checked).all() or np.any(checked < 0):
            raise ValueError("Sampling and statistical weights must be finite and nonnegative")
        return array

    weights = sample_weights(mask, 1) * sample_weights(data_weights, 1)
    if not bool(jnp.any(weights > 0)):
        raise ValueError("No acquired samples have positive weight")
    if not bool(jax.jit(lambda data, weight: jnp.all(jnp.where(weight > 0, jnp.isfinite(data), True)))(y, weights)):
        raise ValueError("Acquired samples must be finite")
    weight_sum = jnp.sum(weights) * (int(np.prod(y.shape)) // int(np.prod(weights.shape)))
    shifts = None
    if frame_shifts is not None:
        if ndim != 2 or independent_coils:
            raise ValueError("Frame motion requires two-dimensional sensitivity-combined images")
        shifts = jnp.asarray(frame_shifts)
        full_weights = jnp.broadcast_to(weights, y.shape)
        y = jnp.moveaxis(y, frame_axis, 0)
        weights = jnp.moveaxis(full_weights, frame_axis, 0)
        if shifts.shape != (y.shape[0], 2):
            raise ValueError("frame_shifts must have shape (frames, 2)")

    def shift_image(image, shift, *, cval=shift_cval):
        return _shift_image_last2(image, shift, order=shift_order,
            mode=shift_mode, cval=cval)

    if initial_image is None:
        backprojection = op.adjoint(jnp.where(weights > 0, y, 0), kspace_has_coils=True,
            density_compensation=True, normalize_sensitivities=True)
        x0 = (jnp.mean(jax.vmap(lambda image, shift: shift_image(image, -shift))(backprojection, shifts), axis=0)
              if shifts is not None else backprojection)
    else:
        x0 = jnp.asarray(initial_image, dtype=jnp.result_type(y.dtype, jnp.asarray(initial_image).dtype))
    if tuple(x0.shape[-ndim:]) != op.spatial_shape:
        raise ValueError("initial_image does not match the encoding grid")
    if not bool(jnp.all(jnp.isfinite(x0))):
        raise ValueError("initial_image must be finite")

    def encode(image, *, homogeneous=False):
        if shifts is None:
            return op.forward(image)
        fill = 0. if homogeneous else shift_cval
        return jax.vmap(lambda shift: op.forward(shift_image(image, shift, cval=fill)))(shifts)

    def data_loss(image):
        prediction = encode(image)
        if prediction.shape != y.shape:
            raise ValueError("Encoded image layout does not match measured samples")
        residual = jnp.where(weights > 0, prediction - y, 0)
        return .5 * jnp.sum(weights * jnp.abs(residual)**2) / weight_sum

    homogeneous_data_loss = None
    if (sensitivities is not None and not independent_coils and not op.is_non_cartesian and shifts is None and
            op.readout_coordinates is None and op.readout_phase is None and y.ndim in (ndim + 1, ndim + 2)):
        from .cartesian_loss import cartesian_sense_loss

        data_loss = cartesian_sense_loss(y, weights, op.sensitivities, op.spatial_shape)
        homogeneous_data_loss = cartesian_sense_loss(y, weights, op.sensitivities, op.spatial_shape, homogeneous=True)

    tv_operator = None
    if reg == "tv" and regularizer_weight:
        tv_operator = TVOperator.create(x0.shape, tuple(range(x0.ndim - ndim, x0.ndim)),
            weight=float(regularizer_weight), dtype=jnp.real(x0).dtype)
    if reg == "l1" and regularizer_weight:
        tv_operator = L1Operator(jnp.full(x0.shape, regularizer_weight / x0.size, dtype=jnp.real(x0).dtype))
    def penalty(image):
        if tv_operator is not None:
            return tv_operator.value(image, smoothed=not proximal)
        return (regularizer_weight * jnp.mean(jnp.abs(image)**2) if reg == "l2" else jnp.asarray(0., jnp.real(image).dtype))
    def objective(values):
        return data_loss(values["image"]) + penalty(values["image"])
    def smooth(values):
        return data_loss(values["image"]) + (penalty(values["image"]) if tv_operator is None else 0)

    if step_size is None:
        if proximal:
            l2_curvature = 2 * regularizer_weight / x0.size if reg == "l2" else 0.
            if not op.is_non_cartesian and op.readout_coordinates is None and op.readout_phase is None and sensitivities is None and shifts is None:
                # Unitary Fourier encoding diagonalizes the sample weights.
                curvature = jnp.max(weights) / weight_sum + l2_curvature
            else:
                def homogeneous_loss(image):
                    value = (homogeneous_data_loss(image) if homogeneous_data_loss is not None else
                        .5 * jnp.sum(weights * jnp.abs(encode(image, homogeneous=True))**2) / weight_sum)
                    return value + (regularizer_weight * jnp.mean(jnp.abs(image)**2) if reg == "l2" else 0)
                gradient = jax.grad(homogeneous_loss)
                direction = jax.random.rademacher(jax.random.key(0), x0.shape, dtype=jnp.real(x0).dtype).astype(x0.dtype)
                if jnp.iscomplexobj(x0):
                    direction += 1j * jax.random.rademacher(jax.random.key(1), x0.shape, dtype=jnp.real(x0).dtype)
                direction /= jnp.linalg.norm(direction)
                def power(_, vector):
                    product = jnp.conj(gradient(vector))
                    return product / jnp.maximum(jnp.linalg.norm(product), 1e-20)
                direction = jax.lax.fori_loop(0, 20, power, direction)
                curvature = jnp.real(jnp.vdot(direction, jnp.conj(gradient(direction))))
            step_size = .95 / max(float(curvature), 1e-12)
        else:
            step_size = .02 * max(float(jnp.sqrt(jnp.mean(jnp.abs(x0)**2))), 1e-6)
    if not np.isfinite(step_size) or step_size <= 0:
        raise ValueError("step_size must be finite and positive")
    parameters = {"image": x0}
    step_function = convergence_check = None
    if proximal:
        optimizer, step_function, convergence_check = make_tv_steps(objective, smooth,
            {"image": (-np.inf, np.inf)}, {"image": () if tv_operator is None else (tv_operator,)},
            name, step_size, tolerance=prox_tolerance, max_prox_steps=tv_iterations,
            grad_clip_norm=grad_clip_norm,
            penalty=1. / x0.size if bregman_penalty is None else bregman_penalty,
            device=device)
    else:
        optimizer = make_optimizer(name, step_size, device=device)
        if grad_clip_norm is not None:
            optimizer = optax.chain(optax.clip_by_global_norm(grad_clip_norm), optimizer)
    parameters, losses, info = execute_optimizer(objective, parameters, optimizer,
        num_steps=num_steps, step_function=step_function,
        convergence_check=convergence_check, device=device)
    image = parameters["image"]
    history = losses[1:] + [float(info.final_loss)] if losses else []
    return (image, history) if return_history else image


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
    kspace: Array | np.ndarray, *, frame_count: int, coil_count: int,
    is_non_cartesian: bool, sensitivities: Array | None, spatial_ndim: int = 2,
) -> Array:
    """Normalize array layouts into explicit (frame, receiver, encoding...)."""
    y = jnp.asarray(kspace)
    dimensions = 1 if is_non_cartesian else spatial_ndim
    if y.ndim == dimensions:
        if frame_count != 1 or coil_count != 1:
            raise ValueError("Samples require explicit frame and receiver axes")
        y = y[None, None, ...]
    elif y.ndim == dimensions + 1:
        if y.shape[0] == frame_count and coil_count == 1:
            y = y[:, None, ...]
        elif frame_count == 1 and y.shape[0] == coil_count:
            y = y[None, ...]
        else:
            raise ValueError("Samples do not match declared frame and receiver counts")
    elif y.ndim == dimensions + 2:
        if y.shape[:2] == (frame_count, coil_count):
            pass
        elif y.shape[:2] == (coil_count, frame_count):
            y = jnp.swapaxes(y, 0, 1)
        else:
            raise ValueError("Samples do not match declared frame and receiver counts")
    else:
        raise ValueError("Samples have an invalid encoding layout")
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


@device_function
def solve_model_based(
    kspace: Array | np.ndarray,
    *,
    experiment: Experiment,
    model: Model,
    spatial_shape: tuple[int, ...] | None = None,
    sensitivities: Array | np.ndarray | None = None,
    mask: Array | np.ndarray | None = None,
    data_weights: Array | np.ndarray | None = None,
    completion_kspace: Array | np.ndarray | None = None,
    completion_weight: float = 0.0,
    trajectory: Array | np.ndarray | None = None,
    density: Array | np.ndarray | None = None,
    optimizer_name: str | None = None,
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
    object_support_mask: object | None = None,
    image_phase: Array | np.ndarray | None = None,
    image_to_encoding: Callable | None = None,
    encoding_frame_indices: Array | np.ndarray | None = None,
    readout_coordinates=None, readout_polarity=None, readout_phase=None,
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
    return_reconstruction_image: bool = False,
    proximal_tolerance: float = 1e-8,
    proximal_max_steps: int = 2048,
    bregman_penalty: float = 1.0,
    bregman_inner_steps: int = 5,
    device: DeviceLike = "cpu",
) -> (
    tuple[dict[str, Quantity], list[float], Callable, Array]
    | tuple[dict[str, Quantity], list[float], Callable[[dict, Array | None, dict[str, Any] | None], Array]]
    | tuple[dict[str, Quantity], list[float]]
    | dict[str, Quantity]
):
    """Directly reconstruct parameter maps from k-space via a chained model+encoding operator.

    Minimizes: :math:`\\tfrac12 \\|\\|A(M(\\theta)) - y\\|\\|_2^2 + R(\\theta)`

    Notes
    -----
    - The signal model comes from :class:`mrax.data_struct.model.Model` and is treated as a
      (generally non-linear) operator mapping parameter fields -> image series.
    - The encoding operator :math:`A` is the same as in :func:`solve_variational` (FFT/NUFFT,
      optional sampling mask, optional coil sensitivities).
    - ``roi_mask`` weights regularizers and masks output fields. Physical object
      support is supplied separately with ``object_support_mask``; an analysis ROI
      must not erase signal outside that ROI before Fourier encoding.
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
    - ``return_reconstruction_image=True`` additionally returns the fitted image
      before output ROI masking, and requires ``return_model_callable=True``.
    """

    warn_if_fit_requires_model_override(experiment, model)

    from operator import index

    from mrax.core.fit_execution import execute_optimizer

    num_steps = index(num_steps)
    if return_reconstruction_image and not return_model_callable:
        raise ValueError("return_reconstruction_image requires return_model_callable=True")
    if num_steps < 0:
        raise ValueError("num_steps must be nonnegative")
    for label, value in (("completion_weight", completion_weight), ("reference_weight", reference_weight)):
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{label} must be finite and nonnegative")

    full_spatial_shape, spatial_dim_names = _extract_spatial_metadata(experiment)
    if spatial_shape is None:
        spatial_shape = tuple(int(v) for v in full_spatial_shape)
    spatial_ndim = len(spatial_shape)
    if spatial_ndim not in (1, 2, 3):
        raise ValueError("Model reconstruction requires one to three spatial axes")
    if reference_kspace is not None and spatial_ndim != 2:
        raise ValueError("Joint reference reconstruction currently requires two spatial axes")

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
    params = jax.device_put(params, device)
    unpack = jit_on_device(unpack, device)
    params = _collapse_params_for_constant_dims(params, constant_dims)
    field_keys = set(unpack(params))
    if set(constant_dims) - field_keys:
        raise ValueError("parameter_constant_dims contains unknown parameter fields")

    reg_terms = model.make_regularizer_terms(
        field_shape,
        regularizer_configs=regularizer_configs,
        regularizer_type=regularizer_type,
        reg_weights=regularizer_weights,
    )
    configured_keys = set((regularizer_configs if regularizer_configs is not None else model.regularizer_configs) or {})
    configured_keys.update((regularizer_weights if regularizer_weights is not None else model.reg_weights) or {})
    if configured_keys - field_keys:
        raise ValueError("Regularization contains unknown parameter fields")
    has_tv = any(hasattr(fn, "tv_order") for entries in reg_terms.values() for _, fn in entries)
    selected = (optimizer_name or ("fista" if has_tv else "adam")).lower()
    if selected == "bregman":
        selected = "split_bregman"
    true_tv = has_tv and selected in {"fista", "split_bregman"}
    if has_tv and not true_tv:
        import warnings
        warnings.warn(f"{selected} uses smoothed TV; select FISTA or split_bregman for exact proximal TV", UserWarning, stacklevel=2)
        reg_terms = model.make_regularizer_terms(field_shape, regularizer_configs=regularizer_configs,
            regularizer_type=regularizer_type, reg_weights=regularizer_weights, smoothed_tv=True)
    if true_tv:
        if reference_kspace is not None:
            raise ValueError("Joint reference estimation with exact TV is not yet supported; supply a fixed reference_image")
        if regularizer_prox is not None or any(np.any(np.asarray(v) != 0) for v in jax.tree.leaves(regularizer_weight)) or box_constraints is not None:
            raise ValueError("Exact field TV uses physical parameter bounds and cannot combine raw proximal penalties")
        if optimizer_base_factory is not None and optimizer_base_factory is not optax.sgd:
            raise ValueError("Exact TV requires objective-aware gradient steps")
        from mrax.core.physical_parameters import physical_parametrization
        params, unpack = physical_parametrization(model, params, unpack)

    # Observables: merge study globals (if any) with experiment observables and unwrap to arrays for JIT.
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

    if obs_data_mask is not None:
        mask_data = obs_data_mask if mask_data is None else (mask_data * obs_data_mask)
        obs_field_mask = _reduce_mask_to_field(obs_data_mask, experiment, field_dim_names)
        if obs_field_mask is not None:
            mask_fields = obs_field_mask if mask_fields is None else (mask_fields * obs_field_mask)
    support_mask = None
    if object_support_mask is not None:
        support_data, _ = _prepare_mask_arrays(object_support_mask, roi_mask_dims, experiment,
            field_shape, field_dim_names, spatial_dim_names)
        support_mask = _flatten_to_frames(jnp.asarray(support_data), experiment) > 0
    phase_image = None if image_phase is None else jnp.asarray(image_phase)
    if phase_image is not None:
        if not bool(jnp.all(jnp.isfinite(phase_image))):
            raise ValueError("image_phase must be finite")
        phase_image = jnp.exp(1j * phase_image)

    frame_weights = _prepare_frame_mask(frame_mask, experiment)
    if auto_mask_nan_observables and auto_frame_mask is not None:
        frame_weights = auto_frame_mask if frame_weights is None else (frame_weights * auto_frame_mask)
    frame_weights = jnp.asarray(frame_weights) if frame_weights is not None else None
    if encoding_frame_indices is not None:
        frame_indices = np.asarray(encoding_frame_indices)
        if frame_indices.ndim != 1 or not np.issubdtype(frame_indices.dtype, np.integer) or not len(frame_indices) or np.any(frame_indices < 0) or np.any(frame_indices >= experiment.frame_count):
            raise ValueError("Encoding frame indices must refer to valid signal-model frames")
        if frame_weights is not None:
            frame_weights = frame_weights[jnp.asarray(frame_indices)]

    y = jnp.asarray(kspace)
    # A zero measurement is valid data. Missingness must be declared explicitly.
    if mask is not None and not np.all((np.asarray(mask) == 0) | (np.asarray(mask) == 1)):
        raise ValueError("Model-based sampling masks must be binary; use data_weights for statistical weights")

    op = build_operator(
        spatial_shape=spatial_shape,
        sensitivities=sensitivities,
        mask=mask,
        trajectory=trajectory,
        density=density,
        nufft_eps=nufft_eps,
        readout_coordinates=readout_coordinates, readout_polarity=readout_polarity, readout_phase=readout_phase,
    )
    frame_count = int(experiment.frame_count) if encoding_frame_indices is None else len(frame_indices)
    y_coils = _normalize_kspace_for_model(
        y,
        frame_count=frame_count,
        coil_count=op.coil_count,
        is_non_cartesian=op.is_non_cartesian,
        sensitivities=op.sensitivities,
        spatial_ndim=spatial_ndim,
    )
    if op.is_non_cartesian:
        traj_len = int(np.asarray(op.trajectory).shape[0]) if op.trajectory is not None else 0
        if int(y_coils.shape[-1]) != traj_len:
            raise ValueError("kspace sample count does not match trajectory length")
    else:
        if tuple(int(v) for v in y_coils.shape[-spatial_ndim:]) != spatial_shape:
            raise ValueError(
                f"kspace spatial shape {tuple(int(v) for v in y_coils.shape[-spatial_ndim:])} does not match {spatial_shape}"
            )

    data_finite = jnp.isfinite(y_coils)
    measured_support = None
    if mask is not None:
        try:
            measured_support = jnp.broadcast_to(jnp.asarray(mask, bool), y_coils.shape)
        except ValueError as error:
            raise ValueError("Sampling support must broadcast to measured samples") from error
    likelihood_weights = None
    if data_weights is not None:
        raw_weights = jnp.asarray(data_weights)
        if jnp.iscomplexobj(raw_weights) or not np.isfinite(raw_weights).all() or np.any(np.asarray(raw_weights) < 0):
            raise ValueError("Statistical data weights must be finite, real and nonnegative")
        try:
            likelihood_weights = jnp.broadcast_to(raw_weights, y_coils.shape)
        except ValueError as error:
            raise ValueError("Statistical data weights must broadcast to measured samples") from error
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
            spatial_ndim=spatial_ndim,
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
                    readout_coordinates=readout_coordinates, readout_polarity=readout_polarity, readout_phase=readout_phase,
                )
                use_completion = True

    expected_frames = frame_count
    y_coils, phase_rot = _align_kspace_global_phase_by_frame(
        y_coils,
        expected_frames=int(expected_frames),
        mode=str(frame_phase_correction or "none"),
        return_rotation=True,
    )
    if y_completion_coils is not None and phase_rot is not None:
        y_completion_coils = y_completion_coils * phase_rot

    included = data_finite
    if measured_support is not None:
        included &= measured_support
    if frame_weights is not None:
        included = included & (frame_weights.reshape((frame_count,) + (1,) * (y_coils.ndim - 1)) > 0)
    if likelihood_weights is not None:
        included &= likelihood_weights > 0
    if not bool(jnp.any(included)):
        raise ValueError("No measured samples remain after acquisition and frame masking")
    normalization_weights = included if likelihood_weights is None else included * likelihood_weights
    scale = 1.0 / jnp.maximum(_masked_rms(y_coils, normalization_weights), 1e-6)
    tv_operators = {}
    if true_tv:
        from mrax.opt.tv import TVOperator
        for key, value in params.items():
            field_mask = _mask_for_regularized_field(mask_fields, value)
            tv_operators[key] = tuple(TVOperator.create(value.shape, fn.tv_axes, fn.tv_order,
                field_mask, weight * unpack._mrax_physical_scales[key], value.dtype)
                for weight, fn in reg_terms.get(key, ()) if hasattr(fn, "tv_order"))

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

    def smooth_loss(p):
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
        if phase_image is not None:
            pred_img = pred_img * phase_image
        if support_mask is not None:
            pred_img = jnp.where(support_mask, pred_img, jnp.zeros((), dtype=pred_img.dtype))
        if image_to_encoding is not None:
            pred_img = image_to_encoding(pred_img)
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

        weights = measured_support
        if w_frame is not None:
            weights = w_frame if weights is None else (weights * w_frame)
        weights = data_finite if weights is None else (weights * data_finite)
        if likelihood_weights is not None:
            weights = weights * likelihood_weights

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
                if true_tv and key in p_model and hasattr(reg_fn, "tv_order"):
                    continue
                reg_term = reg_term + weight * reg_fn(field, _mask_for_regularized_field(mask_fields, field))

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

    def loss_fn(p):
        return smooth_loss(p) + sum(op.value(p[key]) for key, ops in tv_operators.items() for op in ops)

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

    if initial_params is not None and true_tv:
        initial_params = unpack._mrax_from_raw(_collapse_params_for_constant_dims(initial_params, constant_dims))
    params = _validate_init(params, initial_params)
    if reference_kspace is not None:
        params_all = {"model": params, "reference_image": params_all["reference_image"]}
    else:
        params_all = params

    if true_tv:
        from mrax.core.physical_parameters import balance_physical_coordinates
        params_all = balance_physical_coordinates(smooth_loss, params_all, unpack,
            tv_operators, balance=num_steps != 0)

    opt = make_optimizer(
        selected,
        step_size,
        regularizer_weight=regularizer_weight,
        regularizer_prox=regularizer_prox,
        base_optimizer_factory=optimizer_base_factory,
        device=device,
        box_constraints=box_constraints,
    )
    step_function = convergence_check = None
    if true_tv:
        from mrax.core.tv_fit import make_tv_steps
        opt, step_function, convergence_check = make_tv_steps(loss_fn, smooth_loss,
            unpack._mrax_physical_bounds, tv_operators, selected, step_size,
            tolerance=proximal_tolerance, max_prox_steps=proximal_max_steps,
            penalty=bregman_penalty, inner_steps=bregman_inner_steps, device=device)
    params_all, losses, info = execute_optimizer(loss_fn, params_all, opt, num_steps=num_steps,
        step_function=step_function, convergence_check=convergence_check, device=device)
    losses = losses[1:] + [float(info.final_loss)] if losses else []

    final_model_params = params_all if reference_kspace is None else params_all["model"]
    fitted_fields_raw = unpack(final_model_params)
    reconstruction_image = None
    if return_reconstruction_image:
        reference = ref_fixed if reference_kspace is None else params_all["reference_image"]
        reconstruction_image = _align_predicted(model_apply(fitted_fields_raw, reference), experiment)
        if reference is not None:
            reconstruction_image = reconstruction_image * reference
        if phase_image is not None:
            reconstruction_image = reconstruction_image * phase_image
        if support_mask is not None:
            reconstruction_image = jnp.where(support_mask, reconstruction_image, 0)
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
        model_callable = bind_device(make_model_callable(experiment, model, study_globals=inferred_globals), device)
        if return_reconstruction_image:
            return fitted_fields, losses, model_callable, reconstruction_image
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
    "solve_variational",
    "solve_model_based",
]
