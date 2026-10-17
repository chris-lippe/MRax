from __future__ import annotations

import math
from typing import Any, Callable, Iterable

import jax
import jax.numpy as jnp
import jax.scipy.signal as jsignal
import numpy as np
import optax

from mrax.devices import DeviceLike, device_function
from mrax.opt.optimizers import make_optimizer
from mrax.opt.regularizers import make_regularizer_fn

ArrayLike = Any


def _apply_complex_mode(
    array: ArrayLike,
    func: Callable[[jnp.ndarray], jnp.ndarray],
    *,
    complex_mode: str,
) -> jnp.ndarray:
    arr = jnp.asarray(array)
    if not jnp.iscomplexobj(arr):
        return func(arr)
    mode = complex_mode.lower()
    if mode == "real_imag":
        return func(jnp.real(arr)) + 1j * func(jnp.imag(arr))
    if mode == "magnitude":
        return func(jnp.abs(arr))
    raise ValueError("complex_mode must be 'real_imag' or 'magnitude'")


def _normalize_tuple(
    value: int | float | Iterable[int | float], ndim: int, *, name: str, cast: type = int
) -> tuple[int, ...] | tuple[float, ...]:
    if np.isscalar(value):
        return (cast(value),) * ndim
    vals = tuple(cast(v) for v in value)
    if len(vals) != ndim:
        raise ValueError(f"{name} must have length {ndim}")
    return vals


def _pad_for_patch(size: int) -> tuple[int, int]:
    left = size // 2
    right = size // 2
    if size % 2 == 0:
        left = max(0, left - 1)
    return left, right


def _spatial_spec(ndim: int) -> str:
    if ndim == 1:
        return "W"
    if ndim == 2:
        return "HW"
    if ndim == 3:
        return "DHW"
    raise ValueError("Only 1D/2D/3D spatial inputs are supported")


def _conv_dimension_numbers(spatial_ndim: int):
    spatial = _spatial_spec(spatial_ndim)
    return ("N" + spatial + "C", spatial + "IO", "N" + spatial + "C")


def _gaussian_kernel1d(sigma: float, truncate: float) -> jnp.ndarray:
    sigma = float(max(sigma, 1e-8))
    radius = int(truncate * sigma + 0.5)
    x = jnp.arange(-radius, radius + 1)
    kernel = jnp.exp(-(x**2) / (2.0 * sigma**2))
    kernel = kernel / jnp.sum(kernel)
    return kernel


def _convolve1d(arr: jnp.ndarray, kernel: jnp.ndarray, axis: int, *, mode: str) -> jnp.ndarray:
    pad = kernel.shape[0] // 2
    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (pad, pad)
    padded = jnp.pad(arr, pad_width, mode=mode)
    kernel_shape = [1] * arr.ndim
    kernel_shape[axis] = kernel.shape[0]
    kernel_nd = kernel.reshape(kernel_shape)
    return jsignal.convolve(padded, kernel_nd, mode="valid")


def _gaussian_filter_jax(
    array: jnp.ndarray,
    sigma: float | Iterable[float],
    *,
    mode: str,
    truncate: float,
) -> jnp.ndarray:
    arr = jnp.asarray(array)
    sigmas = _normalize_tuple(sigma, arr.ndim, name="sigma", cast=float)
    out = arr
    for axis, sig in enumerate(sigmas):
        if sig <= 0:
            continue
        kernel = _gaussian_kernel1d(sig, truncate)
        out = _convolve1d(out, kernel, axis, mode=mode)
    return out


def _uniform_filter_jax(array: jnp.ndarray, size: int | Iterable[int], *, mode: str) -> jnp.ndarray:
    arr = jnp.asarray(array)
    size_tuple = _normalize_tuple(size, arr.ndim, name="size", cast=int)
    pad_width = [_pad_for_patch(s) for s in size_tuple]
    padded = jnp.pad(arr, pad_width, mode=mode)
    window = size_tuple
    summed = jax.lax.reduce_window(
        padded,
        0.0,
        jax.lax.add,
        window_dimensions=window,
        window_strides=(1,) * arr.ndim,
        padding="VALID",
    )
    return summed / float(math.prod(size_tuple))


def _median_filter_jax(array: jnp.ndarray, size: int | Iterable[int], *, mode: str) -> jnp.ndarray:
    arr = jnp.asarray(array)
    size_tuple = _normalize_tuple(size, arr.ndim, name="size", cast=int)
    pad_width = [_pad_for_patch(s) for s in size_tuple]
    padded = jnp.pad(arr, pad_width, mode=mode)
    patches = jax.lax.conv_general_dilated_patches(
        padded[jnp.newaxis, ..., jnp.newaxis],
        size_tuple,
        (1,) * arr.ndim,
        "VALID",
        dimension_numbers=_conv_dimension_numbers(arr.ndim),
    )[0]
    median = jnp.median(patches, axis=-1)
    return median.reshape(arr.shape)


def _nan_gaussian_filter(
    array: jnp.ndarray,
    sigma: float | tuple[float, ...],
    *,
    mode: str,
    truncate: float,
) -> jnp.ndarray:
    arr = jnp.asarray(array)
    if not jnp.isnan(arr).any():
        return _gaussian_filter_jax(arr, sigma, mode=mode, truncate=truncate)
    mask = jnp.isfinite(arr).astype(arr.dtype)
    filled = jnp.where(jnp.isfinite(arr), arr, 0.0)
    filtered = _gaussian_filter_jax(filled, sigma, mode=mode, truncate=truncate)
    weights = _gaussian_filter_jax(mask, sigma, mode=mode, truncate=truncate)
    out = jnp.where(weights > 0, filtered / weights, jnp.nan)
    return out


def denoise_filter(
    array: ArrayLike,
    *,
    method: str = "gaussian",
    sigma: float | tuple[float, ...] = 1.0,
    size: int | tuple[int, ...] = 3,
    mode: str = "reflect",
    truncate: float = 4.0,
    preserve_nan: bool = False,
    complex_mode: str = "real_imag",
) -> jnp.ndarray:
    """Denoise with simple multidimensional filters (JAX implementation)."""

    arr = jnp.asarray(array)
    method_key = method.lower()

    if method_key in ("gaussian", "gauss"):
        if preserve_nan:
            def func(x):
                return _nan_gaussian_filter(x, sigma, mode=mode, truncate=truncate)
        else:
            def func(x):
                return _gaussian_filter_jax(x, sigma, mode=mode, truncate=truncate)
    elif method_key in ("median", "med"):
        if preserve_nan:
            raise ValueError("preserve_nan is only supported for gaussian filtering")
        def func(x):
            return _median_filter_jax(x, size, mode=mode)
    elif method_key in ("uniform", "box", "boxcar", "mean"):
        if preserve_nan:
            raise ValueError("preserve_nan is only supported for gaussian filtering")
        def func(x):
            return _uniform_filter_jax(x, size, mode=mode)
    else:
        raise ValueError("method must be one of {'gaussian', 'median', 'uniform'}")

    return _apply_complex_mode(arr, func, complex_mode=complex_mode)


@device_function
def _regularized_denoise_real(
    data: ArrayLike,
    *,
    regularizer: str | Callable | None,
    weight: float,
    num_steps: int,
    step_size: float,
    optimizer_name: str,
    axes: tuple[int, ...] | None,
    mask: ArrayLike | None,
    init: ArrayLike | None,
    return_history: bool,
    device: DeviceLike = "cpu",
) -> tuple[jnp.ndarray, list[float] | None]:
    arr = jnp.asarray(data, dtype=float)
    init_arr = arr if init is None else jnp.asarray(init, dtype=float)

    if axes is None:
        if arr.ndim <= 1:
            axes = tuple(range(arr.ndim))
        else:
            axes = (arr.ndim - 2, arr.ndim - 1)
    axes = tuple(int(ax) % arr.ndim for ax in axes)

    reg_fn = None
    if isinstance(regularizer, str):
        reg_fn = make_regularizer_fn(regularizer, axes=axes)
    elif callable(regularizer):
        reg_fn = regularizer

    mask_arr = None if mask is None else jnp.asarray(mask)
    target = arr
    params = init_arr

    def reg_value(x):
        if reg_fn is None or weight <= 0:
            return jnp.array(0.0, dtype=x.dtype)
        try:
            return reg_fn(x, mask=mask_arr)
        except TypeError:
            return reg_fn(x)

    def loss_fn(x):
        fidelity = 0.5 * jnp.mean((x - target) ** 2)
        return fidelity + weight * reg_value(x)

    opt = make_optimizer(optimizer_name, step_size=step_size, device=device)
    state = opt.init(params)

    history: list[float] | None = [] if return_history else None
    value_and_grad = jax.value_and_grad(loss_fn)

    if num_steps <= 0:
        return params, history

    for _ in range(num_steps):
        loss, grads = value_and_grad(params)
        updates, state = opt.update(grads, state, params)
        params = optax.apply_updates(params, updates)
        if history is not None:
            history.append(float(loss))

    return params, history


@device_function
def regularized_denoise(
    data: ArrayLike,
    *,
    regularizer: str | Callable | None = "tv",
    weight: float = 0.1,
    num_steps: int = 50,
    step_size: float = 0.1,
    optimizer_name: str = "adam",
    axes: tuple[int, ...] | None = None,
    mask: ArrayLike | None = None,
    init: ArrayLike | None = None,
    complex_mode: str = "real_imag",
    return_history: bool = False,
    device: DeviceLike = "cpu",
) -> jnp.ndarray | tuple[jnp.ndarray, list[float] | dict[str, list[float]]]:
    """Denoise using a regularizer optimized with the project optimizers."""

    arr = jnp.asarray(data)
    if jnp.iscomplexobj(arr):
        mode = complex_mode.lower()
        if mode == "real_imag":
            real, real_hist = _regularized_denoise_real(
                jnp.real(arr),
                regularizer=regularizer,
                weight=weight,
                num_steps=num_steps,
                step_size=step_size,
                optimizer_name=optimizer_name,
                axes=axes,
                mask=mask,
                init=None if init is None else jnp.real(init),
                return_history=return_history,
                device=device,
            )
            imag, imag_hist = _regularized_denoise_real(
                jnp.imag(arr),
                regularizer=regularizer,
                weight=weight,
                num_steps=num_steps,
                step_size=step_size,
                optimizer_name=optimizer_name,
                axes=axes,
                mask=mask,
                init=None if init is None else jnp.imag(init),
                return_history=return_history,
                device=device,
            )
            combined = real + 1j * imag
            if return_history:
                history = {
                    "real": real_hist or [],
                    "imag": imag_hist or [],
                }
                return combined, history
            return combined
        if mode == "magnitude":
            mag, history = _regularized_denoise_real(
                jnp.abs(arr),
                regularizer=regularizer,
                weight=weight,
                num_steps=num_steps,
                step_size=step_size,
                optimizer_name=optimizer_name,
                axes=axes,
                mask=mask,
                init=None if init is None else jnp.abs(init),
                return_history=return_history,
                device=device,
            )
            return (mag, history) if return_history else mag
        raise ValueError("complex_mode must be 'real_imag' or 'magnitude'")

    out, history = _regularized_denoise_real(
        arr,
        regularizer=regularizer,
        weight=weight,
        num_steps=num_steps,
        step_size=step_size,
        optimizer_name=optimizer_name,
        axes=axes,
        mask=mask,
        init=init,
        return_history=return_history,
        device=device,
    )
    return (out, history) if return_history else out


def _runtime_import(module: str):
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise ImportError(
            f"Optional dependency '{module}' is required for this denoiser. "
            "Install it with `pip install 'mrax[denoise-external]'`. "
            "BM3D/BM4D are licensed for non-commercial use; review their licenses before use."
        ) from exc


def _estimate_sigma(data: ArrayLike) -> float:
    arr = jnp.asarray(data, dtype=float)
    diffs = []
    for axis in range(arr.ndim):
        if arr.shape[axis] > 1:
            diffs.append(jnp.diff(arr, axis=axis))
    if not diffs:
        return float(jnp.std(arr))
    stacked = jnp.concatenate([d.reshape(-1) for d in diffs])
    mad = jnp.median(jnp.abs(stacked))
    return float(mad / (0.6745 * jnp.sqrt(2.0)))


def denoise_bm3d(
    array: ArrayLike,
    *args,
    sigma: float | None = None,
    stage: str = "full",
    **kwargs,
) -> jnp.ndarray:
    """BM3D denoising via the external ``bm3d`` package.

    Pass package-specific options through ``kwargs``.
    """

    module = _runtime_import("bm3d")
    bm3d_fn = getattr(module, "bm3d", None)
    if bm3d_fn is None:
        raise ImportError("bm3d.bm3d not found")
    if args:
        return bm3d_fn(np.asarray(array), *args, **kwargs)
    sigma_psd = float(sigma) if sigma is not None else _estimate_sigma(array)
    ext_kwargs = dict(kwargs)
    if "stage_arg" not in ext_kwargs:
        stage_key = stage.lower()
        if stage_key in ("basic", "hard"):
            ext_kwargs["stage_arg"] = module.BM3DStages.HARD_THRESHOLDING
        elif stage_key in ("wiener", "refine"):
            ext_kwargs["stage_arg"] = module.BM3DStages.WIENER_FILTERING
        else:
            ext_kwargs["stage_arg"] = module.BM3DStages.ALL_STAGES
    return bm3d_fn(np.asarray(array), sigma_psd, **ext_kwargs)


def denoise_bm4d(
    array: ArrayLike,
    *args,
    sigma: float | None = None,
    stage: str = "full",
    **kwargs,
) -> jnp.ndarray:
    """BM4D denoising via the external ``bm4d`` package.

    Pass package-specific options through ``kwargs``.
    """

    module = _runtime_import("bm4d")
    bm4d_fn = getattr(module, "bm4d", None)
    if bm4d_fn is None:
        raise ImportError("bm4d.bm4d not found")
    if args:
        return bm4d_fn(np.asarray(array), *args, **kwargs)
    sigma_psd = float(sigma) if sigma is not None else _estimate_sigma(array)
    ext_kwargs = dict(kwargs)
    if "stage_arg" not in ext_kwargs:
        stage_key = stage.lower()
        if stage_key in ("basic", "hard"):
            ext_kwargs["stage_arg"] = module.BM4DStages.HARD_THRESHOLDING
        elif stage_key in ("wiener", "refine"):
            ext_kwargs["stage_arg"] = module.BM4DStages.WIENER_FILTERING
        else:
            ext_kwargs["stage_arg"] = module.BM4DStages.ALL_STAGES
    return bm4d_fn(np.asarray(array), sigma_psd, **ext_kwargs)

__all__ = [
    "denoise_filter",
    "denoise_bm3d",
    "denoise_bm4d",
    "regularized_denoise",
]
