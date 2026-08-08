"""JAX-native spatial phase-unwrapping utilities."""

from __future__ import annotations

import math
from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.fft import dctn, idctn
from jax.scipy.sparse.linalg import cg
from jax.typing import ArrayLike


def _principal_difference(value: jax.Array, period: float) -> jax.Array:
    half_period = period / 2.0
    wrapped = jnp.mod(value + half_period, period) - half_period
    return jnp.where(
        (wrapped == -half_period) & (value > 0),
        half_period,
        wrapped,
    )


def _unwrap_axis(values: jax.Array, *, axis: int, period: float) -> jax.Array:
    """Integrate principal wrapped differences along one axis."""

    delta = _principal_difference(jnp.diff(values, axis=axis), period)
    anchor = jnp.take(values, jnp.asarray([0]), axis=axis)
    return jnp.concatenate((anchor, anchor + jnp.cumsum(delta, axis=axis)), axis=axis)


def _divergence(edges: list[jax.Array], shape: tuple[int, ...]) -> jax.Array:
    """Apply the adjoint of the forward-difference operator."""

    result = jnp.zeros(shape, dtype=edges[0].dtype)
    for axis, edge in enumerate(edges):
        before = [(0, 0)] * len(shape)
        after = [(0, 0)] * len(shape)
        before[axis] = (1, 0)
        after[axis] = (0, 1)
        result += jnp.pad(edge, before) - jnp.pad(edge, after)
    return result


def _laplacian_eigenvalues(shape: tuple[int, ...], dtype: jnp.dtype) -> jax.Array:
    """Return Neumann-Laplacian eigenvalues for an orthonormal DCT-II."""

    eigenvalues = jnp.zeros(shape, dtype=dtype)
    for axis, size in enumerate(shape):
        axis_shape = [1] * len(shape)
        axis_shape[axis] = size
        frequency = jnp.arange(size, dtype=dtype)
        values = 2.0 - 2.0 * jnp.cos(jnp.pi * frequency / float(size))
        eigenvalues += values.reshape(axis_shape)
    return eigenvalues


def _unweighted_solution(phase: jax.Array, *, period: float) -> tuple[jax.Array, list[jax.Array]]:
    gradients = [
        _principal_difference(jnp.diff(phase, axis=axis), period)
        for axis in range(phase.ndim)
    ]
    rhs = _divergence(gradients, phase.shape)
    eigenvalues = _laplacian_eigenvalues(phase.shape, phase.dtype)
    coefficients = dctn(rhs, type=2, norm="ortho")
    coefficients = jnp.where(
        eigenvalues == 0,
        0,
        coefficients / jnp.where(eigenvalues == 0, 1, eigenvalues),
    )
    return idctn(coefficients, type=2, norm="ortho"), gradients


def _unwrap_least_squares(
    phase: jax.Array,
    weights: jax.Array | None,
    *,
    period: float,
    weight_floor: float,
    weight_power: float,
    max_iterations: int,
    tolerance: float,
) -> jax.Array:
    """Unwrap one spatial image with a DCT/weighted least-squares solve."""

    estimate, gradients = _unweighted_solution(phase, period=period)
    if weights is not None:
        edge_weights: list[jax.Array] = []
        diagonal = jnp.zeros_like(phase)
        for axis in range(phase.ndim):
            before = [slice(None)] * phase.ndim
            after = [slice(None)] * phase.ndim
            before[axis] = slice(None, -1)
            after[axis] = slice(1, None)
            edge_weight = weight_floor + (
                weights[tuple(before)] * weights[tuple(after)]
            ) ** weight_power
            edge_weights.append(edge_weight)
            pad_before = [(0, 0)] * phase.ndim
            pad_after = [(0, 0)] * phase.ndim
            pad_before[axis] = (1, 0)
            pad_after[axis] = (0, 1)
            diagonal += jnp.pad(edge_weight, pad_before) + jnp.pad(edge_weight, pad_after)

        rhs = _divergence(
            [weight * gradient for weight, gradient in zip(edge_weights, gradients, strict=True)],
            phase.shape,
        )

        def normal_operator(value: jax.Array) -> jax.Array:
            weighted_gradient = [
                weight * jnp.diff(value, axis=axis)
                for axis, weight in enumerate(edge_weights)
            ]
            # The mean term fixes the otherwise arbitrary constant phase offset.
            return _divergence(weighted_gradient, phase.shape) + jnp.mean(value)

        estimate, _ = cg(
            normal_operator,
            rhs,
            x0=estimate,
            M=lambda residual: residual / (diagonal + 1e-6),
            tol=tolerance,
            maxiter=max_iterations,
        )
        reference = jnp.sum(weights * jnp.exp(1j * (phase - estimate)))
    else:
        reference = jnp.sum(jnp.exp(1j * (phase - estimate)))

    estimate += jnp.angle(reference)
    # Project back onto the input phase's exact modulo-period equivalence class.
    return phase + period * jnp.round((estimate - phase) / period)


@partial(
    jax.jit,
    static_argnames=(
        "spatial_ndim",
        "period",
        "weight_floor",
        "weight_power",
        "max_iterations",
        "tolerance",
    ),
)
def unwrap_phase_image(
    phase: ArrayLike,
    *,
    spatial_ndim: int | None = None,
    weights: ArrayLike | None = None,
    period: float = math.tau,
    weight_floor: float = 1e-3,
    weight_power: float = 4.0,
    max_iterations: int = 50,
    tolerance: float = 1e-5,
) -> jax.Array:
    """Unwrap trailing spatial axes independently for every leading index.

    Two- and three-dimensional data use a fast cosine-transform least-squares
    initializer. When nonnegative ``weights`` are supplied, a fixed-budget
    preconditioned conjugate-gradient refinement emphasizes reliable spatial
    edges and confines inconsistencies to low-weight regions. One-dimensional
    data use direct Itoh integration.

    The result differs from ``phase`` only by integer multiples of ``period``.
    Leading frame, echo, observable, and coil axes are never connected.

    Parameters
    ----------
    phase:
        Real wrapped phase values in radians.
    spatial_ndim:
        Number of trailing spatial axes to unwrap (1, 2, or 3). If omitted, all
        axes are treated as spatial, up to three dimensions.
    weights:
        Optional nonnegative phase-reliability weights broadcastable to
        ``phase``. Reconstruction callers should use image magnitude. Weights
        are normalized independently for each leading batch index.
    period:
        Phase period in the same units as ``phase``. The default is ``2 * pi``.
    weight_floor, weight_power:
        Reliability floor and contrast used for spatial edge weights.
    max_iterations, tolerance:
        Fixed work limit and convergence tolerance for weighted refinement.

    References
    ----------
    D. C. Ghiglia and L. A. Romero, "Robust two-dimensional weighted and
    unweighted phase unwrapping that uses fast transforms and iterative
    methods," JOSA A 11(1), 107-117 (1994),
    https://doi.org/10.1364/JOSAA.11.000107.
    """

    values = jnp.asarray(phase)
    if jnp.issubdtype(values.dtype, jnp.complexfloating):
        raise ValueError("phase must contain real wrapped phase values")
    if not jnp.issubdtype(values.dtype, jnp.inexact) or values.dtype.itemsize < 4:
        values = values.astype(jnp.float32)
    if values.ndim == 0:
        return values
    ndim = values.ndim if spatial_ndim is None else int(spatial_ndim)
    if ndim < 1 or ndim > 3 or ndim > values.ndim:
        raise ValueError(
            "spatial_ndim must be between 1 and 3 and cannot exceed the phase array dimensions."
        )
    if any(int(size) == 0 for size in values.shape[-ndim:]):
        raise ValueError("spatial phase axes cannot be empty")
    if not math.isfinite(period) or period <= 0:
        raise ValueError("period must be a positive finite value")
    if not math.isfinite(weight_floor) or weight_floor <= 0:
        raise ValueError("weight_floor must be a positive finite value")
    if not math.isfinite(weight_power) or weight_power <= 0:
        raise ValueError("weight_power must be a positive finite value")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be a positive finite value")

    if ndim == 1:
        return _unwrap_axis(values, axis=-1, period=period)

    spatial_shape = values.shape[-ndim:]
    batch_shape = values.shape[:-ndim]
    flat_phase = values.reshape((-1,) + spatial_shape)
    flat_weights = None
    if weights is not None:
        weight_values = jnp.broadcast_to(jnp.asarray(weights, dtype=values.dtype), values.shape)
        weight_values = jnp.nan_to_num(jnp.abs(weight_values), nan=0.0, posinf=0.0, neginf=0.0)
        spatial_axes = tuple(range(values.ndim - ndim, values.ndim))
        rms = jnp.sqrt(jnp.mean(weight_values**2, axis=spatial_axes, keepdims=True))
        weight_values = jnp.clip(weight_values / jnp.maximum(3.0 * rms, 1e-6), 0.0, 1.0)
        flat_weights = weight_values.reshape((-1,) + spatial_shape)

    def solve_one(single_phase: jax.Array, single_weights: jax.Array | None) -> jax.Array:
        return _unwrap_least_squares(
            single_phase,
            single_weights,
            period=period,
            weight_floor=weight_floor,
            weight_power=weight_power,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )

    if flat_phase.shape[0] == 1:
        output = solve_one(flat_phase[0], None if flat_weights is None else flat_weights[0])[None]
    elif flat_weights is None:
        output = jax.lax.map(lambda item: solve_one(item, None), flat_phase)
    else:
        output = jax.lax.map(lambda item: solve_one(item[0], item[1]), (flat_phase, flat_weights))
    return output.reshape(batch_shape + spatial_shape)


__all__ = ["unwrap_phase_image"]
