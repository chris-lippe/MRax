"""Linear NUFFT differentiation rules with bounded reconstruction workspace.

The primitive lowers entirely to the portable JAX implementation. Explicit
linear rules prevent autodiff from saving sample-by-neighbour scan residuals.
No native extension or platform-specific lowering is used.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.core import ShapedArray
from jax.extend.core import Primitive
from jax.interpreters import ad, batching, mlir


def _implementation(values, indices, weights, *deapodization, configuration):
    from .nufft import NUFFTPlan

    # Large constant cotangents must not run FFT/spreading in XLA's constant
    # folder. Linear differentiation is defined explicitly below.
    values = jax.lax.optimization_barrier(values)
    shape, grid, count, width, chunk, batch, cached, beta, wide, adjoint = configuration
    plan = NUFFTPlan(shape, grid, count, width, chunk, batch, indices, weights,
                     tuple(deapodization), cached, beta, wide)
    return plan._apply_raw(values, adjoint=adjoint)


def _abstract(values, indices, weights, *deapodization, configuration):
    shape, _, count, _, _, _, _, _, _, adjoint = configuration
    input_shape = (count,) if adjoint else shape
    if tuple(values.shape[-len(input_shape):]) != input_shape:
        raise ValueError("NUFFT input shape does not match the prepared plan")
    output = shape if adjoint else (count,)
    dtype = jnp.result_type(values.dtype, weights.dtype, jnp.complex64)
    return ShapedArray(values.shape[:-len(input_shape)] + output, dtype)


_nufft = Primitive("mrax_nufft")
_nufft.def_impl(jax.jit(_implementation, static_argnames=("configuration",)))
_nufft.def_abstract_eval(_abstract)
mlir.register_lowering(_nufft, mlir.lower_fun(_implementation, multiple_results=False))


def _jvp(primals, tangents, *, configuration):
    if any(not isinstance(tangent, ad.Zero) for tangent in tangents[1:]):
        raise ValueError("NUFFT plans require fixed trajectories and stencil coefficients")
    result = _nufft.bind(*primals, configuration=configuration)
    tangent = tangents[0]
    if isinstance(tangent, ad.Zero):
        tangent = ad.Zero(ShapedArray(result.shape, result.dtype))
    else:
        tangent = _nufft.bind(tangent, *primals[1:], configuration=configuration)
    return result, tangent


def _transpose(cotangent, values, *coefficients, configuration):
    if any(ad.is_undefined_primal(value) for value in coefficients):
        raise ValueError("NUFFT plans require fixed trajectories and stencil coefficients")
    if isinstance(cotangent, ad.Zero):
        result = ad.Zero(values.aval)
    else:
        opposite = configuration[:-1] + (not configuration[-1],)
        # JAX uses a bilinear transpose for complex cotangents; conjugation
        # converts the mathematical Hermitian adjoint into that convention.
        result = jnp.conj(_nufft.bind(jnp.conj(cotangent), *coefficients,
                                    configuration=opposite))
        if not jnp.issubdtype(values.aval.dtype, jnp.complexfloating):
            result = jnp.real(result)
        result = result.astype(values.aval.dtype)
    return (result,) + (None,) * len(coefficients)


def _batch(arguments, axes, *, configuration):
    if any(axis is not None for axis in axes[1:]):
        raise ValueError("NUFFT vmap requires a shared fixed-trajectory plan")
    values = arguments[0]
    axis = axes[0]
    if axis is not None:
        values = jnp.moveaxis(values, axis, 0)
        axis = 0
    return _nufft.bind(values, *arguments[1:], configuration=configuration), axis


ad.primitive_jvps[_nufft] = _jvp
ad.primitive_transposes[_nufft] = _transpose
batching.primitive_batchers[_nufft] = _batch


def apply_nufft(plan, values, *, adjoint):
    """Bind a fixed plan to its portable implementation and linear AD rules."""
    configuration = (plan.spatial_shape, plan.grid_shape, plan.sample_count,
        plan.width, plan.chunk_size, plan.batch_size, plan.cached_stencil,
        plan.shape_parameter, plan.prefer_wide_accumulation, adjoint)
    return _nufft.bind(jnp.asarray(values), plan.indices, plan.weights,
                       *plan.deapodization, configuration=configuration)
