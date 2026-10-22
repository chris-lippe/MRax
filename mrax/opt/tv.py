"""Masked finite-difference TV operators and a box-constrained TV proximal map."""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, prod

import jax
import jax.numpy as jnp
import numpy as np


@jax.custom_jvp
def vector_norm(values):
    return jnp.sqrt(jnp.sum(jnp.abs(values) ** 2, axis=-1))


@vector_norm.defjvp
def _vector_norm_jvp(primals, tangents):
    values, = primals
    tangent, = tangents
    norm = vector_norm(values)
    derivative = jnp.sum(jnp.real(jnp.conj(values) * tangent), axis=-1) / jnp.where(norm > 0, norm, 1)
    return norm, jnp.where(norm > 0, derivative, 0)


def difference(values, axis, order):
    result = jnp.diff(values, n=order, axis=axis)
    padding = [(0, 0)] * values.ndim
    padding[axis] = (0, order)
    return jnp.pad(result, padding)


def difference_adjoint(values, axis, order):
    slices = [slice(None)] * values.ndim
    slices[axis] = slice(0, values.shape[axis] - order)
    result = values[tuple(slices)]
    for _ in range(order):
        left = [(0, 0)] * values.ndim
        right = list(left)
        left[axis], right[axis] = (1, 0), (0, 1)
        result = jnp.pad(result, left) - jnp.pad(result, right)
    return result


@dataclass(frozen=True)
class TVOperator:
    """Isotropic weighted TV; invalid edges never couple across a mask boundary."""
    axes: tuple[int, ...]
    order: int
    valid: tuple[jax.Array, ...]
    weights: jax.Array

    @classmethod
    def create(cls, shape, axes=None, order=1, mask=None, weight=1.0, dtype=jnp.float32):
        if order < 1 or weight < 0 or not isfinite(weight):
            raise ValueError("TV order must be positive and its weight finite and nonnegative")
        selected = tuple(range(len(shape))) if axes is None else tuple(axes)
        if any(axis < -len(shape) or axis >= len(shape) for axis in selected):
            raise ValueError("TV axis is outside the field dimensions")
        selected = tuple(dict.fromkeys(axis % len(shape) for axis in selected if shape[axis] > order))
        if not selected:
            return cls((), order, (), jnp.zeros(shape, dtype=dtype))
        if mask is None:
            valid = []
            for axis in selected:
                edges = jnp.ones(shape, dtype=dtype)
                slices = [slice(None)] * len(shape)
                slices[axis] = slice(shape[axis] - order, None)
                valid.append(edges.at[tuple(slices)].set(0))
            weights = jnp.full(shape, weight / prod(shape), dtype=dtype)
        else:
            mask = jnp.broadcast_to(jnp.asarray(mask, dtype=dtype), shape)
            valid = []
            for axis in selected:
                edge_shape = list(shape)
                edge_shape[axis] -= order
                edges = jnp.ones(edge_shape, dtype=dtype)
                for offset in range(order + 1):
                    slices = [slice(None)] * len(shape)
                    slices[axis] = slice(offset, shape[axis] - order + offset)
                    edges = edges * mask[tuple(slices)]
                padding = [(0, 0)] * len(shape)
                padding[axis] = (0, order)
                valid.append(jnp.pad(edges, padding))
            mean_weight = jnp.mean(jnp.stack(valid, axis=-1), axis=-1)
            weights = weight * mean_weight / (jnp.sum(mean_weight) + 1e-8)
        return cls(selected, order, tuple(value > 0 for value in valid), weights)

    @property
    def norm_bound_squared(self):
        return len(self.axes) * 4.0 ** self.order

    def gradient(self, field):
        if not self.axes:
            return jnp.zeros(field.shape + (0,), dtype=field.dtype)
        return jnp.stack([difference(field, axis, self.order) * valid
                          for axis, valid in zip(self.axes, self.valid, strict=True)], axis=-1)

    def adjoint(self, dual):
        result = jnp.zeros(self.weights.shape, dtype=dual.dtype)
        for index, (axis, valid) in enumerate(zip(self.axes, self.valid, strict=True)):
            result = result + difference_adjoint(dual[..., index] * valid, axis, self.order)
        return result

    def value(self, field, smoothed=False):
        grad = self.gradient(field)
        norm = (jnp.sqrt(jnp.sum(jnp.abs(grad) ** 2, axis=-1) + 1e-12) - 1e-6
                if smoothed else vector_norm(grad))
        return jnp.sum(self.weights * norm)

    def project_dual(self, dual, multiplier=1.0):
        radius = multiplier * self.weights
        norm = vector_norm(dual)
        scale = jnp.minimum(1.0, radius / jnp.where(norm > 0, norm, 1))
        return dual * scale[..., None]


@dataclass(frozen=True)
class L1Operator:
    """Elementwise magnitude penalty sharing the certified proximal interface."""

    weights: jax.Array
    axes: tuple[int, ...] = (0,)
    norm_bound_squared: float = 1.0

    def gradient(self, value):
        return value[..., None]

    def adjoint(self, value):
        return value[..., 0]

    def value(self, value, smoothed=False):
        return jnp.sum(self.weights * jnp.abs(value))

    def project_dual(self, value, multiplier=1.0):
        bound = multiplier * self.weights
        norm = vector_norm(value)
        scale = jnp.minimum(1., bound / jnp.where(norm > 0, norm, 1.))
        return value * scale[..., None]


@dataclass(frozen=True)
class CallbackOperator:
    """Identity splitting for an application-supplied convex proximal map.

    The callback owns its penalty definition; its value is unavailable. Thus
    fitting uses smooth majorization and residual checks, not a composite-value
    monotonicity claim. Thresholds retain the public scalar/array/tree semantics.
    """
    weights: jax.Array
    callback: object
    axes: tuple[int, ...] = (0,)
    norm_bound_squared: float = 1.

    def gradient(self, value):
        return value[..., None]

    def adjoint(self, dual):
        return dual[..., 0]

    def prox(self, value, multiplier):
        return self.callback(value[..., 0], multiplier * self.weights)[..., None]

    def project_dual(self, dual, multiplier=1.):
        return dual - multiplier * self.prox(dual / multiplier, 1 / multiplier)

    def value(self, value, smoothed=False):
        return jnp.asarray(0., jnp.real(value).dtype)


def tv_prox(value, operators, rate, bounds, duals=None, *, tolerance=1e-7, max_steps=512):
    """Solve prox(rate * weighted TV + box indicator) with a checked dual gap.

    All TV groups on a field are solved jointly. The gap certifies the inner
    convex problem; this does not imply global convergence of a nonlinear fit.
    """
    lower, upper = bounds
    def project(array):
        if jnp.iscomplexobj(array):
            if np.any(np.isfinite(lower)) or np.any(np.isfinite(upper)):
                raise ValueError("Complex TV fields require unbounded box constraints")
            return array
        return jnp.clip(array, lower, upper).astype(array.dtype)

    active = tuple(op for op in operators if op.axes)
    if not active:
        return project(value), (), jnp.asarray(0.0, dtype=jnp.real(value).dtype), jnp.asarray(0)
    if len(active) == 1 and isinstance(active[0], CallbackOperator):
        op = active[0]
        # Dykstra computes prox(g + box), including nonseparable custom proxs;
        # a simple clip(prox(g)) is generally not that proximal map.
        def body(state):
            x, p, q, count, _ = state
            y = op.prox((x + p)[..., None], rate)[..., 0]
            p = x + p - y
            candidate = project(y + q)
            q = y + q - candidate
            error = jnp.maximum(jnp.max(jnp.abs(candidate - x)),
                                jnp.max(jnp.abs(candidate - y)))
            return candidate, p, q, count + 1, error
        zero = jnp.zeros_like(value)
        x, p, _, count, error = jax.lax.while_loop(
            lambda s: (s[3] < max_steps) & (s[4] > tolerance * (1 + jnp.max(jnp.abs(s[0])))),
            body, (value, zero, zero, jnp.asarray(0), jnp.asarray(jnp.inf, jnp.real(value).dtype)))
        valid = error <= tolerance * (1 + jnp.max(jnp.abs(x)))
        # This is a fixed-point certificate, not a dual objective gap.
        return x, (op.project_dual(p[..., None], rate),), jnp.where(valid, 0., jnp.inf), count
    if len(active) == 1 and isinstance(active[0], L1Operator):
        # Exact separable solution avoids an unnecessary iterative inner solve
        # and its sqrt(tolerance) error near an L1 kink.
        radius = rate * active[0].weights
        if jnp.iscomplexobj(value):
            norm = jnp.abs(value)
            result = value * jnp.maximum(1 - radius / jnp.where(norm > 0, norm, 1), 0)
        else:
            result = project(jnp.sign(value) * jnp.maximum(jnp.abs(value) - radius, 0))
        dual = active[0].project_dual((value - result)[..., None], rate)
        return result, (dual,), jnp.asarray(0., jnp.real(value).dtype), jnp.asarray(0)
    duals = (tuple(jnp.zeros_like(op.gradient(value)) for op in active)
             if duals is None else tuple(op.project_dual(p, rate) for op, p in zip(active, duals, strict=True)))
    norm_bound = sum(op.norm_bound_squared for op in active)
    real_dtype = jnp.real(value).dtype
    initial_step = jnp.asarray(.99 / norm_bound ** .5, dtype=real_dtype)
    initial = project(value)

    def objective_and_gap(x, ps):
        divergence = sum(op.adjoint(p) for op, p in zip(active, ps, strict=True))
        minimizer = project(value - divergence)

        def gap_at(candidate):
            difference = candidate - minimizer
            gap = .5 * jnp.sum(jnp.abs(difference)**2) + jnp.real(jnp.vdot(minimizer - value + divergence, difference))
            for op, p in zip(active, ps, strict=True):
                grad = op.gradient(candidate)
                complementarity = rate * op.weights * vector_norm(grad) - jnp.real(jnp.sum(jnp.conj(p) * grad, axis=-1))
                gap = gap + jnp.sum(jnp.maximum(complementarity, 0))
            return jnp.maximum(gap, 0)

        original_gap, reconstructed_gap = gap_at(x), gap_at(minimizer)
        use_reconstruction = reconstructed_gap < original_gap
        candidate = jnp.where(use_reconstruction, minimizer, x)
        primal = .5 * jnp.sum(jnp.abs(candidate - value)**2) + rate * sum(op.value(candidate) for op in active)
        return primal, jnp.minimum(original_gap, reconstructed_gap), candidate

    def body(state):
        x, extrapolated, ps, primal_step, dual_step, count, _, _, _ = state
        new_ps = tuple(op.project_dual(p + dual_step * op.gradient(extrapolated), rate)
                       for op, p in zip(active, ps, strict=True))
        divergence = sum(op.adjoint(p) for op, p in zip(active, new_ps, strict=True))
        candidate = project((x - primal_step * divergence + primal_step * value) / (1 + primal_step))
        theta = 1 / jnp.sqrt(1 + 2 * primal_step)
        primal, gap, solution = objective_and_gap(candidate, new_ps)
        return (candidate, candidate + theta * (candidate - x), new_ps,
                primal_step * theta, dual_step / theta, count + 1, primal, gap, solution)

    def condition(state):
        return (state[5] < max_steps) & (state[7] > tolerance * (1 + state[6]))

    state = jax.lax.while_loop(condition, body, (initial, initial, duals, initial_step, initial_step,
                                               jnp.asarray(0), jnp.asarray(0., real_dtype),
                                               jnp.asarray(jnp.inf, real_dtype), initial))
    return state[8], state[2], state[7], state[5]
