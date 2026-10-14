"""Masked finite-difference TV operators and a box-constrained TV proximal map."""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, prod

import jax
import jax.numpy as jnp


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


def tv_prox(value, operators, rate, bounds, duals=None, *, tolerance=1e-7, max_steps=512):
    """Solve prox(rate * weighted TV + box indicator) with a checked dual gap.

    All TV groups on a field are solved jointly. The gap certifies the inner
    convex problem; this does not imply global convergence of a nonlinear fit.
    """
    lower, upper = bounds
    active = tuple(op for op in operators if op.axes)
    if not active:
        return jnp.clip(value, lower, upper), (), jnp.asarray(0.0), jnp.asarray(0)
    duals = (tuple(jnp.zeros_like(op.gradient(value)) for op in active)
             if duals is None else tuple(op.project_dual(p, rate) for op, p in zip(active, duals, strict=True)))
    norm_bound = sum(op.norm_bound_squared for op in active)
    initial_step = jnp.asarray(.99 / norm_bound ** .5, dtype=value.dtype)
    initial = jnp.clip(value, lower, upper)

    def objective_and_gap(x, ps):
        divergence = sum(op.adjoint(p) for op, p in zip(active, ps, strict=True))
        minimizer = jnp.clip(value - divergence, lower, upper)

        def gap_at(candidate):
            difference = candidate - minimizer
            gap = .5 * jnp.sum(difference**2) + jnp.sum((minimizer - value + divergence) * difference)
            for op, p in zip(active, ps, strict=True):
                grad = op.gradient(candidate)
                complementarity = rate * op.weights * vector_norm(grad) - jnp.sum(p * grad, axis=-1)
                gap = gap + jnp.sum(jnp.maximum(complementarity, 0))
            return jnp.maximum(gap, 0)

        original_gap, reconstructed_gap = gap_at(x), gap_at(minimizer)
        use_reconstruction = reconstructed_gap < original_gap
        candidate = jnp.where(use_reconstruction, minimizer, x)
        primal = .5 * jnp.sum((candidate - value)**2) + rate * sum(op.value(candidate) for op in active)
        return primal, jnp.minimum(original_gap, reconstructed_gap), candidate

    def body(state):
        x, extrapolated, ps, primal_step, dual_step, count, _, _, _ = state
        new_ps = tuple(op.project_dual(p + dual_step * op.gradient(extrapolated), rate)
                       for op, p in zip(active, ps, strict=True))
        divergence = sum(op.adjoint(p) for op, p in zip(active, new_ps, strict=True))
        candidate = jnp.clip((x - primal_step * divergence + primal_step * value) / (1 + primal_step), lower, upper)
        theta = 1 / jnp.sqrt(1 + 2 * primal_step)
        primal, gap, solution = objective_and_gap(candidate, new_ps)
        return (candidate, candidate + theta * (candidate - x), new_ps,
                primal_step * theta, dual_step / theta, count + 1, primal, gap, solution)

    def condition(state):
        return (state[5] < max_steps) & (state[7] > tolerance * (1 + state[6]))

    state = jax.lax.while_loop(condition, body, (initial, initial, duals, initial_step, initial_step,
                                               jnp.asarray(0), jnp.asarray(0., value.dtype),
                                               jnp.asarray(jnp.inf, value.dtype), initial))
    return state[8], state[2], state[7], state[5]
