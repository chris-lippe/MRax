"""Limited-memory inverse curvature for a split-Bregman primal subproblem.

Each secant uses two gradients with the same splits and multipliers. Their
linear terms cancel, so history approximates the unchanged primal Hessian
even when the next outer iteration changes the Bregman multiplier.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import optax


def _dot(left, right):
    return sum(jnp.real(jnp.vdot(a, b)) for a, b in
               zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True))


def initial_history(point, capacity):
    dtype = jnp.result_type(*[jnp.real(x).dtype for x in point.values()], jnp.float32)
    return dict(s={k: jnp.zeros((capacity,) + x.shape, x.dtype) for k, x in point.items()},
                y={k: jnp.zeros((capacity,) + x.shape, x.dtype) for k, x in point.items()},
                inverse_curvature=jnp.zeros((capacity,), dtype),
                next=jnp.asarray(0, jnp.int32), count=jnp.asarray(0, jnp.int32),
                updates=jnp.asarray(0, jnp.int32), skipped=jnp.asarray(0, jnp.int32),
                resets=jnp.asarray(0, jnp.int32))


def inverse_product(value, history, initial_inverse):
    """Standard L-BFGS two-loop recursion with an SPD initial inverse."""
    capacity = history['inverse_curvature'].size
    coefficients = jnp.zeros_like(history['inverse_curvature'])

    def backward(index, carry):
        q, coefficients = carry
        slot = (history['next'] - 1 - index) % capacity
        s = jax.tree.map(lambda x: x[slot], history['s'])
        y = jax.tree.map(lambda x: x[slot], history['y'])
        alpha = history['inverse_curvature'][slot] * _dot(s, q)
        alpha = jnp.where(index < history['count'], alpha, 0.)
        q = jax.tree.map(lambda a, b: a - alpha.astype(jnp.real(a).dtype) * b, q, y)
        return q, coefficients.at[slot].set(alpha)

    q, coefficients = jax.lax.fori_loop(0, capacity, backward, (value, coefficients))
    result = initial_inverse(q)

    def forward(index, result):
        slot = (history['next'] - history['count'] + index) % capacity
        s = jax.tree.map(lambda x: x[slot], history['s'])
        y = jax.tree.map(lambda x: x[slot], history['y'])
        beta = history['inverse_curvature'][slot] * _dot(y, result)
        amount = jnp.where(index < history['count'], coefficients[slot] - beta, 0.)
        return jax.tree.map(lambda a, b: a + amount.astype(jnp.real(a).dtype) * b, result, s)

    return jax.lax.fori_loop(0, capacity, forward, result)


def update_history(history, displacement, gradient_change, accepted):
    """Retain finite, positively curved secants from one primal objective."""
    dtype = history['inverse_curvature'].dtype
    scale_s = jnp.max(jnp.stack([jnp.max(jnp.abs(x), initial=0.) for x in displacement.values()]))
    scale_y = jnp.max(jnp.stack([jnp.max(jnp.abs(x), initial=0.) for x in gradient_change.values()]))
    safe_s, safe_y = jnp.where(scale_s > 0, scale_s, 1.), jnp.where(scale_y > 0, scale_y, 1.)
    s = jax.tree.map(lambda x: x / safe_s.astype(jnp.real(x).dtype), displacement)
    y = jax.tree.map(lambda x: x / safe_y.astype(jnp.real(x).dtype), gradient_change)
    cosine = _dot(s, y) / jnp.sqrt(jnp.maximum(_dot(s, s) * _dot(y, y), jnp.finfo(dtype).tiny))
    curvature = _dot(s, y) * scale_s * scale_y
    finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in
                                jax.tree.leaves((displacement, gradient_change))]))
    eps = max([jnp.finfo(dtype).eps] + [jnp.finfo(jnp.real(x).dtype).eps
                                       for x in displacement.values()])
    valid = (accepted & finite & (scale_s > 0) & (scale_y > 0)
             & (cosine > jnp.sqrt(eps)) & jnp.isfinite(curvature)
             & (curvature > 32 * jnp.finfo(dtype).tiny))

    def store(_):
        slot = history['next']
        return {**history,
                's': jax.tree.map(lambda a, b: a.at[slot].set(b), history['s'], displacement),
                'y': jax.tree.map(lambda a, b: a.at[slot].set(b), history['y'], gradient_change),
                'inverse_curvature': history['inverse_curvature'].at[slot].set(1 / curvature),
                'next': (slot + 1) % history['inverse_curvature'].size,
                'count': jnp.minimum(history['count'] + 1, history['inverse_curvature'].size),
                'updates': optax.safe_int32_increment(history['updates'])}

    return jax.lax.cond(valid, store,
        lambda _: {**history, 'skipped': history['skipped'] + accepted.astype(jnp.int32)}, None)
