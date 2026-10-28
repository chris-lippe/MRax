"""Differentiable bounded-influence data losses for any fitting optimizer."""
from __future__ import annotations

from operator import index

import jax.numpy as jnp
import numpy as np


def pseudo_huber_loss(residual, weights, *, delta, group_size=1):
    """Weighted pseudo-Huber loss, optionally grouping consecutive frames.

    ``delta`` is a positive cutoff in the RMS-scaled residual's units. With
    ``group_size > 1``, the whole group's weighted RMS determines its common
    influence weight. The first dimension must contain consecutive groups;
    other dimensions remain independent. Near zero, this agrees with weighted
    mean squared error. It approaches linear growth in the group's RMS for
    large residuals. The rationalized expression avoids cancellation at zero.
    """
    group_size = index(group_size)
    if group_size < 1:
        raise ValueError("group_size must be positive")
    if np.ndim(delta) != 0 or not np.isfinite(delta) or delta <= 0:
        raise ValueError("delta must be finite and positive")
    residual, weights = jnp.asarray(residual), jnp.asarray(weights)
    weights = jnp.broadcast_to(weights, residual.shape)
    if not residual.ndim or residual.shape[0] % group_size:
        raise ValueError("The first residual dimension must contain complete groups")
    shape = (-1, group_size, *residual.shape[1:])
    safe = jnp.where(weights > 0, residual, 0).reshape(shape)
    w = weights.reshape(shape)
    count = jnp.sum(w, axis=1)
    squared = jnp.sum(w * jnp.real(safe * jnp.conj(safe)), axis=1) / jnp.maximum(count, 1e-30)
    value = 2 * squared / (jnp.sqrt(1 + squared / delta**2) + 1)
    return jnp.sum(count * value) / jnp.maximum(jnp.sum(count), 1e-30)
