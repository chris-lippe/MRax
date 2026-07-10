"""Low-level operators and shrinkage primitives used by reconstruction solvers."""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

Array = jnp.ndarray


def soft_threshold(x: Array, threshold: float) -> Array:
    """Element-wise soft thresholding (supports complex inputs)."""
    t = jnp.asarray(threshold)
    magnitude = jnp.abs(x)
    shrink = jnp.maximum(1.0 - t / (magnitude + 1e-12), 0.0)
    scale = jnp.where(t > 0, shrink, 1.0)
    return scale * x


def isotropic_shrinkage(v: Array, threshold: float) -> Array:
    """Vector soft thresholding used for isotropic total variation."""
    t = jnp.asarray(threshold)
    norm = jnp.linalg.norm(v, axis=-1, keepdims=True)
    shrink = jnp.maximum(1.0 - t / (norm + 1e-12), 0.0)
    scale = jnp.where(t > 0, shrink, 1.0)
    return v * scale


def forward_gradient2d(x: Array) -> Array:
    """Forward finite differences with zero Neumann boundary."""
    dx = jnp.pad(jnp.diff(x, axis=0), ((0, 1), (0, 0)))
    dy = jnp.pad(jnp.diff(x, axis=1), ((0, 0), (0, 1)))
    return jnp.stack([dx, dy], axis=-1)


def divergence2d(v: Array) -> Array:
    """Adjoint of forward_gradient2d (negative divergence)."""
    px, py = v[..., 0], v[..., 1]
    px_pad = jnp.pad(px, ((1, 0), (0, 0)))
    py_pad = jnp.pad(py, ((0, 0), (1, 0)))
    return px_pad[1:] - px_pad[:-1] + py_pad[:, 1:] - py_pad[:, :-1]


def conjugate_gradient(
    matvec: Callable[[Array], Array],
    b: Array,
    x0: Array | None = None,
    max_iters: int = 25,
    tol: float = 1e-6,
) -> Array:
    """JAX-compatible conjugate gradient solver for SPD systems."""
    b = jnp.asarray(b)
    x0 = jnp.zeros_like(b) if x0 is None else x0

    r0 = b - matvec(x0)
    p0 = r0
    rs0 = jnp.real(jnp.vdot(r0, r0))

    def cond_fun(val):
        i, _, r, _, rs = val
        return jnp.logical_and(i < max_iters, jnp.sqrt(rs) > tol)

    def body_fun(val):
        i, x, r, p, rsold = val
        Ap = matvec(p)
        alpha = rsold / jnp.vdot(p, Ap)
        x_new = x + alpha * p
        r_new = r - alpha * Ap
        rsnew = jnp.real(jnp.vdot(r_new, r_new))
        beta = rsnew / rsold
        p_new = r_new + beta * p
        return i + 1, x_new, r_new, p_new, rsnew

    init = (0, x0, r0, p0, rs0)
    _, x_final, _, _, _ = jax.lax.while_loop(cond_fun, body_fun, init)
    return x_final


__all__ = [
    "Array",
    "conjugate_gradient",
    "divergence2d",
    "forward_gradient2d",
    "isotropic_shrinkage",
    "soft_threshold",
]
