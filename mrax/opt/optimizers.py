"""Optax-friendly wrappers and proximal variants."""

from __future__ import annotations

from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from mrax.devices import DeviceLike, device_function, place_optimizer

from .ops import soft_threshold

LearningRate = float | Callable[[jax.Array], jax.Array]


def _align_tree(example, value):
    """Broadcast ``value`` to the structure of ``example`` if needed."""
    if jax.tree.structure(example) == jax.tree.structure(value):
        return value
    if isinstance(value, (dict, list, tuple)):
        raise ValueError("Optimizer option tree must match the parameter tree")
    return jax.tree.map(lambda _: value, example)


def _validate_proximal_weights(weights):
    for leaf in jax.tree_util.tree_leaves(weights):
        value = np.asarray(leaf)
        if not np.all(np.isfinite(value) & (value >= 0)):
            raise ValueError("Proximal regularizer weights must be finite and nonnegative")


def _any_positive(weights) -> bool:
    leaves, _ = jax.tree_util.tree_flatten(weights)
    return any(np.any(np.asarray(leaf) > 0.0) for leaf in leaves)


def _with_box_constraints(opt, bounds):
    if bounds is None:
        return opt

    def update_fn(grads, state, params):
        updates, new_state = opt.update(grads, state, params)
        projected = _project_bounds(optax.apply_updates(params, updates), bounds)
        return jax.tree_util.tree_map(lambda new, old: new - old, projected, params), new_state

    def init_fn(params):
        lower, upper = bounds
        lowers, uppers = _align_tree(params, lower), _align_tree(params, upper)
        for x, lo, hi in zip(jax.tree.leaves(params), jax.tree.leaves(lowers), jax.tree.leaves(uppers), strict=True):
            lo, hi = np.asarray(lo), np.asarray(hi)
            if np.any(np.isnan(lo)) or np.any(np.isnan(hi)) or np.any(lo > hi):
                raise ValueError("Box bounds must be ordered and cannot contain NaN")
            if np.broadcast_shapes(np.shape(x), np.shape(lo), np.shape(hi)) != np.shape(x):
                raise ValueError("Box bounds must broadcast to each parameter shape")
        return opt.init(params)

    return optax.GradientTransformation(init_fn, update_fn)


def _project_bounds(params, bounds: Tuple[object, object] | None):
    if bounds is None:
        return params
    lower, upper = bounds
    lower_tree = _align_tree(params, lower)
    upper_tree = _align_tree(params, upper)

    def clip_fn(x, lo, hi):
        return jnp.clip(x, lo, hi)

    return jax.tree_util.tree_map(clip_fn, params, lower_tree, upper_tree)


def _make_split_bregman_optimizer(step_size, regularizer_weight, prox_fn, box_bounds, penalty):
    """Linearized split Bregman with its augmented-Lagrangian coupling."""
    def init_fn(params):
        return {"split": params, "breg": jax.tree.map(jnp.zeros_like, params),
                "count": jnp.zeros((), dtype=jnp.int32)}

    def update_fn(grads, state, params):
        rate = step_size(state["count"]) if callable(step_size) else step_size
        coupled = jax.tree.map(lambda g, x, d, b: g + penalty * (x - d + b),
                               grads, params, state["split"], state["breg"])
        primal = _project_bounds(jax.tree.map(lambda x, g: x - rate * g, params, coupled), box_bounds)
        weights = _align_tree(primal, regularizer_weight)
        proxs = _align_tree(primal, prox_fn)
        split = jax.tree.map(lambda x, b, w, prox: prox(x + b, w / penalty),
                             primal, state["breg"], weights, proxs)
        breg = jax.tree.map(lambda b, x, d: b + x - d, state["breg"], primal, split)
        updates = jax.tree.map(lambda new, old: new - old, primal, params)
        return updates, {"split": split, "breg": breg, "count": optax.safe_int32_increment(state["count"])}

    return optax.GradientTransformation(init_fn, update_fn)


def _make_fista_optimizer(step_size, regularizer_weight, prox_fn, box_bounds):
    """Euclidean FISTA. Caller gradients are evaluated at the extrapolated point."""
    def init_fn(params):
        leaves = jax.tree.leaves(params)
        dtype = jnp.result_type(*[jnp.real(jnp.asarray(x)).dtype for x in leaves]) if leaves else jnp.float32
        return {"x": params, "t": jnp.asarray(1.0, dtype=dtype), "count": jnp.zeros((), jnp.int32)}

    def update_fn(grads, state, params):
        rate = step_size(state["count"]) if callable(step_size) else step_size
        forward = jax.tree.map(lambda x, g: x - rate * g, params, grads)
        weights = _align_tree(forward, regularizer_weight)
        proxs = _align_tree(forward, prox_fn)
        accepted = jax.tree.map(lambda x, w, prox: prox(x, w * rate), forward, weights, proxs)
        accepted = _project_bounds(accepted, box_bounds)
        t = .5 * (1 + jnp.sqrt(1 + 4 * state["t"]**2))
        momentum = (state["t"] - 1) / t
        extrapolated = jax.tree.map(lambda x, old: x + momentum * (x - old), accepted, state["x"])
        extrapolated = _project_bounds(extrapolated, box_bounds)
        updates = jax.tree.map(lambda new, old: new - old, extrapolated, params)
        return updates, {"x": accepted, "t": t, "count": optax.safe_int32_increment(state["count"])}

    return optax.GradientTransformation(init_fn, update_fn)


@device_function
def make_optimizer(
    optimizer_name: str,
    step_size: LearningRate,
    regularizer_weight=0.0,
    regularizer_prox: Callable[[jnp.ndarray, float], jnp.ndarray] | None = None,
    base_optimizer_factory: Callable[[LearningRate], optax.GradientTransformation] | None = None,
    box_constraints: Tuple[object, object] | None = None,
    bregman_penalty: float = 1.0,
    device: DeviceLike = "cpu",
) -> optax.GradientTransformation:
    """Single entry point to build optimizers, with optional proximal variants.

    - ``fista`` uses Euclidean gradient and proximal steps with acceleration.
      ``split_bregman`` uses linearized augmented-Lagrangian primal steps,
      splitting, and dual updates. Both retain their solver at zero weight.
    - Proximal callbacks receive the appropriate scalar, array, or tree threshold.
      Base factories are restricted to plain ``optax.sgd``; Adam is unsupported.
      The fitting API supplies objective-aware backtracking for physical-map TV.
    - All other optimizers are resolved from Optax and only support L2 weight decay plus optional
      box constraints.
    """
    name = optimizer_name.lower()
    if not callable(step_size) and (not np.isfinite(step_size) or step_size < 0):
        raise ValueError("Learning rate must be finite and nonnegative")
    if callable(step_size):
        schedule = step_size
        def step_size(count):
            rate = jnp.asarray(schedule(count))
            return jnp.where(jnp.isfinite(rate) & (rate >= 0), rate, jnp.nan)
    _validate_proximal_weights(regularizer_weight)
    prox_fn = regularizer_prox or soft_threshold
    if name in ("split_bregman", "bregman"):
        if base_optimizer_factory not in (None, optax.sgd):
            raise ValueError("Split Bregman requires Euclidean gradient steps; an Adam base is unsupported")
        if bregman_penalty <= 0 or not np.isfinite(bregman_penalty):
            raise ValueError("bregman_penalty must be finite and positive")
        _validate_proximal_weights(regularizer_weight)
        return place_optimizer(_with_box_constraints(_make_split_bregman_optimizer(
            step_size, regularizer_weight, prox_fn, box_constraints, bregman_penalty
        ), box_constraints), device)

    if name == "fista":
        if base_optimizer_factory not in (None, optax.sgd):
            raise ValueError("FISTA requires Euclidean gradient steps; an Adam base is unsupported")
        _validate_proximal_weights(regularizer_weight)
        return place_optimizer(_with_box_constraints(_make_fista_optimizer(
            step_size, regularizer_weight, prox_fn, box_constraints
        ), box_constraints), device)

    if regularizer_prox is not None:
        raise ValueError("regularizer_prox is only supported for 'split_bregman' or 'fista'.")
    if base_optimizer_factory is not None:
        raise ValueError("base_optimizer_factory is only used with 'split_bregman' or 'fista'.")

    supported = {"adam", "adamw", "sgd", "rmsprop", "adagrad", "adadelta", "adamax", "adamaxw",
                 "amsgrad", "lamb", "lars", "lion", "novograd", "noisy_sgd", "radam", "sm3", "yogi"}
    opt_builder = getattr(optax, name, None) if name in supported else None
    if opt_builder is None or not callable(opt_builder):
        raise ValueError(f"Unknown optimizer '{optimizer_name}'")

    opt = opt_builder(step_size)
    if _any_positive(regularizer_weight):
        def decay_init(params):
            weights = _align_tree(params, regularizer_weight)
            for x, weight in zip(jax.tree.leaves(params), jax.tree.leaves(weights), strict=True):
                if np.broadcast_shapes(np.shape(x), np.shape(weight)) != np.shape(x):
                    raise ValueError("Decay weights must broadcast to parameter shapes")
            return optax.EmptyState()

        def decay_update(grads, state, params):
            return jax.tree.map(lambda g, p, w: g + w * p, grads, params,
                                _align_tree(params, regularizer_weight)), state

        opt = optax.chain(optax.GradientTransformation(decay_init, decay_update), opt)

    return place_optimizer(_with_box_constraints(opt, box_constraints), device)


__all__ = ["make_optimizer"]
