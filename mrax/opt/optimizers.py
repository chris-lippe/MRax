"""Optax-friendly wrappers and proximal variants."""

from __future__ import annotations

from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .ops import soft_threshold

LearningRate = float | Callable[[jax.Array], jax.Array]


def _align_tree(example, value):
    """Broadcast ``value`` to the structure of ``example`` if needed."""
    try:
        return jax.tree_util.tree_map(lambda _, v: v, example, value)
    except Exception:
        return jax.tree_util.tree_map(lambda _: value, example)


def _all_nonpositive(weights) -> bool:
    leaves, _ = jax.tree_util.tree_flatten(weights)
    return all(np.all(np.asarray(leaf) <= 0.0) for leaf in leaves)


def _any_positive(weights) -> bool:
    leaves, _ = jax.tree_util.tree_flatten(weights)
    return any(np.any(np.asarray(leaf) > 0.0) for leaf in leaves)


def _all_zero(weights) -> bool:
    return all(np.all(np.asarray(leaf) == 0.0) for leaf in jax.tree_util.tree_leaves(weights))


def _with_box_constraints(opt, bounds):
    if bounds is None:
        return opt

    def update_fn(grads, state, params):
        updates, new_state = opt.update(grads, state, params)
        projected = _project_bounds(optax.apply_updates(params, updates), bounds)
        return jax.tree_util.tree_map(lambda new, old: new - old, projected, params), new_state

    return optax.GradientTransformation(opt.init, update_fn)


def _project_bounds(params, bounds: Tuple[object, object] | None):
    if bounds is None:
        return params
    lower, upper = bounds
    lower_tree = _align_tree(params, lower)
    upper_tree = _align_tree(params, upper)

    def clip_fn(x, lo, hi):
        lo_finite = jnp.where(jnp.isfinite(lo), lo, -jnp.inf)
        hi_finite = jnp.where(jnp.isfinite(hi), hi, jnp.inf)
        return jnp.clip(x, lo_finite, hi_finite)

    return jax.tree_util.tree_map(clip_fn, params, lower_tree, upper_tree)


def _make_split_bregman_optimizer(
    step_size: LearningRate,
    regularizer_weight,
    prox_fn,
    base_optimizer_factory: Callable[[LearningRate], optax.GradientTransformation],
    box_bounds: Tuple[object, object] | None,
) -> optax.GradientTransformation:
    """Split Bregman updater wrapping a base optimizer."""
    base_opt = base_optimizer_factory(step_size)

    def init_fn(params):
        state = {
            "base": base_opt.init(params),
            "breg": jax.tree_util.tree_map(jnp.zeros_like, params),
        }
        if callable(step_size):
            state["count"] = jnp.zeros((), dtype=jnp.int32)
        return state

    def shrink(param, breg, weight, prox, rate):
        weight = jnp.asarray(weight)
        shrunk = prox(param + breg, weight * rate)
        breg_new = breg + param - shrunk
        return shrunk, breg_new

    def update_fn(grads, state, params):
        base_updates, new_base_state = base_opt.update(grads, state["base"], params)
        tmp_params = optax.apply_updates(params, base_updates)
        weights_tree = _align_tree(tmp_params, regularizer_weight)
        prox_tree = _align_tree(tmp_params, prox_fn)
        rate = step_size(state["count"]) if callable(step_size) else step_size
        pairs = jax.tree_util.tree_map(
            lambda p, b, w, prox: shrink(p, b, w, prox, rate),
            tmp_params, state["breg"], weights_tree, prox_tree,
        )
        def is_pair(value):
            return isinstance(value, tuple)
        new_params = jax.tree_util.tree_map(
            lambda pair: pair[0], pairs, is_leaf=is_pair  # type: ignore[arg-type]
        )
        new_breg = jax.tree_util.tree_map(
            lambda pair: pair[1], pairs, is_leaf=is_pair  # type: ignore[arg-type]
        )
        new_params = _project_bounds(new_params, box_bounds)
        updates = jax.tree_util.tree_map(lambda new, old: new - old, new_params, params)
        new_state = {"base": new_base_state, "breg": new_breg}
        if callable(step_size):
            new_state["count"] = optax.safe_int32_increment(state["count"])
        return updates, new_state

    return optax.GradientTransformation(init_fn, update_fn)


def _make_fista_optimizer(
    step_size: LearningRate,
    regularizer_weight,
    prox_fn,
    base_optimizer_factory: Callable[[LearningRate], optax.GradientTransformation],
    box_bounds: Tuple[object, object] | None,
) -> optax.GradientTransformation:
    """FISTA wrapper around a base optimizer with optional proximal step."""
    if _all_nonpositive(regularizer_weight):
        return _with_box_constraints(base_optimizer_factory(step_size), box_bounds)
    base_opt = base_optimizer_factory(step_size)

    def init_fn(params):
        leaves = jax.tree_util.tree_leaves(params)
        dtype = jnp.result_type(*[jnp.real(jnp.asarray(value)).dtype for value in leaves])
        state = {
            "base": base_opt.init(params),
            "x": params,
            "t": jnp.array(1.0, dtype=dtype),
        }
        if callable(step_size):
            state["count"] = jnp.zeros((), dtype=jnp.int32)
        return state

    def update_fn(grads, state, params):
        base_updates, new_base_state = base_opt.update(grads, state["base"], params)
        y_step = optax.apply_updates(params, base_updates)
        weights_tree = _align_tree(y_step, regularizer_weight)
        prox_tree = _align_tree(y_step, prox_fn)
        rate = step_size(state["count"]) if callable(step_size) else step_size
        x_new = jax.tree_util.tree_map(
            lambda y, w, p: p(y, w * rate), y_step, weights_tree, prox_tree
        )
        t_new = 0.5 * (1.0 + jnp.sqrt(1.0 + 4.0 * state["t"] ** 2))
        momentum = (state["t"] - 1.0) / t_new
        y_new = jax.tree_util.tree_map(
            lambda x_cur, x_prev: x_cur + momentum * (x_cur - x_prev),
            x_new,
            state["x"],
        )
        y_new = _project_bounds(y_new, box_bounds)
        updates = jax.tree_util.tree_map(lambda new, old: new - old, y_new, params)
        new_state = {"base": new_base_state, "x": x_new, "t": t_new}
        if callable(step_size):
            new_state["count"] = optax.safe_int32_increment(state["count"])
        return updates, new_state

    return optax.GradientTransformation(init_fn, update_fn)


def make_optimizer(
    optimizer_name: str,
    step_size: LearningRate,
    regularizer_weight=0.0,
    regularizer_prox: Callable[[jnp.ndarray, float], jnp.ndarray] | None = None,
    base_optimizer_factory: Callable[[LearningRate], optax.GradientTransformation] | None = None,
    box_constraints: Tuple[object, object] | None = None,
) -> optax.GradientTransformation:
    """Single entry point to build optimizers, with optional proximal variants.

    - ``split_bregman`` / ``fista`` support arbitrary proximal regularizers via ``regularizer_prox`` with
      per-parameter weights (scalars, arrays, or pytrees) and optional box constraints.
    - All other optimizers are resolved from Optax and only support L2 weight decay plus optional
      box constraints.
    """
    name = optimizer_name.lower()
    prox_fn = regularizer_prox or soft_threshold
    base_factory = base_optimizer_factory or (lambda lr: optax.adam(lr))

    if name in ("split_bregman", "bregman"):
        if regularizer_prox is None and _all_zero(regularizer_weight):
            return _with_box_constraints(base_factory(step_size), box_constraints)
        return _make_split_bregman_optimizer(
            step_size, regularizer_weight, prox_fn, base_factory, box_constraints
        )

    if name == "fista":
        return _make_fista_optimizer(
            step_size, regularizer_weight, prox_fn, base_factory, box_constraints
        )

    if regularizer_prox is not None:
        raise ValueError("regularizer_prox is only supported for 'split_bregman' or 'fista'.")
    if base_optimizer_factory is not None:
        raise ValueError("base_optimizer_factory is only used with 'split_bregman' or 'fista'.")

    opt_builder = getattr(optax, name, None)
    if opt_builder is None or not callable(opt_builder):
        raise ValueError(f"Unknown optimizer '{optimizer_name}'")

    opt = opt_builder(step_size)
    if _any_positive(regularizer_weight):
        opt = optax.chain(optax.additive_weight_decay(regularizer_weight), opt)

    return _with_box_constraints(opt, box_constraints)


__all__ = ["make_optimizer"]
