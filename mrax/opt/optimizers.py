"""Optax-friendly wrappers and proximal variants."""

from __future__ import annotations

from operator import index
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

    def update_fn(grads, state, params, **extra_args):
        updates, new_state = optax.with_extra_args_support(opt).update(grads, state, params, **extra_args)
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

    update_fn.__dict__.update(getattr(opt.update, '__dict__', {}))
    return optax.GradientTransformationExtraArgs(init_fn, update_fn)


def _project_bounds(params, bounds: Tuple[object, object] | None):
    if bounds is None:
        return params
    lower, upper = bounds
    lower_tree = _align_tree(params, lower)
    upper_tree = _align_tree(params, upper)

    def clip_fn(x, lo, hi):
        return jnp.clip(x, lo, hi)

    return jax.tree_util.tree_map(clip_fn, params, lower_tree, upper_tree)


def _make_proximal_optimizer(name, step_size, regularizer_weight, prox_fn,
                              box_bounds, penalty, *, context, tolerance,
                              max_steps, inner_steps, maximum_rate, grad_clip_norm):
    """One canonical kernel, exposed as an Optax ExtraArgs transformation.

    The fitting executor supplies ``value_fn`` automatically. Direct Optax
    callers should pass a differentiable smooth ``value_fn`` to ``update``.
    Without it fixed-step compatibility mode uses the supplied gradient (at
    state['solver']['y'] for FISTA); objective safeguards require value_fn.
    """
    from ._proximal import make_proximal_kernel
    from .tv import CallbackOperator, L1Operator

    def flatten(tree):
        leaves, definition = jax.tree.flatten(tree)
        return {str(i): x for i, x in enumerate(leaves)}, definition

    def build_kernel(loss, smooth, bounds, operators, checked):
        return make_proximal_kernel(loss, smooth, bounds, operators, name, step_size,
            tolerance=tolerance, max_prox_steps=max_steps, penalty=penalty,
            inner_steps=inner_steps, maximum_rate=maximum_rate,
            grad_clip_norm=grad_clip_norm, objective_check=checked,
            composite_check=context is not None or prox_fn is soft_threshold)

    context_kernel = build_kernel(*context, True) if context is not None else None

    def prepare(params, value_fn=None, grads=None):
        if context is not None:
            def convert(x):
                return x
            return params, convert, context_kernel
        else:
            packed, definition = flatten(params)
            def convert(x):
                return jax.tree.unflatten(definition, [x[str(i)] for i in range(len(packed))])
            weights, _ = flatten(_align_tree(params, regularizer_weight))
            proxs, _ = flatten(_align_tree(params, prox_fn))
            if box_bounds is None:
                bounds = {k: (-np.inf, np.inf) for k in packed}
            else:
                lo, _ = flatten(_align_tree(params, box_bounds[0]))
                hi, _ = flatten(_align_tree(params, box_bounds[1]))
                bounds = {k: (lo[k], hi[k]) for k in packed}
            for key, x in packed.items():
                lo, hi = map(np.asarray, bounds[key])
                if np.any(np.isnan(lo)) or np.any(np.isnan(hi)) or np.any(lo > hi):
                    raise ValueError("Box bounds must be ordered and cannot contain NaN")
                if np.broadcast_shapes(np.shape(x), lo.shape, hi.shape) != np.shape(x):
                    raise ValueError("Box bounds must broadcast to each parameter shape")
                if np.broadcast_shapes(np.shape(x), np.shape(weights[key])) != np.shape(x):
                    raise ValueError("Proximal weights must broadcast to parameter shapes")
            operators = {k: ((L1Operator(jnp.broadcast_to(jnp.asarray(weights[k], jnp.real(x).dtype), x.shape)),)
                              if np.any(np.asarray(weights[k]) > 0) else ()) for k, x in packed.items()}
            checked = value_fn is not None
            if checked:
                def smooth(x):
                    return value_fn(convert(x))
            else:
                flat_grad, _ = flatten(grads if grads is not None else jax.tree.map(jnp.zeros_like, params))
                # JAX complex gradients are conjugate to descent gradients.
                def smooth(x):
                    return sum(jnp.real(jnp.vdot(flat_grad[k], x[k])) for k in x)
            if prox_fn is not soft_threshold:
                operators = {k: (CallbackOperator(jnp.asarray(weights[k], jnp.real(x).dtype), proxs[k]),)
                             for k, x in packed.items()}
            def loss(x):
                return smooth(x) + sum(op.value(x[k]) for k, ops in operators.items() for op in ops)
        kernel = build_kernel(loss, smooth, bounds, operators, checked)
        return packed, convert, kernel

    def init(params):
        packed, _, (kernel, _, _) = prepare(params)
        return {"solver": kernel.init(packed), "accepted": params,
                "value": jnp.asarray(0., jnp.result_type(*[jnp.real(x).dtype for x in jax.tree.leaves(params)], jnp.float32)),
                "status": jnp.asarray(0, jnp.int32)}

    def update(grads, state, params, *, value_fn=None, **extra_args):
        packed, convert, (_, step, _) = prepare(params, value_fn, grads)
        value, accepted, solver, status = step(packed, state["solver"])
        accepted = convert(accepted)
        accepted, solver = jax.lax.cond(status == 0,
            lambda _: (accepted, solver), lambda _: (params, state['solver']), None)
        return jax.tree.map(lambda new, old: new - old, accepted, params), {
            "solver": solver, "accepted": accepted,
            "value": value.astype(state['value'].dtype), "status": status}

    def convergence(params, state, target, *, value_fn=None):
        packed, _, (_, _, check) = prepare(params, value_fn)
        return check(packed, state["solver"], target)

    def restore_state(state):
        # Earlier local checkpoints carried unused copies of parameter arrays.
        if not isinstance(state, dict) or not isinstance(state.get('solver'), dict):
            return state
        obsolete = {'x_ref', 'x'} if name == 'fista' else {'x_ref'}
        return {**state, 'solver': {key: value for key, value in state['solver'].items()
                                    if key not in obsolete}}

    update._mrax_manages_gradient = True
    update._mrax_accepted_state = True
    update._mrax_convergence = convergence
    update._mrax_restore_state = restore_state
    def objective(value_fn):
        if context is not None or prox_fn is not soft_threshold:
            return value_fn
        def total(params):
            weights = _align_tree(params, regularizer_weight)
            return value_fn(params) + sum(jnp.sum(w * jnp.abs(p)) for p, w in
                zip(jax.tree.leaves(params), jax.tree.leaves(weights), strict=True))
        return total
    update._mrax_objective = objective
    return optax.GradientTransformationExtraArgs(init, update)


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
    *,
    proximal_context=None,
    proximal_tolerance: float = 1e-8,
    proximal_max_steps: int = 2048,
    bregman_inner_steps: int = 1,
    proximal_max_step_size: float | None = None,
    grad_clip_norm: float | None = None,
) -> optax.GradientTransformation:
    """Single entry point to build optimizers, with optional proximal variants.

    - ``fista`` uses Euclidean gradient and proximal steps with acceleration.
      ``split_bregman`` uses linearized augmented-Lagrangian primal steps,
      splitting, and dual updates. Both retain their solver at zero weight.
    - Proximal callbacks receive the appropriate scalar, array, or tree threshold.
      Base factories are restricted to plain ``optax.sgd``; Adam is unsupported.
      Pass the differentiable smooth objective as ``value_fn`` to ``update``
      for safeguarded trials; the fitting API supplies it automatically. Direct
      fixed-step callers without it must evaluate FISTA gradients at state.y
      (stored under ``state['solver']``). Check ``state['status']`` for failure.
    - All other optimizers are resolved from Optax and only support L2 weight decay plus optional
      box constraints. ``grad_clip_norm`` optionally clips their full gradient.
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
    if grad_clip_norm is not None and (not np.isfinite(grad_clip_norm) or grad_clip_norm <= 0):
        raise ValueError("Gradient clipping norm must be finite and positive")
    prox_fn = regularizer_prox or soft_threshold
    if name in ("fista", "split_bregman", "bregman"):
        name = "split_bregman" if name == "bregman" else name
        if base_optimizer_factory not in (None, optax.sgd):
            raise ValueError("FISTA and split Bregman require Euclidean gradient steps; an Adam base is unsupported")
        if bregman_penalty <= 0 or not np.isfinite(bregman_penalty):
            raise ValueError("bregman_penalty must be finite and positive")
        if not callable(step_size) and step_size <= 0:
            raise ValueError("Proximal learning rate must be finite and positive")
        if not np.isfinite(proximal_tolerance) or proximal_tolerance <= 0:
            raise ValueError("Proximal tolerance must be finite and positive")
        if index(proximal_max_steps) < 1 or index(bregman_inner_steps) < 1:
            raise ValueError("Proximal iteration limits must be positive integers")
        if proximal_max_step_size is not None and (not np.isfinite(proximal_max_step_size) or proximal_max_step_size <= 0):
            raise ValueError("Maximum proximal step must be finite and positive")
        if grad_clip_norm is not None and (not np.isfinite(grad_clip_norm) or grad_clip_norm <= 0):
            raise ValueError("Gradient clipping norm must be finite and positive")
        return place_optimizer(_make_proximal_optimizer(name, step_size,
            regularizer_weight, prox_fn, box_constraints, bregman_penalty,
            context=proximal_context, tolerance=proximal_tolerance,
            max_steps=proximal_max_steps, inner_steps=bregman_inner_steps,
            maximum_rate=proximal_max_step_size, grad_clip_norm=grad_clip_norm), device)

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
    if grad_clip_norm is not None:
        opt = optax.chain(optax.clip_by_global_norm(grad_clip_norm), opt)
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
