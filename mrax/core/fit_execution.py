"""Device-side optimizer execution and optional fitting diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from operator import index
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax


@dataclass(frozen=True)
class FitState:
    """Raw parameters, optimizer moments and completed updates for resuming a fit.

    Resume with the same model, data, regularization and optimizer state structure.
    Changing a constant learning rate is supported when the state structure matches.
    """

    params: dict[str, Any]
    optimizer_state: Any
    step: int = 0


@dataclass(frozen=True)
class FitInfo:
    """Optional fit status; convergence describes the requested stopping tolerances."""

    state: FitState
    iterations: int
    converged: bool
    termination_reason: str
    final_loss: float | None
    devices: tuple[str, ...]


class FitError(FloatingPointError):
    """A nonfinite fit with its last valid chunk-boundary checkpoint, if available."""

    def __init__(self, message: str, state: FitState | None, losses: list[float]):
        super().__init__(message)
        self.last_valid_state = state
        self.loss_history = list(losses)


def _tree_finite(tree):
    finite = jnp.asarray(True)
    for value in jax.tree_util.tree_leaves(tree):
        finite = finite & jnp.all(jnp.isfinite(value))
    return finite


def _relative_update(params, candidate):
    metric = jnp.asarray(0.0)
    for before, after in zip(
        jax.tree_util.tree_leaves(params), jax.tree_util.tree_leaves(candidate), strict=True
    ):
        rms = jnp.sqrt(jnp.mean(jnp.abs(before) ** 2))
        delta = jnp.sqrt(jnp.mean(jnp.abs(after - before) ** 2))
        metric = jnp.maximum(metric, delta / (1.0 + rms))
    return metric


def _validate_optimizer_state(expected, candidate):
    if jax.tree_util.tree_structure(expected) != jax.tree_util.tree_structure(candidate):
        raise ValueError("initial_state does not match this optimizer's state structure")
    for a, b in zip(
        jax.tree_util.tree_leaves(expected), jax.tree_util.tree_leaves(candidate), strict=True
    ):
        if np.shape(a) != np.shape(b) or jnp.asarray(a).dtype != jnp.asarray(b).dtype:
            raise ValueError("initial_state contains optimizer arrays with incompatible shape or dtype")


def execute_optimizer(
    loss_fn: Callable,
    params: dict,
    optimizer: optax.GradientTransformation,
    *,
    num_steps: int,
    chunk_size: int = 32,
    initial_state: FitState | None = None,
    convergence_tolerance: float | None = None,
    parameter_tolerance: float = 1e-5,
    convergence_patience: int = 5,
    min_steps: int = 0,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> tuple[dict, list[float], FitInfo]:
    """Run fixed-size device chunks; histories contain losses before each update.

    Convergence requires both scaled objective changes and relative raw-parameter
    updates to meet their tolerances for consecutive updates. Stopping counters
    start afresh when resuming. Nonfinite proposals are never accepted; errors
    retain the last verified chunk-boundary checkpoint rather than partial results.
    """
    num_steps, chunk_size = index(num_steps), index(chunk_size)
    convergence_patience, min_steps = index(convergence_patience), index(min_steps)
    if num_steps < 0 or chunk_size < 1 or convergence_patience < 1 or min_steps < 0:
        raise ValueError("num_steps/min_steps must be nonnegative and chunk_size/patience positive")
    for name, value in (
        ("convergence_tolerance", convergence_tolerance), ("parameter_tolerance", parameter_tolerance)
    ):
        if value is not None and (not np.isfinite(value) or value < 0):
            raise ValueError(f"{name} must be finite and nonnegative")
    if initial_state is not None and not isinstance(initial_state, FitState):
        raise TypeError("initial_state must be a FitState")
    start_step = index(initial_state.step) if initial_state is not None else 0
    if start_step < 0:
        raise ValueError("initial_state.step must be nonnegative")
    expected_state = optimizer.init(params)
    opt_state = expected_state
    if initial_state is not None:
        _validate_optimizer_state(expected_state, initial_state.optimizer_state)
        opt_state = initial_state.optimizer_state
    total = max(1, num_steps)
    if progress_callback is not None:
        progress_callback(0, total, "Preparing optimizer")
    devices = tuple(sorted({
        str(device) for value in jax.tree_util.tree_leaves(params)
        for device in jnp.asarray(value).devices()
    }))
    if num_steps == 0:
        if progress_callback is not None:
            progress_callback(1, total, "No optimizer steps requested")
        return params, [], FitInfo(
            FitState(params, opt_state, start_step), 0, False, "no_steps", None, devices
        )

    evaluate_loss = jax.jit(loss_fn)
    initial_loss = evaluate_loss(params)
    if not bool(_tree_finite((params, opt_state))) or not np.isfinite(float(initial_loss)):
        raise FitError("Initial fit loss, parameters or optimizer state are nonfinite", None, [])
    losses: list[float] = []
    completed = 0
    previous_loss = initial_loss
    stable = jnp.zeros((), dtype=jnp.int32)
    progress_interval = max(1, num_steps // 100)

    @jax.jit
    def run_chunk(params, state, previous, stable_count, done, limit):
        carry = (params, state, previous, stable_count, done, jnp.asarray(0, dtype=jnp.int32))

        def body(carry, position):
            current, state, previous, stable_count, done, status = carry

            def update(_):
                value, grads = jax.value_and_grad(loss_fn)(current)
                updates, proposal_state = optimizer.update(grads, state, current)
                proposal = optax.apply_updates(current, updates)
                finite = jnp.isfinite(value) & _tree_finite((grads, proposal, proposal_state))
                next_count = stable_count
                converged = jnp.asarray(False)
                if convergence_tolerance is not None:
                    change = jnp.abs(value - previous) / (1.0 + jnp.abs(previous))
                    small = (change <= convergence_tolerance) & (
                        _relative_update(current, proposal) <= parameter_tolerance
                    )
                    next_count = jnp.where(small, stable_count + 1, 0)
                    converged = (done + 1 >= min_steps) & (next_count >= convergence_patience)
                chosen, chosen_state = jax.lax.cond(
                    finite, lambda _: (proposal, proposal_state), lambda _: (current, state), None
                )
                next_status = jnp.where(finite, jnp.where(converged, 1, 0), 2).astype(jnp.int32)
                next_carry = (
                    chosen, chosen_state, value, next_count, done + finite.astype(jnp.int32),
                    next_status,
                )
                return next_carry, (value, jnp.asarray(True))

            def skip(_):
                return carry, (jnp.zeros_like(previous), jnp.asarray(False))

            return jax.lax.cond((status == 0) & (position < limit), update, skip, None)

        return jax.lax.scan(body, carry, jnp.arange(chunk_size))

    converged = False
    final_loss = float(initial_loss)
    while completed < num_steps:
        limit = min(chunk_size, num_steps - completed)
        if progress_callback is not None:
            next_progress = 1 if completed == 0 else (
                (completed // progress_interval + 1) * progress_interval
            )
            limit = min(limit, max(1, next_progress - completed))
        checkpoint = FitState(params, opt_state, start_step + completed)
        checkpoint_losses = list(losses)
        carry, values = run_chunk(
            params, opt_state, previous_loss, stable,
            jnp.asarray(completed, dtype=jnp.int32), jnp.asarray(limit, dtype=jnp.int32),
        )
        params, opt_state, previous_loss, stable, count, status = carry
        history, executed, count_host, status_host = jax.device_get((*values, count, status))
        if int(status_host) == 2:
            raise FitError(
                f"Nonfinite optimizer loss, gradient or update at step {start_step + int(count_host) + 1}",
                checkpoint, checkpoint_losses,
            )
        final_loss = float(evaluate_loss(params))
        if not np.isfinite(final_loss):
            raise FitError(
                f"Nonfinite objective after step {start_step + int(count_host)}",
                checkpoint, checkpoint_losses,
            )
        losses.extend(float(value) for value in history[executed])
        completed = int(count_host)
        converged = int(status_host) == 1
        if progress_callback is not None and (
            completed == 1 or completed == num_steps or completed % progress_interval == 0
            or converged
        ):
            progress_callback(completed, total, f"Fitting step {completed} of {num_steps}")
        if converged:
            break

    info = FitInfo(
        FitState(params, opt_state, start_step + completed), completed, converged,
        "converged" if converged else "max_steps", final_loss, devices,
    )
    return params, losses, info
