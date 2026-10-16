"""Objective-aware TV fitting with restarted FISTA or linearized split Bregman."""
from __future__ import annotations

from operator import index

import jax
import jax.numpy as jnp
import numpy as np
import optax

from mrax.devices import DeviceLike, bind_device, device_function, place_optimizer
from mrax.opt.tv import tv_prox, vector_norm

from .fit_execution import _tree_finite


def dot(left, right):
    return sum(jnp.real(jnp.vdot(a, b)) for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True))


def subtract(left, right):
    return jax.tree.map(lambda a, b: a - b, left, right)


@device_function
def make_tv_steps(loss, smooth_loss, bounds, operators, name, step_size, *,
                  tolerance=1e-8, max_prox_steps=2048, penalty=1.0, inner_steps=5,
                  device: DeviceLike = "cpu", grad_clip_norm: float | None = None):
    max_prox_steps, inner_steps = index(max_prox_steps), index(inner_steps)
    if not np.isfinite(tolerance) or not np.isfinite(penalty) or tolerance <= 0 or max_prox_steps < 1 or penalty <= 0 or inner_steps < 1:
        raise ValueError("Proximal tolerance, iteration limits and Bregman penalty must be positive")
    if name not in ("fista", "split_bregman"):
        raise ValueError("Unknown TV solver")
    if not callable(step_size) and (not np.isfinite(step_size) or step_size <= 0):
        raise ValueError("TV learning rate must be finite and positive")

    def project(params):
        result = {}
        for key, value in params.items():
            if jnp.iscomplexobj(value):
                if np.isfinite(np.asarray(bounds[key])).any():
                    raise ValueError("Complex fields require unbounded box constraints")
                result[key] = value
            else:
                result[key] = jnp.clip(value, *bounds[key]).astype(value.dtype)
        return result

    def rate_at(count, dtype):
        rate = jnp.asarray(step_size(count) if callable(step_size) else step_size, dtype=dtype)
        return jnp.where(jnp.isfinite(rate) & (rate > 0), rate, jnp.nan)

    def next_rate(state, scheduled):
        # Mean-normalized objectives can legitimately need rates above 1e6.
        # Limit only at the dtype range; backtracking determines stability.
        multiplier = 1.5 * (scheduled / state["schedule_rate"])
        maximum = jnp.finfo(state["rate"].dtype).max / 16
        return jnp.minimum(state["rate"], maximum / jnp.maximum(multiplier, 1.)) * multiplier

    def roundoff(value, point):
        # Preserve objective scaling, including weak raw MR signal amplitudes.
        info = jnp.finfo(value.dtype)
        epsilon = max([info.eps] + [jnp.finfo(jnp.real(field).dtype).eps
                                   for field in jax.tree.leaves(point)])
        return 8 * epsilon * jnp.maximum(jnp.abs(value), info.tiny)

    def initial_duals(params):
        return {key: tuple(jnp.zeros_like(op.gradient(value)) for op in operators[key])
                for key, value in params.items()}

    def init(params):
        leaves = jax.tree.leaves(params)
        dtype = jnp.result_type(*(jnp.real(value).dtype for value in leaves), jnp.float32)
        state = {"rate": rate_at(0, dtype), "schedule_rate": rate_at(0, dtype), "count": jnp.asarray(0, jnp.int32),
                 "stationarity": jnp.asarray(0., dtype=dtype)}
        if name == "fista":
            return {**state, "x": params, "y": params, "t": jnp.asarray(1., state["rate"].dtype),
                    "duals": initial_duals(params), "gap": jnp.asarray(0., state["rate"].dtype)}
        splits = {key: tuple(op.gradient(value) for op in operators[key]) for key, value in params.items()}
        return {**state, "splits": splits, "duals": initial_duals(params),
                "primal_residual": jnp.asarray(0., state["rate"].dtype),
                "dual_residual": jnp.asarray(0., state["rate"].dtype)}

    def proximal(value, rate, duals):
        params, new_duals, gaps, valid = {}, {}, [], jnp.asarray(True)
        for key, field in value.items():
            result, ps, gap, _ = tv_prox(field, operators[key], rate.astype(jnp.real(field).dtype), bounds[key], duals[key],
                                       tolerance=tolerance, max_steps=max_prox_steps)
            params[key], new_duals[key] = result, ps
            gaps.append(gap)
            valid = valid & (gap <= tolerance * (1 + .5 * jnp.sum(jnp.abs(result - field)**2)
                                                 + rate * sum(op.value(result) for op in operators[key])))
        gap = jnp.max(jnp.stack(gaps)) if gaps else jnp.asarray(0., dtype=rate.dtype)
        return params, new_duals, gap, valid

    def forward_backward(point, duals, initial_rate):
        value, grad = jax.value_and_grad(smooth_loss)(point)
        grad = jax.tree.map(jnp.conj, grad)
        if grad_clip_norm is not None:
            multiplier = jnp.minimum(1., grad_clip_norm / jnp.maximum(optax.global_norm(grad), 1e-12))
            grad = jax.tree.map(lambda g: g * multiplier.astype(jnp.real(g).dtype), grad)
        empty = (initial_rate, point, duals, jnp.asarray(jnp.inf, value.dtype),
                 jnp.asarray(False), jnp.asarray(0))

        def trial(state):
            rate, _, _, _, _, count = state
            forward = jax.tree.map(lambda x, g: x - rate.astype(jnp.real(x).dtype) * g, point, grad)
            proposal, ps, gap, prox_valid = proximal(forward, rate, duals)
            candidate_value = smooth_loss(proposal)
            delta = subtract(proposal, point)
            majorant = value + dot(grad, delta) + dot(delta, delta) / (2 * rate)
            allowance = roundoff(value, point)
            accepted = prox_valid & _tree_finite((proposal, ps, grad)) & jnp.isfinite(candidate_value)
            accepted = accepted & (candidate_value <= majorant + allowance)
            return rate, proposal, ps, gap, accepted, count + 1

        def next_trial(state):
            rate, proposal, ps, gap, accepted, count = state
            return trial((rate * .5, proposal, ps, gap, accepted, count))

        result = trial(empty)
        result = jax.lax.while_loop(lambda s: (~s[4]) & (s[5] < 32), next_trial, result)
        return result

    def fista_step(params, state):
        old_loss = loss(params)
        scheduled = rate_at(state["count"], state["rate"].dtype)
        rate = next_rate(state, scheduled)
        rate, proposal, ps, gap, accepted, _ = forward_backward(state["y"], state["duals"], rate)
        allowance = roundoff(old_loss, params)
        restart = (~accepted) | (loss(proposal) > old_loss + allowance)
        retried = restart

        def retry(_):
            return forward_backward(params, state["duals"], rate)

        rate, proposal, ps, gap, accepted, _ = jax.lax.cond(
            restart, retry, lambda _: (rate, proposal, ps, gap, accepted, jnp.asarray(0)), None)
        delta = subtract(proposal, params)
        restart = restart | (dot(subtract(state["y"], proposal), delta) > 0)
        t = jnp.where(restart, 1., .5 * (1 + jnp.sqrt(1 + 4 * state["t"]**2)))
        momentum = jnp.where(restart, 0., (state["t"] - 1) / t)
        y = project(jax.tree.map(lambda x, d: x + momentum.astype(jnp.real(x).dtype) * d, proposal, delta))
        point = jax.tree.map(lambda x, y: jnp.where(retried, x, y), params, state["y"])
        gradient_map = subtract(proposal, point)
        stationarity = jnp.sqrt(dot(gradient_map, gradient_map)) / jnp.maximum(rate, 1e-12)
        new_state = {"rate": rate, "schedule_rate": scheduled,
                     "count": optax.safe_int32_increment(state["count"]), "x": proposal, "y": y, "t": t,
                     "duals": ps, "gap": gap, "stationarity": stationarity}
        return old_loss, proposal, new_state, jnp.where(accepted, 0, 3).astype(jnp.int32)

    def bregman_step(params, state):
        old_loss = loss(params)
        scheduled = rate_at(state["count"], state["rate"].dtype)

        def augmented(point):
            total = smooth_loss(point)
            for key, field in point.items():
                for op, split, dual in zip(operators[key], state["splits"][key], state["duals"][key], strict=True):
                    residual = op.gradient(field) - split + dual
                    total = total + .5 * penalty * jnp.sum(jnp.abs(residual)**2)
            return total

        def inner(carry, _):
            point, rate, valid, _ = carry
            value, grad = jax.value_and_grad(augmented)(point)
            grad = jax.tree.map(jnp.conj, grad)
            if grad_clip_norm is not None:
                multiplier = jnp.minimum(1., grad_clip_norm / jnp.maximum(optax.global_norm(grad), 1e-12))
                grad = jax.tree.map(lambda g: g * multiplier.astype(jnp.real(g).dtype), grad)

            def trial(current):
                rate, _, _, count = current
                proposal = project(jax.tree.map(lambda x, g: x - rate.astype(jnp.real(x).dtype) * g, point, grad))
                delta = subtract(proposal, point)
                upper = value + dot(grad, delta) + dot(delta, delta) / (2 * rate)
                candidate = augmented(proposal)
                epsilon = roundoff(value, point)
                accepted = _tree_finite((proposal, grad)) & jnp.isfinite(candidate) & (candidate <= upper + epsilon)
                return rate, proposal, accepted, count + 1

            result = trial((rate, point, jnp.asarray(False), jnp.asarray(0)))
            result = jax.lax.while_loop(lambda s: (~s[2]) & (s[3] < 32),
                                        lambda s: trial((s[0] * .5, s[1], s[2], s[3])), result)
            rate, proposal, accepted, _ = result
            delta = subtract(proposal, point)
            return (proposal, rate, valid & accepted,
                    jnp.sqrt(dot(delta, delta)) / jnp.maximum(rate, 1e-12)), None

        rate = next_rate(state, scheduled)
        (proposal, rate, valid, stationarity), _ = jax.lax.scan(
            inner, (params, rate, jnp.asarray(True), jnp.asarray(0., rate.dtype)), None, length=inner_steps)
        splits, duals, primal_squared, dual_squared = {}, {}, jnp.asarray(0., rate.dtype), jnp.asarray(0., rate.dtype)
        for key, field in proposal.items():
            new_splits, new_duals = [], []
            divergence = jnp.zeros_like(field)
            for op, old_split, dual in zip(operators[key], state["splits"][key], state["duals"][key], strict=True):
                gradient = op.gradient(field)
                argument = gradient + dual
                norm = vector_norm(argument)
                split = argument * jnp.maximum(1 - op.weights / (penalty * jnp.where(norm > 0, norm, 1)), 0)[..., None]
                residual = gradient - split
                new_splits.append(split)
                new_duals.append(dual + residual)
                primal_squared = primal_squared + jnp.sum(jnp.abs(residual)**2)
                divergence = divergence + penalty * op.adjoint(split - old_split)
            splits[key], duals[key] = tuple(new_splits), tuple(new_duals)
            dual_squared = dual_squared + jnp.sum(jnp.abs(divergence)**2)
        new_state = {"rate": rate, "schedule_rate": scheduled, "count": optax.safe_int32_increment(state["count"]),
                     "stationarity": stationarity, "splits": splits, "duals": duals,
                     "primal_residual": jnp.sqrt(primal_squared), "dual_residual": jnp.sqrt(dual_squared)}
        return old_loss, proposal, new_state, jnp.where(valid, 0, 3).astype(jnp.int32)

    def convergence(params, state, target):
        threshold = target * (1 + jnp.sqrt(dot(params, params)))
        valid = state["stationarity"] <= threshold
        if name != "fista":
            valid = valid & (state["primal_residual"] <= threshold) & (state["dual_residual"] <= threshold)
        return valid

    optimizer = optax.GradientTransformation(init, lambda grads, state, params: (grads, state))
    step = fista_step if name == "fista" else bregman_step
    return place_optimizer(optimizer, device), bind_device(step, device), bind_device(convergence, device)
