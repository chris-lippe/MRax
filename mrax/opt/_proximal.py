"""Objective-aware TV fitting with restarted FISTA or linearized split Bregman."""
from __future__ import annotations

from operator import index

import jax
import jax.numpy as jnp
import numpy as np
import optax

from mrax.devices import DeviceLike
from mrax.opt.tv import tv_prox, vector_norm


def _tree_finite(tree):
    return all_finite(tree)


def all_finite(tree):
    result = jnp.asarray(True)
    for value in jax.tree.leaves(tree):
        result = result & jnp.all(jnp.isfinite(value))
    return result


def dot(left, right):
    return sum(jnp.real(jnp.vdot(a, b)) for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True))


def subtract(left, right):
    return jax.tree.map(lambda a, b: a - b, left, right)


def make_proximal_kernel(loss, smooth_loss, bounds, operators, name, step_size, *,
                  tolerance=1e-8, max_prox_steps=2048, penalty=1.0, inner_steps=1,
                  device: DeviceLike = "cpu", grad_clip_norm: float | None = None,
                  maximum_rate: float | None = None, objective_check=True,
                  composite_check=True):
    max_prox_steps, inner_steps = index(max_prox_steps), index(inner_steps)
    if not np.isfinite(tolerance) or not np.isfinite(penalty) or tolerance <= 0 or max_prox_steps < 1 or penalty <= 0 or inner_steps < 1:
        raise ValueError("Proximal tolerance, iteration limits and Bregman penalty must be positive")
    if name not in ("fista", "split_bregman"):
        raise ValueError("Unknown TV solver")
    if not callable(step_size) and (not np.isfinite(step_size) or step_size <= 0):
        raise ValueError("TV learning rate must be finite and positive")
    if maximum_rate is not None and (not np.isfinite(maximum_rate) or maximum_rate <= 0):
        raise ValueError("Maximum TV step must be finite and positive")
    # Singleton/fully collapsed axes contribute zero TV. Keep the primal-dual
    # state structure consistent even if such operators were supplied.
    operators = {key: tuple(op for op in groups if op.axes) for key, groups in operators.items()}

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
        rate = jnp.minimum(rate, maximum_rate) if maximum_rate is not None else rate
        return jnp.where(jnp.isfinite(rate) & (rate > 0), rate, jnp.nan)

    def next_rate(state, scheduled, params):
        if not objective_check:
            return scheduled
        # Mean-normalized objectives can legitimately need rates above 1e6.
        # Limit only at the dtype range; backtracking determines stability.
        # Do not grow the step when improvement is below floating-point
        # resolution. Otherwise a quadratic can oscillate around its minimum
        # inside the objective roundoff allowance.
        increase = jnp.where(state["last_decrease"] > roundoff(state["last_loss"], params), 1.5,
                             jnp.where(state["last_decrease"] < 0, .5, 1.))
        multiplier = increase * (scheduled / state["schedule_rate"])
        maximum = jnp.finfo(state["rate"].dtype).max / 16
        # Revisit a resolvable trial rate instead of inheriting a collapsed rate
        # forever. Backtracking still has to certify every accepted proposal.
        proposal = jnp.maximum(scheduled, jnp.minimum(state["rate"], maximum / jnp.maximum(multiplier, 1.)) * multiplier)
        return jnp.minimum(proposal, maximum_rate) if maximum_rate is not None else proposal

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
                 "stationarity": jnp.asarray(jnp.finfo(dtype).max, dtype=dtype),
                 "last_loss": jnp.asarray(0., dtype), "last_decrease": jnp.asarray(jnp.finfo(dtype).max, dtype)}
        if name == "fista":
            return {**state, "y": params, "t": jnp.asarray(1., state["rate"].dtype),
                    "duals": initial_duals(params), "gap": jnp.asarray(0., state["rate"].dtype)}
        splits = {key: tuple(op.gradient(value) for op in operators[key]) for key, value in params.items()}
        # Delay data evaluation until the first update: zero-step fits and
        # checkpoint validation must not evaluate a model or a proximal map.
        duals = initial_duals(params)
        return {**state, "penalty": jnp.asarray(penalty, dtype),
                "splits": splits, "duals": duals,
                "initialized": jnp.asarray(False),
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
        allowance = roundoff(value, point)
        gradient_finite = _tree_finite(grad)

        def trial(state):
            rate, _, _, _, _, count = state
            forward = jax.tree.map(lambda x, g: x - rate.astype(jnp.real(x).dtype) * g, point, grad)
            proposal, ps, gap, prox_valid = proximal(forward, rate, duals)
            candidate_value = smooth_loss(proposal)
            delta = subtract(proposal, point)
            majorant = value + dot(grad, delta) + dot(delta, delta) / (2 * rate)
            accepted = prox_valid & gradient_finite & _tree_finite((proposal, ps)) & jnp.isfinite(candidate_value)
            accepted = accepted & ((candidate_value <= majorant + allowance) if objective_check else True)
            return rate, proposal, ps, gap, accepted, count + 1

        def next_trial(state):
            rate, proposal, ps, gap, accepted, count = state
            return trial((rate * .5, proposal, ps, gap, accepted, count))

        result = trial(empty)
        result = jax.lax.while_loop(lambda s: (~s[4]) & (s[5] < 64), next_trial, result)
        rate, proposal, ps, gap, accepted, _ = result
        return rate, proposal, ps, gap, accepted, loss(proposal)

    def fista_step(params, state):
        old_loss = loss(params)
        scheduled = rate_at(state["count"], state["rate"].dtype)
        rate = next_rate(state, scheduled, params)
        rate, proposal, ps, gap, accepted, proposal_loss = forward_backward(state["y"], state["duals"], rate)
        allowance = roundoff(old_loss, params)
        restart = (~accepted) | ((proposal_loss > old_loss + allowance) if objective_check and composite_check else False)
        retried = restart

        def retry(_):
            # The extrapolated point's failed line search must not throttle the
            # independent retry at the last accepted feasible iterate.
            return forward_backward(params, state["duals"], jnp.maximum(rate, scheduled))

        rate, proposal, ps, gap, accepted, proposal_loss = jax.lax.cond(
            restart, retry, lambda _: (rate, proposal, ps, gap, accepted, proposal_loss), None)
        delta = subtract(proposal, params)
        restart = restart | (dot(subtract(state["y"], proposal), delta) > 0)
        t = jnp.where(restart, 1., .5 * (1 + jnp.sqrt(1 + 4 * state["t"]**2)))
        momentum = jnp.where(restart, 0., (state["t"] - 1) / t)
        y = project(jax.tree.map(lambda x, d: x + momentum.astype(jnp.real(x).dtype) * d, proposal, delta))
        point = jax.tree.map(lambda x, y: jnp.where(retried, x, y), params, state["y"])
        gradient_map = subtract(proposal, point)
        stationarity = jnp.sqrt(dot(gradient_map, gradient_map)) / rate
        # A sub-resolution update is not a stationarity certificate.
        stationarity = jnp.where(rate >= scheduled * 1e-8, stationarity, jnp.finfo(rate.dtype).max)
        new_state = {"rate": rate, "schedule_rate": scheduled,
                     "count": optax.safe_int32_increment(state["count"]), "y": y, "t": t,
                     "duals": ps, "gap": gap, "stationarity": stationarity}
        new_state.update(last_loss=old_loss.astype(rate.dtype),
                         last_decrease=(old_loss-proposal_loss).astype(rate.dtype))
        return old_loss, proposal, new_state, jnp.where(accepted, 0, 3).astype(jnp.int32)

    def bregman_step(params, state):
        old_loss = loss(params)
        scheduled = rate_at(state["count"], state["rate"].dtype)

        rho = state["penalty"]

        def initialize(_):
            gradient = jax.tree.map(jnp.conj, jax.grad(smooth_loss)(params))
            forward = jax.tree.map(lambda x, g: x - scheduled.astype(jnp.real(x).dtype) * g, params, gradient)
            _, probe, _, valid = proximal(forward, scheduled, initial_duals(params))
            duals = jax.tree.map(lambda p: p / (scheduled * rho).astype(jnp.real(p).dtype), probe)
            return duals, valid

        initial_dual, probe_valid = jax.lax.cond(state["initialized"],
            lambda _: (state["duals"], jnp.asarray(True)), initialize, None)
        state = {**state, "duals": initial_dual}

        def augmented(point):
            total = smooth_loss(point)
            for key, field in point.items():
                for op, split, dual in zip(operators[key], state["splits"][key], state["duals"][key], strict=True):
                    residual = op.gradient(field) - split
                    # Omit rho*||dual||^2/2: it is constant in the primal
                    # subproblem and can swamp meaningful changes as rho falls.
                    total = total + rho * (.5 * jnp.sum(jnp.abs(residual)**2)
                        + jnp.real(jnp.vdot(dual, residual)))
            return total

        def inner(carry, _):
            point, rate, valid, _ = carry
            value, grad = jax.value_and_grad(augmented)(point)
            grad = jax.tree.map(jnp.conj, grad)
            if grad_clip_norm is not None:
                multiplier = jnp.minimum(1., grad_clip_norm / jnp.maximum(optax.global_norm(grad), 1e-12))
                grad = jax.tree.map(lambda g: g * multiplier.astype(jnp.real(g).dtype), grad)
            gradient_finite = _tree_finite(grad)
            epsilon = roundoff(value, point)

            def trial(current):
                rate, _, _, count = current
                proposal = project(jax.tree.map(lambda x, g: x - rate.astype(jnp.real(x).dtype) * g, point, grad))
                delta = subtract(proposal, point)
                upper = value + dot(grad, delta) + dot(delta, delta) / (2 * rate)
                candidate = augmented(proposal)
                accepted = gradient_finite & _tree_finite(proposal) & jnp.isfinite(candidate) & (candidate <= upper + epsilon)
                return rate, proposal, accepted, count + 1

            result = trial((rate, point, jnp.asarray(False), jnp.asarray(0)))
            result = jax.lax.while_loop(lambda s: (~s[2]) & (s[3] < 64),
                                        lambda s: trial((s[0] * .5, s[1], s[2], s[3])), result)
            rate, proposal, accepted, _ = result
            delta = subtract(proposal, point)
            return (proposal, rate, valid & accepted,
                    jnp.sqrt(dot(delta, delta)) / rate), None

        rate = next_rate(state, scheduled, params)
        (proposal, rate, valid, stationarity), _ = jax.lax.scan(
            inner, (params, rate, probe_valid, jnp.asarray(0., rate.dtype)), None, length=inner_steps)
        splits, duals, primal_squared, dual_squared = {}, {}, jnp.asarray(0., rate.dtype), jnp.asarray(0., rate.dtype)
        for key, field in proposal.items():
            new_splits, new_duals = [], []
            divergence = jnp.zeros_like(field)
            for op, old_split, dual in zip(operators[key], state["splits"][key], state["duals"][key], strict=True):
                gradient = op.gradient(field)
                argument = gradient + dual
                norm = vector_norm(argument)
                leaf_rho = rho.astype(jnp.real(argument).dtype)
                split = op.prox(argument, 1 / leaf_rho) if hasattr(op, "prox") else (
                    argument * jnp.maximum(1 - op.weights / (leaf_rho * jnp.where(norm > 0, norm, 1)), 0)[..., None])
                residual = gradient - split
                new_splits.append(split)
                new_duals.append(dual + residual)
                primal_squared = primal_squared + jnp.sum(jnp.abs(residual)**2)
                divergence = divergence + leaf_rho * op.adjoint(split - old_split)
            splits[key], duals[key] = tuple(new_splits), tuple(new_duals)
            dual_squared = dual_squared + jnp.sum(jnp.abs(divergence)**2)
        # Retain the inner projected-gradient diagnostic. The exact KKT test
        # below uses the updated multiplier only when stopping is requested;
        # fixed-budget runs avoid a redundant smooth gradient each update.
        stationarity = jnp.where(rate >= scheduled * 1e-8, stationarity, jnp.finfo(rate.dtype).max)
        # Residual balancing changes the augmented penalty, not the TV weight.
        # Rescale the scaled multiplier so the unscaled multiplier is invariant.
        residual_floor = 32 * jnp.finfo(rate.dtype).eps * (1 + jnp.sqrt(dot(proposal, proposal)))
        rebalance = ((state["count"] + 1) % 10 == 0) & (jnp.maximum(primal_squared, dual_squared) > residual_floor**2)
        factor = jnp.where(primal_squared > 100 * dual_squared, 2.,
                           jnp.where(dual_squared > 100 * primal_squared, .5, 1.))
        new_rho = jnp.where(rebalance, jnp.clip(rho * factor, penalty * 1e-6, penalty * 1e3), rho)
        duals = jax.tree.map(lambda d: d * (rho / new_rho).astype(jnp.real(d).dtype), duals)
        new_state = {"rate": rate, "schedule_rate": scheduled, "count": optax.safe_int32_increment(state["count"]),
                     "initialized": jnp.asarray(True), "penalty": new_rho, "stationarity": stationarity, "splits": splits, "duals": duals,
                     "primal_residual": jnp.sqrt(primal_squared), "dual_residual": jnp.sqrt(dual_squared)}
        new_state.update(last_loss=old_loss.astype(rate.dtype),
                         last_decrease=(old_loss-loss(proposal)).astype(rate.dtype))
        return old_loss, proposal, new_state, jnp.where(valid, 0, 3).astype(jnp.int32)

    def convergence(params, state, target):
        threshold = target * (1 + jnp.sqrt(dot(params, params)))
        valid = state["stationarity"] <= threshold
        if name == 'fista' and grad_clip_norm is not None:
            # Clipping can make an update tiny far from a stationary point.
            # When stopping is requested, certify with the unmodified gradient.
            rate = rate_at(state['count'], state['rate'].dtype)
            gradient = jax.tree.map(jnp.conj, jax.grad(smooth_loss)(params))
            forward = jax.tree.map(lambda x, g: x - rate.astype(jnp.real(x).dtype) * g, params, gradient)
            probe, _, _, certified = proximal(forward, rate, state['duals'])
            change = subtract(params, probe)
            valid = valid & certified & (jnp.sqrt(dot(change, change)) / rate <= threshold)
        if name != "fista":
            # Evaluate at the accepted point with the UPDATED (and, if rho
            # changed, consistently rescaled) multiplier.
            rho = state['penalty']
            gradient = jax.tree.map(jnp.conj, jax.grad(smooth_loss)(params))
            multiplier = {key: rho.astype(jnp.real(field).dtype) * sum((op.adjoint(d) for op, d in
                zip(operators[key], state['duals'][key], strict=True)), jnp.zeros_like(field))
                for key, field in params.items()}
            kkt = jax.tree.map(lambda g, d: g + d, gradient, multiplier)
            rate = rate_at(state['count'], state['rate'].dtype)
            probe = project(jax.tree.map(lambda x, g: x-rate.astype(jnp.real(x).dtype)*g, params, kkt))
            change = subtract(params, probe)
            valid = valid & (jnp.sqrt(dot(change, change)) / rate <= threshold)
            norm_dx = sum(jnp.sum(jnp.abs(op.gradient(params[key]))**2)
                for key, ops in operators.items() for op in ops)
            norm_d = dot(state['splits'], state['splits'])
            primal_threshold = target * (1+jnp.sqrt(jnp.maximum(norm_dx,norm_d)))
            dual_threshold = target * (1+jnp.sqrt(dot(multiplier,multiplier)))
            valid = valid & (state['primal_residual'] <= primal_threshold) & (state['dual_residual'] <= dual_threshold)
        return valid

    optimizer = optax.GradientTransformation(init, lambda grads, state, params: (grads, state))
    step = fista_step if name == "fista" else bregman_step
    return optimizer, step, convergence
