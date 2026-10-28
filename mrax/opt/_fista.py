"""FISTA with Beck--Teboulle backtracking (doi:10.1137/080716542).

The accepted point x and the extrapolated gradient point y are distinct.
The line search certifies the smooth majorant for each proximal-gradient step.
"""
from __future__ import annotations

from operator import index

import jax
import jax.numpy as jnp
import numpy as np
import optax

from mrax.opt.tv import tv_prox


def _dot(a, b):
    return sum(jnp.real(jnp.vdot(x, y)) for x, y in
               zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True))


def _finite(tree):
    leaves = jax.tree.leaves(tree)
    return jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in leaves])) if leaves else jnp.asarray(True)


def make_fista_kernel(loss, smooth_loss, bounds, operators, step_size, *,
                      tolerance=1e-8, max_prox_steps=2048, maximum_rate=None,
                      grad_clip_norm=None, objective_check=True, restart="none",
                      backtracking_factor=2., max_backtracking=32,
                      backtracking="classic", step_growth=1.5):
    """Build a pure, JIT-compatible FISTA kernel and a stopping certificate.

    ``restart='none'`` is the paper recurrence. ``'gradient'`` additionally
    resets momentum on the standard gradient/displacement inner-product test.
    A nonfinite extrapolated gradient always retries at the accepted point.
    An exhausted line search returns status 3 and preserves the input state.
    The step schedule specifies trial steps; backtracking may reduce them.
    ``backtracking='full'`` uses Algorithm 3.5 of Scheinberg et al.
    (doi:10.1007/s10208-014-9189-9), recomputing extrapolation on rate changes.
    """
    max_prox_steps, max_backtracking = index(max_prox_steps), index(max_backtracking)
    if restart not in ("none", "gradient"):
        raise ValueError("FISTA restart must be 'none' or 'gradient'")
    if backtracking not in ("classic", "full"):
        raise ValueError("FISTA backtracking must be 'classic' or 'full'")
    if not np.isfinite(step_growth) or step_growth < 1:
        raise ValueError("FISTA trial-step growth must be finite and at least one")
    if not np.isfinite(backtracking_factor) or backtracking_factor <= 1:
        raise ValueError("FISTA backtracking factor must exceed one")
    if max_prox_steps < 1 or max_backtracking < 1 or tolerance <= 0 or not np.isfinite(tolerance):
        raise ValueError("FISTA proximal tolerance and iteration limits must be positive")
    if not callable(step_size) and (not np.isfinite(step_size) or step_size < 0):
        raise ValueError("FISTA step size must be finite and nonnegative")
    if maximum_rate is not None and (not np.isfinite(maximum_rate) or maximum_rate <= 0):
        raise ValueError("Maximum FISTA step must be finite and positive")
    if grad_clip_norm is not None and (not np.isfinite(grad_clip_norm) or grad_clip_norm <= 0):
        raise ValueError("FISTA gradient clipping norm must be finite and positive")
    operators = {k: tuple(op for op in ops if op.axes) for k, ops in operators.items()}

    def rate_at(count, dtype):
        rate = jnp.asarray(step_size(count) if callable(step_size) else step_size, dtype)
        if maximum_rate is not None:
            rate = jnp.minimum(rate, maximum_rate)
        return jnp.where(jnp.isfinite(rate) & (rate >= 0), rate, jnp.nan)

    def init(params):
        dtype = jnp.result_type(*[jnp.real(x).dtype for x in jax.tree.leaves(params)], jnp.float32)
        rate = rate_at(jnp.asarray(0, jnp.int32), dtype)
        rate = jnp.where(rate == 0, 1., rate)
        return dict(y=params, previous=params, t=jnp.asarray(1., dtype), rate=rate, schedule_rate=rate,
                    count=jnp.asarray(0, jnp.int32),
                    duals={k: tuple(jnp.zeros_like(op.gradient(x)) for op in operators[k])
                           for k, x in params.items()},
                    gap=jnp.asarray(0., dtype), stationarity=jnp.asarray(jnp.finfo(dtype).max, dtype),
                    cached_loss=jnp.asarray(0., dtype), safe_rate=rate, grow=jnp.asarray(True),
                    backtracks=jnp.asarray(0, jnp.int32), prox_steps=jnp.asarray(0, jnp.int32),
                    restarts=jnp.asarray(0, jnp.int32))

    def proximal(point, rate, duals):
        proposal, ps = {}, {}
        gap, count, valid = jnp.asarray(0., rate.dtype), jnp.asarray(0), jnp.asarray(True)
        for key, field in point.items():
            x, p, g, n = tv_prox(field, operators[key], rate.astype(jnp.real(field).dtype),
                                 bounds[key], duals[key], tolerance=tolerance,
                                 max_steps=max_prox_steps, method="fgp")
            proposal[key], ps[key] = x, p
            primal = .5 * jnp.sum(jnp.abs(x - field)**2)
            primal += rate * sum(op.value(x) for op in operators[key])
            valid &= g <= tolerance * (1 + primal)
            gap, count = jnp.maximum(gap, g), count + n
        return proposal, ps, gap, count, valid

    def search(point, duals, rate, trial_limit=max_backtracking):
        value, gradient = jax.value_and_grad(smooth_loss)(point)
        value = jnp.asarray(value)
        gradient = jax.tree.map(jnp.conj, gradient)
        direction = gradient
        dtype = value.dtype
        eps = max([jnp.finfo(dtype).eps] + [jnp.finfo(jnp.real(x).dtype).eps for x in jax.tree.leaves(point)])
        # Scale floating-point comparison allowances with the objective.
        allowance = 16 * eps * jnp.maximum(jnp.abs(value), jnp.finfo(dtype).tiny)
        good_gradient = _finite((gradient, value))

        def trial(carry):
            rate, _, _, _, work, _, trials = carry
            forward = jax.tree.map(lambda x, g: x - rate.astype(jnp.real(x).dtype) * g, point, direction)
            x, ps, gap, steps, certified = proximal(forward, rate, duals)
            delta = jax.tree.map(lambda a, b: a - b, x, point)
            candidate = jnp.asarray(smooth_loss(x))
            # Compare differences to avoid adding tiny terms to the objective.
            remainder = candidate - value - _dot(gradient, delta)
            accepted = good_gradient & certified & _finite((x, ps, candidate))
            if objective_check:
                quadratic = _dot(delta, delta) / (2 * rate)
                # IEEE Inf <= Inf evaluates true; certificates require finite values.
                accepted &= _finite((remainder, quadratic)) & (remainder <= quadratic + allowance)
            return rate, x, ps, gap, work + steps, accepted, trials + 1

        initial = (rate, point, duals, jnp.asarray(0., rate.dtype), jnp.asarray(0),
                   jnp.asarray(False), jnp.asarray(0))
        first = trial(initial)

        def retry(carry):
            return trial((carry[0] / backtracking_factor, *carry[1:]))

        return jax.lax.while_loop(lambda s: (~s[5]) & good_gradient & (s[6] < trial_limit), retry, first)

    def trial_cap(params, scheduled):
        if grad_clip_norm is None:
            return jnp.asarray(jnp.finfo(scheduled.dtype).max / 16, scheduled.dtype)
        gradient = jax.grad(smooth_loss)(params)
        norm = jnp.sqrt(_dot(gradient, gradient))
        # Clip the forward-displacement estimate by shrinking the step, with
        # the SAME step used in prox. Clipping only grad(f) would change the
        # balance between f and g and can create a false composite optimum.
        return scheduled * jnp.minimum(1., grad_clip_norm / jnp.maximum(norm, 1e-30))

    def precision_limited(point, proposal, duals, rate, scheduled):
        # A tiny rate can make a nonstationary proximal step round back to
        # its input. Classify this as a precision limit. Ordinary near-optimum rounding is excluded by an
        # independent composite KKT residual, including the box normal cone.
        if not jax.tree.leaves(point):
            return jnp.asarray(False)
        dtype = rate.dtype
        eps = max([jnp.finfo(dtype).eps] +
                  [jnp.finfo(jnp.real(x).dtype).eps for x in jax.tree.leaves(point)])
        rounded = jnp.all(jnp.stack([jnp.all(a == b) for a, b in
            zip(jax.tree.leaves(point), jax.tree.leaves(proposal), strict=True)]))
        suspect = rounded & (rate < scheduled * eps)

        def check(_):
            gradient = jax.tree.map(jnp.conj, jax.grad(smooth_loss)(point))
            kkt = {}
            for key, x in point.items():
                g = gradient[key] + sum((op.adjoint(p) / rate for op, p in
                    zip(operators[key], duals[key], strict=True)), jnp.zeros_like(x))
                if not jnp.iscomplexobj(x):
                    lower, upper = bounds[key]
                    g = jnp.where(x <= lower, jnp.minimum(g, 0), g)
                    g = jnp.where(x >= upper, jnp.maximum(g, 0), g)
                    g = jnp.where(jnp.asarray(lower) == jnp.asarray(upper), 0, g)
                kkt[key] = g
            threshold = jnp.sqrt(eps) * (1 + jnp.sqrt(_dot(point, point)))
            return (~_finite(kkt)) | (jnp.sqrt(_dot(kkt, kkt)) > threshold)
        return jax.lax.cond(suspect, check, lambda _: jnp.asarray(False), None)

    def step(params, state):
        scheduled = rate_at(state["count"], state["rate"].dtype)
        # Retain the certified Lipschitz estimate. An explicit schedule change
        # rescales that estimate; every trial requires majorization.
        rate = state["rate"] * (scheduled / state["schedule_rate"])
        cap = trial_cap(params, scheduled)
        rate = jnp.minimum(rate, cap)
        result = search(state["y"], state["duals"], rate)
        retried = ~result[5]
        restarted = retried
        # A domain/line-search failure resets momentum while retaining certified
        # curvature. Start the retry at the safer scale for domain-limited models.
        restart_rate = jnp.minimum(rate, state["safe_rate"])
        result = jax.lax.cond(restarted,
            lambda _: search(params, state["duals"], jnp.minimum(restart_rate, cap)), lambda _: result, None)
        rate, x, ps, gap, work, accepted, trials = result
        point = jax.tree.map(lambda a, b: jnp.where(retried, a, b), params, state["y"])
        limited = accepted & precision_limited(point, x, ps, rate, scheduled)
        accepted &= ~limited
        delta = jax.tree.map(lambda a, b: a - b, x, params)
        if restart == "gradient":
            restarted |= _dot(jax.tree.map(lambda y, a: y - a, state["y"], x), delta) > 0
        t = .5 * (1 + jnp.sqrt(1 + 4 * state["t"]**2))
        t = jnp.where(restarted, 1., t)
        momentum = jnp.where(restarted, 0., (state["t"] - 1) / t)
        # Extrapolate y freely; enforce the box in prox(g + I_box).
        y = jax.tree.map(lambda a, d: a + momentum.astype(jnp.real(a).dtype) * d, x, delta)
        norm = jnp.sqrt(_dot(jax.tree.map(lambda a, b: a - b, x, point),
                              jax.tree.map(lambda a, b: a - b, x, point))) / rate
        # Require a resolvable residual for the stationarity check.
        norm = jnp.where(rate >= scheduled * 1e-8, norm, jnp.finfo(rate.dtype).max)
        new = dict(y=y, previous=params, t=t, rate=rate, schedule_rate=scheduled,
                   cached_loss=state["cached_loss"], safe_rate=state["safe_rate"], grow=state["grow"],
                   count=optax.safe_int32_increment(state["count"]), duals=ps, gap=gap,
                   stationarity=norm, backtracks=trials - 1, prox_steps=work,
                   restarts=state["restarts"] + restarted.astype(jnp.int32))
        new = jax.tree.map(lambda a, b: jnp.where(accepted, a, b), new, state)
        x = jax.tree.map(lambda a, b: jnp.where(accepted, a, b), x, params)
        return loss(params), x, new, jnp.where(limited, 4, jnp.where(accepted, 0, 3)).astype(jnp.int32)

    def full_step(params, state):
        # Scheinberg--Goldfarb--Bai Algorithm 3.5: when a trial step changes,
        # recompute BOTH momentum and the extrapolated gradient point.
        scheduled = rate_at(state["count"], state["rate"].dtype)
        rate = state["rate"] * (scheduled / state["schedule_rate"])
        maximum = maximum_rate if maximum_rate is not None else jnp.finfo(rate.dtype).max / 16
        maximum = jnp.minimum(maximum, trial_cap(params, scheduled))
        # Near an optimum, reuse the last resolvably improving step.
        factor = jnp.where((state["count"] > 0) & state["grow"], step_growth, 1.)
        rate = jnp.where(state["grow"], rate, jnp.minimum(rate, state["safe_rate"]))
        rate = jnp.minimum(rate * factor, maximum)
        previous_delta = jax.tree.map(lambda x, old: x - old, params, state["previous"])

        def trial(carry):
            rate, _, _, _, work, _, trials, _, _ = carry
            t = .5 * (1 + jnp.sqrt(1 + 4 * state["t"]**2 * (state["rate"] / rate)))
            t = jnp.where(state["count"] == 0, 1., t)
            momentum = (state["t"] - 1) / t
            point = jax.tree.map(lambda x, d: x + momentum.astype(jnp.real(x).dtype) * d,
                                 params, previous_delta)
            result = search(point, state["duals"], rate, trial_limit=1)
            return (*result[:4], work + result[4], result[5], trials + 1, point, t)

        initial = (rate, params, state["duals"], jnp.asarray(0., rate.dtype), jnp.asarray(0),
                   jnp.asarray(False), jnp.asarray(0), params, state["t"])
        first = trial(initial)
        def retry(carry):
            return trial((carry[0] / backtracking_factor, *carry[1:]))
        result = jax.lax.while_loop(lambda s: (~s[5]) & (s[6] < max_backtracking), retry, first)
        retried = ~result[5]
        def reset(_):
            restart_rate = jnp.minimum(state["rate"], state["safe_rate"])
            r = search(params, state["duals"], jnp.minimum(restart_rate, maximum))
            return (*r[:4], result[4] + r[4], r[5], result[6] + r[6], params, jnp.asarray(1., rate.dtype))
        result = jax.lax.cond(retried, reset, lambda _: result, None)
        rate, x, ps, gap, work, accepted, trials, point, t = result
        limited = accepted & precision_limited(point, x, ps, rate, scheduled)
        accepted &= ~limited
        delta = jax.tree.map(lambda a, b: a - b, x, params)
        restarted = retried
        if restart == "gradient":
            restarted |= _dot(jax.tree.map(lambda y, a: y - a, point, x), delta) > 0
        t = jnp.where(restarted, 1., t)
        next_rate = jnp.minimum(rate * step_growth, maximum)
        next_t = .5 * (1 + jnp.sqrt(1 + 4 * t**2 * rate / next_rate))
        y = jax.tree.map(lambda a, d: a + ((t - 1) / next_t).astype(jnp.real(a).dtype) * d, x, delta)
        displacement = jax.tree.map(lambda a, b: a - b, x, point)
        norm = jnp.sqrt(_dot(displacement, displacement)) / rate
        norm = jnp.where(rate >= scheduled * 1e-8, norm, jnp.finfo(rate.dtype).max)
        old_loss = jax.lax.cond(state["count"] > 0, lambda _: state["cached_loss"],
                                lambda _: jnp.asarray(loss(params), rate.dtype), None)
        new_loss = jnp.asarray(loss(x), rate.dtype)
        epsilon = max([jnp.finfo(old_loss.dtype).eps] +
                      [jnp.finfo(jnp.real(x).dtype).eps for x in jax.tree.leaves(params)])
        grow = jnp.abs(old_loss - new_loss) > 16 * epsilon * jnp.maximum(jnp.abs(old_loss), jnp.finfo(old_loss.dtype).tiny)
        # Full backtracking must be able to recover a collapsed local step.
        # Recover gradually rather than imposing the nominal rate as a floor
        # on every trial (which repeats expensive rejected proximal solves).
        grow |= rate < scheduled * 1e-3
        new = dict(y=y, previous=params, t=t, rate=rate, schedule_rate=scheduled,
                   cached_loss=new_loss, safe_rate=jnp.where(grow, rate, state["safe_rate"]), grow=grow,
                   count=optax.safe_int32_increment(state["count"]), duals=ps, gap=gap,
                   stationarity=norm, backtracks=trials - 1, prox_steps=work,
                   restarts=state["restarts"] + restarted.astype(jnp.int32))
        new = jax.tree.map(lambda a, b: jnp.where(accepted, a, b), new, state)
        x = jax.tree.map(lambda a, b: jnp.where(accepted, a, b), x, params)
        return old_loss, x, new, jnp.where(limited, 4, jnp.where(accepted, 0, 3)).astype(jnp.int32)

    def convergence(params, state, target):
        # Certify at the returned point, rather than y, with the actual smooth
        # gradient even when a caller requested clipping during the iterations.
        gradient = jax.tree.map(jnp.conj, jax.grad(smooth_loss)(params))
        rate = state["rate"]
        forward = jax.tree.map(lambda x, g: x - rate.astype(jnp.real(x).dtype) * g, params, gradient)
        probe, duals, _, _, valid = proximal(forward, rate, state["duals"])
        delta = jax.tree.map(lambda a, b: a - b, probe, params)
        residual = jnp.sqrt(_dot(delta, delta)) / rate
        # Evaluate the equivalent KKT residual with cancellation-resistant
        # float32 arithmetic. A tiny rate can round x - rate*grad back to x
        # while the true projected gradient is still large.
        kkt = {}
        for key, x in params.items():
            g = gradient[key] + sum((op.adjoint(p) / rate for op, p in
                zip(operators[key], duals[key], strict=True)), jnp.zeros_like(x))
            if not jnp.iscomplexobj(x):
                lower, upper = bounds[key]
                g = jnp.where(x <= lower, jnp.minimum(g, 0), g)
                g = jnp.where(x >= upper, jnp.maximum(g, 0), g)
                g = jnp.where(jnp.asarray(lower) == jnp.asarray(upper), 0, g)
            kkt[key] = g
        residual = jnp.maximum(residual, jnp.sqrt(_dot(kkt, kkt)))
        resolvable = rate >= rate_at(state["count"], rate.dtype) * 1e-8
        return (rate_at(state["count"], rate.dtype) > 0) & valid & resolvable & (
            residual <= target * (1 + jnp.sqrt(_dot(params, params))))

    active_step = full_step if backtracking == "full" and objective_check else step

    def scheduled_step(params, state):
        def pause(_):
            # Zero pauses the update, including the proximal term. Retain the last positive Lipschitz estimate
            # and reset extrapolation before any later positive step.
            value = jnp.asarray(loss(params), state["cached_loss"].dtype)
            new = {**state, "y": params, "previous": params, "t": jnp.ones_like(state["t"]),
                   "cached_loss": value.astype(state["cached_loss"].dtype),
                   "grow": jnp.asarray(False), "safe_rate": state["rate"],
                   "count": optax.safe_int32_increment(state["count"])}
            return value, params, new, jnp.asarray(0, jnp.int32)
        return jax.lax.cond(rate_at(state["count"], state["rate"].dtype) == 0,
                            pause, lambda _: active_step(params, state), None)

    optimizer = optax.GradientTransformation(init, lambda grads, state, params: (grads, state))
    return optimizer, scheduled_step, convergence
