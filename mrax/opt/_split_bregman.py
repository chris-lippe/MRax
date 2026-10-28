"""Split Bregman: augmented primal solve, shrinkage, then multiplier update.

The recurrence is Goldstein--Osher, doi:10.1137/080725891, section 3. Primal
subproblems use safeguarded Newton/CG for small systems and limited-memory
quasi-Newton steps for large systems.
"""
from __future__ import annotations

from operator import index

import jax
import jax.numpy as jnp
import numpy as np
import optax

from ._bregman_preconditioner import (
    apply_preconditioner,
    build_preconditioner,
    initial_preconditioner,
    shifted_preconditioner,
)
from ._bregman_quasi_newton import initial_history, inverse_product, update_history
from .tv import vector_norm


def _dot(left, right):
    return sum(jnp.real(jnp.vdot(a, b)) for a, b in
               zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True))


def _finite(tree):
    leaves = jax.tree.leaves(tree)
    return (jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in leaves]))
            if leaves else jnp.asarray(True))


def _norm(tree):
    leaves = jax.tree.leaves(tree)
    if not leaves:
        return jnp.asarray(0., jnp.float32)
    scale = jnp.max(jnp.stack([jnp.max(jnp.abs(x), initial=0.) for x in leaves]))
    safe = jnp.where(scale > 0, scale, 1.)
    return scale * jnp.sqrt(sum(jnp.sum(jnp.abs(x / safe) ** 2) for x in leaves))


def _add(point, direction, rate):
    return jax.tree.map(lambda x, d: x + rate.astype(jnp.real(x).dtype) * d,
                        point, direction)


def _dimension(point):
    return sum(x.size * (2 if jnp.iscomplexobj(x) else 1) for x in point.values())


def make_split_bregman_kernel(loss, smooth_loss, bounds, operators, step_size, *,
                              tolerance=1e-8, max_prox_steps=2048, penalty=1.,
                              inner_steps=1, maximum_rate=None, grad_clip_norm=None,
                              objective_check=True):
    """Return the common split-Bregman recurrence and its KKT stopping check.

    ``inner_steps`` is the number of inexact primal iterations before each
    shrinkage/multiplier update. Systems up to 32 real unknowns use Newton/CG;
    larger systems use an eight-pair L-BFGS inverse on the augmented objective.
    CG uses at most ``max_prox_steps`` Hessian
    products in total, including curvature retries, and a relative residual
    target. Preconditioner probes are additional work. Large subproblems use inexact
    Newton forcing; negative curvature triggers a positively shifted solve.
    ``step_size`` seeds the primal line search; accepted rates may grow to one.
    The augmented penalty is fixed. The split is the original operator output;
    regularization weights remain in the shrinkage objective.
    """
    inner_steps, max_prox_steps = index(inner_steps), index(max_prox_steps)
    if (inner_steps < 1 or max_prox_steps < 1 or not np.isfinite(tolerance)
            or tolerance <= 0 or not np.isfinite(penalty) or penalty <= 0):
        raise ValueError("Split Bregman tolerances, penalty and iteration limits must be positive")
    if not callable(step_size) and (not np.isfinite(step_size) or step_size < 0):
        raise ValueError("Split Bregman step size must be finite and nonnegative")
    if maximum_rate is not None and (not np.isfinite(maximum_rate) or maximum_rate <= 0):
        raise ValueError("Maximum Split Bregman step must be finite and positive")
    operators = {key: tuple(op for op in groups if op.axes)
                 for key, groups in operators.items()}
    # Canonical split coordinates A=D, d=Du, g(d)=sum(w*|d|).
    # Zero-weight standard operators have zero augmented contribution.
    scales = {key: tuple(jnp.asarray(1., op.weights.dtype) if hasattr(op, "prox")
                         else jnp.where(jnp.max(op.weights, initial=0.) > 0, 1., 0.)
                         for op in groups)
              for key, groups in operators.items()}

    def project(point):
        result = {}
        for key, x in point.items():
            if jnp.iscomplexobj(x):
                if np.isfinite(np.asarray(bounds[key])).any():
                    raise ValueError("Complex fields require unbounded box constraints")
                result[key] = x
            else:
                result[key] = jnp.clip(x, *bounds[key]).astype(x.dtype)
        return result

    def projected_gradient(point, gradient):
        result = {}
        for key, x in point.items():
            g = gradient[key]
            if not jnp.iscomplexobj(x):
                lo, hi = bounds[key]
                g = jnp.where(x <= lo, jnp.minimum(g, 0), g)
                g = jnp.where(x >= hi, jnp.maximum(g, 0), g)
                g = jnp.where(jnp.asarray(lo) == jnp.asarray(hi), 0, g)
            result[key] = g
        return result

    def rate_at(count, dtype):
        rate = jnp.asarray(step_size(count) if callable(step_size) else step_size, dtype)
        if maximum_rate is not None:
            rate = jnp.minimum(rate, maximum_rate)
        return jnp.where(jnp.isfinite(rate) & (rate >= 0), rate, jnp.nan)

    def forward(point):
        return {key: tuple(c.astype(jnp.real(x).dtype) * op.gradient(x)
                           for op, c in zip(operators[key], scales[key], strict=True))
                for key, x in point.items()}

    def adjoint(values, point):
        return {key: sum((op.adjoint(c.astype(jnp.real(q).dtype) * q)
                          for op, c, q in zip(operators[key], scales[key], values[key], strict=True)),
                         jnp.zeros_like(x)) for key, x in point.items()}

    def init(params):
        dtype = jnp.result_type(*[jnp.real(x).dtype for x in jax.tree.leaves(params)], jnp.float32)
        rate = rate_at(jnp.asarray(0, jnp.int32), dtype)
        rate = jnp.where(rate == 0, 1., rate)
        splits = forward(params)
        return dict(splits=splits, duals=jax.tree.map(jnp.zeros_like, splits),
                    count=jnp.asarray(0, jnp.int32), rate=rate, schedule_rate=rate,
                    penalty=jnp.asarray(penalty, dtype), layout_version=jnp.asarray(6, jnp.int32),
                    preconditioner=initial_preconditioner(params),
                    history=initial_history(params, 8 if _dimension(params) > 32 else 0),
                    primal_method=jnp.asarray(1 if _dimension(params) > 32 else 0, jnp.int32),
                    initialized=jnp.asarray(False), stationarity=jnp.asarray(jnp.finfo(dtype).max, dtype),
                    primal_residual=jnp.asarray(0., dtype), dual_residual=jnp.asarray(0., dtype),
                    inner_residual=jnp.asarray(0., dtype), cg_iterations=jnp.asarray(0, jnp.int32),
                    cg_residual=jnp.asarray(0., dtype),
                    damping=jnp.asarray(0., dtype), damping_restarts=jnp.asarray(0, jnp.int32),
                    backtracks=jnp.asarray(0, jnp.int32), curvature_truncated=jnp.asarray(False))

    def direction(point, gradient, product, preconditioner, damping):
        projected = projected_gradient(point, gradient)
        # The active set stays fixed during this linear solve; projection and
        # Armijo decrease subsequently verify the actual constrained proposal.
        free = {}
        for key, x in point.items():
            g = gradient[key]
            if jnp.iscomplexobj(x):
                free[key] = jnp.ones(x.shape, bool)
            else:
                lo, hi = bounds[key]
                free[key] = ((jnp.asarray(lo) != jnp.asarray(hi)) &
                             ~((x <= lo) & (g > 0)) & ~((x >= hi) & (g < 0)))
        def mask(value):
            return jax.tree.map(lambda x, yes: jnp.where(yes, x, 0), value, free)

        norm_g = _norm(projected)
        safe = jnp.where(norm_g > 0, norm_g, 1.)
        residual = jax.tree.map(lambda g: -g / safe.astype(jnp.real(g).dtype), projected)
        zero = jax.tree.map(jnp.zeros_like, point)
        dtype = norm_g.dtype
        eps = max([jnp.finfo(dtype).eps] + [jnp.finfo(jnp.real(x).dtype).eps
                                           for x in jax.tree.leaves(point)])
        target = jnp.asarray(max(np.sqrt(tolerance), 32 * eps), dtype)
        dimension = sum(x.size * (2 if jnp.iscomplexobj(x) else 1)
                        for x in jax.tree.leaves(point))
        limit = min(max_prox_steps, max(1, dimension))
        if dimension > max_prox_steps:
            # Use inexact forcing for large Newton equations far from stationarity.
            # This forcing tends to the requested accuracy as the gradient
            # vanishes. Exact TV shrinkage and the Armijo/KKT certificates are
            # unchanged; the CG residual is reported even for an inexact solve.
            forcing = jnp.minimum(.5, jnp.sqrt(norm_g / (1 + norm_g)))
            target = jnp.maximum(target, forcing)
        def solve(shift, budget):
            conditioned = jax.lax.cond(shift > 0,
                lambda _: shifted_preconditioner(preconditioner, shift),
                lambda _: preconditioner, None)

            def precondition(value):
                return mask(apply_preconditioner(value, conditioned))
            z = precondition(residual)
            energy = jnp.asarray(_dot(residual, z), dtype)
            valid_energy = jnp.isfinite(energy) & ((energy > 0) | (norm_g == 0))
            initial = (zero, residual, z, jnp.where(energy > 0, 1., 0.).astype(dtype),
                       energy, jnp.asarray(0, jnp.int32),
                       ~valid_energy, jnp.asarray(0., dtype))

            def body(carry):
                x, r, p, rr, rz, count, truncated, minimum = carry
                hp = mask(product(p))
                pp = _dot(p, p)
                rayleigh = _dot(p, hp) / jnp.maximum(pp, jnp.finfo(dtype).tiny)
                hp = _add(hp, p, shift)
                curvature = _dot(p, hp)
                positive = (jnp.isfinite(curvature) & (curvature > 0) & _finite(hp)
                            & jnp.isfinite(rz) & (rz > 0))
                alpha = jnp.where(positive, rz / jnp.where(positive, curvature, 1.), 0.)
                candidate = _add(x, p, alpha)
                new_r = _add(r, hp, -alpha)
                new_z = precondition(new_r)
                new_rz = jnp.asarray(_dot(new_r, new_z), dtype)
                new_rr = new_rz / jnp.maximum(energy, jnp.finfo(dtype).tiny)
                beta = new_rz / jnp.where(rz > 0, rz, 1.)
                new_p = _add(new_z, p, beta)
                return (candidate, new_r, new_p, new_rr, new_rz, count + 1,
                        truncated | ~positive | ~jnp.isfinite(new_rz) | (new_rz < 0),
                        jnp.minimum(minimum, rayleigh))

            return jax.lax.while_loop(
                lambda s: (s[3] > target ** 2) & (s[5] < jnp.minimum(limit, budget)) & ~s[6], body, initial)

        # A negative-curvature truncated Newton vector can be arbitrarily large
        # in a nearly singular field mode. Re-solve a positively shifted Newton
        # equation instead. This is numerical damping of the primal direction:
        # Armijo still evaluates the original, unshifted augmented subproblem.
        result = solve(damping, max_prox_steps)
        initial = (damping, result, result[5], jnp.asarray(0, jnp.int32))

        def regularize(carry):
            shift, solved, work, restarts = carry
            floor = jnp.asarray(32 * eps, dtype)
            shift = jnp.maximum(jnp.maximum(4 * shift, -2 * solved[7]), floor)
            new = solve(shift, max_prox_steps - work)
            return shift, new, work + new[5], restarts + 1

        damping, result, count, restarts = jax.lax.while_loop(
            lambda s: s[1][6] & (s[3] < 8) & (s[2] < max_prox_steps), regularize, initial)
        solved, _, _, rr, _, _, truncated, _ = result
        proposal = jax.tree.map(lambda d: d * safe.astype(jnp.real(d).dtype), solved)
        descent = _finite(proposal) & (_dot(gradient, proposal) < 0)
        proposal = jax.tree.map(lambda d, g: jnp.where(descent, d, -g), proposal, projected)
        return (proposal, norm_g, count, truncated, jnp.sqrt(jnp.maximum(rr, 0.)),
                damping, restarts)

    def step(params, state):
        old_loss = loss(params)
        scheduled = rate_at(state["count"], state["rate"].dtype)
        large_system = _dimension(params) > 32

        def active(_):
            rho = state["penalty"]

            def augmented(point):
                residual = jax.tree.map(lambda a, d: a - d, forward(point), state["splits"])
                # The omitted squared-dual term is constant in the primal solve.
                return smooth_loss(point) + rho * (
                    .5 * _dot(residual, residual) + _dot(state["duals"], residual))

            def gradient_fn(x):
                return jax.tree.map(jnp.conj, jax.grad(augmented)(x))

            current_value, current_gradient = jax.value_and_grad(augmented)(project(params))
            current_gradient = jax.tree.map(jnp.conj, current_gradient)
            current_norm = _norm(current_gradient)
            reference = state["preconditioner"]["gradient_reference"]
            cache_eps = max([jnp.finfo(current_norm.dtype).eps] +
                            [jnp.finfo(jnp.real(x).dtype).eps for x in params.values()])
            growth_threshold = jnp.maximum(4 * reference,
                jnp.sqrt(cache_eps) * (1 + _norm(params)))
            refresh = (~state["preconditioner"]["ready"] | ~jnp.isfinite(current_norm)
                       | (current_norm > growth_threshold) | (state["count"] % (50 if large_system else 10) == 0) | (
                (state["count"] % 5 == 0) &
                (state["cg_iterations"] > max(16, 4 * len(params)))))

            def rebuild(_):
                _, product = jax.linearize(gradient_fn, project(params))
                built = build_preconditioner(project(params), product, state["preconditioner"])
                return {**built, "gradient_reference": current_norm.astype(state["rate"].dtype)}

            preconditioner = jax.lax.cond(refresh, rebuild,
                                         lambda _: state["preconditioner"], None)
            history_start = state['history']
            if large_system:
                cleared = initial_history(params, 8)
                cleared['updates'] = history_start['updates']
                cleared['skipped'] = history_start['skipped']
                cleared['resets'] = optax.safe_int32_increment(history_start['resets'])
                history_start = jax.lax.cond(refresh & state['preconditioner']['ready'],
                    lambda _: cleared, lambda _: history_start, None)
            rate = state["rate"] * scheduled / state["schedule_rate"]
            rate = jnp.where(state["initialized"], rate * 1.5, scheduled)
            rate = jnp.minimum(rate, maximum_rate) if maximum_rate is not None else rate
            rate = jnp.minimum(rate, 1.) if objective_check else rate

            def inner(carry, index):
                (point, alpha, valid, precision, iterations, backtracks, truncated,
                 _residual, _cg_residual, damping, restarts, history, local_inverse) = carry
                if large_system:
                    def fresh(_):
                        value, gradient = jax.value_and_grad(augmented)(point)
                        return value, jax.tree.map(jnp.conj, gradient)
                    value, gradient = jax.lax.cond(index == 0,
                        lambda _: (current_value, current_gradient), fresh, None)
                    projected = projected_gradient(point, gradient)
                    norm_g = _norm(projected)
                    def inverse(q):
                        return apply_preconditioner(q, local_inverse)
                    search = jax.tree.map(lambda x: -x, inverse_product(projected, history, inverse))
                    # Restrict the inverse to the current bound-active subspace.
                    free = {}
                    for key, x in point.items():
                        lo, hi = bounds[key]
                        g = gradient[key]
                        free[key] = (jnp.ones(x.shape, bool) if jnp.iscomplexobj(x) else
                            ~(((x <= lo) & (g > 0)) | ((x >= hi) & (g < 0))
                              | (jnp.asarray(lo) == jnp.asarray(hi))))
                    search = jax.tree.map(lambda x, yes: jnp.where(yes, x, 0), search, free)
                    descent = _finite(search) & ((_dot(gradient, search) < 0) | (norm_g == 0))
                    reset = initial_history(point, 8)
                    reset['updates'] = history['updates']
                    reset['skipped'] = history['skipped']
                    reset['resets'] = optax.safe_int32_increment(history['resets'])
                    history = jax.tree.map(lambda a, b: jnp.where(descent, a, b), history, reset)
                    fallback = jax.tree.map(lambda x, yes: jnp.where(yes, -x, 0), inverse(projected), free)
                    search = jax.tree.map(lambda a, b: jnp.where(descent, a, b), search, fallback)
                    used = restarted = jnp.asarray(0, jnp.int32)
                    negative = jnp.asarray(False)
                    cg_residual = jnp.asarray(0., alpha.dtype)
                else:
                    value = augmented(point)
                    gradient, product = jax.linearize(gradient_fn, point)
                    search, norm_g, used, negative, cg_residual, damping, restarted = direction(
                        point, gradient, product, local_inverse, damping)
                eps = max([jnp.finfo(value.dtype).eps] + [jnp.finfo(jnp.real(x).dtype).eps
                                                        for x in jax.tree.leaves(point)])
                allowance = 8 * eps * jnp.maximum(jnp.abs(value), jnp.finfo(value.dtype).tiny)

                def line_search(vector, seed):
                    if grad_clip_norm is not None:
                        seed = jnp.minimum(seed, grad_clip_norm / jnp.maximum(_norm(gradient),
                                                                            jnp.finfo(seed.dtype).tiny))

                    def trial(current):
                        trial_alpha, _, _, number = current
                        proposal = project(_add(point, vector, trial_alpha))
                        delta = jax.tree.map(lambda a, b: a - b, proposal, point)
                        slope = _dot(gradient, delta)
                        candidate = augmented(proposal)
                        accepted = (_finite(gradient) & _finite(proposal) & jnp.isfinite(candidate)
                                    & (slope <= allowance) & (candidate <= value + 1e-4 * slope + allowance))
                        return trial_alpha, proposal, accepted, number + 1

                    result = trial((seed, point, jnp.asarray(False), jnp.asarray(0, jnp.int32)))
                    trial_alpha, proposal, accepted, number = jax.lax.while_loop(
                        lambda s: ~s[2] & (s[3] < 64),
                        lambda s: trial((s[0] * .5, s[1], s[2], s[3])), result)
                    leaves = jax.tree.leaves(point)
                    rounded = (jnp.all(jnp.stack([jnp.all(a == b) for a, b in
                               zip(leaves, jax.tree.leaves(proposal), strict=True)]))
                               if leaves else jnp.asarray(False))
                    delta = jax.tree.map(lambda a, b: a - b, proposal, point)
                    unresolved = _norm(delta) <= 8 * eps * jnp.maximum(_norm(point), jnp.finfo(value.dtype).tiny)
                    limited = (rounded | unresolved) & (
                        norm_g > jnp.sqrt(eps) * (1 + _norm(point)))
                    return trial_alpha, proposal, accepted, number, limited, delta

                result = line_search(search, alpha)
                if large_system:
                    # A stale line seed or inverse can round a nonstationary
                    # step to zero. Retry once on the SAME augmented function,
                    # rebuilding the inverse and resetting only primal memory.
                    def retry(_):
                        _, product = jax.linearize(gradient_fn, point)
                        built = build_preconditioner(point, product, local_inverse)
                        built = {**built, 'gradient_reference': _norm(gradient).astype(alpha.dtype)}
                        cleared = initial_history(point, 8)
                        cleared['updates'] = history['updates']
                        cleared['skipped'] = history['skipped']
                        cleared['resets'] = optax.safe_int32_increment(history['resets'])
                        vector = jax.tree.map(lambda q, yes: jnp.where(yes, -q, 0),
                                              apply_preconditioner(projected, built), free)
                        seed = jnp.minimum(scheduled, 1.) if objective_check else scheduled
                        repeated = line_search(vector, seed)
                        repeated = (*repeated[:3], repeated[3] + result[3], *repeated[4:])
                        return repeated, cleared, built

                    recoverable = (valid & ~precision & _finite(gradient)
                                   & (result[4] | ~result[2]))
                    result, history, local_inverse = jax.lax.cond(recoverable, retry,
                        lambda _: (result, history, local_inverse), None)
                alpha, proposal, accepted, number, limited, delta = result
                new_gradient = gradient_fn(proposal)
                valid = valid & _finite(new_gradient)
                if large_system:
                    # The two gradients use the SAME d,b: their linear shifts
                    # cancel, avoiding false secants from a changing multiplier.
                    gradient_change = jax.tree.map(lambda new, old: new - old, new_gradient, gradient)
                    history = update_history(history, delta, gradient_change,
                                              valid & accepted & ~limited)
                return (proposal, alpha, valid & accepted & ~negative, precision | limited,
                        iterations + used, backtracks + number - 1, truncated | negative | (restarted > 0),
                        norm_g.astype(alpha.dtype), cg_residual.astype(alpha.dtype),
                         damping, restarts + restarted, history, local_inverse), None

            initial = (project(params), rate, _finite(params) & jnp.isfinite(old_loss),
                       jnp.asarray(False), jnp.asarray(0, jnp.int32), jnp.asarray(0, jnp.int32),
                       jnp.asarray(False), jnp.asarray(0., rate.dtype), jnp.asarray(0., rate.dtype),
                        state["damping"] * .5, jnp.asarray(0, jnp.int32), history_start, preconditioner)
            (proposal, rate, valid, precision, used, backtracks, truncated, residual,
             cg_residual, damping, restarts, history, preconditioner), _ = jax.lax.scan(
                inner, initial, jnp.arange(inner_steps), length=inner_steps)
            gradients = forward(proposal)
            splits, duals = {}, {}
            for key, groups in operators.items():
                new_d, new_b = [], []
                for op, c, gradient, b in zip(groups, scales[key], gradients[key],
                                             state["duals"][key], strict=True):
                    argument = gradient + b
                    safe = jnp.where(c > 0, c, 1.).astype(jnp.real(argument).dtype)
                    if hasattr(op, "prox"):
                        d = safe * op.prox(argument / safe, 1 / (rho * safe ** 2))
                    else:
                        norm = vector_norm(argument)
                        threshold = op.weights / (rho * safe)
                        d = argument * jnp.maximum(1 - threshold / jnp.where(norm > 0, norm, 1), 0)[..., None]
                    new_d.append(d.astype(argument.dtype))
                    new_b.append((b + gradient - d).astype(argument.dtype))
                splits[key], duals[key] = tuple(new_d), tuple(new_b)
            primal = jax.tree.map(lambda a, d: a - d, gradients, splits)
            split_change = jax.tree.map(lambda a, b: a - b, splits, state["splits"])
            dual = jax.tree.map(lambda x: rho.astype(jnp.real(x).dtype) * x,
                                adjoint(split_change, proposal))
            valid = valid & _finite(splits) & _finite(duals) & jnp.isfinite(loss(proposal))
            status = jnp.where(precision, 4, jnp.where(valid, 0, 3)).astype(jnp.int32)
            updated = {**state, "rate": rate, "schedule_rate": scheduled,
                       "count": optax.safe_int32_increment(state["count"]), "initialized": jnp.asarray(True),
                       "splits": splits, "duals": duals, "stationarity": residual,
                       "inner_residual": residual, "cg_iterations": used, "backtracks": backtracks,
                       "cg_residual": cg_residual, "preconditioner": preconditioner,
                       "damping": damping, "damping_restarts": restarts,
                       "history": history,
                       "curvature_truncated": truncated,
                       "primal_residual": _norm(primal).astype(rate.dtype),
                       "dual_residual": _norm(dual).astype(rate.dtype)}
            accepted, new_state = jax.lax.cond(status == 0, lambda _: (proposal, updated),
                                              lambda _: (params, state), None)
            return old_loss, accepted, new_state, status

        def pause(_):
            return old_loss, params, {**state, "count": optax.safe_int32_increment(state["count"])}, jnp.asarray(0, jnp.int32)

        return jax.lax.cond(scheduled == 0, pause, active, None)

    def convergence(params, state, target):
        gradient = jax.tree.map(jnp.conj, jax.grad(smooth_loss)(params))
        multiplier = adjoint(state["duals"], params)
        total = jax.tree.map(lambda g, d: g + state["penalty"].astype(jnp.real(d).dtype) * d,
                             gradient, multiplier)
        kkt = projected_gradient(params, total)
        primal_scale = jnp.maximum(_norm(forward(params)), _norm(state["splits"]))
        dual_scale = state["penalty"] * _norm(multiplier)
        return (_finite(kkt) & (rate_at(state["count"], state["rate"].dtype) > 0)
                & (_norm(kkt) <= target * (1 + _norm(params)))
                & (state["primal_residual"] <= target * (1 + primal_scale))
                & (state["dual_residual"] <= target * (1 + dual_scale)))

    optimizer = optax.GradientTransformation(init, lambda grads, state, params: (grads, state))
    return optimizer, step, convergence
