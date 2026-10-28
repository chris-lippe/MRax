"""Synthetic fista contracts and numerical references."""


import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from scipy.optimize import minimize

from mrax.core import run_fit
from mrax.core.fit_execution import execute_optimizer
from mrax.data_struct import DataSet, Experiment, Model, Quantity
from mrax.opt._fista import make_fista_kernel
from mrax.opt.optimizers import fista_extrapolated_params, make_optimizer
from mrax.opt.tv import TVOperator, tv_prox


def _paper_reference(matrix, target, initial, weight, rate, steps):
    """Literal Algorithm 4.1, with a known Lipschitz constant and L1 prox."""
    x, y, t = initial.copy(), initial.copy(), 1.
    trace = []
    for _ in range(steps):
        z = y - rate * matrix.T @ (matrix @ y - target)
        new = np.sign(z) * np.maximum(np.abs(z) - rate * weight, 0)
        new_t = (1 + np.sqrt(1 + 4 * t * t)) / 2
        y = new + (t - 1) / new_t * (new - x)
        x, t = new, new_t
        trace.append(x.copy())
    return np.asarray(trace)


def _weighted_fista_problem():
    experiment = Experiment(dims={"frame": 3, "x": 5, "y": 1})
    experiment.observables["offsets"] = Quantity(jnp.arange(3), "1")
    experiment._mrax_dim_observables = {"frame": "offsets"}
    model = Model(name="synthetic_options", parameter_units={"level": "1", "gain": "1"},
        parameter_specs={"level": {"transform": "bounded", "bounds": (-2., 2.)},
                         "gain": {"trainable": False, "value": 2.}},
        parametrization_factory=lambda shape: ({"level": jnp.full(shape, .1)},
            lambda p: {"level": p["level"], "gain": jnp.full_like(p["level"], 2.)}),
        model_apply=lambda p, offsets, observables=None: jnp.broadcast_to(
            p["level"] * p["gain"], experiment.shape))
    values = jnp.broadcast_to(jnp.asarray([1., 3., 99.])[:, None, None], experiment.shape)
    return experiment, model, DataSet(values)





def test_fista_matches_original_paper_trace():
    rng = np.random.default_rng(24)
    matrix = rng.normal(size=(14, 9)).astype(np.float32)
    target = rng.normal(size=14).astype(np.float32)
    rate = .9 / np.linalg.norm(matrix, 2)**2
    initial = np.zeros(9, np.float32)
    expected = _paper_reference(matrix, target, initial, .05, rate, 40)
    def loss(p):
        return .5 * jnp.sum((matrix @ p["x"] - target)**2)
    opt = make_optimizer("fista", rate, regularizer_weight=.05, fista_restart="none", fista_backtracking="classic")

    def run():
        def body(carry, _):
            p, state = carry
            updates, state = opt.update(None, state, p, value_fn=loss)
            p = optax.apply_updates(p, updates)
            return (p, state), p["x"]
        p = {"x": jnp.asarray(initial)}
        return jax.lax.scan(body, (p, opt.init(p)), None, length=40)[1]

    actual = jax.jit(run)()
    np.testing.assert_allclose(actual, expected, atol=8e-7, rtol=1e-5)


def test_full_backtracking_matches_independent_published_recurrence():
    matrix = np.asarray([[1., .3], [.3, .1]], dtype=np.float64)
    target = np.asarray([1., -.2])
    x, previous, t, last_rate = np.zeros(2), np.zeros(2), 1., 4.
    expected = []
    for iteration in range(50):
        rate = last_rate * (1.5 if iteration else 1.)
        while True:
            current_t = 1. if not iteration else .5 * (1 + np.sqrt(1 + 4 * t*t * last_rate/rate))
            y = x + ((t-1)/current_t) * (x-previous)
            grad = matrix.T @ (matrix @ y-target)
            forward = y-rate*grad
            new = np.sign(forward) * np.maximum(np.abs(forward)-rate*.01, 0)
            f_y = .5 * np.sum((matrix @ y-target)**2)
            f_new = .5 * np.sum((matrix @ new-target)**2)
            d = new-y
            if f_new-f_y-grad@d <= d@d/(2*rate) + 1e-15:
                break
            rate /= 2
        previous, x, t, last_rate = x, new, current_t, rate
        expected.append(x.copy())
    enable = getattr(jax, "enable_x64", None)
    if enable is None:
        from jax.experimental import enable_x64 as enable
    with enable():
        def loss(p):
            return .5 * jnp.sum((matrix @ p["x"] - target)**2)
        opt = make_optimizer("fista", 4., regularizer_weight=.01, fista_restart="none", fista_backtracking="full")
        def run():
            def body(carry, _):
                p, state = carry
                update, state = opt.update(None, state, p, value_fn=loss)
                p = optax.apply_updates(p, update)
                return (p, state), p["x"]
            p = {"x": jnp.zeros(2, dtype=jnp.float64)}
            return jax.lax.scan(body, (p, opt.init(p)), None, length=50)[1]
        actual = np.asarray(jax.jit(run)())
        # Near the solution, roundoff-aware line-search decisions can differ
        # between independent reduction implementations. The resolvable trace
        # must agree tightly; the complete trace agrees to numerical accuracy.
        resolved = np.linalg.norm(np.asarray(expected) - expected[-1], axis=1) > 1e-5
        assert np.count_nonzero(resolved) >= 10
        np.testing.assert_allclose(actual[resolved], np.asarray(expected)[resolved], rtol=2e-10, atol=2e-11)
        np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=2e-8)


@pytest.mark.parametrize("backtracking", ["classic", "full"])
def test_optax_chain_schedule_and_exact_checkpoint_resume(backtracking):
    def loss(p):
        return .5 * jnp.sum((p["x"] - 1.)**2)
    p = {"x": jnp.zeros(4)}
    schedule = optax.piecewise_constant_schedule(.2, {17: .5})
    opt = optax.chain(make_optimizer("fista", schedule, fista_backtracking=backtracking))
    # ExtraArgs callbacks survive an ordinary Optax chain.
    def step(carry, _):
        p, state = carry
        updates, state = opt.update(jax.grad(loss)(p), state, p, value_fn=loss)
        return (optax.apply_updates(p, updates), state), None
    run = jax.jit(lambda p, state, n: jax.lax.scan(step, (p, state), None, length=n)[0], static_argnums=2)
    whole = run(p, opt.init(p), 40)
    first = run(p, opt.init(p), 13)
    resumed = run(*first, 27)
    for a, b in zip(jax.tree.leaves(whole), jax.tree.leaves(resumed), strict=True):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("masked", [False, True])
def test_fgp_box_tv_prox_matches_independent_convex_epigraph(order, masked):
    values = np.asarray([-.8, .9, -.3, 1.2, .2], dtype=np.float64)
    mask = np.asarray([1, 1, 0, 1, 1]) if masked else None
    # One spatial dimension gives a linear-program epigraph for |D x|.
    enable = getattr(jax, "enable_x64", None)
    if enable is None:
        from jax.experimental import enable_x64 as enable
    with enable():
        op = TVOperator.create(values.shape, order=order, mask=mask, weight=1.7, dtype=jnp.float64)
        matrix = np.stack([np.asarray(op.gradient(jnp.eye(5)[i]))[:, 0] for i in range(5)], axis=1)
        weights = np.asarray(op.weights)
        def objective(v):
            return .5 * np.sum((v[:5] - values)**2) + weights @ v[5:]
        def jacobian(v):
            return np.r_[v[:5] - values, weights]
        constraints = [{"type": "ineq", "fun": lambda v: v[5:] - matrix @ v[:5]},
                       {"type": "ineq", "fun": lambda v: v[5:] + matrix @ v[:5]}]
        reference = minimize(objective, np.r_[np.zeros(5), np.ones(5)], jac=jacobian,
                             constraints=constraints, bounds=[(-.5, .7)] * 5 + [(0, None)] * 5,
                             method="SLSQP", options={"ftol": 1e-12, "maxiter": 1000})
        assert reference.success
        solve = jax.jit(lambda v: tv_prox(v, (op,), 1., (-.5, .7), tolerance=1e-12,
                                         max_steps=4096, method="fgp"))
        actual, duals, gap, count = solve(jnp.asarray(values))
        np.testing.assert_allclose(actual, reference.x[:5], atol=3e-6, rtol=0)
        assert float(gap) <= 1e-12 * (1 + reference.fun) * 1.01
        warm, _, warm_gap, warm_count = jax.jit(lambda v: tv_prox(
            v, (op,), 1., (-.5, .7), duals, tolerance=1e-12, method="fgp"))(jnp.asarray(values))
        np.testing.assert_array_equal(warm, actual)
        assert int(warm_count) == 0 and float(warm_gap) <= float(gap)


def test_fista_does_not_project_extrapolation_or_require_monotone_objective():
    # A box-constrained iterate can be feasible while the paper's y is outside.
    def loss(p):
        return .5 * (p["x"] - 1.5)**2
    p = {"x": jnp.asarray(0.)}
    opt = make_optimizer("fista", .5, box_constraints=(0., 1.), fista_restart="none", fista_backtracking="classic")
    state = opt.init(p)
    saw_outside = False
    for _ in range(20):
        update, state = opt.update(None, state, p, value_fn=loss)
        p = optax.apply_updates(p, update)
        assert 0 <= float(p["x"]) <= 1.
        saw_outside |= float(state["solver"]["y"]["0"]) > 1.
    assert saw_outside


def test_exhausted_search_returns_failure_without_accepting_or_mutating_state():
    p = {"x": jnp.asarray(0.)}
    def smooth(p):
        return jnp.where(p["x"] >= 0, (p["x"] + 1.)**2, jnp.nan)
    opt, step, _ = make_fista_kernel(smooth, smooth, {"x": (-1., 1.)}, {"x": ()}, 1., max_backtracking=2)
    state = opt.init(p)
    _, result, new, status = jax.jit(step)(p, state)
    assert int(status) == 3
    for a, b in zip(jax.tree.leaves((p, state)), jax.tree.leaves((result, new)), strict=True):
        np.testing.assert_array_equal(a, b)


def test_fista_weak_scale_and_clipping_do_not_false_stop():
    p = {"x": jnp.asarray(0.)}
    def loss(p):
        return .5 * (p["x"] - 2.)**2
    opt = make_optimizer("fista", .1, grad_clip_norm=.01)
    result, _, info = execute_optimizer(loss, p, opt, num_steps=30,
                                        convergence_tolerance=1e-6, convergence_patience=2)
    assert not info.converged
    assert float(result["x"]) > 0


@pytest.mark.parametrize("backtracking", ["classic", "full"])
def test_clipping_preserves_the_native_tv_composite_optimum(backtracking):
    target = jnp.asarray([-2., 2.])
    params = {"x": jnp.zeros(2)}
    # MRax uses the mean of the two pixel norms, including the zero boundary
    # difference: weight 1 therefore gives .5 * abs(x[1] - x[0]).
    operator = TVOperator.create((2,),weight=1.)
    assert float(operator.value(target)) == pytest.approx(2.)
    def smooth(p):
        return .5 * jnp.sum((p["x"]-target)**2)
    def loss(p):
        return smooth(p) + operator.value(p["x"])
    optimizer, step, _ = make_fista_kernel(loss,smooth,{"x":(-3.,3.)},{"x":(operator,)},.1,
        grad_clip_norm=.1,restart="gradient",backtracking=backtracking)
    def body(carry,_):
        _, p, state, status = step(*carry)
        return (p,state),status
    (result,_),statuses = jax.jit(lambda:jax.lax.scan(body,
        (params,optimizer.init(params)),None,length=1000))()
    assert not np.any(statuses)
    np.testing.assert_allclose(result["x"],[-1.5,1.5],atol=4e-4,rtol=0)


def test_rounded_to_zero_float32_step_is_not_a_stationarity_certificate():
    p = {"x": jnp.asarray(1e8, jnp.float32)}
    def loss(p):
        return .5 * (p["x"] - jnp.asarray(1e8 + 8, jnp.float32))**2
    opt, _, check = make_fista_kernel(loss, loss, {"x": (-np.inf, np.inf)}, {"x": ()}, 1e-8)
    assert not bool(jax.jit(check)(p, opt.init(p), 1e-12))


def test_full_backtracking_does_not_grow_roundoff_oscillations_at_l1_solution():
    def smooth(p):
        return .5 * (p["x"] - 2.)**2
    opt = make_optimizer("fista", .1, regularizer_weight=.5, fista_backtracking="full")
    result, _, _ = execute_optimizer(smooth, {"x": jnp.asarray(0.)}, opt, num_steps=400)
    np.testing.assert_allclose(result["x"], 1.5, rtol=0, atol=2e-5)


def test_full_backtracking_recovers_a_tiny_inherited_step_without_a_nominal_floor():
    def smooth(p):
        return .5 * (p["x"] - 2.)**2
    opt, step, _ = make_fista_kernel(smooth, smooth, {"x": (-np.inf, np.inf)}, {"x": ()},
                                    .1, backtracking="full", restart="gradient")
    p = {"x": jnp.asarray(0.)}
    state = opt.init(p)
    state.update(rate=jnp.asarray(1e-14), safe_rate=jnp.asarray(1e-14),
                 count=jnp.asarray(10, jnp.int32), cached_loss=jnp.asarray(2.), grow=jnp.asarray(False))
    def run():
        def body(carry, _):
            _, x, s, status = step(*carry)
            return (x, s), status
        return jax.lax.scan(body, (p, state), None, length=120)
    (result, final), status = jax.jit(run)()
    assert not np.any(status)
    assert float(final["rate"]) > 1e-5
    np.testing.assert_allclose(result["x"], 2., rtol=0, atol=2e-5)


def test_fgp_joint_isotropic_groups_match_independent_two_dimensional_reference():
    enable = getattr(jax, "enable_x64", None)
    if enable is None:
        from jax.experimental import enable_x64 as enable
    with enable():
        values = np.asarray([[-.9, .8], [.2, 1.1]])
        operators = (TVOperator.create(values.shape, weight=.8, dtype=jnp.float64),
                     TVOperator.create(values.shape, axes=(0,), weight=.3, dtype=jnp.float64))
        matrices = [np.stack([np.asarray(op.gradient(jnp.eye(4)[i].reshape(2, 2))).reshape(4, -1)
                             for i in range(4)], axis=2) for op in operators]
        weights = np.concatenate([np.asarray(op.weights).ravel() for op in operators])
        def objective(v):
            return .5 * np.sum((v[:4] - values.ravel())**2) + weights @ v[4:]
        def constraints(v):
            norms = np.concatenate([np.linalg.norm(matrix @ v[:4], axis=1) for matrix in matrices])
            return v[4:] - norms
        reference = minimize(objective, np.r_[np.zeros(4), np.ones(8)],
            bounds=[(-.5, .7)] * 4 + [(0, None)] * 8,
            constraints=[{"type": "ineq", "fun": constraints}], method="SLSQP",
            options={"ftol": 1e-12, "maxiter": 2000})
        assert reference.success
        actual, _, gap, _ = jax.jit(lambda v: tv_prox(v, operators, 1., (-.5, .7),
            method="fgp", tolerance=1e-12, max_steps=4096))(jnp.asarray(values))
        np.testing.assert_allclose(actual.ravel(), reference.x[:4], atol=5e-6)
        assert float(gap) <= 1e-12 * (1 + reference.fun) * 1.01


def test_fgp_complex_tv_matches_two_pixel_radial_solution():
    enable = getattr(jax, "enable_x64", None)
    if enable is None:
        from jax.experimental import enable_x64 as enable
    with enable():
        value = jnp.asarray([-1.-1.j, 1.+1.j], dtype=jnp.complex128)
        op = TVOperator.create(value.shape, weight=.5, dtype=jnp.float64)
        x, _, gap, _ = jax.jit(lambda v: tv_prox(v, (op,), 1., (-np.inf, np.inf),
            method="fgp", tolerance=1e-12))(value)
        expected = np.asarray(value) * (1 - .25 / np.sqrt(2))
        np.testing.assert_allclose(x, expected, atol=1e-7, rtol=0)
        assert float(gap) < 1e-12


@pytest.mark.parametrize("injection", ["name", "optimizer", "factory"])
@pytest.mark.parametrize("packed", [False, True])
def test_native_tv_fit_accepts_same_optimizer_injection_interfaces(injection, packed):
    experiment = Experiment(dims={"frame": 1, "x": 5, "y": 1})
    def factory(shape):
        return {"level": jnp.zeros(shape)}, lambda p: p
    model = Model(name="synthetic_masked", parametrization_factory=factory,
                  model_apply=lambda p, offsets, observables=None: p["level"][None],
                  parameter_units={"level": "1"},
                  parameter_specs={"level": {"transform": "bounded", "bounds": (-2., 2.)}})
    data = DataSet(jnp.asarray([[[-1.], [-1.], [99.], [1.], [1.]]]))
    options = dict(regularizer_configs={"level": [{"type": "tv", "weight": .05}]},
                   mask=jnp.asarray([[1.], [1.], [0.], [1.], [1.]]), fit_active_only=packed,
                   step_size=.5, num_steps=80, return_fit_info=True)
    if injection == "name":
        options["optimizer_name"] = "fista"
    elif injection == "optimizer":
        options["optimizer"] = make_optimizer("fista", .5)
    else:
        options["optimizer_factory"] = lambda rate: make_optimizer("fista", rate)
    fields, _, info = run_fit(experiment, model, data, **options)
    assert info.regularization_mode == "proximal_tv"
    assert info.optimizer_name == "fista"
    np.testing.assert_allclose(fields["level"].values[:, 0], [-1., -1., 0., 1., 1.], atol=2e-4)


def test_named_and_injected_fista_have_identical_default_proximal_accuracy():
    experiment,model,_ = _weighted_fista_problem()
    target=jnp.asarray([-1.,-.3,.2,1.,.5])
    data=DataSet(jnp.broadcast_to(2*target[None,:,None],experiment.shape))
    options=dict(step_size=.3,num_steps=35,return_fit_info=True,
        regularizer_configs={"level":[{"type":"tv","weight":.2}]})
    results=[run_fit(experiment,model,data,optimizer_name="fista",**options),
             run_fit(experiment,model,data,optimizer=make_optimizer("fista",.3),**options),
             run_fit(experiment,model,data,optimizer_factory=lambda rate:make_optimizer("fista",rate),**options)]
    for fields,history,info in results[1:]:
        np.testing.assert_array_equal(fields["level"].values,results[0][0]["level"].values)
        np.testing.assert_array_equal(history,results[0][1])
        for first,second in zip(jax.tree.leaves(info.state.optimizer_state),
                                jax.tree.leaves(results[0][2].state.optimizer_state),strict=True):
            np.testing.assert_array_equal(first,second)


def test_direct_fista_rejects_gradients_at_an_unspecified_point():
    optimizer = make_optimizer("fista", .1)
    params = {"x": jnp.asarray(0.)}
    with pytest.raises(ValueError, match="extrapolated point"):
        optimizer.update({"x": jnp.asarray(-1.)}, optimizer.init(params), params)


@pytest.mark.parametrize("rate", [0., .1])
def test_direct_fista_accepts_a_python_scalar_constant_objective(rate):
    optimizer = make_optimizer("fista", rate)
    params = {"x": jnp.asarray([1., -1.])}
    update, state = jax.jit(lambda p, s: optimizer.update(None, s, p, value_fn=lambda q: 0.))(
        params, optimizer.init(params))
    np.testing.assert_array_equal(update["x"], jnp.zeros(2))
    assert int(state["status"]) == 0


def test_fista_custom_coupled_prox_and_box_match_independent_convex_optimum():
    target = np.asarray([1.1, .2])
    reference = minimize(lambda x: .5 * np.sum((x-target)**2) + .2 * np.sum(x)**2,
                         np.zeros(2), bounds=[(0., .3), (0., .7)], method="SLSQP",
                         options={"ftol": 1e-12})
    assert reference.success
    enable = getattr(jax, "enable_x64", None)
    if enable is None:
        from jax.experimental import enable_x64 as enable
    with enable():
        def prox(value, threshold):
            return value - threshold * jnp.sum(value) / (1 + 2 * threshold)
        optimizer = make_optimizer("fista", .2, regularizer_prox=prox, regularizer_weight=.4,
            box_constraints=(jnp.zeros(2), jnp.asarray([.3, .7])), proximal_tolerance=1e-12)
        params = {"x": jnp.zeros(2, dtype=jnp.float64)}
        def loss(p):
            return .5 * jnp.sum((p["x"] - target)**2)
        def body(carry, _):
            p, state = carry
            updates, state = optimizer.update(None, state, p, value_fn=loss)
            return (optax.apply_updates(p, updates), state), state["status"]
        (result, _), statuses = jax.jit(lambda: jax.lax.scan(body,
            (params, optimizer.init(params)), None, length=100))()
        assert not np.any(statuses)
        np.testing.assert_allclose(result["x"], reference.x, atol=2e-6, rtol=0)


def test_explicit_fixed_step_optax_loop_matches_original_fista():
    optimizer = make_optimizer("fista", .2, fista_restart="none", fista_backtracking="classic")
    params = {"x": jnp.asarray(0.)}
    state = optimizer.init(params)
    expected, extrapolated, momentum = 0., 0., 1.
    for _ in range(15):
        next_momentum = .5 * (1 + np.sqrt(1 + 4 * momentum**2))
        y = fista_extrapolated_params(params,state)["x"]
        updates, state = optimizer.update({"x": y - 1.}, state, params,
                                         gradient_at_extrapolated=True)
        params = optax.apply_updates(params, updates)
        accepted = extrapolated - .2 * (extrapolated - 1.)
        np.testing.assert_allclose(params["x"], accepted, atol=2e-7, rtol=0)
        extrapolated = accepted + (momentum - 1) / next_momentum * (accepted - expected)
        expected, momentum = accepted, next_momentum


def test_fixed_step_gradient_point_preserves_nested_optax_parameter_trees():
    params = {"block": [jnp.zeros(2),jnp.zeros(1)]}
    optimizer = make_optimizer("fista",.2,fista_backtracking="classic",fista_restart="none")
    def loss(p):
        return .5 * sum(jnp.sum((leaf-1.)**2) for leaf in jax.tree.leaves(p))
    def body(carry,_):
        p,state=carry
        point=fista_extrapolated_params(p,state)
        assert jax.tree.structure(point)==jax.tree.structure(p)
        updates,state=optimizer.update(jax.grad(loss)(point),state,p,gradient_at_extrapolated=True)
        p=optax.apply_updates(p,updates)
        return (p,state),jnp.concatenate(jax.tree.leaves(p))
    _,trace=jax.jit(lambda:jax.lax.scan(body,(params,optimizer.init(params)),None,length=20))()
    expected=_paper_reference(np.eye(3),np.ones(3),np.zeros(3),0.,.2,20)
    np.testing.assert_allclose(trace,expected,atol=3e-7,rtol=0)


def test_extrapolated_point_handles_numeric_dictionary_keys_without_reordering():
    params={str(i):jnp.asarray(float(i)) for i in range(12)}
    optimizer=make_optimizer("fista",.2)
    point=fista_extrapolated_params(params,optimizer.init(params))
    for key in params:
        np.testing.assert_array_equal(point[key],params[key])


@pytest.mark.parametrize("backtracking", ["classic", "full"])
def test_zero_schedule_pauses_and_resumes_without_false_convergence(backtracking):
    def schedule(count):
        return jnp.where((count < 2) | ((count >= 5) & (count < 8)), 0., .1)
    def loss(p):
        return .5 * (p["x"] - 2.)**2
    params = {"x": jnp.asarray(0.)}
    optimizer, step, convergence = make_fista_kernel(loss, loss, {"x": (-3., 3.)}, {"x": ()},
        schedule, backtracking=backtracking)
    state = optimizer.init(params)
    advance = jax.jit(step)
    for iteration in range(12):
        previous = params
        _, params, state, status = advance(params, state)
        assert int(status) == 0
        assert all(np.all(np.isfinite(value)) for value in jax.tree.leaves(state))
        if iteration < 2 or 5 <= iteration < 8:
            np.testing.assert_array_equal(params["x"], previous["x"])
        if int(schedule(state["count"]) == 0):
            assert not bool(jax.jit(convergence)(params, state, jnp.asarray(1e-5)))
    assert float(params["x"]) > 0
    frozen = make_optimizer("fista", 0.)
    updates, final = jax.jit(lambda p, s: frozen.update(None, s, p, value_fn=loss))(
        params, frozen.init(params))
    np.testing.assert_array_equal(updates["x"], 0.)
    assert int(final["status"]) == 0


def test_explicit_fista_fit_rejects_wrappers_that_lose_native_tv_binding():
    experiment, model, data = _weighted_fista_problem()
    with pytest.raises(ValueError, match="native fitting interface"):
        run_fit(experiment, model, data, optimizer_name="fista",
            optimizer=optax.chain(make_optimizer("fista", .3)),
            regularizer_configs={"level": [{"type": "tv", "weight": .02}]}, num_steps=3)


def test_native_tv_does_not_silently_override_explicit_raw_coordinates():
    experiment, model, data = _weighted_fista_problem()
    with pytest.raises(ValueError, match="Native field TV"):
        run_fit(experiment, model, data, optimizer_name="fista", parameter_space="raw",
            regularizer_configs={"level": [{"type": "tv", "weight": .02}]}, num_steps=3)


@pytest.mark.parametrize("injection", ["name", "optimizer", "factory"])
@pytest.mark.parametrize("option", ["weights", "pooling", "reference", "normalization", "missing", "metric", "derived"])
def test_native_tv_options_match_weighted_least_squares_reference(injection, option):
    experiment, model, data = _weighted_fista_problem()
    options = dict(regularizer_configs={"level": [{"type": "tv", "weight": .02}]},
                   step_size=.3, num_steps=100, proximal_tolerance=1e-8,
                   frame_mask=jnp.asarray([1., .25, 0.]),
                   data_variance=jnp.asarray([1., 2., 1.]), return_fit_info=True)
    target = (1. + .125 * 3.) / (1.125 * 2.)
    if option == "weights":
        options.update(mask=jnp.asarray([[1., 1., 0., 1., 1.]]), mask_dims=("y", "x"),
                       fit_active_only=True, mask_fill_value=jnp.nan)
    elif option == "pooling":
        model.parameter_constant_dims = {"level": ("x", "y")}
    elif option == "reference":
        options.update(reference_image=jnp.full((5, 1), 2.), return_model_callable=True)
        target /= 2
    elif option == "normalization":
        options["data_normalization"] = "full"
    elif option == "missing":
        data = DataSet(jnp.asarray(data.data).at[2].set(jnp.nan))
    elif option == "metric":
        options["physical_metric"] = "initial_gradient"
    else:
        model.derived_maps = lambda fields, observables: {"double": 2 * fields["level"].values}
        model.derived_map_units = {"double": "1"}
        options["compute_derived_maps"] = True
    if injection == "name":
        options["optimizer_name"] = "fista"
    elif injection == "optimizer":
        options["optimizer"] = make_optimizer("fista", .3, proximal_tolerance=1e-8)
    else:
        options["optimizer_factory"] = lambda rate: make_optimizer("fista", rate, proximal_tolerance=1e-8)
    result = run_fit(experiment, model, data, **options)
    fields, info = result[0], result[-1]
    observed = np.asarray(fields["level"].values)
    np.testing.assert_allclose(observed[np.isfinite(observed)], target, atol=3e-4, rtol=0)
    np.testing.assert_allclose(np.asarray(fields["gain"].values)[np.isfinite(observed)], 2., atol=0)
    assert info.regularization_mode == "proximal_tv" and info.optimizer_name == "fista"
    assert "gain" not in info.state.params
    if option == "weights":
        assert np.isnan(observed[2, 0])
        assert sum(v.size for v in info.state.params.values()) == 4
    if option == "pooling":
        assert info.state.params["level"].size == 1
    if option == "reference":
        prediction = result[2]({key: value.values for key, value in fields.items()})
        np.testing.assert_allclose(prediction, target * 4, atol=1e-3)
    if option == "derived":
        np.testing.assert_allclose(fields["double"].values, 2*target, atol=6e-4)


@pytest.mark.parametrize("injection", ["name", "optimizer", "factory"])
def test_native_tv_initial_parameters_and_zero_step_fit_do_not_evaluate_the_model(injection):
    experiment, model, data = _weighted_fista_problem()
    def forbidden_prediction(fields, offsets, observables=None):
        raise AssertionError("Zero-step fitting must not evaluate the model")
    model.model_apply = forbidden_prediction
    options = dict(regularizer_configs={"level": [{"type": "tv", "weight": .02}]},
                   step_size=.3,num_steps=0,initial_params={"level":jnp.full((5,1),.4)},
                   return_fit_info=True)
    if injection == "name":
        options["optimizer_name"] = "fista"
    elif injection == "optimizer":
        options["optimizer"] = make_optimizer("fista",.3)
    else:
        options["optimizer_factory"] = lambda rate: make_optimizer("fista",rate)
    fields,history,info = run_fit(experiment,model,data,**options)
    assert history == [] and info.termination_reason == "no_steps"
    assert info.state.step == 0
    np.testing.assert_allclose(fields["level"].values,.4,atol=1e-7,rtol=0)


@pytest.mark.parametrize("injection", ["name", "optimizer", "factory"])
def test_native_tv_resume_schedule_progress_and_convergence(injection):
    experiment, model, data = _weighted_fista_problem()
    rate = optax.piecewise_constant_schedule(.3, {9: .5})
    options = dict(regularizer_configs={"level": [{"type": "tv", "weight": .02}]},
                   step_size=rate, frame_mask=jnp.asarray([1., 0., 0.]), return_fit_info=True,
                   chunk_size=4, proximal_tolerance=1e-8)
    if injection == "name":
        options["optimizer_name"] = "fista"
    elif injection == "optimizer":
        options["optimizer"] = make_optimizer("fista", rate, proximal_tolerance=1e-8)
    else:
        options["optimizer_factory"] = lambda rate: make_optimizer("fista", rate, proximal_tolerance=1e-8)
    whole, history, whole_info = run_fit(experiment, model, data, num_steps=30, **options)
    _, first_history, first_info = run_fit(experiment, model, data, num_steps=11, **options)
    resumed, last_history, last_info = run_fit(experiment, model, data, num_steps=19,
        initial_state=first_info.state, **options)
    np.testing.assert_array_equal(resumed["level"].values, whole["level"].values)
    np.testing.assert_array_equal(first_history + last_history, history)
    assert last_info.state.step == whole_info.state.step == 30
    progress = []
    _, _, info = run_fit(experiment, model, data, num_steps=150,
        convergence_tolerance=1e-6, parameter_tolerance=1e-5, convergence_patience=3, min_steps=10,
        progress_callback=lambda step, total, text: progress.append(step), **options)
    assert info.converged and 10 <= info.iterations < 150
    assert progress[0] == 0 and progress[-1] == info.iterations


@pytest.mark.parametrize("backtracking", ["classic", "full"])
def test_domain_restart_retains_certified_curvature(backtracking):
    def loss(p):
        return -jnp.log(p["x"][0])
    p = {"x": jnp.asarray([1e-8])}
    opt, step, _ = make_fista_kernel(loss, loss, {"x": (1e-12, 1.)}, {"x": ()}, .02,
                                      restart="gradient", backtracking=backtracking)
    s = opt.init(p)
    s.update(previous={"x": jnp.asarray([1.])}, y={"x": jnp.asarray([-1.])},
             t=jnp.asarray(10.), rate=jnp.asarray(1e-20), safe_rate=jnp.asarray(1e-20),
             grow=jnp.asarray(True), count=jnp.asarray(10, jnp.int32), cached_loss=loss(p))
    _, q, ns, status = jax.jit(step)(p, s)
    assert int(status) == 0 and int(ns["count"]) == 11
    assert float(loss(q)) < float(loss(p))
    assert np.isfinite(float(ns["rate"])) and float(ns["rate"]) > 0


@pytest.mark.parametrize("backtracking", ["classic", "full"])
def test_invalid_accepted_domain_preserves_checkpoint(backtracking):
    def loss(p):
        return -jnp.log(p["x"][0])
    p = {"x": jnp.asarray([-1.])}
    opt, step, _ = make_fista_kernel(loss, loss, {"x": (-2., 2.)}, {"x": ()}, .02,
                                      restart="gradient", backtracking=backtracking)
    state = opt.init(p)
    _, q, new, status = jax.jit(step)(p, state)
    assert int(status) == 3
    for a, b in zip(jax.tree.leaves((p, state)), jax.tree.leaves((q, new)), strict=True):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("backtracking", ["classic", "full"])
def test_unresolvable_nonstationary_step_is_not_counted(backtracking):
    def loss(p):
        return 1e10 * p["x"][0]
    p = {"x": jnp.asarray([1.], jnp.float32)}
    opt, step, _ = make_fista_kernel(loss, loss, {"x": (-2., 2.)}, {"x": ()}, .02,
                                      restart="gradient", backtracking=backtracking)
    state = opt.init(p)
    state.update(rate=jnp.asarray(1e-30, jnp.float32), safe_rate=jnp.asarray(1e-30, jnp.float32))
    _, q, new, status = jax.jit(step)(p, state)
    assert int(status) == 4
    for a, b in zip(jax.tree.leaves((p, state)), jax.tree.leaves((q, new)), strict=True):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("backtracking", ["classic", "full"])
def test_overflowed_majorization_comparison_is_not_a_certificate(backtracking):
    def loss(p):
        return jnp.sum(jnp.sin(1e20 * p['x']))
    p = {'x': jnp.zeros(4, jnp.float32)}
    opt, step, _ = make_fista_kernel(loss, loss, {'x': (-np.inf, np.inf)}, {'x': ()}, .02,
        backtracking=backtracking, max_backtracking=1)
    state = opt.init(p)
    _, q, new, status = jax.jit(step)(p, state)
    assert int(status) == 3
    for a, b in zip(jax.tree.leaves((p, state)), jax.tree.leaves((q, new)), strict=True):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("backtracking", ["classic", "full"])
def test_rounding_guard_uses_least_precise_parameter_dtype(backtracking):
    enable = getattr(jax, 'enable_x64', None)
    if enable is None:
        from jax.experimental import enable_x64
        enable = enable_x64
    with enable():
        def loss(p):
            return p['single'][0] + p['double'][0]**2
        p = {'single': jnp.asarray([1.], jnp.float32), 'double': jnp.asarray([0.], jnp.float64)}
        opt, step, _ = make_fista_kernel(loss, loss, {k: (-2., 2.) for k in p}, {k: () for k in p}, .02,
            backtracking=backtracking)
        state = opt.init(p)
        state.update(rate=jnp.asarray(1e-10, jnp.float64), safe_rate=jnp.asarray(1e-10, jnp.float64))
        _, q, new, status = jax.jit(step)(p, state)
        assert int(status) == 4
        for a, b in zip(jax.tree.leaves((p, state)), jax.tree.leaves((q, new)), strict=True):
            np.testing.assert_array_equal(a, b)


def test_fista_reuses_trial_objective():
    calls = []

    def smooth(params):
        return .5 * jnp.sum((params['x'] - 2.)**2)

    def objective(params):
        jax.debug.callback(lambda: calls.append(1), ordered=True)
        return smooth(params)

    params = {'x': jnp.zeros(2)}
    optimizer, step, _ = make_fista_kernel(objective, smooth, {'x': (-np.inf, np.inf)},
        {'x': ()}, .1, restart="gradient", backtracking="full")
    _, _, _, status = jax.jit(step)(params, optimizer.init(params))
    jax.effects_barrier()
    assert int(status) == 0
    assert len(calls) == 2
