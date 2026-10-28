"""Synthetic split bregman contracts and numerical references."""

import copy

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from mrax.core import build_loss, run_fit
from mrax.core.data_losses import pseudo_huber_loss
from mrax.data_struct import DataSet, Experiment, Model
from mrax.opt._bregman_preconditioner import (
    apply_preconditioner,
    build_preconditioner,
    initial_preconditioner,
    shifted_preconditioner,
)
from mrax.opt._bregman_quasi_newton import initial_history, inverse_product, update_history
from mrax.opt._split_bregman import make_split_bregman_kernel
from mrax.opt.optimizers import make_optimizer
from mrax.opt.tv import TVOperator
from mrax.reco.variational import _squared_magnitude, build_operator, solve_variational


def _same_tree(left, right):
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        np.testing.assert_array_equal(a, b)


def _scan(optimizer, loss, initial, count):
    @jax.jit
    def solve(params, state):
        def update(carry, _):
            point, state = carry
            updates, state = optimizer.update(None, state, point, value_fn=loss)
            return (optax.apply_updates(point, updates), state), state["status"]
        return jax.lax.scan(update, (params, state), None, length=count)
    return solve(initial, optimizer.init(initial))


def test_split_bregman_matches_independent_dense_recurrence():
    with jax.enable_x64():
        matrix = np.asarray([[4., 1.], [1., 3.]])
        target = np.asarray([1.2, -.7])
        weights = np.asarray([.3, .1])
        rho = .8
        scale = 1.

        def loss(p):
            residual = p["x"] - target
            return .5 * residual @ matrix @ residual

        optimizer = make_optimizer("split_bregman", 1., regularizer_weight={"x": weights},
                                   bregman_penalty=rho, proximal_tolerance=1e-14)
        params = {"x": jnp.zeros(2, jnp.float64)}
        state = optimizer.init(params)
        x = np.zeros(2)
        split = scale * x
        dual = np.zeros(2)
        np.testing.assert_array_equal(state["solver"]["duals"]["0"][0][..., 0], dual)
        for _ in range(8):
            x = np.linalg.solve(matrix + rho * scale ** 2 * np.eye(2),
                                matrix @ target + rho * scale * (split - dual))
            argument = scale * x + dual
            split = np.sign(argument) * np.maximum(np.abs(argument) - weights / (rho * scale), 0)
            dual = dual + scale * x - split
            updates, state = optimizer.update(None, state, params, value_fn=loss)
            params = optax.apply_updates(params, updates)
            assert int(state["status"]) == 0
            np.testing.assert_allclose(params["x"], x, atol=3e-13, rtol=0)
            np.testing.assert_allclose(state["solver"]["splits"]["0"][0][..., 0], split, atol=3e-13)
            np.testing.assert_allclose(state["solver"]["duals"]["0"][0][..., 0], dual, atol=3e-13)
            assert float(state["solver"]["penalty"]) == rho


@pytest.mark.parametrize("complex_field", [False, True])
def test_true_tv_two_pixel_solution_and_kkt(complex_field):
    dtype = jnp.complex64 if complex_field else jnp.float32
    target = jnp.asarray([-1., 1.], dtype)
    if complex_field:
        target = target * (1 + 1j) / jnp.sqrt(2.)
    operator = TVOperator.create((2,), weight=.5)

    def smooth(p):
        return .5 * jnp.sum(jnp.abs(p["x"] - target) ** 2)

    def loss(p):
        return smooth(p) + operator.value(p["x"])

    bounds = (-np.inf, np.inf) if complex_field else (-2., 2.)
    optimizer = make_optimizer("split_bregman", 1.,
        proximal_context=(loss, smooth, {"x": bounds}, {"x": (operator,)}))
    params = {"x": jnp.zeros_like(target)}
    state = optimizer.init(params)

    @jax.jit
    def solve(params, state):
        def update(carry, _):
            point, state = carry
            _, state = optimizer.update(None, state, point, value_fn=smooth)
            return (state["accepted"], state), state["status"]
        return jax.lax.scan(update, (params, state), None, length=80)

    (result, state), statuses = solve(params, state)
    np.testing.assert_array_equal(statuses, 0)
    np.testing.assert_allclose(result["x"], .75 * target, atol=8e-6, rtol=0)
    assert bool(optimizer.update._mrax_convergence(result, state, 2e-5, value_fn=smooth))


def test_schedule_pause_freezes_primal_split_and_multiplier_and_resumes():
    def rate(count):
        return jnp.where((count == 1) | (count == 2), 0., 1.)

    def loss(p):
        return .5 * jnp.sum((p["x"] - 2.) ** 2)

    optimizer = make_optimizer("split_bregman", rate, regularizer_weight=.2)
    point = {"x": jnp.asarray([0., 1.])}
    state = optimizer.init(point)
    changes, state = optimizer.update(None, state, point, value_fn=loss)
    point = optax.apply_updates(point, changes)
    for _ in range(2):
        before = copy.copy(state)
        changes, state = optimizer.update(None, state, point, value_fn=loss)
        np.testing.assert_array_equal(changes["x"], 0)
        for name in ["splits", "duals", "rate", "penalty"]:
            _same_tree(before["solver"][name], state["solver"][name])
    changes, state = optimizer.update(None, state, point, value_fn=loss)
    assert np.any(np.asarray(changes["x"]) != 0)
    assert int(state["solver"]["count"]) == 4


def test_missing_objective_rejected_and_nonfinite_trial_preserves_checkpoint():
    optimizer = make_optimizer("split_bregman", .1, regularizer_weight=.1)
    point = {"x": jnp.asarray([0., 1.])}
    state = optimizer.init(point)
    with pytest.raises(ValueError, match="requires value_fn"):
        optimizer.update(jax.tree.map(jnp.zeros_like, point), state, point)
    updates, failed = jax.jit(lambda p, s: optimizer.update(
        None, s, p, value_fn=lambda p: jnp.asarray(jnp.nan)))(point, state)
    assert int(failed["status"]) == 3
    _same_tree(failed["solver"], state["solver"])
    np.testing.assert_array_equal(updates["x"], 0)


def test_split_bregman_rejects_incompatible_state_schema():
    optimizer = make_optimizer("split_bregman", .1)
    state = optimizer.init({"x": jnp.asarray(0.)})
    state["solver"].pop("layout_version")
    with pytest.raises(ValueError, match="checkpoint layout changed"):
        optimizer.update._mrax_restore_state(state)


@pytest.mark.parametrize('spatial_shape', [(2, 1), (8, 8)])
def test_named_injected_factory_tv_fits_agree_without_extra_options(spatial_shape):
    experiment = Experiment(dims={"frame": 1, "x": spatial_shape[0], "y": spatial_shape[1]})

    def factory(shape):
        return {"level": jnp.zeros(shape)}, lambda p: p

    model = Model(name="synthetic_image", parametrization_factory=factory,
                  model_apply=lambda fields, offsets, observables=None: fields["level"][None],
                  parameter_units={"level": "1"},
                  parameter_specs={"level": {"transform": "bounded", "bounds": (-2., 2.)}})
    data = DataSet(jnp.linspace(-1., 1., np.prod(spatial_shape)).reshape((1,) + spatial_shape))
    regularizers = {"level": [{"type": "tv", "axes": (0,), "weight": .5}]}
    routes = [dict(optimizer_name="split_bregman"),
              dict(optimizer=make_optimizer("split_bregman", .02)),
              dict(optimizer_factory=lambda rate: make_optimizer("split_bregman", rate))]
    results = []
    for route in routes:
        fields, _, info = run_fit(experiment, model, data, step_size=.02, num_steps=40,
                                  regularizer_configs=regularizers, return_fit_info=True, **route)
        results.append((fields["level"].values, info.state.optimizer_state))
    _same_tree(results[0], results[1])
    _same_tree(results[0], results[2])


def test_zero_rate_is_a_pause_and_zero_steps_do_not_evaluate_objective():
    def unused(_):
        raise AssertionError("initialization evaluated an objective")

    optimizer = make_optimizer("split_bregman", 0.,
        proximal_context=(unused, unused, {"x": (-1., 1.)}, {"x": ()}))
    state = optimizer.init({"x": jnp.asarray(0.)})
    assert int(state["solver"]["count"]) == 0


def test_primal_preconditioner_inverts_anisotropic_quadratic_curvature():
    with jax.enable_x64():
        shape = (24, 5)
        first = np.diff(np.eye(shape[0]), axis=0)
        second = np.diff(np.eye(shape[1]), axis=0)
        matrix = (.2 * np.eye(np.prod(shape))
                  + 3000 * np.kron(first.T @ first, np.eye(shape[1]))
                  + .7 * np.kron(np.eye(shape[0]), second.T @ second))
        point = {"x": jnp.zeros(shape, jnp.float64)}

        def product(value):
            return {"x": (matrix @ value["x"].ravel()).reshape(shape)}

        state = jax.jit(lambda p: build_preconditioner(p, product, initial_preconditioner(p)))(point)
        rhs = {"x": jnp.sin(jnp.arange(np.prod(shape), dtype=jnp.float64)).reshape(shape)}
        result = jax.jit(apply_preconditioner)(rhs, state)["x"]
        np.testing.assert_allclose(matrix @ np.asarray(result).ravel(), np.asarray(rhs["x"]).ravel(),
                                   rtol=0, atol=2e-8)
        assert bool(state["valid"])


def test_primal_preconditioner_keeps_coupled_real_and_complex_modes():
    with jax.enable_x64():
        point = {"field": jnp.zeros((4, 3), jnp.complex128), "scalar": jnp.asarray(0., jnp.float64)}

        def smooth(p):
            residual = p["field"] - (1 + 2j) * p["scalar"]
            return .5 * jnp.real(jnp.vdot(residual, residual)) + .5 * p["scalar"] ** 2

        def gradient(p):
            return jax.tree.map(jnp.conj, jax.grad(smooth)(p))
        _, product = jax.linearize(gradient, point)
        state = build_preconditioner(point, product, initial_preconditioner(point))
        rhs = {"field": jnp.full((4, 3), .3 + .7j, jnp.complex128),
               "scalar": jnp.asarray(.2, jnp.float64)}
        solution = apply_preconditioner(rhs, state)
        reconstructed = product(solution)
        for key in rhs:
            np.testing.assert_allclose(reconstructed[key], rhs[key], atol=2e-10, rtol=0)
        assert jnp.real(sum(jnp.vdot(rhs[k], solution[k]) for k in rhs)) > 0


def test_invalid_curvature_uses_a_finite_identity_preconditioner():
    point = {"x": jnp.ones(3)}
    state = build_preconditioner(point,
        lambda value: {"x": jnp.full_like(value["x"], jnp.nan)}, initial_preconditioner(point))
    assert not bool(state["valid"])
    _same_tree(apply_preconditioner(point, state), point)


def test_coupled_nearly_singular_primal_agrees_with_cpu_on_gpu():
    try:
        gpu = jax.devices("gpu")[0]
    except RuntimeError:
        pytest.skip("GPU backend unavailable")
    cpu = jax.devices("cpu")[0]
    solutions = []
    with jax.enable_x64():
        target = .2 + .1 * np.sin(np.arange(64).reshape(8, 8))
        for device in (cpu, gpu):
            with jax.default_device(device):
                observed = jnp.asarray(target)
                initial = {key: jnp.zeros((8, 8), jnp.float64) for key in ("a", "b")}

                def smooth(p, observed=observed):
                    residual = p["a"] + p["b"] - observed
                    difference = p["a"] - p["b"]
                    return .5 * jnp.mean(residual ** 2) + 5e-13 * jnp.mean(difference ** 2)

                bounds = {key: (-1., 1.) for key in initial}
                operators = {key: () for key in initial}
                optimizer = make_optimizer("split_bregman", .2, device=device,
                    proximal_context=(smooth, smooth, bounds, operators))
                (fitted, _state), statuses = _scan(optimizer, smooth, initial, 24)
                np.testing.assert_array_equal(statuses, np.zeros(24, dtype=np.int32))
                solutions.append({key: np.asarray(value) for key, value in fitted.items()})
        for key in initial:
            np.testing.assert_allclose(solutions[0][key], target / 2, atol=1e-6, rtol=0)
            np.testing.assert_allclose(solutions[1][key], solutions[0][key], atol=1e-6, rtol=0)


def test_builtin_complex_least_squares_has_the_quadratic_hessian_at_exact_fit():
    experiment = Experiment(dims={"frame": 1, "x": 2, "y": 1})

    def factory(shape):
        return {"z": jnp.full(shape, 1 + 2j, jnp.complex64)}, lambda p: p

    model = Model(name="synthetic_complex", parametrization_factory=factory,
                  model_apply=lambda fields, offsets, observables=None: fields["z"][None],
                  parameter_units={"z": "1"})
    data = DataSet(jnp.full((1, 2, 1), 1 + 2j, jnp.complex64))
    loss, params, _ = build_loss(experiment, model, data)

    def gradient(p):
        return jax.tree.map(jnp.conj, jax.grad(loss)(p))

    _, product = jax.linearize(gradient, params)
    direction = {"z": jnp.full_like(params["z"], 3 + 4j)}
    np.testing.assert_allclose(product(direction)["z"], .2 * direction["z"], atol=4e-7, rtol=0)


def test_grouped_pseudo_huber_keeps_complex_quadratic_curvature_at_zero():
    residual = jnp.zeros((4, 3), jnp.complex64)
    weights = jnp.ones_like(jnp.real(residual))

    def loss(value):
        return pseudo_huber_loss(value, weights, delta=.5, group_size=2)

    _, product = jax.linearize(lambda p: jnp.conj(jax.grad(loss)(p)), residual)
    direction = jnp.full_like(residual, .3 + .7j)
    np.testing.assert_allclose(product(direction), direction / 6, atol=2e-8, rtol=0)


def test_negative_curvature_damps_the_primal_equation_without_changing_objective():
    def loss(point):
        x = point['x']
        return jnp.sum(.25 * x ** 4 - .5 * x ** 2)

    optimizer = make_optimizer('split_bregman', 1.)
    point = {'x': jnp.asarray([.1, -.1])}
    state = optimizer.init(point)
    update = jax.jit(lambda p, s: optimizer.update(None, s, p, value_fn=loss))
    previous = float(loss(point))
    for index in range(12):
        change, state = update(point, state)
        point = optax.apply_updates(point, change)
        assert int(state['status']) == 0
        value = float(loss(point))
        assert value <= previous + 2e-7
        if index == 0:
            assert float(state['solver']['damping']) > 0
            assert int(state['solver']['damping_restarts']) > 0
        previous = value
    np.testing.assert_allclose(jnp.abs(point['x']), 1., atol=2e-5, rtol=0)


def test_newton_shift_matches_independent_inverse_of_the_shifted_curvature():
    with jax.enable_x64():
        point = {'x': jnp.zeros(2, jnp.float64), 'y': jnp.asarray(0., jnp.float64)}

        def smooth(p):
            return jnp.sum((p['x'] - p['y']) ** 2) + .5 * p['y'] ** 2

        _, product = jax.linearize(jax.grad(smooth), point)
        base = build_preconditioner(point, product, initial_preconditioner(point))
        shift = jnp.asarray(.7, jnp.float64)
        state = shifted_preconditioner(base, shift)
        rhs = {'x': jnp.asarray([.3, -.2]), 'y': jnp.asarray(.4)}
        solved = apply_preconditioner(rhs, state)
        reconstructed = product(solved)
        for key in point:
            np.testing.assert_allclose(reconstructed[key] + shift * solved[key], rhs[key],
                                       rtol=0, atol=3e-12)


def test_inexact_large_primal_solve_reaches_independent_separable_solution():
    diagonal = jnp.geomspace(1., 1000., 128)
    target = jnp.sin(jnp.arange(128) * .17)
    weight = .03

    def loss(p):
        return .5 * jnp.sum(diagonal * (p['x'] - target) ** 2)

    optimizer = make_optimizer('split_bregman', 1., regularizer_weight=weight,
                               proximal_max_steps=4)
    point = {'x': jnp.zeros_like(target)}
    (result, state), status = _scan(optimizer, loss, point, 250)
    np.testing.assert_array_equal(status, 0)
    expected = jnp.sign(target) * jnp.maximum(jnp.abs(target) - weight / diagonal, 0)
    np.testing.assert_allclose(result['x'], expected, atol=3e-4, rtol=0)
    assert int(state['solver']['cg_iterations']) <= 4


def test_local_equilibration_handles_an_isolated_stiff_location():
    with jax.enable_x64():
        diagonal = jnp.geomspace(.1, 100., 64).at[17].set(1e12)
        point = {'x': jnp.zeros(64, jnp.float64)}

        def product(value):
            return {'x': diagonal * value['x']}

        state = build_preconditioner(point, product, initial_preconditioner(point))
        rhs = {'x': jnp.sin(jnp.arange(64, dtype=jnp.float64))}
        reconstructed = product(apply_preconditioner(rhs, state))
        # The symmetric diagonal factors span sqrt(condition); allow their
        # floating-point amplification when transforms mix the field modes.
        rounding = 64 * np.finfo(float).eps * np.sqrt(float(diagonal.max() / diagonal.min()))
        np.testing.assert_allclose(reconstructed['x'], rhs['x'], atol=rounding, rtol=0)
        assert bool(state['valid'])


def test_curvature_retries_share_the_requested_cg_work_limit():
    def loss(point):
        return jnp.sum(.25 * point['x'] ** 4 - .5 * point['x'] ** 2)

    optimizer = make_optimizer('split_bregman', 1., proximal_max_steps=1)
    point = {'x': jnp.asarray([.1, -.1])}
    state = optimizer.init(point)
    changes, failed = jax.jit(lambda p, s: optimizer.update(None, s, p, value_fn=loss))(point, state)
    assert int(failed['status']) == 3
    np.testing.assert_array_equal(changes['x'], 0)
    _same_tree(failed['solver'], state['solver'])


@pytest.mark.parametrize('capacity', [2, 4])
def test_limited_memory_inverse_matches_independent_dense_bfgs(capacity):
    with jax.enable_x64():
        matrix = np.asarray([[4., 1., .2], [1., 2., .3], [.2, .3, 3.]])
        initial = np.diag([1., 2., 3.])
        history = initial_history({'x': jnp.zeros(3, jnp.float64)}, capacity)
        pairs = []
        rhs = jnp.asarray([.4, -.2, .7], jnp.float64)
        for displacement in [np.asarray([1., 2., -1.]), np.asarray([0., 1., 1.]), np.asarray([2., -1., 0.])]:
            change = matrix @ displacement
            pairs.append((displacement, change))
            history = update_history(history, {'x': displacement}, {'x': change}, jnp.asarray(True))
            expected = initial.copy()
            for s, y in pairs[-capacity:]:
                rho = 1 / (s @ y)
                left = np.eye(3) - rho * np.outer(s, y)
                expected = left @ expected @ left.T + rho * np.outer(s, s)
            actual = inverse_product({'x': rhs}, history, lambda p: {'x': initial @ p['x']})
            np.testing.assert_allclose(actual['x'], expected @ rhs, atol=3e-12, rtol=0)


@pytest.mark.parametrize('complex_field', [False, True])
def test_large_quasi_newton_primal_reaches_independent_magnitude_solution(complex_field):
    dtype = jnp.complex64 if complex_field else jnp.float32
    target = jnp.full(64, 1 + 2j if complex_field else 1., dtype)

    def loss(point):
        residual = point['x'] - target
        return .5 * jnp.real(jnp.vdot(residual, residual))

    optimizer = make_optimizer('split_bregman', .1, regularizer_weight=.2)
    (result, state), status = _scan(optimizer, loss, {'x': jnp.zeros_like(target)}, 100)
    np.testing.assert_array_equal(status, 0)
    expected = target * (1 - .2 / jnp.abs(target))
    np.testing.assert_allclose(result['x'], expected, atol=3e-5, rtol=0)
    assert int(state['solver']['primal_method']) == 1
    assert int(state['solver']['history']['updates']) > 0


def test_primal_secants_exclude_the_changing_bregman_linear_term():
    diagonal = jnp.linspace(.5, 3., 64)
    target = jnp.ones(64)

    def loss(point):
        return .5 * jnp.sum(diagonal * (point['x'] - target) ** 2)

    optimizer = make_optimizer('split_bregman', 1., regularizer_weight=.2)
    (_, state), status = _scan(optimizer, loss, {'x': jnp.zeros(64)}, 5)
    np.testing.assert_array_equal(status, 0)
    history = state['solver']['history']
    assert int(history['count']) > 0
    # Identity splitting adds rho*I to the fixed data Hessian. Its linear
    # shift changes with d,b; stored secants retain the fixed primal curvature.
    for i in range(int(history['count'])):
        np.testing.assert_allclose(history['y']['0'][i],
            (diagonal + 1.) * history['s']['0'][i], atol=4e-7, rtol=3e-5)


def test_large_nonconvex_primal_keeps_descent_and_skips_negative_secants():
    def loss(point):
        return jnp.sum(.25 * point['x'] ** 4 - .5 * point['x'] ** 2)

    optimizer = make_optimizer('split_bregman', 1.)
    point = {'x': jnp.full(64, .1)}
    state = optimizer.init(point)
    update = jax.jit(lambda p, s: optimizer.update(None, s, p, value_fn=loss))
    previous = float(loss(point))
    for _ in range(30):
        changes, state = update(point, state)
        point = optax.apply_updates(point, changes)
        assert int(state['status']) == 0
        value = float(loss(point))
        assert value <= previous + 2e-5
        previous = value
    np.testing.assert_allclose(point['x'], 1., atol=3e-5, rtol=0)
    assert int(state['solver']['history']['skipped']) > 0


@pytest.mark.parametrize('size', [2, 64])
def test_finite_trial_with_invalid_derivative_preserves_last_valid_state(size):
    @jax.custom_jvp
    def pathological(value):
        return .5 * jnp.sum((value - .3) ** 2)

    @pathological.defjvp
    def derivative(primals, tangents):
        value, = primals
        tangent, = tangents
        gradient = jnp.where(jnp.any(value > .1), jnp.nan, value - .3)
        return pathological(value), jnp.sum(gradient * tangent)

    def loss(point):
        return pathological(point['x'])

    optimizer = make_optimizer('split_bregman', 1.)
    point = {'x': jnp.zeros(size)}
    state = optimizer.init(point)
    changes, failed = jax.jit(lambda p, s: optimizer.update(None, s, p, value_fn=loss))(point, state)
    assert int(failed['status']) == 3
    np.testing.assert_array_equal(changes['x'], 0)
    _same_tree(failed['solver'], state['solver'])


def test_large_complex_variational_tv_preserves_the_independent_solution():
    side = 8
    target = jnp.where(jnp.arange(side)[:, None] < side // 2, 1j, -1j)
    target = jnp.broadcast_to(target, (side, side))
    operator = build_operator(spatial_shape=target.shape)
    result, history = solve_variational(operator.forward(target), spatial_shape=target.shape,
        initial_image=jnp.zeros(target.shape), regularizer='tv', regularizer_weight=.1,
        optimizer_name='split_bregman', num_steps=400, return_history=True)
    np.testing.assert_allclose(result, target * (1 - .2 / side), atol=5e-4, rtol=0)
    tv = (jnp.sum(jnp.abs(jnp.diff(result, axis=0))) +
          jnp.sum(jnp.abs(jnp.diff(result, axis=1)))) / side ** 2
    expected = .5 * jnp.mean(jnp.abs(result - target) ** 2) + .1 * tv
    np.testing.assert_allclose(history[-1], expected, atol=2e-6, rtol=0)


def test_reconstruction_quadratic_has_the_correct_complex_hessian_at_zero():
    point = jnp.zeros((4, 3), jnp.complex64)

    def loss(value):
        return .5 * jnp.sum(_squared_magnitude(value))

    _, product = jax.linearize(lambda p: jnp.conj(jax.grad(loss)(p)), point)
    direction = jnp.full_like(point, .3 + .7j)
    np.testing.assert_allclose(product(direction), direction, atol=2e-7, rtol=0)


@pytest.mark.parametrize('stale', ['rate', 'history'])
def test_primal_restart_recovers_a_rounded_nonstationary_step(stale):
    operator = TVOperator.create((8, 8), weight=.1)

    def smooth(point):
        return .25 * jnp.sum(point['x'] ** 4)

    def loss(point):
        return smooth(point) + operator.value(point['x'])

    optimizer, step, _ = make_split_bregman_kernel(loss, smooth, {'x': (-np.inf, np.inf)},
        {'x': (operator,)}, .02)
    point = {'x': jnp.ones((8, 8), jnp.float32)}
    update = jax.jit(step)
    _, point, state, status = update(point, optimizer.init(point))
    assert int(status) == 0
    if stale == 'rate':
        state = {**state, 'rate': jnp.asarray(1e-30, jnp.float32)}
    else:
        # A valid positive-curvature secant of this SAME quartic can be a
        # poor local inverse after a large move. The injected state remains valid.
        far = jnp.full_like(point['x'], 4096.)
        secant = {'x': point['x'] - far}
        change = {'x': point['x'] ** 3 - far ** 3}
        history = update_history(initial_history(point, 8), secant, change, jnp.asarray(True))
        assert int(history['count']) == 1
        state = {**state, 'history': history}
    before = copy.copy(state)
    _, result, recovered, status = update(point, state)
    assert int(status) == 0
    assert float(loss(result)) < float(loss(point))
    # The large-system H0 is an SPD approximation. Check the original Armijo
    # inequality independently.
    slope = float(jnp.sum(point['x'] ** 3 * (result['x'] - point['x'])))
    assert slope < 0
    assert float(smooth(result)) <= float(smooth(point)) + 1e-4 * slope
    np.testing.assert_allclose(recovered['rate'], .02, atol=1e-9, rtol=0)
    assert int(recovered['preconditioner']['builds']) == int(before['preconditioner']['builds']) + 1
    assert int(recovered['history']['resets']) == int(before['history']['resets']) + 1
    assert int(recovered['count']) == int(before['count']) + 1
    np.testing.assert_array_equal(recovered['splits']['x'][0], operator.gradient(result['x']))
    np.testing.assert_array_equal(recovered['duals']['x'][0], 0)


def test_primal_restart_does_not_accept_an_unresolved_precision_limit():
    def loss(point):
        return .25 * jnp.sum(point['x'] ** 4)

    optimizer = make_optimizer('split_bregman', .02, proximal_max_step_size=1e-30)
    point = {'x': jnp.ones(64, jnp.float32)}
    state = optimizer.init(point)
    changes, failed = jax.jit(lambda p, s: optimizer.update(None, s, p, value_fn=loss))(point, state)
    assert int(failed['status']) == 4
    np.testing.assert_array_equal(changes['x'], 0)
    _same_tree(failed['solver'], state['solver'])
