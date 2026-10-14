import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from mrax.core import FitError, FitState, build_loss, run_fit
from mrax.data_struct import DataSet, Experiment, Model
from mrax.library.default_models import (
    _bounded,
    _inverse_transform,
    _resolve_bool_observable,
    make_AB_MT_n_model,
)
from mrax.opt.optimizers import make_optimizer


def _constant_problem(target=0.3, initial=0.0, signal=None):
    experiment = Experiment(dims={"frame": 3, "x": 2, "y": 2})

    def factory(shape):
        return {"level": jnp.full(shape, initial)}, lambda params: params

    def apply(fields, _offsets, _observables=None):
        level = fields["level"] if signal is None else signal(fields["level"])
        return jnp.broadcast_to(level, experiment.shape)

    model = Model(
        name="synthetic_level", model_apply=apply, parametrization_factory=factory,
        parameter_units={"level": "1"},
    )
    return experiment, model, DataSet(jnp.full(experiment.shape, target))


def test_chunked_fit_preserves_reference_updates_and_history():
    exp, model, data = _constant_problem()
    loss, params, _ = build_loss(exp, model, data)
    optimizer = optax.adam(0.03)
    state = optimizer.init(params)

    @jax.jit
    def step(params, state):
        value, grads = jax.value_and_grad(loss)(params)
        updates, state = optimizer.update(grads, state, params)
        return optax.apply_updates(params, updates), state, value

    expected_losses = []
    for _ in range(17):
        params, state, value = step(params, state)
        expected_losses.append(float(value))
    fields, history, info = run_fit(
        exp, model, data, num_steps=17, step_size=0.03, chunk_size=5, return_fit_info=True,
    )
    np.testing.assert_allclose(fields["level"].values, params["level"], rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(history, expected_losses, rtol=1e-5, atol=1e-7)
    assert info.iterations == 17 and info.state.step == 17
    assert info.termination_reason == "max_steps" and not info.converged
    assert info.final_loss == pytest.approx(float(loss(info.state.params)), rel=1e-6)


def test_scheduled_fit_resume_preserves_optimizer_moments_and_count():
    exp, model, data = _constant_problem()
    rate = optax.piecewise_constant_schedule(0.03, {5: 0.5})
    kwargs = {"step_size": rate, "return_fit_info": True, "chunk_size": 4}
    whole, whole_history, _ = run_fit(exp, model, data, num_steps=16, **kwargs)
    _, first_history, first = run_fit(exp, model, data, num_steps=7, **kwargs)
    resumed, last_history, last = run_fit(
        exp, model, data, num_steps=9, initial_state=first.state, **kwargs,
    )
    np.testing.assert_allclose(resumed["level"].values, whole["level"].values, atol=1e-7)
    np.testing.assert_allclose(first_history + last_history, whole_history, rtol=1e-5, atol=1e-7)
    assert last.state.step == 16 and last.iterations == 9


def test_fit_convergence_is_opt_in_and_respects_minimum_steps():
    exp, model, data = _constant_problem()
    fields, history, info = run_fit(
        exp, model, data, optimizer=optax.sgd(0.3), num_steps=100,
        convergence_tolerance=1e-7, parameter_tolerance=1e-5,
        convergence_patience=3, min_steps=5, return_fit_info=True,
    )
    assert 5 <= len(history) < 100
    assert info.converged and info.termination_reason == "converged"
    np.testing.assert_allclose(fields["level"].values, 0.3, atol=1e-4)
    assert info.final_loss < 1e-6


def test_nonfinite_initial_objective_is_reported():
    exp, model, data = _constant_problem(initial=-1.0, signal=jnp.log)
    with pytest.raises(FitError, match="Initial") as captured:
        run_fit(exp, model, data, num_steps=4)
    assert captured.value.last_valid_state is None
    assert captured.value.loss_history == []


def test_nonfinite_gradient_preserves_valid_checkpoint():
    exp, model, data = _constant_problem(initial=0.0, signal=jnp.sqrt)
    with pytest.raises(FitError, match="step 1") as captured:
        run_fit(exp, model, data, num_steps=4, chunk_size=2)
    checkpoint = captured.value.last_valid_state
    assert checkpoint.step == 0
    np.testing.assert_array_equal(checkpoint.params["level"], 0.0)
    assert captured.value.loss_history == []


def test_fit_zero_steps_returns_initialization_without_evaluating_model():
    exp, model, data = _constant_problem(initial=-1.0, signal=jnp.log)
    progress = []
    fields, history, callable_model, info = run_fit(
        exp, model, data, num_steps=0, return_model_callable=True, return_fit_info=True,
        progress_callback=lambda step, total, message: progress.append((step, total)),
    )
    assert history == [] and callable(callable_model)
    assert info.termination_reason == "no_steps" and info.final_loss is None
    np.testing.assert_array_equal(fields["level"].values, -1.0)
    assert progress == [(0, 1), (1, 1)]


def test_chunking_preserves_progress_and_callback_cancellation():
    exp, model, data = _constant_problem()
    steps = []
    run_fit(exp, model, data, num_steps=13, chunk_size=5,
            progress_callback=lambda step, total, message: steps.append(step))
    assert steps == list(range(14))

    def cancel(step, total, message):
        if step == 2:
            raise RuntimeError("cancelled")

    with pytest.raises(RuntimeError, match="cancelled"):
        run_fit(exp, model, data, num_steps=13, progress_callback=cancel)


def test_fit_rejects_incompatible_or_ambiguous_resume_state():
    exp, model, data = _constant_problem()
    _, _, info = run_fit(exp, model, data, num_steps=2, return_fit_info=True)
    with pytest.raises(ValueError, match="not both"):
        run_fit(exp, model, data, initial_state=info.state, initial_params=info.state.params)
    bad = FitState(info.state.params, {"wrong": jnp.zeros(())}, step=2)
    with pytest.raises(ValueError, match="state structure"):
        run_fit(exp, model, data, initial_state=bad)


def test_symmetric_bounded_initialization_preserves_small_values():
    model = make_AB_MT_n_model(param_overrides={"a": {"init": 1e-8}})
    params, unpack = model.parametrization_factory((2, 2))
    np.testing.assert_allclose(unpack(params)["a"], 1e-8, rtol=1e-5, atol=0.0)
    raw = jnp.asarray([-1e-8, 0.0, 1e-8], dtype=jnp.float32)
    transformed = jax.jit(lambda value: _bounded(value, -1.0, 1.0))(raw)
    assert float(transformed[0]) < 0 and float(transformed[2]) > 0
    gradient = jax.grad(lambda value: _bounded(value, -1.0, 1.0))(jnp.asarray(0.0))
    assert float(gradient) == pytest.approx(0.5)


def test_bounded_inverse_roundtrip_and_general_array_bounds():
    values = jnp.asarray([-0.8, -1e-8, 0.0, 1e-8, 0.8])
    raw = _inverse_transform(values, {"transform": "bounded", "bounds": (-1.0, 1.0)})
    np.testing.assert_allclose(_bounded(raw, -1.0, 1.0), values, rtol=1e-5, atol=0.0)
    lo, hi = jnp.asarray([1.0, 2.0]), jnp.asarray([3.0, 6.0])
    np.testing.assert_allclose(_bounded(jnp.zeros(2), lo, hi), [2.0, 4.0])
    clipped = _inverse_transform(jnp.asarray([-1.0, 1.0]),
                                 {"transform": "bounded", "bounds": (-1.0, 1.0)})
    assert np.all(np.isfinite(clipped))


@pytest.mark.parametrize("flag, expected", [(False, False), (True, True), ("<Off>", False),
                                            ("On", True), (0, False), (1, True)])
def test_static_flags_are_safe_in_jitted_closures(flag, expected):
    fn = jax.jit(lambda x: x if _resolve_bool_observable({"flag": flag}, "flag", False) else -x)
    assert float(fn(jnp.asarray(2.0))) == (2.0 if expected else -2.0)


def test_dynamic_or_ambiguous_flags_have_clear_errors():
    fn = jax.jit(lambda value: _resolve_bool_observable({"flag": value}, "flag", False))
    with pytest.raises(ValueError, match="static"):
        fn(jnp.asarray(True))
    with pytest.raises(ValueError, match="one static"):
        _resolve_bool_observable({"flag": np.asarray([True, False])}, "flag", False)


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.complex128])
def test_fista_acceleration_uses_parameter_real_precision(dtype):
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        optimizer = make_optimizer("fista", 0.1, regularizer_weight=0.2)
        state = optimizer.init({"p": jnp.ones(2, dtype=dtype)})
        assert state["t"].dtype == jnp.float64
    finally:
        jax.config.update("jax_enable_x64", previous)


@pytest.mark.parametrize("name", ["split_bregman", "fista"])
def test_proximal_schedule_tracks_gradient_step_count(name):
    def rate(count):
        return jnp.where(count == 0, 0.1, 0.05)

    optimizer = make_optimizer(name, rate, regularizer_weight=0.2,
                               base_optimizer_factory=optax.sgd)
    params = {"p": jnp.asarray(1.0)}
    state = optimizer.init(params)
    for _ in range(2):
        updates, state = optimizer.update({"p": jnp.zeros(())}, state, params)
        params = optax.apply_updates(params, updates)
    assert int(state["count"]) == 2
    if name == "split_bregman":
        assert float(params["p"]) == pytest.approx(0.99)
    else:
        assert float(state["x"]["p"]) == pytest.approx(0.97)


def test_zero_weight_bregman_fast_path_retains_custom_prox_and_bounds():
    plain = make_optimizer("split_bregman", 0.1)
    params = {"p": jnp.asarray(1.0)}
    assert jax.tree_util.tree_structure(plain.init(params)) == jax.tree_util.tree_structure(
        optax.adam(0.1).init(params))
    custom = make_optimizer("split_bregman", 0.1, regularizer_prox=lambda value, weight: value + 1,
                            base_optimizer_factory=optax.sgd)
    updates, _ = custom.update({"p": jnp.zeros(())}, custom.init(params), params)
    assert float(optax.apply_updates(params, updates)["p"]) == pytest.approx(2.0)
    bounded = make_optimizer("split_bregman", 0.1, box_constraints=(0.0, 0.5))
    updates, _ = bounded.update({"p": jnp.zeros(())}, bounded.init(params), params)
    assert float(optax.apply_updates(params, updates)["p"]) == pytest.approx(0.5)


def test_cpu_cuda_objective_and_gradient_agree_at_identical_parameters():
    try:
        cpu, gpu = jax.devices("cpu")[0], jax.devices("gpu")[0]
    except RuntimeError:
        pytest.skip("CPU and CUDA backends are required")
    results = []
    for device in (cpu, gpu):
        with jax.default_device(device):
            exp, model, data = _constant_problem(initial=0.12)
            loss, params, _ = build_loss(exp, model, data)
            params = jax.device_put(params, device)
            value, grad = jax.jit(jax.value_and_grad(loss))(params)
            assert device in value.devices()
            results.append((float(value), np.asarray(grad["level"])))
            fields, _, info = run_fit(
                exp, model, data, optimizer=optax.sgd(0.3), num_steps=64,
                convergence_tolerance=1e-7, min_steps=5, return_fit_info=True,
            )
            assert device in fields["level"].values.devices()
            assert info.converged
            np.testing.assert_allclose(fields["level"].values, 0.3, atol=1e-4)
    np.testing.assert_allclose(results[0][0], results[1][0], rtol=1e-5)
    np.testing.assert_allclose(results[0][1], results[1][1], rtol=1e-5, atol=1e-7)
