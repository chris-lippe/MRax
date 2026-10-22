import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from mrax.core import build_loss, run_fit
from mrax.core.fit_execution import execute_optimizer
from mrax.core.fit_sanity import debug_fit_prototype
from mrax.core.tv_fit import make_tv_steps
from mrax.data_struct import DataSet, Experiment, Model
from mrax.opt.optimizers import make_optimizer
from mrax.opt.tv import TVOperator, tv_prox


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
@pytest.mark.parametrize("name", ["fista", "split_bregman"])
@pytest.mark.parametrize("use_tv", [False, True])
def test_proximal_fitting_preserves_mixed_parameter_dtypes(backend, name, use_tv):
    enable = getattr(jax, "enable_x64", None)
    if enable is None:
        from jax.experimental import enable_x64

        enable = enable_x64
    device = _device(backend)
    with enable(), jax.default_device(device):
        params = {"single": jnp.zeros((3, 4), jnp.float32),
                  "double": jnp.zeros((3, 4), jnp.float64)}

        def loss(fields):
            return (.5 * jnp.sum((fields["single"] - 1.)**2)
                    + .5 * jnp.sum((fields["double"] - 2.)**2))

        operators = {key: (TVOperator.create(value.shape, weight=.001, dtype=value.dtype),)
                     if use_tv else () for key, value in params.items()}
        optimizer, step, convergence = make_tv_steps(loss, loss,
            {key: (np.float64(-10), np.float64(10)) for key in params},
            operators, name, .1, device=device)
        fitted, _, _ = execute_optimizer(loss, params, optimizer, num_steps=40,
            step_function=step, convergence_check=convergence, device=device)
        for key, target in [("single", 1.), ("double", 2.)]:
            assert fitted[key].dtype == params[key].dtype
            np.testing.assert_allclose(fitted[key], target, atol=3e-4)


def _device(backend):
    try:
        return jax.devices(backend)[0]
    except RuntimeError:
        pytest.skip(f"{backend} is unavailable")


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
@pytest.mark.parametrize("name", ["fista", "split_bregman"])
def test_proximal_optimizers_recover_penalized_scalar_optimum(backend, name):
    device = _device(backend)
    optimizer = make_optimizer(name, .1, regularizer_weight=.5, device=device)
    params = {"x": jax.device_put(jnp.asarray(0.), device)}

    @jax.jit
    def solve(params):
        def update(carry, _):
            params, state = carry
            grads = {"x": params["x"] - 2}
            updates, state = optimizer.update(grads, state, params,
                value_fn=lambda p: .5 * (p['x'] - 2)**2)
            return (optax.apply_updates(params, updates), state), None
        return jax.lax.scan(update, (params, optimizer.init(params)), None, length=1000)[0][0]

    result = solve(params)
    assert result["x"].device == device
    np.testing.assert_allclose(result["x"], 1.5, atol=2e-5, rtol=0)


@pytest.mark.parametrize("name", ["fista", "split_bregman"])
def test_adam_bases_are_rejected_and_zero_weight_keeps_named_solver(name):
    with pytest.raises(ValueError, match="Adam base"):
        make_optimizer(name, .1, base_optimizer_factory=optax.adam)
    optimizer = make_optimizer(name, .1)
    state = optimizer.init({"x": jnp.asarray(0.)})
    assert "y" in state['solver'] if name == "fista" else "splits" in state['solver']


@pytest.mark.parametrize("weight", [-.1, np.inf, np.nan])
@pytest.mark.parametrize("name", ["fista", "split_bregman"])
def test_invalid_proximal_weights_are_rejected(name, weight):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        make_optimizer(name, .1, regularizer_weight=weight)


@pytest.mark.parametrize("order", [1, 2])
def test_masked_tv_operator_adjoint_and_disconnected_regions(order):
    field = (jnp.arange(20, dtype=jnp.float32).reshape(4, 5) / 3)**2
    mask = jnp.asarray([[1, 1, 1, 0, 1], [1, 1, 1, 0, 1], [1, 1, 1, 0, 1], [0, 0, 0, 0, 0]])
    operator = TVOperator.create(field.shape, order=order, mask=mask)
    dual = jnp.arange(operator.gradient(field).size, dtype=field.dtype).reshape(operator.gradient(field).shape) / 10
    np.testing.assert_allclose(jnp.sum(operator.gradient(field) * dual),
                               jnp.sum(field * operator.adjoint(dual)), atol=1e-5)
    assert float(jnp.linalg.norm(operator.gradient(field))) > 0
    changed = field.at[:, 3].set(1000.)
    np.testing.assert_array_equal(operator.value(field), operator.value(changed))


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_tv_prox_matches_two_pixel_solution_and_certifies_gap(backend):
    device = _device(backend)
    value = jax.device_put(jnp.asarray([-1., 1.]), device)
    operator = TVOperator.create(value.shape, weight=.5)
    solve = jax.jit(lambda value: tv_prox(value, (operator,), 1., (-2., 2.), tolerance=1e-8))
    result, _, gap, _ = solve(value)
    np.testing.assert_allclose(result, [-.75, .75], rtol=0, atol=2e-4)
    assert float(gap) <= 1e-8 * 1.5


def _image_problem(tv=True):
    experiment = Experiment(dims={"frame": 1, "x": 2, "y": 1})

    def factory(shape):
        return {"level_raw": jnp.zeros(shape)}, lambda p: {"level": 2 * jnp.tanh(p["level_raw"] / 2)}

    model = Model(name="synthetic_image", parametrization_factory=factory,
                  model_apply=lambda fields, offsets, observables=None: fields["level"][None],
                  parameter_units={"level": "1"},
                  parameter_specs={"level": {"transform": "bounded", "bounds": (-2., 2.)}})
    data = DataSet(jnp.asarray([[[-1.], [1.]]]))
    regularizers = {"level": [{"type": "tv", "axes": (0,), "weight": .5}]} if tv else {}
    return experiment, model, data, regularizers


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
@pytest.mark.parametrize("name", [None, "fista", "split_bregman"])
def test_true_tv_fitting_in_physical_coordinates(backend, name):
    device = _device(backend)
    with jax.default_device(device):
        exp, model, data, regularizers = _image_problem()
        fields, history, info = run_fit(exp, model, data, optimizer_name=name, num_steps=200,
                                        regularizer_configs=regularizers, return_fit_info=True, device=device)
    # The automatic finite-data mask normalizes over the one valid edge.
    np.testing.assert_allclose(fields["level"].values[:, 0], [-.5, .5], atol=5e-4, rtol=0)
    assert info.optimizer_name == (name or "fista")
    assert info.regularization_mode == "proximal_tv"
    assert info.parameter_space == "physical"
    assert fields["level"].values.device == device
    assert np.all(np.isfinite(history))
    np.testing.assert_allclose(info.final_loss, .75, atol=2e-6)


def test_default_without_tv_is_adam_and_explicit_adam_warns_with_tv():
    exp, model, data, regularizers = _image_problem(False)
    _, _, info = run_fit(exp, model, data, regularizer_configs=regularizers, num_steps=2, return_fit_info=True)
    assert info.optimizer_name == "adam"
    assert info.regularization_mode == "none"
    exp, model, data, regularizers = _image_problem()
    with pytest.warns(RuntimeWarning, match="smoothed TV"):
        _, _, info = run_fit(exp, model, data, optimizer_name="adam", regularizer_configs=regularizers,
                             num_steps=2, return_fit_info=True)
    assert info.optimizer_name == "adam"
    assert info.regularization_mode == "smoothed_tv"
    assert info.parameter_space == "raw"


def test_proximal_resume_preserves_primal_dual_state():
    exp, model, data, regularizers = _image_problem()
    kwargs = dict(optimizer_name="split_bregman", regularizer_configs=regularizers, return_fit_info=True)
    whole, _, _ = run_fit(exp, model, data, num_steps=40, **kwargs)
    _, _, info = run_fit(exp, model, data, num_steps=20, **kwargs)
    resumed, _, info = run_fit(exp, model, data, num_steps=20, initial_state=info.state, **kwargs)
    np.testing.assert_allclose(resumed["level"].values, whole["level"].values, atol=1e-6)
    assert info.state.step == 40


def test_flat_exact_tv_has_finite_selected_subgradient():
    operator = TVOperator.create((3, 3))
    value, gradient = jax.value_and_grad(operator.value)(jnp.ones((3, 3)))
    assert value == 0
    np.testing.assert_array_equal(gradient, jnp.zeros((3, 3)))


def test_exact_tv_physical_loss_matches_field_regularization():
    exp, model, data, regularizers = _image_problem()
    loss, params, unpack = build_loss(exp, model, data, regularizer_configs=regularizers, parameter_space="physical")
    params = {"level": jnp.asarray([[-.75], [.75]]) / unpack._mrax_physical_scales["level"]}
    np.testing.assert_allclose(loss(params), .8125, atol=1e-6)


def test_zero_step_tv_and_resume_keep_explicit_coordinate_scales():
    exp, model, data, regularizers = _image_problem()
    def invalid_apply(*args):
        raise AssertionError("zero steps evaluated the model")
    original = model.model_apply
    model.model_apply = invalid_apply
    fields, history, info = run_fit(exp, model, data, num_steps=0, regularizer_configs=regularizers, return_fit_info=True)
    assert history == [] and info.state.parameter_scales
    model.model_apply = original
    _, _, resumed = run_fit(exp, model, data, num_steps=2, initial_state=info.state,
                            regularizer_configs=regularizers, return_fit_info=True)
    assert resumed.state.parameter_scales == info.state.parameter_scales
    assert resumed.state.step == 2


def test_debug_prototype_uses_actual_tv_solver_and_adam_policy():
    exp, model, data, regularizers = _image_problem()
    report = debug_fit_prototype(exp, model, data, regularizer_configs=regularizers)
    assert report.prototype_ran, report.prototype_error
    assert report.optimizer_name == "fista"
    with pytest.warns(RuntimeWarning, match="smoothed TV"):
        report = debug_fit_prototype(exp, model, data, optimizer_name="adam", regularizer_configs=regularizers)
    assert report.prototype_ran, report.prototype_error
    assert report.optimizer_name == "adam"


@pytest.mark.parametrize("name", ["fista", "split_bregman"])
def test_tv_fit_with_all_fields_fixed_keeps_the_fixed_maps(name):
    exp, model, data, regularizers = _image_problem()
    model.parameter_specs["level"]["trainable"] = False
    fields, history, info = run_fit(exp, model, data, optimizer_name=name, num_steps=2,
                                    regularizer_configs=regularizers, return_fit_info=True)
    np.testing.assert_array_equal(fields["level"].values, jnp.zeros((2, 1)))
    assert len(history) == 2 and info.state.params == {}
