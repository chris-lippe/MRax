"""Synthetic optimizer contracts and numerical references."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from mrax.core import FitError, build_loss, run_fit
from mrax.core.fit_execution import execute_optimizer
from mrax.core.loss_builder import (
    _mask_for_regularized_field,
    _reduce_mask_to_field,
)
from mrax.core.variance import _prepare_mask
from mrax.data_struct import DataSet, Experiment, Model, Quantity
from mrax.opt.optimizers import make_optimizer


def _constant_problem(target=0.3, initial=0.0, signal=None):
    experiment = Experiment(dims={"frame": 3, "x": 2, "y": 2})

    def factory(shape):
        return {"level": jnp.full(shape, initial)}, lambda params: params

    def apply(fields, _offsets, _observables=None):
        level = fields["level"] if signal is None else signal(fields["level"])
        return jnp.broadcast_to(level, experiment.shape)

    model = Model(
        name="synthetic_level",
        model_apply=apply,
        parametrization_factory=factory,
        parameter_units={"level": "1"},
    )
    return experiment, model, DataSet(jnp.full(experiment.shape, target))


def _problem(values, initial=0.0, **options):
    values = jnp.asarray(values)
    exp = Experiment(dims={"frame": values.shape[0], "x": values.shape[1], "y": values.shape[2]})
    exp.observables["offsets"] = Quantity(jnp.arange(values.shape[0]), "1")
    exp._mrax_dim_observables = {"frame": "offsets"}
    model = Model(
        name="synthetic_level",
        parameter_units={"level": "1"},
        parametrization_factory=lambda shape: (
            {"level": jnp.full(shape, initial)},
            lambda params: params,
        ),
        model_apply=lambda fields, offsets, observables=None: jnp.broadcast_to(
            fields["level"], exp.shape
        ),
        **options,
    )
    return exp, model, DataSet(values)


def test_optimizer_matches_optax_and_resumes_schedule():
    rate = optax.piecewise_constant_schedule(0.1, {400: 0.1, 800: 0.1})
    params = {"level": jnp.asarray([0.0, 4.0])}
    target = jnp.asarray([2.0, -1.0])

    def loss(p):
        return 0.5 * jnp.sum((p["level"] - target) ** 2)

    device = params["level"].device
    optimizer = make_optimizer("adam", rate, device=device)
    result, history, _ = execute_optimizer(loss, params, optimizer, num_steps=1200, device=device)
    native = optax.adam(rate)

    def update(carry, _):
        p, state = carry
        updates, state = native.update(jax.grad(loss)(p), state, p)
        return (optax.apply_updates(p, updates), state), None

    expected, _ = jax.jit(
        lambda: jax.lax.scan(update, (params, native.init(params)), None, length=1200)
    )()
    np.testing.assert_array_equal(result["level"], expected[0]["level"])
    np.testing.assert_allclose(result["level"], target, atol=0.02)
    _, first, checkpoint = execute_optimizer(loss, params, optimizer, num_steps=437, device=device)
    resumed, tail, _ = execute_optimizer(
        loss,
        checkpoint.state.params,
        optimizer,
        num_steps=763,
        initial_state=checkpoint.state,
        device=device,
    )
    np.testing.assert_array_equal(resumed["level"], result["level"])
    np.testing.assert_array_equal(first + tail, history)


def test_complex_gradient_recovers_complex_target():
    target = 1.0 + 2.0j

    def loss(p):
        return 0.5 * jnp.abs(p["level"] - target) ** 2

    params = {"level": jnp.asarray(0.0j, dtype=jnp.complex64)}
    result, _, _ = execute_optimizer(loss, params, make_optimizer("fista", 0.1), num_steps=200)
    np.testing.assert_allclose(result["level"], target, atol=0.003)


@pytest.mark.parametrize("name", ["fista", "split_bregman"])
def test_objective_aware_steps_are_invariant_to_loss_scaling(name):
    scale = 1e-14
    target = jnp.asarray([1e-4 + 2e-4j, -3e-4 + 1e-4j], dtype=jnp.complex64)

    def loss(params):
        return scale * 0.5 * jnp.sum(jnp.abs(params["level"] - target) ** 2)

    initial = {"level": jnp.zeros_like(target)}
    optimizer = make_optimizer(
        name,
        0.9 / scale,
        bregman_penalty=scale,
        proximal_context=(loss, loss, {"level": (-np.inf, np.inf)}, {"level": ()}),
    )
    result, _, info = execute_optimizer(loss, initial, optimizer, num_steps=40)
    np.testing.assert_allclose(result["level"], target, rtol=2e-5, atol=2e-9)
    assert float(info.final_loss) < float(loss(initial)) * 1e-8


def test_named_masks_preserve_pixels_in_fitting_and_variance_estimation():
    exp, model, data = _problem(jnp.arange(1, 13.0).reshape(2, 2, 3))
    mask = jnp.asarray([[1, 0, 0], [0, 1, 0]])
    canonical, params, unpack = build_loss(exp, model, data, mask=mask)
    reversed_loss, _, reversed_unpack = build_loss(
        exp, model, data, mask=mask.T, mask_dims=("y", "x")
    )
    np.testing.assert_array_equal(unpack._mrax_mask_fields, reversed_unpack._mrax_mask_fields)
    np.testing.assert_allclose(canonical(params), reversed_loss(params))
    np.testing.assert_array_equal(_prepare_mask(mask.T, ("y", "x"), exp)[0], mask > 0)
    with pytest.raises(ValueError, match="unique"):
        build_loss(exp, model, data, mask=jnp.ones((2, 2)), mask_dims=("x", "x"))


def test_partially_missing_frames_retain_output_and_shared_regularizer_support():
    exp, model, data = _problem([[[1.0], [2.0]], [[np.nan], [2.0]]])
    fields, _ = run_fit(
        exp, model, data, optimizer_name="sgd", num_steps=3, mask_fill_value=jnp.nan
    )
    assert np.isfinite(fields["level"].values).all()
    support = _mask_for_regularized_field(jnp.asarray([[1, 0, 0], [1, 1, 0]]), jnp.ones((1, 3)))
    np.testing.assert_array_equal(support, [[1, 1, 0]])
    exp = Experiment(dims={"x": 2, "frame": 3, "y": 4, "t": 5})
    mask = jnp.ones(exp.shape).at[1, :, 2, 3].set(0)
    reduced = _reduce_mask_to_field(mask, exp, ["t", "x", "y"])
    expected = np.ones((5, 2, 4), dtype=bool)
    expected[3, 1, 2] = False
    np.testing.assert_array_equal(reduced, expected)


def test_nonfinite_gradient_preserves_valid_checkpoint():
    exp, model, data = _constant_problem(initial=0.0, signal=jnp.sqrt)
    with pytest.raises(FitError, match="step 1") as captured:
        run_fit(exp, model, data, num_steps=4, chunk_size=2)
    checkpoint = captured.value.last_valid_state
    assert checkpoint.step == 0
    np.testing.assert_array_equal(checkpoint.params["level"], 0.0)
    assert captured.value.loss_history == []


def test_chunking_preserves_progress_and_callback_cancellation():
    exp, model, data = _constant_problem()
    steps = []
    run_fit(
        exp,
        model,
        data,
        num_steps=13,
        chunk_size=5,
        progress_callback=lambda step, total, message: steps.append(step),
    )
    assert steps == list(range(14))

    def cancel(step, total, message):
        if step == 2:
            raise RuntimeError("cancelled")

    with pytest.raises(RuntimeError, match="cancelled"):
        run_fit(exp, model, data, num_steps=13, progress_callback=cancel)


@pytest.mark.gpu
def test_cpu_cuda_objective_and_gradient_agree_at_identical_parameters():
    try:
        cpu, gpu = jax.devices("cpu")[0], jax.devices("gpu")[0]
    except RuntimeError:
        pytest.skip("CPU and CUDA backends are required")
    results = []
    for device in (cpu, gpu):
        with jax.default_device(device):
            exp, model, data = _constant_problem(initial=0.12)
            loss, params, _ = build_loss(exp, model, data, device=device)
            params = jax.device_put(params, device)
            value, grad = jax.jit(jax.value_and_grad(loss))(params)
            assert device in value.devices()
            results.append((float(value), np.asarray(grad["level"])))
            fields, _, info = run_fit(
                exp,
                model,
                data,
                optimizer=optax.sgd(0.3),
                num_steps=64,
                convergence_tolerance=1e-7,
                min_steps=5,
                return_fit_info=True,
                device=device,
            )
            assert device in fields["level"].values.devices()
            assert info.converged
            np.testing.assert_allclose(fields["level"].values, 0.3, atol=1e-4)
    np.testing.assert_allclose(results[0][0], results[1][0], rtol=1e-5)
    np.testing.assert_allclose(results[0][1], results[1][1], rtol=1e-5, atol=1e-7)
