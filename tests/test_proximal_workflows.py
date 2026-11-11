"""Synthetic proximal contracts and numerical references."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mrax.core import run_fit
from mrax.data_struct import DataSet, Experiment, Model
from mrax.opt.tv import TVOperator, tv_prox


def _device(backend):
    try:
        return jax.devices(backend)[0]
    except RuntimeError:
        pytest.skip(f"{backend} is unavailable")


def _image_problem(tv=True):
    experiment = Experiment(dims={"frame": 1, "x": 2, "y": 1})

    def factory(shape):
        return {"level_raw": jnp.zeros(shape)}, lambda p: {
            "level": 2 * jnp.tanh(p["level_raw"] / 2)
        }

    model = Model(
        name="synthetic_image",
        parametrization_factory=factory,
        model_apply=lambda fields, offsets, observables=None: fields["level"][None],
        parameter_units={"level": "1"},
        parameter_specs={"level": {"transform": "bounded", "bounds": (-2.0, 2.0)}},
    )
    data = DataSet(jnp.asarray([[[-1.0], [1.0]]]))
    regularizers = {"level": [{"type": "tv", "axes": (0,), "weight": 0.5}]} if tv else {}
    return experiment, model, data, regularizers


@pytest.mark.parametrize("order", [1, 2])
def test_masked_tv_operator_adjoint_and_disconnected_regions(order):
    field = (jnp.arange(20, dtype=jnp.float32).reshape(4, 5) / 3) ** 2
    mask = jnp.asarray([[1, 1, 1, 0, 1], [1, 1, 1, 0, 1], [1, 1, 1, 0, 1], [0, 0, 0, 0, 0]])
    operator = TVOperator.create(field.shape, order=order, mask=mask)
    dual = (
        jnp.arange(operator.gradient(field).size, dtype=field.dtype).reshape(
            operator.gradient(field).shape
        )
        / 10
    )
    np.testing.assert_allclose(
        jnp.sum(operator.gradient(field) * dual), jnp.sum(field * operator.adjoint(dual)), atol=1e-5
    )
    assert float(jnp.linalg.norm(operator.gradient(field))) > 0
    changed = field.at[:, 3].set(1000.0)
    np.testing.assert_array_equal(operator.value(field), operator.value(changed))


@pytest.mark.parametrize("backend", ["cpu"])
def test_tv_prox_matches_two_pixel_solution_and_certifies_gap(backend):
    device = _device(backend)
    value = jax.device_put(jnp.asarray([-1.0, 1.0]), device)
    operator = TVOperator.create(value.shape, weight=0.5)
    solve = jax.jit(lambda value: tv_prox(value, (operator,), 1.0, (-2.0, 2.0), tolerance=1e-8))
    result, _, gap, _ = solve(value)
    np.testing.assert_allclose(result, [-0.75, 0.75], rtol=0, atol=2e-4)
    assert float(gap) <= 1e-8 * 1.5


@pytest.mark.parametrize("backend", ["cpu"])
@pytest.mark.parametrize("name", ["fista", "split_bregman"])
def test_true_tv_fitting_in_physical_coordinates(backend, name):
    device = _device(backend)
    with jax.default_device(device):
        exp, model, data, regularizers = _image_problem()
        fields, history, info = run_fit(
            exp,
            model,
            data,
            optimizer_name=name,
            num_steps=200,
            regularizer_configs=regularizers,
            return_fit_info=True,
            device=device,
        )
    # The automatic finite-data mask normalizes over the one valid edge.
    np.testing.assert_allclose(fields["level"].values[:, 0], [-0.5, 0.5], atol=5e-4, rtol=0)
    assert info.optimizer_name == (name or "fista")
    assert info.regularization_mode == "proximal_tv"
    assert info.parameter_space == "physical"
    assert fields["level"].values.device == device
    assert np.all(np.isfinite(history))
    np.testing.assert_allclose(info.final_loss, 0.75, atol=2e-6)
