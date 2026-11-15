"""Synthetic proximal contracts and numerical references."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mrax.core import run_fit
from mrax.data_struct import DataSet, Experiment, Model
from mrax.opt.tv import TVOperator


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


def test_masked_tv_operator_adjoint_and_disconnected_regions():
    field = (jnp.arange(20, dtype=jnp.float32).reshape(4, 5) / 3) ** 2
    mask = jnp.asarray([[1, 1, 1, 0, 1], [1, 1, 1, 0, 1], [1, 1, 1, 0, 1], [0, 0, 0, 0, 0]])
    operator = TVOperator.create(field.shape, order=2, mask=mask)
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


@pytest.mark.parametrize("name", ["fista", "split_bregman"])
def test_true_tv_fitting_in_physical_coordinates(name):
    device = jax.devices("cpu")[0]
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
