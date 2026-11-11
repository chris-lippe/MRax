"""Synthetic device contracts and numerical references."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mrax import get_available_devices
from mrax.core import (
    run_fit,
)
from mrax.data_struct import DataSet, Experiment, Model, Quantity
from mrax.devices import resolve_device
from mrax.preprocessing import regularized_denoise
from mrax.reco.variational import build_operator, solve_variational


def _device(backend):
    devices = get_available_devices(backend)
    if not devices:
        pytest.skip(f"{backend} is unavailable")
    return devices[0]


def _problem():
    exp = Experiment(dims={"frame": 1, "x": 2, "y": 1})
    exp.observables["gain"] = Quantity(jnp.asarray(1.0), "1")
    model = Model(
        name="synthetic_level",
        parameter_units={"level": "1"},
        parameter_specs={"level": {"transform": "identity"}},
        parametrization_factory=lambda shape: ({"level": jnp.zeros(shape)}, lambda p: p),
        model_apply=lambda fields, offsets, observables=None: (
            fields["level"][None] * getattr(observables["gain"], "values", observables["gain"])
        ),
    )
    return exp, model, DataSet(jnp.ones(exp.shape))


def _assert_device(tree, device):
    arrays = [x for x in jax.tree.leaves(tree) if isinstance(x, jax.Array)]
    assert arrays and all(x.devices() == {device} for x in arrays)


def test_device_discovery_returns_local_devices_cpu_first():
    found = get_available_devices()
    assert found and found[0].platform == "cpu"
    assert len(found) == len(set(found))
    assert get_available_devices("cpu") == jax.local_devices(backend="cpu")
    assert get_available_devices("cuda") == get_available_devices("gpu")
    for device in found:
        assert resolve_device(device) == device


@pytest.mark.parametrize("selector", ["cpu:-1", "unknown"])
def test_invalid_or_unavailable_device_is_rejected_before_fitting(selector):
    exp, model, data = _problem()
    with pytest.raises(ValueError):
        run_fit(exp, model, data, device=selector, num_steps=0)


@pytest.mark.parametrize("backend", ["cpu"])
def test_reconstruction_and_denoising_use_selected_device(backend):
    device = _device(backend)
    target = jnp.ones((2, 2))
    op = build_operator(spatial_shape=(2, 2), kspace=jnp.zeros((2, 2)))
    image = solve_variational(
        op.forward(target),
        spatial_shape=(2, 2),
        regularizer="l1",
        regularizer_weight=0.1,
        num_steps=60,
        device=device,
    )
    _assert_device(image, device)
    np.testing.assert_allclose(image, 0.9, atol=0.001)
    denoised = regularized_denoise(target, num_steps=4, device=device)
    _assert_device(denoised, device)
