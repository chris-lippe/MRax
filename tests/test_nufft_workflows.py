"""Synthetic nufft contracts and numerical references."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mrax.reco.nufft import prepare_nufft
from mrax.reco.variational import solve_variational


def _matrix(shape, coordinates):
    positions = np.stack(
        np.meshgrid(*[np.arange(n) - n // 2 for n in shape], indexing="ij"), axis=-1
    )
    return np.exp(
        -2j * np.pi * (coordinates.astype("float64") @ positions.reshape(-1, len(shape)).T)
    ) / np.sqrt(np.prod(shape))


def _complex(rng, shape):
    return (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype("complex64")


@pytest.fixture(params=["cpu"])
def device(request):
    try:
        selected = jax.devices(request.param)[0]
    except RuntimeError:
        pytest.skip(f"{request.param} is unavailable")
    with jax.default_device(selected):
        yield selected


@pytest.mark.parametrize("shape", [(7, 8)])
def test_nufft_matches_direct_dft_and_true_adjoint(shape, device):
    rng = np.random.default_rng(18)
    trajectory = rng.uniform(-0.5, 0.5, (43, len(shape))).astype("float32")
    trajectory[:3] = np.asarray([-0.5, 0.0, 0.5])[:, None]
    trajectory[-1] = trajectory[4]
    matrix = _matrix(shape, trajectory)
    image = _complex(rng, (2, 3) + shape)
    samples = _complex(rng, (2, 3, len(trajectory)))
    plan = prepare_nufft(shape, trajectory, chunk_size=11, batch_size=4)
    encoded = jax.jit(plan.forward)(image)
    decoded = jax.jit(plan.adjoint)(samples)
    expected = image.reshape(2, 3, -1) @ matrix.T
    expected_adjoint = (samples @ matrix.conj()).reshape(image.shape)
    np.testing.assert_allclose(encoded, expected, rtol=1e-5, atol=8e-6)
    np.testing.assert_allclose(decoded, expected_adjoint, rtol=1e-5, atol=8e-6)
    np.testing.assert_allclose(
        jnp.vdot(encoded, samples), jnp.vdot(image, decoded), rtol=5e-6, atol=1e-5
    )
    assert encoded.device == device


def test_nufft_gradients_match_dft_and_directional_derivative(device):
    shape = (5, 6)
    rng = np.random.default_rng(7)
    trajectory = rng.uniform(-0.5, 0.5, (31, 2)).astype("float32")
    plan = prepare_nufft(shape, trajectory, chunk_size=9)
    image = jnp.asarray(_complex(rng, shape))
    target = jnp.asarray(_complex(rng, (31,)))
    matrix = _matrix(shape, trajectory)

    def loss(value):
        return jnp.sum(jnp.abs(plan.forward(value) - target) ** 2) / 2

    gradient = jax.jit(jax.grad(loss))(image)
    expected = np.conj(matrix.conj().T @ (matrix @ np.asarray(image).ravel() - target))
    np.testing.assert_allclose(gradient, expected.reshape(shape), rtol=2e-5, atol=8e-6)
    direction = jnp.asarray(_complex(rng, shape))
    _, tangent = jax.jvp(plan.forward, (image,), (direction,))
    np.testing.assert_allclose(tangent, plan.forward(direction), atol=4e-6)
    h = 0.002
    numerical = (loss(image + h * direction) - loss(image - h * direction)) / (2 * h)
    np.testing.assert_allclose(numerical, jnp.real(jnp.sum(gradient * direction)), rtol=0.002)


def test_iterative_noncartesian_reconstruction_recovers_independent_data(device):
    shape = (6, 7)
    rng = np.random.default_rng(8)
    trajectory = rng.uniform(-0.5, 0.5, (180, 2)).astype("float32")
    image = np.zeros(shape, "complex64")
    image[1:5, 2:5] = 1.0 + 0.2j
    samples = (_matrix(shape, trajectory) @ image.ravel()).astype("complex64")
    result, history = solve_variational(
        samples,
        spatial_shape=shape,
        trajectory=trajectory,
        regularizer="none",
        optimizer_name="fista",
        num_steps=160,
        initial_image=np.zeros(shape, "complex64"),
        return_history=True,
        device=device,
    )
    assert history[-1] < history[0] * 1e-6
    np.testing.assert_allclose(result, image, atol=2e-4)
