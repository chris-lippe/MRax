"""Independent synthetic checks for the portable nonuniform Fourier operator."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mrax.core import run_fit
from mrax.data_struct import DataSet, Experiment, Model, Quantity
from mrax.reco.nufft import prepare_nufft
from mrax.reco.variational import build_operator, fftnc, solve_model_based, solve_variational


def _matrix(shape, coordinates):
    positions = np.stack(np.meshgrid(
        *[np.arange(n) - n // 2 for n in shape], indexing="ij"), axis=-1)
    return np.exp(-2j * np.pi * (coordinates.astype("float64")
                  @ positions.reshape(-1, len(shape)).T)) / np.sqrt(np.prod(shape))


def _complex(rng, shape):
    return (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype("complex64")


def _enable_x64(enabled=True):
    enable = getattr(jax, "enable_x64", None)
    if enable is None:
        from jax.experimental import enable_x64

        enable = enable_x64
    return enable(enabled)


@pytest.fixture(params=["cpu", "gpu"])
def device(request):
    try:
        selected = jax.devices(request.param)[0]
    except RuntimeError:
        pytest.skip(f"{request.param} is unavailable")
    with jax.default_device(selected):
        yield selected


@pytest.mark.parametrize("shape", [(9,), (7, 8), (8, 7), (3, 4, 5), (2, 2)])
def test_nufft_matches_direct_dft_and_true_adjoint(shape, device):
    rng = np.random.default_rng(18)
    trajectory = rng.uniform(-.5, .5, (43, len(shape))).astype("float32")
    trajectory[:3] = np.asarray([-.5, 0., .5])[:, None]
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
    np.testing.assert_allclose(jnp.vdot(encoded, samples), jnp.vdot(image, decoded),
                               rtol=5e-6, atol=1e-5)
    assert encoded.device == device


def test_cartesian_locations_have_centered_unitary_fft_scaling(device):
    shape = (7, 8)
    trajectory = np.stack(np.meshgrid(
        *[np.fft.fftshift(np.fft.fftfreq(n)) for n in shape], indexing="ij"), axis=-1)
    rng = np.random.default_rng(71)
    image = _complex(rng, shape)
    plan = prepare_nufft(shape, trajectory.reshape(-1, 2).astype("float32"))
    np.testing.assert_allclose(plan.forward(image).reshape(shape), fftnc(image), atol=5e-6)
    np.testing.assert_allclose(plan.adjoint(plan.forward(image)), image, atol=8e-6)


def test_nufft_gradients_match_dft_and_directional_derivative(device):
    shape = (5, 6)
    rng = np.random.default_rng(7)
    trajectory = rng.uniform(-.5, .5, (31, 2)).astype("float32")
    plan = prepare_nufft(shape, trajectory, chunk_size=9)
    image = jnp.asarray(_complex(rng, shape))
    target = jnp.asarray(_complex(rng, (31,)))
    matrix = _matrix(shape, trajectory)
    def loss(value):
        return jnp.sum(jnp.abs(plan.forward(value) - target)**2) / 2
    gradient = jax.jit(jax.grad(loss))(image)
    expected = np.conj(matrix.conj().T @ (matrix @ np.asarray(image).ravel() - target))
    np.testing.assert_allclose(gradient, expected.reshape(shape), rtol=2e-5, atol=8e-6)
    direction = jnp.asarray(_complex(rng, shape))
    _, tangent = jax.jvp(plan.forward, (image,), (direction,))
    np.testing.assert_allclose(tangent, plan.forward(direction), atol=4e-6)
    h = .002
    numerical = (loss(image + h * direction) - loss(image - h * direction)) / (2 * h)
    np.testing.assert_allclose(numerical, jnp.real(jnp.sum(gradient * direction)), rtol=.002)


def test_coil_encoding_and_density_are_explicit(device):
    rng = np.random.default_rng(6)
    shape = (5, 6)
    trajectory = rng.uniform(-.5, .5, (47, 2)).astype("float32")
    sensitivities = _complex(rng, (2,) + shape)
    weights = np.linspace(.2, 2., 47).astype("float32")
    operator = build_operator(spatial_shape=shape, trajectory=trajectory,
        sensitivities=sensitivities, density=weights)
    image = _complex(rng, (3,) + shape)
    samples = _complex(rng, (3, 2, 47))
    matrix = _matrix(shape, trajectory)
    expected = (image[:, None] * sensitivities).reshape(3, 2, -1) @ matrix.T
    np.testing.assert_allclose(operator.forward(image), expected, rtol=2e-5, atol=1e-5)
    np.testing.assert_allclose(jnp.vdot(operator.forward(image), samples),
        jnp.vdot(image, operator.adjoint(samples)), rtol=5e-6, atol=2e-5)
    np.testing.assert_allclose(operator.adjoint(samples, density_compensation=True),
        operator.adjoint(samples * weights), rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("eps", [1e-10, 1e-12])
def test_nufft_double_precision_and_accuracy_selection(eps):
    with _enable_x64(), jax.default_device(jax.devices("cpu")[0]):
        rng = np.random.default_rng(3)
        trajectory = rng.uniform(-.5, .5, (61, 2))
        image = rng.normal(size=(9, 10)) + 1j * rng.normal(size=(9, 10))
        expected = _matrix(image.shape, trajectory) @ image.ravel()
        loose = prepare_nufft(image.shape, trajectory, eps=1e-3)
        tight = prepare_nufft(image.shape, trajectory, eps=eps)
        streamed = prepare_nufft(image.shape, trajectory, eps=eps, cache_budget_bytes=0)
        loose_error = np.linalg.norm(loose.forward(image) - expected)
        tight_error = np.linalg.norm(tight.forward(image) - expected)
        assert tight.width > loose.width
        assert tight_error < loose_error / 1000
        assert tight_error / np.linalg.norm(expected) < max(3 * eps, 3e-12)
        np.testing.assert_allclose(tight.forward(image), expected, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(streamed.forward(image), expected, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("x64", [False, True])
def test_long_coherent_adjoint_preserves_single_precision_accuracy(device, x64):
    count = 131073
    plan = prepare_nufft((8, 8), np.zeros((count, 2), "float32"), cache_budget_bytes=0)
    samples = np.full(count, 1. + .3j, "complex64")
    expected = samples.astype("complex128").sum() / 8
    with _enable_x64(x64):
        result = jax.jit(plan.adjoint)(samples)
    relative_error = np.linalg.norm(np.asarray(result) - expected) / (abs(expected) * 8)
    assert result.dtype == jnp.complex64
    assert relative_error < (4e-6 if x64 and device.platform == "cpu" else 2e-4)


@pytest.mark.parametrize("shape", [(7, 8), (3, 4, 5)])
def test_compact_and_cached_stencils_agree_with_direct_dft(shape, device):
    rng = np.random.default_rng(73)
    coordinates = rng.uniform(-.55, .55, (43, len(shape))).astype("float32")
    image = _complex(rng, (2, 3) + shape)
    samples = _complex(rng, (2, 3, 43))
    cached = prepare_nufft(shape, coordinates, chunk_size=11, cache_budget_bytes=2**20)
    compact = prepare_nufft(shape, coordinates, chunk_size=13, cache_budget_bytes=512)
    assert cached.cached_stencil and not compact.cached_stencil
    assert compact.indices.nbytes + compact.weights.nbytes < cached.indices.nbytes + cached.weights.nbytes
    matrix = _matrix(shape, coordinates)
    np.testing.assert_allclose(jax.jit(compact.forward)(image),
        image.reshape(2, 3, -1) @ matrix.T, rtol=2e-5, atol=1e-5)
    np.testing.assert_allclose(compact.adjoint(samples), cached.adjoint(samples),
        rtol=2e-5, atol=1e-5)
    with pytest.raises(ValueError, match="cache budget"):
        prepare_nufft(shape, coordinates, cache_budget_bytes=-1)


@pytest.mark.parametrize("cache_bytes", [0, 2**20])
def test_nufft_preserves_single_precision_with_x64_enabled(cache_bytes, device):
    with _enable_x64():
        points = np.random.default_rng(43).uniform(-.5, .5, (19, 2)).astype("float32")
        plan = prepare_nufft((4, 5), points, cache_budget_bytes=cache_bytes)
        image = jnp.ones((4, 5), dtype=jnp.complex64)
        encoded = jax.jit(plan.forward)(image)
        assert encoded.dtype == jnp.complex64
        assert plan.adjoint(encoded).dtype == jnp.complex64
        gradient = jax.grad(lambda x: jnp.sum(abs(plan.forward(x))**2))(image)
        assert gradient.dtype == jnp.complex64


def test_nufft_vmap_and_hessian_use_the_linear_operator(device):
    rng = np.random.default_rng(23)
    shape = (5, 6)
    coordinates = rng.uniform(-.5, .5, (71, 2)).astype("float32")
    matrix = _matrix(shape, coordinates)
    plan = prepare_nufft(shape, coordinates, cache_budget_bytes=0)
    image = jnp.asarray(rng.normal(size=shape), dtype=jnp.float32)
    direction = jnp.asarray(rng.normal(size=shape), dtype=jnp.float32)
    batched = jax.jit(jax.vmap(plan.forward))(jnp.stack([image, direction]))
    expected = np.stack([image, direction]).reshape(2, -1) @ matrix.T
    np.testing.assert_allclose(batched, expected, rtol=2e-5, atol=1e-5)

    def loss(value):
        return jnp.sum(abs(plan.forward(value))**2) / 2

    _, hessian_vector = jax.jvp(jax.grad(loss), (image,), (direction,))
    expected = np.real(matrix.conj().T @ (matrix @ np.asarray(direction).ravel()))
    np.testing.assert_allclose(hessian_vector, expected.reshape(shape), rtol=3e-5, atol=2e-5)
    samples = jnp.asarray(_complex(rng, (71,)))
    adjoint_gradient = jax.grad(lambda y: jnp.sum(abs(plan.adjoint(y))**2) / 2)(samples)
    expected = np.conj(matrix @ (matrix.conj().T @ np.asarray(samples)))
    np.testing.assert_allclose(adjoint_gradient, expected, rtol=3e-5, atol=2e-5)


def test_nufft_gradient_workspace_does_not_expand_all_sample_neighbours(device):
    coordinates = np.random.default_rng(31).uniform(-.5, .5, (65537, 3)).astype("float32")
    plan = prepare_nufft((16, 16, 16), coordinates, cache_budget_bytes=0)

    def loss(image):
        return jnp.sum(abs(plan.forward(image))**2) / 2

    executable = jax.jit(jax.value_and_grad(loss)).lower(
        jax.ShapeDtypeStruct((2, 16, 16, 16), jnp.complex64)).compile()
    analysis = executable.memory_analysis()
    if analysis is None:
        pytest.skip("Backend does not expose compiled memory statistics")
    # Saving all expanded scan stencils needs hundreds of MiB for this input.
    assert analysis.temp_size_in_bytes < 64 * 1024**2


@pytest.mark.parametrize("optimizer", ["fista", "split_bregman"])
@pytest.mark.parametrize("regularizer", ["l2", "tv"])
def test_single_precision_proximal_reconstruction_with_x64_enabled(optimizer, regularizer, device):
    with _enable_x64():
        shape = (4, 5)
        coordinates = np.random.default_rng(97).uniform(-.5, .5, (80, 2)).astype("float32")
        image = np.zeros(shape, "complex64")
        image[1:3, 1:4] = 1. + .2j
        samples = (_matrix(shape, coordinates) @ image.ravel()).astype("complex64")[None, None]
        reconstructed, history = solve_variational(samples, spatial_shape=shape,
            trajectory=coordinates, independent_coils=True, optimizer_name=optimizer,
            regularizer=regularizer, regularizer_weight=.001, step_size=50., num_steps=20,
            initial_image=np.zeros((1, 1) + shape, "complex64"), return_history=True, device=device)
        assert reconstructed.dtype == jnp.complex64
        assert np.isfinite(np.asarray(reconstructed)).all() and history[-1] < history[0]


@pytest.mark.parametrize("kind", ["radial", "spiral"])
def test_common_mri_trajectories_and_periodic_coordinates(kind, device):
    radius = np.linspace(-.5, .5, 31, dtype="float32")
    if kind == "radial":
        angle = np.arange(11, dtype="float32") * np.pi / 11
        points = np.stack([np.cos(angle)[:, None] * radius,
                           np.sin(angle)[:, None] * radius], axis=-1).reshape(-1, 2)
    else:
        angle = np.linspace(0., 12 * np.pi, 341, dtype="float32")
        points = np.stack([np.cos(angle), np.sin(angle)], axis=-1) * np.linspace(0., .5, 341)[:, None]
    points = points.astype("float32")
    image = _complex(np.random.default_rng(12), (17, 18))
    expected = _matrix(image.shape, points) @ image.ravel()
    plan = prepare_nufft(image.shape, points, chunk_size=63)
    np.testing.assert_allclose(plan.forward(image), expected, rtol=2e-5, atol=1e-5)
    shifted = prepare_nufft(image.shape, points + np.asarray([1., -1.], "float32"))
    np.testing.assert_allclose(shifted.forward(image), expected, rtol=3e-5, atol=2e-5)


@pytest.mark.parametrize("trajectory,kwargs", [
    (np.zeros((0, 2)), {}), (np.zeros((3, 3)), {}),
    (np.asarray([[np.nan, 0.]]), {}), (np.zeros((3, 2), "complex64"), {}),
    (np.zeros((3, 2)), {"eps": 1e-15}), (np.zeros((3, 2)), {"eps": .1}),
    (np.zeros((3, 2)), {"chunk_size": 0}), (np.zeros((3, 2)), {"batch_size": 0}),
])
def test_invalid_nufft_preparation_fails_explicitly(trajectory, kwargs):
    with pytest.raises(ValueError):
        prepare_nufft((4, 5), trajectory, **kwargs)


def test_unsupported_trajectory_gradients_and_low_precision_are_explicit():
    trajectory = jnp.zeros((3, 2), dtype=jnp.float32)
    with pytest.warns(RuntimeWarning, match="roundoff"):
        prepare_nufft((4, 5), trajectory, eps=1e-9)
    with pytest.raises(ValueError, match="trajectory gradients"):
        jax.grad(lambda points: jnp.sum(prepare_nufft((4, 5), points).forward(
            jnp.ones((4, 5)))).real)(trajectory)


def test_iterative_noncartesian_reconstruction_recovers_independent_data(device):
    shape = (6, 7)
    rng = np.random.default_rng(8)
    trajectory = rng.uniform(-.5, .5, (180, 2)).astype("float32")
    image = np.zeros(shape, "complex64")
    image[1:5, 2:5] = 1. + .2j
    samples = (_matrix(shape, trajectory) @ image.ravel()).astype("complex64")
    result, history = solve_variational(samples, spatial_shape=shape,
        trajectory=trajectory, regularizer="none", optimizer_name="fista", num_steps=160,
        initial_image=np.zeros(shape, "complex64"), return_history=True, device=device)
    assert history[-1] < history[0] * 1e-6
    np.testing.assert_allclose(result, image, atol=2e-4)


def test_noncartesian_model_based_decay_agrees_with_image_fit(device):
    shape = (3, 4)
    times = jnp.asarray([0., .5, 1., 2., 3.])[:, None, None]
    pattern = jnp.asarray(np.arange(12).reshape(shape) / 24 + .5, dtype=jnp.float32)

    def apply(fields, offsets, observables=None):
        return fields["amplitude"][None] * jnp.exp(-fields["rate"][None] * times) * pattern

    def factory(shape):
        return {"amplitude": jnp.full(shape, .8), "rate": jnp.full(shape, .2)}, lambda p: p

    model = Model(name="synthetic_decay", parametrization_factory=factory, model_apply=apply,
        parameter_units={"amplitude": "1", "rate": "1"},
        parameter_constant_dims={"amplitude": ("x", "y"), "rate": ("x", "y")})
    experiment = Experiment(dims={"frame": 5, "x": 3, "y": 4},
        observables={"echo_times": Quantity(times[:, 0, 0], "s")})
    image = apply({"amplitude": jnp.full(shape, 1.3), "rate": jnp.full(shape, .7)}, None)
    trajectory = np.random.default_rng(25).uniform(-.5, .5, (80, 2)).astype("float32")
    samples = (np.asarray(image).reshape(5, -1) @ _matrix(shape, trajectory).T)[:, None]
    fields, _ = run_fit(experiment, model, DataSet(image), num_steps=600, device=device)
    fitted = solve_model_based(samples, experiment=experiment, model=model,
        spatial_shape=shape, trajectory=trajectory, num_steps=600, device=device)
    for name, truth in [("amplitude", 1.3), ("rate", .7)]:
        np.testing.assert_allclose(fields[name].values, truth, atol=3e-4)
        np.testing.assert_allclose(fitted[name].values, fields[name].values, atol=3e-4)
