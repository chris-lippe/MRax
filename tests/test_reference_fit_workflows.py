"""Synthetic checks for reference-scaled fitting and dynamic model fields."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mrax.core import build_loss, run_fit
from mrax.core.loss_builder import make_model_callable
from mrax.data_struct import (
    AcquisitionData,
    DataSet,
    Experiment,
    Model,
    Quantity,
    ReconstructionRecipe,
)
from mrax.library.default_models import _make_parametrization_factory, make_AB_MT_n_model
from mrax.reco import build_operator, solve_model_based
from mrax.reco.variational import _model_encoding_loss


@pytest.fixture(params=["cpu", "gpu"])
def device(request):
    try:
        selected = jax.devices(request.param)[0]
    except RuntimeError:
        pytest.skip(f"{request.param} is unavailable")
    with jax.default_device(selected):
        yield selected


def _relative_problem():
    specs = {"level": {"transform": "identity", "init": .5, "unit": "1"}}

    def apply(fields, offsets, observables=None):
        times = observables["TE"]
        times = times.values if isinstance(times, Quantity) else times
        return fields["level"][None] * jnp.exp(-jnp.asarray(times)[:, None, None] / 10)

    model = Model(name="synthetic_relative", parametrization_factory=_make_parametrization_factory(specs),
                  model_apply=apply, parameter_units={"level": "1"}, parameter_specs=specs)
    experiment = Experiment(dims={"echo": 3, "x": 4, "y": 5},
                            observables={"TE": Quantity(jnp.asarray([1., 3., 7.]), "ms")})
    experiment._mrax_dim_observables = {"echo": "TE"}
    return experiment, model


@pytest.mark.parametrize("sampled", [False, True])
@pytest.mark.parametrize("coil_count", [1, 3])
def test_reference_forward_fitting_matches_regularized_model_reconstruction(device, sampled, coil_count):
    experiment, model = _relative_problem()
    truth = jnp.asarray(np.arange(20).reshape(4, 5) / 15 + .5, jnp.float32)
    reference = jnp.asarray(np.linspace(.2, 2., 20).reshape(4, 5), jnp.float32)
    relative = make_model_callable(experiment, model)({"level": truth})
    data = relative * reference
    roi = np.ones((4, 5), bool)
    roi[0, :2] = False
    configs = {"level": [{"type": "tv", "weight": .03},
                         {"type": "smoothness", "weight": .02}]}
    frame_mask = jnp.asarray([1., 0., 1.])
    fields, _, predict = run_fit(experiment, model, DataSet(data), reference_image=reference,
        mask=roi, frame_mask=frame_mask, regularizer_configs=configs, num_steps=80, optimizer_name="fista",
        return_model_callable=True, device=device)
    sensitivities = jnp.ones((coil_count, 4, 5), jnp.complex64) / coil_count**.5
    phase = jnp.asarray([-.2, .1, .3])[:, None, None]
    encoded = build_operator(spatial_shape=(4, 5), sensitivities=sensitivities).forward(
        jnp.where(roi, data, 0) * jnp.exp(1j * phase))
    trajectory = None
    if sampled:
        trajectory = np.stack(np.meshgrid(*[np.fft.fftshift(np.fft.fftfreq(n))
            for n in (4, 5)], indexing="ij"), axis=-1).reshape(-1, 2).astype("float32")
        encoded = encoded.reshape((3, coil_count, -1))
    reconstructed, _, model_prediction, image_prediction = solve_model_based(encoded, experiment=experiment, model=model,
        trajectory=trajectory, spatial_shape=(4, 5), roi_mask=roi, object_support_mask=roi,
        sensitivities=sensitivities, image_phase=phase,
        frame_mask=frame_mask,
        reference_image=reference, regularizer_configs=configs, optimizer_name="fista",
        num_steps=80, encoding_batch_size=2, proximal_tolerance=1e-10, proximal_max_steps=4096, device=device,
        return_model_callable=True, return_reconstruction_image=True)
    np.testing.assert_allclose(reconstructed["level"].values[roi], fields["level"].values[roi],
                               rtol=2e-4, atol=2e-4)
    expected = make_model_callable(experiment, model)(fields) * reference
    np.testing.assert_allclose(predict(fields), expected, atol=1e-6)
    np.testing.assert_allclose(model_prediction(reconstructed), image_prediction, rtol=2e-6, atol=2e-6)


def test_forward_reference_validates_shape_and_allows_zero_reference():
    experiment, model = _relative_problem()
    reference = jnp.ones((4, 5)).at[1, 2].set(0)
    callable_ = make_model_callable(experiment, model, reference_image=Quantity(reference[None], "1"))
    predicted = callable_({"level": jnp.ones((4, 5))})
    assert np.isfinite(np.asarray(predicted)).all()
    np.testing.assert_array_equal(predicted[:, 1, 2], 0)
    loss, params, _ = build_loss(experiment, model, DataSet(predicted), reference_image=reference)
    assert np.isfinite(float(jax.jit(loss)(params)))
    for invalid in [np.ones((2, 3)), np.full((4, 5), np.nan)]:
        with pytest.raises(ValueError, match="reference_image"):
            make_model_callable(experiment, model, reference_image=invalid)


def test_signal_forward_objective_matches_inverse_variance_weighted_relative_fit():
    experiment, model = _relative_problem()
    reference = jnp.linspace(.3, 2., 20).reshape(4, 5)
    relative = make_model_callable(experiment, model)({"level": jnp.ones((4, 5))}) + .01
    signal = relative * reference
    roi = np.ones((4, 5), bool)
    roi[0] = False
    frames = np.asarray([1., 0., 1.])
    included = np.broadcast_to(roi, relative.shape) & (frames[:, None, None] > 0)
    factor = float(jnp.mean(signal[included]**2) /
                   (jnp.mean(relative[included]**2) * jnp.mean(jnp.broadcast_to(reference**2, relative.shape)[included])))
    regularizers = {"level": [{"type": "tv", "weight": .03},
                              {"type": "smoothness", "weight": .02}]}
    scaled_regularizers = {key: [{**entry, "weight": entry["weight"] * factor} for entry in entries]
                           for key, entries in regularizers.items()}
    forward, params, _ = build_loss(experiment, model, DataSet(signal), mask=roi,
        frame_mask=frames, reference_image=reference, regularizer_configs=regularizers)
    weighted, _, _ = build_loss(experiment, model, DataSet(relative), mask=roi,
        frame_mask=frames, data_variance=1 / reference**2, regularizer_configs=scaled_regularizers)
    params = {"level_raw": jnp.linspace(.4, 1.5, 20).reshape(4, 5)}
    expected_value, expected_gradient = jax.value_and_grad(forward)(params)
    actual_value, actual_gradient = jax.value_and_grad(weighted)(params)
    np.testing.assert_allclose(actual_value, expected_value * factor, rtol=2e-6)
    np.testing.assert_allclose(actual_gradient["level_raw"], expected_gradient["level_raw"] * factor, rtol=2e-6, atol=1e-7)


@pytest.mark.parametrize("sampled", [False, True])
def test_batched_encoding_preserves_loss_gradients_and_curvature(device, sampled):
    rng = np.random.default_rng(24)
    shape = (5, 4, 5)
    image = jnp.asarray(rng.normal(size=shape) + 1j * rng.normal(size=shape), jnp.complex64)
    sensitivities = jnp.asarray(rng.normal(size=(5, 2, 4, 5)) + 1j * rng.normal(size=(5, 2, 4, 5)), jnp.complex64)
    mask = jnp.asarray(rng.uniform(size=(5, 1, 4, 5)) > .3)
    phase = jnp.asarray(rng.normal(size=(5, 2)), jnp.float32)
    polarity = jnp.broadcast_to(jnp.asarray([1, -1, 1, -1])[:, None], (4, 5))
    trajectory = None
    if sampled:
        trajectory = np.stack(np.meshgrid(*[np.fft.fftshift(np.fft.fftfreq(n)) for n in shape[1:]],
            indexing="ij"), axis=-1).reshape(-1, 2).astype("float32")
        mask, phase, polarity = None, None, None
    options = dict(spatial_shape=shape[1:], sensitivities=sensitivities, trajectory=trajectory,
                   readout_phase=phase, readout_polarity=polarity)
    op = build_operator(mask=mask, **options)
    full = None if sampled else build_operator(**options)
    target = op.forward(image) + .1
    frame_weights = jnp.asarray([1., .5, 0., 1., 1.]).reshape((5,) + (1,) * (target.ndim - 1))
    weights = jnp.broadcast_to(frame_weights, target.shape)
    if mask is not None:
        weights = weights * mask
    missing = None if sampled else (1 - mask) * frame_weights * jnp.ones_like(target.real)
    kwargs = dict(samples=target, weights=weights, denominator=jnp.sum(weights), scale=.3,
                  frame_weights=frame_weights, full_op=full,
                  completion=None if sampled else full.forward(image) - .2,
                  completion_weights=missing,
                  completion_denominator=None if sampled else jnp.sum(missing), completion_weight=.4)
    unbatched = _model_encoding_loss(op, batch_size=None, **kwargs)
    batched = _model_encoding_loss(op, batch_size=2, **kwargs)
    argument = image * .8
    for function in [jax.jit, lambda fn: jax.jit(jax.grad(fn)),
                     lambda fn: jax.jit(lambda x: jax.jvp(jax.grad(fn), (x,), (jnp.ones_like(x),))[1])]:
        np.testing.assert_allclose(function(batched)(argument), function(unbatched)(argument),
                                   rtol=3e-5, atol=2e-7)


def test_encoding_batches_broadcast_a_static_model_image_across_frames(device):
    image = jnp.arange(20, dtype=jnp.float32).reshape(1, 4, 5) / 20
    sensitivities = jnp.ones((5, 2, 4, 5), jnp.complex64) / 2**.5
    op = build_operator(spatial_shape=(4, 5), sensitivities=sensitivities)
    target = op.forward(image) + .1
    weights = jnp.ones_like(target.real)

    def expected(value):
        return jnp.mean(jnp.abs(op.forward(value) - target)**2)

    for batch_size in [None, 2]:
        actual = _model_encoding_loss(op, samples=target, weights=weights,
            denominator=jnp.sum(weights), scale=1., frame_weights=None, batch_size=batch_size)
        value, gradient = jax.jit(jax.value_and_grad(actual))(image)
        expected_value, expected_gradient = jax.jit(jax.value_and_grad(expected))(image)
        np.testing.assert_allclose(value, expected_value, rtol=2e-6)
        np.testing.assert_allclose(gradient, expected_gradient, rtol=2e-6, atol=1e-7)


@pytest.mark.parametrize("spatial_shape", [(2, 3), (2, 3, 2)])
def test_dynamic_saturation_fields_broadcast_over_spatial_axes_only(device, spatial_shape):
    spatial = dict(zip(("x", "y", "z"), spatial_shape))
    dims = {"t": 3, "offset": 4, **spatial}
    observables = {"freq_offsets": Quantity(jnp.asarray([-2., -.8, .7, 2.]), "ppm"),
                   "TR": Quantity(jnp.full((12,), 1200.), "ms"),
                   "sat_time": Quantity(300., "ms"), "sat_b1": Quantity(2., "uT"),
                   "field_strength_mhz": Quantity(120., "MHz"),
                   "r1_map": Quantity(jnp.ones(spatial_shape), "1/s"),
                   "z_prev": Quantity(jnp.full((3, 4) + spatial_shape, .9), "1")}
    experiment = Experiment(dims=dims, observables=observables)
    experiment._mrax_dim_observables = {"t": None, "offset": "freq_offsets"}
    model = make_AB_MT_n_model()
    params, unpack = model.parametrization_factory((3,) + spatial_shape)
    fields = unpack(params)
    predict = make_model_callable(experiment, model)
    assert jax.eval_shape(predict, fields).shape == (12,) + spatial_shape
    actual = jax.jit(predict)(fields)
    expected = []
    for time in range(3):
        local = Experiment(dims={"offset": 4, **spatial}, observables={
            **observables, "TR": Quantity(jnp.full((4,), 1200.), "ms"),
            "z_prev": Quantity(jnp.full((4,) + spatial_shape, .9), "1")})
        local._mrax_dim_observables = {"offset": "freq_offsets"}
        expected.append(make_model_callable(local, model)({k: v[time] for k, v in fields.items()}))
    np.testing.assert_allclose(actual, jnp.concatenate(expected), rtol=2e-6, atol=2e-6)
    gradients = jax.jit(jax.grad(lambda p: jnp.sum(predict(unpack(p)))))(params)
    assert all(np.isfinite(np.asarray(value)).all() for value in gradients.values())


def test_proximal_metric_uses_the_supplied_initial_parameters(monkeypatch):
    import mrax.core.physical_parameters as coordinates

    experiment, model = _relative_problem()
    original = coordinates.balance_physical_coordinates
    balanced = []

    def inspect_initial(loss, params, unpack, operators, **kwargs):
        if kwargs.get("balance", True):
            balanced.append(np.asarray(unpack(params)["level"]))
        return original(loss, params, unpack, operators, **kwargs)

    monkeypatch.setattr(coordinates, "balance_physical_coordinates", inspect_initial)
    run_fit(experiment, model, DataSet(jnp.ones((3, 4, 5))),
        initial_params={"level_raw": jnp.full((4, 5), 3.)}, num_steps=1,
        regularizer_configs={"level": [{"type": "tv", "weight": .01}]})
    assert len(balanced) == 1
    np.testing.assert_allclose(balanced[0], 3.)


@pytest.mark.parametrize("scale", [1., 1e-4, 0.])
@pytest.mark.parametrize("vector", [False, True])
def test_saturation_forward_reference_does_not_bias_previous_signal_at_small_scales(scale, vector):
    reference = jnp.full((2, 3), scale)
    experiment = Experiment(dims={"offset": 3, "x": 2, "y": 3}, observables={
        "freq_offsets": Quantity(jnp.asarray([-2., .8, 2.]), "ppm"),
        "TR": Quantity(jnp.full((3,), 1200.), "ms"),
        "sat_time": Quantity(300., "ms"), "sat_b1": Quantity(2., "uT"),
        "field_strength_mhz": Quantity(120., "MHz"),
        "r1_map": Quantity(jnp.ones((2, 3)), "1/s"),
        "z_prev": Quantity(jnp.asarray([.9, .8, .7]) if vector else jnp.full((3, 2, 3), .9), "1"),
        "z_reference_image": Quantity(reference, "1")})
    experiment._mrax_dim_observables = {"offset": "freq_offsets"}
    model = make_AB_MT_n_model()
    params, unpack = model.parametrization_factory((2, 3))
    expected = make_model_callable(experiment, model)(unpack(params)) * reference
    samples = build_operator(spatial_shape=(2, 3)).forward(expected)
    fields, _, predict, image = solve_model_based(samples, experiment=experiment, model=model,
        reference_image=reference, num_steps=0, return_model_callable=True,
        return_reconstruction_image=True)
    np.testing.assert_allclose(image, expected, rtol=2e-6, atol=1e-12)
    np.testing.assert_allclose(predict(fields), expected, rtol=2e-6, atol=1e-12)
    if scale:
        changed = reference * 2
        expected_experiment = experiment.insert_observables({
            "z_prev": experiment.observables["z_prev"].with_values(experiment.observables["z_prev"].values / 2)})
        expected_changed = make_model_callable(expected_experiment, model)(unpack(params)) * changed
        predicted_changed = make_model_callable(experiment, model, reference_image=changed)(unpack(params))
        np.testing.assert_allclose(predicted_changed, expected_changed, rtol=2e-6, atol=1e-12)


@pytest.mark.parametrize("sampled", [False, True])
def test_model_reconstruction_does_not_allocate_an_unused_fft_preview(monkeypatch, sampled):
    from mrax.reco import gridding, variational
    from mrax.reco.acquisition import reconstruct_acquisition

    experiment, model = _relative_problem()
    samples = np.ones((3, 1, 4, 5), "complex64")
    trajectory = None
    if sampled:
        trajectory = np.zeros((20, 2), "float32")
        samples = samples.reshape(3, 1, 20)
    acquisition = AcquisitionData(samples, (4, 5), ("phase", "read"), trajectory=trajectory,
                                  kind="sampled" if sampled else "cartesian")

    def unexpected_preview(*args, **kwargs):
        raise AssertionError("Model reconstruction does not need an initial coil-image series")

    image = jnp.ones((3, 4, 5))
    monkeypatch.setattr(variational, "solve_model_based", lambda *a, **kw: ({}, [], None, image))
    monkeypatch.setattr(variational, "build_operator", unexpected_preview)
    monkeypatch.setattr(gridding, "grid_kaiser_bessel", unexpected_preview)
    result = reconstruct_acquisition(acquisition, method="model_based", experiment=experiment, model=model)
    np.testing.assert_allclose(result.image, image)


def test_shared_frame_layout_roundtrip_has_bounded_trace_size():
    from mrax.reco.acquisition import format_coil_images
    from mrax.reco.layout import image_to_encoding

    trace_sizes = []
    for frames in [5, 257]:
        recipe = ReconstructionRecipe(frame_permutations=((1, 0),) * frames)
        acquisition = AcquisitionData(np.zeros((frames, 1, 3, 4), "complex64"), (3, 4),
                                      ("phase", "read"), recipe=recipe)
        original = jnp.arange(frames * 12, dtype=jnp.float32).reshape(frames, 1, 3, 4)
        formatted = format_coil_images(original, recipe, spatial_ndim=2)[:, 0]
        def restore(image, acquisition=acquisition):
            return image_to_encoding(image, acquisition)
        np.testing.assert_array_equal(jax.jit(restore)(formatted), original[:, 0])
        trace_sizes.append(len(jax.make_jaxpr(restore)(formatted).jaxpr.eqns))
    assert trace_sizes[1] <= trace_sizes[0] + 2
