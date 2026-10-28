"""Synthetic optimizer contracts and numerical references."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from mrax.core import FitError, FitState, build_loss, run_fit
from mrax.core.fit_execution import execute_optimizer
from mrax.core.loss_builder import (
    _data_mask_from_observables,
    _mask_for_regularized_field,
    _reduce_mask_to_field,
    make_model_callable,
)
from mrax.core.variance import _prepare_mask
from mrax.data_struct import DataSet, Experiment, Model, Quantity
from mrax.library.default_models import (
    _bounded,
    _inverse_transform,
    _resolve_bool_observable,
    make_AB_MT_n_model,
)
from mrax.opt.optimizers import make_optimizer
from mrax.reco.variational import build_operator, solve_variational


@pytest.mark.parametrize("name", [
    "adam", "adamw", "sgd", "rmsprop", "adagrad", "adadelta", "adamax",
    "adamaxw", "amsgrad", "lamb", "lars", "lion", "novograd", "noisy_sgd",
    "radam", "sm3", "yogi",
])
def test_bounded_optax_cpu_gpu_consistency(name):
    try:
        gpu = jax.devices("gpu")[0]
    except RuntimeError:
        pytest.skip("GPU backend unavailable")
    cpu = jax.devices("cpu")[0]
    references = []
    with jax.enable_x64():
        for device in (cpu, gpu):
            with jax.default_device(device):
                target = jnp.arange(20, dtype=jnp.float64).reshape(4, 5) / 64 + .25
                initial = {"x": jnp.full((4, 5), -.125, jnp.float64)}

                def loss(p, target=target):
                    return .5 * jnp.sum((p["x"] - target) ** 2)

                optimizer = make_optimizer(name, .02, box_constraints=(-.5, .75), device=device)
                fitted, history, info = execute_optimizer(
                    loss, initial, optimizer, num_steps=64, chunk_size=8, device=device)
                assert int(info.state.step) == 64
                assert all(value.devices() == {device} for value in
                           jax.tree.leaves((fitted, info.state.optimizer_state))
                           if hasattr(value, "devices"))
                values = np.asarray(fitted["x"])
                assert np.isfinite(values).all()
                assert np.all((values >= -.5) & (values <= .75))
                references.append((values, np.asarray(history)))
        np.testing.assert_allclose(references[1][0], references[0][0], rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(references[1][1], references[0][1], rtol=1e-8, atol=1e-10)


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


def _problem(values, initial=0.0, **options):
    values = jnp.asarray(values)
    exp = Experiment(dims={"frame": values.shape[0], "x": values.shape[1], "y": values.shape[2]})
    exp.observables["offsets"] = Quantity(jnp.arange(values.shape[0]), "1")
    exp._mrax_dim_observables = {"frame": "offsets"}
    model = Model(
        name="synthetic_level",
        parameter_units={"level": "1"},
        parametrization_factory=lambda shape: (
            {"level": jnp.full(shape, initial)}, lambda params: params
        ),
        model_apply=lambda fields, offsets, observables=None: jnp.broadcast_to(
            fields["level"], exp.shape
        ),
        **options,
    )
    return exp, model, DataSet(values)


@pytest.mark.parametrize("name", ["adam", "adamw", "sgd", "rmsprop", "adagrad", "yogi", "lamb"])
def test_optimizer_matches_optax_and_resumes_schedule(name):
    rate = optax.piecewise_constant_schedule(0.5 if name == "adagrad" else 0.1, {400: 0.1, 800: 0.1})
    params = {"level": jnp.asarray([0.0, 4.0])}
    target = jnp.asarray([2.0, -1.0])
    def loss(p):
        return 0.5 * jnp.sum((p["level"] - target) ** 2)
    device = params["level"].device
    optimizer = make_optimizer(name, rate, device=device)
    result, history, _ = execute_optimizer(loss, params, optimizer, num_steps=1200, device=device)
    native = getattr(optax, name)(rate)

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
        loss, checkpoint.state.params, optimizer, num_steps=763, initial_state=checkpoint.state,
        device=device,
    )
    np.testing.assert_array_equal(resumed["level"], result["level"])
    np.testing.assert_array_equal(first + tail, history)


@pytest.mark.parametrize("name", ["sgd", "adam", "fista"])
def test_complex_gradient_recovers_complex_target(name):
    target = 1.0 + 2.0j
    def loss(p):
        return 0.5 * jnp.abs(p["level"] - target) ** 2
    params = {"level": jnp.asarray(0.0j, dtype=jnp.complex64)}
    result, _, _ = execute_optimizer(loss, params, make_optimizer(name, 0.1), num_steps=200)
    np.testing.assert_allclose(result["level"], target, atol=0.003)


@pytest.mark.parametrize('name', ['fista', 'split_bregman'])
@pytest.mark.parametrize('scale', [1., 1e-8, 1e-14])
def test_objective_aware_steps_are_invariant_to_loss_scaling(name, scale):
    target = jnp.asarray([1e-4 + 2e-4j, -3e-4 + 1e-4j], dtype=jnp.complex64)
    def loss(params):
        return scale * .5 * jnp.sum(jnp.abs(params['level']-target)**2)
    initial = {'level': jnp.zeros_like(target)}
    optimizer = make_optimizer(name, .9/scale, bregman_penalty=scale,
        proximal_context=(loss, loss, {'level': (-np.inf, np.inf)}, {'level': ()}))
    result, _, info = execute_optimizer(loss, initial, optimizer, num_steps=40)
    np.testing.assert_allclose(result['level'], target, rtol=2e-5, atol=2e-9)
    assert float(info.final_loss) < float(loss(initial)) * 1e-8


@pytest.mark.parametrize('name', ['fista', 'split_bregman'])
def test_backtracking_rejects_overshoot_for_small_signal_losses(name):
    target = jnp.asarray([1e-7, -2e-7], dtype=jnp.float32)
    def loss(params):
        return .5 * jnp.sum((params['level']-target)**2)
    params = {'level': jnp.zeros_like(target)}
    optimizer = make_optimizer(name, 256.,
        proximal_context=(loss, loss, {'level': (-np.inf, np.inf)}, {'level': ()}))
    result, _, info = execute_optimizer(loss, params, optimizer, num_steps=1)
    assert float(info.final_loss) <= float(loss(params))
    assert np.linalg.norm(np.asarray(result['level'])-target) < np.linalg.norm(target)


def test_weak_signal_3d_sense_removes_aliasing_and_matches_hybrid_factorization():
    from mrax.reco.variational import fftnc, ifftnc

    nz, ny, nx = 4, 10, 6
    z, y, x = np.indices((nz, ny, nx))
    image = ((y >= 3) & (y <= 6) & (x >= 1) & (x <= 4)) * (1e-5 * (1 + .1*z))
    image = image * np.exp(1j * (.2*x + .1*z))
    sensitivities = np.stack((np.ones_like(image),
        (.15 + y/ny)*np.exp(.15j*x), (1.1-y/ny)*np.exp(.1j*z)))
    sensitivities /= np.sqrt(np.sum(np.abs(sensitivities)**2,axis=0))
    data = fftnc(jnp.asarray(sensitivities*image), axes=(-3,-2,-1))[None]
    mask = (np.arange(ny)%2==0)[None,None,None,:,None]
    samples = data*mask
    options = dict(regularizer='none', regularizer_weight=0., optimizer_name='fista', num_steps=150)
    full = solve_variational(samples, spatial_shape=(nz,ny,nx),
        sensitivities=sensitivities, mask=mask, **options)
    np.testing.assert_allclose(full[0], image, rtol=2e-3, atol=2e-8)
    # A fully sampled axis is diagonal in its full-image representation.
    hybrid_samples = jnp.transpose(ifftnc(samples,axes=(2,))[0], (1,0,2,3))
    hybrid_sensitivities = sensitivities.transpose(1,0,2,3)
    hybrid = solve_variational(hybrid_samples, spatial_shape=(ny,nx),
        sensitivities=hybrid_sensitivities, mask=mask[:,:,0], **options)
    np.testing.assert_allclose(hybrid, full[0], rtol=2e-3, atol=2e-8)


def test_coupled_decay_has_analytic_optimum_and_supports_per_leaf_weights():
    params = {"level": jnp.asarray(0.0)}
    def loss(p):
        return 0.5 * (p["level"] - 2.0) ** 2
    result, _, _ = execute_optimizer(
        loss, params, make_optimizer("sgd", 0.1, regularizer_weight=0.5), num_steps=200
    )
    np.testing.assert_allclose(result["level"], 2.0 / 1.5, atol=1e-5)
    params = {"a": jnp.asarray(1.0), "b": jnp.asarray(1.0)}
    opt = make_optimizer("sgd", 0.1, regularizer_weight={"a": 0.2, "b": 0.5})
    updates, _ = opt.update(jax.tree.map(jnp.zeros_like, params), opt.init(params), params)
    np.testing.assert_allclose(updates["a"], -0.02)
    np.testing.assert_allclose(updates["b"], -0.05)


@pytest.mark.parametrize("name", ["clip", "scale", "lbfgs"])
def test_unrelated_or_extra_argument_optax_transforms_are_rejected(name):
    with pytest.raises(ValueError, match="Unknown optimizer"):
        make_optimizer(name, 0.1)


@pytest.mark.parametrize("name", ["sgd", "fista", "split_bregman"])
@pytest.mark.parametrize("bounds", [(np.nan, 1.0), (2.0, 1.0)])
def test_invalid_bounds_are_rejected_for_all_optimizer_families(name, bounds):
    with pytest.raises(ValueError, match="bounds"):
        make_optimizer(name, 0.1, box_constraints=bounds).init({"level": jnp.asarray(0.0)})


@pytest.mark.parametrize("option", ["step_size", "regularizer_weight"])
@pytest.mark.parametrize("value", [-0.1, np.nan, np.inf])
def test_invalid_optimizer_scalars_are_rejected(option, value):
    args = {"step_size": 0.1, "regularizer_weight": 0.0, option: value}
    with pytest.raises(ValueError, match="finite"):
        make_optimizer("adam", **args)


def test_negative_scheduled_rate_preserves_a_valid_checkpoint():
    def loss(p):
        return (p["level"] - 2.0) ** 2
    params = {"level": jnp.asarray(0.0)}
    optimizer = make_optimizer("sgd", lambda count: jnp.where(count < 2, 0.1, -0.1))
    with pytest.raises(FitError) as caught:
        execute_optimizer(loss, params, optimizer, num_steps=4, chunk_size=1)
    assert caught.value.last_valid_state.step == 2
    assert len(caught.value.loss_history) == 2


def test_decay_tree_and_bound_shapes_must_match_parameters():
    with pytest.raises(ValueError, match="tree"):
        make_optimizer("sgd", 0.1, regularizer_weight={"typo": 0.2}).init(
            {"level": jnp.asarray(0.0)}
        )
    with pytest.raises(ValueError, match="broadcast"):
        make_optimizer("sgd", 0.1, box_constraints=(jnp.zeros(3), 1.0)).init(
            {"level": jnp.zeros(2)}
        )


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
    fields, _ = run_fit(exp, model, data, optimizer_name="sgd", num_steps=3, mask_fill_value=jnp.nan)
    assert np.isfinite(fields["level"].values).all()
    support = _mask_for_regularized_field(jnp.asarray([[1, 0, 0], [1, 1, 0]]), jnp.ones((1, 3)))
    np.testing.assert_array_equal(support, [[1, 1, 0]])
    exp = Experiment(dims={"x": 2, "frame": 3, "y": 4, "t": 5})
    mask = jnp.ones(exp.shape).at[1, :, 2, 3].set(0)
    reduced = _reduce_mask_to_field(mask, exp, ["t", "x", "y"])
    expected = np.ones((5, 2, 4), dtype=bool)
    expected[3, 1, 2] = False
    np.testing.assert_array_equal(reduced, expected)


def test_multidimensional_observable_mask_unflattens_frames():
    exp = Experiment(dims={"offset": 2, "t": 3, "x": 2, "y": 1})
    observable = jnp.ones((6, 2, 1)).at[4, 1, 0].set(jnp.nan)
    mask = _data_mask_from_observables({"gain": observable}, exp)
    expected = np.ones(exp.shape, dtype=bool)
    expected[1, 1, 1, 0] = False
    np.testing.assert_array_equal(mask, expected)


@pytest.mark.parametrize("which", ["frame", "spatial"])
def test_fractional_masks_weight_the_residual(which):
    values = [[[1.0]], [[3.0]]] if which == "frame" else [[[1.0], [3.0]]]
    exp, model, data = _problem(values)
    options = {"frame_mask": jnp.asarray([1.0, 0.1])} if which == "frame" else {
        "mask": jnp.asarray([[1.0], [0.1]])
    }
    loss, params, _ = build_loss(exp, model, data, **options)
    assert float(loss(params)) == pytest.approx(1.9 / (1.1 * 5.0), rel=1e-6)


def test_custom_parameter_name_receives_configured_regularizer_weights():
    exp, model, data = _problem([[[1.0], [2.0]]], reg_weights={"level": 0.2})
    loss, params, _ = build_loss(exp, model, data)
    params["level"] = jnp.asarray([[0.0], [1.0]])
    base, _, _ = build_loss(exp, model, data, regularizer_configs={})
    assert float(loss(params) - base(params)) == pytest.approx(0.2, abs=1e-6)


@pytest.mark.parametrize("option", ["unknown_regularizer", "unknown_constant", "negative_weight", "negative_mask", "empty_mask"])
def test_invalid_fitting_options_raise_instead_of_silently_disabling_features(option):
    exp, model, data = _problem([[[1.0], [2.0]]])
    kwargs = {}
    if option == "unknown_regularizer":
        kwargs["regularizer_configs"] = {"typo": [{"type": "tv", "weight": 0.2}]}
    elif option == "unknown_constant":
        model.parameter_constant_dims = {"typo": ("x",)}
    elif option == "negative_weight":
        kwargs["regularizer_weights"] = {"level": -0.1}
    else:
        kwargs["mask"] = jnp.full((2, 1), -1.0 if option == "negative_mask" else 0.0)
    with pytest.raises(ValueError):
        build_loss(exp, model, data, **kwargs)


def test_roi_normalization_is_default_and_full_mode_preserves_old_scaling():
    exp, model, data = _problem([[[1.0], [1000.0]]])
    mask = jnp.asarray([[1.0], [0.0]])
    roi, params, _ = build_loss(exp, model, data, mask=mask)
    full, _, _ = build_loss(exp, model, data, mask=mask, data_normalization="full")
    assert float(roi(params)) == pytest.approx(1.0, rel=1e-6)
    assert float(full(params)) == pytest.approx(2.0 / 1000001.0, rel=1e-6)
    roi_fields, _ = run_fit(exp, model, data, num_steps=1, optimizer_name="sgd",
                            step_size=0.1, mask=mask)
    full_fields, _ = run_fit(exp, model, data, num_steps=1, optimizer_name="sgd",
                             step_size=0.1, mask=mask, data_normalization="full")
    assert float(roi_fields["level"].values[0, 0]) == pytest.approx(0.2, rel=1e-6)
    assert float(full_fields["level"].values[0, 0]) == pytest.approx(0.4 / 1000001.0, rel=1e-6)
    with pytest.raises(ValueError, match="data_normalization"):
        build_loss(exp, model, data, data_normalization="invalid")


def test_roi_normalization_excludes_masked_frames_and_invalid_observables():
    exp, model, data = _problem([[[1.0]], [[1000.0]]])
    loss, params, _ = build_loss(exp, model, data, frame_mask=jnp.asarray([1.0, 0.0]))
    assert float(loss(params)) == pytest.approx(1.0, rel=1e-6)
    exp.observables["gain"] = Quantity(jnp.asarray([1.0, np.nan]), "1")
    exp._mrax_dim_observables = {"frame": "gain"}
    loss, params, _ = build_loss(exp, model, data)
    assert float(loss(params)) == pytest.approx(1.0, rel=1e-6)


def test_rms_weight_compensation_preserves_the_entire_objective():
    exp, model, data = _problem([[[1.0], [2.0], [100.0]]])
    mask = jnp.asarray([[1.0], [1.0], [0.0]])
    factor = ((1.0 + 4.0 + 10000.0) / 3.0) / ((1.0 + 4.0) / 2.0)
    def config(weight):
        return {"level": [{"type": "smoothness", "axes": (0,), "weight": weight}]}
    old, params, _ = build_loss(exp, model, data, mask=mask, regularizer_configs=config(0.2),
                               data_normalization="full")
    new, _, _ = build_loss(exp, model, data, mask=mask, regularizer_configs=config(0.2 * factor))
    for values in ([[0.0], [1.0], [10.0]], [[2.0], [0.0], [0.0]]):
        params["level"] = jnp.asarray(values)
        np.testing.assert_allclose(new(params), factor * old(params), rtol=2e-6)


def test_tiny_uniform_variances_preserve_finite_relative_weights():
    exp, model, data = _problem(jnp.ones((3, 2, 2)))
    loss, params, _ = build_loss(exp, model, data, data_variance=1e-38)
    assert float(loss(params)) == pytest.approx(1.0, rel=1e-6)


def test_shared_model_uses_current_globals_and_explicit_empty_mapping():
    exp, model, data = _problem(jnp.ones((2, 2, 1)), initial=1.0)
    model.model_apply = lambda fields, offsets, observables=None: jnp.broadcast_to(
        fields["level"], exp.shape
    ) * observables.get("gain", 1.0)
    exp.study_globals = {"gain": Quantity(1.0, "1")}
    run_fit(exp, model, data, num_steps=0)
    assert not getattr(model, "_mrax_study_globals", None)
    other = exp.insert_observables({})
    other.study_globals = {"gain": Quantity(2.0, "1")}
    fields, _ = run_fit(other, model, DataSet(jnp.full(exp.shape, 2.0)), num_steps=1)
    np.testing.assert_allclose(fields["level"].values, 1.0, atol=1e-7)
    predicted = make_model_callable(other, model, study_globals={})({"level": jnp.ones((2, 1))})
    np.testing.assert_array_equal(predicted, jnp.ones(exp.shape))


def test_softplus_inverse_handles_large_finite_initial_values():
    raw = _inverse_transform(jnp.asarray([100.0, 1000.0]), {"transform": "softplus"})
    assert np.isfinite(raw).all()
    np.testing.assert_allclose(jax.nn.softplus(raw), [100.0, 1000.0])


@pytest.mark.parametrize("options", [{"proximal_tolerance": np.nan}, {"proximal_max_steps": 1.5}, {"bregman_inner_steps": 0}, {"bregman_penalty": np.inf}])
def test_tv_solver_rejects_invalid_inner_options(options):
    with pytest.raises((ValueError, TypeError)):
        make_optimizer("fista", 0.1, proximal_context=(
            lambda p: p["level"] ** 2, lambda p: p["level"] ** 2,
            {"level": (-2.0, 2.0)}, {}), **options)


@pytest.mark.parametrize("name", ["fista", "split_bregman"])
def test_reconstruction_proximal_l1_matches_declared_mean_penalty(name):
    target = jnp.ones((2, 2))
    op = build_operator(spatial_shape=(2, 2), kspace=jnp.zeros((2, 2)))
    result = solve_variational(op.forward(target), spatial_shape=(2, 2),
                                     initial_image=jnp.zeros((2, 2)), regularizer="l1",
                                     regularizer_weight=0.1, optimizer_name=name,
                                     step_size=0.5, num_steps=300)
    np.testing.assert_allclose(result, 0.9, atol=2e-4)


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
        assert state["solver"]["t"].dtype == jnp.float64
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
        updates, state = optimizer.update({"p": jnp.zeros(())}, state, params,
                                         gradient_at_extrapolated=name == "fista",
                                         **({"value_fn": lambda p: jnp.asarray(0.)}
                                            if name == "split_bregman" else {}))
        params = optax.apply_updates(params, updates)
    assert int(state["solver"]["count"]) == 2
    if name == "fista":
        assert float(params["p"]) == pytest.approx(0.97)
    else:
        # Split Bregman starts with zero multipliers; the first primal solve
        # leaves a constant data term unchanged, then shrinkage creates b.
        assert int(state["status"]) == 0
        assert float(params["p"]) < 1.


def test_zero_weight_bregman_keeps_its_state_custom_prox_and_bounds():
    plain = make_optimizer("split_bregman", 0.1)
    params = {"p": jnp.asarray(1.0)}
    assert "splits" in plain.init(params)['solver'] and "duals" in plain.init(params)['solver']
    custom = make_optimizer("split_bregman", 0.1, regularizer_prox=lambda value, weight: jnp.maximum(value, 0),
                            base_optimizer_factory=optax.sgd)
    negative = {"p": jnp.asarray(-1.)}
    _, state = custom.update({"p": jnp.zeros(())}, custom.init(negative), negative,
                             value_fn=lambda p: jnp.asarray(0.))
    assert float(state["solver"]["splits"]["0"][0][..., 0]) == 0
    bounded = make_optimizer("split_bregman", 0.1, box_constraints=(0.0, 0.5))
    updates, _ = bounded.update({"p": jnp.zeros(())}, bounded.init(params), params,
                                value_fn=lambda p: jnp.asarray(0.))
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
            loss, params, _ = build_loss(exp, model, data, device=device)
            params = jax.device_put(params, device)
            value, grad = jax.jit(jax.value_and_grad(loss))(params)
            assert device in value.devices()
            results.append((float(value), np.asarray(grad["level"])))
            fields, _, info = run_fit(
                exp, model, data, optimizer=optax.sgd(0.3), num_steps=64,
                convergence_tolerance=1e-7, min_steps=5, return_fit_info=True,
                device=device,
            )
            assert device in fields["level"].values.devices()
            assert info.converged
            np.testing.assert_allclose(fields["level"].values, 0.3, atol=1e-4)
    np.testing.assert_allclose(results[0][0], results[1][0], rtol=1e-5)
    np.testing.assert_allclose(results[0][1], results[1][1], rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize('name', ['fista', 'split_bregman'])
@pytest.mark.parametrize('scale', [1., 1e-8, 1e-14])
def test_extraargs_known_convex_optimum_and_resume(name, scale):
    target=jnp.asarray([2., -3.])
    def loss(p):
        return scale * .5 * jnp.sum((p['x']-target)**2)
    # Scaling the entire objective also scales an absolute augmented penalty.
    # rho tunes conditioning while the objective defines the minimizer.
    options=dict(regularizer_weight=.5*scale, box_constraints=(-2., 2.), bregman_penalty=scale)
    # Deliberately unsafe initial rate must be reduced by line search.
    opt=make_optimizer(name, 20./scale, **options)
    start={'x':jnp.zeros(2)}
    result, history, info=execute_optimizer(loss,start,opt,num_steps=1000)
    np.testing.assert_allclose(result['x'],[1.5,-2.],atol=1e-3)
    assert np.isfinite(history).all() and np.isfinite(info.final_loss)
    assert info.final_loss / scale == pytest.approx(2.375, abs=2e-4)
    partial,_,first=execute_optimizer(loss,start,opt,num_steps=37)
    resumed,_,_=execute_optimizer(loss,partial,opt,num_steps=963,initial_state=first.state)
    np.testing.assert_array_equal(resumed['x'],result['x'])


@pytest.mark.parametrize('name',['fista','split_bregman','adam'])
def test_active_fit_equivalence_and_resume(name):
    exp,model,data=_problem([[[1.],[9.],[2.]],[[2.],[8.],[3.]]])
    model.parameter_specs={'level':{'transform':'identity','init':0.,'scale':1.}}
    mask=jnp.asarray([[1.],[0.],[1.]])
    config={'level':[{'type':'tv','weight':.01}]}
    options=dict(optimizer_name=name,num_steps=40,step_size=.02,mask=mask,
                 regularizer_configs=config,return_fit_info=True)
    full,_,_=run_fit(exp,model,data,**options)
    packed,_,info=run_fit(exp,model,data,fit_active_only=True,**options)
    # Default curvature coordinates depend on original full shapes and remain
    # fixed before packing; raw Adam is unchanged on active pixels.
    np.testing.assert_allclose(packed['level'].values,full['level'].values,atol=2e-5)
    options.update(num_steps=20)
    _,_,partial=run_fit(exp,model,data,fit_active_only=True,**options)
    resumed,_,_=run_fit(exp,model,data,fit_active_only=True,initial_state=partial.state,**options)
    np.testing.assert_array_equal(resumed['level'].values,packed['level'].values)
    assert info.state.active_layout
    with pytest.raises(ValueError,match='layout'):
        run_fit(exp,model,data,fit_active_only=True,initial_state=partial.state,
                **dict(options,mask=jnp.asarray([[0.],[1.],[1.]])))


@pytest.mark.parametrize('name',['fista','split_bregman'])
def test_custom_convex_projection_and_box(name):
    # Nonseparable projection onto the sum-zero plane plus a box. Sequential
    # clipping is incorrect; the combined prox must solve the intersection.
    def prox(x, w):
        return x-jnp.mean(x)
    def loss(p):
        return .5*jnp.sum((p['x']-jnp.asarray([4.,1.,-1.]))**2)
    opt=make_optimizer(name,.1,regularizer_prox=prox,box_constraints=(-1.,1.))
    result,_,_=execute_optimizer(loss,{'x':jnp.zeros(3)},opt,num_steps=300)
    np.testing.assert_allclose(result['x'],[1.,0.,-1.],atol=1e-4)


@pytest.mark.parametrize('name',['fista','split_bregman'])
def test_double_precision_loss_with_single_precision_parameters(name):
    enable = getattr(jax, 'enable_x64', None)
    if enable is None:
        from jax.experimental import enable_x64
        enable = enable_x64
    with enable():
        def loss(p):
            return jnp.sum((p['x']-jnp.asarray(2.,jnp.float64))**2)
        start={'x':jnp.zeros((2,),jnp.float32)}
        result,_,_=execute_optimizer(loss,start,make_optimizer(name,.1),num_steps=60)
        assert result['x'].dtype==jnp.float32
        np.testing.assert_allclose(result['x'],2.,atol=2e-3)


@pytest.mark.parametrize('name',['fista','split_bregman'])
def test_gradient_clipping_cannot_certify_a_distant_stationary_point(name):
    def loss(p):
        return .5*jnp.sum((p['x']-2.)**2)
    opt=make_optimizer(name,.1,grad_clip_norm=1e-12)
    result,_,info=execute_optimizer(loss,{'x':jnp.zeros(2)},opt,num_steps=12,
        convergence_tolerance=1e-6,parameter_tolerance=1e-5,convergence_patience=2)
    assert not info.converged and info.termination_reason=='max_steps'
    assert float(jnp.linalg.norm(jax.grad(loss)(result)['x']))>1.


@pytest.mark.parametrize('name', ['fista', 'split_bregman'])
@pytest.mark.parametrize('option,value', [
    ('proximal_tolerance', 0.),
    ('proximal_max_steps', 0),
    ('proximal_max_step_size', -1.),
    ('bregman_inner_steps', 0),
])
def test_raw_fit_validates_proximal_options(name, option, value):
    exp, model, data = _problem([[[1.]], [[2.]]])
    with pytest.raises(ValueError):
        run_fit(exp, model, data, optimizer_name=name, num_steps=0, **{option: value})


@pytest.mark.parametrize('name', ['fista', 'split_bregman'])
def test_raw_fit_honors_proximal_step_cap(name):
    exp, model, data = _problem([[[1.]], [[2.]]])
    _, _, info = run_fit(exp, model, data, optimizer_name=name, num_steps=5,
                         step_size=2., proximal_max_step_size=.01, return_fit_info=True)
    assert float(info.state.optimizer_state['solver']['rate']) <= .010001


def test_raw_bregman_fit_honors_inner_budget():
    exp, model, data = _problem([[[1.]], [[2.]]])
    losses = []
    for inner_steps in (1, 3):
        _, _, info = run_fit(exp, model, data, optimizer_name='split_bregman',
                             step_size=.01, num_steps=3, bregman_inner_steps=inner_steps,
                             return_fit_info=True)
        losses.append(info.final_loss)
    assert losses[1] < losses[0]


@pytest.mark.parametrize('initial,expected_steps,expected_checks', [(0., 5, 0), (2., 2, 2)])
def test_expensive_convergence_check_waits_for_small_updates(initial, expected_steps, expected_checks):
    calls = []

    def loss(params):
        return .5 * jnp.sum((params['x'] - 2.)**2)

    def convergence(params, state, tolerance):
        jax.debug.callback(lambda: calls.append(1), ordered=True)
        return jnp.asarray(True)

    _, _, info = execute_optimizer(loss, {'x': jnp.full(2, initial)}, make_optimizer('sgd', .1),
        num_steps=5, convergence_tolerance=1e-10, parameter_tolerance=1e-10,
        convergence_patience=2, convergence_check=convergence)
    jax.effects_barrier()
    assert info.iterations == expected_steps
    assert len(calls) == expected_checks


@pytest.mark.parametrize('name', ['fista', 'split_bregman'])
def test_checkpoint_resume_accepts_redundant_state_fields(name):
    def loss(params):
        return .5 * jnp.sum((params['x'] - 2.)**2)

    start = {'x': jnp.zeros(2)}
    optimizer = make_optimizer(name, .1, regularizer_weight=.2)
    expected, _, _ = execute_optimizer(loss, start, optimizer, num_steps=12)
    params, _, partial = execute_optimizer(loss, start, optimizer, num_steps=5)
    old_solver = {**partial.state.optimizer_state['solver'], 'x_ref': {'0': params['x']}}
    if name == 'fista':
        old_solver['x'] = {'0': params['x']}
    old_state = replace(partial.state,
                        optimizer_state={**partial.state.optimizer_state, 'solver': old_solver})
    resumed, _, info = execute_optimizer(loss, params, optimizer, num_steps=7, initial_state=old_state)
    np.testing.assert_array_equal(resumed['x'], expected['x'])
    assert info.state.step == 12
