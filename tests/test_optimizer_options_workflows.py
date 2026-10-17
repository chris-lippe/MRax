"""Synthetic regression tests for optimizer options and fitting masks."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from mrax.core import FitError, build_loss, run_fit
from mrax.core.fit_execution import execute_optimizer
from mrax.core.loss_builder import (
    _data_mask_from_observables,
    _mask_for_regularized_field,
    _reduce_mask_to_field,
    make_model_callable,
)
from mrax.core.tv_fit import make_tv_steps
from mrax.core.variance import _prepare_mask
from mrax.data_struct import DataSet, Experiment, Model, Quantity
from mrax.library.default_models import _inverse_transform
from mrax.opt.optimizers import make_optimizer
from mrax.reco.variational import build_operator, solve_variational


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
    optimizer, step, _ = make_tv_steps(loss, loss, {'level': (-np.inf, np.inf)},
        {'level': ()}, name, .9/scale, penalty=scale)
    result, _, info = execute_optimizer(loss, initial, optimizer, num_steps=40, step_function=step)
    np.testing.assert_allclose(result['level'], target, rtol=2e-5, atol=2e-9)
    assert float(info.final_loss) < float(loss(initial)) * 1e-8


@pytest.mark.parametrize('name', ['fista', 'split_bregman'])
def test_backtracking_rejects_overshoot_for_small_signal_losses(name):
    target = jnp.asarray([1e-7, -2e-7], dtype=jnp.float32)
    def loss(params):
        return .5 * jnp.sum((params['level']-target)**2)
    params = {'level': jnp.zeros_like(target)}
    optimizer, step, _ = make_tv_steps(loss, loss, {'level':(-np.inf,np.inf)},
        {'level':()}, name, 256.)
    result, _, info = execute_optimizer(loss, params, optimizer, num_steps=1, step_function=step)
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
    # A fully sampled axis is diagonal in its image representation; no ROI is used.
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


@pytest.mark.parametrize("options", [{"tolerance": np.nan}, {"max_prox_steps": 1.5}, {"inner_steps": 0}, {"penalty": np.inf}])
def test_tv_solver_rejects_invalid_inner_options(options):
    with pytest.raises((ValueError, TypeError)):
        make_tv_steps(lambda p: p["level"] ** 2, lambda p: p["level"] ** 2,
                      {"level": (-2.0, 2.0)}, {}, "fista", 0.1, **options)


@pytest.mark.parametrize("name", ["fista", "split_bregman"])
def test_reconstruction_proximal_l1_matches_declared_mean_penalty(name):
    target = jnp.ones((2, 2))
    op = build_operator(spatial_shape=(2, 2), kspace=jnp.zeros((2, 2)))
    result = solve_variational(op.forward(target), spatial_shape=(2, 2),
                                     initial_image=jnp.zeros((2, 2)), regularizer="l1",
                                     regularizer_weight=0.1, optimizer_name=name,
                                     step_size=0.5, num_steps=300)
    np.testing.assert_allclose(result, 0.9, atol=2e-4)
