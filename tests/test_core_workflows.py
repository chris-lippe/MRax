"""Synthetic core contracts and numerical references."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mrax.analysis.dti import dti_design_rank, fit_dti_log_linear
from mrax.analysis.roi import ROI, collect_roi_stats
from mrax.core.data_losses import pseudo_huber_loss
from mrax.core.loss_builder import build_loss, make_model_callable, run_fit
from mrax.core.study_runner import execute_study_plan, plan_study_fits
from mrax.core.variance import estimate_data_variance
from mrax.data_struct import (
    DataSet,
    Experiment,
    Model,
    Quantity,
    SpatialGeometry,
    Study,
    join_experiments,
    load_study_results,
    save_study_results,
)
from mrax.io.study_selection import auto_select_scans, pick_scan_by_type, scan_text
from mrax.library import build_default_experiment, default_model_for_type, make_t2_se_model
from mrax.library.default_models import (
    make_AB_MT_n_model,
    make_dki_kurtosis_model,
    make_dti_tensor_model,
    make_fair_asl_model,
)
from mrax.opt.regularizers import total_variation_penalty
from mrax.preprocessing import denoise_filter, regularized_denoise
from mrax.reco import unwrap_phase_image


def _small_t2_study():
    te = jnp.asarray([8.0, 24.0, 48.0, 96.0], dtype=jnp.float32)
    model = make_t2_se_model(scale=1.0)
    experiment = build_default_experiment(
        "mse",
        frame_count=int(te.size),
        spatial_shape=(2, 2),
        observables={"TE": Quantity(te, "ms")},
    )
    truth = {
        "amplitude": jnp.full((2, 2), 1.2, dtype=jnp.float32),
        "t2": jnp.full((2, 2), 70.0, dtype=jnp.float32),
        "bias": jnp.full((2, 2), 0.03, dtype=jnp.float32),
    }
    data = make_model_callable(experiment, model)(truth)
    dataset = DataSet(
        data=data,
        name="t2_demo",
        spatial_geometry=SpatialGeometry(
            affine=np.diag([0.4, 0.5, 1.0, 1.0]),
            spatial_shape=(2, 2),
            voxel_sizes=(0.4, 0.5),
            source="synthetic",
            status="known",
        ),
    )
    study = Study(
        [experiment],
        [dataset],
        [model],
        names=["t2_demo"],
        study_globals={"field_strength_mhz": Quantity(jnp.asarray(400.0), "MHz")},
    )
    return study, model


def _constant_signal_problem():
    experiment = Experiment(
        dims={"frame": 2, "x": 1, "y": 1},
        observables={"sample_index": Quantity(jnp.arange(2), "1")},
    )

    def model_apply(fields, _offsets, _observables=None):
        return jnp.broadcast_to(fields["level"], experiment.shape)

    def parametrization_factory(spatial_shape):
        params = {"level": jnp.zeros(spatial_shape)}
        return params, lambda values: values

    model = Model(
        name="constant_signal",
        model_apply=model_apply,
        parametrization_factory=parametrization_factory,
        parameter_units={"level": "a.u."},
    )
    dataset = DataSet(data=jnp.asarray([0.0, 2.0]).reshape(experiment.shape))
    return experiment, model, dataset


def test_spatial_denoising_preserves_shape_and_reduces_variation():
    noisy = jnp.ones((5, 5), dtype=jnp.float32).at[2, 2].set(5.0)
    filtered = denoise_filter(noisy, method="gaussian", sigma=0.8)
    denoised, history = regularized_denoise(
        noisy, regularizer="tv", weight=0.01, num_steps=3, step_size=0.05, return_history=True
    )
    assert float(total_variation_penalty(jnp.ones((4, 4), dtype=jnp.float32))) == pytest.approx(0.0)
    assert filtered.shape == noisy.shape
    assert denoised.shape == noisy.shape
    assert len(history) == 3
    assert np.isfinite(np.asarray(denoised)).all()
    assert float(total_variation_penalty(filtered)) < float(total_variation_penalty(noisy))


@pytest.mark.parametrize("kind", ["ir", "vfa", "mse", "fair_asl", "wasabi", "ab_mt_n"])
def test_builtin_observables_broadcast_over_volumetric_parameter_maps(kind):
    from mrax.library.default_models import (
        make_AB_MT_n_model,
        make_wasabi_model,
    )

    special = {
        "wasabi": make_wasabi_model,
        "ab_mt_n": make_AB_MT_n_model,
    }
    model = special[kind]() if kind in special else default_model_for_type(kind)
    if kind == "wasabi":
        assert model.derived_map_units == {"B1_map": "%", "B0_map": "ppm"}
    elif kind == "fair_asl":
        assert model.derived_map_units["perfusion_map"] == "mL/100g/min"
    observations = {
        "TI": Quantity(jnp.array([60.0, 220.0, 900.0]), "ms"),
        "TE": Quantity(jnp.array([9.0, 27.0, 45.0]), "ms"),
        "TR": Quantity(jnp.array([900.0, 1100.0, 1300.0]), "ms"),
        "flip_angle": Quantity(jnp.array([5.0, 12.0, 25.0]), "deg"),
        "fair_label": Quantity(jnp.array([0.0, 1.0, 0.0]), "1"),
        "freq_offsets": Quantity(jnp.array([-1.0, 0.2, 1.5]), "ppm"),
        "field_strength_mhz": Quantity(120.0, "MHz"),
        "sat_time": Quantity(jnp.array([3.0, 4.0, 5.0]), "ms"),
        "sat_b1": Quantity(jnp.array([3.2, 3.8, 4.2]), "uT"),
        "r1_map": Quantity(jnp.ones((2, 3, 4)), "1/s"),
        "z_prev": Quantity(jnp.array([0.9, 0.95, 1.0]), "1"),
    }
    params, unpack = model.parametrization_factory((2, 3, 4))
    fields = unpack(params)
    predicted = np.asarray(model.model_apply(fields, None, observations, _allow_direct=True))
    assert predicted.shape == (3, 2, 3, 4)
    assert np.isfinite(predicted).all()
    planar_fields = {key: value.reshape((6, 4)) for key, value in fields.items()}
    planar_observations = {**observations, "r1_map": Quantity(jnp.ones((6, 4)), "1/s")}
    planar = model.model_apply(planar_fields, None, planar_observations, _allow_direct=True)
    np.testing.assert_allclose(predicted.reshape((3, 6, 4)), planar, rtol=1e-6, atol=1e-7)
    if kind == "wasabi":
        offsets = np.linspace(-4, 4, 81, dtype=np.float32)
        detuning = offsets * 400 * 2 * np.pi
        rf = 2 * np.pi * 42.58 * 4
        spectrum = np.abs(1.55 - 2 * rf**2 / (rf**2 + detuning**2)
                          * np.sin(np.sqrt(rf**2 + detuning**2) * 0.005 / 2)**2) + 1e-6
        dynamic = Experiment(
            dims={"t": 60, "b1": 1, "offset": 81, "x": 4, "y": 3}, exp_type="sat_trans",
            observables={"freq_offsets": Quantity(offsets, "ppm"), "sat_time": Quantity(5, "ms"),
                         "sat_b1": Quantity(4, "uT"), "field_strength_mhz": Quantity(400, "MHz")},
        )
        model.parameter_constant_dims = {"c": ("t",)}
        fitted, history = run_fit(
            dynamic, model, DataSet(data=jnp.broadcast_to(jnp.asarray(spectrum)[None, None, :, None, None], dynamic.shape)),
            num_steps=2, step_size=0.02, data_normalization="full",
            regularizer_configs={"b1": [
                {"type": "tv", "weight": 0.001, "axes": [1, 2]},
                {"type": "smoothness", "weight": 0.01, "axes": [0]},
            ]},
        )
        np.testing.assert_allclose(fitted["b1"].values, 4, atol=1e-3)
        assert np.isfinite(history).all() and max(history) < 1e-10
    if kind == "ab_mt_n":
        from mrax.analysis.saturation_transfer import experiment_with_ab_mt_z_prev

        experiment = Experiment(dims={"slice_package": 2, "t": 2, "offset": 3, "x": 1, "y": 1}, observables={})
        measured = jnp.arange(12, dtype=jnp.float32).reshape(experiment.shape)
        prepared = experiment_with_ab_mt_z_prev(experiment, measured)
        np.testing.assert_array_equal(
            prepared.observables["z_prev"].values.reshape(2, 6),
            [[0, 0, 1, 2, 3, 4], [6, 6, 7, 8, 9, 10]],
        )


@pytest.mark.parametrize(
    "order,storage",
    [(("ti", "fair"), "axis"), (("fair", "ti"), "grid"), (("ti", "fair"), "frame")],
)
def test_model_callable_expands_independent_observable_axes(order, storage):
    ti = np.asarray([80.0, 360.0, 1500.0], dtype=np.float32)
    labels = np.asarray([0.0, 1.0], dtype=np.float32)
    sizes = {"ti": len(ti), "fair": len(labels)}
    dims = {name: sizes[name] for name in order}
    dims.update(x=2, y=4)
    grids = np.meshgrid(*(ti if name == "ti" else labels for name in order), indexing="ij")
    ti_grid, label_grid = (grids[order.index(name)] for name in ("ti", "fair"))
    ti_obs, label_obs = (ti, labels) if storage == "axis" else (ti_grid, label_grid)
    if storage == "frame":
        ti_obs, label_obs = ti_obs.reshape(-1), label_obs.reshape(-1)
    experiment = Experiment(
        dims=dims,
        exp_type="fair_asl",
        observables={
            "TI": Quantity(ti_obs, "ms"),
            "fair_label": Quantity(label_obs, "1"),
        },
    )
    model = make_fair_asl_model()
    fields = {
        "amplitude_sel": jnp.full((2, 4), 1.2),
        "amplitude_non": jnp.full((2, 4), 0.9),
        "t1_sel": jnp.full((2, 4), 850.0),
        "t1_non": jnp.full((2, 4), 1400.0),
    }
    amp = np.where(label_grid >= 0.5, 0.9, 1.2)
    t1 = np.where(label_grid >= 0.5, 1400.0, 850.0)
    signal = np.abs(amp - 2 * amp * np.exp(-ti_grid / t1)) + 1e-6
    expected = np.broadcast_to(signal.reshape((-1, 1, 1)), (len(ti) * 2, 2, 4))
    predict = make_model_callable(experiment, model)
    np.testing.assert_allclose(predict(fields), expected, rtol=2e-6, atol=2e-7)
    gradient = jax.grad(lambda values: jnp.sum(predict(values)))(fields)
    assert all(np.isfinite(value).all() for value in gradient.values())
    if storage == "axis":
        repeated = Experiment(dims={"slice_package": 2, **dims}, exp_type="fair_asl",
                              observables=experiment.observables)
        repeated_data = np.tile(expected, (2, 1, 1))
        loss, params, _ = build_loss(repeated, model, DataSet(data=jnp.asarray(repeated_data.reshape(repeated.shape))))
        assert np.isfinite(float(loss(params)))
        assert all(np.isfinite(value).all() for value in jax.grad(loss)(params).values())


def test_loss_combines_inverse_variance_weighting_with_rms_scaling():
    experiment, model, dataset = _constant_signal_problem()
    params = {"level": jnp.zeros((1, 1))}

    uniform_loss, _, _ = build_loss(experiment, model, dataset)
    scalar_variance_loss, _, _ = build_loss(experiment, model, dataset, data_variance=7.0)
    weighted_loss, _, _ = build_loss(
        experiment, model, dataset, data_variance=jnp.asarray([1.0, 4.0])
    )
    rescaled_variance_loss, _, _ = build_loss(
        experiment, model, dataset, data_variance=jnp.asarray([10.0, 40.0])
    )

    assert float(uniform_loss(params)) == pytest.approx(1.0)
    assert float(scalar_variance_loss(params)) == pytest.approx(1.0)
    assert float(weighted_loss(params)) == pytest.approx(0.4)
    assert float(rescaled_variance_loss(params)) == pytest.approx(0.4)

    optimum = {"level": jnp.full((1, 1), 0.4)}
    gradient = jax.grad(weighted_loss)(optimum)["level"]
    assert np.allclose(np.asarray(gradient), 0.0, atol=1e-6)


def test_loss_rejects_nonpositive_data_variance():
    experiment, model, dataset = _constant_signal_problem()

    with pytest.raises(ValueError, match="data_variance"):
        build_loss(experiment, model, dataset, data_variance=jnp.asarray([1.0, 0.0]))


def test_auto_data_variance_estimates_heteroscedastic_frame_noise():
    rng = np.random.default_rng(7)
    sigma = np.asarray([0.5, 1.0, 2.0], dtype=np.float32)
    data = 4.0 + rng.normal(size=(3, 96, 96)).astype(np.float32) * sigma[:, None, None]
    experiment = Experiment(
        dims={"frame": 3, "x": 96, "y": 96},
        observables={"sample_index": Quantity(jnp.arange(3), "1")},
    )
    dataset = DataSet(data=jnp.asarray(data))

    estimate = estimate_data_variance(experiment, dataset, shrinkage=0.0)
    estimated_relative = estimate.relative_variance / estimate.relative_variance[0]
    expected_relative = sigma**2 / sigma[0] ** 2

    assert estimate.method == "wavelet_mad"
    assert np.all(estimate.sample_count > 2000)
    assert np.allclose(estimated_relative, expected_relative, rtol=0.15)


def test_fit_study_storage_and_roi_workflow(tmp_path):
    study, model = _small_t2_study()
    model.derived_maps = lambda fields: {"offset_map": Quantity(fields["t2"].values + 7, "ms")}
    mask = np.asarray([[True, False], [True, False]])
    plan = plan_study_fits(study, target_names=["t2_demo"])

    results, field_cache = execute_study_plan(
        plan,
        run_fit_kwargs={"num_steps": 12, "step_size": 0.03, "mask": mask, "compute_derived_maps": True},
    )

    assert results[0]["loss_history"][-1] < results[0]["loss_history"][0]
    assert {"amplitude", "t2", "bias"} <= set(field_cache["t2_demo"])
    assert results[0]["spatial_geometry"]["source"] == "synthetic"
    derived = results[0]["fields"]["offset_map"]
    assert derived.si_unit == "ms"
    np.testing.assert_array_equal(derived.values[~mask], 0)
    np.testing.assert_allclose(derived.values[mask], field_cache["t2_demo"]["t2"].values[mask] + 7)

    manifest = save_study_results(
        study,
        [
            {
                "name": "t2_demo",
                "model": model.name,
                "fields": {"t2": Quantity(field_cache["t2_demo"]["t2"], "ms")},
            }
        ],
        tmp_path / "study",
        study_name="workflow",
    )
    loaded = load_study_results(manifest)
    loaded_map = loaded["experiments"]["t2_demo"]["fitted_maps"]["t2"]
    rows = collect_roi_stats(
        {"t2": loaded_map},
        [ROI("all", np.ones((2, 2), dtype=bool), dims=("x", "y"))],
        stats=("mean", "count"),
    )

    assert loaded["manifest"]["study_name"] == "workflow"
    assert loaded["experiments"]["t2_demo"]["spatial_geometry"]["status"] == "known"
    assert loaded_map.si_unit == "ms"
    assert next(row for row in rows if row["stat"] == "count")["value"] == 4
    assert np.isfinite(next(row for row in rows if row["stat"] == "mean")["value"])


def test_phase_unwrapping_is_spatial_and_independent_per_frame():
    x = np.linspace(0.0, 5.0 * np.pi, 24, dtype=np.float32)
    y = np.linspace(0.0, 1.5 * np.pi, 16, dtype=np.float32)
    truth = x[:, None] + y[None, :]
    wrapped = np.angle(np.exp(1j * truth)).astype(np.float32)
    shifted_truth = truth + 0.4
    shifted_wrapped = np.angle(np.exp(1j * shifted_truth)).astype(np.float32)
    frames = np.stack([wrapped, shifted_wrapped], axis=0)

    unwrapped = jax.jit(lambda value: unwrap_phase_image(value, spatial_ndim=2))(frames)

    assert unwrapped.shape == frames.shape
    for frame, expected in zip(unwrapped, (truth, shifted_truth), strict=True):
        offset = float(frame[0, 0] - expected[0, 0])
        np.testing.assert_allclose(frame - offset, expected, atol=1e-4)


def test_dti_log_linear_workflow_recovers_tensor_maps():
    bvals = np.asarray([0.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0], dtype=float)
    bvecs = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=float,
    )
    bvecs[4:] /= np.linalg.norm(bvecs[4:], axis=1, keepdims=True)
    truth = {
        "s0": np.full((2, 2), 900.0, dtype=float),
        "dxx": np.full((2, 2), 0.0015, dtype=float),
        "dyy": np.full((2, 2), 0.0009, dtype=float),
        "dzz": np.full((2, 2), 0.0006, dtype=float),
        "dxy": np.full((2, 2), 0.0001, dtype=float),
        "dxz": np.full((2, 2), -0.00005, dtype=float),
        "dyz": np.full((2, 2), 0.00008, dtype=float),
    }
    model = make_dti_tensor_model(scale=1000.0)
    experiment = build_default_experiment(
        "dti",
        frame_count=int(bvals.size),
        spatial_shape=(2, 2),
        observables={
            "bvals": Quantity(jnp.asarray(bvals), "s/mm^2"),
            "bvecs": Quantity(jnp.asarray(bvecs), "1"),
        },
    )
    data = make_model_callable(experiment, model)(
        {key: jnp.asarray(value) for key, value in truth.items()}
    )

    assert dti_design_rank(experiment.observables["bvals"], experiment.observables["bvecs"]) == 7
    fields = fit_dti_log_linear(
        data, experiment.observables["bvals"], experiment.observables["bvecs"]
    )

    for key, value in truth.items():
        assert np.allclose(fields[key], value, atol=1e-6)
    assert float(np.nanmean(fields["fa"])) > 0.25
    for tensor_model in (model, make_dki_kurtosis_model(scale=1000.0)):
        repeated = Experiment(
            dims={"slice_package": 2, "frame": len(bvals), "x": 2, "y": 2},
            exp_type=tensor_model.name,
            observables={"bvals": Quantity(np.tile(bvals, 2), "s/mm^2"),
                         "bvecs": Quantity(np.tile(bvecs, (2, 1)), "1")},
        )
        repeated_fields = {name: jnp.stack([value, value * 1.1]) for name, value in truth.items()}
        if tensor_model.name == "dki":
            repeated_fields["kurtosis"] = jnp.full((2, 2, 2), 0.5)
        predicted = np.asarray(make_model_callable(repeated, tensor_model)(repeated_fields))
        reference = []
        for group in range(2):
            group_fields = {name: np.asarray(value[group]) for name, value in repeated_fields.items()}
            tensor = np.stack([
                np.stack([group_fields["dxx"], group_fields["dxy"], group_fields["dxz"]], -1),
                np.stack([group_fields["dxy"], group_fields["dyy"], group_fields["dyz"]], -1),
                np.stack([group_fields["dxz"], group_fields["dyz"], group_fields["dzz"]], -1),
            ], -2)
            apparent = np.einsum("ni,...ij,nj->n...", bvecs, tensor, bvecs)
            exponent = -bvals[:, None, None] * apparent
            if tensor_model.name == "dki":
                exponent += bvals[:, None, None] ** 2 * apparent ** 2 * 0.5 / 6
            reference.append(group_fields["s0"] * np.exp(exponent) + 1e-6)
        np.testing.assert_allclose(predicted, np.concatenate(reference), rtol=2e-6, atol=2e-6)
        if tensor_model.name == "dti":
            from mrax.gui import GuiSession

            session = GuiSession()
            session.add_experiment("Tensor groups", predicted.reshape(repeated.shape),
                                   dims=repeated.dims, exp_type="dti", model_name="dti",
                                   observables=repeated.observables)
            fitted = session.run_current_fit(use_roi=False, num_steps=2)
            assert np.shape(fitted["fields"]["dxx"].values) == (2, 2, 2)
            np.testing.assert_allclose(fitted["fields"]["dxx"].values, repeated_fields["dxx"], rtol=0.03)
            session.add_experiment("Diffusion reference", predicted[:1],
                                   dims={"frame": 1, "x": 2, "y": 2}, exp_type="dti", model_name="dti",
                                   observables={"bvals": Quantity(bvals[:1], "s/mm^2"),
                                                "bvecs": Quantity(bvecs[:1], "1")})
            assert not session.run_fit_sanity_check()["ok"]
            with pytest.raises(ValueError, match="more than one diffusion volume"):
                session.run_preview_fit(num_steps=1)
            with pytest.raises(ValueError, match="more than one diffusion volume"):
                session.run_current_fit(num_steps=1)


def test_joint_experiment_and_scan_selection_workflow():
    exp_a = Experiment(
        {"echo": 2, "x": 2, "y": 2},
        {"TE": Quantity(jnp.asarray([10.0, 20.0]), "ms")},
        "mse",
    )
    exp_b = Experiment(
        {"echo": 1, "x": 2, "y": 2},
        {"TE": Quantity(jnp.asarray([30.0]), "ms")},
        "mse",
    )
    joined = join_experiments(
        [exp_a, exp_b],
        [DataSet(jnp.ones((2, 2, 2)), "a"), DataSet(jnp.full((1, 2, 2), 2.0), "b")],
        names=["a", "b"],
    )
    selected_exp, selected_ds = joined.select("b")
    assert joined.experiment.frame_count == 3
    assert selected_exp.frame_count == 1
    np.testing.assert_allclose(np.asarray(selected_ds.data), 2.0)

    metadata = {
        "experiments": {
            "1": {"exp_type": "mse", "method": {"Method": "MSME"}},
            "2": {"exp_type": "dce", "method": {"ProtocolName": "contrast"}},
        }
    }
    study = SimpleNamespace(
        names=["1", "2"], experiments=[exp_a, Experiment({"t": 2, "x": 2, "y": 2}, exp_type="dce")]
    )
    assert pick_scan_by_type(study, "mse") == "1"
    fair, t2, wasabi, saturation = auto_select_scans(study, metadata)
    assert (fair, t2, wasabi, saturation) == (None, "1", None, [])
    assert "msme" in scan_text(metadata["experiments"]["1"])


def test_group_pseudo_huber_bounds_spectral_influence_and_ignores_missing():
    weights = jnp.ones((8, 2))
    baseline = jnp.zeros((8, 2))

    def f(r):
        return pseudo_huber_loss(r, weights, delta=0.1, group_size=4)

    gradient = jax.jit(jax.grad(f))
    norm = []
    for amplitude in (1.0, 100.0, 10000.0):
        r = baseline.at[:4, 0].set(amplitude)
        norm.append(float(jnp.linalg.norm(gradient(r)[:4, 0])))
    assert max(norm) / min(norm) < 1.01
    r = jnp.asarray([[0.01, jnp.nan], [0.02, jnp.inf]])
    w = jnp.asarray([[1.0, 0.0], [1.0, 0.0]])
    assert np.isfinite(float(pseudo_huber_loss(r, w, delta=0.1, group_size=2)))


@pytest.mark.parametrize(
    "spin_lock,expected,expected_gradient",
    [
        (
            False,
            [0.9581428766, 0.1448954195, 0.3239752054],
            [-1.3465567827, -1.0522072315, -1.1726071835],
        ),
        (
            True,
            [1.0181690454, 0.5023628473, 0.3643416464],
            [-1.4368637800, -4.2174105644, -1.3498690128],
        ),
    ],
)
def test_ab_mt_signal_and_gradient_reference(spin_lock, expected, expected_gradient):
    # Reference signal and derivatives for independent synthetic inputs.
    model = make_AB_MT_n_model(pools=[])
    params, unpack = model.parametrization_factory((1, 1))
    fields = {
        **unpack(params),
        "a": jnp.full((1, 1), 0.01),
        "b": jnp.full((1, 1), 0.04),
        "sin_a_eff": jnp.full((1, 1), 0.8),
        "t_delay": jnp.full((1, 1), 20.0),
    }
    observations = {
        "freq_offsets": Quantity(jnp.array([-2.0, 0.5, 3.0]), "ppm"),
        "field_strength_mhz": Quantity(120.0, "MHz"),
        "sat_time": Quantity(jnp.array([200.0, 100.0, 300.0]), "ms"),
        "sat_b1": Quantity(jnp.array([1.5, 2.5, 3.5]), "uT"),
        "TR": Quantity(jnp.array([900.0, 1000.0, 1100.0]), "ms"),
        "r1_map": Quantity(jnp.ones((1, 1)), "1/s"),
        "spin_lock": spin_lock,
    }

    def signal(b):
        return model.model_apply({**fields, "b": b}, None, observations, _allow_direct=True)

    np.testing.assert_allclose(signal(fields["b"]).reshape(-1), expected, rtol=3e-6)
    np.testing.assert_allclose(
        jax.jacrev(signal)(fields["b"]).reshape(-1), expected_gradient, rtol=3e-6
    )
