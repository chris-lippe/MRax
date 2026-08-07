import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import mrax.core.study_runner as study_runner
from mrax import __version__
from mrax.analysis.dti import dti_design_rank, fit_dti_log_linear
from mrax.analysis.roi import ROI, collect_roi_stats
from mrax.core.loss_builder import build_loss, make_model_callable
from mrax.core.study_runner import execute_study_plan, plan_study_fits
from mrax.core.variance import estimate_data_variance
from mrax.data_struct import (
    DataSet,
    Experiment,
    Model,
    Quantity,
    SpatialGeometry,
    Study,
    load_study_results,
    save_study_results,
)
from mrax.library import build_default_experiment, default_model_for_type, make_t2_se_model
from mrax.library.default_models import make_dti_tensor_model
from mrax.opt.regularizers import total_variation_penalty
from mrax.preprocessing import denoise_filter, regularized_denoise
from mrax.reco import fftnc, ifftnc, reconstruct_ifft
from mrax.versioning import MRaxVersionIncompatibilityError


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


def test_loss_combines_inverse_variance_weighting_with_rms_scaling():
    experiment, model, dataset = _constant_signal_problem()
    params = {"level": jnp.zeros((1, 1))}

    uniform_loss, _, _ = build_loss(experiment, model, dataset)
    scalar_variance_loss, _, _ = build_loss(
        experiment, model, dataset, data_variance=7.0
    )
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


@pytest.mark.parametrize("variance", ([1.0, 0.0], [1.0, np.nan], [1.0, 2.0, 3.0]))
def test_loss_rejects_invalid_data_variance(variance):
    experiment, model, dataset = _constant_signal_problem()

    with pytest.raises(ValueError, match="data_variance"):
        build_loss(experiment, model, dataset, data_variance=jnp.asarray(variance))


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


def test_auto_data_variance_uses_magnitude_background_when_available():
    rng = np.random.default_rng(17)
    sigma = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
    grid_x, grid_y = np.ogrid[:96, :96]
    tissue = (grid_x - 48) ** 2 + (grid_y - 48) ** 2 < 30**2
    frames = []
    for frame_sigma in sigma:
        real_noise = rng.normal(scale=frame_sigma, size=(96, 96))
        imag_noise = rng.normal(scale=frame_sigma, size=(96, 96))
        frame = np.sqrt(real_noise**2 + imag_noise**2)
        frame[tissue] = np.abs(50.0 + real_noise[tissue])
        frames.append(frame)
    data = np.asarray(frames, dtype=np.float32)
    experiment = Experiment(
        dims={"frame": 3, "x": 96, "y": 96},
        observables={"sample_index": Quantity(jnp.arange(3), "1")},
    )
    dataset = DataSet(data=jnp.asarray(data))

    estimate = estimate_data_variance(experiment, dataset, min_samples=128, shrinkage=0.0)
    estimated_relative = estimate.variance / estimate.variance[0]
    expected_relative = sigma**2 / sigma[0] ** 2

    assert estimate.method == "background_rayleigh"
    assert np.all(estimate.sample_count >= 128)
    assert np.allclose(estimated_relative, expected_relative, rtol=0.25)


def test_auto_data_variance_builds_same_loss_as_explicit_estimate():
    rng = np.random.default_rng(11)
    sigma = np.asarray([0.5, 1.5], dtype=np.float32)
    data = 2.0 + rng.normal(size=(2, 32, 32)).astype(np.float32) * sigma[:, None, None]
    experiment = Experiment(
        dims={"frame": 2, "x": 32, "y": 32},
        observables={"sample_index": Quantity(jnp.arange(2), "1")},
    )
    dataset = DataSet(data=jnp.asarray(data))

    def model_apply(fields, _offsets, _observables=None):
        return jnp.broadcast_to(fields["level"], experiment.shape)

    def parametrization_factory(spatial_shape):
        params = {"level": jnp.zeros(spatial_shape)}
        return params, lambda values: values

    model = Model(
        name="auto_variance_constant_signal",
        model_apply=model_apply,
        parametrization_factory=parametrization_factory,
        parameter_units={"level": "a.u."},
    )
    estimate = estimate_data_variance(experiment, dataset)
    auto_loss, params, unpack = build_loss(experiment, model, dataset, data_variance="auto")
    explicit_loss, _, _ = build_loss(
        experiment,
        model,
        dataset,
        data_variance=estimate.variance,
    )

    assert np.asarray(unpack._mrax_variance_estimate.variance).shape == (2,)
    assert float(auto_loss(params)) == pytest.approx(float(explicit_loss(params)), rel=1e-6)


def test_loss_rejects_unknown_data_variance_mode():
    experiment, model, dataset = _constant_signal_problem()

    with pytest.raises(ValueError, match="must be 'auto'"):
        build_loss(experiment, model, dataset, data_variance="estimate")


def test_fit_study_storage_and_roi_workflow(tmp_path):
    study, model = _small_t2_study()
    plan = plan_study_fits(study, target_names=["t2_demo"])

    results, field_cache = execute_study_plan(
        plan,
        run_fit_kwargs={"num_steps": 12, "step_size": 0.03},
    )

    assert results[0]["loss_history"][-1] < results[0]["loss_history"][0]
    assert {"amplitude", "t2", "bias"} <= set(field_cache["t2_demo"])
    assert results[0]["spatial_geometry"]["source"] == "synthetic"

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


def test_mrax_save_versions_json_geometry_but_not_mrd_payload(tmp_path):
    import ismrmrd

    study, model = _small_t2_study()
    manifest_path = save_study_results(
        study,
        [{"name": "t2_demo", "model": model.name, "fields": {}}],
        tmp_path / "versioned",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata_path = manifest_path.parent / manifest["experiments"]["t2_demo"]["metadata"]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert manifest["mrax_version"] == __version__
    assert metadata["mrax_version"] == __version__
    assert manifest["spatial_geometry"]["t2_demo"]["status"] == "known"
    assert metadata["spatial_geometry"]["source"] == "synthetic"

    image_path = manifest_path.parent / metadata["paths"]["mr_images"][0]["path"]
    dataset = ismrmrd.Dataset(str(image_path), "dataset", create_if_needed=False)
    image = dataset.read_image("image", 0)
    dataset.close()
    assert "spatial_geometry" not in str(image.attribute_string)

    major, minor, _patch = (int(value) for value in __version__.split(".")[:3])
    compatible = f"{major}.{minor}.999"
    manifest["mrax_version"] = compatible
    metadata["mrax_version"] = compatible
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    assert "t2_demo" in load_study_results(manifest_path)["experiments"]

    incompatible = f"0.{minor + 1}.0" if major == 0 else f"{major + 1}.0.0"
    manifest["mrax_version"] = incompatible
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(MRaxVersionIncompatibilityError, match="MRax version incompatibility"):
        load_study_results(manifest_path)

    manifest.pop("mrax_version")
    metadata.pop("mrax_version")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.warns(RuntimeWarning, match="no 'mrax_version' tag"):
        loaded = load_study_results(manifest_path)
    assert "t2_demo" in loaded["experiments"]


def test_execute_study_plan_preserves_dict_run_fit_kwargs(monkeypatch):
    study, _model = _small_t2_study()
    plan = plan_study_fits(study)
    seen = {}

    def fake_run_fit(experiment, model, dataset, **kwargs):
        seen.update(kwargs)
        return {"t2": Quantity(jnp.ones((2, 2)), "ms")}, [1.0]

    monkeypatch.setattr(study_runner, "run_fit", fake_run_fit)

    results, field_cache = study_runner.execute_study_plan(
        plan,
        run_fit_kwargs={
            "num_steps": {"t2_demo": 3},
            "regularizer_weights": {"t2": 0.1},
            "study_globals": {"field_strength_mhz": Quantity(jnp.asarray(400.0), "MHz")},
        },
    )

    assert seen["num_steps"] == 3
    assert seen["regularizer_weights"] == {"t2": 0.1}
    assert "field_strength_mhz" in seen["study_globals"]
    assert results[0]["loss_history"] == [1.0]
    assert "t2" in field_cache["t2_demo"]


def test_models_dimensions_reconstruction_and_preprocessing_workflow():
    assert default_model_for_type("mse").name == "t2_se"
    assert default_model_for_type("mge").name == "t2star_se"

    offsets = Quantity(jnp.asarray([1.0, 2.0, 1.0, 2.0], dtype=jnp.float32), "ppm")
    experiment = build_default_experiment(
        "sat_trans",
        frame_count=4,
        spatial_shape=(2, 2),
        observables={"freq_offsets": offsets},
    )
    assert experiment.dims["offset"] == 2
    assert experiment.dims["t"] == 2

    image = jnp.arange(16, dtype=jnp.float32).reshape((4, 4))
    kspace = fftnc(image)
    assert np.allclose(np.asarray(ifftnc(kspace)), np.asarray(image), atol=1e-5)
    assert np.allclose(np.asarray(reconstruct_ifft(kspace)), np.asarray(image), atol=1e-5)

    noisy = jnp.ones((5, 5), dtype=jnp.float32).at[2, 2].set(5.0)
    filtered = denoise_filter(noisy, method="gaussian", sigma=0.8)
    denoised, history = regularized_denoise(
        noisy,
        regularizer="tv",
        weight=0.01,
        num_steps=3,
        step_size=0.05,
        return_history=True,
    )
    assert float(total_variation_penalty(jnp.ones((4, 4), dtype=jnp.float32))) == pytest.approx(0.0)
    assert filtered.shape == noisy.shape
    assert denoised.shape == noisy.shape
    assert len(history) == 3
    assert np.isfinite(np.asarray(denoised)).all()


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
    data = make_model_callable(experiment, model)({key: jnp.asarray(value) for key, value in truth.items()})

    assert dti_design_rank(experiment.observables["bvals"], experiment.observables["bvecs"]) == 7
    fields = fit_dti_log_linear(data, experiment.observables["bvals"], experiment.observables["bvecs"])

    for key, value in truth.items():
        assert np.allclose(fields[key], value, atol=1e-6)
    assert float(np.nanmean(fields["fa"])) > 0.25
