import jax.numpy as jnp
import numpy as np
import pytest

import mrax.core.study_runner as study_runner
from mrax.analysis.dti import dti_design_rank, fit_dti_log_linear
from mrax.analysis.roi import ROI, collect_roi_stats
from mrax.core.loss_builder import make_model_callable
from mrax.core.study_runner import execute_study_plan, plan_study_fits
from mrax.data_struct import DataSet, Quantity, Study, load_study_results, save_study_results
from mrax.library import build_default_experiment, default_model_for_type, make_t2_se_model
from mrax.library.default_models import make_dti_tensor_model
from mrax.opt.regularizers import total_variation_penalty
from mrax.preprocessing import denoise_filter, regularized_denoise
from mrax.reco import fftnc, ifftnc, reconstruct_ifft


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
    dataset = DataSet(data=data, name="t2_demo")
    study = Study(
        [experiment],
        [dataset],
        [model],
        names=["t2_demo"],
        study_globals={"field_strength_mhz": Quantity(jnp.asarray(400.0), "MHz")},
    )
    return study, model


def test_fit_study_storage_and_roi_workflow(tmp_path):
    study, model = _small_t2_study()
    plan = plan_study_fits(study, target_names=["t2_demo"])

    results, field_cache = execute_study_plan(
        plan,
        run_fit_kwargs={"num_steps": 12, "step_size": 0.03},
    )

    assert results[0]["loss_history"][-1] < results[0]["loss_history"][0]
    assert {"amplitude", "t2", "bias"} <= set(field_cache["t2_demo"])

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
    assert loaded_map.si_unit == "ms"
    assert next(row for row in rows if row["stat"] == "count")["value"] == 4
    assert np.isfinite(next(row for row in rows if row["stat"] == "mean")["value"])


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
