
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from mrax.analysis.dce import analyze_dce_series, replace_isolated_spikes
from mrax.analysis.group import compare_group_stats, summarize_group_stats
from mrax.analysis.stats import describe_values, fdr_bh, nonparametric_test, ttest
from mrax.data_struct import DataSet, Experiment, Quantity, join_experiments
from mrax.io.nifti import export_nifti_4d, export_nifti_series, load_nifti, save_nifti
from mrax.io.study_selection import auto_select_scans, pick_scan_by_type, scan_text
from mrax.reco.pipeline import FFTStage, POCSStage, RecoPipeline, ZeroFillStage
from mrax.reco.processing import (
    apply_channel_scale_phase,
    apply_readout_phase_shift,
    apply_samplewise_correction,
    crop_along_axis,
    estimate_radial_density,
    fid_profiles_to_kspace,
    readout_crop_bounds,
)
from mrax.reco.variational import reconstruct_ifft, reconstruct_variational


def test_nifti_single_volume_and_series_round_trip(tmp_path):
    image = np.arange(12, dtype=np.float32).reshape(3, 4)
    affine = np.diag([1.5, 2.0, 1.0, 1.0])
    path = tmp_path / "image.nii.gz"
    save_nifti(path, image, affine)
    loaded, loaded_affine = load_nifti(path)
    np.testing.assert_allclose(loaded, image)
    np.testing.assert_allclose(loaded_affine, affine)

    experiment = Experiment(
        dims={"echo": 2, "x": 3, "y": 4},
        observables={"TE": Quantity(jnp.asarray([10.0, 20.0]), "ms")},
        exp_type="mse",
    )
    dataset = DataSet(jnp.stack([jnp.asarray(image), jnp.asarray(image + 1)]), name="nifti")
    outputs = export_nifti_series(experiment, dataset, tmp_path / "series", affine=affine)
    output_4d = export_nifti_4d(experiment, dataset, tmp_path / "series_4d.nii.gz", affine=affine)

    assert len(outputs) == 2
    assert all(path.is_file() for path in outputs)
    assert output_4d.is_file()


def test_reconstruction_pipeline_and_processing_helpers():
    image = np.zeros((8, 8), dtype=np.float32)
    image[2:6, 3:5] = 1.0
    kspace = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(image), norm="ortho"))
    mask = np.zeros_like(image, dtype=bool)
    mask[2:, :] = True
    measured = kspace * mask

    state = RecoPipeline(
        [ZeroFillStage(target_shape=(12, 12)), POCSStage(iterations=2), FFTStage("inverse")]
    ).run(measured, metadata={"mask": mask})
    copied = state.copy()
    assert state.image.shape == (12, 12)
    assert copied.metadata == state.metadata
    assert copied.kspace is not state.kspace

    channels = np.ones((2, 3, 4), dtype=np.complex64)
    corrected = apply_channel_scale_phase(
        channels, scale=[1.0, 2.0, 3.0], phase_degrees=[0.0, 90.0, 180.0]
    )
    np.testing.assert_allclose(np.abs(corrected[:, :, 0]), [[1.0, 2.0, 3.0]] * 2)
    np.testing.assert_allclose(
        apply_samplewise_correction(channels, [1.0, 2.0, 3.0, 4.0])[0, 0],
        [1.0, 2.0, 3.0, 4.0],
    )
    shifted = apply_readout_phase_shift(channels, shift_samples=0.5)
    assert shifted.shape == channels.shape
    bounds = readout_crop_bounds(8, output_size=4)
    assert bounds == (2, 6)
    assert crop_along_axis(np.arange(8), output_size=4).tolist() == [2, 3, 4, 5]
    assert fid_profiles_to_kspace(np.ones((2, 8)), transform="fft").shape == (2, 8)
    density = estimate_radial_density(np.asarray([[0.0, 0.0], [0.25, 0.0], [0.0, 0.5]]))
    assert density.shape == (3,)
    assert np.isfinite(density).all()


def test_cartesian_variational_reconstruction_smoke():
    image = jnp.zeros((8, 8)).at[2:6, 2:6].set(1.0)
    kspace = jnp.fft.fftshift(
        jnp.fft.fftn(jnp.fft.ifftshift(image), norm="ortho")
    )
    baseline = reconstruct_ifft(kspace)
    reconstructed, history = reconstruct_variational(
        kspace,
        mask=jnp.ones_like(kspace, dtype=bool),
        num_steps=2,
        step_size=0.01,
        regularizer="l2",
        regularizer_weight=0.0,
        return_history=True,
    )
    np.testing.assert_allclose(np.asarray(baseline), np.asarray(image), atol=1e-5)
    assert reconstructed.shape == image.shape
    assert len(history) == 2


def test_dce_group_and_statistics_workflow():
    series = np.ones((6, 2, 2), dtype=float)
    series[3] = 4.0
    series[4:] = 1.2
    cleaned = replace_isolated_spikes(series, threshold=1.5)
    np.testing.assert_allclose(cleaned[3], 1.1)

    result = analyze_dce_series(
        series,
        np.full((2, 2), 1200.0),
        label_mask=np.ones((2, 2), dtype=int),
        frame_time_ms=1000.0,
        baseline_frames=3,
        min_baseline_frames=1,
        smooth_sigma_frames=0.0,
        spike_threshold=10.0,
    )
    assert result.concentration_mM.shape == series.shape
    assert result.labels == [1]
    assert result.timecourses_mM.shape == (6, 1)

    rows = [
        {"group": "a", "experiment": "dce", "map": "auc", "roi": "all", "stat": "mean", "value": 1.0},
        {"group": "a", "experiment": "dce", "map": "auc", "roi": "all", "stat": "mean", "value": 2.0},
        {"group": "b", "experiment": "dce", "map": "auc", "roi": "all", "stat": "mean", "value": 4.0},
        {"group": "b", "experiment": "dce", "map": "auc", "roi": "all", "stat": "mean", "value": 5.0},
    ]
    summaries = summarize_group_stats(rows)
    comparisons = compare_group_stats(rows, p_adjust="fdr_bh")
    assert len(summaries) == 2
    assert len(comparisons) == 1
    assert "t_p_fdr_bh" in comparisons[0]
    assert describe_values([1.0, 2.0, np.nan])["count"] == 2.0
    assert len(fdr_bh([0.01, 0.04, 0.03])) == 3
    assert np.isfinite(ttest([1, 2, 3], [4, 5, 6])[1])
    assert nonparametric_test([1, 2], [3, 4])[2] == "mannwhitney"


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
    study = SimpleNamespace(names=["1", "2"], experiments=[exp_a, Experiment({"t": 2, "x": 2, "y": 2}, exp_type="dce")])
    assert pick_scan_by_type(study, "mse") == "1"
    fair, t2, wasabi, saturation = auto_select_scans(study, metadata)
    assert (fair, t2, wasabi, saturation) == (None, "1", None, [])
    assert "msme" in scan_text(metadata["experiments"]["1"])
