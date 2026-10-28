"""Synthetic preprocessing contracts and numerical references."""


import jax.numpy as jnp
import numpy as np

from mrax.analysis.dce import analyze_dce_series, replace_isolated_spikes
from mrax.analysis.group import compare_group_stats, summarize_group_stats
from mrax.analysis.stats import describe_values, fdr_bh, nonparametric_test, ttest
from mrax.analysis.timecourse_processing import (
    estimate_timecourse_baseline,
    mean_timecourses_by_label,
)
from mrax.data_struct import DataSet, Experiment
from mrax.preprocessing.drift import (
    linear_drift_correct_dataset,
    linear_drift_correct_framestack,
    negative_mean_squared_error,
)


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


def test_linear_drift_corrects_synthetic_additive_framestack():
    times = np.linspace(0.0, 1.0, 6)
    frames = np.stack(
        [np.full((4, 4), -2.5 * time, dtype=np.float32) for time in times],
        axis=0,
    )

    result = linear_drift_correct_framestack(
        frames,
        transform=AdditiveTransform(),
        metric=negative_mean_squared_error,
        times=times,
        max_parameter_change=4.0,
        optimizer_name="sgd",
        step_size=0.6,
        num_steps=40,
    )

    assert result.diagnostics.success
    assert result.parameters.shape == (6, 1)
    assert np.allclose(result.parameters[:, 0], 2.5 * times, atol=1e-4)
    assert np.allclose(result.corrected, 0.0, atol=1e-4)
    assert result.similarities_after[-1] > result.similarities_before[-1]


def test_linear_drift_dataset_workflow_preserves_experiment_shape():
    experiment = Experiment(dims={"t": 5, "x": 2, "y": 2}, exp_type="dce")
    times = np.linspace(0.0, 1.0, 5)
    data = np.stack(
        [np.full((2, 2), -1.5 * time, dtype=np.float32) for time in times],
        axis=0,
    )

    result = linear_drift_correct_dataset(
        experiment,
        DataSet(jnp.asarray(data), name="drift_demo"),
        dimensions=("t",),
        transform=AdditiveTransform(),
        metric=negative_mean_squared_error,
        times=times,
        max_parameter_change=3.0,
        optimizer_name="sgd",
        step_size=0.6,
        num_steps=40,
    )

    assert result.dataset.shape == experiment.shape
    assert result.dimensions == ("t",)
    assert result.group_dimensions == ()
    assert np.allclose(np.asarray(result.dataset.data), 0.0, atol=1e-4)


def test_tail_theil_sen_uses_late_baseline_trend():
    times = np.array([50.0, 100.0, 150.0, 200.0, 250.0])
    values = np.array([20.0, 14.0, 8.0, 7.0, 6.0])

    baseline = estimate_timecourse_baseline(
        values,
        baseline_frames=5,
        method="tail_theil_sen",
        times=times,
        reference_time=300.0,
        tail_frames=3,
    )

    assert np.isclose(baseline, 5.0)


def test_mean_timecourses_supports_robust_timed_baseline():
    times = np.array([50.0, 100.0, 150.0, 200.0, 250.0, 350.0])
    series = np.array([20.0, 14.0, 8.0, 7.0, 6.0, 9.0])[:, None, None]
    mask = np.ones((1, 1), dtype=int)

    timecourses, labels = mean_timecourses_by_label(
        series,
        mask,
        baseline_frames=5,
        baseline_method="tail_theil_sen",
        baseline_times=times,
        baseline_reference_time=300.0,
        baseline_tail_frames=3,
        smooth_sigma=0.0,
    )

    assert labels == [1]
    assert np.isclose(timecourses[-1, 0], 4.0)


def test_mean_baseline_remains_default():
    series = np.array([1.0, 3.0, 5.0])[:, None, None]
    mask = np.ones((1, 1), dtype=int)

    timecourses, _ = mean_timecourses_by_label(
        series,
        mask,
        baseline_frames=2,
        smooth_sigma=0.0,
    )

    np.testing.assert_allclose(timecourses[:, 0], [-1.0, 1.0, 3.0])


class AdditiveTransform:
    def identity_parameters(self, image):
        return jnp.zeros(1, dtype=jnp.float32)

    def apply(self, image, parameters):
        return jnp.asarray(image) + jnp.asarray(parameters)[0]
