import numpy as np

from mrax.analysis.timecourse_processing import (
    estimate_timecourse_baseline,
    mean_timecourses_by_label,
)


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
