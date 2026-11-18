"""Synthetic preprocessing contracts and numerical references."""

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.ndimage import shift

from mrax.analysis.dce import analyze_dce_series, replace_isolated_spikes
from mrax.analysis.group import compare_group_stats, summarize_group_stats
from mrax.analysis.stats import describe_values, fdr_bh, nonparametric_test, ttest
from mrax.analysis.timecourse_processing import (
    mean_timecourses_by_label,
)
from mrax.data_struct import DataSet, Experiment
from mrax.preprocessing.drift import (
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
        {
            "group": "a",
            "experiment": "dce",
            "map": "auc",
            "roi": "all",
            "stat": "mean",
            "value": 1.0,
        },
        {
            "group": "a",
            "experiment": "dce",
            "map": "auc",
            "roi": "all",
            "stat": "mean",
            "value": 2.0,
        },
        {
            "group": "b",
            "experiment": "dce",
            "map": "auc",
            "roi": "all",
            "stat": "mean",
            "value": 4.0,
        },
        {
            "group": "b",
            "experiment": "dce",
            "map": "auc",
            "roi": "all",
            "stat": "mean",
            "value": 5.0,
        },
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


def test_registration_complex_values_volume_similarity_and_contrast_templates():
    sitk = pytest.importorskip("SimpleITK")
    from mrax.data_struct import SpatialGeometry
    from mrax.gui.registration import (
        RegistrationOptions,
        registration_method_factory,
        registration_presets,
        run_registration,
    )
    from mrax.preprocessing.motion import motion_correct_dataset, register_image

    x, y = np.indices((28, 32))
    image = (
        np.exp(-((x - 11) ** 2 / 17 + (y - 17) ** 2 / 29))
        + 0.4 * np.exp(-((x - 20) ** 2 / 8 + (y - 8) ** 2 / 11))
    ).astype(np.float32) * 1e5
    moving = shift(image, (2, -1), order=1) * (1 + 2j)
    options = RegistrationOptions(transform="translation", metric="mean_squares", iterations=120)
    result = run_registration(image * (1 + 2j), moving, options)
    np.testing.assert_allclose(
        result.corrected.imag, result.corrected.real * 2, rtol=2e-5, atol=0.05
    )
    assert result.corrected.real.max() > 50000
    assert (
        np.mean(abs(result.corrected - image * (1 + 2j)) ** 2)
        < np.mean(abs(moving - image * (1 + 2j)) ** 2) / 5
    )
    volume = np.stack([image * (1 + 0.1 * np.cos(index)) for index in range(8)], axis=-1)
    result = run_registration(
        volume, volume, RegistrationOptions(transform="similarity", metric="mean_squares")
    )
    np.testing.assert_allclose(result.corrected, volume, rtol=1e-5, atol=0.1)
    observed_spacing = []

    def spaced_factory(fixed, moving):
        observed_spacing.append(fixed.GetSpacing())
        return registration_method_factory(options)(fixed, moving)

    motion_correct_dataset(
        Experiment(dims={"frame": 2, "x": 28, "y": 32, "z": 8}, exp_type="anatomical"),
        DataSet(
            data=np.stack([volume, volume]),
            spatial_geometry=SpatialGeometry(
                spatial_shape=volume.shape, voxel_sizes=(0.8, 1.3, 2.1)
            ),
        ),
        registration_method=spaced_factory,
    )
    assert observed_spacing == [(2.1, 1.3, 0.8)]
    with pytest.raises(ValueError, match="mask"):
        run_registration(image, image, options, fixed_mask=np.zeros_like(image, dtype=bool))
    bad = sitk.ImageRegistrationMethod()
    bad.SetMetricAsMeanSquares()
    bad.SetOptimizerAsGradientDescent(learningRate=0, numberOfIterations=0)
    bad.SetInitialTransform(sitk.TranslationTransform(2, [8.0, 0.0]))
    with pytest.raises(ValueError, match="alignment check"):
        register_image(image, image, registration_method=bad, validate_alignment=True)
    with pytest.raises(ValueError, match="motion limit"):
        register_image(image, image, registration_method=bad, max_displacement_voxels=2)
    for preset in registration_presets():
        result = run_registration(
            image,
            moving,
            RegistrationOptions(**preset["settings"]),
            fixed_mask=image > image.max() * 0.05,
        )
        expected = np.array([13.0, 12.0]) + [-1, 2]
        assert (
            np.linalg.norm(np.asarray(result.transform.TransformPoint((13.0, 12.0))) - expected)
            < 0.6
        )
        assert result.diagnostics.alignment_check["maximum_displacement_voxels"] < 6
        assert result.corrected.real.max() > 50000
    padded = np.pad(image / 1e5, ((0, 0), (20, 0)))
    result = register_image(
        image / 1e5,
        padded,
        registration_method=registration_method_factory(
            RegistrationOptions(transform="translation", metric="mean_squares")
        ),
        moving_pre_transform=sitk.TranslationTransform(2, (20.0, 0.0)),
        max_displacement_voxels=0.5,
    )
    np.testing.assert_allclose(result.corrected, image / 1e5, atol=1e-4)
    gx, gy = np.indices((96, 60))
    anatomy = (
        np.exp(-((gx - 35) ** 2 + (gy - 14) ** 2) / 45)
        + 0.8 * np.exp(-((gx - 52) ** 2 + (gy - 44) ** 2) / 60)
        + 0.2 * np.exp(-((gx - 20) ** 2 + (gy - 30) ** 2) / 30)
    ).astype(np.float32)
    region = ((gx - 35) ** 2 + (gy - 14) ** 2 < 144) | ((gx - 52) ** 2 + (gy - 44) ** 2 < 144)
    truth = sitk.Euler2DTransform()
    truth.SetCenter((29.5, 47.5))
    truth.SetTranslation((7, -3))
    truth.SetAngle(np.deg2rad(5))
    reference = sitk.GetImageFromArray(np.sqrt(anatomy))
    displaced = sitk.GetArrayFromImage(
        sitk.Resample(reference, reference, truth, sitk.sitkLinear, 0)
    )
    result = run_registration(
        anatomy,
        displaced,
        RegistrationOptions(prealign_translation=True, max_displacement_voxels=20),
        fixed_mask=region.astype(np.uint16) * 256,
    )
    for point in ((14.0, 35.0), (44.0, 52.0), (30.0, 20.0)):
        assert (
            np.linalg.norm(
                np.asarray(result.transform.TransformPoint(point))
                - truth.GetInverse().TransformPoint(point)
            )
            < 0.4
        )
    source = np.stack(
        [
            np.stack([shift(image * scale, (t, -0.5 * t), order=1) for scale in (1.0, 0.02)])
            for t in range(3)
        ]
    )
    ordered = source.transpose(2, 0, 3, 1)
    result = motion_correct_dataset(
        Experiment(dims={"x": 28, "t": 3, "y": 32, "offset": 2}, exp_type="anatomical"),
        DataSet(data=ordered),
        dimensions=("t",),
        estimation_dimensions=("offset",),
        registration_method=registration_method_factory(options),
        normalize_intensity=True,
    )
    corrected = np.asarray(result.corrected).transpose(1, 3, 0, 2)
    np.testing.assert_allclose(corrected[:, 1], corrected[:, 0] * 0.02, rtol=2e-5, atol=0.05)
    assert np.mean((corrected[-1, 0] - image) ** 2) < np.mean((source[-1, 0] - image) ** 2) / 5
    assert result.estimation_dimensions == ("offset",)

    from scipy.spatial.transform import Rotation

    spacing = np.array([0.8, 1.3, 2.1])
    coordinates = np.indices((18, 26, 22))[::-1] * spacing[:, None, None, None]
    center = np.array([8.0, 16.0, 18.0])
    rotation = Rotation.from_rotvec([0.03, -0.04, 0.06]).as_matrix()
    translation = np.array([0.5, -1.0, 1.5])

    def density(points):
        first = (points - np.array([6.0, 13.0, 15.0])[:, None, None, None]) / np.array(
            [2.0, 4.0, 6.0]
        )[:, None, None, None]
        second = (points - np.array([12.0, 24.0, 26.0])[:, None, None, None]) / np.array(
            [2.0, 3.0, 4.0]
        )[:, None, None, None]
        return (
            np.exp(-np.sum(first**2, axis=0)) + 0.7 * np.exp(-np.sum(second**2, axis=0))
        ).astype(np.float32)

    inverse_points = (
        rotation.T
        @ (coordinates - center[:, None, None, None] - translation[:, None, None, None]).reshape(
            3, -1
        )
    ).reshape(coordinates.shape) + center[:, None, None, None]
    result = run_registration(
        density(coordinates),
        density(inverse_points),
        RegistrationOptions(metric="mean_squares", iterations=220, prealign_translation=True),
        fixed_spacing=spacing,
    )
    for point in (np.array([8.0, 10.0, 8.0]), np.array([14.0, 18.0, 12.0])):
        expected = rotation @ (point * spacing - center) + center + translation
        actual = np.asarray(result.transform.TransformPoint(point)) * spacing
        assert np.linalg.norm(actual - expected) < 0.3


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


class AdditiveTransform:
    def identity_parameters(self, image):
        return jnp.zeros(1, dtype=jnp.float32)

    def apply(self, image, parameters):
        return jnp.asarray(image) + jnp.asarray(parameters)[0]
