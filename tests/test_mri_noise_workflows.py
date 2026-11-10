"""Independent synthetic MRI noise fields; no imported measurement fixtures."""

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from mrax.preprocessing import denoise_bm3d, denoise_bm4d, estimate_noise, estimate_noise_sigma


@pytest.mark.parametrize("ndim", [2, 3])
def test_local_estimate_removes_affine_signal_and_retains_raw_intensity_units(ndim):
    shape = (64, 64) if ndim == 2 else (30, 30, 20)
    grid = np.indices(shape)
    image = 20 + sum(
        (axis + 1) * coordinates / shape[axis] for axis, coordinates in enumerate(grid)
    )
    image += np.random.default_rng(19).normal(0, 0.12, shape)
    source = image.copy()
    report = estimate_noise(image)
    assert report["sigma"] == pytest.approx(0.12, rel=0.12)
    assert estimate_noise_sigma(image * 1e6) == pytest.approx(report["sigma"] * 1e6)
    np.testing.assert_array_equal(image, source)


def test_automatic_magnitude_estimate_uses_signal_not_quieter_background():
    rng = np.random.default_rng(41)
    truth = np.zeros((96, 96))
    truth[24:72, 24:72] = 2
    sigma = np.where(truth > 0, 0.15, 0.02)
    image = np.hypot(
        truth + rng.normal(size=truth.shape) * sigma, rng.normal(size=truth.shape) * sigma
    )
    report = estimate_noise(image)
    assert report["sigma"] == pytest.approx(0.15, rel=0.18)
    assert report["method"] == "local_signal_patches"


def test_automatic_estimate_does_not_count_smooth_texture_as_random_noise():
    y, x = np.indices((128, 128))
    truth = 5 + 0.3 * np.sin(2 * np.pi * x / 8) + 0.2 * np.sin(2 * np.pi * y / 10)
    image = truth + np.random.default_rng(2).normal(0, 0.05, truth.shape)
    report = estimate_noise(image)
    assert report["sigma"] == pytest.approx(0.05, rel=0.2)
    assert any("image structure" in warning for warning in report["warnings"])


@pytest.mark.parametrize("coils", [1, 4, 12])
def test_noise_only_magnitude_roi_recovers_underlying_gaussian_sigma(coils):
    rng = np.random.default_rng(94)
    image = np.sqrt(np.sum(rng.normal(0, 750, (2 * coils, 64, 64)) ** 2, axis=0))
    mask = np.zeros(image.shape, dtype=bool)
    mask[8:56, 8:56] = True
    original_mask = mask.copy()
    report = estimate_noise(image, mask=mask, model="background", coils=coils)
    assert report["sigma"] == pytest.approx(750, rel=0.04)
    np.testing.assert_array_equal(mask, original_mask)


def test_correlated_noise_is_not_measured_only_as_adjacent_pixel_differences():
    rng = np.random.default_rng(200)
    noise = gaussian_filter(rng.normal(size=(128, 128)), 0.8)
    noise *= 0.1 / noise.std()
    report = estimate_noise(5 + noise, model="gaussian")
    assert report["sigma"] == pytest.approx(0.1, rel=0.25)
    assert any("Correlated" in warning for warning in report["warnings"])


def test_complex_estimation_reports_real_and_imaginary_noise_separately():
    rng = np.random.default_rng(42)
    image = 4 + rng.normal(0, 0.08, (64, 64)) + 1j * rng.normal(0, 0.2, (64, 64))
    report = estimate_noise(image)
    assert report["components"]["real"]["sigma"] == pytest.approx(0.08, rel=0.15)
    assert report["components"]["imaginary"]["sigma"] == pytest.approx(0.2, rel=0.15)


def test_zero_real_component_does_not_hide_imaginary_noise():
    image = 1j * np.random.default_rng(58).normal(3, 0.2, (64, 64))
    report = estimate_noise(image)
    assert report["method"] == "complex_components"
    assert report["components"]["real"]["sigma"] == 0
    assert report["components"]["imaginary"]["sigma"] == pytest.approx(0.2, rel=0.15)


def test_large_images_bound_patch_work_and_sparse_rois_use_no_full_residual_field(monkeypatch):
    from mrax.preprocessing import noise

    image = np.random.default_rng(8).normal(5, 0.1, (1024, 1024))
    calls = []
    original = noise._patch

    def counted(values, selected):
        calls.append(values.shape)
        return original(values, selected)

    monkeypatch.setattr(noise, "_patch", counted)
    report = estimate_noise(image)
    assert len(calls) == 256
    assert report["sigma"] == pytest.approx(0.1, rel=0.15)
    mask = np.zeros_like(image, dtype=bool)
    mask[::32, ::32] = True
    report = estimate_noise(image, mask=mask, model="gaussian")
    assert report["sigma"] == pytest.approx(0.1, rel=0.15)
    assert any("Few local patches" in warning for warning in report["warnings"])


def test_roi_validation_padding_missing_values_and_constant_images():
    image = np.ones((32, 32))
    assert estimate_noise_sigma(image) == 0
    assert estimate_noise_sigma(np.zeros_like(image)) == 0
    image[:8] = np.nan
    image[8:16] = 0
    mask = np.ones_like(image, dtype=bool)
    original = mask.copy()
    assert estimate_noise_sigma(image, mask=mask) == 0
    np.testing.assert_array_equal(mask, original)
    for options, message in [
        ({"mask": np.zeros_like(mask)}, "16 finite"),
        ({"mask": mask[:2]}, "match"),
        ({"model": "background"}, "noise-only"),
        ({"coils": 0}, "coils"),
        ({"model": "other"}, "Choose"),
    ]:
        with pytest.raises(ValueError, match=message):
            estimate_noise(image, **options)
    with pytest.raises(ValueError, match="negative"):
        estimate_noise(-np.ones_like(image), mask=mask, model="background")


@pytest.mark.parametrize("method,shape", [("bm3d", (48, 48)), ("bm4d", (24, 24, 16))])
@pytest.mark.parametrize("region", ["automatic", "signal", "background"])
def test_real_bm_filters_reduce_magnitude_mri_noise_using_auto_and_roi_estimates(
    method, shape, region
):
    module = pytest.importorskip(method)
    function = denoise_bm3d if method == "bm3d" else denoise_bm4d
    profile = getattr(module, method.upper() + "Profile")()
    profile.num_threads = 1
    truth = np.ones(shape)
    truth[tuple(slice(0, n // 4) for n in shape)] = 0
    rng = np.random.default_rng(77)
    image = np.hypot(truth + rng.normal(0, 0.08, shape), rng.normal(0, 0.08, shape))
    mask = None if region == "automatic" else truth == (1 if region == "signal" else 0)
    model = "background" if region == "background" else "auto"
    original = image.copy()
    for scale in (1, 1e6):
        result = (
            np.asarray(function(image * scale, noise_mask=mask, noise_model=model, profile=profile))
            / scale
        )
        signal = truth > 0
        assert (
            np.mean((result[signal] - truth[signal]) ** 2)
            < np.mean((image[signal] - truth[signal]) ** 2) * 0.6
        )
        assert np.all(np.isfinite(result))
        assert result.shape == shape
    np.testing.assert_array_equal(image, original)
