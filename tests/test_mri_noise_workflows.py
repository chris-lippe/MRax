"""Independent synthetic MRI noise fields."""

import numpy as np
import pytest

from mrax.preprocessing import estimate_noise, estimate_noise_sigma


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


def test_complex_estimation_reports_real_and_imaginary_noise_separately():
    rng = np.random.default_rng(42)
    image = 4 + rng.normal(0, 0.08, (64, 64)) + 1j * rng.normal(0, 0.2, (64, 64))
    report = estimate_noise(image)
    assert report["components"]["real"]["sigma"] == pytest.approx(0.08, rel=0.15)
    assert report["components"]["imaginary"]["sigma"] == pytest.approx(0.2, rel=0.15)


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
