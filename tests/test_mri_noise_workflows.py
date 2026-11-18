"""Independent synthetic MRI noise fields."""

import numpy as np
import pytest

from mrax.preprocessing import estimate_noise


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


def test_noise_only_magnitude_roi_recovers_underlying_gaussian_sigma():
    coils = 4
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
