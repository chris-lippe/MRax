"""Noise estimation and real external denoisers on independent synthetic images."""

import numpy as np
import pytest

from mrax.preprocessing.denoise import _estimate_sigma, denoise_bm3d, denoise_bm4d


def phantom(ndim):
    shape = (48, 48) if ndim == 2 else (32, 32, 12)
    mask = np.zeros(shape, dtype=bool)
    mask[tuple(slice(n // 4, 3 * n // 4) for n in shape)] = True
    truth = mask.astype(float)
    noise = np.random.default_rng(42).normal(0, 0.08, shape)
    return truth, np.where(mask, truth + noise, 0)


@pytest.mark.parametrize("ndim", [2, 3])
def test_noise_estimate_ignores_zero_filled_background_and_preserves_intensity_units(ndim):
    _, noisy = phantom(ndim)
    assert np.count_nonzero(noisy) < noisy.size / 2
    estimate = _estimate_sigma(noisy)
    assert estimate == pytest.approx(0.08, rel=0.18)
    assert _estimate_sigma(noisy * 1e6) == pytest.approx(estimate * 1e6)


@pytest.mark.parametrize("value", [0, 17])
def test_constant_images_do_not_invent_noise(value):
    assert _estimate_sigma(np.full((16, 16, 4), value)) == 0


@pytest.mark.parametrize("method,ndim", [("bm3d", 2), ("bm4d", 3)])
@pytest.mark.parametrize("sigma", [None, 0.08])
@pytest.mark.parametrize("stage", ["full", "basic"])
def test_external_denoisers_reduce_noise_consistently_at_both_intensity_scales(method, ndim, sigma, stage):
    module = pytest.importorskip(method)
    profile = getattr(module, method.upper() + "Profile")()
    profile.num_threads = 1
    function = denoise_bm3d if method == "bm3d" else denoise_bm4d
    truth, noisy = phantom(ndim)
    original = noisy.copy()
    outputs = []
    for scale in (1, 1e6):
        result = function(
            noisy * scale,
            sigma=None if sigma is None else sigma * scale,
            stage=stage,
            profile=profile,
        ) / scale
        assert result.shape == noisy.shape
        assert np.all(np.isfinite(result))
        assert np.mean((result - truth) ** 2) < 0.25 * np.mean((noisy - truth) ** 2)
        assert np.max(np.abs(result - noisy)) > 0.01
        outputs.append(result)
    assert np.linalg.norm(outputs[0] - outputs[1]) / np.linalg.norm(outputs[0]) < 0.002
    np.testing.assert_array_equal(noisy, original)


@pytest.mark.parametrize("method", ["bm3d", "bm4d"])
def test_zero_sigma_is_an_exact_non_mutating_noop_and_invalid_sigma_is_rejected(method):
    pytest.importorskip(method)
    function = denoise_bm3d if method == "bm3d" else denoise_bm4d
    data = phantom(2 if method == "bm3d" else 3)[1]
    result = function(data, sigma=0)
    np.testing.assert_array_equal(result, data)
    assert not np.shares_memory(result, data)
    for sigma in (-1, np.nan, np.inf):
        with pytest.raises(ValueError, match="Noise sigma"):
            function(data, sigma=sigma)
