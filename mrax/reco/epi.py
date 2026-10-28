"""Source-independent readout calibration and EPI phase correction.

Navigator phase fitting follows the hybrid-space approach described by Poser
et al., DOI 10.1002/mrm.24222. The optional referenceless entropy search follows
the coarse-search/refinement strategy of Skare et al., ISMRM 2006, abstract 2349.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from .variational import fftnc, ifftnc


def resample_readout(samples, coordinates, target, *, kernel="linear", radius=8):
    """Interpolate complex ADC values at explicit increasing readout coordinates.

    Coordinates and targets must share units. Out-of-support values are zero;
    resampling retains the original coordinate scale.
    """
    array = jnp.asarray(samples)
    source = np.asarray(coordinates)
    destination = np.asarray(target)
    if source.shape != (array.shape[-1],) or source.size < 2 or not np.isfinite(source).all() or np.any(np.diff(source) <= 0):
        raise ValueError("Readout coordinates must be finite and strictly increasing")
    if destination.ndim != 1 or not np.isfinite(destination).all():
        raise ValueError("Readout target coordinates must be a finite vector")
    if kernel == "lanczos":
        from operator import index

        radius = index(radius)
        if radius < 2:
            raise ValueError("Windowed-sinc interpolation requires a radius of at least two samples")
        # Interpolate the inverse coordinate map, then resample the band-limited
        # ADC stream. The window radius is measured in ADC samples.
        position = np.interp(destination, source, np.arange(len(source)))
        neighbours = np.floor(position).astype(int)[:, None] + np.arange(-radius + 1, radius + 1)[None, :]
        delta = position[:, None] - neighbours
        weights = np.sinc(delta) * np.sinc(delta / radius)
        weights *= (np.abs(delta) < radius) & (neighbours >= 0) & (neighbours < len(source))
        weights /= np.sum(weights, axis=-1, keepdims=True)
        interpolated = jnp.sum(array[..., np.clip(neighbours, 0, len(source) - 1)] * jnp.asarray(weights), axis=-1)
        return jnp.where((destination >= source[0]) & (destination <= source[-1]), interpolated, 0)
    if kernel != "linear":
        raise ValueError("Readout interpolation kernel must be linear or lanczos")
    def interpolate(line):
        return jnp.interp(destination, source, line.real, left=0., right=0.) + 1j * jnp.interp(destination, source, line.imag, left=0., right=0.)
    return jax.vmap(interpolate)(array.reshape((-1, array.shape[-1]))).reshape(array.shape[:-1] + (len(destination),))


def estimate_navigator_phase(positive, negative):
    """Fit negative/positive phase in radians as offset + slope*x/FOV.

    Inputs are aligned, same-TE navigator readouts with identical shapes. Leading
    repetitions/receivers are pooled with signal weights. For a three-echo
    navigator, interpolate the two outer echoes to the middle echo's TE first.
    """
    positive = np.asarray(positive)
    negative = np.asarray(negative)
    if positive.shape != negative.shape or positive.ndim < 1 or positive.shape[-1] < 3:
        raise ValueError("Navigator polarities must have matching readout shapes")
    if not np.isfinite(positive).all() or not np.isfinite(negative).all():
        raise ValueError("Navigators must be finite")
    pos = np.asarray(ifftnc(positive, axes=(-1,)))
    neg = np.asarray(ifftnc(negative, axes=(-1,)))
    cross = np.sum((neg * np.conj(pos)).reshape((-1, pos.shape[-1])), axis=0)
    weights = np.abs(cross)
    included = weights > np.max(weights) * .05
    if np.count_nonzero(included) < 3:
        raise ValueError("Navigators have insufficient common signal for phase fitting")
    x = (np.arange(pos.shape[-1]) - pos.shape[-1] // 2) / pos.shape[-1]
    phase = np.unwrap(np.angle(cross[included]))
    design = np.stack((np.ones_like(x[included]), x[included]), axis=-1)
    weight = np.sqrt(weights[included] / np.max(weights))
    coefficients = np.linalg.lstsq(design * weight[:, None], phase * weight, rcond=None)[0]
    coefficients[0] = (coefficients[0] + np.pi) % (2 * np.pi) - np.pi
    return coefficients


def regrid_readouts(samples, coordinates, polarity, acquired, *, kernel="adjoint", radius=8):
    """Construct an FFT initializer from explicitly located readouts.

    Iterative objectives must retain the original coordinates and samples.
    Readouts can be stored in any sample order. Acquired coordinates and support
    remain authoritative.
    """
    array = jnp.asarray(samples)
    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.shape != (2, array.shape[-1]) or not np.isfinite(coordinates).all():
        raise ValueError("Regridding requires two finite readout coordinate vectors")
    negative = jnp.broadcast_to(jnp.asarray(polarity) < 0, array.shape)
    valid = jnp.broadcast_to(jnp.asarray(acquired, bool), array.shape)
    if not np.array_equal(np.asarray(valid), np.broadcast_to(np.asarray(valid)[..., :1], array.shape)):
        raise ValueError("Readout interpolation requires complete ADC profiles; retain sparse coordinates for iterative encoding")
    target = np.fft.fftshift(np.fft.fftfreq(array.shape[-1]))
    if kernel == "adjoint":
        # Direct one-dimensional Fourier quadrature avoids an interpolation
        # kernel's spatial noise modulation. Density is Voronoi interval length
        # in Fourier-grid units; coincident ADC locations share their cell.
        position = np.arange(array.shape[-1]) - array.shape[-1] // 2
        parts = []
        for opposite, source in enumerate(coordinates):
            unique, inverse, counts = np.unique(source, return_inverse=True, return_counts=True)
            if len(unique) < 2:
                raise ValueError("Readout quadrature requires at least two distinct Fourier coordinates")
            edges = np.concatenate(([unique[0] - (unique[1] - unique[0]) / 2],
                (unique[1:] + unique[:-1]) / 2, [unique[-1] + (unique[-1] - unique[-2]) / 2]))
            density = np.diff(edges)[inverse] / counts[inverse] * array.shape[-1]
            matrix = np.exp(2j * np.pi * source[:, None] * position) / np.sqrt(len(position))
            values = jnp.where(negative == bool(opposite), array, 0) * jnp.asarray(density)
            parts.append(jnp.einsum("...m,mn->...n", values, jnp.asarray(matrix, dtype=array.dtype), precision=jax.lax.Precision.HIGHEST))
        preview = fftnc(parts[0] + parts[1], axes=(-1,))
        return preview, valid
    values, supports = [], []
    for source in coordinates:
        order = np.argsort(source, kind="stable")
        ordered = source[order]
        unique = np.concatenate(([True], np.diff(ordered) > np.finfo(source.dtype).eps * 4))
        order, ordered = order[unique], ordered[unique]
        values.append(resample_readout(array[..., order], ordered, target, kernel=kernel, radius=radius))
        supports.append((target >= ordered[0]) & (target <= ordered[-1]))
    result = jnp.where(negative, values[1], values[0])
    support = valid & jnp.where(negative, supports[1], supports[0])
    return jnp.where(support, result, 0), support


def correct_readout_phase(samples, negative_mask, coefficients):
    """Apply a negative-readout phase correction in readout hybrid space."""
    array = jnp.asarray(samples)
    mask = jnp.broadcast_to(jnp.asarray(negative_mask, bool), array.shape)
    if np.shape(coefficients) != (2,) or not np.isfinite(coefficients).all():
        raise ValueError("Readout phase requires two finite radian coefficients")
    if not np.array_equal(np.asarray(mask), np.broadcast_to(np.asarray(mask)[..., :1], array.shape)):
        raise ValueError("Readout polarity must be constant along each readout")
    x = (jnp.arange(array.shape[-1]) - array.shape[-1] // 2) / array.shape[-1]
    phase = coefficients[0] + coefficients[1] * x
    hybrid = ifftnc(array, axes=(-1,))
    return fftnc(hybrid * jnp.where(mask, jnp.exp(-1j * phase), 1), axes=(-1,))


def _epi_entropy(parameters, positive, negative, x):
    image = positive + negative * jnp.exp(-1j * (parameters[0] + parameters[1] * x))
    power = jnp.sum(jnp.abs(image)**2, axis=0)
    probability = power / jnp.maximum(jnp.sum(power), 1e-30)
    return -jnp.sum(probability * jnp.log(jnp.maximum(probability, 1e-30)))


_epi_entropy_gradient = jax.jit(jax.value_and_grad(_epi_entropy))
_epi_entropy_candidates = jax.jit(jax.vmap(_epi_entropy, in_axes=(0, None, None, None)))
_epi_entropy = jax.jit(_epi_entropy)


def correct_epi_entropy(samples, negative_mask, *, spatial_ndim=2, phase_prior=(0., 0.),
                        slope_bound=20., max_iterations=100):
    """Estimate a common offset/slope from coil RSS entropy, then correct data.

    Input is one frame ``(coil, *encoding_shape)``. The offset search is limited
    to the pi-wide branch around the supplied prior. Entropy admits both the
    object and a half-FOV translated solution. A navigator or
    justified phase prior is needed to resolve that ambiguity. This optional
    estimator assumes adequate phase FOV and consistent motion between shots.
    """
    array = jnp.asarray(samples)
    if array.ndim != spatial_ndim + 1 or spatial_ndim not in (2, 3):
        raise ValueError("Entropy EPI correction requires one two- or three-dimensional coil frame")
    if not np.isfinite(array).all() or not np.any(np.abs(array) > 0):
        raise ValueError("Entropy EPI correction requires finite nonzero signal")
    mask = np.broadcast_to(np.asarray(negative_mask, bool), array.shape)
    if not np.array_equal(mask, np.broadcast_to(mask[..., :1], array.shape)):
        raise ValueError("Readout polarity must be constant along each readout")
    if not mask.any() or mask.all():
        raise ValueError("Entropy phase estimation requires both readout polarities")
    prior = np.asarray(phase_prior, float)
    if prior.shape != (2,) or not np.isfinite(prior).all() or not np.isfinite(slope_bound) or slope_bound <= 0 or max_iterations < 1:
        raise ValueError("Invalid EPI phase search parameters")
    axes = tuple(range(-spatial_ndim, 0))
    positive = ifftnc(jnp.where(mask, 0, array), axes=axes)
    negative = ifftnc(jnp.where(mask, array, 0), axes=axes)
    x = (jnp.arange(array.shape[-1]) - array.shape[-1] // 2) / array.shape[-1]

    def evaluate(parameters):
        return _epi_entropy(parameters, positive, negative, x)

    def value_and_grad(parameters):
        return _epi_entropy_gradient(parameters, positive, negative, x)

    offsets = prior[0] + np.linspace(-np.pi / 2, np.pi / 2, 5)
    slopes = prior[1] + np.linspace(-slope_bound, slope_bound, 21)
    # Coarse search avoids starting refinement at a local sidelobe minimum.
    candidates = np.asarray([(offset, slope) for offset in offsets for slope in slopes])
    values = np.asarray(_epi_entropy_candidates(jnp.asarray(candidates), positive, negative, x))
    initial = candidates[int(np.argmin(values))]
    def objective(parameters):
        value, gradient = value_and_grad(jnp.asarray(parameters))
        return float(value), np.asarray(gradient, float)
    fit = minimize(objective, initial, jac=True, method="L-BFGS-B",
        bounds=[(prior[0] - np.pi / 2, prior[0] + np.pi / 2),
                (prior[1] - slope_bound, prior[1] + slope_bound)],
        options={"maxiter": int(max_iterations), "ftol": 1e-12, "gtol": 1e-6})
    fitted = np.asarray(fit.x)
    before, after = float(evaluate(jnp.zeros(2))), float(evaluate(jnp.asarray(fitted)))
    if not np.isfinite(after) or after >= before:
        fitted = np.zeros(2)
        after = before
    corrected = correct_readout_phase(array, mask, fitted)
    return corrected, {"coefficients_radians": fitted.tolist(), "entropy_before": before,
        "entropy_after": after, "converged": bool(fit.success), "offset_branch_prior": prior[0].item()}


def calibrate_epi_navigators(acquisition, frame):
    """Fit aligned three-echo navigators at a common TE, using explicit timing."""
    auxiliary = acquisition.auxiliary_samples
    values = jnp.asarray(auxiliary['epi_navigators'][frame])
    polarity = np.asarray(acquisition.auxiliary_coordinates['epi_navigator_polarity'][frame])
    times = np.asarray(acquisition.auxiliary_coordinates['epi_navigator_times'][frame], float)
    if (values.shape != (3, acquisition.coil_count, acquisition.encoding_shape[-1]) or
            polarity.shape != (3,) or not np.isin(polarity, [-1, 1]).all() or times.shape != (3,) or
            not np.isfinite(times).all() or np.any(np.diff(times) <= 0) or polarity[0] == 0 or
            polarity[0] != polarity[2] or polarity[0] != -polarity[1]):
        raise ValueError("EPI navigator calibration requires three time-ordered alternating readouts")
    if acquisition.readout_coordinates is not None and acquisition.recipe.readout_regrid:
        values, _ = regrid_readouts(values[:, :, None, :], acquisition.readout_coordinates,
            polarity[:, None, None, None], True, kernel=acquisition.recipe.readout_interpolation,
            radius=acquisition.recipe.readout_kernel_radius)
        values = values[:, :, 0]
    hybrid = ifftnc(values, axes=(-1,))
    fraction = float((times[1] - times[0]) / (times[2] - times[0]))
    phase_delta = jnp.angle(hybrid[2] * jnp.conj(hybrid[0]))
    amplitude = jnp.abs(hybrid[0])**(1-fraction) * jnp.abs(hybrid[2])**fraction
    reference = fftnc(amplitude * jnp.exp(1j * (jnp.angle(hybrid[0]) + fraction * phase_delta)), axes=(-1,))
    positive, negative = (reference, values[1]) if polarity[0] > 0 else (values[1], reference)
    return estimate_navigator_phase(positive, negative)
