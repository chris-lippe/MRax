"""Factor explicit Fourier trajectories into phase grids and readout coordinates."""
from __future__ import annotations

import numpy as np

from mrax.data_struct import AcquisitionData, ReconstructionRecipe


def prepare_line_acquisition(samples, physical, fov, shape, output_shape, adc_counts, *, frame_indices=None, auxiliary=None):
    """Return a Cartesian phase encoding with exact readout vectors, or None.

    Physical coordinate columns are read, phase, partition. Only constant-phase,
    monotonic ADC profiles sharing two polarity-dependent coordinate vectors
    qualify. More general trajectories retain their sampled representation.
    """
    if not adc_counts or len(set(adc_counts)) != 1:
        return None
    points = int(adc_counts[0])
    if points < output_shape[-1] or points < 2 or len(physical) % points:
        return None
    profiles = physical.reshape((-1, points, len(shape)))
    if np.max(np.ptp(profiles[..., 1:], axis=1), initial=0.) > 1e-2:
        return None
    delta = np.diff(profiles[..., 0], axis=1)
    signs = np.sign(np.median(delta, axis=1)).astype(int)
    if np.any(signs == 0) or np.any(delta * signs[:, None] <= 0):
        return None
    phase_float = profiles[:, 0, 1:] * np.asarray(fov[1:len(shape)])
    phase = np.rint(phase_float).astype(int)[:, ::-1] + np.asarray(shape[:-1]) // 2
    if np.max(np.abs(phase_float - np.rint(phase_float))) > 2e-3:
        return None
    # The positive Nyquist frequency equals the negative DFT boundary.
    phase = np.where(phase == np.asarray(shape[:-1]), 0, phase)
    if np.any(phase < 0) or np.any(phase >= np.asarray(shape[:-1])):
        raise ValueError("Phase coordinates exceed the declared encoding grid")
    read = profiles[..., 0] * float(fov[0]) / shape[-1]
    data = samples.reshape((samples.shape[0], len(profiles), points))
    negative = signs < 0
    read[negative] = read[negative, ::-1]
    vectors = []
    for sign in (1, -1):
        selected = np.flatnonzero(signs == sign)
        vector = read[selected[0]] if len(selected) else None
        if vector is not None and not np.allclose(read[selected], vector, atol=2e-6, rtol=1e-5):
            return None
        vectors.append(vector)
    if vectors[0] is None:
        vectors[0] = vectors[1]
    if vectors[1] is None:
        vectors[1] = vectors[0]
    if frame_indices is None:
        _, inverse, visits = np.unique(phase, axis=0, return_inverse=True, return_counts=True)
        if len(set(visits)) != 1:
            raise ValueError("Nonuniform Cartesian repetitions require explicit frame labels")
        counters = np.zeros(len(visits), int)
        frames = np.empty(len(profiles), int)
        for i, key in enumerate(inverse):
            frames[i] = counters[key]
            counters[key] += 1
    else:
        labels = np.asarray(frame_indices).reshape((-1, points))
        if np.any(labels != labels[:, :1]):
            return None
        frames = labels[:, 0].astype(int)
    encoding = tuple(shape[:-1]) + (points,)
    values = np.zeros((int(frames.max()) + 1, samples.shape[0]) + encoding, samples.dtype)
    counts = np.zeros((values.shape[0], 1) + encoding, np.uint16)
    polarity = np.zeros((values.shape[0], 1) + encoding[:-1] + (1,), np.int8)
    for i, frame in enumerate(frames):
        key = (int(frame), slice(None), *phase[i])
        pol_key = (int(frame), 0, *phase[i], 0)
        if polarity[pol_key] and polarity[pol_key] != signs[i]:
            raise ValueError("Opposite-polarity repeats require distinct frame labels or an explicit sampled acquisition")
        values[key] += data[:, i, ::-1] if negative[i] else data[:, i]
        counts[(int(frame), 0, *phase[i])] += 1
        polarity[pol_key] = signs[i]
    values /= np.maximum(counts, 1)
    coordinates = np.asarray(vectors)
    if np.allclose(coordinates, np.fft.fftshift(np.fft.fftfreq(points))[None], atol=2e-6, rtol=1e-5):
        coordinates = None
    scale = float(np.sqrt(np.prod(output_shape) / np.prod(encoding)))
    recipe = ReconstructionRecipe(output_shape=tuple(output_shape), output_scale=scale)
    return AcquisitionData(values, encoding, ('partition', 'phase', 'read')[-len(shape):],
        acquired=counts > 0, statistical_weights=counts if np.any(counts > 1) else None,
        readout_polarity=polarity, readout_coordinates=coordinates,
        auxiliary_samples=auxiliary or {}, recipe=recipe)
