"""Decode delayed radial readouts on their measured k-space support."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from mrax.data_struct.acquisition import ReadoutModel


def prepare_delayed_readouts(records, target_trajectory, *, folder, method, acqp, reco, loop_order=("polarity", "interleave", "frame", "coil", "read")):
    """Resolve polarity groups, ADC delay and explicit measured coordinates.

    A Fourier profile inversion is only a suggested reconstruction operation.
    The returned primary stream contains the original measurements from every
    imaging polarity; reference and skipped ADC samples remain auxiliary data.
    """
    from .bruker_parameters import parse_jcamp
    from .bruker_sampling import _parse_bruker_inverse_fid_delay

    settings_path = Path(folder) / "pdata" / str(reco.get("_pdata_index", "1")) / "methreco"
    settings = {**method, **(parse_jcamp(settings_path) if settings_path.exists() else {}), **reco}
    if str(settings.get("UseOrigin", "No")).lower() in {"yes", "true", "1"}:
        raise ValueError("Delayed readouts with origin-reference profiles require an explicit event profile")
    points = int(settings["RecoRegridNPoints"])
    profiles = int(settings["RecoRegridNInterleaves"])
    frames = int(acqp.get("NI", 1)) * int(acqp.get("NR", 1))
    groups = 1 if str(settings.get("HalfAcquisition", "No")).lower() in {"yes", "true", "1"} else 2
    source = records.samples
    if len(source) != groups * profiles * frames:
        raise ValueError("Delayed readout groups must account for every raw record")
    oversampling = float(settings["OverSampling"])
    if not np.isfinite(oversampling) or oversampling <= 0:
        raise ValueError("ADC oversampling must be finite and positive")
    skip = int(settings.get("RecoSkip", 0))
    if not 0 <= skip < source.shape[-1]:
        raise ValueError("Delayed readout skip exceeds the acquired profile")
    delay = _parse_bruker_inverse_fid_delay(reco)
    if delay is None:
        raise ValueError("Delayed readout acquisition delay is not explicitly declared")
    # Nonzero digital/analog filter delays must be resolved rather than hidden
    # in an empirical spatial shift. Their sample-time semantics are distinct.
    import re

    stage = str(reco.get("RecoStageNodes", ""))
    for key in ("groupDelay", "filterDelay"):
        match = re.search(rf"{key}\s*=\s*([-+\d.eE]+)", stage)
        if match is None or float(match[1]) != 0:
            raise ValueError("Nonzero or unresolved receiver filter delays need explicit calibration")
    coordinates = (np.arange(source.shape[-1], dtype=float)[skip:] + delay) / (oversampling * points)
    if groups == 2:
        coordinates = np.concatenate((coordinates, -coordinates))
    target = np.asarray(target_trajectory).reshape((profiles, points, -1))
    direction = target[:, -1] - target[:, 0]
    direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
    trajectory = (direction[:, None, :] * coordinates[None, :, None]).reshape((-1, target.shape[-1]))
    canonical = ("polarity", "interleave", "frame", "coil", "read")
    loop_order = tuple(loop_order)
    if len(loop_order) != len(canonical) or set(loop_order) != set(canonical):
        raise ValueError("Delayed profile loops must identify every acquired axis exactly once")
    sizes = dict(zip(canonical, (groups, profiles, frames, source.shape[1], source.shape[-1]), strict=True))
    grouped = source.reshape(tuple(sizes[key] for key in loop_order)).transpose(tuple(loop_order.index(key) for key in canonical))
    samples = grouped[..., skip:].transpose((2, 3, 1, 0, 4)).reshape((frames, source.shape[1], -1))
    auxiliary = dict(records.auxiliary_samples)
    if skip:
        auxiliary["skipped_adc"] = source[..., :skip]
    cutoff = float(settings.get("RecoRegularise", 0))
    cutoff = cutoff if cutoff > 0 else np.finfo(float).eps * max(len(coordinates), points)
    readout = ReadoutModel(coordinates, points, target_trajectory, points, cutoff)
    return samples, trajectory.astype(np.float32), readout, auxiliary
