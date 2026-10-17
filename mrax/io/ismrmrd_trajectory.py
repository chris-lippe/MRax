"""Validate stored MRD coordinates against executed sequence and encoding grids."""
from __future__ import annotations

import numpy as np


def convert_coordinates(coords, units, xyz, encoding, ndim, axes=None):
    coords = np.asarray(coords[:, :ndim], float).copy()
    if axes is not None:
        coords = coords[:, axes]
    if units == 'cycles/fov':
        coords /= np.asarray(xyz[:ndim])
    elif units == 'cycles/m':
        fov = np.asarray([getattr(encoding.encodedSpace.fieldOfView_mm, k) / 1000 for k in 'xyz'[:ndim]])
        coords *= fov / np.asarray(xyz[:ndim])
    elif units == 'radians/pixel':
        coords /= 2 * np.pi
    elif units not in (None, 'cycles/pixel'):
        raise ValueError("Unsupported ISMRMRD trajectory_units")
    if not np.isfinite(coords).all() or np.max(np.abs(coords)) > .75:
        raise ValueError("Trajectory exceeds cycles/pixel bounds; specify its physical trajectory_units explicitly")
    return coords


def sequence_trajectory_encoding(records, heads, indices, info, encoding, xyz, ndim, units):
    """Infer physical units only from agreement with the matching Pulseq ADCs.

    A declared Cartesian trajectory becomes a grid only after every coordinate
    agrees with its MRD counter and readout centre. Otherwise retain the samples.
    """
    if (info is None or info.trajectory_physical is None or len(info.adc_counts) != len(heads)
            or not np.array_equal(info.adc_counts, heads['number_of_samples'])):
        return units, None, False, False, ()
    physical = np.asarray(info.trajectory_physical)
    if physical.ndim != 2 or physical.shape[1] < ndim:
        return units, None, False, False, ()
    starts = np.r_[0, np.cumsum(info.adc_counts)]
    matched = np.ones(ndim, bool)
    absent = np.ones(ndim, bool)
    varying = np.zeros(ndim, bool)
    for j in indices:
        if int(heads[j]['trajectory_dimensions']) < ndim:
            return units, None, False, False, ()
        stored = records[int(j)]['traj'].reshape(-1, int(heads[j]['trajectory_dimensions']))
        expected = physical[starts[j]:starts[j + 1], :ndim]
        matched &= np.all(np.isclose(stored[:, :ndim], expected, atol=1e-3, rtol=1e-5), axis=0)
        absent &= np.all(np.abs(stored[:, :ndim]) < 1e-3, axis=0)
        varying |= np.ptp(stored[:, :ndim], axis=0) > 1e-2
    missing = tuple(int(i) for i in np.flatnonzero(~matched))
    declared = str(getattr(encoding.trajectory, 'value', encoding.trajectory)).lower()
    if missing and (declared != 'cartesian' or not np.all(absent[list(missing)]) or not np.any(matched & varying)):
        return units, None, False, False, ()
    units = units or 'cycles/m'
    axes = None
    channel_order = info.definitions.get('channel_order', ())
    channel_order = channel_order.split() if isinstance(channel_order, str) else list(channel_order)
    if len(channel_order) >= ndim and set(channel_order[:ndim]) == set('xyz'[:ndim]):
        axes = ['xyz'.index(k) for k in channel_order[:ndim]]
    if declared != 'cartesian':
        return units, axes, True, False, ()
    if axes is None:
        # Pulseq gradients use physical channels; MRD counters use read/phase.
        spreads = [np.ptp(physical[starts[j]:starts[j + 1], :ndim], axis=0) for j in indices]
        read = int(np.argmax(np.median(spreads, axis=0)))
        axes = [read] + [i for i in range(ndim) if i != read]
    from mrax.io.ismrmrd_measurement import _phase_index

    for j in indices:
        head = heads[j]
        stored = records[int(j)]['traj'].reshape(-1, int(head['trajectory_dimensions']))
        if missing:
            stored = stored.copy()
            stored[:, missing] = physical[starts[j]:starts[j + 1], missing]
        pre, post = int(head['discard_pre']), int(head['discard_post'])
        end = len(stored) - post
        coords = convert_coordinates(stored[pre:end], units, xyz, encoding, ndim, axes) * np.asarray(xyz[:ndim])
        expected = np.empty_like(coords)
        expected[:, 0] = np.arange(pre, end) - int(head['center_sample'])
        if int(head['flags']) & (1 << 21):
            expected[:, 0] = expected[::-1, 0]
        for dim, key in enumerate(('kspace_encoding_step_1', 'kspace_encoding_step_2')[:ndim - 1], 1):
            limit = getattr(encoding.encodingLimits, key, None) if encoding.encodingLimits else None
            counter = int(head['idx'][key.replace('encoding', 'encode')])
            expected[:, dim] = _phase_index(counter, limit, xyz[dim]) - xyz[dim] // 2
        if not np.allclose(coords, expected, atol=1e-3, rtol=0):
            if missing:
                raise ValueError("Incomplete MRD trajectory cannot be verified against sequence and Cartesian counters")
            return units, axes, True, False, ()
    return units, axes, True, True, missing
