"""Quality-guided phase integration on a maximum-reliability spanning forest."""
from __future__ import annotations

from itertools import product

import jax.numpy as jnp
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree


def quality_guided_unwrap(phase, *, spatial_ndim, mask=None, weights=None, period=2 * np.pi):
    """Unwrap valid spatial neighbours within each independent leading index.

    A curvature reliability measure ranks edges; a maximum-reliability spanning
    forest chooses integration paths. Each connected component starts at its
    strongest voxel with a principal-valued phase anchor. Disconnected components
    retain an independent integer-period ambiguity. Sparse graph construction and
    sorting run on the CPU; the result is returned in the current JAX context.
    """
    values = np.asarray(phase)
    if np.iscomplexobj(values) or spatial_ndim not in (1, 2, 3) or values.ndim < spatial_ndim or not np.isfinite(period) or period <= 0:
        raise ValueError("Quality-guided unwrapping requires real spatial phase and a positive period")
    included = np.ones(values.shape, bool) if mask is None else np.broadcast_to(np.asarray(mask, bool), values.shape)
    signal = np.ones(values.shape) if weights is None else np.broadcast_to(np.asarray(weights), values.shape)
    if not np.isfinite(values[included]).all() or not np.isfinite(signal[included]).all() or np.any(signal[included] < 0):
        raise ValueError("Included phase and reliability weights must be finite and weights nonnegative")
    shape = values.shape[-spatial_ndim:]
    batch = values.reshape((-1,) + shape)
    masks = included.reshape(batch.shape)
    signals = signal.reshape(batch.shape)
    output = np.zeros_like(batch, dtype=np.result_type(values.dtype, np.float32))

    def principal(delta):
        return (delta + period / 2) % period - period / 2

    directions = [step for step in product((-1, 0, 1), repeat=spatial_ndim)
        if step > (0,) * spatial_ndim]
    for destination, wrapped, valid, magnitude in zip(output, batch, masks, signals, strict=True):
        locations = np.flatnonzero(valid)
        if not len(locations):
            continue
        compact = np.full(valid.size, -1, int)
        compact[locations] = np.arange(len(locations))
        compact = compact.reshape(shape)
        curvature = np.full(shape, len(directions) * (period / 2)**2)
        for step in directions:
            center = tuple(slice(1, -1) if displacement else slice(None) for displacement in step)
            before = tuple(slice(0, -2) if displacement == 1 else slice(2, None) if displacement == -1 else slice(None) for displacement in step)
            after = tuple(slice(2, None) if displacement == 1 else slice(0, -2) if displacement == -1 else slice(None) for displacement in step)
            reliable = valid[center] & valid[before] & valid[after]
            difference = principal(wrapped[after] - wrapped[center]) - principal(wrapped[center] - wrapped[before])
            curvature[center] += np.where(reliable, difference**2 - (period / 2)**2, 0)
        quality = 1 / np.sqrt(np.maximum(curvature, 0) + 1e-8)
        source, target, costs = [], [], []
        for axis in range(spatial_ndim):
            first, second = [slice(None)] * spatial_ndim, [slice(None)] * spatial_ndim
            first[axis], second[axis] = slice(None, -1), slice(1, None)
            first, second = tuple(first), tuple(second)
            active = valid[first] & valid[second]
            source.append(compact[first][active])
            target.append(compact[second][active])
            costs.append(1 / (quality[first][active] + quality[second][active]))
        graph = coo_matrix((np.concatenate(costs), (np.concatenate(source), np.concatenate(target))),
            shape=(len(locations), len(locations))).tocsr()
        forest = minimum_spanning_tree(graph)
        forest = forest + forest.T
        _, labels = connected_components(forest, directed=False)
        flat_phase = wrapped.ravel()[locations]
        flat_signal = magnitude.ravel()[locations]
        offsets = np.zeros(len(locations), np.int64)
        by_component = np.argsort(labels, kind="stable")
        boundaries = np.concatenate(([0], np.flatnonzero(np.diff(labels[by_component])) + 1, [len(locations)]))
        visited = np.zeros(len(locations), bool)
        for begin, end in zip(boundaries[:-1], boundaries[1:], strict=True):
            members = by_component[begin:end]
            seed = members[np.argmax(flat_signal[members])]
            stack = [seed]
            visited[seed] = True
            while stack:
                previous = stack.pop()
                for child in forest.indices[forest.indptr[previous]:forest.indptr[previous + 1]]:
                    if visited[child]:
                        continue
                    delta = principal(flat_phase[child] - flat_phase[previous])
                    offsets[child] = offsets[previous] + int(np.rint((flat_phase[previous] - flat_phase[child] + delta) / period))
                    visited[child] = True
                    stack.append(child)
        destination.ravel()[locations] = flat_phase + period * offsets
    return jnp.asarray(output.reshape(values.shape))
