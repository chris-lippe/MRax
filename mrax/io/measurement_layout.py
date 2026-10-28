"""Regular-volume assembly using explicit slice labels and physical planes."""
from __future__ import annotations

import numpy as np


def slice_stack_layout(coordinates, positions, normals):
    """Return slice count, frame order, volume centres and spacing, or None.

    Every repeat must contain the same distinct slice labels and uniformly
    spaced parallel planes. Incomplete or moving slice sets remain separate
    frames rather than being padded or assigned fabricated geometry.
    """
    if 'slice' not in coordinates:
        return None
    labels = np.asarray(coordinates['slice'])
    slices = len(np.unique(labels))
    if slices <= 1:
        return None
    positions, normals = np.asarray(positions, float), np.asarray(normals, float)
    if positions.shape != (len(labels), 3) or normals.shape != positions.shape or not np.isfinite(positions).all():
        return None
    if not np.allclose(np.linalg.norm(normals, axis=1), 1., atol=1e-3):
        return None
    columns = [np.asarray(value) for key, value in coordinates.items() if key not in ('slice', 'time')]
    keys = np.column_stack(columns) if columns else np.zeros((len(labels), 1))
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    groups, centres, spacings = [], [], []
    for group in np.unique(inverse):
        selected = np.flatnonzero(inverse == group)
        if len(selected) != slices or len(np.unique(labels[selected])) != slices:
            return None
        normal = normals[selected[0]]
        if not np.allclose(normals[selected], normal, atol=1e-3):
            return None
        selected = selected[np.argsort(positions[selected] @ normal)]
        delta = np.diff(positions[selected], axis=0)
        spacing = float(np.mean(delta @ normal))
        if spacing <= 0 or not np.allclose(delta, normal * spacing, atol=1e-3, rtol=1e-3):
            return None
        groups.append(selected)
        centres.append((positions[selected[0]] + positions[selected[-1]]) / 2)
        spacings.append(spacing)
    if not np.allclose(spacings, spacings[0], atol=1e-3, rtol=1e-3):
        return None
    return slices, np.stack(groups), np.asarray(centres), spacings[0]


def sequence_slice_positions(labels, positions, normals, definitions):
    """Resolve repeated prototype planes using explicitly declared slice offsets.

    Existing distinct scanner positions take precedence. Pulseq slice offsets
    are relative to the prototype centre, in metres unless SlicePositionsUnit
    declares millimetres. Unspecified offsets leave repeated planes unresolved.
    """
    import warnings

    labels = np.asarray(labels, int)
    positions = np.asarray(positions, float)
    if len(np.unique(labels)) <= 1 or not np.allclose(positions, positions[0], atol=1e-3):
        return positions
    offsets = definitions.get('SlicePositions')
    if offsets is None:
        warnings.warn("Distinct slice labels share one scanner plane; provide sequence SlicePositions for volume geometry", RuntimeWarning, stacklevel=3)
        return positions
    offsets = np.asarray(offsets, float)
    units = str(definitions.get('SlicePositionsUnit', 'm')).lower()
    if offsets.ndim != 1 or not np.isfinite(offsets).all() or np.min(labels) < 0 or np.max(labels) >= len(offsets) or units not in ('m', 'mm'):
        raise ValueError("SlicePositions must cover every slice label with finite relative offsets in m or mm")
    normals = np.asarray(normals, float)
    if normals.shape != positions.shape or not np.allclose(np.linalg.norm(normals, axis=1), 1., atol=1e-3):
        raise ValueError("Declared slice offsets require unit scanner slice directions")
    return positions + offsets[labels, None] * (1000. if units == 'm' else 1.) * normals
