"""Native-grid rectangle rasterization."""

from __future__ import annotations

from typing import Any

import numpy as np

from mrax.analysis import ROI
from mrax.gui.masks import embed_mask_slice
from mrax.gui.session import GuiSession


def add_rectangle_roi(session: GuiSession, name: str, bounds: list[float]) -> ROI:
    """Rasterize viewer-native bounds in native voxel coordinates."""
    if session.roi(name) is not None or any(roi.name == name for roi in session.shared_rois):
        raise ValueError("An ROI with this name already exists")
    plane_mask = _rect_mask(tuple(session.frame_image().shape), bounds)
    if not np.any(plane_mask):
        raise ValueError("The rectangle does not contain any native pixels")
    mask = embed_mask_slice(
        plane_mask,
        spatial_shape=session.spatial_shape,
        spatial_dims=session.spatial_dim_names,
        indices=session.active_indices,
        plane_dims=session.image_plane_dims,
    )
    roi = ROI(
        name,
        mask,
        dims=tuple(session.spatial_dim_names),
        metadata={
            "kind": "rect",
            "bounds": list(bounds),
            "plane_dims": list(session.image_plane_dims),
        },
    )
    session.add_roi(roi)
    return roi


def _rotate_points(
    points: Any, angle_degrees: float, *, origin: tuple[float, float] = (0.0, 0.0)
) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if arr.size == 0:
        return arr.reshape((-1, 2))
    theta = np.deg2rad(float(angle_degrees))
    c, s = float(np.cos(theta)), float(np.sin(theta))
    origin_arr = np.asarray(origin, dtype=float)
    rel = arr.reshape((-1, 2)) - origin_arr
    rotated = np.column_stack([rel[:, 0] * c - rel[:, 1] * s, rel[:, 0] * s + rel[:, 1] * c])
    return rotated + origin_arr


def _rect_mask(
    shape: tuple[int, int],
    bounds: list[float] | tuple[float, float, float, float],
    *,
    angle: float = 0.0,
) -> np.ndarray:
    h, w = shape
    x, y, bw, bh = [float(v) for v in bounds]
    if abs(float(angle)) > 1e-9:
        xx, yy = np.meshgrid(np.arange(w, dtype=float) + 0.5, np.arange(h, dtype=float) + 0.5)
        local = _rotate_points(
            np.column_stack([xx.ravel() - x, yy.ravel() - y]),
            -float(angle),
            origin=(0.0, 0.0),
        )
        inside = (
            (local[:, 0] >= 0.0)
            & (local[:, 0] <= max(bw, 0.0))
            & (local[:, 1] >= 0.0)
            & (local[:, 1] <= max(bh, 0.0))
        )
        return inside.reshape((h, w))
    x0 = max(0, int(np.floor(x)))
    y0 = max(0, int(np.floor(y)))
    x1 = min(w, int(np.ceil(x + bw)))
    y1 = min(h, int(np.ceil(y + bh)))
    mask = np.zeros((h, w), dtype=bool)
    if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = True
    return mask
