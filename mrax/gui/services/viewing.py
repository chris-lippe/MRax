"""Quantitative image display transforms."""

from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

import numpy as np

from mrax.analysis import ROI

from ..masks import mask_slice_for_display

_LOG = logging.getLogger(__name__)


def _first_spatial_image(array: Any) -> np.ndarray:
    arr = np.asarray(array)
    while arr.ndim > 3:
        arr = np.asarray(arr[0])
    if arr.ndim == 3 and 1 in arr.shape:
        arr = np.squeeze(arr)
    if arr.ndim not in (2, 3):
        raise ValueError(f"Expected a 2D or 3D spatial image, got shape {arr.shape}")
    return np.asarray(arr)


def _registration_preview_plane(array: Any) -> np.ndarray:
    arr = _first_spatial_image(array)
    if arr.ndim == 3:
        arr = np.asarray(arr[:, :, int(arr.shape[2] // 2)])
    if np.iscomplexobj(arr):
        arr = np.abs(arr)
    return np.asarray(arr)


def _equalized_overlay_units(reference: Any, foreground: Any) -> tuple[np.ndarray, np.ndarray]:
    ref = _image_to_scalar_for_blend(reference)
    fg = _image_to_scalar_for_blend(foreground)
    rows = min(int(ref.shape[0]), int(fg.shape[0]))
    cols = min(int(ref.shape[1]), int(fg.shape[1]))
    if rows <= 0 or cols <= 0:
        raise ValueError("reference and current view have no overlapping pixels")
    return (
        _histogram_equalized_unit(ref[:rows, :cols]),
        _histogram_equalized_unit(fg[:rows, :cols]),
    )


def _image_to_scalar_for_blend(image: Any) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        rgb = np.asarray(arr[..., :3], dtype=float)
        finite = rgb[np.isfinite(rgb)]
        if finite.size and (float(np.nanmin(finite)) < 0.0 or float(np.nanmax(finite)) > 1.0):
            rgb = np.stack([_robust_display_image(rgb[..., idx]) for idx in range(3)], axis=-1)
        rgb = np.clip(np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        return np.asarray(
            0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2], dtype=np.float32
        )
    plane = _registration_preview_plane(arr)
    if np.iscomplexobj(plane):
        plane = np.abs(plane)
    return np.asarray(plane, dtype=np.float32)


def _histogram_equalized_unit(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=float)
    out = np.zeros(arr.shape, dtype=np.float32)
    finite = np.isfinite(arr)
    values = arr[finite]
    if values.size == 0:
        return out
    if values.size == 1 or float(np.nanmax(values)) <= float(np.nanmin(values)):
        return out
    lo, hi = np.percentile(values, [0.5, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(values))
        hi = float(np.nanmax(values))
    if hi <= lo:
        return out
    clipped = np.clip(values, lo, hi)
    unique, inverse, counts = np.unique(clipped, return_inverse=True, return_counts=True)
    if unique.size <= 1:
        return out
    cumulative_before = np.concatenate([[0], np.cumsum(counts[:-1])]).astype(float)
    ranks_by_value = cumulative_before / max(1.0, float(clipped.size - 1))
    ranks = ranks_by_value[inverse]
    # Keep display headroom for overlay-slider changes across flat regions.
    equalized = 0.96 * ranks
    out[finite] = equalized.astype(np.float32, copy=False)
    return out


def _visible_roi_slice_masks_for_display(
    rois: Sequence[ROI],
    *,
    display_shape: tuple[int, int],
    indices: Mapping[str, int] | None = None,
    plane_dims: Sequence[str] = ("x", "y"),
) -> list[tuple[ROI, np.ndarray]]:
    shape = tuple(int(v) for v in display_shape)
    index_map = dict(indices or {})
    out: list[tuple[ROI, np.ndarray]] = []
    for roi in rois:
        metadata_indices = dict((roi.metadata or {}).get("slice_indices") or {})
        sliced_dims = {"x", "y", "z"} - {str(dim) for dim in plane_dims}
        slice_dim = next(iter(sliced_dims), "z")
        if slice_dim in metadata_indices and slice_dim in index_map:
            try:
                if int(metadata_indices[slice_dim]) != int(index_map[slice_dim]):
                    continue
            except Exception:
                _LOG.debug("Ignored optional GUI operation failure", exc_info=True)
        try:
            mask = np.asarray(
                mask_slice_for_display(
                    roi.mask,
                    dims=roi.dims,
                    indices=index_map,
                    plane_dims=plane_dims,
                ),
                dtype=bool,
            )
        except Exception:
            continue
        if tuple(mask.shape) == shape and np.any(mask):
            out.append((roi, mask))
    return out


def _label_mask_from_visible_roi_slice_masks(
    visible: Sequence[tuple[ROI, np.ndarray]],
    *,
    label_values: Mapping[str, int] | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if not visible:
        return np.zeros((0, 0), dtype=np.uint16), []
    shape = tuple(int(v) for v in np.asarray(visible[0][1]).shape)
    values = {
        roi.name: max(1, int((label_values or {}).get(roi.name, idx)))
        for idx, (roi, _mask) in enumerate(visible, start=1)
    }
    max_label = max(values.values(), default=0)
    dtype = np.uint16 if max_label <= np.iinfo(np.uint16).max else np.int32
    label_mask = np.zeros(shape, dtype=dtype)
    labels: list[dict[str, Any]] = []
    for idx, (roi, mask) in enumerate(visible, start=1):
        arr = np.asarray(mask) != 0
        if tuple(arr.shape) != shape:
            continue
        if not np.any(arr):
            continue
        label = values.get(roi.name, idx)
        label_mask[arr] = label
        labels.append(
            {
                "label": int(label),
                "name": roi.name,
                "voxel_count": int(np.count_nonzero(arr)),
                "metadata": dict(roi.metadata),
            }
        )
    return label_mask, labels


def _side_by_side_images(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = _robust_display_image(left)
    right = _robust_display_image(right)
    gap = max(2, int(round(min(left.shape) * 0.02)))
    fill = np.nanmin([np.nanmin(left), np.nanmin(right)]) if left.size and right.size else 0.0
    spacer = np.full((left.shape[0], gap), fill, dtype=float)
    return np.concatenate([left, spacer, right], axis=1)


def _robust_display_image(image: np.ndarray) -> np.ndarray:
    """Scale one comparison pane independently for display."""

    arr = np.asarray(image, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=float)
    lo, hi = np.percentile(finite, [1.0, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(finite))
        hi = float(np.nanmax(finite))
    if hi <= lo:
        return np.zeros_like(arr, dtype=float)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
