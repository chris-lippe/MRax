"""ROI mask persistence for the MRax GUI."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from mrax.analysis import ROI, match_spatial_mask

MASK_SET_VERSION = 1


@dataclass(frozen=True)
class MaskRecord:
    """Serializable ROI mask metadata used by the GUI."""

    roi: ROI
    color: str = "#2dd4bf"
    source_experiment: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _coerce_roi(item: ROI | MaskRecord) -> tuple[ROI, dict[str, Any]]:
    if isinstance(item, MaskRecord):
        metadata = dict(item.metadata)
        metadata.setdefault("color", item.color)
        if item.source_experiment is not None:
            metadata.setdefault("source_experiment", item.source_experiment)
        return item.roi, metadata
    return item, dict(item.metadata)


def _paths(path: str | Path) -> tuple[Path, Path]:
    base = Path(path)
    if base.suffix == ".json":
        return base.with_suffix(".npz"), base
    if base.suffix == ".npz":
        return base, base.with_suffix(".json")
    return base.with_suffix(".npz"), base.with_suffix(".json")


def save_roi_set(
    rois: Sequence[ROI | MaskRecord],
    path: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Save ROI masks as ``.npz`` data plus sidecar JSON metadata.

    Returns ``(npz_path, json_path)``.
    """

    npz_path, json_path = _paths(path)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, np.ndarray] = {}
    entries: list[dict[str, Any]] = []
    for idx, item in enumerate(rois):
        roi, extra = _coerce_roi(item)
        key = f"mask_{idx}"
        mask = np.asarray(roi.mask)
        arrays[key] = mask
        entries.append(
            {
                "name": roi.name,
                "key": key,
                "shape": list(mask.shape),
                "dims": list(roi.dims) if roi.dims is not None else None,
                "metadata": extra,
            }
        )

    np.savez_compressed(npz_path, **arrays)
    payload = {
        "version": MASK_SET_VERSION,
        "metadata": dict(metadata or {}),
        "rois": entries,
        "npz": npz_path.name,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return npz_path, json_path


def load_roi_set(path: str | Path) -> list[ROI]:
    """Load ROI masks saved by :func:`save_roi_set`.

    Passing either the ``.npz`` or ``.json`` sidecar path is supported.
    """

    npz_path, json_path = _paths(path)
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if "npz" in payload:
            npz_path = json_path.parent / payload["npz"]
        entries = list(payload.get("rois", []))
    else:
        entries = []

    with np.load(npz_path, allow_pickle=False) as data:
        if not entries:
            entries = [
                {
                    "name": key,
                    "key": key,
                    "dims": None,
                    "metadata": {},
                }
                for key in data.files
            ]
        rois: list[ROI] = []
        for entry in entries:
            key = entry["key"]
            mask = np.asarray(data[key])
            dims = entry.get("dims")
            rois.append(
                ROI(
                    str(entry.get("name") or key),
                    mask,
                    dims=tuple(dims) if dims else None,
                    metadata=dict(entry.get("metadata") or {}),
                )
            )
    return rois


def load_mask_array(path: str | Path, *, key: str | None = None) -> np.ndarray:
    """Load a raw NumPy mask from ``.npy`` or ``.npz``."""

    loaded = np.load(Path(path), allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        with loaded as data:
            chosen = key or data.files[0]
            return np.asarray(data[chosen])
    return np.asarray(loaded)


def import_mask(
    mask: Any,
    *,
    name: str = "roi",
    spatial_shape: Sequence[int] | None = None,
    dims: Sequence[str] | None = None,
    allow_transpose: bool = True,
    metadata: dict[str, Any] | None = None,
) -> ROI:
    """Create an MRax :class:`ROI` from a raw mask array or NumPy mask file."""

    if isinstance(mask, (str, Path)):
        arr = load_mask_array(mask)
    else:
        arr = np.asarray(mask)
    roi_dims = tuple(dims) if dims is not None else None
    if spatial_shape is not None:
        try:
            arr = match_spatial_mask(arr, spatial_shape, allow_transpose=allow_transpose)
        except ValueError:
            target = tuple(int(v) for v in spatial_shape)
            if arr.ndim == 2 and len(target) >= 3 and roi_dims is not None and len(roi_dims) >= 3:
                plane_shape = target[:2]
                arr = match_spatial_mask(arr, plane_shape, allow_transpose=allow_transpose)
                roi_dims = roi_dims[:2]
            else:
                raise
    return ROI(
        name,
        np.asarray(arr),
        dims=roi_dims,
        metadata=dict(metadata or {}),
    )


def roi_by_name(rois: Iterable[ROI], name: str) -> ROI | None:
    """Return the first ROI with ``name`` from an iterable."""

    for roi in rois:
        if roi.name == name:
            return roi
    return None


def mask_slice_for_display(
    mask: Any,
    *,
    dims: Sequence[str] | None = None,
    indices: Mapping[str, int] | None = None,
    plane_dims: Sequence[str] = ("x", "y"),
) -> np.ndarray:
    """Return the displayed 2D plane from a 2D/3D spatial mask."""

    arr = np.asarray(mask)
    if arr.ndim <= 2:
        return arr
    dim_names = _mask_dims(arr.ndim, dims)
    plane = tuple(str(dim) for dim in plane_dims)
    keep_axes = [dim_names.index(dim) for dim in plane if dim in dim_names]
    if len(keep_axes) != 2:
        while arr.ndim > 2:
            arr = np.take(arr, 0, axis=0)
        return arr

    index_map = dict(indices or {})
    slicer: list[object] = []
    for axis, dim in enumerate(dim_names):
        if axis in keep_axes:
            slicer.append(slice(None))
            continue
        value = int(index_map.get(dim, 0))
        slicer.append(max(0, min(value, arr.shape[axis] - 1)))
    selected = np.asarray(arr[tuple(slicer)])
    selected_dims = [dim for axis, dim in enumerate(dim_names) if axis in keep_axes]
    if tuple(selected_dims) != plane:
        order = [selected_dims.index(dim) for dim in plane]
        selected = np.transpose(selected, order)
    return selected


def embed_mask_slice(
    mask_2d: Any,
    *,
    spatial_shape: Sequence[int],
    spatial_dims: Sequence[str] = ("x", "y", "z"),
    indices: Mapping[str, int] | None = None,
    fill: bool = False,
    plane_dims: Sequence[str] = ("x", "y"),
) -> np.ndarray:
    """Embed a 2D mask in a spatial mask, optionally filling all extra slices."""

    target = tuple(int(v) for v in spatial_shape)
    dim_names = tuple(str(dim) for dim in spatial_dims)
    if len(target) != len(dim_names):
        raise ValueError("spatial_shape and spatial_dims must describe the same number of axes")
    if len(target) <= 2:
        return np.asarray(match_spatial_mask(mask_2d, target))

    plane = tuple(str(dim) for dim in plane_dims)
    if any(dim not in dim_names for dim in plane):
        raise ValueError(f"Plane dims {plane} are not present in spatial dims {dim_names}")
    plane_shape = tuple(target[dim_names.index(dim)] for dim in plane)
    plane_mask = np.asarray(match_spatial_mask(mask_2d, plane_shape))
    out = np.zeros(target, dtype=plane_mask.dtype)
    index_map = dict(indices or {})
    slicer: list[object] = []
    for axis, dim in enumerate(dim_names):
        if dim in plane:
            slicer.append(slice(None))
        elif fill:
            slicer.append(slice(None))
        else:
            value = int(index_map.get(dim, 0))
            slicer.append(max(0, min(value, target[axis] - 1)))
    selection_dims = [dim for dim, part in zip(dim_names, slicer) if isinstance(part, slice)]
    plane_in_selection = [dim for dim in selection_dims if dim in plane]
    value = plane_mask
    if tuple(plane_in_selection) != plane:
        order = [plane.index(dim) for dim in plane_in_selection]
        value = np.transpose(value, order)
    value_shape = [value.shape[plane_in_selection.index(dim)] if dim in plane_in_selection else 1 for dim in selection_dims]
    out[tuple(slicer)] = value.reshape(tuple(value_shape))
    return out


def replace_mask_slice(
    mask: Any,
    mask_2d: Any,
    *,
    dims: Sequence[str] | None = None,
    indices: Mapping[str, int] | None = None,
    mode: str = "replace",
    plane_dims: Sequence[str] = ("x", "y"),
) -> np.ndarray:
    """Replace, add, or erase the displayed 2D plane in a spatial mask."""

    arr = np.asarray(mask).copy()
    if arr.ndim <= 2:
        plane_mask = np.asarray(match_spatial_mask(mask_2d, arr.shape))
        if mode == "add":
            out = arr.copy()
            if plane_mask.dtype == bool:
                out[np.asarray(plane_mask, dtype=bool)] = 1 if out.dtype != bool else True
            else:
                out[np.asarray(plane_mask) != 0] = plane_mask[np.asarray(plane_mask) != 0]
            return out
        if mode == "erase":
            out = arr.copy()
            out[np.asarray(plane_mask) != 0] = 0
            return out
        return plane_mask

    dim_names = _mask_dims(arr.ndim, dims)
    plane = tuple(str(dim) for dim in plane_dims)
    if any(dim not in dim_names for dim in plane):
        raise ValueError(f"Plane dims {plane} are not present in mask dims {dim_names}")
    plane_shape = tuple(arr.shape[dim_names.index(dim)] for dim in plane)
    plane_mask = np.asarray(match_spatial_mask(mask_2d, plane_shape))
    index_map = dict(indices or {})
    slicer: list[object] = []
    for axis, dim in enumerate(dim_names):
        if dim in plane:
            slicer.append(slice(None))
        else:
            value = int(index_map.get(dim, 0))
            slicer.append(max(0, min(value, arr.shape[axis] - 1)))
    key = tuple(slicer)
    if mode == "add":
        target = arr[key].copy()
        if plane_mask.dtype == bool:
            target[np.asarray(plane_mask, dtype=bool)] = 1 if target.dtype != bool else True
        else:
            target[np.asarray(plane_mask) != 0] = plane_mask[np.asarray(plane_mask) != 0]
        arr[key] = target
    elif mode == "erase":
        target = arr[key].copy()
        target[np.asarray(plane_mask) != 0] = 0
        arr[key] = target
    else:
        arr[key] = plane_mask
    return arr


def expand_mask_along_slices(
    mask: Any,
    *,
    dims: Sequence[str] | None = None,
    slice_dim: str = "z",
    source_index: int | None = None,
    target_indices: Sequence[int] | None = None,
) -> np.ndarray:
    """Copy one mask slice across selected slices of a 3D mask."""

    arr = np.asarray(mask, dtype=bool).copy()
    dim_names = _mask_dims(arr.ndim, dims)
    if slice_dim not in dim_names:
        return arr
    axis = dim_names.index(slice_dim)
    size = int(arr.shape[axis])
    if size <= 0:
        return arr

    if source_index is None:
        moved = np.moveaxis(arr, axis, 0)
        nonempty = np.flatnonzero(np.any(moved, axis=tuple(range(1, moved.ndim))))
        source_index = int(nonempty[0]) if nonempty.size else 0
    source_index = max(0, min(int(source_index), size - 1))
    targets = range(size) if target_indices is None else [max(0, min(int(idx), size - 1)) for idx in target_indices]
    source = np.take(arr, source_index, axis=axis)
    for idx in targets:
        arr = _put_slice(arr, source, axis=axis, index=int(idx))
    return arr


def interpolate_mask_along_slices(
    mask: Any,
    *,
    dims: Sequence[str] | None = None,
    slice_dim: str = "z",
    extrapolate: bool = True,
    interpolation: str = "linear",
    extrapolation: str | None = None,
    threshold: float = 0.0,
) -> np.ndarray:
    """Interpolate a binary mask between non-empty slices.

    ``interpolation`` controls slices between annotated masks. ``extrapolation``
    controls slices before the first and after the last annotated mask. Supported
    strategies are ``"linear"``, ``"spline"``, and ``"constant"``; extrapolation
    also accepts ``"none"``. Linear and spline modes interpolate signed distance
    fields so ROI boundaries morph smoothly instead of switching voxel labels.
    """

    arr = np.asarray(mask, dtype=bool)
    dim_names = _mask_dims(arr.ndim, dims)
    if slice_dim not in dim_names:
        return arr.copy()
    axis = dim_names.index(slice_dim)
    moved = np.moveaxis(arr, axis, 0)
    if moved.ndim < 2:
        return arr.copy()
    valid = np.any(moved, axis=tuple(range(1, moved.ndim)))
    source_indices = np.flatnonzero(valid)
    if source_indices.size == 0:
        return arr.copy()
    interpolation = _normalize_z_fill_strategy(interpolation, allow_none=False)
    if extrapolation is None:
        extrapolation = "constant" if extrapolate else "none"
    extrapolation = _normalize_z_fill_strategy(extrapolation, allow_none=True)
    if source_indices.size == 1:
        if extrapolation != "none":
            expanded = np.repeat(moved[source_indices[0] : source_indices[0] + 1], moved.shape[0], axis=0)
            return np.moveaxis(expanded.astype(bool), 0, axis)
        return arr.copy()

    x = np.arange(moved.shape[0], dtype=float)
    source_fields = np.stack([_signed_distance_field(moved[index]) for index in source_indices], axis=0)
    flat_sources = source_fields.reshape((source_indices.size, -1))
    flat_out = _interpolate_flat_sources(
        x,
        source_indices.astype(float),
        flat_sources,
        interpolation=interpolation,
        extrapolation=extrapolation,
    )
    interpolated = flat_out.reshape(moved.shape) >= float(threshold)
    return np.moveaxis(interpolated, 0, axis)


def _signed_distance_field(mask: Any) -> np.ndarray:
    arr = np.asarray(mask, dtype=bool)
    if not np.any(arr):
        return -np.ones(arr.shape, dtype=float)
    if np.all(arr):
        return np.ones(arr.shape, dtype=float)
    try:
        from scipy import ndimage

        inside = ndimage.distance_transform_edt(arr)
        outside = ndimage.distance_transform_edt(~arr)
        return np.asarray(inside - outside, dtype=float)
    except Exception:
        return arr.astype(float) * 2.0 - 1.0


def _normalize_z_fill_strategy(value: str | None, *, allow_none: bool) -> str:
    strategy = str(value or "").strip().lower()
    aliases = {
        "nearest": "constant",
        "edge": "constant",
        "hold": "constant",
        "cubic": "spline",
        "none": "none",
        "off": "none",
    }
    strategy = aliases.get(strategy, strategy)
    allowed = {"linear", "spline", "constant"}
    if allow_none:
        allowed.add("none")
    return strategy if strategy in allowed else "linear"


def _interpolate_flat_sources(
    x: np.ndarray,
    source_x: np.ndarray,
    flat_sources: np.ndarray,
    *,
    interpolation: str,
    extrapolation: str,
) -> np.ndarray:
    flat_out = np.full((x.size, flat_sources.shape[1]), -np.inf, dtype=float)
    first = int(source_x[0])
    last = int(source_x[-1])
    between = (x >= source_x[0]) & (x <= source_x[-1])
    outside_left = x < source_x[0]
    outside_right = x > source_x[-1]
    flat_out[between] = _evaluate_z_strategy(x[between], source_x, flat_sources, interpolation)
    if extrapolation == "none":
        return flat_out
    if extrapolation == "constant":
        flat_out[outside_left] = flat_sources[0]
        flat_out[outside_right] = flat_sources[-1]
        return flat_out
    if extrapolation == "linear":
        left_slope = (flat_sources[1] - flat_sources[0]) / max(float(source_x[1] - source_x[0]), 1e-12)
        right_slope = (flat_sources[-1] - flat_sources[-2]) / max(float(source_x[-1] - source_x[-2]), 1e-12)
        flat_out[outside_left] = flat_sources[0] + (x[outside_left][:, None] - float(first)) * left_slope
        flat_out[outside_right] = flat_sources[-1] + (x[outside_right][:, None] - float(last)) * right_slope
        return flat_out
    flat_out[outside_left | outside_right] = _evaluate_z_strategy(
        x[outside_left | outside_right],
        source_x,
        flat_sources,
        "spline",
        extrapolate=True,
    )
    return flat_out


def _evaluate_z_strategy(
    values: np.ndarray,
    source_x: np.ndarray,
    flat_sources: np.ndarray,
    strategy: str,
    *,
    extrapolate: bool = False,
) -> np.ndarray:
    if values.size == 0:
        return np.zeros((0, flat_sources.shape[1]), dtype=float)
    if strategy == "constant":
        nearest = np.abs(values[:, None] - source_x[None, :]).argmin(axis=1)
        return flat_sources[nearest]
    if strategy == "spline":
        try:
            from scipy.interpolate import CubicSpline

            spline = CubicSpline(source_x, flat_sources, axis=0, extrapolate=bool(extrapolate))
            return np.asarray(spline(values), dtype=float)
        except Exception:
            strategy = "linear"
    out = np.empty((values.size, flat_sources.shape[1]), dtype=float)
    for col in range(flat_sources.shape[1]):
        out[:, col] = np.interp(
            values,
            source_x,
            flat_sources[:, col],
            left=float(flat_sources[0, col]),
            right=float(flat_sources[-1, col]),
        )
    return out


def stitch_masks_along_slices(
    masks: Iterable[ROI | Any],
    *,
    spatial_shape: Sequence[int],
    spatial_dims: Sequence[str] = ("x", "y", "z"),
    indices: Mapping[str, int] | None = None,
    slice_dim: str = "z",
    plane_dims: Sequence[str] = ("x", "y"),
) -> np.ndarray:
    """Combine 2D slice masks and 3D masks into one spatial mask."""

    target = tuple(int(v) for v in spatial_shape)
    dim_names = tuple(str(dim) for dim in spatial_dims)
    if len(target) != len(dim_names):
        raise ValueError("spatial_shape and spatial_dims must describe the same number of axes")

    stitched = np.zeros(target, dtype=bool)
    for item in masks:
        if isinstance(item, ROI):
            mask = np.asarray(item.mask, dtype=bool)
            mask_dims = tuple(item.dims) if item.dims is not None else None
            metadata = dict(item.metadata or {})
        else:
            mask = np.asarray(item, dtype=bool)
            mask_dims = None
            metadata = {}
        if mask.ndim == len(target):
            if mask_dims is not None and tuple(mask_dims) != dim_names:
                order = [mask_dims.index(dim) for dim in dim_names]
                mask = np.transpose(mask, order)
            if tuple(mask.shape) != target:
                mask = match_spatial_mask(mask, target)
            stitched |= np.asarray(mask, dtype=bool)
            continue
        if mask.ndim == 2 and len(target) >= 3:
            item_indices = dict(indices or {})
            item_indices.update(dict(metadata.get("slice_indices") or {}))
            stitched |= embed_mask_slice(
                mask,
                spatial_shape=target,
                spatial_dims=dim_names,
                indices=item_indices,
                plane_dims=plane_dims,
            )
            continue
        try:
            stitched |= np.broadcast_to(mask, target).astype(bool)
        except ValueError as exc:
            raise ValueError(f"Cannot stitch mask shape {mask.shape} into spatial shape {target}") from exc
    return stitched


def _mask_dims(ndim: int, dims: Sequence[str] | None) -> tuple[str, ...]:
    if dims is not None:
        dim_names = tuple(str(dim) for dim in dims)
        if len(dim_names) != int(ndim):
            raise ValueError("mask dims must match mask.ndim")
        return dim_names
    if ndim == 2:
        return ("x", "y")
    if ndim == 3:
        return ("x", "y", "z")
    return tuple(f"dim_{idx}" for idx in range(int(ndim)))


def _put_slice(arr: np.ndarray, value: np.ndarray, *, axis: int, index: int) -> np.ndarray:
    slicer: list[object] = [slice(None)] * arr.ndim
    slicer[int(axis)] = int(index)
    arr[tuple(slicer)] = value
    return arr


__all__ = [
    "MASK_SET_VERSION",
    "MaskRecord",
    "embed_mask_slice",
    "expand_mask_along_slices",
    "import_mask",
    "interpolate_mask_along_slices",
    "load_mask_array",
    "load_roi_set",
    "mask_slice_for_display",
    "replace_mask_slice",
    "roi_by_name",
    "save_roi_set",
    "stitch_masks_along_slices",
]
