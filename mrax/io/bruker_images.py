from __future__ import annotations

import warnings
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np

from mrax.data_struct import SpatialGeometry

from .bruker_parameters import (
    _as_positive_int,
    _as_string_list,
    _contains_token,
    _flatten_numeric,
    _normalize_size,
    bruker_word_dtype,
    parse_jcamp,
)

_LPS_TO_RAS = np.diag([-1.0, -1.0, 1.0])

def _bruker_orientation(visu: dict) -> list[np.ndarray]:
    vals = _flatten_numeric(visu.get("VisuCoreOrientation"))
    if not vals:
        return []
    count = len(vals) // 9
    if count <= 0:
        return []
    matrices: list[np.ndarray] = []
    for i in range(count):
        mat = np.array(vals[i * 9 : (i + 1) * 9], dtype=float).reshape(3, 3).T
        matrices.append(mat)
    return matrices


def _bruker_image_orientation_metadata(visu: dict) -> dict[str, object]:
    """Normalize decoded NumPy image axes to anatomy and acquisition plane."""

    orientations = _bruker_orientation(visu)
    if not orientations:
        return {}
    orientation = np.asarray(orientations[0], dtype=float)
    if orientation.shape != (3, 3) or not np.isfinite(orientation).all():
        return {}
    norms = np.linalg.norm(orientation, axis=0)
    if np.any(norms <= 1e-8):
        return {}
    dominant = np.argmax(np.abs(orientation / norms[None, :]), axis=0)
    vendor_labels = tuple("xyz"[int(index)] for index in dominant)
    if len(set(vendor_labels)) != 3:
        return {}
    first_axis, second_axis = _bruker_decoded_in_plane_axis_order(visu)
    labels = (
        vendor_labels[first_axis],
        vendor_labels[second_axis],
        vendor_labels[2],
    )
    native_plane = {"x": "sagittal", "y": "coronal", "z": "axial"}[labels[2]]
    return {
        "image_axis_anatomical_labels": list(labels),
        "native_image_plane": native_plane,
    }


def _bruker_decoded_in_plane_axis_order(visu: dict) -> tuple[int, int]:
    """Return Bruker orientation columns matching decoded NumPy axes 0 and 1."""

    dim_val = visu.get("VisuCoreDim")
    dim = int(dim_val) if isinstance(dim_val, (int, float, np.generic)) else 0
    size_vals = _normalize_size(visu.get("VisuCoreSize"), dim=dim if dim > 0 else None)
    if dim >= 3 or len(size_vals) >= 3:
        return (0, 1)
    frame_count = _as_positive_int(visu.get("VisuCoreFrameCount"), 1) or 1
    frames_are_slices = (
        len(size_vals) == 2
        and frame_count > 1
        and _contains_token(visu.get("VisuFGOrderDesc"), "FG_SLICE")
        and not bool(visu.get("_mrax_mixed_slice_geometry", False))
    )
    reshape_order = "F" if _contains_token(visu.get("VisuFGOrderDesc"), "FG_DIFFUSION") else "C"
    if frames_are_slices and size_vals[0] != size_vals[1]:
        reshape_order = "F"
    swapped = reshape_order == "C"

    transpositions = _bruker_frame_transpositions(
        visu.get("_mrax_frame_transposition", visu.get("VisuCoreTransposition")),
        frame_count,
    )
    if transpositions.size and int(transpositions[0]) > 0:
        swapped = not swapped
    return (1, 0) if swapped else (0, 1)


def _bruker_position(visu: dict) -> list[np.ndarray]:
    vals = _flatten_numeric(visu.get("VisuCorePosition"))
    if not vals:
        return []
    count = len(vals) // 3
    if count <= 0:
        return []
    vectors: list[np.ndarray] = []
    for i in range(count):
        vec = np.array(vals[i * 3 : (i + 1) * 3], dtype=float)
        vectors.append(vec)
    return vectors


def _bruker_frame_transpositions(value: object, frame_count: int) -> np.ndarray:
    """Broadcast Bruker frame transposition metadata to one value per frame."""

    count = max(int(frame_count), 0)
    if count == 0:
        return np.zeros((0,), dtype=int)

    vals = _flatten_numeric(value)
    if not vals:
        return np.zeros((count,), dtype=int)

    arr = np.asarray(vals, dtype=int).ravel()
    if arr.size == 1:
        return np.full((count,), int(arr[0]), dtype=int)
    if arr.size >= count:
        return arr[:count]
    if count % int(arr.size) == 0:
        return np.tile(arr, count // int(arr.size))
    pad_val = int(arr[-1]) if arr.size > 0 else 0
    return np.pad(arr, (0, count - arr.size), constant_values=pad_val)


def _visu_fg_order_entries(visu: dict) -> list[list[object]]:
    """Return parsed ``VisuFGOrderDesc`` entries as ``[count, kind, ...]`` lists."""

    value = visu.get("VisuFGOrderDesc")
    if value is None:
        return []
    if isinstance(value, list) and value and all(isinstance(item, list) for item in value):
        return [list(item) for item in value if item]
    if isinstance(value, list) and len(value) >= 2:
        if len(value) <= 5 and isinstance(value[0], (int, float)) and isinstance(value[1], str):
            return [list(value)]
        entries: list[list[object]] = []
        idx = 0
        while idx + 1 < len(value):
            if isinstance(value[idx], (int, float)) and isinstance(value[idx + 1], str):
                entries.append(list(value[idx : idx + 5]))
                idx += 5
            else:
                idx += 1
        return entries
    return []


def _visu_fg_count(visu: dict, token: str) -> int | None:
    needle = token.upper()
    for entry in _visu_fg_order_entries(visu):
        if len(entry) >= 2 and needle in str(entry[1]).upper():
            try:
                count = int(round(float(entry[0])))
            except (TypeError, ValueError):
                return None
            return count if count > 0 else None
    return None


def _bruker_has_mixed_slice_geometry(visu: dict, frame_count: int) -> bool:
    """Detect 2D frame groups whose frames do not share one slice-stack geometry."""

    if int(frame_count) <= 1 or not _contains_token(visu.get("VisuFGOrderDesc"), "FG_SLICE"):
        return False

    orientations = _bruker_orientation(visu)
    if orientations:
        if len(orientations) == 1:
            orientations = orientations * int(frame_count)
        if len(orientations) >= int(frame_count):
            unique_orientations = {
                tuple(np.round(np.asarray(mat, dtype=float), 6).ravel().tolist())
                for mat in orientations[: int(frame_count)]
            }
            if len(unique_orientations) > 1:
                return True

    transpositions = _bruker_frame_transpositions(visu.get("VisuCoreTransposition"), int(frame_count))
    return bool(np.unique(transpositions).size > 1)


def _apply_framewise_transposition(data: np.ndarray | None, transpositions: Sequence[int] | None) -> np.ndarray | None:
    """Apply Bruker frame-wise image transposition metadata to a frame stack."""

    if data is None or transpositions is None:
        return data

    arr = np.asarray(data)
    if arr.ndim < 3:
        return arr

    trans = np.asarray(transpositions, dtype=int).ravel()
    if trans.size != int(arr.shape[0]) or not np.any(trans > 0):
        return arr

    unique = np.unique(trans)
    if unique.size == 1 and int(unique[0]) > 0:
        return np.swapaxes(arr, -2, -1)
    if arr.shape[-2] != arr.shape[-1]:
        warnings.warn(
            "Skipping mixed frame-wise transposition for non-square Bruker frames; "
            "a single array shape cannot represent both transposed and non-transposed frames.",
            RuntimeWarning,
            stacklevel=2,
        )
        return arr

    out = np.array(arr, copy=True)
    for frame_idx, trans_val in enumerate(trans.tolist()):
        if int(trans_val) > 0:
            out[frame_idx] = np.swapaxes(out[frame_idx], -2, -1)
    return out


def _bruker_voxel_sizes(visu: dict) -> tuple[float, float, float]:
    dim_val = visu.get("VisuCoreDim")
    dim = int(dim_val) if isinstance(dim_val, (int, float, np.generic)) else 0
    size_vals = _normalize_size(visu.get("VisuCoreSize"), dim=dim if dim > 0 else None)
    extent_vals = _normalize_size(visu.get("VisuCoreExtent"), dim=dim if dim > 0 else None)
    thickness_vals = _normalize_size(visu.get("VisuCoreFrameThickness"), dim=1)
    is_multislice_2d = dim == 2 or len(size_vals) == 2 or len(extent_vals) == 2
    slice_spacing = _bruker_slice_spacing_from_positions(visu) if is_multislice_2d else None

    if is_multislice_2d:
        if len(size_vals) == 1:
            size_vals = size_vals * 2
        if len(extent_vals) == 1:
            extent_vals = extent_vals * 2
        vx = (
            extent_vals[0] / size_vals[0]
            if extent_vals and size_vals and size_vals[0]
            else 1.0
        )
        vy = (
            extent_vals[1] / size_vals[1]
            if len(extent_vals) > 1 and len(size_vals) > 1 and size_vals[1]
            else 1.0
        )
        vz = (
            slice_spacing
            if slice_spacing is not None
            else thickness_vals[0]
            if thickness_vals
            else 1.0
        )
        return (float(vx), float(vy), float(vz))

    if len(size_vals) < 3:
        if len(size_vals) == 1:
            size_vals = size_vals * 3
        elif len(size_vals) == 2:
            size_vals = size_vals + [1.0]
    if len(extent_vals) < 3:
        if len(extent_vals) == 1:
            extent_vals = extent_vals * 3
        elif len(extent_vals) == 2:
            extra = thickness_vals[0] if thickness_vals else extent_vals[-1]
            extent_vals = extent_vals + [extra]

    vox = []
    for i in range(3):
        denom = size_vals[i] if i < len(size_vals) else 1.0
        vox.append(extent_vals[i] / denom if denom else 1.0)
    return (float(vox[0]), float(vox[1]), float(vox[2]))


def _bruker_slice_spacing_from_positions(visu: dict) -> float | None:
    """Return center-to-center slice spacing when Bruker exports slice positions."""

    positions = _bruker_position(visu)
    orientations = _bruker_orientation(visu)
    if len(positions) < 2 or not orientations:
        return None
    normal = np.asarray(orientations[0], dtype=float)[:, 2]
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm <= 1e-8:
        return None
    coordinates = np.asarray(positions, dtype=float) @ (normal / normal_norm)
    unique_coordinates = np.unique(np.round(coordinates, decimals=6))
    if unique_coordinates.size < 2:
        return None
    differences = np.diff(np.sort(unique_coordinates))
    differences = differences[np.isfinite(differences) & (differences > 1e-6)]
    if differences.size == 0:
        return None
    return float(np.median(differences))


def bruker_affine_from_visu(visu: dict, *, frame_index: int | None = None) -> np.ndarray | list[np.ndarray]:
    orientations = _bruker_orientation(visu)
    positions = _bruker_position(visu)
    voxel_sizes = _bruker_voxel_sizes(visu)

    if not orientations:
        orientations = [np.eye(3)]
    if not positions:
        positions = [np.zeros(3)]

    frame_count = max(len(orientations), len(positions))
    if len(orientations) == 1:
        orientations = orientations * frame_count
    if len(positions) == 1:
        positions = positions * frame_count

    def _build_affine(orientation: np.ndarray, position: np.ndarray) -> np.ndarray:
        affine = np.eye(4, dtype=float)
        affine[:3, :3] = (_LPS_TO_RAS @ orientation) @ np.diag(voxel_sizes)
        affine[:3, 3] = _LPS_TO_RAS @ position
        return affine

    affines = [_build_affine(orientations[i], positions[i]) for i in range(frame_count)]
    if frame_index is not None:
        if frame_index < 0 or frame_index >= len(affines):
            raise IndexError("frame_index out of range for VisuCorePosition/Orientation")
        return affines[frame_index]
    return affines[0] if len(affines) == 1 else affines


def bruker_spatial_geometry(
    visu: dict,
    *,
    spatial_shape: Sequence[int] | None = None,
) -> SpatialGeometry:
    """Return Bruker image geometry without changing native image/k-space axes."""

    vendor_voxel_sizes = _bruker_voxel_sizes(visu)
    first_axis, second_axis = _bruker_decoded_in_plane_axis_order(visu)
    voxel_sizes = (
        vendor_voxel_sizes[first_axis],
        vendor_voxel_sizes[second_axis],
        vendor_voxel_sizes[2],
    )
    orientations = _bruker_orientation(visu)
    spatial_ndim = len(tuple(spatial_shape or ())) or 3
    orientation_metadata = _bruker_image_orientation_metadata(visu)
    raw_labels = orientation_metadata.get("image_axis_anatomical_labels")
    axis_labels = (
        tuple(str(value) for value in raw_labels[:spatial_ndim])
        if isinstance(raw_labels, list) and len(raw_labels) >= spatial_ndim
        else None
    )
    if not orientations:
        return SpatialGeometry(
            spatial_shape=spatial_shape,
            voxel_sizes=voxel_sizes[:spatial_ndim],
            axis_anatomical_labels=axis_labels,
            source="bruker",
            status="unknown",
        )
    affines = bruker_affine_from_visu(visu)
    return SpatialGeometry.from_affines(
        affines,
        spatial_shape=spatial_shape,
        voxel_sizes=voxel_sizes[:spatial_ndim],
        axis_anatomical_labels=axis_labels,
        source="bruker",
        status="known" if _bruker_position(visu) else "partial",
    )


def _dtype_from_word(word: str) -> np.dtype:
    return bruker_word_dtype(word)


def _bruker_visu_frame_scaling_values(value: object, *, frame_count: int, default: float) -> list[float]:
    values = _flatten_numeric(value)
    if not values:
        values = [default]
    if len(values) == 1:
        values = values * frame_count
    elif len(values) != frame_count and frame_count == 1:
        values = [values[-1]]
    return [float(v) for v in values]


def load_2dseq(pdata_dir: Path, *, image_type: str = "auto") -> Tuple[np.ndarray, dict, tuple[int, ...]]:
    visu = parse_jcamp(pdata_dir / "visu_pars")
    word = visu["VisuCoreWordType"]
    dtype = np.dtype(_dtype_from_word(word))
    if visu.get("VisuCoreByteOrder", "").lower() == "bigendian":
        dtype = dtype.newbyteorder(">")
    size = visu["VisuCoreSize"]
    visu_dim = _as_positive_int(visu.get("VisuCoreDim"), None)
    if isinstance(size, (int, float)):
        size = [int(size)]
    if isinstance(size, list):
        if len(size) == 1 and visu_dim != 1:
            size = [size[0], size[0]]
        spatial = [int(s) for s in size[-3:] if int(s) > 0]
    else:
        spatial = []
    if not spatial and isinstance(size, list):
        spatial = [int(s) for s in size[:3] if int(s) > 0]
    if len(spatial) == 1 and visu_dim != 1:
        spatial = [spatial[0], spatial[0]]
    spatial_shape = tuple(spatial[-3:]) if len(spatial) >= 2 else tuple(spatial)
    raw_spatial_shape = spatial_shape
    frame_count = int(visu["VisuCoreFrameCount"])
    slopes = _bruker_visu_frame_scaling_values(
        visu.get("VisuCoreDataSlope", [1.0]),
        frame_count=frame_count,
        default=1.0,
    )
    offsets = _bruker_visu_frame_scaling_values(
        visu.get("VisuCoreDataOffs", [0.0]),
        frame_count=frame_count,
        default=0.0,
    )

    def _fg_order_contains(token: str) -> bool:
        return _contains_token(visu.get("VisuFGOrderDesc"), token)

    mixed_slice_geometry = _bruker_has_mixed_slice_geometry(visu, frame_count)
    frames_are_slices = (
        len(raw_spatial_shape) == 2
        and frame_count > 1
        and _fg_order_contains("FG_SLICE")
        and not mixed_slice_geometry
    )

    raw = np.fromfile(pdata_dir / "2dseq", dtype=dtype)
    prod = int(np.prod(spatial_shape))
    expected = frame_count * prod
    if raw.size != expected:
        raise ValueError(
            f"2dseq size mismatch: found {raw.size} values, expected {expected} "
            f"from VisuCoreFrameCount={frame_count} and VisuCoreSize={raw_spatial_shape}."
        )
    raw = raw.reshape((frame_count, prod))
    scaled = np.empty_like(raw, dtype=np.float32)
    for i in range(frame_count):
        scaled[i] = raw[i] * float(slopes[i]) + float(offsets[i])

    frame_types = [entry.upper() for entry in _as_string_list(visu.get("VisuCoreFrameType"))]
    requested_type = str(image_type or "auto").lower()
    if requested_type not in ("auto", "complex", "real", "reel", "magnitude", "phase", "imaginary"):
        raise ValueError("image_type must be one of: auto, complex, real, magnitude, phase, imaginary")

    if requested_type == "imaginary":
        scaled = (1j * scaled).astype(np.complex64)
    elif requested_type == "complex" or (
        requested_type == "auto"
        and frame_count % 2 == 0
        and (
            "COMPLEX_IMAGE" in frame_types
            or _fg_order_contains("FG_COMPLEX")
            or ("REAL_IMAGE" in frame_types and "IMAGINARY_IMAGE" in frame_types)
        )
    ):
        if _fg_order_contains("FG_COMPLEX") and frame_count % 2 == 0:
            real_first = not (frame_types[:2] == ["IMAGINARY_IMAGE", "REAL_IMAGE"])
            real = scaled[0::2] if real_first else scaled[1::2]
            imag = scaled[1::2] if real_first else scaled[0::2]
        else:
            half = frame_count // 2
            real = scaled[:half]
            imag = scaled[half : 2 * half]
        scaled = (real + 1j * imag).astype(np.complex64)
        frame_count = int(scaled.shape[0])

    reshape_order = "F" if _fg_order_contains("FG_DIFFUSION") else "C"
    transpositions = _bruker_frame_transpositions(visu.get("VisuCoreTransposition"), frame_count)
    if len(raw_spatial_shape) == 1:
        data = scaled.reshape((frame_count, int(raw_spatial_shape[0])), order=reshape_order)
        spatial_shape = (int(raw_spatial_shape[0]),)
    elif len(raw_spatial_shape) == 2:
        frame_spatial_shape = raw_spatial_shape if frames_are_slices else (raw_spatial_shape[1], raw_spatial_shape[0])
        x, y = frame_spatial_shape
        if frames_are_slices and len(frame_spatial_shape) >= 2 and int(frame_spatial_shape[0]) != int(frame_spatial_shape[1]):
            # Bruker 2D slice stacks, like 3D 2dseq volumes, store the read
            # dimension fastest. C-order corrupts non-square slices by
            # interleaving read/phase samples into the displayed image plane;
            # keep square stacks on the historical path because C/F ordering is
            # otherwise ambiguous and existing datasets depend on that layout.
            reshape_order = "F"
        frames = scaled.reshape((frame_count, x, y), order=reshape_order)
        frames = _apply_framewise_transposition(frames, transpositions)
        if frames_are_slices:
            # Treat frames as z only when all frames share one geometry. If Bruker
            # records another FG dimension, keep that dimension as the MRax frame axis.
            slice_count = _visu_fg_count(visu, "FG_SLICE")
            if (
                slice_count is not None
                and 0 < int(slice_count) < int(frame_count)
                and int(frame_count) % int(slice_count) == 0
            ):
                grouped_frame_count = int(frame_count) // int(slice_count)
                entries = _visu_fg_order_entries(visu)
                counts = [int(round(float(entry[0]))) for entry in entries if entry]
                slice_entry = next(
                    (
                        idx
                        for idx, entry in enumerate(entries)
                        if len(entry) >= 2 and "FG_SLICE" in str(entry[1]).upper()
                    ),
                    None,
                )
                if (
                    slice_entry is not None
                    and counts
                    and int(np.prod(counts)) == int(frame_count)
                ):
                    group_shape = tuple(reversed(counts))
                    grouped = frames.reshape(group_shape + tuple(frames.shape[1:]))
                    slice_axis = len(counts) - 1 - int(slice_entry)
                    grouped = np.moveaxis(grouped, slice_axis, len(counts) - 1)
                    grouped = grouped.reshape(
                        (grouped_frame_count, int(slice_count)) + tuple(frames.shape[1:])
                    )
                else:
                    # Bruker frame-group descriptors list the fastest-varying
                    # group first, so FG_SLICE occupies the inner axis.
                    grouped = frames.reshape(
                        (grouped_frame_count, int(slice_count)) + tuple(frames.shape[1:])
                    )
                data = np.moveaxis(grouped, 1, -1)
                spatial_shape = tuple(int(v) for v in frames.shape[1:]) + (int(slice_count),)
                frame_count = grouped_frame_count
            else:
                data = np.moveaxis(frames, 0, -1)[None, ...]
                spatial_shape = tuple(int(v) for v in frames.shape[1:]) + (int(frame_count),)
            if str(visu.get("VisuCoreDiskSliceOrder", "")).lower() == "disk_reverse_slice_order":
                data = np.flip(data, axis=-1)
        else:
            data = frames
            spatial_shape = tuple(int(v) for v in frames.shape[1:])
    elif len(spatial_shape) == 3:
        x, y, z = spatial_shape
        # 3D Bruker 2dseq volumes are stored with the read dimension varying
        # fastest. NumPy's C-order reshape would make z vary fastest and
        # interleave partitions into a grid-like slice artifact.
        data = scaled.reshape((frame_count, x, y, z), order="F")
        if str(visu.get("VisuCoreDiskSliceOrder", "")).lower() == "disk_reverse_slice_order":
            data = np.flip(data, axis=-1)
    else:
        x, y = spatial_shape
        data = scaled.reshape((frame_count, x, y), order=reshape_order)

    visu["_mrax_mixed_slice_geometry"] = bool(mixed_slice_geometry)
    visu["_mrax_frame_transposition"] = transpositions.tolist()
    visu["_mrax_fg_order_entries"] = _visu_fg_order_entries(visu)

    return data, visu, spatial_shape

def _bruker_visu_scaling_metadata(visu: dict, *, frame_count: int) -> dict:
    slopes = _bruker_visu_frame_scaling_values(
        visu.get("VisuCoreDataSlope", [1.0]),
        frame_count=frame_count,
        default=1.0,
    )
    offsets = _bruker_visu_frame_scaling_values(
        visu.get("VisuCoreDataOffs", [0.0]),
        frame_count=frame_count,
        default=0.0,
    )
    slope_arr = np.asarray(slopes, dtype=np.float64)
    offset_arr = np.asarray(offsets, dtype=np.float64)
    return {
        "source": "visu_pars",
        "parameters": ["VisuCoreDataSlope", "VisuCoreDataOffs"],
        "applied_to": "DataSet.data",
        "formula": "scaled = stored_2dseq * VisuCoreDataSlope + VisuCoreDataOffs",
        "stored_frame_count": int(frame_count),
        "slope_min": float(np.min(slope_arr)),
        "slope_max": float(np.max(slope_arr)),
        "offset_min": float(np.min(offset_arr)),
        "offset_max": float(np.max(offset_arr)),
        "framewise": bool(
            slope_arr.size > 1
            and (not np.allclose(slope_arr, slope_arr[0]) or not np.allclose(offset_arr, offset_arr[0]))
        ),
    }


"""Bruker images translation; MRI reconstruction is separate."""

