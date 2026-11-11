from __future__ import annotations

from dataclasses import replace
from itertools import permutations
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np

from mrax.data_struct import SpatialGeometry

from .bruker_frames import (
    frame_group_entries,
    frame_parameter,
    output_frame_grid,
    package_plane,
    slice_packages,
)
from .bruker_parameters import (
    _as_string_list,
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
    if len(vals) % 9:
        raise ValueError("VisuCoreOrientation must contain complete 3 by 3 matrices")
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

    if visu.get("_mrax_package_frame_indices"):
        return {"native_image_plane": visu["_mrax_slice_packages"][0]["native_image_plane"]}
    orientations = _bruker_orientation(visu)
    if not orientations:
        return {}
    grid = visu.get("_mrax_output_frame_indices")
    index = int(grid[0][0]) if grid else 0
    expanded = frame_parameter(visu, "VisuCoreOrientation", 9)
    orientation = expanded[index].reshape(3, 3).T
    if orientation.shape != (3, 3) or not np.isfinite(orientation).all():
        return {}
    norms = np.linalg.norm(orientation, axis=0)
    if np.any(norms <= 1e-8):
        return {}
    unit = np.abs(orientation / norms[None, :])
    dominant = max(permutations(range(3)), key=lambda axes: sum(unit[axis, i] for i, axis in enumerate(axes)))
    vendor_labels = tuple("xyz"[int(index)] for index in dominant)
    if len(set(vendor_labels)) != 3:
        return {}
    labels = vendor_labels
    native_plane = {"x": "sagittal", "y": "coronal", "z": "axial"}[labels[2]]
    return {
        "image_axis_anatomical_labels": list(labels),
        "native_image_plane": native_plane,
    }


def _bruker_position(visu: dict) -> list[np.ndarray]:
    vals = _flatten_numeric(visu.get("VisuCorePosition"))
    if not vals:
        return []
    if len(vals) % 3:
        raise ValueError("VisuCorePosition must contain complete patient-coordinate vectors")
    count = len(vals) // 3
    if count <= 0:
        return []
    vectors: list[np.ndarray] = []
    for i in range(count):
        vec = np.array(vals[i * 3 : (i + 1) * 3], dtype=float)
        vectors.append(vec)
    return vectors




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
    """Map decoded pixel centres to RAS, once per output volume rather than slice."""
    cached_grid = visu.get("_mrax_output_frame_indices")
    grid = np.asarray(cached_grid, dtype=int) if cached_grid is not None else output_frame_grid(
        visu, visu.get("_mrax_selected_package"))[0]
    orientations = frame_parameter(visu, "VisuCoreOrientation", 9, default=np.eye(3).ravel())
    positions = frame_parameter(visu, "VisuCorePosition", 3, default=np.zeros(3))
    voxel_sizes = _bruker_voxel_sizes(visu)
    affines = []
    for row in grid:
        first = int(row[0])
        direction = orientations[first].reshape(3, 3).T
        if not np.isfinite(direction).all() or not np.allclose(direction.T @ direction, np.eye(3), atol=1e-3):
            raise ValueError("Bruker image orientation must describe orthonormal physical axes")
        affine = np.eye(4)
        affine[:3, :3] = _LPS_TO_RAS @ direction @ np.diag(voxel_sizes)
        affine[:3, 3] = _LPS_TO_RAS @ positions[first]
        if int(visu.get("VisuCoreDim", 2)) == 2:
            package = int(visu.get("_mrax_selected_package") or 0)
            distances = _flatten_numeric(visu.get("VisuCoreSlicePacksSliceDist"))
            thicknesses = _flatten_numeric(visu.get("VisuCoreFrameThickness"))
            step_size = distances[min(package, len(distances) - 1)] if distances else 0
            if step_size <= 0:
                step_size = thicknesses[0] if thicknesses else 1.
            affine[:3, 2] = _LPS_TO_RAS @ direction[:, 2] * step_size
        if len(row) > 1 and _bruker_position(visu):
            centres = positions[row]
            step = centres[1] - centres[0]
            if np.linalg.norm(step) <= 1e-8 or not np.allclose(
                centres, centres[0] + np.arange(len(row))[:, None] * step, atol=1e-4,
            ):
                raise ValueError("Slice centres do not form a regular spatial stack")
            if abs(float(step @ direction[:, 2])) <= 1e-8:
                raise ValueError("Slice centres do not advance along their orientation normal")
            if not np.allclose(orientations[row], orientations[first], atol=1e-5):
                raise ValueError("A spatial slice stack must share one orientation")
            affine[:3, 2] = _LPS_TO_RAS @ step
        affines.append(affine)
    if frame_index is not None:
        if frame_index < 0 or frame_index >= len(affines):
            raise IndexError("frame_index out of range for the decoded Bruker volumes")
        return affines[frame_index]
    return affines[0] if len(affines) == 1 else affines


def bruker_spatial_geometry(
    visu: dict,
    *,
    spatial_shape: Sequence[int] | None = None,
) -> SpatialGeometry:
    """Native labels/spacing with the matching NIfTI-boundary RAS affine."""
    package_grids = visu.get("_mrax_package_frame_indices")
    if package_grids:
        shapes, affines = [], []
        for package, grid in enumerate(package_grids):
            selected = {
                **visu,
                "_mrax_package_frame_indices": None,
                "_mrax_output_frame_indices": grid,
                "_mrax_selected_package": package,
            }
            shape = tuple(spatial_shape[:2]) + (len(grid[0]),)
            geometry = bruker_spatial_geometry(selected, spatial_shape=shape)
            for frame in range(len(grid)):
                affines.append(geometry.affine_for_frame(frame))
                shapes.append(shape)
        return replace(
            SpatialGeometry.from_affines(affines, spatial_shape=spatial_shape, source="bruker"),
            frame_shapes=tuple(shapes),
        )
    spatial_ndim = len(tuple(spatial_shape or ())) or 3
    metadata = _bruker_image_orientation_metadata(visu)
    labels = metadata.get("image_axis_anatomical_labels")
    axis_labels = tuple(labels[:spatial_ndim]) if isinstance(labels, list) else None
    if not _bruker_orientation(visu):
        return SpatialGeometry(
            spatial_shape=spatial_shape, voxel_sizes=_bruker_voxel_sizes(visu)[:spatial_ndim],
            axis_anatomical_labels=axis_labels, source="bruker", status="unknown",
        )
    affines = bruker_affine_from_visu(visu)
    first = np.asarray(affines[0] if isinstance(affines, list) else affines)
    spacing = tuple(np.linalg.norm(first[:3, :3], axis=0)[:spatial_ndim])
    # SpatialGeometry maps the array at MRax's NIfTI boundary, which reverses
    # native spatial axes. Keep native pixel order and describe that boundary
    # explicitly rather than attaching an XYZ affine to a ZYX export.
    def nifti_affine(matrix):
        matrix = np.asarray(matrix).copy()
        order = tuple(reversed(range(spatial_ndim))) + tuple(range(spatial_ndim, 3))
        matrix[:3, :3] = matrix[:3, :3][:, order]
        return matrix

    affines = [nifti_affine(matrix) for matrix in affines] if isinstance(affines, list) else nifti_affine(affines)
    return SpatialGeometry.from_affines(
        affines, spatial_shape=spatial_shape, voxel_sizes=spacing,
        axis_anatomical_labels=axis_labels, source="bruker",
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


def load_2dseq(
    pdata_dir: Path, *, image_type: str = "auto", slice_package: int | None = None,
) -> Tuple[np.ndarray, dict, tuple[int, ...]]:
    """Decode X-fastest pixels and explicit frame groups into native image axes."""
    pdata_dir = Path(pdata_dir)
    visu = parse_jcamp(pdata_dir / "visu_pars")
    dim = int(visu.get("VisuCoreDim", 2))
    spatial_shape = tuple(int(size) for size in _flatten_numeric(visu["VisuCoreSize"]))
    if len(spatial_shape) != dim or min(spatial_shape) < 1:
        raise ValueError("VisuCoreSize must identify each positive frame dimension")
    count = int(visu.get("VisuCoreFrameCount", 1))
    dtype = np.dtype(_dtype_from_word(visu["VisuCoreWordType"]))
    dtype = dtype.newbyteorder(">" if str(visu.get("VisuCoreByteOrder", "")).lower() == "bigendian" else "<")
    raw = np.fromfile(pdata_dir / "2dseq", dtype=dtype)
    expected = count * int(np.prod(spatial_shape))
    if raw.size != expected:
        raise ValueError(f"2dseq size mismatch: found {raw.size} values, expected {expected}")
    raw = raw.reshape((count, -1))
    slopes = frame_parameter(visu, "VisuCoreDataSlope", default=1)
    offsets = frame_parameter(visu, "VisuCoreDataOffs", default=0)
    scaled = raw.astype(np.float32) * slopes.astype(np.float32) + offsets.astype(np.float32)
    transpositions = frame_parameter(visu, "VisuCoreTransposition", default=0)[:, 0]
    if np.any(transpositions != transpositions.astype(int)) or np.any((transpositions < 0) | (transpositions > dim)):
        raise ValueError("VisuCoreTransposition exceeds the frame dimensions")
    frames = scaled.reshape((count,) + spatial_shape, order="F")
    for code in np.unique(transpositions.astype(int)):
        if not code:
            continue
        first, second = (0, dim - 1) if code == dim else (code - 1, code)
        if spatial_shape[first] == spatial_shape[second]:
            # Equal-sized frames already use the image coordinates of
            # VisuCoreOrientation. Transposition is a storage-shape declaration,
            # not an extra anatomical rotation.
            continue
        stored_shape = list(spatial_shape)
        stored_shape[first], stored_shape[second] = stored_shape[second], stored_shape[first]
        selected = np.flatnonzero(transpositions == code)
        decoded = scaled[selected].reshape((len(selected),) + tuple(stored_shape), order="F")
        frames[selected] = np.swapaxes(decoded, first + 1, second + 1)

    requested = str(image_type or "auto").lower()
    if requested not in {"auto", "complex", "real", "reel", "magnitude", "phase", "imaginary"}:
        raise ValueError("image_type must be one of: auto, complex, real, magnitude, phase, imaginary")
    types = [item.upper() for item in _as_string_list(visu.get("VisuCoreFrameType"))]
    groups = frame_group_entries(visu)
    complex_group = next((i for i, row in enumerate(groups) if "FG_COMPLEX" in str(row[1]).upper()), None)
    combine = requested == "complex" or (requested == "auto" and
        ("COMPLEX_IMAGE" in types or complex_group is not None or
         ("REAL_IMAGE" in types and "IMAGINARY_IMAGE" in types)))
    real_indices = np.arange(count)
    if requested == "imaginary":
        frames = (1j * frames).astype(np.complex64)
    elif combine:
        if count % 2:
            raise ValueError("Complex images require matching real and imaginary frames")
        if complex_group is not None:
            counts = [int(row[0]) for row in groups]
            if counts[complex_group] != 2 or int(np.prod(counts)) != count:
                raise ValueError("The complex frame group must identify real and imaginary components")
            grid = np.arange(count).reshape(tuple(reversed(counts)))
            axis = len(counts) - 1 - complex_group
            real_first = types[:2] != ["IMAGINARY_IMAGE", "REAL_IMAGE"]
            real_indices = np.take(grid, 0 if real_first else 1, axis=axis).ravel()
            imag_indices = np.take(grid, 1 if real_first else 0, axis=axis).ravel()
        else:
            real_indices, imag_indices = np.arange(count // 2), np.arange(count // 2, count)
        frames = (frames[real_indices] + 1j * frames[imag_indices]).astype(np.complex64)

    inverse = np.full(count, -1, dtype=int)
    inverse[real_indices] = np.arange(len(real_indices))
    packages = slice_packages(visu)
    if slice_package is None and len(packages) > 1:
        grids, volumes = [], []
        maximum = max(len(package) for package in packages)
        for package in range(len(packages)):
            grid, _ = output_frame_grid(visu, package)
            grid = grid[inverse[grid[:, 0]] >= 0]
            if np.any(inverse[grid] < 0):
                raise ValueError("Complex components do not share the same spatial slice layout")
            volume = np.moveaxis(frames[inverse[grid]], 1, -1)
            volumes.append(
                np.pad(
                    volume,
                    ((0, 0), (0, 0), (0, 0), (0, maximum - grid.shape[1])),
                    constant_values=np.nan,
                )
            )
            grids.append(grid.tolist())
        data = np.concatenate(volumes)
        spatial_shape += (maximum,)
        visu["_mrax_package_frame_indices"] = grids
        visu["_mrax_selected_package"] = None
        visu["_mrax_slice_packages"] = [
            {
                "index": package,
                "slice_count": len(indices),
                "native_image_plane": package_plane(visu, package),
            }
            for package, indices in enumerate(packages)
        ]
        return data, visu, spatial_shape
    grid, has_slices = output_frame_grid(visu, slice_package)
    grid = grid[inverse[grid[:, 0]] >= 0]
    if np.any(inverse[grid] < 0):
        raise ValueError("Complex components do not share the same spatial slice layout")
    data = frames[inverse[grid]]
    if dim == 2 and has_slices and grid.shape[1] > 1:
        data = np.moveaxis(data, 1, -1)
        spatial_shape = spatial_shape + (grid.shape[1],)
    else:
        data = data[:, 0]
        if dim == 3 and str(visu.get("VisuCoreDiskSliceOrder", "")).lower() == "disk_reverse_slice_order":
            data = np.flip(data, axis=-1)
    visu["_mrax_output_frame_indices"] = grid.tolist()
    visu["_mrax_selected_package"] = slice_package
    visu["_mrax_mixed_slice_geometry"] = False
    visu["_mrax_frame_transposition"] = transpositions.astype(int).tolist()
    visu["_mrax_fg_order_entries"] = groups
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

