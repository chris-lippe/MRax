"""Bruker frame groups and spatial slice packages, independent of pixel values."""

from __future__ import annotations

import numpy as np

from .bruker_parameters import _flatten_numeric


def frame_group_entries(visu: dict) -> list[list[object]]:
    value = visu.get("VisuFGOrderDesc")
    if isinstance(value, list) and value and all(isinstance(item, list) for item in value):
        return [list(item) for item in value if item]
    if isinstance(value, list) and len(value) >= 2:
        if len(value) <= 5 and isinstance(value[0], (int, float)) and isinstance(value[1], str):
            return [list(value)]
        entries, index = [], 0
        while index + 1 < len(value):
            if isinstance(value[index], (int, float)) and isinstance(value[index + 1], str):
                entries.append(list(value[index : index + 5]))
                index += 5
            else:
                index += 1
        return entries
    return []


def frame_parameter(visu: dict, key: str, width: int = 1, *, default=None) -> np.ndarray:
    """Expand a scalar or frame-group-dependent parameter into stored frame order."""
    count = int(visu.get("VisuCoreFrameCount", 1))
    values = np.asarray(_flatten_numeric(visu.get(key)), dtype=float)
    if not values.size:
        if default is None:
            return np.empty((0, width))
        values = np.asarray(default, dtype=float).reshape(-1)
    if values.size % width:
        raise ValueError(f"{key} does not contain complete parameter values")
    if not np.isfinite(values).all():
        raise ValueError(f"{key} must contain finite parameter values")
    values = values.reshape((-1, width))
    if len(values) == count:
        return values
    if len(values) == 1:
        return np.broadcast_to(values, (count, width))
    groups = frame_group_entries(visu)
    counts = [int(entry[0]) for entry in groups]
    if groups and np.prod(counts) == count:
        dependencies = visu.get("VisuGroupDepVals", [])
        if (
            isinstance(dependencies, list)
            and dependencies
            and not isinstance(dependencies[0], list)
        ):
            dependencies = [dependencies]
        dependent = []
        for index, entry in enumerate(groups):
            if len(entry) < 5:
                continue
            first, length = int(entry[3]), int(entry[4])
            if any(
                isinstance(dep, list) and dep and str(dep[0]).strip("<>") == key
                for dep in dependencies[first : first + length]
            ):
                dependent.append(index)
        if not dependent:
            # Older writers omit dependencies for slice geometry.
            candidates = [i for i, size in enumerate(counts) if size == len(values)]
            slices = [i for i in candidates if "FG_SLICE" in str(groups[i][1]).upper()]
            dependent = (
                slices
                if key in {"VisuCoreOrientation", "VisuCorePosition", "VisuCoreTransposition"}
                else candidates
            )
            if len(dependent) != 1:
                dependent = []
        if dependent and np.prod([counts[i] for i in dependent]) == len(values):
            coordinates = np.unravel_index(np.arange(count), tuple(counts), order="F")
            indices = np.ravel_multi_index(
                tuple(coordinates[i] for i in dependent),
                tuple(counts[i] for i in dependent),
                order="F",
            )
            return values[indices]
    raise ValueError(f"{key} cannot be aligned with the declared frame groups")


def slice_frame_grid(visu: dict) -> tuple[np.ndarray, bool]:
    """Return [non-spatial frame, slice] indices. Descriptor zero varies fastest."""
    count = int(visu.get("VisuCoreFrameCount", 1))
    groups = frame_group_entries(visu)
    slice_group = next(
        (i for i, row in enumerate(groups) if "FG_SLICE" in str(row[1]).upper()), None
    )
    if slice_group is None:
        # Some older exports identify the slice group without its full descriptor.
        if "FG_SLICE" in str(visu.get("VisuFGOrderDesc", "")).upper():
            return np.arange(count).reshape(1, count), True
        return np.arange(count).reshape(count, 1), False
    counts = [int(row[0]) for row in groups]
    if min(counts) < 1 or int(np.prod(counts)) != count:
        raise ValueError("Bruker frame groups do not account for every stored frame")
    grid = np.arange(count).reshape(tuple(reversed(counts)))
    grid = np.moveaxis(grid, len(counts) - 1 - slice_group, -1)
    return grid.reshape((-1, counts[slice_group])), True


def slice_packages(visu: dict) -> list[tuple[int, ...]]:
    """Identify homogeneous spatial packages without treating slices as time."""
    if int(visu.get("VisuCoreDim", 2)) != 2:
        return [()]
    grid, has_slices = slice_frame_grid(visu)
    if not has_slices:
        return [()]
    count = grid.shape[1]
    declared = _flatten_numeric(visu.get("VisuCoreSlicePacksSlices"))
    if declared:
        if len(declared) % 2:
            raise ValueError("Slice packages require a first slice index and slice count")
        packages = [
            tuple(range(int(first), int(first + length)))
            for first, length in np.asarray(declared).reshape((-1, 2))
        ]
        if sorted(index for package in packages for index in package) != list(range(count)):
            raise ValueError("Slice packages must partition the slice frame group exactly once")
        return packages
    orientations = frame_parameter(visu, "VisuCoreOrientation", 9)
    if not len(orientations):
        return [tuple(range(count))]
    matrices = orientations[grid[0]]
    packages, current = [], []
    for index, matrix in enumerate(matrices):
        if current and not np.allclose(matrix, matrices[current[0]], atol=1e-5):
            packages.append(tuple(current))
            current = []
        current.append(index)
    if current:
        packages.append(tuple(current))
    return packages


def output_frame_grid(visu: dict, slice_package: int | None = None) -> tuple[np.ndarray, bool]:
    """Select one package and order its slices along their measured normal."""
    grid, has_slices = slice_frame_grid(visu)
    if int(visu.get("VisuCoreDim", 2)) != 2:
        return np.arange(int(visu.get("VisuCoreFrameCount", 1))).reshape(-1, 1), False
    packages = slice_packages(visu)
    if len(packages) > 1 and slice_package is None:
        raise ValueError(
            "This Bruker reconstruction has multiple slice packages. Select slice_package or import the study"
        )
    selected = 0 if slice_package is None else int(slice_package)
    if not 0 <= selected < len(packages):
        raise ValueError("The selected Bruker slice package does not exist")
    if has_slices:
        grid = grid[:, packages[selected]]
        positions = frame_parameter(visu, "VisuCorePosition", 3)
        orientations = frame_parameter(visu, "VisuCoreOrientation", 9)
        if len(positions) and len(orientations):
            grid = grid.copy()
            for row in grid:
                matrices = orientations[row].reshape((-1, 3, 3))
                if not np.allclose(matrices, matrices[0], atol=1e-5):
                    raise ValueError(
                        "One spatial slice package contains incompatible image orientations"
                    )
                normal = matrices[0, 2]
                row[:] = row[np.argsort(positions[row] @ normal, kind="stable")]
        elif str(visu.get("VisuCoreDiskSliceOrder", "")).lower() == "disk_reverse_slice_order":
            grid = grid[:, ::-1]
    return grid, has_slices


def package_plane(visu: dict, slice_package: int = 0) -> str:
    grid, _ = output_frame_grid(visu, slice_package)
    orientations = frame_parameter(visu, "VisuCoreOrientation", 9)
    if not len(orientations):
        return "unknown"
    normal = orientations[int(grid[0, 0])].reshape(3, 3)[2]
    return ("sagittal", "coronal", "axial")[int(np.argmax(np.abs(normal)))]
