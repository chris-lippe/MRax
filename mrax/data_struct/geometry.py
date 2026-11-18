from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Any, Mapping, Sequence

import numpy as np

SPATIAL_GEOMETRY_VERSION = 1
NATIVE_AXIS_CONVENTION = "native_v1"
NIFTI_AFFINE_CONVENTION = "nifti_xyz_ras"

AffineTuple = tuple[tuple[float, float, float, float], ...]


def _normalized_affine(value: object | None) -> AffineTuple | None:
    if value is None:
        return None
    try:
        affine = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if affine.shape != (4, 4) or not np.all(np.isfinite(affine)):
        return None
    return tuple(tuple(float(item) for item in row) for row in affine)  # type: ignore[return-value]


def _normalized_shape(value: object | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    try:
        shape = tuple(int(item) for item in value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return shape if shape and all(item > 0 for item in shape) else None


def _normalized_spacing(value: object | None) -> tuple[float, ...] | None:
    if value is None:
        return None
    try:
        spacing = tuple(float(item) for item in value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not spacing or not np.all(np.isfinite(spacing)) or any(item <= 0 for item in spacing):
        return None
    return spacing


def _normalized_axis_labels(
    value: object | None,
    *,
    spatial_shape: tuple[int, ...] | None,
) -> tuple[str, ...] | None:
    if value is None:
        return None
    try:
        labels = tuple(str(item).strip().lower() for item in value)  # type: ignore[arg-type]
    except TypeError:
        return None
    if not labels or any(label not in {"x", "y", "z"} for label in labels):
        return None
    if len(set(labels)) != len(labels):
        return None
    if spatial_shape is not None and len(labels) != len(spatial_shape):
        return None
    return labels


@dataclass(frozen=True)
class SpatialGeometry:
    """Physical geometry kept separate from an image's native array ordering.

    MRax image and k-space arrays remain in their acquisition/import order. ``affine``
    maps the XYZ array produced by the NIfTI I/O boundary into RAS world coordinates.
    Raw k-space axes use their encoding geometry. ``status='unknown'`` records
    unresolved orientation separately from an identity affine.
    """

    affine: AffineTuple | np.ndarray | Sequence[Sequence[float]] | None = None
    frame_affines: tuple[AffineTuple, ...] | Sequence[object] = ()
    spatial_shape: tuple[int, ...] | Sequence[int] | None = None
    voxel_sizes: tuple[float, ...] | Sequence[float] | None = None
    axis_anatomical_labels: tuple[str, ...] | Sequence[str] | None = None
    source: str = "unknown"
    status: str = "unknown"
    version: int = SPATIAL_GEOMETRY_VERSION
    axis_convention: str = NATIVE_AXIS_CONVENTION
    affine_convention: str = NIFTI_AFFINE_CONVENTION
    frame_shapes: tuple[tuple[int, ...], ...] | Sequence[object] = ()
    frame_dims: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        if int(self.version) != SPATIAL_GEOMETRY_VERSION:
            raise ValueError(
                f"Unsupported spatial geometry version {self.version}; "
                f"expected {SPATIAL_GEOMETRY_VERSION}"
            )
        if str(self.axis_convention) != NATIVE_AXIS_CONVENTION:
            raise ValueError(f"Unsupported spatial axis convention '{self.axis_convention}'")
        if str(self.affine_convention) != NIFTI_AFFINE_CONVENTION:
            raise ValueError(f"Unsupported affine convention '{self.affine_convention}'")
        affine = _normalized_affine(self.affine)
        frame_affines = tuple(
            normalized
            for value in self.frame_affines
            if (normalized := _normalized_affine(value)) is not None
        )
        status = str(self.status or "unknown").strip().lower()
        if status not in {"known", "partial", "unknown"}:
            raise ValueError("SpatialGeometry.status must be 'known', 'partial', or 'unknown'")
        if affine is None and not frame_affines and status == "known":
            status = "unknown"
        object.__setattr__(self, "affine", affine)
        object.__setattr__(self, "frame_affines", frame_affines)
        shapes = tuple(_normalized_shape(value) for value in self.frame_shapes)
        if any(value is None for value in shapes):
            raise ValueError("Frame shapes must contain positive spatial dimensions")
        object.__setattr__(self, "frame_shapes", shapes)
        frame_dims = tuple((str(name), int(size)) for name, size in self.frame_dims)
        if len(dict(frame_dims)) != len(frame_dims) or any(size < 1 for _, size in frame_dims):
            raise ValueError("Frame dimensions must have distinct names and positive sizes")
        if frame_dims and shapes and np.prod([size for _, size in frame_dims]) != len(shapes):
            raise ValueError("Frame dimensions must account for every measured spatial extent")
        object.__setattr__(self, "frame_dims", frame_dims)
        spatial_shape = _normalized_shape(self.spatial_shape)
        if spatial_shape and any(
            len(shape) != len(spatial_shape)
            or any(actual > limit for actual, limit in zip(shape, spatial_shape))
            for shape in shapes
        ):
            raise ValueError("Measured frame shapes must fit their storage grid")
        object.__setattr__(self, "spatial_shape", spatial_shape)
        object.__setattr__(self, "voxel_sizes", _normalized_spacing(self.voxel_sizes))
        object.__setattr__(
            self,
            "axis_anatomical_labels",
            _normalized_axis_labels(self.axis_anatomical_labels, spatial_shape=spatial_shape),
        )
        object.__setattr__(self, "source", str(self.source or "unknown"))
        object.__setattr__(self, "status", status)

    @property
    def has_orientation(self) -> bool:
        return self.affine is not None or bool(self.frame_affines)

    def affine_for_frame(self, frame_index: int = 0) -> np.ndarray | None:
        if self.frame_affines:
            index = max(0, min(int(frame_index), len(self.frame_affines) - 1))
            return np.asarray(self.frame_affines[index], dtype=float)
        if self.affine is not None:
            return np.asarray(self.affine, dtype=float)
        return None

    def to_metadata(self) -> dict[str, Any]:
        return {
            "version": int(self.version),
            "axis_convention": self.axis_convention,
            "affine_convention": self.affine_convention,
            "status": self.status,
            "source": self.source,
            "spatial_shape": None if self.spatial_shape is None else list(self.spatial_shape),
            "voxel_sizes": None if self.voxel_sizes is None else list(self.voxel_sizes),
            "axis_anatomical_labels": (
                None if self.axis_anatomical_labels is None else list(self.axis_anatomical_labels)
            ),
            "affine": None if self.affine is None else [list(row) for row in self.affine],
            "frame_affines": [[list(row) for row in affine] for affine in self.frame_affines],
            "frame_shapes": [list(shape) for shape in self.frame_shapes],
            "frame_dims": [list(dim) for dim in self.frame_dims],
        }

    def shape_for_frame(self, frame_index: int = 0) -> tuple[int, ...] | None:
        """Measured spatial extent, excluding padding used for unequal stacks."""
        if self.frame_shapes:
            return tuple(
                self.frame_shapes[max(0, min(int(frame_index), len(self.frame_shapes) - 1))]
            )
        return self.spatial_shape

    def frame_index(self, coordinates, *, default=0) -> int:
        if not self.frame_dims:
            return int(default)
        return int(
            np.ravel_multi_index(
                tuple(coordinates.get(name, 0) for name, _ in self.frame_dims),
                tuple(size for _, size in self.frame_dims),
            )
        )

    def measured_mask(self) -> np.ndarray | None:
        """Valid acquired voxels when differently sized stacks share storage."""
        if not self.frame_shapes or all(shape == self.spatial_shape for shape in self.frame_shapes):
            return None
        mask = np.zeros((len(self.frame_shapes),) + tuple(self.spatial_shape), dtype=bool)
        for frame, shape in enumerate(self.frame_shapes):
            mask[(frame,) + tuple(slice(0, size) for size in shape)] = True
        return mask

    def for_frame(self, index: int = 0) -> "SpatialGeometry":
        affine = self.affine_for_frame(index)
        shape = self.shape_for_frame(index)
        if affine is None or shape is None:
            return self
        ndim = len(shape)
        order = tuple(reversed(range(ndim))) + tuple(range(ndim, 3))
        native = affine[:3, :3][:, order]
        spacing = np.linalg.norm(native, axis=0)
        unit = np.abs(native / np.maximum(spacing, 1e-12))
        axes = max(
            permutations(range(3)),
            key=lambda candidate: sum(unit[axis, i] for i, axis in enumerate(candidate)),
        )
        return SpatialGeometry(
            affine=affine,
            spatial_shape=shape,
            voxel_sizes=tuple(spacing[:ndim]),
            axis_anatomical_labels=tuple("xyz"[axis] for axis in axes[:ndim]),
            source=self.source,
            status=self.status,
        )

    def select_frames(self, indices, *, frame_dims=()) -> "SpatialGeometry":
        indices = tuple(int(index) for index in indices)
        geometry = SpatialGeometry.from_affines(
            [self.affine_for_frame(index) for index in indices],
            spatial_shape=self.spatial_shape,
            voxel_sizes=self.voxel_sizes,
            axis_anatomical_labels=self.axis_anatomical_labels,
            source=self.source,
            status=self.status,
        )
        from dataclasses import replace

        return replace(
            geometry,
            frame_shapes=tuple(self.shape_for_frame(index) for index in indices)
            if self.frame_shapes
            else (),
            frame_dims=tuple(frame_dims),
        )

    @classmethod
    def from_metadata(cls, value: Mapping[str, object] | None) -> "SpatialGeometry":
        metadata = dict(value or {})
        return cls(
            affine=metadata.get("affine"),
            frame_affines=metadata.get("frame_affines") or (),
            frame_shapes=metadata.get("frame_shapes") or (),
            frame_dims=metadata.get("frame_dims") or (),
            spatial_shape=metadata.get("spatial_shape"),
            voxel_sizes=metadata.get("voxel_sizes"),
            axis_anatomical_labels=metadata.get("axis_anatomical_labels"),
            source=str(metadata.get("source") or "unknown"),
            status=str(metadata.get("status") or "unknown"),
            version=int(metadata.get("version") or SPATIAL_GEOMETRY_VERSION),
            axis_convention=str(metadata.get("axis_convention") or NATIVE_AXIS_CONVENTION),
            affine_convention=str(metadata.get("affine_convention") or NIFTI_AFFINE_CONVENTION),
        )

    @classmethod
    def from_affines(
        cls,
        affines: Sequence[object] | object | None,
        *,
        spatial_shape: Sequence[int] | None = None,
        voxel_sizes: Sequence[float] | None = None,
        axis_anatomical_labels: Sequence[str] | None = None,
        source: str,
        status: str = "known",
    ) -> "SpatialGeometry":
        single = _normalized_affine(affines)
        if single is not None:
            return cls(
                affine=single,
                spatial_shape=spatial_shape,
                voxel_sizes=voxel_sizes,
                axis_anatomical_labels=axis_anatomical_labels,
                source=source,
                status=status,
            )
        values = affines if isinstance(affines, Sequence) else ()
        normalized = tuple(
            affine
            for value in values
            if (affine := _normalized_affine(value)) is not None
        )
        if not normalized:
            return cls(
                spatial_shape=spatial_shape,
                voxel_sizes=voxel_sizes,
                axis_anatomical_labels=axis_anatomical_labels,
                source=source,
                status="unknown",
            )
        common = normalized[0] if all(item == normalized[0] for item in normalized[1:]) else None
        return cls(
            affine=common,
            frame_affines=() if common is not None else normalized,
            spatial_shape=spatial_shape,
            voxel_sizes=voxel_sizes,
            axis_anatomical_labels=axis_anatomical_labels,
            source=source,
            status=status,
        )


@dataclass(frozen=True)
class ResolvedSpatialGeometry:
    affine: np.ndarray
    source: str
    identity_fallback: bool


def geometries_are_compatible(
    target: SpatialGeometry | None,
    reference: SpatialGeometry | None,
) -> bool:
    """Return whether a reference geometry can safely describe the target grid."""

    if target is None or reference is None:
        return False
    compared = False
    if target.spatial_shape is not None and reference.spatial_shape is not None:
        compared = True
        if tuple(target.spatial_shape) != tuple(reference.spatial_shape):
            return False
    if target.voxel_sizes is not None and reference.voxel_sizes is not None:
        compared = True
        if len(target.voxel_sizes) != len(reference.voxel_sizes):
            return False
        if not np.allclose(target.voxel_sizes, reference.voxel_sizes, rtol=1e-4, atol=1e-6):
            return False
    if (
        target.axis_anatomical_labels is not None
        and reference.axis_anatomical_labels is not None
    ):
        compared = True
        if target.axis_anatomical_labels != reference.axis_anatomical_labels:
            return False
    return compared


def resolve_spatial_geometry(
    geometry: SpatialGeometry | None,
    *,
    reference: SpatialGeometry | None = None,
    frame_index: int = 0,
) -> ResolvedSpatialGeometry:
    """Resolve export geometry, using a compatible reference before identity."""

    if geometry is not None:
        affine = geometry.affine_for_frame(frame_index)
        if affine is not None:
            return ResolvedSpatialGeometry(affine, geometry.source, False)
    if geometries_are_compatible(geometry, reference):
        affine = reference.affine_for_frame(frame_index) if reference is not None else None
        if affine is not None:
            return ResolvedSpatialGeometry(affine, f"reference:{reference.source}", False)
    return ResolvedSpatialGeometry(np.eye(4, dtype=float), "identity", True)
