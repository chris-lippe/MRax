from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from mrax.data_struct import DataSet, Experiment, SpatialGeometry, resolve_spatial_geometry
from mrax.data_struct.dimensions import SPATIAL_DIMS

_LOG = logging.getLogger(__name__)


def _import_nibabel():
    try:
        import nibabel as nib  # type: ignore
    except Exception as exc:  # pragma: no cover - invalid runtime environment
        raise ImportError(
            "nibabel is required for NIfTI I/O. Reinstall MRax with `pip install mrax`."
        ) from exc
    return nib


def mrax_to_nifti_volume(volume: np.ndarray) -> np.ndarray:
    """Convert a MRax-ordered spatial volume to NIfTI (XYZ) order."""

    arr = np.asarray(volume)
    if arr.ndim not in (2, 3):
        raise ValueError("mrax_to_nifti_volume expects 2D or 3D spatial input")
    arr = np.transpose(arr, tuple(range(arr.ndim))[::-1])
    if arr.ndim == 2:
        arr = arr[..., None]
    return arr


def nifti_to_mrax_volume(volume: np.ndarray) -> np.ndarray:
    """Convert a NIfTI spatial volume into MRax spatial axis order."""

    arr = np.asarray(volume)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim not in (2, 3):
        raise ValueError("nifti_to_mrax_volume expects 2D or 3D spatial input")
    arr = np.transpose(arr, tuple(range(arr.ndim))[::-1])
    return arr


def load_nifti(path: str | Path, *, dtype: np.dtype | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Load a NIfTI file and return data in MRax spatial order plus affine."""

    nib = _import_nibabel()
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj)
    if dtype is not None:
        data = data.astype(dtype)
    data = nifti_to_mrax_volume(data)
    return data, img.affine


def nifti_spatial_geometry(
    affine: np.ndarray,
    *,
    spatial_shape: Sequence[int] | None = None,
) -> SpatialGeometry:
    """Describe a loaded NIfTI grid while leaving its MRax array order unchanged."""

    matrix = np.asarray(affine, dtype=float)
    nifti_voxel_sizes = tuple(float(value) for value in np.linalg.norm(matrix[:3, :3], axis=0))
    spatial_ndim = len(tuple(spatial_shape or ())) or 3
    voxel_sizes = tuple(reversed(nifti_voxel_sizes[:spatial_ndim]))
    directions = matrix[:3, :spatial_ndim]
    norms = np.linalg.norm(directions, axis=0)
    axis_labels: tuple[str, ...] | None = None
    if np.all(norms > 1e-8):
        dominant = np.argmax(np.abs(directions / norms[None, :]), axis=0)
        nifti_labels = tuple("xyz"[int(index)] for index in dominant)
        if len(set(nifti_labels)) == spatial_ndim:
            axis_labels = tuple(reversed(nifti_labels))
    return SpatialGeometry(
        affine=matrix,
        spatial_shape=spatial_shape,
        voxel_sizes=voxel_sizes,
        axis_anatomical_labels=axis_labels,
        source="nifti",
        status="known",
    )


def save_nifti(
    path: str | Path,
    data: np.ndarray,
    affine: np.ndarray,
    *,
    dtype: np.dtype | None = None,
    slope: float | None = None,
    intercept: float | None = None,
) -> None:
    """Save data (MRax spatial order) to NIfTI with a specified affine."""

    nib = _import_nibabel()
    arr = mrax_to_nifti_volume(data)
    if dtype is not None:
        arr = arr.astype(dtype)
    img = nib.Nifti1Image(arr, affine, dtype=dtype)
    img.set_qform(affine, code=1)
    img.set_sform(None)
    try:
        img.header.set_xyzt_units(xyz="mm")
    except Exception:
        _LOG.debug("Ignored optional operation failure", exc_info=True)
    if slope is not None or intercept is not None:
        img.header.set_slope_inter(slope, intercept)
    nib.save(img, str(path))


def _iter_non_spatial_indices(dims: Mapping[str, int]) -> Iterable[tuple[int, ...]]:
    non_spatial = [name for name in dims.keys() if name not in SPATIAL_DIMS]
    if not non_spatial:
        yield ()
        return
    shape = [int(dims[name]) for name in non_spatial]
    yield from np.ndindex(*shape)


def export_nifti_series(
    experiment: Experiment,
    dataset: DataSet,
    out_dir: str | Path,
    *,
    affine: np.ndarray | None = None,
    affines: Sequence[np.ndarray] | None = None,
    slope: float | None = None,
    intercept: float | None = None,
    slopes: Sequence[float] | None = None,
    intercepts: Sequence[float] | None = None,
    raw_dtype: np.dtype | None = None,
    stem: str | None = None,
    overwrite: bool = True,
    reference_geometry: SpatialGeometry | None = None,
) -> list[Path]:
    """Export each frame to an individual NIfTI file.

    Data is assumed to be in MRax spatial order (x/y/z). Use ``mrax_to_nifti_volume``
    if you need to reorder axes explicitly. Pass an affine (or per-frame affines) to
    set world coordinates. Optional ``slope``/``intercept`` plus ``raw_dtype`` allow
    storing scaled integer data similar to vendor exports.
    """

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    data = np.asarray(dataset.data)
    dim_names = list(experiment.dims.keys())
    spatial_axes = [i for i, name in enumerate(dim_names) if name in SPATIAL_DIMS]
    non_spatial_axes = [i for i, name in enumerate(dim_names) if name not in SPATIAL_DIMS]

    if not spatial_axes:
        raise ValueError("Experiment does not declare spatial dimensions; cannot export NIfTI.")

    outputs: list[Path] = []
    spatial_shape = tuple(int(data.shape[axis]) for axis in spatial_axes)
    non_spatial_sizes = [experiment.dims[dim_names[ax]] for ax in non_spatial_axes]
    frame_count = int(np.prod(non_spatial_sizes)) if non_spatial_sizes else 1

    for idx in _iter_non_spatial_indices(experiment.dims):
        slices = [slice(None)] * data.ndim
        for axis, index in zip(non_spatial_axes, idx):
            slices[axis] = index
        volume = np.asarray(data[tuple(slices)])
        volume = mrax_to_nifti_volume(volume)

        if idx:
            flat_index = int(np.ravel_multi_index(idx, non_spatial_sizes))
        else:
            flat_index = 0

        if affines is not None:
            if flat_index >= len(affines):
                raise ValueError("affines length does not match frame count")
            affine_local = np.asarray(affines[flat_index], dtype=float)
        elif affine is not None:
            affine_local = np.asarray(affine, dtype=float)
        else:
            geometry = dataset.spatial_geometry or SpatialGeometry(
                spatial_shape=spatial_shape,
                source="unknown",
            )
            resolved = resolve_spatial_geometry(
                geometry,
                reference=reference_geometry,
                frame_index=flat_index,
            )
            affine_local = resolved.affine
            if resolved.identity_fallback:
                _LOG.warning(
                    "NIfTI export for %s has no orientation metadata; using identity affine",
                    dataset.name,
                )

        slope_local = slope
        intercept_local = intercept
        if slopes is not None:
            if len(slopes) != frame_count:
                raise ValueError("slopes length does not match frame count")
            slope_local = float(slopes[flat_index])
        if intercepts is not None:
            if len(intercepts) != frame_count:
                raise ValueError("intercepts length does not match frame count")
            intercept_local = float(intercepts[flat_index])

        fname = stem or str(dataset.name or "scan")
        out_file = out_path / f"{fname}_{flat_index + 1}.nii"
        if out_file.exists() and not overwrite:
            outputs.append(out_file)
            continue

        nib = _import_nibabel()
        if slope_local is not None:
            if raw_dtype is None:
                raise ValueError("raw_dtype is required when slope/intercept is provided")
            safe_slope = float(slope_local) if float(slope_local) != 0.0 else 1.0
            raw = np.rint((volume.astype(np.float64) - (intercept_local or 0.0)) / safe_slope)
            if np.issubdtype(raw_dtype, np.integer):
                info = np.iinfo(raw_dtype)
                raw = np.clip(raw, info.min, info.max).astype(raw_dtype)
            else:
                raw = raw.astype(raw_dtype)
            img = nib.Nifti1Image(raw, affine_local)
        else:
            img = nib.Nifti1Image(volume, affine_local)
        img.set_qform(affine_local, code=1)
        img.set_sform(None)
        try:
            img.header.set_xyzt_units(xyz="mm")
        except Exception:
            _LOG.debug("Ignored optional operation failure", exc_info=True)
        if slope_local is not None or intercept_local is not None:
            img.header.set_slope_inter(slope_local, intercept_local)
        outputs.append(out_file)
        nib.save(img, str(out_file))

    return outputs


def export_nifti_4d(
    experiment: Experiment,
    dataset: DataSet,
    out_path: str | Path,
    *,
    affine: np.ndarray | None = None,
    slopes: Sequence[float] | None = None,
    intercepts: Sequence[float] | None = None,
    slope: float | None = None,
    intercept: float | None = None,
    raw_dtype: np.dtype | None = None,
    stem: str | None = None,
    overwrite: bool = True,
    reference_geometry: SpatialGeometry | None = None,
) -> Path:
    """Export all frames into a single 4D NIfTI file."""

    out_file = Path(out_path)
    if out_file.is_dir():
        fname = stem or str(dataset.name or "scan")
        out_file = out_file / f"{fname}.nii"

    if out_file.exists() and not overwrite:
        return out_file

    data = np.asarray(dataset.data)
    dim_names = list(experiment.dims.keys())
    spatial_axes = [i for i, name in enumerate(dim_names) if name in SPATIAL_DIMS]
    if not spatial_axes:
        raise ValueError("Experiment does not declare spatial dimensions; cannot export NIfTI.")

    spatial_shape = tuple(int(data.shape[ax]) for ax in spatial_axes)
    frame_count = int(np.prod([experiment.dims[name] for name in dim_names if name not in SPATIAL_DIMS])) or 1
    frames = data.reshape((frame_count, *spatial_shape))

    nifti_frames = np.stack([mrax_to_nifti_volume(frame) for frame in frames], axis=0)
    nifti_stack = np.moveaxis(nifti_frames, 0, -1)

    slope_local = slope
    intercept_local = intercept
    if slopes is not None:
        slope_local = float(slopes[0])
        if len(slopes) > 1 and not np.allclose(slopes, slopes[0]):
            slope_local = None
    if intercepts is not None:
        intercept_local = float(intercepts[0])
        if len(intercepts) > 1 and not np.allclose(intercepts, intercepts[0]):
            intercept_local = None
    if slope_local is None and slope is not None:
        slope_local = slope
    if intercept_local is None and intercept is not None:
        intercept_local = intercept

    if affine is not None:
        affine_local = np.asarray(affine, dtype=float)
    else:
        geometry = dataset.spatial_geometry or SpatialGeometry(
            spatial_shape=spatial_shape,
            source="unknown",
        )
        resolved = resolve_spatial_geometry(
            geometry,
            reference=reference_geometry,
        )
        affine_local = resolved.affine
        if resolved.identity_fallback:
            _LOG.warning(
                "NIfTI export for %s has no orientation metadata; using identity affine",
                dataset.name,
            )
    nib = _import_nibabel()

    if slope_local is not None:
        if raw_dtype is None:
            raise ValueError("raw_dtype is required when slope/intercept is provided")
        safe_slope = float(slope_local) if float(slope_local) != 0.0 else 1.0
        raw = np.rint((nifti_stack.astype(np.float64) - (intercept_local or 0.0)) / safe_slope)
        if np.issubdtype(raw_dtype, np.integer):
            info = np.iinfo(raw_dtype)
            raw = np.clip(raw, info.min, info.max).astype(raw_dtype)
        else:
            raw = raw.astype(raw_dtype)
        img = nib.Nifti1Image(raw, affine_local)
    else:
        img = nib.Nifti1Image(nifti_stack, affine_local)

    img.set_qform(affine_local, code=1)
    img.set_sform(None)
    try:
        img.header.set_xyzt_units(xyz="mm")
    except Exception:
        _LOG.debug("Ignored optional operation failure", exc_info=True)
    if slope_local is not None or intercept_local is not None:
        img.header.set_slope_inter(slope_local, intercept_local)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(out_file))
    return out_file
