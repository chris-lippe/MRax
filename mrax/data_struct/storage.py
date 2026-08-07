from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Any, Dict

import ismrmrd
import jax.numpy as jnp
import numpy as np

from mrax import MRAX_VERSION, MRAX_VERSION_TAG
from mrax.versioning import missing_mrax_version_message, validate_mrax_file_version

from .dimensions import SPATIAL_DIMS
from .experiment import Experiment
from .geometry import SpatialGeometry
from .quantity import Quantity
from .study import Study

MANIFEST_NAME = "study_manifest.json"
_LOG = logging.getLogger(__name__)


def _resolve_manifest_path(base: Path, value: object, *, label: str) -> Path:
    """Resolve a manifest path while keeping it inside the manifest directory."""

    root = Path(base).resolve()
    candidate = (root / str(value)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Unsafe {label} path outside manifest directory: {value}") from exc
    return candidate


def _ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def _record_io_warning(container: dict[str, Any], message: str) -> str:
    text = f"warning: skipped {message}"
    container.setdefault("warnings", []).append(text)
    warnings.warn(text, RuntimeWarning, stacklevel=3)
    return text


def _record_skipped_experiment(
    manifest: dict[str, Any],
    name: str,
    stage: str,
    exc: BaseException,
) -> None:
    message = f"{name}/{stage}: {type(exc).__name__}: {exc}"
    text = _record_io_warning(manifest, message)
    manifest.setdefault("skipped_experiments", {})[str(name)] = {
        "name": str(name),
        "status": "skipped",
        "reason": str(stage),
        "message": str(exc),
        "error_type": type(exc).__name__,
        "warning": text,
    }
    summary = manifest.get("summary")
    rows = summary.get("experiments", []) if isinstance(summary, dict) else []
    for row in rows:
        if isinstance(row, dict) and str(row.get("name")) == str(name):
            row["status"] = "skipped"
            row["issue"] = str(exc)
            row["error_type"] = type(exc).__name__
            break


def save_ismrmrd_array(path: Path, array: np.ndarray, group: str):
    """Write an array to an ISMRMRD file under the given group name."""
    _ensure_parent(path)
    # Make writes idempotent: replace the target group instead of appending, so rerunning
    # exports (or changing reconstruction shapes) does not accumulate incompatible entries.
    if path.exists():
        try:
            import h5py

            with h5py.File(path, "a") as f:
                root = f.get("dataset")
                if root is not None and group in root:
                    del root[group]
        except Exception:
            # Fall back to append semantics if the file cannot be opened/edited.
            _LOG.debug("Ignored optional operation failure", exc_info=True)
    dset = ismrmrd.Dataset(str(path), "dataset")
    img = ismrmrd.Image.from_array(np.asarray(array), transpose=False)
    dset.append_image(group, img)
    dset.close()


def load_ismrmrd_array(path: Path, group: str, expected_shape: tuple[int, ...] | None = None) -> np.ndarray:
    """Read the first array stored under a group name from an ISMRMRD file."""
    dset = ismrmrd.Dataset(str(path), "dataset", create_if_needed=False)
    img = dset.read_image(group, 0)
    arr = np.asarray(img.data)
    dset.close()
    if expected_shape:
        try:
            arr = arr.reshape(expected_shape)
        except ValueError:
            _LOG.debug("Ignored optional operation failure", exc_info=True)
    return arr


def _clear_ismrmrd_dataset_item(path: Path, name: str) -> None:
    if not path.exists():
        return
    try:
        import h5py

        with h5py.File(path, "a") as f:
            root = f.get("dataset")
            if root is not None and name in root:
                del root[name]
    except Exception:
        _LOG.debug("Ignored optional operation failure", exc_info=True)


def _set_acquisition_phase_indices(acq: ismrmrd.Acquisition, phase_idx: tuple[int, ...]) -> None:
    if phase_idx:
        acq.idx.kspace_encode_step_1 = int(phase_idx[0])
    if len(phase_idx) >= 2:
        acq.idx.kspace_encode_step_2 = int(phase_idx[1])
    if len(phase_idx) > 2:
        for idx, value in enumerate(phase_idx[2:10]):
            acq.idx.user[idx] = int(value)


def _phase_indices_from_acquisition(
    acq: ismrmrd.Acquisition,
    phase_shape: tuple[int, ...],
    serial_index: int,
) -> tuple[int, ...]:
    if not phase_shape:
        return ()
    values: list[int] = []
    values.append(int(acq.idx.kspace_encode_step_1))
    if len(phase_shape) >= 2:
        values.append(int(acq.idx.kspace_encode_step_2))
    if len(phase_shape) > 2:
        values.extend(int(v) for v in acq.idx.user[: len(phase_shape) - 2])
    candidate = tuple(values[: len(phase_shape)])
    if len(candidate) == len(phase_shape) and all(0 <= idx < size for idx, size in zip(candidate, phase_shape)):
        return candidate
    return tuple(int(v) for v in np.unravel_index(serial_index, phase_shape))


def _set_acquisition_flags(
    acq: ismrmrd.Acquisition,
    phase_idx: tuple[int, ...],
    phase_shape: tuple[int, ...],
    scan_counter: int,
    total_acquisitions: int,
) -> None:
    if not phase_shape or phase_idx[0] == 0:
        acq.setFlag(ismrmrd.ACQ_FIRST_IN_ENCODE_STEP1)
    if not phase_shape or phase_idx[0] == phase_shape[0] - 1:
        acq.setFlag(ismrmrd.ACQ_LAST_IN_ENCODE_STEP1)
    if len(phase_shape) >= 2:
        if phase_idx[1] == 0:
            acq.setFlag(ismrmrd.ACQ_FIRST_IN_ENCODE_STEP2)
        if phase_idx[1] == phase_shape[1] - 1:
            acq.setFlag(ismrmrd.ACQ_LAST_IN_ENCODE_STEP2)
    if scan_counter == total_acquisitions - 1:
        acq.setFlag(ismrmrd.ACQ_LAST_IN_MEASUREMENT)


def save_ismrmrd_kspace(path: Path, array: np.ndarray, *, has_coil_axis: bool = False) -> None:
    """Write raw k-space as standard ISMRMRD acquisition readouts.

    ISMRMRD raw data are stored as one acquisition per readout line, with the
    acquisition payload shaped ``(active_channels, number_of_samples)``.
    ``array`` uses MRax's in-memory convention where the final axis is the
    readout/sample axis and an optional leading axis stores receiver coils.
    """

    _ensure_parent(path)
    _clear_ismrmrd_dataset_item(path, "data")
    arr = np.asarray(array, dtype=np.complex64)
    if arr.ndim == 0:
        arr = arr.reshape((1,))

    if has_coil_axis:
        if arr.ndim == 1:
            arr = arr.reshape((1, arr.shape[0]))
        if arr.ndim < 2:
            raise ValueError(f"K-space with a coil axis must be at least 2D, got shape {arr.shape}.")
        active_channels = int(arr.shape[0])
        phase_shape = tuple(int(v) for v in arr.shape[1:-1])
        readout = int(arr.shape[-1])
    else:
        active_channels = 1
        phase_shape = tuple(int(v) for v in arr.shape[:-1])
        readout = int(arr.shape[-1])

    if readout <= 0:
        raise ValueError(f"K-space readout axis is empty for shape {arr.shape}.")
    if active_channels <= 0:
        raise ValueError(f"K-space channel axis is empty for shape {arr.shape}.")
    if readout > np.iinfo(np.uint16).max or active_channels > np.iinfo(np.uint16).max:
        raise ValueError("ISMRMRD acquisition headers store sample/channel counts as uint16.")

    total_acquisitions = int(np.prod(phase_shape, dtype=np.int64)) if phase_shape else 1
    dset = ismrmrd.Dataset(str(path), "dataset")
    try:
        for scan_counter, phase_idx in enumerate(np.ndindex(*phase_shape) if phase_shape else [()]):
            if has_coil_axis:
                line = arr[(slice(None),) + tuple(phase_idx) + (slice(None),)]
            else:
                line = arr[tuple(phase_idx) + (slice(None),)].reshape((1, readout))
            acq = ismrmrd.Acquisition.from_array(np.asarray(line, dtype=np.complex64))
            acq.scan_counter = int(scan_counter)
            acq.center_sample = int(readout // 2)
            _set_acquisition_phase_indices(acq, tuple(int(v) for v in phase_idx))
            _set_acquisition_flags(
                acq,
                tuple(int(v) for v in phase_idx),
                phase_shape,
                scan_counter,
                total_acquisitions,
            )
            dset.append_acquisition(acq)
    finally:
        dset.close()


def load_ismrmrd_kspace(path: Path, expected_shape: tuple[int, ...] | None = None) -> np.ndarray:
    """Read raw k-space from standard ISMRMRD acquisitions."""

    dset = ismrmrd.Dataset(str(path), "dataset", create_if_needed=False)
    try:
        try:
            count = int(dset.number_of_acquisitions())
        except LookupError:
            count = 0
        if count <= 0:
            raise ValueError(f"No ISMRMRD acquisitions found in k-space file: {path}")

        first = dset.read_acquisition(0)
        active_channels = int(first.active_channels)
        readout = int(first.number_of_samples)
        if expected_shape:
            shape = tuple(int(v) for v in expected_shape)
        else:
            shape = (count, active_channels, readout) if active_channels > 1 else (count, readout)

        if shape and len(shape) >= 2 and shape[0] == active_channels and int(np.prod(shape[1:-1])) == count:
            has_coil_axis = True
            phase_shape = tuple(shape[1:-1])
            output = np.empty(shape, dtype=first.data.dtype)
        elif shape and shape[-1] == readout and active_channels == 1 and int(np.prod(shape[:-1])) == count:
            has_coil_axis = False
            phase_shape = tuple(shape[:-1])
            output = np.empty(shape, dtype=first.data.dtype)
        else:
            has_coil_axis = True
            phase_shape = (count,)
            shape = (active_channels, count, readout)
            output = np.empty(shape, dtype=first.data.dtype)

        for scan_counter in range(count):
            acq = first if scan_counter == 0 else dset.read_acquisition(scan_counter)
            phase_idx = _phase_indices_from_acquisition(acq, phase_shape, scan_counter)
            if has_coil_axis:
                output[(slice(None),) + phase_idx + (slice(None),)] = np.asarray(acq.data)
            else:
                output[phase_idx + (slice(None),)] = np.asarray(acq.data)[0]
        return output
    finally:
        try:
            dset.close()
        except Exception:
            _LOG.debug("Ignored optional operation failure", exc_info=True)


def _axes_from_experiment(experiment: Experiment) -> tuple[list[str], list[str]]:
    """Split dims into spatial vs observable axes preserving order."""
    dim_names = list(experiment.dims.keys())
    spatial = [name for name in dim_names if name in SPATIAL_DIMS]
    observable = [name for name in dim_names if name not in SPATIAL_DIMS]
    return spatial, observable


def _is_non_imaging_experiment(experiment: Experiment, extra_meta: object | None = None) -> bool:
    if bool(getattr(experiment, "_mrax_non_imaging", False)):
        return True
    meta = extra_meta if isinstance(extra_meta, dict) else {}
    if bool(meta.get("non_imaging")):
        return True
    exp_type = str(experiment.exp_type or "").strip().lower()
    return exp_type in {"spectroscopy", "mrs", "fid", "non_imaging", "non-imaging"}


def _format_obs_component(name: str, idx: int, width: int) -> str:
    return f"{name}{idx:0{width}d}"


def _axis_widths(experiment: Experiment) -> dict[str, int]:
    """Width for zero-padding observable indices in filenames."""
    return {name: max(1, len(str(size - 1))) for name, size in experiment.dims.items()}


def _observables_to_manifest(source: Experiment | dict[str, Quantity]) -> dict[str, dict[str, object]]:
    """Serialize observables (values + units) to a manifest-friendly dict."""
    if hasattr(source, "observables"):
        observables = source.observables
    else:
        observables = source
    serialized: dict[str, dict[str, object]] = {}
    for name, qty in observables.items():
        arr = np.asarray(qty.values)
        if arr.size == 1:
            values: object = arr.reshape(()).item()
        else:
            values = arr.tolist()
        serialized[name] = {
            "values": values,
            "unit": qty.si_unit,
        }
    return serialized


def _metadata_to_jsonable(value: Any) -> Any:
    """Best-effort conversion for optional metadata saved beside arrays."""
    if isinstance(value, Quantity):
        return {
            "values": _metadata_to_jsonable(value.values),
            "unit": value.si_unit,
        }
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        arr = np.asarray(value)
        if arr.size <= 200:
            return arr.tolist()
        return {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _metadata_to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        if len(value) > 200:
            return {
                "length": len(value),
                "items_preview": [_metadata_to_jsonable(item) for item in value[:50]],
            }
        return [_metadata_to_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        arr = np.asarray(value)
        if arr.shape:
            return _metadata_to_jsonable(arr)
    except Exception:
        _LOG.debug("Ignored optional operation failure", exc_info=True)
    return str(value)


def _frame_values(values: object, frame_count: int, default: float) -> list[float]:
    if values is None:
        return [default] * frame_count
    if hasattr(values, "tolist") and not isinstance(values, list):
        values = values.tolist()
    if not isinstance(values, list):
        values = [values]
    if len(values) == 1:
        return [float(values[0])] * frame_count
    if len(values) != frame_count:
        if frame_count == 1:
            return [float(values[-1])]
        if len(values) < frame_count:
            return [float(v) for v in values] + [float(values[-1])] * (frame_count - len(values))
        return [float(v) for v in values[:frame_count]]
    return [float(v) for v in values]


def _nifti_params_from_metadata(
    experiment: Experiment,
    metadata: dict | None,
    dataset: object | None = None,
) -> dict[str, object]:
    geometry_params: dict[str, object] = {}
    geometry = getattr(dataset, "spatial_geometry", None)
    geometry_declared = isinstance(geometry, SpatialGeometry)
    if geometry is None and isinstance(metadata, dict):
        geometry_meta = metadata.get("spatial_geometry")
        if isinstance(geometry_meta, dict):
            geometry = SpatialGeometry.from_metadata(geometry_meta)
            geometry_declared = True
    if isinstance(geometry, SpatialGeometry) and geometry.has_orientation:
        if geometry.frame_affines:
            geometry_params["affines"] = [
                np.asarray(value, dtype=float) for value in geometry.frame_affines
            ]
        affine = geometry.affine_for_frame()
        if affine is not None and "affines" not in geometry_params:
            geometry_params["affine"] = affine
    if not metadata:
        return geometry_params
    visu = metadata.get("visu")
    if not isinstance(visu, dict):
        return geometry_params
    if "VisuCoreSize" not in visu or "VisuCoreExtent" not in visu:
        return geometry_params
    try:
        from mrax.io.bruker_import import bruker_affine_from_visu, bruker_word_dtype
    except Exception:
        return geometry_params

    affines = bruker_affine_from_visu(visu)
    if isinstance(affines, list):
        affine = None
        affines_arg = affines
    else:
        affine = affines
        affines_arg = None

    frame_count = experiment.frame_count
    slopes = _frame_values(visu.get("VisuCoreDataSlope", [1.0]), frame_count, 1.0)
    intercepts = _frame_values(visu.get("VisuCoreDataOffs", [0.0]), frame_count, 0.0)

    try:
        raw_dtype = bruker_word_dtype(visu["VisuCoreWordType"])
    except Exception:
        raw_dtype = None
    if raw_dtype is not None and str(visu.get("VisuCoreByteOrder", "")).lower() == "bigendian":
        raw_dtype = raw_dtype.newbyteorder(">")
    if raw_dtype is None:
        slopes = None
        intercepts = None

    return {
        "affine": geometry_params.get("affine") if geometry_declared else affine,
        "affines": geometry_params.get("affines") if geometry_declared else affines_arg,
        "slopes": slopes,
        "intercepts": intercepts,
        "raw_dtype": raw_dtype,
    }


def _match_observable_axis(axis: str, experiment: Experiment) -> str | None:
    """Best-effort mapping from a dim name to a known observable name."""
    if axis in experiment.observables:
        return axis
    if f"{axis}s" in experiment.observables:
        return f"{axis}s"
    if axis.endswith("s") and axis[:-1] in experiment.observables:
        return axis[:-1]

    size = experiment.dims.get(axis)
    if size is None:
        return None
    candidates = [name for name, qty in experiment.observables.items() if qty.shape and qty.shape[0] == size]
    if len(candidates) == 1:
        return candidates[0]
    return None


def _value_from_quantity(quantity: Quantity, idx: int) -> object:
    arr = np.asarray(quantity.values)
    if arr.shape and idx < arr.shape[0]:
        val = arr[idx]
        if np.ndim(val) == 0:
            try:
                return val.item()
            except Exception:
                return float(val)
        return val.tolist()
    return arr.tolist()


def _save_volume_slices(
    array: np.ndarray,
    experiment: Experiment,
    exp_dir: Path,
    manifest_base: Path,
    file_stem: str,
    file_extension: str,
    group: str,
    subdir: str,
) -> list[dict[str, object]]:
    """Write per-observable slices to individual ISMRMRD files."""
    dim_names = list(experiment.dims.keys())
    if array.ndim != len(dim_names):
        raise ValueError(
            f"Array with shape {array.shape} does not match experiment dims {dim_names}."
        )
    spatial_axes, observable_axes = _axes_from_experiment(experiment)
    if len(spatial_axes) < 2:
        raise ValueError("Experiments must define at least x and y dimensions to save images.")
    observable_idxs = [dim_names.index(axis) for axis in observable_axes]
    slice_dir = exp_dir / subdir
    slice_dir.mkdir(parents=True, exist_ok=True)
    widths = _axis_widths(experiment)

    obs_sizes = [experiment.dims[ax] for ax in observable_axes]
    if not observable_axes:
        obs_sizes = [1]

    slices: list[dict[str, object]] = []
    for obs_idx in np.ndindex(*obs_sizes):
        # obs_idx empty tuple when there are no observables.
        idx_map: dict[str, int] = {}
        slicespec = [slice(None)] * array.ndim
        for axis_name, axis_idx, dim in zip(observable_axes, obs_idx, observable_idxs):
            idx_map[axis_name] = int(axis_idx)
            slicespec[dim] = axis_idx

        slice_arr = np.asarray(array[tuple(slicespec)])
        if idx_map:
            suffix = "_".join(
                _format_obs_component(name, idx_map[name], widths[name]) for name in observable_axes
            )
            fname = f"{file_stem}_{suffix}{file_extension}"
        else:
            fname = f"{file_stem}{file_extension}"
        out_path = slice_dir / fname
        save_ismrmrd_array(out_path, slice_arr, group=group)

        # Map observable indices to their corresponding values (if present) for readability.
        obs_values: dict[str, object] = {}
        for axis_name, idx in idx_map.items():
            obs_key = _match_observable_axis(axis_name, experiment)
            if obs_key:
                obs_values[obs_key] = _value_from_quantity(experiment.observables[obs_key], idx)

        slices.append(
            {
                "path": _relative(out_path, manifest_base),
                "observable_indices": idx_map,
                "observable_values": obs_values or {},
                "shape": list(slice_arr.shape),
            }
        )
    return slices


def _save_non_image_array(
    array: np.ndarray,
    exp_dir: Path,
    manifest_base: Path,
    *,
    file_stem: str,
    subdir: str,
) -> str:
    path = exp_dir / subdir / f"{file_stem}.npy"
    _ensure_parent(path)
    np.save(path, np.asarray(array))
    return _relative(path, manifest_base)


def _load_sliced_array(
    base: Path,
    slice_entries: list[dict[str, object]],
    group: str,
    full_shape: tuple[int, ...],
    dim_names: list[str],
    extra_axis: int | None = None,
) -> np.ndarray:
    """Reassemble a volume from per-observable slice files."""
    full_array: np.ndarray | None = None
    for entry in slice_entries:
        arr = load_ismrmrd_array(
            _resolve_manifest_path(base, entry["path"], label="array"),
            group=group,
            expected_shape=tuple(entry.get("shape", ())),
        )
        if full_array is None:
            full_array = np.zeros(full_shape, dtype=arr.dtype)
        idx = _index_from_entry(entry, dim_names)
        if extra_axis is not None and len(full_shape) == len(dim_names) + 1:
            extra_axis = max(0, min(extra_axis, len(idx)))
            idx.insert(extra_axis, slice(None))
        full_array[tuple(idx)] = arr
    if full_array is None:
        full_array = np.empty(full_shape)
    return full_array


def _load_sliced_kspace_array(
    base: Path,
    slice_entries: list[dict[str, object]],
    full_shape: tuple[int, ...],
    dim_names: list[str],
    extra_axis: int | None = None,
) -> np.ndarray:
    """Reassemble k-space slices from ISMRMRD acquisition records."""
    full_array: np.ndarray | None = None
    for entry in slice_entries:
        arr = load_ismrmrd_kspace(
            _resolve_manifest_path(base, entry["path"], label="k-space"),
            expected_shape=tuple(entry.get("shape", ())),
        )
        if full_array is None:
            full_array = np.zeros(full_shape, dtype=arr.dtype)
        idx = _index_from_entry(entry, dim_names)
        if extra_axis is not None and len(full_shape) == len(dim_names) + 1:
            extra_axis = max(0, min(extra_axis, len(idx)))
            idx.insert(extra_axis, slice(None))
        full_array[tuple(idx)] = arr
    if full_array is None:
        full_array = np.empty(full_shape)
    return full_array


def _index_from_entry(entry: dict[str, object], dim_names: list[str]) -> list[object]:
    """Recreate numpy indexing tuple from manifest entry."""
    indices = entry.get("observable_indices", {}) or {}
    idx: list[object] = []
    for name in dim_names:
        if name in indices:
            idx.append(int(indices[name]))
        else:
            idx.append(slice(None))
    return idx


def _reshape_kspace_for_slicing(
    kspace: np.ndarray,
    experiment: Experiment,
) -> tuple[np.ndarray, bool]:
    """Align k-space array to experiment dims for slicing; return (array, has_coils)."""
    non_spatial = experiment.non_spatial_dim_names
    spatial = experiment.spatial_dim_names
    spatial_shape = experiment.spatial_shape
    frame_count = experiment.frame_count

    if not spatial:
        return kspace, False

    expected = tuple(experiment.dims[name] for name in non_spatial + spatial)
    if kspace.shape == expected:
        return kspace, False

    if kspace.ndim == len(spatial) + 1 and int(kspace.shape[0]) == int(frame_count):
        target = tuple(experiment.dims[name] for name in non_spatial) + spatial_shape
        try:
            return kspace.reshape(target), False
        except Exception:
            return kspace, False

    if kspace.ndim == len(spatial) + 2 and int(kspace.shape[0]) == int(frame_count):
        coil_count = int(kspace.shape[1])
        target = tuple(experiment.dims[name] for name in non_spatial) + (coil_count,) + spatial_shape
        try:
            return kspace.reshape(target), True
        except Exception:
            return kspace, True

    if kspace.ndim == len(non_spatial) + len(spatial) + 1:
        if tuple(int(v) for v in kspace.shape[-len(spatial) :]) == spatial_shape:
            return kspace, True

    return kspace, False


def _relative(path: Path, base: Path) -> str:
    return str(path.relative_to(base))


def _study_summary(study: Study, targets: list[str] | None) -> dict[str, Any]:
    """Structured summary for manifest readability."""
    order = study.execution_order(targets)
    by_name = study.entries_by_name()

    items: list[dict[str, Any]] = []
    for name in order:
        if name not in by_name:
            continue
        experiment, model, dataset = by_name[name]
        items.append(
            {
                "name": name,
                "model": model.name,
                "exp_type": experiment.exp_type,
                "dataset": dataset.name,
                "dims": experiment.dims,
                "observables": sorted(experiment.observables.keys()),
            }
        )
    return {"experiments": items}


def save_study_results(
    study: Study,
    results: list[dict[str, Any]],
    output_dir: str | Path,
    study_name: str = "study",
    targets: list[str] | None = None,
    file_extension: str = ".mrd",
    experiment_metadata: dict[str, Any] | None = None,
    save_nifti: bool = False,
):
    """Persist study inputs/outputs to disk with a manifest that links files."""

    if not file_extension.startswith("."):
        raise ValueError("file_extension must start with '.' (e.g., '.mrd' or '.h5')")

    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)
    exp_root = base / "experiments"
    exp_root.mkdir(exist_ok=True)

    result_map: Dict[str, dict] = {entry["name"]: entry for entry in results}
    study_globals = getattr(study, "study_globals", {}) or {}
    manifest: Dict[str, Any] = {
        "format": "mrax.study",
        "format_version": 1,
        MRAX_VERSION_TAG: MRAX_VERSION,
        "study_name": study_name,
        "experiments": {},
        "targets": targets or [],
        "summary": _study_summary(study, targets),
        "study_globals": _observables_to_manifest(study_globals),
        "warnings": [],
        "skipped_experiments": {},
        "spatial_geometry": {},
    }

    order = study.execution_order(targets)
    required_names = set(order)
    entry_map = study.entries_by_name()

    for name in order:
        if name not in required_names or name not in entry_map:
            continue

        experiment, model, dataset = entry_map[name]
        extra_meta = experiment_metadata.get(name) if experiment_metadata else None
        experiment_to_use = experiment
        is_non_imaging = _is_non_imaging_experiment(experiment_to_use, extra_meta)

        exp_dir = exp_root / name
        non_image_paths: dict[str, str] = {}
        try:
            if is_non_imaging:
                mr_image_slices = []
                non_image_paths["mr_image"] = _save_non_image_array(
                    np.asarray(dataset.data),
                    exp_dir,
                    base,
                    file_stem="mr_data",
                    subdir="non_images",
                )
            else:
                mr_image_slices = _save_volume_slices(
                    np.asarray(dataset.data),
                    experiment_to_use,
                    exp_dir,
                    base,
                    file_stem="mr_image",
                    file_extension=file_extension,
                    group="image",
                    subdir="mr_images",
                )
        except Exception as exc:
            _record_skipped_experiment(manifest, str(name), "mr_image", exc)
            continue

        nifti_paths: dict[str, object] = {}
        if save_nifti and not is_non_imaging:
            try:
                from mrax.io.nifti import export_nifti_series, save_nifti

                nifti_root = exp_dir / "nifti"
                nifti_params = _nifti_params_from_metadata(experiment_to_use, extra_meta, dataset)
                nifti_image_dir = nifti_root / "mr_images"
                nifti_files = export_nifti_series(
                    experiment_to_use,
                    dataset,
                    nifti_image_dir,
                    stem="mr_image",
                    **nifti_params,
                )
                nifti_paths["mr_images"] = [_relative(p, base) for p in nifti_files]

                affine_map = nifti_params.get("affine")
                if affine_map is None and nifti_params.get("affines"):
                    affine_map = nifti_params["affines"][0]
                if affine_map is None:
                    affine_map = np.eye(4, dtype=float)

                nifti_map_paths: Dict[str, str] = {}
                for field_name, arr in result_map.get(name, {}).get("fields", {}).items():
                    data = arr.values if isinstance(arr, Quantity) else arr
                    data = np.asarray(data)
                    if data.ndim not in (2, 3):
                        continue
                    map_path = nifti_root / "fitted_maps" / f"{field_name}.nii"
                    _ensure_parent(map_path)
                    save_nifti(map_path, data, affine_map)
                    nifti_map_paths[field_name] = _relative(map_path, base)
                if nifti_map_paths:
                    nifti_paths["fitted_maps"] = nifti_map_paths
            except Exception as exc:
                _record_io_warning(manifest, f"{name}/nifti: {type(exc).__name__}: {exc}")

        kspace_slices: list[dict[str, object]] = []
        kspace_save_shape: list[int] | None = None
        if getattr(dataset, "kspace", None) is not None:
            try:
                if is_non_imaging:
                    kspace_save_shape = list(np.asarray(dataset.kspace).shape)
                    non_image_paths["raw_kspace"] = _save_non_image_array(
                        np.asarray(dataset.kspace),
                        exp_dir,
                        base,
                        file_stem="raw_kspace",
                        subdir="non_images",
                    )
                else:
                    kspace_arr, has_coils = _reshape_kspace_for_slicing(np.asarray(dataset.kspace), experiment_to_use)
                    kspace_save_shape = list(np.asarray(kspace_arr).shape)
                    dim_names = list(experiment_to_use.dims.keys())
                    non_spatial = experiment_to_use.non_spatial_dim_names
                    for entry in mr_image_slices:
                        idx = _index_from_entry(entry, dim_names)
                        if has_coils and kspace_arr.ndim == len(dim_names) + 1:
                            idx.insert(len(non_spatial), slice(None))
                        slice_arr = kspace_arr[tuple(idx)]
                        path = base / entry["path"]
                        save_ismrmrd_kspace(path, np.asarray(slice_arr), has_coil_axis=has_coils)
                        kspace_slices.append(
                            {
                                "path": entry["path"],
                                "observable_indices": entry.get("observable_indices", {}),
                                "observable_values": entry.get("observable_values", {}),
                                "shape": list(np.asarray(slice_arr).shape),
                                "storage": "ismrmrd_acquisitions",
                            }
                        )
            except Exception as exc:
                _record_io_warning(manifest, f"{name}/raw_kspace: {type(exc).__name__}: {exc}")
                kspace_slices = []
                kspace_save_shape = None

        if name not in result_map:
            _record_skipped_experiment(manifest, str(name), "results", KeyError(f"No results provided for experiment '{name}'"))
            continue
        fields = result_map[name].get("fields", {})

        fitted_paths: Dict[str, str] = {}
        fitted_units: Dict[str, str] = {}
        for field_name, arr in fields.items():
            try:
                map_path = exp_dir / "fitted_maps" / f"{field_name}{file_extension}"
                save_ismrmrd_array(map_path, np.asarray(arr), group="map")
                fitted_paths[field_name] = _relative(map_path, base)
                if isinstance(arr, Quantity):
                    fitted_units[field_name] = arr.si_unit
                else:
                    fitted_units[field_name] = getattr(model, "parameter_units", {}).get(field_name, "")
            except Exception as exc:
                _record_io_warning(manifest, f"{name}/fitted_maps/{field_name}: {type(exc).__name__}: {exc}")

        paths = {
            "mr_images": mr_image_slices,
            "raw_kspace_images": kspace_slices or [],
            "fitted_maps": fitted_paths,
        }
        if non_image_paths:
            paths["non_images"] = non_image_paths
        if save_nifti:
            paths["nifti"] = nifti_paths

        entry_meta = {
            "model": model.name,
            "exp_type": experiment_to_use.exp_type,
            "dataset_name": dataset.name,
            "dims": experiment.dims,
            "observables": sorted(experiment_to_use.observables.keys()),
            "observable_values": _observables_to_manifest(experiment_to_use),
            "observable_units": {k: v.si_unit for k, v in experiment_to_use.observables.items()},
            "paths": paths,
            "shapes": {
                "mr_image": list(np.asarray(dataset.data).shape),
                "raw_kspace": kspace_save_shape if kspace_save_shape is not None else None,
                "fitted_maps": {k: list(np.asarray(fields[k]).shape) for k in fitted_paths},
            },
            "fitted_map_units": fitted_units,
            "plots": {},
        }
        geometry = getattr(dataset, "spatial_geometry", None)
        if geometry is None and experiment_to_use.spatial_shape and not is_non_imaging:
            geometry = SpatialGeometry(
                spatial_shape=experiment_to_use.spatial_shape,
                source="unknown",
                status="unknown",
            )
        if isinstance(geometry, SpatialGeometry):
            entry_meta["spatial_geometry"] = geometry.to_metadata()
            manifest["spatial_geometry"][name] = geometry.to_metadata()
        entry_meta[MRAX_VERSION_TAG] = MRAX_VERSION
        if is_non_imaging:
            entry_meta["non_imaging"] = True
        if extra_meta is not None:
            entry_meta["extra_metadata"] = _metadata_to_jsonable(extra_meta)

        # Write human-readable per-experiment metadata for quick inspection.
        metadata_path = exp_dir / "metadata.json"
        _ensure_parent(metadata_path)
        try:
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(entry_meta, f, indent=2)
        except Exception as exc:
            _record_skipped_experiment(manifest, str(name), "metadata", exc)
            continue

        manifest["experiments"][name] = {"metadata": _relative(metadata_path, base)}

    with open(base / MANIFEST_NAME, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return base / MANIFEST_NAME


def load_study_results(manifest_path: str | Path) -> dict[str, Any]:
    """Load study manifest plus arrays from ISMRMRD files."""

    manifest_file = Path(manifest_path)
    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    version_info = validate_mrax_file_version(manifest, file_kind="MRax study manifest")

    base = manifest_file.parent
    load_warnings: list[str] = []

    def warn_skipped(message: str) -> None:
        text = f"warning: skipped {message}"
        load_warnings.append(text)
        warnings.warn(text, RuntimeWarning, stacklevel=2)

    if version_info.tag_missing:
        text = missing_mrax_version_message("MRax study manifest")
        load_warnings.append(text)
        warnings.warn(text, RuntimeWarning, stacklevel=2)

    study_globals: Dict[str, Quantity] = {}
    for name, payload in (manifest.get("study_globals") or {}).items():
        try:
            values = payload.get("values")
            unit = payload.get("unit", "")
            if values is None:
                continue
            study_globals[name] = Quantity(jnp.asarray(values), unit)
        except Exception as exc:
            warn_skipped(f"study_globals/{name}: {type(exc).__name__}: {exc}")

    experiments: Dict[str, Any] = {}
    for name, entry in manifest.get("experiments", {}).items():
        try:
            meta_file = _resolve_manifest_path(base, entry["metadata"], label="metadata")
            with open(meta_file, "r", encoding="utf-8") as mf:
                entry = json.load(mf)
        except Exception as exc:
            warn_skipped(f"{name}/metadata: {type(exc).__name__}: {exc}")
            continue
        entry_version = validate_mrax_file_version(
            entry,
            file_kind=f"MRax experiment metadata for '{name}'",
        )
        if entry_version.tag_missing and not version_info.tag_missing:
            text = missing_mrax_version_message(f"MRax experiment metadata for '{name}'")
            load_warnings.append(text)
            warnings.warn(text, RuntimeWarning, stacklevel=2)

        try:
            dim_names = list(entry["dims"].keys())
            non_spatial_count = len([n for n in dim_names if n not in SPATIAL_DIMS])
            shapes = entry["shapes"]

            paths = entry["paths"]
            mr_image_slices = paths.get("mr_images", [])
            non_image_paths = paths.get("non_images", {}) or {}
            if mr_image_slices:
                mr_image = _load_sliced_array(
                    base,
                    mr_image_slices,
                    group="image",
                    full_shape=tuple(shapes.get("mr_image", ())),
                    dim_names=dim_names,
                )
            elif non_image_paths.get("mr_image"):
                mr_image = np.load(
                    _resolve_manifest_path(base, non_image_paths["mr_image"], label="image")
                )
            else:
                mr_image = np.asarray([])
        except Exception as exc:
            warn_skipped(f"{name}/mr_image: {type(exc).__name__}: {exc}")
            continue

        kspace = None
        raw_kspace_slices = paths.get("raw_kspace_images", [])
        try:
            if raw_kspace_slices:
                kspace = _load_sliced_kspace_array(
                    base,
                    raw_kspace_slices,
                    full_shape=tuple(shapes.get("raw_kspace", ())),
                    dim_names=dim_names,
                    extra_axis=non_spatial_count,
                )
            elif non_image_paths.get("raw_kspace"):
                kspace = np.load(
                    _resolve_manifest_path(base, non_image_paths["raw_kspace"], label="k-space")
                )
        except Exception as exc:
            warn_skipped(f"{name}/raw_kspace: {type(exc).__name__}: {exc}")
            kspace = None

        fitted_maps = {}
        fitted_shapes = shapes.get("fitted_maps", {})
        fitted_units = entry.get("fitted_map_units", {})
        for key, rel_path in paths.get("fitted_maps", {}).items():
            try:
                arr = load_ismrmrd_array(
                    _resolve_manifest_path(base, rel_path, label="fitted map"),
                    group="map",
                    expected_shape=tuple(fitted_shapes.get(key, ())),
                )
                unit = fitted_units.get(key, "")
                if unit:
                    fitted_maps[key] = Quantity(jnp.asarray(arr), unit)
                else:
                    fitted_maps[key] = arr
            except Exception as exc:
                warn_skipped(f"{name}/fitted_maps/{key}: {type(exc).__name__}: {exc}")

        observable_values = entry.get("observable_values", {})
        observable_units = entry.get("observable_units", {})
        observables: dict[str, Quantity] = {}
        for obs_name, payload in observable_values.items():
            try:
                unit = payload.get("unit") or observable_units.get(obs_name, "")
                values = payload.get("values")
                if values is not None:
                    observables[obs_name] = Quantity(jnp.asarray(values), unit)
            except Exception as exc:
                warn_skipped(f"{name}/observables/{obs_name}: {type(exc).__name__}: {exc}")

        experiments[name] = {
            "mr_image": mr_image,
            "raw_kspace": kspace,
            "fitted_maps": fitted_maps,
            "model": entry.get("model"),
            "exp_type": entry.get("exp_type"),
            "data_role": (entry.get("extra_metadata") or {}).get("data_role") or entry.get("exp_type"),
            "observable_units": entry.get("observable_units", {}),
            "observables": observables,
            "fitted_map_units": fitted_units,
            "extra_metadata": entry.get("extra_metadata"),
            "spatial_geometry": entry.get("spatial_geometry")
            or (manifest.get("spatial_geometry") or {}).get(name),
            "non_imaging": bool(entry.get("non_imaging")),
            "plots": entry.get("plots", {}),
        }

    return {
        "manifest": manifest,
        "experiments": experiments,
        "study_globals": study_globals,
        "warnings": list(manifest.get("warnings") or []) + load_warnings,
    }


def load_precomputed_fields(manifest_path: str | Path) -> dict[str, dict]:
    """Convenience loader to seed cached fields before executing a planned study fit."""

    data = load_study_results(manifest_path)
    fields = {}
    for name, entry in data["experiments"].items():
        if entry["fitted_maps"]:
            fields[name] = entry["fitted_maps"]
    return fields
