"""Shared study and experiment import helpers."""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

import jax.numpy as jnp
import numpy as np

from mrax.data_struct import Quantity
from mrax.data_struct.dimensions import (
    dim_observable_map,
    frame_count_from_dims,
    non_spatial_names,
    spatial_names,
)
from mrax.versioning import missing_mrax_version_message, validate_mrax_file_version

from ..models import normalize_model_key
from ..services.common import _metadata_to_jsonable as _metadata_to_jsonable
from ..services.package import _find_study_manifest as _find_study_manifest

_LOG = logging.getLogger(__name__)


def _detect_import_vendor(root: Path) -> str:
    if _looks_like_bruker_study(root):
        return "Bruker"
    if (
        root.is_file()
        and root.suffix.lower() == ".seq"
        or root.is_dir()
        and any(p.is_file() and p.suffix.lower() == ".seq" for p in root.rglob("*"))
    ):
        return "Pulseq"
    if _looks_like_dicom_study(root):
        return "DICOM"
    return "Unknown"


def _looks_like_bruker_study(root: Path) -> bool:
    if not root.is_dir():
        return False
    from mrax.io.bruker_parameters import discover_bruker_experiments

    return any(
        (entry / "method").is_file() or (entry / "acqp").is_file()
        for entry in discover_bruker_experiments(root)
    )


def _looks_like_dicom_study(root: Path) -> bool:
    if not root.is_dir():
        return False
    try:
        from mrax.io.dicom import discover_dicom_series

        return bool(discover_dicom_series(root, recursive=True))
    except (OSError, ValueError):
        return False


def _detect_bruker_import_summary(root: Path, pdata_index="1") -> list[dict[str, Any]]:
    try:
        from mrax.io.bruker_metadata import _resolve_pdata_index, detect_type
        from mrax.io.bruker_parameters import discover_bruker_experiments, parse_jcamp
        from mrax.library.default_experiments import default_model_for_type
    except Exception:
        return []

    rows: list[dict[str, Any]] = []
    model_names = {}
    for scan_dir in discover_bruker_experiments(root):
        method_path = scan_dir / "method"
        selected_pdata = _resolve_pdata_index(pdata_index, scan_dir.name)
        if (
            not selected_pdata
            or selected_pdata in {".", ".."}
            or "/" in selected_pdata
            or "\\" in selected_pdata
        ):
            raise ValueError("Choose a reconstruction inside the experiment pdata folder")
        pdata_root = scan_dir / "pdata"
        choices = (
            sorted(
                (p.name for p in pdata_root.iterdir() if p.is_dir() and (p / "2dseq").is_file()),
                key=lambda name: (not name.isdigit(), int(name) if name.isdigit() else name),
            )
            if pdata_root.is_dir()
            else []
        )
        pdata_dir = pdata_root / selected_pdata
        visu_path = pdata_dir / "visu_pars"
        if not method_path.is_file():
            continue
        try:
            method = parse_jcamp(method_path)
        except Exception:
            method = {}
        try:
            visu = parse_jcamp(visu_path) if visu_path.is_file() else {}
        except Exception:
            visu = {}
        frame_count = _detect_bruker_frame_count(visu, method)
        try:
            exp_type = detect_type(visu, method, frame_count=frame_count)
        except Exception:
            exp_type = "anatomical"
        if exp_type not in model_names:
            model_names[exp_type] = (
                normalize_model_key(default_model_for_type(exp_type).name) or "identity"
            )
        rows.append(
            {
                "scan": scan_dir.name,
                "pdata_index": selected_pdata,
                "reconstructions": choices,
                "importable": (pdata_dir / "2dseq").is_file(),
                "image_bytes": (pdata_dir / "2dseq").stat().st_size
                if (pdata_dir / "2dseq").is_file()
                else 0,
                "has_raw": any((scan_dir / name).is_file() for name in ("fid", "ser")),
                "method": method.get("Method") or method.get("PVM_Method") or "",
                "protocol": visu.get("VisuAcquisitionProtocol")
                or method.get("ACQ_scan_name")
                or "",
                "frame_count": frame_count,
                "detected_type": exp_type,
                "detected_model": model_names[exp_type],
                "field_strength_mhz": _detect_bruker_field_strength_mhz(method, visu, scan_dir),
            }
        )
    return rows


def _detect_bruker_frame_count(visu: dict[str, Any], method: dict[str, Any]) -> int | None:
    for key in ("VisuCoreFrameCount", "VisuFGOrderDescDim", "PVM_NRepetitions"):
        value = visu.get(key, method.get(key))
        numeric = _numeric_values(value)
        if numeric:
            count = int(round(numeric[0]))
            if count > 0:
                return count
    return None


def _detect_bruker_field_strength_mhz(
    method: dict[str, Any], visu: dict[str, Any], scan_dir: Path
) -> float | None:
    values = []
    for key in ("PVM_FrqWork", "BF1", "SFO1", "ACQ_imaging_frequency"):
        values.extend(_numeric_values(method.get(key)))
        values.extend(_numeric_values(visu.get(key)))
    acqp_path = scan_dir / "acqp"
    if acqp_path.is_file():
        try:
            from mrax.io.bruker_parameters import parse_jcamp

            acqp = parse_jcamp(acqp_path)
            for key in ("BF1", "SFO1", "ACQ_imaging_frequency"):
                values.extend(_numeric_values(acqp.get(key)))
        except Exception:
            _LOG.debug("Ignored optional GUI operation failure", exc_info=True)
    for value in values:
        if 1.0 <= float(value) <= 2000.0:
            return float(value)
    return None


def _numeric_values(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, (int, float, np.number)):
        return [float(value)]
    if isinstance(value, np.ndarray):
        return [float(v) for v in value.reshape((-1,)) if _is_number(v)]
    if isinstance(value, (list, tuple)):
        out: list[float] = []
        for item in value:
            out.extend(_numeric_values(item))
        return out
    try:
        return [float(value)]
    except Exception:
        return []


def _is_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except Exception:
        return False


def _default_dim_order(ndim: int, source: str = "") -> tuple[str, ...]:
    ndim = int(ndim)
    if ndim <= 0:
        return ("sample",)
    if ndim == 1:
        return ("sample",)
    if ndim == 2:
        return ("x", "y")
    if ndim == 3 and source == "nifti":
        return ("x", "y", "z")
    if ndim == 3:
        return ("frame", "x", "y")
    if ndim == 4:
        return ("frame", "z", "x", "y")
    return tuple([f"axis_{idx}" for idx in range(ndim - 2)] + ["x", "y"])


def _normalize_memory_order(memory_order: str | None) -> str:
    order = str(memory_order or "C").strip().upper()
    if order in {"C", "ROW", "ROW-MAJOR", "ROW_MAJOR"}:
        return "C"
    if order in {"F", "FORTRAN", "COLUMN", "COLUMN-MAJOR", "COLUMN_MAJOR"}:
        return "F"
    raise ValueError("Memory order must be C / row-major or Fortran / column-major")


def _dtype_with_endian(dtype: str, endian: str | None = "native") -> np.dtype:
    np_dtype = np.dtype(str(dtype or "float32"))
    endian_key = str(endian or "native").strip().lower()
    if np_dtype.itemsize <= 1 or endian_key in {"", "native", "=", "|"}:
        return np_dtype
    if endian_key in {"little", "little endian", "le", "<"}:
        return np_dtype.newbyteorder("<")
    if endian_key in {"big", "big endian", "be", ">"}:
        return np_dtype.newbyteorder(">")
    raise ValueError("Raw endian must be native, little endian, or big endian")


def _reshape_import_array(
    array: Any, shape: Sequence[int], *, memory_order: str = "C"
) -> np.ndarray:
    arr = np.asarray(array)
    target = tuple(int(size) for size in shape)
    if not target:
        return arr
    if int(np.prod(target, dtype=np.int64)) != int(arr.size):
        raise ValueError(f"Cannot reshape array with {arr.size} value(s) to {target}.")
    return arr.reshape(target, order=_normalize_memory_order(memory_order))


def _load_raw_binary_array(
    path: Path,
    *,
    dtype: str,
    shape: Sequence[int],
    endian: str = "native",
    byte_offset: int = 0,
    memory_order: str = "C",
) -> np.ndarray:
    if path is None or not Path(path).is_file():
        raise FileNotFoundError("Choose a raw binary data file")
    target = tuple(int(size) for size in shape)
    if not target:
        raise ValueError("Raw binary import requires an explicit array shape")
    np_dtype = _dtype_with_endian(dtype, endian)
    with open(path, "rb") as handle:
        handle.seek(max(0, int(byte_offset)))
        arr = np.fromfile(handle, dtype=np_dtype)
    expected = int(np.prod(target, dtype=np.int64))
    if int(arr.size) != expected:
        raise ValueError(
            f"Raw file contains {arr.size} {np_dtype} value(s), expected {expected} for shape {target}."
        )
    return arr.reshape(target, order=_normalize_memory_order(memory_order))


def _load_numpy_array_file(
    path: Path,
    *,
    shape: Sequence[int] = (),
    npz_key: str | None = None,
    memory_order: str = "C",
) -> np.ndarray:
    if path is None or not Path(path).is_file():
        raise FileNotFoundError("Choose a NumPy .npy or .npz file")
    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        with loaded as npz:
            keys = list(npz.files)
            if not keys:
                raise ValueError(f"NPZ file {path} does not contain arrays")
            key = npz_key or keys[0]
            if key not in npz.files:
                raise KeyError(
                    f"NPZ file {path} does not contain array key '{key}'. Available: {', '.join(keys)}"
                )
            arr = np.asarray(npz[key])
    else:
        arr = np.asarray(loaded)
    return _reshape_import_array(arr, shape, memory_order=memory_order)


def load_experiment_array(
    *,
    source: str,
    path: Path | None,
    shape: Sequence[int],
    dtype: str,
    endian: str = "native",
    byte_offset: int = 0,
    memory_order: str = "C",
    npz_key: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    source = str(source or "blank")
    metadata: dict[str, Any] = {"import_format": source}
    if source == "blank":
        target = tuple(int(size) for size in shape)
        if not target:
            raise ValueError("Blank image import requires an array shape")
        return np.zeros(target, dtype=np.float32), metadata
    if path is None:
        raise FileNotFoundError("Choose a data file")
    metadata["source_path"] = str(path)
    if source == "nifti":
        from mrax.io.nifti import load_nifti, nifti_spatial_geometry

        arr, affine = load_nifti(path)
        metadata["affine"] = _metadata_to_jsonable(np.asarray(affine))
        metadata["spatial_geometry"] = nifti_spatial_geometry(
            affine,
            spatial_shape=arr.shape,
        ).to_metadata()
        if shape:
            metadata["memory_order"] = _normalize_memory_order(memory_order)
        return _reshape_import_array(arr, shape, memory_order=memory_order), metadata
    if source == "numpy":
        if npz_key:
            metadata["npz_key"] = npz_key
        if shape:
            metadata["memory_order"] = _normalize_memory_order(memory_order)
        return _load_numpy_array_file(
            path, shape=shape, npz_key=npz_key, memory_order=memory_order
        ), metadata
    if source == "raw":
        metadata.update(
            {
                "dtype": dtype,
                "endian": str(endian or "native"),
                "byte_offset": int(byte_offset),
                "memory_order": _normalize_memory_order(memory_order),
            }
        )
        return _load_raw_binary_array(
            path,
            dtype=dtype,
            endian=endian,
            shape=shape,
            byte_offset=byte_offset,
            memory_order=memory_order,
        ), metadata
    raise ValueError(f"Unsupported experiment data source '{source}'")


def load_dicom_experiment(path: Path) -> dict[str, Any]:
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError("Choose a readable DICOM image file")

    from mrax.io.dicom import load_dicom_series
    from mrax.library.default_experiments import default_model_for_type

    experiment, dataset, dicom_metadata = load_dicom_series(path)
    exp_type = str(experiment.exp_type or "anatomical")
    model = default_model_for_type(exp_type)
    extra_metadata = dict(dicom_metadata)
    extra_metadata.update(
        {
            "import_format": "dicom",
            "source_path": str(path),
        }
    )
    return {
        "kind": "experiment",
        "source": "dicom",
        "path": path,
        "name": str(dataset.name or path.stem or "dicom"),
        "data": np.asarray(dataset.data),
        "dims": dict(experiment.dims),
        "exp_type": exp_type,
        "model_name": str(model.name or "identity"),
        "observables": dict(experiment.observables),
        "observable_compatibility": [],
        "extra_metadata": extra_metadata,
    }


def _observables_from_json_text(text: str) -> dict[str, Quantity]:
    payload = json.loads(str(text or "{}"))
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise ValueError("Observables JSON must be an object keyed by observable name")
    observables: dict[str, Quantity] = {}
    for name, row in payload.items():
        if isinstance(row, Mapping):
            unit = str(row.get("unit") or row.get("si_unit") or "").strip()
            values = row.get("values", row.get("value"))
        elif isinstance(row, (list, tuple)) and len(row) == 2 and isinstance(row[1], str):
            values, unit = row
            unit = str(unit).strip()
        else:
            raise ValueError(
                f'Observable \'{name}\' must be {{"unit": "...", "values": ...}} or [values, unit].'
            )
        if not unit:
            raise ValueError(f"Observable '{name}' is missing a unit")
        observables[str(name)] = Quantity(jnp.asarray(values), unit)
    return observables


def _observables_to_json_text(observables: Mapping[str, Any]) -> str:
    payload = {}
    for name, value in dict(observables or {}).items():
        if isinstance(value, Quantity):
            payload[str(name)] = {
                "unit": value.si_unit,
                "values": _metadata_to_jsonable(np.asarray(value.values)),
            }
        else:
            payload[str(name)] = {"unit": "", "values": _metadata_to_jsonable(np.asarray(value))}
    return json.dumps(payload, indent=2, sort_keys=True)


def _apply_observable_dim_compatibility(
    observables: Mapping[str, Quantity],
    dims: Mapping[str, int],
    *,
    exp_type: str | None = None,
    model_name: str | None = None,
    policy: str = "fail",
) -> tuple[dict[str, Quantity], list[str]]:
    """Validate/adjust observable leading axes against imported non-spatial dims."""

    policy_key = _normalize_observable_mismatch_policy(policy)
    dims_map = {str(key): int(value) for key, value in dict(dims or {}).items()}
    if not observables or not dims_map:
        return dict(observables or {}), [
            "observable compatibility: no non-scalar observables to check"
        ]
    updated: dict[str, Quantity] = {}
    lines: list[str] = []
    failures: list[str] = []
    adjusted = 0

    for name, qty in require_quantity_mapping(observables).items():
        arr = np.asarray(qty.values)
        target = _observable_target_for_import(
            name, arr, dims_map, exp_type=exp_type, model_name=model_name
        )
        if target is None:
            updated[name] = qty
            continue
        axis_names, target_shape = target
        axis_count = len(target_shape)
        current_shape = tuple(int(v) for v in arr.shape[:axis_count])
        if current_shape == target_shape:
            updated[name] = qty
            continue
        description = (
            f"{name}: leading {', '.join(axis_names)} shape {current_shape} "
            f"does not match imported dims {target_shape}"
        )
        if policy_key == "fail":
            failures.append(description)
            updated[name] = qty
            continue
        try:
            resized, actions = _resize_observable_leading_axes(arr, target_shape, policy_key)
        except ValueError as exc:
            failures.append(f"{description}; {exc}")
            updated[name] = qty
            continue
        updated[name] = Quantity(jnp.asarray(resized), qty.si_unit)
        adjusted += 1
        detail = "; ".join(actions) if actions else "no change"
        lines.append(f"observable adjusted: {description}; {detail}")

    if failures:
        suffix = " Select a trim/extend policy or edit the observables JSON."
        raise ValueError("Observable shape mismatch. " + " | ".join(failures) + suffix)
    if adjusted:
        lines.insert(
            0,
            f"observable compatibility: adjusted {adjusted} observable(s) using policy '{policy_key}'",
        )
    else:
        lines.append("observable compatibility: checked, no mismatches")
    return updated, lines


def require_quantity_mapping(observables: Mapping[str, Quantity]) -> dict[str, Quantity]:
    out: dict[str, Quantity] = {}
    for key, value in dict(observables or {}).items():
        if not isinstance(value, Quantity):
            raise TypeError(f"Observable '{key}' must be a Quantity")
        out[str(key)] = value
    return out


def _normalize_observable_mismatch_policy(policy: str | None) -> str:
    key = str(policy or "fail").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "none": "fail",
        "error": "fail",
        "trim_longer": "trim",
        "extend_shorter": "extend",
        "trim_extend": "trim_extend",
        "trim_or_extend": "trim_extend",
        "fit": "trim_extend",
    }
    key = aliases.get(key, key)
    if key not in {"fail", "trim", "extend", "trim_extend"}:
        raise ValueError("Observable mismatch policy must be fail, trim, extend, or trim_extend")
    return key


def _observable_target_for_import(
    name: str,
    arr: np.ndarray,
    dims: Mapping[str, int],
    *,
    exp_type: str | None = None,
    model_name: str | None = None,
) -> tuple[tuple[str, ...], tuple[int, ...]] | None:
    arr = np.asarray(arr)
    if arr.ndim == 0:
        return None
    dims_map = {str(key): int(value) for key, value in dict(dims or {}).items()}
    non_spatial = tuple(non_spatial_names(dims_map))
    if not non_spatial:
        return None
    spatial_shape = tuple(int(dims_map[key]) for key in spatial_names(dims_map))
    if spatial_shape and _observable_name_is_spatial_like(name):
        return None
    frame_count = int(frame_count_from_dims(dims_map))

    target_dim = _observable_target_dim_name(
        name, dims_map, exp_type=exp_type, model_name=model_name
    )
    if target_dim:
        if target_dim == "frame":
            return (("frame",), (frame_count,))
        if target_dim in non_spatial:
            return ((target_dim,), (int(dims_map[target_dim]),))

    if arr.ndim >= len(non_spatial) and len(non_spatial) > 1 and arr.ndim == len(non_spatial):
        return (non_spatial, tuple(int(dims_map[key]) for key in non_spatial))

    if arr.shape[0] == 1:
        return None
    if frame_count > 1:
        return (("frame",), (frame_count,))
    return None


def _observable_target_dim_name(
    name: str,
    dims: Mapping[str, int],
    *,
    exp_type: str | None = None,
    model_name: str | None = None,
) -> str | None:
    dims_map = {str(key): int(value) for key, value in dict(dims or {}).items()}
    non_spatial = set(non_spatial_names(dims_map))
    norm = _normalize_observable_name(name)
    dim_maps: list[Mapping[str, str | None]] = []
    try:
        dim_maps.append(dim_observable_map(exp_type, model_name))
    except Exception:
        _LOG.debug("Ignored optional GUI operation failure", exc_info=True)
    try:
        dim_maps.append(dim_observable_map(exp_type, exp_type))
    except Exception:
        _LOG.debug("Ignored optional GUI operation failure", exc_info=True)
    for dim_map in dim_maps:
        for dim_name, obs_name in dict(dim_map or {}).items():
            if obs_name and _normalize_observable_name(obs_name) == norm:
                return str(dim_name) if str(dim_name) in non_spatial else "frame"

    for dim_name in non_spatial:
        dim_norm = _normalize_observable_name(dim_name)
        if norm in {dim_norm, f"{dim_norm}s"}:
            return str(dim_name)

    aliases = {
        "te": "te",
        "echo_time": "te",
        "echotime": "te",
        "ti": "ti",
        "inversion_time": "ti",
        "inversiontime": "ti",
        "tr": "tr",
        "repetition_time": "tr",
        "repetitiontime": "tr",
        "flip": "flip",
        "flip_angle": "flip",
        "flipangle": "flip",
        "sat_b1": "b1",
        "saturation_b1": "b1",
        "freq_offsets": "offset",
        "freq_offset": "offset",
        "offsets": "offset",
        "offset": "offset",
        "fair_label": "fair",
        "fairlabel": "fair",
        "time": "t",
        "times": "t",
        "water_offset_time": "t",
        "bval": "frame",
        "bvals": "frame",
        "bvec": "frame",
        "bvecs": "frame",
    }
    target = aliases.get(norm)
    if target is None:
        return None
    if target == "frame":
        return "frame"
    if target in non_spatial:
        return target
    if "frame" in non_spatial:
        return "frame"
    return target if target in dims_map else None


def _normalize_observable_name(name: str) -> str:
    return str(name or "").strip().lower().replace("-", "_").replace(" ", "_")


def _observable_name_is_spatial_like(name: str) -> bool:
    norm = _normalize_observable_name(name)
    return (
        norm.endswith("_map") or norm.endswith("_image") or norm in {"b0_map", "b1_map", "r1_map"}
    )


def _resize_observable_leading_axes(
    arr: np.ndarray,
    target_shape: Sequence[int],
    policy: str,
) -> tuple[np.ndarray, list[str]]:
    out = np.asarray(arr)
    target = tuple(int(v) for v in target_shape)
    if out.ndim < len(target):
        raise ValueError(
            f"observable has {out.ndim} dimensions, cannot align to target axes {target}"
        )
    allow_trim = policy in {"trim", "trim_extend"}
    allow_extend = policy in {"extend", "trim_extend"}
    actions: list[str] = []
    for axis, target_size in enumerate(target):
        current_size = int(out.shape[axis])
        if current_size == target_size:
            continue
        if current_size > target_size:
            if not allow_trim:
                raise ValueError(
                    f"axis {axis} is longer ({current_size}>{target_size}) and policy does not trim"
                )
            slicer = [slice(None)] * out.ndim
            slicer[axis] = slice(0, target_size)
            out = out[tuple(slicer)]
            actions.append(f"trimmed axis {axis} from {current_size} to {target_size}")
            continue
        if not allow_extend:
            raise ValueError(
                f"axis {axis} is shorter ({current_size}<{target_size}) and policy does not extend"
            )
        if current_size <= 0:
            raise ValueError(f"axis {axis} is empty and cannot be extended")
        pad_count = target_size - current_size
        last_slice = np.take(out, [-1], axis=axis)
        padding = np.repeat(last_slice, pad_count, axis=axis)
        out = np.concatenate([out, padding], axis=axis)
        actions.append(
            f"extended axis {axis} from {current_size} to {target_size} using last value"
        )
    return out, actions


def _load_experiment_metadata_template_from_file(
    path: Path,
    *,
    experiment_name: str | None = None,
) -> dict[str, Any]:
    metadata_path = Path(path)
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata file does not exist: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if _looks_like_study_manifest_payload(payload):
        version_info = validate_mrax_file_version(payload, file_kind="MRax study manifest")
        if version_info.tag_missing:
            warnings.warn(
                missing_mrax_version_message("MRax study manifest"),
                RuntimeWarning,
                stacklevel=2,
            )
        experiments = dict(payload.get("experiments") or {})
        if experiment_name:
            name = str(experiment_name)
            if name not in experiments:
                choices = ", ".join(sorted(str(key) for key in experiments))
                raise KeyError(f"Experiment '{name}' is not in manifest. Available: {choices}")
        elif len(experiments) == 1:
            name = next(iter(experiments.keys()))
        else:
            choices = ", ".join(sorted(str(key) for key in experiments))
            raise ValueError(f"Manifest contains multiple experiments; enter one of: {choices}")
        entry = dict(experiments[name] or {})
        rel_metadata = entry.get("metadata")
        if not rel_metadata:
            raise ValueError(f"Manifest experiment '{name}' has no metadata file entry")
        child_path = (metadata_path.parent / str(rel_metadata)).resolve()
        with child_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        child_version = validate_mrax_file_version(
            payload,
            file_kind=f"MRax experiment metadata for '{name}'",
        )
        if child_version.tag_missing:
            warnings.warn(
                missing_mrax_version_message(f"MRax experiment metadata for '{name}'"),
                RuntimeWarning,
                stacklevel=2,
            )
        return _experiment_metadata_template_from_payload(
            payload,
            label=f"{metadata_path.name}:{name}",
            fallback_name=name,
        )
    version_info = validate_mrax_file_version(payload, file_kind="MRax experiment metadata")
    if version_info.tag_missing:
        warnings.warn(
            missing_mrax_version_message("MRax experiment metadata"),
            RuntimeWarning,
            stacklevel=2,
        )
    return _experiment_metadata_template_from_payload(
        payload,
        label=str(metadata_path),
        fallback_name=metadata_path.parent.name or metadata_path.stem,
    )


def _looks_like_study_manifest_payload(payload: Any) -> bool:
    if not isinstance(payload, Mapping):
        return False
    experiments = payload.get("experiments")
    if not isinstance(experiments, Mapping):
        return False
    return any(isinstance(entry, Mapping) and "metadata" in entry for entry in experiments.values())


def _experiment_metadata_template_from_payload(
    payload: Mapping[str, Any],
    *,
    label: str | None = None,
    fallback_name: str | None = None,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError("Metadata template must be a JSON object")
    exp_info = payload.get("experiment") if isinstance(payload.get("experiment"), Mapping) else {}
    exp_info = dict(exp_info or {})
    exp_type = (
        exp_info.get("type")
        or exp_info.get("exp_type")
        or payload.get("exp_type")
        or payload.get("type")
        or "anatomical"
    )
    model = exp_info.get("model") or payload.get("model") or "identity"
    dims = _dims_from_metadata_template_payload(exp_info.get("dims") or payload.get("dims") or {})
    observable_values = payload.get("observable_values")
    if observable_values is None:
        observable_fallback = payload.get("observables") or {}
        observable_values = observable_fallback if isinstance(observable_fallback, Mapping) else {}
    observables = _observables_from_metadata_payload(
        observable_values, payload.get("observable_units") or {}
    )
    extra_metadata = payload.get("extra_metadata")
    if extra_metadata is None:
        extra_metadata = payload.get("importer_metadata")
    if extra_metadata is None:
        extra_metadata = {}
    name = (
        exp_info.get("name") or payload.get("name") or payload.get("dataset_name") or fallback_name
    )
    if not dims and not observables and "exp_type" not in payload and not exp_info:
        raise ValueError("Metadata JSON does not contain experiment metadata or observable values")
    return {
        "label": label or str(name or "metadata file"),
        "name": str(name or ""),
        "exp_type": str(exp_type or "anatomical"),
        "model": str(model or "identity"),
        "dims": dims,
        "observables": observables,
        "extra_metadata": _metadata_to_jsonable(dict(extra_metadata or {})),
    }


def _dims_from_metadata_template_payload(payload: Any) -> dict[str, int]:
    if not payload:
        return {}
    if not isinstance(payload, Mapping):
        raise ValueError("Template dimensions must be a JSON object")
    dims: dict[str, int] = {}
    for key, value in payload.items():
        dim = str(key).strip()
        if not dim:
            raise ValueError("Template dimension names cannot be empty")
        size = int(value)
        if size <= 0:
            raise ValueError(f"Template dimension '{dim}' must be positive")
        dims[dim] = size
    return dims


def _observables_from_metadata_payload(
    observable_values: Any,
    observable_units: Mapping[str, Any] | None = None,
) -> dict[str, Quantity]:
    if not observable_values:
        return {}
    if not isinstance(observable_values, Mapping):
        raise ValueError("Observable metadata must be a JSON object keyed by observable name")
    units = {str(key): str(value) for key, value in dict(observable_units or {}).items()}
    observables: dict[str, Quantity] = {}
    for name, row in observable_values.items():
        key = str(name)
        if isinstance(row, Quantity):
            observables[key] = Quantity(jnp.asarray(np.asarray(row.values)), row.si_unit)
            continue
        if isinstance(row, Mapping):
            unit = str(row.get("unit") or row.get("si_unit") or units.get(key, "")).strip()
            values = row.get("values", row.get("value"))
        else:
            unit = str(units.get(key, "")).strip()
            values = row
        if values is None:
            raise ValueError(f"Observable '{key}' is missing values")
        if not unit:
            raise ValueError(f"Observable '{key}' is missing a unit")
        observables[key] = Quantity(jnp.asarray(values), unit)
    return observables


def _detect_load_kind(path: Path) -> str:
    path = Path(path)
    if path.is_dir():
        if (path / "mrax_package.json").is_file():
            return "bundle_folder"
        manifest = _find_study_manifest(path)
        if manifest is not None:
            return "manifest"
        raise ValueError(f"Unsupported MRax folder: {path}")
    suffix = path.suffix.lower()
    if suffix == ".mrax":
        return "bundle"
    if suffix == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            if path.name == "study_manifest.json":
                return "manifest"
            raise
        if payload.get("format") == "mrax.gui.state":
            return "gui_state"
        if payload.get("format") == "mrax.gui.package":
            return "bundle_folder"
        if "rois" in payload and "npz" in payload:
            return "roi"
        if "manifest" in payload or "experiments" in payload or path.name == "study_manifest.json":
            return "manifest"
        raise ValueError(
            f"JSON file is not a recognized MRax manifest, GUI state, or ROI set: {path}"
        )
    if suffix == ".npy":
        return "mask"
    if suffix == ".npz":
        sidecar = path.with_suffix(".json")
        if sidecar.is_file():
            try:
                payload = json.loads(sidecar.read_text(encoding="utf-8"))
                if "rois" in payload and "npz" in payload:
                    return "roi"
            except Exception:
                _LOG.debug("Ignored optional GUI operation failure", exc_info=True)
        return "roi"
    raise ValueError(f"Unsupported load file extension: {path.suffix}")
