"""Analysis object export using MRax geometry and units."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from mrax.analysis import ROI
from mrax.data_struct import Quantity, SpatialGeometry
from mrax.data_struct.dimensions import spatial_names
from mrax.io.image_export import export_image_files, image_frames
from mrax.reco.components import available_reconstruction_variants

from ..masks import save_roi_set
from ..session import GuiSession
from .common import _array_summary, _metadata_to_jsonable


def _single_export_file_types(record: Mapping[str, Any]) -> list[dict[str, str]]:
    arr = np.asarray(record.get("array"))
    kind = str(record.get("kind") or "")
    formats = [
        {"id": "npy", "label": "NumPy array (.npy)", "suffix": ".npy"},
        {"id": "npz", "label": "Compressed NumPy archive (.npz)", "suffix": ".npz"},
    ]
    if kind == "roi" and record.get("roi") is not None:
        formats.insert(
            0, {"id": "roi_set", "label": "MRax ROI set (.npz + .json)", "suffix": ".npz"}
        )
    if arr.ndim in (1, 2) and not np.iscomplexobj(arr):
        formats.append({"id": "csv", "label": "CSV table (.csv)", "suffix": ".csv"})
    if kind != "k_space" and not np.iscomplexobj(arr):
        try:
            next(image_frames(arr, record.get("dims")))
        except (ValueError, StopIteration):
            pass
        else:
            formats.append({"id": "nifti", "label": "NIfTI frames (.nii)", "suffix": ".nii"})
            formats.append({"id": "dicom", "label": "DICOM image series (.dcm)", "suffix": ".dcm"})
    formats.append({"id": "json", "label": "Metadata JSON (.json)", "suffix": ".json"})
    return formats


def _quantity_array_and_unit(value: Any) -> tuple[np.ndarray, str | None]:
    if isinstance(value, Quantity):
        return np.asarray(value.values), getattr(value, "si_unit", None)
    return np.asarray(value), None


def _export_object_records_for_session(session: GuiSession) -> list[dict[str, Any]]:
    if not session.is_loaded:
        return []
    records: list[dict[str, Any]] = []
    experiments = (session.data or {}).get("experiments") or {}
    for exp_name, exp_data in experiments.items():
        group = f"Experiment: {exp_name}"
        dims = dict(
            exp_data.get("dims") or session.summary_by_name.get(str(exp_name), {}).get("dims") or {}
        )
        extra = exp_data.get("extra_metadata") or {}
        source_grid = (
            exp_data.get("pipeline_source_grid") or extra.get("pipeline_source_grid") or {}
        )
        source_dims = source_grid.get("dims") or dims
        fit_dims = (exp_data.get("fit_grid") or extra.get("fit_grid") or {}).get("dims") or dims
        if exp_data.get("mr_image") is not None:
            current_image = np.asarray(exp_data["mr_image"])
            records.append(
                _export_record(
                    "image",
                    f"{exp_name} / image",
                    current_image,
                    group=group,
                    experiment=exp_name,
                    name="mr_image",
                    dims=dims,
                )
            )
            original_image = exp_data.get("original_mr_image")
            if original_image is not None:
                original_array = np.asarray(original_image)
                if tuple(original_array.shape) != tuple(current_image.shape) or not np.array_equal(
                    original_array, current_image
                ):
                    records.append(
                        _export_record(
                            "original_image",
                            f"{exp_name} / original image",
                            original_array,
                            group=group,
                            experiment=exp_name,
                            name="original_mr_image",
                            dims=source_dims,
                        )
                    )
        processed = dict(exp_data.get("processed") or {})
        for variant_name, component, coil_index, array in available_reconstruction_variants(
            processed,
            spatial_ndim=len(spatial_names(source_dims)),
        ):
            channel_label = "combined" if coil_index is None else f"coil {coil_index + 1}"
            record = _export_record(
                "reconstruction",
                f"{exp_name} / MRax reconstruction / {channel_label} / {component}",
                array,
                group=group,
                experiment=exp_name,
                name=variant_name,
                dims=source_dims,
                unit="rad" if component == "phase" else "",
                metadata={
                    "component": component,
                    "coil": None if coil_index is None else int(coil_index) + 1,
                    "phase_unwrapped": bool(
                        component == "phase" and not variant_name.endswith("_wrapped")
                    ),
                    "coil_combination": "vendor magnitude / coherent phase"
                    if coil_index is None
                    else "individual",
                },
            )
            record["nifti_reference"] = {
                "experiment": str(exp_name),
                "affine": _experiment_nifti_affine(session, str(exp_name)),
            }
            records.append(record)
        if exp_data.get("raw_kspace") is not None:
            records.append(
                _export_record(
                    "k_space",
                    f"{exp_name} / k-space",
                    np.asarray(exp_data["raw_kspace"]),
                    group=group,
                    experiment=exp_name,
                    name="raw_kspace",
                    dims=source_dims,
                )
            )
        for map_name, value in (exp_data.get("fitted_maps") or {}).items():
            arr, unit = _quantity_array_and_unit(value)
            records.append(
                _export_record(
                    "map",
                    f"{exp_name} / map / {map_name}",
                    arr,
                    group=group,
                    experiment=exp_name,
                    name=str(map_name),
                    dims=fit_dims,
                    unit=unit or str((exp_data.get("fitted_map_units") or {}).get(map_name) or ""),
                )
            )

    roi_by_name: dict[str, ROI] = {}
    for roi in [*list(session.shared_rois), *list(session.rois)]:
        roi_by_name.setdefault(str(roi.name), roi)
    for roi in roi_by_name.values():
        record = _export_record(
            "roi",
            f"ROI / {roi.name}",
            np.asarray(roi.mask),
            group="ROIs",
            name=roi.name,
            dims=tuple(roi.dims or ()),
            metadata=dict(roi.metadata),
            roi=roi,
        )
        record["nifti_reference"] = _roi_nifti_reference(session, roi.metadata)
        records.append(record)
    for record in records:
        if record.get("kind") != "roi":
            experiment = experiments.get(record.get("experiment"), {})
            extra = experiment.get("extra_metadata") or {}
            reference = record.setdefault("nifti_reference", {})
            kind = record.get("kind")
            grid = (
                experiment.get("fit_grid") or extra.get("fit_grid")
                if kind == "map"
                else experiment.get("pipeline_source_grid") or extra.get("pipeline_source_grid")
                if kind in {"original_image", "reconstruction", "k_space"}
                else None
            )
            if grid:
                geometry = grid.get("geometry")
                reference["geometry"] = geometry
                reference["affine"] = (
                    None
                    if geometry is None
                    else SpatialGeometry.from_metadata(geometry).affine_for_frame(0)
                )
            else:
                reference.setdefault(
                    "affine", _experiment_nifti_affine(session, record.get("experiment"))
                )
                reference["geometry"] = experiment.get("spatial_geometry")
    return records


def _export_record(
    kind: str,
    label: str,
    array: Any,
    *,
    group: str,
    name: str,
    experiment: str | None = None,
    dims: Mapping[str, Any] | Sequence[str] | None = None,
    unit: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    roi: ROI | None = None,
) -> dict[str, Any]:
    arr = np.asarray(array)
    record = {
        "id": "/".join(part for part in (experiment, kind, name) if part),
        "group": str(group),
        "kind": str(kind),
        "label": str(label),
        "name": str(name),
        "experiment": None if experiment is None else str(experiment),
        "array": arr,
        "shape": tuple(int(v) for v in arr.shape),
        "dtype": str(arr.dtype),
        "dims": dict(dims) if isinstance(dims, Mapping) else list(dims or ()),
        "unit": unit or "",
        "metadata": dict(metadata or {}),
    }
    if roi is not None:
        record["roi"] = roi
    return record


def _single_export_record_metadata(record: Mapping[str, Any]) -> dict[str, Any]:
    arr = np.asarray(record.get("array"))
    metadata = {
        "id": record.get("id"),
        "label": record.get("label"),
        "kind": record.get("kind"),
        "name": record.get("name"),
        "experiment": record.get("experiment"),
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "dims": record.get("dims"),
        "unit": record.get("unit") or "",
        "summary": _array_summary(arr, unit=str(record.get("unit") or "") or None),
    }
    extra = record.get("metadata")
    if extra:
        metadata["metadata"] = _metadata_to_jsonable(extra)
    if record.get("kind") == "roi" and record.get("roi") is not None:
        roi = record["roi"]
        metadata["roi"] = {
            "name": roi.name,
            "dims": list(roi.dims or ()),
            "metadata": _metadata_to_jsonable(dict(roi.metadata)),
        }
    reference = record.get("nifti_reference")
    if reference:
        metadata["nifti_reference"] = {
            key: value for key, value in dict(reference).items() if key != "affine"
        }
    return _metadata_to_jsonable(metadata)


def _write_single_export_object(
    record: Mapping[str, Any], path: Path, file_type: str
) -> list[Path]:
    fmt = str(file_type)
    path = Path(path).expanduser()
    arr = np.asarray(record.get("array"))
    metadata = _single_export_record_metadata(record)
    if fmt == "roi_set":
        roi = record.get("roi")
        if roi is None:
            raise ValueError("Selected object is not an ROI")
        if path.suffix.lower() not in {".npz", ".json"}:
            path = path.with_suffix(".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        roi_npz, roi_json = save_roi_set([roi], path, metadata=metadata)
        return [roi_npz, roi_json]
    suffix_by_format = {
        "npy": ".npy",
        "npz": ".npz",
        "csv": ".csv",
        "nifti": ".nii",
        "dicom": ".dcm",
        "json": ".json",
    }
    suffix = suffix_by_format.get(fmt)
    if suffix is None:
        raise ValueError(f"Unsupported export file type '{file_type}'")
    if path.suffix.lower() != suffix:
        path = path.with_suffix(suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "npy":
        np.save(path, arr)
    elif fmt == "npz":
        np.savez_compressed(path, data=arr, metadata=json.dumps(metadata))
    elif fmt == "csv":
        if arr.ndim == 1:
            value = arr.reshape((-1, 1))
        elif arr.ndim == 2:
            value = arr
        else:
            raise ValueError("CSV export supports 1D and 2D arrays only")
        if np.iscomplexobj(value):
            raise ValueError("CSV export does not support complex arrays")
        np.savetxt(path, value, delimiter=",")
    elif fmt in {"nifti", "dicom"}:
        reference = dict(record.get("nifti_reference") or {})
        return export_image_files(
            path,
            arr,
            dims=record.get("dims"),
            affine=_validated_nifti_affine(reference.get("affine")),
            geometry=reference.get("geometry"),
            format=fmt,
            unit=str(record.get("unit") or ""),
            description=str(record.get("name") or "MRax image"),
        )
    elif fmt == "json":
        path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return [path]


def _validated_nifti_affine(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        affine = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if affine.shape != (4, 4) or not np.all(np.isfinite(affine)):
        return None
    return affine


def _experiment_nifti_affine(
    session: GuiSession,
    experiment_name: str | None,
    *,
    frame_index: int = 0,
) -> np.ndarray | None:
    if not experiment_name or not session.data:
        return None
    exp_data = (session.data.get("experiments") or {}).get(str(experiment_name))
    if not isinstance(exp_data, Mapping):
        return None

    geometry_value = exp_data.get("spatial_geometry")
    if not isinstance(geometry_value, Mapping):
        extra_value = exp_data.get("extra_metadata")
        if isinstance(extra_value, Mapping):
            geometry_value = extra_value.get("spatial_geometry")
    if isinstance(geometry_value, Mapping):
        geometry_affine = SpatialGeometry.from_metadata(geometry_value).affine_for_frame(
            frame_index
        )
        if geometry_affine is not None:
            return geometry_affine
        return None

    metadata_sources = [
        exp_data,
        exp_data.get("extra_metadata") or {},
        exp_data.get("processed") or {},
        exp_data.get("kspace_metadata") or {},
    ]
    for metadata in metadata_sources:
        if not isinstance(metadata, Mapping):
            continue
        affines = metadata.get("affines")
        if affines is not None:
            try:
                candidates = list(affines)
            except TypeError:
                candidates = []
            if candidates:
                index = max(0, min(int(frame_index), len(candidates) - 1))
                affine = _validated_nifti_affine(candidates[index])
                if affine is not None:
                    return affine
        affine = _validated_nifti_affine(metadata.get("affine"))
        if affine is not None:
            return affine

    extra = exp_data.get("extra_metadata") or {}
    visu = extra.get("visu") if isinstance(extra, Mapping) else None
    if isinstance(visu, Mapping):
        try:
            from mrax.io.bruker_images import bruker_affine_from_visu

            affine_value = bruker_affine_from_visu(dict(visu))
        except Exception:
            return None
        if isinstance(affine_value, list):
            if not affine_value:
                return None
            index = max(0, min(int(frame_index), len(affine_value) - 1))
            affine = affine_value[index]
        else:
            affine = affine_value
        return _validated_nifti_affine(affine)
    return None


def _roi_nifti_reference(session: GuiSession, metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    roi_metadata = dict(metadata or {})
    declared_experiment = str(roi_metadata.get("source_experiment") or "").strip()
    reference_experiment = declared_experiment or str(session.selected_experiment or "").strip()
    indices = roi_metadata.get("slice_indices") or {}
    try:
        frame_index = int(indices.get("frame", 0)) if isinstance(indices, Mapping) else 0
    except (TypeError, ValueError):
        frame_index = 0
    affine = _experiment_nifti_affine(session, reference_experiment, frame_index=frame_index)
    return {
        "experiment": reference_experiment or None,
        "selection": "mask metadata" if declared_experiment else "active experiment",
        "frame_index": max(0, frame_index),
        "affine": np.eye(4, dtype=float) if affine is None else affine,
        "identity_fallback": affine is None,
    }
