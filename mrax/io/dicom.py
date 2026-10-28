"""Vendor-neutral DICOM image import for MRax.

The general layer in this module only interprets standard DICOM data elements and
functional groups. Vendor-specific conventions are supplied through
``DicomMetadataAdapter`` implementations, such as the Bruker adapter in
``mrax.io.dicom_bruker``.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, cast, runtime_checkable

import jax.numpy as jnp
import numpy as np
from pydicom import dcmread
from pydicom.dataset import Dataset as PydicomDataset

from mrax.data_struct import DataSet, Experiment, Quantity, SpatialGeometry, Study
from mrax.library.default_experiments import default_model_for_type

from .import_definitions import (
    ImportDefinition,
    NormalizedMetadata,
    build_imported_experiment_dataset,
    detect_experiment_type,
    extract_observables_from_metadata,
    needs_manual_type_override,
)

_LPS_TO_RAS = np.diag([-1.0, -1.0, 1.0])
_NORMALIZED_FIELDS = {item.name for item in fields(NormalizedMetadata)}
_MAX_EXPERIMENT_NAME_LENGTH = 40


@dataclass(frozen=True)
class DicomSeriesInfo:
    """Header-only description of one discoverable DICOM image series."""

    series_instance_uid: str
    study_instance_uid: str
    modality: str
    manufacturer: str
    description: str
    file_count: int
    frame_count: int


@dataclass(frozen=True)
class DicomMetadataPatch:
    """Vendor adapter additions to normalized and diagnostic metadata."""

    normalized: Mapping[str, object] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)


@runtime_checkable
class DicomMetadataAdapter(Protocol):
    """Extension point for origin-specific DICOM metadata conventions."""

    name: str

    def matches(self, datasets: Sequence[PydicomDataset]) -> bool:
        """Return whether this adapter recognizes the DICOM origin."""

        ...

    def extract(
        self,
        datasets: Sequence[PydicomDataset],
        frame_metadata: Sequence[Mapping[str, object]],
    ) -> DicomMetadataPatch:
        """Return metadata additions for the shared pixel-assembly pipeline."""

        ...


@dataclass
class _FrameRecord:
    pixels: np.ndarray
    order: tuple[int, int]
    metadata: dict[str, object]


def discover_dicom_series(
    source: str | Path | Sequence[str | Path],
    *,
    recursive: bool = True,
    force: bool = False,
) -> list[DicomSeriesInfo]:
    """Inspect DICOM headers and return image-series summaries.

    Summaries contain only technical series metadata.
    Readable DICOM images enter discovery.
    """

    grouped = _read_grouped_headers(source, recursive=recursive, force=force)
    output: list[DicomSeriesInfo] = []
    for uid, entries in grouped.items():
        headers = [dataset for _path, dataset in entries]
        first = headers[0]
        output.append(
            DicomSeriesInfo(
                series_instance_uid=uid if not uid.startswith("__missing__") else "",
                study_instance_uid=str(getattr(first, "StudyInstanceUID", "") or ""),
                modality=str(getattr(first, "Modality", "") or ""),
                manufacturer=str(getattr(first, "Manufacturer", "") or ""),
                description=_series_description(first),
                file_count=len(entries),
                frame_count=sum(_declared_frame_count(dataset) for dataset in headers),
            )
        )
    return output


def load_dicom_series(
    source: str | Path | Sequence[str | Path],
    *,
    series_instance_uid: str | None = None,
    name: str | None = None,
    override_type: str | None = None,
    metadata_adapters: Sequence[DicomMetadataAdapter] | None = None,
    recursive: bool = True,
    force: bool = False,
    include_identifiers: bool = False,
    metadata_overrides: Mapping[str, object] | None = None,
) -> tuple[Experiment, DataSet, dict[str, object]]:
    """Load one classic or enhanced DICOM image series into MRax containers.

    Standard rescale transforms, functional-group geometry, slice ordering, temporal
    dimensions, and common MR observables are handled by the general layer. When the
    input contains more than one series, ``series_instance_uid`` is required.
    """

    grouped = _read_grouped_headers(source, recursive=recursive, force=force)
    if not grouped:
        raise FileNotFoundError("No readable DICOM image series found")
    selected_uid = _select_series_uid(grouped, series_instance_uid)
    entries = grouped[selected_uid]
    datasets = [dcmread(path, force=force) for path, _header in entries]
    return _materialize_dicom_series(
        datasets,
        name=name,
        override_type=override_type,
        metadata_adapters=metadata_adapters,
        include_identifiers=include_identifiers,
        metadata_overrides=metadata_overrides,
    )


def load_dicom_study(
    source: str | Path | Sequence[str | Path],
    *,
    type_overrides: Mapping[str, str] | None = None,
    metadata_adapters: Sequence[DicomMetadataAdapter] | None = None,
    recursive: bool = True,
    force: bool = False,
    include_identifiers: bool = False,
    series_groups: Mapping[str, Sequence[str]] | None = None,
    metadata_overrides: Mapping[str, Mapping[str, object]] | None = None,
) -> tuple[Study, dict[str, object]]:
    """Load DICOM series, optionally assembling explicitly named contrast sweeps.

    ``series_groups`` maps experiment names to series UIDs. Selected series must
    share study, frame of reference, pixel layout and physical geometry. Unlisted
    series remain separate. Explicit compatible UIDs govern series merging.
    """

    from mrax.io.measurement import image_signal_scale

    grouped = _read_grouped_headers(source, recursive=recursive, force=force)
    if not grouped:
        raise FileNotFoundError("No readable DICOM image series found")
    source_series_count = len(grouped)
    grouped = _combine_series(grouped, series_groups or {})

    experiments: list[Experiment] = []
    datasets_out: list[DataSet] = []
    models = []
    names: list[str] = []
    experiment_metadata: dict[str, object] = {}
    used_names: set[str] = set()
    field_strengths: list[float] = []
    for index, (uid, entries) in enumerate(grouped.items(), start=1):
        datasets = [dcmread(path, force=force) for path, _header in entries]
        base_name = uid if uid in (series_groups or {}) else _compact_series_name(datasets[0], fallback_index=index)
        entry_name = _unique_name(base_name, used_names)
        override = (type_overrides or {}).get(uid) or (type_overrides or {}).get(entry_name)
        experiment, dataset, metadata = _materialize_dicom_series(
            datasets,
            name=entry_name,
            override_type=override,
            metadata_adapters=metadata_adapters,
            include_identifiers=include_identifiers,
            metadata_overrides=(metadata_overrides or {}).get(uid, (metadata_overrides or {}).get(entry_name)),
        )
        experiments.append(experiment)
        datasets_out.append(dataset)
        models.append(default_model_for_type(str(experiment.exp_type or "anatomical"),
            scale=image_signal_scale(dataset.data), observables=experiment.observables, image_data=dataset.data))
        names.append(entry_name)
        experiment_metadata[entry_name] = metadata
        strength = _finite_float(metadata.get("magnetic_field_strength_t"))
        if strength is not None and strength > 0:
            field_strengths.append(strength)

    study_globals: dict[str, Quantity] = {}
    if field_strengths:
        study_globals["field_strength_mhz"] = Quantity(
            jnp.asarray(field_strengths[0] * 42.57747892),
            "MHz",
        )
    study = Study(
        experiments,
        datasets_out,
        models,
        names=names,
        study_globals=study_globals,
    )
    return study, {
        "format": "dicom",
        "series_count": len(names),
        "source_series_count": source_series_count,
        "experiments": experiment_metadata,
        "identifiers_included": bool(include_identifiers),
    }


def _combine_series(grouped, groups):
    selected: set[str] = set()
    combined = {}
    for name, uids in groups.items():
        if isinstance(uids, str) or not uids:
            raise ValueError("DICOM series_groups requires nonempty sequences of series UIDs")
        if not name or name in grouped:
            raise ValueError("DICOM group names must be nonempty and distinct from series UIDs")
        entries = []
        signatures = []
        for uid in uids:
            if uid not in grouped or uid in selected:
                raise ValueError("DICOM grouped series is missing or selected more than once")
            selected.add(uid)
            series = grouped[uid]
            signatures.append(_series_geometry_signature(series))
            entries.extend(series)
        if any(signature != signatures[0] for signature in signatures[1:]):
            raise ValueError("DICOM grouped series must share study, frame of reference, image layout and geometry")
        combined[name] = entries
    combined.update({uid: entries for uid, entries in grouped.items() if uid not in selected})
    return combined


def _series_geometry_signature(entries):
    signatures = set()
    for _path, header in entries:
        for frame in range(_declared_frame_count(header)):
            row = _standard_frame_metadata(header, frame)
            geometry = tuple(tuple(np.round(row.get(key) or (), 5)) for key in (
                "image_orientation_patient", "image_position_patient", "pixel_spacing"))
            if any(not values for values in geometry):
                raise ValueError("Explicit DICOM series grouping requires complete physical geometry")
            signatures.add((str(getattr(header, "StudyInstanceUID", "")),
                str(getattr(header, "FrameOfReferenceUID", "")), int(header.Rows), int(header.Columns),
                str(getattr(header, "ComplexImageComponent", "")), tuple(getattr(header, "ImageType", ())),
                _rounded(row.get("slice_thickness")), *geometry))
    return frozenset(signatures)


def _source_files(source: str | Path | Sequence[str | Path], *, recursive: bool) -> list[Path]:
    values = [source] if isinstance(source, (str, Path)) else list(source)
    files: list[Path] = []
    for value in values:
        path = Path(value).expanduser()
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            iterator = path.rglob("*") if recursive else path.glob("*")
            files.extend(candidate for candidate in iterator if candidate.is_file())
    return sorted(dict.fromkeys(files))


def _read_grouped_headers(
    source: str | Path | Sequence[str | Path],
    *,
    recursive: bool,
    force: bool,
) -> dict[str, list[tuple[Path, PydicomDataset]]]:
    grouped: dict[str, list[tuple[Path, PydicomDataset]]] = defaultdict(list)
    missing_index = 0
    for path in _source_files(source, recursive=recursive):
        try:
            dataset = dcmread(path, stop_before_pixels=True, force=force)
        except Exception:
            continue
        if not _is_dicom_image(dataset):
            continue
        uid = str(getattr(dataset, "SeriesInstanceUID", "") or "")
        if not uid:
            missing_index += 1
            uid = f"__missing__{missing_index}"
        grouped[uid].append((path, dataset))
    for entries in grouped.values():
        entries.sort(key=lambda item: _dataset_order(item[1]))
    return dict(grouped)


def _is_dicom_image(dataset: PydicomDataset) -> bool:
    return bool(
        getattr(dataset, "SOPClassUID", None)
        and getattr(dataset, "Rows", None)
        and getattr(dataset, "Columns", None)
    )


def _select_series_uid(
    grouped: Mapping[str, Sequence[tuple[Path, PydicomDataset]]],
    requested: str | None,
) -> str:
    if requested is not None:
        if requested not in grouped:
            raise KeyError(f"Unknown DICOM series instance UID '{requested}'")
        return requested
    if len(grouped) != 1:
        raise ValueError("Multiple DICOM series found; choose one with series_instance_uid")
    return next(iter(grouped))


def _materialize_dicom_series(
    datasets: Sequence[PydicomDataset],
    *,
    name: str | None,
    override_type: str | None,
    metadata_adapters: Sequence[DicomMetadataAdapter] | None,
    include_identifiers: bool,
    metadata_overrides: Mapping[str, object] | None = None,
) -> tuple[Experiment, DataSet, dict[str, object]]:
    if not datasets:
        raise ValueError("A DICOM series must contain at least one dataset")
    input_dataset_count = len(datasets)
    adapters = list(metadata_adapters) if metadata_adapters is not None else _builtin_metadata_adapters()
    matched_adapter = next((adapter for adapter in adapters if adapter.matches(datasets)), None)
    selector = getattr(matched_adapter, "select_datasets", None)
    if callable(selector):
        datasets = tuple(selector(datasets))
        if not datasets:
            raise ValueError("The DICOM metadata adapter rejected every image object in the series")
    records: list[_FrameRecord] = []
    for dataset_index, dataset in enumerate(datasets):
        records.extend(_decode_dataset_frames(dataset, dataset_index))
    if not records:
        raise ValueError("DICOM series contains no decodable image frames")

    volumes = _group_frame_records(records)
    data, spatial_shape, volume_metadata, affines = _assemble_volumes(volumes)
    standard_values = _normalized_standard_metadata(datasets[0], volume_metadata)
    adapter_patch = (
        matched_adapter.extract(datasets, volume_metadata)
        if matched_adapter is not None
        else DicomMetadataPatch()
    )
    normalized_values = dict(standard_values)
    normalized_values.update(
        {
            key: value
            for key, value in adapter_patch.normalized.items()
            if key in _NORMALIZED_FIELDS and value is not None
        }
    )
    overrides = dict(metadata_overrides or {})
    if set(overrides) - _NORMALIZED_FIELDS:
        raise ValueError("Unknown normalized DICOM metadata fields")
    if 'frame_count' in overrides and int(overrides['frame_count']) != len(volumes):
        raise ValueError("DICOM metadata frame_count must match the decoded volume count")
    normalized_values.update(overrides)
    normalized = NormalizedMetadata(**cast(Any, normalized_values))
    exp_type = str(override_type or detect_experiment_type(normalized))
    observables = extract_observables_from_metadata(exp_type, normalized, len(volumes))
    from mrax.data_struct.dimensions import get_default_dim_specs

    specs = get_default_dim_specs(exp_type) if exp_type in {'ir', 'mse', 'mge', 'vfa', 'vtr'} else ()
    columns = [np.asarray(observables[spec.observable].values).reshape(-1)
        for spec in specs if spec.observable in observables
        and np.asarray(observables[spec.observable].values).size == len(volumes)]
    order = np.lexsort(tuple(reversed(columns))) if columns else np.arange(len(volumes))
    data = data[order]
    volume_metadata = [volume_metadata[i] for i in order]
    affines = [affines[i] for i in order]
    observables = {key: value.with_values(jnp.asarray(np.asarray(value.values)[order]))
        if np.asarray(value.values).ndim and np.asarray(value.values).shape[0] == len(volumes) else value
        for key, value in observables.items()}
    dataset_name = str(name or _compact_series_name(datasets[0]))
    first_geometry_row = volume_metadata[0] if volume_metadata else {}
    pixel_spacing = tuple(first_geometry_row.get("pixel_spacing") or ())
    slice_spacing = _slice_spacing(first_geometry_row)
    first_affine = next((affine for affine in affines if affine is not None), None)
    if first_affine is not None:
        position_spacing = float(np.linalg.norm(np.asarray(first_affine)[:3, 2]))
        if np.isfinite(position_spacing) and position_spacing > 0:
            slice_spacing = position_spacing
    voxel_sizes = (
        (*tuple(float(value) for value in pixel_spacing[:2]), float(slice_spacing))
        if len(spatial_shape) == 3 and len(pixel_spacing) >= 2
        else tuple(float(value) for value in pixel_spacing[: len(spatial_shape)])
    )
    if len(spatial_shape) == 3:
        # Native rows/columns/slices reverse to slices/columns/rows at NIfTI I/O.
        affines = [None if affine is None else np.asarray(affine)[:, [2, 0, 1, 3]] for affine in affines]
    spatial_geometry = SpatialGeometry.from_affines(
        affines,
        spatial_shape=spatial_shape,
        voxel_sizes=voxel_sizes or None,
        axis_anatomical_labels=_dicom_axis_anatomical_labels(
            first_geometry_row,
            spatial_ndim=len(spatial_shape),
        ),
        source="dicom",
    )
    experiment, dataset = build_imported_experiment_dataset(
        ImportDefinition(
            name=dataset_name,
            data=data,
            frame_count=len(volumes),
            spatial_shape=spatial_shape,
            exp_type=exp_type,
            observables=observables,
            spatial_geometry=spatial_geometry,
        )
    )

    first = datasets[0]
    metadata: dict[str, object] = {
        "format": "dicom",
        "adapter": matched_adapter.name if matched_adapter is not None else "standard",
        "modality": str(getattr(first, "Modality", "") or ""),
        "manufacturer": str(getattr(first, "Manufacturer", "") or ""),
        "manufacturer_model_name": str(getattr(first, "ManufacturerModelName", "") or ""),
        "series_description": _series_description(first),
        "protocol_name": str(getattr(first, "ProtocolName", "") or ""),
        "sequence_name": _sequence_name(first),
        "sop_class_uid": str(getattr(first, "SOPClassUID", "") or ""),
        "source_file_count": len(datasets),
        "auxiliary_source_file_count": input_dataset_count - len(datasets),
        "source_frame_count": len(records),
        "frame_count": len(volumes),
        "spatial_shape": spatial_shape,
        "affine": affines[0] if _all_affines_equal(affines) else None,
        "affines": None if _all_affines_equal(affines) else affines,
        "spatial_geometry": spatial_geometry.to_metadata(),
        "magnetic_field_strength_t": _finite_float(getattr(first, "MagneticFieldStrength", None)),
        "manual_type_override_required": bool(
            override_type is None and needs_manual_type_override(normalized, exp_type)
        ),
        "identifiers_included": bool(include_identifiers),
        "vendor": dict(adapter_patch.metadata),
    }
    if include_identifiers:
        metadata["study_instance_uid"] = str(getattr(first, "StudyInstanceUID", "") or "")
        metadata["series_instance_uid"] = str(getattr(first, "SeriesInstanceUID", "") or "")
        metadata["frame_of_reference_uid"] = str(getattr(first, "FrameOfReferenceUID", "") or "")
    experiment._mrax_dicom_metadata = metadata  # type: ignore[attr-defined]
    return experiment, dataset, metadata


def _decode_dataset_frames(dataset: PydicomDataset, dataset_index: int) -> list[_FrameRecord]:
    samples_per_pixel = int(getattr(dataset, "SamplesPerPixel", 1) or 1)
    if samples_per_pixel != 1:
        raise ValueError("MRax DICOM import currently supports scalar grayscale image data only")
    pixels = np.asarray(dataset.pixel_array)
    if pixels.ndim == 2:
        frame_arrays = pixels[None, ...]
    elif pixels.ndim == 3:
        frame_arrays = pixels
    else:
        raise ValueError(f"Unsupported DICOM pixel array shape {pixels.shape}")
    records: list[_FrameRecord] = []
    for frame_index, frame in enumerate(frame_arrays):
        metadata = _standard_frame_metadata(dataset, frame_index)
        slope = _finite_float(metadata.get("rescale_slope"))
        intercept = _finite_float(metadata.get("rescale_intercept"))
        scaled = np.asarray(frame)
        if slope is not None or intercept is not None:
            scaled = scaled.astype(np.float64) * (1.0 if slope is None else slope)
            scaled = scaled + (0.0 if intercept is None else intercept)
        records.append(
            _FrameRecord(
                pixels=scaled,
                order=(_dataset_order(dataset)[0], frame_index + dataset_index * len(frame_arrays)),
                metadata=metadata,
            )
        )
    return records


def _standard_frame_metadata(dataset: PydicomDataset, frame_index: int) -> dict[str, object]:
    dimension_keywords = _dimension_index_keywords(dataset)
    dimension_values = _value_from_functional_groups(
        dataset,
        frame_index,
        "FrameContentSequence",
        "DimensionIndexValues",
    )
    dimensions: dict[str, int] = {}
    if dimension_values is not None:
        raw_values = (
            list(cast(Sequence[object], dimension_values))
            if _is_sequence_value(dimension_values)
            else [dimension_values]
        )
        for keyword, value in zip(dimension_keywords, raw_values):
            try:
                dimensions[keyword] = int(cast(Any, value))
            except (TypeError, ValueError):
                continue

    metadata: dict[str, object] = {
        "dimensions": dimensions,
        "image_orientation_patient": _numeric_tuple(
            _functional_or_top(dataset, frame_index, "PlaneOrientationSequence", "ImageOrientationPatient")
        ),
        "image_position_patient": _numeric_tuple(
            _functional_or_top(dataset, frame_index, "PlanePositionSequence", "ImagePositionPatient")
        ),
        "pixel_spacing": _numeric_tuple(
            _functional_or_top(dataset, frame_index, "PixelMeasuresSequence", "PixelSpacing")
        ),
        "slice_thickness": _finite_float(
            _functional_or_top(dataset, frame_index, "PixelMeasuresSequence", "SliceThickness")
        ),
        "spacing_between_slices": _finite_float(
            _functional_or_top(dataset, frame_index, "PixelMeasuresSequence", "SpacingBetweenSlices")
        ),
        "in_stack_position": _integer_value(
            _functional_or_top(dataset, frame_index, "FrameContentSequence", "InStackPositionNumber")
        ),
        "temporal_position": _integer_value(
            _functional_or_top(dataset, frame_index, "FrameContentSequence", "TemporalPositionIndex")
            or getattr(dataset, "TemporalPositionIdentifier", None)
        ),
        "frame_acquisition_number": _integer_value(
            _functional_or_top(dataset, frame_index, "FrameContentSequence", "FrameAcquisitionNumber")
        ),
        "echo_number": _integer_value(getattr(dataset, "EchoNumbers", None)),
        "echo_time": _finite_float(
            _functional_or_top(dataset, frame_index, "MREchoSequence", "EffectiveEchoTime")
            or getattr(dataset, "EchoTime", None)
        ),
        "repetition_time": _finite_float(
            _functional_or_top(
                dataset,
                frame_index,
                "MRTimingAndRelatedParametersSequence",
                "RepetitionTime",
            )
            or getattr(dataset, "RepetitionTime", None)
        ),
        "inversion_time": _finite_float(
            _functional_or_top(dataset, frame_index, "MRModifierSequence", "InversionTimes")
            or getattr(dataset, "InversionTime", None)
        ),
        "flip_angle": _finite_float(
            _functional_or_top(
                dataset,
                frame_index,
                "MRTimingAndRelatedParametersSequence",
                "FlipAngle",
            )
            or getattr(dataset, "FlipAngle", None)
        ),
        "diffusion_b_value": _finite_float(
            _functional_or_top(dataset, frame_index, "MRDiffusionSequence", "DiffusionBValue")
        ),
        "diffusion_gradient_patient": _numeric_tuple(
            _nested_functional_value(
                dataset,
                frame_index,
                "MRDiffusionSequence",
                "DiffusionGradientDirectionSequence",
                "DiffusionGradientOrientation",
            )
        ),
        "frame_comments": str(
            _functional_or_top(dataset, frame_index, "FrameContentSequence", "FrameComments") or ""
        ),
        "frame_label": str(
            _functional_or_top(dataset, frame_index, "FrameContentSequence", "FrameLabel") or ""
        ),
        "rescale_slope": _finite_float(
            _functional_or_top(dataset, frame_index, "PixelValueTransformationSequence", "RescaleSlope")
            or getattr(dataset, "RescaleSlope", None)
        ),
        "rescale_intercept": _finite_float(
            _functional_or_top(dataset, frame_index, "PixelValueTransformationSequence", "RescaleIntercept")
            or getattr(dataset, "RescaleIntercept", None)
        ),
    }
    metadata["slice_coordinate"] = _slice_coordinate(metadata)
    metadata["diffusion_gradient_image"] = _diffusion_gradient_image(metadata)
    return metadata


def _functional_or_top(
    dataset: PydicomDataset,
    frame_index: int,
    sequence_keyword: str,
    value_keyword: str,
) -> object | None:
    value = _value_from_functional_groups(dataset, frame_index, sequence_keyword, value_keyword)
    return value if value is not None else getattr(dataset, value_keyword, None)


def _value_from_functional_groups(
    dataset: PydicomDataset,
    frame_index: int,
    sequence_keyword: str,
    value_keyword: str,
) -> object | None:
    per_frame = getattr(dataset, "PerFrameFunctionalGroupsSequence", ()) or ()
    if frame_index < len(per_frame):
        value = _sequence_item_value(per_frame[frame_index], sequence_keyword, value_keyword)
        if value is not None:
            return value
    shared = getattr(dataset, "SharedFunctionalGroupsSequence", ()) or ()
    if shared:
        return _sequence_item_value(shared[0], sequence_keyword, value_keyword)
    return None


def _nested_functional_value(
    dataset: PydicomDataset,
    frame_index: int,
    sequence_keyword: str,
    nested_sequence_keyword: str,
    value_keyword: str,
) -> object | None:
    for groups in (
        getattr(dataset, "PerFrameFunctionalGroupsSequence", ()) or (),
        getattr(dataset, "SharedFunctionalGroupsSequence", ()) or (),
    ):
        if not groups:
            continue
        group = groups[min(frame_index, len(groups) - 1)]
        sequence = getattr(group, sequence_keyword, ()) or ()
        if not sequence:
            continue
        nested = getattr(sequence[0], nested_sequence_keyword, ()) or ()
        if nested and hasattr(nested[0], value_keyword):
            return getattr(nested[0], value_keyword)
    return None


def _sequence_item_value(dataset: PydicomDataset, sequence_keyword: str, value_keyword: str) -> object | None:
    sequence = getattr(dataset, sequence_keyword, ()) or ()
    if sequence and hasattr(sequence[0], value_keyword):
        return getattr(sequence[0], value_keyword)
    return None


def _dimension_index_keywords(dataset: PydicomDataset) -> list[str]:
    from pydicom.datadict import keyword_for_tag

    output: list[str] = []
    for item in getattr(dataset, "DimensionIndexSequence", ()) or ():
        pointer = getattr(item, "DimensionIndexPointer", None)
        output.append(keyword_for_tag(pointer) if pointer is not None else "")
    return output


def _group_frame_records(records: Sequence[_FrameRecord]) -> list[list[_FrameRecord]]:
    ordered = sorted(records, key=lambda record: record.order)
    slice_keys = [_slice_key(record.metadata) for record in ordered]
    if any(key is None for key in slice_keys):
        return [[record] for record in ordered]
    unique_slices = sorted({cast(float, key) for key in slice_keys})
    if len(unique_slices) <= 1:
        return [[record] for record in ordered]

    by_signature: dict[tuple[object, ...], list[_FrameRecord]] = defaultdict(list)
    for record in ordered:
        by_signature[_frame_signature(record.metadata)].append(record)
    volumes: list[list[_FrameRecord]] = []
    for signature_records in by_signature.values():
        by_slice: dict[float, list[_FrameRecord]] = defaultdict(list)
        for record in signature_records:
            key = _slice_key(record.metadata)
            if key is not None:
                by_slice[key].append(record)
        if set(by_slice) != set(unique_slices):
            return [[record] for record in ordered]
        occurrence_counts = {len(items) for items in by_slice.values()}
        if len(occurrence_counts) != 1:
            return [[record] for record in ordered]
        for occurrence in range(next(iter(occurrence_counts))):
            volume = [sorted(by_slice[key], key=lambda record: record.order)[occurrence] for key in unique_slices]
            volumes.append(volume)
    return sorted(volumes, key=lambda volume: min(record.order for record in volume))


def _frame_signature(metadata: Mapping[str, object]) -> tuple[object, ...]:
    raw_dimensions = metadata.get("dimensions")
    dimensions = dict(raw_dimensions) if isinstance(raw_dimensions, Mapping) else {}
    non_spatial_dimensions = tuple(
        sorted(
            (key, int(value))
            for key, value in dimensions.items()
            if key not in {"InStackPositionNumber", "ImagePositionPatient", "StackID"}
        )
    )
    if non_spatial_dimensions:
        return ("dimensions", *non_spatial_dimensions)
    return (
        "values",
        metadata.get("temporal_position"),
        metadata.get("echo_number"),
        _rounded(metadata.get("echo_time")),
        _rounded(metadata.get("inversion_time")),
        _rounded(metadata.get("flip_angle")),
        _rounded(metadata.get("diffusion_b_value")),
    )


def _assemble_volumes(
    volumes: Sequence[Sequence[_FrameRecord]],
) -> tuple[np.ndarray, tuple[int, ...], list[dict[str, object]], list[np.ndarray | None]]:
    arrays: list[np.ndarray] = []
    metadata: list[dict[str, object]] = []
    affines: list[np.ndarray | None] = []
    for volume in volumes:
        if not volume:
            continue
        # DICOM rows/columns already match MRax's in-memory image-plane ordering.
        # NIfTI conversion performs the corresponding axis reversal at its I/O boundary.
        spatial = [np.asarray(record.pixels) for record in volume]
        array = spatial[0] if len(spatial) == 1 else np.stack(spatial, axis=-1)
        arrays.append(array)
        representative = dict(volume[0].metadata)
        representative["slice_count"] = len(volume)
        metadata.append(representative)
        affines.append(_dicom_affine(volume))
    if not arrays:
        raise ValueError("DICOM series contains no image volumes")
    shapes = {tuple(array.shape) for array in arrays}
    if len(shapes) != 1:
        raise ValueError(f"DICOM series contains inconsistent spatial shapes: {sorted(shapes)}")
    return np.stack(arrays, axis=0), tuple(arrays[0].shape), metadata, affines


def _dicom_affine(volume: Sequence[_FrameRecord]) -> np.ndarray | None:
    first = volume[0].metadata
    orientation = np.asarray(first.get("image_orientation_patient") or (), dtype=float)
    position = np.asarray(first.get("image_position_patient") or (), dtype=float)
    spacing = np.asarray(first.get("pixel_spacing") or (), dtype=float)
    if orientation.size != 6 or position.size != 3 or spacing.size < 2:
        return None
    affine_lps = np.eye(4, dtype=float)
    x_direction = orientation[:3]
    y_direction = orientation[3:]
    normal = np.cross(x_direction, y_direction)
    affine_lps[:3, 0] = x_direction * float(spacing[1])
    affine_lps[:3, 1] = y_direction * float(spacing[0])
    if len(volume) > 1:
        last_position = np.asarray(volume[1].metadata.get("image_position_patient") or (), dtype=float)
        if last_position.size == 3:
            affine_lps[:3, 2] = last_position - position
        else:
            affine_lps[:3, 2] = normal * _slice_spacing(first)
    else:
        affine_lps[:3, 2] = normal * _slice_spacing(first)
    affine_lps[:3, 3] = position
    affine = np.eye(4, dtype=float)
    affine[:3, :] = _LPS_TO_RAS @ affine_lps[:3, :]
    return affine


def _dicom_axis_anatomical_labels(
    metadata: Mapping[str, object],
    *,
    spatial_ndim: int,
) -> tuple[str, ...] | None:
    """Map native DICOM array axes (rows, columns, slices) to anatomy."""

    orientation = np.asarray(metadata.get("image_orientation_patient") or (), dtype=float)
    if orientation.size != 6 or spatial_ndim not in {2, 3}:
        return None
    directions = [orientation[3:], orientation[:3]]
    if spatial_ndim == 3:
        directions.append(np.cross(orientation[:3], orientation[3:]))
    direction_matrix = np.column_stack(directions)
    norms = np.linalg.norm(direction_matrix, axis=0)
    if np.any(norms <= 1e-8):
        return None
    dominant = np.argmax(np.abs(direction_matrix / norms[None, :]), axis=0)
    labels = tuple("xyz"[int(index)] for index in dominant)
    return labels if len(set(labels)) == spatial_ndim else None


def _normalized_standard_metadata(
    first: PydicomDataset,
    frame_metadata: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    comments = [str(row.get("frame_comments") or row.get("frame_label") or "") for row in frame_metadata]
    return {
        "method_name": _sequence_name(first),
        "sequence_name": _sequence_name(first),
        "protocol_name": str(getattr(first, "ProtocolName", "") or ""),
        "comment": str(getattr(first, "ImageComments", "") or ""),
        "frame_comments": tuple(comments),
        "frame_count": len(frame_metadata),
        "echo_times": _frame_values(frame_metadata, "echo_time"),
        "repetition_times": _frame_values(frame_metadata, "repetition_time", collapse_constant=True),
        "inversion_times": _frame_values(frame_metadata, "inversion_time"),
        "flip_angles": _frame_values(frame_metadata, "flip_angle", collapse_constant=True),
        "diffusion_bvals": _frame_values(frame_metadata, "diffusion_b_value"),
        "diffusion_bvecs": _frame_vectors(frame_metadata, "diffusion_gradient_image"),
    }


def _frame_values(
    metadata: Sequence[Mapping[str, object]],
    key: str,
    *,
    collapse_constant: bool = False,
) -> list[float] | float | None:
    values = [_finite_float(row.get(key)) for row in metadata]
    if not values or all(value is None for value in values):
        return None
    fallback = next(value for value in values if value is not None)
    complete = [fallback if value is None else value for value in values]
    if collapse_constant and np.allclose(complete, complete[0]):
        return float(complete[0])
    return [float(value) for value in complete]


def _frame_vectors(metadata: Sequence[Mapping[str, object]], key: str) -> np.ndarray | None:
    values = [row.get(key) for row in metadata]
    if not values or any(value is None for value in values):
        return None
    array = np.asarray(values, dtype=float)
    return array if array.shape == (len(values), 3) else None


def _slice_coordinate(metadata: Mapping[str, object]) -> float | None:
    orientation = np.asarray(metadata.get("image_orientation_patient") or (), dtype=float)
    position = np.asarray(metadata.get("image_position_patient") or (), dtype=float)
    if orientation.size == 6 and position.size == 3:
        return float(np.dot(np.cross(orientation[:3], orientation[3:]), position))
    in_stack = _finite_float(metadata.get("in_stack_position"))
    return in_stack


def _slice_key(metadata: Mapping[str, object]) -> float | None:
    coordinate = _finite_float(metadata.get("slice_coordinate"))
    return None if coordinate is None else round(coordinate, 5)


def _slice_spacing(metadata: Mapping[str, object]) -> float:
    return float(
        _finite_float(metadata.get("spacing_between_slices"))
        or _finite_float(metadata.get("slice_thickness"))
        or 1.0
    )


def _diffusion_gradient_image(metadata: Mapping[str, object]) -> tuple[float, float, float] | None:
    gradient = np.asarray(metadata.get("diffusion_gradient_patient") or (), dtype=float)
    orientation = np.asarray(metadata.get("image_orientation_patient") or (), dtype=float)
    if gradient.size != 3 or orientation.size != 6:
        return None
    axes = np.column_stack((orientation[:3], orientation[3:], np.cross(orientation[:3], orientation[3:])))
    rotated = axes.T @ gradient
    norm = float(np.linalg.norm(rotated))
    if norm > 0:
        rotated = rotated / norm
    return (float(rotated[0]), float(rotated[1]), float(rotated[2]))


def _builtin_metadata_adapters() -> list[DicomMetadataAdapter]:
    from .dicom_bruker import BrukerDicomMetadataAdapter

    return [BrukerDicomMetadataAdapter()]


def _declared_frame_count(dataset: PydicomDataset) -> int:
    try:
        return max(1, int(getattr(dataset, "NumberOfFrames", 1) or 1))
    except (TypeError, ValueError):
        return 1


def _dataset_order(dataset: PydicomDataset) -> tuple[int, int]:
    return (
        _integer_value(getattr(dataset, "AcquisitionNumber", None)) or 0,
        _integer_value(getattr(dataset, "InstanceNumber", None)) or 0,
    )


def _series_description(dataset: PydicomDataset) -> str:
    return str(getattr(dataset, "SeriesDescription", "") or getattr(dataset, "ProtocolName", "") or "")


def _sequence_name(dataset: PydicomDataset) -> str:
    return str(
        getattr(dataset, "PulseSequenceName", "")
        or getattr(dataset, "SequenceName", "")
        or getattr(dataset, "ScanningSequence", "")
        or ""
    )


def _compact_series_name(
    dataset: PydicomDataset,
    *,
    fallback_index: int | None = None,
) -> str:
    description = _series_description(dataset) or _sequence_name(dataset)
    base = re.sub(r"[^A-Za-z0-9]+", "_", description).strip("_").lower()
    if not base:
        modality = re.sub(
            r"[^A-Za-z0-9]+",
            "_",
            str(getattr(dataset, "Modality", "") or "dicom").strip(),
        ).strip("_").lower()
        base = modality or "dicom"
    series_number = _integer_value(getattr(dataset, "SeriesNumber", None))
    if series_number is not None and not re.match(rf"^{series_number}(?:_|$)", base):
        base = f"{series_number}_{base}"
    elif series_number is None and fallback_index is not None and base == "dicom":
        base = f"dicom_{fallback_index}"
    return base[:_MAX_EXPERIMENT_NAME_LENGTH].rstrip("_") or "dicom"


def _all_affines_equal(affines: Sequence[np.ndarray | None]) -> bool:
    if not affines:
        return False
    first = affines[0]
    return all(
        (first is None and affine is None)
        or (first is not None and affine is not None and np.allclose(affine, first))
        for affine in affines[1:]
    )


def _unique_name(value: str, used: set[str]) -> str:
    base = (str(value or "dicom_series").strip() or "dicom_series")[
        :_MAX_EXPERIMENT_NAME_LENGTH
    ].rstrip("_")
    candidate = base
    index = 2
    while candidate in used:
        suffix = f"_{index}"
        candidate = f"{base[: _MAX_EXPERIMENT_NAME_LENGTH - len(suffix)].rstrip('_')}{suffix}"
        index += 1
    used.add(candidate)
    return candidate


def _numeric_tuple(value: object) -> tuple[float, ...] | None:
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=float).reshape((-1,))
    except (TypeError, ValueError):
        return None
    if not array.size or not np.all(np.isfinite(array)):
        return None
    return tuple(float(item) for item in array)


def _finite_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        result = float(cast(Any, value))
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _integer_value(value: object) -> int | None:
    try:
        return int(cast(Any, value)) if value is not None else None
    except (TypeError, ValueError):
        return None


def _rounded(value: object) -> float | None:
    result = _finite_float(value)
    return None if result is None else round(result, 6)


def _is_sequence_value(value: object) -> bool:
    return isinstance(value, (list, tuple)) or (
        hasattr(value, "__iter__") and not isinstance(value, (str, bytes))
    )


__all__ = [
    "DicomMetadataAdapter",
    "DicomMetadataPatch",
    "DicomSeriesInfo",
    "discover_dicom_series",
    "load_dicom_series",
    "load_dicom_study",
]
