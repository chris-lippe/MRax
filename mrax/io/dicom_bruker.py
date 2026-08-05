"""Bruker-specific metadata adaptation for standards-based DICOM import."""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence, cast

import numpy as np
from pydicom.dataset import Dataset as PydicomDataset

from .dicom import DicomMetadataPatch


class BrukerDicomMetadataAdapter:
    """Interpret Bruker conventions layered on Enhanced MR/PET DICOM objects.

    Bruker exports used by this adapter contain standards-based functional groups;
    pixel decoding and geometry therefore remain in the general DICOM layer. This
    adapter is limited to origin detection, sequence naming, frame annotations, and
    Bruker-specific experiment hints.
    """

    name = "bruker"

    def matches(self, datasets: Sequence[PydicomDataset]) -> bool:
        return bool(
            datasets
            and all("bruker" in str(getattr(dataset, "Manufacturer", "") or "").lower() for dataset in datasets)
        )

    def select_datasets(
        self,
        datasets: Sequence[PydicomDataset],
    ) -> Sequence[PydicomDataset]:
        """Select the primary Bruker image stack from one DICOM series.

        ParaVision may assign the same Series Instance UID to a quantitative
        enhanced object and smaller auxiliary parallel-imaging objects. Enhanced
        objects that belong to one standard DICOM concatenation stay together;
        otherwise the most complete original quantitative stack is selected.
        Classic single-frame series are left untouched.
        """

        if sum(_declared_frame_count(dataset) > 1 for dataset in datasets) < 2:
            return datasets

        groups: dict[str, list[PydicomDataset]] = {}
        classic: list[PydicomDataset] = []
        for index, dataset in enumerate(datasets):
            if _declared_frame_count(dataset) <= 1:
                classic.append(dataset)
                continue
            concatenation_uid = str(getattr(dataset, "ConcatenationUID", "") or "")
            key = f"concat:{concatenation_uid}" if concatenation_uid else f"object:{index}"
            groups.setdefault(key, []).append(dataset)
        if classic:
            groups["classic"] = classic
        if len(groups) <= 1:
            return datasets
        return max(groups.values(), key=_dataset_group_rank)

    def extract(
        self,
        datasets: Sequence[PydicomDataset],
        frame_metadata: Sequence[Mapping[str, object]],
    ) -> DicomMetadataPatch:
        if not datasets:
            return DicomMetadataPatch()
        first = datasets[0]
        pulse_sequence = str(
            getattr(first, "PulseSequenceName", "")
            or getattr(first, "SequenceName", "")
            or ""
        )
        protocol = str(getattr(first, "ProtocolName", "") or "")
        description = str(getattr(first, "SeriesDescription", "") or "")
        frame_comments = tuple(
            str(row.get("frame_comments") or row.get("frame_label") or "")
            for row in frame_metadata
        )
        combined = " ".join((pulse_sequence, protocol, description, *frame_comments)).lower()

        sat_trans_on = _bruker_sat_trans_enabled(first, combined)
        normalized: dict[str, object] = {
            "method_name": pulse_sequence or protocol or description,
            "sequence_name": pulse_sequence,
            "protocol_name": protocol,
            "comment": description,
            "frame_comments": frame_comments,
            "scan_time_ms": _acquisition_duration_ms(first),
            "sat_trans_on": sat_trans_on,
            "freq_offsets_ppm": _bruker_frame_offsets_ppm(frame_comments),
            "fair_labels": _bruker_fair_labels(frame_comments),
        }
        metadata = {
            "profile": _bruker_export_profile(first),
            "pulse_sequence_name": pulse_sequence,
            "acquisition_contrast": str(getattr(first, "AcquisitionContrast", "") or ""),
            "complex_image_component": str(getattr(first, "ComplexImageComponent", "") or ""),
            "image_type": _string_values(getattr(first, "ImageType", ())),
            "dimension_labels": _dimension_labels(first),
            "standard_functional_groups": bool(
                getattr(first, "SharedFunctionalGroupsSequence", None)
                or getattr(first, "PerFrameFunctionalGroupsSequence", None)
            ),
            "unavailable_observables": (
                ["sat_b1", "sat_time"] if sat_trans_on else []
            ),
        }
        return DicomMetadataPatch(
            normalized={key: value for key, value in normalized.items() if value is not None},
            metadata=metadata,
        )


def _bruker_export_profile(dataset: PydicomDataset) -> str:
    modality = str(getattr(dataset, "Modality", "") or "").lower()
    enhanced = bool(getattr(dataset, "PerFrameFunctionalGroupsSequence", None))
    if modality == "mr":
        return "enhanced_mr" if enhanced else "classic_mr"
    if modality in {"pt", "pet"}:
        return "enhanced_pet" if enhanced else "classic_pet"
    return f"enhanced_{modality}" if enhanced and modality else modality or "dicom"


def _declared_frame_count(dataset: PydicomDataset) -> int:
    try:
        return max(1, int(getattr(dataset, "NumberOfFrames", 1) or 1))
    except (TypeError, ValueError):
        return 1


def _dataset_group_rank(datasets: Sequence[PydicomDataset]) -> tuple[int, int, int, int]:
    image_type = {
        value.upper()
        for dataset in datasets
        for value in _string_values(getattr(dataset, "ImageType", ()))
    }
    original = int("ORIGINAL" in image_type or "DERIVED" not in image_type)
    non_parallel = int("NON_PARALLEL" in image_type)
    dimension_count = max(
        (len(getattr(dataset, "DimensionIndexSequence", ()) or ()) for dataset in datasets),
        default=0,
    )
    frame_count = sum(_declared_frame_count(dataset) for dataset in datasets)
    return original, non_parallel, dimension_count, frame_count


def _bruker_sat_trans_enabled(dataset: PydicomDataset, combined: str) -> bool | None:
    value = str(getattr(dataset, "MagnetizationTransfer", "") or "").strip().lower()
    if value in {"yes", "y", "true", "1"}:
        return True
    if any(token in combined for token in ("cest", "wasabi", "sat_trans", "sattrans")):
        return True
    if value in {"no", "n", "false", "0"}:
        return False
    return None


def _bruker_frame_offsets_ppm(comments: Sequence[str]) -> list[float] | None:
    offsets: list[float] = []
    for comment in comments:
        match = re.search(r"([-+]?\d+(?:\.\d+)?)\s*ppm\b", str(comment), re.IGNORECASE)
        if match is None:
            return None
        offsets.append(float(match.group(1)))
    return offsets or None


def _bruker_fair_labels(comments: Sequence[str]) -> list[int] | None:
    labels: list[int] = []
    for comment in comments:
        normalized = re.sub(r"[^a-z]", "", str(comment).lower())
        if "nonselective" in normalized or "global" in normalized:
            labels.append(1)
        elif "selective" in normalized or "slice" in normalized:
            labels.append(0)
        else:
            return None
    return labels or None


def _acquisition_duration_ms(dataset: PydicomDataset) -> float | None:
    value = getattr(dataset, "AcquisitionDuration", None)
    if value is None:
        return None
    try:
        duration = float(cast(Any, value))
    except (TypeError, ValueError):
        return None
    return duration * 1000.0 if np.isfinite(duration) and duration >= 0 else None


def _dimension_labels(dataset: PydicomDataset) -> list[str]:
    return [
        str(getattr(item, "DimensionDescriptionLabel", "") or "")
        for item in getattr(dataset, "DimensionIndexSequence", ()) or ()
        if str(getattr(item, "DimensionDescriptionLabel", "") or "")
    ]


def _string_values(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    try:
        return [str(item) for item in value]  # type: ignore[union-attr]
    except TypeError:
        return [str(value)]


__all__ = ["BrukerDicomMetadataAdapter"]
