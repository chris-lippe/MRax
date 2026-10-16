from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from mrax.data_struct import AcquisitionData, DataSet, Experiment, Model
from mrax.devices import device_function
from mrax.io.import_definitions import (
    NormalizedMetadata,
    detect_experiment_type,
    needs_manual_type_override,
    normalize_sequence_type,
    reshape_epi_samples,
    reshape_rare_samples,
)


@dataclass(frozen=True)
class PulseqSequenceInfo:
    """Parsed Pulseq sequence metadata and ADC trajectory."""

    path: Path
    definitions: Mapping[str, object] = field(default_factory=dict)
    trajectory: np.ndarray | None = None
    trajectory_physical: np.ndarray | None = None
    adc_times: np.ndarray | None = None
    adc_counts: tuple[int, ...] = ()
    sequence_kind: str = "unknown"
    sequence_type: str = "standard"
    spatial_shape: tuple[int, ...] | None = None
    normalized_metadata: NormalizedMetadata = field(default_factory=NormalizedMetadata)


def _parse_scalar(token: str) -> object:
    text = token.strip()
    if text == "":
        return text
    try:
        if any(c in text.lower() for c in (".", "e")):
            return float(text)
        return int(text)
    except ValueError:
        return text


def _parse_definition_value(raw: str) -> object:
    parts = raw.strip().split()
    if not parts:
        return ""
    parsed = [_parse_scalar(part) for part in parts]
    if len(parsed) == 1:
        return parsed[0]
    return parsed


def _read_sections(path: Path) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for raw_line in path.read_text(errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1].strip().upper()
            sections.setdefault(current, [])
            continue
        if current is not None:
            sections[current].append(line)
    return sections


def _parse_pulseq_text(path: Path) -> tuple[dict[str, object], tuple[int, ...]]:
    sections = _read_sections(path)
    definitions: dict[str, object] = {}
    for line in sections.get("DEFINITIONS", []):
        parts = line.split(None, 1)
        if len(parts) == 1:
            definitions[parts[0]] = ""
        else:
            definitions[parts[0]] = _parse_definition_value(parts[1])

    adc_samples_by_id: dict[int, int] = {}
    for line in sections.get("ADC", []):
        parts = line.split()
        if len(parts) >= 2:
            adc_samples_by_id[int(parts[0])] = int(float(parts[1]))

    adc_counts: list[int] = []
    for line in sections.get("BLOCKS", []):
        parts = line.split()
        if len(parts) >= 7:
            adc_id = int(float(parts[6]))
            if adc_id > 0 and adc_id in adc_samples_by_id:
                adc_counts.append(adc_samples_by_id[adc_id])
    return definitions, tuple(adc_counts)


def _numpy_definitions(definitions: Mapping[str, object] | None) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in dict(definitions or {}).items():
        if isinstance(value, np.ndarray):
            out[str(key)] = value.tolist()
        elif isinstance(value, np.generic):
            out[str(key)] = value.item()
        else:
            out[str(key)] = value
    return out


def _definition_list(definitions: Mapping[str, object], *keys: str) -> list[float]:
    for key in keys:
        if key not in definitions:
            continue
        value = definitions[key]
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if isinstance(value, (list, tuple)):
            vals = []
            for item in value:
                try:
                    vals.append(float(item))
                except (TypeError, ValueError):
                    continue
            if vals:
                return vals
        try:
            return [float(value)]
        except (TypeError, ValueError):
            continue
    return []


def _coerce_positive_int(value: object) -> int | None:
    try:
        resolved = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return resolved if resolved > 0 else None


def _definition_count_hint(definitions: Mapping[str, object], *keys: str) -> int | None:
    values = _definition_list(definitions, *keys)
    return len(values) if len(values) > 1 else None


def _infer_spatial_shape(definitions: Mapping[str, object], adc_counts: Sequence[int]) -> tuple[int, int] | None:
    matrix = _definition_list(definitions, "Matrix", "matrix", "ReconMatrix", "recon_matrix")
    if len(matrix) >= 2:
        return (int(round(matrix[1])), int(round(matrix[0])))
    if len(adc_counts) > 1 and len(set(int(v) for v in adc_counts)) == 1:
        return (len(adc_counts), int(adc_counts[0]))
    if len(adc_counts) == 1:
        return (1, int(adc_counts[0]))
    return None


def _infer_trajectory_shape(definitions, adc_counts, physical):
    matrix = _definition_list(definitions, "Matrix", "matrix", "ReconMatrix", "recon_matrix")
    if len(matrix) >= 2:
        return tuple(int(round(v)) for v in reversed(matrix[:3] if len(matrix) > 2 and matrix[2] > 1 else matrix[:2]))
    fov = _definition_list(definitions, "FOV", "fov")
    if physical is not None and len(fov) >= physical.shape[1]:
        extent = np.max(np.abs(physical), axis=0) * np.asarray(fov[:physical.shape[1]]) * 2
        if adc_counts and len(set(adc_counts)) == 1 and int(adc_counts[0]) > 1:
            profiles = physical.reshape((-1, int(adc_counts[0]), physical.shape[1]))
            spread = np.ptp(profiles, axis=1)
            read = int(np.argmax(np.median(spread, axis=0)))
            other = [i for i in range(physical.shape[1]) if i != read]
            if np.max(spread[:, other], initial=0.) < 1e-2:
                step = np.median(np.abs(np.diff(profiles[..., read], axis=-1)))
                extent[read] = (np.median(spread[:, read]) + step) * fov[read]
        if np.all(extent > 1):
            return tuple(int(round(v)) for v in extent[::-1])
    return _infer_spatial_shape(definitions, adc_counts)


def _infer_pulseq_frame_count_hint(
    definitions: Mapping[str, object],
    sidecar_metadata: Mapping[str, object] | None = None,
) -> int | None:
    """Infer a likely frame count from sidecar hints or observable definitions."""

    metadata = dict(sidecar_metadata or {})
    explicit = (
        _coerce_positive_int(metadata.get("frame_count"))
        or _coerce_positive_int(metadata.get("num_frames"))
        or _coerce_positive_int(metadata.get("frames"))
    )
    if explicit is not None:
        return explicit

    candidates = [
        _definition_count_hint(definitions, "TE", "EchoTime"),
        _definition_count_hint(definitions, "TR", "RepetitionTime"),
        _definition_count_hint(definitions, "TI", "InversionTime"),
        _definition_count_hint(definitions, "FlipAngle", "FA"),
    ]
    positive = [int(v) for v in candidates if v is not None and int(v) > 1]
    return max(positive) if positive else None


def _tile_density(density: np.ndarray | None, repetitions: int) -> np.ndarray | None:
    if density is None:
        return None
    if repetitions <= 1:
        return np.asarray(density, dtype=np.float32)
    return np.tile(np.asarray(density, dtype=np.float32), int(repetitions))


def _match_non_cartesian_samples_to_trajectory(
    info: PulseqSequenceInfo,
    samples: np.ndarray,
    *,
    density: np.ndarray | None,
    sidecar_metadata: Mapping[str, object] | None = None,
    sample_axis: int = -1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict[str, object]]:
    """Match Pulseq non-Cartesian samples to the base ADC trajectory.

    This helper covers three common cases:

    - samples already match one trajectory pass: ``(..., num_samples)``
    - a serial stream contains several frames with the same trajectory repeated
    - a serial stream contains several contiguous trajectory passes that should stay
      in one frame (for example repeated interleaves)
    """

    if info.trajectory is None:
        raise ValueError("Pulseq non-Cartesian import requires a calculated ADC trajectory")

    arr = np.asarray(samples)
    axis = int(sample_axis)
    if axis < 0:
        axis += arr.ndim
    if axis < 0 or axis >= arr.ndim:
        raise ValueError("sample_axis is out of range for the provided non-Cartesian samples")
    if axis != arr.ndim - 1:
        arr = np.moveaxis(arr, axis, -1)
        axis = arr.ndim - 1

    base_trajectory = np.asarray(info.trajectory, dtype=np.float32)
    base_sample_count = int(base_trajectory.shape[0])
    if base_sample_count <= 0:
        raise ValueError("Pulseq non-Cartesian trajectory must contain at least one sample")

    sample_count = int(arr.shape[-1])
    frame_count_hint = _infer_pulseq_frame_count_hint(info.definitions, sidecar_metadata)
    reps_per_frame = (
        _coerce_positive_int((sidecar_metadata or {}).get("trajectory_repetitions_per_frame"))
        or _coerce_positive_int((sidecar_metadata or {}).get("interleaves_per_frame"))
        or _coerce_positive_int((sidecar_metadata or {}).get("shots_per_frame"))
    )
    matching_meta: dict[str, object] = {
        "sample_axis": int(axis),
        "base_trajectory_samples": int(base_sample_count),
        "frame_count_hint": None if frame_count_hint is None else int(frame_count_hint),
        "trajectory_repetitions_per_frame": None if reps_per_frame is None else int(reps_per_frame),
    }

    if sample_count == base_sample_count:
        matching_meta["mode"] = "single_trajectory"
        return arr, base_trajectory, density, matching_meta

    if sample_count % base_sample_count != 0:
        raise ValueError(
            "Pulseq non-Cartesian sample count does not match the calculated trajectory: "
            f"{sample_count} samples vs {base_sample_count} per trajectory pass"
        )

    trajectory_repetitions = sample_count // base_sample_count
    matching_meta["trajectory_repetitions"] = int(trajectory_repetitions)

    if reps_per_frame is not None:
        if trajectory_repetitions % int(reps_per_frame) != 0:
            raise ValueError(
                "trajectory_repetitions_per_frame does not divide the number of repeated "
                f"trajectory passes ({trajectory_repetitions})"
            )
        expanded_trajectory = np.tile(base_trajectory, (int(reps_per_frame), 1))
        expanded_density = _tile_density(density, int(reps_per_frame))
        expected_per_frame = int(expanded_trajectory.shape[0])
        if sample_count == expected_per_frame:
            matching_meta["mode"] = "expanded_trajectory_per_frame"
            return arr, expanded_trajectory, expanded_density, matching_meta
        if arr.ndim == 1:
            frame_count = trajectory_repetitions // int(reps_per_frame)
            arr = arr.reshape((int(frame_count), expected_per_frame))
            matching_meta["mode"] = "reshaped_serial_frames_with_expanded_trajectory"
            matching_meta["frame_count"] = int(frame_count)
            return arr, expanded_trajectory, expanded_density, matching_meta

    if arr.ndim == 1 and frame_count_hint is not None and trajectory_repetitions == int(frame_count_hint):
        arr = arr.reshape((int(frame_count_hint), base_sample_count))
        matching_meta["mode"] = "reshaped_serial_frames_from_frame_hint"
        matching_meta["frame_count"] = int(frame_count_hint)
        return arr, base_trajectory, density, matching_meta

    expanded_trajectory = np.tile(base_trajectory, (int(trajectory_repetitions), 1))
    expanded_density = _tile_density(density, int(trajectory_repetitions))
    matching_meta["mode"] = "expanded_trajectory_serial"
    return arr, expanded_trajectory, expanded_density, matching_meta


def _split_adc_trajectory(trajectory: np.ndarray, adc_counts: Sequence[int]) -> list[np.ndarray]:
    counts = [int(v) for v in adc_counts if int(v) > 0]
    if not counts or sum(counts) != int(trajectory.shape[0]):
        return []
    lines: list[np.ndarray] = []
    pos = 0
    for count in counts:
        lines.append(trajectory[pos : pos + count])
        pos += count
    return lines


def detect_pulseq_kspace_kind(trajectory: np.ndarray | None, adc_counts: Sequence[int] = ()) -> str:
    """Infer whether a Pulseq ADC trajectory is Cartesian, EPI, or non-Cartesian."""

    if trajectory is None:
        return "unknown"
    traj = np.asarray(trajectory, dtype=np.float32)
    if traj.ndim != 2 or traj.shape[1] < 2 or traj.shape[0] == 0:
        return "unknown"
    lines = _split_adc_trajectory(traj, adc_counts)
    if lines:
        read_signs = []
        line_like = True
        for line in lines:
            if line.shape[0] < 2:
                continue
            delta = line[-1, 0] - line[0, 0]
            if abs(float(delta)) > 1e-7:
                read_signs.append(float(np.sign(delta)))
            if float(np.nanmax(line[:, 1]) - np.nanmin(line[:, 1])) > 1e-3:
                line_like = False
        if line_like and len(read_signs) >= 2 and len(set(read_signs)) > 1:
            return "epi"
        if line_like:
            return "cartesian"

    unique_x = np.unique(np.round(traj[:, 0], 5))
    unique_y = np.unique(np.round(traj[:, 1], 5))
    if unique_x.size * unique_y.size == traj.shape[0]:
        return "cartesian"
    return "non_cartesian"


def _normalize_trajectory(
    trajectory_physical: np.ndarray,
    *,
    spatial_shape: tuple[int, ...] | None,
    definitions: Mapping[str, object],
    normalize: bool,
) -> np.ndarray:
    traj = np.asarray(trajectory_physical, dtype=np.float32)
    if not normalize:
        return traj
    fov = _definition_list(definitions, "FOV", "fov")
    if spatial_shape is not None and len(fov) >= traj.shape[1]:
        scale = np.asarray(fov[:traj.shape[1]]) / np.asarray(spatial_shape[::-1])
        return (traj * scale[None, :]).astype(np.float32)
    raise ValueError("Pulseq trajectory normalization requires FOV and an encoding matrix")


def _metadata_from_pulseq(definitions: Mapping[str, object], frame_count: int, path: Path, *, definition_paths=()) -> NormalizedMetadata:
    from mrax.io.pulseq_metadata import sequence_metadata

    return sequence_metadata(definitions, frame_count, path, definition_paths=definition_paths)


def _detect_pulseq_sequence_type(definitions: Mapping[str, object], path: Path) -> str:
    combined = " ".join(
        str(definitions.get(key, "")) for key in ("Name", "SequenceName", "ProtocolName")
    )
    combined = f"{combined} {path.stem}".lower()
    if any(token in combined for token in ("rare", "turbo se", "turbo_se", "turbose", "tse", "fse")):
        return "rare"
    return "standard"


def _manual_override_warning_message(meta: NormalizedMetadata, path: Path) -> str:
    return (
        "Could not determine the experiment type automatically for "
        f"{path}. Falling back to 'anatomical'. "
        f"Method='{meta.method_name}', Sequence='{meta.sequence_name}', Protocol='{meta.protocol_name}'. "
        "Provide a manual override (override_type=...) or a model_map_by_type entry for quantitative fitting."
    )


def _fit_override_warning_message(
    meta: NormalizedMetadata,
    path: Path,
    *,
    exp_type: str,
    default_identity_model: bool,
    manual_type_override_required: bool,
) -> str:
    if manual_type_override_required:
        message = _manual_override_warning_message(meta, path)
        if default_identity_model:
            message += " The imported default model is 'identity'; fitting it only estimates an amplitude image."
        return message + " A manual experiment/model override is required before quantitative fitting."
    return (
        f"Experiment imported from {path} uses the default identity model for exp_type='{exp_type}'. "
        "This is expected for localizers and other non-quantitative scans, but fitting it only estimates "
        "an amplitude image. Provide model_map_by_type or pass a custom model to run_fit for quantitative fitting."
    )


def _manual_fit_override_warning_message(meta: NormalizedMetadata, path: Path) -> str:
    return (
        _manual_override_warning_message(meta, path)
        + " A manual experiment/model override is required before quantitative fitting."
    )


def _attach_fit_override_warning(
    experiment: Experiment,
    _model: Model,
    *,
    identity_message: str | None = None,
    manual_message: str | None = None,
) -> None:
    if identity_message:
        experiment._mrax_default_identity_fit_warning = identity_message
    if manual_message:
        experiment._mrax_manual_type_override_warning = manual_message


def _coerce_slice_order(value: object) -> list[int] | None:
    """Normalize a metadata-provided slice order."""

    if value is None:
        return None
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        return None
    out: list[int] = []
    for item in value:
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            return None
    return out if out else None


def read_pulseq_sidecar_metadata(seq_path: str | Path) -> dict[str, object]:
    """Read optional Pulseq sidecar metadata from JSON files next to the sequence."""

    path = Path(seq_path)
    candidates = [
        path.with_suffix(".json"),
        path.with_name(path.stem + "_metadata.json"),
        path.with_name(path.name + ".json"),
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            loaded = json.loads(candidate.read_text())
        except Exception as exc:
            warnings.warn(
                f"Could not parse Pulseq sidecar metadata {candidate}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            return {}
        if isinstance(loaded, dict):
            metadata = dict(loaded)
            metadata.setdefault("path", str(candidate))
            return metadata
    return {}


def read_pulseq(
    seq_path: str | Path,
    *,
    normalize_trajectory: bool = True,
    definition_paths=(),
    definition_overrides: Mapping | None = None,
    calculate_trajectory: bool = True,
) -> PulseqSequenceInfo:
    """Read a Pulseq ``.seq`` file and return definitions plus ADC trajectory."""

    path = Path(seq_path)
    definitions, adc_counts = _parse_pulseq_text(path)
    trajectory_physical: np.ndarray | None = None
    adc_times: np.ndarray | None = None
    try:
        if calculate_trajectory:
            from pypulseq import Sequence  # type: ignore

            sequence = Sequence()
            from mrax.io.pulseq_compat import read_sequence

            read_sequence(sequence, path)
            definitions = {**definitions, **_numpy_definitions(getattr(sequence, "definitions", {}))}
            k_adc, _k_full, _t_excitation, _t_refocusing, t_adc = sequence.calculate_kspace()
            k_adc_arr = np.asarray(k_adc, dtype=np.float32)
            if k_adc_arr.ndim == 2 and k_adc_arr.shape[0] >= 2 and k_adc_arr.shape[1] > 0:
                active_dims = 3 if np.ptp(k_adc_arr[2]) > 1e-3 else 2
                trajectory_physical = k_adc_arr[:active_dims].T
            adc_times_arr = np.asarray(t_adc, dtype=np.float64)
            if adc_times_arr.size:
                adc_times = adc_times_arr.reshape(-1)
    except ImportError:
        warnings.warn(
            "pypulseq is not installed; returning Pulseq definitions without calculated trajectory.",
            RuntimeWarning,
            stacklevel=2,
        )
    except Exception as exc:
        warnings.warn(
            f"Pulseq trajectory calculation failed for {path}: {exc}. Returning definitions only.",
            RuntimeWarning,
            stacklevel=2,
        )

    sidecar = read_pulseq_sidecar_metadata(path)
    definitions = {**definitions, **dict(sidecar.get("sequence_definitions", {})), **dict(definition_overrides or {})}
    fov = _definition_list(definitions, "FOV", "fov")
    fov_unit = str(definitions.get("FOVUnit", "")).strip().lower()
    if fov_unit not in ("", "m", "mm"):
        raise ValueError("Pulseq FOVUnit must be m or mm")
    if fov and (fov_unit == "mm" or not fov_unit and max(fov) > 10):
        if not fov_unit:
            warnings.warn("Pulseq FOV appears to use millimetres; converting to metres. Declare FOVUnit to make the convention explicit", RuntimeWarning, stacklevel=2)
        definitions = {**definitions, "FOV": [v / 1000. for v in fov], "FOVUnit": "m"}
    spatial_shape = _infer_trajectory_shape(definitions, adc_counts, trajectory_physical)
    trajectory = (
        _normalize_trajectory(
            trajectory_physical,
            spatial_shape=spatial_shape,
            definitions=definitions,
            normalize=normalize_trajectory,
        )
        if trajectory_physical is not None and (not normalize_trajectory or spatial_shape is not None and len(_definition_list(definitions, "FOV", "fov")) >= trajectory_physical.shape[1])
        else None
    )
    kind = detect_pulseq_kspace_kind(trajectory, adc_counts)
    sequence_type = normalize_sequence_type(_detect_pulseq_sequence_type(definitions, path))
    normalized_meta = _metadata_from_pulseq(definitions, frame_count=1, path=path, definition_paths=definition_paths)
    return PulseqSequenceInfo(
        path=path,
        definitions=definitions,
        trajectory=trajectory,
        trajectory_physical=trajectory_physical,
        adc_times=adc_times,
        adc_counts=tuple(int(v) for v in adc_counts),
        sequence_kind=kind,
        sequence_type=sequence_type,
        spatial_shape=spatial_shape,
        normalized_metadata=normalized_meta,
    )






def default_interleaved_slice_order(slice_count: int) -> list[int]:
    """Return a simple even-then-odd slice order to reduce adjacent-slice crosstalk."""

    count = int(slice_count)
    if count <= 0:
        return []
    return list(range(0, count, 2)) + list(range(1, count, 2))


def resolve_rare_slice_order(
    slice_count: int,
    *,
    explicit: Sequence[int] | None = None,
    sidecar_metadata: Mapping[str, object] | None = None,
) -> list[int]:
    """Resolve RARE slice order from explicit input, metadata, or default interleaving."""

    count = int(slice_count)
    if count <= 0:
        return []

    if explicit is not None:
        resolved = [int(v) for v in explicit]
    else:
        metadata = dict(sidecar_metadata or {})
        resolved = (
            _coerce_slice_order(metadata.get("rare_slice_order"))
            or _coerce_slice_order(metadata.get("slice_order"))
            or default_interleaved_slice_order(count)
        )

    if len(resolved) != count:
        raise ValueError("slice_order length must match the number of repeated RARE acquisitions")
    if sorted(resolved) != list(range(count)):
        raise ValueError("slice_order must be a permutation of frame indices 0..N-1")
    return resolved


def _expand_rare_trajectory_for_repetitions(
    info: PulseqSequenceInfo,
    repetition_count: int,
    *,
    rare_factor: int,
    slice_order: Sequence[int] | None,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Repeat the per-line RARE trajectory for interleaved slice/frame acquisitions."""

    if info.trajectory is None:
        raise ValueError("RARE trajectory expansion requires a calculated trajectory")
    repetition_count = int(repetition_count)
    if repetition_count <= 1:
        return np.asarray(info.trajectory, dtype=np.float32), tuple(int(v) for v in info.adc_counts)

    if rare_factor <= 0:
        raise ValueError("rare_factor must be positive")
    lines = _split_adc_trajectory(np.asarray(info.trajectory, dtype=np.float32), info.adc_counts)
    if not lines:
        raise ValueError("Could not split the RARE trajectory into ADC lines")

    resolved_slice_order = resolve_rare_slice_order(repetition_count, explicit=slice_order)

    shot_blocks: list[tuple[list[np.ndarray], tuple[int, ...]]] = []
    pos = 0
    counts = tuple(int(v) for v in info.adc_counts)
    while pos < len(lines):
        block_lines = lines[pos : pos + int(rare_factor)]
        block_counts = counts[pos : pos + int(rare_factor)]
        shot_blocks.append((block_lines, block_counts))
        pos += int(rare_factor)

    expanded_lines: list[np.ndarray] = []
    expanded_counts: list[int] = []
    for block_lines, block_counts in shot_blocks:
        for _slice_idx in resolved_slice_order:
            for count, line in zip(block_counts, block_lines):
                expanded_lines.append(np.asarray(line, dtype=np.float32))
                expanded_counts.append(int(count))
    return np.concatenate(expanded_lines, axis=0), tuple(expanded_counts)


def _prepare_pulseq_acquisition(
    info: PulseqSequenceInfo,
    kspace: np.ndarray,
    *,
    spatial_shape: tuple[int, ...],
    density: np.ndarray | None,
    sequence_type: str | None,
    rare_slice_order: Sequence[int] | None,
    sidecar_metadata: Mapping[str, object] | None = None,
    sample_axis: int = -1,
    coil_axis: int | None = None,
) -> tuple[AcquisitionData, dict]:
    resolved_sequence_type = normalize_sequence_type(sequence_type or info.sequence_type)
    kind = info.sequence_kind if info.sequence_kind != "unknown" else "non_cartesian"
    meta = {
        "pulseq_path": str(info.path),
        "adc_counts": tuple(info.adc_counts),
        "trajectory_units": "cycles/pixel",
        "sequence_type": resolved_sequence_type,
    }
    serial = np.asarray(kspace)
    if not serial.size:
        raise ValueError("Pulseq samples cannot be empty")
    if info.trajectory_physical is not None and serial.shape[sample_axis] == len(info.trajectory_physical):
        from dataclasses import replace

        from mrax.io.pulseq_samples import prepare_pulseq_samples

        if coil_axis is None:
            frames = np.moveaxis(serial, sample_axis, -1).reshape((-1, 1, serial.shape[sample_axis]))
        else:
            if coil_axis % serial.ndim == sample_axis % serial.ndim:
                raise ValueError("coil_axis and sample_axis must be different")
            axes = [i for i in range(serial.ndim) if i not in (coil_axis % serial.ndim, sample_axis % serial.ndim)]
            axes += [coil_axis % serial.ndim, sample_axis % serial.ndim]
            ordered = serial.transpose(axes)
            frames = ordered.reshape((-1,) + ordered.shape[-2:])
        prepared = [prepare_pulseq_samples(info, frame, spatial_shape=spatial_shape, density="auto" if density is None else density) for frame in frames]
        first = prepared[0]
        options = {"samples": np.concatenate([a.samples for a in prepared], axis=0)}
        if first.acquired is not None:
            options["acquired"] = np.concatenate([np.broadcast_to(a.acquired, (a.frame_count, 1) + a.samples.shape[2:]) for a in prepared], axis=0)
        return replace(first, **options), meta
    if (
        kind == "cartesian"
        and resolved_sequence_type == "rare"
        and info.trajectory is not None
        and info.adc_counts
    ):
        serial = np.asarray(kspace)
        base_sample_count = int(sum(info.adc_counts))
        if base_sample_count <= 0:
            raise ValueError("RARE import requires positive ADC counts")
        if serial.shape[-1] % base_sample_count != 0:
            raise ValueError(
                "RARE serial sample count does not match the Pulseq trajectory: "
                f"{serial.shape[-1]} samples vs {base_sample_count} per trajectory pass"
            )
        repetition_count = serial.shape[-1] // base_sample_count
        rare_factor_vals = _definition_list(info.definitions, "RAREFactor", "EchoTrainLength")
        rare_factor = int(round(rare_factor_vals[0])) if rare_factor_vals else 1
        rare_trajectory, rare_adc_counts = _expand_rare_trajectory_for_repetitions(
            info,
            repetition_count,
            rare_factor=rare_factor,
            slice_order=rare_slice_order,
        )
        meta["trajectory_repetitions"] = int(repetition_count)
        if rare_slice_order is not None:
            meta["slice_order"] = [int(v) for v in rare_slice_order]
        values = reshape_rare_samples(serial, trajectory=rare_trajectory,
            adc_counts=rare_adc_counts, spatial_shape=spatial_shape, frame_order=rare_slice_order)
        acquired = reshape_rare_samples(np.ones(serial.shape, bool), trajectory=rare_trajectory,
            adc_counts=rare_adc_counts, spatial_shape=spatial_shape, frame_order=rare_slice_order)
        return _canonical_acquisition(values, spatial_shape, acquired=acquired), meta
    if kind == "epi" and info.adc_counts and len(set(info.adc_counts)) == 1:
        values = reshape_epi_samples(kspace, samples_per_line=int(info.adc_counts[0]),
            line_count=len(info.adc_counts), reverse_odd_lines=True)
        return _canonical_acquisition(values, spatial_shape), meta
    if kind == "cartesian" and info.adc_counts and len(set(info.adc_counts)) == 1:
        grid = np.asarray(kspace)
        if grid.shape[-1] == int(sum(info.adc_counts)):
            grid = grid.reshape(grid.shape[:-1] + (len(info.adc_counts), int(info.adc_counts[0])))
        return _canonical_acquisition(grid, spatial_shape), meta
    if info.trajectory is None:
        raise ValueError("Pulseq non-Cartesian import requires a calculated ADC trajectory")
    matched_samples, matched_trajectory, matched_density, matching_meta = _match_non_cartesian_samples_to_trajectory(
        info,
        np.asarray(kspace),
        density=None if density is None else np.asarray(density, dtype=np.float32),
        sidecar_metadata=sidecar_metadata,
        sample_axis=sample_axis,
    )
    meta.update(
        {
            "sample_matching": matching_meta,
            "trajectory_samples": int(np.asarray(matched_trajectory).shape[0]),
        }
    )
    return _canonical_acquisition(matched_samples, spatial_shape,
        trajectory=matched_trajectory[:, ::-1], density=matched_density), meta


@device_function
def import_pulseq_experiment(
    seq_path: str | Path,
    *,
    measurement_path: str | Path | None = None,
    reader_options: Mapping | None = None,
    metadata_overrides: Mapping | None = None,
    device="cpu",
    definition_paths=(),
    kspace: np.ndarray | None = None,
    image_data: np.ndarray | None = None,
    spatial_shape: tuple[int, ...] | None = None,
    density: np.ndarray | None = None,
    sample_axis: int = -1,
    coil_axis: int | None = None,
    sequence_type: str | None = None,
    rare_slice_order: Sequence[int] | None = None,
    override_type: str | None = None,
    name: str | None = None,
    model_map_by_type: Mapping[str, Model | Callable[[], Model]] | None = None,
) -> tuple[Experiment, DataSet, Model, dict]:
    """Build MRax containers from a Pulseq sequence and optional acquired data.

    Pulseq files define the sequence and ADC trajectory, but not measured signal
    samples. Supply ``image_data`` for the experiment and optionally ``kspace``.
    Raw-only workflows use ``read_pulseq_acquisition`` and reconstruct separately.
    """

    if measurement_path is not None or Path(seq_path).suffix.lower() != ".seq":
        from mrax.io.measurement import build_measurement_experiment
        from mrax.io.pulseq_measurement import read_pulseq_measurement

        measurement = read_pulseq_measurement(measurement_path or seq_path,
            sequence_path=seq_path if measurement_path is not None else None,
            reader_options=reader_options, metadata_overrides=metadata_overrides, definition_paths=definition_paths)
        return build_measurement_experiment(measurement, image_data=image_data,
            override_type=override_type, name=name, model_map_by_type=model_map_by_type, device=device)
    if reader_options is not None:
        raise ValueError("reader_options require a measurement file")
    info = read_pulseq(seq_path, definition_paths=definition_paths)
    sidecar_metadata = read_pulseq_sidecar_metadata(seq_path)
    resolved_shape = spatial_shape or info.spatial_shape
    if resolved_shape is None:
        if image_data is None:
            raise ValueError("spatial_shape is required when it cannot be inferred from the Pulseq file")
        arr = np.asarray(image_data)
        if arr.ndim < 2:
            raise ValueError("image_data must include at least two spatial dimensions")
        resolved_shape = (int(arr.shape[-2]), int(arr.shape[-1]))
    resolved_shape = tuple(int(value) for value in resolved_shape)

    acquisition = None
    raw_metadata = {}
    if kspace is not None:
        resolved_rare_slice_order = None
        if normalize_sequence_type(sequence_type or info.sequence_type) == "rare":
            serial = np.asarray(kspace)
            base_sample_count = int(sum(info.adc_counts))
            if base_sample_count > 0 and serial.shape[-1] % base_sample_count == 0:
                repetition_count = serial.shape[-1] // base_sample_count
                if repetition_count > 1:
                    resolved_rare_slice_order = resolve_rare_slice_order(
                        repetition_count,
                        explicit=rare_slice_order,
                        sidecar_metadata=sidecar_metadata,
                    )
        acquisition, raw_metadata = _prepare_pulseq_acquisition(
            info,
            np.asarray(kspace),
            spatial_shape=resolved_shape,
            density=density,
            sequence_type=sequence_type,
            rare_slice_order=resolved_rare_slice_order,
            sidecar_metadata=sidecar_metadata,
            sample_axis=sample_axis,
            coil_axis=coil_axis,
        )
    else:
        resolved_rare_slice_order = (
            [int(v) for v in rare_slice_order] if rare_slice_order is not None else _coerce_slice_order(
                sidecar_metadata.get("rare_slice_order")
            ) or _coerce_slice_order(sidecar_metadata.get("slice_order"))
        )

    if image_data is None:
        raise ValueError("Image data are required for an experiment; prepare raw samples with read_pulseq_acquisition and reconstruct_acquisition first")

    from dataclasses import replace

    from mrax.data_struct import SpatialGeometry
    from mrax.io.measurement import MeasurementData, build_measurement_experiment

    data_arr = np.asarray(image_data)
    if acquisition is not None and acquisition.domain == "kspace":
        resolved_shape = tuple(acquisition.recipe.output_shape or acquisition.encoding_shape)
    if tuple(data_arr.shape[-len(resolved_shape):]) != resolved_shape:
        raise ValueError("Image shape does not match the prepared encoding or declared spatial_shape")
    frame_count = int(np.prod(data_arr.shape[:-len(resolved_shape)])) if data_arr.ndim > len(resolved_shape) else 1
    normalized_meta = _metadata_from_pulseq(info.definitions, frame_count=frame_count, path=info.path, definition_paths=definition_paths)
    if metadata_overrides:
        normalized_meta = replace(normalized_meta, **metadata_overrides)
    exp_type = override_type or detect_experiment_type(normalized_meta)
    manual_type_override_required = override_type is None and needs_manual_type_override(normalized_meta, exp_type)
    geometry = acquisition.geometry if acquisition is not None else None
    if geometry is None:
        fov = _definition_list(info.definitions, "FOV", "fov")
        spacing = tuple(np.asarray(fov[:len(resolved_shape)])[::-1] * 1000. / np.asarray(resolved_shape)) if len(fov) >= len(resolved_shape) else None
        geometry = SpatialGeometry(spatial_shape=resolved_shape, voxel_sizes=spacing, source="pulseq")
    experiment, dataset, model, _metadata = build_measurement_experiment(MeasurementData(
        image_data=data_arr, acquisition=acquisition, spatial_geometry=geometry,
        normalized_metadata=normalized_meta, metadata=raw_metadata), name=name or info.path.stem,
        override_type=exp_type, model_map_by_type=model_map_by_type, device=device)
    model_from_default = not model_map_by_type or exp_type not in model_map_by_type

    default_identity_model = model_from_default and str(getattr(model, "name", "")).lower() == "identity"
    identity_message = None
    manual_message = None
    if default_identity_model:
        identity_message = _fit_override_warning_message(
            normalized_meta,
            info.path,
            exp_type=exp_type,
            default_identity_model=default_identity_model,
            manual_type_override_required=manual_type_override_required,
        )
    if manual_type_override_required:
        manual_message = _manual_fit_override_warning_message(normalized_meta, info.path)
    if identity_message or manual_message:
        _attach_fit_override_warning(
            experiment,
            model,
            identity_message=identity_message,
            manual_message=manual_message,
        )

    metadata = {
        "path": info.path,
        "type": exp_type,
        "pulseq": {
            "definitions": dict(info.definitions),
            "sidecar_metadata": sidecar_metadata,
            "adc_counts": tuple(info.adc_counts),
            "sequence_kind": info.sequence_kind,
            "sequence_type": normalize_sequence_type(sequence_type or info.sequence_type),
            "slice_order": None if resolved_rare_slice_order is None else [int(v) for v in resolved_rare_slice_order],
            "spatial_shape": resolved_shape,
            "sample_axis": int(sample_axis),
            "adc_times": info.adc_times,
            "trajectory_physical": info.trajectory_physical,
        },
    }
    return experiment, dataset, model, metadata


__all__ = [
    "PulseqSequenceInfo",
    "default_interleaved_slice_order",
    "detect_pulseq_kspace_kind",
    "import_pulseq_experiment",
    "read_pulseq",
    "read_pulseq_acquisition",
    "read_pulseq_sidecar_metadata",
    "resolve_rare_slice_order",
]


def _canonical_acquisition(values, shape, *, trajectory=None, density=None, acquired=None):
    """Give prepared Pulseq values explicit frame/receiver/encoding axes."""
    values = np.asarray(values)
    original_shape = values.shape
    ndim = 1 if trajectory is not None else len(shape)
    if values.ndim == ndim:
        values = values[None, None]
    elif values.ndim == ndim + 1:
        values = values[:, None]
    if values.ndim != ndim + 2:
        raise ValueError("Pulseq samples require explicit frame and receiver axes")
    if acquired is not None and np.shape(acquired) == original_shape:
        acquired = np.asarray(acquired).reshape(values.shape)
    return AcquisitionData(values.astype(np.result_type(values.dtype, np.complex64), copy=False),
        tuple(shape), ("partition", "phase", "read")[-len(shape):], kind="sampled" if trajectory is not None else "cartesian",
        trajectory=trajectory, density=density, acquired=acquired)


def read_pulseq_acquisition(seq_path, samples, *, spatial_shape=None, density=None,
                            sequence_type=None, rare_slice_order=None, sample_axis=-1, coil_axis=None):
    """Prepare measured samples without creating a reconstructed image."""
    info = read_pulseq(seq_path)
    shape = spatial_shape or info.spatial_shape
    if shape is None:
        raise ValueError("An encoding shape is required for raw Pulseq samples")
    sidecar = read_pulseq_sidecar_metadata(seq_path)
    acquisition, _ = _prepare_pulseq_acquisition(info, np.asarray(samples), spatial_shape=tuple(shape),
        density=density, sequence_type=sequence_type, rare_slice_order=rare_slice_order,
        sidecar_metadata=sidecar, sample_axis=sample_axis, coil_axis=coil_axis)
    return acquisition
