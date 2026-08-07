from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import jax.numpy as jnp
import numpy as np

from mrax.data_struct import DataSet, Experiment, Quantity, SpatialGeometry
from mrax.data_struct.dimensions import reshape_frames_to_dims, reshape_observables_to_dims
from mrax.library.default_experiments import build_default_experiment, filter_observables_for_type
from mrax.reco.pipeline import FFTStage, RecoPipeline
from mrax.reco.processing import estimate_radial_density

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class NormalizedMetadata:
    """Normalized acquisition metadata used for experiment typing and observables.

    Vendor-specific importers should translate scanner metadata into this schema before
    creating MRax :class:`Experiment` and :class:`DataSet` objects. The fields are limited
    to concepts the internal factory understands; raw scanner dictionaries should remain
    importer metadata and not leak into downstream model code.
    """

    method_name: str = ""
    sequence_name: str = ""
    protocol_name: str = ""
    comment: str = ""
    frame_comments: Sequence[str] = ()
    frame_count: int | None = None
    fair_labels: Sequence[int] | np.ndarray | None = None
    sat_trans_on: bool | None = None
    scan_time_ms: float | None = None
    echo_times: Sequence[float] | float | None = None
    repetition_times: Sequence[float] | float | None = None
    inversion_times: Sequence[float] | float | None = None
    flip_angles: Sequence[float] | float | None = None
    freq_offsets_ppm: Sequence[float] | float | None = None
    sat_b1_uT: Sequence[float] | float | None = None
    sat_time_ms: Sequence[float] | float | None = None
    diffusion_bvals: Sequence[float] | float | None = None
    diffusion_bvecs: Sequence[Sequence[float]] | np.ndarray | None = None


@dataclass(frozen=True)
class KSpaceImportDefinition:
    """Vendor-neutral raw k-space payload.

    ``kind`` is one of ``"cartesian"``, ``"epi"``, or ``"non_cartesian"``. Cartesian
    and EPI data use grid-shaped k-space with spatial axes last. Non-Cartesian data
    uses samples with the sample axis last and a matching ``trajectory`` array with
    columns ``(kx, ky)`` in cycles/pixel. ``sequence_type`` describes the sequence-
    specific acquisition ordering used to interpret the samples (for example
    ``"rare"`` for Turbo Spin Echo / RARE style line ordering).
    """

    samples: np.ndarray
    kind: str = "cartesian"
    sequence_type: str = "standard"
    trajectory: np.ndarray | None = None
    density: np.ndarray | None = None
    sampling_mask: np.ndarray | None = None
    spatial_shape: tuple[int, ...] | None = None
    sample_axis: int = -1
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ImportDefinition:
    """Vendor-neutral payload used to materialize MRax experiment and dataset objects."""

    name: str
    data: np.ndarray
    frame_count: int
    spatial_shape: tuple[int, ...]
    exp_type: str
    observables: Mapping[str, Quantity] = field(default_factory=dict)
    processed: Mapping[str, object] | None = None
    kspace: np.ndarray | KSpaceImportDefinition | None = None
    trajectory: np.ndarray | None = None
    density: np.ndarray | None = None
    sampling_mask: np.ndarray | None = None
    kspace_kind: str = "cartesian"
    sequence_type: str = "standard"
    kspace_metadata: Mapping[str, object] = field(default_factory=dict)
    spatial_geometry: SpatialGeometry | None = None


def build_imported_experiment_dataset(definition: ImportDefinition) -> tuple[Experiment, DataSet]:
    """Build MRax containers from a normalized import definition."""

    experiment = build_default_experiment(
        definition.exp_type,
        definition.frame_count,
        definition.spatial_shape,
        definition.observables,
    )
    dim_map = getattr(experiment, "_mrax_dim_observables", None)
    experiment.observables = reshape_observables_to_dims(
        experiment.observables,
        experiment.dims,
        dim_map,
    )

    data = reshape_frames_to_dims(jnp.asarray(definition.data), experiment.dims)
    processed = dict(definition.processed or {})
    for key, value in list(processed.items()):
        if key == "sampling_mask":
            continue
        arr = np.asarray(value)
        if arr.ndim >= 1 and int(arr.shape[0]) == int(definition.frame_count):
            processed[key] = jnp.asarray(reshape_frames_to_dims(arr, experiment.dims))

    kspace = definition.kspace
    trajectory = definition.trajectory
    density = definition.density
    sampling_mask = definition.sampling_mask
    kspace_kind = definition.kspace_kind
    sequence_type = normalize_sequence_type(definition.sequence_type)
    kspace_metadata = dict(definition.kspace_metadata or {})
    if isinstance(kspace, KSpaceImportDefinition):
        kspace_def = kspace
        kspace = kspace_def.samples
        trajectory = kspace_def.trajectory if trajectory is None else trajectory
        density = kspace_def.density if density is None else density
        sampling_mask = kspace_def.sampling_mask if sampling_mask is None else sampling_mask
        kspace_kind = kspace_def.kind
        sequence_type = normalize_sequence_type(kspace_def.sequence_type)
        if kspace_def.spatial_shape is not None:
            kspace_metadata.setdefault("spatial_shape", tuple(int(v) for v in kspace_def.spatial_shape))
        kspace_metadata.setdefault("sample_axis", int(kspace_def.sample_axis))
        kspace_metadata.update(dict(kspace_def.metadata or {}))
    if sampling_mask is None and processed and "sampling_mask" in processed:
        sampling_mask = np.asarray(processed["sampling_mask"])
    if kspace is not None:
        kspace_metadata.setdefault("kind", kspace_kind)
        kspace_metadata.setdefault("sequence_type", sequence_type)

    dataset = DataSet(
        data=jnp.asarray(data),
        name=definition.name,
        processed=processed or None,
        kspace=jnp.asarray(kspace) if kspace is not None else None,
        trajectory=jnp.asarray(trajectory) if trajectory is not None else None,
        density=jnp.asarray(density) if density is not None else None,
        sampling_mask=jnp.asarray(sampling_mask) if sampling_mask is not None else None,
        kspace_metadata=kspace_metadata or None,
        spatial_geometry=definition.spatial_geometry,
    )
    return experiment, dataset


def _has_variation(vals) -> bool:
    vals_list = [v for v in _as_list(vals) if v is not None]
    return len(vals_list) > 1 and len(set(vals_list)) > 1


def _as_list(val) -> list:
    if val is None:
        return []
    if isinstance(val, (list, tuple)):
        return list(val)
    if isinstance(val, np.ndarray):
        return list(val.ravel())
    return [val]


def _has_value(val) -> bool:
    if val is None:
        return False
    if isinstance(val, (list, tuple)):
        return len(val) > 0
    if isinstance(val, np.ndarray):
        return val.size > 0
    return True


def _has_text(*values: object) -> bool:
    return any(isinstance(val, str) and val.strip() for val in values)


def _contains_keyword_token(text: str, token: str) -> bool:
    pattern = rf"(^|[\s_/\-]){re.escape(token)}($|[\s_/\-])"
    return re.search(pattern, str(text)) is not None


def _first_finite_scalar(val) -> float | None:
    if val is None:
        return None
    arr = np.asarray(_as_list(val), dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(arr[0])


def _trim_to_frames(val, frame_count: int):
    if isinstance(val, (list, tuple)):
        if len(val) > frame_count:
            return list(val)[-frame_count:]
        return val
    if isinstance(val, np.ndarray) and val.ndim == 1 and val.size > frame_count:
        return val[-frame_count:].tolist()
    return val


def _parse_offsets_from_comments(comments: Sequence[str]) -> list[float]:
    offsets: list[float] = []
    for entry in comments:
        try:
            match = re.search(r"([-+0-9.]+)", str(entry))
            if match:
                offsets.append(float(match.group(1)))
        except Exception:
            continue
    return offsets


def _coerce_bvecs(values: object, frame_count: int) -> np.ndarray | None:
    if values is None:
        return None
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        return None
    if arr.ndim == 1:
        if arr.size % 3 == 0:
            arr = arr.reshape((-1, 3))
        elif arr.size == 3 and frame_count > 1:
            arr = np.tile(arr.reshape((1, 3)), (frame_count, 1))
        else:
            return None
    if arr.ndim == 2 and arr.shape[0] == 3 and arr.shape[1] != 3:
        arr = arr.T
    if arr.ndim != 2 or arr.shape[1] != 3:
        return None
    if arr.shape[0] >= frame_count:
        arr = arr[-frame_count:]
    elif arr.shape[0] == 1 and frame_count > 1:
        arr = np.tile(arr, (frame_count, 1))
    else:
        return None
    return arr


def _infer_t1_mapping_type(meta: NormalizedMetadata, combined: str) -> str | None:
    """Infer a T1 mapping subtype from the strongest available evidence."""

    flip_list = _as_list(meta.flip_angles)
    tr_list = _as_list(meta.repetition_times)
    ti_list = _as_list(meta.inversion_times)

    has_t1map = "t1map" in combined
    has_ir_keyword = any(k in combined for k in ("invrec", "inversion_recovery", "ir_"))
    has_vtr_keyword = any(k in combined for k in ("vtr", "variable tr"))
    has_vfa_keyword = any(k in combined for k in ("vfa", "variable flip", "flipangle"))
    has_var_tr = _has_variation(tr_list)
    has_var_flip = _has_variation(flip_list)
    has_var_ti = _has_variation(ti_list)
    has_ti = _has_value(meta.inversion_times)

    # Prefer explicit subtype evidence over generic "t1map" protocol labels.
    if has_vtr_keyword and has_var_tr:
        return "vtr"
    if has_vfa_keyword and has_var_flip:
        return "vfa"
    if has_ir_keyword and (has_ti or has_t1map):
        return "ir"

    if not has_t1map:
        return None

    if has_var_ti:
        return "ir"
    if has_var_tr and not has_var_flip:
        return "vtr"
    if has_var_flip and not has_var_tr:
        return "vfa"
    return None


def needs_manual_type_override(meta: NormalizedMetadata, exp_type: str) -> bool:
    """Return True when fallback typing should require an override before fitting."""

    if exp_type != "anatomical":
        return False

    frame_count = int(meta.frame_count or 0)
    has_quantitative_hints = any(
        (
            _has_variation(meta.echo_times),
            _has_variation(meta.repetition_times),
            _has_variation(meta.inversion_times),
            _has_variation(meta.flip_angles),
            _has_value(meta.freq_offsets_ppm),
            _has_value(meta.diffusion_bvals),
            _has_value(meta.diffusion_bvecs),
            bool(meta.frame_comments),
        )
    )
    has_named_sequence = _has_text(meta.method_name, meta.sequence_name, meta.protocol_name, meta.comment)
    return has_quantitative_hints or (frame_count > 1 and has_named_sequence)


def detect_experiment_type(meta: NormalizedMetadata) -> str:
    """Infer experiment type from normalized acquisition metadata."""

    combined = " ".join(
        s for s in (meta.method_name, meta.sequence_name, meta.comment, meta.protocol_name) if s
    ).lower()
    combined_no_comment = " ".join(
        s for s in (meta.method_name, meta.sequence_name, meta.protocol_name) if s
    ).lower()
    sat_on = bool(meta.sat_trans_on) if meta.sat_trans_on is not None else False
    frame_comment = " ".join(meta.frame_comments).lower()

    if any(k in combined for k in ("spectro", "spectroscopy", "mrs", "press", "steam")) or _contains_keyword_token(
        combined, "fid"
    ):
        return "spectroscopy"

    if (
        sat_on
        or any(k in combined for k in ("sat_trans", "sattrans", "cesl", "cest", "wasabi", "wobble"))
        or "cest" in frame_comment
        or "sat" in frame_comment
    ):
        return "sat_trans"

    if _has_value(meta.diffusion_bvals) or _has_value(meta.diffusion_bvecs):
        return "dti"
    if any(k in combined_no_comment for k in ("dti", "dwi", "diffusion", "dw_epi", "dw-")) or any(
        k in frame_comment for k in ("dti", "dwi", "diffusion")
    ):
        return "dti"

    flip_list = _as_list(meta.flip_angles)
    tr_list = _as_list(meta.repetition_times)

    if _has_value(meta.fair_labels):
        return "fair_asl"
    t1_type = _infer_t1_mapping_type(meta, combined)
    if t1_type is not None:
        return t1_type
    if "fair" in combined:
        return "fair_asl"
    if any(k in combined for k in ("vtr", "variable tr")) and _has_variation(tr_list):
        return "vtr"
    if any(k in combined for k in ("vfa", "variable flip", "flipangle", "flash")) and _has_variation(flip_list):
        return "vfa"

    scan_time_ms = _first_finite_scalar(meta.scan_time_ms)
    frame_count = int(meta.frame_count or 0)
    looks_flash_like = any(k in combined_no_comment for k in ("flash", "fisp", "gre", "gradient_echo"))
    if (
        looks_flash_like
        and frame_count >= 100
        and scan_time_ms is not None
        and scan_time_ms >= 60_000.0
    ):
        return "dce"

    if any(k in combined for k in ("msme", "multi spin", "spin_echo")) or _contains_keyword_token(combined, "mse"):
        return "mse"
    if any(k in combined for k in ("t2star", "t2*", "multi gradient", "ge3d", "gradient_echo")) or _contains_keyword_token(combined, "mge"):
        return "mge"

    return "anatomical"


def extract_observables_from_metadata(
    exp_type: str, meta: NormalizedMetadata, frame_count: int
) -> dict[str, Quantity]:
    """Collect common observables (TE/TR/TI/flip/offsets) from normalized metadata."""

    obs: dict[str, Quantity] = {}

    te = _trim_to_frames(meta.echo_times, frame_count)
    if isinstance(te, (list, tuple)) and te:
        te_arr = jnp.asarray(te, dtype=jnp.float32)
        if te_arr.ndim > 0 and te_arr.shape[0] == frame_count + 1:
            te_arr = te_arr[1:]
        obs["TE"] = Quantity(te_arr, "ms")

    tr_val = meta.repetition_times
    if tr_val is not None:
        tr_arr = jnp.asarray(tr_val if isinstance(tr_val, (list, tuple)) else [tr_val], dtype=jnp.float32)
        if tr_arr.ndim == 0:
            tr_arr = tr_arr[None]
        obs["TR"] = Quantity(tr_arr, "ms")

    ti = _trim_to_frames(meta.inversion_times, frame_count)
    if isinstance(ti, (list, tuple)) and ti:
        ti_arr = jnp.asarray(ti, dtype=jnp.float32)
        if ti_arr.ndim > 0 and ti_arr.shape[0] == frame_count + 1:
            ti_arr = ti_arr[1:]
        obs["TI"] = Quantity(ti_arr, "ms")

    fa = _trim_to_frames(meta.flip_angles, frame_count)
    if _has_value(fa):
        fa_arr = jnp.asarray(fa, dtype=jnp.float32)
        if fa_arr.ndim == 0:
            fa_arr = fa_arr[None]
        obs["flip_angle"] = Quantity(fa_arr, "deg")

    if exp_type == "dce" and meta.scan_time_ms is not None:
        obs["scan_time"] = Quantity(jnp.asarray(float(meta.scan_time_ms), dtype=jnp.float32), "ms")

    if exp_type == "fair_asl":
        fair_labels = _trim_to_frames(meta.fair_labels, frame_count)
        if _has_value(fair_labels):
            obs["fair_label"] = Quantity(jnp.asarray(fair_labels, dtype=jnp.float32), "1")

    if exp_type == "sat_trans":
        offsets = meta.freq_offsets_ppm
        if not _has_value(offsets):
            offsets = _parse_offsets_from_comments(meta.frame_comments)
        offsets = _trim_to_frames(offsets, frame_count)
        if _has_value(offsets):
            obs["freq_offsets"] = Quantity(jnp.asarray(offsets, dtype=jnp.float32), "ppm")

        if _has_value(meta.sat_b1_uT):
            sat_b1 = _trim_to_frames(meta.sat_b1_uT, frame_count)
            obs["sat_b1"] = Quantity(jnp.asarray(sat_b1, dtype=jnp.float32), "uT")

        if _has_value(meta.sat_time_ms):
            sat_time = _trim_to_frames(meta.sat_time_ms, frame_count)
            obs["sat_time"] = Quantity(jnp.asarray(sat_time, dtype=jnp.float32), "ms")

    if exp_type in ("dti", "diffusion"):
        bvals = _trim_to_frames(meta.diffusion_bvals, frame_count)
        if _has_value(bvals):
            obs["bvals"] = Quantity(jnp.asarray(bvals, dtype=jnp.float32), "s/mm^2")
        bvecs = _coerce_bvecs(meta.diffusion_bvecs, frame_count)
        if bvecs is not None:
            obs["bvecs"] = Quantity(jnp.asarray(bvecs, dtype=jnp.float32), "1")

    return filter_observables_for_type(exp_type, obs)


def _normalize_kspace_kind(kind: str) -> str:
    normalized = str(kind or "cartesian").lower().replace("-", "_")
    aliases = {
        "noncartesian": "non_cartesian",
        "non_cart": "non_cartesian",
        "nufft": "non_cartesian",
        "radial": "non_cartesian",
        "spiral": "non_cartesian",
        "echo_planar": "epi",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in ("cartesian", "epi", "non_cartesian"):
        raise ValueError("k-space kind must be one of: cartesian, epi, non_cartesian")
    return normalized


def normalize_sequence_type(sequence_type: str | None) -> str:
    """Return a normalized sequence-ordering label for importer-specific logic."""

    normalized = str(sequence_type or "standard").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "": "standard",
        "default": "standard",
        "generic": "standard",
        "cartesian": "standard",
        "gre": "standard",
        "flash": "standard",
        "tse": "rare",
        "fse": "rare",
        "turbose": "rare",
        "turbo_se": "rare",
        "turbo_spin_echo": "rare",
        "fast_spin_echo": "rare",
        "turbo_spin_echo_rare": "rare",
    }
    return aliases.get(normalized, normalized)


def sampling_mask_from_kspace(kspace: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """Return a boolean mask marking acquired Cartesian/EPI k-space samples."""

    return np.abs(np.asarray(kspace)) > float(threshold)


def reshape_epi_samples(
    samples: np.ndarray,
    *,
    samples_per_line: int,
    line_count: int | None = None,
    reverse_odd_lines: bool = True,
) -> np.ndarray:
    """Reshape serial EPI ADC samples into ``(..., ky, kx)`` order.

    Pulseq and vendor raw streams commonly store EPI data in acquisition order:
    line 0 left-to-right, line 1 right-to-left, and so on. Setting
    ``reverse_odd_lines=True`` reverses the readout direction on odd ky lines so the
    returned grid has a consistent kx ordering.
    """

    arr = np.asarray(samples)
    n_read = int(samples_per_line)
    if n_read <= 0:
        raise ValueError("samples_per_line must be positive")
    total = int(arr.shape[-1])
    if line_count is None:
        if total % n_read != 0:
            raise ValueError("serial EPI sample count is not divisible by samples_per_line")
        line_count = total // n_read
    n_lines = int(line_count)
    if n_lines <= 0:
        raise ValueError("line_count must be positive")
    expected = n_lines * n_read
    if total != expected:
        raise ValueError(f"Expected {expected} EPI samples, found {total}")
    grid = arr.reshape(arr.shape[:-1] + (n_lines, n_read))
    if reverse_odd_lines and n_lines > 1:
        grid = np.array(grid, copy=True)
        grid[..., 1::2, :] = grid[..., 1::2, ::-1]
    return grid


def _split_trajectory_lines(trajectory: np.ndarray, adc_counts: Sequence[int]) -> list[np.ndarray]:
    counts = np.asarray(adc_counts, dtype=int).ravel()
    if counts.size == 0 or np.any(counts <= 0):
        raise ValueError("adc_counts must contain positive line lengths")
    traj = np.asarray(trajectory, dtype=np.float32)
    if traj.ndim != 2 or traj.shape[1] < 2:
        raise ValueError("trajectory must have shape (num_samples, 2)")
    if int(traj.shape[0]) != int(np.sum(counts)):
        raise ValueError("trajectory sample count does not match adc_counts")

    lines: list[np.ndarray] = []
    pos = 0
    for count in counts.tolist():
        nxt = pos + int(count)
        lines.append(traj[pos:nxt, :2])
        pos = nxt
    return lines


def reshape_rare_samples(
    samples: np.ndarray,
    *,
    trajectory: np.ndarray,
    adc_counts: Sequence[int],
    spatial_shape: tuple[int, int],
    frame_order: Sequence[int] | None = None,
    reverse_negative_readout: bool = True,
) -> np.ndarray:
    """Reshape serial RARE/TurboSE samples into Cartesian ``(..., ky, kx)`` order.

    RARE acquisitions often acquire k-space lines in an echo-train order that does not
    match the final ky ordering. This helper uses the ADC trajectory to assign each
    acquired line to its Cartesian ky index, while preserving repeated occurrences as
    separate frames (for example interleaved slices or repetitions).
    """

    arr = np.asarray(samples)
    if arr.ndim == 0:
        raise ValueError("samples must be at least 1D")

    ny, nx = (int(spatial_shape[0]), int(spatial_shape[1]))
    if ny <= 0 or nx <= 0:
        raise ValueError("spatial_shape must contain positive (ky, kx) sizes")

    line_lengths = np.asarray(adc_counts, dtype=int).ravel()
    if line_lengths.size == 0 or np.any(line_lengths <= 0):
        raise ValueError("adc_counts must contain positive line lengths")
    if not np.all(line_lengths == line_lengths[0]):
        raise ValueError("RARE reshaping currently requires a constant ADC sample count per line")

    samples_per_line = int(line_lengths[0])
    if samples_per_line != nx:
        raise ValueError(
            f"RARE readout length {samples_per_line} does not match spatial_shape readout {nx}"
        )

    total_samples = int(np.sum(line_lengths))
    if int(arr.shape[-1]) != total_samples:
        raise ValueError(
            f"Expected {total_samples} serial RARE samples on the last axis, found {arr.shape[-1]}"
        )

    lines = arr.reshape(arr.shape[:-1] + (line_lengths.size, samples_per_line))
    traj_lines = _split_trajectory_lines(trajectory, line_lengths)
    ky_values = np.round(
        np.asarray([float(np.median(line[:, 1])) for line in traj_lines], dtype=np.float64),
        decimals=6,
    )
    ky_unique = np.unique(ky_values)
    ky_unique.sort()
    if ky_unique.size == 0:
        raise ValueError("Could not infer ky ordering from the RARE trajectory")
    if ky_unique.size > ny:
        raise ValueError(f"Trajectory contains {ky_unique.size} unique ky lines but spatial_shape expects {ny}")

    ky_offset = (ny - int(ky_unique.size)) // 2
    seen_per_ky = np.zeros((int(ky_unique.size),), dtype=int)
    frame_count = 0
    for ky in ky_values.tolist():
        ky_idx = int(np.argmin(np.abs(ky_unique - ky)))
        seen_per_ky[ky_idx] += 1
        frame_count = max(frame_count, int(seen_per_ky[ky_idx]))

    if frame_order is None:
        frame_order_arr = np.arange(frame_count, dtype=int)
    else:
        frame_order_arr = np.asarray(frame_order, dtype=int).ravel()
        if frame_order_arr.size < frame_count:
            raise ValueError("frame_order must include one destination frame for each repeated ky occurrence")
        expected = np.arange(int(frame_order_arr.size), dtype=int)
        if np.any(np.sort(frame_order_arr) != expected):
            raise ValueError("frame_order must be a permutation of contiguous frame indices starting at 0")

    output_frame_count = int(frame_order_arr.size)
    output = np.zeros((output_frame_count,) + arr.shape[:-1] + (ny, nx), dtype=arr.dtype)
    frame_occurrence = np.zeros((int(ky_unique.size),), dtype=int)
    for line_idx, ky in enumerate(ky_values.tolist()):
        ky_idx = int(np.argmin(np.abs(ky_unique - ky)))
        acq_frame_idx = int(frame_occurrence[ky_idx])
        frame_occurrence[ky_idx] += 1
        frame_idx = int(frame_order_arr[acq_frame_idx])
        dst_ky = ky_idx + ky_offset
        line = lines[..., line_idx, :]
        if reverse_negative_readout and traj_lines[line_idx].shape[0] >= 2:
            if float(traj_lines[line_idx][-1, 0] - traj_lines[line_idx][0, 0]) < 0.0:
                line = line[..., ::-1]
        output[frame_idx, ..., dst_ky, :] = line

    return output[0] if frame_count == 1 else output


def build_kspace_definition(
    kspace: np.ndarray,
    *,
    kind: str = "cartesian",
    sequence_type: str = "standard",
    trajectory: np.ndarray | None = None,
    density: np.ndarray | None = None,
    sampling_mask: np.ndarray | None = None,
    spatial_shape: tuple[int, ...] | None = None,
    sample_axis: int = -1,
    metadata: Mapping[str, object] | None = None,
) -> KSpaceImportDefinition:
    """Validate and describe raw k-space for an importer payload."""

    kspace_kind = _normalize_kspace_kind(kind)
    normalized_sequence_type = normalize_sequence_type(sequence_type)
    samples = np.asarray(kspace)
    sample_axis = int(sample_axis)
    if sample_axis < 0:
        sample_axis += samples.ndim
    if sample_axis < 0 or sample_axis >= samples.ndim:
        raise ValueError("sample_axis is out of range for k-space samples")

    traj = None if trajectory is None else np.asarray(trajectory, dtype=np.float32)
    dens = None if density is None else np.asarray(density, dtype=np.float32)
    mask = None if sampling_mask is None else np.asarray(sampling_mask, dtype=bool)

    if kspace_kind == "non_cartesian":
        if traj is None:
            raise ValueError("non-Cartesian k-space requires a trajectory")
        if traj.ndim != 2 or traj.shape[1] not in (2, 3):
            raise ValueError("trajectory must have shape (num_samples, 2) or (num_samples, 3)")
        if spatial_shape is not None and len(spatial_shape) != int(traj.shape[1]):
            raise ValueError("spatial_shape dimensionality must match the non-Cartesian trajectory")
        if sample_axis != samples.ndim - 1:
            samples = np.moveaxis(samples, sample_axis, -1)
            sample_axis = samples.ndim - 1
        sample_count = int(samples.shape[sample_axis])
        if sample_count != int(traj.shape[0]):
            raise ValueError(
                f"k-space sample axis has {sample_count} samples but trajectory has {traj.shape[0]}"
            )
        if dens is None:
            dens = estimate_radial_density(
                traj,
                spatial_shape=spatial_shape
                if spatial_shape is not None and len(spatial_shape) == int(traj.shape[1])
                else None,
            )
        elif dens.shape != (sample_count,):
            raise ValueError("density must have shape (num_samples,)")
    elif traj is not None:
        raise ValueError("trajectory is only valid for non-Cartesian k-space definitions")

    if mask is None and kspace_kind in ("cartesian", "epi"):
        mask = sampling_mask_from_kspace(samples)

    return KSpaceImportDefinition(
        samples=samples,
        kind=kspace_kind,
        sequence_type=normalized_sequence_type,
        trajectory=traj,
        density=dens,
        sampling_mask=mask,
        spatial_shape=spatial_shape,
        sample_axis=sample_axis,
        metadata=dict(metadata or {}),
    )


def build_cartesian_kspace_definition(
    kspace: np.ndarray,
    *,
    sequence_type: str = "standard",
    sampling_mask: np.ndarray | None = None,
    spatial_shape: tuple[int, ...] | None = None,
    metadata: Mapping[str, object] | None = None,
) -> KSpaceImportDefinition:
    """Create a Cartesian k-space definition."""

    return build_kspace_definition(
        kspace,
        kind="cartesian",
        sequence_type=sequence_type,
        sampling_mask=sampling_mask,
        spatial_shape=spatial_shape,
        metadata=metadata,
    )


def build_epi_kspace_definition(
    kspace: np.ndarray,
    *,
    samples_per_line: int | None = None,
    line_count: int | None = None,
    reverse_odd_lines: bool = True,
    sequence_type: str = "standard",
    sampling_mask: np.ndarray | None = None,
    spatial_shape: tuple[int, ...] | None = None,
    metadata: Mapping[str, object] | None = None,
) -> KSpaceImportDefinition:
    """Create an EPI k-space definition, optionally reshaping serial ADC samples."""

    samples = np.asarray(kspace)
    meta = dict(metadata or {})
    if samples_per_line is not None:
        samples = reshape_epi_samples(
            samples,
            samples_per_line=int(samples_per_line),
            line_count=line_count,
            reverse_odd_lines=reverse_odd_lines,
        )
        meta.update(
            {
                "samples_per_line": int(samples_per_line),
                "line_count": int(samples.shape[-2]),
                "reverse_odd_lines": bool(reverse_odd_lines),
            }
        )
    return build_kspace_definition(
        samples,
        kind="epi",
        sequence_type=sequence_type,
        sampling_mask=sampling_mask,
        spatial_shape=spatial_shape,
        metadata=meta,
    )


def build_rare_kspace_definition(
    samples: np.ndarray,
    *,
    trajectory: np.ndarray,
    adc_counts: Sequence[int],
    sampling_mask: np.ndarray | None = None,
    spatial_shape: tuple[int, int],
    frame_order: Sequence[int] | None = None,
    metadata: Mapping[str, object] | None = None,
) -> KSpaceImportDefinition:
    """Create a Cartesian k-space definition for RARE/TurboSE acquisitions."""

    reshaped = reshape_rare_samples(
        samples,
        trajectory=trajectory,
        adc_counts=adc_counts,
        spatial_shape=spatial_shape,
        frame_order=frame_order,
    )
    return build_kspace_definition(
        reshaped,
        kind="cartesian",
        sequence_type="rare",
        sampling_mask=sampling_mask,
        spatial_shape=spatial_shape,
        metadata=metadata,
    )


def build_non_cartesian_kspace_definition(
    samples: np.ndarray,
    *,
    trajectory: np.ndarray,
    density: np.ndarray | None = None,
    spatial_shape: tuple[int, ...] | None = None,
    sample_axis: int = -1,
    sequence_type: str = "standard",
    metadata: Mapping[str, object] | None = None,
) -> KSpaceImportDefinition:
    """Create a non-Cartesian k-space definition with trajectory and density."""

    return build_kspace_definition(
        samples,
        kind="non_cartesian",
        sequence_type=sequence_type,
        trajectory=trajectory,
        density=density,
        spatial_shape=spatial_shape,
        sample_axis=sample_axis,
        metadata=metadata,
    )


def apply_encoding_steps(
    kspace: np.ndarray,
    enc_steps: Sequence[int],
    axis: int = -2,
    *,
    target_size: int | None = None,
) -> np.ndarray:
    """Reorder k-space lines according to acquisition encoding steps."""

    steps = np.asarray(enc_steps, dtype=int).ravel()
    if steps.size == 0:
        return kspace

    axis = axis if axis >= 0 else kspace.ndim + axis

    desired = int(target_size) if target_size is not None else int(kspace.shape[axis])
    desired = max(desired, 1)

    shape = list(kspace.shape)
    shape[axis] = desired
    filled = np.zeros(shape, dtype=kspace.dtype)

    s_min = int(steps.min())
    s_max = int(steps.max())
    if s_min >= 0 and s_max < desired:
        offset = 0
    elif s_min >= -(desired // 2) and s_max <= (desired // 2) - 1:
        offset = desired // 2
    else:
        offset = -s_min

    slicer_src = [slice(None)] * kspace.ndim
    slicer_dst = [slice(None)] * kspace.ndim
    for src_idx, ky in enumerate(steps.tolist()):
        dst = int(ky) + offset
        if 0 <= dst < desired:
            slicer_src[axis] = src_idx
            slicer_dst[axis] = dst
            filled[tuple(slicer_dst)] = kspace[tuple(slicer_src)]
    return filled


def reshape_raw_kspace(
    samples: np.ndarray,
    *,
    frame_count: int,
    spatial_shape: tuple[int, ...],
    coil_count: int | None = None,
    is_multi_echo: bool = False,
    enc_steps: Sequence[int] | None = None,
    target_ky: int | None = None,
) -> np.ndarray:
    """Reshape serial complex samples into k-space, applying optional ordering hints."""

    base_shape = tuple(int(s) for s in spatial_shape)
    if not base_shape:
        raise ValueError("Cannot reshape raw k-space samples without a spatial shape.")
    total_points = int(np.prod(base_shape))
    if total_points == 0:
        raise ValueError("Invalid spatial shape for raw k-space reconstruction.")
    repeats = samples.size // total_points
    if repeats == 0:
        raise ValueError(
            f"Expected at least {total_points} complex samples for one frame, found {samples.size}."
        )

    inferred_coils = 1
    if repeats % frame_count == 0:
        inferred_coils = repeats // frame_count
    if coil_count is None:
        coil_count = inferred_coils
    if coil_count <= 0 or coil_count * frame_count != repeats:
        coil_count = inferred_coils

    frames = frame_count if coil_count == repeats // frame_count else repeats
    usable = samples[: frames * coil_count * total_points]

    def _frame_smooth(arr: np.ndarray) -> float:
        if arr.shape[0] <= 1:
            return 0.0
        mag = np.abs(arr)
        return float(np.mean(np.abs(np.diff(mag, axis=0))))

    def _ky_smooth(arr: np.ndarray) -> float:
        mag = np.abs(arr)
        return float(np.mean(np.abs(np.diff(mag, axis=2))))

    def _score(arr: np.ndarray) -> float:
        fr = _frame_smooth(arr)
        ky = _ky_smooth(arr)
        return ky * (fr if fr > 0 else 1.0)

    if len(base_shape) == 2:
        ky, kx = base_shape
    else:
        ky = kx = None

    enc_steps_arr = None
    if enc_steps is not None:
        try:
            enc_steps_arr = np.asarray(enc_steps, dtype=int).ravel()
            if enc_steps_arr.size == 0:
                enc_steps_arr = None
        except Exception:
            enc_steps_arr = None

    candidates: list[np.ndarray] = []
    if coil_count > 1:
        block_major = usable.reshape((frames, coil_count) + base_shape)
        candidates.append(block_major)
        if len(base_shape) == 2:
            line_major = usable.reshape((frames, ky, coil_count, kx))
            line_major = np.moveaxis(line_major, 2, 1)
            candidates.append(line_major)
            if is_multi_echo:
                ky_major = np.transpose(usable.reshape((ky, frames, coil_count, kx)), (1, 2, 0, 3))
                ky_major_alt = np.transpose(usable.reshape((ky, coil_count, frames, kx)), (2, 1, 0, 3))
                candidates.extend([ky_major, ky_major_alt])
    else:
        frame_major = usable.reshape((frames,) + base_shape)
        if frame_major.ndim == 3:
            candidates.append(frame_major[:, None, :, :])
        else:
            candidates.append(frame_major[:, None] + np.zeros((frames, 1) + base_shape, dtype=frame_major.dtype))
        if is_multi_echo and len(base_shape) == 2 and ky is not None and kx is not None:
            ky_major = np.transpose(usable.reshape((ky, frames, kx)), (1, 0, 2))
            candidates.append(ky_major[:, None, :, :])

    def _score_candidate(arr: np.ndarray) -> float:
        scored = arr
        if enc_steps_arr is not None:
            try:
                scored = apply_encoding_steps(scored, enc_steps_arr, axis=-2, target_size=target_ky)
            except Exception:
                scored = arr
        if is_multi_echo and len(candidates) > 1:
            return _score(scored)
        return _ky_smooth(scored)

    if not candidates:
        kspace = (
            usable.reshape((frames, coil_count) + base_shape)
            if coil_count > 1
            else usable.reshape((frames,) + base_shape)
        )
    else:
        if len(candidates) > 1:
            kspace = min(candidates, key=_score_candidate)
        else:
            kspace = candidates[0]

    if coil_count == 1 and kspace.ndim == 4:
        kspace = kspace[:, 0, ...]

    if enc_steps_arr is not None:
        try:
            kspace = apply_encoding_steps(kspace, enc_steps_arr, axis=-2, target_size=target_ky)
        except Exception:
            _LOG.debug("Ignored optional operation failure", exc_info=True)
    return kspace


def default_reco_pipeline(mask: np.ndarray | None) -> RecoPipeline:
    """Default reconstruction chain for Cartesian importers."""

    return RecoPipeline([FFTStage(direction="inverse", axes=(-2, -1))])


__all__ = [
    "ImportDefinition",
    "KSpaceImportDefinition",
    "NormalizedMetadata",
    "apply_encoding_steps",
    "build_cartesian_kspace_definition",
    "build_epi_kspace_definition",
    "build_imported_experiment_dataset",
    "build_kspace_definition",
    "build_non_cartesian_kspace_definition",
    "build_rare_kspace_definition",
    "default_reco_pipeline",
    "detect_experiment_type",
    "extract_observables_from_metadata",
    "needs_manual_type_override",
    "normalize_sequence_type",
    "reshape_epi_samples",
    "reshape_rare_samples",
    "reshape_raw_kspace",
    "sampling_mask_from_kspace",
]
