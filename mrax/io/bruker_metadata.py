from __future__ import annotations

import logging
import re
import warnings
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple

import jax.numpy as jnp
import numpy as np

from mrax.data_struct import Experiment, Model, Quantity
from mrax.data_struct.quantity import require_quantity_dict
from mrax.io.import_definitions import (
    NormalizedMetadata,
    detect_experiment_type,
    extract_observables_from_metadata,
    normalize_sequence_type,
)

from .bruker_images import _bruker_orientation
from .bruker_parameters import _flatten_numeric

RecipeResult = Tuple[str | None, dict | None, Model | Callable[[], Model] | None]
_LOG = logging.getLogger(__name__)
ExperimentRecipe = Callable[[dict, dict, int], str | RecipeResult]
PDataIndex = str | int | Dict[str | int, str | int]
DEFAULT_PDATA_INDEX = "1"

def _bruker_non_imaging_role(
    *,
    exp_type: str | None,
    sequence_type: str | None,
    spatial_shape: Sequence[int],
    method: dict,
    visu: dict,
) -> str | None:
    if str(exp_type or "").strip().lower() in {"spectroscopy", "mrs", "fid"}:
        return "spectroscopy"
    if str(sequence_type or "").strip().lower() in {"spectroscopy", "mrs", "fid"}:
        return "spectroscopy"
    tokens = [
        method.get("Method"),
        method.get("PVM_Method"),
        method.get("ACQ_scan_name"),
        visu.get("VisuAcqSequenceName"),
        visu.get("VisuAcquisitionProtocol"),
        visu.get("VisuSeriesExperimentComment"),
    ]
    text = " ".join(str(token or "") for token in tokens).lower()
    if any(keyword in text for keyword in ("spectro", "spectroscopy", "mrs", "press", "steam")) or re.search(
        r"(?<![a-z0-9])fid(?![a-z0-9])", text
    ):
        return "spectroscopy"
    if len(tuple(spatial_shape)) < 2:
        return "non_imaging"
    return None


def _resolve_pdata_index(pdata_index: PDataIndex, scan_no: str) -> str:
    """Resolve a pdata index for a given scan, supporting per-scan overrides."""
    if isinstance(pdata_index, dict):
        if scan_no in pdata_index:
            return str(pdata_index[scan_no])
        try:
            scan_int = int(scan_no)
            if scan_int in pdata_index:
                return str(pdata_index[scan_int])
        except ValueError:
            _LOG.debug("Ignored optional operation failure", exc_info=True)
        if "default" in pdata_index:
            return str(pdata_index["default"])
        return DEFAULT_PDATA_INDEX
    return str(pdata_index)


def _normalize_study_globals(
    study_globals: Optional[Dict[str, Quantity]] = None,
    field_strength_mhz: float | Quantity | None = None,
) -> Dict[str, Quantity]:
    """Coerce study-level globals and ensure values are Quantities."""
    globals_map: Dict[str, Quantity] = {}
    if study_globals:
        globals_map.update(study_globals)
    if field_strength_mhz is not None:
        if isinstance(field_strength_mhz, Quantity):
            globals_map["field_strength_mhz"] = field_strength_mhz
        else:
            globals_map["field_strength_mhz"] = Quantity(
                jnp.asarray(float(field_strength_mhz)),
                "MHz",
            )
    if globals_map:
        globals_map = require_quantity_dict(globals_map, "Study global")
    return globals_map


def _normalize_fair_mode_text(mode: object) -> str:
    if mode is None:
        return ""
    text = str(mode).upper()
    return re.sub(r"[_\\s-]+", "", text)


def _fair_mode_interleaved(mode: object) -> bool:
    text = _normalize_fair_mode_text(mode)
    return "INTERLEAVED" in text


def _fair_mode_single_map(mode: object) -> bool:
    text = _normalize_fair_mode_text(mode)
    if not text or "INTERLEAVED" in text:
        return False
    return "NONSELECTIVE" in text or "SELECTIVE" in text


def _override_fair_exp_type(exp_type: str, fair_mode: object) -> str:
    if _fair_mode_interleaved(fair_mode):
        return "fair_asl"
    if _fair_mode_single_map(fair_mode):
        return "ir"
    return exp_type


def _infer_bruker_fair_labels(fair_mode: object, frame_count: int) -> np.ndarray | None:
    """Infer FAIR label/control frame labels from Bruker ``PVM_FairMode``."""

    if fair_mode is None or frame_count <= 0:
        return None
    mode = str(fair_mode).upper()
    if "INTERLEAVED2" in mode or ("INTERLEAVED" in mode and "2" in mode):
        start = 0
        if "NON" in mode and "FIRST" in mode:
            start = 1
        if "SEL" in mode and "FIRST" in mode and "NON" not in mode:
            start = 0
        return (np.arange(frame_count, dtype=np.int32) + start) % 2
    if "INTERLEAVED" in mode:
        start = 0
        if "NON" in mode and "FIRST" in mode:
            start = 1
        if "SEL" in mode and "FIRST" in mode and "NON" not in mode:
            start = 0
        return (np.arange(frame_count, dtype=np.int32) + start) % 2
    if "NON" in mode and "SEL" in mode:
        return np.ones(frame_count, dtype=np.int32)
    if "SEL" in mode:
        return np.zeros(frame_count, dtype=np.int32)
    if "NON" in mode:
        return np.ones(frame_count, dtype=np.int32)
    return None


def _maybe_reorder_fair_frames(
    data: np.ndarray,
    observables: dict,
    *,
    fair_mode: object,
) -> tuple[np.ndarray, dict, np.ndarray | None]:
    obs = dict(observables)
    fair_label = obs.get("fair_label")
    ti_obs = obs.get("TI")
    if fair_label is None or ti_obs is None:
        return data, obs, None
    labels = np.asarray(fair_label.values)
    if labels.ndim != 1:
        return data, obs, None
    frame_count = int(labels.shape[0])
    ti_vals = np.asarray(ti_obs.values)
    if ti_vals.ndim == 0:
        return data, obs, None
    if ti_vals.ndim != 1:
        return data, obs, None
    if ti_vals.shape[0] == frame_count:
        ti_frame = ti_vals
    elif frame_count % int(ti_vals.shape[0]) == 0:
        reps = frame_count // int(ti_vals.shape[0])
        if _fair_mode_interleaved(fair_mode):
            ti_frame = np.repeat(ti_vals, reps)
        else:
            ti_frame = np.tile(ti_vals, reps)
    else:
        return data, obs, None
    order = np.lexsort((labels, ti_frame))
    if order.ndim != 1 or order.size != frame_count or np.all(order == np.arange(frame_count)):
        return data, obs, None
    data_reordered = np.asarray(data)[order]
    for key, qty in list(obs.items()):
        arr = np.asarray(qty.values)
        if arr.ndim >= 1 and int(arr.shape[0]) == frame_count:
            obs[key] = qty.with_values(arr[order])
    return data_reordered, obs, order


def _sat_trans_enabled(method: dict) -> bool:
    sat_flag = method.get("PVM_SatTransOnOff")
    sat_token = sat_flag[0] if isinstance(sat_flag, list) and sat_flag else sat_flag
    return str(sat_token or "").lower() not in ("", "off", "no", "0", "false", "none")


def _bruker_spin_lock(method: dict) -> bool | None:
    """Parse the explicit values of an optional static spin-lock flag."""
    value = method.get("SpinLock")
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.size == 0 or not np.all(arr == arr.reshape(-1)[0]):
        raise ValueError("SpinLock must have one Boolean value")
    flag = arr.reshape(-1)[0]
    if isinstance(flag, (str, np.str_)):
        token = str(flag).strip().strip("<>").lower()
        if token in ("on", "yes", "true", "1", "1.0"):
            return True
        if token in ("off", "no", "false", "0", "0.0"):
            return False
    elif flag == 0 or flag == 1:
        return bool(flag)
    raise ValueError("SpinLock must have one Boolean value")


def _bruker_sat_time_ms(method: dict) -> object | None:
    pulse_obj = method.get("PVM_SatTransPulse")
    if pulse_obj is None:
        return None
    pulse_keys = [
        "Length",
        "Bandwidth",
        "Flipangle",
        "Calculated",
        "Sharpness",
        "Bwfac",
        "Sint",
        "Pint",
        "Type",
        "Rpfac",
        "Pow",
        "Shape",
    ]
    if isinstance(pulse_obj, dict):
        return pulse_obj.get("Length")
    if isinstance(pulse_obj, (list, tuple)) and len(pulse_obj) >= 1:
        for i, key in enumerate(pulse_keys):
            if key == "Length" and i < len(pulse_obj):
                return pulse_obj[i]
    return None


def _read_timestamp_file(exp_dir: Path) -> np.ndarray | None:
    candidates = (
        exp_dir / "TimeStamps.txt",
        exp_dir / "timestamps.txt",
        exp_dir / "time_stamps.txt",
    )
    for path in candidates:
        if not path.exists():
            continue
        try:
            lines = path.read_text().splitlines()
        except OSError:
            continue
        times = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            try:
                times.append(float(parts[0]))
            except (ValueError, TypeError):
                continue
        if times:
            return np.asarray(times, dtype=float)
    return None


def _infer_tr_from_timestamps(
    times_s: np.ndarray,
    *,
    nominal_tr_ms: float | None,
    rep_count: int | None,
) -> np.ndarray | None:
    if times_s.size < 2:
        return None
    diffs = np.diff(times_s)
    if diffs.size == 0:
        return None
    adc_step = float(np.median(diffs))
    threshold = adc_step * 5.0
    if nominal_tr_ms is not None:
        threshold = max(threshold, 0.5 * (float(nominal_tr_ms) * 1e-3))
    boundaries = np.where(diffs > threshold)[0] + 1
    rep_times = np.concatenate(([times_s[0]], times_s[boundaries]))
    if rep_times.size < 2:
        return None
    tr = np.diff(rep_times)
    if tr.size == 0:
        return None
    first = tr[0]
    tr_full = np.concatenate(([first], tr))
    if rep_count is not None and rep_count > 0:
        if tr_full.size < rep_count:
            pad = np.full((rep_count - tr_full.size,), tr_full[-1], dtype=float)
            tr_full = np.concatenate((tr_full, pad))
        elif tr_full.size > rep_count:
            tr_full = tr_full[:rep_count]
    return tr_full


def _sat_trans_rep_count(exp_type: str, dims: dict, frame_count: int) -> int | None:
    if exp_type != "sat_trans":
        return None
    t_len = int(dims.get("t", 1))
    offset_len = int(dims.get("offset", 1))
    rep_count = offset_len * t_len
    if rep_count <= 0 or rep_count > frame_count:
        return None
    return rep_count


def _parse_bvals_from_comments(comments: object) -> list[float]:
    if comments is None:
        return []
    if isinstance(comments, str):
        comments = [comments]
    if not isinstance(comments, (list, tuple)):
        return []
    bvals: list[float] = []
    for entry in comments:
        try:
            match = re.search(r"[Bb]\s*([-+0-9.]+(?:[eE][-+]?\d+)?)", str(entry))
            if match:
                bvals.append(float(match.group(1)))
        except Exception:
            continue
    return bvals


def _reshape_diffusion_array(values: object, *, frame_count: int, components: int) -> np.ndarray | None:
    flat = _flatten_numeric(values)
    if not flat:
        return None
    needed = frame_count * components
    if len(flat) < needed:
        return None
    arr = np.asarray(flat[:needed], dtype=float)
    return arr.reshape((frame_count, components))


def _normalize_bvecs(bvecs: np.ndarray, bvals: np.ndarray | None = None, b0_threshold: float = 50.0) -> np.ndarray:
    out = np.asarray(bvecs, dtype=float).copy()
    if out.ndim != 2 or out.shape[1] != 3:
        return out
    norms = np.linalg.norm(out, axis=1)
    nonzero = norms > 0
    out[nonzero] = out[nonzero] / norms[nonzero][:, None]
    if bvals is not None:
        out = out.copy()
        out[bvals < b0_threshold] = 0.0
    return out


def _bvecs_from_bmatrix(bmatrix: np.ndarray, bvals: np.ndarray | None = None, b0_threshold: float = 50.0) -> np.ndarray:
    mats = np.asarray(bmatrix, dtype=float)
    if mats.ndim != 3 or mats.shape[1:] != (3, 3):
        raise ValueError("Diffusion B-matrix must have shape (N, 3, 3)")
    vectors = np.zeros((mats.shape[0], 3), dtype=float)
    for idx, mat in enumerate(mats):
        sym = 0.5 * (mat + mat.T)
        try:
            vals, vecs = np.linalg.eigh(sym)
        except np.linalg.LinAlgError:
            continue
        axis = int(np.argmax(np.abs(vals)))
        if abs(float(vals[axis])) <= 1e-12:
            continue
        vectors[idx] = vecs[:, axis]
    return _normalize_bvecs(vectors, bvals, b0_threshold=b0_threshold)


def _bruker_diffusion_bvals(visu: dict, method: dict, frame_count: int) -> np.ndarray | None:
    bmat = _reshape_diffusion_array(visu.get("VisuAcqDiffusionBMatrix"), frame_count=frame_count, components=9)
    if bmat is not None:
        bmat = bmat.reshape((frame_count, 3, 3))
        return np.trace(bmat, axis1=1, axis2=2)
    comments = _parse_bvals_from_comments(visu.get("VisuFGElemComment", []))
    if comments and len(comments) >= frame_count:
        return np.asarray(comments[-frame_count:], dtype=float)
    bvals = _flatten_numeric(method.get("PVM_DwEffBval"))
    if bvals and len(bvals) >= frame_count:
        return np.asarray(bvals[-frame_count:], dtype=float)
    return None


def _rotate_bvecs_to_image(bvecs: np.ndarray, visu: dict) -> np.ndarray:
    orientations = _bruker_orientation(visu)
    if not orientations:
        return bvecs
    rot = orientations[0]
    if rot.shape != (3, 3):
        return bvecs
    # VisuCoreOrientation maps image axes -> scanner (LPS). Convert scanner vectors to image axes.
    return bvecs @ rot


def _bruker_diffusion_bvecs(
    visu: dict,
    method: dict,
    frame_count: int,
    bvals: np.ndarray | None = None,
    *,
    rotate_to_image: bool = True,
) -> np.ndarray | None:
    bmat = _reshape_diffusion_array(visu.get("VisuAcqDiffusionBMatrix"), frame_count=frame_count, components=9)
    if bmat is not None:
        bvecs = _bvecs_from_bmatrix(bmat.reshape((frame_count, 3, 3)), bvals)
        return _rotate_bvecs_to_image(bvecs, visu) if rotate_to_image else bvecs
    for key in ("VisuAcqDiffusionGradOrient",):
        arr = _reshape_diffusion_array(visu.get(key), frame_count=frame_count, components=3)
        if arr is not None:
            bvecs = _normalize_bvecs(arr, bvals)
            return _rotate_bvecs_to_image(bvecs, visu) if rotate_to_image else bvecs
    arr = _reshape_diffusion_array(method.get("PVM_DwGradVec"), frame_count=frame_count, components=3)
    if arr is not None:
        bvecs = _normalize_bvecs(arr, bvals)
        return _rotate_bvecs_to_image(bvecs, visu) if rotate_to_image else bvecs
    dirs = _flatten_numeric(method.get("PVM_DwDir"))
    if dirs:
        dir_arr = np.asarray(dirs, dtype=float)
        if dir_arr.size % 3 == 0:
            dir_arr = dir_arr.reshape((-1, 3))
            ao = method.get("PVM_DwAoImages")
            ao_count = int(ao) if isinstance(ao, (int, float, np.generic)) else 0
            total = ao_count + dir_arr.shape[0]
            if total == frame_count:
                zeros = np.zeros((ao_count, 3), dtype=float)
                bvecs = _normalize_bvecs(np.vstack([zeros, dir_arr]), bvals)
                return _rotate_bvecs_to_image(bvecs, visu) if rotate_to_image else bvecs
            if dir_arr.shape[0] == frame_count:
                bvecs = _normalize_bvecs(dir_arr, bvals)
                return _rotate_bvecs_to_image(bvecs, visu) if rotate_to_image else bvecs
    return None


def _normalize_bruker_metadata(
    visu: dict,
    method: dict,
    *,
    frame_count: int | None = None,
    rotate_bvecs_to_image: bool = True,
) -> NormalizedMetadata:
    frame_comments = visu.get("VisuFGElemComment", [])
    if isinstance(frame_comments, str):
        frame_comments = [frame_comments]
    elif not isinstance(frame_comments, (list, tuple)):
        frame_comments = []
    diffusion_bvals = None
    diffusion_bvecs = None
    if frame_count is not None:
        diffusion_bvals = _bruker_diffusion_bvals(visu, method, frame_count)
        diffusion_bvecs = _bruker_diffusion_bvecs(
            visu,
            method,
            frame_count,
            diffusion_bvals,
            rotate_to_image=rotate_bvecs_to_image,
        )
    fair_mode = method.get("PVM_FairMode")
    return NormalizedMetadata(
        method_name=str(method.get("Method", "") or ""),
        sequence_name=str(visu.get("VisuAcqSequenceName", "") or ""),
        protocol_name=str(visu.get("VisuAcquisitionProtocol", "") or ""),
        comment=str(visu.get("VisuSeriesExperimentComment", "") or ""),
        frame_comments=tuple(str(v) for v in frame_comments),
        frame_count=int(frame_count) if frame_count is not None else None,
        fair_labels=(
            _infer_bruker_fair_labels(fair_mode, frame_count)
            if frame_count is not None
            else None
        ),
        sat_trans_on=_sat_trans_enabled(method),
        scan_time_ms=(
            _flatten_numeric(method.get("PVM_ScanTime") or visu.get("VisuAcqScanTime"))[0]
            if _flatten_numeric(method.get("PVM_ScanTime") or visu.get("VisuAcqScanTime"))
            else None
        ),
        echo_times=visu.get("VisuAcqEchoTime", []),
        repetition_times=visu.get("VisuAcqRepetitionTime") or method.get("PVM_RepetitionTime"),
        inversion_times=visu.get("VisuAcqInversionTime", []),
        flip_angles=visu.get("VisuAcqFlipAngle") or method.get("PVM_ExcPulseAngle"),
        sat_b1_uT=method.get("PVM_SatTransPulseAmpl_uT"),
        sat_time_ms=_bruker_sat_time_ms(method),
        spin_lock=_bruker_spin_lock(method),
        diffusion_bvals=diffusion_bvals,
        diffusion_bvecs=diffusion_bvecs,
    )


def detect_type(
    visu: dict,
    method: dict,
    frame_count: int | None = None,
    *,
    rotate_bvecs_to_image: bool = True,
) -> str:
    """Best-effort detection of the experiment type from Bruker metadata."""

    metadata = _normalize_bruker_metadata(
        visu,
        method,
        frame_count=frame_count,
        rotate_bvecs_to_image=rotate_bvecs_to_image,
    )
    exp_type = detect_experiment_type(metadata)
    return _override_fair_exp_type(exp_type, method.get("PVM_FairMode"))


def extract_observables(
    exp_type: str,
    visu: dict,
    frame_count: int,
    method: dict | None = None,
    *,
    rotate_bvecs_to_image: bool = True,
) -> dict:
    metadata = _normalize_bruker_metadata(
        visu,
        method or {},
        frame_count=frame_count,
        rotate_bvecs_to_image=rotate_bvecs_to_image,
    )
    return extract_observables_from_metadata(exp_type, metadata, frame_count)


def _apply_recipe(
    recipe: ExperimentRecipe | None, visu: dict, method: dict, frame_count: int
) -> Tuple[str | None, dict | None, Model | Callable[[], Model] | None]:
    if recipe is None:
        return None, None, None
    try:
        result = recipe(visu, method, frame_count)
    except Exception as exc:  # pragma: no cover - user-provided hook
        warnings.warn(f"Experiment recipe failed with {exc}; falling back to defaults.", RuntimeWarning, stacklevel=2)
        return None, None, None

    if isinstance(result, tuple):
        recipe_type = result[0]
        recipe_obs = result[1] if len(result) > 1 else None
        recipe_model = result[2] if len(result) > 2 else None
        return recipe_type, recipe_obs, recipe_model
    return result, None, None


def _manual_override_warning_message(meta: NormalizedMetadata, source: Path) -> str:
    return (
        "Could not determine the experiment type automatically for "
        f"{source}. Falling back to 'anatomical'. "
        f"Method='{meta.method_name}', Sequence='{meta.sequence_name}', Protocol='{meta.protocol_name}'. "
        "Provide a manual override (override_type=...) or a recipe/model override for quantitative fitting."
    )


def _fit_override_warning_message(
    meta: NormalizedMetadata,
    source: Path,
    *,
    exp_type: str,
    default_identity_model: bool,
    manual_type_override_required: bool,
) -> str:
    if manual_type_override_required:
        message = _manual_override_warning_message(meta, source)
        if default_identity_model:
            message += " The imported default model is 'identity'; fitting it only estimates an amplitude image."
        return message + " A manual experiment/model override is required before quantitative fitting."
    return (
        f"Experiment imported from {source} uses the default identity model for exp_type='{exp_type}'. "
        "This is expected for localizers and other non-quantitative scans, but fitting it only estimates "
        "an amplitude image. Provide model_map_by_type/model_map_by_name, recipe=..., or pass a custom "
        "model to run_fit for quantitative fitting."
    )


def _manual_fit_override_warning_message(meta: NormalizedMetadata, source: Path) -> str:
    return (
        _manual_override_warning_message(meta, source)
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

def _bruker_sequence_type(method: dict, visu: dict | None = None) -> str:
    """Infer a sequence-ordering label from Bruker metadata."""

    tokens = [
        method.get("Method"),
        method.get("PVM_Method"),
        method.get("ACQ_scan_name"),
        None if visu is None else visu.get("VisuAcqSequenceName"),
        None if visu is None else visu.get("VisuAcquisitionProtocol"),
    ]
    text = " ".join(str(token or "") for token in tokens).lower()
    if any(keyword in text for keyword in ("spectro", "spectroscopy", "mrs", "press", "steam")) or re.search(
        r"(?<![a-z0-9])fid(?![a-z0-9])", text
    ):
        return "spectroscopy"
    if "epi" in text:
        return "epi"
    if any(keyword in text for keyword in ("rare", "turbo", "tse", "fse", "spin_echo")):
        return "rare"
    if any(keyword in text for keyword in ("flash", "fisp", "gre", "gradient_echo")):
        return "standard"
    if "spiral" in text:
        return "spiral"
    if any(keyword in text for keyword in ("radial", "ute", "zte")):
        return "radial"
    return normalize_sequence_type(method.get("Method"))


"""Bruker metadata translation; MRI reconstruction is separate."""

