from __future__ import annotations

import logging
import os
import re
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import jax.numpy as jnp
import numpy as np

from mrax.data_struct import DataSet, Experiment, Model, Quantity, Study
from mrax.data_struct.quantity import require_quantity_dict
from mrax.io.import_definitions import (
    ImportDefinition,
    KSpaceImportDefinition,
    NormalizedMetadata,
    apply_encoding_steps,
    build_imported_experiment_dataset,
    build_non_cartesian_kspace_definition,
    detect_experiment_type,
    extract_observables_from_metadata,
    needs_manual_type_override,
    normalize_sequence_type,
    reshape_raw_kspace,
)
from mrax.library.default_experiments import default_model_for_type
from mrax.reco.pipeline import FFTStage, RecoPipeline, RecoState, cartesian_ifft, make_sampling_mask
from mrax.reco.processing import (
    apply_channel_scale_phase,
    estimate_radial_density,
    fid_profiles_to_kspace,
    readout_crop_bounds,
)
from mrax.reco.variational import RegridApodizationCorrection, reconstruct_regrid

RecipeResult = Tuple[str | None, dict | None, Model | Callable[[], Model] | None]
_LOG = logging.getLogger(__name__)
ExperimentRecipe = Callable[[dict, dict, int], str | RecipeResult]

_LPS_TO_RAS = np.diag([-1.0, -1.0, 1.0])


def _parse_value(raw: str):
    raw = raw.strip()

    def _parse_scalar(tok: str):
        tok = tok.strip()
        if tok == "":
            return tok
        if tok.startswith("<") and tok.endswith(">"):
            return tok[1:-1]
        if re.fullmatch(r"[-+]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", tok):
            return float(tok) if ("." in tok or "e" in tok.lower()) else int(tok)
        return tok

    # Strip Bruker array-shape prefixes like "( 2 )" or "( 2, 30 )" and keep the payload.
    m_shape = re.match(r"^\(\s*([0-9]+(?:\s*,\s*[0-9]+)*)\s*\)\s*(.*)$", raw, re.S)
    if m_shape:
        payload = m_shape.group(2).strip()
        if payload:
            dims = [int(x) for x in re.split(r"\s*,\s*", m_shape.group(1).strip()) if x.strip() != ""]
            if dims and all(d > 0 for d in dims):
                raw = payload

    # Expand Bruker RLE segments like "@30*(0.5) @30*(0.5658)".
    if "@" in raw:
        rle = re.compile(r"@([0-9]+)\*\(([^)]*)\)")
        if rle.search(raw):
            expanded: list[object] = []
            pos = 0
            for m in rle.finditer(raw):
                prefix = raw[pos : m.start()].strip()
                if prefix:
                    prefix_val = _parse_value(prefix)
                    if isinstance(prefix_val, list):
                        expanded.extend(prefix_val)
                    else:
                        expanded.append(prefix_val)

                n = int(m.group(1))
                inner = m.group(2).strip()
                inner_val = _parse_value(inner)
                if isinstance(inner_val, list):
                    expanded.extend(inner_val * n)
                else:
                    expanded.extend([inner_val] * n)
                pos = m.end()

            suffix = raw[pos:].strip()
            if suffix:
                suffix_val = _parse_value(suffix)
                if isinstance(suffix_val, list):
                    expanded.extend(suffix_val)
                else:
                    expanded.append(suffix_val)
            return expanded

    tokens = re.findall(r"<[^>]*>|\([^)]*\)|[^\s]+", raw)
    if not tokens:
        return raw

    parsed: list[object] = []
    for tok in tokens:
        if tok.startswith("(") and tok.endswith(")") and "," in tok:
            parts = [p.strip() for p in tok[1:-1].split(",") if p.strip() != ""]
            parsed.append([_parse_scalar(p) for p in parts])
        else:
            parsed.append(_parse_scalar(tok))

    if len(parsed) == 1:
        return parsed[0]
    return parsed


def _parse_jcamp_file(path: Path, entries: Dict[str, object] | None = None) -> Dict[str, object]:
    entries = dict(entries or {})
    lines = path.read_text(errors="replace").splitlines()
    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        if not line.startswith("##$") or "=" not in line:
            idx += 1
            continue

        key, raw = line[3:].split("=", 1)
        key = key.strip()
        value_lines = [raw.split("$$", 1)[0].strip()]
        idx += 1
        while idx < len(lines) and not lines[idx].startswith("##"):
            cont = lines[idx].strip()
            if cont and not cont.startswith("$$"):
                value_lines.append(cont.split("$$", 1)[0].strip())
            idx += 1

        if key and key != "END":
            raw_value = "\n".join(part for part in value_lines if part != "")
            entries[key] = _parse_value(raw_value)
    return entries


def parse_jcamp(path: Path, *, update_with_output_files: bool = True) -> Dict[str, object]:
    """Parse a Bruker JCAMP parameter file.

    ParaVision 360 may write companion ``*.out`` files whose values supersede
    the base file. This mirrors Bruker's MATLAB ``readBrukerParamFile`` behavior.
    """

    path = Path(path)
    entries = _parse_jcamp_file(path)
    output_path = path.with_name(path.name + ".out")
    if update_with_output_files and output_path.exists():
        entries = _parse_jcamp_file(output_path, entries)
    return entries


_BRUKER_WORD_DTYPE = {
    "_8BIT_UNSGN_INT": np.uint8,
    "_8BIT_SGN_INT": np.int8,
    "_16BIT_UNSGN_INT": np.uint16,
    "_16BIT_SGN_INT": np.int16,
    "_32BIT_UNSGN_INT": np.uint32,
    "_32BIT_SGN_INT": np.int32,
    "_32BIT_FLOAT": np.float32,
    "_64BIT_FLOAT": np.float64,
}


def bruker_word_dtype(word: str) -> np.dtype:
    key = str(word)
    if key not in _BRUKER_WORD_DTYPE:
        raise ValueError(f"Unknown VisuCoreWordType {word}")
    return _BRUKER_WORD_DTYPE[key]


def _flatten_numeric(value: object) -> list[float]:
    if value is None:
        return []
    if isinstance(value, (np.generic, int, float)):
        return [float(value)]
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return [float(value)]
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        out: list[float] = []
        for item in value:
            out.extend(_flatten_numeric(item))
        return out
    return []


def _normalize_size(value: object, *, dim: int | None = None) -> list[float]:
    vals = _flatten_numeric(value)
    if not vals:
        return []
    vals = [float(v) for v in vals if v is not None]
    if dim is None or dim <= 0:
        return vals
    vals = [v for v in vals if v > 0]
    if not vals:
        return []
    if len(vals) >= dim:
        return vals[-dim:]
    if len(vals) == 1:
        return vals * dim
    return vals + [vals[-1]] * (dim - len(vals))


def _bruker_spatial_shape(visu: dict) -> tuple[int, ...]:
    dim_val = visu.get("VisuCoreDim")
    dim = int(dim_val) if isinstance(dim_val, (int, float, np.generic)) else 0
    size_vals = _normalize_size(visu.get("VisuCoreSize"), dim=dim if dim > 0 else None)
    if not size_vals:
        return ()
    if dim <= 0:
        dim = 3 if len(size_vals) >= 3 else len(size_vals)
    if len(size_vals) < dim:
        if len(size_vals) == 1:
            size_vals = size_vals * dim
        else:
            size_vals = size_vals + [size_vals[-1]] * (dim - len(size_vals))
    return tuple(int(round(v)) for v in size_vals[:dim])


def _bruker_orientation(visu: dict) -> list[np.ndarray]:
    vals = _flatten_numeric(visu.get("VisuCoreOrientation"))
    if not vals:
        return []
    count = len(vals) // 9
    if count <= 0:
        return []
    matrices: list[np.ndarray] = []
    for i in range(count):
        mat = np.array(vals[i * 9 : (i + 1) * 9], dtype=float).reshape(3, 3).T
        matrices.append(mat)
    return matrices


def _bruker_image_orientation_metadata(visu: dict) -> dict[str, object]:
    """Normalize Bruker image axes to anatomical axes and acquisition plane."""

    orientations = _bruker_orientation(visu)
    if not orientations:
        return {}
    orientation = np.asarray(orientations[0], dtype=float)
    if orientation.shape != (3, 3) or not np.isfinite(orientation).all():
        return {}
    norms = np.linalg.norm(orientation, axis=0)
    if np.any(norms <= 1e-8):
        return {}
    dominant = np.argmax(np.abs(orientation / norms[None, :]), axis=0)
    labels = tuple("xyz"[int(index)] for index in dominant)
    if len(set(labels)) != 3:
        return {}
    native_plane = {"x": "sagittal", "y": "coronal", "z": "axial"}[labels[2]]
    return {
        "image_axis_anatomical_labels": list(labels),
        "native_image_plane": native_plane,
    }


def _bruker_position(visu: dict) -> list[np.ndarray]:
    vals = _flatten_numeric(visu.get("VisuCorePosition"))
    if not vals:
        return []
    count = len(vals) // 3
    if count <= 0:
        return []
    vectors: list[np.ndarray] = []
    for i in range(count):
        vec = np.array(vals[i * 3 : (i + 1) * 3], dtype=float)
        vectors.append(vec)
    return vectors


def _bruker_frame_transpositions(value: object, frame_count: int) -> np.ndarray:
    """Broadcast Bruker frame transposition metadata to one value per frame."""

    count = max(int(frame_count), 0)
    if count == 0:
        return np.zeros((0,), dtype=int)

    vals = _flatten_numeric(value)
    if not vals:
        return np.zeros((count,), dtype=int)

    arr = np.asarray(vals, dtype=int).ravel()
    if arr.size == 1:
        return np.full((count,), int(arr[0]), dtype=int)
    if arr.size >= count:
        return arr[:count]
    if count % int(arr.size) == 0:
        return np.tile(arr, count // int(arr.size))
    pad_val = int(arr[-1]) if arr.size > 0 else 0
    return np.pad(arr, (0, count - arr.size), constant_values=pad_val)


def _bruker_frame_package_indices(method: dict, frame_count: int) -> np.ndarray | None:
    """Return natural-order Bruker package ids derived from ``PVM_SPackArrNSlices``."""

    counts = [int(round(v)) for v in _flatten_numeric(method.get("PVM_SPackArrNSlices")) if float(v) > 0]
    if not counts or sum(counts) != int(frame_count):
        return None

    package_ids = np.empty((int(frame_count),), dtype=int)
    start = 0
    for pkg_idx, count in enumerate(counts):
        stop = start + int(count)
        package_ids[start:stop] = int(pkg_idx)
        start = stop
    return package_ids


def _visu_fg_order_entries(visu: dict) -> list[list[object]]:
    """Return parsed ``VisuFGOrderDesc`` entries as ``[count, kind, ...]`` lists."""

    value = visu.get("VisuFGOrderDesc")
    if value is None:
        return []
    if isinstance(value, list) and value and all(isinstance(item, list) for item in value):
        return [list(item) for item in value if item]
    if isinstance(value, list) and len(value) >= 2:
        if len(value) <= 5 and isinstance(value[0], (int, float)) and isinstance(value[1], str):
            return [list(value)]
        entries: list[list[object]] = []
        idx = 0
        while idx + 1 < len(value):
            if isinstance(value[idx], (int, float)) and isinstance(value[idx + 1], str):
                entries.append(list(value[idx : idx + 5]))
                idx += 5
            else:
                idx += 1
        return entries
    return []


def _visu_fg_count(visu: dict, token: str) -> int | None:
    needle = token.upper()
    for entry in _visu_fg_order_entries(visu):
        if len(entry) >= 2 and needle in str(entry[1]).upper():
            try:
                count = int(round(float(entry[0])))
            except (TypeError, ValueError):
                return None
            return count if count > 0 else None
    return None


def _bruker_has_mixed_slice_geometry(visu: dict, frame_count: int) -> bool:
    """Detect 2D frame groups whose frames do not share one slice-stack geometry."""

    if int(frame_count) <= 1 or not _contains_token(visu.get("VisuFGOrderDesc"), "FG_SLICE"):
        return False

    orientations = _bruker_orientation(visu)
    if orientations:
        if len(orientations) == 1:
            orientations = orientations * int(frame_count)
        if len(orientations) >= int(frame_count):
            unique_orientations = {
                tuple(np.round(np.asarray(mat, dtype=float), 6).ravel().tolist())
                for mat in orientations[: int(frame_count)]
            }
            if len(unique_orientations) > 1:
                return True

    transpositions = _bruker_frame_transpositions(visu.get("VisuCoreTransposition"), int(frame_count))
    return bool(np.unique(transpositions).size > 1)


def _apply_framewise_transposition(data: np.ndarray | None, transpositions: Sequence[int] | None) -> np.ndarray | None:
    """Apply Bruker frame-wise image transposition metadata to a frame stack."""

    if data is None or transpositions is None:
        return data

    arr = np.asarray(data)
    if arr.ndim < 3:
        return arr

    trans = np.asarray(transpositions, dtype=int).ravel()
    if trans.size != int(arr.shape[0]) or not np.any(trans > 0):
        return arr

    unique = np.unique(trans)
    if unique.size == 1 and int(unique[0]) > 0:
        return np.swapaxes(arr, -2, -1)
    if arr.shape[-2] != arr.shape[-1]:
        warnings.warn(
            "Skipping mixed frame-wise transposition for non-square Bruker frames; "
            "a single array shape cannot represent both transposed and non-transposed frames.",
            RuntimeWarning,
            stacklevel=2,
        )
        return arr

    out = np.array(arr, copy=True)
    for frame_idx, trans_val in enumerate(trans.tolist()):
        if int(trans_val) > 0:
            out[frame_idx] = np.swapaxes(out[frame_idx], -2, -1)
    return out


def _bruker_voxel_sizes(visu: dict) -> tuple[float, float, float]:
    dim_val = visu.get("VisuCoreDim")
    dim = int(dim_val) if isinstance(dim_val, (int, float, np.generic)) else 0
    size_vals = _normalize_size(visu.get("VisuCoreSize"), dim=dim if dim > 0 else None)
    extent_vals = _normalize_size(visu.get("VisuCoreExtent"), dim=dim if dim > 0 else None)
    thickness_vals = _normalize_size(visu.get("VisuCoreFrameThickness"), dim=1)

    if dim == 2 or len(size_vals) == 2 or len(extent_vals) == 2:
        if len(size_vals) == 1:
            size_vals = size_vals * 2
        if len(extent_vals) == 1:
            extent_vals = extent_vals * 2
        vx = extent_vals[0] / size_vals[0] if size_vals and size_vals[0] else 1.0
        vy = extent_vals[1] / size_vals[1] if len(size_vals) > 1 and size_vals[1] else 1.0
        vz = thickness_vals[0] if thickness_vals else 1.0
        return (float(vx), float(vy), float(vz))

    if len(size_vals) < 3:
        if len(size_vals) == 1:
            size_vals = size_vals * 3
        elif len(size_vals) == 2:
            size_vals = size_vals + [1.0]
    if len(extent_vals) < 3:
        if len(extent_vals) == 1:
            extent_vals = extent_vals * 3
        elif len(extent_vals) == 2:
            extra = thickness_vals[0] if thickness_vals else extent_vals[-1]
            extent_vals = extent_vals + [extra]

    vox = []
    for i in range(3):
        denom = size_vals[i] if i < len(size_vals) else 1.0
        vox.append(extent_vals[i] / denom if denom else 1.0)
    return (float(vox[0]), float(vox[1]), float(vox[2]))


def bruker_affine_from_visu(visu: dict, *, frame_index: int | None = None) -> np.ndarray | list[np.ndarray]:
    orientations = _bruker_orientation(visu)
    positions = _bruker_position(visu)
    voxel_sizes = _bruker_voxel_sizes(visu)

    if not orientations:
        orientations = [np.eye(3)]
    if not positions:
        positions = [np.zeros(3)]

    frame_count = max(len(orientations), len(positions))
    if len(orientations) == 1:
        orientations = orientations * frame_count
    if len(positions) == 1:
        positions = positions * frame_count

    def _build_affine(orientation: np.ndarray, position: np.ndarray) -> np.ndarray:
        affine = np.eye(4, dtype=float)
        affine[:3, :3] = (_LPS_TO_RAS @ orientation) @ np.diag(voxel_sizes)
        affine[:3, 3] = _LPS_TO_RAS @ position
        return affine

    affines = [_build_affine(orientations[i], positions[i]) for i in range(frame_count)]
    if frame_index is not None:
        if frame_index < 0 or frame_index >= len(affines):
            raise IndexError("frame_index out of range for VisuCorePosition/Orientation")
        return affines[frame_index]
    return affines[0] if len(affines) == 1 else affines


def _dtype_from_word(word: str) -> np.dtype:
    return bruker_word_dtype(word)


def _dtype_from_acq_word(word: str) -> np.dtype:
    """Map ACQ_word_size (e.g., _32_BIT) to a NumPy dtype."""

    mapping = {
        "_8_BIT": np.int8,
        "_16_BIT": np.int16,
        "_32_BIT": np.int32,
        "_64_BIT": np.int64,
    }
    if word not in mapping:
        raise ValueError(f"Unknown ACQ_word_size {word}")
    return mapping[word]


def _dtype_from_go_word(word: str) -> np.dtype:
    """Map GO_raw_data_format values to a NumPy dtype."""

    mapping = {
        "GO_16BIT_SGN_INT": np.int16,
        "GO_32BIT_SGN_INT": np.int32,
        "GO_32BIT_FLOAT": np.float32,
    }
    if word not in mapping:
        raise ValueError(f"Unknown GO_raw_data_format {word}")
    return mapping[word]


PDataIndex = str | int | Dict[str | int, str | int]
DEFAULT_PDATA_INDEX = "1"


def _find_rawdat_file(exp_dir: Path) -> Path | None:
    """Locate a Bruker rawdat file using common names."""

    for name in (
        "rawdata.job0",
        "rawdat.job0",
        "rawdata.job1",
        "rawdat.job1",
        "rawdata.job2",
        "rawdat.job2",
        "rawdata.job",
        "rawdat.job",
        "fid",
        "ser",
    ):
        candidate = exp_dir / name
        if candidate.exists():
            return candidate
    return None


def _read_rawdat_complex(raw_path: Path, dtype: np.dtype) -> np.ndarray:
    """Read interleaved real/imaginary samples from a rawdat.job file."""

    raw = np.fromfile(raw_path, dtype=dtype)
    if raw.size % 2 != 0:
        raw = raw[:-1]
    paired = raw.reshape((-1, 2)).astype(np.float32)
    return paired[:, 0] + 1j * paired[:, 1]


def _as_reco_string_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for item in value:
            out.extend(_as_reco_string_list(item))
        return out
    return [str(value)]


def _apply_reco_qopts(kspace: np.ndarray, reco: dict) -> np.ndarray:
    """Apply Bruker quadrature options in internal ``(..., ky, kx)`` ordering."""

    qopts = _as_reco_string_list(reco.get("RECO_qopts"))
    if not qopts:
        return kspace

    out = np.asarray(kspace)
    # Bruker dimensions are read, phase1, phase2, ...; internal 2D k-space is ky, kx.
    axes = [-1, -2, -3, -4]
    for idx, option in enumerate(qopts):
        opt = option.upper()
        if idx >= len(axes) or abs(axes[idx]) > out.ndim:
            continue
        axis = axes[idx]
        if opt in ("COMPLEX_CONJUGATE", "CONJ_AND_QNEG"):
            out = np.conj(out)
        if opt in ("QUAD_NEGATION", "CONJ_AND_QNEG"):
            signs = np.ones(out.shape[axis], dtype=out.dtype)
            signs[1::2] = -1
            shape = [1] * out.ndim
            shape[axis] = out.shape[axis]
            out = out * signs.reshape(shape)
    return out


def _resolve_bruker_target_kx(reco: dict, target_kx: int | None = None) -> int | None:
    """Resolve the Bruker readout FT size in MRax ``(..., ky, kx)`` ordering."""

    if target_kx is not None:
        try:
            resolved = int(target_kx)
        except (TypeError, ValueError):
            resolved = 0
        return resolved if resolved > 0 else None

    ft_size = reco.get("RECO_ft_size") or reco.get("RECO_size")
    vals = _flatten_numeric(ft_size)
    if not vals:
        return None
    resolved = int(round(vals[0]))
    return resolved if resolved > 0 else None


def _bruker_spatial_ndim_for_array(arr: np.ndarray, reco: dict) -> int:
    """Infer whether a Bruker Cartesian array carries 2D or 3D spatial axes."""

    size_vals = _flatten_numeric(reco.get("RECO_ft_size") or reco.get("RECO_size"))
    if len(size_vals) >= 3 and int(round(size_vals[2])) > 1 and np.asarray(arr).ndim >= 4:
        return 3
    return 2


def _pixel_shift_from_rotate(val: float, size: int) -> int:
    """Convert Bruker's ``RECO_rotate`` entry into a pixel shift."""

    if size <= 0:
        return 0
    # ParaVision stores RECO_rotate close to 0.5 for centered images.
    if -2.0 <= float(val) <= 2.0:
        return int(np.round((float(val) - 0.5) * size))
    return int(np.round(val))


def _apply_bruker_rotate_to_kspace(kspace: np.ndarray, reco: dict) -> tuple[np.ndarray, dict]:
    """Apply Bruker ``RECO_rotate`` as a k-space phase ramp."""

    arr = np.asarray(kspace)
    rotate = reco.get("RECO_rotate")
    if rotate is None:
        return arr, {"applied": False, "shifts": None}

    rot_vals = np.asarray(_flatten_numeric(rotate), dtype=float).ravel()
    frame_count = int(arr.shape[0]) if arr.ndim > 0 else 0
    if rot_vals.size < 2 or frame_count <= 0 or arr.ndim < 3:
        return arr, {"applied": False, "shifts": None}

    spatial_ndim = _bruker_spatial_ndim_for_array(arr, reco)
    spatial_shape = tuple(int(v) for v in arr.shape[-spatial_ndim:])
    if any(v <= 0 for v in spatial_shape):
        return arr, {"applied": False, "shifts": None}

    if rot_vals.size >= spatial_ndim * frame_count:
        rotate_by_dim = [
            rot_vals[dim_idx * frame_count : (dim_idx + 1) * frame_count]
            for dim_idx in range(spatial_ndim)
        ]
    elif (
        rot_vals.size % spatial_ndim == 0
        and rot_vals.size // spatial_ndim > 1
        and frame_count % (rot_vals.size // spatial_ndim) == 0
    ):
        values_per_dim = int(rot_vals.size // spatial_ndim)
        repeat_count = int(frame_count // values_per_dim)
        rotate_by_dim = [
            np.tile(
                rot_vals[dim_idx * values_per_dim : (dim_idx + 1) * values_per_dim],
                repeat_count,
            )
            for dim_idx in range(spatial_ndim)
        ]
    elif rot_vals.size >= spatial_ndim:
        rotate_by_dim = [
            np.full((frame_count,), float(rot_vals[dim_idx]), dtype=float)
            for dim_idx in range(spatial_ndim)
        ]
    else:
        return arr, {"applied": False, "shifts": None}

    # Bruker stores read/phase/slice; MRax k-space uses slice/phase/read for 3D
    # and phase/read for 2D.
    shifts_by_internal_dim = []
    for value_series, size in zip(reversed(rotate_by_dim[:spatial_ndim]), spatial_shape):
        shifts_by_internal_dim.append(
            np.asarray([_pixel_shift_from_rotate(v, size) for v in value_series], dtype=np.float32)
        )

    if not any(np.any(shifts != 0) for shifts in shifts_by_internal_dim):
        return arr, {
            "applied": False,
            "shifts": [shifts.tolist() for shifts in shifts_by_internal_dim],
            "spatial_ndim": spatial_ndim,
        }

    axes = tuple(range(-spatial_ndim, 0))
    k_unc = np.fft.ifftshift(arr, axes=axes)
    phase_arg = 0.0
    for axis_idx, (size, shifts) in enumerate(zip(spatial_shape, shifts_by_internal_dim)):
        shift_shape = [1] * arr.ndim
        shift_shape[0] = frame_count
        coord_shape = [1] * arr.ndim
        coord_shape[arr.ndim - spatial_ndim + axis_idx] = size
        shift_arr = shifts.reshape(shift_shape)
        coord = np.arange(size, dtype=np.float32).reshape(coord_shape)
        phase_arg = phase_arg + shift_arr * coord / float(size)

    phase = np.exp(-2j * np.pi * phase_arg)
    out = np.fft.fftshift(k_unc * phase, axes=axes)
    return out, {
        "applied": True,
        "shifts": [shifts.tolist() for shifts in shifts_by_internal_dim],
        "spatial_ndim": spatial_ndim,
    }


def _apply_bruker_channel_corrections(
    data: np.ndarray,
    reco: dict,
    *,
    min_ndim: int = 4,
) -> tuple[np.ndarray, dict]:
    """Apply Bruker receiver scaling and phase offsets along axis 1."""

    arr = np.asarray(data)
    if arr.ndim < int(min_ndim):
        return arr, {"scale_applied": False, "phase_applied": False}

    channel_count = int(arr.shape[1])
    scale_applied = False
    phase_applied = False

    scale_vals = np.asarray(_flatten_numeric(reco.get("RecoScaleChan")), dtype=np.float32).ravel()
    scale_arg = None
    if scale_vals.size == channel_count:
        scale_arg = scale_vals
        scale_applied = True

    phase_vals = np.asarray(_flatten_numeric(reco.get("RecoPhaseChan")), dtype=np.float32).ravel()
    phase_arg = None
    if phase_vals.size == channel_count:
        phase_arg = phase_vals
        phase_applied = True

    out = apply_channel_scale_phase(
        arr,
        scale=scale_arg,
        phase_degrees=phase_arg,
        coil_axis=1,
    )

    return out, {"scale_applied": scale_applied, "phase_applied": phase_applied}


def _resolve_bruker_cutoff_params(
    reco: dict,
    frame_count: int,
    spatial_ndim: int = 2,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Resolve Bruker ``RECO_size``/``RECO_offset`` into internal spatial order."""

    size_vals = np.asarray(_flatten_numeric(reco.get("RECO_size")), dtype=int).ravel()
    spatial_ndim = int(spatial_ndim)
    if spatial_ndim <= 0 or size_vals.size < spatial_ndim:
        return None, None

    # Bruker stores read/phase/slice. Internal arrays use slice/phase/read for 3D
    # and phase/read for 2D.
    crop_shape = np.asarray(list(reversed(size_vals[:spatial_ndim])), dtype=int)
    if np.any(crop_shape <= 0):
        return None, None

    offset_vals = np.asarray(_flatten_numeric(reco.get("RECO_offset")), dtype=int).ravel()
    if offset_vals.size == 0:
        offsets = np.zeros((spatial_ndim, int(frame_count)), dtype=int)
        return crop_shape, offsets

    dim_count = max(int(size_vals.size), spatial_ndim)
    if offset_vals.size >= dim_count * int(frame_count):
        offset_matrix = offset_vals[: dim_count * int(frame_count)].reshape((dim_count, int(frame_count)))
    elif offset_vals.size >= dim_count:
        offset_matrix = np.repeat(offset_vals[:dim_count, None], int(frame_count), axis=1)
    else:
        return crop_shape, None

    return crop_shape, offset_matrix[:spatial_ndim][::-1].astype(int)


def _apply_bruker_image_cutoff(image: np.ndarray, reco: dict) -> tuple[np.ndarray, dict]:
    """Apply Bruker ``RECO_size``/``RECO_offset`` cropping in image space."""

    arr = np.asarray(image)
    if arr.ndim < 3:
        return arr, {"applied": False}

    spatial_ndim = _bruker_spatial_ndim_for_array(arr, reco)
    crop_shape, offsets = _resolve_bruker_cutoff_params(reco, int(arr.shape[0]), spatial_ndim=spatial_ndim)
    if crop_shape is None or offsets is None:
        return arr, {"applied": False}

    target_shape = tuple(int(v) for v in crop_shape)
    if target_shape == tuple(int(v) for v in arr.shape[-spatial_ndim:]) and not np.any(offsets):
        return arr, {"applied": False, "crop_shape": crop_shape.tolist()}

    out_shape = arr.shape[:-spatial_ndim] + target_shape
    out = np.zeros(out_shape, dtype=arr.dtype)
    for frame_idx in range(int(arr.shape[0])):
        src_slices: list[object] = [slice(None)] * arr.ndim
        dst_slices: list[object] = [slice(None)] * out.ndim
        src_slices[0] = frame_idx
        dst_slices[0] = frame_idx
        valid = True
        for axis_idx, target in enumerate(target_shape):
            src_axis = arr.ndim - spatial_ndim + axis_idx
            dst_axis = out.ndim - spatial_ndim + axis_idx
            offset_vec = offsets[axis_idx]
            start = int(offset_vec[min(frame_idx, offset_vec.size - 1)])
            stop = min(start + int(target), int(arr.shape[src_axis]))
            length = max(stop - start, 0)
            if length <= 0:
                valid = False
                break
            src_slices[src_axis] = slice(start, stop)
            dst_slices[dst_axis] = slice(0, length)
        if not valid:
            continue
        out[tuple(dst_slices)] = arr[tuple(src_slices)]
    return out, {
        "applied": True,
        "crop_shape": crop_shape.tolist(),
        "offsets": offsets.tolist(),
    }


def _bruker_unhandled_reco_steps(reco: dict) -> list[str]:
    """List Bruker reco filters that are currently surfaced as metadata only."""

    active: list[str] = []
    stage_text = _stage_node_text(reco)

    for token in (
        "RecoInverseFidFilter",
        "RecoStabCorrFilter",
        "RecoRegridFilter",
        "RecoFTShiftFilter",
        "RecoFTFilter",
        "RecoShuffleProfilesFilter",
        "RecoReverseFilter",
        "RecoHalfFourierFilter",
    ):
        if token in stage_text:
            active.append(token)

    wdw_mode = [item.upper() for item in _as_reco_string_list(reco.get("RECO_wdw_mode"))]
    if any(item != "NO_WDW" for item in wdw_mode):
        active.append("RECO_wdw_mode")

    pc_mode = [item.upper() for item in _as_reco_string_list(reco.get("RECO_pc_mode"))]
    if any(item not in ("NO_PC", "NO_PHASE_CORR") for item in pc_mode):
        active.append("RECO_pc_mode")

    ppc_mode = [item.upper() for item in _as_reco_string_list(reco.get("RECO_ppc_mode"))]
    if any(item not in ("NO_PPC", "NO_PHASE_CORR") for item in ppc_mode):
        active.append("RECO_ppc_mode")

    return active


def prepare_bruker_cartesian_kspace(
    kspace: np.ndarray,
    reco: dict,
    *,
    target_kx: int | None = None,
) -> tuple[np.ndarray, dict]:
    """Apply Bruker Cartesian raw-data corrections and return reconstruction-ready k-space.

    The returned array stays in MRax's internal Cartesian ordering
    ``(frame, coil?, ky, kx)`` and is suitable for direct FFT-based reconstruction.
    The helper applies the Bruker metadata that act in k-space or as per-channel
    global factors:

    - ``RECO_qopts`` quadrature options
    - read-direction zero-filling/placement implied by ``RECO_ft_size``
    - ``RECO_rotate`` as a phase ramp
    - ``RecoScaleChan`` / ``RecoPhaseChan`` receive-channel normalization
    """

    corrected = np.asarray(kspace)
    applied_steps: list[str] = []

    qopts = _as_reco_string_list(reco.get("RECO_qopts"))
    if qopts:
        corrected = _apply_reco_qopts(corrected, reco)
        applied_steps.append("RECO_qopts")

    resolved_target_kx = _resolve_bruker_target_kx(reco, target_kx=target_kx)
    readout_padding = None
    if resolved_target_kx is not None and corrected.ndim >= 2 and int(corrected.shape[-1]) != int(resolved_target_kx):
        kx = int(corrected.shape[-1])
        if int(resolved_target_kx) > kx:
            pad_total = int(resolved_target_kx) - kx
            pad_width = [(0, 0)] * corrected.ndim
            # Bruker's convertFrameToCKData places partial-Fourier readout samples at
            # the end of the read dimension.
            pad_width[-1] = (pad_total, 0)
            corrected = np.pad(corrected, pad_width, mode="constant")
            readout_padding = {"pad_left": int(pad_total), "pad_right": 0}
            applied_steps.append("readout_zero_fill")
        else:
            warnings.warn(
                f"Raw k-space readout has {kx} points, larger than target {resolved_target_kx}; keeping raw size.",
                RuntimeWarning,
                stacklevel=2,
            )

    corrected, rotate_meta = _apply_bruker_rotate_to_kspace(corrected, reco)
    if rotate_meta.get("applied"):
        applied_steps.append("RECO_rotate")

    corrected, channel_meta = _apply_bruker_channel_corrections(corrected, reco)
    if channel_meta.get("scale_applied"):
        applied_steps.append("RecoScaleChan")
    if channel_meta.get("phase_applied"):
        applied_steps.append("RecoPhaseChan")

    return corrected, {
        "applied_kspace_corrections": applied_steps,
        "qopts": qopts,
        "target_kx": resolved_target_kx,
        "readout_padding": readout_padding,
        "rotate": rotate_meta,
        "channel_corrections": channel_meta,
        "unhandled_reco_steps": _bruker_unhandled_reco_steps(reco),
    }


def prepare_bruker_non_cartesian_kspace(
    samples: np.ndarray,
    reco: dict,
) -> tuple[np.ndarray, dict]:
    """Apply Bruker receiver corrections to non-Cartesian sample streams.

    Non-Cartesian samples stay in MRax order ``(frame, coil?, sample)``. ParaVision's
    regridding pipeline applies the receive-channel scale and phase factors before
    channel combination, so those factors belong on ``DataSet.kspace`` just like the
    Cartesian path.
    """

    corrected = np.asarray(samples)
    applied_steps: list[str] = []
    corrected, channel_meta = _apply_bruker_channel_corrections(corrected, reco, min_ndim=3)
    if channel_meta.get("scale_applied"):
        applied_steps.append("RecoScaleChan")
    if channel_meta.get("phase_applied"):
        applied_steps.append("RecoPhaseChan")
    return corrected, {
        "applied_kspace_corrections": applied_steps,
        "channel_corrections": channel_meta,
        "unhandled_reco_steps": _bruker_unhandled_reco_steps(reco),
    }


def format_bruker_cartesian_image(
    image: np.ndarray,
    reco: dict,
    *,
    frame_transposition: Sequence[int] | None = None,
) -> np.ndarray:
    """Apply Bruker post-FFT image-domain formatting steps to a Cartesian image stack.

    This helper expects the image produced from vendor-corrected Cartesian k-space
    and applies the image-domain steps that define the standard Bruker magnitude
    output:

    - FFT normalization adjustment from MRax's orthonormal FFT to Bruker's FFT
    - ``RECO_size`` / ``RECO_offset`` cutoff
    - sum-of-squares coil combination
    - ``RECO_transposition`` / ``VisuCoreTransposition``
    """

    arr = np.asarray(image)
    spatial_ndim = _bruker_spatial_ndim_for_array(arr, reco)
    if arr.ndim >= spatial_ndim:
        fft_norm = float(np.sqrt(np.prod(arr.shape[-spatial_ndim:])))
        if fft_norm > 0:
            arr = arr / fft_norm

    arr, _ = _apply_bruker_image_cutoff(arr, reco)

    if arr.ndim >= 4:
        magnitude = np.sqrt(np.sum(np.abs(arr) ** 2, axis=1)).astype(np.float32)
    else:
        magnitude = np.abs(arr).astype(np.float32)

    if spatial_ndim == 3 and magnitude.ndim >= 4:
        # Internal Bruker k-space is reconstructed as (slice, phase, read);
        # imported image data is exposed as (read, phase, slice).
        magnitude = np.transpose(magnitude, (0, 3, 2, 1))

    transpositions = frame_transposition if frame_transposition is not None else reco.get("RECO_transposition")
    if spatial_ndim == 3:
        return magnitude
    return _apply_framewise_transposition(magnitude, transpositions)


def reconstruct_bruker_cartesian_kspace(
    kspace: np.ndarray,
    reco: dict,
    *,
    frame_transposition: Sequence[int] | None = None,
    assume_corrections_applied: bool = False,
    target_kx: int | None = None,
) -> np.ndarray:
    """Reconstruct a Bruker Cartesian k-space stack using the standard vendor steps."""

    corrected = np.asarray(kspace)
    if not assume_corrections_applied:
        corrected, _ = prepare_bruker_cartesian_kspace(corrected, reco, target_kx=target_kx)
    spatial_ndim = _bruker_spatial_ndim_for_array(corrected, reco)
    image = cartesian_ifft(corrected, axes=tuple(range(-spatial_ndim, 0)))
    return format_bruker_cartesian_image(
        image,
        reco,
        frame_transposition=frame_transposition,
    )


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


def _as_positive_int(value: object, default: int | None = None) -> int | None:
    vals = _flatten_numeric(value)
    if vals:
        resolved = int(round(vals[0]))
    else:
        try:
            resolved = int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return default
    return resolved if resolved > 0 else default


def _resolve_local_bruker_file(exp_dir: Path, value: object, default_name: str) -> Path:
    candidates: list[Path] = []
    if value is not None:
        for item in _as_reco_string_list(value):
            item_path = Path(str(item))
            candidates.append(item_path)
            candidates.append(exp_dir / item_path.name)
    candidates.append(exp_dir / default_name)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return exp_dir / default_name


def _bruker_regrid_metadata(method: dict, reco: dict) -> dict | None:
    """Return ParaVision RecoRegridN metadata for Bruker non-Cartesian scans."""

    regrid_flag = str(reco.get("RecoRegridN", "") or "").strip().lower()
    has_regrid = regrid_flag in ("on", "yes", "true", "1") or reco.get("RecoRegridNPoints") is not None
    method_text = " ".join(str(method.get(k, "") or "") for k in ("Method", "PVM_Method", "ACQ_scan_name")).lower()
    looks_non_cartesian = any(token in method_text for token in ("spiral", "ute", "zte", "radial"))
    if not has_regrid and not looks_non_cartesian:
        return None

    points = _as_positive_int(reco.get("RecoRegridNPoints"))
    interleaves = _as_positive_int(reco.get("RecoRegridNInterleaves"), 1)
    dim = _as_positive_int(reco.get("RecoRegridNDim"), 2)
    if points is None or interleaves is None or dim is None:
        return None

    traj_type = str(reco.get("RecoRegridNTrajType") or "").strip()
    return {
        "points_per_interleave": int(points),
        "interleaves": int(interleaves),
        "dim": int(dim),
        "trajectory_type": traj_type or "unknown",
        "trajectory_file": reco.get("RecoRegridNTrajFile"),
        "density_file": reco.get("RecoRegridNTrajFileDC"),
        "oversampling": _as_positive_int(reco.get("RecoRegridNOver")),
        "ft_size": reco.get("RecoRegridNFtSize"),
    }


def read_bruker_regrid_trajectory(
    exp_dir: Path,
    *,
    method: dict | None = None,
    reco: dict | None = None,
) -> tuple[np.ndarray, dict]:
    """Read a Bruker ParaVision ``RecoRegridN`` trajectory file.

    The returned trajectory is shaped ``(num_samples, dim)`` and uses the values
    written by ParaVision, which are already normalized around the MRax cycles/pixel
    convention for validated ParaVision 360 3.6 2D regridding scans.
    """

    exp_dir = Path(exp_dir)
    method = parse_jcamp(exp_dir / "method") if method is None else method
    if reco is None:
        pdata_dir = exp_dir / "pdata" / DEFAULT_PDATA_INDEX
        reco = parse_jcamp(pdata_dir / "reco") if (pdata_dir / "reco").exists() else {}
    regrid = _bruker_regrid_metadata(method, reco)
    if regrid is None:
        raise ValueError(f"No Bruker RecoRegridN trajectory metadata found for {exp_dir}")

    traj_path = _resolve_local_bruker_file(exp_dir, regrid.get("trajectory_file"), "traj")
    points = int(regrid["points_per_interleave"])
    interleaves = int(regrid["interleaves"])
    dim = int(regrid["dim"])
    expected = dim * points * interleaves
    raw = np.fromfile(traj_path, dtype="<f8")
    if int(raw.size) < expected:
        raise ValueError(
            f"Trajectory file {traj_path} has {raw.size} values, expected at least {expected} "
            f"from dim={dim}, points={points}, interleaves={interleaves}."
        )
    trajectory = raw[:expected].reshape((interleaves, points, dim)).reshape((interleaves * points, dim))
    meta = {
        **regrid,
        "trajectory_path": str(traj_path),
        "trajectory_samples": int(trajectory.shape[0]),
    }
    return trajectory.astype(np.float32), meta


def _stage_node_text(reco: dict) -> str:
    nodes = reco.get("RecoStageNodes")
    if isinstance(nodes, str):
        return nodes
    if not isinstance(nodes, (list, tuple)):
        return ""
    parts: list[str] = []
    for item in nodes:
        if isinstance(item, (list, tuple)):
            parts.append(" ".join(str(part) for part in item))
        else:
            parts.append(str(item))
    return " ".join(parts)


def _normalize_non_cartesian_density(density: np.ndarray, spatial_shape: tuple[int, ...]) -> np.ndarray:
    weights = np.asarray(density, dtype=np.float32).ravel()
    weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)
    if not np.any(weights > 0.0):
        return np.ones_like(weights, dtype=np.float32)

    target = float(np.prod(tuple(int(v) for v in spatial_shape)))
    total = float(np.sum(weights))
    if total > 0.0 and np.isfinite(total) and target > 0.0:
        weights = weights * (target / total)
    return weights.astype(np.float32)


def _density_from_jacobian_trajectory(
    density_trajectory: np.ndarray,
    *,
    points_per_interleave: int,
    interleaves: int,
    spatial_shape: tuple[int, ...],
) -> np.ndarray:
    traj = np.asarray(density_trajectory, dtype=np.float32)
    if traj.ndim != 2 or traj.shape[1] < 2:
        raise ValueError("Bruker density trajectory must have shape (sample, dim>=2)")
    expected = int(points_per_interleave) * int(interleaves)
    if int(traj.shape[0]) != expected:
        raise ValueError(f"Bruker density trajectory has {traj.shape[0]} samples, expected {expected}")
    if traj.shape[1] != 2:
        return estimate_radial_density(traj[:, : len(spatial_shape)], spatial_shape=spatial_shape)

    segments = traj[:, :2].reshape((int(interleaves), int(points_per_interleave), 2))
    weights: list[np.ndarray] = []
    for segment in segments:
        derivative = np.gradient(segment, axis=0)
        jac = np.abs(segment[:, 0] * derivative[:, 1] - segment[:, 1] * derivative[:, 0])
        weights.append(jac.astype(np.float32))
    return _normalize_non_cartesian_density(np.concatenate(weights), spatial_shape)


def _bruker_density_correction_type(reco: dict, regrid_meta: dict) -> str:
    text = _stage_node_text(reco)
    match = re.search(r"RecoRegridDensCorrFilter\s+\w+\{[^}]*type\s*=\s*\"([^\"]+)\"", text)
    if match:
        return match.group(1).strip().lower()

    traj_type = str(regrid_meta.get("trajectory_type", "") or "").lower()
    dens_mode = str(reco.get("RecoRegridNDensCorr", "") or "").lower()
    if "radial" in traj_type:
        return "rho"
    if "spiral" in traj_type and "pre" in dens_mode:
        return "jacobian"
    return "rho"


def _read_regrid_real_trajectory_file(path: Path, *, points: int, interleaves: int, dim: int) -> np.ndarray:
    expected = int(dim) * int(points) * int(interleaves)
    raw = np.fromfile(path, dtype="<f8")
    if int(raw.size) < expected:
        raise ValueError(
            f"Bruker trajectory file {path} has {raw.size} values, expected at least {expected} "
            f"from dim={dim}, points={points}, interleaves={interleaves}."
        )
    return raw[:expected].reshape((int(interleaves), int(points), int(dim))).reshape((-1, int(dim)))


def read_bruker_regrid_density(
    exp_dir: Path,
    trajectory: np.ndarray,
    *,
    spatial_shape: tuple[int, ...],
    method: dict | None = None,
    reco: dict | None = None,
) -> tuple[np.ndarray, dict]:
    """Read or derive ParaVision ``RecoRegridN`` density compensation weights.

    Bruker records the density compensation operation in ``RecoStageNodes``. Radial
    trajectories use ``type="rho"`` and spiral trajectories commonly use
    ``type="jacobian"`` with a companion ``trajDC`` trajectory. The returned weights
    are shaped ``(sample,)`` and normalized like MRax's generic density estimates.
    """

    exp_dir = Path(exp_dir)
    method = parse_jcamp(exp_dir / "method") if method is None else method
    if reco is None:
        pdata_dir = exp_dir / "pdata" / DEFAULT_PDATA_INDEX
        reco = parse_jcamp(pdata_dir / "reco") if (pdata_dir / "reco").exists() else {}
    regrid = _bruker_regrid_metadata(method, reco)
    if regrid is None:
        raise ValueError(f"No Bruker RecoRegridN density metadata found for {exp_dir}")

    points = int(regrid["points_per_interleave"])
    interleaves = int(regrid["interleaves"])
    dim = int(regrid["dim"])
    traj_dims = len(tuple(int(v) for v in spatial_shape))
    if traj_dims not in (2, 3):
        raise ValueError("spatial_shape must describe a 2D or 3D Bruker regrid target")
    traj_for_density = np.asarray(trajectory, dtype=np.float32)[:, :traj_dims]
    correction_type = _bruker_density_correction_type(reco, regrid)
    source_path: Path | None = None
    source = "trajectory"

    if correction_type == "jacobian":
        candidate = _resolve_local_bruker_file(exp_dir, regrid.get("density_file"), "trajDC")
        if candidate.exists():
            source_path = candidate
            density_traj = _read_regrid_real_trajectory_file(candidate, points=points, interleaves=interleaves, dim=dim)
            density = _density_from_jacobian_trajectory(
                density_traj,
                points_per_interleave=points,
                interleaves=interleaves,
                spatial_shape=spatial_shape,
            )
            source = "trajDC"
        else:
            density = estimate_radial_density(traj_for_density, spatial_shape=spatial_shape)
            source = "trajectory_fallback"
    elif correction_type == "rho":
        density = estimate_radial_density(traj_for_density, spatial_shape=spatial_shape)
    else:
        density = estimate_radial_density(traj_for_density, spatial_shape=spatial_shape)
        source = "trajectory_fallback"

    meta = {
        "correction_type": correction_type,
        "source": source,
        "density_path": None if source_path is None else str(source_path),
        "density_samples": int(density.shape[0]),
        "normalized_to_spatial_pixels": int(np.prod(spatial_shape)),
    }
    return np.asarray(density, dtype=np.float32), meta


def _first_n_ints(value: object, count: int, default: tuple[int, ...] | None = None) -> tuple[int, ...] | None:
    vals = [int(round(v)) for v in _flatten_numeric(value)]
    if len(vals) >= int(count):
        return tuple(int(v) for v in vals[: int(count)])
    if len(vals) == 1 and int(count) > 0:
        return tuple(int(vals[0]) for _ in range(int(count)))
    return default


def _first_two_ints(value: object, default: tuple[int, int] | None = None) -> tuple[int, int] | None:
    vals = _first_n_ints(value, 2, default)
    if vals is None:
        return None
    return (int(vals[0]), int(vals[1]))


def _first_finite_float(value: object) -> float | None:
    for val in _flatten_numeric(value):
        val_f = float(val)
        if np.isfinite(val_f):
            return val_f
    return None


def bruker_vendor_regrid_options(
    reco: dict,
    *,
    spatial_shape: tuple[int, ...],
) -> tuple[dict, dict]:
    """Build optional MRax regridding options that mirror ParaVision image steps.

    The returned options can be passed to :func:`mrax.reco.variational.reconstruct_regrid`.
    They implement the reproducible parts of Bruker's ``RecoStageNodes`` chain:
    oversampled regridding, phase checkerboarding, center cutoff, approximate
    Kaiser-Bessel deapodization, image support masking, RSS coil combination, and
    magnitude output.
    """

    final_shape = tuple(int(v) for v in spatial_shape)
    spatial_ndim = len(final_shape)
    if spatial_ndim not in (2, 3):
        raise ValueError("spatial_shape must be 2D or 3D for Bruker regrid options")
    grid_shape = _first_n_ints(reco.get("RecoRegridNFtSize"), spatial_ndim, final_shape) or final_shape
    offset = _first_n_ints(reco.get("RecoRegridNOffset"), spatial_ndim)
    if offset is None and grid_shape != final_shape:
        offset = tuple((int(grid) - int(final)) // 2 for grid, final in zip(grid_shape, final_shape))
    stage_text = _stage_node_text(reco)

    kernel_width = _first_finite_float(reco.get("RecoRegridNKernelWidth"))
    shape_parameter = _first_finite_float(reco.get("RecoRegridNKernelShapeParameter"))
    apodization = None
    if "RecoRegridApodCorrFilter" in stage_text and kernel_width is not None and shape_parameter is not None:
        apodization = RegridApodizationCorrection(
            kernel_width=float(kernel_width),
            shape_parameter=float(shape_parameter),
            full_shape=grid_shape,
            crop_offset=offset,
        )

    opts = {
        "grid_shape": grid_shape,
        "crop_offset": offset,
        "phase_correction": "checkerboard" if "RecoPhasCorrFilter" in stage_text else None,
        "apodization_correction": apodization,
        "image_mask": "ellipse" if "RecoMaskFilter" in stage_text else None,
        "coil_combine": "rss" if "RecoSumOfSquaresFilter" in stage_text else "sum",
        "magnitude": "RecoMagnitudeFilter" in stage_text,
    }
    meta = {
        "grid_shape": list(grid_shape),
        "crop_offset": None if offset is None else list(offset),
        "phase_correction": opts["phase_correction"],
        "apodization_correction": None
        if apodization is None
        else {
            "type": "kaiser_bessel",
            "kernel_width": float(kernel_width),
            "shape_parameter": float(shape_parameter),
            "strength": float(apodization.strength),
        },
        "image_mask": opts["image_mask"],
        "coil_combine": opts["coil_combine"],
        "magnitude": bool(opts["magnitude"]),
        "implemented_steps": [
            step
            for step, enabled in (
                ("oversampled_regrid", grid_shape != final_shape),
                ("phase_checkerboard", opts["phase_correction"] is not None),
                ("cutoff", offset is not None and grid_shape != final_shape),
                ("apodization_correction", apodization is not None),
                ("mask", opts["image_mask"] is not None),
                ("sum_of_squares", opts["coil_combine"] == "rss"),
                ("magnitude", bool(opts["magnitude"])),
            )
            if enabled
        ],
    }
    return opts, meta


def _non_cartesian_segment_keep_indices(
    *,
    raw_points_per_interleave: int,
    kept_points_per_interleave: int,
    interleaves: int,
    method: dict,
    reco: dict | None = None,
) -> np.ndarray:
    """Return raw per-object indices that match ParaVision's retained trajectory."""

    segment = int(raw_points_per_interleave)
    kept = int(kept_points_per_interleave)
    if segment <= 0 or kept <= 0 or int(interleaves) <= 0:
        raise ValueError("Non-Cartesian raw segment sizes must be positive")
    if kept > segment:
        raise ValueError("Kept non-Cartesian points cannot exceed raw points per interleave")

    stab = _bruker_stab_corr_metadata(reco or {})
    if stab and _as_positive_int(stab.get("echoSize")) == kept:
        nav = _as_positive_int(stab.get("navSize"), 0) or 0
        pre = _as_positive_int(stab.get("junkSize"), 0) or 0
        post = _as_positive_int(stab.get("junkEndSize"), None)
    else:
        nav = _as_positive_int(method.get("PVM_SpiralNavSize"), 0) or 0
        pre = _as_positive_int(method.get("PVM_SpiralPreSize"), 0) or 0
        post = _as_positive_int(method.get("PVM_SpiralPostSize"), None)
    start = int(nav) + int(pre)
    if start + kept > segment:
        if post is not None and int(post) + kept <= segment:
            start = segment - int(post) - kept
        else:
            start = max(segment - kept, 0)

    indices: list[int] = []
    for interleave_idx in range(int(interleaves)):
        offset = interleave_idx * segment
        indices.extend(range(offset + start, offset + start + kept))
    return np.asarray(indices, dtype=int)


def _bruker_stab_corr_metadata(reco: dict) -> dict:
    """Extract navigator, junk, and segment metadata from RecoStabCorrFilter nodes."""

    for text in re.findall(r"RecoStabCorrFilter\s+\w+\{[^}]*\}", _stage_node_text(reco)):
        if not text:
            continue

        meta: dict[str, object] = {}
        for key in ("navSize", "junkSize", "echoSize", "junkEndSize", "nEchoes", "nSegments"):
            match = re.search(rf"{key}\s*=\s*([-+]?\d+)", text)
            if match:
                meta[key] = int(match.group(1))
        for key in ("useNav", "drift", "amplitude"):
            match = re.search(rf"{key}\s*=\s*(true|false)", text, re.I)
            if match:
                meta[key] = match.group(1).lower() == "true"
        return meta
    return {}


def _bruker_epi_stage_has_filter(reco: dict, name: str) -> bool:
    return str(name) in _stage_node_text(reco)


def _bruker_b0_correction_path(exp_dir: Path, reco: dict) -> Path | None:
    text = _stage_node_text(reco)
    if "TRAJ_B0CORR" not in text and not (exp_dir / "b0").exists():
        return None
    match = re.search(r"TRAJ_B0CORR_Q\d+\{[^}]*filename\s*=\s*\"([^\"]+)\"", text)
    if match:
        return _resolve_local_bruker_file(exp_dir, match.group(1), "b0")
    candidate = exp_dir / "b0"
    return candidate if candidate.exists() else None


def _read_bruker_complex_correction(
    path: Path,
    *,
    points_per_interleave: int,
    interleaves: int,
) -> np.ndarray:
    expected = int(points_per_interleave) * int(interleaves)
    raw = np.fromfile(path, dtype="<f8")
    if int(raw.size) < expected * 2:
        raise ValueError(f"Bruker correction file {path} has {raw.size} values, expected at least {expected * 2}.")
    values = raw[: expected * 2]
    correction = values[0::2] + 1j * values[1::2]
    return correction.reshape((int(interleaves), int(points_per_interleave))).reshape(expected).astype(np.complex64)


def _apply_bruker_b0_correction(
    samples: np.ndarray,
    *,
    exp_dir: Path,
    reco: dict,
    regrid_meta: dict,
) -> tuple[np.ndarray, dict]:
    """Apply ParaVision's per-sample complex B0/demodulation correction when present."""

    correction_path = _bruker_b0_correction_path(exp_dir, reco)
    if correction_path is None or not correction_path.exists():
        return np.asarray(samples), {"applied": False, "reason": "missing_b0_correction"}

    correction = _read_bruker_complex_correction(
        correction_path,
        points_per_interleave=int(regrid_meta["points_per_interleave"]),
        interleaves=int(regrid_meta["interleaves"]),
    )
    arr = np.asarray(samples)
    if int(arr.shape[-1]) != int(correction.shape[0]):
        raise ValueError(
            f"B0 correction has {correction.shape[0]} samples but k-space has {arr.shape[-1]} samples."
        )
    return arr * correction.reshape((1,) * (arr.ndim - 1) + (correction.shape[0],)), {
        "applied": True,
        "path": str(correction_path),
        "samples": int(correction.shape[0]),
    }


def _read_bruker_pv360_raw_job_blocks(
    raw_path: Path,
    dtype: np.dtype,
    *,
    method: dict,
    acqp: dict,
) -> tuple[np.ndarray, dict] | None:
    """Read PV360 ``rawdata.job*`` blocks as ``(scan, coil, complex_point)``."""

    jobs = acqp.get("ACQ_jobs")
    if not isinstance(jobs, (list, tuple)) or len(jobs) < 1:
        return None

    job_scan_size = _as_positive_int(jobs[0])
    if job_scan_size is None or job_scan_size % 2 != 0:
        return None

    coil_count = _as_positive_int(method.get("PVM_EncNReceivers") or method.get("PVM_NReceivers"), 1)
    if coil_count is None or coil_count <= 0:
        return None

    raw = np.fromfile(raw_path, dtype=dtype)
    block_width = int(job_scan_size) * int(coil_count)
    if block_width <= 0 or raw.size < block_width:
        return None
    scan_count = int(raw.size) // block_width
    if scan_count <= 0:
        return None

    usable = raw[: scan_count * block_width].reshape((scan_count, int(coil_count), int(job_scan_size)))
    real = usable[:, :, 0::2].astype(np.float32)
    imag = usable[:, :, 1::2].astype(np.float32)
    blocks = real + 1j * imag
    return blocks, {
        "storage": "pv360_raw_job",
        "job_scan_size": int(job_scan_size),
        "scan_count": int(scan_count),
        "coil_count": int(coil_count),
        "raw_points_per_scan": int(job_scan_size // 2),
    }


def _split_bruker_sort_maps(reco: dict, interleaves: int, frame_count: int) -> tuple[np.ndarray, np.ndarray, dict]:
    sort_size_vals = np.asarray(_flatten_numeric(reco.get("RecoSortSize")), dtype=int).ravel()
    sort_maps_vals = np.asarray(_flatten_numeric(reco.get("RecoSortMaps")), dtype=int).ravel()
    interleave_map = np.arange(int(interleaves), dtype=int)
    frame_map = np.arange(int(frame_count), dtype=int)
    applied = False

    if sort_size_vals.size >= 2 and sort_maps_vals.size >= int(sort_size_vals[0] + sort_size_vals[1]):
        if int(sort_size_vals[0]) == int(interleaves) and int(sort_size_vals[1]) == int(frame_count):
            maybe_interleave = sort_maps_vals[: int(interleaves)]
            maybe_frame = sort_maps_vals[int(interleaves) : int(interleaves) + int(frame_count)]
            if np.array_equal(np.sort(maybe_interleave), np.arange(int(interleaves))):
                interleave_map = maybe_interleave.astype(int)
                applied = True
            if np.array_equal(np.sort(maybe_frame), np.arange(int(frame_count))):
                frame_map = maybe_frame.astype(int)
                applied = True

    def _compact_map(values: np.ndarray) -> list[int] | dict:
        vals = np.asarray(values, dtype=int).ravel()
        if vals.size <= 256:
            return vals.tolist()
        identity = bool(np.array_equal(vals, np.arange(vals.size, dtype=int)))
        return {
            "size": int(vals.size),
            "identity": identity,
            "head": vals[:8].tolist(),
            "tail": vals[-8:].tolist(),
        }

    return interleave_map, frame_map, {
        "sort_size": sort_size_vals.tolist(),
        "sort_maps_applied": bool(applied),
        "interleave_map": _compact_map(interleave_map),
        "frame_map": _compact_map(frame_map),
    }


def _bruker_inverse_fid_active(reco: dict) -> bool:
    return "RecoInverseFidFilter" in _stage_node_text(reco)


def _parse_bruker_inverse_fid_delay(reco: dict) -> float | None:
    match = re.search(
        r"RecoInverseFidFilter\s+\w+\{[^}]*acqDelay\s*=\s*([-+]?\d+(?:\.\d+)?)",
        _stage_node_text(reco),
    )
    if not match:
        return None
    return float(match.group(1))


def _parse_bruker_inverse_fid_readout_shift(reco: dict) -> float | None:
    match = re.search(
        (
            r"RecoFTShiftFilter\s+\w+\{[^}]*winDirection\s*=\s*0"
            r"[^}]*shift\s*=\s*([-+]?\d+(?:\.\d+)?)"
        ),
        _stage_node_text(reco),
    )
    if not match:
        return None
    return float(match.group(1))


def _apply_bruker_inverse_fid_profile_approximation(
    blocks: np.ndarray,
    *,
    expected_scans: int,
    kept_points_per_interleave: int,
    reco: dict,
) -> tuple[np.ndarray | None, dict]:
    """Approximate ParaVision's ZTE ``RecoInverseFidFilter`` profile synthesis.

    ZTE raw profiles are time-domain FIDs. ParaVision shuffles profiles and applies
    ``RecoInverseFidFilter`` followed by a half-sample shift and a 1D inverse FT before
    the 3D radial regrid step. The proprietary inverse-FID regularization is not fully
    documented, but transforming the shuffled FID branch to the readout-frequency domain
    recovers the object structure and avoids treating time samples as k-space samples.
    """

    if not _bruker_inverse_fid_active(reco):
        return None, {"applied": False}

    arr = np.asarray(blocks)
    scans = int(expected_scans)
    kept = int(kept_points_per_interleave)
    raw_points = int(arr.shape[-1])
    if scans <= 0 or kept <= 0 or raw_points < kept:
        return None, {"applied": False, "reason": "invalid_inverse_fid_shape"}

    if int(arr.shape[0]) >= scans * 2:
        # The RECOPASS shuffle branch reduces the two stored profile groups to the
        # trajectory profile count. In validated ParaVision 360 3.6 ZTE data, the
        # second group matches the stored trajectory ordering after the inverse-FID
        # frequency transform.
        source_start = scans
        source_stop = scans * 2
        source_group = "second_half"
    else:
        source_start = 0
        source_stop = scans
        source_group = "first_available"

    source = arr[source_start:source_stop]
    if int(source.shape[0]) != scans:
        return None, {
            "applied": False,
            "reason": "not_enough_inverse_fid_source_profiles",
            "available_profiles": int(source.shape[0]),
            "expected_profiles": int(scans),
        }

    readout_shift = _parse_bruker_inverse_fid_readout_shift(reco)
    crop_start, crop_stop = readout_crop_bounds(raw_points, kept)
    try:
        profile = fid_profiles_to_kspace(
            source,
            transform="fft_ifftshift",
            output_points=kept,
            readout_shift=readout_shift,
            dtype=np.complex64,
        )
    except ValueError as exc:
        return None, {"applied": False, "reason": f"inverse_fid_profile_transform_failed: {exc}"}

    return profile, {
        "applied": True,
        "approximation": True,
        "source_group": source_group,
        "source_scan_start": int(source_start),
        "source_scan_stop": int(source_stop),
        "input_points_per_profile": int(raw_points),
        "output_points_per_profile": int(kept),
        "crop_start": int(crop_start),
        "crop_stop": int(crop_stop),
        "transform": "fft(ifftshift(fid_profile))",
        "recorded_acq_delay": _parse_bruker_inverse_fid_delay(reco),
        "readout_shift": readout_shift,
        "readout_shift_applied": bool(readout_shift is not None and readout_shift != 0.0),
    }


def _reshape_bruker_non_cartesian_blocks(
    blocks: np.ndarray,
    *,
    frame_count: int,
    trajectory_samples: int,
    method: dict,
    reco: dict,
    regrid_meta: dict,
) -> tuple[np.ndarray, dict]:
    """Sort PV360 raw job blocks into ``(frame, coil, sample)`` order."""

    arr = np.asarray(blocks)
    if arr.ndim != 3:
        raise ValueError("PV360 raw job blocks must have shape (scan, coil, point)")

    frames = int(frame_count)
    interleaves = int(regrid_meta["interleaves"])
    kept_per_interleave = int(regrid_meta["points_per_interleave"])
    expected_scans = frames * interleaves
    expected_kept = kept_per_interleave * interleaves
    if frames <= 0 or interleaves <= 0:
        raise ValueError("frame_count and interleaves must be positive for non-Cartesian sorting")
    if int(arr.shape[0]) < expected_scans:
        raise ValueError(f"PV360 raw job has {arr.shape[0]} scans, expected at least {expected_scans}")
    if int(trajectory_samples) != expected_kept:
        raise ValueError(
            f"Trajectory contains {trajectory_samples} samples but RecoRegridN describes {expected_kept}"
        )

    raw_per_interleave = int(arr.shape[-1])
    inverse_fid_profiles, inverse_fid_meta = _apply_bruker_inverse_fid_profile_approximation(
        arr,
        expected_scans=expected_scans,
        kept_points_per_interleave=kept_per_interleave,
        reco=reco,
    )
    if inverse_fid_profiles is None:
        keep = _non_cartesian_segment_keep_indices(
            raw_points_per_interleave=raw_per_interleave,
            kept_points_per_interleave=kept_per_interleave,
            interleaves=1,
            method=method,
            reco=reco,
        )
        raw = arr[:expected_scans].reshape((interleaves, frames, int(arr.shape[1]), raw_per_interleave))
        trim_start = int(keep[0]) if keep.size else None
        trim_stop = int(keep[-1] + 1) if keep.size else None
        discarded_points = int(raw_per_interleave - kept_per_interleave)
    else:
        keep = np.arange(kept_per_interleave, dtype=int)
        raw = inverse_fid_profiles.reshape((interleaves, frames, int(arr.shape[1]), kept_per_interleave))
        trim_start = int(inverse_fid_meta.get("crop_start")) if inverse_fid_meta.get("crop_start") is not None else None
        trim_stop = int(inverse_fid_meta.get("crop_stop")) if inverse_fid_meta.get("crop_stop") is not None else None
        discarded_points = int(raw_per_interleave - kept_per_interleave)
    interleave_map, frame_map, sort_meta = _split_bruker_sort_maps(reco, interleaves, frames)
    sorted_samples = np.zeros(
        (frames, int(arr.shape[1]), interleaves, kept_per_interleave),
        dtype=arr.dtype,
    )
    for acq_interleave in range(interleaves):
        dst_interleave = int(interleave_map[acq_interleave])
        for acq_frame in range(frames):
            dst_frame = int(frame_map[acq_frame])
            sorted_samples[dst_frame, :, dst_interleave, :] = raw[acq_interleave, acq_frame, :, :][:, keep]

    return sorted_samples.reshape((frames, int(arr.shape[1]), expected_kept)), {
        "raw_points_per_frame": int(raw_per_interleave * interleaves),
        "raw_points_per_interleave": int(raw_per_interleave),
        "kept_points_per_interleave": int(kept_per_interleave),
        "interleaves": int(interleaves),
        "raw_scan_count": int(arr.shape[0]),
        "used_scan_count": int(expected_scans),
        "ignored_scan_count": int(max(int(arr.shape[0]) - int(expected_scans), 0)),
        "discarded_points_per_interleave": discarded_points,
        "sample_order": "pv360_scan_channel_point",
        "trim_start": trim_start,
        "trim_stop": trim_stop,
        "profile_preprocessing": {
            "inverse_fid": inverse_fid_meta,
        },
        "stab_correction": _bruker_stab_corr_metadata(reco),
        "reco_sort": sort_meta,
    }


def _reshape_bruker_non_cartesian_samples(
    samples: np.ndarray,
    *,
    frame_count: int,
    coil_count: int,
    trajectory_samples: int,
    method: dict,
    reco: dict | None = None,
    regrid_meta: dict,
) -> tuple[np.ndarray, dict]:
    """Sort Bruker non-Cartesian raw samples into ``(frame, coil, sample)`` order."""

    frames = int(frame_count)
    coils = max(int(coil_count), 1)
    interleaves = int(regrid_meta["interleaves"])
    kept_per_interleave = int(regrid_meta["points_per_interleave"])
    expected_kept = kept_per_interleave * interleaves
    if int(trajectory_samples) != expected_kept:
        raise ValueError(
            f"Trajectory contains {trajectory_samples} samples but RecoRegridN describes {expected_kept}"
        )
    if frames <= 0:
        raise ValueError("frame_count must be positive for non-Cartesian sorting")

    arr = np.asarray(samples)
    raw_per_frame = int(arr.size) // (frames * coils)
    total = frames * raw_per_frame * coils
    if raw_per_frame <= 0 or total > int(arr.size):
        raise ValueError("Not enough raw samples for non-Cartesian sorting")
    if raw_per_frame % interleaves != 0:
        raise ValueError(
            f"Raw non-Cartesian object has {raw_per_frame} samples, not divisible by {interleaves} interleaves"
        )

    raw_per_interleave = raw_per_frame // interleaves
    keep = _non_cartesian_segment_keep_indices(
        raw_points_per_interleave=raw_per_interleave,
        kept_points_per_interleave=kept_per_interleave,
        interleaves=interleaves,
        method=method,
        reco=reco,
    )
    usable = arr[:total].reshape((frames, raw_per_frame, coils))
    sorted_samples = np.moveaxis(usable[:, keep, :], 2, 1)
    return sorted_samples, {
        "raw_points_per_frame": int(raw_per_frame),
        "raw_points_per_interleave": int(raw_per_interleave),
        "kept_points_per_interleave": int(kept_per_interleave),
        "interleaves": int(interleaves),
        "discarded_points_per_interleave": int(raw_per_interleave - kept_per_interleave),
        "sample_order": "frame_sample_coil",
    }


def _resolve_bruker_raw_frame_count(
    *,
    frame_count: int,
    spatial_shape: tuple[int, ...],
    method: dict,
    acqp: dict,
) -> int:
    """Infer the raw object count stored in a Bruker ``rawdata.job*`` file."""

    candidates: list[int] = [int(frame_count)]

    ni = acqp.get("NI")
    try:
        if ni is not None:
            candidates.append(int(ni))
    except (TypeError, ValueError):
        _LOG.debug("Ignored optional operation failure", exc_info=True)

    obj_order = method.get("PVM_ObjOrderList") or acqp.get("ACQ_obj_order")
    if isinstance(obj_order, list) and obj_order:
        candidates.append(int(len(obj_order)))

    nslices = _flatten_numeric(method.get("PVM_SPackArrNSlices") or acqp.get("NSLICES"))
    if nslices and all(v > 0 for v in nslices):
        candidates.append(int(round(sum(nslices))))

    spat_dim = str(method.get("PVM_SpatDimEnum", "") or "").upper()
    if len(spatial_shape) == 3 and spat_dim != "3D":
        candidates.append(int(frame_count) * int(spatial_shape[-1]))

    return max(candidate for candidate in candidates if candidate > 0)


def _resolve_bruker_frame_order(method: dict, acqp: dict, frame_count: int) -> np.ndarray | None:
    """Return acquisition-to-output frame mapping from Bruker object-order metadata."""

    obj_order = method.get("PVM_ObjOrderList") or acqp.get("ACQ_obj_order")
    if obj_order is None:
        return None

    try:
        order = np.asarray(obj_order, dtype=int).ravel()
    except Exception:
        return None
    if order.size == int(frame_count):
        if np.any(np.sort(order) != np.arange(int(frame_count), dtype=int)):
            return None
        return order

    if order.size <= 0 or int(frame_count) % int(order.size) != 0:
        return None
    if np.any(np.sort(order) != np.arange(int(order.size), dtype=int)):
        return None
    echo_images = _as_positive_int(method.get("PVM_NEchoImages"), 1)
    if echo_images is None or echo_images <= 1 or int(order.size) * int(echo_images) != int(frame_count):
        return None
    expanded = []
    for _src_slice, dst_slice in enumerate(order.tolist()):
        for echo_idx in range(int(echo_images)):
            expanded.append(int(echo_idx) * int(order.size) + int(dst_slice))
    return np.asarray(expanded, dtype=int)


def _reorder_bruker_frames(data: np.ndarray | None, frame_order: np.ndarray | None) -> np.ndarray | None:
    """Map acquisition-order Bruker frames into natural object order."""

    if data is None or frame_order is None:
        return data
    arr = np.asarray(data)
    if arr.ndim == 0 or int(arr.shape[0]) != int(frame_order.size):
        return arr
    reordered = np.zeros_like(arr)
    reordered[frame_order] = arr
    return reordered


def _pack_bruker_2d_frame_images(
    image: np.ndarray | None,
    *,
    frame_count: int,
    raw_frame_count: int,
    spatial_shape: tuple[int, ...],
    prefer_transpose: bool = False,
) -> np.ndarray | None:
    """Pack 2D per-object reconstructions into MRax ``(frame, x, y, z)`` layout."""

    if image is None or len(spatial_shape) != 3:
        return image
    arr = np.asarray(image)
    x, y, z = (int(spatial_shape[0]), int(spatial_shape[1]), int(spatial_shape[2]))
    expected_objects = int(frame_count) * z
    if (
        arr.ndim == 4
        and int(arr.shape[0]) == int(frame_count)
        and tuple(int(v) for v in arr.shape[1:]) == (x, y, z)
    ):
        return arr
    if arr.ndim != 3 or int(arr.shape[0]) != expected_objects or int(raw_frame_count) != expected_objects:
        return image
    if tuple(int(v) for v in arr.shape[1:]) == (x, y):
        frames = np.transpose(arr, (0, 2, 1)) if prefer_transpose else arr
    elif tuple(int(v) for v in arr.shape[1:]) == (y, x):
        frames = np.transpose(arr, (0, 2, 1))
    else:
        return image
    packed = frames.reshape((int(frame_count), z, x, y))
    return np.moveaxis(packed, 1, -1)


def _pack_bruker_2d_frame_kspace(
    kspace: np.ndarray | None,
    *,
    frame_count: int,
    raw_frame_count: int,
    spatial_shape: tuple[int, ...],
) -> np.ndarray | None:
    """Separate flattened 2D slice objects into frame and z k-space axes."""

    if kspace is None or len(spatial_shape) != 3:
        return kspace
    arr = np.asarray(kspace)
    z = int(spatial_shape[-1])
    expected_objects = int(frame_count) * z
    if arr.ndim < 3 or int(arr.shape[0]) != expected_objects or int(raw_frame_count) != expected_objects:
        return arr

    packed = arr.reshape((int(frame_count), z) + tuple(arr.shape[1:]))
    if arr.ndim >= 4:
        # Keep receiver coils immediately after frame, followed by z, ky, kx.
        packed = np.moveaxis(packed, 1, 2)
    return packed


def _encoding_step_destinations(enc_steps: Sequence[int], target_size: int) -> np.ndarray:
    """Map Bruker phase-encode steps to destination ky indices."""

    steps = np.asarray(enc_steps, dtype=int).ravel()
    if steps.size == 0:
        return np.zeros((0,), dtype=int)

    desired = max(int(target_size), 1)
    s_min = int(steps.min())
    s_max = int(steps.max())
    if s_min >= 0 and s_max < desired:
        offset = 0
    elif s_min >= -(desired // 2) and s_max <= (desired // 2) - 1:
        offset = desired // 2
    else:
        offset = -s_min
    return steps + int(offset)


def _reshape_bruker_standard_kspace(
    samples: np.ndarray,
    *,
    frame_count: int,
    coil_count: int,
    ky_count: int,
    kx_count: int,
    sample_order: str = "frame_phase_coil_read",
) -> np.ndarray:
    """Reshape Bruker multi-slice Cartesian raw data into ``frame, coil, ky, kx``."""

    total = int(frame_count) * int(ky_count) * int(coil_count) * int(kx_count)
    if int(samples.size) < total:
        raise ValueError(f"Expected at least {total} raw samples for standard Cartesian sorting, found {samples.size}.")
    usable = np.asarray(samples[:total])
    if int(coil_count) > 1:
        if sample_order == "phase_coil_frame_read":
            reshaped = usable.reshape((int(ky_count), int(coil_count), int(frame_count), int(kx_count)))
            return np.transpose(reshaped, (2, 1, 0, 3))
        if sample_order == "phase_frame_coil_read":
            reshaped = usable.reshape((int(ky_count), int(frame_count), int(coil_count), int(kx_count)))
            return np.transpose(reshaped, (1, 2, 0, 3))
        reshaped = usable.reshape((int(frame_count), int(ky_count), int(coil_count), int(kx_count)))
        return np.moveaxis(reshaped, 2, 1)
    if sample_order in {"phase_coil_frame_read", "phase_frame_coil_read"}:
        reshaped = usable.reshape((int(ky_count), int(frame_count), int(kx_count)))
        return np.transpose(reshaped, (1, 0, 2))
    return usable.reshape((int(frame_count), int(ky_count), int(kx_count)))


def _reshape_bruker_standard_2d_raw_job(
    blocks: np.ndarray,
    *,
    frame_count: int,
    raw_frame_count: int,
    slice_count: int,
    ky_count: int,
    kx_count: int,
    reco: dict,
) -> tuple[np.ndarray, dict]:
    """Sort PV360 Cartesian blocks stored as repetition, phase, slice, coil, read."""

    arr = np.asarray(blocks)
    frames = int(frame_count)
    slices = int(slice_count)
    ky = int(ky_count)
    kx = int(kx_count)
    expected_objects = frames * slices
    expected_scans = expected_objects * ky
    if arr.ndim != 3:
        raise ValueError("PV360 Cartesian raw blocks must have shape (scan, coil, read)")
    if frames <= 0 or slices <= 0 or ky <= 0 or kx <= 0:
        raise ValueError("PV360 Cartesian frame and encoding sizes must be positive")
    if int(raw_frame_count) != expected_objects:
        raise ValueError("PV360 Cartesian raw object count does not match frame_count * slice_count")
    if int(arr.shape[0]) < expected_scans or int(arr.shape[-1]) != kx:
        raise ValueError("PV360 Cartesian raw block shape does not match the recorded encoding sizes")

    sort_range = np.asarray(_flatten_numeric(reco.get("RecoSortRange")), dtype=int).ravel()
    sort_size = np.asarray(_flatten_numeric(reco.get("RecoSortSize")), dtype=int).ravel()
    if (
        sort_range.size < 2
        or sort_size.size < 2
        or tuple(sort_range[:2]) != (ky, slices)
        or tuple(sort_size[:2]) != (ky, slices)
    ):
        raise ValueError("PV360 Cartesian RecoSort metadata does not describe phase and slice ordering")

    phase_map, slice_map, sort_meta = _split_bruker_sort_maps(reco, ky, slices)
    phase_sources = np.argsort(phase_map)
    slice_sources = np.argsort(slice_map)
    raw = arr[:expected_scans].reshape((frames, ky, slices, int(arr.shape[1]), kx))
    sorted_raw = raw[:, phase_sources, ...][:, :, slice_sources, ...]
    kspace = np.transpose(sorted_raw, (0, 2, 3, 1, 4)).reshape(
        (expected_objects, int(arr.shape[1]), ky, kx)
    )
    return kspace, {
        "sample_order": "pv360_repetition_phase_slice_channel_read",
        "raw_scan_count": int(arr.shape[0]),
        "used_scan_count": int(expected_scans),
        "ignored_scan_count": int(max(int(arr.shape[0]) - expected_scans, 0)),
        "reco_sort": sort_meta,
    }


def _bruker_standard_2d_sample_order(method: dict, raw_frame_count: int) -> str:
    """Infer the 2D Cartesian raw sample order for Bruker multi-frame scans."""

    nspacks_vals = _flatten_numeric(method.get("PVM_NSPacks"))
    slices_vals = _flatten_numeric(method.get("PVM_SPackArrNSlices"))
    enc_zf_vals = _flatten_numeric(method.get("PVM_EncZf"))
    multi_rec = str(method.get("PVM_EncUseMultiRec") or "").strip().lower() == "yes"
    method_text = " ".join(
        str(method.get(key, "") or "")
        for key in ("Method", "PVM_Method", "ACQ_scan_name", "PVM_ScanSequence")
    ).lower()
    try:
        nspacks = int(round(nspacks_vals[0])) if nspacks_vals else 0
    except (TypeError, ValueError):
        nspacks = 0
    has_read_phase_zero_fill = any(float(v) > 1.0 for v in enc_zf_vals[:2])
    if (
        nspacks > 1
        and len(slices_vals) == nspacks
        and int(raw_frame_count) == nspacks
        and all(int(round(v)) == 1 for v in slices_vals)
    ):
        if not has_read_phase_zero_fill:
            return "phase_frame_coil_read"
        return "phase_coil_frame_read"
    if int(raw_frame_count) > 1 and any(
        keyword in method_text for keyword in ("flash", "fisp", "gre", "gradient_echo")
    ):
        return "frame_phase_coil_read"
    if multi_rec and int(raw_frame_count) > 1 and not has_read_phase_zero_fill:
        return "phase_frame_coil_read"
    return "frame_phase_coil_read"


def _reshape_bruker_3d_standard_kspace(
    samples: np.ndarray,
    *,
    frame_count: int,
    coil_count: int,
    kz_count: int,
    ky_count: int,
    kx_count: int,
) -> np.ndarray:
    """Reshape Bruker 3D Cartesian raw data stored as ``partition, phase, coil, read``."""

    total = int(frame_count) * int(kz_count) * int(ky_count) * int(coil_count) * int(kx_count)
    if int(samples.size) < total:
        raise ValueError(f"Expected at least {total} raw samples for 3D Cartesian sorting, found {samples.size}.")
    usable = np.asarray(samples[:total])
    if int(coil_count) > 1:
        reshaped = usable.reshape(
            (int(frame_count), int(kz_count), int(ky_count), int(coil_count), int(kx_count))
        )
        return np.moveaxis(reshaped, 3, 1)
    reshaped = usable.reshape((int(frame_count), int(kz_count), int(ky_count), int(kx_count)))
    return reshaped[:, None, ...]


def _reshape_bruker_rare_kspace(
    samples: np.ndarray,
    *,
    frame_count: int,
    coil_count: int,
    rare_factor: int,
    enc_steps: Sequence[int],
    target_ky: int,
    kx_count: int,
    sample_order: str = "shot_frame_echo_coil_read",
) -> np.ndarray:
    """Reshape Bruker RARE/TurboSE raw data into ``frame, coil, ky, kx``."""

    steps = np.asarray(enc_steps, dtype=int).ravel()
    etl = int(rare_factor)
    if etl <= 0:
        raise ValueError("rare_factor must be positive")
    if steps.size == 0 or steps.size % etl != 0:
        raise ValueError("RARE enc_steps must be non-empty and divisible by rare_factor")

    shot_count = steps.size // etl
    total = int(shot_count) * int(frame_count) * int(etl) * int(coil_count) * int(kx_count)
    if int(samples.size) < total:
        raise ValueError(f"Expected at least {total} raw samples for RARE sorting, found {samples.size}.")

    usable = np.asarray(samples[:total])
    if int(coil_count) > 1:
        if sample_order == "frame_shot_echo_coil_read":
            reshaped = usable.reshape(
                (int(frame_count), int(shot_count), int(etl), int(coil_count), int(kx_count))
            )
            reshaped = np.transpose(reshaped, (1, 0, 2, 3, 4))
        else:
            reshaped = usable.reshape(
                (int(shot_count), int(frame_count), int(etl), int(coil_count), int(kx_count))
            )
        lines = np.moveaxis(reshaped, 3, 2)  # -> (shot, frame, coil, echo, kx)
    else:
        if sample_order == "frame_shot_echo_coil_read":
            reshaped = usable.reshape((int(frame_count), int(shot_count), int(etl), int(kx_count)))
            reshaped = np.transpose(reshaped, (1, 0, 2, 3))
        else:
            reshaped = usable.reshape((int(shot_count), int(frame_count), int(etl), int(kx_count)))
        lines = reshaped[:, :, None, :, :]

    destinations = _encoding_step_destinations(steps, int(target_ky)).reshape((int(shot_count), int(etl)))
    out = np.zeros((int(frame_count), max(int(coil_count), 1), int(target_ky), int(kx_count)), dtype=usable.dtype)
    for shot_idx in range(int(shot_count)):
        for echo_idx in range(int(etl)):
            dst_ky = int(destinations[shot_idx, echo_idx])
            if 0 <= dst_ky < int(target_ky):
                out[:, :, dst_ky, :] = lines[shot_idx, :, :, echo_idx, :]
    return out[:, 0, ...] if int(coil_count) == 1 else out


def _bruker_rare_sample_order(method: dict, raw_frame_count: int) -> str:
    """Infer whether RARE frames are outermost or nested inside the shot loop."""

    method_name = str(method.get("Method") or method.get("ACQ_method") or "").upper()
    slices_vals = _flatten_numeric(method.get("PVM_SPackArrNSlices"))
    slice_count = int(round(sum(slices_vals))) if slices_vals else 0
    if int(raw_frame_count) > 1 and slice_count <= 1 and "RAREVTR" in method_name:
        return "frame_shot_echo_coil_read"
    return "shot_frame_echo_coil_read"


def _reshape_bruker_epi_raw_job(
    raw_path: Path,
    dtype: np.dtype,
    *,
    frame_count: int,
    method: dict,
    reco: dict,
    acqp: dict,
) -> tuple[np.ndarray, dict]:
    """Sort PV360 Bruker EPI raw blocks into Cartesian ``(frame, coil, z, y, x)`` k-space.

    ParaVision EPI stores each partition/shot as a raw FID block containing
    navigator/correction echoes plus the acquired phase lines. The proprietary
    ghost and trajectory adjustments are recorded in the metadata; this importer
    performs the deterministic sorting, readout FFT, Bruker zero-fill/cutoff
    layout, and receiver corrections while surfacing the unresolved EPI-specific
    filters in metadata.
    """

    jobs = acqp.get("ACQ_jobs")
    if not isinstance(jobs, (list, tuple)) or len(jobs) < 1:
        raise ValueError("Bruker EPI raw import requires ACQ_jobs metadata.")

    job_scan_size = _as_positive_int(jobs[0])
    coil_count = _as_positive_int(method.get("PVM_EncNReceivers") or method.get("PVM_NReceivers"), 1)
    if job_scan_size is None or job_scan_size % 2 != 0 or coil_count is None or coil_count <= 0:
        raise ValueError("Invalid Bruker EPI raw block geometry.")

    epi_matrix = np.asarray(_flatten_numeric(method.get("PVM_EpiMatrix")), dtype=int).ravel()
    ft_size = np.asarray(_flatten_numeric(reco.get("RECO_ft_size") or reco.get("RECO_size")), dtype=int).ravel()
    if epi_matrix.size < 2 and ft_size.size < 2:
        raise ValueError("Bruker EPI import requires PVM_EpiMatrix or RECO_ft_size.")

    readout = int(epi_matrix[0]) if epi_matrix.size >= 1 and int(epi_matrix[0]) > 0 else int(ft_size[0])
    acquired_phase = int(epi_matrix[1]) if epi_matrix.size >= 2 and int(epi_matrix[1]) > 0 else None
    if epi_matrix.size >= 3 and int(epi_matrix[2]) > 0:
        partition_count = int(epi_matrix[2])
    elif ft_size.size >= 3 and int(ft_size[2]) > 0:
        partition_count = int(ft_size[2])
    else:
        partition_count = 1
    if readout <= 0 or partition_count <= 0:
        raise ValueError("Invalid Bruker EPI matrix dimensions.")

    shots = _as_positive_int(method.get("PVM_EpiNShots"), 1)
    echoes = _as_positive_int(method.get("PVM_EpiNEchoes"))
    raw_points_per_scan = int(job_scan_size) // 2
    if echoes is None:
        if readout <= 0 or raw_points_per_scan % readout != 0:
            raise ValueError("Cannot infer Bruker EPI echo count from raw block size.")
        echoes = raw_points_per_scan // readout
    if shots is None or shots <= 0 or echoes <= 0:
        raise ValueError("Invalid Bruker EPI shot/echo metadata.")
    if raw_points_per_scan < int(echoes) * int(readout):
        raise ValueError("Bruker EPI raw block is smaller than PVM_EpiNEchoes * PVM_EpiMatrix[0].")

    if acquired_phase is None:
        acquired_phase = int(shots) * int(echoes)
    effective_echoes = max(1, int(round(acquired_phase / int(shots))))
    effective_echoes = min(effective_echoes, int(echoes))
    echo_start = effective_echoes if int(echoes) >= 2 * effective_echoes else 0
    if echo_start + effective_echoes > int(echoes):
        echo_start = max(0, int(echoes) - effective_echoes)

    if ft_size.size >= 3 and np.all(ft_size[:3] > 0):
        kx_full, ky_full, kz_full = (int(ft_size[0]), int(ft_size[1]), int(ft_size[2]))
    elif ft_size.size >= 2 and np.all(ft_size[:2] > 0):
        kx_full, ky_full, kz_full = (int(ft_size[0]), int(ft_size[1]), int(partition_count))
    else:
        sort_range = np.asarray(_flatten_numeric(reco.get("RecoSortRange")), dtype=int).ravel()
        ky_full = int(sort_range[0] // partition_count) if sort_range.size else int(acquired_phase)
        kx_full, kz_full = int(readout), int(partition_count)
    if kx_full <= 0 or ky_full <= 0 or kz_full <= 0:
        raise ValueError("Invalid Bruker EPI reconstruction grid.")

    file_words = raw_path.stat().st_size // np.dtype(dtype).itemsize
    block_width = int(job_scan_size) * int(coil_count)
    scan_count = int(file_words) // block_width
    scans_per_frame = int(partition_count) * int(shots)
    required_scans = int(frame_count) * scans_per_frame
    if scan_count < required_scans:
        raise ValueError(
            f"Bruker EPI raw file has {scan_count} scans, expected at least {required_scans} "
            f"for {frame_count} frames."
        )

    sort_size_vals = np.asarray(_flatten_numeric(reco.get("RecoSortSize")), dtype=int).ravel()
    sort_maps_vals = np.asarray(_flatten_numeric(reco.get("RecoSortMaps")), dtype=int).ravel()
    line_count = int(sort_size_vals[0]) if sort_size_vals.size >= 1 and int(sort_size_vals[0]) > 0 else scans_per_frame * effective_echoes
    line_count = min(line_count, scans_per_frame * effective_echoes)
    if sort_maps_vals.size >= line_count:
        destinations = sort_maps_vals[:line_count].astype(int)
    else:
        destinations = np.arange(line_count, dtype=int)
    valid_destinations = (destinations >= 0) & (destinations < kz_full * ky_full)

    raw = np.memmap(
        raw_path,
        dtype=dtype,
        mode="r",
        shape=(scan_count, int(coil_count), int(job_scan_size)),
    )
    kspace = np.zeros((int(frame_count), int(coil_count), kz_full, ky_full, kx_full), dtype=np.complex64)
    points_used = int(echoes) * int(readout)
    prefix_nav_points, prefix_nav_meta = _bruker_epi_prefix_nav_points(method, int(readout), reco=reco)
    read_start = max(0, kx_full - int(readout))
    read_stop = min(kx_full, read_start + int(readout))
    source_read_stop = max(0, read_stop - read_start)

    for frame_idx in range(int(frame_count)):
        scan_start = frame_idx * scans_per_frame
        scan_stop = scan_start + scans_per_frame
        word_start = 2 * int(prefix_nav_points)
        word_stop = 2 * int(prefix_nav_points + points_used)
        if int(job_scan_size) < word_stop:
            raise ValueError("Bruker EPI raw block is smaller than prefix navigator plus image echo train.")
        frame_words = np.asarray(raw[scan_start:scan_stop, :, word_start:word_stop])
        real = frame_words[:, :, 0::2].astype(np.float32, copy=False)
        imag = frame_words[:, :, 1::2].astype(np.float32, copy=False)
        frame_samples = real + 1j * imag
        frame_samples = frame_samples.reshape(
            (int(partition_count), int(shots), int(coil_count), int(echoes), int(readout))
        )
        frame_samples = np.moveaxis(frame_samples, 2, 0)  # -> (coil, z, shot, echo, read)
        acquired = frame_samples[:, :, :, echo_start : echo_start + effective_echoes, :]
        acquired = acquired.reshape((int(coil_count), scans_per_frame * effective_echoes, int(readout)))
        acquired = acquired[:, :line_count, :]
        readout_kspace = np.fft.fftshift(
            np.fft.fft(np.fft.ifftshift(acquired, axes=-1), axis=-1),
            axes=-1,
        ).astype(np.complex64, copy=False)

        frame_flat = kspace[frame_idx].reshape((int(coil_count), kz_full * ky_full, kx_full))
        if source_read_stop > 0:
            frame_flat[:, destinations[valid_destinations], read_start:read_stop] = readout_kspace[
                :, valid_destinations, :source_read_stop
            ]

    return kspace, {
        "storage": "pv360_raw_job",
        "scan_count": int(scan_count),
        "job_scan_size": int(job_scan_size),
        "coil_count": int(coil_count),
        "raw_points_per_scan": raw_points_per_scan,
        "frame_count": int(frame_count),
        "scans_per_frame": int(scans_per_frame),
        "scan_order": "frame_partition_shot",
        "grid_order": "frame_coil_slice_phase_read",
        "epi_matrix": epi_matrix[:3].astype(int).tolist() if epi_matrix.size >= 3 else None,
        "shots": int(shots),
        "raw_echoes": int(echoes),
        "effective_echoes": int(effective_echoes),
        "echo_start": int(echo_start),
        "line_count": int(line_count),
        "valid_line_count": int(np.sum(valid_destinations)),
        "prefix_navigator": prefix_nav_meta,
        "readout_fft": True,
        "readout_placement": {"start": int(read_start), "stop": int(read_stop)},
        "sort_maps": {
            "source": "RecoSortMaps" if sort_maps_vals.size >= line_count else "sequential",
            "count": int(destinations.size),
            "target_size": int(kz_full * ky_full),
        },
        "approximations": [
            "uses acquired EPI echo train selected from raw blocks",
            "does not yet apply vendor ghost/stability navigator correction",
            "does not yet apply PVM_EpiTrajAdjkx/PVM_EpiTrajAdjb0 readout regridding",
        ],
    }


def _reshape_bruker_epi_phase_correction(method: dict, coil_count: int) -> tuple[np.ndarray | None, dict]:
    vals = np.asarray(_flatten_numeric(method.get("PVM_EpiPhaseCorrection")), dtype=np.float32).ravel()
    coils = int(coil_count)
    if vals.size == 0 or coils <= 0:
        return None, {"applied": False, "reason": "missing_PVM_EpiPhaseCorrection"}

    epi_vals = np.asarray(_flatten_numeric(method.get("PVM_EpiMatrix")), dtype=int).ravel()
    is_2d_epi = epi_vals.size < 3 or int(epi_vals[2]) <= 0
    if is_2d_epi and vals.size % (coils * 2) == 0:
        coeffs = vals.reshape((coils, vals.size // (coils * 2), 2))
        return coeffs, {
            "applied": bool(np.any(coeffs != 0)),
            "shape": [int(v) for v in coeffs.shape],
            "layout": "per_receiver_object_constant_linear",
            "units": "cycles_constant_and_linear_over_normalized_centered_readout",
        }

    order = vals.size // coils
    if order <= 0 or vals.size % coils != 0:
        return None, {"applied": False, "reason": "invalid_PVM_EpiPhaseCorrection_shape", "value_count": int(vals.size)}

    coeffs = vals.reshape((coils, order))
    return coeffs, {
        "applied": bool(np.any(coeffs != 0)),
        "shape": [int(v) for v in coeffs.shape],
        "units": "cycles_polynomial_over_normalized_centered_readout",
    }


def _apply_bruker_epi_phase_correction(
    samples: np.ndarray,
    coeffs: np.ndarray | None,
    *,
    readout: int,
    sign: float = 1.0,
    frame_index: int | None = None,
) -> np.ndarray:
    """Apply ParaVision's recorded EPI navigator/stability phase correction.

    ``PVM_EpiPhaseCorrection`` is stored as per-receiver polynomial coefficients
    for PV360 DTI-EPI and as per-receiver/per-object constant-linear pairs for
    validated ParaVision 360 3.6 2D EPI scans. In the Bruker EPI graph this
    belongs to the navigator/stability correction before readout gridding. ParaVision
    does not document the units publicly; the recorded coefficients match a
    cycles-valued phase ramp over centered readout sample coordinates.
    """

    if coeffs is None or not np.any(coeffs):
        return np.asarray(samples)

    arr = np.asarray(samples).copy()
    coils = int(arr.shape[2])
    coeff_arr = np.asarray(coeffs, dtype=np.float32)
    if coeff_arr.ndim == 3 and coeff_arr.shape[-1] == 2:
        if frame_index is not None and 0 <= int(frame_index) < int(coeff_arr.shape[1]):
            coeff_arr = coeff_arr[:, int(frame_index), :]
        else:
            coeff_arr = np.mean(coeff_arr, axis=1)
    if coeff_arr.ndim != 2:
        return arr
    order = int(coeff_arr.shape[1])
    # Empirically matches PV360 DTI-EPI metadata better than pixel-index units.
    x = np.linspace(-0.5, 0.5, int(readout), dtype=np.float32)
    powers = np.stack([x**degree for degree in range(order)], axis=0)
    for coil_idx in range(min(coils, int(coeff_arr.shape[0]))):
        phase_cycles = np.asarray(coeff_arr[coil_idx], dtype=np.float32) @ powers
        correction = np.exp(2j * np.pi * float(sign) * phase_cycles).astype(np.complex64)
        arr[:, :, coil_idx, :, :] *= correction.reshape((1, 1, 1, int(readout)))
    return arr


def _bruker_epi_is_2d_reco(method: dict, reco: dict) -> bool:
    """Return whether Bruker EPI metadata describes a 2D object stack."""

    epi_matrix = np.asarray(_flatten_numeric(method.get("PVM_EpiMatrix")), dtype=int).ravel()
    ft_size = np.asarray(_flatten_numeric(reco.get("RECO_ft_size") or reco.get("RECO_size")), dtype=int).ravel()
    reco_size = np.asarray(_flatten_numeric(reco.get("RECO_size")), dtype=int).ravel()
    if epi_matrix.size >= 3 and int(epi_matrix[2]) > 0:
        return False
    if ft_size.size >= 3 and int(ft_size[2]) > 1:
        return False
    if reco_size.size >= 3 and int(reco_size[2]) > 1:
        return False
    return True


def _apply_bruker_epi_alternating_readout_reversal(samples: np.ndarray, *, first_echo: int = 0) -> np.ndarray:
    """Reverse the alternating EPI readout rows described by ``REV_ALT_ROWS``."""

    arr = np.asarray(samples).copy()
    arr[..., int(first_echo) :: 2, :] = arr[..., int(first_echo) :: 2, ::-1]
    return arr


def _bruker_epi_prefix_nav_points(method: dict, readout: int, *, reco: dict | None = None) -> tuple[int, dict]:
    """Return raw complex points to skip before the EPI image echo train."""

    method_enabled = str(method.get("PVM_EpiPrefixNavYes", "") or "").strip().lower() in {"yes", "on", "true", "1"}
    method_nav_points = _as_positive_int(method.get("PVM_EpiPrefixNavSize"), 0) or 0
    read = int(readout)
    stab = _bruker_stab_corr_metadata(reco or {})
    stab_echo_size = _as_positive_int(stab.get("echoSize"), None)
    stab_matches_readout = bool(stab) and (read <= 0 or stab_echo_size is None or int(stab_echo_size) == read)
    stab_nav_points = (_as_positive_int(stab.get("navSize"), 0) or 0) if stab_matches_readout else 0
    stab_junk_points = (_as_positive_int(stab.get("junkSize"), 0) or 0) if stab_matches_readout else 0
    stab_use_nav = bool(stab.get("useNav", False)) if stab_matches_readout else False

    nav_points = int(method_nav_points) if method_enabled else 0
    nav_source = "PVM_EpiPrefixNavSize" if nav_points > 0 else None
    if nav_points <= 0 and stab_use_nav and int(stab_nav_points) > 0:
        nav_points = int(stab_nav_points)
        nav_source = "RecoStabCorrFilter.navSize"
    if method_enabled and int(stab_nav_points) > 0 and int(stab_nav_points) != int(nav_points):
        nav_points = int(stab_nav_points)
        nav_source = "RecoStabCorrFilter.navSize"

    skipped_points = int(nav_points) + int(stab_junk_points)
    if skipped_points <= 0:
        return 0, {
            "applied": False,
            "enabled": bool(method_enabled or stab_use_nav),
            "source_points": int(method_nav_points),
            "navigator_points": int(nav_points),
            "junk_points": int(stab_junk_points),
            "reason": "disabled_or_missing_prefix_navigator_and_stab_junk",
        }

    meta = {
        "applied": True,
        "enabled": bool(method_enabled or stab_use_nav),
        "source": nav_source or ("RecoStabCorrFilter.junkSize" if int(stab_junk_points) > 0 else None),
        "source_points": int(method_nav_points),
        "navigator_points": int(nav_points),
        "junk_points": int(stab_junk_points),
        "skipped_points": int(skipped_points),
        "navigator_echoes": (float(nav_points) / float(read)) if read > 0 and int(nav_points) > 0 else None,
    }
    if stab_matches_readout and stab:
        meta["stab_corr"] = {
            key: value
            for key, value in stab.items()
            if key in {"navSize", "junkSize", "echoSize", "nEchoes", "nSegments", "useNav"}
        }
    if read > 0 and int(nav_points) > 0 and int(nav_points) % read != 0:
        meta["warning"] = "Navigator prefix size is not readout-aligned; skipping the exact recorded raw point count."

    return int(skipped_points), meta


def _bruker_epi_rotate_shift(rotate_value: float, size: int) -> float:
    """Convert PV360 ``RECO_rotate`` values for EPI FT-shift filters."""

    if int(size) <= 0:
        return 0.0
    return -(float(rotate_value) - 0.5) * float(size)


def _bruker_epi_rotate_mode() -> str:
    mode = str(os.environ.get("MRAX_BRUKER_EPI_ROTATE_MODE", "default") or "default").strip().lower()
    valid = {
        "default",
        "off",
        "read_off",
        "phase_off",
        "slice_off",
        "flip_read",
        "flip_phase",
        "flip_slice",
        "flip_all",
    }
    if mode not in valid:
        warnings.warn(f"Unknown MRAX_BRUKER_EPI_ROTATE_MODE={mode!r}; using default.", RuntimeWarning, stacklevel=2)
        mode = "default"
    return mode


def _apply_bruker_epi_rotate_mode(read_shift: float, phase_shift: float, slice_shift: float, *, mode: str) -> tuple[float, float, float]:
    shifts = [float(read_shift), float(phase_shift), float(slice_shift)]
    if mode == "off":
        return 0.0, 0.0, 0.0
    if mode == "read_off":
        shifts[0] = 0.0
    elif mode == "phase_off":
        shifts[1] = 0.0
    elif mode == "slice_off":
        shifts[2] = 0.0
    elif mode == "flip_read":
        shifts[0] *= -1.0
    elif mode == "flip_phase":
        shifts[1] *= -1.0
    elif mode == "flip_slice":
        shifts[2] *= -1.0
    elif mode == "flip_all":
        shifts = [-value for value in shifts]
    return float(shifts[0]), float(shifts[1]), float(shifts[2])


def _bruker_epi_frame_rotate_shifts(
    reco: dict,
    *,
    frame_count: int,
    kx_full: int,
    ky_full: int,
    kz_full: int,
    spatial_ndim: int,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Resolve ``RECO_rotate`` into per-frame read/phase/slice EPI shifts."""

    frames = int(frame_count)
    read = np.zeros((frames,), dtype=np.float32)
    phase = np.zeros((frames,), dtype=np.float32)
    slice_ = np.zeros((frames,), dtype=np.float32)
    rotate_vals = np.asarray(_flatten_numeric(reco.get("RECO_rotate")), dtype=float).ravel()
    if frames <= 0 or int(spatial_ndim) <= 0 or rotate_vals.size < int(spatial_ndim):
        return read, phase, slice_, {
            "source": None if rotate_vals.size == 0 else rotate_vals.astype(float).tolist(),
            "mode": mode,
            "applied": False,
            "reason": "missing_or_invalid_RECO_rotate",
        }

    dims = int(np.clip(int(spatial_ndim), 1, 3))
    sizes = [int(kx_full), int(ky_full), int(kz_full)]
    if rotate_vals.size >= dims * frames:
        layout = "dimension_major_per_frame"
        rotate_by_dim = [rotate_vals[idx * frames : (idx + 1) * frames] for idx in range(dims)]
    else:
        layout = "single_value_per_dimension"
        rotate_by_dim = [np.full((frames,), float(rotate_vals[idx]), dtype=float) for idx in range(dims)]

    shift_by_dim = [
        np.asarray([_bruker_epi_rotate_shift(float(value), sizes[idx]) for value in rotate_by_dim[idx]], dtype=np.float32)
        for idx in range(dims)
    ]
    if dims >= 1:
        read = shift_by_dim[0]
    if dims >= 2:
        phase = shift_by_dim[1]
    if dims >= 3:
        slice_ = shift_by_dim[2]

    read2, phase2, slice2 = [], [], []
    for idx in range(frames):
        r, p, s = _apply_bruker_epi_rotate_mode(read[idx], phase[idx], slice_[idx], mode=mode)
        read2.append(r)
        phase2.append(p)
        slice2.append(s)
    read = np.asarray(read2, dtype=np.float32)
    phase = np.asarray(phase2, dtype=np.float32)
    slice_ = np.asarray(slice2, dtype=np.float32)
    applied = bool(np.any(read != 0.0) or np.any(phase != 0.0) or np.any(slice_ != 0.0))
    return read, phase, slice_, {
        "source": rotate_vals.astype(float).tolist(),
        "source_layout": layout,
        "mode": mode,
        "applied": applied,
        "spatial_ndim": dims,
        "read": read.astype(float).tolist(),
        "phase": phase.astype(float).tolist(),
        "slice": slice_.astype(float).tolist(),
    }


def _bruker_epi_pc_lin_mode() -> str:
    return "auto_epi_vendor"


def _bruker_epi_pc_lin_shifts(reco: dict, *, mode: str) -> tuple[tuple[float, float, float], dict]:
    """Convert opt-in RECO_pc_lin first-order terms to pixel-shift equivalents."""

    requested_mode = str(mode).strip().lower().replace("-", "_")
    mode_i = requested_mode
    pc_modes = [item.upper() for item in _as_reco_string_list(reco.get("RECO_pc_mode"))]
    vals = np.asarray(_flatten_numeric(reco.get("RECO_pc_lin")), dtype=np.float32)
    reco_size = np.asarray(_flatten_numeric(reco.get("RECO_size")), dtype=int).ravel()
    ft_size = np.asarray(_flatten_numeric(reco.get("RECO_ft_size")), dtype=int).ravel()
    reco_offset = np.asarray(_flatten_numeric(reco.get("RECO_offset")), dtype=int).ravel()
    if mode_i in {"off", "none"}:
        return (0.0, 0.0, 0.0), {"applied": False, "requested_mode": requested_mode, "effective_mode": "off", "mode": "off"}
    if vals.size < 4 or vals.size % 2 != 0:
        return (
            (0.0, 0.0, 0.0),
            {"applied": False, "requested_mode": requested_mode, "effective_mode": "off", "mode": mode_i, "reason": "missing_RECO_pc_lin"},
        )
    pairs = vals.reshape((-1, 2))[:3]
    if pairs.shape[0] < 3:
        if mode_i == "auto_epi_vendor":
            return (
                (0.0, 0.0, 0.0),
                {
                    "applied": False,
                    "requested_mode": requested_mode,
                    "effective_mode": "off",
                    "mode": requested_mode,
                    "source_mode": pc_modes,
                    "source_lin": pairs.astype(float).tolist(),
                    "reason": "2d_read_phase_RECO_pc_lin_not_auto_applied",
                },
            )
        pairs = np.pad(pairs, ((0, 3 - int(pairs.shape[0])), (0, 0)), mode="constant")
    if mode_i == "auto_epi_vendor":
        gate = {
            "slice_RECO_pc_mode_is_FIRST_ORDER_PC": len(pc_modes) >= 3 and pc_modes[2] == "FIRST_ORDER_PC",
            "has_RECO_pc_lin_slice_pair": pairs.shape[0] >= 3,
            "slice_delta_degrees_is_180": bool(pairs.shape[0] >= 3 and np.isclose(float(pairs[2, 1] - pairs[2, 0]), 180.0, atol=1e-3)),
            "has_reco_size_slice": reco_size.size >= 3,
            "has_ft_size_slice": ft_size.size >= 3,
            "has_reco_offset_slice": reco_offset.size >= 3,
        }
        if gate["has_reco_size_slice"] and gate["has_ft_size_slice"] and gate["has_reco_offset_slice"]:
            cutoff = int(ft_size[2]) - int(reco_size[2])
            gate.update(
                {
                    "positive_slice_cutoff": cutoff > 0,
                    "positive_reco_offset_slice": int(reco_offset[2]) > 0,
                    "symmetric_slice_cutoff": cutoff == 2 * int(reco_offset[2]),
                }
            )
        else:
            gate.update(
                {
                    "positive_slice_cutoff": False,
                    "positive_reco_offset_slice": False,
                    "symmetric_slice_cutoff": False,
                }
            )
        gate["passed"] = bool(all(gate.values()))
        if gate["passed"]:
            mode_i = "slice_only_flip"
            auto_reason = "pv360_epi_first_order_pc_lin_slice_180_selected_slice_only_flip"
            fallback_reason = None
        else:
            return (
                (0.0, 0.0, 0.0),
                {
                    "applied": False,
                    "requested_mode": requested_mode,
                    "effective_mode": "off",
                    "mode": requested_mode,
                    "auto_epi_vendor_gate": gate,
                    "auto_epi_vendor_fallback_reason": ",".join(
                        key for key, value in gate.items() if key != "passed" and not value
                    ),
                },
            )
    else:
        gate = None
        auto_reason = None
        fallback_reason = None

    shifts = []
    for axis_idx, pair in enumerate(pairs):
        active = axis_idx < len(pc_modes) and pc_modes[axis_idx] == "FIRST_ORDER_PC"
        delta_degrees = float(pair[1] - pair[0]) if active else 0.0
        shifts.append(delta_degrees / 360.0)

    if mode_i.startswith("read_only"):
        shifts[1] = 0.0
        shifts[2] = 0.0
    elif mode_i.startswith("phase_only"):
        shifts[0] = 0.0
        shifts[2] = 0.0
    elif mode_i.startswith("slice_only"):
        shifts[0] = 0.0
        shifts[1] = 0.0

    if mode_i.endswith("_flip"):
        shifts = [-value for value in shifts]
    applied = bool(any(abs(value) > 0.0 for value in shifts))
    return (float(shifts[0]), float(shifts[1]), float(shifts[2])), {
        "applied": applied,
        "requested_mode": requested_mode,
        "effective_mode": mode_i,
        "mode": mode_i,
        "source_mode": pc_modes,
        "source_lin": pairs.astype(float).tolist(),
        "shift_pixels": [float(value) for value in shifts],
        "normalization": "delta_degrees_div_360",
        "auto_epi_vendor_gate": gate,
        "auto_epi_vendor_decision_reason": auto_reason,
        "auto_epi_vendor_fallback_reason": fallback_reason,
    }


def _bruker_epi_b0_trajectory_phase_options(method: dict | None = None, reco: dict | None = None) -> dict:
    raw_env = os.environ.get("MRAX_BRUKER_EPI_B0_TRAJECTORY_PHASE")
    raw = str(raw_env if raw_env is not None else "auto").strip().lower()
    if raw in {"0", "false", "no", "off", "none"}:
        return {"enabled": False, "mode": "off", "source": "environment"}
    if raw in {"1", "true", "yes", "on", "phase"}:
        return {"enabled": True, "mode": "samplewise_phase", "source": "environment"}
    if raw == "auto":
        has_b0 = method is not None and bool(_flatten_numeric(method.get("PVM_EpiTrajAdjb0")))
        has_regrid = reco is not None and _bruker_epi_stage_has_filter(reco, "RecoRegridFilter")
        return {
            "enabled": bool(has_b0 and has_regrid),
            "mode": "samplewise_phase" if has_b0 and has_regrid else "off",
            "source": "auto_RecoRegridFilter",
        }
    warnings.warn(
        f"Unknown MRAX_BRUKER_EPI_B0_TRAJECTORY_PHASE={raw!r}; using auto.",
        RuntimeWarning,
        stacklevel=2,
    )
    return {"enabled": False, "mode": "off", "source": "environment_invalid"}


def _apply_bruker_epi_b0_trajectory_phase(
    samples: np.ndarray,
    method: dict,
    *,
    reco: dict | None = None,
    sign: float = -1.0,
    enabled: bool | None = None,
) -> tuple[np.ndarray, dict]:
    """Apply the recorded PV360 EPI B0 trajectory phase term when present."""

    options = _bruker_epi_b0_trajectory_phase_options(method, reco)
    if enabled is not None:
        options = {**options, "enabled": bool(enabled), "source": "call_argument"}

    arr = np.asarray(samples)
    if not bool(options["enabled"]):
        return arr, {
            **options,
            "applied": False,
            "reason": "disabled_b0_phase_is_part_of_RecoRegridFilter_model",
        }

    vals = np.asarray(_flatten_numeric(method.get("PVM_EpiTrajAdjb0")), dtype=np.float32).ravel()
    if vals.size != int(arr.shape[-1]):
        return arr, {**options, "applied": False, "reason": "missing_or_mismatched_PVM_EpiTrajAdjb0"}
    # ParaVision stores the B0 trajectory as a phase term for RecoRegridFilter.
    # The values in PV360 EPI method files are radians, not cycles.
    phase = np.exp(1j * float(sign) * vals).astype(np.complex64)
    return arr * phase.reshape((1,) * (arr.ndim - 1) + (int(vals.size),)), {
        **options,
        "applied": True,
        "samples": int(vals.size),
        "units": "radians",
        "source": "PVM_EpiTrajAdjb0",
        "sign": float(sign),
    }


def _bruker_epi_readout_regrid_options(method: dict | None = None, reco: dict | None = None) -> dict:
    mode = os.environ.get("MRAX_BRUKER_EPI_READOUT_REGRID_MODE")
    stage = os.environ.get("MRAX_BRUKER_EPI_READOUT_REGRID_STAGE")
    if mode is not None and str(mode).strip():
        mode_i = str(mode).strip().lower().replace("-", "_")
        default_stage = "after_reversal" if "partial_echo" in mode_i or "echo_centered_partial" in mode_i else "after_b0"
        stage_i = str(stage or default_stage).strip().lower().replace("-", "_")
        return {
            "mode": mode_i,
            "stage": stage_i,
            "enabled": mode_i not in {"off", "none", "false", "0"},
            "source": "environment",
            "kernel": "linear",
            "echo_polarity_aware": "alternating" in mode_i or "measured" in mode_i,
        }
    if method is not None and reco is not None:
        epi_vals = _flatten_numeric(method.get("PVM_EpiMatrix"))
        readout = int(round(epi_vals[0])) if epi_vals and int(round(epi_vals[0])) > 0 else None
        traj = _flatten_numeric(method.get("PVM_EpiTrajAdjkx"))
        if readout is not None and len(traj) == int(readout):
            acquired_phase = int(round(epi_vals[1])) if len(epi_vals) >= 2 and int(round(epi_vals[1])) > 0 else 0
            shots = _as_positive_int(method.get("PVM_EpiNShots"), 1) or 1
            echoes = _as_positive_int(method.get("PVM_EpiNEchoes"), acquired_phase)
            echoes_per_object = acquired_phase // int(shots) if acquired_phase > 0 else 0
            object_count = int(echoes) // max(int(echoes_per_object), 1) if echoes is not None else 1
            traj_arr = np.asarray(traj, dtype=np.float32)
            traj_span = float(np.max(traj_arr) - np.min(traj_arr)) if traj_arr.size else 0.0
            span_fraction = traj_span / float(max(int(readout) - 1, 1))
            is_3d_multishot = not _bruker_epi_is_2d_reco(method, reco) and int(shots) > 1
            if is_3d_multishot and span_fraction < 0.75:
                scale, scale_meta = _bruker_epi_partial_readout_regrid_scale(method)
                return {
                    "mode": "partial_echo_density_cubic",
                    "stage": "after_reversal",
                    "enabled": True,
                    "available": True,
                    "source": "auto_3d_multishot_epi_trajectory_span_guard",
                    "reason": "PVM_EpiTrajAdjkx_span_is_not_a_full_readout_grid",
                    "kernel": "cubic_zero_fill",
                    "echo_polarity_aware": False,
                    "density_compensation": True,
                    "echo_centered": True,
                    "trajectory_scale": float(scale),
                    "trajectory_scale_meta": scale_meta,
                    "object_count": int(object_count),
                    "shots": int(shots),
                    "trajectory_span": traj_span,
                    "trajectory_span_fraction": span_fraction,
                }
            if _bruker_epi_is_2d_reco(method, reco):
                if object_count > 1:
                    mode_i = "cubic_reverse"
                elif int(readout) != int(acquired_phase):
                    mode_i = "cubic"
                else:
                    mode_i = "linear"
                return {
                    "mode": mode_i,
                    "stage": "after_b0",
                    "enabled": True,
                    "source": "auto_2d_epi_PVM_EpiTrajAdjkx",
                    "kernel": "cubic" if "cubic" in mode_i else "linear",
                    "echo_polarity_aware": False,
                    "object_count": int(object_count),
                    "shots": int(shots),
                    "trajectory_span": traj_span,
                    "trajectory_span_fraction": span_fraction,
                }
            return {
                "mode": "measured_linear",
                "stage": "after_b0",
                "enabled": True,
                "source": "auto_RecoRegridFilter_PVM_EpiTrajAdjkx",
                "kernel": "linear",
                "echo_polarity_aware": True,
                "object_count": int(object_count),
                "shots": int(shots),
                "trajectory_span": traj_span,
                "trajectory_span_fraction": span_fraction,
            }
    return {
        "mode": "off",
        "stage": "after_b0",
        "enabled": False,
        "source": "default",
        "kernel": "linear",
        "echo_polarity_aware": False,
    }


def _bruker_epi_normalized_readout_trajectory(method: dict, readout: int, *, reverse: bool) -> tuple[np.ndarray | None, dict]:
    vals = np.asarray(_flatten_numeric(method.get("PVM_EpiTrajAdjkx")), dtype=np.float32).ravel()
    n = int(readout)
    if vals.size != n or n <= 2:
        return None, {"available": False, "reason": "missing_or_mismatched_PVM_EpiTrajAdjkx", "value_count": int(vals.size)}
    traj = -vals if reverse else vals
    order = np.argsort(traj)
    sorted_traj = traj[order].astype(np.float32, copy=False)
    unique_traj, unique_idx = np.unique(sorted_traj, return_index=True)
    if unique_traj.size != n:
        return None, {"available": False, "reason": "non_unique_PVM_EpiTrajAdjkx", "unique_count": int(unique_traj.size)}
    span = float(unique_traj[-1] - unique_traj[0])
    if span <= 0.0 or not np.isfinite(span):
        return None, {"available": False, "reason": "invalid_PVM_EpiTrajAdjkx_span"}
    normalized = (unique_traj - float(unique_traj[0])) / span
    target = np.linspace(0.0, 1.0, n, dtype=np.float32)
    return np.stack([normalized, target, order[unique_idx].astype(np.float32)], axis=0), {
        "available": True,
        "source": "PVM_EpiTrajAdjkx",
        "value_count": n,
        "reverse": bool(reverse),
        "trajectory_span": span,
    }


def _bruker_epi_partial_readout_regrid_scale(method: dict | None = None) -> tuple[float, dict]:
    raw_env = os.environ.get("MRAX_BRUKER_EPI_PARTIAL_READOUT_REGRID_SCALE")
    if raw_env is not None and str(raw_env).strip():
        try:
            scale = float(raw_env)
        except ValueError:
            warnings.warn(
                f"Invalid MRAX_BRUKER_EPI_PARTIAL_READOUT_REGRID_SCALE={raw_env!r}; using auto.",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            if np.isfinite(scale) and scale > 0.0:
                return scale, {"source": "environment"}
    # PV360's partial-span DTI-EPI trajectory behaves like an echo-centered ramp
    # trajectory, but Bruker's RecoRegridFilter also applies a small internal ramp
    # scale. This value matches the measured Bambooshark DTI-EPI geometry while
    # keeping the correction in raw readout trajectory space.
    return 1.04, {"source": "auto_partial_span_epi_default"}


def _interp_complex_same_grid(values: np.ndarray, x: np.ndarray, target: np.ndarray, *, kernel: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.complex64)
    flat = arr.reshape((-1, int(arr.shape[-1])))
    if kernel == "cubic":
        try:
            from scipy.interpolate import CubicSpline

            real = CubicSpline(x, flat.real, axis=-1, bc_type="natural")(target)
            imag = CubicSpline(x, flat.imag, axis=-1, bc_type="natural")(target)
            return (real + 1j * imag).reshape(arr.shape).astype(np.complex64, copy=False)
        except Exception:
            kernel = "linear"
    idx_hi = np.searchsorted(x, target, side="left")
    idx_hi = np.clip(idx_hi, 1, x.size - 1)
    idx_lo = idx_hi - 1
    denom = (x[idx_hi] - x[idx_lo]).astype(np.float32)
    weight = ((target - x[idx_lo]) / (denom + 1e-12)).astype(np.float32)
    out = (1.0 - weight[None, :]) * flat[:, idx_lo] + weight[None, :] * flat[:, idx_hi]
    return out.reshape(arr.shape).astype(np.complex64, copy=False)


def _interp_complex_to_grid(
    values: np.ndarray,
    x: np.ndarray,
    target: np.ndarray,
    *,
    fill: complex = 0j,
    kernel: str = "linear",
) -> np.ndarray:
    arr = np.asarray(values, dtype=np.complex64)
    x = np.asarray(x, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    flat = arr.reshape((-1, int(arr.shape[-1])))
    out = np.empty((flat.shape[0], int(target.size)), dtype=np.complex64)
    fill_value = complex(fill)
    if str(kernel).strip().lower() == "cubic" and x.size >= 4:
        inside = (target >= float(x[0])) & (target <= float(x[-1]))
        if np.any(inside):
            out[:] = fill_value
            try:
                from scipy.interpolate import CubicSpline

                real = CubicSpline(x, flat.real, axis=-1, bc_type="natural")(target[inside])
                imag = CubicSpline(x, flat.imag, axis=-1, bc_type="natural")(target[inside])
                out[:, inside] = real + 1j * imag
                return out.reshape(arr.shape[:-1] + (int(target.size),)).astype(np.complex64, copy=False)
            except Exception:
                _LOG.debug("Ignored optional operation failure", exc_info=True)
    for idx, row in enumerate(flat):
        out[idx] = np.interp(target, x, row.real, left=fill_value.real, right=fill_value.real) + 1j * np.interp(
            target,
            x,
            row.imag,
            left=fill_value.imag,
            right=fill_value.imag,
        )
    return out.reshape(arr.shape[:-1] + (int(target.size),)).astype(np.complex64, copy=False)


def _apply_bruker_epi_readout_regrid(
    samples: np.ndarray,
    method: dict,
    *,
    mode: str,
) -> tuple[np.ndarray, dict]:
    arr = np.asarray(samples)
    mode_i = str(mode).strip().lower().replace("-", "_")
    if mode_i in {"off", "none"}:
        return arr, {"applied": False, "mode": "off"}
    if mode_i in {
        "partial_echo",
        "partial_echo_cubic",
        "partial_echo_density",
        "partial_echo_density_cubic",
        "echo_centered_partial",
        "echo_centered_partial_cubic",
        "echo_centered_partial_density",
        "echo_centered_partial_density_cubic",
    }:
        vals = np.asarray(_flatten_numeric(method.get("PVM_EpiTrajAdjkx")), dtype=np.float32).ravel()
        n = int(arr.shape[-1])
        if vals.size != n or n <= 2:
            return arr, {"applied": False, "mode": mode_i, "reason": "missing_or_mismatched_PVM_EpiTrajAdjkx"}
        echo_position = _as_positive_int(method.get("PVM_EpiEchoPosition"), None)
        echo_index = int(echo_position) - 1 if echo_position is not None else n // 2
        if echo_index < 0 or echo_index >= n:
            return arr, {"applied": False, "mode": mode_i, "reason": "invalid_PVM_EpiEchoPosition"}
        scale, scale_meta = _bruker_epi_partial_readout_regrid_scale(method)
        target = np.arange(n, dtype=np.float32)
        x = (vals - float(vals[echo_index])) * float(scale) + float(n - 1) / 2.0
        order = np.argsort(x)
        x_sorted = x[order].astype(np.float32, copy=False)
        density_meta: dict = {"applied": False, "reason": "mode_without_density"}
        target_weights: np.ndarray | None = None
        if "density" in mode_i:
            weights, density_detail = _bruker_epi_readout_density_weights(method, n)
            if weights is not None:
                weight_sorted = np.asarray(weights, dtype=np.float32)[order]
                target_weights = np.interp(target, x_sorted, weight_sorted, left=0.0, right=0.0).astype(np.float32)
                density_meta = {
                    "applied": True,
                    "mode": "jacobian",
                    "weight_interpolation": "trajectory_to_cartesian_target",
                    **density_detail,
                }
            else:
                density_meta = {"applied": False, **density_detail}
        out = np.empty(arr.shape[:-1] + (n,), dtype=np.complex64)
        for echo_idx in range(int(arr.shape[-2])) if arr.ndim >= 2 else range(1):
            selector = [slice(None)] * arr.ndim
            if arr.ndim >= 2:
                selector[-2] = echo_idx
            block = np.take(arr[tuple(selector)], order, axis=-1)
            out_block = _interp_complex_to_grid(
                block,
                x_sorted,
                target,
                fill=0j,
                kernel="cubic" if "cubic" in mode_i else "linear",
            )
            if target_weights is not None:
                out_block = out_block * target_weights.reshape((1,) * (out_block.ndim - 1) + (n,))
            out[tuple(selector)] = out_block
        return out, {
            "applied": True,
            "mode": mode_i,
            "kernel": "cubic_zero_fill" if "cubic" in mode_i else "linear_zero_fill",
            "stage": "after_reversal",
            "source": "PVM_EpiTrajAdjkx",
            "echo_position": int(echo_position) if echo_position is not None else None,
            "echo_position_base": "one_based",
            "echo_index": int(echo_index),
            "trajectory_scale": float(scale),
            "trajectory_scale_meta": scale_meta,
            "trajectory_span": float(np.max(vals) - np.min(vals)),
            "target_range": [0, int(n - 1)],
            "mapped_range": [float(x_sorted[0]), float(x_sorted[-1])],
            "density_compensation": density_meta,
        }
    echo_polarity_aware = "measured" in mode_i or "alternating" in mode_i
    reverse_all = mode_i.endswith("_reverse") and not echo_polarity_aware
    kernel = "cubic" if "cubic" in mode_i else "linear"

    def _regrid_block(block: np.ndarray, *, reverse: bool) -> tuple[np.ndarray | None, dict]:
        trajectory, traj_meta = _bruker_epi_normalized_readout_trajectory(
            method,
            int(block.shape[-1]),
            reverse=reverse,
        )
        if trajectory is None:
            return None, traj_meta
        x = trajectory[0]
        target = trajectory[1]
        order = trajectory[2].astype(int)
        sorted_block = np.take(block, order, axis=-1)
        return _interp_complex_same_grid(sorted_block, x, target, kernel=kernel), traj_meta

    if echo_polarity_aware and arr.ndim >= 2:
        out = np.empty_like(arr, dtype=np.complex64)
        reverse_count = 0
        forward_count = 0
        last_meta: dict = {}
        for echo_idx in range(int(arr.shape[-2])):
            reverse = echo_idx % 2 == 0
            selector = [slice(None)] * arr.ndim
            selector[-2] = echo_idx
            regridded, traj_meta = _regrid_block(arr[tuple(selector)], reverse=reverse)
            if regridded is None:
                return arr, {"applied": False, "mode": mode_i, **traj_meta}
            out[tuple(selector)] = regridded
            last_meta = traj_meta
            if reverse:
                reverse_count += 1
            else:
                forward_count += 1
        return out, {
            "applied": True,
            "mode": mode_i,
            "kernel": kernel,
            "echo_polarity_aware": True,
            "reverse_echoes": int(reverse_count),
            "forward_echoes": int(forward_count),
            **last_meta,
        }

    out, traj_meta = _regrid_block(arr, reverse=reverse_all)
    if out is None:
        return arr, {"applied": False, "mode": mode_i, **traj_meta}
    return out, {
        "applied": True,
        "mode": mode_i,
        "kernel": kernel,
        "echo_polarity_aware": False,
        **traj_meta,
    }


def _bruker_epi_readout_density_options(
    method: dict | None = None,
    reco: dict | None = None,
    readout_regrid_options: dict | None = None,
) -> dict:
    raw_env = os.environ.get("MRAX_BRUKER_EPI_READOUT_DENSITY_COMPENSATION")
    raw = str(raw_env if raw_env is not None else "auto").strip().lower().replace("-", "_")
    if raw in {"0", "false", "no", "off", "none"}:
        return {"enabled": False, "mode": "off", "source": "environment"}
    force = raw in {"1", "true", "yes", "on", "jacobian", "density"}
    if raw not in {"auto"} and not force:
        warnings.warn(
            f"Unknown MRAX_BRUKER_EPI_READOUT_DENSITY_COMPENSATION={raw!r}; using auto.",
            RuntimeWarning,
            stacklevel=2,
        )
        raw = "auto"

    if method is None or reco is None:
        return {"enabled": False, "mode": "off", "source": "default", "reason": "missing_method_or_reco"}
    epi_vals = _flatten_numeric(method.get("PVM_EpiMatrix"))
    readout = int(round(epi_vals[0])) if epi_vals and int(round(epi_vals[0])) > 0 else None
    traj = np.asarray(_flatten_numeric(method.get("PVM_EpiTrajAdjkx")), dtype=np.float32).ravel()
    if readout is None or traj.size != int(readout):
        return {
            "enabled": False,
            "mode": "off",
            "source": "default",
            "reason": "missing_or_mismatched_PVM_EpiTrajAdjkx",
            "value_count": int(traj.size),
        }
    if not _bruker_epi_stage_has_filter(reco, "RecoRegridFilter") and not force:
        return {"enabled": False, "mode": "off", "source": "auto", "reason": "missing_RecoRegridFilter"}

    span = float(np.max(traj) - np.min(traj)) if traj.size else 0.0
    span_fraction = span / float(max(int(readout) - 1, 1))
    shots = _as_positive_int(method.get("PVM_EpiNShots"), 1) or 1
    is_3d_multishot = not _bruker_epi_is_2d_reco(method, reco) and int(shots) > 1
    regrid_enabled = bool((readout_regrid_options or {}).get("enabled", False))
    partial_span_guard = (
        str((readout_regrid_options or {}).get("reason", ""))
        == "PVM_EpiTrajAdjkx_span_is_not_a_full_readout_grid"
    )
    if force or (not regrid_enabled and is_3d_multishot and partial_span_guard):
        return {
            "enabled": True,
            "mode": "jacobian",
            "source": "environment" if force else "auto_partial_PVM_EpiTrajAdjkx",
            "trajectory_span": span,
            "trajectory_span_fraction": span_fraction,
            "shots": int(shots),
        }
    return {
        "enabled": False,
        "mode": "off",
        "source": "auto",
        "reason": "readout_trajectory_interpolation_or_full_span_preferred",
        "trajectory_span": span,
        "trajectory_span_fraction": span_fraction,
        "shots": int(shots),
    }


def _bruker_epi_readout_density_weights(method: dict, readout: int) -> tuple[np.ndarray | None, dict]:
    vals = np.asarray(_flatten_numeric(method.get("PVM_EpiTrajAdjkx")), dtype=np.float32).ravel()
    n = int(readout)
    if vals.size != n or n <= 2:
        return None, {"reason": "missing_or_mismatched_PVM_EpiTrajAdjkx", "value_count": int(vals.size)}

    weights = np.abs(np.gradient(vals.astype(np.float64))).astype(np.float32)
    valid = np.isfinite(weights) & (weights > 0.0)
    if not np.any(valid):
        return None, {"reason": "invalid_PVM_EpiTrajAdjkx_density"}
    if not np.all(valid):
        weights = weights.copy()
        weights[~valid] = float(np.median(weights[valid]))
    mean_weight = float(np.mean(weights))
    if mean_weight <= 0.0 or not np.isfinite(mean_weight):
        return None, {"reason": "invalid_PVM_EpiTrajAdjkx_density"}
    weights = (weights / mean_weight).astype(np.float32, copy=False)
    return weights, {
        "trajectory_source": "PVM_EpiTrajAdjkx",
        "samples": n,
        "normalization": "mean_1",
        "weight_min": float(np.min(weights)),
        "weight_max": float(np.max(weights)),
        "weight_mean": float(np.mean(weights)),
    }


def _apply_bruker_epi_readout_density_compensation(
    samples: np.ndarray,
    method: dict,
    *,
    enabled: bool,
) -> tuple[np.ndarray, dict]:
    arr = np.asarray(samples)
    if not enabled:
        return arr, {"applied": False, "reason": "disabled"}
    weights, meta = _bruker_epi_readout_density_weights(method, int(arr.shape[-1]))
    if weights is None:
        return arr, {"applied": False, **meta}
    shaped = weights.reshape((1,) * (arr.ndim - 1) + (int(weights.size),))
    return (arr * shaped).astype(np.complex64, copy=False), {
        "applied": True,
        "mode": "jacobian",
        "stage": "after_b0_before_readout_reversal",
        **meta,
    }


def _apply_bruker_epi_echo_time_shifts(
    samples: np.ndarray,
    method: dict,
    object_echo_indices: Sequence[np.ndarray],
    *,
    sign: float = 1.0,
) -> tuple[np.ndarray, dict]:
    """Apply recorded per-shot EPI readout echo-time shifts."""

    if str(method.get("PVM_EpiEchoTimeShifting", "") or "").strip().lower() not in ("yes", "on", "true", "1"):
        return np.asarray(samples), {"applied": False, "reason": "PVM_EpiEchoTimeShifting_disabled"}

    arr = np.asarray(samples).copy()
    if arr.ndim != 5:
        return arr, {"applied": False, "reason": "invalid_epi_sample_shape"}
    shots = int(arr.shape[1])
    shift_keys = ("PVM_EpiEchoShiftA", "PVM_EpiEchoShiftB")
    applied: list[dict] = []

    for object_idx, echo_indices in enumerate(object_echo_indices[: len(shift_keys)]):
        shifts = np.asarray(_flatten_numeric(method.get(shift_keys[object_idx])), dtype=np.float32).ravel()
        if shifts.size != shots or echo_indices.size == 0:
            continue
        for shot_idx, shift in enumerate(shifts.tolist()):
            if float(shift) == 0.0:
                continue
            block = arr[:, shot_idx : shot_idx + 1, :, echo_indices, :]
            arr[:, shot_idx : shot_idx + 1, :, echo_indices, :] = _bruker_epi_axis_phase_ramp(
                block,
                -1,
                float(sign) * float(shift),
            )
        applied.append(
            {
                "object": int(object_idx),
                "source": shift_keys[object_idx],
                "shift_range": [float(np.min(shifts)), float(np.max(shifts))],
            }
        )

    if not applied:
        return arr, {"applied": False, "reason": "missing_or_mismatched_PVM_EpiEchoShift"}
    return arr, {
        "applied": True,
        "source": "PVM_EpiEchoShiftA/B",
        "mode": "per_shot_readout_phase_ramp",
        "sign": float(sign),
        "objects": applied,
    }


def _bruker_epi_stage_has_stab_amplitude(reco: dict) -> bool:
    """Return whether the Bruker EPI graph requests stability amplitude balancing."""

    stage_text = _stage_node_text(reco)
    return bool(re.search(r"RecoStabCorrFilter\s+\w+\{[^}]*amplitude\s*=\s*true", stage_text))


def _bruker_epi_readout_polarity_amplitude_weights(
    readout_profiles: Sequence[np.ndarray],
    *,
    enabled: bool,
) -> tuple[list[np.ndarray] | None, dict]:
    """Estimate per-coil readout polarity amplitude weights for paired EPI objects.

    Bruker records ``RecoStabCorrFilter`` with ``amplitude=true`` for this EPI
    graph. Public EPI ghosting literature describes a readout-direction-dependent
    receiver response that modulates opposed readout polarities differently. The
    proprietary navigator filter is not documented, so this estimates only the
    reproducible amplitude component from the paired polarity/object profiles.
    """

    if not enabled:
        return None, {"applied": False, "reason": "RecoStabCorrFilter_amplitude_not_requested"}
    if len(readout_profiles) != 2:
        return None, {"applied": False, "reason": "requires_two_readout_polarity_objects"}

    first = np.asarray(readout_profiles[0])
    second = np.asarray(readout_profiles[1])
    if first.ndim != 3 or second.shape != first.shape:
        return None, {"applied": False, "reason": "mismatched_readout_profile_shapes"}

    # Profiles are (coil, graph_line, readout_x) after readout orientation has
    # been normalized. Average across the full graph-ordered line set to obtain a
    # stable per-coil readout response estimate for each polarity.
    mag0 = np.mean(np.abs(first), axis=1).astype(np.float32, copy=False)
    mag1 = np.mean(np.abs(second), axis=1).astype(np.float32, copy=False)
    common = (0.5 * (mag0 + mag1)).astype(np.float32, copy=False)
    eps = (0.05 * np.percentile(common, 10.0, axis=1, keepdims=True) + 1e-6).astype(np.float32)
    weights0 = np.clip(common / (mag0 + eps), 0.2, 5.0).astype(np.float32, copy=False)
    weights1 = np.clip(common / (mag1 + eps), 0.2, 5.0).astype(np.float32, copy=False)

    return [weights0, weights1], {
        "applied": True,
        "source": "RecoStabCorrFilter(amplitude=true)",
        "mode": "paired_readout_polarity_profile_balance",
        "weight_clip": [0.2, 5.0],
        "weight_range": [
            float(min(np.min(weights0), np.min(weights1))),
            float(max(np.max(weights0), np.max(weights1))),
        ],
    }


def _bruker_epi_weighted_line_phase_fit(
    reference_kx: np.ndarray,
    moving_kx: np.ndarray,
    *,
    max_abs_shift: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit shared phase offset and readout delay from paired EPI line sets."""

    ref = np.asarray(reference_kx)
    mov = np.asarray(moving_kx)
    readout = int(ref.shape[-1])
    x = np.linspace(-0.5, 0.5, readout, dtype=np.float32)
    window = np.abs(x) <= 0.35
    if int(np.sum(window)) < 3:
        window = np.ones_like(x, dtype=bool)

    cross = np.sum(ref[..., window] * np.conj(mov[..., window]), axis=0)
    phase = np.unwrap(np.angle(cross), axis=-1).astype(np.float32, copy=False)
    weight = np.abs(cross).astype(np.float32, copy=False)
    weight /= np.max(weight, axis=-1, keepdims=True) + 1e-6
    weight = np.maximum(weight, 1e-4)
    xw = x[window].reshape((1, -1))

    sum_w = np.sum(weight, axis=-1)
    sum_x = np.sum(weight * xw, axis=-1)
    sum_xx = np.sum(weight * xw * xw, axis=-1)
    sum_y = np.sum(weight * phase, axis=-1)
    sum_xy = np.sum(weight * xw * phase, axis=-1)
    denom = sum_w * sum_xx - sum_x * sum_x
    valid = np.abs(denom) > 1e-8

    slope = np.zeros_like(sum_w, dtype=np.float32)
    offset = np.zeros_like(sum_w, dtype=np.float32)
    slope[valid] = ((sum_w[valid] * sum_xy[valid] - sum_x[valid] * sum_y[valid]) / denom[valid]).astype(
        np.float32,
        copy=False,
    )
    offset[valid] = ((sum_y[valid] - slope[valid] * sum_x[valid]) / (sum_w[valid] + 1e-8)).astype(
        np.float32,
        copy=False,
    )

    shift_pixels = np.clip(slope / (2.0 * np.pi), -float(max_abs_shift), float(max_abs_shift)).astype(
        np.float32,
        copy=False,
    )
    return offset, shift_pixels, {
        "fit_window_fraction": float(np.mean(window)),
        "valid_line_fraction": float(np.mean(valid)),
        "max_abs_shift_pixels": float(max_abs_shift),
    }


def _bruker_epi_unwrap_phase_field(phase: np.ndarray, *, line_shape: tuple[int, int, int] | None = None) -> np.ndarray:
    arr = np.asarray(phase, dtype=np.float32)
    if line_shape is not None and int(np.prod(line_shape)) == int(arr.size):
        shaped = arr.reshape(line_shape)
        shaped = np.unwrap(shaped, axis=2)
        shaped = np.unwrap(shaped, axis=0)
        return shaped.reshape(arr.shape).astype(np.float32, copy=False)
    return np.unwrap(arr).astype(np.float32, copy=False)


def _bruker_epi_even_odd_stabilization_options(method: dict, reco: dict) -> dict:
    raw = str(os.environ.get("MRAX_BRUKER_EPI_EVEN_ODD_STABILIZATION", "auto") or "auto").strip().lower()
    if raw in {"0", "false", "no", "off", "none"}:
        return {"enabled": False, "mode": "off", "source": "environment"}
    if raw in {"1", "true", "yes", "on"}:
        return {"enabled": True, "mode": "global_cross_spectrum", "source": "environment"}
    if raw != "auto":
        warnings.warn(
            f"Unknown MRAX_BRUKER_EPI_EVEN_ODD_STABILIZATION={raw!r}; using auto.",
            RuntimeWarning,
            stacklevel=2,
        )
    return {
        "enabled": False,
        "mode": "off",
        "source": "default",
        "available": _bruker_epi_is_2d_reco(method, reco),
    }


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return float(default)
    try:
        return float(raw)
    except ValueError:
        warnings.warn(f"Invalid {name}={raw!r}; using {default}.", RuntimeWarning, stacklevel=2)
        return float(default)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return int(default)
    try:
        return int(raw)
    except ValueError:
        warnings.warn(f"Invalid {name}={raw!r}; using {default}.", RuntimeWarning, stacklevel=2)
        return int(default)


def _bruker_epi_ghost_filter_options(method: dict, reco: dict) -> dict:
    raw = str(os.environ.get("MRAX_BRUKER_EPI_GHOST_FILTER", "auto") or "auto").strip().lower()
    common = {
        "strength": float(np.clip(_env_float("MRAX_BRUKER_EPI_GHOST_FILTER_STRENGTH", 1.0), 0.0, 1.0)),
        "max_abs_coefficient": float(
            np.clip(_env_float("MRAX_BRUKER_EPI_GHOST_FILTER_MAX_COEFF", 0.5), 0.0, 1.0)
        ),
        "max_order": int(np.clip(_env_int("MRAX_BRUKER_EPI_GHOST_FILTER_ORDER", 1), 0, 1)),
    }
    if raw in {"0", "false", "no", "off", "none"}:
        return {"enabled": False, "mode": "off", "source": "environment", **common}
    if raw not in {"auto", "1", "true", "yes", "on"}:
        warnings.warn(
            f"Unknown MRAX_BRUKER_EPI_GHOST_FILTER={raw!r}; using auto.",
            RuntimeWarning,
            stacklevel=2,
        )
        raw = "auto"

    stage_text = _stage_node_text(reco)
    epi_matrix = np.asarray(_flatten_numeric(method.get("PVM_EpiMatrix")), dtype=int).ravel()
    readout = int(epi_matrix[0]) if epi_matrix.size >= 1 and int(epi_matrix[0]) > 0 else 0
    prefix_points, prefix_meta = _bruker_epi_prefix_nav_points(method, readout, reco=reco)
    requested = raw in {"1", "true", "yes", "on"}
    available = "RecoEpiGhostFilter" in stage_text and _bruker_epi_is_2d_reco(method, reco)
    enabled = requested or (raw == "auto" and available and int(prefix_points) == 0)
    return {
        "enabled": bool(enabled),
        "mode": "fov2_complex_leakage_subtraction" if enabled else "off",
        "source": "environment" if requested else "auto_RecoEpiGhostFilter",
        "available": bool(available),
        "prefix_navigator": prefix_meta,
        **common,
    }


def _apply_bruker_epi_fov2_ghost_filter(
    coil_image: np.ndarray,
    *,
    enabled: bool,
    phase_axis: int = 2,
    strength: float = 1.0,
    max_abs_coefficient: float = 0.5,
    max_order: int = 1,
) -> tuple[np.ndarray, dict]:
    """Approximate Bruker's ``RecoEpiGhostFilter`` with a clipped FOV/2 leakage model."""

    arr = np.asarray(coil_image)
    if not enabled:
        return arr, {"applied": False, "reason": "disabled"}
    if arr.ndim != 4:
        return arr, {"applied": False, "reason": "invalid_coil_image_shape"}

    axis = int(phase_axis)
    if axis < 0:
        axis += arr.ndim
    if axis <= 0 or axis >= arr.ndim or int(arr.shape[axis]) < 4:
        return arr, {"applied": False, "reason": "invalid_phase_axis", "axis": int(phase_axis)}

    shift = int(arr.shape[axis]) // 2
    if shift <= 0:
        return arr, {"applied": False, "reason": "invalid_fov2_shift"}

    rss = np.sqrt(np.sum(np.abs(arr) ** 2, axis=0)).astype(np.float32, copy=False)
    object_threshold = max(float(np.percentile(rss, 75.0)), 0.25 * float(np.max(rss)))
    object_mask = rss > object_threshold
    ghost_mask = np.roll(object_mask, shift, axis=axis - 1) & ~object_mask & (rss <= np.percentile(rss, 85.0))
    if int(np.sum(ghost_mask)) < max(8, int(0.01 * ghost_mask.size)):
        return arr, {
            "applied": False,
            "reason": "insufficient_background_ghost_mask",
            "ghost_mask_fraction": float(np.mean(ghost_mask)),
        }

    source = np.roll(arr, shift, axis=axis)
    corrected = np.array(arr, copy=True)
    coefficients: list[complex] = []
    coefficient_ranges: list[dict[str, float]] = []
    order = int(np.clip(int(max_order), 0, 1))
    readout_axis = 1
    readout_coord = np.linspace(-0.5, 0.5, int(arr.shape[readout_axis]), dtype=np.float32)
    readout_coord_field = np.broadcast_to(
        readout_coord.reshape((int(arr.shape[readout_axis]), 1, 1)),
        ghost_mask.shape,
    )
    for coil_idx in range(int(arr.shape[0])):
        src = source[coil_idx][ghost_mask]
        dst = arr[coil_idx][ghost_mask]
        if order == 0:
            denom = np.vdot(src, src)
            coeff = 0.0 + 0.0j if float(np.abs(denom)) <= 1e-12 else np.vdot(src, dst) / denom
            coeff_field = np.full((int(arr.shape[readout_axis]),), coeff, dtype=np.complex64)
            coefficients.append(complex(coeff))
        else:
            x = readout_coord_field[ghost_mask].astype(np.float32, copy=False)
            design = np.stack([src, src * x], axis=1)
            try:
                beta = np.linalg.lstsq(design, dst, rcond=None)[0]
            except np.linalg.LinAlgError:
                beta = np.zeros((2,), dtype=np.complex64)
            coeff_field = (beta[0] + beta[1] * readout_coord).astype(np.complex64, copy=False)
            coefficients.append(complex(beta[0]))

        coeff_abs_field = np.abs(coeff_field).astype(np.float32, copy=False)
        over = coeff_abs_field > float(max_abs_coefficient)
        if np.any(over):
            coeff_field = np.where(
                over,
                coeff_field * (float(max_abs_coefficient) / (coeff_abs_field + 1e-12)),
                coeff_field,
            )
            coeff_abs_field = np.abs(coeff_field).astype(np.float32, copy=False)
        coefficient_ranges.append(
            {
                "min": float(np.min(coeff_abs_field)),
                "median": float(np.median(coeff_abs_field)),
                "max": float(np.max(coeff_abs_field)),
            }
        )
        corrected[coil_idx] = arr[coil_idx] - float(strength) * coeff_field.reshape(
            (int(arr.shape[readout_axis]), 1, 1)
        ) * source[coil_idx]

    coeff_abs = np.asarray([abs(c) for c in coefficients], dtype=np.float32)
    return corrected.astype(np.complex64, copy=False), {
        "applied": True,
        "source": "RecoEpiGhostFilter",
        "mode": "fov2_complex_leakage_subtraction",
        "max_order": int(order),
        "axis": axis,
        "fov2_shift": int(shift),
        "strength": float(strength),
        "max_abs_coefficient": float(max_abs_coefficient),
        "ghost_mask_fraction": float(np.mean(ghost_mask)),
        "object_threshold": float(object_threshold),
        "coefficient_abs": {
            "min": float(np.min(coeff_abs)),
            "median": float(np.median(coeff_abs)),
            "max": float(np.max(coeff_abs)),
        },
        "coefficient_abs_by_readout": coefficient_ranges,
    }


def _apply_bruker_epi_even_odd_line_stabilization(
    read_image: np.ndarray,
    *,
    enabled: bool,
    sign: float = -1.0,
    max_abs_shift: float = 4.0,
) -> tuple[np.ndarray, dict]:
    """Estimate and remove a global odd/even EPI readout phase mismatch."""

    arr = np.asarray(read_image)
    if not enabled:
        return arr, {"applied": False, "reason": "disabled"}
    if arr.ndim != 3:
        return arr, {"applied": False, "reason": "invalid_read_image_shape"}
    lines = int(arr.shape[1])
    readout = int(arr.shape[-1])
    pair_count = lines // 2
    if pair_count < 3 or readout <= 3:
        return arr, {"applied": False, "reason": "insufficient_even_odd_lines"}

    even = arr[:, 0 : 2 * pair_count : 2, :]
    odd = arr[:, 1 : 2 * pair_count : 2, :]
    even_kx = _bruker_epi_centered_ft(even, axis=-1, inverse=True)
    odd_kx = _bruker_epi_centered_ft(odd, axis=-1, inverse=True)
    phase_offset, shift_pixels, fit_meta = _bruker_epi_weighted_line_phase_fit(
        even_kx,
        odd_kx,
        max_abs_shift=max_abs_shift,
    )

    phase_unwrapped = _bruker_epi_unwrap_phase_field(phase_offset)
    weights = np.mean(np.abs(even_kx) * np.abs(odd_kx), axis=(0, 2)).astype(np.float32, copy=False)
    weights = np.maximum(weights, 1e-6)
    phase_global = float(np.sum(phase_unwrapped * weights) / np.sum(weights))
    shift_global = float(np.sum(shift_pixels * weights) / np.sum(weights))
    shift_global = float(np.clip(shift_global, -float(max_abs_shift), float(max_abs_shift)))

    x = np.linspace(-0.5, 0.5, readout, dtype=np.float32)
    phase_model = phase_global + 2.0 * np.pi * shift_global * x
    correction = np.exp(1j * float(sign) * phase_model).astype(np.complex64)
    corrected_odd = _bruker_epi_centered_ft(odd_kx * correction.reshape((1, 1, readout)), axis=-1, inverse=False)

    out = arr.astype(np.complex64, copy=True)
    out[:, 1 : 2 * pair_count : 2, :] = corrected_odd.astype(np.complex64, copy=False)
    return out, {
        "applied": True,
        "source": "adjacent_even_odd_line_cross_spectrum",
        "mode": "global_odd_line_phase_and_readout_delay",
        "sign": float(sign),
        "line_pairs": int(pair_count),
        "phase_offset_radians": {
            "raw_min": float(np.min(phase_offset)),
            "raw_median": float(np.median(phase_unwrapped)),
            "raw_max": float(np.max(phase_offset)),
            "global": phase_global,
        },
        "shift_pixels": {
            "raw_min": float(np.min(shift_pixels)),
            "raw_median": float(np.median(shift_pixels)),
            "raw_max": float(np.max(shift_pixels)),
            "global": shift_global,
        },
        "fit": fit_meta,
    }


def _bruker_epi_prefix_nav_orientation_options(prefix_meta: dict, *, object_split: str) -> dict:
    """Return final image orientation correction for prefix-navigator two-object EPI."""

    applied = bool(prefix_meta.get("applied")) and str(object_split) == "interleaved_echo_parity"
    if not applied:
        return {
            "applied": False,
            "reason": "not_prefix_navigator_interleaved_echo_parity_epi",
        }
    return {
        "applied": True,
        "mode": "flip_readout_axis_before_output_axis_formatting",
        "axis": "readout",
        "source": "RecoStabCorrFilter_prefix_navigator_two_object_epi",
    }


def _bruker_epi_fractional_shift(data: np.ndarray, *, axis: int, shift: float) -> np.ndarray:
    """Apply a fractional pixel shift with the Fourier shift theorem."""

    arr = np.asarray(data)
    if arr.ndim == 0 or float(shift) == 0.0:
        return arr
    axis_i = int(axis)
    if axis_i < 0:
        axis_i += arr.ndim
    size = int(arr.shape[axis_i])
    if size <= 1:
        return arr
    freq = np.fft.fftfreq(size).astype(np.float32)
    phase = np.exp(-2j * np.pi * freq * float(shift)).astype(np.complex64)
    shape = [1] * arr.ndim
    shape[axis_i] = size
    shifted = np.fft.ifft(np.fft.fft(arr, axis=axis_i) * phase.reshape(shape), axis=axis_i)
    if np.iscomplexobj(arr):
        return shifted.astype(np.complex64, copy=False)
    return shifted.real.astype(arr.dtype, copy=False)


def _bruker_epi_estimate_readout_shift(
    reference_magnitude: np.ndarray,
    moving_magnitude: np.ndarray,
    *,
    max_abs_shift: float = 4.0,
) -> tuple[float, dict]:
    """Estimate readout-axis shift needed to align paired EPI polarity objects."""

    reference = np.asarray(reference_magnitude, dtype=np.float32)
    moving = np.asarray(moving_magnitude, dtype=np.float32)
    if reference.shape != moving.shape or reference.ndim < 1:
        return 0.0, {"applied": False, "reason": "mismatched_polarity_magnitude_shapes"}

    axes = tuple(range(1, reference.ndim))
    ref_profile = np.mean(reference, axis=axes) if axes else reference
    mov_profile = np.mean(moving, axis=axes) if axes else moving
    ref_profile = np.asarray(ref_profile, dtype=np.float32) - float(np.mean(ref_profile))
    mov_profile = np.asarray(mov_profile, dtype=np.float32) - float(np.mean(mov_profile))
    size = int(ref_profile.size)
    if size <= 2 or not np.any(ref_profile) or not np.any(mov_profile):
        return 0.0, {"applied": False, "reason": "insufficient_readout_profile_signal"}

    corr = np.fft.ifft(np.fft.fft(ref_profile) * np.conj(np.fft.fft(mov_profile))).real
    max_int = int(max(0, min(size // 2 - 1, np.ceil(float(max_abs_shift)))))
    candidate_shifts = np.arange(-max_int, max_int + 1, dtype=int)
    candidate_indices = np.mod(candidate_shifts, size)
    best_pos = int(np.argmax(corr[candidate_indices]))
    best_shift = int(candidate_shifts[best_pos])
    best_idx = int(candidate_indices[best_pos])

    left = float(corr[(best_idx - 1) % size])
    center = float(corr[best_idx])
    right = float(corr[(best_idx + 1) % size])
    denom = left - 2.0 * center + right
    delta = 0.0 if abs(denom) < 1e-12 else 0.5 * (left - right) / denom
    delta = float(np.clip(delta, -0.5, 0.5))
    shift = float(np.clip(best_shift + delta, -float(max_abs_shift), float(max_abs_shift)))
    return shift, {
        "applied": True,
        "mode": "readout_profile_phase_correlation",
        "max_abs_shift_pixels": float(max_abs_shift),
        "correlation_peak": center,
    }


def _bruker_epi_ramp_delay_scale(method: dict, readout: int) -> tuple[float, dict]:
    """Estimate how ramp sampling scales a post-FFT EPI readout-delay estimate."""

    vals = np.asarray(_flatten_numeric(method.get("PVM_EpiTrajAdjkx")), dtype=np.float32).ravel()
    if vals.size != int(readout) or int(readout) <= 1:
        return 1.0, {"applied": False, "reason": "missing_or_mismatched_PVM_EpiTrajAdjkx"}

    span = float(np.max(vals) - np.min(vals))
    if span <= 0.0:
        return 1.0, {"applied": False, "reason": "invalid_PVM_EpiTrajAdjkx_span"}

    target_span = float(int(readout) - 1)
    scale = float(np.clip(np.sqrt(target_span / span), 0.75, 1.5))
    return scale, {
        "applied": True,
        "source": "PVM_EpiTrajAdjkx",
        "mode": "sqrt_cartesian_span_over_measured_trajectory_span",
        "trajectory_span": span,
        "target_span": target_span,
        "scale": scale,
    }


def _bruker_epi_axis_phase_ramp(data: np.ndarray, axis: int, shift: float) -> np.ndarray:
    arr = np.asarray(data)
    if arr.ndim == 0 or float(shift) == 0.0:
        return arr
    axis_i = int(axis)
    if axis_i < 0:
        axis_i += arr.ndim
    size = int(arr.shape[axis_i])
    if size <= 0:
        return arr
    coord = np.arange(size, dtype=np.float32) - float(size) / 2.0
    phase = np.exp(2j * np.pi * float(shift) * coord / float(size)).astype(np.complex64)
    shape = [1] * arr.ndim
    shape[axis_i] = size
    return arr * phase.reshape(tuple(shape))


def _bruker_epi_centered_ft(data: np.ndarray, *, axis: int, inverse: bool) -> np.ndarray:
    transform = np.fft.ifft if inverse else np.fft.fft
    return np.fft.fftshift(
        transform(np.fft.ifftshift(data, axes=axis), axis=axis),
        axes=axis,
    ).astype(np.complex64, copy=False)


def _take_along_axis_index(data: np.ndarray, indices: np.ndarray, *, axis: int) -> np.ndarray:
    """Return ``data`` indexed by a 1D vector along one axis."""

    axis_i = int(axis)
    if axis_i < 0:
        axis_i += int(data.ndim)
    slices: list[object] = [slice(None)] * int(data.ndim)
    slices[axis_i] = np.asarray(indices, dtype=int)
    return data[tuple(slices)]


def _mask_shape_for_axis(mask: np.ndarray, *, axis: int, ndim: int) -> tuple[int, ...]:
    axis_i = int(axis)
    if axis_i < 0:
        axis_i += int(ndim)
    shape = [1] * int(ndim)
    shape[axis_i] = int(mask.size)
    return tuple(shape)


def _parse_bruker_half_fourier_filter(reco: dict, *, direction: int = 1) -> dict:
    """Extract the active Bruker half-Fourier filter for one reconstruction direction."""

    direction_i = int(direction)
    stage_text = _stage_node_text(reco)
    for text in re.findall(r"RecoHalfFourierFilter\s+\w+\{[^}]*\}", stage_text):
        dir_match = re.search(r"direction\s*=\s*([-+]?\d+)", text)
        if dir_match and int(dir_match.group(1)) != direction_i:
            continue
        echo_match = re.search(r"echoPosition\s*=\s*([-+]?\d+)", text)
        alg_match = re.search(r"algorithm\s*=\s*([-+]?\d+)", text)
        return {
            "enabled": True,
            "source": "RecoHalfFourierFilter",
            "direction": direction_i,
            "echo_position": int(echo_match.group(1)) if echo_match else None,
            "algorithm": int(alg_match.group(1)) if alg_match else None,
        }

    half_flags = _as_reco_string_list(reco.get("RecoHalfFT"))
    if len(half_flags) > direction_i and str(half_flags[direction_i]).strip().lower() in {"yes", "on", "true", "1"}:
        positions = np.asarray(_flatten_numeric(reco.get("RecoHalfFTPos")), dtype=int).ravel()
        algorithm = str(reco.get("RecoHalfFTAlgorithm") or "").strip() or None
        return {
            "enabled": True,
            "source": "RecoHalfFT",
            "direction": direction_i,
            "echo_position": int(positions[direction_i]) if positions.size > direction_i else None,
            "algorithm": algorithm,
        }

    return {"enabled": False, "reason": "RecoHalfFourierFilter_not_requested", "direction": direction_i}


def _bruker_epi_half_fourier_mode(method: dict | None = None, reco: dict | None = None) -> str:
    raw = os.environ.get("MRAX_BRUKER_EPI_HALF_FOURIER_MODE")
    if raw is not None and str(raw).strip():
        mode = str(raw).strip().lower().replace("-", "_")
        if mode in {"off", "none", "zero", "zero_fill", "zerofill", "pocs", "homodyne"}:
            return "zero_fill" if mode in {"off", "none", "zero", "zerofill"} else mode
        warnings.warn(
            f"Unknown MRAX_BRUKER_EPI_HALF_FOURIER_MODE={mode!r}; using auto.",
            RuntimeWarning,
            stacklevel=2,
        )
    if reco is not None:
        half_meta = _parse_bruker_half_fourier_filter(reco, direction=1)
        if half_meta.get("algorithm") == 2:
            return "pocs"
    if method is not None and reco is not None and not _bruker_epi_is_2d_reco(method, reco):
        return "homodyne"
    return "zero_fill"


def _bruker_epi_half_fourier_iterations(default: int = 12) -> int:
    return int(default)


def _apply_bruker_epi_phase_half_fourier(
    phase_kspace: np.ndarray,
    *,
    acquired_indices: np.ndarray,
    axis: int,
    echo_position: int | None,
    iterations: int = 12,
) -> tuple[np.ndarray, dict]:
    """Apply a phase-constrained POCS completion along the EPI phase axis.

    At this point in the Bruker EPI graph, readout and slice have already been
    transformed to image space while the phase axis remains partial Fourier
    k-space. The low-resolution phase estimate is therefore a 1D phase-axis
    estimate for each coil/readout/slice location.
    """

    measured = np.asarray(phase_kspace, dtype=np.complex64)
    axis_i = int(axis)
    if axis_i < 0:
        axis_i += int(measured.ndim)
    n = int(measured.shape[axis_i])
    if n <= 1:
        return _bruker_epi_centered_ft(measured, axis=axis_i, inverse=True), {
            "applied": False,
            "reason": "invalid_phase_axis_size",
        }

    acquired = np.unique(np.asarray(acquired_indices, dtype=int).ravel())
    acquired = acquired[(acquired >= 0) & (acquired < n)]
    if acquired.size == 0:
        return _bruker_epi_centered_ft(measured, axis=axis_i, inverse=True), {
            "applied": False,
            "reason": "no_acquired_phase_lines",
        }
    if acquired.size == n:
        return _bruker_epi_centered_ft(measured, axis=axis_i, inverse=True), {
            "applied": False,
            "reason": "fully_sampled_phase_axis",
            "acquired_lines": int(acquired.size),
        }

    first = int(acquired[0])
    last = int(acquired[-1])
    center_from_echo = None
    if echo_position is not None:
        center_from_echo = first + max(int(echo_position) - 1, 0)
    center = int(center_from_echo) if center_from_echo is not None and 0 <= center_from_echo < n else n // 2

    acquired_mask = np.zeros((n,), dtype=bool)
    acquired_mask[acquired] = True
    common_radius = int(min(center - first, last - center))
    if common_radius < 2:
        return _bruker_epi_centered_ft(measured, axis=axis_i, inverse=True), {
            "applied": False,
            "reason": "insufficient_symmetric_center",
            "acquired_range": [first, last],
            "center": center,
        }

    lowres_mask = np.zeros((n,), dtype=bool)
    low_start = max(0, center - common_radius)
    low_stop = min(n, center + common_radius + 1)
    lowres_mask[low_start:low_stop] = True
    lowres_mask &= acquired_mask
    lowres = measured * lowres_mask.reshape(_mask_shape_for_axis(lowres_mask, axis=axis_i, ndim=measured.ndim))
    lowres_image = _bruker_epi_centered_ft(lowres, axis=axis_i, inverse=True)
    phase = np.exp(1j * np.angle(lowres_image)).astype(np.complex64, copy=False)

    estimate = measured.copy()
    all_indices = np.arange(n, dtype=int)
    mirror_indices = np.mod(2 * center - all_indices, n)
    mirror_acquired = acquired_mask[mirror_indices]
    missing = ~acquired_mask
    fill_mask = missing & mirror_acquired
    if np.any(fill_mask):
        mirror_values = np.conj(_take_along_axis_index(measured, mirror_indices, axis=axis_i))
        estimate = np.where(
            fill_mask.reshape(_mask_shape_for_axis(fill_mask, axis=axis_i, ndim=measured.ndim)),
            mirror_values,
            estimate,
        )

    acquired_mask_nd = acquired_mask.reshape(_mask_shape_for_axis(acquired_mask, axis=axis_i, ndim=measured.ndim))
    for _ in range(max(int(iterations), 0)):
        image = _bruker_epi_centered_ft(estimate, axis=axis_i, inverse=True)
        image = np.abs(image).astype(np.float32, copy=False) * phase
        estimate = _bruker_epi_centered_ft(image, axis=axis_i, inverse=False)
        estimate = np.where(acquired_mask_nd, measured, estimate).astype(np.complex64, copy=False)

    image = _bruker_epi_centered_ft(estimate, axis=axis_i, inverse=True)
    return image.astype(np.complex64, copy=False), {
        "applied": True,
        "source": "RecoHalfFourierFilter",
        "mode": "phase_axis_lowres_phase_pocs",
        "axis": int(axis_i),
        "iterations": int(iterations),
        "echo_position": None if echo_position is None else int(echo_position),
        "echo_position_base": "one_based",
        "center_index": int(center),
        "acquired_lines": int(acquired.size),
        "acquired_range": [int(first), int(last)],
        "lowres_range": [int(low_start), int(low_stop - 1)],
        "lowres_lines": int(np.sum(lowres_mask)),
        "fillable_missing_lines": int(np.sum(fill_mask)),
    }


def _apply_bruker_epi_phase_homodyne(
    phase_kspace: np.ndarray,
    *,
    acquired_indices: np.ndarray,
    axis: int,
    echo_position: int | None,
) -> tuple[np.ndarray, dict]:
    measured = np.asarray(phase_kspace, dtype=np.complex64)
    axis_i = int(axis)
    if axis_i < 0:
        axis_i += int(measured.ndim)
    n = int(measured.shape[axis_i])
    acquired = np.unique(np.asarray(acquired_indices, dtype=int).ravel())
    acquired = acquired[(acquired >= 0) & (acquired < n)]
    if n <= 1 or acquired.size == 0 or acquired.size == n:
        return _bruker_epi_centered_ft(measured, axis=axis_i, inverse=True), {
            "applied": False,
            "mode": "homodyne",
            "reason": "fully_sampled_or_invalid_phase_axis",
        }
    first = int(acquired[0])
    last = int(acquired[-1])
    center = int(first + max(int(echo_position) - 1, 0)) if echo_position is not None else n // 2
    if center < 0 or center >= n:
        center = n // 2
    common_radius = int(min(center - first, last - center))
    if common_radius < 2:
        return _bruker_epi_centered_ft(measured, axis=axis_i, inverse=True), {
            "applied": False,
            "mode": "homodyne",
            "reason": "insufficient_symmetric_center",
            "acquired_range": [first, last],
            "center": center,
        }
    lowres_mask = np.zeros((n,), dtype=bool)
    low_start = max(0, center - common_radius)
    low_stop = min(n, center + common_radius + 1)
    lowres_mask[low_start:low_stop] = True
    acquired_mask = np.zeros((n,), dtype=bool)
    acquired_mask[acquired] = True
    lowres_mask &= acquired_mask
    lowres = measured * lowres_mask.reshape(_mask_shape_for_axis(lowres_mask, axis=axis_i, ndim=measured.ndim))
    lowres_image = _bruker_epi_centered_ft(lowres, axis=axis_i, inverse=True)
    phase = np.exp(1j * np.angle(lowres_image)).astype(np.complex64, copy=False)
    zf_image = _bruker_epi_centered_ft(measured, axis=axis_i, inverse=True)
    demodulated = zf_image * np.conj(phase)
    image = (demodulated.real.astype(np.float32, copy=False) * phase).astype(np.complex64, copy=False)
    return image, {
        "applied": True,
        "source": "RecoHalfFourierFilter",
        "mode": "phase_axis_homodyne",
        "axis": int(axis_i),
        "echo_position": None if echo_position is None else int(echo_position),
        "echo_position_base": "one_based",
        "center_index": int(center),
        "acquired_lines": int(acquired.size),
        "acquired_range": [int(first), int(last)],
        "lowres_range": [int(low_start), int(low_stop - 1)],
        "lowres_lines": int(np.sum(lowres_mask)),
    }


def _coil_images_to_cartesian_kspace(coil_images: np.ndarray) -> np.ndarray:
    arr = np.asarray(coil_images)
    axes = tuple(range(arr.ndim - 3, arr.ndim))
    return np.fft.fftshift(
        np.fft.fftn(np.fft.ifftshift(arr, axes=axes), axes=axes, norm="ortho"),
        axes=axes,
    ).astype(np.complex64, copy=False)


def _cartesian_kspace_to_coil_images(kspace: np.ndarray) -> np.ndarray:
    arr = np.asarray(kspace)
    axes = tuple(range(arr.ndim - 3, arr.ndim))
    return np.fft.fftshift(
        np.fft.ifftn(np.fft.ifftshift(arr, axes=axes), axes=axes, norm="ortho"),
        axes=axes,
    ).astype(np.complex64, copy=False)


_BRUKER_EPI_SLICE_CROP_MODES = ("reco_offset", "zero_start", "center", "auto_epi_vendor")

_BRUKER_EPI_UNAVAILABLE_NOTE = (
    "Bruker PV360 EPI raw reconstruction could not be generated for this scan. "
    "See candidate_errors for the rejected reconstruction paths."
)


def _normalize_bruker_epi_slice_crop_mode(mode: str | None = None) -> str:
    if mode is None or str(mode).strip() == "":
        mode = "auto_epi_vendor"
    normalized = str(mode).strip().lower().replace("-", "_")
    if normalized not in _BRUKER_EPI_SLICE_CROP_MODES:
        valid = ", ".join(_BRUKER_EPI_SLICE_CROP_MODES)
        raise ValueError(f"Invalid Bruker EPI slice crop mode {mode!r}; expected one of: {valid}.")
    return normalized


def _bruker_epi_slice_crop_decision(
    *,
    mode: str | None,
    reco_size: Sequence[int] | np.ndarray,
    ft_size: Sequence[int] | np.ndarray,
    reco_offset: Sequence[int] | np.ndarray,
    pre_crop_slices: int,
    output_slices: int,
    sequence_type: str = "epi",
    storage: str = "pv360_raw_job",
) -> dict:
    """Resolve the opt-in Bruker EPI slice crop convention."""

    crop_mode = _normalize_bruker_epi_slice_crop_mode(mode)
    reco_size_arr = np.asarray(reco_size, dtype=int).ravel()
    ft_size_arr = np.asarray(ft_size, dtype=int).ravel()
    reco_offset_arr = np.asarray(reco_offset, dtype=int).ravel()
    reco_offset_slice = int(reco_offset_arr[2]) if reco_offset_arr.size >= 3 else 0
    pre_slices = int(pre_crop_slices)
    out_slices = int(output_slices)
    max_start = pre_slices - out_slices
    if pre_slices <= 0 or out_slices <= 0 or max_start < 0:
        raise ValueError(
            "Invalid Bruker EPI slice crop geometry: "
            f"pre_crop_slices={pre_slices}, output_slices={out_slices}."
        )
    center_start = int(max_start // 2)
    slice_cutoff = int(ft_size_arr[2]) - int(reco_size_arr[2]) if ft_size_arr.size >= 3 and reco_size_arr.size >= 3 else None

    auto_gate = {
        "sequence_type_is_epi": str(sequence_type).strip().lower() == "epi",
        "storage_is_pv360_raw_job": str(storage).strip().lower() == "pv360_raw_job",
        "has_reco_size_slice": bool(reco_size_arr.size >= 3),
        "has_ft_size_slice": bool(ft_size_arr.size >= 3),
        "has_reco_offset_slice": bool(reco_offset_arr.size >= 3),
        "pre_crop_matches_reco_ft_size_slice": bool(ft_size_arr.size >= 3 and pre_slices == int(ft_size_arr[2])),
        "output_matches_reco_size_slice": bool(reco_size_arr.size >= 3 and out_slices == int(reco_size_arr[2])),
        "positive_slice_cutoff": bool(slice_cutoff is not None and slice_cutoff > 0),
        "positive_reco_offset_slice": bool(reco_offset_arr.size >= 3 and int(reco_offset_arr[2]) > 0),
        "symmetric_slice_cutoff": bool(
            reco_size_arr.size >= 3
            and ft_size_arr.size >= 3
            and reco_offset_arr.size >= 3
            and int(ft_size_arr[2]) - int(reco_size_arr[2]) == 2 * int(reco_offset_arr[2])
        ),
    }
    auto_gate["passed"] = bool(all(auto_gate.values()))

    if crop_mode == "reco_offset":
        crop_start = reco_offset_slice
        effective_mode = "reco_offset"
        start_source = "RECO_offset[2]"
        reason = "current_interpretation"
        fallback_reason = None
    elif crop_mode == "zero_start":
        crop_start = 0
        effective_mode = "zero_start"
        start_source = "zero_start"
        reason = "explicit_zero_start"
        fallback_reason = None
    elif crop_mode == "center":
        crop_start = center_start
        effective_mode = "center"
        start_source = "center"
        reason = "explicit_center_crop"
        fallback_reason = None
    else:
        if auto_gate["passed"]:
            crop_start = 0
            effective_mode = "zero_start"
            start_source = "auto_epi_vendor_zero_start"
            reason = "pv360_epi_symmetric_slice_cutoff_selected_zero_start"
            fallback_reason = None
        else:
            crop_start = reco_offset_slice
            effective_mode = "reco_offset"
            start_source = "auto_epi_vendor_reco_offset"
            failed = [key for key, value in auto_gate.items() if key != "passed" and not bool(value)]
            fallback_reason = "auto_epi_vendor_gate_failed:" + ",".join(failed)
            reason = fallback_reason

    crop_start = int(crop_start)
    if crop_start < 0 or crop_start > max_start:
        raise ValueError(
            f"Bruker EPI slice crop mode {crop_mode!r} chose crop_start={crop_start}, "
            f"outside valid range [0, {max_start}] for {pre_slices}->{out_slices} slices."
        )

    offset_used = bool(crop_start == reco_offset_slice)
    notice = None
    if not offset_used:
        if crop_mode == "auto_epi_vendor" and effective_mode == "zero_start":
            notice = (
                f"RECO_offset[2]={reco_offset_slice} was not used as the slice crop start because PV360 EPI "
                f"raw crop/origin policy selected zero_start crop_start={crop_start}."
            )
        else:
            notice = (
                f"RECO_offset[2]={reco_offset_slice} was not used as the slice crop start; "
                f"{start_source} selected crop_start={crop_start}."
            )

    return {
        "requested_slice_crop_mode": crop_mode,
        "slice_crop_mode": crop_mode,
        "effective_slice_crop_mode": effective_mode,
        "crop_mode_effective": effective_mode,
        "slice_crop_start": crop_start,
        "selected_start_source": start_source,
        "selection_reason": reason,
        "auto_epi_vendor_decision_reason": reason if crop_mode == "auto_epi_vendor" else None,
        "auto_epi_vendor_fallback_reason": fallback_reason if crop_mode == "auto_epi_vendor" else None,
        "reco_offset_slice": reco_offset_slice,
        "center_crop_start": center_start,
        "max_crop_start": int(max_start),
        "pre_crop_slices": pre_slices,
        "output_slices": out_slices,
        "reco_offset_used_as_crop_start": offset_used,
        "reco_offset_ignored_for_slice_crop": bool(not offset_used),
        "notice": notice,
        "auto_epi_vendor_gate": auto_gate,
    }


def _reconstruct_bruker_epi_raw_job(
    raw_path: Path,
    dtype: np.dtype,
    *,
    frame_count: int,
    method: dict,
    reco: dict,
    acqp: dict,
    visu: dict | None = None,
    slice_crop_mode: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Reconstruct PV360 EPI raw blocks following the recorded Bruker graph.

    Unlike the generic Cartesian path, Bruker EPI does not first build a full
    ``(kz, ky, kx)`` cube. The readout FT is performed before the sort map. This
    helper stores that sorted measured pre-reconstruction k-space for display and
    export, then applies the remaining slice/phase reconstruction steps only to
    produce the MRax raw image preview.
    """

    jobs = acqp.get("ACQ_jobs")
    if not isinstance(jobs, (list, tuple)) or len(jobs) < 1:
        raise ValueError("Bruker EPI raw import requires ACQ_jobs metadata.")

    job_scan_size = _as_positive_int(jobs[0])
    coil_count = _as_positive_int(method.get("PVM_EncNReceivers") or method.get("PVM_NReceivers"), 1)
    if job_scan_size is None or job_scan_size % 2 != 0 or coil_count is None or coil_count <= 0:
        raise ValueError("Invalid Bruker EPI raw block geometry.")

    epi_matrix = np.asarray(_flatten_numeric(method.get("PVM_EpiMatrix")), dtype=int).ravel()
    ft_size = np.asarray(_flatten_numeric(reco.get("RECO_ft_size") or reco.get("RECO_size")), dtype=int).ravel()
    reco_size = np.asarray(_flatten_numeric(reco.get("RECO_size")), dtype=int).ravel()
    reco_offset = np.asarray(_flatten_numeric(reco.get("RECO_offset")), dtype=int).ravel()
    if epi_matrix.size < 2 or ft_size.size < 2 or reco_size.size < 2:
        raise ValueError("Bruker EPI import requires PVM_EpiMatrix, RECO_ft_size, and RECO_size.")

    readout = int(epi_matrix[0])
    acquired_phase = int(epi_matrix[1])
    if epi_matrix.size >= 3 and int(epi_matrix[2]) > 0:
        partition_count = int(epi_matrix[2])
    elif ft_size.size >= 3 and int(ft_size[2]) > 0:
        partition_count = int(ft_size[2])
    else:
        partition_count = 1

    kx_full, ky_full = (int(ft_size[0]), int(ft_size[1]))
    kz_full = int(ft_size[2]) if ft_size.size >= 3 and int(ft_size[2]) > 0 else int(partition_count)
    x_out, y_out = (int(reco_size[0]), int(reco_size[1]))
    z_out = int(reco_size[2]) if reco_size.size >= 3 and int(reco_size[2]) > 0 else 1
    reco_size3 = np.asarray([x_out, y_out, z_out], dtype=int)
    ft_size3 = np.asarray([kx_full, ky_full, kz_full], dtype=int)
    reco_offset3 = np.zeros((3,), dtype=int)
    if reco_offset.size > 0:
        reco_offset3[: min(3, int(reco_offset.size))] = reco_offset[: min(3, int(reco_offset.size))]
    if min(readout, acquired_phase, partition_count, kx_full, ky_full, kz_full, x_out, y_out, z_out) <= 0:
        raise ValueError("Invalid Bruker EPI reconstruction dimensions.")
    slice_crop = _bruker_epi_slice_crop_decision(
        mode=slice_crop_mode,
        reco_size=reco_size3,
        ft_size=ft_size3,
        reco_offset=reco_offset3,
        pre_crop_slices=kz_full,
        output_slices=z_out,
        sequence_type="epi",
        storage="pv360_raw_job",
    )
    z_offset = int(slice_crop["slice_crop_start"])

    shots = _as_positive_int(method.get("PVM_EpiNShots"), 1)
    echoes = _as_positive_int(method.get("PVM_EpiNEchoes"))
    if shots is None or echoes is None or shots <= 0 or echoes <= 0:
        raise ValueError("Invalid Bruker EPI shot/echo metadata.")
    echoes_per_object = int(acquired_phase) // int(shots)
    object_count = int(echoes) // max(echoes_per_object, 1)
    if echoes_per_object <= 0 or object_count <= 0:
        raise ValueError("Cannot split Bruker EPI echo train into reconstruction objects.")

    file_words = raw_path.stat().st_size // np.dtype(dtype).itemsize
    block_width = int(job_scan_size) * int(coil_count)
    scan_count = int(file_words) // block_width
    scans_per_frame = int(partition_count) * int(shots)
    required_scans = int(frame_count) * scans_per_frame
    if scan_count < required_scans:
        raise ValueError(
            f"Bruker EPI raw file has {scan_count} scans, expected at least {required_scans} "
            f"for {frame_count} frames."
        )

    sort_size_vals = np.asarray(_flatten_numeric(reco.get("RecoSortSize")), dtype=int).ravel()
    sort_maps_vals = np.asarray(_flatten_numeric(reco.get("RecoSortMaps")), dtype=int).ravel()
    line_count = int(sort_size_vals[0]) if sort_size_vals.size >= 1 and int(sort_size_vals[0]) > 0 else int(
        partition_count * acquired_phase
    )
    if sort_maps_vals.size < line_count:
        raise ValueError("Bruker EPI import requires RecoSortMaps for graph-ordered reconstruction.")
    destinations = sort_maps_vals[:line_count].astype(int)
    valid_destinations = (destinations >= 0) & (destinations < ky_full * kz_full)
    acquired_ky = np.unique(destinations[valid_destinations] % ky_full)

    rotate_mode = _bruker_epi_rotate_mode()
    epi_spatial_ndim = 3 if kz_full > 1 and (ft_size.size >= 3 or reco_size.size >= 3) else 2
    read_shifts, phase_shifts, slice_shifts, rotate_meta = _bruker_epi_frame_rotate_shifts(
        reco,
        frame_count=frame_count,
        kx_full=kx_full,
        ky_full=ky_full,
        kz_full=kz_full,
        spatial_ndim=epi_spatial_ndim,
        mode=rotate_mode,
    )
    pc_lin_mode = _bruker_epi_pc_lin_mode()
    (pc_read_shift, pc_phase_shift, pc_slice_shift), pc_lin_meta = _bruker_epi_pc_lin_shifts(
        reco,
        mode=pc_lin_mode,
    )
    read_shifts = (read_shifts + float(pc_read_shift)).astype(np.float32, copy=False)
    phase_shifts = (phase_shifts + float(pc_phase_shift)).astype(np.float32, copy=False)
    slice_shifts = (slice_shifts + float(pc_slice_shift)).astype(np.float32, copy=False)

    channel_scale = np.asarray(_flatten_numeric(reco.get("RecoScaleChan")), dtype=np.float32).ravel()
    if channel_scale.size != int(coil_count):
        channel_scale = np.ones((int(coil_count),), dtype=np.float32)
    channel_phase = np.asarray(_flatten_numeric(reco.get("RecoPhaseChan")), dtype=np.float32).ravel()
    if channel_phase.size != int(coil_count):
        channel_phase = np.zeros((int(coil_count),), dtype=np.float32)

    _phase_coeffs, phase_meta = _reshape_bruker_epi_phase_correction(method, int(coil_count))
    raw = np.memmap(raw_path, dtype=dtype, mode="r", shape=(scan_count, int(coil_count), int(job_scan_size)))
    kspace_out = np.empty((int(frame_count), int(coil_count), x_out, y_out, z_out), dtype=np.complex64)
    raw_image_out = np.empty((int(frame_count), x_out, y_out, z_out), dtype=np.float32)
    debug_pre_crop_enabled = str(os.environ.get("MRAX_BRUKER_EPI_DEBUG_PRE_CROP", "0") or "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    debug_pre_crop_out = (
        np.empty((int(frame_count), x_out, y_out, kz_full), dtype=np.float32) if debug_pre_crop_enabled else None
    )

    points_used = int(echoes) * int(readout)
    prefix_nav_points, prefix_nav_meta = _bruker_epi_prefix_nav_points(method, int(readout), reco=reco)
    if int(job_scan_size) < 2 * int(prefix_nav_points + points_used):
        raise ValueError("Bruker EPI raw block is smaller than prefix navigator plus image echo train.")
    if object_count == 2 and int(echoes) == 2 * int(echoes_per_object):
        object_echo_indices = [np.arange(start, int(echoes), object_count, dtype=int) for start in range(object_count)]
        object_split = "interleaved_echo_parity"
    else:
        object_echo_indices = [
            np.arange(i * echoes_per_object, min((i + 1) * echoes_per_object, int(echoes)), dtype=int)
            for i in range(object_count)
        ]
        object_split = "contiguous_echo_blocks"
    destination_y = destinations[valid_destinations] % ky_full
    destination_z = destinations[valid_destinations] // ky_full
    is_2d_epi_reco = _bruker_epi_is_2d_reco(method, reco)
    stab_amplitude_enabled = (
        is_2d_epi_reco and object_split == "interleaved_echo_parity" and _bruker_epi_stage_has_stab_amplitude(reco)
    )
    half_fourier_filter_meta = _parse_bruker_half_fourier_filter(reco, direction=1)
    half_fourier_mode = _bruker_epi_half_fourier_mode(method, reco)
    half_fourier_iterations = _bruker_epi_half_fourier_iterations()
    output_is_2d_stack = bool(z_out == 1 and (reco_size.size < 3 or ft_size.size < 3 or partition_count == 1))
    readout_regrid_options = _bruker_epi_readout_regrid_options(method, reco)
    readout_regrid_meta: dict = {"applied": False, **readout_regrid_options}
    readout_density_options = _bruker_epi_readout_density_options(method, reco, readout_regrid_options)
    readout_density_meta: dict = {"applied": False, **readout_density_options}
    b0_meta: dict | None = None
    echo_time_shift_meta: dict | None = None
    segment_stability_meta: dict = {
        "applied": False,
        "reason": "RecoStabCorrFilter_navigator_or_segment_internals_not_reimplemented_without_stored_navigator_data",
        "stab_corr": _bruker_stab_corr_metadata(reco),
    }
    amplitude_balance_meta: dict = {
        "applied": False,
        "reason": "not_evaluated",
    }
    amplitude_weight_min = np.inf
    amplitude_weight_max = -np.inf
    even_odd_options = _bruker_epi_even_odd_stabilization_options(method, reco)
    even_odd_stabilization_meta: dict = {
        "applied": False,
        "enabled": bool(even_odd_options["enabled"]),
        "mode": even_odd_options["mode"],
        "source": even_odd_options["source"],
        "reason": "not_evaluated",
    }
    ghost_filter_options = _bruker_epi_ghost_filter_options(method, reco)
    ghost_filter_meta: dict = {
        "applied": False,
        "enabled": bool(ghost_filter_options["enabled"]),
        "mode": ghost_filter_options["mode"],
        "requested_options": ghost_filter_options,
    }
    orientation_meta = _bruker_epi_prefix_nav_orientation_options(prefix_nav_meta, object_split=object_split)

    def _object_read_images_for_frame(
        frame_idx: int,
    ) -> tuple[list[np.ndarray], list[np.ndarray], dict | None, dict | None]:
        nonlocal readout_regrid_meta, readout_density_meta, even_odd_stabilization_meta
        scan_start = int(frame_idx) * scans_per_frame
        scan_stop = scan_start + scans_per_frame
        word_start = 2 * int(prefix_nav_points)
        word_stop = 2 * int(prefix_nav_points + points_used)
        frame_words = np.asarray(raw[scan_start:scan_stop, :, word_start:word_stop])
        real = frame_words[:, :, 0::2].astype(np.float32, copy=False)
        imag = frame_words[:, :, 1::2].astype(np.float32, copy=False)
        frame_samples = real + 1j * imag
        frame_samples = frame_samples.reshape(
            (int(partition_count), int(shots), int(coil_count), int(echoes), int(readout))
        )
        read_shift = float(read_shifts[int(frame_idx)])
        frame_samples = frame_samples[::-1, ...]
        if readout_regrid_options["enabled"] and readout_regrid_options["stage"] == "before_b0":
            frame_samples, readout_regrid_meta = _apply_bruker_epi_readout_regrid(
                frame_samples,
                method,
                mode=str(readout_regrid_options["mode"]),
            )
            readout_regrid_meta = {**readout_regrid_meta, "stage": readout_regrid_options["stage"]}
        frame_samples, frame_b0_meta = _apply_bruker_epi_b0_trajectory_phase(
            frame_samples,
            method,
            reco=reco,
        )
        frame_samples, frame_readout_density_meta = _apply_bruker_epi_readout_density_compensation(
            frame_samples,
            method,
            enabled=bool(readout_density_options["enabled"]),
        )
        readout_density_meta = {
            **readout_density_options,
            **frame_readout_density_meta,
        }
        if (
            not bool(readout_density_options["enabled"])
            and frame_readout_density_meta.get("reason") == "disabled"
            and readout_density_options.get("reason")
        ):
            readout_density_meta["reason"] = readout_density_options["reason"]
        if readout_regrid_options["enabled"] and readout_regrid_options["stage"] == "after_b0":
            frame_samples, readout_regrid_meta = _apply_bruker_epi_readout_regrid(
                frame_samples,
                method,
                mode=str(readout_regrid_options["mode"]),
            )
            readout_regrid_meta = {**readout_regrid_meta, "stage": readout_regrid_options["stage"]}
        frame_samples = _apply_bruker_epi_alternating_readout_reversal(frame_samples, first_echo=0)
        if readout_regrid_options["enabled"] and readout_regrid_options["stage"] == "after_reversal":
            frame_samples, readout_regrid_meta = _apply_bruker_epi_readout_regrid(
                frame_samples,
                method,
                mode=str(readout_regrid_options["mode"]),
            )
            readout_regrid_meta = {**readout_regrid_meta, "stage": readout_regrid_options["stage"]}
        frame_samples, frame_echo_time_shift_meta = _apply_bruker_epi_echo_time_shifts(
            frame_samples,
            method,
            object_echo_indices,
        )
        frame_samples = _apply_bruker_epi_phase_correction(
            frame_samples,
            _phase_coeffs,
            readout=readout,
            frame_index=frame_idx,
        )
        if readout_regrid_options["enabled"] and readout_regrid_options["stage"] == "after_echo_phase":
            frame_samples, readout_regrid_meta = _apply_bruker_epi_readout_regrid(
                frame_samples,
                method,
                mode=str(readout_regrid_options["mode"]),
            )
            readout_regrid_meta = {**readout_regrid_meta, "stage": readout_regrid_options["stage"]}

        frame_object_read_images: list[np.ndarray] = []
        frame_object_raw_lines: list[np.ndarray] = []
        for echo_indices in object_echo_indices:
            if echo_indices.size <= 0:
                continue

            echo_block = frame_samples[:, :, :, echo_indices, :]
            lines = np.moveaxis(echo_block, 2, 0).reshape(
                int(coil_count),
                int(partition_count) * int(shots) * int(echo_indices.size),
                int(readout),
            )
            lines = lines[:, :line_count, :]
            lines = _bruker_epi_axis_phase_ramp(lines, -1, read_shift)
            frame_object_raw_lines.append(np.asarray(lines, dtype=np.complex64))
            read_image = _bruker_epi_centered_ft(lines, axis=-1, inverse=False)
            if int(readout) > 0:
                read_image = read_image / float(readout)
            read_image, frame_even_odd_meta = _apply_bruker_epi_even_odd_line_stabilization(
                read_image,
                enabled=bool(even_odd_options["enabled"]),
            )
            even_odd_stabilization_meta = {
                **frame_even_odd_meta,
                "enabled": bool(even_odd_options["enabled"]),
                "requested_options": even_odd_options,
            }
            frame_object_read_images.append(read_image)
        return frame_object_read_images, frame_object_raw_lines, frame_b0_meta, frame_echo_time_shift_meta

    half_fourier_meta: dict = {"applied": False, "reason": "not_evaluated"}
    ramp_delay_scale, ramp_delay_scale_meta = _bruker_epi_ramp_delay_scale(method, readout)
    timing_alignment_shifts: list[float] = []
    timing_alignment_applied = False
    timing_alignment_detail: dict = {"applied": False, "reason": "not_evaluated"}
    for frame_idx in range(int(frame_count)):
        phase_shift = float(phase_shifts[int(frame_idx)])
        slice_shift = float(slice_shifts[int(frame_idx)])
        object_read_images, object_raw_lines, b0_meta, echo_time_shift_meta = _object_read_images_for_frame(frame_idx)

        amplitude_weights, frame_amplitude_meta = _bruker_epi_readout_polarity_amplitude_weights(
            object_read_images,
            enabled=stab_amplitude_enabled,
        )
        amplitude_balance_meta = frame_amplitude_meta
        if amplitude_weights is not None:
            amplitude_weight_min = min(amplitude_weight_min, *(float(np.min(w)) for w in amplitude_weights))
            amplitude_weight_max = max(amplitude_weight_max, *(float(np.max(w)) for w in amplitude_weights))

        object_coil_images: list[np.ndarray] = []
        pre_crop_object_coil_images: list[np.ndarray] = []
        object_measured_kspaces: list[np.ndarray] = []
        for object_idx, read_image in enumerate(object_read_images):
            if amplitude_weights is not None and object_idx < len(amplitude_weights):
                read_image = read_image * amplitude_weights[object_idx][:, None, :]

            measured_grid = np.zeros((int(coil_count), kx_full, ky_full, kz_full), dtype=np.complex64)
            lines = object_raw_lines[object_idx] if object_idx < len(object_raw_lines) else read_image
            measured_lines_x_line = np.moveaxis(lines, 2, 1)
            measured_grid[:, :, destination_y, destination_z] = measured_lines_x_line[:, :, valid_destinations]
            measured_grid = _bruker_epi_axis_phase_ramp(measured_grid, -1, slice_shift)
            measured_grid = _bruker_epi_axis_phase_ramp(measured_grid, -2, phase_shift)

            grid = np.zeros((int(coil_count), kx_full, ky_full, kz_full), dtype=np.complex64)
            read_image_x_line = np.moveaxis(read_image, 2, 1)
            grid[:, :, destination_y, destination_z] = read_image_x_line[:, :, valid_destinations]
            grid = _bruker_epi_axis_phase_ramp(grid, -1, slice_shift)
            image = _bruker_epi_centered_ft(grid, axis=-1, inverse=True)
            image = _bruker_epi_axis_phase_ramp(image, -2, phase_shift)
            if half_fourier_filter_meta.get("enabled") and half_fourier_mode == "pocs":
                image, half_fourier_meta = _apply_bruker_epi_phase_half_fourier(
                    image,
                    acquired_indices=acquired_ky,
                    axis=-2,
                    echo_position=half_fourier_filter_meta.get("echo_position"),
                    iterations=half_fourier_iterations,
                )
                half_fourier_meta = {
                    **half_fourier_filter_meta,
                    **half_fourier_meta,
                    "selected_mode": half_fourier_mode,
                    "iterations": int(half_fourier_iterations),
                }
            elif half_fourier_filter_meta.get("enabled") and half_fourier_mode == "homodyne":
                image, half_fourier_meta = _apply_bruker_epi_phase_homodyne(
                    image,
                    acquired_indices=acquired_ky,
                    axis=-2,
                    echo_position=half_fourier_filter_meta.get("echo_position"),
                )
                half_fourier_meta = {
                    **half_fourier_filter_meta,
                    **half_fourier_meta,
                    "selected_mode": half_fourier_mode,
                    "iterations": int(half_fourier_iterations),
                }
            else:
                image = _bruker_epi_centered_ft(image, axis=-2, inverse=True)
                half_fourier_meta = {
                    **half_fourier_filter_meta,
                    "applied": False,
                    "mode": "zero_filled_phase_ifft",
                    "selected_mode": half_fourier_mode,
                    "iterations": int(half_fourier_iterations),
                }

            for coil_idx in range(int(coil_count)):
                channel_correction = channel_scale[coil_idx] * np.exp(1j * np.deg2rad(channel_phase[coil_idx]))
                image[coil_idx] *= channel_correction
                measured_grid[coil_idx] *= channel_correction

            image_full = image[:, :x_out, :y_out, :]
            image = image_full[:, :, :, z_offset : z_offset + z_out]
            if image.shape[-1] != z_out:
                raise ValueError("Bruker EPI RECO_size/slice_crop_mode crop is outside the reconstructed volume.")
            measured_grid = measured_grid[:, :x_out, :y_out, z_offset : z_offset + z_out]
            if measured_grid.shape[-1] != z_out:
                raise ValueError("Bruker EPI RECO_size/slice_crop_mode crop is outside the reconstructed volume.")
            object_measured_kspaces.append(np.asarray(measured_grid, dtype=np.complex64))
            if debug_pre_crop_out is not None:
                pre_crop_object_coil_images.append(np.asarray(image_full, dtype=np.complex64))
            object_coil_images.append(np.asarray(image, dtype=np.complex64))

        if not object_coil_images:
            raise ValueError("Bruker EPI echo split did not produce any image objects.")
        if object_split == "interleaved_echo_parity" and len(object_coil_images) == 2:
            reference_mag = np.sqrt(np.sum(np.abs(object_coil_images[0]) ** 2, axis=0)).astype(np.float32)
            moving_mag = np.sqrt(np.sum(np.abs(object_coil_images[1]) ** 2, axis=0)).astype(np.float32)
            shift, timing_alignment_detail = _bruker_epi_estimate_readout_shift(reference_mag, moving_mag)
            if timing_alignment_detail.get("applied"):
                shift = float(shift) * float(ramp_delay_scale)
                object_coil_images[1] = _bruker_epi_fractional_shift(object_coil_images[1], axis=1, shift=shift)
                if debug_pre_crop_out is not None and len(pre_crop_object_coil_images) == 2:
                    pre_crop_object_coil_images[1] = _bruker_epi_fractional_shift(
                        pre_crop_object_coil_images[1],
                        axis=1,
                        shift=shift,
                    )
                timing_alignment_shifts.append(float(shift))
                timing_alignment_applied = True

        ghost_corrected_images: list[np.ndarray] = []
        frame_ghost_meta: dict | None = None
        for image in object_coil_images:
            corrected_image, frame_ghost_meta = _apply_bruker_epi_fov2_ghost_filter(
                image,
                enabled=bool(ghost_filter_options["enabled"]),
                phase_axis=2,
                strength=float(ghost_filter_options["strength"]),
                max_abs_coefficient=float(ghost_filter_options["max_abs_coefficient"]),
                max_order=int(ghost_filter_options["max_order"]),
            )
            ghost_corrected_images.append(corrected_image)
        object_coil_images = ghost_corrected_images
        ghost_filter_meta = {
            **(frame_ghost_meta or ghost_filter_meta),
            "enabled": bool(ghost_filter_options["enabled"]),
            "requested_options": ghost_filter_options,
            "frames": int(frame_idx + 1),
        }

        if bool(orientation_meta.get("applied")):
            object_coil_images = [np.flip(image, axis=1) for image in object_coil_images]
            if debug_pre_crop_out is not None:
                pre_crop_object_coil_images = [np.flip(image, axis=1) for image in pre_crop_object_coil_images]

        object_magnitudes = [
            np.sqrt(np.sum(np.abs(image) ** 2, axis=0)).astype(np.float32, copy=False)
            for image in object_coil_images
        ]
        raw_image_out[frame_idx] = np.mean(np.stack(object_magnitudes, axis=0), axis=0).astype(np.float32)
        if debug_pre_crop_out is not None:
            pre_crop_magnitudes = [
                np.sqrt(np.sum(np.abs(image) ** 2, axis=0)).astype(np.float32, copy=False)
                for image in pre_crop_object_coil_images
            ]
            debug_pre_crop_out[frame_idx] = np.mean(np.stack(pre_crop_magnitudes, axis=0), axis=0).astype(np.float32)
        if object_measured_kspaces:
            kspace_out[frame_idx] = object_measured_kspaces[0]
        else:
            kspace_out[frame_idx] = _coil_images_to_cartesian_kspace(object_coil_images[0])

    if amplitude_balance_meta.get("applied"):
        amplitude_balance_meta = {
            **amplitude_balance_meta,
            "frames": int(frame_count),
            "weight_range": [float(amplitude_weight_min), float(amplitude_weight_max)],
        }
    if timing_alignment_applied:
        shift_arr = np.asarray(timing_alignment_shifts, dtype=np.float32)
        timing_alignment_meta = {
            "applied": True,
            "source": "RecoStabCorrFilter(useNav=true)",
            "mode": "readout_profile_phase_correlation_fractional_shift",
            "axis": "readout",
            "reference_object": 0,
            "moving_object": 1,
            "frames": int(shift_arr.size),
            "shift_pixels": {
                "min": float(np.min(shift_arr)),
                "median": float(np.median(shift_arr)),
                "max": float(np.max(shift_arr)),
            },
            "per_frame_shift_pixels": np.round(shift_arr, 6).astype(float).tolist(),
            "ramp_delay_scale": ramp_delay_scale_meta,
            "detail": timing_alignment_detail,
        }
    else:
        timing_alignment_meta = timing_alignment_detail

    sampling_mask = np.zeros((x_out, y_out, z_out), dtype=bool)
    sampling_mask[:, np.clip(acquired_ky, 0, y_out - 1), :] = True

    raw_image_result = raw_image_out[..., 0] if output_is_2d_stack else raw_image_out
    kspace_result = kspace_out[..., 0] if output_is_2d_stack else kspace_out
    sampling_mask_result = sampling_mask[..., 0] if output_is_2d_stack else sampling_mask
    output_axis_swap = bool(output_is_2d_stack and int(x_out) == int(y_out))
    kspace_axis_order = "read_phase" if output_is_2d_stack else "slice_read_phase"
    if output_axis_swap:
        raw_image_result = np.swapaxes(raw_image_result, -2, -1)
        kspace_result = np.swapaxes(kspace_result, -2, -1)
        sampling_mask_result = np.swapaxes(sampling_mask_result, -2, -1)
        kspace_axis_order = "phase_read"
    elif not output_is_2d_stack:
        kspace_result = np.moveaxis(kspace_result, -1, -3)
        sampling_mask_result = np.moveaxis(sampling_mask_result, -1, -3)
    debug_pre_crop_result = (
        debug_pre_crop_out[..., 0]
        if output_is_2d_stack and isinstance(debug_pre_crop_out, np.ndarray) and debug_pre_crop_out.shape[-1] == 1
        else debug_pre_crop_out
    )
    if output_axis_swap and isinstance(debug_pre_crop_result, np.ndarray):
        debug_pre_crop_result = np.swapaxes(debug_pre_crop_result, -2, -1)

    return kspace_result, raw_image_result, sampling_mask_result, {
        "storage": "pv360_raw_job",
        "scan_count": int(scan_count),
        "job_scan_size": int(job_scan_size),
        "coil_count": int(coil_count),
        "raw_points_per_scan": int(job_scan_size // 2),
        "frame_count": int(frame_count),
        "scans_per_frame": int(scans_per_frame),
        "scan_order": "frame_reverse_partition_shot",
        "grid_order": "frame_coil_read_phase" if output_is_2d_stack else "frame_coil_slice_read_phase",
        "kspace_axis_order": kspace_axis_order,
        "epi_matrix": [int(readout), int(acquired_phase), int(partition_count)],
        "shots": int(shots),
        "raw_echoes": int(echoes),
        "echoes_per_object": int(echoes_per_object),
        "object_count": int(object_count),
        "reconstruction_candidate": "graph_ordered_epi",
        "candidate_layout": "2d_object_frames" if output_is_2d_stack else "3d_object_frames",
        "output_axis_swap": "square_2d_epi_read_phase_to_visu_axes" if output_axis_swap else None,
        "object_split": object_split,
        "line_count": int(line_count),
        "valid_line_count": int(np.sum(valid_destinations)),
        "raw_receiver_layout": "receiver_contiguous_complex_pairs",
        "prefix_navigator": prefix_nav_meta,
        "partition_order": "raw_partitions_reversed_before_sorting",
        "readout_reversal": {
            "source": "RecoReverseFilter",
            "segment": _as_positive_int(reco.get("RecoReverseSegment")),
            "applied": True,
            "mode": "even_echoes",
        },
        "segment_stability": segment_stability_meta,
        "readout_polarity_amplitude_balance": amplitude_balance_meta,
        "even_odd_line_stabilization": even_odd_stabilization_meta,
        "readout_polarity_timing_alignment": timing_alignment_meta,
        "readout_regrid": readout_regrid_meta,
        "readout_density_compensation": readout_density_meta,
        "epi_ghost_filter": ghost_filter_meta,
        "final_orientation": orientation_meta,
        "half_fourier": {
            **half_fourier_filter_meta,
            **half_fourier_meta,
            "selected_mode": half_fourier_mode,
            "iterations": int(half_fourier_iterations),
        },
        "readout_ft": {
            "mode": "fft",
            "normalization": "divide_by_readout_samples",
            "scale": 1.0 / float(readout) if int(readout) > 0 else None,
        },
        "slice_ft": "ifft",
        "phase_ft": (
            f"RecoHalfFourierFilter_phase_{half_fourier_mode}"
            if half_fourier_meta.get("applied")
            else "ifft_zero_filled_partial_fourier"
        ),
        "object_combination": "mean_of_object_magnitudes",
        "coil_combination": "rss",
        "kspace_output": "sorted_raw_pre_reconstruction_first_epi_object_kspace",
        "sort_maps": {
            "source": "RecoSortMaps",
            "count": int(destinations.size),
            "target_size": int(ky_full * kz_full),
        },
        "epi_phase_correction": {
            **phase_meta,
            "mode": "per_receiver_readout_phase_ramp" if phase_meta.get("applied") else None,
        },
        "b0_trajectory_phase": b0_meta or {"applied": False},
        "echo_time_shifts": echo_time_shift_meta or {"applied": False},
        "rotate": {
            **rotate_meta,
            "pc_lin_added": [float(pc_read_shift), float(pc_phase_shift), float(pc_slice_shift)],
            "combined_read": read_shifts.astype(float).tolist(),
            "combined_phase": phase_shifts.astype(float).tolist(),
            "combined_slice": slice_shifts.astype(float).tolist(),
        },
        "pc_lin": pc_lin_meta,
        "reco_crop": {
            "debug_pre_crop_enabled": bool(debug_pre_crop_enabled),
            "pre_crop_shape": [int(frame_count), int(x_out), int(y_out), int(kz_full)],
            "output_shape": [int(frame_count), int(x_out), int(y_out), int(z_out)],
            "RECO_size": reco_size3.astype(int).tolist(),
            "RECO_ft_size": ft_size3.astype(int).tolist(),
            "RECO_offset": reco_offset3.astype(int).tolist(),
            "current_interpretation": "RECO_offset[2]_as_zero_based_slice_crop_start_after_slice_ft",
            "reco_offset": reco_offset3.astype(int).tolist(),
            **slice_crop,
        },
        "approximations": [
            "RecoStabCorrFilter phase/navigator internals are reported but not fully reproduced",
            "RecoRegridFilter ramp interpolation kernel is not fully reproduced; measured trajectory span scales the polarity delay estimate",
        ],
        "_debug_pre_crop_raw_image": debug_pre_crop_result,
    }


def _reconstruct_bruker_epi_best_effort_raw_job(
    raw_path: Path,
    dtype: np.dtype,
    *,
    frame_count: int,
    method: dict,
    reco: dict,
    acqp: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Return the best generic EPI raw preview when graph-ordered reconstruction cannot run."""

    kspace, sorting_meta = _reshape_bruker_epi_raw_job(
        raw_path,
        dtype,
        frame_count=frame_count,
        method=method,
        reco=reco,
        acqp=acqp,
    )
    arr = np.asarray(kspace)
    if arr.ndim != 5:
        raise ValueError(f"Best-effort Bruker EPI candidate expected 5D k-space, got shape {arr.shape}.")

    if int(arr.shape[2]) == 1 and len(_flatten_numeric(reco.get("RECO_size"))) < 3:
        kspace_for_reco = arr[:, :, 0, :, :]
        fft_axes = (-2, -1)
        candidate_layout = "2d_object_frames"
    else:
        kspace_for_reco = arr
        fft_axes = (-3, -2, -1)
        candidate_layout = "3d_object_frames"

    target_kx = int(kspace_for_reco.shape[-1]) if kspace_for_reco.ndim >= 2 else None
    kspace_for_reco, correction_meta = prepare_bruker_cartesian_kspace(
        kspace_for_reco,
        reco,
        target_kx=target_kx,
    )
    frame_transposition = _bruker_frame_transpositions(reco.get("RECO_transposition"), frame_count)
    image = cartesian_ifft(kspace_for_reco, axes=fft_axes)
    raw_image_out = format_bruker_cartesian_image(
        image,
        reco,
        frame_transposition=frame_transposition,
    )
    sampling_mask = make_sampling_mask(kspace_for_reco)
    spatial_ndim = len(fft_axes)
    if sampling_mask.ndim > spatial_ndim:
        sampling_mask = np.any(sampling_mask, axis=tuple(range(sampling_mask.ndim - spatial_ndim)))
    return np.asarray(kspace_for_reco), np.asarray(raw_image_out, dtype=np.float32), sampling_mask, {
        **sorting_meta,
        "reconstruction_candidate": "best_effort_cartesian_epi",
        "candidate_layout": candidate_layout,
        "fft_axes": list(fft_axes),
        "kspace_corrections": correction_meta,
        "approximations": list(sorting_meta.get("approximations", []))
        + [
            "best-effort Cartesian EPI preview used because graph-ordered EPI reconstruction was not applicable",
            "does not fully reproduce proprietary Bruker EPI navigator, ramp-sampling, ghost, or stability filters",
        ],
    }


def _load_bruker_raw_kspace(
    exp_dir: Path,
    frame_count: int,
    spatial_shape: tuple[int, ...],
    method: dict,
    reco: dict,
    acqp: dict | None = None,
    visu: dict | None = None,
    pipeline: RecoPipeline | Callable[[dict], RecoPipeline] | None = None,
    slice_crop_mode: str | None = None,
) -> tuple[np.ndarray | KSpaceImportDefinition | None, np.ndarray | None, np.ndarray | None, dict]:
    raw_path = _find_rawdat_file(exp_dir)
    if raw_path is None or not raw_path.exists():
        return None, None, None, {}

    acqp = (
        parse_jcamp(exp_dir / "acqp")
        if acqp is None and (exp_dir / "acqp").exists()
        else (acqp or {})
    )
    dtype_word = (
        acqp.get("ACQ_word_size")
        or acqp.get("GO_raw_data_format")
        or reco.get("RECO_wordtype")
        or method.get("PVM_DigDtype")
        or method.get("VisuCoreWordType")
    )
    acq_word = acqp.get("ACQ_word_size")
    dtype = np.dtype(np.float32)
    if dtype_word:
        try:
            if isinstance(dtype_word, str) and dtype_word.startswith("GO_"):
                dtype = np.dtype(_dtype_from_go_word(dtype_word))
            elif isinstance(dtype_word, str) and dtype_word.endswith("_BIT"):
                dtype = np.dtype(_dtype_from_acq_word(dtype_word))
            else:
                dtype = np.dtype(_dtype_from_word(dtype_word))
        except Exception:
            dtype = np.dtype(np.float32)
    byte_order = acqp.get("ACQ_byte_order") or acqp.get("BYTORDA")
    if byte_order and str(byte_order).lower().startswith("big"):
        dtype = dtype.newbyteorder(">")

    raw_frame_count = _resolve_bruker_raw_frame_count(
        frame_count=frame_count,
        spatial_shape=spatial_shape,
        method=method,
        acqp=acqp,
    )
    frame_order = _resolve_bruker_frame_order(method, acqp, raw_frame_count)
    package_indices = _bruker_frame_package_indices(method, raw_frame_count)
    mixed_slice_geometry = bool(visu and _bruker_has_mixed_slice_geometry(visu, raw_frame_count))
    if len(spatial_shape) < 2:
        return None, None, None, {
            "rawdat_path": str(raw_path),
            "raw_frame_count": int(raw_frame_count),
            "frame_order": None if frame_order is None else frame_order.tolist(),
            "frame_packages": None if package_indices is None else package_indices.tolist(),
            "mixed_slice_geometry": mixed_slice_geometry,
            "sequence_type": _bruker_sequence_type(method, visu),
            "kspace_ready": False,
            "unsupported_reason": "1D Bruker raw reconstruction is not supported",
            "acq_word_size": acq_word,
        }
    raw_shape = spatial_shape
    inp_size = reco.get("RECO_inp_size") or method.get("PVM_EncMatrix")
    if isinstance(inp_size, list) and len(inp_size) >= 2 and all(isinstance(v, (int, float)) for v in inp_size[:2]):
        # Bruker uses (read, phase, partition) ordering for sizes. Our internal
        # Cartesian convention keeps readout last and phase at axis=-2; for 3D
        # scans the partition/slice-encode axis therefore sits before phase/read.
        if len(spatial_shape) == 3 and len(inp_size) >= 3 and isinstance(inp_size[2], (int, float)):
            raw_shape = (int(inp_size[2]), int(inp_size[1]), int(inp_size[0]))
        else:
            raw_shape = (int(inp_size[1]), int(inp_size[0]))

    sat_on = _sat_trans_enabled(method)
    echo_images = method.get("PVM_NEchoImages")
    is_multi_echo = False
    if echo_images is not None and not sat_on:
        try:
            is_multi_echo = int(echo_images) == int(raw_frame_count) and int(echo_images) > 1
        except (TypeError, ValueError):
            is_multi_echo = False

    coil_count = None
    coil_token = method.get("PVM_EncNReceivers") or method.get("PVM_NReceivers")
    if coil_token is not None:
        try:
            coil_count = int(coil_token)
        except (TypeError, ValueError):
            coil_count = None
    receivers = int(coil_count) if coil_count is not None else 1

    sequence_type = _bruker_sequence_type(method, visu)
    regrid_meta = _bruker_regrid_metadata(method, reco)
    if regrid_meta is not None:
        base_meta = {
            "rawdat_path": str(raw_path),
            "raw_frame_count": int(raw_frame_count),
            "frame_order": None if frame_order is None else frame_order.tolist(),
            "frame_packages": None if package_indices is None else package_indices.tolist(),
            "mixed_slice_geometry": mixed_slice_geometry,
            "sequence_type": sequence_type,
            "kspace_ready": False,
            "kind": "non_cartesian",
            "regrid": dict(regrid_meta),
            "acq_word_size": acq_word,
        }
        spatial_ndim = int(regrid_meta["dim"])
        if spatial_ndim not in (2, 3):
            return None, None, None, {
                **base_meta,
                "unsupported_reason": "Only 2D and 3D Bruker non-Cartesian trajectories are currently imported.",
            }

        trajectory, trajectory_meta = read_bruker_regrid_trajectory(exp_dir, method=method, reco=reco)
        raw_blocks = _read_bruker_pv360_raw_job_blocks(raw_path, dtype, method=method, acqp=acqp)
        if raw_blocks is not None:
            blocks, raw_storage_meta = raw_blocks
            kspace, sorting_meta = _reshape_bruker_non_cartesian_blocks(
                blocks,
                frame_count=raw_frame_count,
                trajectory_samples=int(trajectory.shape[0]),
                method=method,
                reco=reco,
                regrid_meta=regrid_meta,
            )
            sorting_meta["raw_storage"] = raw_storage_meta
            frame_order_for_output = None
        else:
            samples = _read_rawdat_complex(raw_path, dtype)
            kspace, sorting_meta = _reshape_bruker_non_cartesian_samples(
                samples,
                frame_count=raw_frame_count,
                coil_count=receivers,
                trajectory_samples=int(trajectory.shape[0]),
                method=method,
                reco=reco,
                regrid_meta=regrid_meta,
            )
            frame_order_for_output = frame_order
        if len(spatial_shape) >= spatial_ndim:
            spatial_nc = tuple(int(v) for v in spatial_shape[:spatial_ndim])
        else:
            spatial_nc = tuple(int(v) for v in raw_shape[:spatial_ndim])
        trajectory_nc = np.asarray(trajectory[:, :spatial_ndim], dtype=np.float32)

        kspace, b0_meta = _apply_bruker_b0_correction(
            kspace,
            exp_dir=exp_dir,
            reco=reco,
            regrid_meta=regrid_meta,
        )
        kspace, correction_meta = prepare_bruker_non_cartesian_kspace(kspace, reco)
        applied_corrections = []
        if b0_meta.get("applied"):
            applied_corrections.append("B0Correction")
        applied_corrections.extend(correction_meta.get("applied_kspace_corrections", []))
        metadata_only_steps = list(correction_meta.get("unhandled_reco_steps", []))
        inverse_fid_meta = sorting_meta.get("profile_preprocessing", {}).get("inverse_fid", {})
        if inverse_fid_meta.get("applied"):
            applied_corrections.append("RecoInverseFidFilter(approximate)")
            if inverse_fid_meta.get("readout_shift_applied"):
                applied_corrections.append("RecoFTShiftFilter(readout approximate)")
            metadata_only_steps = [
                step
                for step in metadata_only_steps
                if step not in {"RecoInverseFidFilter", "RecoShuffleProfilesFilter"}
            ]
        kspace = _reorder_bruker_frames(kspace, frame_order_for_output)

        density, density_meta = read_bruker_regrid_density(
            exp_dir,
            trajectory,
            spatial_shape=spatial_nc,
            method=method,
            reco=reco,
        )
        _, vendor_regrid_meta = bruker_vendor_regrid_options(reco, spatial_shape=spatial_nc)
        mrax_density = estimate_radial_density(trajectory_nc, spatial_shape=spatial_nc)
        kspace_def = build_non_cartesian_kspace_definition(
            np.asarray(kspace),
            trajectory=trajectory_nc,
            density=np.asarray(mrax_density, dtype=np.float32),
            spatial_shape=spatial_nc,
            sample_axis=-1,
            sequence_type=sequence_type,
            metadata={
                **base_meta,
                "kspace_ready": True,
                "trajectory": trajectory_meta,
                "density": {
                    "correction_type": "rho",
                    "source": "trajectory",
                    "density_samples": int(mrax_density.shape[0]),
                    "normalized_to_spatial_pixels": int(np.prod(spatial_nc)),
                    "usage": "DataSet.density for MRax NUFFT/reconstruction",
                },
                "vendor_density": density_meta,
                "vendor_regrid_options": vendor_regrid_meta,
                "sample_sorting": sorting_meta,
                "vendor_corrections": {
                    "kspace": applied_corrections,
                    "image": [
                        "density_compensation",
                        "regridding",
                        "cutoff",
                        "apodization_correction",
                        "mask",
                        "sum_of_squares",
                        "magnitude",
                    ],
                    "metadata_only": metadata_only_steps,
                },
                "b0_correction": b0_meta,
                "kspace_corrections_applied": applied_corrections,
            },
        )
        raw_image_out = np.asarray(
            reconstruct_regrid(
                kspace_def.samples,
                np.asarray(kspace_def.trajectory),
                spatial_shape=spatial_nc,
                density=np.asarray(density, dtype=np.float32),
            )
        )
        raw_image_out = np.abs(raw_image_out).astype(np.float32)
        if spatial_ndim == 2:
            raw_image_out = _pack_bruker_2d_frame_images(
                raw_image_out,
                frame_count=frame_count,
                raw_frame_count=raw_frame_count,
                spatial_shape=spatial_shape,
                prefer_transpose=True,
            )
        return kspace_def, raw_image_out, None, dict(kspace_def.metadata)

    if sequence_type == "epi":
        candidate_errors: list[dict[str, str]] = []
        try:
            kspace, raw_image_out, sampling_mask, sorting_meta = _reconstruct_bruker_epi_raw_job(
                raw_path,
                dtype,
                frame_count=raw_frame_count,
                method=method,
                reco=reco,
                acqp=acqp,
                visu=visu,
                slice_crop_mode=slice_crop_mode,
            )
        except ValueError as exc:
            candidate_errors.append({"candidate": "graph_ordered_epi", "reason": str(exc)})
            try:
                kspace, raw_image_out, sampling_mask, sorting_meta = _reconstruct_bruker_epi_best_effort_raw_job(
                    raw_path,
                    dtype,
                    frame_count=raw_frame_count,
                    method=method,
                    reco=reco,
                    acqp=acqp,
                )
            except ValueError as fallback_exc:
                candidate_errors.append({"candidate": "best_effort_cartesian_epi", "reason": str(fallback_exc)})
                return None, None, None, {
                    "rawdat_path": str(raw_path),
                    "raw_frame_count": int(raw_frame_count),
                    "frame_order": None if frame_order is None else frame_order.tolist(),
                    "frame_packages": None if package_indices is None else package_indices.tolist(),
                    "mixed_slice_geometry": mixed_slice_geometry,
                    "sequence_type": sequence_type,
                    "kind": "epi",
                    "kspace_ready": False,
                    "reconstruction_status": "unavailable",
                    "reconstruction_candidate": None,
                    "candidate_errors": candidate_errors,
                    "raw_reconstruction_note": _BRUKER_EPI_UNAVAILABLE_NOTE,
                    "acq_word_size": acq_word,
                }
        if candidate_errors:
            sorting_meta = {
                **sorting_meta,
                "candidate_errors": candidate_errors,
            }

        debug_pre_crop_raw_image = sorting_meta.pop("_debug_pre_crop_raw_image", None)

        frame_transposition = _bruker_frame_transpositions(reco.get("RECO_transposition"), raw_frame_count)
        if not np.any(frame_transposition > 0) and visu is not None:
            frame_transposition = _bruker_frame_transpositions(visu.get("VisuCoreTransposition"), raw_frame_count)

        kspace = _reorder_bruker_frames(kspace, frame_order)
        raw_image_out = _reorder_bruker_frames(raw_image_out, frame_order)
        if isinstance(debug_pre_crop_raw_image, np.ndarray):
            debug_pre_crop_raw_image = _reorder_bruker_frames(debug_pre_crop_raw_image, frame_order)
        frame_transposition = _reorder_bruker_frames(frame_transposition, frame_order)
        raw_image_out = _pack_bruker_2d_frame_images(
            raw_image_out,
            frame_count=frame_count,
            raw_frame_count=raw_frame_count,
            spatial_shape=spatial_shape,
        )
        amplitude_balance_applied = bool(
            sorting_meta.get("readout_polarity_amplitude_balance", {}).get("applied", False)
        )
        timing_alignment_applied = bool(
            sorting_meta.get("readout_polarity_timing_alignment", {}).get("applied", False)
        )
        segment_stability_applied = bool(sorting_meta.get("segment_stability", {}).get("applied", False))
        half_fourier_applied = bool(sorting_meta.get("half_fourier", {}).get("applied", False))
        ramp_delay_scale_applied = bool(
            sorting_meta.get("readout_polarity_timing_alignment", {})
            .get("ramp_delay_scale", {})
            .get("applied", False)
        )
        readout_regrid_applied = bool(sorting_meta.get("readout_regrid", {}).get("applied", False))
        readout_density_applied = bool(sorting_meta.get("readout_density_compensation", {}).get("applied", False))
        b0_trajectory_phase_applied = bool(sorting_meta.get("b0_trajectory_phase", {}).get("applied", False))
        pc_lin_applied = bool(sorting_meta.get("pc_lin", {}).get("applied", False))
        echo_time_shift_applied = bool(sorting_meta.get("echo_time_shifts", {}).get("applied", False))
        epi_phase_correction_applied = bool(sorting_meta.get("epi_phase_correction", {}).get("applied", False))
        ghost_filter_applied = bool(sorting_meta.get("epi_ghost_filter", {}).get("applied", False))
        vendor_image_steps = [
            "receiver_contiguous_raw_unpack",
            "RecoReverseFilter(alternating readout rows)",
            "readout_ft",
            "RecoSortMaps",
            "slice_ft",
        ]
        regrid_anchor = "RecoReverseFilter(alternating readout rows)"
        if b0_trajectory_phase_applied:
            vendor_image_steps.insert(1, "RecoRegridFilter(B0 trajectory phase)")
            regrid_anchor = "RecoRegridFilter(B0 trajectory phase)"
        vendor_image_steps.append("RecoHalfFourierFilter(phase POCS)" if half_fourier_applied else "phase_ft")
        if segment_stability_applied:
            vendor_image_steps.insert(1, "RecoStabCorrFilter(segment amplitude/phase stability)")
        if amplitude_balance_applied:
            vendor_image_steps.insert(1, "RecoStabCorrFilter(readout polarity amplitude balance)")
        if timing_alignment_applied:
            insert_idx = 2 if amplitude_balance_applied else 1
            vendor_image_steps.insert(insert_idx, "RecoStabCorrFilter(readout polarity timing alignment)")
        if ramp_delay_scale_applied:
            vendor_image_steps.insert(
                vendor_image_steps.index(regrid_anchor) + 1,
                "RecoRegridFilter(readout trajectory delay scaling)",
            )
        if readout_regrid_applied:
            vendor_image_steps.insert(
                vendor_image_steps.index(regrid_anchor) + 1,
                "RecoRegridFilter(readout trajectory interpolation auto/opt-in)",
            )
        if readout_density_applied:
            vendor_image_steps.insert(
                vendor_image_steps.index(regrid_anchor) + 1,
                "RecoRegridFilter(readout trajectory density compensation)",
            )
        if pc_lin_applied:
            vendor_image_steps.insert(vendor_image_steps.index("slice_ft"), "RECO_pc_lin(first-order opt-in)")
        if echo_time_shift_applied:
            vendor_image_steps.insert(vendor_image_steps.index("readout_ft"), "PVM_EpiEchoTimeShifting")
        if epi_phase_correction_applied:
            vendor_image_steps.insert(vendor_image_steps.index("readout_ft"), "PVM_EpiPhaseCorrection")
        if ghost_filter_applied:
            vendor_image_steps.append("RecoEpiGhostFilter(FOV/2 leakage correction)")
        if reco.get("RECO_size") is not None or reco.get("RECO_offset") is not None:
            vendor_image_steps.append("RECO_size/RECO_offset")
        vendor_image_steps.extend(["sum_of_squares", "magnitude", "echo_object_average"])
        if sorting_meta.get("prefix_navigator", {}).get("applied"):
            vendor_image_steps.insert(1, "PVM_EpiPrefixNavSize(skip prefix navigator)")

        applied_corrections = [
            "receiver_contiguous_raw_unpack",
            "reverse_partition_scan_order",
            "RecoReverseFilter(alternating readout rows)",
            "RecoSortMaps",
            "RecoFTFilter(readout)",
            "RecoFTFilter(slice)",
            "RecoScaleChan",
            "RecoPhaseChan",
        ]
        regrid_correction_anchor = "RecoReverseFilter(alternating readout rows)"
        if b0_trajectory_phase_applied:
            applied_corrections.insert(2, "RecoRegridFilter(B0 trajectory phase)")
            regrid_correction_anchor = "RecoRegridFilter(B0 trajectory phase)"
        applied_corrections.insert(
            applied_corrections.index("RecoScaleChan"),
            "RecoHalfFourierFilter(phase POCS)" if half_fourier_applied else "RecoFTFilter(phase)",
        )
        if segment_stability_applied:
            applied_corrections.insert(2, "RecoStabCorrFilter(segment amplitude/phase stability)")
        if amplitude_balance_applied:
            applied_corrections.insert(2, "RecoStabCorrFilter(readout polarity amplitude balance)")
        if timing_alignment_applied:
            insert_idx = 3 if amplitude_balance_applied else 2
            applied_corrections.insert(insert_idx, "RecoStabCorrFilter(readout polarity timing alignment)")
        if ramp_delay_scale_applied:
            applied_corrections.insert(
                applied_corrections.index(regrid_correction_anchor) + 1,
                "RecoRegridFilter(readout trajectory delay scaling)",
            )
        if readout_regrid_applied:
            applied_corrections.insert(
                applied_corrections.index(regrid_correction_anchor) + 1,
                "RecoRegridFilter(readout trajectory interpolation auto/opt-in)",
            )
        if readout_density_applied:
            applied_corrections.insert(
                applied_corrections.index(regrid_correction_anchor) + 1,
                "RecoRegridFilter(readout trajectory density compensation)",
            )
        if pc_lin_applied:
            applied_corrections.insert(applied_corrections.index("RecoFTFilter(slice)"), "RECO_pc_lin(first-order opt-in)")
        if echo_time_shift_applied:
            applied_corrections.insert(applied_corrections.index("RecoFTFilter(readout)"), "PVM_EpiEchoTimeShifting")
        if epi_phase_correction_applied:
            applied_corrections.insert(
                applied_corrections.index("RecoFTFilter(readout)"),
                "PVM_EpiPhaseCorrection",
            )
        if ghost_filter_applied:
            applied_corrections.append("RecoEpiGhostFilter(FOV/2 leakage correction)")
        if sorting_meta.get("prefix_navigator", {}).get("applied"):
            applied_corrections.insert(1, "PVM_EpiPrefixNavSize(skip prefix navigator)")
        metadata_only_steps = [
            step
            for step in _bruker_unhandled_reco_steps(reco)
            if step
            not in {
                "RecoFTShiftFilter",
                "RecoFTFilter",
                "RecoReverseFilter",
                "RecoStabCorrFilter",
                "RecoRegridFilter",
                "RecoHalfFourierFilter",
            }
        ]
        if sorting_meta.get("half_fourier", {}).get("enabled") and not half_fourier_applied:
            metadata_only_steps.append("RecoHalfFourierFilter")
        stage_text = _stage_node_text(reco)
        for step in ("RECO_pc_mode",):
            if step not in metadata_only_steps and step in stage_text:
                metadata_only_steps.append(step)
        if "RecoStabCorrFilter" in stage_text:
            metadata_only_steps.append("RecoStabCorrFilter(phase/navigator residual)")
        if "RecoEpiGhostFilter" in stage_text and not ghost_filter_applied:
            metadata_only_steps.append("RecoEpiGhostFilter")
        if not b0_trajectory_phase_applied and _flatten_numeric(method.get("PVM_EpiTrajAdjb0")):
            metadata_only_steps.append("PVM_EpiTrajAdjb0(part of proprietary RecoRegridFilter model)")

        meta = {
            "rawdat_path": str(raw_path),
            "raw_frame_count": int(raw_frame_count),
            "frame_order": None if frame_order is None else frame_order.tolist(),
            "frame_packages": None if package_indices is None else package_indices.tolist(),
            "frame_transposition": frame_transposition.tolist(),
            "mixed_slice_geometry": mixed_slice_geometry,
            "sequence_type": sequence_type,
            "kind": "epi",
            "kspace_ready": True,
            "reconstruction_status": "reconstructed",
            "reconstruction_candidate": sorting_meta.get("reconstruction_candidate", "graph_ordered_epi"),
            "candidate_errors": sorting_meta.get("candidate_errors", []),
            "raw_reconstruction_note": (
                "Bruker EPI raw reconstruction. MRax applies the recorded PV360 raw-space corrections: "
                "receiver-contiguous raw unpacking, prefix navigator skipping, reverse partition scan order, "
                "readout-polarity amplitude balancing, measured readout trajectory interpolation or density "
                "compensation when enabled plus B0 phase, alternating readout reversal, "
                "echo-time shifting, recorded phase correction, sort maps, FT, phase-axis half-Fourier handling, "
                "recorded RecoEpiGhostFilter FOV/2 ghost correction when available, RECO cutoff, and "
                "coil/object combination. The proprietary RecoRegridFilter ramp interpolation kernel and some "
                "RecoStabCorrFilter navigator internals remain approximated or metadata-only."
            ),
            "sample_sorting": sorting_meta,
            "vendor_corrections": {
                "kspace": applied_corrections,
                "image": vendor_image_steps,
                "metadata_only": metadata_only_steps,
            },
            "reco_parameters": {
                k: reco[k]
                for k in (
                    "RECO_size",
                    "RECO_inp_size",
                    "RECO_ft_size",
                    "RECO_offset",
                    "RECO_fov",
                    "RECO_ft_mode",
                    "RECO_qopts",
                    "RECO_rotate",
                    "RECO_transposition",
                    "RECO_pc_mode",
                    "RECO_pc_lin",
                    "RECO_ppc_mode",
                    "RECO_wdw_mode",
                    "RecoHalfFT",
                    "RecoHalfFTAlgorithm",
                    "RecoHalfFTPos",
                    "RECO_wordtype",
                    "RecoScaleChan",
                    "RecoPhaseChan",
                    "RecoReverseSegment",
                    "RecoSortDim",
                    "RecoSortSize",
                    "RecoSortRange",
                )
                if k in reco
            },
            "acq_word_size": acq_word,
            "rotate_applied": sorting_meta.get("rotate"),
            "fft_norm_correction": None,
            "sampling_mask_fraction": float(np.mean(sampling_mask)),
            "sampling_mask_shape": tuple(int(v) for v in sampling_mask.shape),
            "kspace_corrections_applied": applied_corrections,
            "_debug_pre_crop_raw_image": debug_pre_crop_raw_image,
        }
        return kspace, raw_image_out, sampling_mask, meta

    raw_blocks_result = _read_bruker_pv360_raw_job_blocks(
        raw_path,
        dtype,
        method=method,
        acqp=acqp,
    )
    if raw_blocks_result is None:
        raw_blocks = None
        raw_storage_meta: dict[str, object] = {}
        samples = _read_rawdat_complex(raw_path, dtype)
    else:
        raw_blocks, raw_storage_meta = raw_blocks_result
        samples = np.asarray(raw_blocks).reshape((-1,))

    enc_steps = reco.get("RECO_EncSteps1") or reco.get("RECO_enc_steps1") or method.get("PVM_EncSteps1")
    enc_steps2 = reco.get("RECO_EncSteps2") or reco.get("RECO_enc_steps2") or method.get("PVM_EncSteps2")
    target_ky = None
    target_kz = None
    ft_size = reco.get("RECO_ft_size") or reco.get("RECO_size")
    if isinstance(ft_size, list) and len(ft_size) >= 2 and all(isinstance(v, (int, float)) for v in ft_size[:2]):
        # Bruker lists sizes as (read, phase); our k-space uses (phase, read) ordering.
        target_ky = int(ft_size[1])
        target_kx = int(ft_size[0])
        if len(ft_size) >= 3 and isinstance(ft_size[2], (int, float)):
            target_kz = int(ft_size[2])
    else:
        target_kx = None

    if enc_steps is not None and len(raw_shape) in (2, 3):
        try:
            enc_steps_arr = np.asarray(enc_steps, dtype=int).ravel()
            if enc_steps_arr.size > 0:
                ky_acquired = int(enc_steps_arr.size)
                ky_axis = 0 if len(raw_shape) == 2 else 1
                if ky_acquired > 1 and ky_acquired != int(raw_shape[ky_axis]):
                    shape_vals = list(raw_shape)
                    shape_vals[ky_axis] = ky_acquired
                    raw_shape = tuple(shape_vals)
        except Exception:
            _LOG.debug("Ignored optional operation failure", exc_info=True)

    if enc_steps2 is not None and len(raw_shape) == 3:
        try:
            enc_steps2_arr = np.asarray(enc_steps2, dtype=int).ravel()
            if enc_steps2_arr.size > 0:
                kz_acquired = int(enc_steps2_arr.size)
                if kz_acquired > 1 and kz_acquired != int(raw_shape[0]):
                    raw_shape = (kz_acquired, int(raw_shape[1]), int(raw_shape[2]))
        except Exception:
            _LOG.debug("Ignored optional operation failure", exc_info=True)

    kspace = None
    sample_sorting_meta: dict[str, object] = {}
    frame_order_for_output = frame_order
    ky_acquired = int(raw_shape[0]) if len(raw_shape) == 2 else None
    kx_acquired = int(raw_shape[1]) if len(raw_shape) == 2 else None
    if len(raw_shape) == 2 and ky_acquired is not None and kx_acquired is not None:
        if sequence_type == "rare":
            rare_factor = method.get("PVM_RareFactor")
            try:
                etl = int(rare_factor)
            except (TypeError, ValueError):
                etl = 0
            if etl > 1 and enc_steps is not None and target_ky is not None:
                try:
                    rare_sample_order = _bruker_rare_sample_order(method, raw_frame_count)
                    kspace = _reshape_bruker_rare_kspace(
                        samples,
                        frame_count=raw_frame_count,
                        coil_count=receivers,
                        rare_factor=etl,
                        enc_steps=enc_steps,
                        target_ky=int(target_ky),
                        kx_count=kx_acquired,
                        sample_order=rare_sample_order,
                    )
                    sample_sorting_meta = {
                        "sample_order": rare_sample_order,
                        "rare_factor": int(etl),
                    }
                except ValueError:
                    kspace = None
        elif (
            raw_blocks is not None
            and len(spatial_shape) == 3
            and int(raw_frame_count) == int(frame_count) * int(spatial_shape[-1])
        ):
            try:
                kspace, sample_sorting_meta = _reshape_bruker_standard_2d_raw_job(
                    raw_blocks,
                    frame_count=frame_count,
                    raw_frame_count=raw_frame_count,
                    slice_count=int(spatial_shape[-1]),
                    ky_count=ky_acquired,
                    kx_count=kx_acquired,
                    reco=reco,
                )
                sample_sorting_meta["raw_storage"] = raw_storage_meta
                frame_order_for_output = None
            except ValueError:
                kspace = None
        if kspace is None and receivers > 1 and raw_frame_count > 1 and not is_multi_echo:
            try:
                standard_sample_order = _bruker_standard_2d_sample_order(method, raw_frame_count)
                kspace = _reshape_bruker_standard_kspace(
                    samples,
                    frame_count=raw_frame_count,
                    coil_count=receivers,
                    ky_count=ky_acquired,
                    kx_count=kx_acquired,
                    sample_order=standard_sample_order,
                )
                sample_sorting_meta = {"sample_order": standard_sample_order}
            except ValueError:
                kspace = None
    elif len(raw_shape) == 3:
        kz_acquired = int(raw_shape[0])
        ky_acquired_3d = int(raw_shape[1])
        kx_acquired_3d = int(raw_shape[2])
        try:
            kspace = _reshape_bruker_3d_standard_kspace(
                samples,
                frame_count=raw_frame_count,
                coil_count=receivers,
                kz_count=kz_acquired,
                ky_count=ky_acquired_3d,
                kx_count=kx_acquired_3d,
            )
            sample_sorting_meta = {"sample_order": "partition_phase_coil_read"}
        except ValueError:
            kspace = None

    encoding_corrections: list[str] = []
    if kspace is None:
        kspace = reshape_raw_kspace(
            samples,
            frame_count=raw_frame_count,
            spatial_shape=raw_shape,
            coil_count=coil_count,
            is_multi_echo=is_multi_echo,
            enc_steps=enc_steps,
            target_ky=target_ky,
        )
        sample_sorting_meta = {"sample_order": "generic_heuristic"}
        if enc_steps is not None:
            encoding_corrections.append("RECO_EncSteps1/PVM_EncSteps1")
    elif enc_steps is not None and sequence_type != "rare":
        kspace = apply_encoding_steps(
            np.asarray(kspace),
            np.asarray(enc_steps, dtype=int).ravel(),
            axis=-2,
            target_size=target_ky,
        )
        encoding_corrections.append("RECO_EncSteps1/PVM_EncSteps1")

    if enc_steps2 is not None and len(raw_shape) == 3:
        kspace = apply_encoding_steps(
            np.asarray(kspace),
            np.asarray(enc_steps2, dtype=int).ravel(),
            axis=-3,
            target_size=target_kz,
        )
        encoding_corrections.append("RECO_EncSteps2/PVM_EncSteps2")

    kspace, correction_meta = prepare_bruker_cartesian_kspace(
        np.asarray(kspace),
        reco,
        target_kx=target_kx,
    )

    frame_transposition = _bruker_frame_transpositions(reco.get("RECO_transposition"), raw_frame_count)
    if not np.any(frame_transposition > 0) and visu is not None:
        frame_transposition = _bruker_frame_transpositions(visu.get("VisuCoreTransposition"), raw_frame_count)

    mask = make_sampling_mask(kspace)
    fft_axes = tuple(range(-len(raw_shape), 0)) if len(raw_shape) >= 2 else (-2, -1)
    builder = pipeline
    if builder is None:
        reco_pipeline = RecoPipeline([FFTStage(direction="inverse", axes=fft_axes)])
    elif isinstance(builder, RecoPipeline):
        reco_pipeline = builder
    else:
        reco_pipeline = builder({"mask": mask, "method": method, "reco": reco})
    state = RecoState(
        kspace=kspace,
        mask=mask,
        metadata={"spatial_shape": spatial_shape, "rawdat_path": str(raw_path)},
    )
    # Run reconstruction for convenience/visualization, but always return the unfilled
    # k-space (zeros outside the sampling mask). This keeps model-based reconstruction
    # well-defined when using forward masking.
    kspace_unfilled = np.asarray(kspace)
    result = reco_pipeline.run(state.copy())
    image = result.image if result.image is not None else cartesian_ifft(result.kspace, axes=fft_axes)

    kspace_unfilled = _reorder_bruker_frames(kspace_unfilled, frame_order_for_output)
    if mask is not None:
        mask = _reorder_bruker_frames(mask, frame_order_for_output)
    if image is not None:
        image = _reorder_bruker_frames(image, frame_order_for_output)
    frame_transposition = _reorder_bruker_frames(frame_transposition, frame_order_for_output)

    kspace_out = _pack_bruker_2d_frame_kspace(
        kspace_unfilled,
        frame_count=frame_count,
        raw_frame_count=raw_frame_count,
        spatial_shape=spatial_shape,
    )
    mask = _pack_bruker_2d_frame_kspace(
        mask,
        frame_count=frame_count,
        raw_frame_count=raw_frame_count,
        spatial_shape=spatial_shape,
    )

    raw_image_out = (
        format_bruker_cartesian_image(
            image,
            reco,
            frame_transposition=frame_transposition,
        )
        if image is not None
        else None
    )

    raw_image_out = _pack_bruker_2d_frame_images(
        raw_image_out,
        frame_count=frame_count,
        raw_frame_count=raw_frame_count,
        spatial_shape=spatial_shape,
    )

    vendor_image_steps = ["fft_norm_correction"]
    if reco.get("RECO_size") is not None or reco.get("RECO_offset") is not None:
        vendor_image_steps.append("RECO_size/RECO_offset")
    vendor_image_steps.append("sum_of_squares" if image is not None and np.asarray(image).ndim >= 4 else "magnitude")
    if np.any(frame_transposition > 0):
        vendor_image_steps.append("RECO_transposition")

    meta = {
        "rawdat_path": str(raw_path),
        "raw_frame_count": int(raw_frame_count),
        "frame_order": None if frame_order is None else frame_order.tolist(),
        "frame_packages": None if package_indices is None else package_indices.tolist(),
        "frame_transposition": frame_transposition.tolist(),
        "mixed_slice_geometry": mixed_slice_geometry,
        "sequence_type": sequence_type,
        "kspace_ready": True,
        "vendor_corrections": {
            "kspace": encoding_corrections + correction_meta.get("applied_kspace_corrections", []),
            "image": vendor_image_steps,
            "metadata_only": correction_meta.get("unhandled_reco_steps", []),
        },
        "reco_parameters": {
            k: reco[k]
            for k in (
                "RECO_size",
                "RECO_inp_size",
                "RECO_ft_size",
                "RECO_offset",
                "RECO_fov",
                "RECO_ft_mode",
                "RECO_qopts",
                "RECO_rotate",
                "RECO_transposition",
                "RECO_pc_mode",
                "RECO_pc_lin",
                "RECO_ppc_mode",
                "RECO_wdw_mode",
                "RECO_wordtype",
                "RecoScaleChan",
                "RecoPhaseChan",
            )
            if k in reco
        },
        "stages": [getattr(stage, "name", stage.__class__.__name__) for stage in reco_pipeline.stages],
        "sample_sorting": sample_sorting_meta,
        "kspace_layout": (
            ["frame", "coil", "z", "ky", "kx"]
            if kspace_out is not None and np.asarray(kspace_out).ndim == 5
            else None
        ),
        "fft_axes": list(fft_axes),
        "acq_word_size": acq_word,
        "rotate_applied": correction_meta.get("rotate"),
        "reco_sw": reco.get("RECO_sw"),
        "fft_norm_correction": (
            float(np.sqrt(np.prod(kspace_out.shape[-len(raw_shape) :]))) if kspace_out is not None else None
        ),
        "kspace_corrections_applied": encoding_corrections + correction_meta.get("applied_kspace_corrections", []),
    }
    sampling_mask = None
    if mask is not None and not np.all(mask):
        sampling_mask = mask
        spatial_ndim = len(spatial_shape)
        if sampling_mask.ndim > spatial_ndim:
            lead_axes = tuple(range(sampling_mask.ndim - spatial_ndim))
            sampling_mask = np.any(sampling_mask, axis=lead_axes)
        meta["sampling_mask_fraction"] = float(np.mean(sampling_mask))
        meta["sampling_mask_shape"] = tuple(int(v) for v in sampling_mask.shape)
    return kspace_out, raw_image_out, sampling_mask, meta


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


def _contains_token(value: object, token: str) -> bool:
    needle = token.lower()
    if value is None:
        return False
    if isinstance(value, str):
        return needle in value.lower()
    if isinstance(value, (list, tuple)):
        return any(_contains_token(item, token) for item in value)
    return needle in str(value).lower()


def _as_string_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for item in value:
            out.extend(_as_string_list(item))
        return out
    return [str(value)]


def _bruker_visu_frame_scaling_values(value: object, *, frame_count: int, default: float) -> list[float]:
    values = _flatten_numeric(value)
    if not values:
        values = [default]
    if len(values) == 1:
        values = values * frame_count
    elif len(values) != frame_count and frame_count == 1:
        values = [values[-1]]
    return [float(v) for v in values]


def _bruker_visu_scaling_metadata(visu: dict, *, frame_count: int) -> dict:
    slopes = _bruker_visu_frame_scaling_values(
        visu.get("VisuCoreDataSlope", [1.0]),
        frame_count=frame_count,
        default=1.0,
    )
    offsets = _bruker_visu_frame_scaling_values(
        visu.get("VisuCoreDataOffs", [0.0]),
        frame_count=frame_count,
        default=0.0,
    )
    slope_arr = np.asarray(slopes, dtype=np.float64)
    offset_arr = np.asarray(offsets, dtype=np.float64)
    return {
        "source": "visu_pars",
        "parameters": ["VisuCoreDataSlope", "VisuCoreDataOffs"],
        "applied_to": "DataSet.data",
        "formula": "scaled = stored_2dseq * VisuCoreDataSlope + VisuCoreDataOffs",
        "stored_frame_count": int(frame_count),
        "slope_min": float(np.min(slope_arr)),
        "slope_max": float(np.max(slope_arr)),
        "offset_min": float(np.min(offset_arr)),
        "offset_max": float(np.max(offset_arr)),
        "framewise": bool(
            slope_arr.size > 1
            and (not np.allclose(slope_arr, slope_arr[0]) or not np.allclose(offset_arr, offset_arr[0]))
        ),
    }


def load_2dseq(pdata_dir: Path, *, image_type: str = "auto") -> Tuple[np.ndarray, dict, tuple[int, ...]]:
    visu = parse_jcamp(pdata_dir / "visu_pars")
    word = visu["VisuCoreWordType"]
    dtype = np.dtype(_dtype_from_word(word))
    if visu.get("VisuCoreByteOrder", "").lower() == "bigendian":
        dtype = dtype.newbyteorder(">")
    size = visu["VisuCoreSize"]
    visu_dim = _as_positive_int(visu.get("VisuCoreDim"), None)
    if isinstance(size, (int, float)):
        size = [int(size)]
    if isinstance(size, list):
        if len(size) == 1 and visu_dim != 1:
            size = [size[0], size[0]]
        spatial = [int(s) for s in size[-3:] if int(s) > 0]
    else:
        spatial = []
    if not spatial and isinstance(size, list):
        spatial = [int(s) for s in size[:3] if int(s) > 0]
    if len(spatial) == 1 and visu_dim != 1:
        spatial = [spatial[0], spatial[0]]
    spatial_shape = tuple(spatial[-3:]) if len(spatial) >= 2 else tuple(spatial)
    raw_spatial_shape = spatial_shape
    frame_count = int(visu["VisuCoreFrameCount"])
    slopes = _bruker_visu_frame_scaling_values(
        visu.get("VisuCoreDataSlope", [1.0]),
        frame_count=frame_count,
        default=1.0,
    )
    offsets = _bruker_visu_frame_scaling_values(
        visu.get("VisuCoreDataOffs", [0.0]),
        frame_count=frame_count,
        default=0.0,
    )

    def _fg_order_contains(token: str) -> bool:
        return _contains_token(visu.get("VisuFGOrderDesc"), token)

    mixed_slice_geometry = _bruker_has_mixed_slice_geometry(visu, frame_count)
    frames_are_slices = (
        len(raw_spatial_shape) == 2
        and frame_count > 1
        and _fg_order_contains("FG_SLICE")
        and not mixed_slice_geometry
    )

    raw = np.fromfile(pdata_dir / "2dseq", dtype=dtype)
    prod = int(np.prod(spatial_shape))
    expected = frame_count * prod
    if raw.size != expected:
        raise ValueError(
            f"2dseq size mismatch: found {raw.size} values, expected {expected} "
            f"from VisuCoreFrameCount={frame_count} and VisuCoreSize={raw_spatial_shape}."
        )
    raw = raw.reshape((frame_count, prod))
    scaled = np.empty_like(raw, dtype=np.float32)
    for i in range(frame_count):
        scaled[i] = raw[i] * float(slopes[i]) + float(offsets[i])

    frame_types = [entry.upper() for entry in _as_string_list(visu.get("VisuCoreFrameType"))]
    requested_type = str(image_type or "auto").lower()
    if requested_type not in ("auto", "complex", "real", "reel", "magnitude", "phase", "imaginary"):
        raise ValueError("image_type must be one of: auto, complex, real, magnitude, phase, imaginary")

    if requested_type == "imaginary":
        scaled = (1j * scaled).astype(np.complex64)
    elif requested_type == "complex" or (
        requested_type == "auto"
        and frame_count % 2 == 0
        and (
            "COMPLEX_IMAGE" in frame_types
            or _fg_order_contains("FG_COMPLEX")
            or ("REAL_IMAGE" in frame_types and "IMAGINARY_IMAGE" in frame_types)
        )
    ):
        if _fg_order_contains("FG_COMPLEX") and frame_count % 2 == 0:
            real_first = not (frame_types[:2] == ["IMAGINARY_IMAGE", "REAL_IMAGE"])
            real = scaled[0::2] if real_first else scaled[1::2]
            imag = scaled[1::2] if real_first else scaled[0::2]
        else:
            half = frame_count // 2
            real = scaled[:half]
            imag = scaled[half : 2 * half]
        scaled = (real + 1j * imag).astype(np.complex64)
        frame_count = int(scaled.shape[0])

    reshape_order = "F" if _fg_order_contains("FG_DIFFUSION") else "C"
    transpositions = _bruker_frame_transpositions(visu.get("VisuCoreTransposition"), frame_count)
    if len(raw_spatial_shape) == 1:
        data = scaled.reshape((frame_count, int(raw_spatial_shape[0])), order=reshape_order)
        spatial_shape = (int(raw_spatial_shape[0]),)
    elif len(raw_spatial_shape) == 2:
        frame_spatial_shape = raw_spatial_shape if frames_are_slices else (raw_spatial_shape[1], raw_spatial_shape[0])
        x, y = frame_spatial_shape
        if frames_are_slices and len(frame_spatial_shape) >= 2 and int(frame_spatial_shape[0]) != int(frame_spatial_shape[1]):
            # Bruker 2D slice stacks, like 3D 2dseq volumes, store the read
            # dimension fastest. C-order corrupts non-square slices by
            # interleaving read/phase samples into the displayed image plane;
            # keep square stacks on the historical path because C/F ordering is
            # otherwise ambiguous and existing datasets depend on that layout.
            reshape_order = "F"
        frames = scaled.reshape((frame_count, x, y), order=reshape_order)
        frames = _apply_framewise_transposition(frames, transpositions)
        if frames_are_slices:
            # Treat frames as z only when all frames share one geometry. If Bruker
            # records another FG dimension, keep that dimension as the MRax frame axis.
            slice_count = _visu_fg_count(visu, "FG_SLICE")
            if (
                slice_count is not None
                and 0 < int(slice_count) < int(frame_count)
                and int(frame_count) % int(slice_count) == 0
            ):
                grouped_frame_count = int(frame_count) // int(slice_count)
                entries = _visu_fg_order_entries(visu)
                counts = [int(round(float(entry[0]))) for entry in entries if entry]
                slice_entry = next(
                    (
                        idx
                        for idx, entry in enumerate(entries)
                        if len(entry) >= 2 and "FG_SLICE" in str(entry[1]).upper()
                    ),
                    None,
                )
                if (
                    slice_entry is not None
                    and counts
                    and int(np.prod(counts)) == int(frame_count)
                ):
                    group_shape = tuple(reversed(counts))
                    grouped = frames.reshape(group_shape + tuple(frames.shape[1:]))
                    slice_axis = len(counts) - 1 - int(slice_entry)
                    grouped = np.moveaxis(grouped, slice_axis, len(counts) - 1)
                    grouped = grouped.reshape(
                        (grouped_frame_count, int(slice_count)) + tuple(frames.shape[1:])
                    )
                else:
                    # Bruker frame-group descriptors list the fastest-varying
                    # group first, so FG_SLICE occupies the inner axis.
                    grouped = frames.reshape(
                        (grouped_frame_count, int(slice_count)) + tuple(frames.shape[1:])
                    )
                data = np.moveaxis(grouped, 1, -1)
                spatial_shape = tuple(int(v) for v in frames.shape[1:]) + (int(slice_count),)
                frame_count = grouped_frame_count
            else:
                data = np.moveaxis(frames, 0, -1)[None, ...]
                spatial_shape = tuple(int(v) for v in frames.shape[1:]) + (int(frame_count),)
            if str(visu.get("VisuCoreDiskSliceOrder", "")).lower() == "disk_reverse_slice_order":
                data = np.flip(data, axis=-1)
        else:
            data = frames
            spatial_shape = tuple(int(v) for v in frames.shape[1:])
    elif len(spatial_shape) == 3:
        x, y, z = spatial_shape
        # 3D Bruker 2dseq volumes are stored with the read dimension varying
        # fastest. NumPy's C-order reshape would make z vary fastest and
        # interleave partitions into a grid-like slice artifact.
        data = scaled.reshape((frame_count, x, y, z), order="F")
        if str(visu.get("VisuCoreDiskSliceOrder", "")).lower() == "disk_reverse_slice_order":
            data = np.flip(data, axis=-1)
    else:
        x, y = spatial_shape
        data = scaled.reshape((frame_count, x, y), order=reshape_order)

    visu["_mrax_mixed_slice_geometry"] = bool(mixed_slice_geometry)
    visu["_mrax_frame_transposition"] = transpositions.tolist()
    visu["_mrax_fg_order_entries"] = _visu_fg_order_entries(visu)

    return data, visu, spatial_shape


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


def load_bruker_experiment(
    exp_dir: Path,
    override_type: Optional[str] = None,
    model_map_by_type: Optional[Dict[str, Model | Callable[[], Model]]] = None,
    model_map_by_name: Optional[Dict[str, Model | Callable[[], Model]]] = None,
    study_globals: Optional[Dict[str, Quantity]] = None,
    pdata_index: PDataIndex = DEFAULT_PDATA_INDEX,
    recipe: ExperimentRecipe | None = None,
    load_raw_kspace: bool = False,
    reco_pipeline: RecoPipeline | Callable[[dict], RecoPipeline] | None = None,
    prefer_raw_image: bool = False,
    rotate_bvecs_to_image: bool = True,
    slice_crop_mode: str | None = None,
) -> Tuple[Experiment, DataSet, Model, dict]:
    resolved_pdata_index = _resolve_pdata_index(pdata_index, exp_dir.name)
    method = parse_jcamp(exp_dir / "method")
    acqp = parse_jcamp(exp_dir / "acqp") if (exp_dir / "acqp").exists() else {}
    pdata_dir = exp_dir / "pdata" / resolved_pdata_index
    reco_params = parse_jcamp(pdata_dir / "reco") if (pdata_dir / "reco").exists() else {}
    data, visu, spatial_shape = load_2dseq(pdata_dir)
    study_globals = _normalize_study_globals(study_globals)
    recipe_type, recipe_obs, recipe_model = _apply_recipe(recipe, visu, method, frame_count=data.shape[0])
    frame_count = data.shape[0]
    normalized_meta = _normalize_bruker_metadata(
        visu,
        method,
        frame_count=frame_count,
        rotate_bvecs_to_image=rotate_bvecs_to_image,
    )
    exp_type = override_type or recipe_type or detect_experiment_type(normalized_meta)
    manual_type_override_required = False
    if override_type is None and recipe_type is None:
        exp_type = _override_fair_exp_type(exp_type, method.get("PVM_FairMode"))
        manual_type_override_required = needs_manual_type_override(normalized_meta, exp_type)
    obs = (
        recipe_obs
        if recipe_obs is not None
        else extract_observables_from_metadata(exp_type, normalized_meta, frame_count)
    )
    fair_order = None
    if exp_type == "fair_asl":
        data, obs, fair_order = _maybe_reorder_fair_frames(
            data,
            obs,
            fair_mode=method.get("PVM_FairMode"),
        )
    experiment, _ = build_imported_experiment_dataset(
        ImportDefinition(
            name=str(exp_dir.name),
            data=np.asarray(data),
            frame_count=frame_count,
            spatial_shape=spatial_shape,
            exp_type=exp_type,
            observables=obs,
        )
    )
    if exp_type == "sat_trans":
        timestamps = _read_timestamp_file(exp_dir)
        if timestamps is not None:
            nominal_tr_ms = None
            tr_vals = _flatten_numeric(normalized_meta.repetition_times)
            if tr_vals:
                nominal_tr_ms = float(tr_vals[0])
            rep_count = _sat_trans_rep_count(exp_type, experiment.dims, frame_count)
            tr_full_s = _infer_tr_from_timestamps(
                timestamps,
                nominal_tr_ms=nominal_tr_ms,
                rep_count=rep_count,
            )
            if tr_full_s is not None:
                obs = dict(obs)
                obs["TR"] = Quantity(np.asarray(tr_full_s, dtype=np.float32) * 1e3, "ms")
                experiment, _ = build_imported_experiment_dataset(
                    ImportDefinition(
                        name=str(exp_dir.name),
                        data=np.asarray(data),
                        frame_count=frame_count,
                        spatial_shape=spatial_shape,
                        exp_type=exp_type,
                        observables=obs,
                    )
                )
    dataset_data = np.asarray(data)
    processed: dict[str, object] = {}
    raw_kspace = None
    raw_image = None
    debug_pre_crop_raw_image = None
    sampling_mask = None
    raw_meta: dict | None = None
    sequence_type = _bruker_sequence_type(method, visu)
    if load_raw_kspace:
        try:
            raw_kspace, raw_image, sampling_mask, raw_meta = _load_bruker_raw_kspace(
                exp_dir,
                frame_count=frame_count,
                spatial_shape=spatial_shape,
                method=method,
                reco=reco_params,
                acqp=acqp,
                visu=visu,
                pipeline=reco_pipeline,
                slice_crop_mode=slice_crop_mode,
            )
        except (OSError, ValueError) as exc:
            warnings.warn(
                f"Raw k-space import failed for {exp_dir}: {exc}. Continuing with 2dseq data.",
                RuntimeWarning,
                stacklevel=2,
            )
            raw_meta = {"error": str(exc)}
        if fair_order is not None:
            if (
                isinstance(raw_kspace, np.ndarray)
                and raw_kspace.ndim >= 1
                and raw_kspace.shape[0] == fair_order.size
            ):
                raw_kspace = np.asarray(raw_kspace)[fair_order]
            if raw_image is not None and raw_image.ndim >= 1 and raw_image.shape[0] == fair_order.size:
                raw_image = np.asarray(raw_image)[fair_order]
        if raw_meta is not None:
            debug_pre_crop_raw_image = raw_meta.pop("_debug_pre_crop_raw_image", None)
        if (
            fair_order is not None
            and debug_pre_crop_raw_image is not None
            and np.asarray(debug_pre_crop_raw_image).ndim >= 1
            and np.asarray(debug_pre_crop_raw_image).shape[0] == fair_order.size
        ):
            debug_pre_crop_raw_image = np.asarray(debug_pre_crop_raw_image)[fair_order]
        if raw_image is not None:
            if prefer_raw_image and tuple(raw_image.shape) == tuple(dataset_data.shape):
                dataset_data = np.asarray(raw_image)
            else:
                processed["raw_reconstruction"] = np.asarray(raw_image)
        if sampling_mask is not None:
            processed["sampling_mask"] = np.asarray(sampling_mask)
    else:
        raw_meta = {
            "status": "not_loaded" if _find_rawdat_file(exp_dir) is not None else "unavailable",
            "lazy_available": _find_rawdat_file(exp_dir) is not None,
        }

    experiment, dataset = build_imported_experiment_dataset(
        ImportDefinition(
            name=str(exp_dir.name),
            data=dataset_data,
            frame_count=frame_count,
            spatial_shape=spatial_shape,
            exp_type=exp_type,
            observables=obs,
            processed=processed,
            kspace=raw_kspace,
            sequence_type=sequence_type,
            kspace_metadata=raw_meta or {},
        )
    )
    if debug_pre_crop_raw_image is not None:
        dataset.processed = dict(dataset.processed or {})
        dataset.processed["raw_reconstruction_pre_reco_crop"] = np.asarray(debug_pre_crop_raw_image)
    experiment.study_globals = study_globals
    model_from_default = False
    if recipe_model is not None:
        candidate = recipe_model
        model = candidate() if callable(candidate) else candidate
    elif model_map_by_name and exp_dir.name in model_map_by_name:
        candidate = model_map_by_name[exp_dir.name]
        model = candidate() if callable(candidate) else candidate
    elif model_map_by_type and exp_type in model_map_by_type:
        candidate = model_map_by_type[exp_type]
        model = candidate() if callable(candidate) else candidate
    else:
        model = default_model_for_type(exp_type)
        model_from_default = True

    default_identity_model = model_from_default and str(getattr(model, "name", "")).lower() == "identity"
    identity_message = None
    manual_message = None
    if default_identity_model:
        identity_message = _fit_override_warning_message(
            normalized_meta,
            exp_dir,
            exp_type=exp_type,
            default_identity_model=default_identity_model,
            manual_type_override_required=manual_type_override_required,
        )
    if manual_type_override_required:
        manual_message = _manual_fit_override_warning_message(normalized_meta, exp_dir)
    if identity_message or manual_message:
        _attach_fit_override_warning(
            experiment,
            model,
            identity_message=identity_message,
            manual_message=manual_message,
        )

    # Attach default observables for models that expect them when called directly.
    global_obs = study_globals
    default_obs = {**global_obs, **experiment.observables}
    if hasattr(model, "model_apply"):
        base_apply = model.model_apply

        def model_apply_with_defaults(
            fields,
            offsets,
            observables=None,
            base_apply=base_apply,
            default_obs=default_obs,
        ):
            merged_obs = default_obs if observables is None else {**default_obs, **observables}
            return base_apply(fields, offsets, merged_obs)

        model.model_apply = model_apply_with_defaults  # type: ignore[assignment]
    if hasattr(model, "derived_maps"):
        base_derived = model.derived_maps  # type: ignore[attr-defined]

        def derived_with_defaults(fields, observables=None, base_derived=base_derived, default_obs=default_obs):
            merged_obs = default_obs if observables is None else {**default_obs, **observables}
            return base_derived(fields, merged_obs)

        model.derived_maps = derived_with_defaults  # type: ignore[attr-defined]
    scaling_frame_count = _as_positive_int(visu.get("VisuCoreFrameCount"), frame_count) or int(frame_count)
    image_scaling = _bruker_visu_scaling_metadata(visu, frame_count=scaling_frame_count)
    if int(frame_count) != int(scaling_frame_count):
        image_scaling["dataset_frame_count"] = int(frame_count)
    non_imaging_role = _bruker_non_imaging_role(
        exp_type=exp_type,
        sequence_type=sequence_type,
        spatial_shape=spatial_shape,
        method=method,
        visu=visu,
    )
    if non_imaging_role:
        try:
            experiment._mrax_non_imaging = True
            experiment._mrax_data_role = non_imaging_role
        except Exception:
            _LOG.debug("Ignored optional operation failure", exc_info=True)

    metadata = {
        "scan_no": str(exp_dir.name),
        "path": exp_dir,
        "type": exp_type,
        "pdata_index": resolved_pdata_index,
        "visu": visu,
        "method": method,
        "acqp": acqp,
        "reco": reco_params,
        "image_scaling": image_scaling,
        "mixed_slice_geometry": bool(visu.get("_mrax_mixed_slice_geometry", False)),
        "sequence_type": sequence_type,
        "raw_reconstruction": raw_meta or {},
        **_bruker_image_orientation_metadata(visu),
    }
    if non_imaging_role:
        metadata["non_imaging"] = True
        metadata["data_role"] = non_imaging_role
    return experiment, dataset, model, metadata


def load_bruker_study(
    study_root: str | Path,
    type_overrides: Optional[Dict[str, str]] = None,
    model_map_by_type: Optional[Dict[str, Model | Callable[[], Model]]] = None,
    model_map_by_name: Optional[Dict[str, Model | Callable[[], Model]]] = None,
    study_globals: Optional[Dict[str, Quantity]] = None,
    field_strength_mhz: float | Quantity | None = None,
    pdata_index: PDataIndex = DEFAULT_PDATA_INDEX,
    recipe: ExperimentRecipe | None = None,
    load_raw_kspace: bool = False,
    reco_pipeline: RecoPipeline | Callable[[dict], RecoPipeline] | None = None,
    prefer_raw_image: bool = False,
    rotate_bvecs_to_image: bool = True,
    slice_crop_mode: str | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
):
    root = Path(study_root)
    study_globals = _normalize_study_globals(study_globals, field_strength_mhz)
    if "field_strength_mhz" not in study_globals:
        warnings.warn(
            "No field_strength_mhz provided for this study. Set field_strength_mhz=... or include it in study_globals; "
            "models that require it will raise errors when fitted.",
            RuntimeWarning,
            stacklevel=2,
        )
    entries: List[dict] = []
    skipped_entries: List[dict] = []
    scan_dirs = sorted(
        [entry for entry in root.iterdir() if entry.is_dir() and entry.name.isdigit()],
        key=lambda p: (int(p.name), p.name),
    )
    total_steps = max(1, len(scan_dirs) + 1)
    if progress_callback is not None:
        progress_callback(0, total_steps, "Scanning Bruker study")
    for scan_idx, entry in enumerate(scan_dirs, start=1):
        if progress_callback is not None:
            progress_callback(scan_idx, total_steps, f"Importing Bruker scan {entry.name}")
        pdata_idx = _resolve_pdata_index(pdata_index, entry.name)
        pdata = entry / "pdata" / pdata_idx / "2dseq"
        method = entry / "method"
        missing_files = [str(path) for path in (method, pdata) if not path.exists()]
        if missing_files:
            warnings.warn(
                f"warning: skipped Bruker scan {entry.name}: missing file(s): {', '.join(missing_files)}",
                RuntimeWarning,
                stacklevel=2,
            )
            skipped_entries.append(
                {
                    "scan_no": entry.name,
                    "path": str(entry),
                    "pdata_index": pdata_idx,
                    "status": "skipped",
                    "reason": "missing_file",
                    "missing_files": missing_files,
                    "message": "Missing required Bruker file(s): " + ", ".join(missing_files),
                }
            )
            if progress_callback is not None:
                progress_callback(scan_idx, total_steps, f"Skipped Bruker scan {entry.name}")
            continue
        override = None
        if type_overrides and entry.name in type_overrides:
            override = type_overrides[entry.name]
        try:
            experiment, dataset, model, meta = load_bruker_experiment(
                entry,
                override,
                model_map_by_type,
                model_map_by_name,
                study_globals=study_globals,
                pdata_index=pdata_idx,
                recipe=recipe,
                load_raw_kspace=load_raw_kspace,
                reco_pipeline=reco_pipeline,
                prefer_raw_image=prefer_raw_image,
                rotate_bvecs_to_image=rotate_bvecs_to_image,
                slice_crop_mode=slice_crop_mode,
            )
            entries.append(
                {
                    "scan_no": meta["scan_no"],
                    "experiment": experiment,
                    "dataset": dataset,
                    "model": model,
                    "visu": meta["visu"],
                    "method": meta["method"],
                    "acqp": meta.get("acqp", {}),
                    "reco": meta.get("reco", {}),
                    "image_scaling": meta.get("image_scaling", {}),
                    "raw_reconstruction": meta.get("raw_reconstruction", {}),
                    "image_axis_anatomical_labels": meta.get("image_axis_anatomical_labels"),
                    "native_image_plane": meta.get("native_image_plane"),
                    "type": meta["type"],
                    "path": meta["path"],
                    "pdata_index": meta["pdata_index"],
                }
            )
        except Exception as exc:
            missing = []
            if isinstance(exc, FileNotFoundError) and getattr(exc, "filename", None):
                missing = [str(exc.filename)]
            warnings.warn(
                f"warning: skipped Bruker scan {entry.name}: {type(exc).__name__}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            skipped_entries.append(
                {
                    "scan_no": entry.name,
                    "path": str(entry),
                    "pdata_index": pdata_idx,
                    "status": "skipped",
                    "reason": "missing_file" if missing else "import_failed",
                    "missing_files": missing,
                    "message": str(exc),
                    "error_type": type(exc).__name__,
                }
            )
            if progress_callback is not None:
                progress_callback(scan_idx, total_steps, f"Skipped Bruker scan {entry.name}")
            continue
        if progress_callback is not None:
            progress_callback(scan_idx, total_steps, f"Imported Bruker scan {entry.name}")

    if progress_callback is not None:
        progress_callback(total_steps, total_steps, "Building MRax study")
    study = Study(
        experiments=[e["experiment"] for e in entries],
        datasets=[e["dataset"] for e in entries],
        models=[e["model"] for e in entries],
        names=[e["scan_no"] for e in entries],
        study_globals=study_globals,
    )
    metadata = {
        "subject": root.name,
        "experiments": {
            e["scan_no"]: {
                "scan_no": e["scan_no"],
                "study_subject": root.name,
                "type": e["type"],
                "path": str(e["path"]),
                "pdata_index": e["pdata_index"],
                "status": "loaded",
                "visu": e["visu"],
                "method": e["method"],
                "acqp": e.get("acqp", {}),
                "reco": e.get("reco", {}),
                "image_scaling": e.get("image_scaling", {}),
                "raw_reconstruction": e.get("raw_reconstruction", {}),
                "image_axis_anatomical_labels": e.get("image_axis_anatomical_labels"),
                "native_image_plane": e.get("native_image_plane"),
            }
            for e in entries
        },
        "skipped_experiments": {e["scan_no"]: dict(e) for e in skipped_entries},
        "import_issues": [dict(e) for e in skipped_entries],
    }
    return study, metadata


__all__ = [
    "bruker_affine_from_visu",
    "bruker_vendor_regrid_options",
    "bruker_word_dtype",
    "format_bruker_cartesian_image",
    "load_2dseq",
    "load_bruker_study",
    "load_bruker_experiment",
    "parse_jcamp",
    "prepare_bruker_cartesian_kspace",
    "prepare_bruker_non_cartesian_kspace",
    "read_bruker_regrid_density",
    "read_bruker_regrid_trajectory",
    "reconstruct_bruker_cartesian_kspace",
]
