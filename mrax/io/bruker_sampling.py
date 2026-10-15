"""Bruker sampling translation; MRI reconstruction is separate."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from mrax.reco.processing import apply_channel_scale_phase, estimate_radial_density

from .bruker_parameters import (
    _as_positive_int,
    _as_reco_string_list,
    _flatten_numeric,
    _stage_node_text,
    parse_jcamp,
)

DEFAULT_PDATA_INDEX = "1"

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
    normalize: bool = True,
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
        # Rotated spiral arms span (time, rotation angle). The angular tangent
        # is (-ky, kx), so their area Jacobian is abs(k dot dk/dt).
        jac = np.abs(np.sum(segment * derivative, axis=-1))
        weights.append(jac.astype(np.float32))
    result = np.concatenate(weights)
    return _normalize_non_cartesian_density(result, spatial_shape) if normalize else result


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
                normalize=False,
            )
            source = "trajDC"
        else:
            density = estimate_radial_density(traj_for_density, spatial_shape=spatial_shape)
            source = "trajectory_fallback"
    elif correction_type == "rho":
        from mrax.reco.density import radial_profile_density

        density = radial_profile_density(traj_for_density, points_per_readout=points,
            center_out=not _bruker_inverse_fid_active(reco))
    else:
        density = estimate_radial_density(traj_for_density, spatial_shape=spatial_shape)
        source = "trajectory_fallback"

    meta = {
        "correction_type": correction_type,
        "source": source,
        "density_path": None if source_path is None else str(source_path),
        "density_samples": int(density.shape[0]),
        "normalization": "radial_volume_per_profile" if correction_type == "rho" else "trajectory_jacobian",
    }
    return np.asarray(density, dtype=np.float32), meta


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


def _reshape_bruker_non_cartesian_blocks(
    blocks: np.ndarray,
    *,
    frame_count: int,
    trajectory_samples: int,
    method: dict,
    reco: dict,
    regrid_meta: dict,
    loop_order=("interleave", "frame", "coil", "read"),
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
    if _bruker_inverse_fid_active(reco):
        raise ValueError("Delayed profiles require the explicit readout-model resolver")
    inverse_fid_meta = {"applied": False}
    keep = _non_cartesian_segment_keep_indices(
        raw_points_per_interleave=raw_per_interleave,
        kept_points_per_interleave=kept_per_interleave,
        interleaves=1,
        method=method,
        reco=reco,
    )
    canonical = ("interleave", "frame", "coil", "read")
    loop_order = tuple(loop_order)
    if len(loop_order) != len(canonical) or set(loop_order) != set(canonical):
        raise ValueError("Sampled profile loops must identify every acquired axis exactly once")
    sizes = dict(zip(canonical, (interleaves, frames, int(arr.shape[1]), raw_per_interleave), strict=True))
    raw = arr[:expected_scans].reshape(tuple(sizes[key] for key in loop_order)).transpose(tuple(loop_order.index(key) for key in canonical))
    trim_start = int(keep[0]) if keep.size else None
    trim_stop = int(keep[-1] + 1) if keep.size else None
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
