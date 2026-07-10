from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence


def scan_text(info: Mapping[str, Any]) -> str:
    """Flatten Bruker method/visu fields into a lowercase search string."""

    method = info.get("method", {}) or {}
    visu = info.get("visu", {}) or {}
    keys = (
        "Method",
        "PVM_Protocol",
        "PVM_ScanSequence",
        "PVM_EncSequence",
        "VisuAcqComment",
        "VisuAcqProtocol",
        "VisuAcqSequenceName",
        "VisuSubjectId",
    )
    parts: list[str] = []
    for key in keys:
        if key in method:
            parts.append(str(method.get(key)))
        if key in visu:
            parts.append(str(visu.get(key)))
    return " ".join(p for p in parts if p).lower()


def pick_scan_by_type(study: Any, exp_type: str, *, min_frames: int = 1) -> str | None:
    """Pick the first scan name matching an experiment type and minimum frame count."""

    for name, exp in zip(getattr(study, "names", []), getattr(study, "experiments", [])):
        if getattr(exp, "exp_type", None) != exp_type:
            continue
        frame_count = int(getattr(exp, "frame_count", 0))
        if frame_count >= min_frames:
            return name
    return None


def _first_finite_scalar(value: object | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        scalar = float(value)
        return scalar if scalar == scalar else None
    elif hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        for entry in value:
            scalar = _first_finite_scalar(entry)
            if scalar is not None:
                return scalar
        return None
    try:
        scalar = float(value)
    except Exception:
        return None
    return scalar if scalar == scalar else None


def pick_dce_scan(
    study: Any,
    meta: Mapping[str, Any] | None,
    *,
    min_frames: int = 100,
    min_scan_time_ms: float = 60_000.0,
    flash_keywords: Sequence[str] = ("flash", "gre", "gradient_echo", "fisp"),
) -> str | None:
    """Pick a dynamic FLASH-style DCE scan from a study."""

    direct = pick_scan_by_type(study, "dce", min_frames=min_frames)
    if direct is not None:
        return direct

    meta = meta or {}
    exp_meta = meta.get("experiments", {}) if isinstance(meta, Mapping) else {}
    names = list(getattr(study, "names", []))
    experiments = list(getattr(study, "experiments", []))

    candidates: list[tuple[int, str]] = []
    for idx, (name, exp) in enumerate(zip(names, experiments)):
        exp_type = getattr(exp, "exp_type", None)
        if exp_type not in ("anatomical", "mge", "vfa", "dce"):
            continue
        frame_count = int(getattr(exp, "frame_count", 0) or 0)
        if frame_count < min_frames:
            continue

        info = exp_meta.get(name, {})
        text = scan_text(info)
        if not any(keyword in text for keyword in flash_keywords):
            continue

        method = info.get("method", {}) or {}
        visu = info.get("visu", {}) or {}
        scan_time_ms = _first_finite_scalar(method.get("PVM_ScanTime") or visu.get("VisuAcqScanTime"))
        if scan_time_ms is None or scan_time_ms < float(min_scan_time_ms):
            continue

        candidates.append((idx, name))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def auto_select_scans(
    study: Any,
    meta: Mapping[str, Any] | None,
    *,
    fair_exp_types: Sequence[str] = ("fair_asl",),
    t2_exp_types: Sequence[str] = ("mse",),
    sat_trans_type: str = "sat_trans",
    wasabi_keywords: Iterable[str] = ("wasabi", "wobble", "b0", "b1"),
    min_frames: int = 1,
) -> tuple[str | None, str | None, str | None, list[str]]:
    """Heuristically select FAIR, T2, WASABI, and sat-trans scans from a study."""

    meta = meta or {}
    exp_meta = meta.get("experiments", {}) if isinstance(meta, Mapping) else {}
    names = list(getattr(study, "names", []))
    experiments = list(getattr(study, "experiments", []))

    texts = {name: scan_text(exp_meta.get(name, {})) for name in names}

    fair_scans = [
        name
        for name, exp in zip(names, experiments)
        if getattr(exp, "exp_type", None) in fair_exp_types and int(getattr(exp, "frame_count", 0)) >= min_frames
    ]
    if not fair_scans:
        fair_scans = [name for name in names if "fair" in texts.get(name, "")]
    fair_scan = fair_scans[0] if fair_scans else None

    t2_scan = None
    for t2_type in t2_exp_types:
        t2_scan = pick_scan_by_type(study, t2_type, min_frames=min_frames)
        if t2_scan:
            break

    wasabi = None
    for name in names:
        info = exp_meta.get(name, {})
        exp_type = info.get("type")
        if exp_type != sat_trans_type:
            continue
        text = texts.get(name, "")
        if any(keyword in text for keyword in wasabi_keywords):
            wasabi = name
            break

    sat_dyn = [
        name
        for name in names
        if exp_meta.get(name, {}).get("type") == sat_trans_type and name != wasabi
    ]

    return fair_scan, t2_scan, wasabi, sat_dyn


__all__ = ["auto_select_scans", "pick_dce_scan", "pick_scan_by_type", "scan_text"]
