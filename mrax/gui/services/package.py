"""Workspace package manifests and safe entry resolution."""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Any, Sequence

from mrax import MRAX_VERSION, MRAX_VERSION_TAG
from mrax.versioning import missing_mrax_version_message, validate_mrax_file_version


def _write_mrax_package_manifest(
    root: Path, entries: dict[str, str], *, warnings: Sequence[str] | None = None
) -> Path:
    path = Path(root) / "mrax_package.json"
    payload = {
        "format": "mrax.gui.package",
        "version": 1,
        MRAX_VERSION_TAG: MRAX_VERSION,
        "entries": entries,
    }
    if warnings:
        payload["warnings"] = list(warnings)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _mrax_package_entries(root: Path) -> dict[str, Path]:
    root = Path(root).resolve()
    manifest = root / "mrax_package.json"
    entries: dict[str, Path] = {}
    if manifest.is_file():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if payload.get("format") == "mrax.gui.package":
            package_version = validate_mrax_file_version(
                payload,
                file_kind="MRax package manifest",
            )
            if package_version.tag_missing:
                warnings.warn(
                    missing_mrax_version_message("MRax package manifest"),
                    RuntimeWarning,
                    stacklevel=2,
                )
            for key, value in dict(payload.get("entries") or {}).items():
                candidate = (root / str(value).replace("\\", "/")).resolve()
                if not _is_relative_to(candidate, root):
                    raise ValueError(
                        f"Unsafe package entry path outside package directory: {value}"
                    )
                entries[str(key)] = candidate
    if "gui_state" not in entries and (root / "mrax_gui_state.json").is_file():
        entries["gui_state"] = root / "mrax_gui_state.json"
    if "study_manifest" not in entries:
        study_manifest = _find_study_manifest(root)
        if study_manifest is not None:
            entries["study_manifest"] = study_manifest
    if "roi_npz" not in entries:
        roi_path = _find_roi_file(root)
        if roi_path is not None:
            entries["roi_npz" if roi_path.suffix.lower() == ".npz" else "roi_json"] = roi_path
    return entries


def _find_study_manifest(root: Path) -> Path | None:
    root = Path(root)
    direct = root / "study_manifest.json"
    if direct.is_file():
        return direct
    matches = sorted(root.rglob("study_manifest.json"))
    return matches[0] if matches else None


def _find_roi_file(root: Path) -> Path | None:
    root = Path(root)
    for candidate in (root / "mrax_rois.npz", root / "mrax_rois.json"):
        if candidate.is_file():
            return candidate
    for candidate in sorted(root.rglob("*.json")):
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        if "rois" in payload and "npz" in payload:
            return candidate
    for candidate in sorted(root.rglob("*.npz")):
        if candidate.with_suffix(".json").is_file():
            return candidate
    return None


def _package_relative(path: Path, root: Path) -> str:
    try:
        return Path(os.path.relpath(path, root)).as_posix()
    except ValueError:
        return str(path)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _resolve_state_manifest_path(payload: dict[str, Any], state_path: Path) -> Path | None:
    value = payload.get("data_manifest")
    if not value:
        return None
    candidate = Path(str(value).replace("\\", "/"))
    if not candidate.is_absolute():
        candidate = state_path.parent / candidate
    return candidate if candidate.is_file() else None
