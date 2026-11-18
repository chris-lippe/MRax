"""Portable workspace and session persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from mrax.gui.session import GuiSession

from .state import _serialize_gui_state
from .workspace import StudyWorkspace


def save_session_state(
    session: GuiSession,
    path: Path,
    *,
    preview_results: dict[str, dict[str, Any]],
    fit_results: dict[str, dict[str, Any]],
    drift_results=None,
    study_manifest: Path | None = None,
    fit_manifest: Path | None = None,
) -> Path:
    """Write portable analysis state without browser preferences."""
    session._save_current_view_state()
    path = Path(path)
    from mrax.io.output_guard import require_new_file

    array_path = path.with_name(f"{path.stem}_arrays.npz")
    require_new_file(path)
    require_new_file(array_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload, arrays = _serialize_gui_state(
        session,
        preview_results=preview_results,
        fit_results=fit_results,
        drift_results=drift_results,
        data_manifest=study_manifest,
        fit_manifest=fit_manifest,
        state_path=path,
    )
    np.savez_compressed(array_path, **arrays)
    payload["array_file"] = array_path.name
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def save_workspace_bundle(
    workspace: StudyWorkspace, path: Path, *, progress_callback=None, overwrite=False
) -> Path:
    from .workspace_io import save_workspace_bundle as save

    return save(workspace, path, progress_callback=progress_callback, overwrite=overwrite)


def load_workspace_bundle(path: Path, extraction_root: Path, *, key: str) -> StudyWorkspace:
    from .workspace_io import load_workspace_bundle as load

    return load(path, extraction_root, key=key)
