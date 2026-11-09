"""Study workspace state for the browser application."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..session import GuiSession, VisualViewState


@dataclass
class StudyWorkspace:
    """Cached GUI workspace for one loaded or in-memory study."""

    key: str
    label: str
    session: GuiSession
    fit_results_by_experiment: dict[str, dict[str, Any]] = field(default_factory=dict)
    preview_results_by_experiment: dict[str, dict[str, Any]] = field(default_factory=dict)
    registration_results: dict[str, dict[str, Any]] = field(default_factory=dict)
    drift_results: dict[str, dict[str, Any]] = field(default_factory=dict)
    visual_undo_stacks: dict[str, list[VisualViewState]] = field(default_factory=dict)
    view_label: str = "Image"
    plot_x_axis_key: str = "auto"
    dirty: bool = False
    dirty_reason: str = ""
    display_settings: dict[str, Any] = field(default_factory=dict)
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    roi_history: list[dict[str, Any]] = field(default_factory=list, repr=False)
    pipelines: dict[str, dict[str, Any]] = field(default_factory=dict)
    pipeline_mask_cache: dict = field(default_factory=dict, repr=False)
    source_files: list[dict[str, Any]] = field(default_factory=list, repr=False)
