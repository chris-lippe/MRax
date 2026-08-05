"""PySide6/pyqtgraph desktop application for MRax."""

from __future__ import annotations

import argparse
import base64
import datetime as _dt
import json
import logging
import os
import platform
import shutil
import sys
import tempfile
import traceback
import warnings
import zipfile
from collections import deque
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping, Sequence

import jax.numpy as jnp
import numpy as np

from mrax.analysis import ROI, broadcast_mask
from mrax.data_struct import DataSet, Experiment, Quantity
from mrax.data_struct.dimensions import (
    dim_observable_map,
    frame_count_from_dims,
    non_spatial_names,
    reshape_dims_to_frames,
    reshape_frames_to_dims,
    spatial_names,
)

from ._deps import GUI_INSTALL_HINT, MissingGuiDependencyError, import_pyqtgraph, import_qt
from .masks import (
    embed_mask_slice,
    import_mask,
    interpolate_mask_along_slices,
    load_mask_array,
    load_roi_set,
    mask_slice_for_display,
    replace_mask_slice,
    save_roi_set,
    stitch_masks_along_slices,
)
from .models import (
    model_available_pools,
    model_choices,
    model_default_pools,
    model_parameter_specs,
    normalize_model_key,
)
from .registration import (
    DriftCorrectionOptions,
    RegistrationOptions,
    alignment_metrics,
    apply_transform_to_image,
    colorize_image,
    drift_choice_sets,
    inverse_transform,
    manual_affine_transform,
    options_from_mapping,
    parse_sequence,
    registration_choice_sets,
    registration_overlay,
    resample_mask_with_transform,
    run_drift_correction,
    run_registration,
    transform_metadata,
)
from .session import (
    ExperimentViewState,
    FitSettings,
    FrameMask,
    GuiSession,
    VisualViewState,
    _display_slice,
    frame_axis_values,
)

_LOG = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the MRax desktop GUI.")
    parser.add_argument("manifest", nargs="?", help="Optional path to a study_manifest.json file.")
    parser.add_argument(
        "--quit-after-ms",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser


def _utc_timestamp() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _make_gui_diagnostic_event(
    level: str,
    message: Any,
    *,
    detail: Any = "",
    context: Mapping[str, Any] | None = None,
    timestamp: str | None = None,
) -> dict[str, Any]:
    return {
        "timestamp": timestamp or _utc_timestamp(),
        "level": str(level or "info").lower(),
        "message": str(message),
        "detail": "" if detail is None else str(detail),
        "context": _metadata_to_jsonable(dict(context or {})),
    }


def _format_gui_diagnostic_events(
    events: Sequence[Mapping[str, Any]],
    *,
    include_details: bool = False,
) -> list[str]:
    if not events:
        return ["No captured application warnings or errors."]
    lines: list[str] = []
    for event in events:
        timestamp = str(event.get("timestamp") or "")
        level = str(event.get("level") or "info").upper()
        message = str(event.get("message") or "")
        prefix = f"{timestamp} " if timestamp else ""
        lines.append(f"{prefix}[{level}] {message}")
        context = event.get("context") or {}
        if isinstance(context, Mapping) and context:
            context_text = ", ".join(f"{key}={value}" for key, value in context.items() if value not in (None, ""))
            if context_text:
                lines.append(f"    context: {context_text}")
        detail = str(event.get("detail") or "")
        if include_details and detail:
            lines.extend(f"    {line}" if line else "" for line in detail.rstrip().splitlines())
    return lines


def _format_gui_diagnostic_report(
    *,
    events: Sequence[Mapping[str, Any]],
    session_state: Mapping[str, Any],
    session_diagnostics: Sequence[str],
    user_message: str = "",
    attachments: Mapping[str, Any] | None = None,
) -> str:
    problem_events = [
        event
        for event in events
        if str(event.get("level") or "").lower() in {"warning", "error", "critical"}
    ]
    latest = problem_events[-1] if problem_events else None
    source = session_state.get("source") or session_state.get("manifest_path") or "unknown"
    lines = [
        "MRax GUI Diagnostic Report",
        f"Generated: {_utc_timestamp()}",
        "",
        "Summary",
        (
            f"- Captured {len(problem_events)} warning/error event(s)."
            if problem_events
            else "- No GUI-captured warning/error events are currently recorded."
        ),
    ]
    if latest is not None:
        lines.append(f"- Latest issue: [{str(latest.get('level') or '').upper()}] {latest.get('message')}")
    message = str(user_message or "").strip()
    if message:
        lines.extend(["", "User Message"])
        lines.extend(message.splitlines())
    attachment_items = {
        str(key): value
        for key, value in dict(attachments or {}).items()
        if value not in (None, "")
    }
    if attachment_items:
        lines.extend(["", "Attachments"])
        lines.extend(f"- {key}: {value}" for key, value in attachment_items.items())
    lines.extend(
        [
            "",
            "Reproduction Context",
            f"- Source: {source}",
            f"- Source kind: {session_state.get('source_kind') or 'unknown'}",
            f"- Selected experiment: {session_state.get('selected_experiment') or 'none'}",
            f"- View: {session_state.get('view') or 'unknown'}",
            f"- Active frame: {session_state.get('active_frame')}",
            f"- Active z: {session_state.get('active_z')}",
            f"- Active indices: {session_state.get('active_indices') or {}}",
            f"- Active map: {session_state.get('active_map') or 'none'}",
            f"- ROI names: {', '.join(session_state.get('roi_names') or []) or 'none'}",
            f"- Dirty state: {session_state.get('dirty')}",
            f"- Dirty reason: {session_state.get('dirty_reason') or 'none'}",
            "",
            "Suggested Reproduction Steps",
            "1. Start the MRax GUI in the environment listed below.",
            f"2. Load or import the source listed above: {source}.",
            "3. Select the experiment, view, frame, z slice, map, and ROI state listed in Reproduction Context.",
            "4. Repeat the action that produced the latest warning/error event in Application Events.",
            "",
            "Environment",
            f"- Python: {sys.version.replace(chr(10), ' ')}",
            f"- Platform: {platform.platform()}",
            f"- Working directory: {Path.cwd()}",
            f"- Command: {' '.join(sys.argv)}",
            "",
            "Session Diagnostics",
        ]
    )
    lines.extend(f"- {line}" for line in (session_diagnostics or ["No session diagnostics available."]))
    lines.extend(["", "Application Events"])
    lines.extend(_format_gui_diagnostic_events(events, include_details=True))
    return "\n".join(lines).rstrip() + "\n"


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
    feature_page_index: int = 0
    dirty: bool = False
    dirty_reason: str = ""


WorkspaceMetadataTemplateItem = tuple[str, str, str]


def _workspace_experiment_metadata_template_items(
    workspaces: Mapping[str, StudyWorkspace],
    *,
    active_key: str | None = None,
) -> list[tuple[str, WorkspaceMetadataTemplateItem]]:
    """Return combo-box labels/data for all experiments in loaded study workspaces."""

    items: list[tuple[str, WorkspaceMetadataTemplateItem]] = []
    for workspace_key, workspace in (workspaces or {}).items():
        session = workspace.session
        if not session.data:
            continue
        label = str(workspace.label or session.study_name or workspace_key or "Study")
        prefix = f"Current study: {label}" if workspace_key == active_key else label
        for exp_name in session.experiment_names:
            items.append((f"{prefix} / {exp_name}", ("workspace", str(workspace_key), str(exp_name))))
    return items


def _is_workspace_metadata_template_item(value: Any) -> bool:
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        return False
    return str(value[0]) == "workspace"


def main(argv: list[str] | None = None) -> int:
    _apply_local_runtime_env()
    _reexec_with_local_lib_bundle_if_needed(argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return _run_gui(args)
    except MissingGuiDependencyError as exc:
        print(str(exc), file=sys.stderr)
        return 2


def _local_lib_dir() -> Path:
    return Path(
        os.environ.get(
            "MRAX_GUI_LOCAL_LIB_DIR",
            Path.home() / ".local" / "mrax-gui-libs" / "usr" / "lib" / "x86_64-linux-gnu",
        )
    )


def _local_bundle_root() -> Path:
    lib_dir = _local_lib_dir()
    if lib_dir.name == "x86_64-linux-gnu" and lib_dir.parent.name == "lib":
        return lib_dir.parents[2]
    return lib_dir


def _apply_local_runtime_env() -> None:
    root = _local_bundle_root()
    if os.environ.get("DISPLAY") and not os.environ.get("QT_QPA_PLATFORM"):
        # VS Code/dev containers often expose both DISPLAY and WAYLAND_DISPLAY.
        # Qt tries Wayland first, emits a plugin warning, then falls back. Prefer
        # XCB directly when an X display is available.
        os.environ["QT_QPA_PLATFORM"] = "xcb"
    fonts = root / "etc" / "fonts"
    if fonts.is_dir() and not os.environ.get("FONTCONFIG_PATH"):
        os.environ["FONTCONFIG_PATH"] = str(fonts)
    x11 = root / "usr" / "share" / "X11"
    if x11.is_dir():
        if not os.environ.get("XLOCALEDIR"):
            os.environ["XLOCALEDIR"] = str(x11 / "locale")
        if not os.environ.get("XCOMPOSEFILE"):
            compose = x11 / "locale" / "en_US.UTF-8" / "Compose"
            if compose.is_file():
                os.environ["XCOMPOSEFILE"] = str(compose)
        if not os.environ.get("XKB_CONFIG_ROOT"):
            xkb = x11 / "xkb"
            if xkb.is_dir():
                os.environ["XKB_CONFIG_ROOT"] = str(xkb)


def _reexec_with_local_lib_bundle_if_needed(argv: list[str] | None = None) -> None:
    """Restart with a user-local native library bundle when one is available.

    This supports restricted containers where MRax can be installed with
    pip but Qt's native Debian libraries cannot be installed system-wide.
    """

    marker = "MRAX_GUI_LOCAL_LIBS_REEXEC"
    if os.environ.get(marker) == "1":
        return
    lib_dir = _local_lib_dir()
    if not lib_dir.is_dir():
        return
    current = os.environ.get("LD_LIBRARY_PATH", "")
    lib_str = str(lib_dir)
    if lib_str in current.split(":"):
        return

    env = dict(os.environ)
    env[marker] = "1"
    env["LD_LIBRARY_PATH"] = lib_str if not current else f"{lib_str}:{current}"
    args = sys.argv[1:] if argv is None else list(argv)
    os.execvpe(sys.executable, [sys.executable, "-m", "mrax.gui.app", *args], env)


def _run_gui(args: argparse.Namespace) -> int:
    QtCore, QtGui, QtWidgets = import_qt()
    pg = import_pyqtgraph()
    pg.setConfigOptions(antialias=True, imageAxisOrder="row-major")

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    app.setApplicationName("MRax GUI")
    app.setOrganizationName("MRax")
    try:
        app.setStyle("Fusion")
    except Exception:
        _LOG.debug("Ignored optional GUI operation failure", exc_info=True)
    _install_application_font(QtGui, app)
    MainWindow = _build_main_window(QtCore, QtGui, QtWidgets, pg)
    window = MainWindow(startup_path=args.manifest)
    window.show()
    if args.quit_after_ms is not None:
        QtCore.QTimer.singleShot(max(0, int(args.quit_after_ms)), app.quit)
    return int(app.exec())


def _install_application_font(QtGui, app) -> None:
    """Install a known UI font when the host/container has no usable default."""

    candidates = [
        _local_bundle_root() / "usr" / "share" / "fonts" / "truetype" / "dejavu" / "DejaVuSans.ttf",
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    try:
        import matplotlib

        mpl_font = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
        candidates.append(mpl_font)
    except Exception:
        _LOG.debug("Ignored optional GUI operation failure", exc_info=True)

    for font_path in candidates:
        if not font_path.is_file():
            continue
        font_id = QtGui.QFontDatabase.addApplicationFont(str(font_path))
        if font_id < 0:
            continue
        families = QtGui.QFontDatabase.applicationFontFamilies(font_id)
        if families:
            app.setFont(QtGui.QFont(families[0], 10))
            return


class _MainWindowMixin:
    def __init__(self, startup_path: str | None = None):
        super().__init__()
        self._study_workspaces: dict[str, StudyWorkspace] = {}
        self._active_study_key: str | None = None
        self.session = GuiSession()
        self.last_fit_result: dict[str, Any] | None = None
        self.last_preview_fit_result: dict[str, Any] | None = None
        self._fit_results_by_experiment: dict[str, dict[str, Any]] = {}
        self._preview_results_by_experiment: dict[str, dict[str, Any]] = {}
        self.registration_results: dict[str, dict[str, Any]] = {}
        self.drift_results: dict[str, dict[str, Any]] = {}
        self._registration_preview_result: dict[str, Any] | None = None
        self.feature_page_labels: list[str] = []
        self._visual_undo_stacks: dict[str, list[VisualViewState]] = {}
        self._updating_visual_state_controls = False
        self.diagnostic_events: deque[dict[str, Any]] = deque(maxlen=300)
        self._previous_warning_showwarning = None
        self._previous_excepthook = None
        self._diagnostic_hooks_active = False
        self.overlay_items: list[Any] = []
        self.roi_items: dict[str, Any] = {}
        self.roi_item_kinds: dict[str, str] = {}
        self.brush_mode: str | None = None
        self._polygon_points: list[tuple[float, float]] = []
        self._polygon_preview = None
        self._polygon_start_preview = None
        self._brush_preview = None
        self._roi_tool_settings_widget = None
        self._roi_tool_settings_section = None
        self._brush_size_label = None
        self._brush_label_label = None
        self._wand_tolerance_label = None
        self._lasso_support_label = None
        self.lasso_support_spin = None
        self._lasso_spline_label = None
        self.lasso_spline_combo = None
        self.lasso_snap_check = None
        self.brush_label_spin = None
        self.brush_auto_subtract_check = None
        self._label_mask_controls_section = None
        self.label_mask_label_combo = None
        self.label_mask_value_spin = None
        self._advanced_roi_actions_section = None
        self._advanced_roi_buttons: dict[str, Any] = {}
        self._drawing_controls_widget = None
        self._drawing_status_label = None
        self._finish_drawing_button = None
        self._undo_drawing_button = None
        self._cancel_drawing_button = None
        self._last_brush_stamp: tuple[Any, ...] | None = None
        self._brush_edit_checkpoint: dict[str, Any] | None = None
        self._brush_undo_stack: list[dict[str, Any]] = []
        self._brush_stroke_checkpoint: dict[str, Any] | None = None
        self._brush_stroke_active = False
        self._image_pan_locked = False
        self._syncing_roi = False
        self._task_running = False
        self._busy_message: str | None = None
        self._central_splitter = None
        self.progress_bar = None
        self._updating_frames = False
        self._displayed_image: np.ndarray | None = None
        self._displayed_image_shape: tuple[int, ...] | None = None
        self._displayed_view_key: str | None = None
        self._displayed_image_label = "value"
        self._displayed_image_unit: str | None = None
        self._dimension_counter_labels: dict[str, tuple[Any, str, int]] = {}
        self._dimension_detail_label = None
        self._plot_x_axis_key = "auto"
        self._bundle_extract_dirs: list[Path] = []
        self._hover_status_message: str | None = None
        self._dirty = False
        self._dirty_reason = ""
        self._ui_scale = 1.0
        self._ui_scale_bucket = 100
        self.gui_config_path = _gui_config_path()
        self.gui_config = _load_gui_config(self.gui_config_path)
        self._create_workspace("Untitled", self.session, activate=True)
        self.setWindowTitle("MRax[*]")
        self.resize(1320, 860)
        self._build_ui()
        self._install_gui_diagnostic_hooks()
        self._apply_theme()
        QtWidgets.QApplication.instance().installEventFilter(self)
        if startup_path:
            self.load_manifest(Path(startup_path))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not hasattr(self, "_ui_scale_bucket"):
            return
        scale = self._scale_for_window_size()
        bucket = int(round(scale * 20.0)) * 5
        if bucket != self._ui_scale_bucket:
            self._ui_scale = scale
            self._ui_scale_bucket = bucket
            self._apply_theme()

    def eventFilter(self, obj, event):
        event_type = getattr(QtCore.QEvent, "Type", QtCore.QEvent)
        if self.is_busy and event.type() == event_type.StatusTip:
            return True
        if isinstance(obj, QtWidgets.QPushButton):
            if self.is_busy:
                if event.type() == event_type.Leave:
                    self._hover_status_message = None
                return super().eventFilter(obj, event)
            if event.type() == event_type.Enter:
                tip = _button_status_tip(obj)
                if tip:
                    self._hover_status_message = tip
                    self.statusBar().showMessage(tip)
            elif event.type() == event_type.Leave and self._hover_status_message:
                if self.statusBar().currentMessage() == self._hover_status_message:
                    self.statusBar().clearMessage()
                self._hover_status_message = None
        if event.type() == event_type.KeyPress and self.brush_mode is not None:
            key = event.key()
            if _qt_key_matches(QtCore, key, "Key_Escape"):
                self._cancel_active_roi_tool()
                return True
            if self.brush_mode in {"polygon", "lasso"} and _qt_key_matches(QtCore, key, "Key_Backspace", "Key_Delete"):
                if self._polygon_points:
                    self._polygon_points.pop()
                    self._update_polygon_preview()
                    self._refresh_drawing_controls()
                return True
            if _qt_key_matches(QtCore, key, "Key_Return", "Key_Enter"):
                self._finish_active_roi_tool()
                return True
        mouse_release = getattr(event_type, "MouseButtonRelease", None)
        if (
            mouse_release is not None
            and event.type() == mouse_release
            and self.brush_mode in {"brush", "erase"}
        ):
            button = _qt_mouse_button(QtCore, "LeftButton")
            try:
                released_left = event.button() == button or _qt_enum_value(event.button()) == _qt_enum_value(button)
            except Exception:
                released_left = True
            if released_left:
                self._finish_brush_stroke()
        return super().eventFilter(obj, event)

    def closeEvent(self, event):
        if self.is_busy:
            self._information_box(
                "Task running",
                "MRax is currently running a task. Wait for it to finish before closing.",
            )
            event.ignore()
            return
        if self.session.data and self._dirty:
            standard_button = getattr(QtWidgets.QMessageBox, "StandardButton", QtWidgets.QMessageBox)
            save_button = standard_button.Save
            discard_button = standard_button.Discard
            cancel_button = standard_button.Cancel
            reply = self._question_box(
                "Save changes?",
                "The current MRax session has unsaved changes. Save before closing?",
                save_button | discard_button | cancel_button,
                save_button,
            )
            if reply == save_button:
                if self.save_dialog():
                    self._restore_gui_diagnostic_hooks()
                    event.accept()
                else:
                    event.ignore()
                return
            if reply == cancel_button:
                event.ignore()
                return
        self._restore_gui_diagnostic_hooks()
        event.accept()

    def _install_gui_diagnostic_hooks(self) -> None:
        if self._diagnostic_hooks_active:
            return
        self._previous_warning_showwarning = warnings.showwarning
        self._previous_excepthook = sys.excepthook
        warnings.showwarning = self._capture_python_warning
        sys.excepthook = self._capture_unhandled_exception
        self._diagnostic_hooks_active = True

    def _restore_gui_diagnostic_hooks(self) -> None:
        if not self._diagnostic_hooks_active:
            return
        if self._previous_warning_showwarning is not None:
            warnings.showwarning = self._previous_warning_showwarning
        if self._previous_excepthook is not None:
            sys.excepthook = self._previous_excepthook
        self._diagnostic_hooks_active = False

    def _capture_python_warning(
        self,
        message,
        category,
        filename,
        lineno,
        file=None,
        line=None,
    ) -> None:
        category_name = getattr(category, "__name__", str(category))
        detail = warnings.formatwarning(message, category, filename, lineno, line).rstrip()
        self.record_diagnostic_event(
            "warning",
            str(message),
            detail=detail,
            context={
                "category": category_name,
                "file": filename,
                "line": lineno,
            },
        )

    def _capture_unhandled_exception(self, exc_type, exc, tb) -> None:
        detail = "".join(traceback.format_exception(exc_type, exc, tb))
        self.record_diagnostic_event(
            "error",
            f"Unhandled {getattr(exc_type, '__name__', exc_type)}: {exc}",
            detail=detail,
            context={"source": "sys.excepthook"},
        )

    def record_diagnostic_event(
        self,
        level: str,
        message: Any,
        *,
        detail: Any = "",
        context: Mapping[str, Any] | None = None,
        update_status: bool = True,
    ) -> dict[str, Any]:
        event = _make_gui_diagnostic_event(level, message, detail=detail, context=context)
        self.diagnostic_events.append(event)
        if update_status and hasattr(self, "statusBar"):
            status_message = str(message)
            if len(status_message) > 180:
                status_message = f"{status_message[:177]}..."
            try:
                self.statusBar().showMessage(f"{str(level).capitalize()}: {status_message}. See Diagnostics.")
            except Exception:
                _LOG.debug("Ignored optional GUI operation failure", exc_info=True)
        if hasattr(self, "diagnostics_text"):
            try:
                self.update_diagnostics()
            except Exception:
                _LOG.debug("Ignored optional GUI operation failure", exc_info=True)
        try:
            self._update_diagnostics_attention()
        except Exception:
            _LOG.debug("Ignored optional GUI operation failure", exc_info=True)
        return event

    def _diagnostic_problem_count(self) -> int:
        return sum(
            1
            for event in self.diagnostic_events
            if str(event.get("level") or "").lower() in {"warning", "error", "critical"}
        )

    def _update_diagnostics_attention(self) -> None:
        button = getattr(self, "feature_nav_buttons", {}).get("Diagnostics")
        if button is None:
            return
        count = self._diagnostic_problem_count()
        if count:
            button.setText("\u26a0 Diagnostics")
            button.setToolTip(f"{count} warning/error diagnostic event(s) recorded.")
        else:
            button.setText("Diagnostics")
            button.setToolTip("")

    def mark_dirty(self, reason: str = ""):
        if not self.session.data:
            return
        self._dirty = True
        self._dirty_reason = reason or self._dirty_reason
        self.setWindowModified(True)
        if self._active_study_key in self._study_workspaces:
            workspace = self._study_workspaces[self._active_study_key]
            workspace.dirty = True
            workspace.dirty_reason = self._dirty_reason
        self.refresh_study_selector()

    def clear_dirty(self):
        self._dirty = False
        self._dirty_reason = ""
        self.setWindowModified(False)
        if self._active_study_key in self._study_workspaces:
            workspace = self._study_workspaces[self._active_study_key]
            workspace.dirty = False
            workspace.dirty_reason = ""
        self.refresh_study_selector()

    def _scale_for_window_size(self) -> float:
        size = self.size()
        base_area = 1320.0 * 860.0
        area = max(1.0, float(size.width()) * float(size.height()))
        return float(np.clip(np.sqrt(area / base_area), 0.92, 1.65))

    def _scaled_px(self, value: float) -> int:
        return max(1, int(round(float(value) * self._ui_scale)))

    def _center_dialog(self, dialog) -> None:
        preserve_size = bool(dialog.property("mrax_preserve_initial_size"))
        requested_size = dialog.size() if preserve_size else None
        try:
            dialog.adjustSize()
        except Exception:
            _LOG.debug("Ignored optional GUI operation failure", exc_info=True)
        if requested_size is not None:
            dialog.resize(
                max(int(dialog.width()), int(requested_size.width())),
                max(int(dialog.height()), int(requested_size.height())),
            )
        parent_geo = self.frameGeometry()
        geo = dialog.frameGeometry()
        if geo.width() <= 0 or geo.height() <= 0:
            hint = dialog.sizeHint()
            dialog.resize(max(geo.width(), hint.width()), max(geo.height(), hint.height()))
            geo = dialog.frameGeometry()
        geo.moveCenter(parent_geo.center())
        screen = self.screen() or QtWidgets.QApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            max_x = max(available.left(), available.right() - geo.width() + 1)
            max_y = max(available.top(), available.bottom() - geo.height() + 1)
            geo.moveTo(
                min(max(geo.x(), available.left()), max_x),
                min(max(geo.y(), available.top()), max_y),
            )
        dialog.move(geo.topLeft())

    def _exec_dialog(self, dialog):
        self._center_dialog(dialog)
        return dialog.exec()

    def _show_centered_dialog(self, dialog) -> None:
        self._center_dialog(dialog)
        dialog.show()
        self._center_dialog(dialog)

    def _unique_study_key(self, label: str) -> str:
        base = _safe_filename(str(label or "study")) or "study"
        key = base
        idx = 2
        while key in self._study_workspaces:
            key = f"{base}_{idx}"
            idx += 1
        return key

    def _workspace_label_for_session(self, session: GuiSession, fallback: str = "Untitled") -> str:
        if session.data:
            name = session.study_name
            if name:
                return name
            if session.manifest_path is not None:
                return session.manifest_path.parent.name or session.manifest_path.stem
        return fallback

    def _create_workspace(
        self,
        label: str,
        session: GuiSession,
        *,
        activate: bool = False,
    ) -> str:
        key = self._unique_study_key(label)
        workspace = StudyWorkspace(
            key=key,
            label=str(label or "Untitled"),
            session=session,
            fit_results_by_experiment=dict(getattr(self, "_fit_results_by_experiment", {}) or {}),
            preview_results_by_experiment=dict(getattr(self, "_preview_results_by_experiment", {}) or {}),
            registration_results=dict(getattr(self, "registration_results", {}) or {}),
            drift_results=dict(getattr(self, "drift_results", {}) or {}),
            visual_undo_stacks=dict(getattr(self, "_visual_undo_stacks", {}) or {}),
            view_label=getattr(self, "view_combo", None).currentText() if hasattr(self, "view_combo") else "Image",
            plot_x_axis_key=getattr(self, "_plot_x_axis_key", "auto"),
            feature_page_index=getattr(self.feature_stack, "currentIndex", lambda: 0)() if hasattr(self, "feature_stack") else 0,
            dirty=bool(getattr(self, "_dirty", False)),
            dirty_reason=str(getattr(self, "_dirty_reason", "") or ""),
        )
        self._study_workspaces[key] = workspace
        if activate or self._active_study_key is None:
            self._active_study_key = key
            self._restore_workspace_to_window(workspace, update_widgets=False)
        return key

    def _capture_active_workspace(self) -> None:
        key = self._active_study_key
        if key is None or key not in self._study_workspaces:
            return
        if self.session.selected_experiment:
            self.session._save_current_view_state()
        workspace = self._study_workspaces[key]
        workspace.session = self.session
        workspace.fit_results_by_experiment = dict(self._fit_results_by_experiment)
        workspace.preview_results_by_experiment = dict(self._preview_results_by_experiment)
        workspace.registration_results = dict(self.registration_results)
        workspace.drift_results = dict(self.drift_results)
        workspace.visual_undo_stacks = dict(self._visual_undo_stacks)
        workspace.view_label = self.view_combo.currentText() if hasattr(self, "view_combo") else workspace.view_label
        workspace.plot_x_axis_key = self._plot_x_axis_key
        workspace.feature_page_index = self.feature_stack.currentIndex() if hasattr(self, "feature_stack") else workspace.feature_page_index
        workspace.dirty = bool(self._dirty)
        workspace.dirty_reason = str(self._dirty_reason or "")
        workspace.label = self._workspace_label_for_session(self.session, workspace.label)

    def _restore_workspace_to_window(self, workspace: StudyWorkspace, *, update_widgets: bool = True) -> None:
        self.session = workspace.session
        self._fit_results_by_experiment = workspace.fit_results_by_experiment
        self._preview_results_by_experiment = workspace.preview_results_by_experiment
        self.registration_results = workspace.registration_results
        self.drift_results = workspace.drift_results
        self._visual_undo_stacks = workspace.visual_undo_stacks
        self._plot_x_axis_key = workspace.plot_x_axis_key
        self._dirty = bool(workspace.dirty)
        self._dirty_reason = str(workspace.dirty_reason or "")
        self.last_fit_result = self._fit_results_by_experiment.get(self.session.selected_experiment or "")
        self.last_preview_fit_result = self._preview_results_by_experiment.get(self.session.selected_experiment or "")
        self.setWindowModified(self._dirty)
        if not update_widgets:
            return
        if hasattr(self, "view_combo"):
            self.view_combo.blockSignals(True)
            if workspace.view_label in [self.view_combo.itemText(idx) for idx in range(self.view_combo.count())]:
                self.view_combo.setCurrentText(workspace.view_label)
            else:
                self.view_combo.setCurrentText("Image")
            self.view_combo.blockSignals(False)
        if hasattr(self, "feature_stack"):
            self._select_feature_page(workspace.feature_page_index)
        self.populate_from_session()
        self.refresh_study_selector()

    def refresh_study_selector(self):
        combo = getattr(self, "study_combo", None)
        if combo is None:
            return
        combo.blockSignals(True)
        try:
            combo.clear()
            for key, workspace in self._study_workspaces.items():
                suffix = " *" if workspace.dirty else ""
                combo.addItem(f"{workspace.label}{suffix}", key)
            if self._active_study_key is not None:
                idx = combo.findData(self._active_study_key)
                if idx >= 0:
                    combo.setCurrentIndex(idx)
        finally:
            combo.blockSignals(False)

    def on_study_changed(self, _index: int):
        key = self.study_combo.currentData() if hasattr(self, "study_combo") else None
        if not key or key == self._active_study_key:
            return
        self.switch_study(str(key))

    def switch_study(self, key: str):
        if self.is_busy:
            return
        if key not in self._study_workspaces:
            return
        self._capture_active_workspace()
        self._active_study_key = key
        self._restore_workspace_to_window(self._study_workspaces[key])
        self.statusBar().showMessage(f"Switched to study: {self._study_workspaces[key].label}")

    def _replace_or_add_loaded_workspace(self, session: GuiSession, label: str) -> None:
        self._capture_active_workspace()
        current_key = self._active_study_key
        current = self._study_workspaces.get(current_key or "")
        can_replace_current = (
            current is not None
            and not current.session.data
            and not current.dirty
            and len(self._study_workspaces) == 1
        )
        self.session = session
        self._fit_results_by_experiment = {}
        self._preview_results_by_experiment = {}
        self.registration_results = {}
        self.drift_results = {}
        self._visual_undo_stacks = {}
        self.last_fit_result = None
        self.last_preview_fit_result = None
        self._plot_x_axis_key = "auto"
        self._dirty = False
        self._dirty_reason = ""
        if can_replace_current:
            current.label = str(label or "Study")
            current.session = session
            current.fit_results_by_experiment = {}
            current.preview_results_by_experiment = {}
            current.registration_results = {}
            current.drift_results = {}
            current.visual_undo_stacks = {}
            current.view_label = "Image"
            current.plot_x_axis_key = "auto"
            current.feature_page_index = 0
            current.dirty = False
            current.dirty_reason = ""
            self._restore_workspace_to_window(current)
        else:
            key = self._create_workspace(str(label or "Study"), session, activate=False)
            self._active_study_key = key
            self._restore_workspace_to_window(self._study_workspaces[key])

    def _information_box(self, title: str, text: str):
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Information)
        box.setWindowTitle(title)
        box.setText(text)
        return self._exec_dialog(box)

    def _question_box(self, title: str, text: str, buttons, default_button):
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Question)
        box.setWindowTitle(title)
        box.setText(text)
        box.setStandardButtons(buttons)
        box.setDefaultButton(default_button)
        return self._exec_dialog(box)

    def _get_open_file_name(self, title: str, directory: str, name_filter: str) -> tuple[str, str]:
        dialog = QtWidgets.QFileDialog(self, title, directory, name_filter)
        dialog.setOptions(_file_dialog_options(QtWidgets))
        file_mode = getattr(QtWidgets.QFileDialog, "FileMode", QtWidgets.QFileDialog)
        accept_mode = getattr(QtWidgets.QFileDialog, "AcceptMode", QtWidgets.QFileDialog)
        dialog.setFileMode(file_mode.ExistingFile)
        dialog.setAcceptMode(accept_mode.AcceptOpen)
        if self._exec_dialog(dialog) != QtWidgets.QDialog.Accepted:
            return "", ""
        files = dialog.selectedFiles()
        return (files[0] if files else ""), dialog.selectedNameFilter()

    def _get_save_file_name(self, title: str, directory: str, name_filter: str) -> tuple[str, str]:
        dialog = QtWidgets.QFileDialog(self, title, directory, name_filter)
        dialog.setOptions(_file_dialog_options(QtWidgets))
        file_mode = getattr(QtWidgets.QFileDialog, "FileMode", QtWidgets.QFileDialog)
        accept_mode = getattr(QtWidgets.QFileDialog, "AcceptMode", QtWidgets.QFileDialog)
        dialog.setFileMode(file_mode.AnyFile)
        dialog.setAcceptMode(accept_mode.AcceptSave)
        if self._exec_dialog(dialog) != QtWidgets.QDialog.Accepted:
            return "", ""
        files = dialog.selectedFiles()
        return (files[0] if files else ""), dialog.selectedNameFilter()

    def _get_existing_directory(self, title: str, directory: str) -> str:
        dialog = QtWidgets.QFileDialog(self, title, directory)
        dialog.setOptions(_file_dialog_options(QtWidgets))
        file_mode = getattr(QtWidgets.QFileDialog, "FileMode", QtWidgets.QFileDialog)
        accept_mode = getattr(QtWidgets.QFileDialog, "AcceptMode", QtWidgets.QFileDialog)
        option = getattr(QtWidgets.QFileDialog, "Option", QtWidgets.QFileDialog)
        dialog.setFileMode(file_mode.Directory)
        dialog.setAcceptMode(accept_mode.AcceptOpen)
        dialog.setOption(option.ShowDirsOnly, True)
        if self._exec_dialog(dialog) != QtWidgets.QDialog.Accepted:
            return ""
        files = dialog.selectedFiles()
        return files[0] if files else ""

    def _build_ui(self):
        central = QtWidgets.QSplitter()
        self._central_splitter = central
        central.setOrientation(QtCore.Qt.Horizontal)
        self.setCentralWidget(central)

        left = self._build_left_panel()
        center = self._build_center_panel()
        right = self._build_right_panel()
        central.addWidget(left)
        central.addWidget(center)
        central.addWidget(right)
        central.setStretchFactor(0, 0)
        central.setStretchFactor(1, 1)
        central.setStretchFactor(2, 0)
        central.setSizes([260, 760, 340])
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setFixedWidth(220)
        self.progress_bar.hide()
        self.statusBar().addPermanentWidget(self.progress_bar)
        self.refresh_study_selector()
        self.statusBar().showMessage(f"MRax GUI ready. {GUI_INSTALL_HINT}")

    def _build_left_panel(self):
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        layout.addWidget(QtWidgets.QLabel("Study"))
        self.study_combo = QtWidgets.QComboBox()
        self.study_combo.currentIndexChanged.connect(self.on_study_changed)
        layout.addWidget(self.study_combo)

        study_row = QtWidgets.QHBoxLayout()
        new_study_btn = QtWidgets.QPushButton("New Study")
        new_study_btn.setStatusTip("Create a new empty in-memory study workspace.")
        new_study_btn.clicked.connect(self.new_empty_study_dialog)
        rename_study_btn = QtWidgets.QPushButton("Rename Study")
        rename_study_btn.setStatusTip("Rename the active study workspace and saved study name.")
        rename_study_btn.clicked.connect(self.rename_study_dialog)
        study_row.addWidget(new_study_btn)
        study_row.addWidget(rename_study_btn)
        layout.addLayout(study_row)

        file_row = QtWidgets.QHBoxLayout()
        load_btn = QtWidgets.QPushButton("Load...")
        load_btn.clicked.connect(self.load_dialog)
        save_btn = QtWidgets.QPushButton("Save...")
        save_btn.clicked.connect(self.save_dialog)
        file_row.addWidget(load_btn)
        file_row.addWidget(save_btn)
        layout.addLayout(file_row)

        import_row = QtWidgets.QHBoxLayout()
        import_btn = QtWidgets.QPushButton("Import...")
        import_btn.clicked.connect(self.import_study_dialog)
        export_btn = QtWidgets.QPushButton("Export...")
        export_btn.clicked.connect(self.export_dialog)
        import_row.addWidget(import_btn)
        import_row.addWidget(export_btn)
        layout.addLayout(import_row)

        layout.addWidget(QtWidgets.QLabel("Experiments"))
        exp_action_row = QtWidgets.QHBoxLayout()
        add_exp_btn = QtWidgets.QPushButton("Add Experiment")
        add_exp_btn.setStatusTip(
            "Add blank, NIfTI, DICOM, NumPy, or raw binary image data to the active study."
        )
        add_exp_btn.clicked.connect(self.add_experiment_dialog)
        rename_exp_btn = QtWidgets.QPushButton("Rename")
        rename_exp_btn.setStatusTip("Rename the selected experiment in the active study.")
        rename_exp_btn.clicked.connect(self.rename_experiment_dialog)
        edit_exp_btn = QtWidgets.QPushButton("Edit")
        edit_exp_btn.setStatusTip("Edit experiment type, model, observables, and extra metadata.")
        edit_exp_btn.clicked.connect(self.edit_experiment_metadata_dialog)
        exp_action_row.addWidget(add_exp_btn)
        exp_action_row.addWidget(rename_exp_btn)
        exp_action_row.addWidget(edit_exp_btn)
        layout.addLayout(exp_action_row)
        self.experiment_list = QtWidgets.QListWidget()
        self.experiment_list.currentTextChanged.connect(self.on_experiment_changed)
        layout.addWidget(self.experiment_list, 2)

        layout.addWidget(QtWidgets.QLabel("View"))
        self.view_combo = QtWidgets.QComboBox()
        self.view_combo.addItems(["Image", "MRax reconstruction", "Vendor vs MRax", "Vendor-MRax diff", "K-space", "Map"])
        self.view_combo.currentTextChanged.connect(self.on_view_changed)
        layout.addWidget(self.view_combo)

        self.kspace_mode = QtWidgets.QComboBox()
        self.kspace_mode.addItems(["log_magnitude", "magnitude", "phase", "real", "imag"])
        self.kspace_mode.currentTextChanged.connect(self.on_kspace_mode_changed)
        layout.addWidget(self.kspace_mode)

        self.kspace_centering = QtWidgets.QComboBox()
        self.kspace_centering.addItems(["auto", "none", "fftshift"])
        self.kspace_centering.currentTextChanged.connect(self.on_kspace_centering_changed)
        layout.addWidget(self.kspace_centering)

        layout.addWidget(QtWidgets.QLabel("Maps"))
        self.map_list = QtWidgets.QListWidget()
        self.map_list.currentTextChanged.connect(self.on_map_changed)
        layout.addWidget(self.map_list, 1)

        layout.addStretch(1)
        return panel

    def _add_button_grid(self, grid, entries, *, columns: int = 3):
        buttons = {}
        for idx, (label, slot) in enumerate(entries):
            button = QtWidgets.QPushButton(label)
            button.clicked.connect(slot)
            grid.addWidget(button, idx // columns, idx % columns)
            buttons[label] = button
        return buttons

    def _build_center_panel(self):
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        toolbar = QtWidgets.QVBoxLayout()
        action_row = QtWidgets.QHBoxLayout()
        reset_btn = QtWidgets.QPushButton("Reset View")
        reset_btn.clicked.connect(self.reset_image_view)
        enlarge_img = QtWidgets.QPushButton("Enlarge Image")
        enlarge_img.clicked.connect(self.open_enlarged_image)
        enlarge_plot = QtWidgets.QPushButton("Enlarge Plot")
        enlarge_plot.clicked.connect(self.open_enlarged_plot)
        summary_btn = QtWidgets.QPushButton("Study Summary")
        summary_btn.clicked.connect(self.open_summary_window)
        self.plot_x_combo = QtWidgets.QComboBox()
        self.plot_x_combo.setMinimumWidth(140)
        self.plot_x_combo.currentIndexChanged.connect(self.on_plot_x_axis_changed)
        action_row.addWidget(reset_btn)
        action_row.addWidget(enlarge_img)
        action_row.addWidget(enlarge_plot)
        action_row.addWidget(summary_btn)
        action_row.addWidget(QtWidgets.QLabel("Plot X"))
        action_row.addWidget(self.plot_x_combo)
        action_row.addStretch(1)
        toolbar.addLayout(action_row)
        self.dimension_controls_widget = QtWidgets.QWidget()
        self.dimension_controls_layout = QtWidgets.QGridLayout(self.dimension_controls_widget)
        self.dimension_controls_layout.setContentsMargins(0, 0, 0, 0)
        toolbar.addWidget(self.dimension_controls_widget)
        layout.addLayout(toolbar)

        splitter = QtWidgets.QSplitter()
        splitter.setOrientation(QtCore.Qt.Vertical)
        self.image_view = pg.ImageView()
        self.image_view.ui.roiBtn.hide()
        self.image_view.ui.menuBtn.hide()
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.showGrid(x=True, y=True, alpha=0.25)
        splitter.addWidget(self.image_view)
        splitter.addWidget(self.plot_widget)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, 1)

        self.image_view.getImageItem().scene().sigMouseClicked.connect(self.on_image_clicked)
        self.image_view.getImageItem().scene().sigMouseMoved.connect(self.on_image_hovered)
        return panel

    def _build_right_panel(self):
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        nav = QtWidgets.QWidget()
        nav_layout = QtWidgets.QGridLayout(nav)
        nav_layout.setContentsMargins(0, 0, 0, 0)
        nav_layout.setSpacing(4)
        self.feature_nav_buttons: dict[str, Any] = {}
        self.feature_stack = QtWidgets.QStackedWidget()
        pages = (
            ("Inspector", self._build_inspector_tab()),
            ("View State", self._build_visual_state_page()),
            ("ROIs", self._build_roi_page()),
            ("Fit", self._build_fit_tab()),
            ("Registration", self._build_registration_tab()),
            ("Metadata", self._build_metadata_tab()),
            ("Summary", self._build_summary_tab()),
            ("Diagnostics", self._build_diagnostics_tab()),
        )
        self.feature_page_labels = [label for label, _page in pages]
        for idx, (label, page) in enumerate(pages):
            button = QtWidgets.QPushButton(label)
            button.setCheckable(True)
            button.clicked.connect(lambda _checked=False, index=idx: self._select_feature_page(index))
            nav_layout.addWidget(button, idx // 4, idx % 4)
            self.feature_nav_buttons[label] = button
            self.feature_stack.addWidget(page)
        layout.addWidget(nav)
        layout.addWidget(self.feature_stack, 1)
        self.tabs = self.feature_stack
        self._select_feature_page(0)
        self._update_diagnostics_attention()
        return panel

    def _select_feature_page(self, index: int):
        if not hasattr(self, "feature_stack"):
            return
        idx = max(0, min(int(index), self.feature_stack.count() - 1))
        previous = self.feature_stack.currentIndex()
        leaving_registration = (
            previous != idx
            and 0 <= previous < len(getattr(self, "feature_page_labels", []))
            and self.feature_page_labels[previous] == "Registration"
        )
        if leaving_registration:
            self._disable_registration_reference_overlay(update=False)
            self._clear_registration_preview(update=False)
        self.feature_stack.setCurrentIndex(idx)
        for pos, button in enumerate(getattr(self, "feature_nav_buttons", {}).values()):
            button.setChecked(pos == idx)
        if leaving_registration and self.session.is_loaded:
            self.update_view()

    def _build_visual_state_page(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setSpacing(8)
        visual_box = QtWidgets.QGroupBox("View State")
        visual_layout = QtWidgets.QFormLayout(visual_box)
        self.visual_original_check = QtWidgets.QCheckBox("Show original data")
        self.visual_original_check.setChecked(True)
        self.visual_original_check.toggled.connect(self.on_visual_state_controls_changed)
        self.visual_display_mode_combo = QtWidgets.QComboBox()
        self.visual_display_mode_combo.addItem("Current experiment view", "current")
        self.visual_display_mode_combo.currentIndexChanged.connect(self.on_visual_state_controls_changed)
        self.visual_mask_enabled_check = QtWidgets.QCheckBox("Apply binary mask")
        self.visual_mask_enabled_check.toggled.connect(self.on_visual_state_controls_changed)
        self.visual_mask_combo = QtWidgets.QComboBox()
        self.visual_mask_combo.currentIndexChanged.connect(self.on_visual_state_controls_changed)
        self.visual_drift_combo = QtWidgets.QComboBox()
        self.visual_drift_combo.currentIndexChanged.connect(self.on_visual_state_controls_changed)
        self.visual_transform_combo = QtWidgets.QComboBox()
        self.visual_transform_combo.currentIndexChanged.connect(self.on_visual_state_controls_changed)
        self.visual_transform_direction_combo = QtWidgets.QComboBox()
        self.visual_transform_direction_combo.addItem("Moving to reference", "moving_to_reference")
        self.visual_transform_direction_combo.addItem("Reference to moving", "reference_to_moving")
        self.visual_transform_direction_combo.currentIndexChanged.connect(self.on_visual_state_controls_changed)
        self.visual_color_combo = QtWidgets.QComboBox()
        for value in registration_choice_sets()["channel"]:
            self.visual_color_combo.addItem(value, value)
        self.visual_color_combo.currentIndexChanged.connect(self.on_visual_state_controls_changed)
        self.visual_alpha_spin = QtWidgets.QDoubleSpinBox()
        self.visual_alpha_spin.setRange(0.0, 1.0)
        self.visual_alpha_spin.setDecimals(2)
        self.visual_alpha_spin.setSingleStep(0.05)
        self.visual_alpha_spin.setValue(1.0)
        self.visual_alpha_spin.valueChanged.connect(self.on_visual_state_controls_changed)
        self.visual_reference_channel_combo = QtWidgets.QComboBox()
        self.visual_moving_channel_combo = QtWidgets.QComboBox()
        for combo in (self.visual_reference_channel_combo, self.visual_moving_channel_combo):
            for value in registration_choice_sets()["channel"]:
                combo.addItem(value, value)
            combo.currentIndexChanged.connect(self.on_visual_state_controls_changed)
        self.visual_reference_channel_combo.setCurrentText("red")
        self.visual_moving_channel_combo.setCurrentText("green")
        self.visual_reference_alpha_spin = QtWidgets.QDoubleSpinBox()
        self.visual_moving_alpha_spin = QtWidgets.QDoubleSpinBox()
        for spin in (self.visual_reference_alpha_spin, self.visual_moving_alpha_spin):
            spin.setRange(0.0, 1.0)
            spin.setDecimals(2)
            spin.setSingleStep(0.05)
            spin.setValue(1.0)
            spin.valueChanged.connect(self.on_visual_state_controls_changed)
        visual_layout.addRow(self.visual_original_check)
        visual_layout.addRow("Display", self.visual_display_mode_combo)
        visual_layout.addRow(self.visual_mask_enabled_check)
        visual_layout.addRow("Mask", self.visual_mask_combo)
        visual_layout.addRow("Drift correction", self.visual_drift_combo)
        visual_layout.addRow("Transform", self.visual_transform_combo)
        visual_layout.addRow("Direction", self.visual_transform_direction_combo)
        visual_layout.addRow("Channel", self.visual_color_combo)
        visual_layout.addRow("Opacity", self.visual_alpha_spin)
        visual_buttons = QtWidgets.QHBoxLayout()
        visual_undo = QtWidgets.QPushButton("Undo Visual")
        visual_undo.clicked.connect(self.undo_visual_state)
        visual_reset = QtWidgets.QPushButton("Reset Visual")
        visual_reset.clicked.connect(self.reset_visual_state)
        visual_restore_image = QtWidgets.QPushButton("Restore Source Image")
        visual_restore_image.clicked.connect(self.restore_original_image_data)
        visual_buttons.addWidget(visual_undo)
        visual_buttons.addWidget(visual_reset)
        visual_buttons.addWidget(visual_restore_image)
        visual_layout.addRow(visual_buttons)
        layout.addWidget(visual_box)
        layout.addStretch(1)
        return widget

    def _build_roi_page(self):
        widget = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(widget)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(content)
        layout.setSpacing(8)

        manage_box = QtWidgets.QGroupBox("ROI Management")
        manage_layout = QtWidgets.QGridLayout(manage_box)
        roi_manage_buttons = self._add_button_grid(
            manage_layout,
            (
                ("Import...", self.import_roi_dialog),
                ("New ROI", self.add_empty_roi),
                ("Rename", self.rename_selected_roi),
                ("Remove", self.remove_selected_roi),
                ("Disable All", self.clear_rois),
            ),
            columns=2,
        )
        roi_manage_buttons["Import..."].setStatusTip("Import an MRax ROI set, NIfTI mask, NumPy mask, or raw binary mask.")
        layout.addWidget(manage_box)

        layout.addWidget(QtWidgets.QLabel("ROIs"))
        self.roi_list = QtWidgets.QListWidget()
        selection_mode = getattr(QtWidgets.QAbstractItemView, "SelectionMode", QtWidgets.QAbstractItemView)
        self.roi_list.setSelectionMode(selection_mode.ExtendedSelection)
        self.roi_list.currentTextChanged.connect(self.on_roi_changed)
        self.roi_list.itemChanged.connect(self.on_roi_item_check_changed)
        self.roi_list.itemSelectionChanged.connect(self.on_roi_selection_changed)
        self.roi_list.setMinimumHeight(self._scaled_px(140))
        layout.addWidget(self.roi_list, 1)

        roi_view_box = QtWidgets.QGroupBox("ROI View State")
        roi_view_layout = QtWidgets.QGridLayout(roi_view_box)
        roi_view_mask = QtWidgets.QPushButton("Mask View With ROI")
        roi_view_mask.clicked.connect(self.apply_selected_roi_as_visual_mask)
        roi_view_clear = QtWidgets.QPushButton("Clear View Mask")
        roi_view_clear.clicked.connect(self.clear_visual_mask)
        roi_view_layout.addWidget(roi_view_mask, 0, 0)
        roi_view_layout.addWidget(roi_view_clear, 0, 1)
        layout.addWidget(roi_view_box)

        editing_box = QtWidgets.QGroupBox("ROI Editing")
        editing_layout = QtWidgets.QVBoxLayout(editing_box)
        self._roi_tool_settings_widget = QtWidgets.QWidget()
        brush_settings = QtWidgets.QGridLayout(self._roi_tool_settings_widget)
        brush_settings.setContentsMargins(0, 0, 0, 0)
        self._brush_size_label = QtWidgets.QLabel("Brush")
        self.brush_size_spin = QtWidgets.QSpinBox()
        self.brush_size_spin.setRange(1, 128)
        self.brush_size_spin.setValue(4)
        self.brush_size_spin.setSuffix(" px")
        self._brush_label_label = QtWidgets.QLabel("Label")
        self.brush_label_spin = QtWidgets.QSpinBox()
        self.brush_label_spin.setRange(1, 1_000_000)
        self.brush_label_spin.setValue(1)
        self._wand_tolerance_label = QtWidgets.QLabel("Tolerance")
        self.wand_tolerance_spin = QtWidgets.QDoubleSpinBox()
        self.wand_tolerance_spin.setRange(0.0, 100.0)
        self.wand_tolerance_spin.setDecimals(1)
        self.wand_tolerance_spin.setSingleStep(1.0)
        self.wand_tolerance_spin.setValue(8.0)
        self.wand_tolerance_spin.setSuffix(" %")
        self._lasso_support_label = QtWidgets.QLabel("Support points")
        self.lasso_support_spin = QtWidgets.QSpinBox()
        self.lasso_support_spin.setRange(6, 512)
        self.lasso_support_spin.setValue(64)
        self.lasso_support_spin.setSuffix(" pts")
        self._lasso_spline_label = QtWidgets.QLabel("Spline")
        self.lasso_spline_combo = QtWidgets.QComboBox()
        self.lasso_spline_combo.addItem("Cubic", "cubic")
        self.lasso_spline_combo.addItem("Linear", "linear")
        self.lasso_snap_check = QtWidgets.QCheckBox("Snap lasso to edges")
        self.lasso_snap_check.setChecked(False)
        self.brush_auto_subtract_check = QtWidgets.QCheckBox("Auto subtract from other ROIs")
        self.brush_auto_subtract_check.setChecked(False)
        brush_settings.addWidget(self._brush_size_label, 0, 0)
        brush_settings.addWidget(self.brush_size_spin, 0, 1)
        brush_settings.addWidget(self._brush_label_label, 1, 0)
        brush_settings.addWidget(self.brush_label_spin, 1, 1)
        brush_settings.addWidget(self.brush_auto_subtract_check, 2, 0, 1, 2)
        brush_settings.addWidget(self._wand_tolerance_label, 3, 0)
        brush_settings.addWidget(self.wand_tolerance_spin, 3, 1)
        brush_settings.addWidget(self._lasso_support_label, 4, 0)
        brush_settings.addWidget(self.lasso_support_spin, 4, 1)
        brush_settings.addWidget(self._lasso_spline_label, 5, 0)
        brush_settings.addWidget(self.lasso_spline_combo, 5, 1)
        brush_settings.addWidget(self.lasso_snap_check, 6, 0, 1, 2)
        self._roi_tool_settings_section = _section_box(QtWidgets, "Tool Settings", self._roi_tool_settings_widget)
        self._roi_tool_settings_section.setVisible(False)
        editing_layout.addWidget(self._roi_tool_settings_section)

        label_controls = QtWidgets.QWidget()
        label_layout = QtWidgets.QFormLayout(label_controls)
        label_layout.setContentsMargins(0, 0, 0, 0)
        self.label_mask_label_combo = QtWidgets.QComboBox()
        self.label_mask_label_combo.currentIndexChanged.connect(self.on_label_mask_label_changed)
        self.label_mask_value_spin = QtWidgets.QSpinBox()
        self.label_mask_value_spin.setRange(1, 1_000_000)
        self.label_mask_value_spin.setValue(1)
        change_label_btn = QtWidgets.QPushButton("Change Label")
        change_label_btn.clicked.connect(self.change_selected_label_value)
        label_layout.addRow("Label", self.label_mask_label_combo)
        label_layout.addRow("Value", self.label_mask_value_spin)
        label_layout.addRow(change_label_btn)
        self._label_mask_controls_section = _section_box(QtWidgets, "Label Mask", label_controls)
        self._label_mask_controls_section.setVisible(False)
        editing_layout.addWidget(self._label_mask_controls_section)

        self._drawing_controls_widget = QtWidgets.QWidget()
        drawing_controls = QtWidgets.QGridLayout(self._drawing_controls_widget)
        drawing_controls.setContentsMargins(0, 0, 0, 0)
        self._drawing_status_label = QtWidgets.QLabel("Drawing: idle")
        self._drawing_status_label.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        self._finish_drawing_button = QtWidgets.QPushButton("Done")
        self._undo_drawing_button = QtWidgets.QPushButton("Undo")
        self._cancel_drawing_button = QtWidgets.QPushButton("Cancel")
        self._finish_drawing_button.clicked.connect(self.finish_drawing_roi)
        self._undo_drawing_button.clicked.connect(self.undo_drawing_point)
        self._cancel_drawing_button.clicked.connect(self.cancel_drawing_roi)
        drawing_controls.addWidget(self._drawing_status_label, 0, 0, 1, 3)
        drawing_controls.addWidget(self._finish_drawing_button, 1, 0)
        drawing_controls.addWidget(self._undo_drawing_button, 1, 1)
        drawing_controls.addWidget(self._cancel_drawing_button, 1, 2)
        self._drawing_controls_widget.setVisible(False)
        editing_layout.addWidget(self._drawing_controls_widget)

        primary_tools = QtWidgets.QWidget()
        primary_layout = QtWidgets.QGridLayout(primary_tools)
        primary_layout.setContentsMargins(0, 0, 0, 0)
        self._add_button_grid(
            primary_layout,
            (
                ("Shape...", self.add_flexible_shape_roi),
                ("Lasso", self.add_lasso_roi),
                ("Brush", self.enable_brush),
                ("Erase", self.enable_erase),
                ("Wand", self.enable_wand),
                ("Auto", self.auto_threshold_roi),
            ),
        )
        editing_layout.addWidget(primary_tools)

        advanced_tools = QtWidgets.QWidget()
        advanced_layout = QtWidgets.QGridLayout(advanced_tools)
        advanced_layout.setContentsMargins(0, 0, 0, 0)
        self._advanced_roi_buttons = self._add_button_grid(
            advanced_layout,
            (
                ("Consolidate", self.consolidate_selected_roi),
                ("Combine", self.combine_rois),
                ("Subtract", self.subtract_rois),
                ("Shift...", self.shift_selected_rois),
                ("Copy Z", self.copy_selected_rois_to_z),
                ("Delete Z", self.delete_selected_roi_slice_z),
                ("Apply All", self.apply_rois_to_all_experiments),
                ("Interp Z", self.interpolate_selected_roi_z),
            ),
        )
        self._advanced_roi_actions_section = _section_box(QtWidgets, "Advanced ROI Actions", advanced_tools)
        self._advanced_roi_actions_section.setVisible(False)
        editing_layout.addWidget(self._advanced_roi_actions_section)
        layout.addWidget(editing_box)
        layout.addStretch(1)
        scroll.setWidget(content)
        outer.addWidget(scroll)
        self._refresh_drawing_controls()
        self._refresh_roi_action_visibility()
        return widget

    def _build_inspector_tab(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        self.inspector = QtWidgets.QTextEdit()
        self.inspector.setReadOnly(True)
        layout.addWidget(self.inspector)
        return widget

    def _build_metadata_tab(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        controls = QtWidgets.QHBoxLayout()
        self.metadata_filter_edit = QtWidgets.QLineEdit()
        self.metadata_filter_edit.setPlaceholderText("Filter metadata")
        self.metadata_filter_edit.textChanged.connect(self.filter_metadata_tree)
        expand_btn = QtWidgets.QPushButton("Expand")
        expand_btn.clicked.connect(lambda: self.metadata_tree.expandAll())
        collapse_btn = QtWidgets.QPushButton("Collapse")
        collapse_btn.clicked.connect(lambda: self.metadata_tree.collapseAll())
        controls.addWidget(self.metadata_filter_edit, 1)
        controls.addWidget(expand_btn)
        controls.addWidget(collapse_btn)
        layout.addLayout(controls)

        splitter = QtWidgets.QSplitter()
        splitter.setOrientation(QtCore.Qt.Vertical)
        self.metadata_tree = QtWidgets.QTreeWidget()
        self.metadata_tree.setColumnCount(3)
        self.metadata_tree.setHeaderLabels(["Name", "Value", "Type"])
        self.metadata_tree.currentItemChanged.connect(self.on_metadata_item_changed)
        self.metadata_detail = QtWidgets.QPlainTextEdit()
        self.metadata_detail.setReadOnly(True)
        splitter.addWidget(self.metadata_tree)
        splitter.addWidget(self.metadata_detail)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, 1)
        return widget

    def _build_summary_tab(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        self.summary_table = QtWidgets.QTableWidget()
        layout.addWidget(self.summary_table, 2)
        self.summary_plot = pg.PlotWidget()
        self.summary_plot.showGrid(x=True, y=True, alpha=0.25)
        layout.addWidget(self.summary_plot, 1)
        return widget

    def _build_fit_tab(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setSpacing(8)

        layout.addWidget(QtWidgets.QLabel("Model"))
        self.model_combo = QtWidgets.QComboBox()
        for choice in model_choices():
            self.model_combo.addItem(choice.label, choice.key)
        self.model_combo.currentIndexChanged.connect(self.refresh_advanced_fit_controls)
        self.model_combo.currentIndexChanged.connect(self.refresh_sat_trans_reference_combo)
        self.model_combo.currentIndexChanged.connect(self.refresh_ab_mt_map_combos)
        layout.addWidget(self.model_combo)

        core_box = QtWidgets.QGroupBox("Fit")
        form = QtWidgets.QFormLayout(core_box)
        self.optimizer_combo = QtWidgets.QComboBox()
        self.optimizer_combo.addItems(["adam", "adamw", "sgd", "rmsprop", "fista", "split_bregman"])
        self.steps_spin = QtWidgets.QSpinBox()
        self.steps_spin.setRange(0, 1_000_000)
        self.steps_spin.setValue(200)
        self.step_size_spin = QtWidgets.QDoubleSpinBox()
        self.step_size_spin.setDecimals(5)
        self.step_size_spin.setRange(1e-6, 10.0)
        self.step_size_spin.setValue(0.02)
        form.addRow("Optimizer", self.optimizer_combo)
        form.addRow("Steps", self.steps_spin)
        form.addRow("Step size", self.step_size_spin)
        self.fit_use_view_state_check = QtWidgets.QCheckBox("Use view-state masks/transforms")
        self.fit_use_view_state_check.setToolTip(
            "When enabled, fitting uses the current experiment after view-state binary masks and transforms are applied."
        )
        form.addRow("Fit input", self.fit_use_view_state_check)
        layout.addWidget(core_box)

        summary_box = QtWidgets.QGroupBox("Details")
        summary_layout = QtWidgets.QGridLayout(summary_box)
        summary_layout.setColumnStretch(1, 1)
        self.fit_inputs_summary_label = QtWidgets.QLabel("No model-specific inputs")
        self.fit_inputs_summary_label.setWordWrap(True)
        self.fit_options_summary_label = QtWidgets.QLabel("Default data options")
        self.fit_options_summary_label.setWordWrap(True)
        self.fit_parameters_summary_label = QtWidgets.QLabel("Model parameters")
        self.fit_parameters_summary_label.setWordWrap(True)
        self.fit_frames_summary_label = QtWidgets.QLabel("All frames included")
        self.fit_frames_summary_label.setWordWrap(True)
        self.fit_inputs_button = QtWidgets.QPushButton("Model Inputs...")
        self.fit_inputs_button.clicked.connect(self.open_fit_inputs_dialog)
        self.fit_options_button = QtWidgets.QPushButton("Data Options...")
        self.fit_options_button.clicked.connect(self.open_fit_options_dialog)
        self.fit_parameters_button = QtWidgets.QPushButton("Parameters...")
        self.fit_parameters_button.clicked.connect(self.open_fit_parameters_dialog)
        self.fit_frames_button = QtWidgets.QPushButton("Frames...")
        self.fit_frames_button.clicked.connect(self.open_fit_frames_dialog)
        for row, (button, label) in enumerate(
            (
                (self.fit_inputs_button, self.fit_inputs_summary_label),
                (self.fit_options_button, self.fit_options_summary_label),
                (self.fit_parameters_button, self.fit_parameters_summary_label),
                (self.fit_frames_button, self.fit_frames_summary_label),
            )
        ):
            summary_layout.addWidget(button, row, 0)
            summary_layout.addWidget(label, row, 1)
        layout.addWidget(summary_box)

        self.fit_options_dialog = QtWidgets.QDialog(self)
        self.fit_options_dialog.setWindowTitle("Fit Data Options")
        self.fit_options_dialog.resize(self._scaled_px(420), self._scaled_px(260))
        options_layout = QtWidgets.QVBoxLayout(self.fit_options_dialog)
        options_form = QtWidgets.QFormLayout()
        self.mask_fill_spin = QtWidgets.QDoubleSpinBox()
        self.mask_fill_spin.setDecimals(6)
        self.mask_fill_spin.setRange(-1e12, 1e12)
        self.mask_fill_spin.setValue(0.0)
        options_form.addRow("Mask fill", self.mask_fill_spin)
        self.auto_nan_check = QtWidgets.QCheckBox()
        self.auto_nan_check.setChecked(True)
        options_form.addRow("Auto-mask NaN observables", self.auto_nan_check)
        self.derived_maps_combo = QtWidgets.QComboBox()
        self.derived_maps_combo.addItems(["auto", "yes", "no"])
        options_form.addRow("Derived maps", self.derived_maps_combo)
        self.mask_fill_spin.valueChanged.connect(lambda _value: self.refresh_fit_breakout_summaries())
        self.auto_nan_check.toggled.connect(lambda _checked: self.refresh_fit_breakout_summaries())
        self.derived_maps_combo.currentIndexChanged.connect(lambda _idx: self.refresh_fit_breakout_summaries())
        self.fit_use_view_state_check.toggled.connect(lambda _checked: self.refresh_fit_breakout_summaries())
        options_layout.addLayout(options_form)
        options_layout.addStretch(1)
        options_close = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        options_close.rejected.connect(self.fit_options_dialog.hide)
        options_layout.addWidget(options_close)

        self.fit_inputs_dialog = QtWidgets.QDialog(self)
        self.fit_inputs_dialog.setWindowTitle("Model Inputs")
        self.fit_inputs_dialog.resize(self._scaled_px(520), self._scaled_px(260))
        inputs_layout = QtWidgets.QVBoxLayout(self.fit_inputs_dialog)
        self.fit_inputs_empty_label = QtWidgets.QLabel("The selected model does not need extra study-level inputs.")
        self.fit_inputs_empty_label.setWordWrap(True)
        inputs_layout.addWidget(self.fit_inputs_empty_label)
        inputs_form = QtWidgets.QFormLayout()
        self.sat_trans_reference_combo = QtWidgets.QComboBox()
        self.sat_trans_reference_combo.setToolTip(
            "Single-frame saturation-transfer scan used to normalize this scan before fitting."
        )
        self.sat_trans_reference_combo.currentIndexChanged.connect(
            lambda _idx: self.refresh_fit_breakout_summaries()
        )
        self.sat_trans_reference_label = QtWidgets.QLabel("sat_trans reference")
        inputs_form.addRow(self.sat_trans_reference_label, self.sat_trans_reference_combo)
        self.ab_mt_t1_combo = QtWidgets.QComboBox()
        self.ab_mt_b1_combo = QtWidgets.QComboBox()
        self.ab_mt_b0_combo = QtWidgets.QComboBox()
        for combo in (self.ab_mt_t1_combo, self.ab_mt_b1_combo, self.ab_mt_b0_combo):
            combo.setToolTip("Fitted map from this study used as an AB-MT prior observable.")
            combo.currentIndexChanged.connect(lambda _idx: self.refresh_fit_breakout_summaries())
        self.ab_mt_t1_label = QtWidgets.QLabel("AB-MT T1 map")
        self.ab_mt_b1_label = QtWidgets.QLabel("AB-MT B1 map")
        self.ab_mt_b0_label = QtWidgets.QLabel("AB-MT B0 map")
        inputs_form.addRow(self.ab_mt_t1_label, self.ab_mt_t1_combo)
        inputs_form.addRow(self.ab_mt_b1_label, self.ab_mt_b1_combo)
        inputs_form.addRow(self.ab_mt_b0_label, self.ab_mt_b0_combo)
        inputs_layout.addLayout(inputs_form)
        inputs_layout.addStretch(1)
        inputs_close = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        inputs_close.rejected.connect(self.fit_inputs_dialog.hide)
        inputs_layout.addWidget(inputs_close)
        self._set_fit_dependency_controls_visible(False, False)

        self.fit_parameters_dialog = QtWidgets.QDialog(self)
        self.fit_parameters_dialog.setWindowTitle("Model Parameters")
        self.fit_parameters_dialog.resize(self._scaled_px(980), self._scaled_px(620))
        advanced_layout = QtWidgets.QVBoxLayout(self.fit_parameters_dialog)
        self.pool_list = QtWidgets.QListWidget()
        self.pool_list.setMaximumHeight(92)
        self.pool_list.itemChanged.connect(lambda _item: self.refresh_parameter_table())
        self.pool_list.setVisible(False)
        advanced_layout.addWidget(self.pool_list)
        self.custom_pool_widget = QtWidgets.QWidget()
        custom_pool_layout = QtWidgets.QHBoxLayout(self.custom_pool_widget)
        custom_pool_layout.setContentsMargins(0, 0, 0, 0)
        self.custom_pool_edit = QtWidgets.QLineEdit()
        self.custom_pool_edit.setPlaceholderText("extra AB-MT pool name")
        add_custom_pool = QtWidgets.QPushButton("Add Pool")
        add_custom_pool.clicked.connect(self.add_custom_pool)
        remove_custom_pool = QtWidgets.QPushButton("Remove Pool")
        remove_custom_pool.clicked.connect(self.remove_selected_pool)
        custom_pool_layout.addWidget(self.custom_pool_edit, 1)
        custom_pool_layout.addWidget(add_custom_pool)
        custom_pool_layout.addWidget(remove_custom_pool)
        self.custom_pool_widget.setVisible(False)
        advanced_layout.addWidget(self.custom_pool_widget)
        self.param_table = QtWidgets.QTableWidget()
        self.param_table.setColumnCount(10)
        self.param_table.setHorizontalHeaderLabels(
            [
                "Parameter",
                "Initial",
                "Lower",
                "Upper",
                "Unit",
                "Reg type",
                "Reg weight",
                "Reg axes",
                "Force constant dims",
                "Fixed",
            ]
        )
        advanced_layout.addWidget(self.param_table, 1)
        reset_params = QtWidgets.QPushButton("Reset Model Defaults")
        reset_params.clicked.connect(self.refresh_advanced_fit_controls)
        advanced_layout.addWidget(reset_params)
        parameters_close = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        parameters_close.rejected.connect(self.fit_parameters_dialog.hide)
        advanced_layout.addWidget(parameters_close)

        self.fit_frames_dialog = QtWidgets.QDialog(self)
        self.fit_frames_dialog.setWindowTitle("Frame Inclusion")
        self.fit_frames_dialog.resize(self._scaled_px(440), self._scaled_px(520))
        frames_layout = QtWidgets.QVBoxLayout(self.fit_frames_dialog)
        self.frame_mask_list = QtWidgets.QListWidget()
        self.frame_mask_list.itemChanged.connect(self.on_frame_mask_changed)
        frames_layout.addWidget(self.frame_mask_list, 1)
        frames_close = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        frames_close.rejected.connect(self.fit_frames_dialog.hide)
        frames_layout.addWidget(frames_close)

        buttons = QtWidgets.QGridLayout()
        sanity = QtWidgets.QPushButton("Fit Sanity Check")
        sanity.clicked.connect(self.run_fit_sanity_check)
        preview = QtWidgets.QPushButton("Preview ROI Fit")
        preview.clicked.connect(self.run_preview_fit)
        full = QtWidgets.QPushButton("Run Full-Map Fit")
        full.clicked.connect(self.run_full_fit)
        save = QtWidgets.QPushButton("Save Fit")
        save.clicked.connect(self.save_fit_dialog)
        buttons.addWidget(sanity, 0, 0, 1, 2)
        buttons.addWidget(preview, 1, 0)
        buttons.addWidget(full, 1, 1)
        buttons.addWidget(save, 2, 0, 1, 2)
        layout.addLayout(buttons)
        results = QtWidgets.QGroupBox("Fit results")
        results_layout = QtWidgets.QVBoxLayout(results)
        self.fit_results_text = QtWidgets.QTextEdit()
        self.fit_results_text.setReadOnly(True)
        self.fit_results_text.setMinimumHeight(110)
        self.fit_results_text.setPlaceholderText("Run a full-map fit to summarize fitted maps and losses.")
        results_layout.addWidget(self.fit_results_text)
        layout.addWidget(results, 1)
        layout.addStretch(1)
        return widget

    def _build_registration_tab(self):
        widget = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(widget)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(content)
        layout.setSpacing(8)

        source_box = QtWidgets.QGroupBox("Image Selection")
        source_form = QtWidgets.QFormLayout(source_box)
        self.reg_reference_experiment_combo = QtWidgets.QComboBox()
        self.reg_reference_source_combo = QtWidgets.QComboBox()
        self.reg_reference_map_combo = QtWidgets.QComboBox()
        self.reg_reference_frame_spin = QtWidgets.QSpinBox()
        self.reg_current_moving_label = QtWidgets.QLabel("Current view")
        self.reg_current_moving_label.setWordWrap(True)
        self.reg_reference_source_combo.addItem("Image", "image")
        self.reg_reference_source_combo.addItem("MRax reconstruction", "reconstruction")
        self.reg_reference_source_combo.addItem("Fitted map", "map")
        self.reg_reference_frame_spin.setRange(0, 999999)
        self.reg_reference_frame_spin.setValue(0)
        self.reg_reference_frame_spin.setPrefix("frame ")
        source_form.addRow("Reference experiment", self.reg_reference_experiment_combo)
        source_form.addRow("Reference image", self.reg_reference_source_combo)
        source_form.addRow("Reference map", self.reg_reference_map_combo)
        source_form.addRow("Reference frame", self.reg_reference_frame_spin)
        source_form.addRow("Moving image", self.reg_current_moving_label)
        layout.addWidget(source_box)

        option_box = QtWidgets.QGroupBox("SimpleITK Options")
        option_form = QtWidgets.QFormLayout(option_box)
        choices = registration_choice_sets()
        self.reg_transform_combo = QtWidgets.QComboBox()
        self.reg_metric_combo = QtWidgets.QComboBox()
        self.reg_optimizer_combo = QtWidgets.QComboBox()
        self.reg_interpolator_combo = QtWidgets.QComboBox()
        self.reg_initializer_combo = QtWidgets.QComboBox()
        self.reg_sampling_combo = QtWidgets.QComboBox()
        for key, combo in (
            ("transform", self.reg_transform_combo),
            ("metric", self.reg_metric_combo),
            ("optimizer", self.reg_optimizer_combo),
            ("interpolator", self.reg_interpolator_combo),
            ("initializer", self.reg_initializer_combo),
            ("sampling_strategy", self.reg_sampling_combo),
        ):
            for value in choices[key]:
                combo.addItem(value.replace("_", " "), value)
        self.reg_transform_combo.setCurrentText("rigid")
        self.reg_metric_combo.setCurrentText("mattes mi")
        self.reg_optimizer_combo.setCurrentText("gradient descent")
        self.reg_interpolator_combo.setCurrentText("linear")
        self.reg_initializer_combo.setCurrentText("geometry")
        self.reg_sampling_combo.setCurrentText("random")

        self.reg_sampling_percentage_spin = QtWidgets.QDoubleSpinBox()
        self.reg_sampling_percentage_spin.setRange(0.0, 1.0)
        self.reg_sampling_percentage_spin.setDecimals(3)
        self.reg_sampling_percentage_spin.setSingleStep(0.05)
        self.reg_sampling_percentage_spin.setValue(0.2)
        self.reg_histogram_bins_spin = QtWidgets.QSpinBox()
        self.reg_histogram_bins_spin.setRange(2, 4096)
        self.reg_histogram_bins_spin.setValue(50)
        self.reg_learning_rate_spin = QtWidgets.QDoubleSpinBox()
        self.reg_learning_rate_spin.setDecimals(6)
        self.reg_learning_rate_spin.setRange(1e-9, 1e9)
        self.reg_learning_rate_spin.setValue(1.0)
        self.reg_min_step_spin = QtWidgets.QDoubleSpinBox()
        self.reg_min_step_spin.setDecimals(8)
        self.reg_min_step_spin.setRange(1e-12, 1e6)
        self.reg_min_step_spin.setValue(1e-4)
        self.reg_iterations_spin = QtWidgets.QSpinBox()
        self.reg_iterations_spin.setRange(0, 1_000_000)
        self.reg_iterations_spin.setValue(200)
        self.reg_convergence_spin = QtWidgets.QDoubleSpinBox()
        self.reg_convergence_spin.setDecimals(10)
        self.reg_convergence_spin.setRange(0.0, 1e6)
        self.reg_convergence_spin.setValue(1e-6)
        self.reg_convergence_window_spin = QtWidgets.QSpinBox()
        self.reg_convergence_window_spin.setRange(1, 10000)
        self.reg_convergence_window_spin.setValue(10)
        self.reg_shrink_edit = QtWidgets.QLineEdit("2,1")
        self.reg_smoothing_edit = QtWidgets.QLineEdit("1,0")
        option_form.addRow("Transform", self.reg_transform_combo)
        option_form.addRow("Metric", self.reg_metric_combo)
        option_form.addRow("Optimizer", self.reg_optimizer_combo)
        option_form.addRow("Interpolator", self.reg_interpolator_combo)
        option_form.addRow("Initializer", self.reg_initializer_combo)
        option_form.addRow("Sampling", self.reg_sampling_combo)
        option_form.addRow("Sampling fraction", self.reg_sampling_percentage_spin)
        option_form.addRow("Histogram bins", self.reg_histogram_bins_spin)
        option_form.addRow("Learning rate", self.reg_learning_rate_spin)
        option_form.addRow("Min step", self.reg_min_step_spin)
        option_form.addRow("Iterations", self.reg_iterations_spin)
        option_form.addRow("Convergence", self.reg_convergence_spin)
        option_form.addRow("Convergence window", self.reg_convergence_window_spin)
        option_form.addRow("Shrink levels", self.reg_shrink_edit)
        option_form.addRow("Smoothing sigmas", self.reg_smoothing_edit)
        layout.addWidget(option_box)

        metric_mask_box = QtWidgets.QGroupBox("Registration Masks")
        metric_mask_form = QtWidgets.QFormLayout(metric_mask_box)
        self.reg_reference_metric_mask_combo = QtWidgets.QComboBox()
        self.reg_moving_metric_mask_combo = QtWidgets.QComboBox()
        metric_mask_form.addRow("Reference metric mask", self.reg_reference_metric_mask_combo)
        metric_mask_form.addRow("Moving metric mask", self.reg_moving_metric_mask_combo)
        layout.addWidget(metric_mask_box)

        overlay_box = QtWidgets.QGroupBox("Reference Overlay")
        overlay_form = QtWidgets.QFormLayout(overlay_box)
        self.reg_reference_channel_combo = QtWidgets.QComboBox()
        self.reg_moving_channel_combo = QtWidgets.QComboBox()
        for combo in (self.reg_reference_channel_combo, self.reg_moving_channel_combo):
            for value in choices["channel"]:
                combo.addItem(value, value)
            combo.currentIndexChanged.connect(self.on_registration_overlay_controls_changed)
        self.reg_reference_channel_combo.setCurrentText("red")
        self.reg_moving_channel_combo.setCurrentText("green")
        self.reg_reference_alpha_spin = QtWidgets.QDoubleSpinBox()
        self.reg_moving_alpha_spin = QtWidgets.QDoubleSpinBox()
        for spin in (self.reg_reference_alpha_spin, self.reg_moving_alpha_spin):
            spin.setRange(0.0, 1.0)
            spin.setDecimals(2)
            spin.setSingleStep(0.05)
            spin.setValue(1.0)
        self.reg_reference_overlay_check = QtWidgets.QCheckBox("Overlay reference background")
        self.reg_reference_overlay_check.toggled.connect(self.on_registration_overlay_controls_changed)
        self.reg_overlay_blend_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.reg_overlay_blend_slider.setRange(0, 100)
        self.reg_overlay_blend_slider.setValue(60)
        self.reg_overlay_blend_slider.valueChanged.connect(self.on_registration_overlay_controls_changed)
        self.reg_overlay_blend_label = QtWidgets.QLabel("")
        overlay_form.addRow(self.reg_reference_overlay_check)
        overlay_form.addRow("Reference channel", self.reg_reference_channel_combo)
        overlay_form.addRow("Current channel", self.reg_moving_channel_combo)
        overlay_form.addRow("Blend", self.reg_overlay_blend_slider)
        overlay_form.addRow("", self.reg_overlay_blend_label)
        self._refresh_registration_overlay_blend_label()
        layout.addWidget(overlay_box)

        run_box = QtWidgets.QGroupBox("Run And Diagnose")
        run_layout = QtWidgets.QGridLayout(run_box)
        run_btn = QtWidgets.QPushButton("Run SimpleITK Registration")
        run_btn.clicked.connect(self.run_simpleitk_registration)
        preview_btn = QtWidgets.QPushButton("Preview Stored Transform")
        preview_btn.clicked.connect(self.preview_selected_registration_transform)
        view_transform_btn = QtWidgets.QPushButton("Use Transform In View State")
        view_transform_btn.clicked.connect(self.use_selected_registration_transform_in_view_state)
        view_overlay_btn = QtWidgets.QPushButton("Use Transform + Show Overlay")
        view_overlay_btn.clicked.connect(self.use_selected_registration_overlay_in_view_state)
        run_layout.addWidget(run_btn, 0, 0)
        run_layout.addWidget(preview_btn, 0, 1)
        run_layout.addWidget(view_transform_btn, 1, 0)
        run_layout.addWidget(view_overlay_btn, 1, 1)
        layout.addWidget(run_box)

        manual_box = QtWidgets.QGroupBox("Manual Alignment")
        manual_form = QtWidgets.QFormLayout(manual_box)
        self.reg_shift_x_spin = QtWidgets.QDoubleSpinBox()
        self.reg_shift_y_spin = QtWidgets.QDoubleSpinBox()
        for spin in (self.reg_shift_x_spin, self.reg_shift_y_spin):
            spin.setRange(-100000.0, 100000.0)
            spin.setDecimals(2)
            spin.setSingleStep(1.0)
            spin.setSuffix(" px")
        self.reg_rotation_spin = QtWidgets.QDoubleSpinBox()
        self.reg_rotation_spin.setRange(-360.0, 360.0)
        self.reg_rotation_spin.setDecimals(3)
        self.reg_rotation_spin.setSingleStep(1.0)
        self.reg_rotation_spin.setSuffix(" deg")
        self.reg_scale_x_spin = QtWidgets.QDoubleSpinBox()
        self.reg_scale_y_spin = QtWidgets.QDoubleSpinBox()
        for spin in (self.reg_scale_x_spin, self.reg_scale_y_spin):
            spin.setRange(0.01, 100.0)
            spin.setDecimals(4)
            spin.setSingleStep(0.01)
            spin.setValue(1.0)
        manual_form.addRow("Shift X", self.reg_shift_x_spin)
        manual_form.addRow("Shift Y", self.reg_shift_y_spin)
        manual_form.addRow("Rotation", self.reg_rotation_spin)
        manual_form.addRow("Stretch X", self.reg_scale_x_spin)
        manual_form.addRow("Stretch Y", self.reg_scale_y_spin)
        manual_buttons = QtWidgets.QHBoxLayout()
        manual_preview = QtWidgets.QPushButton("Preview Manual")
        manual_preview.clicked.connect(self.preview_manual_registration)
        manual_store = QtWidgets.QPushButton("Store Manual Transform")
        manual_store.clicked.connect(self.store_manual_registration)
        manual_reset = QtWidgets.QPushButton("Reset Manual")
        manual_reset.clicked.connect(self.reset_manual_registration_controls)
        manual_buttons.addWidget(manual_preview)
        manual_buttons.addWidget(manual_store)
        manual_buttons.addWidget(manual_reset)
        manual_form.addRow(manual_buttons)
        layout.addWidget(manual_box)

        drift_box = QtWidgets.QGroupBox("MRax Linear Drift Correction")
        drift_form = QtWidgets.QFormLayout(drift_box)
        drift_choices = drift_choice_sets()
        self.reg_drift_transform_combo = QtWidgets.QComboBox()
        self.reg_drift_metric_combo = QtWidgets.QComboBox()
        self.reg_drift_reference_combo = QtWidgets.QComboBox()
        self.reg_drift_interpolation_combo = QtWidgets.QComboBox()
        self.reg_drift_optimizer_combo = QtWidgets.QComboBox()
        for value in drift_choices["transform"]:
            self.reg_drift_transform_combo.addItem(value.replace("_", " "), value)
        for value in drift_choices["metric"]:
            self.reg_drift_metric_combo.addItem(value.replace("_", " "), value)
        for value in drift_choices["reference"]:
            self.reg_drift_reference_combo.addItem(value.replace("_", " "), value)
        for value in drift_choices["interpolation"]:
            self.reg_drift_interpolation_combo.addItem(value.replace("_", " "), value)
        for value in drift_choices["optimizer"]:
            self.reg_drift_optimizer_combo.addItem(value.replace("_", " "), value)
        self.reg_drift_dimensions_edit = QtWidgets.QLineEdit("")
        self.reg_drift_step_size_spin = QtWidgets.QDoubleSpinBox()
        self.reg_drift_step_size_spin.setDecimals(6)
        self.reg_drift_step_size_spin.setRange(1e-9, 1e6)
        self.reg_drift_step_size_spin.setValue(0.05)
        self.reg_drift_steps_spin = QtWidgets.QSpinBox()
        self.reg_drift_steps_spin.setRange(0, 1_000_000)
        self.reg_drift_steps_spin.setValue(200)
        self.reg_drift_fit_intercept_check = QtWidgets.QCheckBox()
        self.reg_drift_mask_combo = QtWidgets.QComboBox()
        drift_form.addRow("Transform", self.reg_drift_transform_combo)
        drift_form.addRow("Metric", self.reg_drift_metric_combo)
        drift_form.addRow("Reference", self.reg_drift_reference_combo)
        drift_form.addRow("Dimensions", self.reg_drift_dimensions_edit)
        drift_form.addRow("Interpolation", self.reg_drift_interpolation_combo)
        drift_form.addRow("Optimizer", self.reg_drift_optimizer_combo)
        drift_form.addRow("Step size", self.reg_drift_step_size_spin)
        drift_form.addRow("Steps", self.reg_drift_steps_spin)
        drift_form.addRow("Fit intercept", self.reg_drift_fit_intercept_check)
        drift_form.addRow("Mask", self.reg_drift_mask_combo)
        drift_buttons = QtWidgets.QHBoxLayout()
        drift_run = QtWidgets.QPushButton("Run MRax Drift Correction")
        drift_run.clicked.connect(self.run_mrax_drift_correction)
        drift_buttons.addWidget(drift_run)
        drift_form.addRow(drift_buttons)
        layout.addWidget(drift_box)

        mask_box = QtWidgets.QGroupBox("Mask Transform Preview")
        mask_form = QtWidgets.QFormLayout(mask_box)
        self.reg_available_transform_combo = QtWidgets.QComboBox()
        self.reg_mask_roi_combo = QtWidgets.QComboBox()
        self.reg_mask_direction_combo = QtWidgets.QComboBox()
        self.reg_mask_direction_combo.addItem("Reference ROI to moving grid", "reference_to_moving")
        self.reg_mask_direction_combo.addItem("Moving ROI to reference grid", "moving_to_reference")
        mask_form.addRow("Available transform", self.reg_available_transform_combo)
        mask_form.addRow("ROI", self.reg_mask_roi_combo)
        mask_form.addRow("Direction", self.reg_mask_direction_combo)
        mask_buttons = QtWidgets.QHBoxLayout()
        mask_preview = QtWidgets.QPushButton("Preview Mask Transform")
        mask_preview.clicked.connect(self.preview_registration_mask_transform)
        mask_add = QtWidgets.QPushButton("Add Transformed ROI")
        mask_add.clicked.connect(self.add_transformed_registration_roi)
        mask_buttons.addWidget(mask_preview)
        mask_buttons.addWidget(mask_add)
        mask_form.addRow(mask_buttons)
        layout.addWidget(mask_box)

        self.registration_diagnostics_text = QtWidgets.QTextEdit()
        self.registration_diagnostics_text.setReadOnly(True)
        self.registration_diagnostics_text.setMinimumHeight(self._scaled_px(140))
        layout.addWidget(self.registration_diagnostics_text, 1)
        layout.addStretch(1)
        scroll.setWidget(content)
        outer.addWidget(scroll)

        for combo in (
            self.reg_reference_experiment_combo,
            self.reg_reference_source_combo,
        ):
            combo.currentIndexChanged.connect(self.refresh_registration_tab)
        return widget

    def _build_diagnostics_tab(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        self.diagnostics_text = QtWidgets.QTextEdit()
        self.diagnostics_text.setReadOnly(True)
        layout.addWidget(self.diagnostics_text)
        actions = QtWidgets.QHBoxLayout()
        refresh = QtWidgets.QPushButton("Refresh Diagnostics")
        refresh.clicked.connect(self.update_diagnostics)
        report = QtWidgets.QPushButton("Save Diagnostic Report")
        report.clicked.connect(self.save_diagnostic_report_dialog)
        clear = QtWidgets.QPushButton("Clear Event Log")
        clear.clicked.connect(self.clear_diagnostic_events)
        actions.addWidget(refresh)
        actions.addWidget(report)
        actions.addWidget(clear)
        actions.addStretch(1)
        layout.addLayout(actions)
        return widget

    def _apply_theme(self):
        font_px = self._scaled_px(12)
        button_height = self._scaled_px(28)
        pad_y = self._scaled_px(4)
        pad_x = self._scaled_px(8)
        radius = self._scaled_px(4)
        progress_height = self._scaled_px(16)
        if self.progress_bar is not None:
            self.progress_bar.setFixedWidth(self._scaled_px(220))
        self.setStyleSheet(
            f"""
            QWidget {{ font-family: "DejaVu Sans", "Sans Serif"; font-size: {font_px}px; }}
            QPushButton {{ min-height: {button_height}px; padding: {pad_y}px {pad_x}px; }}
            QPushButton:checked {{
                background: #dbeafe;
                border: 1px solid #60a5fa;
                color: #111827;
            }}
            QPushButton:disabled, QComboBox:disabled, QLineEdit:disabled,
            QListWidget:disabled, QTableWidget:disabled, QTextEdit:disabled {{
                color: #7c8797;
                background: #eef1f5;
                border-color: #d5dae2;
            }}
            QListWidget, QTextEdit, QTableWidget, QLineEdit, QComboBox {{
                border: 1px solid #c8cdd4;
                border-radius: {radius}px;
                background: #ffffff;
            }}
            QProgressBar {{
                border: 1px solid #aeb7c4;
                border-radius: {radius}px;
                text-align: center;
                min-height: {progress_height}px;
                background: #f8fafc;
            }}
            QProgressBar::chunk {{
                background: #2563eb;
                border-radius: {max(1, radius - 1)}px;
            }}
            QTabWidget::pane {{ border: 1px solid #c8cdd4; }}
            QLabel {{ color: #27313f; }}
            QAbstractItemView::item:selected {{
                background: #dbeafe;
                color: #111827;
            }}
            QAbstractItemView::item:hover {{
                background: #eff6ff;
                color: #111827;
            }}
            QComboBox QAbstractItemView {{
                selection-background-color: #dbeafe;
                selection-color: #111827;
            }}
            QTableWidget::item:selected {{
                background: #dbeafe;
                color: #111827;
            }}
            """
        )

    def open_manifest_dialog(self):
        path, _ = self._get_open_file_name(
            "Load MRax study",
            str(Path.cwd()),
            "MRax manifest (study_manifest.json);;JSON (*.json);;All files (*)",
        )
        if path:
            self.load_manifest(Path(path))

    def load_dialog(self):
        if self.is_busy:
            return
        path, _ = self._get_open_file_name(
            "Load MRax file",
            str(Path.cwd()),
            "MRax files (*.mrax *.json *.npz *.npy);;MRax bundles (*.mrax);;JSON (*.json);;NumPy (*.npz *.npy);;All files (*)",
        )
        if not path:
            return
        load_path = Path(path)
        self.run_guarded_task(
            lambda progress: self._load_path_for_task(load_path, progress),
            self.on_load_finished,
            f"Loading {load_path.name}...",
            error_title="Could not load file",
        )

    def load_path_autodetect(self, path: Path):
        result = self._load_path_data(path, lambda _current, _total=100, _detail="": None)
        self.populate_from_session()
        if result.get("kind") in {"roi", "mask"}:
            self.mark_dirty("ROI set changed")
        else:
            self.clear_dirty()
        self.statusBar().showMessage(str(result.get("message") or f"Loaded {path}"))
        return result

    def _load_path_for_task(self, path: Path, progress: Callable[[int, int, str], None]):
        return self._load_path_data(path, progress)

    def _load_path_data(self, path: Path, progress: Callable[[int, int, str], None]):
        path = Path(path)
        progress(0, 4, "Detecting file type")
        kind = _detect_load_kind(path)
        progress(1, 4, "Loading data")
        if kind == "bundle":
            progress(1, 4, "Extracting MRax bundle")
            self._load_mrax_bundle_data(path)
            message = f"Loaded MRax bundle from {path}"
        elif kind == "bundle_folder":
            self._load_mrax_package_data(path if path.is_dir() else path.parent)
            message = f"Loaded MRax package folder from {path}"
        elif kind == "gui_state":
            self._load_gui_state_data(path)
            message = f"Loaded GUI state from {path}"
        elif kind == "manifest":
            self._load_manifest_data(path)
            message = f"Loaded {path}"
        elif kind == "roi":
            self._load_roi_data(path)
            message = f"Loaded ROI set from {path}"
        elif kind == "mask":
            self._load_roi_data(path, raw_mask=True)
            message = f"Loaded raw mask from {path}"
        else:
            raise ValueError(f"Could not identify MRax file type for {path}")
        progress(4, 4, "Load complete")
        return {"path": path, "kind": kind, "message": message}

    def on_load_finished(self, result: dict[str, Any]):
        self.populate_from_session()
        if result.get("kind") in {"roi", "mask"}:
            self.mark_dirty("ROI set changed")
        else:
            self.clear_dirty()
        self.statusBar().showMessage(str(result.get("message") or "Load complete."))

    def import_study_dialog(self):
        if self.is_busy:
            return
        directory = self._get_existing_directory(
            "Choose study folder to import",
            str(Path.cwd()),
        )
        if not directory:
            return
        root = Path(directory)
        vendor = _detect_import_vendor(root)
        if vendor == "DICOM":
            self.run_guarded_task(
                lambda progress: self._load_dicom(root, progress),
                self.on_dicom_import_finished,
                "Importing DICOM folder into MRax structures...",
                error_title="Could not import DICOM folder",
            )
            return
        if vendor != "Bruker":
            self.show_error(
                "Unsupported import folder",
                "The selected folder does not contain a recognizable Bruker study or DICOM image series.",
            )
            return
        options = self._show_import_options_dialog(root, vendor)
        if options is None:
            return
        self.run_guarded_task(
            lambda progress: self._load_bruker(root, options, progress),
            self.on_bruker_import_finished,
            "Importing Bruker study into MRax structures...",
            error_title="Could not import Bruker study",
        )

    def _show_import_options_dialog(self, root: Path, vendor: str):
        summary = _detect_bruker_import_summary(root) if vendor == "Bruker" else []
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Import Study")
        dialog.resize(960, 620)
        layout = QtWidgets.QVBoxLayout(dialog)

        form = QtWidgets.QFormLayout()
        vendor_combo = QtWidgets.QComboBox()
        vendor_combo.addItems(["Bruker"])
        vendor_combo.setCurrentText(vendor if vendor == "Bruker" else "Bruker")
        form.addRow("Vendor", vendor_combo)

        field_spin = QtWidgets.QDoubleSpinBox()
        field_spin.setDecimals(6)
        field_spin.setRange(0.0, 2000.0)
        field_spin.setSpecialValueText("unset")
        detected_field = next((row.get("field_strength_mhz") for row in summary if row.get("field_strength_mhz")), None)
        if detected_field:
            field_spin.setValue(float(detected_field))
        form.addRow("Field strength MHz", field_spin)

        pdata_edit = QtWidgets.QLineEdit("1")
        form.addRow("pdata index", pdata_edit)
        raw_kspace_check = QtWidgets.QCheckBox()
        raw_kspace_check.setChecked(False)
        form.addRow("Load/reconstruct raw k-space now", raw_kspace_check)
        prefer_raw_check = QtWidgets.QCheckBox()
        prefer_raw_check.setChecked(False)
        prefer_raw_check.setEnabled(False)
        form.addRow("Prefer raw image", prefer_raw_check)
        raw_kspace_check.toggled.connect(prefer_raw_check.setEnabled)
        layout.addLayout(form)

        layout.addWidget(QtWidgets.QLabel("Detected scans"))
        table = QtWidgets.QTableWidget()
        table.setColumnCount(7)
        table.setHorizontalHeaderLabels(
            ["scan", "method", "protocol", "frames", "detected type", "exp type", "model"]
        )
        table.setRowCount(len(summary))
        type_widgets: dict[str, Any] = {}
        model_widgets: dict[str, Any] = {}
        model_keys = [(choice.key, choice.label) for choice in model_choices()]
        for row_idx, row in enumerate(summary):
            scan = str(row.get("scan") or "")
            detected_type = str(row.get("detected_type") or "anatomical")
            detected_model = normalize_model_key(row.get("detected_model") or detected_type) or "identity"
            values = [
                scan,
                str(row.get("method") or ""),
                str(row.get("protocol") or ""),
                str(row.get("frame_count") or ""),
                detected_type,
            ]
            for col, value in enumerate(values):
                table.setItem(row_idx, col, QtWidgets.QTableWidgetItem(value))
            type_combo = QtWidgets.QComboBox()
            type_combo.setEditable(True)
            type_combo.addItems(_known_exp_types(detected_type))
            type_combo.setCurrentText(detected_type)
            table.setCellWidget(row_idx, 5, type_combo)
            type_widgets[scan] = type_combo
            model_combo = QtWidgets.QComboBox()
            model_combo.addItem("auto", "")
            for key, label in model_keys:
                model_combo.addItem(label, key)
            for idx in range(model_combo.count()):
                if model_combo.itemData(idx) == detected_model:
                    model_combo.setCurrentIndex(idx)
                    break
            type_combo.currentTextChanged.connect(
                lambda text, combo=model_combo: _set_model_combo_from_type(combo, text)
            )
            table.setCellWidget(row_idx, 6, model_combo)
            model_widgets[scan] = model_combo
        table.resizeColumnsToContents()
        layout.addWidget(table, 1)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if self._exec_dialog(dialog) != QtWidgets.QDialog.Accepted:
            return None

        field_value = float(field_spin.value())
        return {
            "vendor": vendor_combo.currentText(),
            "field_strength_mhz": None if field_value <= 0 else field_value,
            "pdata_index": pdata_edit.text().strip() or "1",
            "load_raw_kspace": raw_kspace_check.isChecked(),
            "prefer_raw_image": raw_kspace_check.isChecked() and prefer_raw_check.isChecked(),
            "slice_crop_mode": None,
            "type_overrides": {
                scan: widget.currentText().strip()
                for scan, widget in type_widgets.items()
                if widget.currentText().strip()
            },
            "model_overrides_by_name": {
                scan: str(widget.currentData())
                for scan, widget in model_widgets.items()
                if widget.currentData()
            },
        }

    def _load_bruker(
        self,
        path: Path,
        options: dict[str, Any],
        progress: Callable[[int, int, str], None] | None = None,
    ):
        if progress is not None:
            progress(0, 1, "Preparing Bruker import")
        session = GuiSession()
        session.load_bruker_study(
            path,
            field_strength_mhz=options.get("field_strength_mhz"),
            pdata_index=options.get("pdata_index") or "1",
            load_raw_kspace=bool(options.get("load_raw_kspace", False)),
            prefer_raw_image=bool(options.get("prefer_raw_image", False)),
            slice_crop_mode=options.get("slice_crop_mode"),
            type_overrides=options.get("type_overrides") or {},
            model_overrides_by_name=options.get("model_overrides_by_name") or {},
            progress_callback=progress,
        )
        self._replace_or_add_loaded_workspace(session, self._workspace_label_for_session(session, path.name))
        return path

    def on_bruker_import_finished(self, path: Path):
        self._fit_results_by_experiment.clear()
        self._preview_results_by_experiment.clear()
        self.registration_results.clear()
        self.drift_results.clear()
        self._visual_undo_stacks.clear()
        self.last_fit_result = None
        self.last_preview_fit_result = None
        self.populate_from_session()
        self.mark_dirty("imported Bruker study")
        issue_count = len((self.session.import_metadata or {}).get("import_issues") or [])
        issue_text = f" {issue_count} scan(s) were skipped; see summary." if issue_count else ""
        if issue_count:
            self.record_diagnostic_event(
                "warning",
                f"Bruker import skipped {issue_count} scan(s).",
                detail=json.dumps((self.session.import_metadata or {}).get("import_issues") or [], indent=2),
                context={"source": str(path), "operation": "Bruker import"},
                update_status=False,
            )
        self.statusBar().showMessage(
            f"Imported Bruker study from {path}. Source files were not modified.{issue_text}"
        )

    def _load_dicom(
        self,
        path: Path,
        progress: Callable[[int, int, str], None] | None = None,
    ):
        session = GuiSession()
        session.load_dicom_study(path, progress_callback=progress)
        self._replace_or_add_loaded_workspace(session, self._workspace_label_for_session(session, path.name))
        return path

    def on_dicom_import_finished(self, path: Path):
        self._fit_results_by_experiment.clear()
        self._preview_results_by_experiment.clear()
        self.registration_results.clear()
        self.drift_results.clear()
        self._visual_undo_stacks.clear()
        self.last_fit_result = None
        self.last_preview_fit_result = None
        self.populate_from_session()
        self.mark_dirty("imported DICOM study")
        count = len(self.session.experiment_names)
        self.statusBar().showMessage(
            f"Imported {count} DICOM experiment(s) from {path}. Source files were not modified."
        )

    def save_imported_study_dialog(self):
        if self.session.study is None:
            self.statusBar().showMessage("No imported in-memory study is available to save.")
            return
        directory = self._get_existing_directory(
            "Choose MRax output folder",
            str(Path.cwd()),
        )
        if not directory:
            return
        try:
            manifest = self.session.save_imported_study(directory)
        except Exception:
            self.show_error("Could not save imported study", traceback.format_exc())
            return
        self.statusBar().showMessage(f"Saved imported study as MRax manifest: {manifest}")

    def _text_input_dialog(self, title: str, label: str, default: str = "") -> str | None:
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(title)
        layout = QtWidgets.QVBoxLayout(dialog)
        form = QtWidgets.QFormLayout()
        edit = QtWidgets.QLineEdit(default)
        form.addRow(label, edit)
        layout.addLayout(form)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if self._exec_dialog(dialog) != QtWidgets.QDialog.Accepted:
            return None
        return edit.text().strip()

    def new_empty_study_dialog(self):
        if self.is_busy:
            return
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("New Empty Study")
        layout = QtWidgets.QVBoxLayout(dialog)
        form = QtWidgets.QFormLayout()
        name_edit = QtWidgets.QLineEdit("Untitled Study")
        field_spin = QtWidgets.QDoubleSpinBox()
        field_spin.setDecimals(6)
        field_spin.setRange(0.0, 2000.0)
        field_spin.setSpecialValueText("unset")
        form.addRow("Study name", name_edit)
        form.addRow("Field strength MHz", field_spin)
        layout.addLayout(form)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if self._exec_dialog(dialog) != QtWidgets.QDialog.Accepted:
            return
        session = GuiSession()
        field_value = float(field_spin.value())
        session.create_empty_study(
            name_edit.text().strip() or "Untitled Study",
            field_strength_mhz=None if field_value <= 0.0 else field_value,
        )
        self._replace_or_add_loaded_workspace(session, session.study_name)
        self.mark_dirty("new empty study")
        self.populate_from_session()
        self.statusBar().showMessage(f"Created empty study: {session.study_name}")

    def rename_study_dialog(self):
        current = self.session.study_name if self.session.data else "Untitled Study"
        name = self._text_input_dialog("Rename Study", "Study name", current)
        if name is None:
            return
        try:
            self.session.rename_study(name)
        except Exception:
            self.show_error("Could not rename study", traceback.format_exc())
            return
        if self._active_study_key in self._study_workspaces:
            self._study_workspaces[self._active_study_key].label = self.session.study_name
        self.mark_dirty("study renamed")
        self.populate_from_session()
        self.statusBar().showMessage(f"Renamed study to {self.session.study_name}.")

    def rename_experiment_dialog(self):
        if not self.session.is_loaded:
            self.statusBar().showMessage("Select an experiment to rename.")
            return
        old = str(self.session.selected_experiment or "")
        name = self._text_input_dialog("Rename Experiment", "Experiment name", old)
        if name is None:
            return
        try:
            new = self.session.rename_experiment(old, name)
            self._rename_experiment_sidecars(old, new)
        except Exception:
            self.show_error("Could not rename experiment", traceback.format_exc())
            return
        self.mark_dirty("experiment renamed")
        self.populate_from_session()
        self.statusBar().showMessage(f"Renamed experiment {old} to {new}.")

    def _rename_experiment_sidecars(self, old: str, new: str) -> None:
        for mapping in (self._fit_results_by_experiment, self._preview_results_by_experiment, self._visual_undo_stacks):
            if old in mapping and new not in mapping:
                mapping[new] = mapping.pop(old)
        for result in self.drift_results.values():
            metadata = result.get("metadata")
            if isinstance(metadata, MutableMapping) and metadata.get("experiment") == old:
                metadata["experiment"] = new
        for result in self.registration_results.values():
            metadata = result.get("metadata")
            if not isinstance(metadata, MutableMapping):
                continue
            for key in ("moving_experiment", "reference_experiment"):
                if metadata.get(key) == old:
                    metadata[key] = new

    def _make_import_preview_panel(
        self,
        sections: Sequence[tuple[str, str, int]],
    ) -> tuple[object, dict[str, object]]:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        fields: dict[str, object] = {}
        for key, title, height in sections:
            box = QtWidgets.QGroupBox(title)
            box_layout = QtWidgets.QVBoxLayout(box)
            box_layout.setContentsMargins(8, 8, 8, 8)
            edit = QtWidgets.QPlainTextEdit()
            edit.setReadOnly(True)
            wrap_mode = getattr(QtWidgets.QPlainTextEdit, "LineWrapMode", QtWidgets.QPlainTextEdit)
            edit.setLineWrapMode(wrap_mode.NoWrap)
            try:
                edit.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
            except Exception:
                _LOG.debug("Ignored optional GUI operation failure", exc_info=True)
            edit.setFixedHeight(self._scaled_px(height))
            box_layout.addWidget(edit)
            layout.addWidget(box)
            fields[key] = edit
        layout.addStretch(1)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        frame_shape = getattr(QtWidgets.QFrame, "Shape", QtWidgets.QFrame).NoFrame
        scroll.setFrameShape(frame_shape)
        scroll.setWidget(panel)
        return scroll, fields

    def _set_import_preview_fields(self, fields: Mapping[str, object], values: Mapping[str, Any]) -> None:
        for key, widget in fields.items():
            text = values.get(key, "")
            if isinstance(text, (list, tuple)):
                text = "\n".join(str(line) for line in text)
            widget.setPlainText(str(text or ""))

    def add_experiment_dialog(self):
        if self.is_busy:
            return
        self._capture_active_workspace()
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Add Experiment")
        dialog_min = (self._scaled_px(980), self._scaled_px(680))
        dialog.setMinimumSize(*dialog_min)
        dialog.resize(self._scaled_px(1120), self._scaled_px(760))
        dialog.setProperty("mrax_preserve_initial_size", True)
        root_layout = QtWidgets.QVBoxLayout(dialog)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(10)
        content_splitter = QtWidgets.QSplitter(dialog)
        content_splitter.setOrientation(QtCore.Qt.Horizontal)
        content_splitter.setChildrenCollapsible(False)
        left_scroll = QtWidgets.QScrollArea(dialog)
        left_scroll.setWidgetResizable(True)
        left_scroll.setMinimumWidth(self._scaled_px(560))
        frame_shape = getattr(QtWidgets.QFrame, "Shape", QtWidgets.QFrame).NoFrame
        left_scroll.setFrameShape(frame_shape)
        left_panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(left_panel)
        layout.setContentsMargins(0, 0, 8, 0)
        layout.setSpacing(10)
        left_scroll.setWidget(left_panel)
        right_panel = QtWidgets.QWidget()
        right_panel.setMinimumWidth(self._scaled_px(360))
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(8, 0, 0, 0)
        right_layout.setSpacing(8)
        content_splitter.addWidget(left_scroll)
        content_splitter.addWidget(right_panel)
        content_splitter.setStretchFactor(0, 3)
        content_splitter.setStretchFactor(1, 2)
        content_splitter.setSizes([self._scaled_px(660), self._scaled_px(420)])
        root_layout.addWidget(content_splitter, 1)
        form = QtWidgets.QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        field_growth = getattr(QtWidgets.QFormLayout, "FieldGrowthPolicy", QtWidgets.QFormLayout).AllNonFixedFieldsGrow
        form.setFieldGrowthPolicy(field_growth)

        source_combo = QtWidgets.QComboBox()
        source_combo.addItem("Blank image", "blank")
        source_combo.addItem("NIfTI image", "nifti")
        source_combo.addItem("DICOM image", "dicom")
        source_combo.addItem("NumPy array (.npy/.npz)", "numpy")
        source_combo.addItem("Raw binary", "raw")
        form.addRow("Source", source_combo)

        file_row = QtWidgets.QHBoxLayout()
        file_edit = QtWidgets.QLineEdit("")
        browse_btn = QtWidgets.QPushButton("Browse...")
        file_row.addWidget(file_edit, 1)
        file_row.addWidget(browse_btn)
        form.addRow("File", file_row)

        name_edit = QtWidgets.QLineEdit(self._next_experiment_name("experiment"))
        form.addRow("Experiment name", name_edit)

        exp_type_combo = QtWidgets.QComboBox()
        exp_type_combo.setEditable(True)
        exp_type_combo.addItems(_known_exp_types("anatomical"))
        exp_type_combo.setCurrentText("anatomical")
        form.addRow("Experiment type", exp_type_combo)

        model_combo = QtWidgets.QComboBox()
        for choice in model_choices():
            model_combo.addItem(choice.label, choice.key)
        model_combo.setCurrentIndex(max(0, model_combo.findData("identity")))
        form.addRow("Model", model_combo)
        exp_type_combo.currentTextChanged.connect(lambda text: _set_model_combo_from_type(model_combo, text))

        metadata_source_combo = QtWidgets.QComboBox()
        metadata_source_combo.addItem("Manual JSON fields", "manual")
        for label, item_data in _workspace_experiment_metadata_template_items(
            self._study_workspaces,
            active_key=self._active_study_key,
        ):
            metadata_source_combo.addItem(label, item_data)
        metadata_source_combo.addItem("MRax metadata/manifest file", "file")
        self._configure_wide_combo(metadata_source_combo, contents=28)
        form.addRow("Metadata template", metadata_source_combo)

        metadata_file_row = QtWidgets.QHBoxLayout()
        metadata_file_edit = QtWidgets.QLineEdit("")
        metadata_file_browse_btn = QtWidgets.QPushButton("Browse...")
        metadata_file_row.addWidget(metadata_file_edit, 1)
        metadata_file_row.addWidget(metadata_file_browse_btn)
        form.addRow("Metadata file", metadata_file_row)

        metadata_file_experiment_edit = QtWidgets.QLineEdit("")
        metadata_file_experiment_edit.setPlaceholderText("optional experiment name for study_manifest.json")
        form.addRow("File experiment", metadata_file_experiment_edit)

        apply_metadata_btn = QtWidgets.QPushButton("Apply metadata template")
        form.addRow("", apply_metadata_btn)

        observable_mismatch_combo = QtWidgets.QComboBox()
        observable_mismatch_combo.addItem("Fail on mismatch", "fail")
        observable_mismatch_combo.addItem("Trim longer observables", "trim")
        observable_mismatch_combo.addItem("Extend shorter observables", "extend")
        observable_mismatch_combo.addItem("Trim or extend observables", "trim_extend")
        self._configure_wide_combo(observable_mismatch_combo, contents=24)
        form.addRow("Observable mismatch", observable_mismatch_combo)

        shape_edit = QtWidgets.QLineEdit("1,64,64")
        form.addRow("Array shape", shape_edit)
        order_edit = QtWidgets.QLineEdit("frame,x,y")
        form.addRow("Dimension order", order_edit)

        dtype_combo = QtWidgets.QComboBox()
        for dtype in ("float32", "float64", "int16", "uint16", "int32", "uint8", "complex64", "complex128"):
            dtype_combo.addItem(dtype)
        form.addRow("Raw dtype", dtype_combo)
        endian_combo = QtWidgets.QComboBox()
        endian_combo.addItem("Native endian", "native")
        endian_combo.addItem("Little endian", "little")
        endian_combo.addItem("Big endian", "big")
        form.addRow("Raw endian", endian_combo)
        offset_spin = QtWidgets.QSpinBox()
        offset_spin.setRange(0, 2_000_000_000)
        offset_spin.setSuffix(" bytes")
        form.addRow("Raw byte offset", offset_spin)
        memory_order_combo = QtWidgets.QComboBox()
        memory_order_combo.addItem("C / row-major", "C")
        memory_order_combo.addItem("Fortran / column-major", "F")
        form.addRow("Memory order", memory_order_combo)
        npz_key_edit = QtWidgets.QLineEdit("")
        form.addRow("NPZ array key", npz_key_edit)
        settings_box = QtWidgets.QGroupBox("Experiment Data")
        settings_layout = QtWidgets.QVBoxLayout(settings_box)
        settings_layout.addLayout(form)
        layout.addWidget(settings_box)

        observables_edit = QtWidgets.QPlainTextEdit("{}")
        observables_edit.setMinimumHeight(self._scaled_px(96))
        observables_edit.setMaximumHeight(self._scaled_px(140))
        layout.addWidget(_section_box(QtWidgets, "Observables JSON", observables_edit))

        metadata_edit = QtWidgets.QPlainTextEdit("{}")
        metadata_edit.setMinimumHeight(self._scaled_px(96))
        metadata_edit.setMaximumHeight(self._scaled_px(140))
        layout.addWidget(_section_box(QtWidgets, "Extra Metadata JSON", metadata_edit))
        layout.addStretch(1)

        preview_panel, preview_fields = self._make_import_preview_panel(
            (
                ("status", "Status", 72),
                ("file", "File and Source", 96),
                ("array", "Array and Dimensions", 136),
                ("raw", "Raw Diagnostics", 148),
                ("metadata", "Metadata and Observables", 120),
            )
        )
        right_layout.addWidget(preview_panel, 1)
        preview_timer = QtCore.QTimer(dialog)
        preview_timer.setSingleShot(True)

        def update_source_controls(*_args):
            source = str(source_combo.currentData() or "blank")
            inferred = source == "dicom"
            file_edit.setEnabled(source != "blank")
            browse_btn.setEnabled(source != "blank")
            dtype_combo.setEnabled(source == "raw")
            endian_combo.setEnabled(source == "raw")
            offset_spin.setEnabled(source == "raw")
            memory_order_combo.setEnabled(source in {"nifti", "numpy", "raw"})
            npz_key_edit.setEnabled(source == "numpy")
            shape_edit.setEnabled(not inferred)
            order_edit.setEnabled(not inferred)
            exp_type_combo.setEnabled(not inferred)
            model_combo.setEnabled(not inferred)
            observables_edit.setEnabled(not inferred)
            metadata_edit.setEnabled(not inferred)
            observable_mismatch_combo.setEnabled(not inferred)
            metadata_source_combo.setEnabled(not inferred)
            if source == "blank" and not shape_edit.text().strip():
                shape_edit.setText("1,64,64")
            if source in {"nifti", "numpy"} and order_edit.text().strip() in {"frame,x,y", "x,y,z"}:
                order_edit.clear()
            if source in {"nifti", "numpy"} and shape_edit.text().strip() == "1,64,64":
                shape_edit.clear()
            if source in {"blank", "raw"} and order_edit.text().strip() in {"x,y,z", ""}:
                order_edit.setText("frame,x,y")
            if source == "raw" and not shape_edit.text().strip():
                shape_edit.setText("1,64,64")
            update_metadata_template_controls()
            schedule_preview()

        def browse_source():
            source = str(source_combo.currentData() or "blank")
            filters = {
                "nifti": "NIfTI (*.nii *.nii.gz);;All files (*)",
                "dicom": "DICOM (*.dcm *.dicom *.ima);;All files (*)",
                "numpy": "NumPy (*.npy *.npz);;All files (*)",
                "raw": "Raw binary (*);;All files (*)",
            }
            path, _ = self._get_open_file_name("Choose experiment data", str(Path.cwd()), filters.get(source, "All files (*)"))
            if path:
                file_edit.setText(path)
                suggested_name = Path(path).stem
                if source == "dicom":
                    try:
                        imported = _load_dicom_experiment_from_dialog(Path(path))
                        suggested_name = str(imported["name"])
                        exp_type = str(imported["exp_type"])
                        if exp_type_combo.findText(exp_type) < 0:
                            exp_type_combo.addItem(exp_type)
                        exp_type_combo.setCurrentText(exp_type)
                        model_index = model_combo.findData(str(imported["model_name"]))
                        if model_index >= 0:
                            model_combo.setCurrentIndex(model_index)
                        observables_edit.setPlainText(
                            _observables_to_json_text(imported["observables"])
                        )
                        metadata_edit.setPlainText(
                            json.dumps(
                                _metadata_to_jsonable(imported["extra_metadata"]),
                                indent=2,
                                sort_keys=True,
                            )
                        )
                    except Exception as exc:
                        self.statusBar().showMessage(f"Could not inspect DICOM file: {exc}")
                if not name_edit.text().strip() or name_edit.text().startswith("experiment"):
                    name_edit.setText(self._next_experiment_name(suggested_name))
                schedule_preview()

        def schedule_preview(*_args):
            preview_timer.start(250)

        def metadata_template_payload() -> dict[str, Any] | None:
            source = metadata_source_combo.currentData()
            if source is None or str(source) == "manual":
                return None
            if _is_workspace_metadata_template_item(source):
                _kind, workspace_key, exp_name = source
                workspace = self._study_workspaces.get(str(workspace_key))
                if workspace is None:
                    raise KeyError(f"Unknown study workspace '{workspace_key}'")
                template = _experiment_metadata_template_from_session(workspace.session, str(exp_name))
                template["label"] = f"{workspace.label} / {exp_name}"
                return template
            if str(source) == "file":
                path_text = metadata_file_edit.text().strip()
                if not path_text:
                    raise FileNotFoundError("Choose a metadata.json or study_manifest.json file")
                return _load_experiment_metadata_template_from_file(
                    Path(path_text).expanduser(),
                    experiment_name=metadata_file_experiment_edit.text().strip() or None,
                )
            raise ValueError(f"Unsupported metadata template source '{source}'")

        def apply_metadata_template():
            try:
                template = metadata_template_payload()
            except Exception as exc:
                self.statusBar().showMessage(f"Could not apply metadata template: {exc}")
                update_preview()
                return
            if template is None:
                self.statusBar().showMessage("Manual metadata fields are active.")
                return
            exp_type = str(template.get("exp_type") or "anatomical")
            if exp_type_combo.findText(exp_type) < 0:
                exp_type_combo.addItem(exp_type)
            exp_type_combo.setCurrentText(exp_type)
            model_key = normalize_model_key(template.get("model") or "identity") or "identity"
            model_idx = model_combo.findData(model_key)
            if model_idx >= 0:
                model_combo.setCurrentIndex(model_idx)
            dims = dict(template.get("dims") or {})
            if dims:
                order_edit.setText(",".join(str(key) for key in dims.keys()))
                if str(source_combo.currentData() or "blank") in {"blank", "raw"}:
                    shape_edit.setText(",".join(str(value) for value in dims.values()))
            observables_edit.setPlainText(_observables_to_json_text(template.get("observables") or {}))
            metadata_edit.setPlainText(
                json.dumps(_metadata_to_jsonable(template.get("extra_metadata") or {}), indent=2, sort_keys=True)
            )
            label = str(template.get("label") or template.get("name") or "metadata template")
            self.statusBar().showMessage(f"Applied metadata template from {label}.")
            schedule_preview()

        def update_metadata_template_controls(*_args):
            source = metadata_source_combo.currentData()
            inferred = str(source_combo.currentData() or "blank") == "dicom"
            is_file = source == "file" and not inferred
            metadata_file_edit.setEnabled(is_file)
            metadata_file_browse_btn.setEnabled(is_file)
            metadata_file_experiment_edit.setEnabled(is_file)
            apply_metadata_btn.setEnabled(
                not inferred and source is not None and str(source) != "manual"
            )
            schedule_preview()

        def on_metadata_template_source_changed(*_args):
            update_metadata_template_controls()
            source = metadata_source_combo.currentData()
            if _is_workspace_metadata_template_item(source):
                apply_metadata_template()

        def browse_metadata_template():
            path, _ = self._get_open_file_name(
                "Choose MRax experiment metadata",
                str(Path.cwd()),
                "MRax JSON (*.json);;All files (*)",
            )
            if path:
                metadata_file_edit.setText(path)
                apply_metadata_template()

        def preview_payload() -> dict[str, Any]:
            source = str(source_combo.currentData() or "blank")
            path = Path(file_edit.text()).expanduser() if file_edit.text().strip() else None
            if source == "dicom":
                if path is None:
                    raise FileNotFoundError("Choose a DICOM image file")
                return _load_dicom_experiment_from_dialog(path)
            shape = _parse_shape_text(shape_edit.text())
            order = _parse_dim_order(order_edit.text())
            data, metadata = _load_experiment_array_from_dialog(
                source=source,
                path=path,
                shape=shape,
                dtype=dtype_combo.currentText(),
                endian=str(endian_combo.currentData() or "native"),
                byte_offset=int(offset_spin.value()),
                memory_order=str(memory_order_combo.currentData() or "C"),
                npz_key=npz_key_edit.text().strip() or None,
            )
            dims = (
                _dims_from_order_and_shape(order, data.shape)
                if order
                else _dims_from_order_and_shape(_default_dim_order(data.ndim, source), data.shape)
            )
            observables_raw = _observables_from_json_text(observables_edit.toPlainText())
            observables, observable_compatibility_lines = _apply_observable_dim_compatibility(
                observables_raw,
                dims,
                exp_type=exp_type_combo.currentText().strip() or "anatomical",
                model_name=str(model_combo.currentData() or "identity"),
                policy=str(observable_mismatch_combo.currentData() or "fail"),
            )
            extra_metadata = _metadata_from_json_text(metadata_edit.toPlainText())
            extra_metadata.update(metadata)
            return {
                "kind": "experiment",
                "source": source,
                "path": path,
                "data": data,
                "dims": dims,
                "observables": observables,
                "observable_compatibility": observable_compatibility_lines,
                "extra_metadata": extra_metadata,
            }

        def update_preview():
            source = str(source_combo.currentData() or "blank")
            path = Path(file_edit.text()).expanduser() if file_edit.text().strip() else None
            shape: tuple[int, ...] = ()
            order: tuple[str, ...] = ()
            try:
                shape = _parse_shape_text(shape_edit.text())
            except Exception:
                _LOG.debug("Ignored optional GUI operation failure", exc_info=True)
            try:
                order = _parse_dim_order(order_edit.text())
            except Exception:
                _LOG.debug("Ignored optional GUI operation failure", exc_info=True)
            status_lines: list[str] = []
            raw_lines: list[str] = ["Not used for this source."] if source != "raw" else []
            file_lines = _import_file_preview_lines(path, source=source)
            file_lines.extend(
                [
                    f"experiment name: {name_edit.text().strip() or '(auto)'}",
                    f"experiment type: {exp_type_combo.currentText().strip() or 'anatomical'}",
                    f"model: {str(model_combo.currentData() or 'identity')}",
                    f"memory order: {str(memory_order_combo.currentData() or 'C')}",
                    f"metadata template: {metadata_source_combo.currentText()}",
                ]
            )
            if metadata_source_combo.currentData() == "file":
                file_lines.append(f"metadata file: {metadata_file_edit.text().strip() or '(none)'}")
                file_lines.append(
                    f"file experiment: {metadata_file_experiment_edit.text().strip() or '(auto/single)'}"
                )
            array_lines: list[str] = [
                f"requested shape: {shape if shape else '(use file shape)'}",
                f"dimension order: {', '.join(order) if order else '(auto)'}",
            ]
            metadata_lines: list[str] = []
            if source == "raw":
                raw_lines = (
                    _raw_import_diagnostic_lines(
                        path,
                        dtype=dtype_combo.currentText(),
                        endian=str(endian_combo.currentData() or "native"),
                        shape=shape,
                        byte_offset=int(offset_spin.value()),
                        order=order,
                    )
                )
            try:
                payload = preview_payload()
                data = np.asarray(payload["data"])
                array_lines.extend(_array_detail_lines(data))
                array_lines.append("dims: " + ", ".join(f"{k}={v}" for k, v in payload["dims"].items()))
                metadata_lines.extend(_mapping_detail_lines(payload["extra_metadata"], title="import metadata"))
                metadata_lines.extend(_observables_detail_lines(payload["observables"]))
                metadata_lines.extend(payload.get("observable_compatibility") or [])
                status_lines.append("settings are importable")
                status_lines.append("OK will add an experiment to the active study.")
            except Exception as exc:
                status_lines.append("settings are not importable")
                status_lines.append(f"mismatch note: {exc}")
                if not metadata_lines:
                    metadata_lines.append("Metadata and observables will be checked after the array settings are valid.")
            self._set_import_preview_fields(
                preview_fields,
                {
                    "status": status_lines,
                    "file": file_lines,
                    "array": array_lines,
                    "raw": raw_lines,
                    "metadata": metadata_lines,
                },
            )

        source_combo.currentIndexChanged.connect(update_source_controls)
        metadata_source_combo.currentIndexChanged.connect(on_metadata_template_source_changed)
        metadata_file_browse_btn.clicked.connect(browse_metadata_template)
        apply_metadata_btn.clicked.connect(apply_metadata_template)
        browse_btn.clicked.connect(browse_source)
        preview_timer.timeout.connect(update_preview)
        for widget in (file_edit, shape_edit, order_edit, npz_key_edit):
            widget.textChanged.connect(schedule_preview)
        metadata_file_edit.textChanged.connect(schedule_preview)
        metadata_file_experiment_edit.textChanged.connect(schedule_preview)
        dtype_combo.currentIndexChanged.connect(schedule_preview)
        endian_combo.currentIndexChanged.connect(schedule_preview)
        memory_order_combo.currentIndexChanged.connect(schedule_preview)
        observable_mismatch_combo.currentIndexChanged.connect(schedule_preview)
        offset_spin.valueChanged.connect(schedule_preview)
        observables_edit.textChanged.connect(schedule_preview)
        metadata_edit.textChanged.connect(schedule_preview)
        update_source_controls()
        update_metadata_template_controls()
        update_preview()

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        accepted_payload: dict[str, Any] = {}

        def try_accept_import():
            try:
                payload = preview_payload()
            except Exception as exc:
                update_preview()
                current = preview_fields["status"].toPlainText().rstrip()
                suffix = f"\nImport not applied: {exc}"
                preview_fields["status"].setPlainText((current + suffix).strip())
                self.statusBar().showMessage(f"Import settings mismatch: {exc}")
                return
            accepted_payload.clear()
            accepted_payload.update(payload)
            dialog.accept()

        buttons.accepted.connect(try_accept_import)
        buttons.rejected.connect(dialog.reject)
        root_layout.addWidget(buttons)
        if self._exec_dialog(dialog) != QtWidgets.QDialog.Accepted:
            return

        try:
            name = self.session.add_experiment(
                name_edit.text().strip(),
                accepted_payload["data"],
                dims=accepted_payload["dims"],
                exp_type=str(
                    accepted_payload.get("exp_type")
                    or exp_type_combo.currentText().strip()
                    or "anatomical"
                ),
                model_name=str(
                    accepted_payload.get("model_name")
                    or model_combo.currentData()
                    or "identity"
                ),
                observables=accepted_payload["observables"],
                extra_metadata=accepted_payload["extra_metadata"],
                data_role="image",
                non_imaging=False,
                select=True,
            )
        except Exception:
            self.show_error("Could not add experiment", traceback.format_exc())
            return
        self.mark_dirty("experiment added")
        self.populate_from_session()
        self.statusBar().showMessage(f"Added experiment '{name}'.")

    def edit_experiment_metadata_dialog(self):
        if not self.session.is_loaded:
            self.statusBar().showMessage("Select an experiment to edit.")
            return
        exp_name = str(self.session.selected_experiment or "")
        exp_data = self.session.current_experiment
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(f"Edit Experiment: {exp_name}")
        dialog.resize(self._scaled_px(760), self._scaled_px(620))
        layout = QtWidgets.QVBoxLayout(dialog)
        form = QtWidgets.QFormLayout()
        exp_type_combo = QtWidgets.QComboBox()
        exp_type_combo.setEditable(True)
        exp_type_combo.addItems(_known_exp_types(str(exp_data.get("exp_type") or "anatomical")))
        exp_type_combo.setCurrentText(str(exp_data.get("exp_type") or "anatomical"))
        form.addRow("Experiment type", exp_type_combo)
        model_combo = QtWidgets.QComboBox()
        for choice in model_choices():
            model_combo.addItem(choice.label, choice.key)
        current_model = normalize_model_key(exp_data.get("model") or "identity") or "identity"
        idx = model_combo.findData(current_model)
        model_combo.setCurrentIndex(idx if idx >= 0 else 0)
        form.addRow("Model", model_combo)
        layout.addLayout(form)
        layout.addWidget(QtWidgets.QLabel("Observables JSON"))
        observables_edit = QtWidgets.QPlainTextEdit(_observables_to_json_text(exp_data.get("observables") or {}))
        layout.addWidget(observables_edit, 1)
        layout.addWidget(QtWidgets.QLabel("Extra Metadata JSON"))
        metadata_edit = QtWidgets.QPlainTextEdit(json.dumps(_metadata_to_jsonable(exp_data.get("extra_metadata") or {}), indent=2, sort_keys=True))
        layout.addWidget(metadata_edit, 1)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if self._exec_dialog(dialog) != QtWidgets.QDialog.Accepted:
            return
        try:
            self.session.update_experiment_metadata(
                exp_name,
                exp_type=exp_type_combo.currentText().strip() or None,
                model_name=str(model_combo.currentData() or "identity"),
                observables=_observables_from_json_text(observables_edit.toPlainText()),
                extra_metadata=_metadata_from_json_text(metadata_edit.toPlainText()),
                replace_observables=True,
            )
        except Exception:
            self.show_error("Could not update experiment metadata", traceback.format_exc())
            return
        self.mark_dirty("experiment metadata edited")
        self.populate_from_session()
        self.statusBar().showMessage(f"Updated metadata for {exp_name}.")

    def _next_experiment_name(self, stem: str) -> str:
        base = _safe_filename(str(stem or "experiment")).replace("-", "_") or "experiment"
        existing = set(self.session.experiment_names)
        candidate = base
        idx = 2
        while candidate in existing:
            candidate = f"{base}_{idx}"
            idx += 1
        return candidate

    def load_manifest(self, path: Path):
        try:
            self._load_manifest_data(path)
        except Exception:
            self.show_error("Could not load manifest", traceback.format_exc())
            return
        self.populate_from_session()
        self.clear_dirty()
        self.statusBar().showMessage(f"Loaded {path}")

    def _load_manifest_data(self, path: Path):
        session = GuiSession()
        session.load_manifest(path)
        label = self._workspace_label_for_session(session, Path(path).parent.name or Path(path).stem)
        self._replace_or_add_loaded_workspace(session, label)

    def populate_from_session(self):
        self._displayed_image_shape = None
        self._displayed_view_key = None
        self.last_fit_result = self._fit_results_by_experiment.get(self.session.selected_experiment or "")
        self.last_preview_fit_result = self._preview_results_by_experiment.get(self.session.selected_experiment or "")
        self.refresh_study_selector()
        self.experiment_list.blockSignals(True)
        self.experiment_list.clear()
        self.experiment_list.addItems(self.session.experiment_names)
        if self.session.selected_experiment:
            matches = self.experiment_list.findItems(
                self.session.selected_experiment,
                QtCore.Qt.MatchExactly,
            )
            if matches:
                self.experiment_list.setCurrentItem(matches[0])
        self.experiment_list.blockSignals(False)

        self.refresh_map_list()
        self.refresh_visual_state_controls()
        self.kspace_mode.blockSignals(True)
        self.kspace_mode.setCurrentText(self.session.active_kspace_mode)
        self.kspace_mode.blockSignals(False)
        self.kspace_centering.blockSignals(True)
        self.kspace_centering.setCurrentText(self.session.active_kspace_centering)
        self.kspace_centering.blockSignals(False)
        self.refresh_roi_list()
        self.refresh_plot_x_axis_combo()
        self.refresh_frame_controls()
        self.refresh_model_combo()
        self.refresh_sat_trans_reference_combo()
        self.refresh_ab_mt_map_combos()
        self.refresh_frame_mask_list()
        self.refresh_registration_tab()
        self.update_view()
        self.update_inspector()
        self.update_metadata()
        self.update_summary()
        self.update_diagnostics()
        self.update_fit_results_summary()

    def refresh_model_combo(self):
        key = normalize_model_key(self.session.fit_settings.model_name or self.session.current_model_name)
        if not key:
            return
        self.model_combo.blockSignals(True)
        for idx in range(self.model_combo.count()):
            if self.model_combo.itemData(idx) == key:
                self.model_combo.setCurrentIndex(idx)
                break
        self.model_combo.blockSignals(False)
        self.optimizer_combo.setCurrentText(self.session.fit_settings.optimizer_name)
        self.steps_spin.setValue(int(self.session.fit_settings.num_steps))
        self.step_size_spin.setValue(float(self.session.fit_settings.step_size))
        self.mask_fill_spin.setValue(float(self.session.fit_settings.mask_fill_value or 0.0))
        self.auto_nan_check.setChecked(bool(self.session.fit_settings.auto_mask_nan_observables))
        self.fit_use_view_state_check.setChecked(bool(self.session.fit_settings.use_view_state_data))
        if self.session.fit_settings.compute_derived_maps is None:
            self.derived_maps_combo.setCurrentText("auto")
        else:
            self.derived_maps_combo.setCurrentText("yes" if self.session.fit_settings.compute_derived_maps else "no")
        self.refresh_sat_trans_reference_combo()
        self.refresh_ab_mt_map_combos()
        self.refresh_advanced_fit_controls()

    def refresh_sat_trans_reference_combo(self, *_args):
        exp_type = self.session.current_experiment.get("exp_type") if self.session.is_loaded else None
        model_key = str(self.model_combo.currentData() or "")
        visible = _sat_trans_reference_visible(model_key, exp_type)
        choices = self.session.sat_trans_reference_choices() if visible else []
        selected = self.session.fit_settings.sat_trans_reference_scan
        self.sat_trans_reference_combo.blockSignals(True)
        self.sat_trans_reference_combo.clear()
        self.sat_trans_reference_combo.addItem("auto", None)
        for name in choices:
            self.sat_trans_reference_combo.addItem(name, name)
        if selected:
            idx = self.sat_trans_reference_combo.findData(selected)
            if idx >= 0:
                self.sat_trans_reference_combo.setCurrentIndex(idx)
            else:
                self.sat_trans_reference_combo.addItem(f"{selected} (missing)", selected)
                self.sat_trans_reference_combo.setCurrentIndex(self.sat_trans_reference_combo.count() - 1)
        self.sat_trans_reference_combo.setEnabled(visible and bool(choices))
        self.sat_trans_reference_combo.blockSignals(False)
        self._set_sat_trans_reference_visible(visible)

    def refresh_ab_mt_map_combos(self, *_args):
        model_key = str(self.model_combo.currentData() or "")
        visible = _ab_mt_prior_maps_visible(model_key)
        enabled = visible and self.session.is_loaded
        configs = (
            (self.ab_mt_t1_combo, "t1", self.session.fit_settings.ab_mt_t1_map_ref),
            (self.ab_mt_b1_combo, "b1", self.session.fit_settings.ab_mt_b1_map_ref),
            (self.ab_mt_b0_combo, "b0", self.session.fit_settings.ab_mt_b0_map_ref),
        )
        for combo, kind, selected in configs:
            choices = self.session.ab_mt_map_choices(kind) if enabled else []
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("auto", None)
            for choice in choices:
                combo.addItem(choice["label"], choice["ref"])
            if selected:
                idx = combo.findData(selected)
                if idx >= 0:
                    combo.setCurrentIndex(idx)
                else:
                    combo.addItem(f"{selected} (missing)", selected)
                    combo.setCurrentIndex(combo.count() - 1)
            combo.setEnabled(enabled and bool(choices))
            combo.blockSignals(False)
        self._set_ab_mt_prior_maps_visible(visible)

    def _set_fit_dependency_controls_visible(self, sat_trans: bool, ab_mt: bool):
        self._set_sat_trans_reference_visible(sat_trans)
        self._set_ab_mt_prior_maps_visible(ab_mt)

    def _set_sat_trans_reference_visible(self, visible: bool):
        self._fit_sat_trans_dependency_visible = bool(visible)
        self.sat_trans_reference_label.setVisible(bool(visible))
        self.sat_trans_reference_combo.setVisible(bool(visible))
        self._refresh_fit_inputs_visibility()
        self.refresh_fit_breakout_summaries()

    def _set_ab_mt_prior_maps_visible(self, visible: bool):
        self._fit_ab_mt_dependency_visible = bool(visible)
        for widget in (
            self.ab_mt_t1_label,
            self.ab_mt_t1_combo,
            self.ab_mt_b1_label,
            self.ab_mt_b1_combo,
            self.ab_mt_b0_label,
            self.ab_mt_b0_combo,
        ):
            widget.setVisible(bool(visible))
        self._refresh_fit_inputs_visibility()
        self.refresh_fit_breakout_summaries()

    def _refresh_fit_inputs_visibility(self):
        if hasattr(self, "fit_inputs_empty_label"):
            has_dependency = bool(
                getattr(self, "_fit_sat_trans_dependency_visible", False)
                or getattr(self, "_fit_ab_mt_dependency_visible", False)
            )
            self.fit_inputs_empty_label.setVisible(not has_dependency)
            self.fit_inputs_button.setVisible(has_dependency)

    def open_fit_inputs_dialog(self):
        self._open_fit_breakout_dialog(self.fit_inputs_dialog)

    def open_fit_options_dialog(self):
        self._open_fit_breakout_dialog(self.fit_options_dialog)

    def open_fit_parameters_dialog(self):
        self._open_fit_breakout_dialog(self.fit_parameters_dialog)

    def open_fit_frames_dialog(self):
        self._open_fit_breakout_dialog(self.fit_frames_dialog)

    def _open_fit_breakout_dialog(self, dialog):
        if dialog is None:
            return
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def refresh_fit_breakout_summaries(self):
        if not hasattr(self, "fit_parameters_summary_label"):
            return
        if not all(
            hasattr(self, name)
            for name in (
                "sat_trans_reference_combo",
                "ab_mt_t1_combo",
                "mask_fill_spin",
                "auto_nan_check",
                "derived_maps_combo",
                "param_table",
            )
        ):
            return
        model_key = str(self.model_combo.currentData() or "")
        exp_type = self.session.current_experiment.get("exp_type") if self.session.is_loaded else None
        sat_visible = _sat_trans_reference_visible(model_key, exp_type)
        ab_visible = _ab_mt_prior_maps_visible(model_key)
        inputs: list[str] = []
        if sat_visible:
            value = self.sat_trans_reference_combo.currentData()
            inputs.append(f"sat_trans reference: {value or 'auto'}")
        if ab_visible:
            inputs.append(
                "AB-MT priors: "
                f"T1={self.ab_mt_t1_combo.currentText() or 'auto'}, "
                f"B1={self.ab_mt_b1_combo.currentText() or 'auto'}, "
                f"B0={self.ab_mt_b0_combo.currentText() or 'auto'}"
            )
        self.fit_inputs_summary_label.setText("; ".join(inputs) if inputs else "No model-specific inputs")
        self.fit_inputs_button.setVisible(bool(inputs))

        self.fit_options_summary_label.setText(
            "view state: "
            f"{'on' if self.fit_use_view_state_check.isChecked() else 'off'}, "
            f"mask fill: {float(self.mask_fill_spin.value()):.6g}, "
            f"NaN auto-mask: {'on' if self.auto_nan_check.isChecked() else 'off'}, "
            f"derived: {self.derived_maps_combo.currentText()}"
        )

        pools = self.selected_pool_names() if hasattr(self, "pool_list") else ()
        pool_text = f"; pools: {', '.join(pools)}" if pools else ""
        self.fit_parameters_summary_label.setText(
            f"{self.param_table.rowCount()} parameter row(s){pool_text}"
        )

        if self.session.frame_mask is None:
            self.fit_frames_summary_label.setText("No frame inclusion mask")
        else:
            included = self.session.frame_mask.as_array()
            total = int(included.size)
            selected = int(np.count_nonzero(included))
            excluded = total - selected
            self.fit_frames_summary_label.setText(
                f"{selected}/{total} frame(s) included"
                if excluded
                else f"All {total} frame(s) included"
            )

    def refresh_advanced_fit_controls(self, *_args):
        key = str(self.model_combo.currentData() or "")
        available = list(model_available_pools(key))
        if _pool_editor_visible(key):
            for pool in _gui_config_pool_names(self.gui_config, key):
                if pool not in available:
                    available.append(pool)
        defaults = self.session.fit_settings.selected_pools or model_default_pools(key)
        for pool in defaults:
            if pool not in available:
                available.append(pool)
        pool_controls_visible = bool(available) or _pool_editor_visible(key)
        self.pool_list.setVisible(pool_controls_visible)
        self.custom_pool_widget.setVisible(_pool_editor_visible(key))
        self.pool_list.blockSignals(True)
        self.pool_list.clear()
        for pool in available:
            item = QtWidgets.QListWidgetItem(pool)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Checked if pool in defaults else QtCore.Qt.Unchecked)
            self.pool_list.addItem(item)
        self.pool_list.blockSignals(False)
        self.refresh_parameter_table()

    def add_custom_pool(self):
        key = str(self.model_combo.currentData() or "")
        if not _pool_editor_visible(key):
            return
        names = _sanitize_pool_names(self.custom_pool_edit.text())
        if not names:
            self.statusBar().showMessage("Enter at least one pool name.")
            return
        existing = {self.pool_list.item(row).text() for row in range(self.pool_list.count())}
        self.pool_list.blockSignals(True)
        try:
            for name in names:
                if name in existing:
                    matches = self.pool_list.findItems(name, QtCore.Qt.MatchExactly)
                    if matches:
                        matches[0].setCheckState(QtCore.Qt.Checked)
                    continue
                item = QtWidgets.QListWidgetItem(name)
                item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
                item.setCheckState(QtCore.Qt.Checked)
                self.pool_list.addItem(item)
                existing.add(name)
                template = _gui_config_pool_templates(self.gui_config, key).get(
                    name,
                    _default_pool_override(key, name),
                )
                self.session.fit_settings.pool_overrides.setdefault(
                    name,
                    template,
                )
                self.gui_config.setdefault("custom_pools", {}).setdefault(key, {}).setdefault(name, template)
        finally:
            self.pool_list.blockSignals(False)
        _save_gui_config(self.gui_config, self.gui_config_path)
        self.custom_pool_edit.clear()
        self.refresh_parameter_table()
        self.statusBar().showMessage(f"Added pool(s): {', '.join(names)}")

    def remove_selected_pool(self):
        key = str(self.model_combo.currentData() or "")
        if not _pool_editor_visible(key):
            return
        item = self.pool_list.currentItem()
        if item is None:
            self.statusBar().showMessage("Select a pool to remove or disable.")
            return
        name = item.text()
        built_in = set(model_available_pools(key))
        if name in built_in:
            item.setCheckState(QtCore.Qt.Unchecked)
            self.statusBar().showMessage(f"Disabled built-in pool '{name}' for this fit.")
        else:
            row = self.pool_list.row(item)
            self.pool_list.takeItem(row)
            self.session.fit_settings.pool_overrides.pop(name, None)
            self.statusBar().showMessage(
                f"Removed pool '{name}' from this fit. It remains in GUI settings for reuse."
            )
        self.refresh_parameter_table()

    def selected_pool_names(self) -> tuple[str, ...]:
        return tuple(
            self.pool_list.item(row).text()
            for row in range(self.pool_list.count())
            if self.pool_list.item(row).checkState() == QtCore.Qt.Checked
        )

    def refresh_parameter_table(self):
        key = str(self.model_combo.currentData() or "")
        pools = self.selected_pool_names() or None
        pool_overrides = _pool_overrides_with_custom_defaults(
            key,
            pools or (),
            self.session.fit_settings.pool_overrides or {},
            self.gui_config,
        )
        specs = model_parameter_specs(
            key,
            pools=pools,
            param_overrides=self.session.fit_settings.param_overrides or None,
            pool_overrides=pool_overrides or None,
            parameter_constant_dims=self.session.fit_settings.parameter_constant_dims or None,
        )
        self.param_table.setRowCount(len(specs))
        for row, (name, spec) in enumerate(specs.items()):
            name_item = QtWidgets.QTableWidgetItem(str(name))
            name_item.setFlags(name_item.flags() & ~QtCore.Qt.ItemIsEditable)
            self.param_table.setItem(row, 0, name_item)
            self.param_table.setItem(row, 1, QtWidgets.QTableWidgetItem(_format_optional_float(spec.get("initial"))))
            bounds = spec.get("bounds")
            lower = bounds[0] if isinstance(bounds, (list, tuple)) and len(bounds) == 2 else ""
            upper = bounds[1] if isinstance(bounds, (list, tuple)) and len(bounds) == 2 else ""
            self.param_table.setItem(row, 2, QtWidgets.QTableWidgetItem(_format_optional_float(lower)))
            self.param_table.setItem(row, 3, QtWidgets.QTableWidgetItem(_format_optional_float(upper)))
            unit_item = QtWidgets.QTableWidgetItem(str(spec.get("unit") or ""))
            unit_item.setFlags(unit_item.flags() & ~QtCore.Qt.ItemIsEditable)
            self.param_table.setItem(row, 4, unit_item)
            reg_cfg = (self.session.fit_settings.regularizer_configs or {}).get(name, [{}])
            reg_entry = reg_cfg[0] if reg_cfg else {}
            reg_type_combo = QtWidgets.QComboBox()
            reg_type_combo.addItems(["none", "smoothness", "tv", "tv2"])
            reg_type_combo.setCurrentText(str(reg_entry.get("type") or "none"))
            reg_type_combo.currentTextChanged.connect(lambda _text: self.refresh_optimizer_choices())
            self.param_table.setCellWidget(row, 5, reg_type_combo)
            weight_edit = QtWidgets.QLineEdit(_format_optional_float(reg_entry.get("weight", 0.0)) or "0")
            weight_validator = QtGui.QDoubleValidator(0.0, 1.0e12, 12, weight_edit)
            notation = getattr(QtGui.QDoubleValidator, "ScientificNotation", None)
            if notation is None and hasattr(QtGui.QDoubleValidator, "Notation"):
                notation = QtGui.QDoubleValidator.Notation.ScientificNotation
            if notation is not None:
                weight_validator.setNotation(notation)
            weight_edit.setValidator(weight_validator)
            weight_edit.setToolTip("Regularizer loss weight. Use 0 to disable this regularizer.")
            weight_edit.editingFinished.connect(lambda: self.refresh_optimizer_choices())
            self.param_table.setCellWidget(row, 6, weight_edit)
            axes = reg_entry.get("axes")
            axes_combo = QtWidgets.QComboBox()
            axes_combo.setEditable(True)
            axes_combo.addItems(_axis_choice_labels(self.session.experiment_dims))
            axes_combo.setCurrentText(_format_csv(axes))
            self.param_table.setCellWidget(row, 7, axes_combo)
            dims = (self.session.fit_settings.parameter_constant_dims or {}).get(name)
            dims_combo = QtWidgets.QComboBox()
            dims_combo.setEditable(True)
            dims_combo.addItems(_constant_dim_choice_labels(self.session.experiment_dims))
            dims_combo.setCurrentText(_format_csv(dims))
            dims_combo.setToolTip(
                "Force this parameter to be constant across selected dimensions. "
                "Leave blank for a spatially varying fitted map."
            )
            self.param_table.setCellWidget(row, 8, dims_combo)
            fixed_check = QtWidgets.QCheckBox()
            fixed_check.setToolTip("Hold this parameter at the Initial value during fitting.")
            fixed_check.setChecked(_parameter_fixed_from_overrides(name, key, pools or (), self.session.fit_settings))
            self.param_table.setCellWidget(row, 9, fixed_check)
        self.param_table.resizeColumnsToContents()
        self.refresh_optimizer_choices(preferred=self.session.fit_settings.optimizer_name)
        self.refresh_fit_breakout_summaries()

    def refresh_optimizer_choices(self, preferred: str | None = None):
        current = preferred or self.optimizer_combo.currentText() or self.session.fit_settings.optimizer_name
        reg_types = self._selected_regularizer_types_from_table()
        choices = _optimizer_choices_for_regularizers(reg_types)
        self.optimizer_combo.blockSignals(True)
        self.optimizer_combo.clear()
        self.optimizer_combo.addItems(choices)
        if current in choices:
            self.optimizer_combo.setCurrentText(current)
        else:
            self.optimizer_combo.setCurrentText(choices[0])
        self.optimizer_combo.blockSignals(False)
        if current and current not in choices:
            self.statusBar().showMessage(
                f"Optimizer '{current}' does not support selected regularizer(s); using {choices[0]}."
            )

    def _selected_regularizer_types_from_table(self) -> set[str]:
        types: set[str] = set()
        for row in range(self.param_table.rowCount()):
            widget = self.param_table.cellWidget(row, 5)
            value = (
                widget.currentText()
                if widget is not None and hasattr(widget, "currentText")
                else (self.param_table.item(row, 5).text() if self.param_table.item(row, 5) else "")
            )
            reg_type = str(value or "").strip().lower()
            reg_weight = _regularizer_weight_from_cell(self.param_table, row, 6)
            if _regularizer_config_enabled(reg_type, reg_weight):
                types.add(reg_type)
        return types

    def refresh_map_list(self):
        self.map_list.blockSignals(True)
        self.map_list.clear()
        self.map_list.addItems(self.session.map_names)
        if self.session.active_map:
            matches = self.map_list.findItems(self.session.active_map, QtCore.Qt.MatchExactly)
            if matches:
                self.map_list.setCurrentItem(matches[0])
        self.map_list.blockSignals(False)
        self.map_list.setEnabled(_view_key(self.view_combo.currentText()) == "map" and bool(self.session.map_names))

    def refresh_visual_state_controls(self):
        if not hasattr(self, "visual_original_check"):
            return
        self._updating_visual_state_controls = True
        try:
            state = self.session.visual_state
            self.visual_original_check.setChecked(bool(state.show_original))
            display_idx = self.visual_display_mode_combo.findData(state.display_mode)
            self.visual_display_mode_combo.setCurrentIndex(display_idx if display_idx >= 0 else 0)
            self.visual_mask_enabled_check.setChecked(bool(state.mask_enabled))
            roi_items = [("None", None)] + [(roi.name, roi.name) for roi in self.session.rois]
            _set_combo_data_items(self.visual_mask_combo, roi_items, state.mask_roi)
            drift_items = [("None", None)] + [
                (self._drift_result_label(key, value), key)
                for key, value in sorted(self.drift_results.items())
                if self._drift_result_applies_to_current_experiment(value)
            ]
            _set_combo_data_items(self.visual_drift_combo, drift_items, state.drift_key)
            transform_items = [("None", None)] + [
                (self._registration_result_label(key, value), key)
                for key, value in sorted(self.registration_results.items())
            ]
            _set_combo_data_items(self.visual_transform_combo, transform_items, state.transform_key)
            idx = self.visual_transform_direction_combo.findData(state.transform_direction)
            self.visual_transform_direction_combo.setCurrentIndex(idx if idx >= 0 else 0)
            color_idx = self.visual_color_combo.findData(state.color_channel)
            self.visual_color_combo.setCurrentIndex(color_idx if color_idx >= 0 else self.visual_color_combo.findData("gray"))
            self.visual_alpha_spin.setValue(float(np.clip(state.color_alpha, 0.0, 1.0)))
            ref_idx = self.visual_reference_channel_combo.findData(state.reference_channel)
            self.visual_reference_channel_combo.setCurrentIndex(
                ref_idx if ref_idx >= 0 else self.visual_reference_channel_combo.findData("red")
            )
            mov_idx = self.visual_moving_channel_combo.findData(state.moving_channel)
            self.visual_moving_channel_combo.setCurrentIndex(
                mov_idx if mov_idx >= 0 else self.visual_moving_channel_combo.findData("green")
            )
            self.visual_reference_alpha_spin.setValue(float(np.clip(state.reference_alpha, 0.0, 1.0)))
            self.visual_moving_alpha_spin.setValue(float(np.clip(state.moving_alpha, 0.0, 1.0)))
        finally:
            self._updating_visual_state_controls = False

    def _push_visual_undo(self):
        exp_name = self.session.selected_experiment
        if not exp_name:
            return
        stack = self._visual_undo_stacks.setdefault(exp_name, [])
        stack.append(replace(self.session.visual_state))
        if len(stack) > 50:
            del stack[:-50]

    def on_visual_state_controls_changed(self, *_args):
        if getattr(self, "_updating_visual_state_controls", False) or not self.session.is_loaded:
            return
        self._clear_registration_preview(update=False)
        self._push_visual_undo()
        show_original = bool(self.visual_original_check.isChecked())
        try:
            sender = self.sender()
        except Exception:
            sender = None
        if sender is not self.visual_original_check:
            show_original = False
            self.visual_original_check.blockSignals(True)
            self.visual_original_check.setChecked(False)
            self.visual_original_check.blockSignals(False)
        mask_enabled = bool(self.visual_mask_enabled_check.isChecked())
        if sender is self.visual_mask_combo:
            mask_enabled = self.visual_mask_combo.currentData() is not None
            self.visual_mask_enabled_check.blockSignals(True)
            self.visual_mask_enabled_check.setChecked(mask_enabled)
            self.visual_mask_enabled_check.blockSignals(False)
        self.session.visual_state = VisualViewState(
            show_original=show_original,
            display_mode=str(self.visual_display_mode_combo.currentData() or "current"),
            mask_enabled=mask_enabled,
            mask_roi=self.visual_mask_combo.currentData(),
            drift_key=self.visual_drift_combo.currentData(),
            transform_key=self.visual_transform_combo.currentData(),
            transform_direction=str(self.visual_transform_direction_combo.currentData() or "moving_to_reference"),
            color_channel=str(self.visual_color_combo.currentData() or "gray"),
            color_alpha=float(self.visual_alpha_spin.value()),
            reference_channel=str(self.visual_reference_channel_combo.currentData() or "red"),
            moving_channel=str(self.visual_moving_channel_combo.currentData() or "green"),
            reference_alpha=float(self.visual_reference_alpha_spin.value()),
            moving_alpha=float(self.visual_moving_alpha_spin.value()),
        )
        self.mark_dirty("visual view state changed")
        self.update_view()

    def undo_visual_state(self):
        exp_name = self.session.selected_experiment
        stack = self._visual_undo_stacks.get(exp_name or "", [])
        if not stack:
            self.statusBar().showMessage("No visual state change to undo.")
            return
        self.session.visual_state = stack.pop()
        self.refresh_visual_state_controls()
        self.update_view()
        self.mark_dirty("visual view state changed")
        self.statusBar().showMessage("Restored previous visual view state.")

    def reset_visual_state(self):
        if not self.session.is_loaded:
            return
        self._push_visual_undo()
        self.session.visual_state = VisualViewState()
        self.refresh_visual_state_controls()
        self.update_view()
        self.mark_dirty("visual view state changed")
        self.statusBar().showMessage("Visual view state reset. Original data is shown.")

    def restore_original_image_data(self):
        if not self.session.is_loaded:
            return
        changed = self.session.restore_original_image_data()
        self.session.visual_state = replace(self.session.visual_state, show_original=True)
        self.refresh_frame_controls()
        self.refresh_registration_tab()
        self.refresh_visual_state_controls()
        self.update_view()
        if changed:
            self.mark_dirty("original image data restored")
            self.statusBar().showMessage("Restored current experiment image data from the original source snapshot.")
        else:
            self.statusBar().showMessage("Current experiment image data already matches the original source snapshot.")

    def apply_selected_roi_as_visual_mask(self):
        if not self.session.is_loaded:
            return
        roi = self._selected_roi_for_volume_action()
        if roi is None:
            self.statusBar().showMessage("Select an ROI before masking the view.")
            return
        self._push_visual_undo()
        state = replace(
            self.session.visual_state,
            show_original=False,
            mask_enabled=True,
            mask_roi=roi.name,
        )
        self.session.visual_state = state
        self.refresh_visual_state_controls()
        self.update_view()
        self.mark_dirty("visual view state changed")
        self.statusBar().showMessage(f"View masked with ROI '{roi.name}'.")

    def clear_visual_mask(self):
        if not self.session.is_loaded:
            return
        self._push_visual_undo()
        self.session.visual_state = replace(
            self.session.visual_state,
            mask_enabled=False,
            mask_roi=None,
        )
        self.refresh_visual_state_controls()
        self.update_view()
        self.mark_dirty("visual view state changed")
        self.statusBar().showMessage("View-state mask cleared.")

    def use_selected_registration_transform_in_view_state(self):
        result = self._selected_registration_result()
        if result is None:
            self.statusBar().showMessage("No registration transform is available.")
            return
        self._clear_registration_preview(update=False)
        self._push_visual_undo()
        self.session.visual_state = replace(
            self.session.visual_state,
            show_original=False,
            display_mode="current",
            transform_key=str(result["key"]),
            transform_direction="moving_to_reference",
        )
        self.refresh_visual_state_controls()
        self.update_view()
        self.mark_dirty("visual view state changed")
        self.statusBar().showMessage("Selected registration transform is active in the view state.")

    def use_selected_registration_overlay_in_view_state(self):
        result = self._selected_registration_result()
        if result is None:
            self.statusBar().showMessage("No registration transform is available.")
            return
        self._clear_registration_preview(update=False)
        self._push_visual_undo()
        self.session.visual_state = replace(
            self.session.visual_state,
            show_original=False,
            display_mode="current",
            transform_key=str(result["key"]),
            transform_direction="moving_to_reference",
        )
        self.refresh_visual_state_controls()
        if hasattr(self, "reg_reference_overlay_check") and self._registration_page_active():
            self.reg_reference_overlay_check.setChecked(True)
        self.update_view()
        self.mark_dirty("visual view state changed")
        self.statusBar().showMessage("Registration transform is active; reference overlay is enabled for this tab.")

    def refresh_roi_list(self):
        self._updating_rois = True
        self.roi_list.blockSignals(True)
        self.roi_list.clear()
        by_name = {roi.name: roi for roi in self.session.shared_rois}
        for roi in self.session.rois:
            by_name.setdefault(roi.name, roi)
        active = {roi.name for roi in self.session.rois}
        for name in sorted(by_name):
            item = QtWidgets.QListWidgetItem(name)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Checked if name in active else QtCore.Qt.Unchecked)
            self.roi_list.addItem(item)
        if by_name:
            selected = self.session.fit_settings.selected_roi
            if selected not in by_name:
                selected = next(iter(sorted(active))) if active else next(iter(sorted(by_name)))
            self.session.fit_settings.selected_roi = selected
            matches = self.roi_list.findItems(selected, QtCore.Qt.MatchExactly)
            if matches:
                self.roi_list.setCurrentItem(matches[0])
        self.roi_list.blockSignals(False)
        self._updating_rois = False
        self._refresh_roi_action_visibility()
        self.refresh_label_mask_controls()
        self.refresh_registration_tab()
        self.refresh_visual_state_controls()

    def refresh_frame_controls(self):
        _clear_layout(self.dimension_controls_layout)
        self._dimension_counter_labels = {}
        self._dimension_detail_label = None
        if not self.session.is_loaded:
            return
        view = _view_key(self.view_combo.currentText())
        controls = self.session.dimension_controls(view)
        if not controls:
            label = QtWidgets.QLabel("No extra dimensions")
            self.dimension_controls_layout.addWidget(label, 0, 0)
            return
        for row, control in enumerate(controls):
            key = str(control["key"])
            label = str(control["label"])
            size = int(control["size"])
            value = self._active_dimension_value(key)
            value = max(0, min(value, max(0, size - 1)))
            text = QtWidgets.QLabel(self._dimension_control_text(key, label, value, size))
            text.setFixedWidth(86)
            text.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            slider.setMinimum(0)
            slider.setMaximum(max(0, size - 1))
            slider.setValue(value)
            slider.valueChanged.connect(
                lambda val, k=key: self.on_dimension_changed(k, int(val))
            )
            self.dimension_controls_layout.addWidget(text, row, 0)
            self.dimension_controls_layout.addWidget(slider, row, 1)
            self._dimension_counter_labels[key] = (text, label, size)
        self.dimension_controls_layout.setColumnStretch(1, 1)
        detail = QtWidgets.QLabel("")
        detail.setWordWrap(True)
        detail.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        detail.setMinimumHeight(18)
        self.dimension_controls_layout.addWidget(detail, len(controls), 0, 1, 2)
        self._dimension_detail_label = detail
        self._update_dimension_details()

    def refresh_frame_mask_list(self):
        self._updating_frames = True
        self.frame_mask_list.clear()
        if self.session.frame_mask is None:
            self._updating_frames = False
            return
        included = self.session.frame_mask.as_array()
        for idx, value in enumerate(included):
            item = QtWidgets.QListWidgetItem(f"Frame {idx}")
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Checked if value else QtCore.Qt.Unchecked)
            self.frame_mask_list.addItem(item)
        self._updating_frames = False
        self.refresh_fit_breakout_summaries()

    def refresh_registration_tab(self, *_args):
        if getattr(self, "_updating_registration_tab", False):
            return
        if not hasattr(self, "reg_reference_experiment_combo"):
            return
        self._updating_registration_tab = True
        try:
            names = list(self.session.experiment_names) if self.session.is_loaded else []
            ref_current = self.reg_reference_experiment_combo.currentData()
            if ref_current not in names:
                ref_current = names[0] if names else None
            _set_combo_data_items(self.reg_reference_experiment_combo, [(name, name) for name in names], ref_current)

            ref_name = self.reg_reference_experiment_combo.currentData()
            _set_combo_text_items(
                self.reg_reference_map_combo,
                self._registration_map_names(ref_name),
                self.reg_reference_map_combo.currentText(),
            )
            self._refresh_registration_frame_spin(
                ref_name,
                self.reg_reference_source_combo.currentData(),
                self.reg_reference_frame_spin,
            )
            ref_is_map = self.reg_reference_source_combo.currentData() == "map"
            self.reg_reference_map_combo.setEnabled(bool(ref_is_map and self.reg_reference_map_combo.count()))
            self.reg_reference_frame_spin.setEnabled(not ref_is_map)
            if hasattr(self, "reg_current_moving_label"):
                self.reg_current_moving_label.setText(self._current_registration_moving_label())
            self.refresh_registration_transform_combo()
            registration_roi_items = _registration_roi_choice_items(
                self.session.rois,
                self.session.shared_rois,
                include_none=True,
            )
            roi_names = [
                str(value)
                for _label, value in registration_roi_items
                if value is not None
            ]
            _set_combo_data_items(
                self.reg_reference_metric_mask_combo,
                registration_roi_items,
                self.reg_reference_metric_mask_combo.currentData(),
            )
            _set_combo_data_items(
                self.reg_moving_metric_mask_combo,
                registration_roi_items,
                self.reg_moving_metric_mask_combo.currentData(),
            )
            if hasattr(self, "reg_drift_mask_combo"):
                _set_combo_data_items(
                    self.reg_drift_mask_combo,
                    registration_roi_items,
                    self.reg_drift_mask_combo.currentData(),
                )
            _set_combo_text_items(
                self.reg_mask_roi_combo,
                roi_names,
                self.reg_mask_roi_combo.currentText() or self.session.fit_settings.selected_roi,
            )
        finally:
            self._updating_registration_tab = False

    def refresh_registration_transform_combo(self):
        if not hasattr(self, "reg_available_transform_combo"):
            return
        current = self.reg_available_transform_combo.currentData()
        items = [
            (self._registration_result_label(key, value), key)
            for key, value in sorted(self.registration_results.items())
        ]
        _set_combo_data_items(self.reg_available_transform_combo, items, current)
        self.refresh_visual_state_controls()

    def _registration_map_names(self, experiment_name: str | None) -> list[str]:
        if not experiment_name or not self.session.data:
            return []
        exp_data = self.session.data.get("experiments", {}).get(str(experiment_name)) or {}
        return list((exp_data.get("fitted_maps") or {}).keys())

    def _refresh_registration_frame_spin(self, experiment_name: str | None, source: str | None, spin) -> None:
        try:
            count = self._registration_frame_count(experiment_name, source)
        except Exception:
            count = 1
        count = max(1, int(count))
        spin.setMaximum(max(0, count - 1))
        if spin.value() >= count:
            spin.setValue(max(0, count - 1))

    def _registration_frame_count(self, experiment_name: str | None, source: str | None) -> int:
        if not experiment_name or not self.session.data:
            return 1
        exp_data = self.session.data.get("experiments", {}).get(str(experiment_name)) or {}
        if source == "reconstruction":
            arr = np.asarray((exp_data.get("processed") or {}).get("raw_reconstruction"))
        else:
            arr = np.asarray(exp_data.get("mr_image"))
        dims = {str(k): int(v) for k, v in dict(exp_data.get("dims") or {}).items()}
        if source == "map":
            return 1
        if dims:
            try:
                return int(reshape_dims_to_frames(arr, dims).shape[0])
            except Exception:
                _LOG.debug("Ignored optional GUI operation failure", exc_info=True)
        return int(arr.shape[0]) if arr.ndim >= 3 else 1

    def _registration_source_image(
        self,
        experiment_name: str | None,
        source: str | None,
        frame_index: int,
        map_name: str | None,
    ) -> np.ndarray:
        if not experiment_name or not self.session.data:
            raise RuntimeError("No experiment is selected for registration")
        exp_data = self.session.data.get("experiments", {}).get(str(experiment_name)) or {}
        source_key = str(source or "image")
        if source_key == "map":
            maps = exp_data.get("fitted_maps") or {}
            selected = map_name or (next(iter(maps), None))
            if not selected or selected not in maps:
                raise RuntimeError(f"Experiment {experiment_name} has no selected fitted map")
            arr, _unit = _quantity_array_and_unit(maps[selected])
            return _first_spatial_image(arr)
        if source_key == "reconstruction":
            processed = exp_data.get("processed") or {}
            if processed.get("raw_reconstruction") is None:
                raise RuntimeError(f"Experiment {experiment_name} has no MRax reconstruction")
            arr = np.asarray(processed["raw_reconstruction"])
        else:
            if (
                str(experiment_name) == str(self.session.selected_experiment)
                and bool(self.session.visual_state.show_original)
            ):
                original = exp_data.get("original_mr_image")
                arr = np.asarray(original if original is not None else exp_data.get("mr_image"))
            else:
                arr = np.asarray(exp_data.get("mr_image"))
        dims = {str(k): int(v) for k, v in dict(exp_data.get("dims") or {}).items()}
        if dims:
            try:
                frames = np.asarray(reshape_dims_to_frames(arr, dims))
                idx = max(0, min(int(frame_index), int(frames.shape[0]) - 1))
                return np.asarray(frames[idx])
            except Exception:
                _LOG.debug("Ignored optional GUI operation failure", exc_info=True)
        if arr.ndim <= 2:
            return np.asarray(arr)
        idx = max(0, min(int(frame_index), int(arr.shape[0]) - 1))
        return _first_spatial_image(arr[idx])

    def _current_registration_moving_label(self) -> str:
        if not self.session.is_loaded:
            return "No current image"
        view = _view_key(self.view_combo.currentText())
        pieces = [str(self.session.selected_experiment or "current"), view.replace("_", " ")]
        if view == "map" and self.session.active_map:
            pieces.append(str(self.session.active_map))
        pieces.append(f"frame {int(self.session.active_frame)}")
        indices = {
            key: value
            for key, value in self._current_slice_indices().items()
            if key not in {"frame"}
        }
        if indices:
            pieces.append(", ".join(f"{key}={value}" for key, value in sorted(indices.items())))
        return " | ".join(pieces)

    def _current_registration_moving_image(self) -> tuple[np.ndarray, dict[str, Any]]:
        if not self.session.is_loaded:
            raise RuntimeError("No current moving image is loaded")
        view = _view_key(self.view_combo.currentText())
        if view == "k_space":
            image = self.session.kspace_image()
            source = "k_space"
        elif view == "mrax_reconstruction":
            image = self.session.reconstruction_image()
            source = "mrax_reconstruction"
        elif view == "vendor_vs_mrax":
            raise RuntimeError("The side-by-side vendor/MRax comparison view cannot be used as a moving registration image")
        elif view == "vendor_mrax_diff":
            vendor, mrax = _matched_2d_pair(
                self.session.frame_image(original=bool(self.session.visual_state.show_original)),
                self.session.reconstruction_image(),
            )
            image = vendor - mrax
            source = "vendor_mrax_diff"
        elif view == "map":
            image = self.session.map_image()
            source = "map"
        else:
            image = self.session.frame_image(original=bool(self.session.visual_state.show_original))
            source = "image"
        metadata = {
            "moving_experiment": str(self.session.selected_experiment),
            "moving_source": source,
            "moving_view": view,
            "moving_frame": int(self.session.active_frame),
            "moving_map": self.session.active_map if view == "map" else "",
            "moving_indices": dict(self._current_slice_indices()),
            "moving_source_policy": "current_view",
            "moving_uses_original_image": bool(self.session.visual_state.show_original and source in {"image", "vendor_mrax_diff"}),
        }
        return np.asarray(_first_spatial_image(image)), metadata

    def _selected_registration_images(self) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        ref_name = self.reg_reference_experiment_combo.currentData()
        fixed = self._registration_source_image(
            ref_name,
            self.reg_reference_source_combo.currentData(),
            self.reg_reference_frame_spin.value(),
            self.reg_reference_map_combo.currentText(),
        )
        moving, moving_metadata = self._current_registration_moving_image()
        metadata = {
            "reference_experiment": str(ref_name),
            "reference_source": self.reg_reference_source_combo.currentData(),
            "reference_frame": int(self.reg_reference_frame_spin.value()),
            "reference_map": self.reg_reference_map_combo.currentText(),
            **moving_metadata,
        }
        return np.asarray(fixed), np.asarray(moving), metadata

    def _registration_options_from_controls(self) -> RegistrationOptions:
        shrink = parse_sequence(self.reg_shrink_edit.text(), int) or (1,)
        sigmas = parse_sequence(self.reg_smoothing_edit.text(), float) or (0.0,)
        return options_from_mapping(
            {
                "transform": self.reg_transform_combo.currentData(),
                "metric": self.reg_metric_combo.currentData(),
                "optimizer": self.reg_optimizer_combo.currentData(),
                "interpolator": self.reg_interpolator_combo.currentData(),
                "initializer": self.reg_initializer_combo.currentData(),
                "sampling_strategy": self.reg_sampling_combo.currentData(),
                "sampling_percentage": self.reg_sampling_percentage_spin.value(),
                "histogram_bins": self.reg_histogram_bins_spin.value(),
                "learning_rate": self.reg_learning_rate_spin.value(),
                "min_step": self.reg_min_step_spin.value(),
                "iterations": self.reg_iterations_spin.value(),
                "convergence_minimum": self.reg_convergence_spin.value(),
                "convergence_window": self.reg_convergence_window_spin.value(),
                "shrink_factors": shrink,
                "smoothing_sigmas": sigmas,
            }
        )

    def _registration_roi_mask_for_image(
        self,
        roi_name: str | None,
        image_shape: Sequence[int],
        role: str,
    ) -> np.ndarray | None:
        if not roi_name:
            return None
        roi = _registration_roi_by_name(
            str(roi_name),
            self.session.rois,
            self.session.shared_rois,
        )
        if roi is None:
            raise RuntimeError(f"Select an available ROI for the {role} registration mask")
        target_shape = tuple(int(v) for v in image_shape)
        mask = np.asarray(roi.mask, dtype=bool)
        if mask.ndim > len(target_shape):
            mask = mask_slice_for_display(mask, dims=roi.dims, indices=self._current_slice_indices())
        if tuple(int(v) for v in mask.shape) != target_shape:
            raise RuntimeError(
                f"{role.capitalize()} mask ROI '{roi.name}' has shape {mask.shape}; "
                f"selected {role} image has shape {target_shape}"
            )
        if not np.any(mask):
            raise RuntimeError(f"{role.capitalize()} mask ROI '{roi.name}' is empty")
        return np.asarray(mask, dtype=bool)

    def _registration_overlay_from_controls(self, fixed: np.ndarray, moving: np.ndarray) -> np.ndarray:
        fixed_channel = "red"
        moving_channel = "green"
        fixed_alpha = 1.0
        moving_alpha = 1.0
        if hasattr(self, "reg_reference_channel_combo"):
            fixed_channel = str(self.reg_reference_channel_combo.currentData() or fixed_channel)
            moving_channel = str(self.reg_moving_channel_combo.currentData() or moving_channel)
            fixed_alpha = float(self.reg_reference_alpha_spin.value())
            moving_alpha = float(self.reg_moving_alpha_spin.value())
        return registration_overlay(
            fixed,
            moving,
            fixed_channel=fixed_channel,
            moving_channel=moving_channel,
            fixed_alpha=fixed_alpha,
            moving_alpha=moving_alpha,
        )

    def _registration_page_active(self) -> bool:
        if not hasattr(self, "feature_stack"):
            return False
        idx = int(self.feature_stack.currentIndex())
        labels = getattr(self, "feature_page_labels", [])
        return 0 <= idx < len(labels) and labels[idx] == "Registration"

    def _registration_reference_overlay_enabled(self) -> bool:
        return (
            self._registration_page_active()
            and hasattr(self, "reg_reference_overlay_check")
            and bool(self.reg_reference_overlay_check.isChecked())
        )

    def _registration_reference_overlay_blend(self) -> float:
        if not hasattr(self, "reg_overlay_blend_slider"):
            return 0.6
        return float(np.clip(self.reg_overlay_blend_slider.value() / 100.0, 0.0, 1.0))

    def _refresh_registration_overlay_blend_label(self):
        if not hasattr(self, "reg_overlay_blend_label"):
            return
        foreground = int(round(self._registration_reference_overlay_blend() * 100.0))
        background = 100 - foreground
        self.reg_overlay_blend_label.setText(f"reference {background}% / current view {foreground}%")

    def _disable_registration_reference_overlay(self, *, update: bool = True) -> None:
        if not hasattr(self, "reg_reference_overlay_check"):
            return
        if not self.reg_reference_overlay_check.isChecked():
            return
        self.reg_reference_overlay_check.blockSignals(True)
        self.reg_reference_overlay_check.setChecked(False)
        self.reg_reference_overlay_check.blockSignals(False)
        if update and self.session.is_loaded:
            self.update_view()

    def on_registration_overlay_controls_changed(self, *_args):
        self._refresh_registration_overlay_blend_label()
        if hasattr(self, "reg_reference_overlay_check") and self.reg_reference_overlay_check.isChecked():
            if not self._registration_page_active():
                self._disable_registration_reference_overlay(update=False)
                return
        if self.session.is_loaded:
            self.update_view()

    def _apply_registration_reference_overlay(
        self,
        foreground: np.ndarray,
        view: str,
        display_levels: tuple[float, float] | None,
    ) -> tuple[np.ndarray, tuple[float, float] | None]:
        if not self._registration_reference_overlay_enabled():
            return np.asarray(foreground), display_levels
        try:
            reference = self._registration_source_image(
                self.reg_reference_experiment_combo.currentData(),
                self.reg_reference_source_combo.currentData(),
                self.reg_reference_frame_spin.value(),
                self.reg_reference_map_combo.currentText(),
            )
            blended = _blend_reference_background(
                reference,
                foreground,
                foreground_fraction=self._registration_reference_overlay_blend(),
                reference_channel=str(self.reg_reference_channel_combo.currentData() or "red"),
                foreground_channel=str(self.reg_moving_channel_combo.currentData() or "green"),
            )
        except Exception as exc:
            self.statusBar().showMessage(f"Reference overlay skipped for {view}: {exc}")
            return np.asarray(foreground), display_levels
        return blended, None

    def preview_selected_registration_overlay(self):
        if not self.session.is_loaded:
            return
        try:
            fixed, moving, metadata = self._selected_registration_images()
            fixed_plane = _registration_preview_plane(fixed)
            moving_plane = _registration_preview_plane(moving)
        except Exception:
            self.show_error("Could not preview registration overlay", traceback.format_exc())
            return
        if hasattr(self, "reg_reference_overlay_check"):
            self.reg_reference_overlay_check.setChecked(True)
        self.registration_diagnostics_text.setPlainText(
            json.dumps(
                {
                    "metadata": metadata,
                    "overlay": {
                        "reference_background": True,
                        "foreground_fraction": self._registration_reference_overlay_blend(),
                        "reference_channel": self.reg_reference_channel_combo.currentData(),
                        "current_channel": self.reg_moving_channel_combo.currentData(),
                    },
                    "metrics": alignment_metrics(fixed_plane, moving_plane),
                },
                indent=2,
            )
        )
        self.update_view()
        self.statusBar().showMessage("Reference overlay enabled for the current view.")

    def run_simpleitk_registration(self):
        if self.is_busy or not self.session.is_loaded:
            return
        try:
            fixed, moving, metadata = self._selected_registration_images()
            fixed_mask = self._registration_roi_mask_for_image(
                self.reg_reference_metric_mask_combo.currentData(),
                fixed.shape,
                "reference",
            )
            moving_mask = self._registration_roi_mask_for_image(
                self.reg_moving_metric_mask_combo.currentData(),
                moving.shape,
                "moving",
            )
            metadata = {
                **metadata,
                "reference_metric_mask": self.reg_reference_metric_mask_combo.currentData(),
                "moving_metric_mask": self.reg_moving_metric_mask_combo.currentData(),
            }
            options = self._registration_options_from_controls()
        except Exception:
            self.show_error("Could not configure registration", traceback.format_exc())
            return
        self.run_guarded_task(
            lambda progress: self._run_registration_for_task(
                fixed,
                moving,
                metadata,
                options,
                fixed_mask,
                moving_mask,
                progress,
            ),
            self.on_registration_finished,
            "Running SimpleITK registration...",
            error_title="Could not run registration",
        )

    def _run_registration_for_task(
        self,
        fixed: np.ndarray,
        moving: np.ndarray,
        metadata: dict[str, Any],
        options: RegistrationOptions,
        fixed_mask: np.ndarray | None,
        moving_mask: np.ndarray | None,
        progress: Callable[[int, int, str], None],
    ) -> dict[str, Any]:
        progress(0, 3, "Configuring SimpleITK registration")
        before = alignment_metrics(fixed, moving, mask=fixed_mask)
        progress(1, 3, "Optimizing transform")
        result = run_registration(fixed, moving, options, fixed_mask=fixed_mask, moving_mask=moving_mask)
        progress(2, 3, "Preparing registration preview")
        corrected = np.asarray(result.corrected)
        after = alignment_metrics(fixed, corrected, mask=fixed_mask)
        key = self._new_registration_key(metadata, "sitk")
        progress(3, 3, "Registration complete")
        return {
            "key": key,
            "kind": "simpleitk",
            "fixed": np.asarray(fixed),
            "moving": np.asarray(moving),
            "corrected": corrected,
            "transform": result.transform,
            "metadata": metadata,
            "options": options.__dict__,
            "diagnostics": {
                "metric_value": result.diagnostics.metric_value,
                "optimizer_stop_condition": result.diagnostics.optimizer_stop_condition,
                "iteration_metrics": list(result.diagnostics.iteration_metrics),
                "transform_parameters": list(result.diagnostics.transform_parameters),
                "pre": before,
                "post": after,
                "reference_mask_pixels": None if fixed_mask is None else int(np.count_nonzero(fixed_mask)),
                "moving_mask_pixels": None if moving_mask is None else int(np.count_nonzero(moving_mask)),
                "transform": transform_metadata(result.transform),
            },
        }

    def on_registration_finished(self, result: dict[str, Any]):
        key = str(result["key"])
        self.registration_results[key] = result
        self.refresh_registration_transform_combo()
        self._show_registration_result_preview(result)
        self.registration_diagnostics_text.setPlainText(json.dumps(_registration_result_summary(result), indent=2))
        self.mark_dirty("registration transform added")
        self.statusBar().showMessage(f"Stored registration transform: {self._registration_result_label(key, result)}")

    def preview_manual_registration(self):
        if not self.session.is_loaded:
            return
        try:
            result = self._manual_registration_result(store=False)
        except Exception:
            self.show_error("Could not preview manual registration", traceback.format_exc())
            return
        self._show_registration_result_preview(result)
        self.registration_diagnostics_text.setPlainText(json.dumps(_registration_result_summary(result), indent=2))

    def store_manual_registration(self):
        if not self.session.is_loaded:
            return
        try:
            result = self._manual_registration_result(store=True)
        except Exception:
            self.show_error("Could not store manual transform", traceback.format_exc())
            return
        key = str(result["key"])
        self.registration_results[key] = result
        self.refresh_registration_transform_combo()
        self._show_registration_result_preview(result)
        self.registration_diagnostics_text.setPlainText(json.dumps(_registration_result_summary(result), indent=2))
        self.mark_dirty("manual registration transform added")
        self.statusBar().showMessage(f"Stored manual transform: {self._registration_result_label(key, result)}")

    def _manual_registration_result(self, *, store: bool) -> dict[str, Any]:
        fixed, moving, metadata = self._selected_registration_images()
        transform = manual_affine_transform(
            fixed.shape,
            shift_x=float(self.reg_shift_x_spin.value()),
            shift_y=float(self.reg_shift_y_spin.value()),
            rotation_degrees=float(self.reg_rotation_spin.value()),
            scale_x=float(self.reg_scale_x_spin.value()),
            scale_y=float(self.reg_scale_y_spin.value()),
        )
        corrected = apply_transform_to_image(moving, fixed, transform, interpolator="linear")
        before = alignment_metrics(fixed, moving)
        after = alignment_metrics(fixed, corrected)
        key = self._new_registration_key(metadata, "manual") if store else "__manual_preview__"
        return {
            "key": key,
            "kind": "manual",
            "fixed": np.asarray(fixed),
            "moving": np.asarray(moving),
            "corrected": np.asarray(corrected),
            "transform": transform,
            "metadata": metadata,
            "manual": {
                "shift_x": float(self.reg_shift_x_spin.value()),
                "shift_y": float(self.reg_shift_y_spin.value()),
                "rotation_degrees": float(self.reg_rotation_spin.value()),
                "scale_x": float(self.reg_scale_x_spin.value()),
                "scale_y": float(self.reg_scale_y_spin.value()),
            },
            "diagnostics": {
                "pre": before,
                "post": after,
                "transform": transform_metadata(transform),
            },
        }

    def reset_manual_registration_controls(self):
        self.reg_shift_x_spin.setValue(0.0)
        self.reg_shift_y_spin.setValue(0.0)
        self.reg_rotation_spin.setValue(0.0)
        self.reg_scale_x_spin.setValue(1.0)
        self.reg_scale_y_spin.setValue(1.0)

    def run_mrax_drift_correction(self):
        if self.is_busy or not self.session.is_loaded:
            return
        try:
            experiment, dataset = self._current_drift_inputs()
            data = np.asarray(dataset.data)
            frames = np.asarray(reshape_dims_to_frames(data, experiment.dims))
            if frames.ndim not in (3, 4):
                raise RuntimeError("MRax drift correction requires a frame stack with 2D or 3D spatial frames")
            frame_shape = tuple(int(v) for v in frames.shape[1:])
            options = self._drift_options_from_controls()
            mask = self._registration_roi_mask_for_image(
                self.reg_drift_mask_combo.currentData(),
                frame_shape,
                "drift",
            )
            self._validate_drift_mask_for_options(mask, options)
            metadata = {
                "experiment": str(self.session.selected_experiment),
                "frame_shape": list(frame_shape),
                "frame_count": int(frames.shape[0]),
                "active_frame": int(self.session.active_frame),
                "mask": self.reg_drift_mask_combo.currentData(),
                "options": _drift_options_to_dict(options),
            }
        except Exception:
            self.show_error("Could not configure MRax drift correction", traceback.format_exc())
            return
        self.run_guarded_task(
            lambda progress: self._run_mrax_drift_correction_for_task(
                experiment,
                dataset,
                options,
                mask,
                metadata,
                progress,
            ),
            self.on_mrax_drift_correction_finished,
            "Running MRax drift correction...",
            error_title="Could not run MRax drift correction",
        )

    def _current_drift_inputs(self) -> tuple[Experiment, DataSet]:
        entry = self.session.current_study_entry()
        if entry is not None:
            return entry[0], entry[2]
        return self.session.build_experiment(), self.session.build_dataset()

    def _drift_options_from_controls(self) -> DriftCorrectionOptions:
        reference = str(self.reg_drift_reference_combo.currentData() or "first")
        reference_value: str | int
        if reference == "current":
            reference_value = int(self.session.active_frame)
        else:
            reference_value = reference
        dimensions_text = str(self.reg_drift_dimensions_edit.text() or "").strip()
        dimensions = _parse_name_sequence(dimensions_text) if dimensions_text else None
        return DriftCorrectionOptions(
            transform=str(self.reg_drift_transform_combo.currentData() or "translation"),
            metric=str(self.reg_drift_metric_combo.currentData() or "ncc"),
            reference=reference_value,
            dimensions=dimensions,
            interpolation=str(self.reg_drift_interpolation_combo.currentData() or "linear"),
            optimizer=str(self.reg_drift_optimizer_combo.currentData() or "adam"),
            step_size=float(self.reg_drift_step_size_spin.value()),
            num_steps=int(self.reg_drift_steps_spin.value()),
            fit_intercept=bool(self.reg_drift_fit_intercept_check.isChecked()),
        )

    def _validate_drift_mask_for_options(
        self,
        mask: np.ndarray | None,
        options: DriftCorrectionOptions,
    ) -> None:
        if mask is None:
            return
        if str(options.metric or "").lower().replace("-", "_") != "mutual_information":
            return
        pixel_count = int(np.count_nonzero(mask))
        if pixel_count < 16:
            raise RuntimeError(
                "Mutual-information drift correction needs a larger mask. "
                f"The selected mask contains {pixel_count} pixel(s); use at least 16, "
                "or switch the drift metric to normalized cross-correlation or negative MSE."
            )

    def _run_mrax_drift_correction_for_task(
        self,
        experiment: Experiment,
        dataset: DataSet,
        options: DriftCorrectionOptions,
        mask: np.ndarray | None,
        metadata: dict[str, Any],
        progress: Callable[[int, int, str], None],
    ) -> dict[str, Any]:
        progress(0, 2, "Fitting MRax linear drift model")
        result = run_drift_correction(experiment, dataset, options, mask=mask)
        progress(1, 2, "Preparing drift-corrected data")
        key = self._new_drift_key(metadata)
        corrected = np.asarray(result.corrected)
        summary = _linear_drift_result_summary(result, metadata)
        failed_groups = _failed_linear_drift_groups(result)
        if failed_groups:
            raise RuntimeError(
                "MRax drift correction stopped before applying changes because "
                f"{len(failed_groups)} group(s) had non-finite optimizer diagnostics. "
                "The original data was not changed. Try normalized cross-correlation, "
                "negative MSE, a lower step size, fewer transform degrees of freedom, "
                "or a larger mask.\n\n"
                + "\n".join(failed_groups[:6])
            )
        if not np.any(np.isfinite(corrected)):
            raise RuntimeError(
                "MRax drift correction produced no finite output values. "
                "The original data was not changed."
            )
        progress(2, 2, "MRax drift correction complete")
        return {
            "key": key,
            "kind": "mrax_linear_drift",
            "corrected": corrected,
            "metadata": metadata,
            "summary": summary,
        }

    def on_mrax_drift_correction_finished(self, result: dict[str, Any]):
        key = str(result["key"])
        self.drift_results[key] = result
        self._push_visual_undo()
        self.session.visual_state = replace(
            self.session.visual_state,
            show_original=False,
            drift_key=key,
        )
        self.refresh_frame_controls()
        self.refresh_registration_tab()
        self.refresh_visual_state_controls()
        self.update_view()
        self.registration_diagnostics_text.setPlainText(json.dumps(result["summary"], indent=2))
        self.mark_dirty("MRax drift correction view state changed")
        self.statusBar().showMessage(
            f"Stored MRax linear drift correction for {result['metadata'].get('experiment')} in the view state."
        )

    def _new_drift_key(self, metadata: Mapping[str, Any]) -> str:
        experiment = _safe_filename(str(metadata.get("experiment") or "experiment"))
        base = f"{experiment}_mrax_linear_drift"
        key = base
        index = 2
        while key in self.drift_results:
            key = f"{base}_{index}"
            index += 1
        return key

    def _drift_result_label(self, key: str, result: Mapping[str, Any]) -> str:
        meta = dict(result.get("metadata") or {})
        options = dict(meta.get("options") or {})
        metric = options.get("metric") or "drift"
        return f"{meta.get('experiment', '?')} ({metric}, {key})"

    def _drift_result_applies_to_current_experiment(self, result: Mapping[str, Any]) -> bool:
        selected = self.session.selected_experiment
        experiment = (result.get("metadata") or {}).get("experiment")
        return selected is not None and (experiment is None or str(experiment) == str(selected))

    def preview_selected_registration_transform(self):
        result = self._selected_registration_result()
        if result is None:
            self.statusBar().showMessage("No registration transform is available.")
            return
        self._show_registration_result_preview(result)
        self.registration_diagnostics_text.setPlainText(json.dumps(_registration_result_summary(result), indent=2))

    def _selected_registration_result(self, transform_key: str | None = None) -> dict[str, Any] | None:
        key = transform_key or self.reg_available_transform_combo.currentData()
        if key is None:
            return None
        return self.registration_results.get(str(key))

    def _show_registration_result_preview(self, result: Mapping[str, Any]) -> None:
        self._registration_preview_result = dict(result)
        if self.session.is_loaded:
            self.update_view()
        self.statusBar().showMessage(
            "Registration transform preview is active. Use the overlay switch separately to compare with the reference."
        )

    def _active_registration_preview_result(self) -> dict[str, Any] | None:
        if not self._registration_page_active():
            return None
        result = self._registration_preview_result
        if not result:
            return None
        transform = result.get("transform")
        if transform is None:
            return None
        return result

    def _clear_registration_preview(self, *, update: bool = True) -> None:
        if self._registration_preview_result is None:
            return
        self._registration_preview_result = None
        if update and self.session.is_loaded:
            self.update_view()

    def preview_registration_mask_transform(self, *, transform_key: str | None = None, direction: str | None = None):
        try:
            roi, transformed, target_image, result = self._transformed_registration_roi(
                transform_key=transform_key,
                direction=direction,
            )
        except Exception:
            self.show_error("Could not preview mask transform", traceback.format_exc())
            return
        self._show_mask_transform_preview(target_image, transformed.mask)
        self.registration_diagnostics_text.setPlainText(
            json.dumps(
                {
                    "roi": roi.name,
                    "transformed_roi": transformed.name,
                    "transform": self._registration_result_label(str(result["key"]), result),
                    "direction": direction or self.reg_mask_direction_combo.currentData(),
                    "source_pixels": int(np.count_nonzero(roi.mask)),
                    "transformed_pixels": int(np.count_nonzero(transformed.mask)),
                },
                indent=2,
            )
        )

    def add_transformed_registration_roi(self):
        try:
            _roi, transformed, _target_image, result = self._transformed_registration_roi()
        except Exception:
            self.show_error("Could not add transformed ROI", traceback.format_exc())
            return
        self.session.add_roi(transformed)
        self.session.fit_settings.selected_roi = transformed.name
        self.refresh_roi_list()
        self.update_view(preserve_image_view=True)
        self.mark_dirty("ROI set changed")
        self.statusBar().showMessage(
            f"Added transformed ROI '{transformed.name}' using {self._registration_result_label(str(result['key']), result)}."
        )

    def _transformed_registration_roi(
        self,
        *,
        transform_key: str | None = None,
        direction: str | None = None,
        roi_name: str | None = None,
    ) -> tuple[ROI, ROI, np.ndarray, dict[str, Any]]:
        result = self._selected_registration_result(transform_key)
        if result is None:
            raise RuntimeError("No stored registration transform is selected")
        selected_roi_name = roi_name or self.reg_mask_roi_combo.currentText()
        roi = _registration_roi_by_name(
            selected_roi_name,
            self.session.rois,
            self.session.shared_rois,
        )
        if roi is None:
            raise RuntimeError("Select an ROI to transform")
        direction_key = str(direction or self.reg_mask_direction_combo.currentData() or "reference_to_moving")
        inverse = direction_key == "reference_to_moving"
        reference = np.asarray(result["moving"] if inverse else result["fixed"])
        mask = np.asarray(roi.mask)
        if mask.ndim > reference.ndim:
            mask = mask_slice_for_display(mask, dims=roi.dims, indices=self._current_slice_indices())
        transformed_mask = resample_mask_with_transform(mask, reference, result["transform"], inverse=inverse, output_dtype=bool)
        suffix = "to_moving" if inverse else "to_reference"
        registration_metadata = dict(result.get("metadata") or {})
        target_experiment = registration_metadata.get(
            "moving_experiment" if inverse else "reference_experiment"
        )
        transformed = ROI(
            _unique_roi_name(
                _registration_roi_union(self.session.rois, self.session.shared_rois),
                f"{roi.name}_{suffix}",
            ),
            np.asarray(transformed_mask, dtype=bool),
            dims=tuple(self.session.spatial_dim_names[: np.asarray(transformed_mask).ndim]),
            metadata={
                **dict(roi.metadata),
                "shape": "mask",
                "generated_by": "mrax.gui.registration",
                "source_experiment": target_experiment or roi.metadata.get("source_experiment"),
                "source_roi": roi.name,
                "registration_transform": str(result["key"]),
                "registration_direction": direction_key,
            },
        )
        return roi, transformed, reference, result

    def _show_mask_transform_preview(self, image: np.ndarray, mask: np.ndarray) -> None:
        display = _registration_preview_plane(image)
        mask = _registration_preview_plane(mask).astype(bool)
        self._displayed_image = display
        self._displayed_image_shape = tuple(int(v) for v in display.shape[-2:])
        self._displayed_view_key = "registration_mask_preview"
        self.image_view.setImage(display, autoLevels=True, autoRange=True)
        view_box = self.image_view.getView()
        for old in self.overlay_items:
            try:
                view_box.removeItem(old)
            except Exception:
                _LOG.debug("Ignored optional GUI operation failure", exc_info=True)
        self.overlay_items.clear()
        rgba = np.zeros(np.asarray(mask).shape + (4,), dtype=np.ubyte)
        rgba[..., 0] = 255
        rgba[..., 1] = 180
        rgba[..., 3] = np.asarray(mask, dtype=bool).astype(np.ubyte) * 110
        item = pg.ImageItem(rgba)
        item.setZValue(20)
        view_box.addItem(item)
        self.overlay_items.append(item)
        self.statusBar().showMessage("Mask-transform preview: transformed mask is shown in orange.")

    def _registration_result_label(self, key: str, result: Mapping[str, Any]) -> str:
        meta = dict(result.get("metadata") or {})
        return (
            f"{meta.get('moving_experiment', '?')} -> {meta.get('reference_experiment', '?')} "
            f"({result.get('kind', 'transform')}, {key})"
        )

    def _new_registration_key(self, metadata: Mapping[str, Any], kind: str) -> str:
        moving = _safe_filename(str(metadata.get("moving_experiment") or "moving"))
        reference = _safe_filename(str(metadata.get("reference_experiment") or "reference"))
        base = f"{moving}_to_{reference}_{kind}"
        key = base
        index = 2
        while key in self.registration_results:
            key = f"{base}_{index}"
            index += 1
        return key

    def on_experiment_changed(self, name: str):
        if not name or not self.session.is_loaded:
            return
        prior_view = self.view_combo.currentText()
        try:
            self.session.select_experiment(name)
        except Exception:
            self.show_error("Could not select experiment", traceback.format_exc())
            return
        if prior_view == "Map" and not self.session.map_names:
            self.view_combo.blockSignals(True)
            self.view_combo.setCurrentText("Image")
            self.view_combo.blockSignals(False)
        self.populate_from_session()

    def on_map_changed(self, name: str):
        if name and self.view_combo.currentText() == "Map":
            self.session.active_map = name
            self.refresh_registration_tab()
            self.update_view()

    def on_roi_changed(self, name: str):
        if name:
            self.session.fit_settings.selected_roi = name
            self.update_view(preserve_image_view=True)
        self._refresh_roi_action_visibility()
        self.refresh_label_mask_controls()

    def on_roi_selection_changed(self):
        self._refresh_roi_action_visibility()
        self.refresh_label_mask_controls()

    def on_roi_item_check_changed(self, item):
        if getattr(self, "_updating_rois", False) or item is None:
            return
        name = item.text()
        if item.checkState() == QtCore.Qt.Checked:
            self.session.enable_shared_roi(name)
            self.session.fit_settings.selected_roi = name
        else:
            self.session.remove_roi(name)
            if self.session.fit_settings.selected_roi == name:
                self.session.fit_settings.selected_roi = self.session.rois[0].name if self.session.rois else None
        self.mark_dirty("ROI set changed")
        self.update_view(preserve_image_view=True)
        self._refresh_roi_action_visibility()
        self.refresh_label_mask_controls()

    def refresh_label_mask_controls(self):
        section = getattr(self, "_label_mask_controls_section", None)
        combo = getattr(self, "label_mask_label_combo", None)
        spin = getattr(self, "label_mask_value_spin", None)
        if section is None or combo is None or spin is None:
            return
        roi = self.session.roi() if self.session.is_loaded else None
        rows = _label_rows_for_roi(roi)
        visible = bool(rows)
        self._updating_label_mask_controls = True
        combo.blockSignals(True)
        try:
            combo.clear()
            for row in rows:
                label = int(row["label"])
                name = str(row.get("name") or f"label {label}")
                count = int(row.get("voxel_count") or 0)
                combo.addItem(f"{label}: {name} ({count} px)", label)
            if rows:
                active_label = int((roi.metadata or {}).get("active_label") or rows[0]["label"])
                idx = combo.findData(active_label)
                combo.setCurrentIndex(idx if idx >= 0 else 0)
                current = int(combo.currentData() or rows[0]["label"])
                spin.setValue(current)
                if getattr(self, "brush_label_spin", None) is not None:
                    self.brush_label_spin.setValue(current)
        finally:
            combo.blockSignals(False)
            self._updating_label_mask_controls = False
        section.setVisible(visible)

    def on_label_mask_label_changed(self, *_args):
        if getattr(self, "_updating_label_mask_controls", False):
            return
        label = self.label_mask_label_combo.currentData() if self.label_mask_label_combo is not None else None
        if label is None:
            return
        value = max(1, int(label))
        self.label_mask_value_spin.setValue(value)
        if getattr(self, "brush_label_spin", None) is not None:
            self.brush_label_spin.setValue(value)

    def change_selected_label_value(self):
        if not self.session.is_loaded:
            return
        roi = self.session.roi()
        if roi is None or not _roi_is_label_mask(roi):
            self.statusBar().showMessage("Select a label-mask ROI first.")
            return
        old_label = self.label_mask_label_combo.currentData() if self.label_mask_label_combo is not None else None
        if old_label is None:
            self.statusBar().showMessage("Select a label value to change.")
            return
        old_value = int(old_label)
        new_value = max(1, int(self.label_mask_value_spin.value()))
        if old_value == new_value:
            self.statusBar().showMessage(f"Label {old_value} is already selected.")
            return
        mask = np.asarray(roi.mask).copy()
        selected = mask == old_value
        if not np.any(selected):
            self.statusBar().showMessage(f"Label {old_value} is not present in ROI '{roi.name}'.")
            self.refresh_label_mask_controls()
            return
        dtype = np.promote_types(mask.dtype, np.min_scalar_type(new_value))
        if dtype != mask.dtype:
            mask = mask.astype(dtype, copy=False)
        mask[selected] = new_value
        meta = dict(roi.metadata)
        meta["shape"] = "label_mask"
        meta["mask_type"] = "label"
        meta["active_label"] = new_value
        relabeled_rows = []
        for row in list(meta.get("labels") or []):
            if not isinstance(row, Mapping):
                continue
            updated_row = dict(row)
            try:
                if int(updated_row.get("label")) == old_value:
                    updated_row["label"] = new_value
            except Exception:
                _LOG.debug("Ignored optional GUI operation failure", exc_info=True)
            relabeled_rows.append(updated_row)
        if relabeled_rows:
            meta["labels"] = relabeled_rows
        meta["labels"] = _label_rows_from_mask(mask, meta)
        self._replace_roi(ROI(roi.name, mask, dims=roi.dims, metadata=meta))
        self.refresh_roi_list()
        self.update_view(preserve_image_view=True)
        self.update_plot()
        self.statusBar().showMessage(f"Changed label {old_value} to {new_value} in ROI '{roi.name}'.")

    def on_kspace_mode_changed(self, mode: str):
        self.session.active_kspace_mode = mode
        self.update_view()

    def on_kspace_centering_changed(self, mode: str):
        self.session.active_kspace_centering = mode
        self.session._slice_cache.clear()
        self.update_view()

    def refresh_plot_x_axis_combo(self):
        self.plot_x_combo.blockSignals(True)
        self.plot_x_combo.clear()
        choices = self._plot_x_axis_choices()
        for label, key in choices:
            self.plot_x_combo.addItem(label, key)
        keys = [key for _label, key in choices]
        selected = self._plot_x_axis_key if self._plot_x_axis_key in keys else "auto"
        idx = keys.index(selected) if selected in keys else 0
        self.plot_x_combo.setCurrentIndex(idx)
        self._plot_x_axis_key = str(self.plot_x_combo.currentData() or "auto")
        self.plot_x_combo.blockSignals(False)

    def _plot_x_axis_choices(self) -> list[tuple[str, str]]:
        choices = [("Auto", "auto"), ("Frame", "frame")]
        if not self.session.is_loaded:
            return choices
        try:
            experiment = self.session.build_experiment()
            frame_count = self.session.frame_count
            for name, qty in dict(experiment.observables).items():
                arr = _quantity_values(qty)
                if arr.ndim == 1 and int(arr.shape[0]) == frame_count:
                    choices.append((name, name))
                elif arr.ndim == 2 and int(arr.shape[0]) == frame_count:
                    if int(arr.shape[1]) == 3 and name.lower() in {"bvec", "bvecs"}:
                        choices.extend(
                            [
                                (f"{name} x", f"{name}:0"),
                                (f"{name} y", f"{name}:1"),
                                (f"{name} z", f"{name}:2"),
                            ]
                        )
                    else:
                        for col in range(min(int(arr.shape[1]), 6)):
                            choices.append((f"{name}[{col}]", f"{name}:{col}"))
        except Exception:
            _LOG.debug("Ignored optional GUI operation failure", exc_info=True)
        return choices

    def on_plot_x_axis_changed(self, _idx: int):
        self._plot_x_axis_key = str(self.plot_x_combo.currentData() or "auto")
        self.update_plot()

    def _plot_axis_values(self, length: int) -> tuple[str, np.ndarray]:
        try:
            experiment = self.session.build_experiment()
            key = str(getattr(self, "_plot_x_axis_key", "auto") or "auto")
            if key == "frame":
                return "frame", np.arange(int(length))
            if key == "auto":
                if str(experiment.exp_type or "").lower() in {"dti", "dki", "diffusion"}:
                    obs = dict(experiment.observables)
                    bvals = obs.get("bvals") or obs.get("bval")
                    if bvals is not None:
                        arr = _quantity_values(bvals).reshape((-1,))
                        if arr.size == int(length):
                            return "bvals", arr
                x_name, x = frame_axis_values(experiment)
                x = np.asarray(x).reshape((-1,))
                if x.size == int(length):
                    return x_name, x
                return "frame", np.arange(int(length))
            obs_name, sep, col_text = key.partition(":")
            qty = dict(experiment.observables).get(obs_name)
            if qty is not None:
                arr = _quantity_values(qty)
                if sep:
                    col = int(col_text)
                    if arr.ndim == 2 and int(arr.shape[0]) == int(length) and col < int(arr.shape[1]):
                        return f"{obs_name}[{col}]", np.asarray(arr[:, col]).reshape((-1,))
                elif arr.ndim == 1 and int(arr.shape[0]) == int(length):
                    return obs_name, np.asarray(arr).reshape((-1,))
        except Exception:
            _LOG.debug("Ignored optional GUI operation failure", exc_info=True)
        return "frame", np.arange(int(length))

    def on_view_changed(self, _text: str):
        self.refresh_map_list()
        self.refresh_frame_controls()
        self.refresh_registration_tab()
        self.update_view()

    def on_dimension_changed(self, key: str, value: int):
        self.session.set_active_index(key, value)
        self._update_dimension_counter(key, value)
        self.refresh_registration_tab()
        self.update_view()

    def _active_dimension_value(self, key: str) -> int:
        if key == "frame":
            return int(self.session.active_frame)
        if key == "z":
            return int(self.session.active_z)
        return int(self.session.active_indices.get(key, 0))

    def _update_dimension_counter(self, key: str, value: int | None = None):
        row = self._dimension_counter_labels.get(key)
        if row is None:
            return
        label_widget, label, size = row
        current = self._active_dimension_value(key) if value is None else int(value)
        current = max(0, min(current, max(0, int(size) - 1)))
        label_widget.setText(self._dimension_control_text(key, label, current, size))
        self._update_dimension_details()

    def _dimension_control_text(self, key: str, label: str, value: int, size: int) -> str:
        return f"{label}: {value + 1} / {size}"

    def _update_dimension_details(self):
        if self._dimension_detail_label is None:
            return
        details = []
        for key, (_label_widget, label, size) in self._dimension_counter_labels.items():
            value = self._active_dimension_value(key)
            detail = self._dimension_control_detail(key, label, value, size)
            if detail:
                details.append(detail)
        self._dimension_detail_label.setText(" | ".join(details))

    def _dimension_control_detail(self, key: str, _label: str, value: int, _size: int) -> str:
        if key != "frame":
            return ""
        try:
            experiment = self.session.build_experiment()
            if str(experiment.exp_type or "").lower() not in {"dti", "dki", "diffusion"}:
                return ""
            bvals = dict(experiment.observables).get("bvals") or dict(experiment.observables).get("bval")
            bvecs = dict(experiment.observables).get("bvecs") or dict(experiment.observables).get("bvec")
            pieces: list[str] = []
            if bvals is not None:
                b_arr = _quantity_values(bvals).reshape((-1,))
                if value < b_arr.size:
                    pieces.append(f"b={float(b_arr[value]):.6g}")
            if bvecs is not None:
                g_arr = _quantity_values(bvecs)
                if g_arr.ndim >= 2 and value < int(g_arr.shape[0]) and int(g_arr.shape[-1]) >= 3:
                    gx, gy, gz = [float(v) for v in np.asarray(g_arr[value, :3]).reshape((-1,))]
                    pieces.append(f"g=({gx:.3g}, {gy:.3g}, {gz:.3g})")
            return " | ".join(pieces)
        except Exception:
            return ""

    def _current_slice_indices(self) -> dict[str, int]:
        indices = dict(self.session.active_indices)
        indices["frame"] = int(self.session.active_frame)
        indices["z"] = int(self.session.active_z)
        return indices

    def on_frame_mask_changed(self, item):
        if self._updating_frames or self.session.frame_mask is None:
            return
        row = self.frame_mask_list.row(item)
        if item.checkState() == QtCore.Qt.Checked:
            self.session.frame_mask.include(row)
        else:
            self.session.frame_mask.exclude(row)
        self.mark_dirty("frame inclusion changed")
        self.refresh_fit_breakout_summaries()
        self.update_plot()

    def _capture_image_view_state(self) -> dict[str, Any] | None:
        if not hasattr(self, "image_view") or self.image_view is None:
            return None
        return _capture_pyqtgraph_image_view_state(self.image_view)

    def _restore_image_view_state(self, state: Mapping[str, Any] | None) -> None:
        if not state or not hasattr(self, "image_view") or self.image_view is None:
            return
        _restore_pyqtgraph_image_view_state(self.image_view, state)

    def update_view(self, *, preserve_image_view: bool = False):
        if not self.session.is_loaded:
            return
        view = _view_key(self.view_combo.currentText())
        use_original_image = bool(self.session.visual_state.show_original)
        prior_view_key = self._displayed_view_key
        prior_image_shape = self._displayed_image_shape
        prior_view_state = self._capture_image_view_state() if preserve_image_view else None
        if self._view_needs_lazy_raw_data(view):
            self.run_guarded_task(
                lambda progress: self.session.load_raw_data_for_experiment(progress_callback=progress),
                self.on_lazy_raw_data_loaded,
                "Loading raw k-space for selected scan...",
                error_title="Could not load raw k-space",
            )
            return
        try:
            if view == "k_space":
                image = self.session.kspace_image()
                self._displayed_image_label = f"k-space {self.session.active_kspace_mode}"
                self._displayed_image_unit = None
            elif view == "mrax_reconstruction":
                image = self.session.reconstruction_image()
                self._displayed_image_label = "MRax reconstruction"
                self._displayed_image_unit = None
            elif view == "vendor_vs_mrax":
                vendor, mrax = _matched_2d_pair(
                    self.session.frame_image(original=use_original_image),
                    self.session.reconstruction_image(),
                )
                image = _side_by_side_images(vendor, mrax)
                self._displayed_image_label = "vendor | MRax"
                self._displayed_image_unit = None
            elif view == "vendor_mrax_diff":
                vendor, mrax = _matched_2d_pair(
                    self.session.frame_image(original=use_original_image),
                    self.session.reconstruction_image(),
                )
                image = vendor - mrax
                self._displayed_image_label = "vendor - MRax"
                self._displayed_image_unit = None
            elif view == "map":
                image = self.session.map_image()
                map_name = self.session.active_map or "map"
                self._displayed_image_label = map_name
                self._displayed_image_unit = self.session.map_unit(map_name)
            else:
                image = self.session.frame_image(original=use_original_image)
                self._displayed_image_label = "intensity"
                self._displayed_image_unit = None
        except Exception as exc:
            self.statusBar().showMessage(str(exc))
            return
        displayed = np.asarray(image)
        display_levels = None
        if view == "map":
            displayed, display_levels = self._prepare_map_display(displayed)
        displayed, display_levels = self._apply_visual_view_state(displayed, view, display_levels)
        displayed, display_levels = self._apply_registration_reference_overlay(displayed, view, display_levels)
        displayed_shape = tuple(int(v) for v in displayed.shape)
        auto_range = displayed_shape != self._displayed_image_shape or view != self._displayed_view_key
        can_restore_view = bool(
            preserve_image_view
            and prior_view_state
            and prior_view_key == view
            and prior_image_shape == displayed_shape
        )
        if can_restore_view:
            auto_range = False
        self._displayed_image = displayed
        self._displayed_image_shape = displayed_shape
        self._displayed_view_key = view
        try:
            viewer_levels = (
                _image_view_state_levels(prior_view_state)
                if can_restore_view
                else _viewer_display_levels(self._displayed_image, display_levels)
            )
            if viewer_levels is None:
                self.image_view.setImage(self._displayed_image, autoLevels=True, autoRange=auto_range)
            else:
                self.image_view.setImage(
                    self._displayed_image,
                    autoLevels=False,
                    levels=viewer_levels,
                    autoRange=auto_range,
                )
        except TypeError:
            self.image_view.setImage(self._displayed_image, autoLevels=True)
            if auto_range:
                self.reset_image_view()
        self.refresh_overlays()
        self.update_plot()
        self.update_inspector()
        if can_restore_view:
            self._restore_image_view_state(prior_view_state)

    def _apply_visual_view_state(
        self,
        image: np.ndarray,
        view: str,
        display_levels: tuple[float, float] | None,
    ) -> tuple[np.ndarray, tuple[float, float] | None]:
        state = self.session.visual_state
        preview_result = self._active_registration_preview_result()
        if state.show_original and preview_result is None:
            return np.asarray(image), display_levels
        arr = np.asarray(image)
        try:
            display_mode = str(state.display_mode or "current")
            if display_mode != "current":
                display_mode = "current"
            if not state.show_original and state.drift_key:
                arr = self._apply_visual_drift_result(arr, view, state)
                display_levels = None
            if preview_result is not None:
                arr = self._apply_registration_result_transform(
                    arr,
                    preview_result,
                    direction="moving_to_reference",
                )
                display_levels = None
            elif state.transform_key:
                arr = self._apply_visual_transform(arr, state)
                display_levels = None
            if not state.show_original and state.mask_enabled and state.mask_roi:
                arr = self._apply_visual_mask(arr, str(state.mask_roi))
                display_levels = None
            is_rgb = arr.ndim == 3 and arr.shape[-1] in (3, 4)
            if (
                not state.show_original
                and not is_rgb
                and (state.color_channel != "gray" or float(state.color_alpha) < 1.0)
            ):
                arr = colorize_image(arr, channel=state.color_channel, alpha=state.color_alpha)
                display_levels = None
        except Exception as exc:
            self.statusBar().showMessage(f"Visual state skipped for {view}: {exc}")
            return np.asarray(image), display_levels
        return np.asarray(arr), display_levels

    def _apply_visual_drift_result(self, image: np.ndarray, view: str, state: VisualViewState) -> np.ndarray:
        if view != "image":
            return np.asarray(image)
        result = self.drift_results.get(str(state.drift_key or ""))
        if result is None:
            raise RuntimeError("selected drift correction is not available")
        metadata = dict(result.get("metadata") or {})
        selected = self.session.selected_experiment
        experiment = metadata.get("experiment")
        if selected is not None and experiment is not None and str(selected) != str(experiment):
            raise RuntimeError(
                f"drift correction source is {experiment}; current experiment is {selected}"
            )
        corrected = np.asarray(result.get("corrected"))
        if corrected.size == 0:
            raise RuntimeError("selected drift correction has no corrected image data")
        return _display_slice(
            corrected,
            dims=self.session.experiment_dims,
            indices=self._current_slice_indices(),
            kind="image",
        )

    def _visual_registration_display(self, state: VisualViewState) -> np.ndarray:
        result = self.registration_results.get(str(state.transform_key or ""))
        if result is None:
            raise RuntimeError("selected registration transform is not available")
        mode = str(state.display_mode or "current")
        fixed = _registration_preview_plane(result.get("fixed"))
        moving = _registration_preview_plane(result.get("moving"))
        corrected = _registration_preview_plane(result.get("corrected"))
        if mode == "registration_reference":
            return np.asarray(fixed)
        if mode == "registration_moving":
            return np.asarray(moving)
        if mode == "registration_transformed_moving":
            return np.asarray(corrected)
        if mode == "registration_overlay_original":
            return registration_overlay(
                fixed,
                moving,
                fixed_channel=state.reference_channel,
                moving_channel=state.moving_channel,
                fixed_alpha=state.reference_alpha,
                moving_alpha=state.moving_alpha,
            )
        if mode == "registration_overlay_transformed":
            return registration_overlay(
                fixed,
                corrected,
                fixed_channel=state.reference_channel,
                moving_channel=state.moving_channel,
                fixed_alpha=state.reference_alpha,
                moving_alpha=state.moving_alpha,
            )
        if mode == "registration_overlay_moving_grid":
            fixed_to_moving = apply_transform_to_image(
                fixed,
                moving,
                inverse_transform(result["transform"]),
                interpolator="linear",
            )
            return registration_overlay(
                moving,
                fixed_to_moving,
                fixed_channel=state.moving_channel,
                moving_channel=state.reference_channel,
                fixed_alpha=state.moving_alpha,
                moving_alpha=state.reference_alpha,
            )
        raise RuntimeError(f"Unsupported visual display mode '{mode}'")

    def _apply_visual_transform(self, image: np.ndarray, state: VisualViewState) -> np.ndarray:
        result = self.registration_results.get(str(state.transform_key or ""))
        if result is None:
            raise RuntimeError("selected registration transform is not available")
        return self._apply_registration_result_transform(
            image,
            result,
            direction=str(state.transform_direction or "moving_to_reference"),
        )

    def _apply_registration_result_transform(
        self,
        image: np.ndarray,
        result: Mapping[str, Any],
        *,
        direction: str,
    ) -> np.ndarray:
        metadata = dict(result.get("metadata") or {})
        selected = self.session.selected_experiment
        if direction == "reference_to_moving":
            source = metadata.get("reference_experiment")
            reference = np.asarray(result["moving"])
            transform = inverse_transform(result["transform"])
        else:
            source = metadata.get("moving_experiment")
            reference = np.asarray(result["fixed"])
            transform = result["transform"]
        if selected is not None and source is not None and str(selected) != str(source):
            raise RuntimeError(
                f"transform source is {source}; current experiment is {selected}"
            )
        source_image = _first_spatial_image(image)
        reference_image = _first_spatial_image(reference)
        return apply_transform_to_image(source_image, reference_image, transform, interpolator="linear")

    def _apply_visual_mask(self, image: np.ndarray, roi_name: str) -> np.ndarray:
        roi = self.session.roi(roi_name)
        if roi is None:
            raise RuntimeError(f"ROI '{roi_name}' is not enabled for this experiment")
        arr = np.asarray(image)
        display = _registration_preview_plane(arr) if arr.ndim > 3 else arr
        mask = np.asarray(roi.mask, dtype=bool)
        if mask.ndim > 2 and display.ndim in (2, 3):
            mask = mask_slice_for_display(mask, dims=roi.dims, indices=self._current_slice_indices())
        if display.ndim == 3 and display.shape[-1] in (3, 4):
            target_shape = tuple(int(v) for v in display.shape[:2])
        else:
            target_shape = tuple(int(v) for v in display.shape)
        if tuple(int(v) for v in mask.shape) != target_shape:
            raise RuntimeError(f"ROI '{roi_name}' mask shape {mask.shape} does not match displayed image shape {target_shape}")
        if display.ndim == 3 and display.shape[-1] in (3, 4):
            return np.where(mask[..., None], display, 0)
        return np.where(mask, display, 0)

    def _visual_state_fit_kwargs(self, *, include_roi: bool = True) -> dict[str, Any]:
        if not self.session.fit_settings.use_view_state_data:
            return {}
        state = self.session.visual_state
        if state.show_original:
            if not self.session.image_data_is_original():
                experiment, _model, dataset, _study = self.session.build_fit_inputs()
                return {
                    "input_experiment": experiment,
                    "input_dataset": self._view_state_dataset(
                        dataset,
                        np.asarray(self.session.original_image_array),
                        original_applied=True,
                    ),
                }
            return {}

        kwargs: dict[str, Any] = {}
        experiment: Experiment | None = None
        dataset: DataSet | None = None
        transformed = False
        transform_context: tuple[np.ndarray, Any] | None = None

        if state.drift_key or state.transform_key:
            experiment, _model, dataset, _study = self.session.build_fit_inputs()
            data = np.asarray(dataset.data)
            if state.drift_key:
                data = self._visual_state_drift_corrected_data(state)
            if state.transform_key:
                data, experiment, transform_context = self._visual_state_transformed_fit_data(
                    data,
                    experiment,
                    state,
                )
            dataset = self._view_state_dataset(
                dataset,
                data,
                original_applied=False,
                drift_applied=bool(state.drift_key),
                transform_applied=bool(state.transform_key),
            )
            kwargs["input_experiment"] = experiment
            kwargs["input_dataset"] = dataset
            transformed = True

        if include_roi:
            source_roi: ROI | None = None
            if state.mask_enabled and state.mask_roi:
                source_roi = self.session.roi(str(state.mask_roi))
                if source_roi is None:
                    raise RuntimeError(f"ROI '{state.mask_roi}' is not enabled for this experiment")
            elif transformed:
                source_roi = self.session.roi()
            if source_roi is not None and (state.mask_enabled or transformed):
                kwargs["input_roi"] = self._visual_state_fit_roi(
                    source_roi,
                    transform_context=transform_context,
                    target_spatial_dims=tuple(experiment.spatial_dim_names) if experiment is not None else None,
                )
        return kwargs

    def _visual_state_drift_corrected_data(self, state: VisualViewState) -> np.ndarray:
        result = self.drift_results.get(str(state.drift_key or ""))
        if result is None:
            raise RuntimeError("selected drift correction is not available")
        metadata = dict(result.get("metadata") or {})
        selected = self.session.selected_experiment
        experiment = metadata.get("experiment")
        if selected is not None and experiment is not None and str(selected) != str(experiment):
            raise RuntimeError(
                f"drift correction source is {experiment}; current experiment is {selected}"
            )
        corrected = np.asarray(result.get("corrected"))
        if tuple(corrected.shape) != tuple(np.asarray(self.session.image_array).shape):
            raise RuntimeError(
                f"drift-corrected data shape {corrected.shape} does not match current image shape {self.session.image_array.shape}"
            )
        return corrected

    def _visual_state_transformed_fit_data(
        self,
        data: np.ndarray,
        experiment: Experiment,
        state: VisualViewState,
    ) -> tuple[np.ndarray, Experiment, tuple[np.ndarray, Any]]:
        reference_image, transform = self._visual_state_fit_transform_context(state)
        spatial_dims = tuple(experiment.spatial_dim_names)
        if len(spatial_dims) != 2:
            raise RuntimeError("View-state fitting with transforms currently supports 2-D spatial data")
        frames = np.asarray(reshape_dims_to_frames(data, experiment.dims))
        if frames.ndim != 3:
            raise RuntimeError("View-state fitting could not reshape data into 2-D frames")
        target = _first_spatial_image(reference_image)
        transformed_frames = np.stack(
            [
                apply_transform_to_image(_first_spatial_image(frame), target, transform, interpolator="linear")
                for frame in frames
            ],
            axis=0,
        )
        updated_dims = self._experiment_dims_with_spatial_shape(experiment, transformed_frames.shape[1:])
        transformed_data = reshape_frames_to_dims(transformed_frames, updated_dims)
        return transformed_data, self._copy_experiment_with_dims(experiment, updated_dims), (target, transform)

    def _visual_state_fit_transform_context(self, state: VisualViewState) -> tuple[np.ndarray, Any]:
        result = self.registration_results.get(str(state.transform_key or ""))
        if result is None:
            raise RuntimeError("selected registration transform is not available")
        metadata = dict(result.get("metadata") or {})
        selected = self.session.selected_experiment
        direction = str(state.transform_direction or "moving_to_reference")
        if direction == "reference_to_moving":
            source = metadata.get("reference_experiment")
            reference = np.asarray(result["moving"])
            transform = inverse_transform(result["transform"])
        else:
            source = metadata.get("moving_experiment")
            reference = np.asarray(result["fixed"])
            transform = result["transform"]
        if selected is not None and source is not None and str(selected) != str(source):
            raise RuntimeError(
                f"transform source is {source}; current experiment is {selected}"
            )
        return reference, transform

    def _visual_state_fit_roi(
        self,
        roi: ROI,
        *,
        transform_context: tuple[np.ndarray, Any] | None,
        target_spatial_dims: tuple[str, ...] | None,
    ) -> ROI:
        mask = np.asarray(roi.mask) != 0
        dims = tuple(roi.dims or tuple(self.session.spatial_dim_names[: mask.ndim]))
        metadata = {
            **dict(roi.metadata),
            "generated_by": "mrax.gui.view_state_fit",
            "source_roi": roi.name,
            "binary_view_state_mask": True,
        }
        if transform_context is not None:
            reference, transform = transform_context
            if mask.ndim != 2:
                raise RuntimeError("View-state fitting can transform only 2-D ROI masks")
            mask = resample_mask_with_transform(mask, reference, transform, output_dtype=bool)
            dims = tuple(target_spatial_dims or self.session.spatial_dim_names[: mask.ndim])
            metadata["registration_transform"] = str(self.session.visual_state.transform_key or "")
            metadata["registration_direction"] = str(self.session.visual_state.transform_direction or "")
        return ROI(
            f"{roi.name}_view_state",
            np.asarray(mask, dtype=bool),
            dims=dims,
            metadata=metadata,
        )

    def _view_state_dataset(
        self,
        dataset: DataSet,
        data: np.ndarray,
        *,
        original_applied: bool = False,
        drift_applied: bool = False,
        transform_applied: bool = False,
    ) -> DataSet:
        return DataSet(
            data=jnp.asarray(data),
            name=f"{dataset.name}_view_state",
            noise_level=dataset.noise_level,
            processed=dict(dataset.processed or {}),
            kspace=None,
            trajectory=dataset.trajectory,
            density=dataset.density,
            sampling_mask=dataset.sampling_mask,
            kspace_metadata={
                **dict(dataset.kspace_metadata or {}),
                "view_state_original_applied": bool(original_applied),
                "view_state_drift_applied": bool(drift_applied),
                "view_state_transform_applied": bool(transform_applied),
            },
        )

    def _experiment_dims_with_spatial_shape(
        self,
        experiment: Experiment,
        spatial_shape: Sequence[int],
    ) -> dict[str, int]:
        spatial_dims = tuple(experiment.spatial_dim_names)
        shape_by_dim = {name: int(size) for name, size in zip(spatial_dims, spatial_shape)}
        return {
            str(name): int(shape_by_dim.get(str(name), int(size)))
            for name, size in experiment.dims.items()
        }

    def _copy_experiment_with_dims(self, experiment: Experiment, dims: Mapping[str, int]) -> Experiment:
        updated = Experiment(
            dims={str(name): int(size) for name, size in dims.items()},
            observables=dict(experiment.observables),
            exp_type=experiment.exp_type,
        )
        for attr in (
            "study_globals",
            "_mrax_dim_observables",
            "_mrax_default_identity_fit_warning",
            "_mrax_manual_type_override_warning",
        ):
            if hasattr(experiment, attr):
                setattr(updated, attr, getattr(experiment, attr))
        return updated

    def _view_needs_lazy_raw_data(self, view: str) -> bool:
        if view == "k_space":
            return self.session.kspace_array is None and self.session.can_lazy_load_raw_data()
        if view in {"mrax_reconstruction", "vendor_vs_mrax", "vendor_mrax_diff"}:
            return self.session.reconstruction_array is None and self.session.can_lazy_load_raw_data()
        return False

    def on_lazy_raw_data_loaded(self, _loaded: bool):
        self.refresh_frame_controls()
        self.update_view()
        self.update_diagnostics()
        self.mark_dirty("raw k-space loaded")
        self.statusBar().showMessage("Raw k-space data loaded for selected scan.")

    def _prepare_map_display(self, image: np.ndarray) -> tuple[np.ndarray, tuple[float, float] | None]:
        arr = np.asarray(image)
        if np.iscomplexobj(arr):
            arr = np.abs(arr)
        else:
            arr = arr.astype(float, copy=True)
        roi = self.session.roi()
        if roi is not None and arr.ndim >= 2:
            try:
                mask = mask_slice_for_display(
                    roi.mask,
                    dims=roi.dims,
                    indices=self._current_slice_indices(),
                )
                mask = np.asarray(mask, dtype=bool)
                if tuple(mask.shape) == tuple(arr.shape[-2:]):
                    arr = arr.copy()
                    arr[~mask] = np.nan
            except Exception:
                _LOG.debug("Ignored optional GUI operation failure", exc_info=True)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return arr, None
        if finite.size >= 16:
            lo, hi = np.nanpercentile(finite, [1.0, 99.0])
            if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
                lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
        else:
            lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
        if not np.isfinite(lo) or not np.isfinite(hi):
            return arr, None
        if lo == hi:
            pad = max(abs(float(lo)) * 0.01, 1e-6)
            lo -= pad
            hi += pad
        return arr, (float(lo), float(hi))

    def refresh_overlays(self):
        view_box = self.image_view.getView()
        for item in self.overlay_items:
            try:
                view_box.removeItem(item)
            except Exception:
                _LOG.debug("Ignored optional GUI operation failure", exc_info=True)
        self.overlay_items.clear()
        for item in list(self.roi_items.values()):
            try:
                view_box.removeItem(item)
            except Exception:
                _LOG.debug("Ignored optional GUI operation failure", exc_info=True)
        self.roi_items.clear()
        self.roi_item_kinds.clear()
        try:
            if not self.session.rois:
                return
            if _view_key(self.view_combo.currentText()) not in {"image", "map"}:
                return
            try:
                displayed = np.asarray(self._displayed_image if self._displayed_image is not None else self.session.frame_image())
                shape = _display_spatial_shape(displayed)
            except Exception:
                return
            indices = self._current_slice_indices()
            editable_name = self.session.fit_settings.selected_roi
            self._syncing_roi = True
            try:
                for roi in self.session.rois:
                    editable = roi.name == editable_name
                    mask = _roi_overlay_mask_for_display(
                        roi,
                        display_shape=shape,
                        indices=indices,
                        editable=editable,
                    )
                    if mask is None:
                        continue
                    roi_item = self._roi_item_from_roi(roi, mask) if editable else None
                    if roi_item is not None:
                        view_box.addItem(roi_item)
                        self.roi_items[roi.name] = roi_item
                        self.roi_item_kinds[roi.name] = str(roi.metadata.get("shape", "mask"))
                        roi_item.sigRegionChanged.connect(
                            lambda *_args, _item=roi_item, name=roi.name: self.on_roi_item_changed(name, _item)
                        )
                        continue
                    outline_item = self._static_roi_outline_item(roi, mask)
                    if outline_item is not None:
                        view_box.addItem(outline_item)
                        self.overlay_items.append(outline_item)
                        continue
                    color = QtGui.QColor(str(roi.metadata.get("color", "#2dd4bf")))
                    rgba = np.zeros(mask.shape + (4,), dtype=np.ubyte)
                    rgba[..., 0] = color.red()
                    rgba[..., 1] = color.green()
                    rgba[..., 2] = color.blue()
                    rgba[..., 3] = mask.astype(np.ubyte) * 90
                    item = pg.ImageItem(rgba)
                    item.setZValue(10)
                    view_box.addItem(item)
                    self.overlay_items.append(item)
            finally:
                self._syncing_roi = False
        finally:
            self._refresh_roi_interaction_state()

    def _static_roi_outline_item(self, roi: ROI, display_mask: np.ndarray):
        meta = dict(roi.metadata)
        shape = str(meta.get("shape", "mask"))
        color = str(meta.get("color", "#2dd4bf"))
        points = _static_roi_outline_points(meta, display_mask)
        if points is None or points.shape[0] < 2:
            return None
        item = pg.PlotDataItem(
            points[:, 0],
            points[:, 1],
            pen=pg.mkPen(color, width=2),
            symbol=None,
        )
        item.setZValue(15)
        if shape == "polygon" and hasattr(item, "setCurveClickable"):
            item.setCurveClickable(False)
        return item

    def _roi_item_from_roi(self, roi: ROI, display_mask: np.ndarray):
        meta = dict(roi.metadata)
        shape = str(meta.get("shape", "mask"))
        color = str(meta.get("color", "#2dd4bf"))
        angle = float(meta.get("angle", 0.0) or 0.0)
        pen = pg.mkPen(color, width=2)
        handle_pen = pg.mkPen("#111827", width=1)
        if shape == "rectangle":
            x, y, w, h = meta.get("bounds", _mask_bounds(display_mask))
            item = pg.RectROI([x, y], [w, h], angle=angle, pen=pen, removable=False)
            try:
                item.addRotateHandle([1.0, 0.0], [0.5, 0.5])
            except Exception:
                _LOG.debug("Ignored optional GUI operation failure", exc_info=True)
        elif shape == "ellipse":
            x, y, w, h = meta.get("bounds", _mask_bounds(display_mask))
            item = pg.EllipseROI([x, y], [w, h], angle=angle, pen=pen, removable=False)
        elif shape == "polygon":
            points = meta.get("points")
            if not points:
                return None
            arr = _sanitize_polygon_points(points)
            if arr.shape[0] < 3:
                return None
            pos = meta.get("pos")
            if pos is None:
                center = np.mean(arr, axis=0) if arr.size else np.zeros((2,), dtype=float)
                local_points = (arr - center).tolist()
                pos = center.tolist()
            else:
                local_points = arr.tolist()
            item = pg.PolyLineROI(local_points, closed=True, pos=pos, angle=angle, pen=pen, removable=False)
        else:
            return None
        item.setZValue(20)
        for handle in getattr(item, "handles", []):
            try:
                handle["item"].setPen(handle_pen)
            except Exception:
                _LOG.debug("Ignored optional GUI operation failure", exc_info=True)
        return item

    def on_roi_item_changed(self, name: str, item):
        if self._syncing_roi:
            return
        roi = self.session.roi(name)
        if roi is None:
            return
        current = np.asarray(self.session.frame_image())
        shape = current.shape[-2], current.shape[-1]
        kind = self.roi_item_kinds.get(name, str(roi.metadata.get("shape", "mask")))
        mask, metadata = _mask_from_roi_item(item, kind, shape)
        meta = dict(roi.metadata)
        meta.update(metadata)
        meta["slice_indices"] = self._current_slice_indices()
        base_mask = self._roi_as_spatial_mask(roi, geometry_indices=self._current_slice_indices())
        updated_mask = replace_mask_slice(
            base_mask,
            mask,
            dims=tuple(self.session.spatial_dim_names),
            indices=self._current_slice_indices(),
        )
        updated_dims = tuple(self.session.spatial_dim_names[: updated_mask.ndim])
        updated = ROI(name, updated_mask, dims=updated_dims, metadata=meta)
        self._replace_roi(updated)
        self.update_plot()

    def _replace_roi(self, roi: ROI):
        self.session.rois = [existing for existing in self.session.rois if existing.name != roi.name]
        self.session.add_roi(roi)
        self.session.fit_settings.selected_roi = roi.name
        self.mark_dirty("ROI set changed")

    def _all_roi_names(self) -> set[str]:
        names = {roi.name for roi in self.session.shared_rois}
        names.update(roi.name for roi in self.session.rois)
        return names

    def _roi_name_dialog(self, title: str, label: str, default: str) -> str | None:
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(title)
        layout = QtWidgets.QVBoxLayout(dialog)
        form = QtWidgets.QFormLayout()
        name_edit = QtWidgets.QLineEdit(str(default or ""))
        name_edit.selectAll()
        form.addRow(label, name_edit)
        layout.addLayout(form)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if self._exec_dialog(dialog) != QtWidgets.QDialog.Accepted:
            return None
        return str(name_edit.text() or "").strip()

    def add_empty_roi(self):
        if not self.session.is_loaded:
            return
        existing = [ROI(name, np.zeros((1,), dtype=bool)) for name in sorted(self._all_roi_names())]
        default_name = _unique_roi_name(existing, "roi")
        name = self._roi_name_dialog("New ROI", "ROI name", default_name)
        if name is None:
            return
        if not name:
            self.statusBar().showMessage("ROI name cannot be empty.")
            return
        if name in self._all_roi_names():
            self.statusBar().showMessage(f"ROI '{name}' already exists.")
            return
        spatial_dims = tuple(self.session.spatial_dim_names)
        roi = ROI(
            name,
            np.zeros(tuple(self.session.spatial_shape), dtype=bool),
            dims=spatial_dims,
            metadata={
                "shape": "mask",
                "color": _roi_color(len(self.session.rois)),
                "source_experiment": self.session.selected_experiment,
                "slice_indices": self._current_slice_indices(),
                "generated_by": "mrax.gui.new_roi",
            },
        )
        self.session.add_roi(roi)
        self.session.fit_settings.selected_roi = name
        self.refresh_roi_list()
        self.update_view(preserve_image_view=True)
        self.mark_dirty("ROI set changed")
        self.statusBar().showMessage(f"Created empty ROI '{name}'. New drawing and brush edits will add to it.")

    def rename_selected_roi(self):
        item = self.roi_list.currentItem()
        old_name = item.text() if item is not None else (self.session.roi().name if self.session.roi() is not None else "")
        if not old_name:
            self.statusBar().showMessage("No ROI selected.")
            return
        new_name = self._roi_name_dialog("Rename ROI", "New name", old_name)
        if new_name is None:
            return
        if not new_name:
            self.statusBar().showMessage("ROI name cannot be empty.")
            return
        try:
            changed = self.session.rename_roi(old_name, new_name)
        except ValueError as exc:
            self.statusBar().showMessage(str(exc))
            return
        if not changed:
            return
        self.refresh_roi_list()
        self.update_view(preserve_image_view=True)
        self.mark_dirty("ROI set changed")
        self.statusBar().showMessage(f"Renamed ROI '{old_name}' to '{new_name}'.")

    def apply_rois_to_all_experiments(self):
        if not self.session.rois:
            self.statusBar().showMessage("No ROIs to apply.")
            return
        self.session.apply_rois_to_all_experiments()
        self.refresh_roi_list()
        self.update_view(preserve_image_view=True)
        self.mark_dirty("ROI set changed")
        self.statusBar().showMessage("Applied current ROI set to all experiments in the study.")

    def _selected_roi_names(self) -> list[str]:
        names = [item.text() for item in self.roi_list.selectedItems()]
        if not names and self.roi_list.currentItem() is not None:
            names = [self.roi_list.currentItem().text()]
        seen: set[str] = set()
        ordered: list[str] = []
        for name in names:
            if name not in seen:
                seen.add(name)
                ordered.append(name)
        return ordered

    def _selected_rois_for_action(self) -> list[ROI]:
        rois = [self.session.roi(name) for name in self._selected_roi_names()]
        return [roi for roi in rois if roi is not None]

    def _visible_roi_slice_masks(self, rois: list[ROI] | None = None) -> list[tuple[ROI, np.ndarray]]:
        display = self._roi_source_image()
        if display is None:
            return []
        shape = tuple(int(v) for v in np.asarray(display).shape[-2:])
        return _visible_roi_slice_masks_for_display(
            rois or list(self.session.rois),
            display_shape=shape,
            indices=self._current_slice_indices(),
        )

    def combine_rois(self):
        if len(self.session.rois) < 2:
            self.statusBar().showMessage("At least two active ROIs are needed to combine.")
            return
        options = self._combine_roi_options()
        if options is None:
            return
        selected_names, mode, as_label_mask, label_values = options
        rois = [roi for roi in self.session.rois if roi.name in selected_names]
        if len(rois) < 2:
            self.statusBar().showMessage("Select at least two ROIs to combine.")
            return
        spatial_dims = tuple(self.session.spatial_dim_names)
        if mode == "slice":
            visible = self._visible_roi_slice_masks(rois)
            if len(visible) < 2:
                self.statusBar().showMessage("Selected ROIs are not visible on the current slice.")
                return
            if as_label_mask:
                combined_2d, labels = _label_mask_from_visible_roi_slice_masks(visible, label_values=label_values)
                source_names = [str(row["name"]) for row in labels]
            else:
                combined_2d, source_names = _combine_visible_roi_slice_masks(visible)
                labels = []
            combined = embed_mask_slice(
                combined_2d,
                spatial_shape=tuple(self.session.spatial_shape),
                spatial_dims=spatial_dims,
                indices=self._current_slice_indices(),
            )
            combine_dimension = "current_slice"
        elif mode.startswith("slice_dim:"):
            combine_dimension = mode.split(":", 1)[1]
            if as_label_mask:
                combined, labels = self._label_mask_from_spatial_rois(rois, label_values=label_values)
                source_names = [str(row["name"]) for row in labels]
            else:
                try:
                    combined = stitch_masks_along_slices(
                        rois,
                        spatial_shape=tuple(self.session.spatial_shape),
                        spatial_dims=spatial_dims,
                        indices=self._current_slice_indices(),
                        slice_dim=combine_dimension,
                    )
                except Exception as exc:
                    self.statusBar().showMessage(f"Could not combine ROIs: {exc}")
                    return
                source_names = [roi.name for roi in rois]
                labels = []
        else:
            combine_dimension = "full"
            if as_label_mask:
                combined, labels = self._label_mask_from_spatial_rois(rois, label_values=label_values)
                source_names = [str(row["name"]) for row in labels]
            else:
                combined = np.zeros(tuple(self.session.spatial_shape), dtype=bool)
                source_names = []
                labels = []
                for roi in rois:
                    roi_mask = np.asarray(self._roi_as_spatial_mask(roi)) != 0
                    if tuple(roi_mask.shape) != tuple(combined.shape):
                        continue
                    combined |= roi_mask
                    source_names.append(roi.name)
        if not np.any(np.asarray(combined) != 0):
            self.statusBar().showMessage("Combined ROI would be empty.")
            return
        name = _unique_roi_name(self.session.rois, "labels" if as_label_mask else "combined")
        roi = ROI(
            name,
            combined,
            dims=spatial_dims,
            metadata={
                "shape": "label_mask" if as_label_mask else "mask",
                "mask_type": "label" if as_label_mask else "binary",
                "color": _roi_color(len(self.session.rois)),
                "source_experiment": self.session.selected_experiment,
                "slice_indices": self._current_slice_indices(),
                "source_rois": source_names,
                "labels": labels,
                "combine_dimension": combine_dimension,
                "generated_by": "mrax.gui.combine_label_mask" if as_label_mask else "mrax.gui.combine",
            },
        )
        self.session.add_roi(roi)
        self.session.fit_settings.selected_roi = name
        self.refresh_roi_list()
        self.update_view(preserve_image_view=True)
        self.mark_dirty("ROI set changed")
        kind = "label mask" if as_label_mask else "ROI"
        self.statusBar().showMessage(f"Combined {len(source_names)} ROI(s) into {kind} '{name}'.")

    def _combine_roi_options(self) -> tuple[list[str], str, bool, dict[str, int]] | None:
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Combine ROIs")
        layout = QtWidgets.QVBoxLayout(dialog)
        form = QtWidgets.QFormLayout()
        mode_combo = QtWidgets.QComboBox()
        mode_combo.addItem("Current image slice", "slice")
        for dim in tuple(self.session.spatial_dim_names):
            if dim not in {"x", "y"}:
                mode_combo.addItem(f"Across {dim} slices", f"slice_dim:{dim}")
        mode_combo.addItem("Full spatial mask", "full")
        self._configure_wide_combo(mode_combo, contents=18)
        label_mask_check = QtWidgets.QCheckBox("Create integer label mask")
        label_mask_check.setChecked(False)
        form.addRow("Combine along", mode_combo)
        form.addRow(label_mask_check)
        layout.addLayout(form)

        roi_list = QtWidgets.QListWidget()
        selected_names = set(self._selected_roi_names())
        checked_names = selected_names if len(selected_names) >= 2 else {roi.name for roi in self.session.rois}
        for roi in self.session.rois:
            item = QtWidgets.QListWidgetItem(roi.name)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Checked if roi.name in checked_names else QtCore.Qt.Unchecked)
            roi_list.addItem(item)
        layout.addWidget(QtWidgets.QLabel("ROIs"))
        layout.addWidget(roi_list)

        label_table = QtWidgets.QTableWidget()
        label_table.setColumnCount(2)
        label_table.setHorizontalHeaderLabels(["ROI", "Label"])
        label_table.setRowCount(len(self.session.rois))
        label_spins: dict[str, Any] = {}
        for row, roi in enumerate(self.session.rois):
            name_item = QtWidgets.QTableWidgetItem(roi.name)
            name_item.setFlags(name_item.flags() & ~QtCore.Qt.ItemIsEditable)
            label_table.setItem(row, 0, name_item)
            spin = QtWidgets.QSpinBox()
            spin.setRange(1, 1_000_000)
            spin.setValue(row + 1)
            label_table.setCellWidget(row, 1, spin)
            label_spins[roi.name] = spin
        label_table.resizeColumnsToContents()
        label_table.setVisible(False)
        label_values_label = QtWidgets.QLabel("Label values")
        label_values_label.setVisible(False)
        layout.addWidget(label_values_label)
        layout.addWidget(label_table)

        def sync_label_table_visible():
            visible = bool(label_mask_check.isChecked())
            label_table.setVisible(visible)
            label_values_label.setVisible(visible)

        label_mask_check.toggled.connect(lambda _checked: sync_label_table_visible())
        sync_label_table_visible()
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if self._exec_dialog(dialog) != QtWidgets.QDialog.Accepted:
            return None
        names = [
            roi_list.item(row).text()
            for row in range(roi_list.count())
            if roi_list.item(row).checkState() == QtCore.Qt.Checked
        ]
        as_label_mask = bool(label_mask_check.isChecked())
        label_values = {
            name: int(label_spins[name].value())
            for name in names
            if name in label_spins
        } if as_label_mask else {}
        return names, str(mode_combo.currentData() or "slice"), as_label_mask, label_values

    def combine_visible_rois_slice(self):
        self.combine_rois()

    def _label_mask_from_spatial_rois(
        self,
        rois: Sequence[ROI],
        *,
        label_values: Mapping[str, int] | None = None,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        shape = tuple(self.session.spatial_shape)
        values = {
            roi.name: max(1, int((label_values or {}).get(roi.name, idx)))
            for idx, roi in enumerate(rois, start=1)
        }
        max_label = max(values.values(), default=0)
        dtype = np.uint16 if max_label <= np.iinfo(np.uint16).max else np.int32
        label_mask = np.zeros(shape, dtype=dtype)
        labels: list[dict[str, Any]] = []
        for idx, roi in enumerate(rois, start=1):
            try:
                mask = np.asarray(self._roi_as_spatial_mask(roi)) != 0
            except Exception:
                continue
            if tuple(mask.shape) != shape:
                continue
            if not np.any(mask):
                continue
            label = values.get(roi.name, idx)
            label_mask[mask] = label
            labels.append(
                {
                    "label": int(label),
                    "name": roi.name,
                    "voxel_count": int(np.count_nonzero(mask)),
                    "metadata": dict(roi.metadata),
                }
            )
        return label_mask, labels

    def subtract_rois(self):
        if len(self.session.rois) < 2:
            self.statusBar().showMessage("At least two active ROIs are needed to subtract.")
            return
        options = self._subtract_roi_options()
        if options is None:
            return
        target_name, cutter_names, mode = options
        target = self.session.roi(target_name)
        cutters = [roi for roi in self.session.rois if roi.name in set(cutter_names)]
        if target is None or not cutters:
            self.statusBar().showMessage("Select a target ROI and at least one subtracting ROI.")
            return
        spatial_dims = tuple(self.session.spatial_dim_names)
        base = self._roi_as_spatial_mask(target)
        try:
            if mode == "slice":
                target_plane = np.asarray(
                    mask_slice_for_display(base, dims=spatial_dims, indices=self._current_slice_indices())
                ).copy()
                visible = self._visible_roi_slice_masks(cutters)
                cut_mask, source_names = _combine_visible_roi_slice_masks(visible)
                if tuple(cut_mask.shape) != tuple(target_plane.shape):
                    self.statusBar().showMessage("Subtracting ROIs are not aligned with the target slice.")
                    return
                target_plane[np.asarray(cut_mask, dtype=bool)] = 0
                updated = replace_mask_slice(
                    base,
                    target_plane,
                    dims=spatial_dims,
                    indices=self._current_slice_indices(),
                )
                subtract_dimension = "current_slice"
            else:
                cut_mask = np.zeros(tuple(self.session.spatial_shape), dtype=bool)
                source_names = []
                for cutter in cutters:
                    cutter_mask = np.asarray(self._roi_as_spatial_mask(cutter)) != 0
                    if tuple(cutter_mask.shape) != tuple(cut_mask.shape):
                        continue
                    cut_mask |= cutter_mask
                    source_names.append(cutter.name)
                updated = np.asarray(base).copy()
                updated[cut_mask] = 0
                subtract_dimension = "full"
        except Exception as exc:
            self.statusBar().showMessage(f"Could not subtract ROIs: {exc}")
            return
        meta = _rasterized_roi_metadata(target.metadata, "subtract")
        meta["subtracted_rois"] = source_names
        meta["subtract_dimension"] = subtract_dimension
        if np.asarray(updated).dtype != bool:
            meta["mask_type"] = "label"
        self._replace_roi(ROI(target.name, updated, dims=spatial_dims, metadata=meta))
        self.refresh_roi_list()
        self.update_view(preserve_image_view=True)
        self.mark_dirty("ROI set changed")
        self.statusBar().showMessage(f"Subtracted {len(source_names)} ROI(s) from '{target.name}'.")

    def _subtract_roi_options(self) -> tuple[str, list[str], str] | None:
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Subtract ROIs")
        layout = QtWidgets.QVBoxLayout(dialog)
        form = QtWidgets.QFormLayout()
        target_combo = QtWidgets.QComboBox()
        selected_names = self._selected_roi_names()
        default_target = selected_names[0] if selected_names else (self.session.rois[0].name if self.session.rois else "")
        for roi in self.session.rois:
            target_combo.addItem(roi.name, roi.name)
        idx = target_combo.findData(default_target)
        if idx >= 0:
            target_combo.setCurrentIndex(idx)
        mode_combo = QtWidgets.QComboBox()
        mode_combo.addItem("Current image slice", "slice")
        mode_combo.addItem("Full spatial mask", "full")
        self._configure_wide_combo(target_combo, contents=18)
        self._configure_wide_combo(mode_combo, contents=18)
        form.addRow("Cut this ROI", target_combo)
        form.addRow("Subtract over", mode_combo)
        layout.addLayout(form)

        cutter_list = QtWidgets.QListWidget()
        selected_cutters = set(selected_names[1:] if len(selected_names) >= 2 else [])
        if not selected_cutters:
            selected_cutters = {roi.name for roi in self.session.rois if roi.name != default_target}
        for roi in self.session.rois:
            item = QtWidgets.QListWidgetItem(roi.name)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Checked if roi.name in selected_cutters else QtCore.Qt.Unchecked)
            cutter_list.addItem(item)
        layout.addWidget(QtWidgets.QLabel("Subtracting ROIs"))
        layout.addWidget(cutter_list)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if self._exec_dialog(dialog) != QtWidgets.QDialog.Accepted:
            return None
        target_name = str(target_combo.currentData() or "")
        cutter_names = [
            cutter_list.item(row).text()
            for row in range(cutter_list.count())
            if cutter_list.item(row).checkState() == QtCore.Qt.Checked and cutter_list.item(row).text() != target_name
        ]
        return target_name, cutter_names, str(mode_combo.currentData() or "slice")

    def shift_selected_rois(self):
        rois = self._selected_rois_for_action()
        if not rois:
            self.statusBar().showMessage("No ROI selected.")
            return
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Shift ROIs")
        layout = QtWidgets.QVBoxLayout(dialog)
        form = QtWidgets.QFormLayout()
        dx_spin = QtWidgets.QSpinBox()
        dy_spin = QtWidgets.QSpinBox()
        dz_spin = QtWidgets.QSpinBox()
        for spin in (dx_spin, dy_spin, dz_spin):
            spin.setRange(-10000, 10000)
            spin.setValue(0)
        form.addRow("Horizontal pixels", dx_spin)
        form.addRow("Vertical pixels", dy_spin)
        form.addRow("Z slices", dz_spin)
        layout.addLayout(form)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if self._exec_dialog(dialog) != QtWidgets.QDialog.Accepted:
            return
        dx, dy, dz = int(dx_spin.value()), int(dy_spin.value()), int(dz_spin.value())
        if dx == 0 and dy == 0 and dz == 0:
            return
        spatial_dims = tuple(self.session.spatial_dim_names)
        for roi in rois:
            mask = self._roi_as_spatial_mask(roi)
            shifted = _shift_spatial_mask_visual(mask, spatial_dims, dx=dx, dy=dy, dz=dz)
            meta = _rasterized_roi_metadata(roi.metadata, "shift")
            meta["shift"] = {"dx": dx, "dy": dy, "dz": dz}
            self._replace_roi(ROI(roi.name, shifted, dims=spatial_dims, metadata=meta))
        self.refresh_roi_list()
        self.update_view(preserve_image_view=True)
        self.statusBar().showMessage(f"Shifted {len(rois)} ROI(s).")

    def copy_selected_rois_to_z(self):
        rois = self._selected_rois_for_action()
        if not rois:
            self.statusBar().showMessage("No ROI selected.")
            return
        spatial_dims = tuple(self.session.spatial_dim_names)
        if "z" not in spatial_dims:
            self.statusBar().showMessage("Current experiment has no z dimension.")
            return
        z_axis = spatial_dims.index("z")
        z_count = int(self.session.spatial_shape[z_axis])
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Copy ROI Slice")
        layout = QtWidgets.QVBoxLayout(dialog)
        form = QtWidgets.QFormLayout()
        z_spin = QtWidgets.QSpinBox()
        z_spin.setRange(1, max(1, z_count))
        z_spin.setValue(max(1, min(z_count, int(self.session.active_z) + 1)))
        form.addRow("Target z slice", z_spin)
        layout.addLayout(form)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if self._exec_dialog(dialog) != QtWidgets.QDialog.Accepted:
            return
        target_z = int(z_spin.value()) - 1
        copied = 0
        for roi in rois:
            try:
                plane = np.asarray(
                    mask_slice_for_display(
                        roi.mask,
                        dims=roi.dims,
                        indices=self._current_slice_indices(),
                    ),
                    dtype=bool,
                )
            except Exception:
                continue
            mask = self._roi_as_spatial_mask(roi)
            indices = self._current_slice_indices()
            indices["z"] = target_z
            updated = replace_mask_slice(mask, plane, dims=spatial_dims, indices=indices)
            meta = _rasterized_roi_metadata(roi.metadata, "copy_z")
            meta["copied_to_z"] = target_z
            self._replace_roi(ROI(roi.name, updated, dims=spatial_dims, metadata=meta))
            copied += 1
        self.refresh_roi_list()
        self.update_view(preserve_image_view=True)
        self.statusBar().showMessage(f"Copied {copied} ROI slice(s) to z {target_z + 1}.")

    def delete_selected_roi_slice_z(self):
        rois = self._selected_rois_for_action()
        if not rois:
            self.statusBar().showMessage("No ROI selected.")
            return
        spatial_dims = tuple(self.session.spatial_dim_names)
        if "z" not in spatial_dims:
            self.statusBar().showMessage("Current experiment has no z dimension.")
            return
        indices = self._current_slice_indices()
        z_index = int(indices.get("z", self.session.active_z))
        deleted = 0
        for roi in rois:
            mask = self._roi_as_spatial_mask(roi)
            updated = _delete_mask_slice(mask, dims=spatial_dims, indices=indices)
            meta = _rasterized_roi_metadata(roi.metadata, "delete_z_slice")
            meta["deleted_z"] = z_index
            self._replace_roi(ROI(roi.name, updated, dims=spatial_dims, metadata=meta))
            deleted += 1
        self.refresh_roi_list()
        self.update_view(preserve_image_view=True)
        self.statusBar().showMessage(f"Deleted z slice {z_index + 1} from {deleted} ROI(s).")

    def snap_selected_rois_to_edges(self):
        rois = self._selected_rois_for_action()
        if not rois:
            self.statusBar().showMessage("No ROI selected.")
            return
        image = self._roi_source_image()
        if image is None:
            return
        display_shape = tuple(int(v) for v in np.asarray(image).shape[-2:])
        radius = 2
        snapped = 0
        for roi in rois:
            meta = dict(roi.metadata)
            item = self.roi_items.get(roi.name)
            if item is not None:
                kind = self.roi_item_kinds.get(roi.name, str(meta.get("shape", "mask")))
                _mask, item_meta = _mask_from_roi_item(item, kind, display_shape)
                meta.update(item_meta)
            if str(meta.get("shape")) != "polygon" or not meta.get("points"):
                continue
            points = np.asarray(meta.get("points"), dtype=float)
            snapped_points = _snap_polygon_points_to_gradient(points, image, radius=radius, strength=0.35)
            if snapped_points.shape[0] < 3:
                continue
            mask_2d = _polygon_mask(display_shape, snapped_points)
            updated_mask = replace_mask_slice(
                self._roi_as_spatial_mask(roi),
                mask_2d,
                dims=tuple(self.session.spatial_dim_names),
                indices=self._current_slice_indices(),
            )
            meta["points"] = snapped_points.tolist()
            meta["angle"] = 0.0
            meta["generated_by"] = "mrax.gui.snap_edges"
            meta["snap_radius"] = radius
            self._replace_roi(ROI(roi.name, updated_mask, dims=tuple(self.session.spatial_dim_names), metadata=meta))
            snapped += 1
        if snapped == 0:
            self.statusBar().showMessage("Snap currently applies to selected lasso/polygon ROIs.")
            return
        self.refresh_roi_list()
        self.update_view(preserve_image_view=True)
        self.statusBar().showMessage(f"Snapped {snapped} ROI contour(s) to local intensity edges.")

    def interpolate_selected_roi_z(self):
        spatial_dims = tuple(self.session.spatial_dim_names)
        if "z" not in spatial_dims:
            self.statusBar().showMessage("Current experiment has no z dimension.")
            return
        selected_rois = self._selected_rois_for_action()
        options = self._z_interpolation_options(default_extrapolation="none" if len(selected_rois) >= 2 else "constant")
        if options is None:
            return
        interpolation_strategy, extrapolation_strategy = options
        if len(selected_rois) >= 2:
            try:
                mask = stitch_masks_along_slices(
                    selected_rois,
                    spatial_shape=tuple(self.session.spatial_shape),
                    spatial_dims=spatial_dims,
                    indices=self._current_slice_indices(),
                    slice_dim="z",
                )
            except Exception as exc:
                self.statusBar().showMessage(f"Could not stitch selected ROIs for interpolation: {exc}")
                return
            source_z = sorted(_source_z_index(roi, default=int(self.session.active_z)) for roi in selected_rois)
            if len(set(source_z)) < 2:
                self.statusBar().showMessage("Select ROIs from at least two different z slices to interpolate between them.")
                return
            interpolated = interpolate_mask_along_slices(
                mask,
                dims=spatial_dims,
                slice_dim="z",
                extrapolate=extrapolation_strategy != "none",
                interpolation=interpolation_strategy,
                extrapolation=extrapolation_strategy,
            )
            name = _unique_roi_name(self.session.rois, "interpolated")
            roi = ROI(
                name,
                interpolated,
                dims=spatial_dims,
                metadata={
                    "shape": "mask",
                    "color": _roi_color(len(self.session.rois)),
                    "source_experiment": self.session.selected_experiment,
                    "source_rois": [roi.name for roi in selected_rois],
                    "source_z": source_z,
                    "interpolation_strategy": interpolation_strategy,
                    "extrapolation_strategy": extrapolation_strategy,
                    "generated_by": "mrax.gui.interpolate_selected_z",
                },
            )
            self.session.add_roi(roi)
            self.session.fit_settings.selected_roi = name
            self.refresh_roi_list()
            self.update_view(preserve_image_view=True)
            self.mark_dirty("ROI set changed")
            self.statusBar().showMessage(
                f"Interpolated {len(selected_rois)} selected ROI(s) into '{name}' "
                f"({interpolation_strategy}, extrapolate={extrapolation_strategy})."
            )
            return

        roi = selected_rois[0] if selected_rois else self._selected_roi_for_volume_action()
        if roi is None:
            return
        mask = self._roi_as_spatial_mask(roi)
        interpolated = interpolate_mask_along_slices(
            mask,
            dims=spatial_dims,
            slice_dim="z",
            extrapolate=extrapolation_strategy != "none",
            interpolation=interpolation_strategy,
            extrapolation=extrapolation_strategy,
        )
        meta = dict(roi.metadata)
        meta["generated_by"] = "mrax.gui.interpolate_z"
        meta["interpolation_strategy"] = interpolation_strategy
        meta["extrapolation_strategy"] = extrapolation_strategy
        updated = ROI(roi.name, interpolated, dims=spatial_dims, metadata=meta)
        self._replace_roi(updated)
        self.refresh_roi_list()
        self.update_view(preserve_image_view=True)
        self.statusBar().showMessage(
            f"Interpolated ROI '{roi.name}' along z ({interpolation_strategy}, extrapolate={extrapolation_strategy})."
        )

    def _z_interpolation_options(self, *, default_extrapolation: str = "constant") -> tuple[str, str] | None:
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Interpolate ROI Z")
        layout = QtWidgets.QVBoxLayout(dialog)
        form = QtWidgets.QFormLayout()
        interpolation_combo = QtWidgets.QComboBox()
        for label, value in (
            ("Linear", "linear"),
            ("Spline", "spline"),
            ("Constant", "constant"),
        ):
            interpolation_combo.addItem(label, value)
        extrapolation_combo = QtWidgets.QComboBox()
        for label, value in (
            ("None", "none"),
            ("Constant", "constant"),
            ("Linear", "linear"),
            ("Spline", "spline"),
        ):
            extrapolation_combo.addItem(label, value)
        self._configure_wide_combo(interpolation_combo, contents=12)
        self._configure_wide_combo(extrapolation_combo, contents=12)
        default = str(default_extrapolation or "constant").lower()
        match = extrapolation_combo.findData(default)
        if match >= 0:
            extrapolation_combo.setCurrentIndex(match)
        form.addRow("Between annotated slices", interpolation_combo)
        form.addRow("Outside annotated slices", extrapolation_combo)
        layout.addLayout(form)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if self._exec_dialog(dialog) != QtWidgets.QDialog.Accepted:
            return None
        return str(interpolation_combo.currentData() or "linear"), str(extrapolation_combo.currentData() or "constant")

    def _configure_wide_combo(self, combo, *, contents: int = 12) -> None:
        try:
            combo.setMinimumContentsLength(int(contents))
        except Exception:
            _LOG.debug("Ignored optional GUI operation failure", exc_info=True)
        try:
            policy_enum = getattr(QtWidgets.QComboBox, "SizeAdjustPolicy", QtWidgets.QComboBox)
            combo.setSizeAdjustPolicy(policy_enum.AdjustToContents)
        except Exception:
            _LOG.debug("Ignored optional GUI operation failure", exc_info=True)
        try:
            combo.setMinimumWidth(self._scaled_px(max(150, int(contents) * 11)))
        except Exception:
            _LOG.debug("Ignored optional GUI operation failure", exc_info=True)

    def _selected_roi_for_volume_action(self) -> ROI | None:
        if not self.session.is_loaded:
            return None
        item = self.roi_list.currentItem()
        roi = self.session.roi(item.text()) if item is not None else self.session.roi()
        if roi is None:
            self.statusBar().showMessage("No ROI selected.")
        return roi

    def _roi_as_spatial_mask(self, roi: ROI, *, geometry_indices: Mapping[str, int] | None = None) -> np.ndarray:
        spatial_dims = tuple(self.session.spatial_dim_names)
        spatial_shape = tuple(self.session.spatial_shape)
        mask = np.asarray(roi.mask)
        indices = dict(geometry_indices or roi.metadata.get("slice_indices") or self._current_slice_indices())

        def geometry_spatial_mask() -> np.ndarray:
            plane_shape = tuple(spatial_shape[spatial_dims.index(dim)] for dim in ("x", "y") if dim in spatial_dims)
            geometry_mask = _geometry_mask_from_metadata(roi.metadata, plane_shape)
            if geometry_mask is None:
                raise ValueError(f"ROI '{roi.name}' cannot be aligned to the current image shape")
            return embed_mask_slice(
                geometry_mask,
                spatial_shape=spatial_shape,
                spatial_dims=spatial_dims,
                indices=indices,
            )

        if tuple(mask.shape) == spatial_shape and tuple(roi.dims or spatial_dims) == spatial_dims:
            return mask.copy()
        if mask.ndim == 2 and len(spatial_shape) >= 3:
            try:
                return embed_mask_slice(
                    mask,
                    spatial_shape=spatial_shape,
                    spatial_dims=spatial_dims,
                    indices=indices,
                )
            except Exception:
                return geometry_spatial_mask()
        try:
            return np.asarray(
                broadcast_mask(
                    mask,
                    spatial_shape,
                    mask_dims=roi.dims,
                    target_dims=spatial_dims,
                )
            )
        except Exception:
            return geometry_spatial_mask()

    def update_plot(self):
        self._draw_signal_plot(self.plot_widget)

    def _draw_signal_plot(self, plot):
        plot.clear()
        if not self.session.is_loaded:
            return
        preview = self._preview_results_by_experiment.get(self.session.selected_experiment or "")
        if preview is not None:
            try:
                observed = np.asarray(preview["observed"]).reshape((-1,))
                predicted = np.asarray(preview["predicted"]).reshape((-1,))
                included = np.asarray(preview["included"], dtype=bool)[: observed.size]
                x_name, x = self._plot_axis_values(observed.size)
                x = np.asarray(x).reshape((-1,))
                if x.size != observed.size:
                    x = np.arange(observed.size)
                    x_name = "frame"
                plot.plot(x[included], observed[included], pen=None, symbol="o", symbolBrush="#2563eb")
                if (~included).any():
                    plot.plot(x[~included], observed[~included], pen=None, symbol="x", symbolBrush="#dc2626")
                plot.plot(x[: predicted.size], predicted, pen="#16a34a")
                plot.setTitle(f"Preview fit: {preview['model']}")
                plot.setLabel("bottom", x_name)
                roi = self.session.roi()
                plot.setLabel("left", f"{roi.name if roi is not None else 'ROI'} mean")
                return
            except Exception:
                plot.clear()
        if self.session.is_non_imaging:
            try:
                self._draw_non_imaging_plot(plot)
                return
            except Exception:
                plot.setTitle("Non-imaging plot unavailable")
                return
        roi = self.session.roi()
        try:
            if roi is not None:
                if not self._roi_has_pixels(roi):
                    plot.setTitle(f"ROI '{roi.name}' is empty")
                    plot.setLabel("bottom", "frame")
                    plot.setLabel("left", f"{roi.name} mean")
                    return
                extracted = self.session.extract_timecourse(roi.name, source="image")
                y = np.asarray(extracted["timecourse"].values if hasattr(extracted["timecourse"], "values") else extracted["timecourse"])
                y = y.reshape((-1,))
                x_name, x = self._plot_axis_values(y.size)
                x = np.asarray(x).reshape((-1,))
                if x.size != y.size:
                    x = np.arange(y.size)
                    x_name = "frame"
                included = self.session.frame_mask.as_array()[: y.size] if self.session.frame_mask else np.ones(y.size, dtype=bool)
                plot.plot(x[included], y[included], pen=None, symbol="o", symbolBrush="#2563eb", name="included")
                if (~included).any():
                    plot.plot(x[~included], y[~included], pen=None, symbol="x", symbolBrush="#dc2626", name="excluded")
                plot.setLabel("bottom", x_name)
                plot.setLabel("left", f"{roi.name} mean")
            else:
                image = self.session.image_array
                experiment = self.session.build_experiment()
                flat = reshape_dims_to_frames(image, experiment.dims).reshape((self.session.frame_count, -1))
                y = np.nanmean(flat, axis=1)
                x_name, x = self._plot_axis_values(y.size)
                x = np.asarray(x).reshape((-1,))
                if x.size != y.size:
                    x = np.arange(y.size)
                    x_name = "frame"
                plot.plot(x, y, pen="#2563eb")
                plot.setLabel("bottom", x_name)
                plot.setLabel("left", "mean intensity")
        except Exception:
            plot.setTitle("Plot unavailable")

    def _draw_non_imaging_plot(self, plot):
        vector = np.asarray(self.session.non_imaging_vector()).reshape((-1,))
        if vector.size == 0:
            plot.setTitle("No non-imaging data")
            return
        x = np.arange(int(vector.size))
        if np.iscomplexobj(vector):
            y = np.abs(vector)
            y_label = "magnitude"
        else:
            y = vector.astype(float, copy=False)
            y_label = "signal"
        finite = np.isfinite(y)
        if not np.any(finite):
            plot.setTitle("Non-imaging data contains no finite samples")
            return
        plot.plot(x[finite], y[finite], pen="#2563eb")
        role = self.session.data_role.replace("_", " ")
        plot.setTitle(f"{role}: {self.session.selected_experiment}")
        plot.setLabel("bottom", "sample")
        plot.setLabel("left", y_label)

    def _roi_has_pixels(self, roi: ROI) -> bool:
        try:
            return bool(np.any(np.asarray(self._roi_as_spatial_mask(roi)) != 0))
        except Exception:
            return bool(np.any(np.asarray(roi.mask) != 0))

    def update_inspector(self):
        if not self.session.data:
            self.inspector.setPlainText("Open a study_manifest.json to inspect MRax data.")
            return
        if not self.session.is_loaded:
            rows = list(self.session.summary_by_name.values())
            self.inspector.setPlainText(
                "\n".join(
                    [
                        f"Study: {self.session.study_name}",
                        f"Source: {self.session.source_kind or 'none'}",
                        f"Manifest: {self.session.manifest_path}",
                        f"Experiments: {len(rows)}",
                    ]
                )
            )
            return
        exp = self.session.current_experiment
        lines = [
            f"Study: {self.session.study_name}",
            f"Source: {self.session.source_kind or 'none'}",
            f"Manifest: {self.session.manifest_path}",
            f"Experiment: {self.session.selected_experiment}",
            f"Model: {exp.get('model')}",
            f"Type: {exp.get('exp_type')}",
            f"Data role: {self.session.data_role}",
            f"Non-imaging: {'yes' if self.session.is_non_imaging else 'no'}",
            f"Dims: {self.session.experiment_dims}",
            f"Image shape: {tuple(self.session.image_array.shape)}",
            f"K-space shape: {None if self.session.kspace_array is None else tuple(self.session.kspace_array.shape)}",
            f"K-space display: {self.session.active_kspace_mode}, centering={self.session.active_kspace_centering}",
            f"Maps: {', '.join(self.session.map_names) or 'none'}",
            f"Observables: {', '.join(sorted((exp.get('observables') or {}).keys())) or 'none'}",
            f"ROIs: {', '.join(roi.name for roi in self.session.rois) or 'none'}",
        ]
        if self.session.active_map:
            lines.append(f"Active map unit: {self.session.map_unit(self.session.active_map) or 'unknown'}")
        self.inspector.setPlainText("\n".join(lines))

    def update_metadata(self):
        if not self.session.data:
            self.metadata_tree.clear()
            self.metadata_detail.setPlainText("Open a study to inspect metadata.")
            return
        payload = (
            _experiment_metadata_payload(self.session)
            if self.session.is_loaded
            else {
                "study": {
                    "name": self.session.study_name,
                    "source_kind": self.session.source_kind,
                    "manifest_path": str(self.session.manifest_path or ""),
                    "experiment_count": len(self.session.experiment_names),
                },
                "study_globals": (self.session.data or {}).get("study_globals") or {},
            }
        )
        self.metadata_tree.blockSignals(True)
        try:
            self.metadata_tree.clear()
            for key, value in payload.items():
                self._add_metadata_tree_item(self.metadata_tree, str(key), value, str(key))
            self.metadata_tree.resizeColumnToContents(0)
            self.metadata_tree.resizeColumnToContents(1)
            self.metadata_tree.expandToDepth(1)
        finally:
            self.metadata_tree.blockSignals(False)
        self.filter_metadata_tree(self.metadata_filter_edit.text())
        first = self.metadata_tree.topLevelItem(0)
        if first is not None:
            self.metadata_tree.setCurrentItem(first)
            self.on_metadata_item_changed(first, None)
        else:
            self.metadata_detail.clear()

    def _add_metadata_tree_item(self, parent, name: str, value: Any, path: str):
        summary = _metadata_value_summary(value)
        item = QtWidgets.QTreeWidgetItem([name, summary["value"], summary["type"]])
        item.setData(0, QtCore.Qt.UserRole, path)
        item.setData(1, QtCore.Qt.UserRole, _metadata_to_jsonable(value))
        parent.addTopLevelItem(item) if isinstance(parent, QtWidgets.QTreeWidget) else parent.addChild(item)
        if isinstance(value, dict):
            for key in sorted(value.keys(), key=lambda item_key: str(item_key).lower()):
                child_name = str(key)
                self._add_metadata_tree_item(item, child_name, value[key], f"{path}.{child_name}")
        elif isinstance(value, (list, tuple)) and not _metadata_sequence_is_scalar(value):
            for idx, child in enumerate(value):
                self._add_metadata_tree_item(item, f"[{idx}]", child, f"{path}[{idx}]")
        return item

    def on_metadata_item_changed(self, current, _previous):
        if current is None:
            self.metadata_detail.clear()
            return
        path = current.data(0, QtCore.Qt.UserRole) or current.text(0)
        value = current.data(1, QtCore.Qt.UserRole)
        detail = {
            "path": path,
            "name": current.text(0),
            "type": current.text(2),
            "value": value,
        }
        self.metadata_detail.setPlainText(json.dumps(detail, indent=2, sort_keys=True))

    def filter_metadata_tree(self, text: str):
        needle = str(text or "").strip().lower()
        for idx in range(self.metadata_tree.topLevelItemCount()):
            item = self.metadata_tree.topLevelItem(idx)
            self._filter_metadata_item(item, needle)

    def _filter_metadata_item(self, item, needle: str) -> bool:
        own_text = " ".join(item.text(col) for col in range(item.columnCount())).lower()
        path = str(item.data(0, QtCore.Qt.UserRole) or "").lower()
        matched = not needle or needle in own_text or needle in path
        child_match = False
        for idx in range(item.childCount()):
            child_match = self._filter_metadata_item(item.child(idx), needle) or child_match
        visible = matched or child_match
        item.setHidden(not visible)
        if needle and child_match:
            item.setExpanded(True)
        return visible

    def update_summary(self):
        if not self.session.data:
            self.summary_table.setRowCount(0)
            self.summary_plot.clear()
            return
        rows = list(self.session.summary_by_name.values())
        headers = ["name", "status", "exp_type", "model", "dataset", "dims", "observables", "issue"]
        self.summary_table.setColumnCount(len(headers))
        self.summary_table.setHorizontalHeaderLabels(headers)
        self.summary_table.setRowCount(len(rows))
        for row_idx, row in enumerate(rows):
            for col_idx, key in enumerate(headers):
                val = row.get(key, "")
                if isinstance(val, dict):
                    val = ", ".join(f"{k}={v}" for k, v in val.items())
                elif isinstance(val, (list, tuple)):
                    val = ", ".join(str(v) for v in val)
                self.summary_table.setItem(row_idx, col_idx, QtWidgets.QTableWidgetItem(str(val)))
        self.summary_table.resizeColumnsToContents()
        self.summary_plot.clear()
        counts: dict[str, int] = {}
        for row in rows:
            if row.get("status") and row.get("status") != "loaded":
                continue
            key = str(row.get("exp_type") or "unknown")
            counts[key] = counts.get(key, 0) + 1
        if counts:
            labels = list(counts.keys())
            values = [counts[label] for label in labels]
            bars = pg.BarGraphItem(x=np.arange(len(labels)), height=values, width=0.65, brush="#2563eb")
            self.summary_plot.addItem(bars)
            axis = self.summary_plot.getAxis("bottom")
            axis.setTicks([list(enumerate(labels))])
            self.summary_plot.setLabel("left", "scans")
            self.summary_plot.setTitle("Experiments by type")

    def open_summary_window(self):
        if not self.session.data:
            return
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("MRax Study Summary")
        dialog.resize(900, 600)
        layout = QtWidgets.QVBoxLayout(dialog)
        table = QtWidgets.QTableWidget()
        rows = list(self.session.summary_by_name.values())
        headers = ["name", "status", "exp_type", "model", "dataset", "dims", "observables", "issue"]
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setRowCount(len(rows))
        for row_idx, row in enumerate(rows):
            for col_idx, key in enumerate(headers):
                val = row.get(key, "")
                if isinstance(val, dict):
                    val = ", ".join(f"{k}={v}" for k, v in val.items())
                elif isinstance(val, (list, tuple)):
                    val = ", ".join(str(v) for v in val)
                table.setItem(row_idx, col_idx, QtWidgets.QTableWidgetItem(str(val)))
        table.resizeColumnsToContents()
        layout.addWidget(table)
        self._show_centered_dialog(dialog)
        dialog.setAttribute(QtCore.Qt.WA_DeleteOnClose)

    def reset_image_view(self):
        try:
            self.image_view.getView().autoRange()
        except Exception:
            _LOG.debug("Ignored optional GUI operation failure", exc_info=True)
        try:
            self.plot_widget.autoRange()
        except Exception:
            _LOG.debug("Ignored optional GUI operation failure", exc_info=True)

    def open_enlarged_image(self):
        if not self.session.is_loaded:
            return
        view = _view_key(self.view_combo.currentText())
        try:
            if self._displayed_image is not None:
                image = np.asarray(self._displayed_image)
            elif view == "k_space":
                image = self.session.kspace_image()
            elif view == "mrax_reconstruction":
                image = self.session.reconstruction_image()
            elif view == "vendor_vs_mrax":
                vendor, mrax = _matched_2d_pair(self.session.frame_image(), self.session.reconstruction_image())
                image = _side_by_side_images(vendor, mrax)
            elif view == "vendor_mrax_diff":
                vendor, mrax = _matched_2d_pair(self.session.frame_image(), self.session.reconstruction_image())
                image = vendor - mrax
            elif view == "map":
                image = self.session.map_image()
            else:
                image = self.session.frame_image()
        except Exception as exc:
            self.statusBar().showMessage(str(exc))
            return
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(f"MRax {self.view_combo.currentText()} Viewer")
        dialog.resize(980, 820)
        layout = QtWidgets.QVBoxLayout(dialog)
        image_view = pg.ImageView()
        image_view.ui.roiBtn.hide()
        image_view.ui.menuBtn.hide()
        image_view.setImage(np.asarray(image), autoLevels=True)
        layout.addWidget(image_view)
        reset = QtWidgets.QPushButton("Reset View")
        reset.clicked.connect(lambda: image_view.getView().autoRange())
        layout.addWidget(reset)
        self._show_centered_dialog(dialog)
        dialog.setAttribute(QtCore.Qt.WA_DeleteOnClose)

    def open_enlarged_plot(self):
        if not self.session.is_loaded:
            return
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("MRax Signal Preview")
        dialog.resize(980, 620)
        layout = QtWidgets.QVBoxLayout(dialog)
        plot = pg.PlotWidget()
        plot.showGrid(x=True, y=True, alpha=0.25)
        layout.addWidget(plot)
        self._draw_signal_plot(plot)
        reset = QtWidgets.QPushButton("Reset View")
        reset.clicked.connect(plot.autoRange)
        layout.addWidget(reset)
        self._show_centered_dialog(dialog)
        dialog.setAttribute(QtCore.Qt.WA_DeleteOnClose)

    def update_diagnostics(self):
        lines: list[str] = []
        if not self.session.data:
            lines.append("No manifest loaded.")
        elif not self.session.is_loaded:
            lines.append(f"Study '{self.session.study_name}' has no selected experiment.")
        else:
            lines.extend(f"[{diag.level}] {diag.message}" for diag in self.session.diagnostics())
        lines.extend(["", "Application warnings and errors"])
        lines.extend(_format_gui_diagnostic_events(list(self.diagnostic_events), include_details=True))
        self.diagnostics_text.setPlainText("\n".join(lines))
        self._update_diagnostics_attention()

    def clear_diagnostic_events(self):
        self.diagnostic_events.clear()
        self.update_diagnostics()
        self.statusBar().showMessage("Application diagnostic event log cleared.")

    def save_diagnostic_report_dialog(self):
        default_name = "mrax_diagnostic_report.txt"
        if self.session.selected_experiment:
            default_name = f"{_safe_filename(self.session.selected_experiment)}_mrax_diagnostic_report.txt"
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Save Diagnostic Report")
        dialog.resize(self._scaled_px(680), self._scaled_px(420))
        layout = QtWidgets.QVBoxLayout(dialog)
        form = QtWidgets.QFormLayout()
        report_edit = QtWidgets.QLineEdit(str(Path.cwd() / default_name))
        report_browse = QtWidgets.QPushButton("Browse")
        report_row = QtWidgets.QHBoxLayout()
        report_row.addWidget(report_edit, 1)
        report_row.addWidget(report_browse)
        report_widget = QtWidgets.QWidget()
        report_widget.setLayout(report_row)
        form.addRow("Report file", report_widget)
        message_edit = QtWidgets.QPlainTextEdit()
        message_edit.setPlaceholderText("Optional note for developers: what you tried, what happened, expected behavior, relevant sample or workflow details.")
        message_edit.setMaximumHeight(self._scaled_px(120))
        form.addRow("Manual message", message_edit)
        bundle_check = QtWidgets.QCheckBox("Attach current GUI state and data as MRax bundle")
        bundle_check.setChecked(bool(self.session.is_loaded))
        bundle_check.setEnabled(bool(self.session.is_loaded))
        bundle_edit = QtWidgets.QLineEdit(str((Path.cwd() / default_name).with_suffix(".mrax")))
        bundle_browse = QtWidgets.QPushButton("Browse")
        bundle_row = QtWidgets.QHBoxLayout()
        bundle_row.addWidget(bundle_edit, 1)
        bundle_row.addWidget(bundle_browse)
        bundle_widget = QtWidgets.QWidget()
        bundle_widget.setLayout(bundle_row)
        form.addRow(bundle_check)
        form.addRow("Bundle file", bundle_widget)
        layout.addLayout(form)

        def browse_report():
            path, _ = self._get_save_file_name(
                "Save Diagnostic Report",
                report_edit.text() or str(Path.cwd() / default_name),
                "Text files (*.txt);;All files (*)",
            )
            if path:
                report_edit.setText(path)
                if not bundle_edit.text().strip():
                    bundle_edit.setText(str(Path(path).with_suffix(".mrax")))

        def browse_bundle():
            path, _ = self._get_save_file_name(
                "Save Diagnostic Bundle",
                bundle_edit.text() or str(Path(report_edit.text() or default_name).with_suffix(".mrax")),
                "MRax bundles (*.mrax);;All files (*)",
            )
            if path:
                bundle_edit.setText(path)

        def sync_bundle_enabled(checked: bool):
            bundle_edit.setEnabled(bool(checked))
            bundle_browse.setEnabled(bool(checked))

        report_browse.clicked.connect(browse_report)
        bundle_browse.clicked.connect(browse_bundle)
        bundle_check.toggled.connect(sync_bundle_enabled)
        sync_bundle_enabled(bundle_check.isChecked())
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if self._exec_dialog(dialog) != QtWidgets.QDialog.Accepted:
            return
        report_path = Path(report_edit.text()).expanduser()
        if report_path.suffix == "":
            report_path = report_path.with_suffix(".txt")
        manual_message = message_edit.toPlainText()
        attachments: dict[str, Any] = {}
        try:
            if bundle_check.isChecked():
                bundle_path = Path(bundle_edit.text()).expanduser()
                if bundle_path.suffix.lower() != ".mrax":
                    bundle_path = bundle_path.with_suffix(".mrax")
                saved = self.save_mrax_bundle_file(
                    bundle_path,
                    include_state=True,
                    include_rois=True,
                    include_study=True,
                    include_unloaded_raw=False,
                )
                attachments["mrax_bundle"] = str(bundle_path)
                attachments["mrax_bundle_items"] = len(saved)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                self.build_diagnostic_report(
                    manual_message=manual_message,
                    attachments=attachments,
                ),
                encoding="utf-8",
            )
        except Exception:
            self.show_error("Could not save diagnostic report", traceback.format_exc())
            return
        suffix = f" and bundle {attachments['mrax_bundle']}" if attachments.get("mrax_bundle") else ""
        self.statusBar().showMessage(f"Saved diagnostic report to {report_path}{suffix}")

    def build_diagnostic_report(
        self,
        *,
        manual_message: str = "",
        attachments: Mapping[str, Any] | None = None,
    ) -> str:
        if self.session.is_loaded:
            session_diagnostics = [f"[{diag.level}] {diag.message}" for diag in self.session.diagnostics()]
        else:
            session_diagnostics = ["No manifest loaded."]
        return _format_gui_diagnostic_report(
            events=list(self.diagnostic_events),
            session_state=self._diagnostic_session_state(),
            session_diagnostics=session_diagnostics,
            user_message=manual_message,
            attachments=attachments or {},
        )

    def _diagnostic_session_state(self) -> dict[str, Any]:
        source = None
        if self.session.manifest_path is not None:
            source = str(self.session.manifest_path)
        elif self.session.import_metadata:
            source = str(self.session.import_metadata.get("source") or "")
        return {
            "source": source,
            "manifest_path": "" if self.session.manifest_path is None else str(self.session.manifest_path),
            "source_kind": self.session.source_kind,
            "selected_experiment": self.session.selected_experiment,
            "view": self.view_combo.currentText() if hasattr(self, "view_combo") else "",
            "active_frame": self.session.active_frame,
            "active_z": self.session.active_z,
            "active_indices": dict(self.session.active_indices),
            "active_map": self.session.active_map,
            "roi_names": [roi.name for roi in self.session.rois],
            "dirty": self._dirty,
            "dirty_reason": self._dirty_reason,
        }

    def import_roi_dialog(self):
        if self.is_busy:
            return
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Import ROI / Mask")
        dialog_min = (self._scaled_px(960), self._scaled_px(660))
        dialog.setMinimumSize(*dialog_min)
        dialog.resize(self._scaled_px(1080), self._scaled_px(740))
        dialog.setProperty("mrax_preserve_initial_size", True)
        root_layout = QtWidgets.QVBoxLayout(dialog)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(10)
        content_splitter = QtWidgets.QSplitter(dialog)
        content_splitter.setOrientation(QtCore.Qt.Horizontal)
        content_splitter.setChildrenCollapsible(False)
        left_scroll = QtWidgets.QScrollArea(dialog)
        left_scroll.setWidgetResizable(True)
        left_scroll.setMinimumWidth(self._scaled_px(540))
        frame_shape = getattr(QtWidgets.QFrame, "Shape", QtWidgets.QFrame).NoFrame
        left_scroll.setFrameShape(frame_shape)
        left_panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(left_panel)
        layout.setContentsMargins(0, 0, 8, 0)
        layout.setSpacing(10)
        left_scroll.setWidget(left_panel)
        right_panel = QtWidgets.QWidget()
        right_panel.setMinimumWidth(self._scaled_px(360))
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(8, 0, 0, 0)
        right_layout.setSpacing(8)
        content_splitter.addWidget(left_scroll)
        content_splitter.addWidget(right_panel)
        content_splitter.setStretchFactor(0, 3)
        content_splitter.setStretchFactor(1, 2)
        content_splitter.setSizes([self._scaled_px(640), self._scaled_px(420)])
        root_layout.addWidget(content_splitter, 1)
        form = QtWidgets.QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        field_growth = getattr(QtWidgets.QFormLayout, "FieldGrowthPolicy", QtWidgets.QFormLayout).AllNonFixedFieldsGrow
        form.setFieldGrowthPolicy(field_growth)

        source_combo = QtWidgets.QComboBox()
        source_combo.addItem("MRax ROI set (.json/.npz)", "roi_set")
        source_combo.addItem("NIfTI mask", "nifti")
        source_combo.addItem("NumPy mask (.npy/.npz)", "numpy")
        source_combo.addItem("Raw binary mask", "raw")
        form.addRow("Source", source_combo)

        file_row = QtWidgets.QHBoxLayout()
        file_edit = QtWidgets.QLineEdit("")
        browse_btn = QtWidgets.QPushButton("Browse...")
        file_row.addWidget(file_edit, 1)
        file_row.addWidget(browse_btn)
        form.addRow("File", file_row)

        default_mask_shape = tuple(self.session.spatial_shape) if self.session.is_loaded else (64, 64)
        default_mask_dims = tuple(self.session.spatial_dim_names) if self.session.is_loaded else ("x", "y")
        name_edit = QtWidgets.QLineEdit(_unique_roi_name(list(self.session.rois), "imported_roi"))
        form.addRow("ROI name", name_edit)
        shape_edit = QtWidgets.QLineEdit(",".join(str(size) for size in default_mask_shape))
        form.addRow("Mask shape", shape_edit)
        order_edit = QtWidgets.QLineEdit(",".join(default_mask_dims))
        form.addRow("Dimension order", order_edit)

        dtype_combo = QtWidgets.QComboBox()
        for dtype in ("uint8", "bool", "uint16", "int16", "int32", "float32", "float64"):
            dtype_combo.addItem(dtype)
        form.addRow("Raw dtype", dtype_combo)
        endian_combo = QtWidgets.QComboBox()
        endian_combo.addItem("Native endian", "native")
        endian_combo.addItem("Little endian", "little")
        endian_combo.addItem("Big endian", "big")
        form.addRow("Raw endian", endian_combo)
        offset_spin = QtWidgets.QSpinBox()
        offset_spin.setRange(0, 2_000_000_000)
        offset_spin.setSuffix(" bytes")
        form.addRow("Raw byte offset", offset_spin)
        memory_order_combo = QtWidgets.QComboBox()
        memory_order_combo.addItem("C / row-major", "C")
        memory_order_combo.addItem("Fortran / column-major", "F")
        form.addRow("Memory order", memory_order_combo)
        npz_key_edit = QtWidgets.QLineEdit("")
        form.addRow("NPZ array key", npz_key_edit)
        settings_box = QtWidgets.QGroupBox("ROI / Mask Data")
        settings_layout = QtWidgets.QVBoxLayout(settings_box)
        settings_layout.addLayout(form)
        layout.addWidget(settings_box)

        metadata_edit = QtWidgets.QPlainTextEdit("{}")
        metadata_edit.setMinimumHeight(self._scaled_px(110))
        metadata_edit.setMaximumHeight(self._scaled_px(160))
        layout.addWidget(_section_box(QtWidgets, "Extra ROI Metadata JSON", metadata_edit))
        layout.addStretch(1)

        preview_panel, preview_fields = self._make_import_preview_panel(
            (
                ("status", "Status", 72),
                ("file", "File and Source", 96),
                ("mask", "Mask Array and Dimensions", 156),
                ("raw", "Raw Diagnostics", 148),
                ("roi", "ROI Details", 164),
            )
        )
        right_layout.addWidget(preview_panel, 1)
        preview_timer = QtCore.QTimer(dialog)
        preview_timer.setSingleShot(True)

        def update_source_controls(*_args):
            source = str(source_combo.currentData() or "roi_set")
            is_roi_set = source == "roi_set"
            is_raw = source == "raw"
            name_edit.setEnabled(not is_roi_set)
            shape_edit.setEnabled(not is_roi_set)
            order_edit.setEnabled(not is_roi_set)
            dtype_combo.setEnabled(is_raw)
            endian_combo.setEnabled(is_raw)
            offset_spin.setEnabled(is_raw)
            memory_order_combo.setEnabled(source in {"nifti", "numpy", "raw"})
            npz_key_edit.setEnabled(source == "numpy")
            if source in {"nifti", "numpy"} and shape_edit.text().strip() == ",".join(str(size) for size in default_mask_shape):
                shape_edit.clear()
            if source in {"nifti", "numpy"} and order_edit.text().strip() == ",".join(default_mask_dims):
                order_edit.clear()
            if is_raw and not shape_edit.text().strip():
                shape_edit.setText(",".join(str(size) for size in default_mask_shape))
            if is_raw and not order_edit.text().strip():
                order_edit.setText(",".join(default_mask_dims))
            schedule_preview()

        def browse_source():
            source = str(source_combo.currentData() or "roi_set")
            filters = {
                "roi_set": "MRax ROI sets (*.json *.npz);;All files (*)",
                "nifti": "NIfTI masks (*.nii *.nii.gz);;All files (*)",
                "numpy": "NumPy masks (*.npy *.npz);;All files (*)",
                "raw": "Raw binary masks (*);;All files (*)",
            }
            path, _ = self._get_open_file_name("Choose ROI or mask data", str(Path.cwd()), filters.get(source, "All files (*)"))
            if path:
                file_edit.setText(path)
                if source != "roi_set" and (not name_edit.text().strip() or name_edit.text().startswith("imported_roi")):
                    name_edit.setText(_unique_roi_name(list(self.session.rois), Path(path).stem))
                schedule_preview()

        def schedule_preview(*_args):
            preview_timer.start(250)

        def dims_for_mask_array(array: np.ndarray, source: str, order: Sequence[str]) -> tuple[str, ...]:
            if order:
                if len(order) != int(array.ndim):
                    raise ValueError(f"Dimension order must have {array.ndim} entries for mask shape {array.shape}.")
                return tuple(order)
            spatial_dims = tuple(self.session.spatial_dim_names) if self.session.is_loaded else ()
            if spatial_dims:
                if array.ndim == len(spatial_dims):
                    return spatial_dims
                if array.ndim == 2 and len(spatial_dims) >= 2:
                    return spatial_dims[:2]
            return _default_dim_order(array.ndim, source)

        def rois_from_array(array: np.ndarray, *, source: str, dims: Sequence[str], metadata: Mapping[str, Any]) -> list[ROI]:
            arr = np.asarray(array)
            roi_name = name_edit.text().strip() or _unique_roi_name(list(self.session.rois), "imported_roi")
            spatial_shape = tuple(self.session.spatial_shape) if self.session.is_loaded else None
            meta = dict(metadata or {})
            meta.setdefault("import_format", source)
            if self.session.is_loaded and arr.ndim == 2 and spatial_shape is not None and len(spatial_shape) >= 3:
                meta.setdefault("slice_indices", self._current_slice_indices())
                return [
                    import_mask(
                        arr,
                        name=roi_name,
                        spatial_shape=tuple(self.session.frame_image().shape[-2:]),
                        dims=tuple(dims)[:2] if dims else None,
                        metadata=meta,
                    )
                ]
            return [import_mask(arr, name=roi_name, spatial_shape=spatial_shape, dims=tuple(dims) if dims else None, metadata=meta)]

        def preview_payload() -> dict[str, Any]:
            source = str(source_combo.currentData() or "roi_set")
            path = Path(file_edit.text()).expanduser() if file_edit.text().strip() else None
            extra_metadata = _metadata_from_json_text(metadata_edit.toPlainText())
            if source == "roi_set":
                if path is None or not path.is_file():
                    raise FileNotFoundError("Choose an MRax ROI set .json or .npz file")
                rois = load_roi_set(path)
                if extra_metadata:
                    rois = [
                        ROI(roi.name, roi.mask, dims=roi.dims, metadata={**dict(roi.metadata or {}), **extra_metadata})
                        for roi in rois
                    ]
                return {"kind": "roi", "source": source, "path": path, "rois": rois, "array": None}
            shape = _parse_shape_text(shape_edit.text())
            order = _parse_dim_order(order_edit.text())
            data, metadata = _load_experiment_array_from_dialog(
                source=source,
                path=path,
                shape=shape,
                dtype=dtype_combo.currentText(),
                endian=str(endian_combo.currentData() or "native"),
                byte_offset=int(offset_spin.value()),
                memory_order=str(memory_order_combo.currentData() or "C"),
                npz_key=npz_key_edit.text().strip() or None,
            )
            metadata.update(extra_metadata)
            dims = dims_for_mask_array(np.asarray(data), source, order)
            rois = rois_from_array(np.asarray(data), source=source, dims=dims, metadata=metadata)
            return {
                "kind": "roi",
                "source": source,
                "path": path,
                "data": data,
                "dims": dims,
                "rois": rois,
                "metadata": metadata,
            }

        def update_preview():
            source = str(source_combo.currentData() or "roi_set")
            path = Path(file_edit.text()).expanduser() if file_edit.text().strip() else None
            shape: tuple[int, ...] = ()
            order: tuple[str, ...] = ()
            try:
                shape = _parse_shape_text(shape_edit.text())
            except Exception:
                _LOG.debug("Ignored optional GUI operation failure", exc_info=True)
            try:
                order = _parse_dim_order(order_edit.text())
            except Exception:
                _LOG.debug("Ignored optional GUI operation failure", exc_info=True)
            status_lines: list[str] = []
            file_lines = _import_file_preview_lines(path, source=source)
            file_lines.extend(
                [
                    f"roi name: {name_edit.text().strip() or '(auto)'}",
                    f"memory order: {str(memory_order_combo.currentData() or 'C')}",
                ]
            )
            mask_lines: list[str] = [
                f"requested shape: {shape if shape else '(use file shape)'}",
                f"dimension order: {', '.join(order) if order else '(auto)'}",
            ]
            raw_lines: list[str] = ["Not used for this source."] if source != "raw" else []
            roi_lines: list[str] = []
            if source == "raw":
                raw_lines = _raw_import_diagnostic_lines(
                    path,
                    dtype=dtype_combo.currentText(),
                    endian=str(endian_combo.currentData() or "native"),
                    shape=shape,
                    byte_offset=int(offset_spin.value()),
                    order=order,
                )
            try:
                payload = preview_payload()
                rois = list(payload.get("rois") or [])
                data = payload.get("data")
                if data is not None:
                    mask_lines.extend(_array_detail_lines(data, label="mask"))
                    mask_lines.append("dims: " + ", ".join(payload.get("dims") or ()))
                else:
                    mask_lines.append("MRax ROI set; masks are listed in ROI Details.")
                roi_lines.extend(_roi_detail_lines(rois))
                status_lines.append("settings are importable")
                status_lines.append("OK will add ROI/mask data to the active study.")
            except Exception as exc:
                status_lines.append("settings are not importable")
                status_lines.append(f"mismatch note: {exc}")
                if not roi_lines:
                    roi_lines.append("ROI details will be available after the file and shape settings are valid.")
            self._set_import_preview_fields(
                preview_fields,
                {
                    "status": status_lines,
                    "file": file_lines,
                    "mask": mask_lines,
                    "raw": raw_lines,
                    "roi": roi_lines,
                },
            )

        source_combo.currentIndexChanged.connect(update_source_controls)
        browse_btn.clicked.connect(browse_source)
        preview_timer.timeout.connect(update_preview)
        for widget in (file_edit, shape_edit, order_edit, npz_key_edit):
            widget.textChanged.connect(schedule_preview)
        dtype_combo.currentIndexChanged.connect(schedule_preview)
        endian_combo.currentIndexChanged.connect(schedule_preview)
        memory_order_combo.currentIndexChanged.connect(schedule_preview)
        offset_spin.valueChanged.connect(schedule_preview)
        metadata_edit.textChanged.connect(schedule_preview)
        update_source_controls()
        update_preview()

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        accepted_payload: dict[str, Any] = {}

        def try_accept_import():
            try:
                payload = preview_payload()
            except Exception as exc:
                update_preview()
                current = preview_fields["status"].toPlainText().rstrip()
                suffix = f"\nImport not applied: {exc}"
                preview_fields["status"].setPlainText((current + suffix).strip())
                self.statusBar().showMessage(f"ROI import settings mismatch: {exc}")
                return
            accepted_payload.clear()
            accepted_payload.update(payload)
            dialog.accept()

        buttons.accepted.connect(try_accept_import)
        buttons.rejected.connect(dialog.reject)
        root_layout.addWidget(buttons)
        if self._exec_dialog(dialog) != QtWidgets.QDialog.Accepted:
            return

        try:
            if not self.session.data:
                self.session.create_empty_study(self._workspace_label_for_session(self.session, "Untitled Study"))
            rois = list(accepted_payload.get("rois") or [])
            for roi in rois:
                self.session.add_roi(roi)
        except Exception:
            self.show_error("Could not import ROI/mask data", traceback.format_exc())
            return
        self.mark_dirty("ROI set changed")
        self.refresh_roi_list()
        self.update_view(preserve_image_view=True)
        self.update_inspector()
        self.update_metadata()
        self.statusBar().showMessage(f"Imported {len(rois)} ROI/mask item(s).")

    def load_roi_dialog(self):
        path, _ = self._get_open_file_name(
            "Load ROI set or NumPy mask",
            str(Path.cwd()),
            "ROI sets (*.json *.npz);;NumPy masks (*.npy *.npz);;All files (*)",
        )
        if not path:
            return
        self.load_roi_file(Path(path))

    def load_roi_file(self, path: Path, *, raw_mask: bool = False):
        self._load_roi_data(path, raw_mask=raw_mask)
        self.refresh_roi_list()
        self.update_view(preserve_image_view=True)
        self.mark_dirty("ROI set changed")

    def _preview_rois_for_import(self, path: Path | None) -> list[ROI]:
        if path is None or not Path(path).is_file():
            raise FileNotFoundError("Choose an ROI set or mask file")
        path = Path(path)
        kind = _detect_load_kind(path)
        if kind == "roi" and path.suffix.lower() in {".json", ".npz"}:
            return load_roi_set(path)
        dims = tuple(self.session.spatial_dim_names) if self.session.is_loaded else None
        spatial_shape = tuple(self.session.spatial_shape) if self.session.is_loaded else None
        raw = load_mask_array(path)
        if self.session.is_loaded and raw.ndim == 2 and spatial_shape is not None and len(spatial_shape) >= 3:
            return [
                import_mask(
                    raw,
                    name=path.stem,
                    spatial_shape=tuple(self.session.frame_image().shape[-2:]),
                    dims=dims[:2] if dims else None,
                    metadata={"slice_indices": self._current_slice_indices()},
                )
            ]
        return [import_mask(raw, name=path.stem, spatial_shape=spatial_shape, dims=dims)]

    def _load_roi_data(self, path: Path, *, raw_mask: bool = False):
        if not raw_mask and path.suffix in (".json", ".npz"):
            rois = load_roi_set(path)
        else:
            dims = tuple(self.session.spatial_dim_names) if self.session.is_loaded else None
            spatial_shape = tuple(self.session.spatial_shape) if self.session.is_loaded else None
            if self.session.is_loaded:
                raw = load_mask_array(path)
                if raw.ndim == 2 and spatial_shape is not None and len(spatial_shape) >= 3:
                    plane = import_mask(
                        raw,
                        name=path.stem,
                        spatial_shape=tuple(self.session.frame_image().shape[-2:]),
                        dims=dims[:2] if dims else None,
                        metadata={"slice_indices": self._current_slice_indices()},
                    )
                    rois = [plane]
                else:
                    rois = [import_mask(raw, name=path.stem, spatial_shape=spatial_shape, dims=dims)]
            else:
                rois = [import_mask(path, name=path.stem, spatial_shape=spatial_shape, dims=dims)]
        for roi in rois:
            self.session.add_roi(roi)

    def save_roi_dialog(self):
        if not self.session.rois:
            self.statusBar().showMessage("No ROIs to save.")
            return
        path, _ = self._get_save_file_name(
            "Save ROI set",
            str(Path.cwd() / "mrax_rois.npz"),
            "ROI set (*.npz);;JSON sidecar (*.json)",
        )
        if not path:
            return
        try:
            save_roi_set(
                self.session.rois,
                path,
                metadata={
                    "manifest": str(self.session.manifest_path or ""),
                    "experiment": self.session.selected_experiment,
                },
            )
        except Exception:
            self.show_error("Could not save ROI set", traceback.format_exc())
            return
        self.statusBar().showMessage(f"Saved ROI set to {path}")

    def export_dialog(self):
        if not self.session.is_loaded:
            self.statusBar().showMessage("Load a study before exporting.")
            return
        self.refresh_registration_tab()
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Export MRax Data")
        dialog.resize(self._scaled_px(760), self._scaled_px(620))
        layout = QtWidgets.QVBoxLayout(dialog)
        reference_info = QtWidgets.QLabel(_roi_nifti_reference_summary(self.session))
        reference_info.setWordWrap(True)
        reference_info.setStatusTip(
            "ROI NIfTI files use the affine from the experiment on which each mask was created; "
            "identity is used only when that reference geometry is unavailable."
        )
        layout.addWidget(reference_info)
        tabs = QtWidgets.QTabWidget()
        bulk_tab = QtWidgets.QWidget()
        bulk_layout = QtWidgets.QVBoxLayout(bulk_tab)
        output_row = QtWidgets.QHBoxLayout()
        output_edit = QtWidgets.QLineEdit(str(Path.cwd() / f"{self._default_study_name()}_exports"))
        browse = QtWidgets.QPushButton("Browse")
        browse.clicked.connect(
            lambda: output_edit.setText(
                self._get_existing_directory(
                    "Choose export folder",
                    output_edit.text() or str(Path.cwd()),
                )
                or output_edit.text()
            )
        )
        output_row.addWidget(output_edit, 1)
        output_row.addWidget(browse)
        bulk_layout.addWidget(QtWidgets.QLabel("Export folder"))
        bulk_layout.addLayout(output_row)

        include_images = QtWidgets.QCheckBox("Images")
        include_images.setChecked(True)
        include_kspace = QtWidgets.QCheckBox("K-space")
        include_kspace.setChecked(True)
        include_maps = QtWidgets.QCheckBox("Fitted maps")
        include_maps.setChecked(True)
        include_rois = QtWidgets.QCheckBox("ROIs")
        include_rois.setChecked(bool(self.session.rois))
        numpy_check = QtWidgets.QCheckBox("NumPy files (.npy)")
        numpy_check.setChecked(True)
        nifti_check = QtWidgets.QCheckBox("NIfTI files (.nii)")
        nifti_check.setChecked(False)
        for widget in (include_images, include_kspace, include_maps, include_rois, numpy_check, nifti_check):
            bulk_layout.addWidget(widget)
        transform_rois_check = QtWidgets.QCheckBox("Apply available registration transform to ROIs")
        transform_rois_check.setChecked(False)
        transform_rois_check.setEnabled(bool(self.registration_results and self.session.rois))
        bulk_layout.addWidget(transform_rois_check)
        transform_row = QtWidgets.QHBoxLayout()
        export_transform_combo = QtWidgets.QComboBox()
        export_direction_combo = QtWidgets.QComboBox()
        export_direction_combo.addItem("Reference ROI to moving grid", "reference_to_moving")
        export_direction_combo.addItem("Moving ROI to reference grid", "moving_to_reference")
        for key, value in sorted(self.registration_results.items()):
            export_transform_combo.addItem(self._registration_result_label(key, value), key)
        export_transform_combo.setEnabled(transform_rois_check.isEnabled())
        export_direction_combo.setEnabled(transform_rois_check.isEnabled())
        preview_export_transform = QtWidgets.QPushButton("Preview")
        preview_export_transform.setEnabled(transform_rois_check.isEnabled())
        preview_export_transform.clicked.connect(
            lambda: self.preview_registration_mask_transform(
                transform_key=export_transform_combo.currentData(),
                direction=export_direction_combo.currentData(),
            )
        )
        transform_row.addWidget(export_transform_combo, 1)
        transform_row.addWidget(export_direction_combo)
        transform_row.addWidget(preview_export_transform)
        bulk_layout.addLayout(transform_row)
        bulk_layout.addStretch(1)
        tabs.addTab(bulk_tab, "Study export")

        single_tab = QtWidgets.QWidget()
        single_layout = QtWidgets.QVBoxLayout(single_tab)
        object_tree = QtWidgets.QTreeWidget()
        object_tree.setHeaderLabels(["Object", "Type", "Shape"])
        object_tree.setRootIsDecorated(True)
        object_tree.setAlternatingRowColors(True)
        records = self._exportable_object_records()
        self._populate_export_object_tree(object_tree, records)
        single_layout.addWidget(object_tree, 1)

        nav_row = QtWidgets.QHBoxLayout()
        previous_btn = QtWidgets.QPushButton("Previous")
        next_btn = QtWidgets.QPushButton("Next")
        nav_row.addWidget(previous_btn)
        nav_row.addWidget(next_btn)
        nav_row.addStretch(1)
        single_layout.addLayout(nav_row)

        detail = QtWidgets.QPlainTextEdit()
        detail.setReadOnly(True)
        detail.setMaximumHeight(self._scaled_px(110))
        single_layout.addWidget(detail)

        form = QtWidgets.QFormLayout()
        format_combo = QtWidgets.QComboBox()
        self._configure_wide_combo(format_combo, contents=24)
        form.addRow("File type", format_combo)
        single_layout.addLayout(form)

        single_output_row = QtWidgets.QHBoxLayout()
        single_output_edit = QtWidgets.QLineEdit()
        single_browse = QtWidgets.QPushButton("Browse")
        single_output_row.addWidget(single_output_edit, 1)
        single_output_row.addWidget(single_browse)
        single_layout.addWidget(QtWidgets.QLabel("Output file"))
        single_layout.addLayout(single_output_row)
        tabs.addTab(single_tab, "Single object")
        layout.addWidget(tabs, 1)

        def selected_export_record() -> dict[str, Any] | None:
            item = object_tree.currentItem()
            if item is None:
                return None
            value = item.data(0, QtCore.Qt.UserRole)
            if value is None:
                return None
            try:
                return records[int(value)]
            except Exception:
                return None

        def selected_export_format() -> str | None:
            value = format_combo.currentData()
            return str(value) if value is not None else None

        def refresh_single_details():
            record = selected_export_record()
            format_combo.blockSignals(True)
            format_combo.clear()
            if record is None:
                detail.setPlainText("No exportable object selected.")
                single_output_edit.clear()
                format_combo.blockSignals(False)
                return
            formats = _single_export_file_types(record)
            for fmt in formats:
                format_combo.addItem(str(fmt["label"]), str(fmt["id"]))
            format_combo.blockSignals(False)
            detail.setPlainText(json.dumps(_single_export_record_metadata(record), indent=2, sort_keys=True))
            refresh_single_output_path()

        def refresh_single_output_path():
            record = selected_export_record()
            fmt = selected_export_format()
            if record is None or not fmt:
                return
            single_output_edit.setText(str(_single_export_default_path(record, fmt, Path.cwd())))

        def browse_single_output():
            record = selected_export_record()
            fmt = selected_export_format()
            if record is None or not fmt:
                return
            path, _ = self._get_save_file_name(
                "Export object",
                single_output_edit.text() or str(_single_export_default_path(record, fmt, Path.cwd())),
                _single_export_name_filter(fmt),
            )
            if path:
                single_output_edit.setText(path)

        object_tree.currentItemChanged.connect(lambda *_args: refresh_single_details())
        format_combo.currentIndexChanged.connect(lambda _idx: refresh_single_output_path())
        single_browse.clicked.connect(browse_single_output)
        previous_btn.clicked.connect(lambda: self._select_adjacent_export_object(object_tree, -1))
        next_btn.clicked.connect(lambda: self._select_adjacent_export_object(object_tree, 1))
        self._select_first_export_object(object_tree)
        refresh_single_details()

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if self._exec_dialog(dialog) != QtWidgets.QDialog.Accepted:
            return
        try:
            if tabs.currentWidget() is single_tab:
                record = selected_export_record()
                fmt = selected_export_format()
                if record is None or not fmt:
                    self.statusBar().showMessage("No export object selected.")
                    return
                saved = self.export_single_object(record, Path(single_output_edit.text()).expanduser(), fmt)
            else:
                saved = self.export_current_study(
                    Path(output_edit.text()).expanduser(),
                    include_images=include_images.isChecked(),
                    include_kspace=include_kspace.isChecked(),
                    include_maps=include_maps.isChecked(),
                    include_rois=include_rois.isChecked(),
                    export_numpy=numpy_check.isChecked(),
                    export_nifti=nifti_check.isChecked(),
                    roi_transform_key=(
                        str(export_transform_combo.currentData())
                        if transform_rois_check.isChecked() and export_transform_combo.currentData()
                        else None
                    ),
                    roi_transform_direction=str(export_direction_combo.currentData() or "reference_to_moving"),
                )
        except Exception:
            self.show_error("Could not export MRax data", traceback.format_exc())
            return
        self.statusBar().showMessage(f"Exported {len(saved)} file(s).")

    def _exportable_object_records(self) -> list[dict[str, Any]]:
        return _export_object_records_for_session(self.session)

    def _populate_export_object_tree(self, tree, records: Sequence[dict[str, Any]]) -> None:
        tree.clear()
        groups: dict[str, Any] = {}
        for idx, record in enumerate(records):
            group_name = str(record.get("group") or "Objects")
            group = groups.get(group_name)
            if group is None:
                group = QtWidgets.QTreeWidgetItem([group_name, "", ""])
                group.setExpanded(True)
                tree.addTopLevelItem(group)
                groups[group_name] = group
            item = QtWidgets.QTreeWidgetItem(
                [
                    str(record.get("label") or record.get("name") or record.get("id")),
                    str(record.get("kind") or ""),
                    _shape_text(record.get("array")),
                ]
            )
            item.setData(0, QtCore.Qt.UserRole, idx)
            group.addChild(item)
        tree.resizeColumnToContents(0)

    def _export_object_items(self, tree) -> list[Any]:
        items = []
        for top_idx in range(tree.topLevelItemCount()):
            top = tree.topLevelItem(top_idx)
            for child_idx in range(top.childCount()):
                items.append(top.child(child_idx))
        return items

    def _select_first_export_object(self, tree) -> None:
        items = self._export_object_items(tree)
        if items:
            tree.setCurrentItem(items[0])

    def _select_adjacent_export_object(self, tree, offset: int) -> None:
        items = self._export_object_items(tree)
        if not items:
            return
        current = tree.currentItem()
        try:
            idx = items.index(current)
        except ValueError:
            idx = 0
        idx = max(0, min(len(items) - 1, idx + int(offset)))
        tree.setCurrentItem(items[idx])

    def export_single_object(self, record: Mapping[str, Any], path: Path, file_type: str) -> list[Path]:
        return _write_single_export_object(record, path, file_type)

    def export_current_study(
        self,
        output_dir: Path,
        *,
        include_images: bool,
        include_kspace: bool,
        include_maps: bool,
        include_rois: bool,
        export_numpy: bool,
        export_nifti: bool,
        roi_transform_key: str | None = None,
        roi_transform_direction: str = "reference_to_moving",
    ) -> list[Path]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        saved: list[Path] = []
        manifest: dict[str, Any] = {
            "format": "mrax.gui.export",
            "version": 1,
            "source_manifest": None if self.session.manifest_path is None else str(self.session.manifest_path),
            "experiments": {},
            "warnings": [],
        }
        experiments = (self.session.data or {}).get("experiments") or {}
        for exp_name, exp_data in experiments.items():
            exp_key = _safe_filename(str(exp_name))
            exp_dir = output_dir / exp_key
            exp_dir.mkdir(parents=True, exist_ok=True)
            exp_manifest: dict[str, Any] = {
                "name": str(exp_name),
                "dims": dict(exp_data.get("dims") or {}),
                "files": [],
            }
            if include_images and exp_data.get("mr_image") is not None:
                try:
                    arr = np.asarray(exp_data["mr_image"])
                    if export_numpy:
                        path = exp_dir / "mr_image.npy"
                        np.save(path, arr)
                        saved.append(path)
                        exp_manifest["files"].append({"kind": "image", "format": "npy", "path": _package_relative(path, output_dir)})
                except Exception as exc:
                    manifest["warnings"].append(f"warning: skipped {exp_name}/mr_image NumPy export: {exc}")
                    arr = None
                if export_nifti:
                    saved.extend(self._export_image_nifti(exp_name, exp_data, exp_dir / "nifti" / "mr_image", manifest))
            if include_kspace and exp_data.get("raw_kspace") is not None and export_numpy:
                try:
                    path = exp_dir / "raw_kspace.npy"
                    np.save(path, np.asarray(exp_data["raw_kspace"]))
                    saved.append(path)
                    exp_manifest["files"].append({"kind": "k_space", "format": "npy", "path": _package_relative(path, output_dir)})
                except Exception as exc:
                    manifest["warnings"].append(f"warning: skipped {exp_name}/raw_kspace NumPy export: {exc}")
            if include_maps:
                for map_name, value in (exp_data.get("fitted_maps") or {}).items():
                    try:
                        arr, unit = _quantity_array_and_unit(value)
                    except Exception as exc:
                        manifest["warnings"].append(f"warning: skipped {exp_name}/{map_name} map export: {exc}")
                        continue
                    if export_numpy:
                        try:
                            map_dir = exp_dir / "maps"
                            map_dir.mkdir(parents=True, exist_ok=True)
                            path = map_dir / f"{_safe_filename(str(map_name))}.npy"
                            np.save(path, arr)
                            saved.append(path)
                            exp_manifest["files"].append(
                                {
                                    "kind": "map",
                                    "name": str(map_name),
                                    "format": "npy",
                                    "unit": unit or "",
                                    "path": _package_relative(path, output_dir),
                                }
                            )
                        except Exception as exc:
                            manifest["warnings"].append(f"warning: skipped {exp_name}/{map_name} NumPy export: {exc}")
                    if export_nifti:
                        nifti_path = exp_dir / "nifti" / "maps" / f"{_safe_filename(str(map_name))}.nii"
                        try:
                            if arr.ndim not in (2, 3):
                                raise ValueError("NIfTI map export supports 2D/3D spatial maps only")
                            from mrax.io.nifti import save_nifti

                            nifti_path.parent.mkdir(parents=True, exist_ok=True)
                            save_nifti(nifti_path, arr, np.eye(4))
                        except Exception as exc:
                            manifest["warnings"].append(f"{exp_name}/{map_name}: {exc}")
                        else:
                            saved.append(nifti_path)
                            exp_manifest["files"].append(
                                {
                                    "kind": "map",
                                    "name": str(map_name),
                                    "format": "nifti",
                                    "unit": unit or "",
                                    "path": _package_relative(nifti_path, output_dir),
                                }
                            )
            manifest["experiments"][str(exp_name)] = exp_manifest
        if include_rois and self.session.rois:
            export_rois = self._rois_for_export_transform(
                roi_transform_key,
                roi_transform_direction,
                export_manifest=manifest,
            )
            if export_rois:
                try:
                    roi_npz, roi_json = save_roi_set(
                        export_rois,
                        output_dir / "rois" / "mrax_rois.npz",
                        metadata={
                            "manifest": str(self.session.manifest_path or ""),
                            "experiment": self.session.selected_experiment,
                            "registration_transform": roi_transform_key or "",
                            "registration_direction": roi_transform_direction if roi_transform_key else "",
                        },
                    )
                except Exception as exc:
                    manifest["warnings"].append(f"warning: skipped ROI set export: {exc}")
                else:
                    saved.extend([roi_npz, roi_json])
                    manifest["rois"] = {
                        "npz": _package_relative(roi_npz, output_dir),
                        "json": _package_relative(roi_json, output_dir),
                    }
                    roi_label_files = self._export_roi_label_masks(
                        output_dir / "rois",
                        output_dir,
                        rois=export_rois,
                        export_numpy=export_numpy,
                        export_nifti=export_nifti,
                        export_manifest=manifest,
                    )
                    saved.extend(roi_label_files)
                    manifest["rois"]["label_masks"] = [
                        _package_relative(path, output_dir) for path in roi_label_files
                    ]
        manifest_path = output_dir / "export_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        saved.append(manifest_path)
        if manifest["warnings"]:
            self.record_diagnostic_event(
                "warning",
                f"Export completed with {len(manifest['warnings'])} skipped item(s).",
                detail="\n".join(str(item) for item in manifest["warnings"]),
                context={"output": str(output_dir), "operation": "export"},
                update_status=False,
            )
        return saved

    def _export_image_nifti(
        self,
        exp_name: str,
        exp_data: dict[str, Any],
        output_dir: Path,
        export_manifest: dict[str, Any],
    ) -> list[Path]:
        try:
            from mrax.io.nifti import export_nifti_series

            experiment = Experiment(
                dims={str(k): int(v) for k, v in dict(exp_data.get("dims") or {}).items()},
                observables=dict(exp_data.get("observables") or {}),
                exp_type=exp_data.get("exp_type"),
            )
            dataset = DataSet(data=np.asarray(exp_data["mr_image"]), name=str(exp_name))
            return list(export_nifti_series(experiment, dataset, output_dir, stem="mr_image"))
        except Exception as exc:
            export_manifest["warnings"].append(f"{exp_name}/mr_image: {exc}")
            return []

    def _export_roi_label_masks(
        self,
        roi_dir: Path,
        root_dir: Path,
        *,
        rois: Sequence[ROI] | None = None,
        export_numpy: bool,
        export_nifti: bool,
        export_manifest: dict[str, Any],
    ) -> list[Path]:
        saved: list[Path] = []
        groups = _roi_label_mask_groups(list(rois) if rois is not None else self.session.rois)
        if not groups:
            return saved
        roi_dir.mkdir(parents=True, exist_ok=True)
        multiple = len(groups) > 1
        for group in groups:
            suffix = f"_{group['shape'][0]}x{group['shape'][1]}" if multiple else ""
            label_mask = group["label_mask"]
            binary_stack = group["binary_stack"]
            metadata = {
                "format": "mrax.gui.roi_label_masks",
                "version": 1,
                "shape": list(group["shape"]),
                "labels": group["labels"],
                "note": "label 0 is background; nonzero labels follow the listed ROI order",
            }
            if export_numpy:
                label_path = roi_dir / f"roi_labels{suffix}.npy"
                stack_path = roi_dir / f"roi_binary_stack{suffix}.npy"
                meta_path = roi_dir / f"roi_labels{suffix}.json"
                np.save(label_path, label_mask)
                np.save(stack_path, binary_stack)
                meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
                saved.extend([label_path, stack_path, meta_path])
            if export_nifti:
                try:
                    from mrax.io.nifti import save_nifti

                    references = [
                        _roi_nifti_reference(self.session, row.get("metadata"))
                        for row in group["labels"]
                    ]
                    label_reference = references[0]
                    distinct_references = {
                        (str(row.get("experiment") or ""), bool(row.get("identity_fallback")))
                        for row in references
                    }
                    if len(distinct_references) > 1:
                        export_manifest["warnings"].append(
                            "ROI label mask combines masks from multiple reference experiments; "
                            f"the combined label image uses {label_reference.get('experiment') or 'identity fallback'}."
                        )
                    nifti_dir = roi_dir / "nifti"
                    label_nifti = nifti_dir / f"roi_labels{suffix}.nii"
                    label_nifti.parent.mkdir(parents=True, exist_ok=True)
                    save_nifti(label_nifti, label_mask.astype(np.float32), label_reference["affine"])
                    saved.append(label_nifti)
                    binary_dir = nifti_dir / f"binary{suffix}"
                    for row, mask, reference in zip(group["labels"], binary_stack, references):
                        binary_path = binary_dir / f"{int(row['label']):03d}_{_safe_filename(row['name'])}.nii"
                        binary_path.parent.mkdir(parents=True, exist_ok=True)
                        save_nifti(binary_path, mask.astype(np.float32), reference["affine"])
                        saved.append(binary_path)
                except Exception as exc:
                    export_manifest["warnings"].append(f"ROI label-mask NIfTI export failed: {exc}")
        return saved

    def _rois_for_export_transform(
        self,
        transform_key: str | None,
        direction: str,
        *,
        export_manifest: dict[str, Any] | None = None,
    ) -> list[ROI]:
        if not transform_key:
            return list(self.session.rois)
        transformed: list[ROI] = []
        for roi in self.session.rois:
            try:
                _source, out, _target, _result = self._transformed_registration_roi(
                    transform_key=transform_key,
                    direction=direction,
                    roi_name=roi.name,
                )
            except Exception as exc:
                if export_manifest is not None:
                    export_manifest.setdefault("warnings", []).append(
                        f"warning: skipped ROI '{roi.name}' transform for export: {exc}"
                    )
                continue
            transformed.append(out)
        return transformed

    def save_dialog(self):
        if not self.session.is_loaded:
            self.statusBar().showMessage("Load a study before saving.")
            return False
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Save MRax")
        layout = QtWidgets.QVBoxLayout(dialog)
        bundle_check = QtWidgets.QCheckBox("Save as single .mrax bundle")
        bundle_check.setChecked(True)
        output_row = QtWidgets.QHBoxLayout()
        output_edit = QtWidgets.QLineEdit(str(Path.cwd() / f"{self._default_study_name()}.mrax"))
        browse = QtWidgets.QPushButton("Browse")
        def browse_output():
            if bundle_check.isChecked():
                path, _ = self._get_save_file_name(
                    "Save MRax bundle",
                    output_edit.text() or str(Path.cwd()),
                    "MRax bundle (*.mrax)",
                )
                if path:
                    output_edit.setText(path)
            else:
                path = self._get_existing_directory(
                    "Choose editable MRax package folder",
                    output_edit.text() if output_edit.text() and not output_edit.text().endswith(".mrax") else str(Path.cwd()),
                )
                if path:
                    output_edit.setText(path)

        def on_bundle_toggled(checked: bool):
            current = Path(output_edit.text() or Path.cwd())
            if checked:
                if current.suffix.lower() != ".mrax":
                    output_edit.setText(str(current.with_suffix(".mrax")))
            else:
                if current.suffix.lower() == ".mrax":
                    output_edit.setText(str(current.with_suffix("")))

        browse.clicked.connect(browse_output)
        bundle_check.toggled.connect(on_bundle_toggled)
        output_row.addWidget(output_edit, 1)
        output_row.addWidget(browse)
        layout.addWidget(QtWidgets.QLabel("Output"))
        layout.addLayout(output_row)
        layout.addWidget(bundle_check)
        save_state = QtWidgets.QCheckBox("GUI state")
        save_state.setChecked(True)
        save_rois = QtWidgets.QCheckBox("ROI set")
        save_rois.setChecked(bool(self.session.rois))
        save_study = QtWidgets.QCheckBox("MRax study / fit results")
        save_study.setChecked(True)
        save_unloaded_raw = QtWidgets.QCheckBox("Load and save unloaded raw k-space/reconstructions")
        save_unloaded_raw.setChecked(False)
        save_unloaded_raw.setEnabled(self.session.source_kind == "bruker")
        layout.addWidget(save_state)
        layout.addWidget(save_rois)
        layout.addWidget(save_study)
        layout.addWidget(save_unloaded_raw)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if self._exec_dialog(dialog) != QtWidgets.QDialog.Accepted:
            return False
        try:
            if bundle_check.isChecked():
                bundle_path = Path(output_edit.text()).expanduser()
                if bundle_path.suffix.lower() != ".mrax":
                    bundle_path = bundle_path.with_suffix(".mrax")
                saved = self.save_mrax_bundle_file(
                    bundle_path,
                    include_state=save_state.isChecked(),
                    include_rois=save_rois.isChecked(),
                    include_study=save_study.isChecked(),
                    include_unloaded_raw=save_unloaded_raw.isChecked(),
                )
                self.statusBar().showMessage(f"Saved MRax bundle with {len(saved)} item(s): {bundle_path}")
            else:
                out = Path(output_edit.text()).expanduser()
                saved = self.save_mrax_package_folder(
                    out,
                    include_state=save_state.isChecked(),
                    include_rois=save_rois.isChecked(),
                    include_study=save_study.isChecked(),
                    include_unloaded_raw=save_unloaded_raw.isChecked(),
                )
                self.statusBar().showMessage(f"Saved editable MRax package with {len(saved)} item(s) to {out}")
        except Exception:
            self.show_error("Could not save selected outputs", traceback.format_exc())
            return False
        self.clear_dirty()
        return True

    def _default_study_name(self) -> str:
        if self.session.data:
            name = _safe_filename(self.session.study_name)
            if name:
                return name
        if self.session.manifest_path is not None:
            return _safe_filename(self.session.manifest_path.parent.name or "mrax_study")
        return "mrax_study"

    def save_mrax_bundle_file(
        self,
        path: Path,
        *,
        include_state: bool,
        include_rois: bool,
        include_study: bool,
        include_unloaded_raw: bool = False,
    ) -> list[Path]:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="mrax_bundle_") as tmp:
            package_root = Path(tmp) / path.stem
            saved = self.save_mrax_package_folder(
                package_root,
                include_state=include_state,
                include_rois=include_rois,
                include_study=include_study,
                include_unloaded_raw=include_unloaded_raw,
            )
            _write_mrax_bundle(path, package_root)
        return [path, *saved]

    def save_mrax_package_folder(
        self,
        output_dir: Path,
        *,
        include_state: bool,
        include_rois: bool,
        include_study: bool,
        include_unloaded_raw: bool = False,
    ) -> list[Path]:
        output_dir = Path(output_dir)
        if output_dir.exists() and (output_dir / "mrax_package.json").is_file():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        saved: list[Path] = []
        entries: dict[str, str] = {}
        warnings_list: list[str] = []
        study_manifest = None
        fit_manifest = None
        if include_study or include_state:
            if include_unloaded_raw:
                try:
                    self.session.load_all_lazy_raw_data(progress_callback=self.on_task_progress)
                except Exception as exc:
                    warnings_list.append(f"warning: skipped unloaded raw data import: {exc}")
            try:
                study_manifest = self._save_gui_state_data_snapshot(
                    output_dir / "study",
                    include_unloaded_raw=False,
                )
            except Exception as exc:
                warnings_list.append(f"warning: skipped study snapshot: {exc}")
            if study_manifest is not None:
                entries["study_manifest"] = _package_relative(study_manifest, output_dir)
                saved.append(study_manifest)
            if include_study and self.last_fit_result is not None:
                try:
                    fit_manifest = self.session.save_fit_result(self.last_fit_result, output_dir / "fit_results")
                except Exception as exc:
                    warnings_list.append(f"warning: skipped fit-result snapshot: {exc}")
                else:
                    entries["fit_manifest"] = _package_relative(fit_manifest, output_dir)
                    saved.append(fit_manifest)
        if include_state:
            try:
                state_path = self.save_gui_state_file(
                    output_dir / "mrax_gui_state.json",
                    study_manifest=study_manifest,
                    fit_manifest=fit_manifest,
                    include_data_snapshot=False,
                )
            except Exception as exc:
                warnings_list.append(f"warning: skipped GUI state: {exc}")
            else:
                entries["gui_state"] = _package_relative(state_path, output_dir)
                saved.append(state_path)
        if include_rois and self.session.rois:
            try:
                roi_npz, roi_json = save_roi_set(
                    self.session.rois,
                    output_dir / "mrax_rois.npz",
                    metadata={
                        "manifest": str(study_manifest or self.session.manifest_path or ""),
                        "experiment": self.session.selected_experiment,
                    },
                )
            except Exception as exc:
                warnings_list.append(f"warning: skipped ROI set: {exc}")
            else:
                entries["roi_npz"] = _package_relative(roi_npz, output_dir)
                entries["roi_json"] = _package_relative(roi_json, output_dir)
                saved.extend([roi_npz, roi_json])
        package_manifest = _write_mrax_package_manifest(output_dir, entries, warnings=warnings_list)
        saved.append(package_manifest)
        if warnings_list:
            self.record_diagnostic_event(
                "warning",
                f"MRax package save completed with {len(warnings_list)} skipped item(s).",
                detail="\n".join(warnings_list),
                context={"output": str(output_dir), "operation": "package save"},
                update_status=False,
            )
        return saved

    def save_gui_state_file(
        self,
        path: Path,
        *,
        study_manifest: Path | None = None,
        fit_manifest: Path | None = None,
        include_data_snapshot: bool = True,
    ) -> Path:
        self.session._save_current_view_state()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if study_manifest is None and include_data_snapshot:
            study_manifest = self._save_gui_state_data_snapshot(path.with_name(f"{path.stem}_study"))
        payload, arrays = _serialize_gui_state(
            self.session,
            preview_results=self._preview_results_by_experiment,
            fit_results=self._fit_results_by_experiment,
            drift_results=self.drift_results,
            data_manifest=study_manifest,
            fit_manifest=fit_manifest,
            state_path=path,
        )
        masks_path = path.with_name(f"{path.stem}_arrays.npz")
        np.savez_compressed(masks_path, **arrays)
        payload["array_file"] = masks_path.name
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def _save_gui_state_data_snapshot(
        self,
        output_dir: Path,
        *,
        include_unloaded_raw: bool = False,
    ) -> Path | None:
        output_dir = Path(output_dir)
        if include_unloaded_raw:
            self.session.load_all_lazy_raw_data(progress_callback=self.on_task_progress)
        if self.session.manifest_path is not None and self.session.manifest_path.is_file():
            if output_dir.exists():
                shutil.rmtree(output_dir)
            shutil.copytree(self.session.manifest_path.parent, output_dir)
            copied = output_dir / self.session.manifest_path.name
            return copied if copied.is_file() else None
        if self.session.study is not None:
            return self.session.save_imported_study(output_dir)
        return None

    def load_gui_state_dialog(self):
        path, _ = self._get_open_file_name(
            "Load MRax GUI state",
            str(Path.cwd()),
            "MRax GUI state (*.json);;All files (*)",
        )
        if not path:
            return
        try:
            warnings_list = self.load_gui_state_file(Path(path))
        except Exception:
            self.show_error("Could not load GUI state", traceback.format_exc())
            return
        self.populate_from_session()
        suffix = f" with {len(warnings_list)} warning(s)" if warnings_list else ""
        self.statusBar().showMessage(f"Loaded GUI state from {path}{suffix}")

    def load_gui_state_file(self, path: Path) -> list[str]:
        return self._load_gui_state_data(path)

    def _load_gui_state_data(self, path: Path) -> list[str]:
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        warnings_list: list[str] = []
        manifest = _resolve_state_manifest_path(payload, path)
        if manifest is not None and manifest.is_file():
            session = GuiSession()
            session.load_manifest(manifest)
            self._replace_or_add_loaded_workspace(
                session,
                self._workspace_label_for_session(session, Path(manifest).parent.name or Path(manifest).stem),
            )
        elif not self.session.is_loaded:
            if manifest is None or not manifest.is_file():
                raise FileNotFoundError(
                    "GUI state does not reference a readable MRax study manifest."
                )
        arrays_path = path.with_name(payload.get("array_file", f"{path.stem}_arrays.npz"))
        if not arrays_path.is_file():
            raise FileNotFoundError(f"GUI state array file is missing: {arrays_path}")
        try:
            arrays = dict(np.load(arrays_path, allow_pickle=False))
        except Exception as exc:
            raise RuntimeError(f"Could not load GUI state array file '{arrays_path}': {exc}") from exc
        _restore_gui_state(self.session, payload, arrays, warnings=warnings_list)
        self.drift_results = _restore_drift_results(
            payload.get("drift_results") or {},
            arrays,
            warnings=warnings_list,
        )
        try:
            self._preview_results_by_experiment = dict(payload.get("preview_results") or {})
        except Exception as exc:
            self._preview_results_by_experiment = {}
            warnings_list.append(f"warning: skipped preview results from GUI state: {exc}")
        self._fit_results_by_experiment = {}
        self._visual_undo_stacks.clear()
        self.last_fit_result = None
        self.last_preview_fit_result = self._preview_results_by_experiment.get(self.session.selected_experiment or "")
        self._capture_active_workspace()
        if warnings_list:
            self.record_diagnostic_event(
                "warning",
                f"GUI state loaded with {len(warnings_list)} warning(s).",
                detail="\n".join(warnings_list),
                context={"source": str(path), "operation": "GUI state load"},
                update_status=False,
            )
        return warnings_list

    def load_mrax_bundle_file(self, path: Path):
        self._load_mrax_bundle_data(path)
        self.populate_from_session()

    def _load_mrax_bundle_data(self, path: Path):
        path = Path(path)
        extract_dir = Path(tempfile.mkdtemp(prefix=f"{_safe_filename(path.stem)}_mrax_"))
        _extract_mrax_bundle(path, extract_dir)
        self._bundle_extract_dirs.append(extract_dir)
        self._load_mrax_package_data(extract_dir)

    def load_mrax_package_folder(self, root: Path):
        self._load_mrax_package_data(root)
        self.populate_from_session()

    def _load_mrax_package_data(self, root: Path):
        root = Path(root)
        entries = _mrax_package_entries(root)
        state_path = entries.get("gui_state")
        manifest_path = entries.get("study_manifest") or _find_study_manifest(root)
        roi_path = entries.get("roi_npz") or entries.get("roi_json") or _find_roi_file(root)
        loaded_state = False
        load_warnings: list[str] = []
        if state_path is not None and state_path.is_file():
            try:
                self._load_gui_state_data(state_path)
            except Exception as exc:
                load_warnings.append(f"warning: skipped GUI state load: {exc}")
                loaded = False
            else:
                loaded_state = True
                loaded = True
        else:
            loaded = False
        if not loaded and manifest_path is not None and manifest_path.is_file():
            try:
                self._load_manifest_data(manifest_path)
            except Exception as exc:
                load_warnings.append(f"warning: skipped study manifest load: {exc}")
            else:
                loaded = True
        if roi_path is not None and roi_path.is_file() and (not loaded_state or not self.session.rois):
            try:
                self._load_roi_data(roi_path)
            except Exception as exc:
                load_warnings.append(f"warning: skipped ROI load: {exc}")
            else:
                loaded = True
        if not loaded:
            detail = "; ".join(load_warnings)
            suffix = f" ({detail})" if detail else ""
            raise FileNotFoundError(f"No loadable MRax manifest, GUI state, or ROI set found in {root}{suffix}")
        if load_warnings:
            self.record_diagnostic_event(
                "warning",
                f"MRax package loaded with {len(load_warnings)} skipped item(s).",
                detail="\n".join(load_warnings),
                context={"source": str(root), "operation": "package load"},
                update_status=False,
            )
            self.statusBar().showMessage("; ".join(load_warnings))

    def clear_rois(self):
        self.session.clear_rois()
        self.refresh_roi_list()
        self.update_view(preserve_image_view=True)
        self.mark_dirty("ROI set changed")
        self.statusBar().showMessage("Disabled all shared ROIs for this experiment.")

    def remove_selected_roi(self):
        item = self.roi_list.currentItem()
        if item is None:
            self.statusBar().showMessage("No ROI selected.")
            return
        name = item.text()
        self.session.remove_shared_roi(name)
        self.session.fit_settings.selected_roi = self.session.rois[0].name if self.session.rois else None
        self.refresh_roi_list()
        self.update_view(preserve_image_view=True)
        self.mark_dirty("ROI set changed")
        self.statusBar().showMessage(f"Removed ROI '{name}' from the shared ROI set.")

    def add_rect_roi(self):
        self._add_editable_roi("rectangle")

    def add_circle_roi(self):
        self._add_editable_roi("ellipse")

    def add_flexible_shape_roi(self):
        if not self.session.is_loaded:
            return
        current = np.asarray(self.session.frame_image())
        h, w = current.shape[-2], current.shape[-1]
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Add ROI Shape")
        layout = QtWidgets.QVBoxLayout(dialog)
        form = QtWidgets.QFormLayout()
        shape_combo = QtWidgets.QComboBox()
        for label, key in (
            ("Rectangle", "rectangle"),
            ("Square", "square"),
            ("Ellipse", "ellipse"),
            ("Circle", "circle"),
            ("Triangle", "triangle"),
            ("Pentagon", "pentagon"),
            ("Hexagon", "hexagon"),
            ("Regular polygon", "regular_polygon"),
        ):
            shape_combo.addItem(label, key)
        width_spin = QtWidgets.QSpinBox()
        width_spin.setRange(1, max(1, int(w)))
        width_spin.setValue(max(1, min(int(w), max(4, int(w) // 3))))
        height_spin = QtWidgets.QSpinBox()
        height_spin.setRange(1, max(1, int(h)))
        height_spin.setValue(max(1, min(int(h), max(4, int(h) // 3))))
        sides_spin = QtWidgets.QSpinBox()
        sides_spin.setRange(3, 32)
        sides_spin.setValue(6)
        sides_spin.setEnabled(False)
        angle_spin = QtWidgets.QDoubleSpinBox()
        angle_spin.setRange(-180.0, 180.0)
        angle_spin.setDecimals(1)
        angle_spin.setSingleStep(5.0)
        angle_spin.setSuffix(" deg")
        shape_combo.currentIndexChanged.connect(
            lambda _idx: sides_spin.setEnabled(str(shape_combo.currentData()) == "regular_polygon")
        )
        form.addRow("Shape", shape_combo)
        form.addRow("Width", width_spin)
        form.addRow("Height", height_spin)
        form.addRow("Sides", sides_spin)
        form.addRow("Rotation", angle_spin)
        layout.addLayout(form)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if self._exec_dialog(dialog) != QtWidgets.QDialog.Accepted:
            return
        shape = str(shape_combo.currentData() or "rectangle")
        self._add_editable_roi(
            shape,
            width=int(width_spin.value()),
            height=int(height_spin.value()),
            sides=int(sides_spin.value()) if shape == "regular_polygon" else None,
            angle=float(angle_spin.value()),
        )

    def add_polygon_roi(self):
        if not self.session.is_loaded:
            return
        self.brush_mode = "polygon"
        self._polygon_points = []
        self._last_brush_stamp = None
        self._clear_polygon_preview()
        self._set_tool_settings_expanded(True)
        self._refresh_drawing_controls()
        self.statusBar().showMessage(
            "Polygon drawing active."
        )

    def add_lasso_roi(self):
        if not self.session.is_loaded:
            return
        self.brush_mode = "lasso"
        self._polygon_points = []
        self._last_brush_stamp = None
        self._clear_brush_preview()
        self._clear_polygon_preview()
        self._set_tool_settings_expanded(True)
        self._refresh_drawing_controls()
        self.statusBar().showMessage(
            "Lasso active: hold the left mouse button and trace the boundary; Done finishes the smoothed contour."
        )

    def enable_brush(self):
        self.brush_mode = "brush"
        self._begin_brush_edit_session()
        self._clear_polygon_preview()
        self._set_tool_settings_expanded(True)
        self._refresh_drawing_controls()
        self.statusBar().showMessage("Brush mode: click or drag on the image to add mask pixels.")

    def enable_erase(self):
        self.brush_mode = "erase"
        self._begin_brush_edit_session()
        self._clear_polygon_preview()
        self._set_tool_settings_expanded(True)
        self._refresh_drawing_controls()
        self.statusBar().showMessage("Erase mode: click or drag on the image to remove mask pixels.")

    def enable_wand(self):
        self.brush_mode = "wand"
        self._last_brush_stamp = None
        self._clear_brush_preview()
        self._clear_polygon_preview()
        self._set_tool_settings_expanded(True)
        self._refresh_drawing_controls()
        self.statusBar().showMessage("Wand mode: click a seed pixel to add a connected intensity region.")

    def _begin_brush_edit_session(self):
        self._last_brush_stamp = None
        self._brush_edit_checkpoint = self._capture_roi_edit_state()
        self._brush_undo_stack = []
        self._brush_stroke_checkpoint = None
        self._brush_stroke_active = False

    def _clear_brush_edit_session(self):
        self._last_brush_stamp = None
        self._brush_edit_checkpoint = None
        self._brush_undo_stack = []
        self._brush_stroke_checkpoint = None
        self._brush_stroke_active = False

    def _capture_roi_edit_state(self) -> dict[str, Any]:
        view_states = {}
        for name, state in self.session.view_states.items():
            view_states[name] = {
                "rois": _copy_roi_list(state.rois),
                "selected_roi": state.fit_settings.selected_roi,
            }
        return {
            "rois": _copy_roi_list(self.session.rois),
            "shared_rois": _copy_roi_list(self.session.shared_rois),
            "selected_roi": self.session.fit_settings.selected_roi,
            "view_states": view_states,
        }

    def _restore_roi_edit_state(self, snapshot: Mapping[str, Any] | None):
        if not snapshot:
            return
        self.session.rois = _copy_roi_list(snapshot.get("rois") or [])
        self.session.shared_rois = _copy_roi_list(snapshot.get("shared_rois") or [])
        self.session.fit_settings.selected_roi = snapshot.get("selected_roi")
        saved_states = dict(snapshot.get("view_states") or {})
        for name, state_snapshot in saved_states.items():
            state = self.session.view_states.get(name)
            if state is None:
                continue
            state.rois = _copy_roi_list(state_snapshot.get("rois") or [])
            state.fit_settings.selected_roi = state_snapshot.get("selected_roi")

    def _begin_brush_stroke(self):
        if self.brush_mode not in {"brush", "erase"}:
            return
        if self._brush_edit_checkpoint is None:
            self._brush_edit_checkpoint = self._capture_roi_edit_state()
        if self._brush_stroke_active:
            return
        self._brush_stroke_checkpoint = self._capture_roi_edit_state()
        self._brush_stroke_active = True

    def _finish_brush_stroke(self):
        if not self._brush_stroke_active:
            return
        if self._brush_stroke_checkpoint is not None:
            self._brush_undo_stack.append(self._brush_stroke_checkpoint)
            if len(self._brush_undo_stack) > 50:
                self._brush_undo_stack = self._brush_undo_stack[-50:]
        self._brush_stroke_checkpoint = None
        self._brush_stroke_active = False
        self._last_brush_stamp = None
        self._refresh_drawing_controls()

    def _undo_brush_stroke(self):
        if self.brush_mode not in {"brush", "erase"}:
            return
        self._finish_brush_stroke()
        if not self._brush_undo_stack:
            self.statusBar().showMessage("No brush stroke to undo.")
            self._refresh_drawing_controls()
            return
        snapshot = self._brush_undo_stack.pop()
        self._restore_roi_edit_state(snapshot)
        self.refresh_roi_list()
        self.update_view(preserve_image_view=True)
        self.mark_dirty("ROI set changed")
        self._refresh_drawing_controls()
        self.statusBar().showMessage("Undid last brush stroke.")

    def _cancel_brush_edit_session(self) -> bool:
        if self.brush_mode not in {"brush", "erase"}:
            return False
        checkpoint = self._brush_edit_checkpoint
        self._clear_brush_edit_session()
        if checkpoint is None:
            return False
        self._restore_roi_edit_state(checkpoint)
        self.refresh_roi_list()
        self.update_view(preserve_image_view=True)
        self.mark_dirty("ROI set changed")
        return True

    def _add_editable_roi(
        self,
        shape: str,
        *,
        width: int | None = None,
        height: int | None = None,
        sides: int | None = None,
        angle: float = 0.0,
    ):
        if not self.session.is_loaded:
            return
        current = np.asarray(self.session.frame_image())
        h, w = current.shape[-2], current.shape[-1]
        spatial_dims = tuple(self.session.spatial_dim_names)
        spatial_shape = tuple(self.session.spatial_shape)
        color = _roi_color(len(self.session.rois))
        bw = max(1, min(int(width) if width is not None else max(4, w // 3), w))
        bh = max(1, min(int(height) if height is not None else max(4, h // 3), h))
        shape_key = str(shape)
        if shape_key in {"rectangle", "square"}:
            if shape_key == "square":
                bw = bh = max(1, min(bw, bh))
            bounds = [max(0, (w - bw) // 2), max(0, (h - bh) // 2), bw, bh]
            mask_2d = _rect_mask((h, w), bounds, angle=angle)
            metadata = {"shape": "rectangle", "bounds": bounds, "angle": float(angle), "shape_variant": shape_key}
        elif shape_key in {"ellipse", "circle"}:
            if shape_key == "circle":
                bw = bh = max(1, min(bw, bh))
            bounds = [max(0, (w - bw) // 2), max(0, (h - bh) // 2), bw, bh]
            mask_2d = _ellipse_mask((h, w), bounds, angle=angle)
            metadata = {"shape": "ellipse", "bounds": bounds, "angle": float(angle), "shape_variant": shape_key}
        elif shape_key in {"triangle", "pentagon", "hexagon", "regular_polygon"}:
            side_map = {"triangle": 3, "pentagon": 5, "hexagon": 6}
            n_sides = int(sides or side_map.get(shape_key, 6))
            radius = max(1.0, min(float(bw), float(bh)) / 2.0)
            center = (float(w) / 2.0, float(h) / 2.0)
            points = _regular_polygon_points(center, radius, n_sides, angle=angle)
            mask_2d = _polygon_mask((h, w), points)
            metadata = {
                "shape": "polygon",
                "points": points.tolist(),
                "angle": 0.0,
                "shape_variant": shape_key,
                "sides": n_sides,
            }
        else:
            points = np.asarray(
                [
                    [w * 0.5, h * 0.2],
                    [w * 0.8, h * 0.75],
                    [w * 0.2, h * 0.75],
                ],
                dtype=float,
            )
            mask_2d = _polygon_mask((h, w), points)
            metadata = {"shape": "polygon", "points": points.tolist()}
        mask = embed_mask_slice(
            mask_2d,
            spatial_shape=spatial_shape,
            spatial_dims=spatial_dims,
            indices=self._current_slice_indices(),
        )
        name = _unique_roi_name(self.session.rois, shape_key)
        metadata.update(
            {
                "color": color,
                "source_experiment": self.session.selected_experiment,
                "slice_indices": self._current_slice_indices(),
                "generated_by": f"mrax.gui.{shape_key}",
            }
        )
        roi = ROI(
            name,
            mask,
            dims=spatial_dims,
            metadata=metadata,
        )
        self.session.add_roi(roi)
        self.session.fit_settings.selected_roi = name
        self.refresh_roi_list()
        self.update_view(preserve_image_view=True)
        self.mark_dirty("ROI set changed")
        self.statusBar().showMessage(
            f"Added editable {shape_key} ROI. Drag handles to change position and size."
        )

    def on_image_clicked(self, event):
        if self.brush_mode is None or not self.session.is_loaded:
            return
        coords = self._image_coords_from_scene_pos(event.scenePos(), rounding="round")
        if coords is None:
            return
        x_float, y_float, x, y, h, w = coords
        if self.brush_mode == "lasso":
            right_click = event.button() == QtCore.Qt.RightButton
            double_click = bool(getattr(event, "double", lambda: False)())
            if right_click or double_click:
                self._finish_polygon_roi()
            return
        if self.brush_mode == "polygon":
            right_click = event.button() == QtCore.Qt.RightButton
            double_click = bool(getattr(event, "double", lambda: False)())
            if right_click:
                self._finish_polygon_roi()
                return
            if event.button() != QtCore.Qt.LeftButton:
                return
            if double_click:
                if len(self._polygon_points) < 3:
                    self._append_polygon_point(x_float, y_float, min_distance=0.0)
                    self._update_polygon_preview()
                    self._refresh_drawing_controls()
                self._finish_polygon_roi()
                return
            if self._polygon_click_closes(x_float, y_float):
                self._finish_polygon_roi()
                return
            self._append_polygon_point(x_float, y_float, min_distance=0.0)
            self._update_polygon_preview()
            self._refresh_drawing_controls()
            return
        if event.button() != QtCore.Qt.LeftButton:
            return
        if self.brush_mode == "wand":
            self._apply_wand_at(x, y, (h, w))
            return
        self._apply_brush_at(x, y, (h, w))

    def _image_coords_from_scene_pos(self, scene_pos, *, rounding: str = "floor"):
        if not self.session.is_loaded:
            return None
        pos = self.image_view.getView().mapSceneToView(scene_pos)
        x_float, y_float = float(pos.x()), float(pos.y())
        if rounding == "round":
            x, y = int(round(x_float)), int(round(y_float))
        else:
            x, y = int(np.floor(x_float)), int(np.floor(y_float))
        displayed = np.asarray(self._displayed_image if self._displayed_image is not None else self.session.frame_image())
        if displayed.ndim < 2:
            return None
        h, w = _display_spatial_shape(displayed)
        if not (0 <= y < h and 0 <= x < w):
            return None
        return x_float, y_float, x, y, h, w

    def _active_roi_for_mask_edit(self, display_shape: tuple[int, int], *, prefix: str, generated_by: str) -> ROI:
        roi = self.session.roi()
        spatial_dims = tuple(self.session.spatial_dim_names)
        spatial_shape = tuple(self.session.spatial_shape)
        can_edit = False
        if roi is not None:
            can_edit = _roi_can_address_display_plane(
                roi,
                spatial_shape,
                spatial_dims,
                self._current_slice_indices(),
                display_shape,
            )
            can_edit = can_edit or _geometry_mask_from_metadata(roi.metadata, display_shape) is not None
        if roi is None or not can_edit:
            roi = ROI(
                _unique_roi_name(self.session.rois, prefix),
                np.zeros(spatial_shape, dtype=bool),
                dims=spatial_dims,
                metadata={
                    "color": _roi_color(len(self.session.rois)),
                    "slice_indices": self._current_slice_indices(),
                    "generated_by": generated_by,
                    "shape": "mask",
                },
            )
            self.session.add_roi(roi)
        return roi

    def _roi_edit_label_value(self) -> int:
        try:
            return max(1, int(self.brush_label_spin.value()))
        except Exception:
            return 1

    def _apply_brush_at(self, x: int, y: int, display_shape: tuple[int, int]):
        if self.brush_mode not in {"brush", "erase"}:
            return
        radius = _brush_radius_from_size(self.brush_size_spin.value())
        roi = self._active_roi_for_mask_edit(display_shape, prefix="brush", generated_by="mrax.gui.brush")
        stamp = (self.brush_mode, roi.name, int(x), int(y), float(radius), tuple(display_shape))
        if stamp == self._last_brush_stamp:
            return
        self._begin_brush_stroke()
        self._last_brush_stamp = stamp
        base_mask = self._roi_as_spatial_mask(roi, geometry_indices=self._current_slice_indices())
        mask = np.asarray(
            mask_slice_for_display(
                base_mask,
                dims=tuple(self.session.spatial_dim_names),
                indices=self._current_slice_indices(),
            )
        ).copy()
        brush = _brush_mask(display_shape, (y, x), radius)
        if self.brush_mode == "erase":
            mask[brush] = 0
        else:
            label_value = self._roi_edit_label_value()
            if label_value > 1 and mask.dtype == bool:
                mask = mask.astype(np.uint16)
            mask[brush] = label_value if mask.dtype != bool else True
        meta = _rasterized_roi_metadata(
            roi.metadata,
            "erase" if self.brush_mode == "erase" else "brush",
            brush_radius=radius,
        )
        if mask.dtype != bool:
            meta["mask_type"] = "label"
            meta["active_label"] = self._roi_edit_label_value()
        self._commit_display_mask_edit(roi, mask, metadata=meta)
        if self.brush_mode == "brush" and getattr(self, "brush_auto_subtract_check", None) is not None:
            if self.brush_auto_subtract_check.isChecked():
                self._subtract_stamp_from_other_rois(roi.name, brush, display_shape)

    def _apply_wand_at(self, x: int, y: int, display_shape: tuple[int, int]):
        image = self._roi_source_image()
        if image is None:
            return
        tolerance_fraction = float(self.wand_tolerance_spin.value()) / 100.0
        region, metadata = _seeded_region_mask(
            image,
            (y, x),
            tolerance_fraction=tolerance_fraction,
        )
        if not np.any(region):
            self.statusBar().showMessage("Wand did not find a connected region.")
            return
        roi = self._active_roi_for_mask_edit(display_shape, prefix="wand", generated_by="mrax.gui.wand")
        base_mask = self._roi_as_spatial_mask(roi, geometry_indices=self._current_slice_indices())
        mask = np.asarray(
            mask_slice_for_display(
                base_mask,
                dims=tuple(self.session.spatial_dim_names),
                indices=self._current_slice_indices(),
            )
        ).copy()
        label_value = self._roi_edit_label_value()
        if label_value > 1 and mask.dtype == bool:
            mask = mask.astype(np.uint16)
        if mask.dtype == bool:
            mask |= region
        else:
            mask[region] = label_value
        meta = _rasterized_roi_metadata(roi.metadata, "wand")
        meta.update(metadata)
        if mask.dtype != bool:
            meta["mask_type"] = "label"
            meta["active_label"] = label_value
        self._commit_display_mask_edit(roi, mask, metadata=meta)
        self.statusBar().showMessage(f"Added wand region to ROI '{roi.name}'.")

    def _commit_display_mask_edit(self, roi: ROI, mask_2d: np.ndarray, *, metadata: dict[str, Any]):
        base_mask = self._roi_as_spatial_mask(roi, geometry_indices=self._current_slice_indices())
        if np.asarray(mask_2d).dtype != bool and np.asarray(base_mask).dtype == bool:
            base_mask = np.asarray(base_mask, dtype=np.asarray(mask_2d).dtype)
        updated_mask = replace_mask_slice(
            base_mask,
            mask_2d,
            dims=tuple(self.session.spatial_dim_names),
            indices=self._current_slice_indices(),
        )
        updated_dims = tuple(self.session.spatial_dim_names[: updated_mask.ndim])
        self._replace_roi(ROI(roi.name, updated_mask, dims=updated_dims, metadata=metadata))
        self.refresh_roi_list()
        self.update_view(preserve_image_view=True)

    def _subtract_stamp_from_other_rois(self, target_name: str, stamp_mask: np.ndarray, display_shape: tuple[int, int]) -> None:
        spatial_dims = tuple(self.session.spatial_dim_names)
        changed = 0
        for other in list(self.session.rois):
            if other.name == target_name:
                continue
            try:
                base = self._roi_as_spatial_mask(other, geometry_indices=self._current_slice_indices())
                plane = np.asarray(
                    mask_slice_for_display(base, dims=spatial_dims, indices=self._current_slice_indices())
                ).copy()
            except Exception:
                continue
            if tuple(plane.shape) != tuple(display_shape):
                continue
            if not np.any(np.asarray(plane) != 0):
                continue
            plane[np.asarray(stamp_mask, dtype=bool)] = 0
            if np.array_equal(
                plane,
                np.asarray(mask_slice_for_display(base, dims=spatial_dims, indices=self._current_slice_indices())),
            ):
                continue
            meta = _rasterized_roi_metadata(other.metadata, "auto_subtract")
            meta["auto_subtract_source_roi"] = target_name
            if plane.dtype != bool:
                meta["mask_type"] = "label"
            updated = replace_mask_slice(base, plane, dims=spatial_dims, indices=self._current_slice_indices())
            self._replace_roi(ROI(other.name, updated, dims=spatial_dims, metadata=meta))
            changed += 1
        if changed:
            self.session.fit_settings.selected_roi = target_name
            self.refresh_roi_list()
            self.update_view(preserve_image_view=True)

    def _roi_source_image(self) -> np.ndarray | None:
        view = _view_key(self.view_combo.currentText())
        if view not in {"image", "map"}:
            self.statusBar().showMessage("ROI tools are available on image and map views.")
            return None
        try:
            image = np.asarray(
                self.session.map_image()
                if view == "map"
                else self.session.frame_image(original=bool(self.session.visual_state.show_original))
            )
        except Exception:
            return None
        return image if image.ndim >= 2 else None

    def on_image_hovered(self, scene_pos):
        if self.is_busy:
            return
        if not self.session.is_loaded or self._displayed_image is None:
            return
        if isinstance(scene_pos, (tuple, list)):
            scene_pos = scene_pos[0]
        view = self.image_view.getView()
        if not view.sceneBoundingRect().contains(scene_pos):
            return
        pos = view.mapSceneToView(scene_pos)
        x = int(np.floor(float(pos.x())))
        y = int(np.floor(float(pos.y())))
        arr = np.asarray(self._displayed_image)
        if arr.ndim < 2:
            return
        h, w = _display_spatial_shape(arr)
        if not (0 <= y < h and 0 <= x < w):
            return
        value_text = _format_display_pixel_value(arr[y, x])
        unit = f" {self._displayed_image_unit}" if self._displayed_image_unit else ""
        self.statusBar().showMessage(
            f"x={x}, y={y} | {self._displayed_image_label}: {value_text}{unit}"
        )
        if self.brush_mode in {"brush", "erase"} and self._left_mouse_pressed():
            self._apply_brush_at(x, y, (h, w))
        elif self.brush_mode in {"brush", "erase"}:
            self._finish_brush_stroke()
        elif self.brush_mode == "lasso" and self._left_mouse_pressed():
            min_distance = max(0.75, float(self.brush_size_spin.value()) * 0.18)
            if self._append_polygon_point(float(pos.x()), float(pos.y()), min_distance=min_distance):
                self._update_polygon_preview()
                self._refresh_drawing_controls()
        if self.brush_mode in {"brush", "erase"}:
            self._update_brush_preview(float(pos.x()), float(pos.y()), (h, w))
        else:
            self._clear_brush_preview()

    def _left_mouse_pressed(self) -> bool:
        button = _qt_mouse_button(QtCore, "LeftButton")
        if button is None:
            return False
        try:
            return bool(QtWidgets.QApplication.mouseButtons() & button)
        except Exception:
            return False

    def _update_brush_preview(self, x: float, y: float, display_shape: tuple[int, int]):
        radius = _brush_radius_from_size(self.brush_size_spin.value())
        preview_radius = max(0.5, float(radius))
        theta = np.linspace(0.0, 2.0 * np.pi, 96)
        xs = float(x) + preview_radius * np.cos(theta)
        ys = float(y) + preview_radius * np.sin(theta)
        h, w = display_shape
        visible = -preview_radius <= x <= w + preview_radius and -preview_radius <= y <= h + preview_radius
        if not visible:
            self._clear_brush_preview()
            return
        view_box = self.image_view.getView()
        color = "#ef4444" if self.brush_mode == "erase" else "#22d3ee"
        if self._brush_preview is None:
            self._brush_preview = pg.PlotDataItem(pen=pg.mkPen(color, width=2))
            self._brush_preview.setZValue(35)
            view_box.addItem(self._brush_preview)
        else:
            self._brush_preview.setPen(pg.mkPen(color, width=2))
        self._brush_preview.setData(xs, ys)

    def _clear_brush_preview(self):
        if self._brush_preview is None:
            return
        try:
            self.image_view.getView().removeItem(self._brush_preview)
        except Exception:
            _LOG.debug("Ignored optional GUI operation failure", exc_info=True)
        self._brush_preview = None

    def _refresh_drawing_controls(self):
        active = self.brush_mode is not None
        drawing = self.brush_mode in {"polygon", "lasso"}
        point_count = len(self._polygon_points) if drawing else 0
        if self._drawing_status_label is not None:
            mode_labels = {
                "lasso": "Lasso",
                "polygon": "Polygon",
                "brush": "Brush",
                "erase": "Erase",
                "wand": "Wand",
            }
            mode = mode_labels.get(str(self.brush_mode), "idle")
            self._drawing_status_label.setText(
                f"Drawing: {mode} ({point_count} pts)" if drawing else f"Editing: {mode}" if active else "Editing: idle"
            )
        if self._finish_drawing_button is not None:
            self._finish_drawing_button.setEnabled(active and (not drawing or point_count >= 3))
        if self._undo_drawing_button is not None:
            brush_undo_available = self.brush_mode in {"brush", "erase"} and (
                self._brush_stroke_active or bool(self._brush_undo_stack)
            )
            self._undo_drawing_button.setEnabled((drawing and point_count > 0) or brush_undo_available)
        if self._cancel_drawing_button is not None:
            self._cancel_drawing_button.setEnabled(active)
        if self._drawing_controls_widget is not None:
            self._drawing_controls_widget.setVisible(active)
        self._refresh_roi_action_visibility()
        self._refresh_roi_interaction_state()

    def _set_tool_settings_expanded(self, expanded: bool):
        self._refresh_roi_action_visibility()

    def _refresh_roi_action_visibility(self):
        tool_mode = self.brush_mode
        show_brush_size = tool_mode in {"brush", "erase", "polygon"}
        show_label_controls = tool_mode in {"brush", "wand"}
        show_auto_subtract = tool_mode == "brush"
        show_wand_tolerance = tool_mode == "wand"
        show_lasso_controls = tool_mode == "lasso"
        if self._brush_size_label is not None:
            self._brush_size_label.setVisible(show_brush_size)
            self._brush_size_label.setText("Brush")
        if getattr(self, "brush_size_spin", None) is not None:
            self.brush_size_spin.setVisible(show_brush_size)
        if self._brush_label_label is not None:
            self._brush_label_label.setVisible(show_label_controls)
        if getattr(self, "brush_label_spin", None) is not None:
            self.brush_label_spin.setVisible(show_label_controls)
        if getattr(self, "brush_auto_subtract_check", None) is not None:
            self.brush_auto_subtract_check.setVisible(show_auto_subtract)
        if self._wand_tolerance_label is not None:
            self._wand_tolerance_label.setVisible(show_wand_tolerance)
        if getattr(self, "wand_tolerance_spin", None) is not None:
            self.wand_tolerance_spin.setVisible(show_wand_tolerance)
        if self._lasso_support_label is not None:
            self._lasso_support_label.setVisible(show_lasso_controls)
        if self.lasso_support_spin is not None:
            self.lasso_support_spin.setVisible(show_lasso_controls)
        if self._lasso_spline_label is not None:
            self._lasso_spline_label.setVisible(show_lasso_controls)
        if self.lasso_spline_combo is not None:
            self.lasso_spline_combo.setVisible(show_lasso_controls)
        if self.lasso_snap_check is not None:
            self.lasso_snap_check.setVisible(show_lasso_controls)
        if self._roi_tool_settings_section is not None:
            self._roi_tool_settings_section.setVisible(
                show_brush_size or show_label_controls or show_wand_tolerance or show_lasso_controls
            )

        buttons = self._advanced_roi_buttons
        section = self._advanced_roi_actions_section
        if not buttons or section is None:
            return
        try:
            selected = self._selected_rois_for_action() if self.session.is_loaded else []
        except Exception:
            selected = []
        has_selection = bool(selected)
        has_multi_selection = len(selected) >= 2
        has_rois = bool(self.session.rois)
        has_multiple_active_rois = len(self.session.rois) >= 2
        has_z = bool(self.session.is_loaded and "z" in tuple(self.session.spatial_dim_names))
        visible_by_label = {
            "Consolidate": has_selection,
            "Combine": has_multi_selection or has_multiple_active_rois,
            "Subtract": has_multi_selection or has_multiple_active_rois,
            "Shift...": has_selection,
            "Copy Z": has_selection and has_z,
            "Delete Z": has_selection and has_z,
            "Apply All": has_rois,
            "Interp Z": has_selection and has_z,
        }
        any_visible = False
        for label, button in buttons.items():
            visible = bool(visible_by_label.get(label, True))
            button.setVisible(visible)
            any_visible = any_visible or visible
        section.setVisible(any_visible and self.brush_mode is None)

    def _set_image_pan_locked(self, locked: bool):
        locked = bool(locked)
        if not hasattr(self, "image_view") or self.image_view is None:
            return
        if self._image_pan_locked == locked:
            return
        self._image_pan_locked = locked
        try:
            self.image_view.getView().setMouseEnabled(x=not locked, y=not locked)
        except Exception:
            _LOG.debug("Ignored optional GUI operation failure", exc_info=True)

    def _refresh_roi_interaction_state(self):
        editing_roi = self.brush_mode is not None or bool(self.roi_items)
        self._set_image_pan_locked(editing_roi)

    def finish_drawing_roi(self):
        self._finish_active_roi_tool()

    def undo_drawing_point(self):
        if self.brush_mode in {"brush", "erase"}:
            self._undo_brush_stroke()
            return
        if self.brush_mode not in {"polygon", "lasso"} or not self._polygon_points:
            return
        self._polygon_points.pop()
        self._update_polygon_preview()
        self._refresh_drawing_controls()
        self.statusBar().showMessage("Removed last drawing point.")

    def cancel_drawing_roi(self):
        self._cancel_active_roi_tool()

    def _finish_active_roi_tool(self):
        if self.brush_mode in {"polygon", "lasso"}:
            self._finish_polygon_roi()
            return
        if self.brush_mode is None:
            return
        if self.brush_mode in {"brush", "erase"}:
            self._finish_brush_stroke()
            self._clear_brush_edit_session()
        self.brush_mode = None
        self._last_brush_stamp = None
        self._clear_brush_preview()
        self._set_tool_settings_expanded(False)
        self._refresh_drawing_controls()
        self.statusBar().showMessage("ROI editing finished.")

    def _cancel_active_roi_tool(self):
        if self.brush_mode in {"polygon", "lasso"}:
            self._cancel_polygon_roi()
            return
        if self.brush_mode is None:
            return
        restored = False
        if self.brush_mode in {"brush", "erase"}:
            restored = self._cancel_brush_edit_session()
        self.brush_mode = None
        self._last_brush_stamp = None
        self._clear_brush_preview()
        self._set_tool_settings_expanded(False)
        self._refresh_drawing_controls()
        if restored:
            self.statusBar().showMessage("ROI editing cancelled; brush changes were discarded.")
        else:
            self.statusBar().showMessage("ROI editing cancelled.")

    def _update_polygon_preview(self):
        view_box = self.image_view.getView()
        if self._polygon_preview is None:
            self._polygon_preview = pg.PlotDataItem(
                pen=pg.mkPen("#f59e0b", width=3),
                symbol="o",
                symbolBrush="#22d3ee",
                symbolPen=pg.mkPen("#111827", width=1),
                symbolSize=8,
            )
            self._polygon_preview.setZValue(30)
            view_box.addItem(self._polygon_preview)
        if not self._polygon_points:
            self._polygon_preview.setData([], [])
            self._update_polygon_start_preview(None)
            return
        pts = np.asarray(self._polygon_points, dtype=float)
        if pts.shape[0] >= 3:
            pts = np.vstack([pts, pts[0]])
        self._polygon_preview.setData(pts[:, 0], pts[:, 1])
        self._update_polygon_start_preview(np.asarray(self._polygon_points[0], dtype=float))

    def _update_polygon_start_preview(self, point: np.ndarray | None):
        view_box = self.image_view.getView()
        if point is None:
            if self._polygon_start_preview is not None:
                self._polygon_start_preview.setData([], [])
            return
        if self._polygon_start_preview is None:
            self._polygon_start_preview = pg.PlotDataItem(
                pen=None,
                symbol="o",
                symbolBrush="#22c55e",
                symbolPen=pg.mkPen("#ffffff", width=2),
                symbolSize=13,
            )
            self._polygon_start_preview.setZValue(32)
            view_box.addItem(self._polygon_start_preview)
        self._polygon_start_preview.setData([float(point[0])], [float(point[1])])

    def _append_polygon_point(self, x: float, y: float, *, min_distance: float = 0.0) -> bool:
        if self._polygon_points and min_distance > 0:
            last_x, last_y = self._polygon_points[-1]
            if (float(x) - last_x) ** 2 + (float(y) - last_y) ** 2 < float(min_distance) ** 2:
                return False
        self._polygon_points.append((float(x), float(y)))
        return True

    def _polygon_click_closes(self, x: float, y: float) -> bool:
        if len(self._polygon_points) < 3:
            return False
        first_x, first_y = self._polygon_points[0]
        radius = max(4.0, float(self.brush_size_spin.value()) * 0.5)
        return (float(x) - first_x) ** 2 + (float(y) - first_y) ** 2 <= radius**2

    def _clear_polygon_preview(self):
        if self._polygon_preview is None:
            pass
        else:
            try:
                self.image_view.getView().removeItem(self._polygon_preview)
            except Exception:
                _LOG.debug("Ignored optional GUI operation failure", exc_info=True)
        self._polygon_preview = None
        if self._polygon_start_preview is not None:
            try:
                self.image_view.getView().removeItem(self._polygon_start_preview)
            except Exception:
                _LOG.debug("Ignored optional GUI operation failure", exc_info=True)
            self._polygon_start_preview = None

    def _cancel_polygon_roi(self):
        self.brush_mode = None
        self._polygon_points = []
        self._clear_polygon_preview()
        self._clear_brush_preview()
        self._set_tool_settings_expanded(False)
        self._refresh_drawing_controls()
        self.statusBar().showMessage("Polygon drawing cancelled.")

    def _finish_polygon_roi(self):
        draw_mode = self.brush_mode or "polygon"
        raw_points = np.asarray(self._polygon_points, dtype=float)
        lasso_snap = False
        if draw_mode == "lasso":
            lasso_snap = bool(self.lasso_snap_check.isChecked()) if self.lasso_snap_check is not None else False
            support_points = int(self.lasso_support_spin.value()) if self.lasso_support_spin is not None else 64
            spline_kind = (
                str(self.lasso_spline_combo.currentData() or "cubic")
                if self.lasso_spline_combo is not None
                else "cubic"
            )
            points = _smooth_lasso_support_points(
                raw_points,
                image=self._roi_source_image() if lasso_snap else None,
                snap=lasso_snap,
                snap_radius=2,
                snap_strength=0.20,
                support_points=support_points,
                spline_kind=spline_kind,
            )
        else:
            points = _simplify_polyline(raw_points, tolerance=0.0, closed=True)
            support_points = int(points.shape[0])
            spline_kind = "linear"
        if len(points) < 3:
            self.statusBar().showMessage("Polygon ROI needs at least three vertices.")
            self._refresh_drawing_controls()
            return
        current = np.asarray(self.session.frame_image())
        h, w = current.shape[-2], current.shape[-1]
        mask_2d = _polygon_mask((h, w), points)
        spatial_dims = tuple(self.session.spatial_dim_names)
        prefix = "lasso" if draw_mode == "lasso" else "polygon"
        draw_metadata = {
            "source_experiment": self.session.selected_experiment,
            "slice_indices": self._current_slice_indices(),
            "generated_by": f"mrax.gui.{prefix}",
            "raw_point_count": int(raw_points.shape[0]) if draw_mode == "lasso" else int(points.shape[0]),
            "support_point_count": int(points.shape[0]),
            "requested_support_points": int(support_points),
            "spline_kind": str(spline_kind),
            "snap_to_edges": bool(lasso_snap),
        }
        target_roi = self.session.roi()
        if target_roi is None and self.session.rois:
            target_roi = self.session.rois[0]
        if target_roi is not None:
            if not _roi_can_address_display_plane(
                target_roi,
                tuple(self.session.spatial_shape),
                spatial_dims,
                self._current_slice_indices(),
                (h, w),
            ):
                self.statusBar().showMessage("Selected ROI cannot be edited on the current image plane.")
                self._refresh_drawing_controls()
                return
            updated_mask = replace_mask_slice(
                self._roi_as_spatial_mask(target_roi),
                mask_2d,
                dims=spatial_dims,
                indices=self._current_slice_indices(),
                mode="add",
            )
            meta = _rasterized_roi_metadata(target_roi.metadata, prefix)
            meta.update(draw_metadata)
            meta["appended_contour_count"] = int(meta.get("appended_contour_count", 0) or 0) + 1
            meta["last_added_points"] = points.tolist()
            name = target_roi.name
            self._replace_roi(ROI(name, updated_mask, dims=spatial_dims, metadata=meta))
            message = f"Added {prefix} contour to ROI '{name}'."
        else:
            mask = embed_mask_slice(
                mask_2d,
                spatial_shape=tuple(self.session.spatial_shape),
                spatial_dims=spatial_dims,
                indices=self._current_slice_indices(),
            )
            name = _unique_roi_name(self.session.rois, prefix)
            roi = ROI(
                name,
                mask,
                dims=spatial_dims,
                metadata={
                    "shape": "polygon",
                    "points": points.tolist(),
                    "color": _roi_color(len(self.session.rois)),
                    **draw_metadata,
                },
            )
            self.session.add_roi(roi)
            message = f"Added editable {prefix} ROI. Drag vertices to refine it."
        self.session.fit_settings.selected_roi = name
        self.brush_mode = None
        self._polygon_points = []
        self._clear_polygon_preview()
        self.refresh_roi_list()
        self.update_view(preserve_image_view=True)
        self.mark_dirty("ROI set changed")
        self._set_tool_settings_expanded(False)
        self._refresh_drawing_controls()
        self.statusBar().showMessage(message)

    def consolidate_selected_roi(self):
        roi = self._selected_roi_for_volume_action()
        if roi is None:
            return
        display = self._roi_source_image()
        if display is None:
            return
        shape = tuple(int(v) for v in np.asarray(display).shape[-2:])
        meta = dict(roi.metadata)
        item = self.roi_items.get(roi.name)
        if item is not None:
            kind = self.roi_item_kinds.get(roi.name, str(meta.get("shape", "mask")))
            mask_2d, geometry = _mask_from_roi_item(item, kind, shape)
            meta.update(geometry)
        else:
            try:
                mask_2d = np.asarray(
                    mask_slice_for_display(
                        roi.mask,
                        dims=roi.dims,
                        indices=self._current_slice_indices(),
                    ),
                    dtype=bool,
                )
            except Exception as exc:
                self.statusBar().showMessage(f"Could not consolidate ROI: {exc}")
                return
        if tuple(mask_2d.shape) != shape:
            self.statusBar().showMessage("Selected ROI is not aligned with the current image plane.")
            return
        meta = _rasterized_roi_metadata(meta, "consolidate")
        self._commit_display_mask_edit(roi, mask_2d, metadata=meta)
        self.statusBar().showMessage(f"Consolidated ROI '{roi.name}' into an editable mask.")

    def auto_threshold_roi(self):
        if not self.session.is_loaded:
            return
        image = self._roi_source_image()
        if image is None:
            return
        mask_2d, metadata = _auto_threshold_mask(image)
        if not np.any(mask_2d):
            self.statusBar().showMessage("Auto ROI did not find a thresholded region.")
            return
        spatial_dims = tuple(self.session.spatial_dim_names)
        mask = embed_mask_slice(
            mask_2d,
            spatial_shape=tuple(self.session.spatial_shape),
            spatial_dims=spatial_dims,
            indices=self._current_slice_indices(),
        )
        name = _unique_roi_name(self.session.rois, "auto")
        metadata.update(
            {
                "shape": "mask",
                "color": _roi_color(len(self.session.rois)),
                "source_experiment": self.session.selected_experiment,
                "slice_indices": self._current_slice_indices(),
                "generated_by": "mrax.gui.auto_threshold",
            }
        )
        self.session.add_roi(ROI(name, mask, dims=spatial_dims, metadata=metadata))
        self.session.fit_settings.selected_roi = name
        self.refresh_roi_list()
        self.update_view(preserve_image_view=True)
        self.mark_dirty("ROI set changed")
        self.statusBar().showMessage(f"Added auto-threshold ROI '{name}'.")

    def apply_fit_controls(self):
        self._finish_parameter_table_edits()
        model_key = str(self.model_combo.currentData() or "")
        exp_type = self.session.current_experiment.get("exp_type") if self.session.is_loaded else None
        self.session.fit_settings.model_name = model_key
        self.refresh_optimizer_choices(preferred=self.optimizer_combo.currentText())
        self.session.fit_settings.optimizer_name = self.optimizer_combo.currentText()
        self.session.fit_settings.num_steps = int(self.steps_spin.value())
        self.session.fit_settings.step_size = float(self.step_size_spin.value())
        self.session.fit_settings.mask_fill_value = float(self.mask_fill_spin.value())
        self.session.fit_settings.auto_mask_nan_observables = self.auto_nan_check.isChecked()
        self.session.fit_settings.use_view_state_data = bool(self.fit_use_view_state_check.isChecked())
        derived = self.derived_maps_combo.currentText()
        self.session.fit_settings.compute_derived_maps = None if derived == "auto" else derived == "yes"
        self.session.fit_settings.sat_trans_reference_scan = (
            self.sat_trans_reference_combo.currentData()
            if _sat_trans_reference_visible(model_key, exp_type)
            else None
        )
        if _ab_mt_prior_maps_visible(model_key):
            self.session.fit_settings.ab_mt_t1_map_ref = self.ab_mt_t1_combo.currentData()
            self.session.fit_settings.ab_mt_b1_map_ref = self.ab_mt_b1_combo.currentData()
            self.session.fit_settings.ab_mt_b0_map_ref = self.ab_mt_b0_combo.currentData()
        else:
            self.session.fit_settings.ab_mt_t1_map_ref = None
            self.session.fit_settings.ab_mt_b1_map_ref = None
            self.session.fit_settings.ab_mt_b0_map_ref = None
        self.session.fit_settings.selected_pools = self.selected_pool_names()
        param_overrides, pool_overrides, regularizer_configs, constant_dims = self._parameter_overrides_from_table()
        self.session.fit_settings.param_overrides = param_overrides
        self.session.fit_settings.pool_overrides = pool_overrides
        self.session.fit_settings.regularizer_configs = regularizer_configs
        self.session.fit_settings.parameter_constant_dims = constant_dims
        if constant_dims:
            summary = ", ".join(f"{name}: {_format_csv(dims)}" for name, dims in constant_dims.items())
            self.statusBar().showMessage(
                f"Constant dims active ({summary}); these force parameters constant across those dimensions."
            )

    def _finish_parameter_table_edits(self):
        current = self.param_table.currentItem()
        if current is not None:
            self.param_table.closePersistentEditor(current)
        self.param_table.clearFocus()
        QtWidgets.QApplication.processEvents()

    def _parameter_overrides_from_table(self):
        model_key = str(self.model_combo.currentData() or "")
        selected_pools = set(self.selected_pool_names() or model_default_pools(model_key))
        param_overrides: dict[str, dict[str, object]] = {}
        pool_overrides: dict[str, dict[str, object]] = {}
        regularizer_configs: dict[str, list[dict[str, object]]] = {}
        constant_dims: dict[str, tuple[int | str, ...]] = {}
        for row in range(self.param_table.rowCount()):
            name_item = self.param_table.item(row, 0)
            if name_item is None:
                continue
            name = name_item.text().strip()
            initial = _parse_optional_float(self.param_table.item(row, 1).text() if self.param_table.item(row, 1) else "")
            lower = _parse_optional_float(self.param_table.item(row, 2).text() if self.param_table.item(row, 2) else "")
            upper = _parse_optional_float(self.param_table.item(row, 3).text() if self.param_table.item(row, 3) else "")
            fixed_widget = self.param_table.cellWidget(row, 9)
            fixed = bool(fixed_widget.isChecked()) if fixed_widget is not None and hasattr(fixed_widget, "isChecked") else False
            cfg: dict[str, object] = {}
            if fixed:
                cfg["trainable"] = False
                cfg["value"] = 0.0 if initial is None else initial
                cfg["transform"] = "identity"
            elif initial is not None:
                cfg["init"] = initial
            if not fixed and lower is not None and upper is not None:
                cfg["bounds"] = [lower, upper]
                cfg["transform"] = "bounded"
            if cfg:
                pool_name = (
                    _pool_suffix(name, selected_pools)
                    if model_key in {"multipool_lorentzian", "ab_mt_n", "zaiss_full"}
                    else None
                )
                if pool_name is not None:
                    base_name = name[: -(len(pool_name) + 1)]
                    pool_overrides.setdefault(pool_name, {})[base_name] = cfg
                else:
                    param_overrides[name] = cfg
            reg_type_widget = self.param_table.cellWidget(row, 5)
            reg_type = (
                reg_type_widget.currentText()
                if reg_type_widget is not None and hasattr(reg_type_widget, "currentText")
                else (self.param_table.item(row, 5).text() if self.param_table.item(row, 5) else "")
            ).strip()
            reg_weight = _regularizer_weight_from_cell(self.param_table, row, 6)
            axes_widget = self.param_table.cellWidget(row, 7)
            axes_text = (
                axes_widget.currentText()
                if axes_widget is not None and hasattr(axes_widget, "currentText")
                else (self.param_table.item(row, 7).text() if self.param_table.item(row, 7) else "")
            )
            reg_axes = _parse_axis_values(axes_text, self.session.experiment_dims)
            if _regularizer_config_enabled(reg_type, reg_weight):
                regularizer_configs[name] = [
                    {
                        "type": reg_type,
                        "weight": float(reg_weight),
                        **({"axes": tuple(reg_axes)} if reg_axes else {}),
                    }
                ]
            dims_widget = self.param_table.cellWidget(row, 8)
            dims_text = (
                dims_widget.currentText()
                if dims_widget is not None and hasattr(dims_widget, "currentText")
                else (self.param_table.item(row, 8).text() if self.param_table.item(row, 8) else "")
            )
            dims = tuple(_parse_csv_values(dims_text))
            if dims:
                constant_dims[name] = dims
        return param_overrides, pool_overrides, regularizer_configs, constant_dims

    def run_preview_fit(self):
        if self.is_busy:
            return
        if not self.session.is_loaded:
            return
        self.apply_fit_controls()
        roi = self.session.roi()
        if roi is not None and not self._roi_has_pixels(roi):
            self.statusBar().showMessage(f"ROI '{roi.name}' is empty; preview fit was not run.")
            self.update_plot()
            return
        try:
            view_state_kwargs = self._visual_state_fit_kwargs(include_roi=True)
        except Exception:
            self.show_error("Could not prepare view-state fit data", traceback.format_exc())
            return
        self.run_guarded_task(
            lambda progress: self.session.run_preview_fit(
                progress_callback=progress,
                **view_state_kwargs,
            ),
            self.on_preview_fit_finished,
            "Running ROI preview fit...",
            error_title="Could not run preview fit",
        )

    def on_preview_fit_finished(self, result: dict[str, Any]):
        if self.session.selected_experiment:
            self._preview_results_by_experiment[self.session.selected_experiment] = result
        self.last_preview_fit_result = result
        self.update_plot()
        self.mark_dirty("preview fit run")
        self.statusBar().showMessage("Preview fit complete.")

    def run_fit_sanity_check(self):
        if not self.session.is_loaded:
            return
        self.apply_fit_controls()
        try:
            view_state_kwargs = self._visual_state_fit_kwargs(include_roi=False)
        except Exception:
            self.show_error("Could not prepare view-state fit data", traceback.format_exc())
            return
        result = self.session.run_fit_sanity_check(**view_state_kwargs)
        self.fit_results_text.setPlainText(result.get("report") or "")
        if result.get("ok"):
            self.statusBar().showMessage("Fit sanity check passed.")
        else:
            self.statusBar().showMessage("Fit sanity check found issues.")

    def run_full_fit(self):
        if self.is_busy:
            return
        if not self.session.is_loaded:
            return
        self.apply_fit_controls()
        try:
            view_state_kwargs = self._visual_state_fit_kwargs(include_roi=True)
        except Exception:
            self.show_error("Could not prepare view-state fit data", traceback.format_exc())
            return
        self.run_guarded_task(
            lambda progress: self.session.run_current_fit(
                progress_callback=progress,
                **view_state_kwargs,
            ),
            self.on_full_fit_finished,
            "Running full-map fit...",
            error_title="Could not run full-map fit",
        )

    def on_full_fit_finished(self, result: dict[str, Any]):
        self.last_fit_result = result
        if self.session.selected_experiment:
            self._fit_results_by_experiment[self.session.selected_experiment] = result
        self.session.replace_fitted_maps(result.get("fields") or {})
        self.refresh_map_list()
        self.refresh_registration_tab()
        self.view_combo.setCurrentText("Map")
        self.update_view()
        self.update_fit_results_summary()
        self.mark_dirty("full-map fit run")
        self.statusBar().showMessage("Full-map fit complete. Use Save Fit to persist results.")

    def update_fit_results_summary(self):
        result = self.last_fit_result
        if result is None:
            self.fit_results_text.clear()
            return
        self.fit_results_text.setPlainText(_fit_result_summary_text(result, self.session.rois))

    def save_fit_dialog(self):
        if self.last_fit_result is None:
            self.statusBar().showMessage("No fit result is available to save.")
            return
        directory = self._get_existing_directory(
            "Choose output folder",
            str(Path.cwd()),
        )
        if not directory:
            return
        try:
            manifest = self.session.save_fit_result(self.last_fit_result, directory)
        except Exception:
            self.show_error("Could not save fit result", traceback.format_exc())
            return
        self.statusBar().showMessage(f"Saved fit result to {manifest}")

    @property
    def is_busy(self) -> bool:
        return self._task_running or self._busy_message is not None

    def _begin_busy(self, message: str):
        self._task_running = True
        self._busy_message = message
        self._hover_status_message = None
        if self._central_splitter is not None:
            self._central_splitter.setEnabled(False)
        if self.progress_bar is not None:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat("0%")
            self.progress_bar.setToolTip(message)
            self.progress_bar.show()
        self.statusBar().showMessage(message)
        self._flush_gui_events()

    def _end_busy(self):
        self._busy_message = None
        if self._central_splitter is not None:
            self._central_splitter.setEnabled(True)
        if self.progress_bar is not None:
            self.progress_bar.hide()
            self.progress_bar.setValue(0)
            self.progress_bar.setToolTip("")
        self._task_running = False
        self._flush_gui_events()

    def on_task_progress(self, current: int, total: int, detail: str):
        if self.progress_bar is None:
            return
        total = max(1, int(total))
        current = max(0, min(int(current), total))
        percent = int(round((current / total) * 100.0))
        self.progress_bar.setValue(percent)
        self.progress_bar.setFormat(f"{percent}%")
        if detail:
            self.progress_bar.setToolTip(detail)
        self._flush_gui_events()

    def _flush_gui_events(self):
        app = QtWidgets.QApplication.instance()
        if app is None:
            return
        flags = getattr(QtCore.QEventLoop, "ExcludeUserInputEvents", None)
        if flags is None and hasattr(QtCore.QEventLoop, "ProcessEventsFlag"):
            flags = QtCore.QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents
        try:
            if flags is not None:
                app.processEvents(flags)
            else:
                app.processEvents()
        except TypeError:
            app.processEvents()

    def run_guarded_task(
        self,
        fn: Callable[[Callable[[int, int, str], None]], Any],
        callback: Callable[[Any], None],
        message: str,
        *,
        error_title: str = "Task failed",
    ):
        if self.is_busy:
            return
        self._begin_busy(message)
        try:
            result = fn(self.on_task_progress)
            callback(result)
        except Exception:
            self.show_error(error_title, traceback.format_exc())
            self.statusBar().showMessage("Task failed.")
        finally:
            self._end_busy()

    def show_error(self, title: str, detail: str, *, record: bool = True):
        if record:
            self.record_diagnostic_event("error", title, detail=detail, update_status=False)
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Critical)
        box.setWindowTitle(title)
        box.setText(title)
        box.setDetailedText(detail)
        self._exec_dialog(box)



def _build_main_window(qt_core, qt_gui, qt_widgets, pyqtgraph):
    """Create the concrete Qt window class after optional GUI imports succeed."""

    global QtCore, QtGui, QtWidgets, pg
    QtCore, QtGui, QtWidgets, pg = qt_core, qt_gui, qt_widgets, pyqtgraph

    class MainWindow(_MainWindowMixin, QtWidgets.QMainWindow):
        pass

    return MainWindow


def _roi_color(index: int) -> str:
    colors = (
        "#2dd4bf",
        "#f97316",
        "#a855f7",
        "#22c55e",
        "#e11d48",
        "#0ea5e9",
    )
    return colors[index % len(colors)]


def _set_combo_data_items(combo, items: Sequence[tuple[str, Any]], current: Any | None = None) -> None:
    combo.blockSignals(True)
    try:
        combo.clear()
        for label, value in items:
            combo.addItem(str(label), value)
        if current is not None:
            for idx in range(combo.count()):
                if combo.itemData(idx) == current:
                    combo.setCurrentIndex(idx)
                    break
    finally:
        combo.blockSignals(False)


def _set_combo_text_items(combo, items: Sequence[str], current: str | None = None) -> None:
    combo.blockSignals(True)
    try:
        combo.clear()
        combo.addItems([str(item) for item in items])
        if current:
            idx = combo.findText(str(current))
            if idx >= 0:
                combo.setCurrentIndex(idx)
    finally:
        combo.blockSignals(False)


def _first_spatial_image(array: Any) -> np.ndarray:
    arr = np.asarray(array)
    while arr.ndim > 3:
        arr = np.asarray(arr[0])
    if arr.ndim == 3 and 1 in arr.shape:
        arr = np.squeeze(arr)
    if arr.ndim not in (2, 3):
        raise ValueError(f"Expected a 2D or 3D spatial image, got shape {arr.shape}")
    return np.asarray(arr)


def _registration_preview_plane(array: Any) -> np.ndarray:
    arr = _first_spatial_image(array)
    if arr.ndim == 3:
        arr = np.asarray(arr[:, :, int(arr.shape[2] // 2)])
    if np.iscomplexobj(arr):
        arr = np.abs(arr)
    return np.asarray(arr)


def _blend_reference_background(
    reference: Any,
    foreground: Any,
    *,
    foreground_fraction: float,
    reference_channel: str = "red",
    foreground_channel: str = "green",
) -> np.ndarray:
    """Blend a reference background under the already-rendered current view."""

    bg_unit, fg_unit = _equalized_overlay_units(reference, foreground)
    bg = _unit_to_rgb_for_blend(bg_unit, channel=reference_channel)
    fg = _unit_to_rgb_for_blend(fg_unit, channel=foreground_channel)
    rows = min(int(bg.shape[0]), int(fg.shape[0]))
    cols = min(int(bg.shape[1]), int(fg.shape[1]))
    if rows <= 0 or cols <= 0:
        raise ValueError("reference and current view have no overlapping pixels")
    alpha = float(np.clip(foreground_fraction, 0.0, 1.0))
    out = (1.0 - alpha) * bg[:rows, :cols, :] + alpha * fg[:rows, :cols, :]
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _unit_to_rgb_for_blend(unit: np.ndarray, *, channel: str = "gray") -> np.ndarray:
    return np.clip(unit[..., None] * _blend_channel_vector(channel), 0.0, 1.0).astype(np.float32)


def _equalized_overlay_units(reference: Any, foreground: Any) -> tuple[np.ndarray, np.ndarray]:
    ref = _image_to_scalar_for_blend(reference)
    fg = _image_to_scalar_for_blend(foreground)
    rows = min(int(ref.shape[0]), int(fg.shape[0]))
    cols = min(int(ref.shape[1]), int(fg.shape[1]))
    if rows <= 0 or cols <= 0:
        raise ValueError("reference and current view have no overlapping pixels")
    return (
        _histogram_equalized_unit(ref[:rows, :cols]),
        _histogram_equalized_unit(fg[:rows, :cols]),
    )


def _image_to_scalar_for_blend(image: Any) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        rgb = np.asarray(arr[..., :3], dtype=float)
        finite = rgb[np.isfinite(rgb)]
        if finite.size and (float(np.nanmin(finite)) < 0.0 or float(np.nanmax(finite)) > 1.0):
            rgb = np.stack([_robust_display_image(rgb[..., idx]) for idx in range(3)], axis=-1)
        rgb = np.clip(np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        return np.asarray(0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2], dtype=np.float32)
    plane = _registration_preview_plane(arr)
    if np.iscomplexobj(plane):
        plane = np.abs(plane)
    return np.asarray(plane, dtype=np.float32)


def _histogram_equalized_unit(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=float)
    out = np.zeros(arr.shape, dtype=np.float32)
    finite = np.isfinite(arr)
    values = arr[finite]
    if values.size == 0:
        return out
    if values.size == 1 or float(np.nanmax(values)) <= float(np.nanmin(values)):
        return out
    lo, hi = np.percentile(values, [0.5, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(values))
        hi = float(np.nanmax(values))
    if hi <= lo:
        return out
    clipped = np.clip(values, lo, hi)
    unique, inverse, counts = np.unique(clipped, return_inverse=True, return_counts=True)
    if unique.size <= 1:
        return out
    cumulative_before = np.concatenate([[0], np.cumsum(counts[:-1])]).astype(float)
    ranks_by_value = cumulative_before / max(1.0, float(clipped.size - 1))
    ranks = ranks_by_value[inverse]
    # Keep a little headroom so the overlay slider changes balance without
    # driving large flat regions into display clipping.
    equalized = 0.96 * ranks
    out[finite] = equalized.astype(np.float32, copy=False)
    return out


def _blend_channel_vector(channel: str) -> np.ndarray:
    key = str(channel or "gray").lower().replace("-", "_")
    colors = {
        "red": (1.0, 0.0, 0.0),
        "green": (0.0, 1.0, 0.0),
        "blue": (0.0, 0.35, 1.0),
        "cyan": (0.0, 1.0, 1.0),
        "magenta": (1.0, 0.0, 1.0),
        "yellow": (1.0, 1.0, 0.0),
        "gray": (1.0, 1.0, 1.0),
    }
    return np.asarray(colors.get(key, colors["gray"]), dtype=np.float32)


def _display_spatial_shape(array: Any) -> tuple[int, int]:
    arr = np.asarray(array)
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        return int(arr.shape[0]), int(arr.shape[1])
    if arr.ndim < 2:
        raise ValueError(f"Expected at least a 2D display image, got shape {arr.shape}")
    return int(arr.shape[-2]), int(arr.shape[-1])


def _viewer_display_levels(
    image: Any,
    display_levels: tuple[float, float] | None,
) -> tuple[float, float] | None:
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        return (0.0, 1.0)
    return display_levels


def _capture_pyqtgraph_image_view_state(image_view: Any) -> dict[str, Any] | None:
    """Capture user-controlled image view state before a redraw."""

    state: dict[str, Any] = {}
    try:
        view_box = image_view.getView()
        view_range = view_box.viewRange()
        if len(view_range) >= 2:
            x_range = _finite_range_tuple(view_range[0])
            y_range = _finite_range_tuple(view_range[1])
            if x_range is not None and y_range is not None:
                state["x_range"] = x_range
                state["y_range"] = y_range
    except Exception:
        _LOG.debug("Ignored optional GUI operation failure", exc_info=True)

    levels = _pyqtgraph_image_view_levels(image_view)
    if levels is not None:
        state["levels"] = levels
    return state or None


def _restore_pyqtgraph_image_view_state(image_view: Any, state: Mapping[str, Any]) -> None:
    levels = _image_view_state_levels(state)
    if levels is not None:
        try:
            image_view.setLevels(*levels)
        except TypeError:
            try:
                image_view.setLevels(levels)
            except Exception:
                _LOG.debug("Ignored optional GUI operation failure", exc_info=True)
        except Exception:
            _LOG.debug("Ignored optional GUI operation failure", exc_info=True)
        try:
            image_view.getHistogramWidget().setLevels(*levels)
        except Exception:
            _LOG.debug("Ignored optional GUI operation failure", exc_info=True)

    x_range = _finite_range_tuple(state.get("x_range"))
    y_range = _finite_range_tuple(state.get("y_range"))
    if x_range is None or y_range is None:
        return
    try:
        image_view.getView().setRange(xRange=x_range, yRange=y_range, padding=0)
    except Exception:
        _LOG.debug("Ignored optional GUI operation failure", exc_info=True)


def _image_view_state_levels(state: Mapping[str, Any] | None) -> tuple[float, float] | None:
    if not state:
        return None
    return _finite_range_tuple(state.get("levels"))


def _pyqtgraph_image_view_levels(image_view: Any) -> tuple[float, float] | None:
    try:
        levels = image_view.getHistogramWidget().getLevels()
        result = _finite_range_tuple(levels)
        if result is not None:
            return result
    except Exception:
        _LOG.debug("Ignored optional GUI operation failure", exc_info=True)
    for attr_name in ("getLevels", "levels"):
        try:
            attr = getattr(image_view, attr_name)
            levels = attr() if callable(attr) else attr
            result = _finite_range_tuple(levels)
            if result is not None:
                return result
        except Exception:
            _LOG.debug("Ignored optional GUI operation failure", exc_info=True)
    try:
        image_item = image_view.getImageItem()
    except Exception:
        image_item = None
    if image_item is not None:
        for attr_name in ("getLevels", "levels"):
            try:
                attr = getattr(image_item, attr_name)
                levels = attr() if callable(attr) else attr
                result = _finite_range_tuple(levels)
                if result is not None:
                    return result
            except Exception:
                _LOG.debug("Ignored optional GUI operation failure", exc_info=True)
    return None


def _finite_range_tuple(values: Any) -> tuple[float, float] | None:
    try:
        arr = np.asarray(values, dtype=float).reshape(-1)
    except Exception:
        return None
    if arr.size < 2:
        return None
    lo = float(arr[0])
    hi = float(arr[1])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return None
    return (lo, hi)


def _format_display_pixel_value(value: Any) -> str:
    arr = np.asarray(value)
    if arr.ndim == 0:
        return _format_display_scalar_value(arr.item())
    if arr.size == 0:
        return "[]"
    flat = arr.reshape(-1)
    if flat.size <= 4:
        return "[" + ", ".join(_format_display_scalar_value(item) for item in flat) + "]"
    finite = np.isfinite(flat)
    if np.any(finite):
        finite_values = flat[finite]
        return (
            f"shape={tuple(arr.shape)}, "
            f"min={_format_display_scalar_value(np.min(finite_values))}, "
            f"max={_format_display_scalar_value(np.max(finite_values))}"
        )
    return f"shape={tuple(arr.shape)}"


def _format_display_scalar_value(value: Any) -> str:
    if np.iscomplexobj(value):
        complex_value = complex(value)
        return f"{complex_value.real:.6g}{complex_value.imag:+.6g}j"
    try:
        return f"{float(value):.6g}" if bool(np.isfinite(value)) else str(value)
    except Exception:
        return str(value)


def _registration_result_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "kind": result.get("kind"),
        "metadata": dict(result.get("metadata") or {}),
        "manual": dict(result.get("manual") or {}),
        "options": _metadata_to_jsonable(dict(result.get("options") or {})),
        "diagnostics": _metadata_to_jsonable(dict(result.get("diagnostics") or {})),
    }


def _parse_name_sequence(text: str) -> tuple[str, ...]:
    return tuple(
        item.strip()
        for chunk in str(text or "").replace(";", ",").split(",")
        for item in chunk.split()
        if item.strip()
    )


def _drift_options_to_dict(options: DriftCorrectionOptions) -> dict[str, Any]:
    return {
        "transform": options.transform,
        "metric": options.metric,
        "reference": options.reference,
        "dimensions": None if options.dimensions is None else list(options.dimensions),
        "interpolation": options.interpolation,
        "optimizer": options.optimizer,
        "step_size": float(options.step_size),
        "num_steps": int(options.num_steps),
        "fit_intercept": bool(options.fit_intercept),
    }


def _linear_drift_result_summary(result: Any, metadata: Mapping[str, Any]) -> dict[str, Any]:
    group_rows: list[dict[str, Any]] = []
    for index, group in enumerate(getattr(result, "groups", ()) or ()):
        diagnostics = getattr(group, "diagnostics", None)
        before = np.asarray(getattr(group, "similarities_before", []), dtype=float)
        after = np.asarray(getattr(group, "similarities_after", []), dtype=float)
        group_rows.append(
            {
                "group_index": int(index),
                "reference_index": getattr(group, "reference_index", None),
                "parameters_shape": list(np.asarray(getattr(group, "parameters", [])).shape),
                "coefficients": np.asarray(getattr(group, "coefficients", [])).tolist(),
                "similarity_before_mean": _finite_mean(before),
                "similarity_after_mean": _finite_mean(after),
                "initial_objective": None if diagnostics is None else float(getattr(diagnostics, "initial_objective", np.nan)),
                "final_objective": None if diagnostics is None else float(getattr(diagnostics, "final_objective", np.nan)),
                "iterations": None if diagnostics is None else int(getattr(diagnostics, "iterations", 0)),
                "message": "" if diagnostics is None else str(getattr(diagnostics, "message", "")),
            }
        )
    return _metadata_to_jsonable(
        {
            "kind": "mrax_linear_drift",
            "metadata": dict(metadata or {}),
            "dimensions": list(getattr(result, "dimensions", ()) or ()),
            "group_dimensions": list(getattr(result, "group_dimensions", ()) or ()),
            "group_indices": [list(values) for values in (getattr(result, "group_indices", ()) or ())],
            "corrected_shape": list(np.asarray(getattr(result, "corrected", [])).shape),
            "groups": group_rows,
        }
    )


def _failed_linear_drift_groups(result: Any) -> list[str]:
    out: list[str] = []
    group_indices = list(getattr(result, "group_indices", ()) or ())
    for index, group in enumerate(getattr(result, "groups", ()) or ()):
        diagnostics = getattr(group, "diagnostics", None)
        if diagnostics is None:
            continue
        success = bool(getattr(diagnostics, "success", False))
        final_objective = float(getattr(diagnostics, "final_objective", np.nan))
        if success and np.isfinite(final_objective):
            continue
        group_index = group_indices[index] if index < len(group_indices) else ()
        out.append(
            "group "
            f"{index}"
            f"{'' if not group_index else f' indices={tuple(group_index)}'}: "
            f"{getattr(diagnostics, 'message', '')}"
        )
    return out


def _finite_mean(values: np.ndarray) -> float | None:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None
    return float(np.mean(finite))


def _copy_roi_list(rois: Sequence[ROI]) -> list[ROI]:
    return [
        ROI(
            roi.name,
            np.asarray(roi.mask).copy(),
            dims=None if roi.dims is None else tuple(roi.dims),
            metadata=dict(roi.metadata),
        )
        for roi in rois
    ]


def _registration_roi_union(
    rois: Sequence[ROI],
    shared_rois: Sequence[ROI],
) -> list[ROI]:
    by_name: dict[str, ROI] = {}
    for roi in list(shared_rois) + list(rois):
        name = str(getattr(roi, "name", "") or "")
        if name:
            by_name[name] = roi
    return list(by_name.values())


def _registration_roi_choice_items(
    rois: Sequence[ROI],
    shared_rois: Sequence[ROI],
    *,
    include_none: bool = False,
) -> list[tuple[str, str | None]]:
    items: list[tuple[str, str | None]] = [("None", None)] if include_none else []
    items.extend((roi.name, roi.name) for roi in _registration_roi_union(rois, shared_rois))
    return items


def _registration_roi_by_name(
    name: str | None,
    rois: Sequence[ROI],
    shared_rois: Sequence[ROI],
) -> ROI | None:
    target = str(name or "").strip()
    if not target:
        return None
    for roi in _registration_roi_union(rois, shared_rois):
        if roi.name == target:
            return roi
    return None


def _unique_roi_name(rois: list[ROI] | tuple[ROI, ...], prefix: str) -> str:
    existing = {roi.name for roi in rois}
    idx = 1
    while f"{prefix}_{idx}" in existing:
        idx += 1
    return f"{prefix}_{idx}"


def _qt_enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def _qt_key_matches(QtCore, key: Any, *names: str) -> bool:
    for name in names:
        candidate = getattr(QtCore.Qt, name, None)
        if candidate is None and hasattr(QtCore.Qt, "Key"):
            candidate = getattr(QtCore.Qt.Key, name, None)
        if candidate is None:
            continue
        if key == candidate or _qt_enum_value(key) == _qt_enum_value(candidate):
            return True
    return False


def _qt_mouse_button(QtCore, name: str):
    candidate = getattr(QtCore.Qt, name, None)
    if candidate is None and hasattr(QtCore.Qt, "MouseButton"):
        candidate = getattr(QtCore.Qt.MouseButton, name, None)
    return candidate


def _regular_polygon_points(
    center: tuple[float, float],
    radius: float,
    sides: int,
    *,
    angle: float = 0.0,
) -> np.ndarray:
    n = max(3, int(sides))
    cx, cy = float(center[0]), float(center[1])
    angles = np.linspace(-np.pi / 2.0, 3.0 * np.pi / 2.0, n, endpoint=False) + np.deg2rad(float(angle))
    return np.column_stack([cx + float(radius) * np.cos(angles), cy + float(radius) * np.sin(angles)])


def _brush_mask(shape: tuple[int, int], center_yx: tuple[int, int], radius: int | float) -> np.ndarray:
    h, w = int(shape[0]), int(shape[1])
    y, x = int(center_yx[0]), int(center_yx[1])
    r = max(0.0, float(radius))
    out = np.zeros((h, w), dtype=bool)
    extent = int(np.ceil(r))
    y0, y1 = max(0, y - extent), min(h, y + extent + 1)
    x0, x1 = max(0, x - extent), min(w, x + extent + 1)
    if y1 <= y0 or x1 <= x0:
        return out
    yy, xx = np.ogrid[y0:y1, x0:x1]
    out[y0:y1, x0:x1] = (yy - y) ** 2 + (xx - x) ** 2 <= r**2
    return out


def _brush_radius_from_size(size: int | float) -> float:
    return max(0.0, (float(size) - 1.0) * 0.5)


def _rotate_points(points: Any, angle_degrees: float, *, origin: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if arr.size == 0:
        return arr.reshape((-1, 2))
    theta = np.deg2rad(float(angle_degrees))
    c, s = float(np.cos(theta)), float(np.sin(theta))
    origin_arr = np.asarray(origin, dtype=float)
    rel = arr.reshape((-1, 2)) - origin_arr
    rotated = np.column_stack([rel[:, 0] * c - rel[:, 1] * s, rel[:, 0] * s + rel[:, 1] * c])
    return rotated + origin_arr


def _transform_points(points: Any, *, pos: tuple[float, float] = (0.0, 0.0), angle: float = 0.0) -> np.ndarray:
    arr = np.asarray(points, dtype=float).reshape((-1, 2))
    rotated = _rotate_points(arr, angle, origin=(0.0, 0.0))
    return rotated + np.asarray(pos, dtype=float)


def _shift_array_zero(arr: np.ndarray, shifts: Mapping[int, int]) -> np.ndarray:
    out = np.zeros_like(arr)
    src: list[slice] = [slice(None)] * arr.ndim
    dst: list[slice] = [slice(None)] * arr.ndim
    for axis, shift in shifts.items():
        axis_i = int(axis)
        amount = int(shift)
        size = int(arr.shape[axis_i])
        if amount == 0:
            continue
        if abs(amount) >= size:
            return out
        if amount > 0:
            src[axis_i] = slice(0, size - amount)
            dst[axis_i] = slice(amount, size)
        else:
            src[axis_i] = slice(-amount, size)
            dst[axis_i] = slice(0, size + amount)
    out[tuple(dst)] = arr[tuple(src)]
    return out


def _shift_spatial_mask_visual(
    mask: Any,
    spatial_dims: tuple[str, ...],
    *,
    dx: int = 0,
    dy: int = 0,
    dz: int = 0,
) -> np.ndarray:
    arr = np.asarray(mask)
    shifts: dict[int, int] = {}
    if "y" in spatial_dims:
        shifts[spatial_dims.index("y")] = int(dx)
    if "x" in spatial_dims:
        shifts[spatial_dims.index("x")] = int(dy)
    if "z" in spatial_dims:
        shifts[spatial_dims.index("z")] = int(dz)
    return _shift_array_zero(arr, shifts)


def _visible_roi_slice_masks_for_display(
    rois: Sequence[ROI],
    *,
    display_shape: tuple[int, int],
    indices: Mapping[str, int] | None = None,
) -> list[tuple[ROI, np.ndarray]]:
    shape = tuple(int(v) for v in display_shape)
    index_map = dict(indices or {})
    out: list[tuple[ROI, np.ndarray]] = []
    for roi in rois:
        metadata_indices = dict((roi.metadata or {}).get("slice_indices") or {})
        if "z" in metadata_indices and "z" in index_map:
            try:
                if int(metadata_indices["z"]) != int(index_map["z"]):
                    continue
            except Exception:
                _LOG.debug("Ignored optional GUI operation failure", exc_info=True)
        try:
            mask = np.asarray(
                mask_slice_for_display(
                    roi.mask,
                    dims=roi.dims,
                    indices=index_map,
                ),
                dtype=bool,
            )
        except Exception:
            continue
        if tuple(mask.shape) == shape and np.any(mask):
            out.append((roi, mask))
    return out


def _combine_visible_roi_slice_masks(visible: Sequence[tuple[ROI, np.ndarray]]) -> tuple[np.ndarray, list[str]]:
    if not visible:
        return np.zeros((0, 0), dtype=bool), []
    combined = np.zeros_like(np.asarray(visible[0][1], dtype=bool), dtype=bool)
    source_names: list[str] = []
    for roi, mask in visible:
        arr = np.asarray(mask, dtype=bool)
        if tuple(arr.shape) != tuple(combined.shape):
            continue
        combined |= arr
        source_names.append(roi.name)
    return combined, source_names


def _roi_is_label_mask(roi: ROI | None) -> bool:
    if roi is None:
        return False
    arr = np.asarray(roi.mask)
    metadata = dict(roi.metadata or {})
    if str(metadata.get("mask_type") or "").lower() == "label":
        return True
    if str(metadata.get("shape") or "").lower() == "label_mask":
        return True
    return arr.dtype != bool and bool(np.any(arr != 0))


def _label_rows_for_roi(roi: ROI | None) -> list[dict[str, Any]]:
    if not _roi_is_label_mask(roi):
        return []
    return _label_rows_from_mask(np.asarray(roi.mask), dict(roi.metadata or {}))


def _label_rows_from_mask(mask: Any, metadata: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    arr = np.asarray(mask)
    if arr.size == 0:
        return []
    values: list[int] = []
    for value in np.unique(arr):
        try:
            label = int(value)
        except Exception:
            continue
        if label != 0 and np.isfinite(float(value)):
            values.append(label)
    values = sorted(set(values))
    if not values:
        return []

    label_names: dict[int, list[str]] = {}
    for row in list(dict(metadata or {}).get("labels") or []):
        if not isinstance(row, Mapping):
            continue
        try:
            label = int(row.get("label"))
        except Exception:
            continue
        name = str(row.get("name") or f"label {label}")
        label_names.setdefault(label, [])
        if name not in label_names[label]:
            label_names[label].append(name)

    rows: list[dict[str, Any]] = []
    for label in values:
        names = label_names.get(label) or [f"label {label}"]
        rows.append(
            {
                "label": int(label),
                "name": " + ".join(names),
                "voxel_count": int(np.count_nonzero(arr == label)),
            }
        )
    return rows


def _label_mask_from_visible_roi_slice_masks(
    visible: Sequence[tuple[ROI, np.ndarray]],
    *,
    label_values: Mapping[str, int] | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if not visible:
        return np.zeros((0, 0), dtype=np.uint16), []
    shape = tuple(int(v) for v in np.asarray(visible[0][1]).shape)
    values = {
        roi.name: max(1, int((label_values or {}).get(roi.name, idx)))
        for idx, (roi, _mask) in enumerate(visible, start=1)
    }
    max_label = max(values.values(), default=0)
    dtype = np.uint16 if max_label <= np.iinfo(np.uint16).max else np.int32
    label_mask = np.zeros(shape, dtype=dtype)
    labels: list[dict[str, Any]] = []
    for idx, (roi, mask) in enumerate(visible, start=1):
        arr = np.asarray(mask) != 0
        if tuple(arr.shape) != shape:
            continue
        if not np.any(arr):
            continue
        label = values.get(roi.name, idx)
        label_mask[arr] = label
        labels.append(
            {
                "label": int(label),
                "name": roi.name,
                "voxel_count": int(np.count_nonzero(arr)),
                "metadata": dict(roi.metadata),
            }
        )
    return label_mask, labels


def _delete_mask_slice(
    mask: Any,
    *,
    dims: Sequence[str],
    indices: Mapping[str, int] | None = None,
    plane_dims: Sequence[str] = ("x", "y"),
) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.ndim <= 2:
        return np.zeros(arr.shape, dtype=arr.dtype)
    dim_names = tuple(str(dim) for dim in dims)
    if len(dim_names) != arr.ndim:
        dim_names = tuple(dim_names[: arr.ndim])
    plane = tuple(str(dim) for dim in plane_dims if str(dim) in dim_names)
    if len(plane) != 2:
        plane = tuple(dim_names[:2])
    plane_shape = tuple(arr.shape[dim_names.index(dim)] for dim in plane)
    empty = np.zeros(plane_shape, dtype=arr.dtype)
    return replace_mask_slice(arr, empty, dims=dim_names, indices=indices, plane_dims=plane)


def _source_z_index(roi: ROI, *, default: int = 0) -> int:
    metadata = dict(roi.metadata or {})
    indices = metadata.get("slice_indices") or {}
    try:
        return int(dict(indices).get("z", default))
    except Exception:
        return int(default)


def _rasterized_roi_metadata(
    metadata: Mapping[str, Any] | None,
    operation: str,
    *,
    brush_radius: int | None = None,
) -> dict[str, Any]:
    meta = dict(metadata or {})
    geometry_keys = ("shape", "bounds", "points", "angle", "shape_variant", "sides")
    geometry = {key: meta.get(key) for key in geometry_keys if key in meta and meta.get(key) is not None}
    if geometry and str(geometry.get("shape", "mask")) != "mask":
        meta.setdefault("consolidated_from", geometry)
    for key in ("bounds", "points", "angle", "sides"):
        meta.pop(key, None)
    meta["shape"] = "mask"
    meta["last_edit_operation"] = str(operation)
    meta["generated_by"] = f"mrax.gui.{operation}"
    if brush_radius is not None:
        meta["brush_radius"] = int(brush_radius)
    return meta


def _sanitize_polygon_points(points: Any) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        return np.empty((0, 2), dtype=float)
    arr = arr[np.all(np.isfinite(arr), axis=1)]
    if arr.shape[0] <= 1:
        return arr
    keep = [arr[0]]
    for point in arr[1:]:
        if np.linalg.norm(point - keep[-1]) > 1e-9:
            keep.append(point)
    arr = np.asarray(keep, dtype=float)
    if arr.shape[0] > 2 and np.linalg.norm(arr[0] - arr[-1]) <= 1e-9:
        arr = arr[:-1]
    return arr


def _static_roi_outline_points(metadata: Mapping[str, Any], display_mask: np.ndarray) -> np.ndarray | None:
    shape = str(metadata.get("shape", "mask"))
    angle = float(metadata.get("angle", 0.0) or 0.0)
    if shape == "polygon":
        points = _sanitize_polygon_points(metadata.get("points"))
        if points.shape[0] < 3:
            return None
        return np.vstack([points, points[0]])
    if shape == "rectangle":
        bounds = metadata.get("bounds", _mask_bounds(display_mask))
        x, y, w, h = [float(v) for v in bounds]
        corners = np.asarray([[0.0, 0.0], [w, 0.0], [w, h], [0.0, h], [0.0, 0.0]], dtype=float)
        return _transform_points(corners, pos=(x, y), angle=angle)
    if shape == "ellipse":
        bounds = metadata.get("bounds", _mask_bounds(display_mask))
        x, y, w, h = [float(v) for v in bounds]
        theta = np.linspace(0.0, 2.0 * np.pi, 96)
        local = np.column_stack([w / 2.0 + (w / 2.0) * np.cos(theta), h / 2.0 + (h / 2.0) * np.sin(theta)])
        return _transform_points(local, pos=(x, y), angle=angle)
    return None


def _geometry_mask_from_metadata(metadata: Mapping[str, Any] | None, display_shape: Sequence[int]) -> np.ndarray | None:
    shape_tuple = tuple(int(v) for v in display_shape)
    if len(shape_tuple) != 2:
        return None
    meta = dict(metadata or {})
    shape = str(meta.get("shape", "mask"))
    angle = float(meta.get("angle", 0.0) or 0.0)
    try:
        if shape == "rectangle":
            return _rect_mask(shape_tuple, meta.get("bounds", _mask_bounds(np.zeros(shape_tuple, dtype=bool))), angle=angle)
        if shape == "ellipse":
            return _ellipse_mask(shape_tuple, meta.get("bounds", _mask_bounds(np.zeros(shape_tuple, dtype=bool))), angle=angle)
        if shape == "polygon":
            points = _sanitize_polygon_points(meta.get("points"))
            if points.shape[0] < 3:
                return None
            return _polygon_mask(shape_tuple, points)
    except Exception:
        return None
    return None


def _roi_overlay_mask_for_display(
    roi: ROI,
    *,
    display_shape: Sequence[int],
    indices: Mapping[str, int] | None = None,
    editable: bool = False,
) -> np.ndarray | None:
    shape = tuple(int(v) for v in display_shape)
    if len(shape) != 2:
        return None
    try:
        mask = np.asarray(
            mask_slice_for_display(
                roi.mask,
                dims=roi.dims,
                indices=dict(indices or {}),
            ),
            dtype=bool,
        )
    except Exception:
        mask = None
    if mask is not None and tuple(mask.shape) == shape:
        return mask if np.any(mask) else None
    if not editable:
        return None
    geometry_mask = _geometry_mask_from_metadata(roi.metadata, shape)
    if geometry_mask is None or not np.any(geometry_mask):
        return None
    return np.asarray(geometry_mask, dtype=bool)


def _simplify_polyline(points: np.ndarray, *, tolerance: float = 0.0, closed: bool = False) -> np.ndarray:
    arr = _sanitize_polygon_points(points)
    if arr.shape[0] <= 1:
        return arr
    if arr.shape[0] <= 2 or tolerance <= 0:
        return arr
    simplified = _rdp_simplify(arr, float(tolerance))
    if closed and simplified.shape[0] > 1 and np.linalg.norm(simplified[0] - simplified[-1]) <= float(tolerance):
        simplified = simplified[:-1]
    return simplified


def _rdp_simplify(points: np.ndarray, tolerance: float) -> np.ndarray:
    if points.shape[0] <= 2:
        return points
    start = points[0]
    end = points[-1]
    line = end - start
    norm = float(np.linalg.norm(line))
    if norm <= 1e-12:
        distances = np.linalg.norm(points - start, axis=1)
    else:
        rel = points - start
        distances = np.abs(line[0] * rel[:, 1] - line[1] * rel[:, 0]) / norm
    idx = int(np.argmax(distances))
    if float(distances[idx]) <= float(tolerance):
        return np.vstack([start, end])
    left = _rdp_simplify(points[: idx + 1], tolerance)
    right = _rdp_simplify(points[idx:], tolerance)
    return np.vstack([left[:-1], right])


def _closed_curve_length(points: Any) -> float:
    pts = _sanitize_polygon_points(points)
    if pts.shape[0] < 2:
        return 0.0
    closed = np.vstack([pts, pts[0]])
    return float(np.sum(np.linalg.norm(np.diff(closed, axis=0), axis=1)))


def _moving_average_closed_curve(points: Any, *, radius: int = 2) -> np.ndarray:
    pts = _sanitize_polygon_points(points)
    if pts.shape[0] < 3:
        return pts
    r = max(1, int(radius))
    smoothed = []
    count = pts.shape[0]
    for idx in range(count):
        window = [pts[(idx + offset) % count] for offset in range(-r, r + 1)]
        smoothed.append(np.mean(window, axis=0))
    return np.asarray(smoothed, dtype=float)


def _resample_closed_polyline(points: Any, count: int) -> np.ndarray:
    pts = _sanitize_polygon_points(points)
    target_count = max(3, int(count))
    if pts.shape[0] < 3:
        return pts
    closed = np.vstack([pts, pts[0]])
    distances = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    total = float(np.sum(distances))
    if total <= 1e-12:
        return pts
    cumulative = np.concatenate([[0.0], np.cumsum(distances)])
    samples = np.linspace(0.0, total, target_count, endpoint=False)
    out = []
    for value in samples:
        idx = int(np.searchsorted(cumulative, value, side="right") - 1)
        idx = max(0, min(idx, distances.size - 1))
        segment_length = float(distances[idx])
        ratio = 0.0 if segment_length <= 1e-12 else (float(value) - float(cumulative[idx])) / segment_length
        out.append(closed[idx] + ratio * (closed[idx + 1] - closed[idx]))
    return np.asarray(out, dtype=float)


def _smooth_lasso_support_points(
    points: Any,
    *,
    image: Any | None = None,
    snap: bool = False,
    snap_radius: int = 4,
    snap_strength: float = 0.25,
    support_points: int | None = None,
    spline_kind: str = "cubic",
) -> np.ndarray:
    pts = _sanitize_polygon_points(points)
    if pts.shape[0] < 3:
        return pts
    if snap and image is not None:
        snapped = _snap_polygon_points_to_gradient(pts, image, radius=snap_radius, strength=snap_strength)
        if snapped.shape[0] >= 3:
            pts = snapped
    length = _closed_curve_length(pts)
    target_count = max(3, int(support_points)) if support_points is not None else None
    sample_count = int(
        max(
            48,
            min(
                1024,
                max(
                    (target_count or 0) * 4,
                    pts.shape[0] * 2,
                    length / 1.5 if length > 0 else pts.shape[0],
                ),
            ),
        )
    )
    kind = str(spline_kind or "cubic").lower()
    if kind == "linear":
        curve = _resample_closed_polyline(pts, sample_count)
    else:
        try:
            from scipy import interpolate

            degree = min(3, pts.shape[0] - 1)
            smoothness = max(0.0, pts.shape[0] * 0.08)
            tck, _u = interpolate.splprep([pts[:, 0], pts[:, 1]], s=smoothness, per=True, k=degree)
            u_new = np.linspace(0.0, 1.0, sample_count, endpoint=False)
            x_new, y_new = interpolate.splev(u_new, tck)
            curve = np.column_stack([x_new, y_new])
        except Exception:
            curve = _moving_average_closed_curve(pts, radius=2)
    if target_count is not None:
        support = _resample_closed_polyline(curve, target_count)
        return support if support.shape[0] >= 3 else pts
    tolerance = max(0.75, min(5.0, length / 180.0 if length > 0 else 1.0))
    closed_curve = np.vstack([curve, curve[0]])
    support = _simplify_polyline(closed_curve, tolerance=tolerance, closed=True)
    if support.shape[0] < 3:
        support = _simplify_polyline(closed_curve, tolerance=max(0.35, tolerance * 0.4), closed=True)
    return support if support.shape[0] >= 3 else pts


def _roi_plane_image(image: Any) -> np.ndarray:
    arr = np.asarray(image)
    if np.iscomplexobj(arr):
        arr = np.abs(arr)
    arr = np.asarray(arr, dtype=float)
    if arr.ndim > 2:
        arr = np.asarray(arr.reshape((-1,) + arr.shape[-2:])[0], dtype=float)
    return arr


def _robust_intensity_span(image: Any) -> float:
    arr = _roi_plane_image(image)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0
    if finite.size >= 16:
        lo, hi = np.nanpercentile(finite, [1.0, 99.0])
    else:
        lo, hi = np.nanmin(finite), np.nanmax(finite)
    span = float(hi - lo)
    if not np.isfinite(span) or span <= 0:
        span = float(np.nanmax(finite) - np.nanmin(finite))
    return span if np.isfinite(span) and span > 0 else 0.0


def _otsu_threshold(values: Any, *, bins: int = 256) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return lo
    counts, edges = np.histogram(arr, bins=max(2, int(bins)), range=(lo, hi))
    centers = (edges[:-1] + edges[1:]) / 2.0
    total = counts.sum()
    if total <= 0:
        return float(np.mean(arr))
    weight_bg = np.cumsum(counts).astype(float)
    weight_fg = float(total) - weight_bg
    sum_bg = np.cumsum(counts * centers)
    sum_total = float(sum_bg[-1])
    valid = (weight_bg > 0) & (weight_fg > 0)
    if not np.any(valid):
        return float(np.mean(arr))
    mean_bg = np.zeros_like(centers, dtype=float)
    mean_fg = np.zeros_like(centers, dtype=float)
    mean_bg[valid] = sum_bg[valid] / weight_bg[valid]
    mean_fg[valid] = (sum_total - sum_bg[valid]) / weight_fg[valid]
    score = np.zeros_like(centers, dtype=float)
    score[valid] = weight_bg[valid] * weight_fg[valid] * (mean_bg[valid] - mean_fg[valid]) ** 2
    return float(centers[int(np.argmax(score))])


def _fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask, dtype=bool)
    try:
        from scipy import ndimage

        return np.asarray(ndimage.binary_fill_holes(arr), dtype=bool)
    except Exception:
        return arr


def _largest_component_mask(mask: Any) -> np.ndarray:
    arr = np.asarray(mask, dtype=bool)
    if arr.ndim != 2 or not np.any(arr):
        return np.zeros(arr.shape, dtype=bool)
    try:
        from scipy import ndimage

        labels, count = ndimage.label(arr, structure=np.ones((3, 3), dtype=bool))
        if int(count) == 0:
            return np.zeros(arr.shape, dtype=bool)
        sizes = np.bincount(labels.ravel())
        if sizes.size <= 1:
            return np.zeros(arr.shape, dtype=bool)
        sizes[0] = 0
        return labels == int(np.argmax(sizes))
    except Exception:
        visited = np.zeros(arr.shape, dtype=bool)
        best: list[tuple[int, int]] = []
        for y, x in np.argwhere(arr):
            y_i, x_i = int(y), int(x)
            if visited[y_i, x_i]:
                continue
            component = _component_points(arr, (y_i, x_i), visited=visited)
            if len(component) > len(best):
                best = component
        out = np.zeros(arr.shape, dtype=bool)
        for y, x in best:
            out[y, x] = True
        return out


def _component_points(mask: np.ndarray, seed_yx: tuple[int, int], *, visited: np.ndarray | None = None) -> list[tuple[int, int]]:
    arr = np.asarray(mask, dtype=bool)
    h, w = arr.shape
    sy, sx = int(seed_yx[0]), int(seed_yx[1])
    if not (0 <= sy < h and 0 <= sx < w) or not arr[sy, sx]:
        return []
    seen = np.zeros(arr.shape, dtype=bool) if visited is None else visited
    points: list[tuple[int, int]] = []
    queue: deque[tuple[int, int]] = deque([(sy, sx)])
    seen[sy, sx] = True
    while queue:
        y, x = queue.popleft()
        points.append((y, x))
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                yy, xx = y + dy, x + dx
                if 0 <= yy < h and 0 <= xx < w and arr[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    queue.append((yy, xx))
    return points


def _connected_component_from_seed(mask: Any, seed_yx: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(mask, dtype=bool)
    out = np.zeros(arr.shape, dtype=bool)
    if arr.ndim != 2:
        return out
    sy, sx = int(seed_yx[0]), int(seed_yx[1])
    if not (0 <= sy < arr.shape[0] and 0 <= sx < arr.shape[1]) or not arr[sy, sx]:
        return out
    try:
        from scipy import ndimage

        labels, _count = ndimage.label(arr, structure=np.ones((3, 3), dtype=bool))
        label = int(labels[sy, sx])
        return labels == label if label else out
    except Exception:
        for y, x in _component_points(arr, (sy, sx)):
            out[y, x] = True
        return out


def _seeded_region_mask(
    image: Any,
    seed_yx: tuple[int, int],
    *,
    tolerance_fraction: float = 0.08,
) -> tuple[np.ndarray, dict[str, Any]]:
    arr = _roi_plane_image(image)
    if arr.ndim != 2:
        return np.zeros(arr.shape[-2:], dtype=bool), {"auto_method": "seeded_region"}
    sy, sx = int(seed_yx[0]), int(seed_yx[1])
    h, w = arr.shape
    if not (0 <= sy < h and 0 <= sx < w) or not np.isfinite(arr[sy, sx]):
        return np.zeros((h, w), dtype=bool), {"auto_method": "seeded_region"}
    span = _robust_intensity_span(arr)
    tolerance = max(float(tolerance_fraction) * span, 1e-12)
    seed_value = float(arr[sy, sx])
    candidate = np.isfinite(arr) & (np.abs(arr - seed_value) <= tolerance)
    region = _connected_component_from_seed(candidate, (sy, sx))
    return region, {
        "auto_method": "seeded_region",
        "seed": [int(sy), int(sx)],
        "seed_value": seed_value,
        "tolerance_fraction": float(tolerance_fraction),
        "tolerance": float(tolerance),
    }


def _auto_threshold_mask(image: Any) -> tuple[np.ndarray, dict[str, Any]]:
    arr = _roi_plane_image(image)
    if arr.ndim != 2:
        return np.zeros(arr.shape[-2:], dtype=bool), {"auto_method": "otsu_largest_component"}
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros(arr.shape, dtype=bool), {"auto_method": "otsu_largest_component"}
    threshold = _otsu_threshold(arr[finite])
    if not np.isfinite(threshold):
        return np.zeros(arr.shape, dtype=bool), {"auto_method": "otsu_largest_component"}
    candidate = finite & (arr >= float(threshold))
    component = _largest_component_mask(candidate)
    component = _fill_mask_holes(component)
    return component, {
        "auto_method": "otsu_largest_component",
        "threshold": float(threshold),
        "voxel_count": int(np.count_nonzero(component)),
    }


def _gradient_magnitude(image: Any) -> np.ndarray:
    arr = _roi_plane_image(image)
    if arr.ndim != 2:
        return np.zeros(arr.shape[-2:], dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    gy, gx = np.gradient(arr)
    return np.hypot(gx, gy)


def _snap_polygon_points_to_gradient(points: Any, image: Any, *, radius: int = 4, strength: float = 0.35) -> np.ndarray:
    pts = np.asarray(points, dtype=float).reshape((-1, 2))
    if pts.shape[0] == 0:
        return pts
    gradient = _gradient_magnitude(image)
    h, w = gradient.shape
    r = max(1, int(radius))
    assist = float(np.clip(strength, 0.0, 1.0))
    snapped = []
    for x, y in pts:
        cx = int(round(float(x)))
        cy = int(round(float(y)))
        x0, x1 = max(0, cx - r), min(w, cx + r + 1)
        y0, y1 = max(0, cy - r), min(h, cy + r + 1)
        if x1 <= x0 or y1 <= y0:
            snapped.append([float(x), float(y)])
            continue
        window = gradient[y0:y1, x0:x1]
        if not np.any(np.isfinite(window)):
            snapped.append([float(x), float(y)])
            continue
        yy, xx = np.unravel_index(int(np.nanargmax(window)), window.shape)
        target_x = float(x0 + xx)
        target_y = float(y0 + yy)
        snapped.append([float(x) + assist * (target_x - float(x)), float(y) + assist * (target_y - float(y))])
    return np.asarray(snapped, dtype=float)


def _detect_import_vendor(root: Path) -> str:
    if _looks_like_bruker_study(root):
        return "Bruker"
    if _looks_like_dicom_study(root):
        return "DICOM"
    return "Unknown"


def _looks_like_bruker_study(root: Path) -> bool:
    if not root.is_dir():
        return False
    for entry in root.iterdir():
        if entry.is_dir() and entry.name.isdigit() and (entry / "method").is_file():
            return True
    return False


def _looks_like_dicom_study(root: Path) -> bool:
    if not root.is_dir():
        return False
    try:
        from mrax.io.dicom import discover_dicom_series

        return bool(discover_dicom_series(root, recursive=True))
    except (OSError, ValueError):
        return False


def _detect_bruker_import_summary(root: Path) -> list[dict[str, Any]]:
    try:
        from mrax.io.bruker_import import detect_type, parse_jcamp
    except Exception:
        return []

    rows: list[dict[str, Any]] = []
    for scan_dir in sorted(
        (entry for entry in root.iterdir() if entry.is_dir() and entry.name.isdigit()),
        key=lambda path: (int(path.name), path.name),
    ):
        method_path = scan_dir / "method"
        pdata_dir = scan_dir / "pdata" / "1"
        visu_path = pdata_dir / "visu_pars"
        if not method_path.is_file():
            continue
        try:
            method = parse_jcamp(method_path)
        except Exception:
            method = {}
        try:
            visu = parse_jcamp(visu_path) if visu_path.is_file() else {}
        except Exception:
            visu = {}
        frame_count = _detect_bruker_frame_count(visu, method)
        try:
            exp_type = detect_type(visu, method, frame_count=frame_count)
        except Exception:
            exp_type = "anatomical"
        rows.append(
            {
                "scan": scan_dir.name,
                "method": method.get("Method") or method.get("PVM_Method") or "",
                "protocol": visu.get("VisuAcquisitionProtocol") or method.get("ACQ_scan_name") or "",
                "frame_count": frame_count,
                "detected_type": exp_type,
                "detected_model": normalize_model_key(exp_type) or "identity",
                "field_strength_mhz": _detect_bruker_field_strength_mhz(method, visu, scan_dir),
            }
        )
    return rows


def _detect_bruker_frame_count(visu: dict[str, Any], method: dict[str, Any]) -> int | None:
    for key in ("VisuCoreFrameCount", "VisuFGOrderDescDim", "PVM_NRepetitions"):
        value = visu.get(key, method.get(key))
        numeric = _numeric_values(value)
        if numeric:
            count = int(round(numeric[0]))
            if count > 0:
                return count
    return None


def _detect_bruker_field_strength_mhz(method: dict[str, Any], visu: dict[str, Any], scan_dir: Path) -> float | None:
    values = []
    for key in ("PVM_FrqWork", "BF1", "SFO1", "ACQ_imaging_frequency"):
        values.extend(_numeric_values(method.get(key)))
        values.extend(_numeric_values(visu.get(key)))
    acqp_path = scan_dir / "acqp"
    if acqp_path.is_file():
        try:
            from mrax.io.bruker_import import parse_jcamp

            acqp = parse_jcamp(acqp_path)
            for key in ("BF1", "SFO1", "ACQ_imaging_frequency"):
                values.extend(_numeric_values(acqp.get(key)))
        except Exception:
            _LOG.debug("Ignored optional GUI operation failure", exc_info=True)
    for value in values:
        if 1.0 <= float(value) <= 2000.0:
            return float(value)
    return None


def _numeric_values(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, (int, float, np.number)):
        return [float(value)]
    if isinstance(value, np.ndarray):
        return [float(v) for v in value.reshape((-1,)) if _is_number(v)]
    if isinstance(value, (list, tuple)):
        out: list[float] = []
        for item in value:
            out.extend(_numeric_values(item))
        return out
    try:
        return [float(value)]
    except Exception:
        return []


def _is_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except Exception:
        return False


def _known_exp_types(initial: str) -> list[str]:
    values = [
        initial or "anatomical",
        "anatomical",
        "t1_ir",
        "t1_vtr",
        "t1_vfa",
        "t2_se",
        "t2star_se",
        "fair_asl",
        "dti",
        "dki",
        "wasabi",
        "multipool_lorentzian",
        "ab_mt_n",
        "zaiss_full",
    ]
    unique: list[str] = []
    for value in values:
        if value and value not in unique:
            unique.append(value)
    return unique


def _sat_trans_reference_visible(model_key: str | None, exp_type: str | None) -> bool:
    return exp_type == "sat_trans" and normalize_model_key(model_key) in {
        "wasabi",
        "multipool_lorentzian",
        "ab_mt_n",
        "zaiss_full",
    }


def _ab_mt_prior_maps_visible(model_key: str | None) -> bool:
    return normalize_model_key(model_key) in {"ab_mt_n", "zaiss_full"}


def _pool_editor_visible(model_key: str | None) -> bool:
    return normalize_model_key(model_key) in {"ab_mt_n", "multipool_lorentzian", "zaiss_full"}


def _set_model_combo_from_type(combo, exp_type: str) -> None:
    key = normalize_model_key(exp_type) or "identity"
    for idx in range(combo.count()):
        if combo.itemData(idx) == key:
            combo.setCurrentIndex(idx)
            return


def _format_optional_float(value: Any) -> str:
    if value in ("", None):
        return ""
    try:
        return f"{float(value):.6g}"
    except Exception:
        return str(value)


def _parse_optional_float(value: str) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    return float(text)


def _format_csv(value: Any) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)
    return str(value)


def _parse_csv_values(value: str) -> list[int | str]:
    out: list[int | str] = []
    for part in str(value or "").split(","):
        text = part.strip()
        if not text:
            continue
        try:
            out.append(int(text))
        except ValueError:
            out.append(text)
    return out


def _parse_shape_text(value: str) -> tuple[int, ...]:
    shape: list[int] = []
    for part in str(value or "").replace("x", ",").replace("X", ",").split(","):
        text = part.strip()
        if not text:
            continue
        size = int(text)
        if size <= 0:
            raise ValueError("Shape dimensions must be positive")
        shape.append(size)
    return tuple(shape)


def _parse_dim_order(value: str) -> tuple[str, ...]:
    dims = []
    for part in str(value or "").replace(";", ",").split(","):
        dim = part.strip()
        if not dim:
            continue
        if dim in dims:
            raise ValueError(f"Duplicate dimension '{dim}' in dimension order")
        dims.append(dim)
    return tuple(dims)


def _default_dim_order(ndim: int, source: str = "") -> tuple[str, ...]:
    ndim = int(ndim)
    if ndim <= 0:
        return ("sample",)
    if ndim == 1:
        return ("sample",)
    if ndim == 2:
        return ("x", "y")
    if ndim == 3 and source == "nifti":
        return ("x", "y", "z")
    if ndim == 3:
        return ("frame", "x", "y")
    if ndim == 4:
        return ("frame", "z", "x", "y")
    return tuple([f"axis_{idx}" for idx in range(ndim - 2)] + ["x", "y"])


def _dims_from_order_and_shape(order: Sequence[str], shape: Sequence[int]) -> dict[str, int]:
    names = tuple(str(dim).strip() for dim in order if str(dim).strip())
    sizes = tuple(int(size) for size in shape)
    if len(names) != len(sizes):
        raise ValueError(
            f"Dimension order ({', '.join(names) or 'none'}) must have {len(sizes)} entries for shape {sizes}."
        )
    return {name: size for name, size in zip(names, sizes)}


def _import_file_preview_lines(path: Path | None, *, source: str) -> list[str]:
    lines = [f"source: {source}"]
    if path is None:
        lines.append("file: none selected")
        return lines
    path = Path(path)
    lines.append(f"file: {path}")
    lines.append(f"suffix: {''.join(path.suffixes) or '(none)'}")
    if path.exists():
        if path.is_file():
            lines.append(f"size: {path.stat().st_size} byte(s)")
        else:
            lines.append("size: selected path is not a file")
    else:
        lines.append("size: file does not exist")
    return lines


def _array_detail_lines(array: Any, *, label: str = "array") -> list[str]:
    arr = np.asarray(array)
    lines = [
        f"{label} shape: {arr.shape}",
        f"{label} dimensions: {arr.ndim}",
        f"{label} dtype: {arr.dtype}",
        f"values: {arr.size}",
        f"bytes: {arr.nbytes}",
    ]
    if arr.size == 0:
        lines.append("statistics: empty")
        return lines
    if np.issubdtype(arr.dtype, np.bool_):
        true_count = int(np.count_nonzero(arr))
        lines.append(f"true values: {true_count} / {arr.size} = {_format_fraction(true_count, int(arr.size))}")
        return lines
    if np.issubdtype(arr.dtype, np.integer):
        nonzero = int(np.count_nonzero(arr))
        lines.append(f"nonzero values: {nonzero} / {arr.size} = {_format_fraction(nonzero, int(arr.size))}")
        if arr.size <= 1_000_000:
            unique = np.unique(arr)
            preview = ", ".join(str(int(value)) for value in unique[:8])
            if unique.size > 8:
                preview += ", ..."
            lines.append(f"unique values: {unique.size}; preview: {preview}")
        else:
            lines.append("unique values: skipped for arrays larger than 1,000,000 values")
        lines.append(f"range: {int(np.min(arr))} to {int(np.max(arr))}")
        return lines
    if np.iscomplexobj(arr):
        magnitude = np.abs(arr)
        lines.append(f"magnitude range: {float(np.nanmin(magnitude)):.6g} to {float(np.nanmax(magnitude)):.6g}")
        lines.append(f"magnitude mean: {float(np.nanmean(magnitude)):.6g}")
        return lines
    if np.issubdtype(arr.dtype, np.number):
        finite = np.isfinite(arr)
        finite_count = int(np.count_nonzero(finite))
        lines.append(f"finite values: {finite_count} / {arr.size} = {_format_fraction(finite_count, int(arr.size))}")
        if finite_count:
            finite_values = arr[finite]
            lines.append(f"range: {float(np.min(finite_values)):.6g} to {float(np.max(finite_values)):.6g}")
            lines.append(f"mean: {float(np.mean(finite_values)):.6g}")
        return lines
    lines.append("statistics: non-numeric dtype")
    return lines


def _mapping_detail_lines(mapping: Mapping[str, Any], *, title: str = "metadata") -> list[str]:
    payload = dict(mapping or {})
    if not payload:
        return [f"{title}: none"]
    lines = [f"{title}: {len(payload)} entr{'y' if len(payload) == 1 else 'ies'}"]
    for key in sorted(payload):
        value = payload[key]
        text = json.dumps(_metadata_to_jsonable(value), sort_keys=True)
        if len(text) > 160:
            text = text[:157] + "..."
        lines.append(f"- {key}: {text}")
    return lines


def _observables_detail_lines(observables: Mapping[str, Quantity]) -> list[str]:
    if not observables:
        return ["observables: none"]
    lines = [f"observables: {len(observables)}"]
    for name in sorted(observables):
        quantity = observables[name]
        values = np.asarray(quantity.values)
        lines.append(f"- {name}: shape={values.shape}, dtype={values.dtype}, unit={quantity.si_unit}")
    return lines


def _roi_detail_lines(rois: Sequence[ROI]) -> list[str]:
    if not rois:
        return ["ROI count: 0"]
    lines = [f"ROI count: {len(rois)}"]
    for roi in list(rois)[:12]:
        arr = np.asarray(roi.mask)
        kind = "label mask" if _roi_is_label_mask(roi) else "binary mask"
        nonzero = int(np.count_nonzero(arr))
        dims = ", ".join(roi.dims or ()) or "(none)"
        lines.append(f"- {roi.name}")
        lines.append(f"  type: {kind}")
        lines.append(f"  shape: {arr.shape}")
        lines.append(f"  dtype: {arr.dtype}")
        lines.append(f"  dims: {dims}")
        lines.append(f"  nonzero: {nonzero} / {arr.size} = {_format_fraction(nonzero, int(arr.size))}")
        if _roi_is_label_mask(roi):
            if arr.size <= 1_000_000:
                labels = np.unique(arr)
                labels = labels[labels != 0]
                preview = ", ".join(str(int(value)) for value in labels[:8])
                if labels.size > 8:
                    preview += ", ..."
                lines.append(f"  labels: {labels.size}; preview: {preview or '(none)'}")
            else:
                lines.append("  labels: skipped for masks larger than 1,000,000 values")
    if len(rois) > 12:
        lines.append(f"... {len(rois) - 12} more ROI(s)")
    return lines


def _normalize_memory_order(memory_order: str | None) -> str:
    order = str(memory_order or "C").strip().upper()
    if order in {"C", "ROW", "ROW-MAJOR", "ROW_MAJOR"}:
        return "C"
    if order in {"F", "FORTRAN", "COLUMN", "COLUMN-MAJOR", "COLUMN_MAJOR"}:
        return "F"
    raise ValueError("Memory order must be C / row-major or Fortran / column-major")


def _dtype_with_endian(dtype: str, endian: str | None = "native") -> np.dtype:
    np_dtype = np.dtype(str(dtype or "float32"))
    endian_key = str(endian or "native").strip().lower()
    if np_dtype.itemsize <= 1 or endian_key in {"", "native", "=", "|"}:
        return np_dtype
    if endian_key in {"little", "little endian", "le", "<"}:
        return np_dtype.newbyteorder("<")
    if endian_key in {"big", "big endian", "be", ">"}:
        return np_dtype.newbyteorder(">")
    raise ValueError("Raw endian must be native, little endian, or big endian")


def _format_fraction(numerator: int, denominator: int) -> str:
    if int(denominator) == 0:
        return "n/a"
    return f"{float(numerator) / float(denominator):.3g}"


def _raw_import_diagnostic_lines(
    path: Path | None,
    *,
    dtype: str,
    shape: Sequence[int],
    endian: str = "native",
    byte_offset: int = 0,
    order: Sequence[str] = (),
) -> list[str]:
    lines = ["Raw import diagnostics"]
    target = tuple(int(size) for size in shape)
    dim_names = tuple(str(name).strip() for name in order if str(name).strip())
    if target:
        terms = []
        for idx, size in enumerate(target):
            label = dim_names[idx] if idx < len(dim_names) else f"dim{idx}"
            terms.append(f"{label}={int(size)}")
        expected_values = int(np.prod(target, dtype=np.int64))
        lines.append(f"shape product: {' x '.join(terms)} = {expected_values} value(s)")
    else:
        expected_values = None
        lines.append("shape product: unset")
    try:
        np_dtype = _dtype_with_endian(dtype, endian)
        itemsize = int(np_dtype.itemsize)
        lines.append(f"dtype: {np_dtype} ({itemsize} byte/value, endian={str(endian or 'native')})")
    except Exception as exc:
        lines.append(f"dtype mismatch: {exc}")
        return lines
    if expected_values is not None:
        expected_bytes = int(expected_values * itemsize)
        lines.append(f"expected bytes: {expected_bytes}")
    else:
        expected_bytes = None
    if path is None:
        lines.append("file bytes after offset: no file selected")
        lines.append("status: mismatch")
        return lines
    try:
        total_bytes = int(Path(path).stat().st_size)
    except Exception as exc:
        lines.append(f"file bytes after offset: unavailable ({exc})")
        lines.append("status: mismatch")
        return lines
    offset = max(0, int(byte_offset))
    available_bytes = max(0, total_bytes - offset)
    actual_values, leftover = divmod(available_bytes, itemsize)
    lines.append(f"file bytes: {total_bytes}; offset: {offset}; after offset: {available_bytes}")
    lines.append(f"whole values after offset: {actual_values}" + (f" (+{leftover} leftover byte(s))" if leftover else ""))
    if expected_bytes is None:
        lines.append("status: mismatch")
    elif available_bytes == expected_bytes:
        lines.append(f"byte coverage fraction: {available_bytes} / {expected_bytes} = 1")
        lines.append(f"value coverage fraction: {actual_values} / {expected_values} = 1")
        lines.append("status: match")
    else:
        lines.append(
            f"byte coverage fraction: {available_bytes} / {expected_bytes} = "
            f"{_format_fraction(available_bytes, expected_bytes)}"
        )
        lines.append(
            f"value coverage fraction: {actual_values} / {expected_values} = "
            f"{_format_fraction(actual_values, expected_values or 0)}"
        )
        lines.append("status: mismatch")
    return lines


def _reshape_import_array(array: Any, shape: Sequence[int], *, memory_order: str = "C") -> np.ndarray:
    arr = np.asarray(array)
    target = tuple(int(size) for size in shape)
    if not target:
        return arr
    if int(np.prod(target, dtype=np.int64)) != int(arr.size):
        raise ValueError(f"Cannot reshape array with {arr.size} value(s) to {target}.")
    return arr.reshape(target, order=_normalize_memory_order(memory_order))


def _load_raw_binary_array(
    path: Path,
    *,
    dtype: str,
    shape: Sequence[int],
    endian: str = "native",
    byte_offset: int = 0,
    memory_order: str = "C",
) -> np.ndarray:
    if path is None or not Path(path).is_file():
        raise FileNotFoundError("Choose a raw binary data file")
    target = tuple(int(size) for size in shape)
    if not target:
        raise ValueError("Raw binary import requires an explicit array shape")
    np_dtype = _dtype_with_endian(dtype, endian)
    with open(path, "rb") as handle:
        handle.seek(max(0, int(byte_offset)))
        arr = np.fromfile(handle, dtype=np_dtype)
    expected = int(np.prod(target, dtype=np.int64))
    if int(arr.size) != expected:
        raise ValueError(f"Raw file contains {arr.size} {np_dtype} value(s), expected {expected} for shape {target}.")
    return arr.reshape(target, order=_normalize_memory_order(memory_order))


def _load_numpy_array_file(
    path: Path,
    *,
    shape: Sequence[int] = (),
    npz_key: str | None = None,
    memory_order: str = "C",
) -> np.ndarray:
    if path is None or not Path(path).is_file():
        raise FileNotFoundError("Choose a NumPy .npy or .npz file")
    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        with loaded as npz:
            keys = list(npz.files)
            if not keys:
                raise ValueError(f"NPZ file {path} does not contain arrays")
            key = npz_key or keys[0]
            if key not in npz.files:
                raise KeyError(f"NPZ file {path} does not contain array key '{key}'. Available: {', '.join(keys)}")
            arr = np.asarray(npz[key])
    else:
        arr = np.asarray(loaded)
    return _reshape_import_array(arr, shape, memory_order=memory_order)


def _load_experiment_array_from_dialog(
    *,
    source: str,
    path: Path | None,
    shape: Sequence[int],
    dtype: str,
    endian: str = "native",
    byte_offset: int = 0,
    memory_order: str = "C",
    npz_key: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    source = str(source or "blank")
    metadata: dict[str, Any] = {"import_format": source}
    if source == "blank":
        target = tuple(int(size) for size in shape)
        if not target:
            raise ValueError("Blank image import requires an array shape")
        return np.zeros(target, dtype=np.float32), metadata
    if path is None:
        raise FileNotFoundError("Choose a data file")
    metadata["source_path"] = str(path)
    if source == "nifti":
        from mrax.io.nifti import load_nifti

        arr, affine = load_nifti(path)
        metadata["affine"] = _metadata_to_jsonable(np.asarray(affine))
        if shape:
            metadata["memory_order"] = _normalize_memory_order(memory_order)
        return _reshape_import_array(arr, shape, memory_order=memory_order), metadata
    if source == "numpy":
        if npz_key:
            metadata["npz_key"] = npz_key
        if shape:
            metadata["memory_order"] = _normalize_memory_order(memory_order)
        return _load_numpy_array_file(path, shape=shape, npz_key=npz_key, memory_order=memory_order), metadata
    if source == "raw":
        metadata.update(
            {
                "dtype": dtype,
                "endian": str(endian or "native"),
                "byte_offset": int(byte_offset),
                "memory_order": _normalize_memory_order(memory_order),
            }
        )
        return _load_raw_binary_array(
            path,
            dtype=dtype,
            endian=endian,
            shape=shape,
            byte_offset=byte_offset,
            memory_order=memory_order,
        ), metadata
    raise ValueError(f"Unsupported experiment data source '{source}'")


def _load_dicom_experiment_from_dialog(path: Path) -> dict[str, Any]:
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError("Choose a readable DICOM image file")

    from mrax.io.dicom import load_dicom_series
    from mrax.library.default_experiments import default_model_for_type

    experiment, dataset, dicom_metadata = load_dicom_series(path)
    exp_type = str(experiment.exp_type or "anatomical")
    model = default_model_for_type(exp_type)
    extra_metadata = dict(dicom_metadata)
    extra_metadata.update(
        {
            "import_format": "dicom",
            "source_path": str(path),
        }
    )
    return {
        "kind": "experiment",
        "source": "dicom",
        "path": path,
        "name": str(dataset.name or path.stem or "dicom"),
        "data": np.asarray(dataset.data),
        "dims": dict(experiment.dims),
        "exp_type": exp_type,
        "model_name": str(model.name or "identity"),
        "observables": dict(experiment.observables),
        "observable_compatibility": [],
        "extra_metadata": extra_metadata,
    }


def _observables_from_json_text(text: str) -> dict[str, Quantity]:
    payload = json.loads(str(text or "{}"))
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise ValueError("Observables JSON must be an object keyed by observable name")
    observables: dict[str, Quantity] = {}
    for name, row in payload.items():
        if isinstance(row, Mapping):
            unit = str(row.get("unit") or row.get("si_unit") or "").strip()
            values = row.get("values", row.get("value"))
        elif isinstance(row, (list, tuple)) and len(row) == 2 and isinstance(row[1], str):
            values, unit = row
            unit = str(unit).strip()
        else:
            raise ValueError(
                f"Observable '{name}' must be {{\"unit\": \"...\", \"values\": ...}} or [values, unit]."
            )
        if not unit:
            raise ValueError(f"Observable '{name}' is missing a unit")
        observables[str(name)] = Quantity(jnp.asarray(values), unit)
    return observables


def _observables_to_json_text(observables: Mapping[str, Any]) -> str:
    payload = {}
    for name, value in dict(observables or {}).items():
        if isinstance(value, Quantity):
            payload[str(name)] = {
                "unit": value.si_unit,
                "values": _metadata_to_jsonable(np.asarray(value.values)),
            }
        else:
            payload[str(name)] = {"unit": "", "values": _metadata_to_jsonable(np.asarray(value))}
    return json.dumps(payload, indent=2, sort_keys=True)


def _apply_observable_dim_compatibility(
    observables: Mapping[str, Quantity],
    dims: Mapping[str, int],
    *,
    exp_type: str | None = None,
    model_name: str | None = None,
    policy: str = "fail",
) -> tuple[dict[str, Quantity], list[str]]:
    """Validate/adjust observable leading axes against imported non-spatial dims."""

    policy_key = _normalize_observable_mismatch_policy(policy)
    dims_map = {str(key): int(value) for key, value in dict(dims or {}).items()}
    if not observables or not dims_map:
        return dict(observables or {}), ["observable compatibility: no non-scalar observables to check"]
    updated: dict[str, Quantity] = {}
    lines: list[str] = []
    failures: list[str] = []
    adjusted = 0

    for name, qty in require_quantity_mapping(observables).items():
        arr = np.asarray(qty.values)
        target = _observable_target_for_import(name, arr, dims_map, exp_type=exp_type, model_name=model_name)
        if target is None:
            updated[name] = qty
            continue
        axis_names, target_shape = target
        axis_count = len(target_shape)
        current_shape = tuple(int(v) for v in arr.shape[:axis_count])
        if current_shape == target_shape:
            updated[name] = qty
            continue
        description = (
            f"{name}: leading {', '.join(axis_names)} shape {current_shape} "
            f"does not match imported dims {target_shape}"
        )
        if policy_key == "fail":
            failures.append(description)
            updated[name] = qty
            continue
        try:
            resized, actions = _resize_observable_leading_axes(arr, target_shape, policy_key)
        except ValueError as exc:
            failures.append(f"{description}; {exc}")
            updated[name] = qty
            continue
        updated[name] = Quantity(jnp.asarray(resized), qty.si_unit)
        adjusted += 1
        detail = "; ".join(actions) if actions else "no change"
        lines.append(f"observable adjusted: {description}; {detail}")

    if failures:
        suffix = " Select a trim/extend policy or edit the observables JSON."
        raise ValueError("Observable shape mismatch. " + " | ".join(failures) + suffix)
    if adjusted:
        lines.insert(0, f"observable compatibility: adjusted {adjusted} observable(s) using policy '{policy_key}'")
    else:
        lines.append("observable compatibility: checked, no mismatches")
    return updated, lines


def require_quantity_mapping(observables: Mapping[str, Quantity]) -> dict[str, Quantity]:
    out: dict[str, Quantity] = {}
    for key, value in dict(observables or {}).items():
        if not isinstance(value, Quantity):
            raise TypeError(f"Observable '{key}' must be a Quantity")
        out[str(key)] = value
    return out


def _normalize_observable_mismatch_policy(policy: str | None) -> str:
    key = str(policy or "fail").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "none": "fail",
        "error": "fail",
        "trim_longer": "trim",
        "extend_shorter": "extend",
        "trim_extend": "trim_extend",
        "trim_or_extend": "trim_extend",
        "fit": "trim_extend",
    }
    key = aliases.get(key, key)
    if key not in {"fail", "trim", "extend", "trim_extend"}:
        raise ValueError("Observable mismatch policy must be fail, trim, extend, or trim_extend")
    return key


def _observable_target_for_import(
    name: str,
    arr: np.ndarray,
    dims: Mapping[str, int],
    *,
    exp_type: str | None = None,
    model_name: str | None = None,
) -> tuple[tuple[str, ...], tuple[int, ...]] | None:
    arr = np.asarray(arr)
    if arr.ndim == 0:
        return None
    dims_map = {str(key): int(value) for key, value in dict(dims or {}).items()}
    non_spatial = tuple(non_spatial_names(dims_map))
    if not non_spatial:
        return None
    spatial_shape = tuple(int(dims_map[key]) for key in spatial_names(dims_map))
    if spatial_shape and _observable_name_is_spatial_like(name):
        return None
    frame_count = int(frame_count_from_dims(dims_map))

    target_dim = _observable_target_dim_name(name, dims_map, exp_type=exp_type, model_name=model_name)
    if target_dim:
        if target_dim == "frame":
            return (("frame",), (frame_count,))
        if target_dim in non_spatial:
            return ((target_dim,), (int(dims_map[target_dim]),))

    if arr.ndim >= len(non_spatial) and len(non_spatial) > 1 and arr.ndim == len(non_spatial):
        return (non_spatial, tuple(int(dims_map[key]) for key in non_spatial))

    if arr.shape[0] == 1:
        return None
    if frame_count > 1:
        return (("frame",), (frame_count,))
    return None


def _observable_target_dim_name(
    name: str,
    dims: Mapping[str, int],
    *,
    exp_type: str | None = None,
    model_name: str | None = None,
) -> str | None:
    dims_map = {str(key): int(value) for key, value in dict(dims or {}).items()}
    non_spatial = set(non_spatial_names(dims_map))
    norm = _normalize_observable_name(name)
    dim_maps: list[Mapping[str, str | None]] = []
    try:
        dim_maps.append(dim_observable_map(exp_type, model_name))
    except Exception:
        _LOG.debug("Ignored optional GUI operation failure", exc_info=True)
    try:
        dim_maps.append(dim_observable_map(exp_type, exp_type))
    except Exception:
        _LOG.debug("Ignored optional GUI operation failure", exc_info=True)
    for dim_map in dim_maps:
        for dim_name, obs_name in dict(dim_map or {}).items():
            if obs_name and _normalize_observable_name(obs_name) == norm:
                return str(dim_name) if str(dim_name) in non_spatial else "frame"

    for dim_name in non_spatial:
        dim_norm = _normalize_observable_name(dim_name)
        if norm in {dim_norm, f"{dim_norm}s"}:
            return str(dim_name)

    aliases = {
        "te": "te",
        "echo_time": "te",
        "echotime": "te",
        "ti": "ti",
        "inversion_time": "ti",
        "inversiontime": "ti",
        "tr": "tr",
        "repetition_time": "tr",
        "repetitiontime": "tr",
        "flip": "flip",
        "flip_angle": "flip",
        "flipangle": "flip",
        "sat_b1": "b1",
        "saturation_b1": "b1",
        "freq_offsets": "offset",
        "freq_offset": "offset",
        "offsets": "offset",
        "offset": "offset",
        "fair_label": "fair",
        "fairlabel": "fair",
        "time": "t",
        "times": "t",
        "water_offset_time": "t",
        "bval": "frame",
        "bvals": "frame",
        "bvec": "frame",
        "bvecs": "frame",
    }
    target = aliases.get(norm)
    if target is None:
        return None
    if target == "frame":
        return "frame"
    if target in non_spatial:
        return target
    if "frame" in non_spatial:
        return "frame"
    return target if target in dims_map else None


def _normalize_observable_name(name: str) -> str:
    return str(name or "").strip().lower().replace("-", "_").replace(" ", "_")


def _observable_name_is_spatial_like(name: str) -> bool:
    norm = _normalize_observable_name(name)
    return norm.endswith("_map") or norm.endswith("_image") or norm in {"b0_map", "b1_map", "r1_map"}


def _resize_observable_leading_axes(
    arr: np.ndarray,
    target_shape: Sequence[int],
    policy: str,
) -> tuple[np.ndarray, list[str]]:
    out = np.asarray(arr)
    target = tuple(int(v) for v in target_shape)
    if out.ndim < len(target):
        raise ValueError(f"observable has {out.ndim} dimensions, cannot align to target axes {target}")
    allow_trim = policy in {"trim", "trim_extend"}
    allow_extend = policy in {"extend", "trim_extend"}
    actions: list[str] = []
    for axis, target_size in enumerate(target):
        current_size = int(out.shape[axis])
        if current_size == target_size:
            continue
        if current_size > target_size:
            if not allow_trim:
                raise ValueError(f"axis {axis} is longer ({current_size}>{target_size}) and policy does not trim")
            slicer = [slice(None)] * out.ndim
            slicer[axis] = slice(0, target_size)
            out = out[tuple(slicer)]
            actions.append(f"trimmed axis {axis} from {current_size} to {target_size}")
            continue
        if not allow_extend:
            raise ValueError(f"axis {axis} is shorter ({current_size}<{target_size}) and policy does not extend")
        if current_size <= 0:
            raise ValueError(f"axis {axis} is empty and cannot be extended")
        pad_count = target_size - current_size
        last_slice = np.take(out, [-1], axis=axis)
        padding = np.repeat(last_slice, pad_count, axis=axis)
        out = np.concatenate([out, padding], axis=axis)
        actions.append(f"extended axis {axis} from {current_size} to {target_size} using last value")
    return out, actions


def _experiment_metadata_template_from_session(session: GuiSession, exp_name: str) -> dict[str, Any]:
    name = str(exp_name or "").strip()
    experiments = (session.data or {}).get("experiments") or {}
    if name not in experiments:
        raise KeyError(f"Unknown experiment '{name}'")
    exp_data = dict(experiments[name] or {})
    summary = dict(session.summary_by_name.get(name) or {})
    dims = _dims_from_metadata_template_payload(summary.get("dims") or exp_data.get("dims") or {})
    if not dims:
        image = np.asarray(exp_data.get("mr_image"))
        if image.ndim >= 2:
            dims = _dims_from_order_and_shape(_default_dim_order(image.ndim, "image"), image.shape)
    extra_metadata = dict(exp_data.get("extra_metadata") or {})
    return {
        "label": f"loaded experiment '{name}'",
        "name": name,
        "exp_type": exp_data.get("exp_type") or "anatomical",
        "model": exp_data.get("model") or "identity",
        "dims": dims,
        "observables": _copy_quantity_mapping(exp_data.get("observables") or {}),
        "extra_metadata": _metadata_to_jsonable(extra_metadata),
    }


def _load_experiment_metadata_template_from_file(
    path: Path,
    *,
    experiment_name: str | None = None,
) -> dict[str, Any]:
    metadata_path = Path(path)
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata file does not exist: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if _looks_like_study_manifest_payload(payload):
        experiments = dict(payload.get("experiments") or {})
        if experiment_name:
            name = str(experiment_name)
            if name not in experiments:
                choices = ", ".join(sorted(str(key) for key in experiments))
                raise KeyError(f"Experiment '{name}' is not in manifest. Available: {choices}")
        elif len(experiments) == 1:
            name = next(iter(experiments.keys()))
        else:
            choices = ", ".join(sorted(str(key) for key in experiments))
            raise ValueError(f"Manifest contains multiple experiments; enter one of: {choices}")
        entry = dict(experiments[name] or {})
        rel_metadata = entry.get("metadata")
        if not rel_metadata:
            raise ValueError(f"Manifest experiment '{name}' has no metadata file entry")
        child_path = (metadata_path.parent / str(rel_metadata)).resolve()
        with child_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return _experiment_metadata_template_from_payload(
            payload,
            label=f"{metadata_path.name}:{name}",
            fallback_name=name,
        )
    return _experiment_metadata_template_from_payload(
        payload,
        label=str(metadata_path),
        fallback_name=metadata_path.parent.name or metadata_path.stem,
    )


def _looks_like_study_manifest_payload(payload: Any) -> bool:
    if not isinstance(payload, Mapping):
        return False
    experiments = payload.get("experiments")
    if not isinstance(experiments, Mapping):
        return False
    return any(isinstance(entry, Mapping) and "metadata" in entry for entry in experiments.values())


def _experiment_metadata_template_from_payload(
    payload: Mapping[str, Any],
    *,
    label: str | None = None,
    fallback_name: str | None = None,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError("Metadata template must be a JSON object")
    exp_info = payload.get("experiment") if isinstance(payload.get("experiment"), Mapping) else {}
    exp_info = dict(exp_info or {})
    exp_type = (
        exp_info.get("type")
        or exp_info.get("exp_type")
        or payload.get("exp_type")
        or payload.get("type")
        or "anatomical"
    )
    model = exp_info.get("model") or payload.get("model") or "identity"
    dims = _dims_from_metadata_template_payload(exp_info.get("dims") or payload.get("dims") or {})
    observable_values = payload.get("observable_values")
    if observable_values is None:
        observable_fallback = payload.get("observables") or {}
        observable_values = observable_fallback if isinstance(observable_fallback, Mapping) else {}
    observables = _observables_from_metadata_payload(observable_values, payload.get("observable_units") or {})
    extra_metadata = payload.get("extra_metadata")
    if extra_metadata is None:
        extra_metadata = payload.get("importer_metadata")
    if extra_metadata is None:
        extra_metadata = {}
    name = exp_info.get("name") or payload.get("name") or payload.get("dataset_name") or fallback_name
    if not dims and not observables and "exp_type" not in payload and not exp_info:
        raise ValueError("Metadata JSON does not contain experiment metadata or observable values")
    return {
        "label": label or str(name or "metadata file"),
        "name": str(name or ""),
        "exp_type": str(exp_type or "anatomical"),
        "model": str(model or "identity"),
        "dims": dims,
        "observables": observables,
        "extra_metadata": _metadata_to_jsonable(dict(extra_metadata or {})),
    }


def _dims_from_metadata_template_payload(payload: Any) -> dict[str, int]:
    if not payload:
        return {}
    if not isinstance(payload, Mapping):
        raise ValueError("Template dimensions must be a JSON object")
    dims: dict[str, int] = {}
    for key, value in payload.items():
        dim = str(key).strip()
        if not dim:
            raise ValueError("Template dimension names cannot be empty")
        size = int(value)
        if size <= 0:
            raise ValueError(f"Template dimension '{dim}' must be positive")
        dims[dim] = size
    return dims


def _observables_from_metadata_payload(
    observable_values: Any,
    observable_units: Mapping[str, Any] | None = None,
) -> dict[str, Quantity]:
    if not observable_values:
        return {}
    if not isinstance(observable_values, Mapping):
        raise ValueError("Observable metadata must be a JSON object keyed by observable name")
    units = {str(key): str(value) for key, value in dict(observable_units or {}).items()}
    observables: dict[str, Quantity] = {}
    for name, row in observable_values.items():
        key = str(name)
        if isinstance(row, Quantity):
            observables[key] = Quantity(jnp.asarray(np.asarray(row.values)), row.si_unit)
            continue
        if isinstance(row, Mapping):
            unit = str(row.get("unit") or row.get("si_unit") or units.get(key, "")).strip()
            values = row.get("values", row.get("value"))
        else:
            unit = str(units.get(key, "")).strip()
            values = row
        if values is None:
            raise ValueError(f"Observable '{key}' is missing values")
        if not unit:
            raise ValueError(f"Observable '{key}' is missing a unit")
        observables[key] = Quantity(jnp.asarray(values), unit)
    return observables


def _copy_quantity_mapping(mapping: Mapping[str, Any]) -> dict[str, Quantity]:
    return _observables_from_metadata_payload(dict(mapping or {}))


def _metadata_from_json_text(text: str) -> dict[str, Any]:
    payload = json.loads(str(text or "{}"))
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise ValueError("Metadata JSON must be an object")
    return dict(payload)


def _parse_axis_values(value: str, dims: dict[str, int]) -> list[int]:
    axis_map = _regularizer_axis_map(dims)
    axes: list[int] = []
    for item in _parse_csv_values(value):
        if isinstance(item, int):
            axis = item
        else:
            key = str(item).strip()
            if key not in axis_map:
                raise ValueError(f"Unknown regularizer axis '{key}'. Choose one of: {', '.join(axis_map)}")
            axis = axis_map[key]
        if axis not in axes:
            axes.append(axis)
    return axes


def _regularizer_axis_map(dims: dict[str, int]) -> dict[str, int]:
    spatial = [name for name in ("x", "y", "z") if name in dims]
    names = spatial or list(dims.keys())
    return {name: idx for idx, name in enumerate(names)}


def _axis_choice_labels(dims: dict[str, int]) -> list[str]:
    spatial = [name for name in ("x", "y", "z") if name in dims]
    choices = [""]
    if spatial:
        choices.append(", ".join(spatial[:2]))
        choices.extend(spatial)
    else:
        names = list(dims.keys())
        choices.extend(names)
        if len(names) >= 2:
            choices.append(", ".join(names[-2:]))
    out: list[str] = []
    for choice in choices:
        if choice not in out:
            out.append(choice)
    return out


def _constant_dim_choice_labels(dims: dict[str, int]) -> list[str]:
    choices = [""]
    names = list(dims.keys())
    choices.extend(names)
    spatial = [name for name in ("x", "y", "z") if name in dims]
    if len(spatial) >= 2:
        choices.append(", ".join(spatial[:2]))
    non_spatial = [name for name in names if name not in spatial]
    if non_spatial:
        choices.append(", ".join(non_spatial))
    out: list[str] = []
    for choice in choices:
        if choice not in out:
            out.append(choice)
    return out


def _optimizer_choices_for_regularizers(reg_types: set[str]) -> list[str]:
    if any(reg_type.startswith("tv") for reg_type in reg_types):
        return ["fista", "split_bregman"]
    return ["adam", "adamw", "sgd", "rmsprop", "fista", "split_bregman"]


def _regularizer_config_enabled(reg_type: str | None, weight: float | None) -> bool:
    name = str(reg_type or "").strip().lower()
    if not name or name == "none":
        return False
    return weight is not None and float(weight) > 0.0


def _regularizer_weight_from_cell(table, row: int, column: int) -> float | None:
    widget = table.cellWidget(row, column)
    if widget is not None and hasattr(widget, "value"):
        return float(widget.value())
    if widget is not None and hasattr(widget, "text"):
        return _parse_optional_float(widget.text())
    return _parse_optional_float(table.item(row, column).text() if table.item(row, column) else "")


def _fit_result_summary_text(result: dict[str, Any], rois: list[ROI] | tuple[ROI, ...] = ()) -> str:
    lines: list[str] = []
    model = result.get("model")
    if model:
        lines.append(f"Model: {model}")
    loss_history = result.get("loss_history")
    losses = np.asarray([] if loss_history is None else loss_history, dtype=float).reshape((-1,))
    finite_losses = losses[np.isfinite(losses)]
    if finite_losses.size:
        lines.append(
            "Loss: "
            f"{finite_losses[0]:.6g} -> {finite_losses[-1]:.6g} "
            f"(steps={finite_losses.size})"
        )
    fields = result.get("fields") or {}
    if not fields:
        lines.append("No fitted maps returned.")
        return "\n".join(lines)
    notes = [str(note) for note in result.get("fit_notes") or [] if str(note)]
    if notes:
        if lines:
            lines.append("")
        lines.append("Notes")
        lines.extend(f"- {note}" for note in notes)
    if lines:
        lines.append("")
    lines.append("Fitted maps")
    for name, value in fields.items():
        arr, unit = _quantity_array_and_unit(value)
        stats = _numeric_stats(arr)
        unit_text = f" {unit}" if unit else ""
        if stats["finite"] == 0:
            lines.append(f"{name}: shape={tuple(arr.shape)}, no finite values")
            continue
        lines.append(
            f"{name}: mean={stats['mean']:.6g}{unit_text}, "
            f"std={stats['std']:.6g}, min={stats['min']:.6g}, "
            f"max={stats['max']:.6g}, finite={stats['finite']}/{stats['total']}"
        )
        for roi in rois:
            roi_stats = _roi_numeric_stats(arr, roi)
            if roi_stats is None or roi_stats["finite"] == 0:
                continue
            lines.append(
                f"  {roi.name}: mean={roi_stats['mean']:.6g}{unit_text}, "
                f"std={roi_stats['std']:.6g}, n={roi_stats['finite']}"
            )
    return "\n".join(lines)


def _quantity_array_and_unit(value: Any) -> tuple[np.ndarray, str | None]:
    if isinstance(value, Quantity):
        return np.asarray(value.values), getattr(value, "si_unit", None)
    return np.asarray(value), None


def _quantity_values(value: Any) -> np.ndarray:
    if isinstance(value, Quantity):
        return np.asarray(value.values)
    return np.asarray(value)


def _numeric_stats(value: Any) -> dict[str, float | int]:
    arr = np.asarray(value)
    if np.iscomplexobj(arr):
        arr = np.abs(arr)
    arr = arr.astype(float, copy=False)
    total = int(arr.size)
    finite = np.isfinite(arr)
    finite_values = arr[finite]
    if finite_values.size == 0:
        return {"total": total, "finite": 0, "mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan}
    return {
        "total": total,
        "finite": int(finite_values.size),
        "mean": float(np.mean(finite_values)),
        "std": float(np.std(finite_values)),
        "min": float(np.min(finite_values)),
        "max": float(np.max(finite_values)),
    }


def _roi_numeric_stats(value: Any, roi: ROI) -> dict[str, float | int] | None:
    arr = np.asarray(value)
    if np.iscomplexobj(arr):
        arr = np.abs(arr)
    mask = np.asarray(roi.mask, dtype=bool)
    if arr.ndim < 2:
        return None
    if tuple(arr.shape) == tuple(mask.shape):
        selected = arr[mask]
    elif mask.ndim == 2 and arr.ndim >= 2 and tuple(arr.shape[:2]) == tuple(mask.shape):
        selected = arr[mask, ...]
    elif mask.ndim == 2 and arr.ndim >= 2 and tuple(arr.shape[-2:]) == tuple(mask.shape):
        selected = arr[..., mask]
    else:
        try:
            target_dims = tuple(roi.dims) if roi.dims is not None and len(roi.dims) == arr.ndim else None
            aligned = broadcast_mask(mask, arr.shape, mask_dims=roi.dims, target_dims=target_dims)
            selected = arr[np.asarray(aligned, dtype=bool)]
        except Exception:
            return None
    return _numeric_stats(selected)


def _roi_label_mask_groups(rois: list[ROI] | tuple[ROI, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, ...], list[ROI]] = {}
    for roi in rois:
        mask = np.asarray(roi.mask)
        if mask.ndim < 2:
            continue
        grouped.setdefault(tuple(int(v) for v in mask.shape), []).append(roi)
    out: list[dict[str, Any]] = []
    for shape, group_rois in grouped.items():
        label_mask = np.zeros(shape, dtype=np.int32)
        binary_planes: list[np.ndarray] = []
        labels: list[dict[str, Any]] = []
        used_labels: set[int] = set()
        next_label = 1
        for roi in group_rois:
            raw = np.asarray(roi.mask)
            is_label_mask = raw.dtype != bool and str((roi.metadata or {}).get("mask_type", "")).lower() == "label"
            if is_label_mask or (raw.dtype != bool and np.any(raw > 1)):
                for label_value in sorted(int(v) for v in np.unique(raw) if int(v) != 0):
                    mask = raw == label_value
                    if not np.any(mask):
                        continue
                    used_labels.add(label_value)
                    label_mask[mask] = label_value
                    binary_planes.append(mask.astype(np.uint8))
                    labels.append(
                        {
                            "label": int(label_value),
                            "name": roi.name,
                            "source_label": int(label_value),
                            "dims": list(roi.dims or ()),
                            "metadata": dict(roi.metadata),
                            "voxel_count": int(np.count_nonzero(mask)),
                        }
                    )
                if used_labels:
                    next_label = max(used_labels) + 1
                continue
            while next_label in used_labels:
                next_label += 1
            mask = raw != 0
            if not np.any(mask):
                continue
            label_mask[mask] = next_label
            binary_planes.append(mask.astype(np.uint8))
            labels.append(
                {
                    "label": int(next_label),
                    "name": roi.name,
                    "dims": list(roi.dims or ()),
                    "metadata": dict(roi.metadata),
                    "voxel_count": int(np.count_nonzero(mask)),
                }
            )
            used_labels.add(int(next_label))
            next_label += 1
        if not labels:
            continue
        max_label = int(np.max(label_mask)) if label_mask.size else 0
        label_dtype = np.uint16 if max_label <= np.iinfo(np.uint16).max else np.int32
        binary_stack = np.stack(binary_planes, axis=0).astype(np.uint8) if binary_planes else np.zeros((0, *shape), dtype=np.uint8)
        out.append(
            {
                "shape": shape,
                "label_mask": label_mask.astype(label_dtype, copy=False),
                "binary_stack": binary_stack,
                "labels": labels,
            }
        )
    return out


def _button_status_tip(button) -> str:
    explicit = str(button.statusTip() or "").strip()
    if explicit:
        return explicit
    text = str(button.text() or "").replace("&", "").strip()
    normalized = " ".join(text.split())
    tips = {
        "Load...": "Load a .mrax bundle, editable package, manifest, GUI state, ROI set, or mask.",
        "Save...": "Save the current study, GUI state, ROIs, and fit outputs as a .mrax bundle or folder.",
        "Import...": "Import Bruker or recursive DICOM folder data without changing source files.",
        "Export...": "Export images, k-space, maps, and ROIs as analysis-friendly NumPy or NIfTI files.",
        "Shape...": "Create an editable rotated rectangle, square, circle, ellipse, or regular-polygon ROI.",
        "Lasso": "Draw a freehand ROI by dragging; support count, spline type, and gentle edge snap are configurable.",
        "Add Pool": "Add reusable custom pool templates to the current pool-based model fit.",
        "Remove Pool": "Remove the selected custom pool from this fit or disable a built-in pool.",
        "Brush": "Paint ROI pixels directly on the active image.",
        "Erase": "Erase pixels from the selected ROI.",
        "Wand": "Add a connected intensity-similar region from a clicked seed pixel.",
        "Auto": "Create a largest-component ROI from an automatic intensity threshold.",
        "Snap": "Move selected lasso/polygon vertices to nearby intensity edges.",
        "New ROI": "Create an empty active ROI so subsequent drawing starts a separate region.",
        "Rename": "Rename the selected ROI across the shared ROI set.",
        "Remove": "Delete the selected ROI from the shared ROI set.",
        "Disable All": "Disable all shared ROIs for the current experiment without deleting them.",
        "Consolidate": "Convert the selected geometric ROI into a raster mask for brush edits.",
        "Combine": "Combine chosen ROIs on the current slice, across a slice dimension, or as full masks.",
        "Subtract": "Use one or more ROIs to cut pixels or labels out of another ROI.",
        "Shift...": "Shift selected ROIs horizontally, vertically, or across z slices.",
        "Copy Z": "Copy selected ROI masks from the current z slice to another z slice.",
        "Delete Z": "Clear selected ROI masks on the current z slice without deleting the ROI.",
        "Done": "Finish the active ROI drawing or brush-style editing mode.",
        "Undo": "Remove the last point from the active lasso or polygon drawing.",
        "Clear": "Remove ROIs from the current experiment view.",
        "Apply All": "Apply the current ROI set to all compatible experiments in the study.",
        "Interp Z": "Interpolate and extrapolate selected ROIs along z with separate strategy choices.",
        "Reset View": "Reset the current image or plot view to its full extent.",
        "Undo Visual": "Restore the previous visual-only view state for the current experiment.",
        "Reset Visual": "Show original data and clear visual-only mask, color, and transform modifiers.",
        "Enlarge Image": "Open the current image, k-space, or map view in a larger window.",
        "Enlarge Plot": "Open the signal or fit preview plot in a larger window.",
        "Study Summary": "Open a larger graphical summary of the loaded study.",
        "Reset Model Defaults": "Restore default model parameters, bounds, regularizers, and constants.",
        "Fit Sanity Check": "Run a lightweight validation of the current model, data, and optimizer setup.",
        "Preview ROI Fit": "Fit the selected model to the active ROI mean signal and plot the preview.",
        "Run Full-Map Fit": "Run the selected model fit across all voxels to create fitted parameter maps.",
        "Save Fit": "Save the latest full-map fit output to MRax's manifest layout.",
        "Run SimpleITK Registration": "Register the current viewer image and frame to the shared reference image with the configured SimpleITK options.",
        "Preview Stored Transform": "Apply the selected stored registration as a temporary Registration-page view transform.",
        "Preview Selected Overlay": "Enable the registration-page reference background overlay for the current view.",
        "Preview Manual": "Preview the manual shift, rotation, and stretch controls on the selected moving image.",
        "Store Manual Transform": "Save the current manual alignment as an available registration transform.",
        "Use Transform + Show Overlay": "Apply the selected transform to the current view state and enable the registration reference overlay.",
        "Reset Manual": "Reset manual shift, rotation, and stretch controls to identity.",
        "Run MRax Drift Correction": "Fit and apply MRax linear drift correction to the active experiment frame stack.",
        "Preview Mask Transform": "Preview the selected ROI resampled through the selected registration transform.",
        "Add Transformed ROI": "Create a new ROI from the selected mask transform preview.",
        "Refresh Diagnostics": "Recompute diagnostics for shapes, maps, masks, k-space, and fit setup.",
        "Save Diagnostic Report": "Write a plain-text report with captured warnings, errors, session state, and reproduction context.",
        "Clear Event Log": "Clear captured application warnings and errors from the Diagnostics page.",
        "Browse": "Choose a file or folder for this dialog field.",
        "OK": "Accept the dialog and run the selected action.",
        "Cancel": "Close the dialog without applying changes.",
    }
    return tips.get(normalized, f"Run '{normalized}'." if normalized else "")


def _pool_suffix(name: str, pools: set[str]) -> str | None:
    for pool in sorted(pools, key=len, reverse=True):
        suffix = f"_{pool}"
        if name.endswith(suffix):
            return pool
    return None


def _sanitize_pool_names(text: str) -> list[str]:
    names: list[str] = []
    for raw in str(text or "").replace(";", ",").split(","):
        cleaned = []
        for char in raw.strip():
            if char.isalnum() or char == "_":
                cleaned.append(char.lower())
            elif char in {" ", "-", "."}:
                cleaned.append("_")
        name = "_".join(part for part in "".join(cleaned).split("_") if part)
        if name and name[0].isdigit():
            name = f"pool_{name}"
        if name and name not in names:
            names.append(name)
    return names


def _gui_config_path() -> Path:
    env = os.environ.get("MRAX_GUI_CONFIG")
    if env:
        return Path(env).expanduser()
    root = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")).expanduser()
    return root / "mrax" / "gui_config.json"


def _default_gui_config() -> dict[str, Any]:
    return {
        "format": "mrax.gui.config",
        "version": 1,
        "custom_pools": {
            "ab_mt_n": {},
            "multipool_lorentzian": {},
            "zaiss_full": {},
        },
    }


def _load_gui_config(path: Path | None = None) -> dict[str, Any]:
    config_path = Path(path or _gui_config_path())
    default = _default_gui_config()
    if not config_path.is_file():
        _save_gui_config(default, config_path)
        return default
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        backup = config_path.with_suffix(config_path.suffix + ".invalid")
        try:
            shutil.copy2(config_path, backup)
        except Exception:
            _LOG.debug("Ignored optional GUI operation failure", exc_info=True)
        _save_gui_config(default, config_path)
        return default
    config = _default_gui_config()
    if isinstance(data, dict):
        config.update({key: value for key, value in data.items() if key not in {"format", "version"}})
        config["format"] = "mrax.gui.config"
        config["version"] = 1
    if not isinstance(config.get("custom_pools"), dict):
        config["custom_pools"] = {"ab_mt_n": {}, "multipool_lorentzian": {}, "zaiss_full": {}}
    custom = dict(config.get("custom_pools") or {})
    custom.setdefault("ab_mt_n", {})
    custom.setdefault("multipool_lorentzian", {})
    custom.setdefault("zaiss_full", {})
    config["custom_pools"] = custom
    _save_gui_config(config, config_path)
    return config


def _save_gui_config(config: dict[str, Any], path: Path | None = None) -> Path:
    config_path = Path(path or _gui_config_path())
    config_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _default_gui_config()
    payload.update(dict(config or {}))
    payload["format"] = "mrax.gui.config"
    payload["version"] = 1
    config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return config_path


def _gui_config_pool_templates(config: dict[str, Any] | None, model_key: str | None) -> dict[str, dict[str, object]]:
    key = normalize_model_key(model_key) or ""
    custom = dict((config or {}).get("custom_pools") or {})
    raw = dict(custom.get(key) or {})
    templates: dict[str, dict[str, object]] = {}
    for key, value in raw.items():
        names = _sanitize_pool_names(str(key))
        if not names or not isinstance(value, dict):
            continue
        templates[names[0]] = dict(value)
    return templates


def _gui_config_ab_mt_pool_templates(config: dict[str, Any] | None) -> dict[str, dict[str, object]]:
    return _gui_config_pool_templates(config, "ab_mt_n")


def _gui_config_pool_names(config: dict[str, Any] | None, model_key: str | None) -> tuple[str, ...]:
    return tuple(_gui_config_pool_templates(config, model_key).keys())


def _gui_config_ab_mt_pool_names(config: dict[str, Any] | None) -> tuple[str, ...]:
    return _gui_config_pool_names(config, "ab_mt_n")


def _ab_mt_default_pool_override(_pool_name: str) -> dict[str, object]:
    return {
        "rate_key": "k",
        "f": {"transform": "bounded", "bounds": [0.0001, 0.1], "unit": "1", "init_raw": 0.0},
        "k": {"transform": "bounded", "bounds": [0.8, 3.0], "unit": "ppm", "init_raw": 0.0},
        "dw": {"transform": "bounded", "bounds": [1.0, 5.0], "unit": "ppm", "init_raw": 0.0},
    }


def _multipool_lorentzian_default_pool_override(_pool_name: str) -> dict[str, object]:
    return {
        "amp": {"transform": "bounded", "bounds": [0.01, 0.4], "unit": "1", "init_raw": 0.0},
        "gamma": {"transform": "bounded", "bounds": [0.1, 10.0], "unit": "ppm", "init_raw": 0.0},
        "dw": {"transform": "bounded", "bounds": [1.0, 5.0], "unit": "ppm", "init_raw": 0.0},
    }


def _zaiss_full_default_pool_override(_pool_name: str) -> dict[str, object]:
    return {
        "f": {"transform": "bounded", "bounds": [0.00001, 0.1], "unit": "1", "init_raw": 0.0},
        "k": {"transform": "bounded", "bounds": [0.8, 5.0], "unit": "ppm", "init_raw": 0.0},
        "r2": {"transform": "identity", "trainable": False, "value": 0.0, "unit": "ppm"},
        "dw": {"transform": "bounded", "bounds": [1.0, 5.0], "unit": "ppm", "init_raw": 0.0},
    }


def _default_pool_override(model_key: str | None, pool_name: str) -> dict[str, object]:
    key = normalize_model_key(model_key)
    if key == "ab_mt_n":
        return _ab_mt_default_pool_override(pool_name)
    if key == "multipool_lorentzian":
        return _multipool_lorentzian_default_pool_override(pool_name)
    if key == "zaiss_full":
        return _zaiss_full_default_pool_override(pool_name)
    return {}


def _pool_overrides_with_custom_defaults(
    model_key: str | None,
    pools: tuple[str, ...] | list[str],
    pool_overrides: dict[str, dict[str, object]] | None,
    gui_config: dict[str, Any] | None = None,
) -> dict[str, dict[str, object]]:
    overrides = {str(key): dict(value) for key, value in dict(pool_overrides or {}).items()}
    key = normalize_model_key(model_key)
    if key not in {"ab_mt_n", "multipool_lorentzian", "zaiss_full"}:
        return overrides
    known = set(model_available_pools(key))
    templates = _gui_config_pool_templates(gui_config, key)
    for pool in pools:
        name = str(pool)
        if name and name not in known and name not in overrides:
            overrides[name] = templates.get(name, _default_pool_override(key, name))
    return overrides


def _pool_overrides_with_custom_ab_mt_defaults(
    model_key: str | None,
    pools: tuple[str, ...] | list[str],
    pool_overrides: dict[str, dict[str, object]] | None,
    gui_config: dict[str, Any] | None = None,
) -> dict[str, dict[str, object]]:
    return _pool_overrides_with_custom_defaults(model_key, pools, pool_overrides, gui_config)


def _parameter_fixed_from_overrides(
    name: str,
    model_key: str | None,
    pools: tuple[str, ...] | list[str],
    settings: FitSettings,
) -> bool:
    pool_name = (
        _pool_suffix(name, set(pools or ()))
        if normalize_model_key(model_key) in {"multipool_lorentzian", "ab_mt_n", "zaiss_full"}
        else None
    )
    if pool_name is not None:
        base_name = name[: -(len(pool_name) + 1)]
        cfg = (settings.pool_overrides or {}).get(pool_name, {}).get(base_name, {})
    else:
        cfg = (settings.param_overrides or {}).get(name, {})
    return bool(isinstance(cfg, dict) and cfg.get("trainable") is False)


def _detect_load_kind(path: Path) -> str:
    path = Path(path)
    if path.is_dir():
        if (path / "mrax_package.json").is_file():
            return "bundle_folder"
        manifest = _find_study_manifest(path)
        if manifest is not None:
            return "manifest"
        raise ValueError(f"Unsupported MRax folder: {path}")
    suffix = path.suffix.lower()
    if suffix == ".mrax":
        return "bundle"
    if suffix == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            if path.name == "study_manifest.json":
                return "manifest"
            raise
        if payload.get("format") == "mrax.gui.state":
            return "gui_state"
        if payload.get("format") == "mrax.gui.package":
            return "bundle_folder"
        if "rois" in payload and "npz" in payload:
            return "roi"
        if "manifest" in payload or "experiments" in payload or path.name == "study_manifest.json":
            return "manifest"
        raise ValueError(f"JSON file is not a recognized MRax manifest, GUI state, or ROI set: {path}")
    if suffix == ".npy":
        return "mask"
    if suffix == ".npz":
        sidecar = path.with_suffix(".json")
        if sidecar.is_file():
            try:
                payload = json.loads(sidecar.read_text(encoding="utf-8"))
                if "rois" in payload and "npz" in payload:
                    return "roi"
            except Exception:
                _LOG.debug("Ignored optional GUI operation failure", exc_info=True)
        return "roi"
    raise ValueError(f"Unsupported load file extension: {path.suffix}")


def _write_mrax_package_manifest(root: Path, entries: dict[str, str], *, warnings: Sequence[str] | None = None) -> Path:
    path = Path(root) / "mrax_package.json"
    payload = {
        "format": "mrax.gui.package",
        "version": 1,
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
            for key, value in dict(payload.get("entries") or {}).items():
                candidate = (root / str(value)).resolve()
                if not _is_relative_to(candidate, root):
                    raise ValueError(f"Unsafe package entry path outside package directory: {value}")
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


def _write_mrax_bundle(path: Path, source_dir: Path) -> Path:
    path = Path(path)
    source_dir = Path(source_dir)
    if path.exists():
        path.unlink()
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(source_dir.rglob("*")):
            if file_path.is_file():
                zf.write(file_path, file_path.relative_to(source_dir))
    return path


def _extract_mrax_bundle(path: Path, output_dir: Path) -> Path:
    path = Path(path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "r") as zf:
        for member in zf.infolist():
            target = output_dir / member.filename
            if not _is_relative_to(target.resolve(), output_dir.resolve()):
                raise ValueError(f"Unsafe path in MRax bundle: {member.filename}")
        zf.extractall(output_dir)
    return output_dir


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
        return os.path.relpath(path, root)
    except ValueError:
        return str(path)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _serialize_gui_state(
    session: GuiSession,
    *,
    preview_results: dict[str, dict[str, Any]],
    fit_results: dict[str, dict[str, Any]],
    drift_results: Mapping[str, Mapping[str, Any]] | None = None,
    data_manifest: Path | None = None,
    fit_manifest: Path | None = None,
    state_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    arrays: dict[str, np.ndarray] = {}
    states: dict[str, Any] = {}
    fitted_map_rows: dict[str, list[dict[str, Any]]] = {}
    original_image_rows: dict[str, dict[str, str]] = {}
    shared_roi_rows = _serialize_roi_rows(session.shared_rois, "shared_roi", arrays)
    for exp_name, state in session.view_states.items():
        roi_rows = _serialize_roi_rows(state.rois, f"roi_{_safe_key(exp_name)}", arrays)
        states[exp_name] = {
            "active_frame": state.active_frame,
            "active_z": state.active_z,
            "active_indices": dict(state.active_indices),
            "active_map": state.active_map,
            "active_kspace_mode": state.active_kspace_mode,
            "active_kspace_centering": state.active_kspace_centering,
            "visual_state": _visual_state_to_dict(state.visual_state),
            "fit_settings": _fit_settings_to_dict(state.fit_settings),
            "frame_mask": None if state.frame_mask is None else state.frame_mask.as_array().astype(bool).tolist(),
            "rois": roi_rows,
        }
    for exp_name, exp_data in ((session.data or {}).get("experiments") or {}).items():
        original = exp_data.get("original_mr_image")
        if original is not None:
            key = f"original_{_safe_key(str(exp_name))}_mr_image"
            arrays[key] = np.asarray(original)
            original_image_rows[str(exp_name)] = {"array_key": key}
        rows = []
        for map_name, value in (exp_data.get("fitted_maps") or {}).items():
            key = f"fmap_{_safe_key(str(exp_name))}_{_safe_key(str(map_name))}"
            unit = ""
            array_value = value
            if isinstance(value, Quantity):
                unit = value.si_unit
                array_value = value.values
            arrays[key] = np.asarray(array_value)
            rows.append({"name": str(map_name), "array_key": key, "unit": unit})
        if rows:
            fitted_map_rows[str(exp_name)] = rows
    drift_rows: dict[str, dict[str, Any]] = {}
    for key, result in dict(drift_results or {}).items():
        if str(result.get("kind") or "") != "mrax_linear_drift" or result.get("corrected") is None:
            continue
        array_key = f"drift_{_safe_key(str(key))}_corrected"
        arrays[array_key] = np.asarray(result["corrected"])
        drift_rows[str(key)] = {
            "key": str(key),
            "kind": "mrax_linear_drift",
            "corrected_key": array_key,
            "metadata": _metadata_to_jsonable(dict(result.get("metadata") or {})),
            "summary": _metadata_to_jsonable(dict(result.get("summary") or {})),
        }
    data_manifest_text = _state_relative_path(data_manifest, state_path) if data_manifest is not None else None
    fit_manifest_text = _state_relative_path(fit_manifest, state_path) if fit_manifest is not None else None
    return (
        {
            "format": "mrax.gui.state",
            "version": 2,
            "data_manifest": data_manifest_text,
            "fit_manifest": fit_manifest_text,
            "source_kind": session.source_kind,
            "selected_experiment": session.selected_experiment,
            "shared_rois": shared_roi_rows,
            "view_states": states,
            "original_images": original_image_rows,
            "fitted_maps": fitted_map_rows,
            "drift_results": drift_rows,
            "preview_results": {
                key: _preview_result_to_json(value)
                for key, value in preview_results.items()
            },
            "fit_results": {
                key: {"model": value.get("model"), "name": value.get("name")}
                for key, value in fit_results.items()
            },
        },
        arrays,
    )


def _serialize_roi_rows(rois: list[ROI] | tuple[ROI, ...], prefix: str, arrays: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows = []
    for idx, roi in enumerate(rois):
        key = f"{prefix}_{idx}"
        arrays[key] = np.asarray(roi.mask)
        rows.append(
            {
                "name": roi.name,
                "array_key": key,
                "dims": list(roi.dims or ()),
                "metadata": dict(roi.metadata),
            }
        )
    return rows


def _warn_restore(warnings: list[str] | None, message: str) -> None:
    if warnings is not None:
        warnings.append(str(message))


def _safe_int_value(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_float_value(value: Any, default: float = 1.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_dict_value(value: Any) -> dict[str, Any]:
    try:
        return dict(value or {})
    except Exception:
        return {}


def _restore_roi_rows(
    rows: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    arrays: dict[str, np.ndarray],
    *,
    warnings: list[str] | None = None,
    context: str = "ROI",
) -> list[ROI]:
    rois = []
    for idx, roi_data in enumerate(rows or []):
        if not isinstance(roi_data, Mapping):
            _warn_restore(warnings, f"{context} row {idx} skipped: expected mapping, got {type(roi_data).__name__}.")
            continue
        key = roi_data.get("array_key")
        if key not in arrays:
            _warn_restore(warnings, f"{context} '{roi_data.get('name') or key or idx}' skipped: missing array key '{key}'.")
            continue
        try:
            rois.append(
                ROI(
                    roi_data.get("name") or key,
                    np.asarray(arrays[key]),
                    dims=tuple(roi_data.get("dims") or ()),
                    metadata=dict(roi_data.get("metadata") or {}),
                )
            )
        except Exception as exc:
            _warn_restore(warnings, f"{context} '{roi_data.get('name') or key}' skipped: {exc}")
    return rois


def _restore_drift_results(
    rows: Mapping[str, Mapping[str, Any]],
    arrays: dict[str, np.ndarray],
    *,
    warnings: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    restored: dict[str, dict[str, Any]] = {}
    for key, row in _safe_dict_value(rows).items():
        if not isinstance(row, Mapping):
            _warn_restore(warnings, f"Drift result '{key}' skipped: expected mapping, got {type(row).__name__}.")
            continue
        array_key = row.get("corrected_key")
        if array_key not in arrays:
            _warn_restore(warnings, f"Drift result '{key}' skipped: missing corrected array key '{array_key}'.")
            continue
        result_key = str(row.get("key") or key)
        try:
            restored[result_key] = {
                "key": result_key,
                "kind": "mrax_linear_drift",
                "corrected": np.asarray(arrays[array_key]),
                "metadata": dict(row.get("metadata") or {}),
                "summary": dict(row.get("summary") or {}),
            }
        except Exception as exc:
            _warn_restore(warnings, f"Drift result '{key}' skipped: {exc}")
    return restored


def _visual_state_to_dict(state: VisualViewState) -> dict[str, Any]:
    return {
        "show_original": bool(state.show_original),
        "display_mode": state.display_mode,
        "mask_enabled": bool(state.mask_enabled),
        "mask_roi": state.mask_roi,
        "drift_key": state.drift_key,
        "transform_key": state.transform_key,
        "transform_direction": state.transform_direction,
        "color_channel": state.color_channel,
        "color_alpha": float(state.color_alpha),
        "reference_channel": state.reference_channel,
        "moving_channel": state.moving_channel,
        "reference_alpha": float(state.reference_alpha),
        "moving_alpha": float(state.moving_alpha),
    }


def _visual_state_from_dict(data: Mapping[str, Any]) -> VisualViewState:
    if not isinstance(data, Mapping):
        data = {}
    mask_roi = data.get("mask_roi") or None
    return VisualViewState(
        show_original=bool(data.get("show_original", True)),
        display_mode=str(data.get("display_mode") or "current"),
        mask_enabled=bool(data.get("mask_enabled", bool(mask_roi))),
        mask_roi=mask_roi,
        drift_key=data.get("drift_key") or None,
        transform_key=data.get("transform_key") or None,
        transform_direction=str(data.get("transform_direction") or "moving_to_reference"),
        color_channel=str(data.get("color_channel") or "gray"),
        color_alpha=_safe_float_value(data.get("color_alpha", 1.0), 1.0),
        reference_channel=str(data.get("reference_channel") or "red"),
        moving_channel=str(data.get("moving_channel") or "green"),
        reference_alpha=_safe_float_value(data.get("reference_alpha", 1.0), 1.0),
        moving_alpha=_safe_float_value(data.get("moving_alpha", 1.0), 1.0),
    )


_GUI_STATE_CURRENT_VERSION = 2
_GUI_STATE_KNOWN_KEYS = {
    "format",
    "version",
    "data_manifest",
    "fit_manifest",
    "source_kind",
    "selected_experiment",
    "shared_rois",
    "view_states",
    "original_images",
    "fitted_maps",
    "drift_results",
    "preview_results",
    "fit_results",
    "array_file",
}


def _restore_gui_state(
    session: GuiSession,
    payload: dict[str, Any],
    arrays: dict[str, np.ndarray],
    *,
    warnings: list[str] | None = None,
) -> None:
    fmt = str(payload.get("format") or "")
    if fmt != "mrax.gui.state":
        raise ValueError("GUI state format must be 'mrax.gui.state'.")
    if "version" not in payload:
        raise ValueError("GUI state version is required.")
    version = _safe_int_value(payload.get("version"), -1)
    if version != _GUI_STATE_CURRENT_VERSION:
        raise ValueError(
            f"GUI state version {version} is not supported; expected {_GUI_STATE_CURRENT_VERSION}."
        )
    unknown_keys = sorted(str(key) for key in payload.keys() if str(key) not in _GUI_STATE_KNOWN_KEYS)
    if unknown_keys:
        raise ValueError(
            f"Unsupported GUI state top-level entr{'y' if len(unknown_keys) == 1 else 'ies'}: {', '.join(unknown_keys)}."
        )

    for exp_name, row in _safe_dict_value(payload.get("original_images")).items():
        if exp_name not in session.experiment_names:
            _warn_restore(warnings, f"Original image snapshot for missing experiment '{exp_name}' skipped.")
            continue
        if not isinstance(row, Mapping):
            _warn_restore(warnings, f"Original image snapshot for '{exp_name}' skipped: expected mapping.")
            continue
        key = row.get("array_key")
        if key not in arrays:
            _warn_restore(warnings, f"Original image snapshot for '{exp_name}' skipped: missing array key '{key}'.")
            continue
        try:
            session.data["experiments"][exp_name]["original_mr_image"] = np.asarray(arrays[key]).copy()
        except Exception as exc:
            _warn_restore(warnings, f"Original image snapshot for '{exp_name}' skipped: {exc}")
    session._ensure_original_image_snapshots()

    for exp_name, map_rows in _safe_dict_value(payload.get("fitted_maps")).items():
        if exp_name not in session.experiment_names:
            _warn_restore(warnings, f"Fitted maps for missing experiment '{exp_name}' skipped.")
            continue
        exp_data = session.data["experiments"][exp_name]
        restored_maps = dict(exp_data.get("fitted_maps") or {})
        units = dict(exp_data.get("fitted_map_units") or {})
        for idx, row in enumerate(map_rows or []):
            if not isinstance(row, Mapping):
                _warn_restore(warnings, f"Fitted map row {idx} for '{exp_name}' skipped: expected mapping.")
                continue
            key = row.get("array_key")
            if key not in arrays:
                _warn_restore(warnings, f"Fitted map '{row.get('name') or key or idx}' for '{exp_name}' skipped: missing array key '{key}'.")
                continue
            try:
                map_name = str(row.get("name") or key)
                unit = str(row.get("unit") or "")
                arr = np.asarray(arrays[key])
                restored_maps[map_name] = Quantity(arr, unit) if unit else arr
                if unit:
                    units[map_name] = unit
            except Exception as exc:
                _warn_restore(warnings, f"Fitted map '{row.get('name') or key}' for '{exp_name}' skipped: {exc}")
        exp_data["fitted_maps"] = restored_maps
        exp_data["fitted_map_units"] = units

    session.shared_rois = _restore_roi_rows(
        payload.get("shared_rois") or [],
        arrays,
        warnings=warnings,
        context="shared ROI",
    )
    restored: dict[str, ExperimentViewState] = {}
    for exp_name, state_data in _safe_dict_value(payload.get("view_states")).items():
        if exp_name not in session.experiment_names:
            _warn_restore(warnings, f"View state for missing experiment '{exp_name}' skipped.")
            continue
        if not isinstance(state_data, Mapping):
            _warn_restore(warnings, f"View state for '{exp_name}' skipped: expected mapping.")
            continue
        rois = _restore_roi_rows(
            state_data.get("rois") or [],
            arrays,
            warnings=warnings,
            context=f"ROI in view state '{exp_name}'",
        )
        frame_mask = state_data.get("frame_mask")
        try:
            active_indices = {
                str(k): _safe_int_value(v, 0)
                for k, v in dict(state_data.get("active_indices") or {}).items()
            }
        except Exception as exc:
            _warn_restore(warnings, f"Active indices for '{exp_name}' reset to defaults: {exc}")
            active_indices = {}
        try:
            restored[str(exp_name)] = ExperimentViewState(
                active_frame=_safe_int_value(state_data.get("active_frame"), 0),
                active_z=_safe_int_value(state_data.get("active_z"), 0),
                active_indices=active_indices,
                active_map=state_data.get("active_map"),
                active_kspace_mode=str(state_data.get("active_kspace_mode") or "log_magnitude"),
                active_kspace_centering=str(state_data.get("active_kspace_centering") or "auto"),
                rois=rois,
                fit_settings=_fit_settings_from_dict(state_data.get("fit_settings") or {}),
                frame_mask=None if frame_mask is None else FrameMask(np.asarray(frame_mask, dtype=bool)),
                visual_state=_visual_state_from_dict(state_data.get("visual_state") or {}),
            )
        except Exception as exc:
            _warn_restore(warnings, f"View state for '{exp_name}' skipped: {exc}")
    session.view_states = restored
    selected = payload.get("selected_experiment")
    if selected in session.experiment_names:
        session.selected_experiment = selected
    elif selected:
        _warn_restore(warnings, f"Selected experiment '{selected}' is missing; using the default selected experiment.")
    session._restore_or_reset_current_view_state()


def _resolve_state_manifest_path(payload: dict[str, Any], state_path: Path) -> Path | None:
    value = payload.get("data_manifest")
    if not value:
        return None
    candidate = Path(str(value))
    if not candidate.is_absolute():
        candidate = state_path.parent / candidate
    return candidate if candidate.is_file() else None


def _state_relative_path(path: Path | None, state_path: Path | None) -> str | None:
    if path is None:
        return None
    path = Path(path)
    if state_path is None:
        return str(path)
    try:
        return os.path.relpath(path, Path(state_path).parent)
    except ValueError:
        return str(path)


def _fit_settings_to_dict(settings: FitSettings) -> dict[str, Any]:
    return {
        "model_name": settings.model_name,
        "optimizer_name": settings.optimizer_name,
        "num_steps": settings.num_steps,
        "step_size": settings.step_size,
        "selected_roi": settings.selected_roi,
        "mask_fill_value": settings.mask_fill_value,
        "auto_mask_nan_observables": settings.auto_mask_nan_observables,
        "use_view_state_data": settings.use_view_state_data,
        "compute_derived_maps": settings.compute_derived_maps,
        "sat_trans_reference_scan": settings.sat_trans_reference_scan,
        "ab_mt_t1_map_ref": settings.ab_mt_t1_map_ref,
        "ab_mt_b1_map_ref": settings.ab_mt_b1_map_ref,
        "ab_mt_b0_map_ref": settings.ab_mt_b0_map_ref,
        "selected_pools": list(settings.selected_pools),
        "param_overrides": settings.param_overrides,
        "pool_overrides": settings.pool_overrides,
        "regularizer_configs": settings.regularizer_configs,
        "parameter_constant_dims": {
            key: list(value)
            for key, value in settings.parameter_constant_dims.items()
        },
    }


def _fit_settings_from_dict(data: dict[str, Any] | Mapping[str, Any]) -> FitSettings:
    if not isinstance(data, Mapping):
        data = {}
    constant_dims: dict[str, tuple[Any, ...]] = {}
    raw_constant_dims = _safe_dict_value(data.get("parameter_constant_dims"))
    for key, value in raw_constant_dims.items():
        try:
            constant_dims[str(key)] = tuple(value)
        except Exception:
            continue
    return FitSettings(
        model_name=data.get("model_name"),
        optimizer_name=data.get("optimizer_name") or "adam",
        num_steps=_safe_int_value(data.get("num_steps"), 200),
        step_size=_safe_float_value(data.get("step_size"), 0.02),
        selected_roi=data.get("selected_roi"),
        mask_fill_value=_safe_float_value(data.get("mask_fill_value"), 0.0),
        auto_mask_nan_observables=bool(data.get("auto_mask_nan_observables", True)),
        use_view_state_data=bool(data.get("use_view_state_data", False)),
        compute_derived_maps=data.get("compute_derived_maps"),
        sat_trans_reference_scan=data.get("sat_trans_reference_scan"),
        ab_mt_t1_map_ref=data.get("ab_mt_t1_map_ref"),
        ab_mt_b1_map_ref=data.get("ab_mt_b1_map_ref"),
        ab_mt_b0_map_ref=data.get("ab_mt_b0_map_ref"),
        selected_pools=tuple(data.get("selected_pools") or ()),
        param_overrides=_safe_dict_value(data.get("param_overrides")),
        pool_overrides=_safe_dict_value(data.get("pool_overrides")),
        regularizer_configs=_safe_dict_value(data.get("regularizer_configs")),
        parameter_constant_dims=constant_dims,
    )


def _preview_result_to_json(result: dict[str, Any]) -> dict[str, Any]:
    keys = ("observed", "predicted", "included", "x")
    out = {key: _json_array(result.get(key)) for key in keys if key in result}
    out["x_name"] = result.get("x_name")
    out["model"] = result.get("model")
    return out


def _export_object_records_for_session(session: GuiSession) -> list[dict[str, Any]]:
    if not session.is_loaded:
        return []
    records: list[dict[str, Any]] = []
    experiments = (session.data or {}).get("experiments") or {}
    for exp_name, exp_data in experiments.items():
        group = f"Experiment: {exp_name}"
        dims = dict(exp_data.get("dims") or {})
        if exp_data.get("mr_image") is not None:
            current_image = np.asarray(exp_data["mr_image"])
            records.append(
                _export_record(
                    "image",
                    f"{exp_name} / image",
                    current_image,
                    group=group,
                    experiment=exp_name,
                    name="mr_image",
                    dims=dims,
                )
            )
            original_image = exp_data.get("original_mr_image")
            if original_image is not None:
                original_array = np.asarray(original_image)
                if tuple(original_array.shape) != tuple(current_image.shape) or not np.array_equal(original_array, current_image):
                    records.append(
                        _export_record(
                            "original_image",
                            f"{exp_name} / original image",
                            original_array,
                            group=group,
                            experiment=exp_name,
                            name="original_mr_image",
                            dims=dims,
                        )
                    )
        processed = dict(exp_data.get("processed") or {})
        if processed.get("raw_reconstruction") is not None:
            records.append(
                _export_record(
                    "reconstruction",
                    f"{exp_name} / MRax reconstruction",
                    np.asarray(processed["raw_reconstruction"]),
                    group=group,
                    experiment=exp_name,
                    name="raw_reconstruction",
                    dims=dims,
                )
            )
        if exp_data.get("raw_kspace") is not None:
            records.append(
                _export_record(
                    "k_space",
                    f"{exp_name} / k-space",
                    np.asarray(exp_data["raw_kspace"]),
                    group=group,
                    experiment=exp_name,
                    name="raw_kspace",
                    dims=dims,
                )
            )
        for map_name, value in (exp_data.get("fitted_maps") or {}).items():
            arr, unit = _quantity_array_and_unit(value)
            records.append(
                _export_record(
                    "map",
                    f"{exp_name} / map / {map_name}",
                    arr,
                    group=group,
                    experiment=exp_name,
                    name=str(map_name),
                    dims=dims,
                    unit=unit or str((exp_data.get("fitted_map_units") or {}).get(map_name) or ""),
                )
            )

    roi_by_name: dict[str, ROI] = {}
    for roi in [*list(session.shared_rois), *list(session.rois)]:
        roi_by_name.setdefault(str(roi.name), roi)
    for roi in roi_by_name.values():
        record = _export_record(
            "roi",
            f"ROI / {roi.name}",
            np.asarray(roi.mask),
            group="ROIs",
            name=roi.name,
            dims=tuple(roi.dims or ()),
            metadata=dict(roi.metadata),
            roi=roi,
        )
        record["nifti_reference"] = _roi_nifti_reference(session, roi.metadata)
        records.append(record)
    return records


def _export_record(
    kind: str,
    label: str,
    array: Any,
    *,
    group: str,
    name: str,
    experiment: str | None = None,
    dims: Mapping[str, Any] | Sequence[str] | None = None,
    unit: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    roi: ROI | None = None,
) -> dict[str, Any]:
    arr = np.asarray(array)
    record = {
        "id": "/".join(part for part in (experiment, kind, name) if part),
        "group": str(group),
        "kind": str(kind),
        "label": str(label),
        "name": str(name),
        "experiment": None if experiment is None else str(experiment),
        "array": arr,
        "shape": tuple(int(v) for v in arr.shape),
        "dtype": str(arr.dtype),
        "dims": dict(dims) if isinstance(dims, Mapping) else list(dims or ()),
        "unit": unit or "",
        "metadata": dict(metadata or {}),
    }
    if roi is not None:
        record["roi"] = roi
    return record


def _single_export_file_types(record: Mapping[str, Any]) -> list[dict[str, str]]:
    arr = np.asarray(record.get("array"))
    kind = str(record.get("kind") or "")
    formats = [
        {"id": "npy", "label": "NumPy array (.npy)", "suffix": ".npy"},
        {"id": "npz", "label": "Compressed NumPy archive (.npz)", "suffix": ".npz"},
    ]
    if kind == "roi" and record.get("roi") is not None:
        formats.insert(0, {"id": "roi_set", "label": "MRax ROI set (.npz + .json)", "suffix": ".npz"})
    if arr.ndim in (1, 2) and not np.iscomplexobj(arr):
        formats.append({"id": "csv", "label": "CSV table (.csv)", "suffix": ".csv"})
    if kind != "k_space" and arr.ndim in (2, 3) and not np.iscomplexobj(arr):
        formats.append({"id": "nifti", "label": "NIfTI image (.nii)", "suffix": ".nii"})
    formats.append({"id": "json", "label": "Metadata JSON (.json)", "suffix": ".json"})
    return formats


def _single_export_record_metadata(record: Mapping[str, Any]) -> dict[str, Any]:
    arr = np.asarray(record.get("array"))
    metadata = {
        "id": record.get("id"),
        "label": record.get("label"),
        "kind": record.get("kind"),
        "name": record.get("name"),
        "experiment": record.get("experiment"),
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "dims": record.get("dims"),
        "unit": record.get("unit") or "",
        "summary": _array_summary(arr, unit=str(record.get("unit") or "") or None),
    }
    extra = record.get("metadata")
    if extra:
        metadata["metadata"] = _metadata_to_jsonable(extra)
    if record.get("kind") == "roi" and record.get("roi") is not None:
        roi = record["roi"]
        metadata["roi"] = {
            "name": roi.name,
            "dims": list(roi.dims or ()),
            "metadata": _metadata_to_jsonable(dict(roi.metadata)),
        }
    reference = record.get("nifti_reference")
    if reference:
        metadata["nifti_reference"] = {
            key: value
            for key, value in dict(reference).items()
            if key != "affine"
        }
    return _metadata_to_jsonable(metadata)


def _single_export_default_path(record: Mapping[str, Any], file_type: str, directory: Path) -> Path:
    formats = {str(fmt["id"]): fmt for fmt in _single_export_file_types(record)}
    fmt = formats.get(str(file_type)) or next(iter(formats.values()))
    parts = []
    if record.get("experiment"):
        parts.append(str(record.get("experiment")))
    parts.append(str(record.get("kind") or "object"))
    parts.append(str(record.get("name") or "data"))
    stem = _safe_filename("_".join(parts))
    return Path(directory) / f"{stem}{fmt['suffix']}"


def _single_export_name_filter(file_type: str) -> str:
    labels = {
        "roi_set": "MRax ROI set (*.npz *.json)",
        "npy": "NumPy array (*.npy)",
        "npz": "Compressed NumPy archive (*.npz)",
        "csv": "CSV table (*.csv)",
        "nifti": "NIfTI image (*.nii)",
        "json": "JSON metadata (*.json)",
    }
    return f"{labels.get(str(file_type), 'All files (*)')};;All files (*)"


def _write_single_export_object(record: Mapping[str, Any], path: Path, file_type: str) -> list[Path]:
    fmt = str(file_type)
    path = Path(path).expanduser()
    arr = np.asarray(record.get("array"))
    metadata = _single_export_record_metadata(record)
    if fmt == "roi_set":
        roi = record.get("roi")
        if roi is None:
            raise ValueError("Selected object is not an ROI")
        if path.suffix.lower() not in {".npz", ".json"}:
            path = path.with_suffix(".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        roi_npz, roi_json = save_roi_set([roi], path, metadata=metadata)
        return [roi_npz, roi_json]
    suffix_by_format = {"npy": ".npy", "npz": ".npz", "csv": ".csv", "nifti": ".nii", "json": ".json"}
    suffix = suffix_by_format.get(fmt)
    if suffix is None:
        raise ValueError(f"Unsupported export file type '{file_type}'")
    if path.suffix.lower() != suffix:
        path = path.with_suffix(suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "npy":
        np.save(path, arr)
    elif fmt == "npz":
        np.savez_compressed(path, data=arr, metadata=json.dumps(metadata))
    elif fmt == "csv":
        if arr.ndim == 1:
            value = arr.reshape((-1, 1))
        elif arr.ndim == 2:
            value = arr
        else:
            raise ValueError("CSV export supports 1D and 2D arrays only")
        if np.iscomplexobj(value):
            raise ValueError("CSV export does not support complex arrays")
        np.savetxt(path, value, delimiter=",")
    elif fmt == "nifti":
        if arr.ndim not in (2, 3) or np.iscomplexobj(arr):
            raise ValueError("NIfTI export supports real-valued 2D or 3D arrays only")
        from mrax.io.nifti import save_nifti

        reference = dict(record.get("nifti_reference") or {})
        affine = _validated_nifti_affine(reference.get("affine"))
        if affine is None:
            affine = np.eye(4, dtype=float)
        save_nifti(path, arr.astype(np.float32) if arr.dtype == bool else arr, affine)
    elif fmt == "json":
        path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return [path]


def _validated_nifti_affine(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        affine = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if affine.shape != (4, 4) or not np.all(np.isfinite(affine)):
        return None
    return affine


def _experiment_nifti_affine(
    session: GuiSession,
    experiment_name: str | None,
    *,
    frame_index: int = 0,
) -> np.ndarray | None:
    if not experiment_name or not session.data:
        return None
    exp_data = (session.data.get("experiments") or {}).get(str(experiment_name))
    if not isinstance(exp_data, Mapping):
        return None

    metadata_sources = [
        exp_data,
        exp_data.get("extra_metadata") or {},
        exp_data.get("processed") or {},
        exp_data.get("kspace_metadata") or {},
    ]
    for metadata in metadata_sources:
        if not isinstance(metadata, Mapping):
            continue
        affines = metadata.get("affines")
        if affines is not None:
            try:
                candidates = list(affines)
            except TypeError:
                candidates = []
            if candidates:
                index = max(0, min(int(frame_index), len(candidates) - 1))
                affine = _validated_nifti_affine(candidates[index])
                if affine is not None:
                    return affine
        affine = _validated_nifti_affine(metadata.get("affine"))
        if affine is not None:
            return affine

    extra = exp_data.get("extra_metadata") or {}
    visu = extra.get("visu") if isinstance(extra, Mapping) else None
    if isinstance(visu, Mapping):
        try:
            from mrax.io.bruker_import import bruker_affine_from_visu

            affine_value = bruker_affine_from_visu(dict(visu))
        except Exception:
            return None
        if isinstance(affine_value, list):
            if not affine_value:
                return None
            index = max(0, min(int(frame_index), len(affine_value) - 1))
            affine = affine_value[index]
        else:
            affine = affine_value
        return _validated_nifti_affine(affine)
    return None


def _roi_nifti_reference(session: GuiSession, metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    roi_metadata = dict(metadata or {})
    declared_experiment = str(roi_metadata.get("source_experiment") or "").strip()
    reference_experiment = declared_experiment or str(session.selected_experiment or "").strip()
    indices = roi_metadata.get("slice_indices") or {}
    try:
        frame_index = int(indices.get("frame", 0)) if isinstance(indices, Mapping) else 0
    except (TypeError, ValueError):
        frame_index = 0
    affine = _experiment_nifti_affine(session, reference_experiment, frame_index=frame_index)
    return {
        "experiment": reference_experiment or None,
        "selection": "mask metadata" if declared_experiment else "active experiment",
        "frame_index": max(0, frame_index),
        "affine": np.eye(4, dtype=float) if affine is None else affine,
        "identity_fallback": affine is None,
    }


def _roi_nifti_reference_summary(session: GuiSession) -> str:
    active = str(session.selected_experiment or "none")
    roi_by_name: dict[str, ROI] = {}
    for roi in [*list(session.shared_rois), *list(session.rois)]:
        roi_by_name.setdefault(str(roi.name), roi)
    if not roi_by_name:
        return f"Active reference experiment: {active}. No ROI masks are active."

    grouped: dict[tuple[str, bool], int] = {}
    for roi in roi_by_name.values():
        reference = _roi_nifti_reference(session, roi.metadata)
        name = str(reference.get("experiment") or "unavailable")
        fallback = bool(reference.get("identity_fallback"))
        grouped[(name, fallback)] = grouped.get((name, fallback), 0) + 1
    details = []
    for (name, fallback), count in sorted(grouped.items()):
        status = "identity fallback" if fallback else "reference affine"
        details.append(f"{name}: {count} mask(s), {status}")
    return f"Active reference experiment: {active}. ROI NIfTI references: " + "; ".join(details) + "."


def _shape_text(value: Any) -> str:
    try:
        return "x".join(str(int(v)) for v in np.asarray(value).shape)
    except Exception:
        return ""


def _json_array(value: Any) -> Any:
    try:
        return np.asarray(value).tolist()
    except Exception:
        return value


def _experiment_metadata_payload(session: GuiSession) -> dict[str, Any]:
    if not session.is_loaded:
        return {}
    exp = session.current_experiment
    summary = session.current_summary
    manifest = (session.data or {}).get("manifest") or {}
    source = manifest.get("source") if isinstance(manifest, dict) else None
    payload: dict[str, Any] = {
        "study": {
            "name": manifest.get("study_name") if isinstance(manifest, dict) else None,
            "source_kind": session.source_kind,
            "manifest_path": session.manifest_path,
            "source": source,
        },
        "experiment": {
            "name": session.selected_experiment,
            "type": exp.get("exp_type"),
            "model": exp.get("model"),
            "dims": session.experiment_dims,
            "summary": summary,
            "image_shape": tuple(session.image_array.shape),
            "kspace_shape": None if session.kspace_array is None else tuple(session.kspace_array.shape),
            "maps": session.map_names,
        },
        "observables": exp.get("observables") or {},
        "observable_units": exp.get("observable_units") or {},
        "fitted_map_units": exp.get("fitted_map_units") or {},
        "processed_arrays": dict(exp.get("processed") or {}),
        "kspace_metadata": exp.get("kspace_metadata") or {},
    }
    extra = exp.get("extra_metadata")
    if extra:
        payload["importer_metadata"] = extra
    study_globals = (session.data or {}).get("study_globals") or {}
    if study_globals:
        payload["study_globals"] = study_globals
    return payload


def _view_key(text: str) -> str:
    return str(text or "").strip().lower().replace("-", "_").replace(" ", "_")


def _matched_2d_pair(a: Any, b: Any) -> tuple[np.ndarray, np.ndarray]:
    left = np.asarray(a, dtype=float)
    right = np.asarray(b, dtype=float)
    if left.ndim != 2 or right.ndim != 2:
        raise ValueError("Vendor/MRax comparison requires two 2D image slices.")
    rows = min(left.shape[0], right.shape[0])
    cols = min(left.shape[1], right.shape[1])
    if rows <= 0 or cols <= 0:
        raise ValueError("Vendor/MRax comparison has no overlapping image area.")
    return left[:rows, :cols], right[:rows, :cols]


def _side_by_side_images(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = _robust_display_image(left)
    right = _robust_display_image(right)
    gap = max(2, int(round(min(left.shape) * 0.02)))
    fill = np.nanmin([np.nanmin(left), np.nanmin(right)]) if left.size and right.size else 0.0
    spacer = np.full((left.shape[0], gap), fill, dtype=float)
    return np.concatenate([left, spacer, right], axis=1)


def _robust_display_image(image: np.ndarray) -> np.ndarray:
    """Scale one comparison pane independently for display."""

    arr = np.asarray(image, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=float)
    lo, hi = np.percentile(finite, [1.0, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(finite))
        hi = float(np.nanmax(finite))
    if hi <= lo:
        return np.zeros_like(arr, dtype=float)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def _metadata_value_summary(value: Any) -> dict[str, str]:
    if isinstance(value, Quantity):
        arr = np.asarray(value.values)
        return {
            "value": _array_summary(arr, unit=value.si_unit),
            "type": "Quantity",
        }
    if isinstance(value, np.ndarray):
        return {"value": _array_summary(value), "type": "ndarray"}
    if isinstance(value, Path):
        return {"value": str(value), "type": "Path"}
    if isinstance(value, dict):
        return {"value": f"{len(value)} key(s)", "type": "dict"}
    if isinstance(value, (list, tuple)):
        if _metadata_sequence_is_scalar(value):
            return {"value": _format_metadata_scalar_sequence(value), "type": type(value).__name__}
        return {"value": f"{len(value)} item(s)", "type": type(value).__name__}
    if value is None:
        return {"value": "", "type": "None"}
    if isinstance(value, (str, int, float, bool)):
        return {"value": _truncate_metadata_text(str(value)), "type": type(value).__name__}
    try:
        arr = np.asarray(value)
        if arr.shape:
            return {"value": _array_summary(arr), "type": type(value).__name__}
    except Exception:
        _LOG.debug("Ignored optional GUI operation failure", exc_info=True)
    return {"value": _truncate_metadata_text(str(value)), "type": type(value).__name__}


def _metadata_to_jsonable(value: Any) -> Any:
    if isinstance(value, Quantity):
        return {
            "unit": value.si_unit,
            "values": _metadata_to_jsonable(value.values),
        }
    if isinstance(value, np.ndarray):
        arr = np.asarray(value)
        if arr.size <= 200:
            return arr.tolist()
        return {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "summary": _array_summary(arr),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _metadata_to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        if len(value) > 200:
            return {
                "length": len(value),
                "items_preview": [_metadata_to_jsonable(item) for item in value[:50]],
            }
        return [_metadata_to_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        arr = np.asarray(value)
        if arr.shape:
            return _metadata_to_jsonable(arr)
    except Exception:
        _LOG.debug("Ignored optional GUI operation failure", exc_info=True)
    return str(value)


def _array_summary(value: np.ndarray, *, unit: str | None = None) -> str:
    arr = np.asarray(value)
    unit_text = f" {unit}" if unit else ""
    if arr.size == 0:
        return f"shape={tuple(arr.shape)}, empty{unit_text}"
    if np.issubdtype(arr.dtype, np.number):
        finite = arr[np.isfinite(arr)] if not np.iscomplexobj(arr) else np.abs(arr[np.isfinite(arr)])
        if finite.size:
            return (
                f"shape={tuple(arr.shape)}, dtype={arr.dtype}, "
                f"min={float(np.nanmin(finite)):.6g}, max={float(np.nanmax(finite)):.6g}{unit_text}"
            )
    return f"shape={tuple(arr.shape)}, dtype={arr.dtype}{unit_text}"


def _metadata_sequence_is_scalar(value: list[Any] | tuple[Any, ...]) -> bool:
    if len(value) > 16:
        return False
    scalar_types = (str, int, float, bool, type(None), np.generic)
    return all(isinstance(item, scalar_types) for item in value)


def _format_metadata_scalar_sequence(value: list[Any] | tuple[Any, ...]) -> str:
    return _truncate_metadata_text(", ".join(str(item) for item in value))


def _truncate_metadata_text(value: str, limit: int = 180) -> str:
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def _safe_key(value: str) -> str:
    return base64.urlsafe_b64encode(value.encode("utf-8")).decode("ascii").rstrip("=")


def _safe_filename(value: str) -> str:
    text = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(value).strip())
    text = text.strip("._")
    return text or "mrax"


def _file_dialog_options(QtWidgets):
    option_enum = getattr(QtWidgets.QFileDialog, "Option", QtWidgets.QFileDialog)
    return option_enum.DontUseNativeDialog


def _clear_layout(layout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        child = item.widget()
        if child is not None:
            child.deleteLater()
        child_layout = item.layout()
        if child_layout is not None:
            _clear_layout(child_layout)


def _section_box(QtWidgets, title: str, content):
    box = QtWidgets.QGroupBox(title)
    box.setCheckable(False)
    layout = QtWidgets.QVBoxLayout(box)
    layout.setContentsMargins(8, 6, 8, 8)
    layout.addWidget(content)
    return box


def _mask_bounds(mask: np.ndarray) -> list[float]:
    arr = np.asarray(mask, dtype=bool)
    if arr.ndim > 2:
        arr = mask_slice_for_display(arr)
    points = np.argwhere(arr)
    if points.size == 0:
        h, w = arr.shape[-2], arr.shape[-1]
        return [float(w) * 0.25, float(h) * 0.25, max(1.0, float(w) * 0.5), max(1.0, float(h) * 0.5)]
    y0, x0 = points.min(axis=0)
    y1, x1 = points.max(axis=0)
    return [float(x0), float(y0), float(max(1, x1 - x0 + 1)), float(max(1, y1 - y0 + 1))]


def _roi_can_address_display_plane(
    roi: ROI,
    spatial_shape: tuple[int, ...],
    spatial_dims: tuple[str, ...],
    indices: Mapping[str, int],
    display_shape: tuple[int, int],
) -> bool:
    try:
        broadcast_mask(
            roi.mask,
            spatial_shape,
            mask_dims=roi.dims,
            target_dims=spatial_dims,
        )
        display_mask = mask_slice_for_display(roi.mask, dims=roi.dims, indices=indices)
    except Exception:
        return False
    return tuple(np.asarray(display_mask).shape) == tuple(display_shape)


def _rect_mask(
    shape: tuple[int, int],
    bounds: list[float] | tuple[float, float, float, float],
    *,
    angle: float = 0.0,
) -> np.ndarray:
    h, w = shape
    x, y, bw, bh = [float(v) for v in bounds]
    if abs(float(angle)) > 1e-9:
        xx, yy = np.meshgrid(np.arange(w, dtype=float) + 0.5, np.arange(h, dtype=float) + 0.5)
        local = _rotate_points(
            np.column_stack([xx.ravel() - x, yy.ravel() - y]),
            -float(angle),
            origin=(0.0, 0.0),
        )
        inside = (
            (local[:, 0] >= 0.0)
            & (local[:, 0] <= max(bw, 0.0))
            & (local[:, 1] >= 0.0)
            & (local[:, 1] <= max(bh, 0.0))
        )
        return inside.reshape((h, w))
    x0 = max(0, int(np.floor(x)))
    y0 = max(0, int(np.floor(y)))
    x1 = min(w, int(np.ceil(x + bw)))
    y1 = min(h, int(np.ceil(y + bh)))
    mask = np.zeros((h, w), dtype=bool)
    if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = True
    return mask


def _ellipse_mask(
    shape: tuple[int, int],
    bounds: list[float] | tuple[float, float, float, float],
    *,
    angle: float = 0.0,
) -> np.ndarray:
    h, w = shape
    x, y, bw, bh = [float(v) for v in bounds]
    if bw <= 0 or bh <= 0:
        return np.zeros((h, w), dtype=bool)
    xx, yy = np.meshgrid(np.arange(w, dtype=float) + 0.5, np.arange(h, dtype=float) + 0.5)
    coords = np.column_stack([xx.ravel() - x, yy.ravel() - y])
    if abs(float(angle)) > 1e-9:
        coords = _rotate_points(coords, -float(angle), origin=(0.0, 0.0))
    x_eval = coords[:, 0].reshape((h, w))
    y_eval = coords[:, 1].reshape((h, w))
    cx = bw / 2.0
    cy = bh / 2.0
    rx = max(bw / 2.0, 1e-6)
    ry = max(bh / 2.0, 1e-6)
    return ((x_eval - cx) / rx) ** 2 + ((y_eval - cy) / ry) ** 2 <= 1.0


def _point_component(point: Any, index: int) -> float:
    try:
        return float(point[index])
    except Exception:
        return float(point.x() if int(index) == 0 else point.y())


def _mask_from_roi_item(item, kind: str, shape: tuple[int, int]) -> tuple[np.ndarray, dict[str, Any]]:
    kind = str(kind)
    if kind in {"rectangle", "ellipse"}:
        state = item.saveState() if hasattr(item, "saveState") else item.getState()
        pos = state.get("pos", item.pos())
        size = state.get("size", item.size())
        angle = float(state.get("angle", item.angle() if hasattr(item, "angle") else 0.0) or 0.0)
        bounds = [_point_component(pos, 0), _point_component(pos, 1), _point_component(size, 0), _point_component(size, 1)]
        if kind == "ellipse":
            return _ellipse_mask(shape, bounds, angle=angle), {"shape": "ellipse", "bounds": bounds, "angle": angle}
        return _rect_mask(shape, bounds, angle=angle), {"shape": "rectangle", "bounds": bounds, "angle": angle}
    if kind == "polygon":
        state = item.saveState() if hasattr(item, "saveState") else item.getState()
        pos = state.get("pos", (0.0, 0.0))
        angle = float(state.get("angle", 0.0) or 0.0)
        offset = (_point_component(pos, 0), _point_component(pos, 1))
        local_points = []
        for point in state.get("points", []):
            local_points.append([_point_component(point, 0), _point_component(point, 1)])
        arr = _transform_points(local_points, pos=offset, angle=angle)
        points = arr.tolist()
        if arr.shape[0] < 3:
            return np.zeros(shape, dtype=bool), {"shape": "polygon", "points": points, "angle": 0.0}
        return _polygon_mask(shape, arr), {"shape": "polygon", "points": points, "angle": 0.0}
    return np.zeros(shape, dtype=bool), {"shape": kind}


def _polygon_mask(shape: tuple[int, int], points: np.ndarray) -> np.ndarray:
    h, w = int(shape[0]), int(shape[1])
    out = np.zeros((h, w), dtype=bool)
    pts = _simplify_polyline(np.asarray(points, dtype=float), tolerance=0.0, closed=True)
    if pts.shape[0] < 3 or h <= 0 or w <= 0:
        return out

    x0 = max(0, int(np.floor(float(np.min(pts[:, 0])))))
    x1 = min(w, int(np.ceil(float(np.max(pts[:, 0])))) + 1)
    y0 = max(0, int(np.floor(float(np.min(pts[:, 1])))))
    y1 = min(h, int(np.ceil(float(np.max(pts[:, 1])))) + 1)
    if x1 <= x0 or y1 <= y0:
        return out

    x = np.arange(x0, x1, dtype=float) + 0.5
    y = np.arange(y0, y1, dtype=float) + 0.5
    xx, yy = np.meshgrid(x, y)
    inside = np.zeros((y1 - y0, x1 - x0), dtype=bool)
    x_points = pts[:, 0]
    y_points = pts[:, 1]
    j = len(pts) - 1
    for i in range(len(pts)):
        xi, yi = x_points[i], y_points[i]
        xj, yj = x_points[j], y_points[j]
        intersects = ((yi > yy) != (yj > yy)) & (
            xx < (xj - xi) * (yy - yi) / ((yj - yi) + 1e-12) + xi
        )
        inside ^= intersects
        j = i
    out[y0:y1, x0:x1] = inside
    return out


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
