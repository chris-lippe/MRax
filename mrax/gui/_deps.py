"""Lazy GUI dependency helpers."""

from __future__ import annotations

import importlib

GUI_INSTALL_HINT = "Install MRax with `pip install mrax`."


class MissingGuiDependencyError(RuntimeError):
    """Raised when the required GUI stack cannot be imported."""


def _dependency_error(message: str, exc: BaseException) -> MissingGuiDependencyError:
    err = MissingGuiDependencyError(message)
    err.__cause__ = exc
    return err


def _format_exc(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _qt_probe_details() -> str:
    details = []
    for module_name in ("PySide6.QtCore", "PySide6.QtGui", "PySide6.QtWidgets"):
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            details.append(f"{module_name}: {_format_exc(exc)}")
    return "; ".join(details)


def import_qt():
    """Import Qt through qtpy, raising a clear MRax-specific error on failure."""

    try:
        from qtpy import QtCore, QtGui, QtWidgets
    except Exception as exc:  # pragma: no cover - exercised when Qt cannot import.
        probe = _qt_probe_details()
        detail = f" Underlying Qt errors: {probe}" if probe else f" Underlying error: {_format_exc(exc)}"
        raise _dependency_error(
            "MRax GUI could not import Qt. "
            f"{GUI_INSTALL_HINT} "
            "If the Python packages are installed, install the missing native Qt system libraries."
            f"{detail}",
            exc,
        ) from exc
    return QtCore, QtGui, QtWidgets


def import_pyqtgraph():
    """Import pyqtgraph, raising a clear MRax-specific error on failure."""

    try:
        import pyqtgraph as pg
    except Exception as exc:  # pragma: no cover - exercised when pyqtgraph cannot import.
        raise _dependency_error(
            "MRax GUI could not import pyqtgraph. "
            f"{GUI_INSTALL_HINT} "
            "If the Python package is installed, install the missing native Qt/OpenGL system libraries. "
            f"Underlying error: {_format_exc(exc)}",
            exc,
        ) from exc
    return pg


__all__ = [
    "GUI_INSTALL_HINT",
    "MissingGuiDependencyError",
    "import_pyqtgraph",
    "import_qt",
]
