"""Desktop GUI helpers for MRax.

The public objects in this module intentionally avoid importing Qt until
:func:`launch` starts the desktop application.
"""

from __future__ import annotations

from pathlib import Path

from .masks import MaskRecord, import_mask, load_roi_set, save_roi_set
from .session import FitSettings, FrameMask, GuiSession


def launch(path: str | Path | None = None) -> int:
    """Launch the MRax desktop GUI.

    Parameters
    ----------
    path:
        Optional path to a ``study_manifest.json`` file to open on startup.
    """

    from .app import main

    argv = [] if path is None else [str(path)]
    return main(argv)


__all__ = [
    "FitSettings",
    "FrameMask",
    "GuiSession",
    "MaskRecord",
    "import_mask",
    "launch",
    "load_roi_set",
    "save_roi_set",
]
