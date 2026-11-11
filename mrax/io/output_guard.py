"""Filesystem boundaries for imported sources and explicitly requested outputs."""

from __future__ import annotations

import os
import shutil
import threading
import warnings
from pathlib import Path


def require_empty_directory(path):
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(
            "Choose a new or empty folder. MRax does not merge files into an existing folder"
        )
    return path


def require_new_file(path, *, overwrite=False, warn=False):
    path = Path(path)
    if path.exists() or path.is_symlink():
        if not overwrite or not path.is_file():
            raise ValueError(
                "This file already exists. Choose a new name or explicitly confirm replacement"
            )
        if warn:
            warnings.warn(
                "The explicitly selected export will overwrite an existing file.",
                UserWarning,
                stacklevel=2,
            )
    return path


def publish_new_file(staged, destination):
    """Publish without replacing a file created since the initial check."""
    staged, destination = Path(staged), Path(destination)
    try:
        os.link(staged, destination)
    except FileExistsError as exc:
        raise ValueError("This file now exists. Choose a new name or confirm replacement") from exc
    except OSError:
        try:
            with staged.open("rb") as source, destination.open("xb") as output:
                try:
                    shutil.copyfileobj(source, output)
                except BaseException:
                    destination.unlink(missing_ok=True)
                    raise
        except FileExistsError as exc:
            raise ValueError(
                "This file now exists. Choose a new name or confirm replacement"
            ) from exc


class SourceProtection:
    """Remember read-only input locations without changing their permissions."""

    def __init__(self):
        self._sources: dict[Path, bool] = {}
        self._lock = threading.RLock()

    def protect(self, path, *, directory=None):
        path = Path(path).expanduser().resolve()
        with self._lock:
            self._sources[path] = path.is_dir() if directory is None else bool(directory)

    def is_source(self, path, *, tree=False):
        path = Path(path).expanduser().resolve()
        with self._lock:
            sources = list(self._sources.items())
        return any(
            path == source
            or (directory and path.is_relative_to(source))
            or (tree and source.is_relative_to(path))
            for source, directory in sources
        )

    def check(self, destination, *, directory=False, image_export=False):
        destination = Path(destination)
        if not destination.exists() and not destination.is_symlink():
            return
        if self.is_source(destination, tree=directory) and (directory or not image_export):
            raise ValueError(
                "Imported sources are read-only. Choose a new output location. "
                "Only an explicitly confirmed image export may replace a source image file."
            )
