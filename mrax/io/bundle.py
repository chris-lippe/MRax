"""Portable MRax archives, independent of GUI state."""

from __future__ import annotations

import os
import shutil
import tempfile
import zipfile
from pathlib import Path

from .output_guard import publish_new_file, require_empty_directory, require_new_file


def write_bundle(path, source_dir, *, progress_callback=None, overwrite=False):
    """Atomically write a bundle, reporting bytes compressed."""
    path, root = Path(path), Path(source_dir)
    require_new_file(path, overwrite=overwrite)
    path.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.resolve() != path.resolve())
    total = max(1, sum(p.stat().st_size for p in files))
    done = 0
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".partial", delete=False) as stream:
            temporary = Path(stream.name)
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            for item in files:
                with item.open("rb") as source, archive.open(item.relative_to(root).as_posix(), "w", force_zip64=True) as target:
                    while block := source.read(1024 * 1024):
                        target.write(block)
                        done += len(block)
                        if progress_callback:
                            progress_callback(done, total, "Compressing study")
        if overwrite:
            os.replace(temporary, path)
        else:
            publish_new_file(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    if progress_callback:
        progress_callback(total, total, "Study saved")
    return path


def extract_bundle(path, output_dir):
    """Extract an archive after validating all member paths and file types."""
    root = Path(output_dir).resolve()
    require_empty_directory(root)
    with zipfile.ZipFile(path) as archive:
        names = set()
        for info in archive.infolist():
            target = (root / info.filename.replace("\\", "/")).resolve()
            if not target.is_relative_to(root) or (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Unsafe path in MRax bundle")
            key = os.path.normcase(str(target))
            if key in names:
                raise ValueError("Duplicate path in MRax bundle")
            names.add(key)
        root.mkdir(parents=True, exist_ok=True)
        for info in archive.infolist():
            target = (root / info.filename.replace("\\", "/")).resolve()
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                # Exclusive creation also protects against collisions after validation.
                with archive.open(info) as source, target.open("xb") as destination:
                    shutil.copyfileobj(source, destination)
    return root
