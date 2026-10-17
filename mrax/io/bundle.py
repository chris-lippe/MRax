"""Portable MRax archives, independent of GUI state."""

from __future__ import annotations

import os
import tempfile
import zipfile
from pathlib import Path


def write_bundle(path, source_dir, *, progress_callback=None):
    """Atomically write a bundle, reporting bytes compressed."""
    path, root = Path(path), Path(source_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in root.rglob("*") if p.is_file())
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
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    if progress_callback:
        progress_callback(total, total, "Study saved")
    return path


def extract_bundle(path, output_dir):
    """Extract an archive after validating all member paths and file types."""
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            target = (root / info.filename.replace("\\", "/")).resolve()
            if not target.is_relative_to(root) or (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Unsafe path in MRax bundle")
        archive.extractall(root)
    return root
