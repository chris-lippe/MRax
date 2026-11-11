"""Publish completed files without partial writes or implicit replacement."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from mrax.io.output_guard import publish_new_file


def write_file_atomic(destination, writer, *, overwrite=False):
    destination = Path(destination)
    if not destination.parent.is_dir():
        raise ValueError("Choose an existing destination folder")
    if destination.exists() and (not overwrite or not destination.is_file()):
        raise ValueError("This file already exists. Confirm replacement or choose another name")
    with tempfile.TemporaryDirectory(prefix=".mrax-saving-", dir=destination.parent) as temporary:
        staged = Path(temporary) / destination.name
        writer(staged)
        if not staged.is_file():
            raise ValueError("The operation did not produce the selected file")
        if overwrite:
            os.replace(staged, destination)
        else:
            publish_new_file(staged, destination)
    return destination


def file_identity(path):
    stat = Path(path).stat()
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns]


def export_file_plan(prepared, destination, *, policy):
    """Inspect only generated filenames. Never treat a directory as replaceable."""
    prepared, destination = Path(prepared), Path(destination)
    if destination.is_symlink() or (destination.exists() and not destination.is_dir()):
        raise ValueError("Choose a destination folder")
    if not destination.exists() and not destination.parent.is_dir():
        raise ValueError("Choose an existing parent folder")
    files = []
    conflicts = []
    for source in sorted(prepared.rglob("*")):
        if source.is_symlink():
            raise ValueError("Export files cannot contain symbolic links")
        if not source.is_file():
            continue
        relative = source.relative_to(prepared).as_posix()
        target = destination / relative
        for parent in target.parents:
            if parent == destination:
                break
            if parent.is_symlink() or (parent.exists() and not parent.is_dir()):
                raise ValueError(
                    "An output subfolder is occupied by a file or symbolic link: " + relative
                )
        replaceable, reason = policy(relative, target)
        files.append(relative)
        if target.exists() or target.is_symlink():
            if target.is_symlink() or not target.is_file():
                replaceable, reason = (
                    False,
                    "An existing folder or symbolic link cannot be replaced",
                )
            conflicts.append(
                {
                    "path": relative,
                    "replaceable": replaceable,
                    "reason": reason,
                    "identity": file_identity(target)
                    if target.is_file() and not target.is_symlink()
                    else None,
                }
            )
    if not files:
        raise ValueError("The export did not produce any files")
    return files, conflicts


def publish_export_files(
    prepared, destination, files, *, replace_files=(), progress=None, expected=None, validate=None
):
    """Publish individual files, rolling back this export if a write fails."""
    prepared, destination = Path(prepared), Path(destination)
    replacements = set(replace_files)
    expected = (
        expected
        if expected is not None
        else {relative: file_identity(destination / relative) for relative in replacements}
    )
    created_directories = []
    written = []
    with tempfile.TemporaryDirectory(prefix="mrax_export_rollback_") as temporary:
        backups = Path(temporary)
        try:
            # Preserve replaced files before the first destination write.
            for relative in replacements:
                if file_identity(destination / relative) != expected[relative]:
                    raise ValueError("An output file changed. Review its replacement again")
                backup = backups / relative
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(destination / relative, backup)
            for index, relative in enumerate(files):
                if progress:
                    progress(index, len(files), "Writing export files")
                target = destination / relative
                if relative in replacements and file_identity(target) != expected[relative]:
                    raise ValueError("An output file changed. Review its replacement again")
                if validate:
                    validate(relative, target)
                missing = []
                parent = target.parent
                while not parent.exists():
                    missing.append(parent)
                    parent = parent.parent
                for folder in reversed(missing):
                    folder.mkdir()
                    created_directories.append(folder)
                write_file_atomic(
                    target,
                    lambda path, relative=relative: shutil.copy2(prepared / relative, path),
                    overwrite=relative in replacements,
                )
                written.append(relative)
            if progress:
                progress(len(files), len(files), "Export complete")
        except BaseException:
            for relative in reversed(written):
                target = destination / relative
                if relative in replacements:
                    write_file_atomic(
                        target,
                        lambda path, relative=relative: shutil.copy2(backups / relative, path),
                        overwrite=True,
                    )
                else:
                    target.unlink()
            for folder in reversed(created_directories):
                folder.rmdir()
            raise
    return destination
