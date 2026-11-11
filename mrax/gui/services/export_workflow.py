"""Workspace and object exports to files, folders, or an API archive."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from mrax.gui.masks import save_roi_set
from mrax.io.bundle import write_bundle

from .export import (
    _export_object_records_for_session,
    _single_export_file_types,
    _write_single_export_object,
)
from .file_output import export_file_plan, publish_export_files, write_file_atomic
from .source_protection import check_workspace_output
from .view_data import json_value
from .workspace_io import (
    save_workspace_bundle,
    save_workspace_folder,
    validate_workspace_replacement,
)


def export_destination_options(options):
    kind = str(options.get("kind", "bundle"))
    directory = kind in {"package", "state", "rois"} or (
        kind == "object" and options.get("format") in {"nifti", "dicom", "roi_set"}
    )
    image_export = kind in {"object", "rois"} and options.get("format") not in {"json", "csv"}
    return directory, image_export


def _write_export_files(workspace, options, folder):
    kind = str(options.get("kind", "bundle"))
    if kind in {"package", "state"}:
        # A state file carries its referenced study and arrays for portability.
        save_workspace_folder(workspace, folder)
    elif kind == "rois":
        save_roi_set(workspace.session.shared_rois, folder / "mrax_rois.npz")
    elif kind == "fit":
        fits = workspace.fit_results_by_experiment
        if not fits:
            raise ValueError("Run a full fit before exporting fit results")
        (folder / "fit_results.json").write_text(
            json.dumps(json_value(fits), allow_nan=False, indent=2), encoding="utf-8"
        )
    elif kind == "diagnostics":
        (folder / "diagnostics.json").write_text(
            json.dumps(
                json_value(
                    {
                        "messages": workspace.diagnostics,
                        "registrations": [
                            {
                                k: v
                                for k, v in r.items()
                                if k not in {"fixed", "moving", "corrected", "transform"}
                            }
                            for r in workspace.registration_results.values()
                        ],
                        "drift": [
                            {k: v for k, v in r.items() if k != "corrected"}
                            for r in workspace.drift_results.values()
                        ],
                    }
                ),
                allow_nan=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    elif kind == "object":
        records = _export_object_records_for_session(workspace.session)
        record = next((r for r in records if r["id"] == options.get("object")), None)
        if record is None:
            raise ValueError("Choose an export object")
        format_name = str(options.get("format", "npz"))
        if format_name not in {f["id"] for f in _single_export_file_types(record)}:
            raise ValueError("This format cannot represent the selected object")
        _write_single_export_object(record, folder / "mrax-export", format_name)
    else:
        raise ValueError("Unknown export format")


def _native_metadata(path, *, package=False):
    try:
        if not path.is_file() or path.stat().st_size > 1024**2:
            return False
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return False
        if package:
            return payload.get("format") == "mrax.gui.package" and payload.get("version") == 1
        return "rois" in payload and "npz" in payload and "mrax_version" in payload
    except (OSError, ValueError):
        return False


def export_into_directory(workspace, options, root, progress, *, output_guard=None):
    """Stage once, then add named files to the selected directory."""
    kind = str(options.get("kind", "bundle"))
    destination = Path(options["destination"])
    prepared_root = Path(options.get("_prepared_export_dir") or root)
    prepared = prepared_root / "files"
    pending_path = prepared_root / "pending.json"
    if options.get("_prepared_export_dir"):
        pending = json.loads(pending_path.read_text())
        if pending["workspace"] != workspace.key or pending["options"]["destination"] != str(
            destination
        ):
            raise ValueError("This prepared export belongs to another study or destination")
    else:
        prepared.mkdir()
        _write_export_files(workspace, options, prepared)
        pending = None
    _, image_export = export_destination_options(options)
    native_package = kind in {"package", "state"} and _native_metadata(
        destination / "mrax_package.json", package=True
    )

    def policy(relative, target):
        image_file = image_export and target.suffix.lower() not in {".json", ".csv"}
        try:
            check_workspace_output(workspace, target, image_export=image_file)
            if output_guard:
                output_guard(target, image_export=image_file)
        except ValueError as exc:
            return False, str(exc)
        native_roi = (
            kind in {"rois", "object"}
            and target.suffix.lower() == ".json"
            and _native_metadata(target)
        )
        allowed = image_file or native_package or native_roi
        return (
            allowed,
            ""
            if allowed
            else "Choose another folder to preserve this existing metadata or text file",
        )

    files, conflicts = export_file_plan(prepared, destination, policy=policy)
    if pending is not None:
        previous = {row["path"]: row["identity"] for row in pending["conflicts"]}
        actual = {row["path"]: row["identity"] for row in conflicts}
        if actual != previous or files != pending["files"]:
            raise ValueError("Output files changed after review. Select the folder again")
        requested = options.get("replace_files") or []
        if (
            not isinstance(requested, list)
            or set(requested) != set(actual)
            or any(not row["replaceable"] for row in conflicts)
        ):
            raise ValueError("Confirm exactly the listed, replaceable output files")
    elif conflicts:
        safe_options = {
            key: value
            for key, value in options.items()
            if not key.startswith("_") and key not in {"overwrite", "replace_files"}
        }
        pending_path.write_text(
            json.dumps(
                {
                    "workspace": workspace.key,
                    "options": safe_options,
                    "files": files,
                    "conflicts": conflicts,
                }
            ),
            encoding="utf-8",
        )
        return {
            "saved": False,
            "needs_confirmation": True,
            "kind": kind,
            "path": str(destination),
            "file_count": len(files),
            "conflicts": [
                {key: value for key, value in row.items() if key != "identity"} for row in conflicts
            ],
        }, None
    elif options.get("replace_files"):
        raise ValueError("File replacement requires a prepared export and a reviewed file list")
    replacements = options.get("replace_files") or []

    def validate_write(relative, target):
        allowed, reason = policy(relative, target)
        if relative in replacements and not allowed:
            raise ValueError(reason)

    publish_export_files(
        prepared,
        destination,
        files,
        replace_files=replacements,
        progress=progress,
        expected={row["path"]: row["identity"] for row in conflicts},
        validate=validate_write,
    )
    return {"saved": True, "kind": kind, "path": str(destination), "file_count": len(files)}, None


def export_workspace(workspace, options, root, progress, *, output_guard=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    kind = str(options.get("kind", "bundle"))
    progress(0, 2, "Preparing export")
    if options.get("destination"):
        destination = Path(options["destination"])
        overwrite = bool(options.get("overwrite"))
        directory, image_export = export_destination_options(options)
        if directory:
            return export_into_directory(
                workspace, options, root, progress, output_guard=output_guard
            )
        check_workspace_output(workspace, destination, image_export=image_export)
        if destination.exists() and kind not in {"bundle", "package", "state"} and not image_export:
            raise ValueError(
                "Choose a new file name. Existing metadata and text files are not replaced"
            )
        if kind == "bundle":
            if destination.suffix.lower() != ".mrax":
                raise ValueError("Choose a .mrax file name")
            if destination.exists() and overwrite:
                validate_workspace_replacement(destination)
            write_file_atomic(
                destination,
                lambda path: save_workspace_bundle(workspace, path),
                overwrite=overwrite,
            )
        else:
            suffix = "." + str(options.get("format")) if kind == "object" else ".json"
            if destination.suffix.lower() != suffix:
                raise ValueError("Choose a file name ending in " + suffix)
            prepared = root / "files"
            prepared.mkdir()
            _write_export_files(workspace, options, prepared)
            files = list(prepared.iterdir())
            if len(files) != 1:
                raise ValueError("Choose a folder for this export")
            write_file_atomic(
                destination, lambda path: shutil.copyfile(files[0], path), overwrite=overwrite
            )
        progress(2, 2, "Saved")
        return {"saved": True, "kind": kind, "path": str(destination)}, None
    if kind == "bundle":
        path = save_workspace_bundle(workspace, root / "mrax-workspace.mrax")
    else:
        folder = root / "files"
        folder.mkdir()
        _write_export_files(workspace, options, folder)
        progress(1, 2, "Packing export files")
        path = write_bundle(root / "mrax-export.zip", folder)
    progress(2, 2, "Export ready")
    return {"saved": True, "kind": kind}, Path(path)
