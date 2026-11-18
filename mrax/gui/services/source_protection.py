"""Source provenance survives internal edits and native workspace round trips."""

from pathlib import Path

from mrax.io.output_guard import SourceProtection


def remember_source(workspace, source):
    if source is None:
        return
    path = Path(source).expanduser().resolve()
    reference = {"path": str(path), "directory": path.is_dir()}
    if reference not in workspace.source_files:
        workspace.source_files.append(reference)
    # Protect common metadata sidecars as part of a single-image import.
    if path.is_file():
        sidecar = (
            path.with_name(path.name[:-7] + ".json")
            if path.name.endswith(".nii.gz")
            else path.with_suffix(".json")
        )
        if sidecar != path and sidecar.is_file():
            extra = {"path": str(sidecar), "directory": False}
            if extra not in workspace.source_files:
                workspace.source_files.append(extra)


def source_references(workspace):
    references = list(workspace.source_files)
    data = workspace.session.data or {}
    source = (data.get("manifest") or {}).get("source") or {}
    if source.get("kind") not in {None, "memory"} and source.get("path"):
        path = Path(source["path"]).expanduser().resolve()
        references.append({"path": str(path), "directory": path.is_dir()})
    for row in (data.get("experiments") or {}).values():
        metadata = row.get("extra_metadata") or {}
        if metadata.get("source_path"):
            path = Path(metadata["source_path"]).expanduser().resolve()
            references.append({"path": str(path), "directory": path.is_dir()})
    return references


def check_workspace_output(workspace, destination, *, directory=False, image_export=False):
    guard = SourceProtection()
    for reference in source_references(workspace):
        guard.protect(reference["path"], directory=reference["directory"])
    guard.check(destination, directory=directory, image_export=image_export)


def source_guard(app):
    guard = app.state.folders.sources
    for record in list(app.state.records.values()):
        for reference in source_references(record.workspace):
            guard.protect(reference["path"], directory=reference["directory"])
    return guard


def guard_output(app, destination, *, directory=False, image_export=False):
    source_guard(app).check(destination, directory=directory, image_export=image_export)
