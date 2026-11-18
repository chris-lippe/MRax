"""Brief source descriptions without probing source folders during rendering."""

from pathlib import PurePosixPath


def _path(value):
    return str(value).replace("\\", "/") if value else None


def pipeline_sources(workspace):
    session = workspace.session
    if not session.is_loaded:
        return {}
    name = session.selected_experiment
    saved = (workspace.pipelines.get(name) or {}).get("sources")
    if saved:
        return saved
    entry = session.current_experiment
    imported = ((session.import_metadata or {}).get("experiments") or {}).get(name, {})
    metadata = {**imported, **(entry.get("extra_metadata") or {})}
    source = ((session.data or {}).get("manifest") or {}).get("source") or {}
    kind = session.source_kind or source.get("kind")
    image_path = _path(metadata.get("image_source") or metadata.get("source_path"))
    raw_path = _path(metadata.get("raw_source"))
    format_name = metadata.get("source_format")
    if metadata.get("path") and (kind == "bruker" or metadata.get("visu")):
        folder = PurePosixPath(_path(metadata["path"]))
        image_path = image_path or str(
            folder / "pdata" / str(metadata.get("pdata_index", "1")) / "2dseq"
        )
        format_name = format_name or "Bruker"
        raw_path = raw_path or str(folder)
    format_name = format_name or {
        "bruker": "Bruker",
        "pulseq": "Pulseq",
        "dicom": "DICOM",
        "nifti": "NIfTI",
        "numpy": "NumPy",
        "raw": "Raw binary",
        "blank": "Blank array",
    }.get(metadata.get("import_format") or kind)
    if image_path is None and source.get("kind") not in {None, "memory"}:
        image_path = _path(source.get("path"))
    if image_path is None and workspace.source_files:
        reference = next(
            (
                row
                for row in workspace.source_files
                if not str(row["path"]).lower().endswith(".mrax")
            ),
            workspace.source_files[0],
        )
        image_path = _path(reference["path"])
        if not format_name and image_path.lower().endswith(".mrax"):
            format_name = "MRax workspace"
    if not format_name:
        geometry = metadata.get("spatial_geometry") or {}
        format_name = "Synthetic" if geometry.get("source") == "synthetic" else "In-memory data"

    def describe(source_kind, path):
        if path:
            parts = PurePosixPath(path).parts
            count = (
                5
                if format_name == "Bruker" and source_kind == "image"
                else 3
                if format_name == "Bruker"
                else 2
            )
            location = "/".join(parts[-count:])
        else:
            location = (
                "Generated data"
                if format_name in {"Synthetic", "Blank array"}
                else "No source file recorded"
            )
        return {"kind": source_kind, "format": format_name, "location": location, "path": path}

    return {
        "image": describe("image", image_path),
        "kspace": describe("kspace", raw_path or image_path),
    }
