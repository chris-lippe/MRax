"""Complete workspace packages, including fits, transforms and drift results."""

from __future__ import annotations

import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from mrax.gui.session import GuiSession
from mrax.io.bundle import extract_bundle, write_bundle

from .cloning import clone_workspace
from .package import _mrax_package_entries, _write_mrax_package_manifest
from .registration_state import _restore_drift_results
from .scene import validate_scenes
from .source_protection import check_workspace_output
from .state import _restore_gui_state
from .view_data import json_value
from .workspace import StudyWorkspace


def _inside(root, value):
    candidate = (Path(root) / str(value).replace("\\", "/")).resolve()
    if not candidate.is_relative_to(Path(root).resolve()):
        raise ValueError("A workspace reference leaves its package")
    return candidate


def save_workspace_folder(workspace, root):
    from .persistence import save_session_state

    root = Path(root)
    from mrax.io.output_guard import require_empty_directory

    require_empty_directory(root)
    root.mkdir(parents=True, exist_ok=True)
    session = workspace.session
    if not session.is_loaded:
        raise ValueError("Add an experiment before saving this study")
    # ISMRMRD images have at most four axes. Store larger processed buffers
    # and processing masks losslessly in the workflow archive.
    portable = clone_workspace(workspace).session
    supplemental = {}
    for name, row in (portable.data or {}).get("experiments", {}).items():
        for key, value in list((row.get("processed") or {}).items()):
            if hasattr(value, "shape") and (
                np.asarray(value).ndim > 4 or key.startswith("processing_mask")
            ):
                supplemental.setdefault(name, {})[key] = np.asarray(value)
                row["processed"].pop(key)
    if portable.study is not None:
        for name, _experiment, _model, dataset in portable.study.iter_entries():
            dataset.processed = {
                k: v
                for k, v in (dataset.processed or {}).items()
                if k not in supplemental.get(name, {})
            }
    manifest = portable.save_imported_study(
        root / "study", study_name=session.study_name, bundle=False
    )
    state_path = save_session_state(
        session,
        root / "mrax_gui_state.json",
        preview_results=workspace.preview_results_by_experiment,
        fit_results=workspace.fit_results_by_experiment,
        drift_results=workspace.drift_results,
        study_manifest=manifest,
    )
    entries = {
        "study_manifest": manifest.relative_to(root).as_posix(),
        "gui_state": state_path.name,
    }
    arrays = {}
    registrations = {}
    for index, (key, result) in enumerate(workspace.registration_results.items()):
        row = {
            k: v
            for k, v in result.items()
            if k not in {"fixed", "moving", "corrected", "transform"}
        }
        for name in ("fixed", "moving", "corrected"):
            array_key = f"registration_{index}_{name}"
            arrays[array_key] = np.asarray(result[name])
            row[name + "_array"] = array_key
        transform = result.get("transform")
        if transform is not None:
            import SimpleITK as sitk

            transform_path = root / "transforms" / f"transform_{index:04d}.tfm"
            transform_path.parent.mkdir(exist_ok=True)
            if transform.GetName() == "CompositeTransform":
                transform = sitk.CompositeTransform(transform)
                transform.FlattenTransform()
            sitk.WriteTransform(transform, str(transform_path))
            row["transform_file"] = transform_path.relative_to(root).as_posix()
        registrations[str(key)] = row
    processed_arrays = {}
    for name, values in supplemental.items():
        processed_arrays[name] = {}
        for field, value in values.items():
            array_key = f"processed_{len(arrays)}"
            arrays[array_key] = value
            processed_arrays[name][field] = array_key
    from .pipeline import encode_pipelines

    pipelines = encode_pipelines(workspace, arrays)
    np.savez_compressed(root / "workflow_arrays.npz", **arrays)
    fits = {}
    for name, result in workspace.fit_results_by_experiment.items():
        fits[name] = {k: v for k, v in result.items() if k != "fields"}
    workflow = {
        "format": "mrax.gui.workflow",
        "version": 1,
        "view_label": workspace.view_label,
        "plot_axis": workspace.plot_x_axis_key,
        "display_settings": workspace.display_settings,
        "fits": fits,
        "registrations": registrations,
        "diagnostics": workspace.diagnostics[-200:],
        "array_file": "workflow_arrays.npz",
        "processed_arrays": processed_arrays,
        "pipelines": pipelines,
        "reconstruction_settings": {
            name: {
                key: value
                for key, value in (entry.get("kspace_metadata") or {}).items()
                if key.startswith("reconstruction_") or key == "output_product"
            }
            for name, entry in (session.data or {}).get("experiments", {}).items()
            if (entry.get("kspace_metadata") or {}).get("reconstruction_pipeline")
        },
        "source_kind": session.source_kind,
        "import_metadata": session.import_metadata,
        "source_files": workspace.source_files,
    }
    (root / "workflow.json").write_text(
        json.dumps(json_value(workflow), allow_nan=False, indent=2), encoding="utf-8"
    )
    entries["workflow"] = "workflow.json"
    _write_mrax_package_manifest(root, entries)
    return root / "mrax_package.json"


def validate_workspace_replacement(path):
    """Only a native MRax workspace can be replaced by Save workspace."""
    path = Path(path)
    try:
        with zipfile.ZipFile(path) as archive:
            name = (
                "mrax_package.json"
                if "mrax_package.json" in archive.namelist()
                else "study_manifest.json"
            )
            entry = archive.getinfo(name)
            if entry.file_size > 1024**2:
                raise ValueError("Invalid workspace manifest")
            content = archive.read(entry)
        payload = json.loads(content)
        if not isinstance(payload, dict) or not (
            (payload.get("format") == "mrax.gui.package" and payload.get("version") == 1)
            or (payload.get("format") == "mrax.study" and payload.get("format_version") == 1)
        ):
            raise ValueError("Invalid workspace manifest")
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        raise ValueError("Choose a new name. This destination is not an MRax workspace") from exc


def save_workspace_bundle(workspace, path, *, progress_callback=None, overwrite=False):
    check_workspace_output(workspace, path)
    if Path(path).exists() and overwrite:
        validate_workspace_replacement(path)
    with tempfile.TemporaryDirectory(prefix="mrax_workspace_") as temporary:
        save_workspace_folder(workspace, Path(temporary))
        return write_bundle(
            path, temporary, progress_callback=progress_callback, overwrite=overwrite
        )


def load_workspace_root(root, *, key):
    root = Path(root).resolve()
    entries = _mrax_package_entries(root)
    manifest = entries.get("study_manifest")
    if manifest is None:
        raise ValueError("The package has no study manifest")
    session = GuiSession.from_manifest(manifest)
    workspace = StudyWorkspace(key=key, label=session.study_name, session=session)
    state_path = entries.get("gui_state")
    if state_path:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        arrays_path = _inside(
            root, state_path.relative_to(root).parent / str(payload.get("array_file", ""))
        )
        if not arrays_path.is_file():
            raise ValueError("The state array reference is missing")
        with np.load(arrays_path, allow_pickle=False) as archive:
            arrays = dict(archive)
        warnings = []
        _restore_gui_state(session, payload, arrays, warnings=warnings)
        workspace.drift_results = _restore_drift_results(
            payload.get("drift_results") or {}, arrays, warnings=warnings
        )
        workspace.preview_results_by_experiment = dict(payload.get("preview_results") or {})
        for message in warnings:
            workspace.diagnostics.append(
                {"level": "warning", "operation": "load", "message": message}
            )
    workflow_path = entries.get("workflow")
    if workflow_path:
        workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
        if workflow.get("format") != "mrax.gui.workflow" or workflow.get("version") != 1:
            raise ValueError("Unsupported workflow format or version")
        arrays_path = _inside(root, workflow.get("array_file", "workflow_arrays.npz"))
        with np.load(arrays_path, allow_pickle=False) as archive:
            arrays = dict(archive)
        for name, fields in (workflow.get("processed_arrays") or {}).items():
            if name not in session.data["experiments"]:
                raise ValueError("Processed arrays reference an unknown experiment")
            target = session.data["experiments"][name].setdefault("processed", {})
            for field, array_key in fields.items():
                target[field] = arrays[array_key]
            if session.study is not None and name in session.study.names:
                session.study.datasets[list(session.study.names).index(name)].processed.update(
                    target
                )
        # Keep the cached reconstruction's recipe even if its processing step is removed.
        reconstructions = workflow.get("reconstruction_settings") or {}
        if not isinstance(reconstructions, dict):
            raise ValueError("Invalid reconstruction settings in workspace")
        for name, settings in reconstructions.items():
            if name not in session.data["experiments"] or not isinstance(settings, dict):
                raise ValueError("Invalid reconstruction settings in workspace")
            from mrax.reco.plans import normalize_plan

            if settings.get("reconstruction_pipeline") is not None:
                settings["reconstruction_pipeline"] = normalize_plan(
                    settings["reconstruction_pipeline"]
                )
            entry = session.data["experiments"][name]
            entry["kspace_metadata"] = {**(entry.get("kspace_metadata") or {}), **settings}
            if session.study is not None and name in session.study.names:
                dataset = session.study.datasets[list(session.study.names).index(name)]
                dataset.kspace_metadata = dict(entry["kspace_metadata"])
        for name, row in dict(workflow.get("registrations") or {}).items():
            result = {
                k: v for k, v in row.items() if not k.endswith("_array") and k != "transform_file"
            }
            for field in ("fixed", "moving", "corrected"):
                result[field] = arrays[row[field + "_array"]]
            if row.get("transform_file"):
                import SimpleITK as sitk

                result["transform"] = sitk.ReadTransform(str(_inside(root, row["transform_file"])))
            workspace.registration_results[name] = result
        workspace.display_settings = dict(workflow.get("display_settings") or {})
        references = workflow.get("source_files") or []
        if not isinstance(references, list) or any(
            not isinstance(row, dict)
            or not isinstance(row.get("path"), str)
            or not row["path"]
            or not isinstance(row.get("directory"), bool)
            for row in references
        ):
            raise ValueError("Invalid workspace source references")
        workspace.source_files = references
        source_kind = workflow.get("source_kind", session.source_kind)
        if source_kind not in {"bruker", "pulseq", "dicom", "memory", "manifest"}:
            raise ValueError("Unknown workspace source kind")
        session.source_kind = source_kind
        if workflow.get("import_metadata"):
            session.import_metadata = dict(workflow["import_metadata"])
        workspace.view_label = str(workflow.get("view_label") or "Image")
        workspace.plot_x_axis_key = str(workflow.get("plot_axis") or "auto")
        workspace.diagnostics.extend(list(workflow.get("diagnostics") or [])[-200:])
        from .pipeline import decode_pipelines, install

        for name, row in dict(workflow.get("fits") or {}).items():
            if isinstance(row.get("workflow"), dict):
                row = {**row, "workflow": decode_pipelines(row["workflow"], arrays)}
            fields = ((session.data or {}).get("experiments", {}).get(name, {})).get(
                "fitted_maps", {}
            )
            workspace.fit_results_by_experiment[name] = {**row, "fields": fields}
            if row.get("pipeline_grid"):
                session.data["experiments"][name]["fit_grid"] = row["pipeline_grid"]
        workspace.pipelines = decode_pipelines(workflow.get("pipelines", {}), arrays)
        selected = session.selected_experiment
        for name, pipeline in workspace.pipelines.items():
            if name not in session.experiment_names or len(pipeline["steps"]) > 64:
                raise ValueError("Invalid experiment pipeline")
            session.select_experiment(name)
            session.current_experiment["original_mr_image"] = pipeline["source"]["data"]
            session.current_experiment["pipeline_source_grid"] = {
                "dims": dict(pipeline["source"]["dims"]),
                "geometry": pipeline["source"]["geometry"],
            }
            install(workspace, pipeline["output"] if pipeline["enabled"] else pipeline["source"])
            session._save_current_view_state()
        if selected is not None:
            session.select_experiment(selected)
        validate_scenes(workspace)
    return workspace


def load_workspace_bundle(path, extraction_root, *, key):
    return load_workspace_root(extract_bundle(path, extraction_root), key=key)


def load_workspace_path(path, *, key, extraction_root, reference_root=None):
    path = Path(path)
    if path.is_dir():
        return load_workspace_root(path, key=key)
    if path.suffix.lower() == ".mrax":
        return load_workspace_bundle(path, extraction_root, key=key)
    if path.name == "mrax_package.json":
        return load_workspace_root(path.parent, key=key)
    if path.name == "study_manifest.json":
        session = GuiSession.from_manifest(path)
        return StudyWorkspace(key=key, label=session.study_name, session=session)
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("format") != "mrax.gui.state":
            raise ValueError("Choose an MRax study or state")
        boundary = Path(reference_root).resolve() if reference_root else path.parent
        manifest = _inside(boundary, path.parent / payload.get("data_manifest", ""))
        session = GuiSession.from_manifest(manifest)
        with np.load(
            _inside(boundary, path.parent / payload.get("array_file", "")), allow_pickle=False
        ) as archive:
            arrays = dict(archive)
        messages = []
        _restore_gui_state(session, payload, arrays, warnings=messages)
        workspace = StudyWorkspace(key=key, label=session.study_name, session=session)
        workspace.preview_results_by_experiment = dict(payload.get("preview_results") or {})
        workspace.drift_results = _restore_drift_results(
            payload.get("drift_results") or {}, arrays, warnings=messages
        )
        workspace.diagnostics = [
            {"level": "warning", "operation": "load", "message": m} for m in messages
        ]
        return workspace
    raise ValueError("Unsupported workspace source")
