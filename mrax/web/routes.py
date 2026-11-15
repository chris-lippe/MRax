"""Workspace editing, file access, analysis jobs and exports."""

from __future__ import annotations

import dataclasses
import json
import shutil
import zipfile
from pathlib import Path
from uuid import uuid4

import numpy as np
from fastapi import HTTPException, Query, Request
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from mrax.devices import get_available_devices
from mrax.gui.masks import import_mask, load_roi_set
from mrax.gui.models import (
    model_available_pools,
    model_choices,
    model_default_pools,
    model_fit_inputs,
    model_parameter_specs,
)
from mrax.gui.registration import DriftCorrectionOptions, RegistrationOptions, registration_presets
from mrax.gui.services.analysis_workflows import set_fit_settings, transform_roi, update_metadata
from mrax.gui.services.cloning import clone_workspace
from mrax.gui.services.export import (
    _export_object_records_for_workspace,
    _single_export_file_types,
    _single_export_record_metadata,
)
from mrax.gui.services.export_workflow import export_destination_options
from mrax.gui.services.fit_setup import (
    apply_fit_exclusions,
    check_fit_readiness,
    edit_pool,
    fitting_dimensions,
    fitting_frame_page,
)
from mrax.gui.services.mask_preview import native_mask_preview
from mrax.gui.services.noise_estimation import preview_noise
from mrax.gui.services.processing import processing_catalog, rename_processing_sources
from mrax.gui.services.roi_tools import remember_rois, roi_command
from mrax.gui.services.scene import rename_scene_roi, show_registration_layer
from mrax.gui.services.source_protection import remember_source, source_guard
from mrax.gui.services.timecourse import analyze_timecourse
from mrax.gui.services.view_data import VIEWS, json_value, render_image
from mrax.gui.session import FrameMask, VisualViewState

from .array_headers import read_array_header
from .info import workspace_info
from .schemas import (
    ActionCommand,
    FitCheckCommand,
    FitReadinessResult,
    ImportCommand,
    JobInfo,
    MaskPreview,
    NoiseEstimateCommand,
    ProcessingCatalog,
    TaskCommand,
    TimecourseCommand,
    TimecourseResult,
    WorkspaceInfo,
)


def register_routes(app, get_record, check_revision, add_record):
    def source_path(options):
        if options.get("upload"):
            path = app.state.uploads.get(options["upload"])
            if path is None:
                raise ValueError("Upload expired. Choose the file again")
        else:
            path = app.state.folders.checked_tree(options.get("root"), options.get("path", ""))
        app.state.folders.sources.protect(path)
        return path

    @app.get("/api/catalog")
    def catalog():
        devices = [{"id": f"{d.platform}:{d.id}", "label": str(d)} for d in get_available_devices()]
        return {
            "views": VIEWS,
            "models": [{"key": m.key, "label": m.label} for m in model_choices()],
            "devices": devices,
            "registration": dataclasses.asdict(RegistrationOptions()),
            "registration_presets": registration_presets(),
            "drift": dataclasses.asdict(DriftCorrectionOptions()),
        }

    @app.get("/api/files/roots")
    def roots():
        return app.state.folders.catalog()

    @app.get("/api/files/list")
    def files(
        root: str,
        path: str = "",
        offset: int = Query(0, ge=0),
        search: str = "",
    ):
        return app.state.folders.listing(root, path, offset, search=search)

    @app.post("/api/files/upload")
    async def upload(request: Request, filename: str = "upload.npy", unpack: bool = False):
        from .app import MAX_EXPANDED_BYTES, MAX_UPLOAD_BYTES

        key = uuid4().hex
        folder = app.state.root / "uploads" / key
        folder.mkdir(parents=True)
        name = Path(filename.replace("\\", "/")).name
        if not name or name in {".", ".."}:
            raise ValueError("Choose a file name")
        path = folder / name
        size = 0
        try:
            with path.open("wb") as output:
                async for block in request.stream():
                    size += len(block)
                    if size > MAX_UPLOAD_BYTES:
                        raise HTTPException(
                            413, "Upload exceeds 2 GiB. Connect a local folder instead"
                        )
                    output.write(block)
            if unpack:

                def extract():
                    target = folder / "extracted"
                    target.mkdir()
                    with zipfile.ZipFile(path) as archive:
                        items = archive.infolist()
                        if (
                            len(items) > 200000
                            or sum(item.file_size for item in items) > MAX_EXPANDED_BYTES
                        ):
                            raise ValueError("Expanded archive exceeds 8 GiB")
                        for item in items:
                            relative = Path(item.filename.replace("\\", "/"))
                            destination = (target / relative).resolve()
                            if (
                                relative.is_absolute()
                                or ".." in relative.parts
                                or not destination.is_relative_to(target)
                                or (item.external_attr >> 16) & 0o170000 == 0o120000
                            ):
                                raise ValueError("Unsafe archive entry")
                        archive.extractall(target)
                    children = list(target.iterdir())
                    return children[0] if len(children) == 1 and children[0].is_dir() else target

                path = await run_in_threadpool(extract)
            app.state.uploads[key] = path
            return {"id": key, "name": name, "bytes": size, "directory": path.is_dir()}
        except Exception:
            shutil.rmtree(folder)
            raise

    @app.post("/api/import", response_model=JobInfo)
    def import_study(command: ImportCommand):
        source = (
            None if command.options.get("kind") == "blank" else source_path(command.model_dump())
        )
        if command.workspace_id:
            record = get_record(command.workspace_id)
        else:
            from mrax.gui.services.workspace import StudyWorkspace
            from mrax.gui.session import GuiSession

            session = GuiSession()
            session.create_empty_study("New study")
            key = uuid4().hex
            add_record(StudyWorkspace(key=key, label=session.study_name, session=session))
            record = get_record(key)
        with record.lock:
            if command.workspace_id:
                if command.revision is None:
                    raise ValueError("Refresh the study before importing")
                check_revision(record, command.revision)
            # Resolved source is supplied by this endpoint, never by worker options.
            options = {**command.options, "source": None if source is None else str(source)}
            boundary = (
                app.state.folders.items[command.root]["path"]
                if command.root and not command.upload
                else source
                if source and source.is_dir()
                else source.parent
                if source
                else None
            )
            options["reference_root"] = None if boundary is None else str(boundary)
            return app.state.submit(record, "import", options)

    @app.post("/api/files/preview")
    def preview_source(command: ImportCommand):
        if command.options.get("kind") == "blank":
            return {"kind": "blank", "message": "Declare the shape, dtype and dimension order"}
        path = source_path(command.model_dump())
        from mrax.gui.services.importing import _detect_bruker_import_summary, _detect_import_vendor

        kind = command.options.get("kind", "auto")
        if kind == "auto" and path.is_dir():
            kind = _detect_import_vendor(path).lower()
        if kind == "bruker":
            from mrax.io.bruker_frequency import consistent_bruker_frequency

            rows = _detect_bruker_import_summary(path, command.options.get("pdata_index") or "1")
            return json_value(
                {
                    "kind": kind,
                    "experiments": rows,
                    "field_strength_mhz": consistent_bruker_frequency(row["field_strength_mhz"] for row in rows if row["importable"]),
                }
            )
        if kind == "pulseq":
            from mrax.gui.pulseq_import import plan_pulseq_import

            return json_value(plan_pulseq_import(path).summary)
        if path.suffix.lower() == ".npz":
            arrays = []
            with zipfile.ZipFile(path) as archive:
                for member in archive.infolist():
                    if not member.filename.endswith(".npy"):
                        continue
                    with archive.open(member) as stream:
                        shape, _fortran, dtype = read_array_header(stream)
                    if dtype.hasobject:
                        raise ValueError("Object arrays are not supported")
                    arrays.append(
                        {"key": member.filename[:-4], "shape": list(shape), "dtype": str(dtype)}
                    )
            return {"kind": "numpy", "arrays": arrays}
        if path.suffix.lower() == ".npy":
            value = np.load(path, allow_pickle=False, mmap_mode="r")
            return {"kind": "numpy", "shape": list(value.shape), "dtype": str(value.dtype)}
        return {
            "kind": kind,
            "directory": path.is_dir(),
            "bytes": None if path.is_dir() else path.stat().st_size,
        }

    @app.post("/api/workspaces/{key}/actions", response_model=WorkspaceInfo)
    def action(key: str, command: ActionCommand):
        record = get_record(key)
        with record.lock:
            check_revision(record, command.revision, command.view_revision)
            name, options = command.action, command.options
            workspace = clone_workspace(
                record.workspace,
                share_sources=name.startswith("roi.") or name in {"fit.settings", "fit.pool", "frames.exclude"},
            )
            session = workspace.session
            if name.startswith("roi."):
                operation = name.removeprefix("roi.")
                if operation in {"create", "geometry", "brush", "draw"}:
                    if command.view_revision is None:
                        raise ValueError("ROI drawing requires the displayed view revision")
                    _, editable = render_image(workspace)
                    if not editable:
                        raise ValueError("Draw ROIs on a native image or map grid")
                if operation == "select":
                    if session.roi(options.get("name")) is None:
                        raise ValueError("Select an enabled ROI")
                    session.fit_settings.selected_roi = options["name"]
                elif operation == "transform":
                    remember_rois(workspace)
                    transform_roi(workspace, options)
                elif operation == "load":
                    path = source_path(options)
                    remember_source(workspace, path)
                    if path.is_dir():
                        matches = list(path.rglob("*.json"))
                        path = next(
                            (p for p in matches if "rois" in json.loads(p.read_text())), path
                        )
                    if options.get("mask"):
                        roi = import_mask(
                            path,
                            name=str(options.get("name") or path.stem),
                            dims=None
                            if session.slice_packages
                            else tuple(session.spatial_dim_names),
                        )
                        rois = [roi]
                    else:
                        sidecar = path.with_suffix(".json")
                        if sidecar.is_file():
                            payload = json.loads(sidecar.read_text())
                            linked = (sidecar.parent / str(payload.get("npz", ""))).resolve()
                            if linked.parent != sidecar.parent.resolve():
                                raise ValueError("ROI data must be beside its sidecar")
                        rois = load_roi_set(path)
                    rois = [session.imported_roi(roi) for roi in rois]
                    if any(r.name in [v.name for v in session.shared_rois] for r in rois):
                        raise ValueError("An ROI with this name already exists")
                    remember_rois(workspace)
                    for roi in rois:
                        from mrax.gui.services.pipeline import project_roi

                        roi = project_roi(workspace, roi, inverse=True)
                        session.add_roi(roi)
                else:
                    roi_command(workspace, operation, options)
                if operation != "consolidate":
                    workspace.preview_results_by_experiment.clear()
            elif name == "fit.settings":
                values = dict(options)
                exclusions = values.pop("excluded_frames", None)
                set_fit_settings(workspace, values)
                if exclusions is not None:
                    apply_fit_exclusions(session, exclusions)
            elif name == "fit.pool":
                edit_pool(workspace, options)
            elif name == "frames.exclude":
                excluded = [int(v) for v in options.get("indices", [])]
                if any(not 0 <= v < session.frame_count for v in excluded):
                    raise ValueError("Excluded frame index is out of range")
                mask = np.ones(session.frame_count, dtype=bool)
                mask[excluded] = False
                if not np.any(mask):
                    raise ValueError("Keep at least one included frame")
                session.frame_mask = FrameMask(mask)
                workspace.preview_results_by_experiment.clear()
            elif name == "metadata.update":
                update_metadata(workspace, options)
            elif name == "study.globals":
                from mrax.gui.services.importing import _observables_from_json_text

                values = _observables_from_json_text(json.dumps(options))
                session.data["study_globals"] = values
                if session.study is not None:
                    session.study.study_globals = dict(values)
            elif name == "study.rename":
                session.rename_study(str(options.get("name", "")))
                workspace.label = session.study_name
            elif name == "experiment.rename":
                old = session.selected_experiment
                new = session.rename_experiment(old, str(options.get("name", "")))
                for table in (
                    workspace.fit_results_by_experiment,
                    workspace.preview_results_by_experiment,
                    workspace.display_settings,
                    workspace.visual_undo_stacks,
                    workspace.pipelines,
                ):
                    if old in table:
                        table[new] = table.pop(old)
                if new in workspace.fit_results_by_experiment:
                    workspace.fit_results_by_experiment[new]["name"] = new
                for row in [
                    *workspace.registration_results.values(),
                    *workspace.drift_results.values(),
                ]:
                    for metadata in (
                        row.get("metadata") or {},
                        (row.get("summary") or {}).get("metadata") or {},
                    ):
                        for field in ("reference_experiment", "moving_experiment", "experiment"):
                            if metadata.get(field) == old:
                                metadata[field] = new
            elif name == "registration.show":
                show_registration_layer(workspace, options.get("key"))
                render_image(workspace)
            elif name.startswith("visual."):
                stack = workspace.visual_undo_stacks.setdefault(session.selected_experiment, [])
                if name == "visual.undo":
                    if not stack:
                        raise ValueError("There is no view modification to undo")
                    session.visual_state = stack.pop()
                else:
                    stack.append(dataclasses.replace(session.visual_state))
                    del stack[:-20]
                    if name == "visual.reset":
                        session.visual_state = VisualViewState()
                    elif name == "visual.restore_source":
                        session.restore_original_image_data()
                        session.visual_state = VisualViewState()
                    elif name == "visual.update":
                        allowed = {f.name for f in dataclasses.fields(VisualViewState)}
                        if set(options) - allowed:
                            raise ValueError("Unknown visual setting")
                        state = dataclasses.replace(session.visual_state, **options)
                        if (
                            state.transform_key
                            and state.transform_key not in workspace.registration_results
                        ):
                            raise ValueError("Choose a stored transform")
                        if state.drift_key and state.drift_key not in workspace.drift_results:
                            raise ValueError("Choose a stored drift result")
                        if state.transform_direction not in {
                            "moving_to_reference",
                            "reference_to_moving",
                        }:
                            raise ValueError("Unknown transform direction")
                        if any(
                            not 0 <= float(getattr(state, k)) <= 1
                            for k in ("color_alpha", "reference_alpha", "moving_alpha")
                        ):
                            raise ValueError("Opacity must be between zero and one")
                        session.visual_state = state
                    else:
                        raise ValueError("Unknown view modification")
                render_image(workspace)
            else:
                raise ValueError("Unknown workspace action")
            if name == "roi.rename":
                rename_scene_roi(
                    workspace, options.get("name"), options.get("new_name", "").strip()
                )
            elif name == "roi.remove":
                rename_scene_roi(workspace, options.get("name"))
            elif name == "experiment.rename":
                for settings in workspace.display_settings.values():
                    for layer in settings.get("scene", {}).get("layers", []):
                        if layer.get("experiment") == old:
                            layer["experiment"] = new
                    comparison = settings.get("scene", {}).get("comparison", {})
                    for side in ("left", "right"):
                        if comparison.get(side, {}).get("experiment") == old:
                            comparison[side]["experiment"] = new
                rename_processing_sources(workspace, old, new)
            workspace.session._save_current_view_state()
            if name in {"fit.settings", "fit.pool", "metadata.update", "frames.exclude", "study.globals"}:
                workspace.preview_results_by_experiment.pop(
                    workspace.session.selected_experiment, None
                )
            if (name.startswith("roi.") and name != "roi.consolidate") or name in {
                "fit.settings",
                "fit.pool",
                "metadata.update",
                "frames.exclude",
                "study.globals",
            }:
                for result in workspace.fit_results_by_experiment.values():
                    result["inputs_changed"] = True
            workspace.dirty = True
            # Validate the entire DTO before committing any candidate edits.
            from .app import Record

            result = WorkspaceInfo.model_validate(
                workspace_info(
                    Record(
                        workspace,
                        record.revision + 1,
                        record.view_revision + 1,
                        file_location=record.file_location,
                    )
                )
            )
            record.workspace = workspace
            record.revision += 1
            record.view_revision += 1
            app.state.checkpoint(record)
            return result

    @app.post("/api/workspaces/{key}/tasks", response_model=JobInfo)
    def task(key: str, command: TaskCommand):
        record = get_record(key)
        with record.lock:
            check_revision(record, command.revision)
            options = dict(command.options)
            if (
                command.operation == "reconstruct"
                and not record.workspace.session.raw_data_loaded()
            ):
                source = Path(
                    str(
                        record.workspace.session.current_experiment.get("extra_metadata", {}).get(
                            "path", ""
                        )
                    )
                ).resolve()
                boundaries = [
                    app.state.root,
                    *(row["path"] for row in app.state.folders.items.values()),
                ]
                if not app.state.folders.system and not any(
                    source.is_relative_to(boundary) for boundary in boundaries
                ):
                    raise ValueError(
                        "The original source folder is outside the available file locations"
                    )
            if command.operation in {"save", "export"}:
                if any(name.startswith("_") for name in options):
                    raise ValueError("Internal export options cannot be supplied by the client")
                if "prepared_export" in options:
                    token = options["prepared_export"]
                    if not isinstance(token, str):
                        raise ValueError("Choose a prepared export from this study")
                    prepared = app.state.jobs.prepared_export(token, key, record.revision)
                    pending = json.loads((prepared / "pending.json").read_text())
                    replacements = options.get("replace_files") or []
                    if not isinstance(replacements, list) or not all(
                        isinstance(path, str) for path in replacements
                    ):
                        raise ValueError("Choose output files from the reviewed list")
                    options = {
                        **pending["options"],
                        "replace_files": replacements,
                        "_prepared_export_dir": str(prepared),
                        "_prepared_export_id": token,
                    }
                if options.get("root"):
                    destination = app.state.folders.resolve(
                        options["root"], options.get("path", ""), write=True
                    )
                    directory, image_export = export_destination_options(options)
                    if not directory and (
                        destination == destination.parent
                        or destination in {row["path"] for row in app.state.folders.items.values()}
                    ):
                        raise ValueError("Choose a file name")
                    if (
                        options.get("_prepared_export_dir")
                        and str(destination) != options["destination"]
                    ):
                        raise ValueError("The export destination changed. Select the folder again")
                    options["destination"] = str(destination)
                    guard = source_guard(app)
                    if not directory:
                        guard.check(destination, image_export=image_export)
                else:
                    options.pop("destination", None)
            elif any(
                k in options for k in ("source", "destination", "extraction_root", "artifact_root")
            ):
                raise ValueError("Use the file importer to select a source")
            return app.state.submit(record, command.operation, options)

    @app.get("/api/workspaces/{key}/roi-mask", response_model=MaskPreview)
    def preview_roi_mask(
        key: str,
        name: str = Query(min_length=1, max_length=80),
        axes: str | None = Query(default=None, max_length=161),
        slice_index: int = Query(default=0, ge=0),
    ):
        record = get_record(key)
        with record.lock:
            return native_mask_preview(record.workspace.session, name, axes, slice_index)

    @app.get("/api/workspaces/{key}/inspect")
    def inspect(key: str):
        record = get_record(key)
        with record.lock:
            workspace = record.workspace
            session = workspace.session
            rows = _export_object_records_for_workspace(workspace) if session.is_loaded else []
            return json_value(
                {
                    "summary": list(session.summary_by_name.values()),
                    "study_globals": (session.data or {}).get("study_globals", {}),
                    "metadata": {
                        "observables": session.build_experiment().observables,
                        "metadata": session.current_experiment.get("extra_metadata", {}),
                        "exp_type": session.current_experiment.get("exp_type"),
                        "model_name": session.current_model_name,
                    }
                    if session.is_loaded
                    else {},
                    "objects": [
                        {
                            **_single_export_record_metadata(row),
                            "formats": _single_export_file_types(row),
                        }
                        for row in rows
                    ],
                    "diagnostics": [*workspace.diagnostics, *app.state.recovery_errors],
                    "roi_statistics": session.roi_stats()
                    if session.is_loaded and session.map_names
                    else [],
                    "fit": {
                        k: v
                        for k, v in (
                            workspace.fit_results_by_experiment.get(session.selected_experiment)
                            or {}
                        ).items()
                        if k not in {"fields", "spatial_geometry"}
                    },
                    "registrations": [
                        {
                            k: v
                            for k, v in r.items()
                            if k not in {"fixed", "moving", "corrected", "transform"}
                        }
                        for r in workspace.registration_results.values()
                    ],
                    "drifts": [
                        {k: v for k, v in r.items() if k != "corrected"}
                        for r in workspace.drift_results.values()
                    ],
                }
            )

    @app.get("/api/workspaces/{key}/processing", response_model=ProcessingCatalog)
    def processing(key: str, input_source: str = "image", package: int | None = None):
        record = get_record(key)
        with record.lock:
            return processing_catalog(record.workspace, input_source, package)

    @app.get("/api/workspaces/{key}/reconstruction")
    def reconstruction(key: str):
        from mrax.gui.services.reconstruction import reconstruction_catalog

        record = get_record(key)
        with record.lock:
            return reconstruction_catalog(record.workspace)

    @app.get("/api/workspaces/{key}/pipeline-blocks")
    def pipeline_blocks(key: str):
        from mrax.gui.services.pipeline_blocks import block_catalog

        record = get_record(key)
        with record.lock:
            return block_catalog(record.workspace)

    @app.post("/api/workspaces/{key}/noise-estimate")
    def noise_estimate(key: str, command: NoiseEstimateCommand):
        record = get_record(key)
        with record.lock:
            check_revision(record, command.revision)
            snapshot = clone_workspace(record.workspace, readonly_arrays=True)
        result = preview_noise(snapshot, command.options)
        with record.lock:
            check_revision(record, command.revision)
        return result

    @app.get("/api/workspaces/{key}/roi-statistics")
    def roi_statistics(key: str, map_name: str | None = None):
        record = get_record(key)
        with record.lock:
            from mrax.gui.services.pipeline import display_session

            session = display_session(record.workspace, fitted=True)
            name = map_name or session.active_map
            if name is None:
                return []
            if name not in session.map_names:
                raise ValueError("Choose an available map for ROI statistics")
            rows = session.roi_stats(
                map_names=[name], stats=("count", "mean", "std", "median", "min", "max")
            )
            for row in rows:
                if row["stat"] != "count":
                    row.setdefault("unit", session.map_unit(name))
            return json_value(rows)

    @app.post("/api/workspaces/{key}/analysis/timecourse", response_model=TimecourseResult)
    def timecourse(key: str, command: TimecourseCommand):
        record = get_record(key)
        with record.lock:
            check_revision(record, command.revision, command.view_revision)
            result = analyze_timecourse(record.workspace, command.model_dump())
            return {
                **result,
                "workspace_id": key,
                "revision": record.revision,
                "view_revision": record.view_revision,
            }

    @app.get("/api/workspaces/{key}/parameters")
    def parameters(key: str, model: str | None = None):
        record = get_record(key)
        with record.lock:
            settings = record.workspace.session.fit_settings
            return json_value(
                model_parameter_specs(
                    model or settings.model_name or record.workspace.session.current_model_name,
                    pools=settings.selected_pools or None,
                    param_overrides=settings.param_overrides or None,
                    pool_overrides=settings.pool_overrides or None,
                    parameter_constant_dims=settings.parameter_constant_dims or None,
                )
            )

    @app.post("/api/workspaces/{key}/fit-check", response_model=FitReadinessResult)
    def fit_check(key: str, command: FitCheckCommand):
        record = get_record(key)
        with record.lock:
            check_revision(record, command.revision, command.view_revision)
            snapshot = clone_workspace(record.workspace, readonly_arrays=True)
        result = check_fit_readiness(snapshot, command.model_dump())
        with record.lock:
            check_revision(record, command.revision, command.view_revision)
        return result

    @app.get("/api/workspaces/{key}/fit-frames")
    def fit_frames(key: str, offset: int = Query(0, ge=0), limit: int = Query(32, ge=1, le=100)):
        record = get_record(key)
        with record.lock:
            return {"revision": record.revision,
                    **fitting_frame_page(record.workspace.session, offset, limit)}

    @app.get("/api/workspaces/{key}/fit-options")
    def fit_options(key: str):
        record = get_record(key)
        with record.lock:
            session = record.workspace.session
            model = session.fit_settings.model_name or session.current_model_name
            inputs = model_fit_inputs(model)
            reference_scan = (
                inputs.reference_scan and session.current_experiment.get("exp_type") == "sat_trans"
            )
            return {
                "workspace_id": key,
                "experiment": session.selected_experiment,
                "model": model,
                "inputs": {
                    "reference_scan": reference_scan,
                    "prior_maps": inputs.prior_maps,
                    "derived_maps": inputs.derived_maps,
                },
                "pools": model_available_pools(model, session.fit_settings.pool_overrides),
                "default_pools": model_default_pools(model),
                "field_dimensions": fitting_dimensions(session, model),
                "reference_scans": session.sat_trans_reference_choices() if reference_scan else [],
                "prior_maps": {
                    kind: session.ab_mt_map_choices(kind) for kind in inputs.prior_maps
                },
            }

    @app.get("/api/jobs/{key}/download")
    def download(key: str):
        try:
            item = app.state.jobs.get(key)
        except KeyError as exc:
            raise HTTPException(404, "Job not found") from exc
        if item["status"] != "completed":
            raise HTTPException(409, "Wait for the export to complete")
        path = app.state.jobs.artifact(key)
        if path is None or not path.is_file():
            raise HTTPException(404, "This operation has no downloadable file")
        return FileResponse(path, filename=path.name, media_type="application/octet-stream")
