"""Local browser API. Runtime files live outside the checkout; no arbitrary path API."""

from __future__ import annotations

import copy
import json
import secrets
import shutil
import tempfile
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware

from mrax._version import MRAX_VERSION
from mrax.gui.services.cloning import clone_workspace
from mrax.gui.services.comparison import render_comparison
from mrax.gui.services.persistence import load_workspace_bundle, save_workspace_bundle
from mrax.gui.services.roi_tools import roi_command
from mrax.gui.services.scene import add_registration_layer, update_scene
from mrax.gui.services.synthetic import (
    synthetic_multimodal_session,
    synthetic_t2_session,
    synthetic_tutorial_session,
)
from mrax.gui.services.view_data import VIEWS, display_options, render_image
from mrax.gui.services.workspace import StudyWorkspace
from mrax.gui.session import GuiSession

from .filesystem import FileRoots
from .info import image_for_record, workspace_info
from .job_runtime import Jobs
from .schemas import (
    CloseCommand,
    JobInfo,
    NavigationCommand,
    PreviewCommand,
    RectangleCommand,
    ViewCommand,
    WorkspaceInfo,
)

MAX_UPLOAD_BYTES = 2 * 1024**3
MAX_EXPANDED_BYTES = 8 * 1024**3


@dataclass
class Record:
    workspace: StudyWorkspace
    revision: int = 0
    view_revision: int = 0
    lock: threading.RLock = field(default_factory=threading.RLock)
    file_location: dict[str, str] | None = None
    info_cache: dict = field(default_factory=dict, repr=False)


def update_view(workspace, command, *, navigation=False):
    session = workspace.session
    if command.experiment is not None:
        if command.experiment not in session.experiment_names:
            raise ValueError("Unknown experiment")
        display_options(workspace)["base_view"] = workspace.view_label
        display_options(workspace)["plot_axis"] = workspace.plot_x_axis_key
        session.select_experiment(command.experiment)
        workspace.view_label = display_options(workspace).get("base_view", "Image")
        workspace.plot_x_axis_key = display_options(workspace).get("plot_axis", "auto")
    if command.plane is not None:
        if command.plane != "axial" and not session.supports_image_planes():
            raise ValueError("This experiment has no volume grid")
        session.set_image_plane(command.plane)
    if command.view is not None:
        if command.view not in VIEWS:
            raise ValueError("Unknown image view")
        workspace.view_label = command.view
    kinds = (
        (
            ("kspace",)
            if workspace.view_label == "K-space"
            else ("map",)
            if workspace.view_label == "Map"
            else ("image",)
        )
        if navigation
        else ("image", "map", "kspace")
    )
    controls = {
        c["key"]: int(c["size"]) for kind in kinds for c in session.dimension_controls(kind)
    }
    indices = dict(command.indices or {})
    if command.z is not None:
        indices[session.image_slice_dim] = command.z
    for key, value in indices.items():
        size = controls.get(
            key, session.frame_count if key == "frame" else session.experiment_dims.get(key)
        )
        if size is None or not 0 <= value < size:
            raise ValueError("Dimension index is out of range")
        session.set_active_index(key, value)
    if command.frame is not None:
        if not 0 <= command.frame < session.frame_count:
            raise ValueError("Dimension index is out of range")
        session.set_active_frame(command.frame)
    if command.map_name is not None:
        if command.map_name not in session.map_names:
            raise ValueError("Unknown fitted map")
        session.active_map = command.map_name
    if command.reconstruction_component is not None:
        session.set_reconstruction_component(command.reconstruction_component)
    if "coil" in command.model_fields_set:
        session.set_reconstruction_coil(command.coil)
    if command.unwrap_phase is not None:
        session.set_reconstruction_phase_unwrap(command.unwrap_phase)
    if command.kspace_mode is not None:
        if command.kspace_mode not in {"magnitude", "log_magnitude", "phase", "real", "imag"}:
            raise ValueError("Unknown k-space component")
        session.active_kspace_mode = command.kspace_mode
    if command.kspace_centering is not None:
        session.active_kspace_centering = command.kspace_centering
    if command.plot_axis is not None:
        if command.plot_axis != "auto" and command.plot_axis not in session.experiment_dims:
            raise ValueError("Unknown plot axis")
        workspace.plot_x_axis_key = command.plot_axis
    for key in ("component", "colormap"):
        if getattr(command, key) is not None:
            display_options(workspace)[key] = getattr(command, key)
    if command.scene is not None:
        update_scene(workspace, command.scene.model_dump(include=command.scene.model_fields_set))
    if command.selected_roi is not None:
        if session.roi(command.selected_roi) is None:
            raise ValueError("Select an enabled ROI")
        session.fit_settings.selected_roi = command.selected_roi
    display_options(workspace)["base_view"] = workspace.view_label
    display_options(workspace)["plot_axis"] = workspace.plot_x_axis_key
    if navigation:
        state = session.view_states.get(session.selected_experiment)
        if state is not None:
            state = copy.copy(state)
            state.active_frame = session.active_frame
            state.active_z = session.active_z
            state.active_indices = dict(session.active_indices)
            session.view_states[session.selected_experiment] = state
    else:
        session._save_current_view_state()


def create_app(
    *,
    runtime_dir: Path | None = None,
    read_roots=(),
    write_roots=(),
    state_dir: Path | None = None,
    initial_paths=(),
    filesystem_scope="system",
) -> FastAPI:
    records: dict[str, Record] = {}
    registry_lock = threading.RLock()
    token = secrets.token_urlsafe(32)
    folders = FileRoots(read_roots, write_roots, scope=filesystem_scope)
    recovery = None if state_dir is None else Path(state_dir).expanduser().resolve()

    @asynccontextmanager
    async def lifespan(app):
        with tempfile.TemporaryDirectory(prefix="mrax_web_", dir=runtime_dir) as temporary:
            root = Path(temporary)

            app.state.root = root
            app.state.uploads = {}
            app.state.records = records
            app.state.folders = folders
            app.state.recovery_errors = []
            saver = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mrax-recovery")
            checkpoints = {}
            checkpoint_workers = set()
            checkpoint_lock = threading.RLock()

            def checkpoint(record):
                if recovery is None or not record.workspace.session.is_loaded:
                    return
                key = record.workspace.key
                with checkpoint_lock:
                    checkpoints[key] = (record, clone_workspace(record.workspace))
                    if key in checkpoint_workers:
                        return
                    checkpoint_workers.add(key)

                def write():
                    while True:
                        with checkpoint_lock:
                            entry = checkpoints.pop(key, None)
                            if entry is None:
                                checkpoint_workers.discard(key)
                                return
                        owner, snapshot = entry
                        if records.get(key) is not owner:
                            continue
                        try:
                            save_workspace_bundle(
                                snapshot, recovery / f"{key}.mrax", overwrite=True
                            )
                        except Exception:
                            app.state.recovery_errors.append(
                                {"message": "Recovery checkpoint failed; save a workspace bundle"}
                            )

                saver.submit(write)

            def attach(key, revision, result):
                record = records.get(key)
                if record is None:
                    return False
                with record.lock:
                    if record.revision != revision:
                        return False
                    if result.get("_workspace") is not None:
                        workspace = result["_workspace"]
                        workspace.key = key
                        previous = record.workspace
                        if record.view_revision != result.get(
                            "_view_revision", record.view_revision
                        ):
                            session = previous.session
                            if session.selected_experiment in workspace.session.experiment_names:
                                update_view(
                                    workspace,
                                    ViewCommand(
                                        revision=revision,
                                        experiment=session.selected_experiment,
                                        plane=session.active_image_plane,
                                        indices=dict(session.active_indices),
                                        view=previous.view_label,
                                        map_name=session.active_map,
                                        component=display_options(previous).get("component"),
                                        colormap=display_options(previous).get("colormap"),
                                        reconstruction_component=session.active_reconstruction_component,
                                        coil=session.active_reconstruction_coil,
                                        unwrap_phase=session.active_reconstruction_unwrap_phase,
                                        kspace_mode=session.active_kspace_mode,
                                        kspace_centering=session.active_kspace_centering,
                                        plot_axis=previous.plot_x_axis_key,
                                        selected_roi=session.fit_settings.selected_roi,
                                    ),
                                )
                            for name, state in previous.session.view_states.items():
                                restored = workspace.session.view_states.get(name)
                                if restored is None:
                                    continue
                                maps = workspace.session.data["experiments"][name].get(
                                    "fitted_maps", {}
                                )
                                for field_name, value in vars(state).items():
                                    if field_name.startswith("active_") and (
                                        field_name != "active_map" or value in maps
                                    ):
                                        setattr(restored, field_name, copy.deepcopy(value))
                                if any(
                                    roi.name == state.fit_settings.selected_roi
                                    for roi in restored.rois
                                ):
                                    restored.fit_settings.selected_roi = (
                                        state.fit_settings.selected_roi
                                    )
                            # Display edits made during a worker operation belong to the
                            # user. Preserve them, then attach newly produced layers.
                            workspace.display_settings.update(
                                copy.deepcopy(previous.display_settings)
                            )
                            for key in (
                                workspace.registration_results.keys()
                                - previous.registration_results.keys()
                            ):
                                add_registration_layer(workspace, key)
                        workspace.roi_history = previous.roi_history
                        workspace.dirty = True
                        WorkspaceInfo.model_validate(
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
                    elif result.get("operation") == "save":
                        record.workspace.dirty = False
                    checkpoint(record)
                    return True

            def submit(record, operation, options):
                snapshot = clone_workspace(record.workspace, readonly_arrays=True)
                if operation == "import":
                    options = {
                        **options,
                        "extraction_root": str(root / snapshot.key / "imports" / uuid4().hex),
                    }
                options = {
                    **options,
                    "_view_revision": record.view_revision,
                }
                return app.state.jobs.submit(
                    snapshot.key,
                    record.revision,
                    snapshot,
                    operation=operation,
                    options=options,
                )

            app.state.checkpoint = checkpoint
            app.state.checkpoint_delete = lambda key: (
                saver.submit(lambda: (recovery / f"{key}.mrax").unlink(missing_ok=True))
                if recovery
                else None
            )
            app.state.submit = submit

            def failure(key, operation, detail):
                record = records.get(key)
                if record is not None:
                    with record.lock:
                        record.workspace.diagnostics.append(
                            {"level": "error", "operation": operation, "message": detail}
                        )
                        del record.workspace.diagnostics[:-200]
                        checkpoint(record)

            app.state.jobs = Jobs(
                root / "jobs",
                attach,
                on_failure=failure,
                output_guard=folders.sources.check,
            )
            if recovery is not None:
                recovery.mkdir(parents=True, exist_ok=True)
                for path in list(recovery.glob("*.mrax"))[:8]:
                    try:
                        workspace = load_workspace_bundle(
                            path, root / path.stem / "recovered", key=path.stem
                        )
                        workspace.dirty = True
                        records[path.stem] = Record(workspace)
                    except Exception:
                        app.state.recovery_errors.append(
                            {
                                "message": "A recovery file could not be restored; the original file is retained"
                            }
                        )
            try:
                # Remember an editable bundle location without embedding it in the bundle.
                from mrax.gui.services.workspace_io import load_workspace_path

                for path in initial_paths:
                    if len(records) >= 8:
                        raise ValueError("Close a recovered workspace before opening more studies")
                    key = uuid4().hex
                    workspace = load_workspace_path(path, key=key, extraction_root=root / key)
                    location = None
                    if Path(path).suffix.lower() == ".mrax":
                        source = Path(path).resolve()
                        for root_key, row in folders.items.items():
                            if row["writable"] and (
                                folders.system or source.is_relative_to(row["path"])
                            ):
                                location = {
                                    "root": root_key,
                                    "path": folders.location(root_key, source),
                                }
                                break
                    if location is None:
                        from mrax.gui.services.source_protection import remember_source

                        source = Path(path).resolve()
                        source = source if source.is_dir() else source.parent
                        remember_source(workspace, source)
                        folders.sources.protect(source)
                    records[key] = Record(workspace, file_location=location)
                yield
            finally:
                await run_in_threadpool(app.state.jobs.close)
                await run_in_threadpool(saver.shutdown, wait=True)
                records.clear()

    app = FastAPI(title="MRax local browser API", version=MRAX_VERSION, lifespan=lifespan)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "[::1]"])

    @app.middleware("http")
    async def local_access(request: Request, call_next):
        origin = request.headers.get("origin")
        local_origins = {
            str(request.base_url).rstrip("/"),
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        }
        if origin and origin not in local_origins:
            return JSONResponse({"detail": "Origin is not allowed"}, status_code=403)
        if request.url.path.startswith("/api/") and request.method not in {"GET", "HEAD"}:
            if not secrets.compare_digest(request.headers.get("x-mrax-token", ""), token):
                return JSONResponse({"detail": "Invalid local session token"}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(OSError)
    async def file_error(_request, exc):
        if isinstance(exc, PermissionError):
            message = "You do not have permission to open or save this file or folder"
        elif isinstance(exc, FileNotFoundError):
            message = "The selected file or folder no longer exists"
        elif isinstance(exc, FileExistsError):
            message = "A file or folder with this name already exists"
        else:
            message = "Could not open or save this file. Check the location, permissions and available space."
        return JSONResponse({"detail": message}, status_code=422)

    @app.exception_handler(ValueError)
    async def invalid_input(_request, exc):
        return JSONResponse({"detail": str(exc) or "Check the selected input"}, status_code=422)

    def get_record(key: str) -> Record:
        with registry_lock:
            if key not in records:
                raise HTTPException(404, "Workspace not found")
            return records[key]

    def check_revision(record, revision, view_revision=None):
        if record.revision != revision or (
            view_revision is not None and record.view_revision != view_revision
        ):
            raise HTTPException(409, "The workspace changed; refresh before editing")

    def add_record(workspace, *, file_location=None):
        from .schemas import PreviewResult

        for preview in workspace.preview_results_by_experiment.values():
            PreviewResult.model_validate(preview)
        with registry_lock:
            if len(records) >= 8:
                raise HTTPException(409, "Close a study before opening more workspaces")
            record = Record(workspace, file_location=file_location)
            payload = WorkspaceInfo.model_validate(workspace_info(record))
            records[workspace.key] = record
            return payload

    @app.get("/api/health")
    def health():
        return {"status": "ok", "stage": 7, "compute": "MRax devices"}

    @app.get("/api/session")
    def session():
        return {
            "token": token,
            "recovery_enabled": recovery is not None,
            "recovery_errors": app.state.recovery_errors,
        }

    @app.get("/api/workspaces", response_model=list[WorkspaceInfo])
    def workspaces():
        with registry_lock:
            active = list(records.values())
        result = []
        for record in active:
            with record.lock:
                result.append(workspace_info(record))
        return result

    @app.post("/api/workspaces/new", response_model=WorkspaceInfo)
    def new_workspace():
        session = GuiSession()
        session.create_empty_study("New study")
        return add_record(
            StudyWorkspace(key=uuid4().hex, label=session.study_name, session=session)
        )

    @app.post("/api/workspaces/synthetic", response_model=WorkspaceInfo)
    def synthetic(kind: str = "t2"):
        factories = {
            "t2": synthetic_t2_session,
            "multimodal": synthetic_multimodal_session,
            "tutorial": synthetic_tutorial_session,
        }
        if kind not in factories:
            raise ValueError("Unknown synthetic fixture")
        session = factories[kind]()
        workspace = StudyWorkspace(key=uuid4().hex, label=session.study_name, session=session)
        return add_record(workspace)

    @app.post("/api/workspaces/upload", response_model=WorkspaceInfo)
    async def upload(request: Request):
        key = uuid4().hex
        root = app.state.root / key
        root.mkdir()
        path = root / "upload.mrax"
        size = 0
        opened = False
        try:
            with path.open("wb") as output:
                async for block in request.stream():
                    size += len(block)
                    if size > MAX_UPLOAD_BYTES:
                        raise HTTPException(
                            413, "Upload exceeds 2 GiB; connect a local folder instead"
                        )
                    output.write(block)
            with zipfile.ZipFile(path) as archive:
                items = archive.infolist()
                if (
                    len(items) > 200000
                    or sum(item.file_size for item in items) > MAX_EXPANDED_BYTES
                ):
                    raise HTTPException(413, "Expanded bundle exceeds 8 GiB")
            workspace = await run_in_threadpool(
                load_workspace_bundle, path, root / "extracted", key=key
            )
            result = add_record(workspace)
            opened = True
            return result
        except (ValueError, OSError, zipfile.BadZipFile, KeyError) as exc:
            raise HTTPException(
                422, "Invalid or unsupported MRax bundle; no workspace was opened"
            ) from exc
        finally:
            path.unlink(missing_ok=True)
            if not opened:
                shutil.rmtree(root)

    @app.get("/api/workspaces/{key}", response_model=WorkspaceInfo)
    def workspace(key: str):
        record = get_record(key)
        with record.lock:
            return workspace_info(record)

    @app.delete("/api/workspaces/{key}", status_code=204)
    def close_workspace(key: str, command: CloseCommand):
        record = get_record(key)
        with record.lock:
            check_revision(record, command.revision)
            if app.state.jobs.is_active(key):
                raise HTTPException(409, "Wait for or cancel the active fit before closing")
            if record.workspace.dirty and not command.discard:
                raise HTTPException(409, "Save the workspace or explicitly discard unsaved changes")
            with registry_lock:
                records.pop(key, None)
            app.state.checkpoint_delete(key)
            root = app.state.root / key
            if root.is_dir():
                shutil.rmtree(root)
        return Response(status_code=204)

    @app.patch("/api/workspaces/{key}/view", response_model=WorkspaceInfo)
    def view(key: str, command: ViewCommand):
        record = get_record(key)
        with record.lock:
            check_revision(record, command.revision)
            navigation = command.model_fields_set <= {"revision", "frame", "z", "indices"}
            if navigation:
                candidate = copy.copy(record.workspace)
                candidate.session = copy.copy(record.workspace.session)
                candidate.session.active_indices = dict(candidate.session.active_indices)
                candidate.session.view_states = dict(candidate.session.view_states)
                candidate.session._slice_cache = {}
                candidate.display_settings = dict(candidate.display_settings)
                active = candidate.session.selected_experiment or ""
                candidate.display_settings[active] = dict(
                    candidate.display_settings.get(active, {})
                )
            else:
                candidate = clone_workspace(record.workspace)
            update_view(candidate, command, navigation=navigation)
            candidate.dirty = True
            payload = WorkspaceInfo.model_validate(
                workspace_info(
                    Record(
                        candidate,
                        record.revision,
                        record.view_revision + 1,
                        file_location=record.file_location,
                        info_cache=record.info_cache,
                    )
                )
            )
            record.workspace = candidate
            record.view_revision += 1
            if not navigation:
                app.state.checkpoint(record)
            return payload

    @app.post("/api/workspaces/{key}/navigate")
    def navigate(key: str, command: NavigationCommand):
        from .navigation import navigation_response

        record = get_record(key)
        with record.lock:
            check_revision(record, command.revision, command.view_revision)
            return navigation_response(record, command)

    @app.get("/api/workspaces/{key}/slice")
    def image(key: str, layer: str | None = None):
        record = get_record(key)
        with record.lock:
            array, _ = image_for_record(record, layer)
            dtype = "<f8" if array.dtype.itemsize > 4 else "<f4"
            data = np.asarray(array, dtype=dtype)
            return Response(
                data.tobytes(order="C"),
                media_type="application/octet-stream",
                headers={
                    "X-Shape": json.dumps(list(data.shape)),
                    "X-Dtype": "float64-le" if dtype == "<f8" else "float32-le",
                    "X-View-Revision": str(record.view_revision),
                    "X-Revision": str(record.revision),
                },
            )

    @app.get("/api/workspaces/{key}/comparison")
    def comparison(key: str):
        record = get_record(key)
        with record.lock:
            arrays, metadata = render_comparison(record.workspace)
            return Response(
                b"".join(array.tobytes(order="C") for array in arrays),
                media_type="application/octet-stream",
                headers={
                    "X-Comparison": json.dumps(metadata, ensure_ascii=True),
                    "X-View-Revision": str(record.view_revision),
                    "X-Revision": str(record.revision),
                },
            )

    @app.post("/api/workspaces/{key}/rois", response_model=WorkspaceInfo)
    def rectangle(key: str, command: RectangleCommand):
        record = get_record(key)
        with record.lock:
            check_revision(record, command.revision, command.view_revision)
            candidate = clone_workspace(record.workspace)
            _, editable = render_image(candidate)
            if not editable:
                raise ValueError("Draw ROIs on a native image or map grid")
            roi_command(
                candidate,
                "create",
                {"name": command.name, "kind": "rectangle", "bounds": command.bounds},
            )
            candidate.session.fit_settings.selected_roi = command.name
            record.workspace = candidate
            record.revision += 1
            record.workspace.dirty = True
            record.workspace.preview_results_by_experiment.clear()
            app.state.checkpoint(record)
            return workspace_info(record)

    @app.post("/api/workspaces/{key}/preview-fit", response_model=JobInfo)
    def preview_fit(key: str, command: PreviewCommand):
        record = get_record(key)
        with record.lock:
            check_revision(record, command.revision)
            if record.workspace.session.roi(command.roi_name) is None:
                raise HTTPException(422, "Select an existing ROI before fitting")
            return app.state.submit(
                record,
                "preview",
                {"roi_name": command.roi_name, "steps": command.steps, "device": command.device},
            )

    @app.get("/api/jobs/{key}", response_model=JobInfo)
    def job(key: str):
        try:
            return app.state.jobs.get(key)
        except KeyError as exc:
            raise HTTPException(404, "Job not found") from exc

    @app.post("/api/jobs/{key}/cancel", response_model=JobInfo)
    def cancel_job(key: str):
        try:
            return app.state.jobs.cancel(key)
        except KeyError as exc:
            raise HTTPException(404, "Job not found") from exc

    @app.post("/api/workspaces/{key}/save")
    def save(key: str):
        record = get_record(key)
        path = app.state.root / f"{uuid4().hex}.mrax"
        with record.lock:
            snapshot, revision = clone_workspace(record.workspace), record.revision
        save_workspace_bundle(snapshot, path)
        with record.lock:
            if record.revision == revision:
                record.workspace.dirty = False
        return FileResponse(
            path,
            filename="mrax-workspace.mrax",
            media_type="application/octet-stream",
            background=BackgroundTask(path.unlink, missing_ok=True),
        )

    from .routes import register_routes

    register_routes(app, get_record, check_revision, add_record)
    from .file_operations import register_file_operations

    register_file_operations(app, get_record, check_revision, add_record)
    from .workflow_routes import register_workflow_routes

    register_workflow_routes(app, get_record, check_revision)
    assets = Path(__file__).with_name("static")
    if (assets / "index.html").is_file():
        app.mount("/", StaticFiles(directory=assets, html=True), name="frontend")
    else:

        @app.get("/")
        def missing_assets():
            return JSONResponse(
                {"detail": "Build the browser frontend: cd frontend && npm ci && npm run build"},
                status_code=503,
            )

    @app.exception_handler(IndexError)
    async def index_error(_request, _exc):
        return JSONResponse(
            {"detail": "The selected dimension or coil is outside its available range"},
            status_code=422,
        )

    return app
