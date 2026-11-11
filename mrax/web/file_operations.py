"""Open and save ordinary files on the application's filesystem."""

from __future__ import annotations

import shutil
from uuid import uuid4

from pydantic import BaseModel

from mrax.gui.services.cloning import clone_workspace
from mrax.gui.services.file_output import write_file_atomic
from mrax.gui.services.source_protection import guard_output, remember_source
from mrax.gui.services.workspace_io import (
    load_workspace_path,
    save_workspace_bundle,
    validate_workspace_replacement,
)

from .info import workspace_info
from .schemas import WorkspaceInfo


class FileLocation(BaseModel):
    root: str
    path: str = ""


class SaveFile(FileLocation):
    revision: int
    overwrite: bool = False


class WriteText(SaveFile):
    workspace_id: str
    content: str


def register_file_operations(app, get_record, check_revision, add_record):
    @app.get("/api/files/stat")
    def stat(root: str, path: str = ""):
        return app.state.folders.describe(root, path)

    @app.get("/api/files/text")
    def read_text(root: str, path: str = ""):
        source = app.state.folders.checked_tree(root, path)
        if not source.is_file() or source.stat().st_size > 8 * 1024**2:
            raise ValueError("Choose a text file smaller than 8 MiB")
        try:
            app.state.folders.sources.protect(source)
            return {"name": source.name, "content": source.read_text(encoding="utf-8-sig")}
        except UnicodeError as exc:
            raise ValueError("Choose a UTF-8 text file") from exc

    @app.post("/api/files/folder")
    def new_folder(command: FileLocation):
        destination = app.state.folders.resolve(command.root, command.path, write=True)
        try:
            destination.mkdir()
        except FileExistsError as exc:
            raise ValueError("A file or folder with this name already exists") from exc
        return app.state.folders.describe(command.root, command.path)

    @app.post("/api/files/text")
    def write_text(command: WriteText):
        if len(command.content.encode("utf-8")) > 16 * 1024**2:
            raise ValueError("The text export exceeds 16 MiB")
        record = get_record(command.workspace_id)
        with record.lock:
            check_revision(record, command.revision)
            destination = app.state.folders.resolve(command.root, command.path, write=True)
            guard_output(app, destination)
            if destination.exists():
                raise ValueError(
                    "Choose a new file name. Existing metadata and text files are not replaced"
                )
            write_file_atomic(
                destination,
                lambda path: path.write_text(command.content, encoding="utf-8"),
                overwrite=False,
            )
        return {"saved": True, "path": str(destination)}

    @app.post("/api/workspaces/open", response_model=WorkspaceInfo)
    def open_workspace(command: FileLocation):
        folders = app.state.folders
        source = folders.checked_tree(command.root, command.path)
        key = uuid4().hex
        extraction = app.state.root / key
        try:
            workspace = load_workspace_path(
                source,
                key=key,
                extraction_root=extraction,
                reference_root=folders.items[command.root]["path"] if not folders.system else None,
            )
            location = (
                command.model_dump()
                if source.is_file()
                and source.suffix.lower() == ".mrax"
                and folders.items[command.root]["writable"]
                else None
            )
            if location is None:
                remember_source(workspace, source if source.is_dir() else source.parent)
                folders.sources.protect(source if source.is_dir() else source.parent)
            return add_record(workspace, file_location=location)
        except BaseException:
            shutil.rmtree(extraction, ignore_errors=True)
            raise

    @app.post("/api/workspaces/{key}/save-file", response_model=WorkspaceInfo)
    def save_file(key: str, command: SaveFile):
        record = get_record(key)
        destination = app.state.folders.resolve(command.root, command.path, write=True)
        if destination.suffix.lower() != ".mrax":
            raise ValueError("Save the workspace with a .mrax extension")
        with record.lock:
            check_revision(record, command.revision)
            guard_output(app, destination)
            if destination.exists() and command.overwrite:
                validate_workspace_replacement(destination)
            snapshot = clone_workspace(record.workspace)
            write_file_atomic(
                destination,
                lambda path: save_workspace_bundle(snapshot, path),
                overwrite=command.overwrite,
            )
            record.workspace.dirty = False
            record.file_location = {"root": command.root, "path": command.path}
            app.state.checkpoint(record)
            return workspace_info(record)
