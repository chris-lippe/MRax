"""Session workflow library, ordinary workflow files and explicit target execution."""

from __future__ import annotations

import copy
import threading
from typing import Literal
from uuid import uuid4

from fastapi import HTTPException
from pydantic import Field, model_validator

from mrax.gui.services.reusable_workflows import (
    capture_workflow,
    resource_choices,
    validate_application_options,
    validate_workflow_targets,
)
from mrax.workflows import (
    WorkflowDocument,
    WorkflowKind,
    dumps_workflow,
    load_workflow,
    loads_workflow,
    save_workflow,
)

from .file_operations import FileLocation, SaveFile
from .schemas import Command, JobInfo


class CaptureWorkflow(Command):
    revision: int = Field(ge=0)
    kind: WorkflowKind
    name: str | None = Field(default=None, max_length=160)
    description: str = Field(default="", max_length=8000)
    draft: dict = {}


class EditWorkflow(Command):
    document: WorkflowDocument | None = None
    content: str | None = Field(default=None, max_length=1024 * 1024)
    revision: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def one_document(self):
        if (self.document is None) == (self.content is None):
            raise ValueError("Supply either a document or its JSON text")
        if self.content is not None:
            self.document = loads_workflow(self.content)
        return self


class ApplyWorkflow(Command):
    revision: int = Field(ge=0)
    workflow_revision: int = Field(ge=0)
    experiments: list[str] = Field(min_length=1, max_length=128)
    bindings: dict[str, dict[str, str]] = {}
    mode: Literal["replace", "append"] = "replace"
    run: bool = True
    device: str = Field(default="cpu", max_length=80)


class WorkflowLibraryEntry(Command):
    id: str
    revision: int
    document: WorkflowDocument
    suggested_bindings: dict[str, str]
    origin: dict[str, str] | None


def register_workflow_routes(app, get_record, check_revision):
    library = {}
    lock = threading.RLock()

    def item(key):
        if key not in library:
            raise HTTPException(404, "The workflow is no longer in this session")
        return library[key]

    def check_workflow(key, revision):
        row = item(key)
        if row["revision"] != revision:
            raise HTTPException(409, "The workflow changed; reopen it before continuing")
        return row

    def add(document, *, suggested=None, origin=None):
        dumps_workflow(document)
        with lock:
            if len(library) >= 128:
                raise ValueError("Remove a session workflow before adding another")
            key = uuid4().hex
            row = {
                "id": key,
                "revision": 0,
                "document": document.model_dump(),
                "suggested_bindings": suggested or {},
                "origin": origin,
            }
            library[key] = row
            return copy.deepcopy(row)

    @app.get("/api/workflows", response_model=list[WorkflowLibraryEntry])
    def workflows():
        with lock:
            return copy.deepcopy(list(library.values()))

    @app.post("/api/workflows", response_model=WorkflowLibraryEntry)
    def create(command: EditWorkflow):
        return add(command.document)

    @app.put("/api/workflows/{key}", response_model=WorkflowLibraryEntry)
    def edit(key: str, command: EditWorkflow):
        with lock:
            row = check_workflow(key, command.revision)
            dumps_workflow(command.document)
            row.update(document=command.document.model_dump(), revision=row["revision"] + 1)
            return copy.deepcopy(row)

    @app.delete("/api/workflows/{key}", status_code=204)
    def remove(key: str, revision: int):
        with lock:
            check_workflow(key, revision)
            del library[key]

    @app.post("/api/workflows/open-file", response_model=WorkflowLibraryEntry)
    def open_file(command: FileLocation):
        source = app.state.folders.checked_tree(command.root, command.path)
        if not source.is_file():
            raise ValueError("Choose a workflow file")
        return add(load_workflow(source))

    @app.post("/api/workflows/{key}/save-file")
    def save_file(key: str, command: SaveFile):
        with lock:
            row = check_workflow(key, command.revision)
            document = WorkflowDocument.model_validate(row["document"])
        destination = app.state.folders.resolve(command.root, command.path, write=True)
        from mrax.gui.services.source_protection import guard_output

        guard_output(app, destination)
        save_workflow(document, destination, overwrite=command.overwrite)
        return {"saved": True, "name": destination.name}

    @app.post("/api/workspaces/{key}/workflows", response_model=WorkflowLibraryEntry)
    def capture(key: str, command: CaptureWorkflow):
        record = get_record(key)
        with record.lock:
            check_revision(record, command.revision)
            document, suggested = capture_workflow(
                record.workspace,
                command.kind,
                name=command.name,
                description=command.description,
                draft=command.draft,
            )
            origin = {
                "workspace_id": key,
                "experiment": record.workspace.session.selected_experiment,
            }
        return add(document, suggested=suggested, origin=origin)

    @app.get("/api/workflow-targets")
    def targets():
        # Release the registry before acquiring individual workspace locks.
        rows = list(app.state.records.items())
        result = []
        for key, record in rows:
            with record.lock:
                result.append(
                    {
                        "workspace_id": key,
                        "label": record.workspace.label,
                        "revision": record.revision,
                        "experiments": resource_choices(record.workspace),
                    }
                )
        return result

    def prepared(key, workflow, command):
        record = get_record(key)
        check_revision(record, command.revision)
        if app.state.jobs.is_active(key):
            raise HTTPException(409, "This study already has an active operation")
        with lock:
            row = check_workflow(workflow, command.workflow_revision)
            document = WorkflowDocument.model_validate(row["document"])
        validate_application_options(document, command.mode, command.run)
        return record, document

    @app.post("/api/workspaces/{key}/workflows/{workflow}/validate")
    def validate(key: str, workflow: str, command: ApplyWorkflow):
        record = get_record(key)
        with record.lock:
            _, document = prepared(key, workflow, command)
            checks = validate_workflow_targets(
                record.workspace,
                document,
                command.experiments,
                command.bindings,
            )
        return {"ready": all(row["ready"] for row in checks), "targets": checks}

    @app.post("/api/workspaces/{key}/workflows/{workflow}/apply", response_model=JobInfo)
    def apply(key: str, workflow: str, command: ApplyWorkflow):
        record = get_record(key)
        with record.lock:
            _, document = prepared(key, workflow, command)
            checks = validate_workflow_targets(
                record.workspace,
                document,
                command.experiments,
                command.bindings,
            )
            if not all(row["ready"] for row in checks):
                raise ValueError(
                    "; ".join(
                        row["experiment"] + ": " + ", ".join(row["errors"])
                        for row in checks
                        if not row["ready"]
                    )
                )
            return app.state.submit(
                record,
                "workflow",
                {
                    "document": document.model_dump(),
                    "experiments": command.experiments,
                    "bindings": command.bindings,
                    "mode": command.mode,
                    "run": command.run,
                    "device": command.device,
                },
            )
