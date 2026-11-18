"""Bounded in-memory scientific jobs. Disk output is explicit."""

from __future__ import annotations

import copy
import shutil
import threading
import time
from concurrent.futures import CancelledError, ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4


def _workflow_worker(workspace, root, operation, options, progress, *, output_guard=None):
    try:
        progress(0, 1, "Preparing operation")
        from mrax.gui.services.analysis_workflows import run_operation

        if operation in {"save", "export"}:
            from mrax.gui.services.export_workflow import export_workspace

            summary, artifact = export_workspace(
                workspace, options, root / "export", progress, output_guard=output_guard
            )
            return {
                "operation": operation,
                "value": summary,
                "_artifact": None if artifact is None else str(artifact),
                "_view_revision": options.get("_view_revision"),
                "_pending_export": bool(summary.get("needs_confirmation")),
                "_consumed_export": options.get("_prepared_export_id")
                if summary.get("saved")
                else None,
            }
        workspace, summary = run_operation(workspace, operation, options, progress)
        progress(1, 1, "Operation complete")
        workspace.dirty = True
        return {
            "operation": operation,
            "value": summary,
            "_workspace": workspace,
            "_view_revision": options.get("_view_revision"),
        }
    except InterruptedError:
        return {"cancelled": True}


class Jobs:
    def __init__(self, root, attach, on_failure=None, *, output_guard=None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.attach = attach
        self.on_failure = on_failure
        self.output_guard = output_guard
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mrax-operation")
        self.lock = threading.RLock()
        self.items = {}

    def submit(self, workspace_id, revision, workspace, *, operation="preview", options=None):
        options = dict(options or {})
        with self.lock:
            if (
                sum(
                    row["status"] in {"queued", "running", "cancelling"}
                    for row in self.items.values()
                )
                >= 8
            ):
                raise ValueError("The operation queue is full")
            if self.is_active(workspace_id):
                raise ValueError("This workspace already has an active operation")
            key = uuid4().hex
            root = self.root / key
            root.mkdir()
            cancelled = threading.Event()
            item = {
                "id": key,
                "workspace_id": workspace_id,
                "operation": operation,
                "status": "queued",
                "detail": "Queued",
                "result": None,
                "summary": None,
                "pipeline": None,
                "_cancel": cancelled,
                "_applying": False,
                "_revision": revision,
                "_prepared_export_id": options.get("_prepared_export_id"),
            }
            self.items[key] = item

            def execute():
                previous = 0.0

                def progress(current, total, detail):
                    nonlocal previous
                    if cancelled.is_set():
                        raise InterruptedError("Cancelled")
                    now = time.monotonic()
                    if now - previous >= 0.05 or current == total:
                        with self.lock:
                            if not cancelled.is_set():
                                item.update(
                                    status="running",
                                    progress=min(0.99, max(0, current / max(1, total))),
                                    detail=str(detail),
                                )
                        previous = now

                def pipeline_event(value):
                    with self.lock:
                        item["pipeline"] = value

                def begin_commit():
                    with self.lock:
                        if cancelled.is_set():
                            raise InterruptedError("Cancelled")
                        item["_applying"] = True

                progress.pipeline_event = pipeline_event
                progress.begin_commit = begin_commit

                return _workflow_worker(
                    workspace, root, operation, options, progress, output_guard=self.output_guard
                )

            future = self.pool.submit(execute)
            item["future"] = future

        def finish(done):
            summary = result = None
            try:
                payload = done.result()
                with self.lock:
                    apply = not cancelled.is_set() and not payload.get("cancelled")
                    item["_applying"] = apply
                if not apply:
                    status, detail = "cancelled", "Operation cancelled"
                elif self.attach(workspace_id, revision, payload):
                    status, detail = (
                        "completed",
                        "Fit complete" if operation == "preview" else "Operation complete",
                    )
                    summary = payload["value"]
                    if payload.get("_pending_export"):
                        item["_pending_export"] = True
                        summary = {**summary, "pending_export": key}
                        detail = "Review existing output files"
                    if payload.get("_consumed_export"):
                        consumed = payload["_consumed_export"]
                        if consumed in self.items:
                            self.items[consumed].pop("_pending_export", None)
                            self.items[consumed]["summary"] = {"saved": True}
                        shutil.rmtree(self.root / consumed, ignore_errors=True)
                    if payload.get("_artifact"):
                        item["_artifact"] = payload["_artifact"]
                        summary = {**summary, "download": f"/api/jobs/{key}/download"}
                    result = summary if operation == "preview" else None
                else:
                    status, detail = "stale", "Inputs changed. Result was not applied"
            except (CancelledError, InterruptedError):
                status, detail = "cancelled", "Operation cancelled"
            except Exception as exc:
                status, detail = (
                    "failed",
                    str(exc)[:2000] or "Operation failed. The workspace is unchanged",
                )
                if self.on_failure is not None:
                    self.on_failure(workspace_id, operation, detail)
            with self.lock:
                item.update(status=status, detail=detail, result=result, summary=summary)
                if item["pipeline"] is not None:
                    item["pipeline"]["status"] = status
                    for step in item["pipeline"]["steps"]:
                        if step["status"] == "running":
                            step["status"] = "cancelled" if status == "cancelled" else "failed"
                if status == "completed":
                    item["progress"] = 1.0
                # A Future retains its workspace result. Keeping it here would retain
                # old, potentially large volumes after every completed operation.
                item.pop("future", None)
                terminal = [
                    old
                    for old, row in self.items.items()
                    if old != key and row["status"] not in {"queued", "running", "cancelling"}
                ]
                for old in terminal[:-15]:
                    self.items.pop(old)
                    shutil.rmtree(self.root / old, ignore_errors=True)
            if not item.get("_artifact") and not item.get("_pending_export"):
                shutil.rmtree(root, ignore_errors=True)

        future.add_done_callback(finish)
        return self.get(key)

    def get(self, key):
        with self.lock:
            return {
                k: copy.deepcopy(v) if k == "pipeline" else v
                for k, v in self.items[key].items()
                if k != "future" and not k.startswith("_")
            }

    def cancel(self, key):
        with self.lock:
            item = self.items[key]
            if item.get("_pending_export") and not any(
                row.get("_prepared_export_id") == key
                and row["status"] in {"queued", "running", "cancelling"}
                for row in self.items.values()
            ):
                shutil.rmtree(self.root / key, ignore_errors=True)
                item.pop("_pending_export", None)
                item.update(status="cancelled", detail="Export cancelled", summary={"saved": False})
            if item["status"] in {"queued", "running", "cancelling"} and not item["_applying"]:
                item["_cancel"].set()
                future = item.get("future")
                if future is not None and future.cancel():
                    item.update(status="cancelled", detail="Operation cancelled")
                else:
                    item.update(
                        status="cancelling",
                        detail="Cancellation requested. Waiting for a safe boundary",
                    )
        return self.get(key)

    def prepared_export(self, key, workspace_id, revision):
        """Resolve an app-owned preparation, never a client-supplied filesystem path."""
        with self.lock:
            item = self.items.get(key)
            if not item or not item.get("_pending_export") or item["status"] != "completed":
                raise ValueError("The prepared export expired. Select the folder again")
            if item["workspace_id"] != workspace_id or item["_revision"] != revision:
                raise ValueError("The study changed. Select the export folder again")
            path = self.root / key / "export"
            if not (path / "pending.json").is_file():
                raise ValueError("The prepared export expired. Select the folder again")
            return path

    def artifact(self, key):
        with self.lock:
            path = self.items[key].get("_artifact")
            return None if path is None else Path(path)

    def is_active(self, workspace_id):
        with self.lock:
            return any(
                row["workspace_id"] == workspace_id
                and row["status"] in {"queued", "running", "cancelling"}
                for row in self.items.values()
            )

    def close(self):
        with self.lock:
            for item in self.items.values():
                if item["status"] in {"queued", "running", "cancelling"} and not item["_applying"]:
                    item["_cancel"].set()
        self.pool.shutdown(wait=True, cancel_futures=True)
