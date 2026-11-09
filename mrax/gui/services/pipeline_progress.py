"""Small progress snapshots for the pipeline currently being evaluated."""

import copy
import time


class PipelineRun:
    def __init__(self, experiment, steps, progress, start_at=0):
        self.progress = progress
        self.current = None
        self.previous = 0.0
        self.value = {"experiment": experiment, "status": "running", "steps": []}
        index = 0
        for step in steps:
            status = "pending" if step["enabled"] else "skipped"
            if step["enabled"]:
                if index < start_at:
                    status = "unchanged"
                index += 1
            self.value["steps"].append(
                {
                    "id": step["id"],
                    "client_id": step.get("_draft_id", step["id"]),
                    "label": step["label"],
                    "kind": step["kind"],
                    "family": step["family"],
                    "method": step["method"],
                    "status": status,
                    "progress": 1.0 if status == "unchanged" else 0.0,
                }
            )
        self.emit(force=True)

    def emit(self, *, force=False):
        callback = getattr(self.progress, "pipeline_event", None)
        now = time.monotonic()
        if callback is not None and (force or now - self.previous >= 0.05):
            callback(copy.deepcopy(self.value))
            self.previous = now

    def start(self, step):
        self.current = next(row for row in self.value["steps"] if row["id"] == step["id"])
        self.current.update(status="running", progress=0.0)
        self.emit(force=True)

    def update(self, done, total):
        if self.current is not None:
            self.current["progress"] = min(1.0, max(0.0, done / max(1, total)))
            self.emit()

    def complete(self):
        self.current.update(status="completed", progress=1.0)
        self.current = None
        self.emit(force=True)

    def finish(self):
        self.value["status"] = "completed"
        self.emit(force=True)

    def fail(self, error):
        status = "cancelled" if isinstance(error, InterruptedError) else "failed"
        if self.current is not None:
            self.current["status"] = status
        self.value["status"] = status
        self.emit(force=True)
