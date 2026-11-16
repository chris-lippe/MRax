"""Applied image state and default operation selection."""

import hashlib
import json


def image_signature(steps):
    """Identify the ordered image-producing steps used by a fit."""
    values = [
        (step["id"], step["version"], step["enabled"])
        for step in steps
        if step["enabled"] and step["kind"] != "fit"
    ]
    return hashlib.sha256(json.dumps(values).encode()).hexdigest() if values else "original"


def image_steps(pipeline):
    return [
        step for step in pipeline.get("applied_steps", pipeline["steps"])
        if step["enabled"] and step["kind"] != "fit"
    ]


def primary_step(pipeline, kind, options):
    from .pipeline import _family

    family = "register" if kind == "stored_registration" else _family(
        options.get("method") if kind == "process" else kind
    )
    return next((step for step in pipeline["steps"] if step["family"] == family), None)
