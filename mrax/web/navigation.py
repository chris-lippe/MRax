"""Atomic navigation frames and read-only prefetch, without workspace copies."""

import copy
import json
import struct

import numpy as np
from fastapi.responses import Response

from mrax.gui.services.comparison import (
    comparison_sources,
    render_comparison,
    same_source,
    validate_comparison,
)
from mrax.gui.services.scene import scene_settings

from .info import image_for_record, workspace_info
from .schemas import ViewCommand, WorkspaceInfo


def navigation_response(record, command):
    from .app import Record, update_view

    candidate = copy.copy(record.workspace)
    session = candidate.session = copy.copy(candidate.session)
    session.active_indices = dict(session.active_indices)
    session.view_states = dict(session.view_states)
    session._slice_cache = {}
    candidate.display_settings = dict(candidate.display_settings)
    active = session.selected_experiment or ""
    candidate.display_settings[active] = dict(candidate.display_settings.get(active, {}))
    scene = scene_settings(candidate)
    if scene["comparison"]["enabled"]:
        comparison = scene["comparison"]
        # Reuse the catalog; navigation must not clone every open experiment.
        if (
            record.info_cache.get("revision") != record.revision
            or "catalog" not in record.info_cache
        ):
            workspace_info(record)
        catalog = record.info_cache["catalog"]
        choices = comparison_sources(candidate, catalog)
        selected = [
            next(row for row in choices if same_source(row, comparison[side]))
            for side in ("left", "right")
        ]
        dimensions = {
            dim: min(row["dimensions"].get(dim, 1) for row in selected)
            for dim in set().union(*(row["dimensions"] for row in selected))
        }
        if set(command.indices) - {*dimensions, "frame"}:
            raise ValueError("Choose an available comparison dimension")
        cursor = (
            comparison["indices"] if comparison["indices"] is not None else session.active_indices
        )
        indices = {dim: min(cursor.get(dim, 0), size - 1) for dim, size in dimensions.items()}
        following = [
            row["frames"]
            for side, row in zip(("left", "right"), selected)
            if comparison[side].get("frame") is None and row["frames"] > 1
        ]
        frame = (
            comparison["frame"]
            if comparison["frame"] is not None
            else min(session.active_frame, min(following, default=1) - 1)
        )
        indices.update({key: value for key, value in command.indices.items() if key != "frame"})
        comparison = {
            **comparison,
            "frame": command.indices.get("frame", frame),
            "indices": indices,
        }
        scene["comparison"] = validate_comparison(candidate, comparison, catalog)
        candidate.display_settings[active]["scene"] = scene
    else:
        update_view(
            candidate,
            ViewCommand(revision=command.revision, indices=command.indices),
            navigation=True,
        )
    candidate.dirty = record.workspace.dirty or command.commit
    cache = dict(record.info_cache)
    cache["roi_planes"] = dict(cache.get("roi_planes", {}))
    snapshot = Record(
        candidate,
        record.revision,
        record.view_revision + int(command.commit),
        file_location=record.file_location,
        info_cache=cache,
    )
    workspace = WorkspaceInfo.model_validate(workspace_info(snapshot)).model_dump(mode="json")
    planes, buffers = [], []

    def append(name, array):
        dtype = "<f8" if array.dtype.itemsize > 4 else "<f4"
        array = np.asarray(array, dtype=dtype)
        planes.append(
            {
                "id": name,
                "shape": list(array.shape),
                "dtype": "float64-le" if dtype == "<f8" else "float32-le",
                "length": array.size,
            }
        )
        buffer = array.tobytes(order="C")
        buffers.append(buffer + b"\0" * (-len(buffer) % 8))

    comparison_info = None
    if command.include_images and scene["comparison"]["enabled"]:
        arrays, comparison_info = render_comparison(candidate, catalog=cache.get("catalog"))
        for name, array in zip(("left", "right"), arrays):
            append(name, array)
    elif command.include_images:
        append("image", image_for_record(snapshot)[0])
        for layer in workspace["scene"]["layers"]:
            if layer["visible"] and not layer["unavailable"]:
                append("layer:" + layer["id"], image_for_record(snapshot, layer["id"])[0])
    header = json.dumps(
        {
            "workspace": workspace,
            "planes": planes,
            "comparison": comparison_info,
            "images_included": command.include_images,
        },
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    body = b"".join(
        (struct.pack("<II", 1, len(header)), header, b"\0" * (-len(header) % 8), *buffers)
    )
    if command.commit:
        record.workspace = candidate
        record.view_revision = snapshot.view_revision
        record.info_cache = cache
    return Response(
        body, media_type="application/vnd.mrax.navigation", headers={"Cache-Control": "no-store"}
    )
