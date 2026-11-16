"""Transactional metadata/ROI copies sharing immutable numerical buffers."""

import copy
import dataclasses

import numpy as np


def clone_workspace(workspace, *, readonly_arrays=False, share_sources=False):
    """Copy editable state, with shared inputs for display and ROI commands."""
    memo = {}
    visited = set()
    if share_sources:
        if readonly_arrays:
            raise ValueError("Read-only snapshots require independent acquisition metadata")
        for value in (
            workspace.session.data,
            workspace.session.study,
            workspace.session.import_metadata,
            workspace.pipelines,
            workspace.pipeline_sets,
            workspace.registration_results,
            workspace.drift_results,
        ):
            if value is not None:
                memo[id(value)] = value

    def visit(value):
        if id(value) in visited or id(value) in memo:
            return
        visited.add(id(value))
        if (
            isinstance(value, np.ndarray)
            or (hasattr(value, "block_until_ready") and hasattr(value, "shape"))
            or (hasattr(value, "GetParameters") and hasattr(value, "GetDimension"))
        ):
            if readonly_arrays and isinstance(value, np.ndarray):
                view = value.view()
                view.flags.writeable = False
                memo[id(value)] = view
            else:
                memo[id(value)] = value
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
        elif dataclasses.is_dataclass(value):
            for field in dataclasses.fields(value):
                visit(getattr(value, field.name))

    visit(workspace)
    return copy.deepcopy(workspace, memo)
