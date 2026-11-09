"""Transactional metadata/ROI copies sharing immutable numerical buffers."""

import copy
import dataclasses

import numpy as np


def clone_workspace(workspace, *, readonly_arrays=False):
    memo = {}
    visited = set()

    def visit(value):
        if id(value) in visited:
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
