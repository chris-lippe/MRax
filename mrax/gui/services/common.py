"""Application metadata and numerical summary helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from mrax.data_struct import Quantity

_LOG = logging.getLogger(__name__)


def _metadata_to_jsonable(value: Any) -> Any:
    if isinstance(value, Quantity):
        return {
            "unit": value.si_unit,
            "values": _metadata_to_jsonable(value.values),
        }
    if isinstance(value, np.ndarray):
        arr = np.asarray(value)
        if arr.size <= 200:
            return arr.tolist()
        return {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "summary": _array_summary(arr),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _metadata_to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        if len(value) > 200:
            return {
                "length": len(value),
                "items_preview": [_metadata_to_jsonable(item) for item in value[:50]],
            }
        return [_metadata_to_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        arr = np.asarray(value)
        if arr.shape:
            return _metadata_to_jsonable(arr)
    except Exception:
        _LOG.debug("Ignored optional GUI operation failure", exc_info=True)
    return str(value)


def _array_summary(value: np.ndarray, *, unit: str | None = None) -> str:
    arr = np.asarray(value)
    unit_text = f" {unit}" if unit else ""
    if arr.size == 0:
        return f"shape={tuple(arr.shape)}, empty{unit_text}"
    if np.issubdtype(arr.dtype, np.number):
        finite = (
            arr[np.isfinite(arr)] if not np.iscomplexobj(arr) else np.abs(arr[np.isfinite(arr)])
        )
        if finite.size:
            return (
                f"shape={tuple(arr.shape)}, dtype={arr.dtype}, "
                f"min={float(np.nanmin(finite)):.6g}, max={float(np.nanmax(finite)):.6g}{unit_text}"
            )
    return f"shape={tuple(arr.shape)}, dtype={arr.dtype}{unit_text}"
