"""Registration and drift result persistence."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ..services.common import _metadata_to_jsonable as _metadata_to_jsonable
from ..services.state import _safe_dict_value as _safe_dict_value
from ..services.state import _warn_restore as _warn_restore


def _registration_result_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "kind": result.get("kind"),
        "metadata": dict(result.get("metadata") or {}),
        "manual": dict(result.get("manual") or {}),
        "options": _metadata_to_jsonable(dict(result.get("options") or {})),
        "diagnostics": _metadata_to_jsonable(dict(result.get("diagnostics") or {})),
    }


def _linear_drift_result_summary(result: Any, metadata: Mapping[str, Any]) -> dict[str, Any]:
    group_rows: list[dict[str, Any]] = []
    for index, group in enumerate(getattr(result, "groups", ()) or ()):
        diagnostics = getattr(group, "diagnostics", None)
        before = np.asarray(getattr(group, "similarities_before", []), dtype=float)
        after = np.asarray(getattr(group, "similarities_after", []), dtype=float)
        group_rows.append(
            {
                "group_index": int(index),
                "reference_index": getattr(group, "reference_index", None),
                "parameters_shape": list(np.asarray(getattr(group, "parameters", [])).shape),
                "coefficients": np.asarray(getattr(group, "coefficients", [])).tolist(),
                "similarity_before_mean": _finite_mean(before),
                "similarity_after_mean": _finite_mean(after),
                "initial_objective": None
                if diagnostics is None
                else float(getattr(diagnostics, "initial_objective", np.nan)),
                "final_objective": None
                if diagnostics is None
                else float(getattr(diagnostics, "final_objective", np.nan)),
                "iterations": None
                if diagnostics is None
                else int(getattr(diagnostics, "iterations", 0)),
                "message": "" if diagnostics is None else str(getattr(diagnostics, "message", "")),
            }
        )
    return _metadata_to_jsonable(
        {
            "kind": "mrax_linear_drift",
            "metadata": dict(metadata or {}),
            "dimensions": list(getattr(result, "dimensions", ()) or ()),
            "group_dimensions": list(getattr(result, "group_dimensions", ()) or ()),
            "group_indices": [
                list(values) for values in (getattr(result, "group_indices", ()) or ())
            ],
            "corrected_shape": list(np.asarray(getattr(result, "corrected", [])).shape),
            "groups": group_rows,
        }
    )


def _failed_linear_drift_groups(result: Any) -> list[str]:
    out: list[str] = []
    group_indices = list(getattr(result, "group_indices", ()) or ())
    for index, group in enumerate(getattr(result, "groups", ()) or ()):
        diagnostics = getattr(group, "diagnostics", None)
        if diagnostics is None:
            continue
        success = bool(getattr(diagnostics, "success", False))
        final_objective = float(getattr(diagnostics, "final_objective", np.nan))
        if success and np.isfinite(final_objective):
            continue
        group_index = group_indices[index] if index < len(group_indices) else ()
        out.append(
            "group "
            f"{index}"
            f"{'' if not group_index else f' indices={tuple(group_index)}'}: "
            f"{getattr(diagnostics, 'message', '')}"
        )
    return out


def _finite_mean(values: np.ndarray) -> float | None:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None
    return float(np.mean(finite))


def _restore_drift_results(
    rows: Mapping[str, Mapping[str, Any]],
    arrays: dict[str, np.ndarray],
    *,
    warnings: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    restored: dict[str, dict[str, Any]] = {}
    for key, row in _safe_dict_value(rows).items():
        if not isinstance(row, Mapping):
            _warn_restore(
                warnings,
                f"Drift result '{key}' skipped: expected mapping, got {type(row).__name__}.",
            )
            continue
        array_key = row.get("corrected_key")
        if array_key not in arrays:
            _warn_restore(
                warnings,
                f"Drift result '{key}' skipped: missing corrected array key '{array_key}'.",
            )
            continue
        result_key = str(row.get("key") or key)
        try:
            restored[result_key] = {
                "key": result_key,
                "kind": str(row.get("kind") or "mrax_linear_drift"),
                "corrected": np.asarray(arrays[array_key]),
                "metadata": dict(row.get("metadata") or {}),
                "summary": dict(row.get("summary") or {}),
            }
        except Exception as exc:
            _warn_restore(warnings, f"Drift result '{key}' skipped: {exc}")
    return restored
