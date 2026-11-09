"""Portable analysis-session state, independent of browser preferences."""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from mrax import MRAX_VERSION, MRAX_VERSION_TAG
from mrax.analysis import ROI
from mrax.data_struct import Quantity, SpatialGeometry
from mrax.data_struct.dimensions import spatial_names
from mrax.versioning import missing_mrax_version_message, validate_mrax_file_version

from ..session import ExperimentViewState, FitSettings, FrameMask, GuiSession, VisualViewState
from .common import _metadata_to_jsonable

_GUI_STATE_CURRENT_VERSION = 3


_GUI_STATE_KNOWN_KEYS = {
    "format",
    "version",
    MRAX_VERSION_TAG,
    "spatial_plane_contract",
    "data_manifest",
    "fit_manifest",
    "source_kind",
    "selected_experiment",
    "shared_rois",
    "view_states",
    "original_images",
    "fitted_maps",
    "spatial_geometry",
    "drift_results",
    "preview_results",
    "fit_results",
    "array_file",
}


def _session_spatial_geometry_metadata(
    session: GuiSession,
    experiment_name: str,
    exp_data: Mapping[str, Any],
) -> dict[str, Any]:
    geometry = exp_data.get("spatial_geometry")
    if isinstance(geometry, Mapping):
        return _metadata_to_jsonable(dict(geometry))
    if session.study is not None and experiment_name in (session.study.names or ()):
        index = list(session.study.names).index(experiment_name)
        dataset_geometry = session.study.datasets[index].spatial_geometry
        if dataset_geometry is not None:
            return dataset_geometry.to_metadata()
    dims = dict(session.summary_by_name.get(experiment_name, {}).get("dims") or {})
    shape = tuple(int(dims[name]) for name in spatial_names(dims)) or None
    return SpatialGeometry(
        spatial_shape=shape,
        source="unknown",
        status="unknown",
    ).to_metadata()


def _serialize_gui_state(
    session: GuiSession,
    *,
    preview_results: dict[str, dict[str, Any]],
    fit_results: dict[str, dict[str, Any]],
    drift_results: Mapping[str, Mapping[str, Any]] | None = None,
    data_manifest: Path | None = None,
    fit_manifest: Path | None = None,
    state_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    arrays: dict[str, np.ndarray] = {}
    states: dict[str, Any] = {}
    fitted_map_rows: dict[str, list[dict[str, Any]]] = {}
    spatial_geometry_rows: dict[str, Any] = {}
    original_image_rows: dict[str, dict[str, str]] = {}
    shared_roi_rows = _serialize_roi_rows(session.shared_rois, "shared_roi", arrays)
    for exp_name, state in session.view_states.items():
        roi_rows = _serialize_roi_rows(state.rois, f"roi_{_safe_key(exp_name)}", arrays)
        states[exp_name] = {
            "active_frame": state.active_frame,
            "active_z": state.active_z,
            "active_indices": dict(state.active_indices),
            "active_image_plane": state.active_image_plane,
            "active_map": state.active_map,
            "active_kspace_mode": state.active_kspace_mode,
            "active_kspace_centering": state.active_kspace_centering,
            "active_reconstruction_component": state.active_reconstruction_component,
            "active_reconstruction_coil": state.active_reconstruction_coil,
            "active_reconstruction_unwrap_phase": state.active_reconstruction_unwrap_phase,
            "visual_state": _visual_state_to_dict(state.visual_state),
            "fit_settings": _fit_settings_to_dict(state.fit_settings),
            "frame_mask": None
            if state.frame_mask is None
            else state.frame_mask.as_array().astype(bool).tolist(),
            "rois": roi_rows,
        }
    for exp_name, exp_data in ((session.data or {}).get("experiments") or {}).items():
        spatial_geometry_rows[str(exp_name)] = _session_spatial_geometry_metadata(
            session,
            str(exp_name),
            exp_data,
        )
        original = exp_data.get("original_mr_image")
        if original is not None:
            key = f"original_{_safe_key(str(exp_name))}_mr_image"
            arrays[key] = np.asarray(original)
            original_image_rows[str(exp_name)] = {"array_key": key}
        rows = []
        for map_name, value in (exp_data.get("fitted_maps") or {}).items():
            key = f"fmap_{_safe_key(str(exp_name))}_{_safe_key(str(map_name))}"
            unit = ""
            array_value = value
            if isinstance(value, Quantity):
                unit = value.si_unit
                array_value = value.values
            arrays[key] = np.asarray(array_value)
            rows.append({"name": str(map_name), "array_key": key, "unit": unit})
        if rows:
            fitted_map_rows[str(exp_name)] = rows
    drift_rows: dict[str, dict[str, Any]] = {}
    for key, result in dict(drift_results or {}).items():
        if str(result.get("kind") or "") != "mrax_linear_drift" or result.get("corrected") is None:
            continue
        array_key = f"drift_{_safe_key(str(key))}_corrected"
        arrays[array_key] = np.asarray(result["corrected"])
        drift_rows[str(key)] = {
            "key": str(key),
            "kind": "mrax_linear_drift",
            "corrected_key": array_key,
            "metadata": _metadata_to_jsonable(dict(result.get("metadata") or {})),
            "summary": _metadata_to_jsonable(dict(result.get("summary") or {})),
        }
    data_manifest_text = (
        _state_relative_path(data_manifest, state_path) if data_manifest is not None else None
    )
    fit_manifest_text = (
        _state_relative_path(fit_manifest, state_path) if fit_manifest is not None else None
    )
    return (
        {
            "format": "mrax.gui.state",
            "version": 3,
            MRAX_VERSION_TAG: MRAX_VERSION,
            "spatial_plane_contract": "fixed_xyz_v1",
            "data_manifest": data_manifest_text,
            "fit_manifest": fit_manifest_text,
            "source_kind": session.source_kind,
            "selected_experiment": session.selected_experiment,
            "shared_rois": shared_roi_rows,
            "view_states": states,
            "original_images": original_image_rows,
            "fitted_maps": fitted_map_rows,
            "spatial_geometry": spatial_geometry_rows,
            "drift_results": drift_rows,
            "preview_results": {
                key: _preview_result_to_json(value) for key, value in preview_results.items()
            },
            "fit_results": {
                key: {"model": value.get("model"), "name": value.get("name")}
                for key, value in fit_results.items()
            },
        },
        arrays,
    )


def _serialize_roi_rows(
    rois: list[ROI] | tuple[ROI, ...], prefix: str, arrays: dict[str, np.ndarray]
) -> list[dict[str, Any]]:
    rows = []
    for idx, roi in enumerate(rois):
        key = f"{prefix}_{idx}"
        arrays[key] = np.asarray(roi.mask)
        rows.append(
            {
                "name": roi.name,
                "array_key": key,
                "dims": list(roi.dims or ()),
                "metadata": dict(roi.metadata),
            }
        )
    return rows


def _warn_restore(warnings: list[str] | None, message: str) -> None:
    if warnings is not None and str(message) not in warnings:
        warnings.append(str(message))


def _safe_int_value(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_float_value(value: Any, default: float = 1.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_dict_value(value: Any) -> dict[str, Any]:
    try:
        return dict(value or {})
    except Exception:
        return {}


def _restore_roi_rows(
    rows: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    arrays: dict[str, np.ndarray],
    *,
    warnings: list[str] | None = None,
    context: str = "ROI",
) -> list[ROI]:
    rois = []
    for idx, roi_data in enumerate(rows or []):
        if not isinstance(roi_data, Mapping):
            _warn_restore(
                warnings,
                f"{context} row {idx} skipped: expected mapping, got {type(roi_data).__name__}.",
            )
            continue
        key = roi_data.get("array_key")
        if key not in arrays:
            _warn_restore(
                warnings,
                f"{context} '{roi_data.get('name') or key or idx}' skipped: missing array key '{key}'.",
            )
            continue
        try:
            rois.append(
                ROI(
                    roi_data.get("name") or key,
                    np.asarray(arrays[key]),
                    dims=tuple(roi_data.get("dims") or ()),
                    metadata=dict(roi_data.get("metadata") or {}),
                )
            )
        except Exception as exc:
            _warn_restore(warnings, f"{context} '{roi_data.get('name') or key}' skipped: {exc}")
    return rois


def _visual_state_to_dict(state: VisualViewState) -> dict[str, Any]:
    return {
        "show_original": bool(state.show_original),
        "display_mode": state.display_mode,
        "mask_enabled": bool(state.mask_enabled),
        "mask_roi": state.mask_roi,
        "drift_key": state.drift_key,
        "transform_key": state.transform_key,
        "transform_direction": state.transform_direction,
        "color_channel": state.color_channel,
        "color_alpha": float(state.color_alpha),
        "reference_channel": state.reference_channel,
        "moving_channel": state.moving_channel,
        "reference_alpha": float(state.reference_alpha),
        "moving_alpha": float(state.moving_alpha),
    }


def _visual_state_from_dict(data: Mapping[str, Any]) -> VisualViewState:
    if not isinstance(data, Mapping):
        data = {}
    mask_roi = data.get("mask_roi") or None
    return VisualViewState(
        show_original=bool(data.get("show_original", True)),
        display_mode=str(data.get("display_mode") or "current"),
        mask_enabled=bool(data.get("mask_enabled", bool(mask_roi))),
        mask_roi=mask_roi,
        drift_key=data.get("drift_key") or None,
        transform_key=data.get("transform_key") or None,
        transform_direction=str(data.get("transform_direction") or "moving_to_reference"),
        color_channel=str(data.get("color_channel") or "gray"),
        color_alpha=_safe_float_value(data.get("color_alpha", 1.0), 1.0),
        reference_channel=str(data.get("reference_channel") or "red"),
        moving_channel=str(data.get("moving_channel") or "green"),
        reference_alpha=_safe_float_value(data.get("reference_alpha", 1.0), 1.0),
        moving_alpha=_safe_float_value(data.get("moving_alpha", 1.0), 1.0),
    )


def _restore_gui_state(
    session: GuiSession,
    payload: dict[str, Any],
    arrays: dict[str, np.ndarray],
    *,
    warnings: list[str] | None = None,
) -> None:
    fmt = str(payload.get("format") or "")
    if fmt != "mrax.gui.state":
        raise ValueError("GUI state format must be 'mrax.gui.state'.")
    producer_version = validate_mrax_file_version(payload, file_kind="MRax GUI state")
    if producer_version.tag_missing:
        _warn_restore(warnings, missing_mrax_version_message("MRax GUI state"))
    if "version" not in payload:
        raise ValueError("GUI state version is required.")
    version = _safe_int_value(payload.get("version"), -1)
    if version != _GUI_STATE_CURRENT_VERSION:
        raise ValueError(
            f"GUI state version {version} is not supported; expected {_GUI_STATE_CURRENT_VERSION}."
        )
    if payload.get("spatial_plane_contract") != "fixed_xyz_v1":
        raise ValueError("GUI state spatial_plane_contract must be 'fixed_xyz_v1'.")
    unknown_keys = sorted(
        str(key) for key in payload.keys() if str(key) not in _GUI_STATE_KNOWN_KEYS
    )
    if unknown_keys:
        raise ValueError(
            f"Unsupported GUI state top-level entr{'y' if len(unknown_keys) == 1 else 'ies'}: {', '.join(unknown_keys)}."
        )

    if session.data is None:
        raise RuntimeError("GUI state restoration requires a loaded MRax study.")
    for exp_name, geometry_value in _safe_dict_value(payload.get("spatial_geometry")).items():
        if exp_name not in session.experiment_names:
            _warn_restore(
                warnings,
                f"Spatial geometry for missing experiment '{exp_name}' skipped.",
            )
            continue
        if not isinstance(geometry_value, Mapping):
            _warn_restore(
                warnings,
                f"Spatial geometry for '{exp_name}' skipped: expected mapping.",
            )
            continue
        try:
            geometry = SpatialGeometry.from_metadata(geometry_value)
            session.data["experiments"][exp_name]["spatial_geometry"] = geometry.to_metadata()
            if session.study is not None and exp_name in (session.study.names or ()):
                index = list(session.study.names).index(exp_name)
                session.study.datasets[index].spatial_geometry = geometry
        except Exception as exc:
            _warn_restore(warnings, f"Spatial geometry for '{exp_name}' skipped: {exc}")

    for exp_name, row in _safe_dict_value(payload.get("original_images")).items():
        if exp_name not in session.experiment_names:
            _warn_restore(
                warnings, f"Original image snapshot for missing experiment '{exp_name}' skipped."
            )
            continue
        if not isinstance(row, Mapping):
            _warn_restore(
                warnings, f"Original image snapshot for '{exp_name}' skipped: expected mapping."
            )
            continue
        key = row.get("array_key")
        if key not in arrays:
            _warn_restore(
                warnings,
                f"Original image snapshot for '{exp_name}' skipped: missing array key '{key}'.",
            )
            continue
        try:
            session.data["experiments"][exp_name]["original_mr_image"] = np.asarray(
                arrays[key]
            ).copy()
        except Exception as exc:
            _warn_restore(warnings, f"Original image snapshot for '{exp_name}' skipped: {exc}")
    session._ensure_original_image_snapshots()

    for exp_name, map_rows in _safe_dict_value(payload.get("fitted_maps")).items():
        if exp_name not in session.experiment_names:
            _warn_restore(warnings, f"Fitted maps for missing experiment '{exp_name}' skipped.")
            continue
        exp_data = session.data["experiments"][exp_name]
        restored_maps = dict(exp_data.get("fitted_maps") or {})
        units = dict(exp_data.get("fitted_map_units") or {})
        for idx, row in enumerate(map_rows or []):
            if not isinstance(row, Mapping):
                _warn_restore(
                    warnings, f"Fitted map row {idx} for '{exp_name}' skipped: expected mapping."
                )
                continue
            key = row.get("array_key")
            if key not in arrays:
                _warn_restore(
                    warnings,
                    f"Fitted map '{row.get('name') or key or idx}' for '{exp_name}' skipped: missing array key '{key}'.",
                )
                continue
            try:
                map_name = str(row.get("name") or key)
                unit = str(row.get("unit") or "")
                arr = np.asarray(arrays[key])
                restored_maps[map_name] = Quantity(arr, unit) if unit else arr
                if unit:
                    units[map_name] = unit
            except Exception as exc:
                _warn_restore(
                    warnings,
                    f"Fitted map '{row.get('name') or key}' for '{exp_name}' skipped: {exc}",
                )
        exp_data["fitted_maps"] = restored_maps
        exp_data["fitted_map_units"] = units

    session.shared_rois = _restore_roi_rows(
        payload.get("shared_rois") or [],
        arrays,
        warnings=warnings,
        context="shared ROI",
    )
    restored: dict[str, ExperimentViewState] = {}
    for exp_name, state_data in _safe_dict_value(payload.get("view_states")).items():
        if exp_name not in session.experiment_names:
            _warn_restore(warnings, f"View state for missing experiment '{exp_name}' skipped.")
            continue
        if not isinstance(state_data, Mapping):
            _warn_restore(warnings, f"View state for '{exp_name}' skipped: expected mapping.")
            continue
        rois = _restore_roi_rows(
            state_data.get("rois") or [],
            arrays,
            warnings=warnings,
            context=f"ROI in view state '{exp_name}'",
        )
        frame_mask = state_data.get("frame_mask")
        try:
            active_indices = {
                str(k): _safe_int_value(v, 0)
                for k, v in dict(state_data.get("active_indices") or {}).items()
            }
        except Exception as exc:
            _warn_restore(warnings, f"Active indices for '{exp_name}' reset to defaults: {exc}")
            active_indices = {}
        native_image_plane = session.native_image_plane_for_experiment(str(exp_name))
        active_image_plane = str(state_data.get("active_image_plane") or native_image_plane).lower()
        if active_image_plane not in {"axial", "coronal", "sagittal"}:
            _warn_restore(
                warnings,
                f"Image plane for '{exp_name}' reset to its native orientation: unsupported value.",
            )
            active_image_plane = native_image_plane
        try:
            restored[str(exp_name)] = ExperimentViewState(
                active_frame=_safe_int_value(state_data.get("active_frame"), 0),
                active_z=_safe_int_value(state_data.get("active_z"), 0),
                active_indices=active_indices,
                active_image_plane=active_image_plane,
                active_map=state_data.get("active_map"),
                active_kspace_mode=str(state_data.get("active_kspace_mode") or "log_magnitude"),
                active_kspace_centering=str(state_data.get("active_kspace_centering") or "auto"),
                active_reconstruction_component=str(
                    state_data.get("active_reconstruction_component") or "magnitude"
                ),
                active_reconstruction_coil=(
                    None
                    if state_data.get("active_reconstruction_coil") is None
                    else _safe_int_value(state_data.get("active_reconstruction_coil"), 0)
                ),
                active_reconstruction_unwrap_phase=bool(
                    state_data.get("active_reconstruction_unwrap_phase", True)
                ),
                rois=rois,
                fit_settings=_fit_settings_from_dict(state_data.get("fit_settings") or {}),
                frame_mask=None
                if frame_mask is None
                else FrameMask(np.asarray(frame_mask, dtype=bool)),
                visual_state=_visual_state_from_dict(state_data.get("visual_state") or {}),
            )
        except Exception as exc:
            _warn_restore(warnings, f"View state for '{exp_name}' skipped: {exc}")
    session.view_states = restored
    selected = payload.get("selected_experiment")
    if selected in session.experiment_names:
        session.selected_experiment = selected
    elif selected:
        _warn_restore(
            warnings,
            f"Selected experiment '{selected}' is missing; using the default selected experiment.",
        )
    session._restore_or_reset_current_view_state()


def _state_relative_path(path: Path | None, state_path: Path | None) -> str | None:
    if path is None:
        return None
    path = Path(path)
    if state_path is None:
        return str(path)
    try:
        return Path(os.path.relpath(path, Path(state_path).parent)).as_posix()
    except ValueError:
        return str(path)


def _fit_settings_to_dict(settings: FitSettings) -> dict[str, Any]:
    return {
        "model_name": settings.model_name,
        "optimizer_name": settings.optimizer_name,
        "num_steps": settings.num_steps,
        "step_size": settings.step_size,
        "selected_roi": settings.selected_roi,
        "mask_fill_value": settings.mask_fill_value,
        "auto_mask_nan_observables": settings.auto_mask_nan_observables,
        "use_view_state_data": settings.use_view_state_data,
        "compute_derived_maps": settings.compute_derived_maps,
        "sat_trans_reference_scan": settings.sat_trans_reference_scan,
        "ab_mt_t1_map_ref": settings.ab_mt_t1_map_ref,
        "ab_mt_b1_map_ref": settings.ab_mt_b1_map_ref,
        "ab_mt_b0_map_ref": settings.ab_mt_b0_map_ref,
        "selected_pools": list(settings.selected_pools),
        "param_overrides": settings.param_overrides,
        "pool_overrides": settings.pool_overrides,
        "regularizer_configs": settings.regularizer_configs,
        "parameter_constant_dims": {
            key: list(value) for key, value in settings.parameter_constant_dims.items()
        },
    }


def _fit_settings_from_dict(data: dict[str, Any] | Mapping[str, Any]) -> FitSettings:
    if not isinstance(data, Mapping):
        data = {}
    constant_dims: dict[str, tuple[Any, ...]] = {}
    raw_constant_dims = _safe_dict_value(data.get("parameter_constant_dims"))
    for key, value in raw_constant_dims.items():
        try:
            constant_dims[str(key)] = tuple(value)
        except Exception:
            continue
    return FitSettings(
        model_name=data.get("model_name"),
        optimizer_name=data.get("optimizer_name") or "auto",
        num_steps=_safe_int_value(data.get("num_steps"), 200),
        step_size=_safe_float_value(data.get("step_size"), 0.02),
        selected_roi=data.get("selected_roi"),
        mask_fill_value=_safe_float_value(data.get("mask_fill_value"), 0.0),
        auto_mask_nan_observables=bool(data.get("auto_mask_nan_observables", True)),
        use_view_state_data=bool(data.get("use_view_state_data", False)),
        compute_derived_maps=data.get("compute_derived_maps"),
        sat_trans_reference_scan=data.get("sat_trans_reference_scan"),
        ab_mt_t1_map_ref=data.get("ab_mt_t1_map_ref"),
        ab_mt_b1_map_ref=data.get("ab_mt_b1_map_ref"),
        ab_mt_b0_map_ref=data.get("ab_mt_b0_map_ref"),
        selected_pools=tuple(data.get("selected_pools") or ()),
        param_overrides=_safe_dict_value(data.get("param_overrides")),
        pool_overrides=_safe_dict_value(data.get("pool_overrides")),
        regularizer_configs=_safe_dict_value(data.get("regularizer_configs")),
        parameter_constant_dims=constant_dims,
    )


def _preview_result_to_json(result: dict[str, Any]) -> dict[str, Any]:
    keys = ("observed", "predicted", "included", "x")
    out = {key: _json_array(result.get(key)) for key in keys if key in result}
    out["x_name"] = result.get("x_name")
    out["model"] = result.get("model")
    for key in ("parameters", "relative_rmse", "x_unit"):
        if key in result:
            out[key] = _metadata_to_jsonable(result[key])
    return out


def _json_array(value: Any) -> Any:
    try:
        return np.asarray(value).tolist()
    except Exception:
        return value


def _safe_key(value: str) -> str:
    return base64.urlsafe_b64encode(value.encode("utf-8")).decode("ascii").rstrip("=")
