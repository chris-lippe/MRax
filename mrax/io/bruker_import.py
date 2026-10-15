"""Bruker image, metadata and raw acquisition import. Reconstruction is explicit."""
from __future__ import annotations

import logging
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import jax.numpy as jnp
import numpy as np

from mrax.data_struct import DataSet, Experiment, Model, Quantity, SpatialGeometry, Study
from mrax.devices import DeviceLike, device_function
from mrax.io.import_definitions import (
    ImportDefinition,
    build_imported_experiment_dataset,
    detect_experiment_type,
    extract_observables_from_metadata,
    needs_manual_type_override,
)
from mrax.library.default_experiments import default_model_for_type

from .bruker_acquisition import read_bruker_acquisition
from .bruker_images import (
    _bruker_image_orientation_metadata,
    _bruker_visu_scaling_metadata,
    bruker_spatial_geometry,
    load_2dseq,
)
from .bruker_metadata import (
    _apply_recipe,
    _attach_fit_override_warning,
    _bruker_non_imaging_role,
    _bruker_sequence_type,
    _fit_override_warning_message,
    _infer_tr_from_timestamps,
    _manual_fit_override_warning_message,
    _maybe_reorder_fair_frames,
    _normalize_bruker_metadata,
    _normalize_study_globals,
    _override_fair_exp_type,
    _read_timestamp_file,
    _resolve_pdata_index,
    _sat_trans_rep_count,
)
from .bruker_parameters import (
    _as_positive_int,
    _flatten_numeric,
    discover_bruker_experiments,
    parse_jcamp,
)

RecipeResult = Tuple[str | None, dict | None, Model | Callable[[], Model] | None]
_LOG = logging.getLogger(__name__)
ExperimentRecipe = Callable[[dict, dict, int], str | RecipeResult]
PDataIndex = str | int | Dict[str | int, str | int]
DEFAULT_PDATA_INDEX = "1"

def _attach_acquisition(dataset, folder, pdata_index, *, definition_paths, profiles, strict_raw, image_frame_order=None):
    try:
        acquisition, report = read_bruker_acquisition(folder, pdata_index=pdata_index,
            definition_paths=definition_paths, profiles=profiles)
        if image_frame_order is not None:
            order = np.asarray(image_frame_order, int)
            if order.shape != (acquisition.frame_count,) or acquisition.recipe.stack_slices != 1:
                raise ValueError("Image-frame reordering requires an explicit mapping to acquisition readouts")
            recipe = acquisition.recipe
            options = {}
            for name in ("acquired", "statistical_weights"):
                value = getattr(acquisition, name)
                if value is not None and value.ndim == acquisition.samples.ndim and value.shape[0] == acquisition.frame_count:
                    options[name] = value[order]
            acquisition = replace(acquisition, samples=acquisition.samples[order],
                frame_coordinates={name: np.asarray(value)[order] for name, value in acquisition.frame_coordinates.items()},
                recipe=replace(recipe,
                    frame_image_shifts=tuple(recipe.frame_image_shifts[i] for i in order) if recipe.frame_image_shifts else (),
                    frame_permutations=tuple(recipe.frame_permutations[i] for i in order) if recipe.frame_permutations else ()), **options)
    except (OSError, ValueError) as error:
        if strict_raw:
            raise
        warnings.warn(f"Raw acquisition preparation failed: {error}", RuntimeWarning, stacklevel=3)
        return {"status": "unavailable", "error_type": type(error).__name__, "message": str(error)}
    dataset.acquisition = acquisition
    dataset.kspace = jnp.asarray(acquisition.samples)
    dataset.trajectory = None if acquisition.trajectory is None else jnp.asarray(acquisition.trajectory)
    dataset.density = None if acquisition.density is None else jnp.asarray(acquisition.density)
    dataset.sampling_mask = None if acquisition.acquired is None else jnp.asarray(acquisition.acquired)
    dataset.kspace_metadata = {**report, "status": "loaded", "kind": acquisition.kind,
        "kspace_ready": acquisition.domain == "kspace",
        "acquisition_schema_version": acquisition.schema_version,
        "reconstruction_status": "not_run", "sample_domain": acquisition.domain}
    return dataset.kspace_metadata

@device_function
def import_bruker_experiment(
    exp_dir: Path,
    override_type: Optional[str] = None,
    model_map_by_type: Optional[Dict[str, Model | Callable[[], Model]]] = None,
    model_map_by_name: Optional[Dict[str, Model | Callable[[], Model]]] = None,
    study_globals: Optional[Dict[str, Quantity]] = None,
    pdata_index: PDataIndex = DEFAULT_PDATA_INDEX,
    recipe: ExperimentRecipe | None = None,
    load_raw_kspace: bool = True,
    rotate_bvecs_to_image: bool = True,
    definition_paths=(),
    profiles=None,
    strict_raw: bool = True,
    device: DeviceLike = "cpu",
) -> Tuple[Experiment, DataSet, Model, dict]:
    exp_dir = Path(exp_dir)
    resolved_pdata_index = _resolve_pdata_index(pdata_index, exp_dir.name)
    method = parse_jcamp(exp_dir / "method")
    acqp = parse_jcamp(exp_dir / "acqp") if (exp_dir / "acqp").exists() else {}
    pdata_dir = exp_dir / "pdata" / resolved_pdata_index
    reco_params = parse_jcamp(pdata_dir / "reco") if (pdata_dir / "reco").exists() else {}
    data, visu, spatial_shape = load_2dseq(pdata_dir)
    study_globals = _normalize_study_globals(study_globals)
    recipe_type, recipe_obs, recipe_model = _apply_recipe(recipe, visu, method, frame_count=data.shape[0])
    frame_count = data.shape[0]
    normalized_meta = _normalize_bruker_metadata(
        visu,
        method,
        frame_count=frame_count,
        rotate_bvecs_to_image=rotate_bvecs_to_image,
    )
    exp_type = override_type or recipe_type or detect_experiment_type(normalized_meta)
    manual_type_override_required = False
    if override_type is None and recipe_type is None:
        exp_type = _override_fair_exp_type(exp_type, method.get("PVM_FairMode"))
        manual_type_override_required = needs_manual_type_override(normalized_meta, exp_type)
    obs = (
        recipe_obs
        if recipe_obs is not None
        else extract_observables_from_metadata(exp_type, normalized_meta, frame_count)
    )
    fair_order = None
    if exp_type == "fair_asl":
        data, obs, fair_order = _maybe_reorder_fair_frames(
            data,
            obs,
            fair_mode=method.get("PVM_FairMode"),
        )
    experiment, _ = build_imported_experiment_dataset(
        ImportDefinition(
            name=str(exp_dir.name),
            data=np.asarray(data),
            frame_count=frame_count,
            spatial_shape=spatial_shape,
            exp_type=exp_type,
            observables=obs,
        )
    )
    if exp_type == "sat_trans":
        timestamps = _read_timestamp_file(exp_dir)
        if timestamps is not None:
            nominal_tr_ms = None
            tr_vals = _flatten_numeric(normalized_meta.repetition_times)
            if tr_vals:
                nominal_tr_ms = float(tr_vals[0])
            rep_count = _sat_trans_rep_count(exp_type, experiment.dims, frame_count)
            tr_full_s = _infer_tr_from_timestamps(
                timestamps,
                nominal_tr_ms=nominal_tr_ms,
                rep_count=rep_count,
            )
            if tr_full_s is not None:
                obs = dict(obs)
                obs["TR"] = Quantity(np.asarray(tr_full_s, dtype=np.float32) * 1e3, "ms")
                experiment, _ = build_imported_experiment_dataset(
                    ImportDefinition(
                        name=str(exp_dir.name),
                        data=np.asarray(data),
                        frame_count=frame_count,
                        spatial_shape=spatial_shape,
                        exp_type=exp_type,
                        observables=obs,
                    )
                )
    sequence_type = _bruker_sequence_type(method, visu)
    from .bruker_storage import discover_raw_stream
    stream, _ = discover_raw_stream(exp_dir, acqp)
    raw_meta = {"status": "not_loaded" if stream is not None else "unavailable", "lazy_available": stream is not None}
    experiment, dataset = build_imported_experiment_dataset(ImportDefinition(
        name=str(exp_dir.name), data=np.asarray(data), frame_count=frame_count,
        spatial_shape=spatial_shape, exp_type=exp_type, observables=obs,
        sequence_type=sequence_type, kspace_metadata=raw_meta,
        spatial_geometry=bruker_spatial_geometry(visu, spatial_shape=spatial_shape)))
    experiment.study_globals = study_globals
    model_from_default = False
    if recipe_model is not None:
        candidate = recipe_model
        model = candidate() if callable(candidate) else candidate
    elif model_map_by_name and exp_dir.name in model_map_by_name:
        candidate = model_map_by_name[exp_dir.name]
        model = candidate() if callable(candidate) else candidate
    elif model_map_by_type and exp_type in model_map_by_type:
        candidate = model_map_by_type[exp_type]
        model = candidate() if callable(candidate) else candidate
    else:
        model = default_model_for_type(exp_type)
        model_from_default = True

    default_identity_model = model_from_default and str(getattr(model, "name", "")).lower() == "identity"
    identity_message = None
    manual_message = None
    if default_identity_model:
        identity_message = _fit_override_warning_message(
            normalized_meta,
            exp_dir,
            exp_type=exp_type,
            default_identity_model=default_identity_model,
            manual_type_override_required=manual_type_override_required,
        )
    if manual_type_override_required:
        manual_message = _manual_fit_override_warning_message(normalized_meta, exp_dir)
    if identity_message or manual_message:
        _attach_fit_override_warning(
            experiment,
            model,
            identity_message=identity_message,
            manual_message=manual_message,
        )

    # Attach default observables for models that expect them when called directly.
    global_obs = study_globals
    default_obs = {**global_obs, **experiment.observables}
    if hasattr(model, "model_apply"):
        base_apply = model.model_apply

        def model_apply_with_defaults(
            fields,
            offsets,
            observables=None,
            base_apply=base_apply,
            default_obs=default_obs,
        ):
            merged_obs = default_obs if observables is None else {**default_obs, **observables}
            return base_apply(fields, offsets, merged_obs)

        model.model_apply = model_apply_with_defaults  # type: ignore[assignment]
    if hasattr(model, "derived_maps"):
        base_derived = model.derived_maps  # type: ignore[attr-defined]

        def derived_with_defaults(fields, observables=None, base_derived=base_derived, default_obs=default_obs):
            merged_obs = default_obs if observables is None else {**default_obs, **observables}
            return base_derived(fields, merged_obs)

        model.derived_maps = derived_with_defaults  # type: ignore[attr-defined]
    scaling_frame_count = _as_positive_int(visu.get("VisuCoreFrameCount"), frame_count) or int(frame_count)
    image_scaling = _bruker_visu_scaling_metadata(visu, frame_count=scaling_frame_count)
    if int(frame_count) != int(scaling_frame_count):
        image_scaling["dataset_frame_count"] = int(frame_count)
    non_imaging_role = _bruker_non_imaging_role(
        exp_type=exp_type,
        sequence_type=sequence_type,
        spatial_shape=spatial_shape,
        method=method,
        visu=visu,
    )
    if non_imaging_role:
        try:
            experiment._mrax_non_imaging = True
            experiment._mrax_data_role = non_imaging_role
        except Exception:
            _LOG.debug("Ignored optional operation failure", exc_info=True)

    metadata = {
        "scan_no": str(exp_dir.name),
        "path": exp_dir,
        "type": exp_type,
        "pdata_index": resolved_pdata_index,
        "visu": visu,
        "method": method,
        "acqp": acqp,
        "reco": reco_params,
        "image_scaling": image_scaling,
        "mixed_slice_geometry": bool(visu.get("_mrax_mixed_slice_geometry", False)),
        "sequence_type": sequence_type,
        "image_frame_order": None if fair_order is None else fair_order.tolist(),
        "raw_import": raw_meta,
        "spatial_geometry": (
            dataset.spatial_geometry.to_metadata()
            if dataset.spatial_geometry is not None
            else SpatialGeometry(source="bruker").to_metadata()
        ),
        **_bruker_image_orientation_metadata(visu),
    }
    if non_imaging_role:
        metadata["non_imaging"] = True
        metadata["data_role"] = non_imaging_role
    if load_raw_kspace and stream is not None:
        metadata["raw_import"] = _attach_acquisition(dataset, exp_dir, resolved_pdata_index,
            definition_paths=definition_paths, profiles=profiles, strict_raw=strict_raw,
            image_frame_order=metadata.get("image_frame_order"))
    return experiment, dataset, model, metadata

@device_function
def import_bruker_study(
    study_root: str | Path,
    type_overrides: Optional[Dict[str, str]] = None,
    model_map_by_type: Optional[Dict[str, Model | Callable[[], Model]]] = None,
    model_map_by_name: Optional[Dict[str, Model | Callable[[], Model]]] = None,
    study_globals: Optional[Dict[str, Quantity]] = None,
    field_strength_mhz: float | Quantity | None = None,
    pdata_index: PDataIndex = DEFAULT_PDATA_INDEX,
    recipe: ExperimentRecipe | None = None,
    load_raw_kspace: bool = True,
    rotate_bvecs_to_image: bool = True,
    definition_paths=(),
    profiles=None,
    strict_raw: bool = True,
    device: DeviceLike = "cpu",
    progress_callback: Callable[[int, int, str], None] | None = None,
):
    root = Path(study_root)
    study_globals = _normalize_study_globals(study_globals, field_strength_mhz)
    if "field_strength_mhz" not in study_globals:
        warnings.warn(
            "No field_strength_mhz provided for this study. Set field_strength_mhz=... or include it in study_globals; "
            "models that require it will raise errors when fitted.",
            RuntimeWarning,
            stacklevel=2,
        )
    entries: List[dict] = []
    skipped_entries: List[dict] = []
    scan_dirs = discover_bruker_experiments(root)
    total_steps = max(1, len(scan_dirs) + 1)
    if progress_callback is not None:
        progress_callback(0, total_steps, "Scanning Bruker study")
    for scan_idx, entry in enumerate(scan_dirs, start=1):
        if progress_callback is not None:
            progress_callback(scan_idx, total_steps, f"Importing Bruker scan {entry.name}")
        pdata_idx = _resolve_pdata_index(pdata_index, entry.name)
        pdata = entry / "pdata" / pdata_idx / "2dseq"
        method = entry / "method"
        missing_files = [str(path) for path in (method, pdata) if not path.exists()]
        if missing_files:
            warnings.warn(
                f"warning: skipped Bruker scan {entry.name}: missing file(s): {', '.join(missing_files)}",
                RuntimeWarning,
                stacklevel=2,
            )
            skipped_entries.append(
                {
                    "scan_no": entry.name,
                    "path": str(entry),
                    "pdata_index": pdata_idx,
                    "status": "skipped",
                    "reason": "missing_file",
                    "missing_files": missing_files,
                    "message": "Missing required Bruker file(s): " + ", ".join(missing_files),
                }
            )
            if progress_callback is not None:
                progress_callback(scan_idx, total_steps, f"Skipped Bruker scan {entry.name}")
            continue
        override = None
        if type_overrides and entry.name in type_overrides:
            override = type_overrides[entry.name]
        try:
            experiment, dataset, model, meta = import_bruker_experiment(
                entry,
                override,
                model_map_by_type,
                model_map_by_name,
                study_globals=study_globals,
                pdata_index=pdata_idx,
                recipe=recipe,
                load_raw_kspace=load_raw_kspace,
                rotate_bvecs_to_image=rotate_bvecs_to_image,
                definition_paths=definition_paths, profiles=profiles, strict_raw=strict_raw, device=device,
            )
            entries.append(
                {
                    "scan_no": meta["scan_no"],
                    "experiment": experiment,
                    "dataset": dataset,
                    "model": model,
                    "visu": meta["visu"],
                    "method": meta["method"],
                    "acqp": meta.get("acqp", {}),
                    "reco": meta.get("reco", {}),
                    "image_scaling": meta.get("image_scaling", {}),
                    "raw_import": meta.get("raw_import", {}),
                    "image_frame_order": meta.get("image_frame_order"),
                    "image_axis_anatomical_labels": meta.get("image_axis_anatomical_labels"),
                    "native_image_plane": meta.get("native_image_plane"),
                    "type": meta["type"],
                    "path": meta["path"],
                    "pdata_index": meta["pdata_index"],
                }
            )
        except Exception as exc:
            missing = []
            if isinstance(exc, FileNotFoundError) and getattr(exc, "filename", None):
                missing = [str(exc.filename)]
            warnings.warn(
                f"warning: skipped Bruker scan {entry.name}: {type(exc).__name__}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            skipped_entries.append(
                {
                    "scan_no": entry.name,
                    "path": str(entry),
                    "pdata_index": pdata_idx,
                    "status": "skipped",
                    "reason": "missing_file" if missing else "import_failed",
                    "missing_files": missing,
                    "message": str(exc),
                    "error_type": type(exc).__name__,
                }
            )
            if progress_callback is not None:
                progress_callback(scan_idx, total_steps, f"Skipped Bruker scan {entry.name}")
            continue
        if progress_callback is not None:
            progress_callback(scan_idx, total_steps, f"Imported Bruker scan {entry.name}")

    if progress_callback is not None:
        progress_callback(total_steps, total_steps, "Building MRax study")
    study = Study(
        experiments=[e["experiment"] for e in entries],
        datasets=[e["dataset"] for e in entries],
        models=[e["model"] for e in entries],
        names=[e["scan_no"] for e in entries],
        study_globals=study_globals,
    )
    metadata = {
        "subject": root.name,
        "experiments": {
            e["scan_no"]: {
                "scan_no": e["scan_no"],
                "study_subject": root.name,
                "type": e["type"],
                "path": str(e["path"]),
                "pdata_index": e["pdata_index"],
                "status": "loaded",
                "visu": e["visu"],
                "method": e["method"],
                "acqp": e.get("acqp", {}),
                "reco": e.get("reco", {}),
                "image_scaling": e.get("image_scaling", {}),
                "raw_import": e.get("raw_import", {}),
                "image_frame_order": e.get("image_frame_order"),
                "spatial_geometry": e.get("spatial_geometry"),
                "image_axis_anatomical_labels": e.get("image_axis_anatomical_labels"),
                "native_image_plane": e.get("native_image_plane"),
            }
            for e in entries
        },
        "skipped_experiments": {e["scan_no"]: dict(e) for e in skipped_entries},
        "import_issues": [dict(e) for e in skipped_entries],
    }
    return study, metadata

__all__ = ["import_bruker_experiment", "import_bruker_study"]
