"""Existing toolbox operations as reproducible steps in a study's data workflow."""

from __future__ import annotations

import copy
import importlib.util
import inspect
import math
from dataclasses import replace
from uuid import uuid4

import numpy as np

from mrax.analysis import (
    ROI,
    apply_baseline,
    apply_dilution_correction,
    compute_dti_maps,
    compute_z_spectra,
    nan_gaussian_filter,
    replace_isolated_spikes,
)
from mrax.analysis.roi import broadcast_mask
from mrax.data_struct import Quantity
from mrax.data_struct.dimensions import reshape_dims_to_frames, reshape_frames_to_dims
from mrax.gui.registration import inverse_transform, resample_mask_with_transform
from mrax.preprocessing import denoise_bm3d, denoise_bm4d, denoise_filter, regularized_denoise

from .cloning import clone_workspace
from .scene import fitting_roi, require_common_grid
from .view_data import scalar_component
from .view_state import WorkspaceFitInputs

TENSOR_FIELDS = ("dxx", "dyy", "dzz", "dxy", "dxz", "dyz")
FILTERS = {"gaussian", "median", "uniform", "regularized", "bm3d", "bm4d"}


def _default(function, name):
    return inspect.signature(function).parameters[name].default


def _field(
    key,
    label,
    function=None,
    *,
    value=None,
    kind="number",
    choices=(),
    minimum=None,
    maximum=None,
    nullable=False,
):
    return {
        "key": key,
        "label": label,
        "kind": kind,
        "default": _default(function, key) if function is not None else value,
        "choices": list(choices),
        "minimum": minimum,
        "maximum": maximum,
        "nullable": nullable,
    }


def processing_methods(session):
    gaussian = [
        _field("sigma", "Sigma (voxels)", denoise_filter, minimum=0, maximum=100),
        _field("preserve_nan", "NaN-aware Gaussian", denoise_filter, kind="boolean"),
    ]
    boundary = _field(
        "mode",
        "Boundary handling",
        denoise_filter,
        kind="choice",
        choices=("reflect", "symmetric", "edge", "wrap"),
    )
    complex_mode = _field(
        "complex_mode",
        "Complex data",
        denoise_filter,
        kind="choice",
        choices=("real_imag", "magnitude"),
    )
    methods = []
    for key, label in (
        ("gaussian", "Gaussian denoising"),
        ("median", "Median filtering"),
        ("uniform", "Mean filtering"),
    ):
        fields = (
            gaussian
            if key == "gaussian"
            else [
                _field(
                    "size",
                    "Kernel size (voxels)",
                    denoise_filter,
                    kind="integer",
                    minimum=1,
                    maximum=31,
                )
            ]
        )
        methods.append(
            {
                "id": key,
                "label": label,
                "group": "Denoise",
                "fields": [*fields, boundary, complex_mode],
                "axes": "filter",
                "reference": None,
                "unavailable": None,
            }
        )
    methods.append(
        {
            "id": "regularized",
            "label": "Regularized denoising",
            "group": "Denoise",
            "axes": "filter",
            "reference": None,
            "unavailable": None,
            "fields": [
                _field(
                    "regularizer",
                    "Regularizer",
                    regularized_denoise,
                    kind="choice",
                    choices=("tv", "tv2", "smoothness"),
                ),
                _field(
                    "weight", "Regularization weight", regularized_denoise, minimum=0, maximum=1e6
                ),
                _field(
                    "num_steps",
                    "Denoising iterations",
                    regularized_denoise,
                    kind="integer",
                    minimum=1,
                    maximum=10000,
                ),
                _field(
                    "step_size",
                    "Denoising step size",
                    regularized_denoise,
                    minimum=1e-12,
                    maximum=10,
                ),
                _field(
                    "optimizer_name",
                    "Denoising optimizer",
                    regularized_denoise,
                    kind="choice",
                    choices=("adam", "adamw", "sgd", "rmsprop"),
                ),
                complex_mode,
            ],
        }
    )
    for key, function in (("bm3d", denoise_bm3d), ("bm4d", denoise_bm4d)):
        methods.append(
            {
                "id": key,
                "label": key.upper() + " denoising",
                "group": "Denoise",
                "axes": "filter",
                "reference": None,
                "unavailable": None
                if importlib.util.find_spec(key)
                else f"Requires the optional {key} package (mrax[denoise-external]).",
                "fields": [
                    _field(
                        "sigma",
                        "Noise sigma (automatic if empty)",
                        function,
                        minimum=0,
                        maximum=1e12,
                        nullable=True,
                    ),
                    _field(
                        "stage",
                        "Denoising stage",
                        function,
                        kind="choice",
                        choices=("full", "basic"),
                    ),
                    complex_mode,
                ],
            }
        )
    axis = next(
        (
            name
            for name in session.experiment_dims
            if name not in session.spatial_dim_names and session.experiment_dims[name] > 1
        ),
        None,
    )
    count = session.experiment_dims.get(axis, 1)
    unavailable = (
        None
        if axis is not None and count > 1
        else "Requires a series with a non-spatial acquisition axis."
    )
    baseline = _field(
        "baseline_frames",
        "Baseline samples",
        value=min(_default(apply_baseline, "baseline_frames"), count),
        kind="integer",
        minimum=1,
    )
    methods.extend(
        [
            {
                "id": "baseline",
                "label": "Baseline correction",
                "group": "Time series",
                "axes": "series",
                "reference": None,
                "unavailable": unavailable,
                "fields": [
                    baseline,
                    _field(
                        "mode",
                        "Baseline correction",
                        apply_baseline,
                        kind="choice",
                        choices=("subtract", "divide"),
                    ),
                ],
            },
            {
                "id": "temporal_gaussian",
                "label": "Temporal Gaussian smoothing",
                "group": "Time series",
                "axes": "series",
                "reference": None,
                "unavailable": unavailable,
                "fields": [_field("sigma", "Sigma (samples)", value=1.0, minimum=0, maximum=100)],
            },
            {
                "id": "despike",
                "label": "Isolated-spike correction",
                "group": "Time series",
                "axes": "series",
                "reference": None,
                "unavailable": unavailable
                if count >= 3
                else "Requires at least three samples on the acquisition axis.",
                "fields": [
                    _field(
                        "threshold",
                        "Spike ratio threshold",
                        replace_isolated_spikes,
                        minimum=1.000001,
                        maximum=100,
                    )
                ],
            },
            {
                "id": "normalize",
                "label": "Reference normalization / Z spectra",
                "group": "Normalize",
                "axes": "none",
                "reference": "frame",
                "unavailable": None,
                "fields": [
                    _field(
                        "epsilon",
                        "Minimum reference magnitude",
                        compute_z_spectra,
                        minimum=1e-15,
                        maximum=1e6,
                    )
                ],
            },
            {
                "id": "dilution",
                "label": "Dilution correction",
                "group": "Normalize",
                "axes": "none",
                "reference": "series",
                "unavailable": unavailable,
                "fields": [
                    baseline,
                    _field(
                        "clip_min",
                        "Minimum dilution ratio",
                        apply_dilution_correction,
                        minimum=1e-12,
                        maximum=1e6,
                    ),
                    _field(
                        "smooth_sigma",
                        "Dilution smoothing (samples)",
                        apply_dilution_correction,
                        minimum=0,
                        maximum=100,
                    ),
                ],
            },
            {
                "id": "dti",
                "label": "Diffusion tensor metrics",
                "group": "Derived maps",
                "axes": "none",
                "reference": None,
                "unavailable": None
                if all(name in session.map_names for name in TENSOR_FIELDS)
                else "Requires fitted dxx, dyy, dzz, dxy, dxz and dyz maps.",
                "fields": [],
            },
        ]
    )
    return methods


def processing_catalog(workspace):
    from .pipeline import pipeline_info

    session = workspace.session
    if not session.is_loaded:
        return {
            "methods": [],
            "sources": [],
            "axes": [],
            "spatial_axes": [],
            "series_axes": [],
            "history": [],
        }
    sources = [
        {"id": "view", "label": "Image with View processing"},
        {"id": "image", "label": "Current image data"},
        {"id": "original", "label": "Original image data"},
    ]
    if session.reconstruction_array is not None:
        sources.append({"id": "reconstruction", "label": "MRax reconstruction"})
    sources.extend(
        {"id": "map:" + name, "label": "Map: " + name}
        for name in session.map_names
        if session.map_array(name).shape == session.spatial_shape
    )
    return {
        "methods": processing_methods(session),
        "sources": sources,
        "axes": list(session.experiment_dims),
        "spatial_axes": session.spatial_dim_names,
        "series_axes": [
            name
            for name in session.experiment_dims
            if name not in session.spatial_dim_names and session.experiment_dims[name] > 1
        ],
        "history": pipeline_info(workspace)["steps"],
    }


def _settings(method, supplied):
    fields = {field["key"]: field for field in method["fields"]}
    if not isinstance(supplied, dict) or set(supplied) - fields.keys():
        raise ValueError("Unknown processing parameter")
    values = {key: copy.deepcopy(field["default"]) for key, field in fields.items()}
    values.update(supplied)
    for key, value in values.items():
        field = fields[key]
        if value is None and field["nullable"]:
            continue
        if field["kind"] == "boolean":
            valid = isinstance(value, bool)
        elif field["kind"] == "choice":
            valid = value in field["choices"]
        else:
            valid = (
                isinstance(value, (float, int))
                and not isinstance(value, bool)
                and math.isfinite(value)
            )
            valid = valid and (field["kind"] != "integer" or int(value) == value)
            valid = (
                valid
                and (field["minimum"] is None or value >= field["minimum"])
                and (field["maximum"] is None or value <= field["maximum"])
            )
        if not valid:
            raise ValueError("Check " + field["label"].lower())
        if field["kind"] == "integer":
            values[key] = int(value)
    return values


def _source(workspace, key, component):
    session = workspace.session
    pipeline = workspace.pipelines.get(session.selected_experiment)
    if pipeline and key in {"original", "reconstruction"}:
        from .pipeline import install

        context = clone_workspace(workspace)
        install(context, pipeline["source"])
        session = context.session
    elif key.startswith("map:"):
        from .pipeline import display_session

        session = display_session(workspace, fitted=True)
    experiment = session.build_experiment()
    dims = dict(experiment.dims)
    geometry = session.build_dataset().spatial_geometry
    rois = list(session.rois)
    if (
        pipeline
        and key in {"image", "view"}
        and not getattr(workspace, "_pipeline_evaluating", False)
    ):
        from .pipeline import project_roi

        rois = [project_roi(workspace, roi) for roi in rois]
    model = session.fit_settings.model_name or session.current_model_name
    unit = (session.current_experiment.get("extra_metadata") or {}).get("image_unit", "")
    if key.startswith("map:"):
        name = key.removeprefix("map:")
        if name not in session.map_names or session.map_array(name).shape != session.spatial_shape:
            raise ValueError("Choose a scalar map on the experiment's native grid")
        data = session.map_array(name)
        dims = {name: dims[name] for name in session.spatial_dim_names}
        observables, model, unit = {}, "identity", session.map_unit(name) or ""
    else:
        observables = copy.deepcopy(experiment.observables)
        if key == "image":
            data = session.image_array
        elif key == "original":
            data = session.original_image_array
        elif key == "reconstruction" and session.reconstruction_array is not None:
            data = session.reconstruction_display_array()
            unit = "rad" if session.active_reconstruction_component == "phase" else unit
        elif key == "view":
            context = clone_workspace(workspace)
            context.session.fit_settings.use_view_state_data = True
            kwargs = WorkspaceFitInputs(context)._visual_state_fit_kwargs(include_roi=False)
            dataset = kwargs.get("input_dataset", session.build_dataset())
            data = np.asarray(dataset.data)
            dims = dict(kwargs.get("input_experiment", experiment).dims)
            state = session.visual_state
            if not state.show_original and state.transform_key:
                result = workspace.registration_results[state.transform_key]
                inverse = state.transform_direction == "reference_to_moving"
                reference = result["moving"] if inverse else result["fixed"]
                transform = (
                    inverse_transform(result["transform"]) if inverse else result["transform"]
                )
                if any(np.asarray(roi.mask).ndim != 2 for roi in rois):
                    raise ValueError(
                        "Processing a transformed view currently requires two-dimensional ROI masks"
                    )
                rois = [
                    ROI(
                        roi.name,
                        resample_mask_with_transform(
                            roi.mask, reference, transform, output_dtype=np.asarray(roi.mask).dtype
                        ),
                        dims=tuple(session.spatial_dim_names),
                        metadata={**roi.metadata, "kind": "mask", "consolidated": True},
                    )
                    for roi in rois
                ]
                reference_name = result["metadata"][
                    "moving_experiment" if inverse else "reference_experiment"
                ]
                context.session.select_experiment(reference_name)
                geometry = context.session.build_dataset().spatial_geometry
                if geometry is not None:
                    reference_frame = result["metadata"][
                        "moving_frame" if inverse else "reference_frame"
                    ]
                    geometry = replace(
                        geometry,
                        affine=geometry.affine_for_frame(reference_frame),
                        frame_affines=(),
                        spatial_shape=tuple(dims[dim] for dim in session.spatial_dim_names),
                    )
        else:
            raise ValueError("Choose an available processing source")
    data = np.asarray(data)
    if data.shape != tuple(dims.values()):
        raise ValueError("The input shape does not match its declared dimensions")
    if component != "auto":
        data = scalar_component(data, component)
        if component == "phase":
            unit, model = "rad", "identity"
    return data, dims, observables, model, unit, geometry, rois


def _same_grid(source, dims, spatial_dims, geometry, *, series, reference_frame):
    count = math.prod(size for name, size in dims.items() if name not in spatial_dims)
    for frame in range(count):
        require_common_grid(
            source.spatial_dim_names,
            source.spatial_shape,
            source.build_dataset().spatial_geometry,
            spatial_dims,
            tuple(dims[name] for name in spatial_dims),
            geometry,
            source_frame=frame if series else reference_frame,
            target_frame=frame,
        )


def _reference(workspace, options, dims, unit, geometry, *, series):
    name = options.get("reference_experiment")
    if name not in workspace.session.experiment_names:
        raise ValueError("Choose a reference experiment")
    context = clone_workspace(workspace).session
    context.select_experiment(name)
    index = options.get("reference_frame", 0)
    if (
        not isinstance(index, int)
        or isinstance(index, bool)
        or not 0 <= index < context.frame_count
    ):
        raise ValueError("Reference frame is out of range")
    _same_grid(
        context,
        dims,
        workspace.session.spatial_dim_names,
        geometry,
        series=series,
        reference_frame=index,
    )
    array = np.asarray(context.image_array)
    component = options.get("component", "auto")
    reference_unit = (context.current_experiment.get("extra_metadata") or {}).get("image_unit", "")
    if np.iscomplexobj(array) and component == "auto":
        raise ValueError("Choose an input component explicitly for complex reference normalization")
    if component != "auto":
        array = scalar_component(array, component)
        if component == "phase":
            reference_unit = "rad"
    if unit != reference_unit:
        raise ValueError("Input and reference data must have matching units")
    if series:
        if context.experiment_dims != dims:
            raise ValueError("Dilution reference dimensions must match the input series")
        original_observables = workspace.session.build_experiment().observables
        references = context.build_experiment().observables
        for key, value in original_observables.items():
            if np.asarray(value.values).size > 1 and (
                key not in references
                or value.si_unit != references[key].si_unit
                or np.asarray(value.values).shape != np.asarray(references[key].values).shape
                or not np.allclose(value.values, references[key].values, equal_nan=True)
            ):
                raise ValueError("Dilution reference acquisition coordinates must match the input")
        return reshape_dims_to_frames(array, dims)
    return np.asarray(reshape_dims_to_frames(array, context.experiment_dims))[index]


def _filter(data, axes, method, parameters, progress):
    other = [axis for axis in range(data.ndim) if axis not in axes]
    order = other + axes
    grouped = data.transpose(order)
    shape = grouped.shape
    blocks = grouped.reshape((-1,) + tuple(data.shape[axis] for axis in axes))
    output = None
    for index, block in enumerate(blocks):
        progress(index, len(blocks), f"{method}: block {index + 1} of {len(blocks)}")
        if method in {"gaussian", "median", "uniform"}:
            result = denoise_filter(block, method=method, **parameters)
        elif method == "regularized":
            result = regularized_denoise(block, axes=tuple(range(block.ndim)), **parameters)
        else:
            function = denoise_bm3d if method == "bm3d" else denoise_bm4d
            values = {key: value for key, value in parameters.items() if key != "complex_mode"}
            if np.iscomplexobj(block) and parameters["complex_mode"] == "real_imag":
                result = function(block.real, **values) + 1j * function(block.imag, **values)
            else:
                result = function(np.abs(block) if np.iscomplexobj(block) else block, **values)
        result = np.asarray(result)
        if output is None:
            output = np.empty(blocks.shape, dtype=result.dtype)
        output[index] = result
    progress(len(blocks), len(blocks), "Filtering complete")
    if output is None:
        raise ValueError("The input has no blocks to process")
    return output.reshape(shape).transpose(np.argsort(order))


def _create_result(workspace, data, context, options, method, parameters, source_name):
    from .pipeline import install, snapshot

    session = workspace.session
    _original, dims, observables, model, unit, geometry, _rois = context
    if method["id"] == "normalize" or (
        method["id"] == "baseline" and parameters["mode"] == "divide"
    ):
        unit, model = "1", "identity"
    if data.shape != tuple(dims.values()) or not np.any(np.isfinite(data)):
        raise ValueError("Processing must produce finite samples on the declared input grid")
    state = snapshot(session)
    state.update(
        data=data,
        dims=dims,
        observables=observables,
        model=model,
        unit=unit,
        geometry=None if geometry is None else geometry.to_metadata(),
    )
    install(workspace, state)
    return {"experiment": source_name, "method": method["id"]}


def _dti(workspace, options, progress):
    session = workspace.session
    if any(np.iscomplexobj(session.fitted_maps[name]) for name in TENSOR_FIELDS):
        raise ValueError("Diffusion tensor coefficients must be real")
    units = [session.map_unit(name) for name in TENSOR_FIELDS]
    if any(unit != units[0] for unit in units):
        raise ValueError("Diffusion tensor components must use the same units")
    mask = (
        fitting_roi(workspace, options["roi_names"]).mask
        if options.get("roi_names") is not None
        else None
    )
    progress(0, 1, "Computing diffusion tensor metrics")
    metrics = compute_dti_maps(
        {name: session.fitted_maps[name] for name in TENSOR_FIELDS}, mask=mask
    )
    if not np.any(np.isfinite(metrics.fa)):
        raise ValueError("No finite tensor metrics are available inside the selected region")
    maps = dict(session.fitted_maps)
    created = []
    for key in ("adc", "fa", "md", "ad", "rd", "trace"):
        name = "DTI " + key.upper()
        index = 2
        while name in maps:
            name = f"DTI {key.upper()} {index}"
            index += 1
        unit = "1" if key == "fa" else units[0]
        value = np.asarray(getattr(metrics, key))
        maps[name] = Quantity(value, unit) if unit else value
        created.append(name)
    session.replace_fitted_maps(maps)
    session.active_map = created[1]
    workspace.view_label = "Map"
    history = list(
        (session.current_experiment.get("extra_metadata") or {}).get("processing_history", [])
    )
    history.append(
        {
            "method": "dti",
            "label": "Diffusion tensor metrics",
            "source_experiment": session.selected_experiment,
            "source_maps": list(TENSOR_FIELDS),
            "maps": created,
            "roi_names": options.get("roi_names"),
            "result_experiment": session.selected_experiment,
        }
    )
    if mask is not None:
        mask_key = "processing_mask_" + uuid4().hex[:12]
        snapshot = np.asarray(mask, dtype=np.uint8).copy()
        history[-1]["mask_array"] = mask_key
        session.current_experiment.setdefault("processed", {})[mask_key] = snapshot
        dataset = session.build_dataset()
        dataset.processed = {**(dataset.processed or {}), mask_key: snapshot}
    metadata = session.current_experiment.get("extra_metadata") or {}
    metadata["processing_history"] = history
    session.current_experiment["extra_metadata"] = metadata
    session.import_metadata = session.import_metadata or {}
    session.import_metadata.setdefault("experiments", {}).setdefault(
        session.selected_experiment, {}
    )["processing_history"] = history
    if session.selected_experiment in workspace.fit_results_by_experiment:
        workspace.fit_results_by_experiment[session.selected_experiment]["fields"] = maps
    session._save_current_view_state()
    progress(1, 1, "Diffusion maps available in View and Export")
    return {"experiment": session.selected_experiment, "maps": created, "method": "dti"}


def apply_processing(workspace, options, progress):
    session = workspace.session
    options = dict(options)
    if not session.is_loaded:
        raise ValueError("Load an experiment before processing")
    method = next(
        (row for row in processing_methods(session) if row["id"] == options.get("method")), None
    )
    if method is None or method["unavailable"]:
        raise ValueError(method["unavailable"] if method else "Choose a processing method")
    parameters = _settings(method, options.get("parameters", {}))
    if method["id"] == "dti":
        return _dti(workspace, options, progress)
    component = options.get("component", "auto")
    if component not in {"auto", "magnitude", "phase", "real", "imag"}:
        raise ValueError("Choose a supported input component")
    source_name = session.selected_experiment
    context = _source(workspace, str(options.get("input", "view")), component)
    data, dims = context[:2]
    roi = (
        fitting_roi(workspace, options["roi_names"])
        if options.get("roi_names") is not None
        else None
    )
    if (
        roi is not None
        and options.get("input", "view") == "view"
        and not session.visual_state.show_original
        and session.visual_state.transform_key
    ):
        raise ValueError("Limit transformed processing with transferred ROIs on the target grid")
    key = method["id"]
    if key in FILTERS:
        axes = options.get("axes", session.spatial_dim_names)
        if (
            not isinstance(axes, list)
            or not 1 <= len(axes) <= 3
            or len(set(axes)) != len(axes)
            or any(name not in dims for name in axes)
        ):
            raise ValueError("Choose one to three filter axes from the input dimensions")
        if key in {"bm3d", "bm4d"} and len(axes) != (2 if key == "bm3d" else 3):
            raise ValueError(
                key.upper()
                + " requires exactly "
                + ("two" if key == "bm3d" else "three")
                + " filter axes"
            )
        options["axes"] = list(axes)
        if not np.all(np.isfinite(data)) and not (
            key == "gaussian" and parameters.get("preserve_nan")
        ):
            raise ValueError(
                "This method needs finite input; use NaN-aware Gaussian for missing samples"
            )
        result = _filter(data, [list(dims).index(name) for name in axes], key, parameters, progress)
    else:
        if np.iscomplexobj(data):
            raise ValueError(
                "Choose magnitude, real or imaginary input explicitly for time-series processing"
            )
        axis_name = options.get("axis") or next(
            (name for name in dims if name not in session.spatial_dim_names and dims[name] > 1),
            None,
        )
        options["axis"] = axis_name
        if key in {"baseline", "temporal_gaussian", "despike"} and (
            axis_name not in dims or axis_name in session.spatial_dim_names or dims[axis_name] < 2
        ):
            raise ValueError("Choose a non-spatial acquisition axis with at least two samples")
        progress(0, 1, method["label"])
        if key == "baseline":
            if parameters["baseline_frames"] > dims[axis_name]:
                raise ValueError("Baseline samples exceed the selected acquisition axis")
            result = apply_baseline(data, axis=list(dims).index(axis_name), **parameters)
        elif key == "temporal_gaussian":
            sigmas = tuple(parameters["sigma"] if name == axis_name else 0 for name in dims)
            result = nan_gaussian_filter(data, sigmas)
        elif key == "despike":
            axis = list(dims).index(axis_name)
            if dims[axis_name] < 3:
                raise ValueError("Spike correction needs at least three samples")
            result = np.moveaxis(
                replace_isolated_spikes(np.moveaxis(data, axis, 0), **parameters), 0, axis
            )
        else:
            frames = reshape_dims_to_frames(data, dims)
            if key == "normalize":
                options.setdefault("reference_frame", 0)
            reference = _reference(
                workspace, options, dims, context[4], context[5], series=key == "dilution"
            )
            if key == "normalize":
                result = compute_z_spectra(frames, reference, **parameters)
            elif key == "dilution":
                result = apply_dilution_correction(frames, reference, **parameters)
            else:
                raise ValueError("Unsupported processing method")
            result = reshape_frames_to_dims(np.asarray(result), dims)
        progress(1, 1, "Processing complete")
    result = np.asarray(result)
    if roi is not None:
        mask = broadcast_mask(roi.mask, data.shape, mask_dims=roi.dims, target_dims=tuple(dims))
        changes_unit = key == "normalize" or (key == "baseline" and parameters["mode"] == "divide")
        result = np.where(mask, result, np.nan if changes_unit else data)
    return _create_result(
        workspace,
        result,
        context,
        options,
        method,
        parameters,
        source_name,
    )


def rename_processing_sources(workspace, old, new):
    for pipeline in workspace.pipelines.values():
        for step in pipeline["steps"]:
            for key in ("reference_experiment", "moving_experiment"):
                if step["options"].get(key) == old:
                    step["options"][key] = new
            if old in step.get("references", {}):
                step["references"][new] = step["references"].pop(old)
    for name, entry in (workspace.session.data or {}).get("experiments", {}).items():
        metadata = entry.get("extra_metadata") or {}
        for step in metadata.get("processing_history", []):
            for key in ("source_experiment", "reference_experiment", "result_experiment"):
                if step.get(key) == old:
                    step[key] = new
        if workspace.session.import_metadata is not None and "processing_history" in metadata:
            workspace.session.import_metadata.setdefault("experiments", {})[name] = metadata


def run_processing(workspace, options, progress):
    from .pipeline import apply_step

    return apply_step(workspace, options, progress)
