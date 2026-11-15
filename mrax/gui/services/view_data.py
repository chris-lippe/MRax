"""Quantitative browser views and profiles on MRax's declared spatial axes."""

from __future__ import annotations

import dataclasses

import numpy as np

from mrax.data_struct.dimensions import reshape_dims_to_frames
from mrax.gui.masks import mask_slice_for_display
from mrax.gui.plotting import (
    axis_coordinates,
    image_profile,
    profile_dimension,
    profile_selection,
    scalar_component,
)
from mrax.gui.registration import (
    apply_transform_to_image,
    inverse_transform,
    registration_overlay,
    resample_mask_with_transform,
)
from mrax.gui.session import _display_slice

from .viewing import _side_by_side_images

VIEWS = (
    "Image",
    "MRax reconstruction",
    "Vendor vs MRax",
    "Vendor MRax Diff",
    "K-space",
    "Map",
    "Registration overlay",
)


def json_value(value):
    """JSON values preserve nonfinite samples as null, never fabricated zeroes."""
    if hasattr(value, "values") and hasattr(value, "si_unit"):
        return {"values": json_value(value.values), "unit": value.si_unit}
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if hasattr(value, "__array__"):
        return json_value(np.asarray(value))
    if dataclasses.is_dataclass(value):
        return json_value(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, complex):
        return {"real": json_value(value.real), "imag": json_value(value.imag)}
    if value is None or isinstance(value, str):
        return value
    return str(value)


def display_options(workspace):
    return workspace.display_settings.setdefault(workspace.session.selected_experiment or "", {})


def registration_plane(array, session, *, native_2d=False):
    array = np.asarray(array)
    if array.ndim < 2 or (array.ndim == 2 and not native_2d):
        return array
    if array.ndim == 2:
        return mask_slice_for_display(
            array, dims=session.spatial_dim_names, plane_dims=session.image_plane_dims
        )
    names = tuple(session.spatial_dim_names[: array.ndim])
    dims = dict(zip(names, array.shape))
    return _display_slice(
        array,
        dims=dims,
        indices=session.active_indices,
        kind="image",
        plane_dims=session.image_plane_dims,
    )


def transformed_context(workspace):
    if workspace.session.selected_experiment in workspace.pipelines:
        return None
    state = workspace.session.visual_state
    result = workspace.registration_results.get(state.transform_key or "")
    if result is None:
        return None
    inverse = state.transform_direction == "reference_to_moving"
    metadata = result.get("metadata") or {}
    source = metadata.get("reference_experiment" if inverse else "moving_experiment")
    if source and source != workspace.session.selected_experiment:
        raise ValueError("This transform belongs to another source experiment")
    if (
        result["transform"].GetDimension() == 2
        and metadata.get("plane", workspace.session.active_image_plane)
        != workspace.session.active_image_plane
    ):
        raise ValueError("Use the plane on which this transform was estimated")
    return (
        result,
        (inverse_transform(result["transform"]) if inverse else result["transform"]),
        np.asarray(result["moving"] if inverse else result["fixed"]),
    )


def render_image(workspace):
    from .pipeline import display_session

    session = display_session(workspace)
    if not session.is_loaded:
        return np.zeros((0, 0), dtype=np.float32), False
    options = display_options(workspace)
    view = workspace.view_label
    state = session.visual_state
    native = view in {"Image", "Map"}
    if view == "Image":
        pipeline = workspace.pipelines.get(session.selected_experiment)
        array = session.frame_image(original=state.show_original and pipeline is None)
    elif view == "MRax reconstruction":
        array = session.reconstruction_image()
    elif view == "K-space":
        array = session.kspace_image()
    elif view == "Map":
        array = session.map_image()
    elif view in {"Vendor vs MRax", "Vendor MRax Diff"}:
        vendor = scalar_component(session.frame_image(), "magnitude")
        reconstruction = session.reconstruction_image(component="magnitude", combined=True)
        if vendor.shape != reconstruction.shape:
            raise ValueError("The comparison images are on different grids")
        array = (
            _side_by_side_images(vendor, reconstruction)
            if view == "Vendor vs MRax"
            else vendor - reconstruction
        )
    elif view == "Registration overlay":
        key = options.get("registration_key") or state.transform_key
        result = workspace.registration_results.get(key or "")
        if result is None:
            raise ValueError("Choose a stored registration")
        array = registration_overlay(
            registration_plane(result["fixed"], session),
            registration_plane(result["corrected"], session),
            fixed_channel=state.reference_channel,
            moving_channel=state.moving_channel,
            fixed_alpha=state.reference_alpha,
            moving_alpha=state.moving_alpha,
        )
    else:
        raise ValueError("Unknown image view")
    array = np.asarray(array)
    if view in {"Image", "Map"}:
        array = scalar_component(array, options.get("component", "auto"))
    if (
        not state.show_original
        and view in {"Image", "Map"}
        and session.selected_experiment not in workspace.pipelines
    ):
        if state.drift_key and view == "Image":
            result = workspace.drift_results.get(state.drift_key)
            if result is None:
                raise ValueError("The selected drift result is unavailable")
            metadata = result.get("metadata") or {}
            if metadata.get("experiment") != session.selected_experiment:
                raise ValueError("Drift results belong to another experiment")
            array = scalar_component(
                _display_slice(
                    np.asarray(result["corrected"]),
                    dims=session.experiment_dims,
                    indices=session.active_indices,
                    kind="image",
                    plane_dims=session.image_plane_dims,
                ),
                options.get("component", "auto"),
            )
        context = transformed_context(workspace)
        if context:
            _result, transform, reference = context
            if transform.GetDimension() == 3:
                source_data = (
                    session.map_array()
                    if view == "Map"
                    else np.asarray(workspace.drift_results[state.drift_key]["corrected"])
                    if state.drift_key
                    else session.image_array
                )
                volume = (
                    source_data
                    if view == "Map"
                    else np.asarray(reshape_dims_to_frames(source_data, session.experiment_dims))[
                        session.active_frame
                    ]
                )
                corrected = apply_transform_to_image(
                    scalar_component(volume, options.get("component", "auto")), reference, transform
                )
                array = registration_plane(corrected, session)
            else:
                array = apply_transform_to_image(
                    array, registration_plane(reference, session), transform
                )
            native = False
        if state.mask_enabled and state.mask_roi:
            roi = session.roi(state.mask_roi)
            if roi is None:
                raise ValueError("The display mask is not enabled")
            mask = mask_slice_for_display(
                roi.mask,
                dims=roi.dims,
                indices=session.active_indices,
                plane_dims=session.image_plane_dims,
            )
            if context:
                if context[1].GetDimension() == 3:
                    mask = registration_plane(
                        resample_mask_with_transform(
                            roi.mask, context[2], context[1], output_dtype=roi.mask.dtype
                        ),
                        session,
                    )
                else:
                    mask = resample_mask_with_transform(
                        mask, array, context[1], output_dtype=roi.mask.dtype
                    )
            if mask.shape != array.shape:
                raise ValueError("The display mask has a different grid")
            array = np.where(mask != 0, array, np.nan)
            native = False
    if array.ndim not in {2, 3}:
        raise ValueError("The selected view is not a displayable plane")
    return array, native and len(
        session.spatial_shape
    ) >= 2 and array.shape == session.frame_image().shape


def displayed_roi(workspace, roi, image, editable):
    from .pipeline import display_session, display_steps, project_roi

    roi = project_roi(workspace, roi, steps=display_steps(workspace))
    session = display_session(workspace)
    if workspace.view_label not in {"Image", "Map"}:
        return None
    mask = mask_slice_for_display(
        roi.mask, dims=roi.dims, indices=session.active_indices, plane_dims=session.image_plane_dims
    )
    context = transformed_context(workspace) if not session.visual_state.show_original else None
    if context and context[1].GetDimension() == 2:
        mask = resample_mask_with_transform(mask, image, context[1], output_dtype=roi.mask.dtype)
    elif context:
        mask = registration_plane(
            resample_mask_with_transform(
                roi.mask, context[2], context[1], output_dtype=roi.mask.dtype
            ),
            session,
        )
    return mask if mask.shape == image.shape[:2] else None


def curve_for_workspace(workspace):
    session = workspace.session
    selected = (
        session.roi(session.fit_settings.selected_roi)
        if session.rois and not display_options(workspace).get("plot_whole_image", False)
        else None
    )
    if not session.experiment_dims:
        return None
    if selected is not None and session.selected_experiment in workspace.pipelines:
        from .pipeline import project_roi

        selected = project_roi(workspace, selected)
    experiment = session.build_experiment()
    fitted_inputs = None
    if session.fit_settings.use_view_state_data:
        from .view_state import WorkspaceFitInputs

        fitted_inputs = WorkspaceFitInputs(workspace)._visual_state_fit_kwargs(include_roi=selected is not None)
        experiment = fitted_inputs.get("input_experiment", experiment)
        selected = fitted_inputs.get("input_roi", selected)
    dimension = profile_dimension(experiment, workspace.plot_x_axis_key, non_imaging=session.is_non_imaging)
    label, axis = axis_coordinates(
        experiment,
        dimension,
        session.active_indices,
        geometry=session.current_experiment.get("spatial_geometry"),
        spectral=session.is_non_imaging,
    )
    data = (
        session.original_image_array if session.visual_state.show_original else session.image_array
    )
    if fitted_inputs and fitted_inputs.get("input_dataset") is not None:
        data = np.asarray(fitted_inputs["input_dataset"].data)
    values = image_profile(
        data,
        experiment,
        dimension,
        session.active_indices,
        roi=selected,
        plane_dims=session.image_plane_dims,
        component=display_options(workspace).get("component", "auto"),
    )
    _selection, frame_ids = profile_selection(experiment, dimension, session.active_indices)
    included = (
        session.frame_mask.as_array()[frame_ids]
        if session.frame_mask is not None
        else np.ones(len(values), dtype=bool)
    )
    if len(axis) != len(values):
        raise ValueError("Curve coordinates do not match its selected dimension")
    elapsed_time = label == "Elapsed time (s)"
    observable = next(
        (
            (key, value)
            for key, value in experiment.observables.items()
            if key == dimension or (getattr(value, "dims", None) and dimension in value.dims)
        ),
        None,
    )
    if observable is None and session.is_non_imaging:
        observable = next(((key, value) for key, value in experiment.observables.items()
                           if label in {key, f"{key} ({value.si_unit})"}), None)
    if observable is None and not elapsed_time and not session.is_non_imaging and dimension not in session.spatial_dim_names:
        from mrax.gui.session import frame_axis_values

        frame_label, frame_axis = frame_axis_values(experiment)
        if len(np.asarray(frame_axis)) == len(axis):
            label, axis = frame_label, frame_axis
            observable = (frame_label, experiment.observables.get(frame_label))
    return {
        "x": json_value(np.asarray(axis)),
        "observed": json_value(values),
        "x_name": "Elapsed time" if elapsed_time else label,
        "x_unit": "s" if elapsed_time else getattr(observable[1], "si_unit", None) if observable else None,
        "included": included.tolist(),
        "dimension": dimension,
        "roi_name": selected.name if selected is not None else None,
    }
