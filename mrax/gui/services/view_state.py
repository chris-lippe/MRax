"""View-state adaptation for quantitative fitting."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import jax.numpy as jnp
import numpy as np

from mrax.analysis import ROI
from mrax.data_struct import DataSet, Experiment
from mrax.data_struct.dimensions import reshape_dims_to_frames, reshape_frames_to_dims

from ..registration import apply_transform_to_image, inverse_transform, resample_mask_with_transform
from ..services.viewing import _first_spatial_image as _first_spatial_image
from ..session import VisualViewState


class ViewStateFitMixin:
    def _visual_state_fit_kwargs(
        self, *, include_roi: bool = True, roi_override: ROI | None = None
    ) -> dict[str, Any]:
        if (
            hasattr(self, "workspace")
            and self.session.selected_experiment in self.workspace.pipelines
            and not getattr(self.workspace, "_pipeline_evaluating", False)
        ):
            from .pipeline import project_roi

            result = {}
            if include_roi:
                roi = roi_override if roi_override is not None else self.session.roi()
                if roi is not None:
                    result["input_roi"] = project_roi(self.workspace, roi)
            return result
        if not self.session.fit_settings.use_view_state_data:
            return {}
        state = self.session.visual_state
        if state.show_original:
            if not self.session.image_data_is_original():
                experiment, _model, dataset, _study = self.session.build_fit_inputs()
                return {
                    "input_experiment": experiment,
                    "input_dataset": self._view_state_dataset(
                        dataset,
                        np.asarray(self.session.original_image_array),
                        original_applied=True,
                    ),
                }
            return {}

        kwargs: dict[str, Any] = {}
        experiment: Experiment | None = None
        dataset: DataSet | None = None
        transformed = False
        transform_context: tuple[np.ndarray, Any] | None = None

        if state.drift_key or state.transform_key:
            experiment, _model, dataset, _study = self.session.build_fit_inputs()
            data = np.asarray(dataset.data)
            if state.drift_key:
                data = self._visual_state_drift_corrected_data(state)
            if state.transform_key:
                data, experiment, transform_context = self._visual_state_transformed_fit_data(
                    data,
                    experiment,
                    state,
                )
            dataset = self._view_state_dataset(
                dataset,
                data,
                original_applied=False,
                drift_applied=bool(state.drift_key),
                transform_applied=bool(state.transform_key),
            )
            kwargs["input_experiment"] = experiment
            kwargs["input_dataset"] = dataset
            transformed = True

        if include_roi:
            source_roi: ROI | None = roi_override
            if source_roi is None and state.mask_enabled and state.mask_roi:
                source_roi = self.session.roi(str(state.mask_roi))
                if source_roi is None:
                    raise RuntimeError(f"ROI '{state.mask_roi}' is not enabled for this experiment")
            elif source_roi is None and transformed:
                source_roi = self.session.roi()
            if source_roi is not None and (
                roi_override is not None or state.mask_enabled or transformed
            ):
                kwargs["input_roi"] = self._visual_state_fit_roi(
                    source_roi,
                    transform_context=transform_context,
                    target_spatial_dims=tuple(experiment.spatial_dim_names)
                    if experiment is not None
                    else None,
                )
        return kwargs

    def _visual_state_drift_corrected_data(self, state: VisualViewState) -> np.ndarray:
        result = self.drift_results.get(str(state.drift_key or ""))
        if result is None:
            raise RuntimeError("selected drift correction is not available")
        metadata = dict(result.get("metadata") or {})
        selected = self.session.selected_experiment
        experiment = metadata.get("experiment")
        if selected is not None and experiment is not None and str(selected) != str(experiment):
            raise RuntimeError(
                f"drift correction source is {experiment}. Current experiment is {selected}"
            )
        corrected = np.asarray(result.get("corrected"))
        if tuple(corrected.shape) != tuple(np.asarray(self.session.image_array).shape):
            raise RuntimeError(
                f"drift-corrected data shape {corrected.shape} does not match current image shape {self.session.image_array.shape}"
            )
        return corrected

    def _visual_state_transformed_fit_data(
        self,
        data: np.ndarray,
        experiment: Experiment,
        state: VisualViewState,
    ) -> tuple[np.ndarray, Experiment, tuple[np.ndarray, Any]]:
        reference_image, transform = self._visual_state_fit_transform_context(state)
        spatial_dims = tuple(experiment.spatial_dim_names)
        if len(spatial_dims) != 2:
            raise RuntimeError(
                "View-state fitting with transforms currently supports 2-D spatial data"
            )
        frames = np.asarray(reshape_dims_to_frames(data, experiment.dims))
        if frames.ndim != 3:
            raise RuntimeError("View-state fitting could not reshape data into 2-D frames")
        target = _first_spatial_image(reference_image)
        transformed_frames = np.stack(
            [
                apply_transform_to_image(
                    _first_spatial_image(frame), target, transform, interpolator="linear"
                )
                for frame in frames
            ],
            axis=0,
        )
        updated_dims = self._experiment_dims_with_spatial_shape(
            experiment, transformed_frames.shape[1:]
        )
        transformed_data = reshape_frames_to_dims(transformed_frames, updated_dims)
        return (
            transformed_data,
            self._copy_experiment_with_dims(experiment, updated_dims),
            (target, transform),
        )

    def _visual_state_fit_transform_context(self, state: VisualViewState) -> tuple[np.ndarray, Any]:
        result = self.registration_results.get(str(state.transform_key or ""))
        if result is None:
            raise RuntimeError("selected registration transform is not available")
        metadata = dict(result.get("metadata") or {})
        selected = self.session.selected_experiment
        direction = str(state.transform_direction or "moving_to_reference")
        if direction == "reference_to_moving":
            source = metadata.get("reference_experiment")
            reference = np.asarray(result["moving"])
            transform = inverse_transform(result["transform"])
        else:
            source = metadata.get("moving_experiment")
            reference = np.asarray(result["fixed"])
            transform = result["transform"]
        if selected is not None and source is not None and str(selected) != str(source):
            raise RuntimeError(f"transform source is {source}. Current experiment is {selected}")
        return reference, transform

    def _visual_state_fit_roi(
        self,
        roi: ROI,
        *,
        transform_context: tuple[np.ndarray, Any] | None,
        target_spatial_dims: tuple[str, ...] | None,
    ) -> ROI:
        mask = np.asarray(roi.mask) != 0
        dims = tuple(roi.dims or tuple(self.session.spatial_dim_names[: mask.ndim]))
        metadata = {
            **dict(roi.metadata),
            "generated_by": "mrax.gui.view_state_fit",
            "source_roi": roi.name,
            "binary_view_state_mask": True,
        }
        if transform_context is not None:
            reference, transform = transform_context
            if mask.ndim != 2:
                raise RuntimeError("View-state fitting can transform only 2-D ROI masks")
            mask = resample_mask_with_transform(mask, reference, transform, output_dtype=bool)
            dims = tuple(target_spatial_dims or self.session.spatial_dim_names[: mask.ndim])
            metadata["registration_transform"] = str(self.session.visual_state.transform_key or "")
            metadata["registration_direction"] = str(
                self.session.visual_state.transform_direction or ""
            )
        return ROI(
            f"{roi.name}_view_state",
            np.asarray(mask, dtype=bool),
            dims=dims,
            metadata=metadata,
        )

    def _view_state_dataset(
        self,
        dataset: DataSet,
        data: np.ndarray,
        *,
        original_applied: bool = False,
        drift_applied: bool = False,
        transform_applied: bool = False,
    ) -> DataSet:
        return DataSet(
            data=jnp.asarray(data),
            name=f"{dataset.name}_view_state",
            noise_level=dataset.noise_level,
            processed=dict(dataset.processed or {}),
            kspace=None,
            trajectory=dataset.trajectory,
            density=dataset.density,
            sampling_mask=dataset.sampling_mask,
            kspace_metadata={
                **dict(dataset.kspace_metadata or {}),
                "view_state_original_applied": bool(original_applied),
                "view_state_drift_applied": bool(drift_applied),
                "view_state_transform_applied": bool(transform_applied),
            },
        )

    def _experiment_dims_with_spatial_shape(
        self,
        experiment: Experiment,
        spatial_shape: Sequence[int],
    ) -> dict[str, int]:
        spatial_dims = tuple(experiment.spatial_dim_names)
        shape_by_dim = {name: int(size) for name, size in zip(spatial_dims, spatial_shape)}
        return {
            str(name): int(shape_by_dim.get(str(name), int(size)))
            for name, size in experiment.dims.items()
        }

    def _copy_experiment_with_dims(
        self, experiment: Experiment, dims: Mapping[str, int]
    ) -> Experiment:
        updated = Experiment(
            dims={str(name): int(size) for name, size in dims.items()},
            observables=dict(experiment.observables),
            exp_type=experiment.exp_type,
        )
        for attr in (
            "study_globals",
            "_mrax_dim_observables",
            "_mrax_default_identity_fit_warning",
            "_mrax_manual_type_override_warning",
        ):
            if hasattr(experiment, attr):
                setattr(updated, attr, getattr(experiment, attr))
        return updated


class WorkspaceFitInputs(ViewStateFitMixin):
    def __init__(self, workspace):
        self.workspace = workspace
        self.session = workspace.session
        self.registration_results = workspace.registration_results
        self.drift_results = workspace.drift_results
