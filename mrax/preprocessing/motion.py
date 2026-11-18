from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import jax.numpy as jnp
import numpy as np

from mrax.data_struct.dataset import DataSet
from mrax.data_struct.dimensions import reshape_dims_to_frames, reshape_frames_to_dims
from mrax.data_struct.experiment import Experiment

ReferenceSelector = str | int | np.ndarray
RegistrationMethodFactory = Callable[[Any, Any], Any]
TransformFactory = Callable[[Any, Any], Any]


@dataclass(frozen=True)
class RegistrationDiagnostics:
    """Optimizer diagnostics for one moving image registration."""

    metric_value: float | None
    optimizer_stop_condition: str
    iteration_metrics: tuple[float, ...] = ()
    transform_parameters: tuple[float, ...] = ()
    alignment_check: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ImageRegistrationResult:
    """A registered moving image plus the estimated SimpleITK transform."""

    corrected: np.ndarray
    fixed: np.ndarray
    transform: Any
    diagnostics: RegistrationDiagnostics


@dataclass(frozen=True)
class FrameStackMotionResult:
    """Per-frame registration output for one frame stack."""

    corrected: np.ndarray
    reference_image: np.ndarray
    reference_index: int | None
    transforms: tuple[Any | None, ...]
    diagnostics: tuple[RegistrationDiagnostics, ...]


@dataclass(frozen=True)
class MotionCorrectionResult:
    """Motion-corrected MRax dataset and grouped registration diagnostics."""

    dataset: DataSet
    dimensions: tuple[str, ...]
    group_dimensions: tuple[str, ...]
    group_indices: tuple[tuple[int, ...], ...]
    groups: tuple[FrameStackMotionResult, ...]
    estimation_dimensions: tuple[str, ...] = ()

    @property
    def corrected(self):
        """Corrected dataset values in the source MRax dimension layout."""

        return self.dataset.data

    @property
    def diagnostics(self) -> tuple[RegistrationDiagnostics, ...]:
        """Flattened diagnostics in group-major, frame-minor order."""

        return tuple(diag for group in self.groups for diag in group.diagnostics)


@dataclass(frozen=True)
class ExperimentRegistrationResult:
    """A moving experiment registered onto a reference experiment image grid."""

    experiment: Experiment
    dataset: DataSet
    registration: ImageRegistrationResult


def _require_simpleitk():
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise ImportError(
            "SimpleITK is required for registration. Install `pip install 'mrax[full]'`."
        ) from exc
    return sitk


def _check_spatial_array(array: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(array)
    if arr.ndim not in (2, 3):
        raise ValueError(f"{name} must be a 2D or 3D spatial image, got shape {arr.shape}")
    return arr


def _registration_array(array: np.ndarray, *, name: str, normalize: bool = False) -> np.ndarray:
    arr = _check_spatial_array(array, name=name)
    if np.iscomplexobj(arr):
        arr = np.abs(arr)
    arr = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.all():
        fill = float(np.nanmedian(arr[finite])) if finite.any() else 0.0
        arr = np.where(finite, arr, fill)
    if normalize:
        lo, hi = np.percentile(arr, (1, 99))
        if hi <= lo:
            lo, hi = float(arr.min()), float(arr.max())
        if hi <= lo:
            raise ValueError(f"{name} image has no intensity variation. Choose a different registration frame")
        arr = np.clip((arr - lo) / (hi - lo), 0, 1).astype(np.float32)
    return arr


def _sitk_reference_image(array: np.ndarray):
    sitk = _require_simpleitk()
    return sitk.GetImageFromArray(_registration_array(array, name="reference"))


def _interpolator(sitk, interpolator: str | Any):
    if not isinstance(interpolator, str):
        return interpolator
    key = interpolator.lower()
    if key in ("linear", "bilinear", "trilinear"):
        return sitk.sitkLinear
    if key in ("nearest", "nearest_neighbor", "nn"):
        return sitk.sitkNearestNeighbor
    if key in ("bspline", "b_spline"):
        return sitk.sitkBSpline
    raise ValueError("interpolator must be 'linear', 'nearest', or 'bspline'")


def _resample_array_to_reference(
    array: np.ndarray,
    *,
    reference_image,
    transform,
    interpolator: str | Any,
    default_value: float,
    output_pixel_type: Any | None = None,
) -> np.ndarray:
    sitk = _require_simpleitk()
    arr = _check_spatial_array(array, name="moving")
    interp = _interpolator(sitk, interpolator)
    pixel_type = sitk.sitkFloat32 if output_pixel_type is None else output_pixel_type

    def _resample(values: np.ndarray) -> np.ndarray:
        image = sitk.GetImageFromArray(np.asarray(values, dtype=np.float32))
        resampled = sitk.Resample(
            image,
            reference_image,
            transform,
            interp,
            float(default_value),
            pixel_type,
        )
        return np.asarray(sitk.GetArrayFromImage(resampled))

    if np.iscomplexobj(arr):
        return _resample(np.real(arr)) + 1j * _resample(np.imag(arr))
    return _resample(arr)


def resample_image(
    moving: np.ndarray,
    reference: np.ndarray,
    transform,
    *,
    interpolator: str | Any = "linear",
    default_value: float = 0.0,
    output_pixel_type: Any | None = None,
) -> np.ndarray:
    """Resample one image onto a reference image grid with a SimpleITK transform."""

    reference_source = _check_spatial_array(reference, name="reference")
    moving_source = _check_spatial_array(moving, name="moving")
    if reference_source.ndim != moving_source.ndim:
        raise ValueError("moving and reference images must have the same spatial dimensionality")
    return _resample_array_to_reference(
        moving_source,
        reference_image=_sitk_reference_image(reference_source),
        transform=transform,
        interpolator=interpolator,
        default_value=default_value,
        output_pixel_type=output_pixel_type,
    )


def _transform_parameters(transform) -> tuple[float, ...]:
    if transform is None or not hasattr(transform, "GetParameters"):
        return ()
    return tuple(float(value) for value in transform.GetParameters())


def _identity_diagnostics(label: str) -> RegistrationDiagnostics:
    return RegistrationDiagnostics(
        metric_value=None,
        optimizer_stop_condition=label,
    )


def _default_registration_method(sitk, fixed_image, moving_image, dimension: int):
    voxel_size = min(fixed_image.GetSpacing())
    initial_transform = sitk.Euler2DTransform() if dimension == 2 else sitk.Euler3DTransform()
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_image,
        moving_image,
        initial_transform,
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    method = sitk.ImageRegistrationMethod()
    method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    method.SetMetricSamplingStrategy(method.NONE)
    method.SetInterpolator(sitk.sitkLinear)
    method.SetOptimizerAsRegularStepGradientDescent(
        learningRate=voxel_size,
        minStep=1e-4 * voxel_size,
        numberOfIterations=200,
        gradientMagnitudeTolerance=1e-6,
    )
    method.SetOptimizerScalesFromPhysicalShift()
    method.SetShrinkFactorsPerLevel((2, 1))
    method.SetSmoothingSigmasPerLevel((voxel_size, 0.0))
    method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    method.SetInitialTransform(initial_transform, inPlace=False)
    return method


def _registration_method(
    registration_method: Any | RegistrationMethodFactory | None,
    *,
    sitk,
    fixed_image,
    moving_image,
    dimension: int,
):
    if registration_method is None:
        return _default_registration_method(sitk, fixed_image, moving_image, dimension)
    return (
        registration_method(fixed_image, moving_image)
        if callable(registration_method)
        else registration_method
    )


def _set_initial_transform(
    method,
    initial_transform: Any | TransformFactory | None,
    *,
    fixed_image,
    moving_image,
    in_place: bool,
) -> None:
    if initial_transform is None:
        return
    transform = (
        initial_transform(fixed_image, moving_image)
        if callable(initial_transform)
        else initial_transform
    )
    method.SetInitialTransform(transform, inPlace=bool(in_place))


def _method_metric_value(method) -> float | None:
    try:
        return float(method.GetMetricValue())
    except (AttributeError, RuntimeError):
        return None


def _method_stop_condition(method) -> str:
    try:
        return str(method.GetOptimizerStopConditionDescription())
    except (AttributeError, RuntimeError):
        return ""


def _alignment_scores(fixed, moving, mask):
    valid = np.isfinite(fixed) & np.isfinite(moving)
    valid &= np.asarray(mask, dtype=bool) if mask is not None else fixed > 0.05
    count = int(valid.sum())
    if count < 32:
        return None
    bins = min(32, max(8, int(np.sqrt(count) / 2)))
    joint = np.histogram2d(fixed[valid], moving[valid], bins=bins, range=((0, 1), (0, 1)))[0]
    joint /= max(joint.sum(), 1)
    x, y = joint.sum(axis=1), joint.sum(axis=0)
    product = x[:, None] * y[None, :]
    occupied = (joint > 0) & (product > 0)
    information = float(np.sum(joint[occupied] * np.log(joint[occupied] / product[occupied])))
    entropy_x = -float(np.sum(x[x > 0] * np.log(x[x > 0])))
    entropy_y = -float(np.sum(y[y > 0] * np.log(y[y > 0])))
    return information/max(np.sqrt(entropy_x*entropy_y),1e-12), float(np.mean(moving[valid]>.05))


def register_image(
    fixed: np.ndarray,
    moving: np.ndarray,
    *,
    registration_method: Any | RegistrationMethodFactory | None = None,
    initial_transform: Any | TransformFactory | None = None,
    transform_in_place: bool = False,
    interpolator: str | Any = "linear",
    default_value: float = 0.0,
    output_pixel_type: Any | None = None,
    normalize_intensity: bool = False,
    fixed_mask: np.ndarray | None = None,
    moving_mask: np.ndarray | None = None,
    moving_pre_transform: Any | None = None,
    fixed_spacing: Iterable[float] | None = None,
    validate_alignment: bool = False,
    max_displacement_voxels: float | None = None,
) -> ImageRegistrationResult:
    """Register one moving image to one fixed image with SimpleITK.

    The default is rigid Mattes mutual-information registration. Pass a configured ``SimpleITK.ImageRegistrationMethod`` or a
    factory accepting ``(fixed_image, moving_image)`` to use any SimpleITK
    registration configuration. ``initial_transform`` can likewise be a
    transform or a factory when it should be set by the wrapper.
    ``moving_pre_transform`` maps fixed voxel indices into the moving grid.
    ``fixed_spacing`` gives physical voxel sizes in SimpleITK column-first order.
    Optimization uses physical distances. The returned transform uses voxel
    indices, and the final output is sampled once from the original moving data.
    ``validate_alignment`` rejects large losses of normalized mutual information
    or signal support in the reference region. This is a heuristic sanity check.
    """

    sitk = _require_simpleitk()
    fixed_source = _check_spatial_array(fixed, name="fixed")
    moving_source = _check_spatial_array(moving, name="moving")
    if max_displacement_voxels is not None and (
        not np.isfinite(max_displacement_voxels) or max_displacement_voxels <= 0
    ):
        raise ValueError("The motion limit must be positive and finite")
    if fixed_source.ndim != moving_source.ndim:
        raise ValueError("fixed and moving images must have the same spatial dimensionality")

    fixed_image = sitk.GetImageFromArray(
        _registration_array(fixed_source, name="fixed", normalize=normalize_intensity)
    )
    moving_values = _registration_array(moving_source, name="moving", normalize=normalize_intensity)
    moving_estimation_source = moving_values
    if moving_pre_transform is not None:
        moving_values = _resample_array_to_reference(
            moving_values,
            reference_image=fixed_image,
            transform=moving_pre_transform,
            interpolator="linear",
            default_value=0,
        )
        if moving_mask is not None:
            if np.shape(moving_mask) != moving_source.shape:
                raise ValueError("The moving mask must match its source image grid")
            moving_mask = (
                _resample_array_to_reference(
                    np.asarray(moving_mask, dtype=np.float32),
                    reference_image=fixed_image,
                    transform=moving_pre_transform,
                    interpolator="nearest",
                    default_value=0,
                )
                > 0.5
            )
    moving_image = sitk.GetImageFromArray(moving_values)
    output_reference = sitk.Image(fixed_image)
    scale = None
    if fixed_spacing is not None:
        spacing = tuple(float(value) for value in fixed_spacing)
        if len(spacing) != fixed_source.ndim or not np.isfinite(spacing).all() or min(spacing) <= 0:
            raise ValueError(
                "Registration spacing needs one positive finite value per spatial axis"
            )
        fixed_image.SetSpacing(spacing)
        moving_image.SetSpacing(spacing)
        scale = sitk.ScaleTransform(fixed_source.ndim, spacing)
    method = _registration_method(
        registration_method,
        sitk=sitk,
        fixed_image=fixed_image,
        moving_image=moving_image,
        dimension=fixed_source.ndim,
    )
    _set_initial_transform(
        method,
        initial_transform,
        fixed_image=fixed_image,
        moving_image=moving_image,
        in_place=transform_in_place,
    )
    for mask, image, setter in (
        (fixed_mask, fixed_image, method.SetMetricFixedMask),
        (moving_mask, moving_image, method.SetMetricMovingMask),
    ):
        if mask is not None:
            values = (np.asarray(mask) != 0).astype(np.uint8)
            if values.shape != tuple(reversed(image.GetSize())) or not np.any(values):
                raise ValueError(
                    "Registration masks must match their image grid and contain selected pixels"
                )
            mask_image = sitk.GetImageFromArray(values)
            mask_image.CopyInformation(image)
            setter(mask_image)

    iteration_metrics: list[float] = []

    def _record_iteration_metric() -> None:
        metric_value = _method_metric_value(method)
        if metric_value is not None:
            iteration_metrics.append(metric_value)

    method.AddCommand(sitk.sitkIterationEvent, _record_iteration_metric)
    try:
        final_transform = method.Execute(fixed_image, moving_image)
    except RuntimeError as exc:
        if "All samples map outside" in str(exc) or "No valid points" in str(exc):
            raise ValueError(
                "Registration lost overlap. Try regular-step descent, a smaller step or a broader mask"
            ) from exc
        raise
    if not np.isfinite(final_transform.GetParameters()).all():
        raise ValueError("Registration produced non-finite transform parameters")
    estimated_parameters = _transform_parameters(final_transform)
    if scale is not None:
        final_transform = sitk.CompositeTransform([scale.GetInverse(), final_transform, scale])
    motion_check = {}
    if max_displacement_voxels is not None:
        if fixed_mask is not None and np.any(fixed_mask):
            locations = np.argwhere(fixed_mask)
            lower, upper = locations.min(axis=0), locations.max(axis=0)
        else:
            lower, upper = np.zeros(fixed_source.ndim), np.asarray(fixed_source.shape) - 1
        from itertools import product

        points = [
            tuple(float(value) for value in point[::-1])
            for point in product(*zip(lower, upper, strict=True))
        ]
        displacement = max(
            float(np.linalg.norm(np.asarray(final_transform.TransformPoint(point)) - point))
            for point in points
        )
        motion_check["maximum_displacement_voxels"] = displacement
        if displacement > max_displacement_voxels:
            raise ValueError(
                "Estimated motion exceeds the selected motion limit. Review the images and adjust the limit if needed"
            )
    if moving_pre_transform is not None:
        final_transform = sitk.CompositeTransform([moving_pre_transform, final_transform])
    coverage = _resample_array_to_reference(
        np.ones(moving_source.shape, dtype=np.float32),
        reference_image=output_reference,
        transform=final_transform,
        interpolator="nearest",
        default_value=0,
    )
    region = (
        np.ones(fixed_source.shape, dtype=bool)
        if fixed_mask is None
        else np.asarray(fixed_mask, dtype=bool)
    )
    if not np.any((coverage > 0.5) & region):
        raise ValueError(
            "Registration lost image overlap. Choose another reference or restrict the transform"
        )
    corrected = _resample_array_to_reference(
        moving_source,
        reference_image=output_reference,
        transform=final_transform,
        interpolator=interpolator,
        default_value=default_value,
        output_pixel_type=output_pixel_type,
    )
    if not np.any((coverage > 0.5) & region & np.isfinite(corrected)):
        raise ValueError("Registration has no finite signal in the reference region")
    alignment_check = dict(motion_check)
    if validate_alignment:
        fixed_normalized = _registration_array(fixed_source, name="fixed", normalize=True)
        moving_normalized = (
            moving_estimation_source
            if normalize_intensity
            else _registration_array(moving_source, name="moving", normalize=True)
        )
        initial = moving_pre_transform
        if initial is None:
            initial = sitk.TranslationTransform(
                fixed_source.ndim,
                ((np.asarray(moving_source.shape[::-1]) - fixed_source.shape[::-1]) / 2).tolist(),
            )
        before = _resample_array_to_reference(
            moving_normalized,
            reference_image=output_reference,
            transform=initial,
            interpolator="linear",
            default_value=0,
        )
        after = _resample_array_to_reference(
            moving_normalized,
            reference_image=output_reference,
            transform=final_transform,
            interpolator="linear",
            default_value=0,
        )
        a = _alignment_scores(fixed_normalized, before, fixed_mask)
        b = _alignment_scores(fixed_normalized, after, fixed_mask)
        if a is not None and b is not None:
            alignment_check.update(
                normalized_mi_before=a[0],
                normalized_mi_after=b[0],
                signal_fraction_before=a[1],
                signal_fraction_after=b[1],
            )
            lost_information = b[0] < a[0] - 0.04 and b[0] < 0.8 * a[0]
            lost_signal = a[1] > 0.4 and b[1] < 0.8 * a[1]
            if lost_information or lost_signal:
                raise ValueError(
                    "Registration failed its alignment check. Try mutual information or a structural reference image"
                )
    diagnostics = RegistrationDiagnostics(
        metric_value=_method_metric_value(method),
        optimizer_stop_condition=_method_stop_condition(method),
        iteration_metrics=tuple(iteration_metrics),
        transform_parameters=estimated_parameters,
        alignment_check=alignment_check,
    )
    return ImageRegistrationResult(
        corrected=corrected,
        fixed=np.asarray(fixed_source),
        transform=final_transform,
        diagnostics=diagnostics,
    )


def _select_reference_image(
    frames: np.ndarray,
    reference: ReferenceSelector,
    *,
    name: str,
) -> tuple[np.ndarray, int | None]:
    stack = np.asarray(frames)
    if stack.ndim not in (3, 4):
        raise ValueError(f"{name} must have shape (frame, x, y) or (frame, x, y, z)")
    if stack.shape[0] <= 0:
        raise ValueError(f"{name} must contain at least one frame")
    if isinstance(reference, (int, np.integer)):
        index = int(reference)
        if index < 0 or index >= int(stack.shape[0]):
            raise IndexError(f"{name} reference frame index {index} is out of range")
        return np.asarray(stack[index]), index
    if isinstance(reference, str):
        key = reference.lower()
        if key in ("first", "frame0"):
            return np.asarray(stack[0]), 0
        if key == "mean":
            return np.asarray(np.nanmean(stack, axis=0)), None
        if key == "median":
            return np.asarray(np.nanmedian(stack, axis=0)), None
        raise ValueError("reference must be a frame index, image array, 'first', 'mean', or 'median'")
    selected = _check_spatial_array(np.asarray(reference), name=f"{name} reference")
    if tuple(selected.shape) != tuple(stack.shape[1:]):
        raise ValueError(
            f"{name} reference image shape {selected.shape} does not match frame shape {stack.shape[1:]}"
        )
    return selected, None


def motion_correct_framestack(
    frames: np.ndarray,
    *,
    reference: ReferenceSelector = "first",
    progress_callback: Callable[[int, int], None] | None = None,
    **registration_kwargs,
) -> FrameStackMotionResult:
    """Register a ``(frame, *spatial)`` stack to one reference image."""

    stack = np.asarray(frames)
    reference_image, reference_index = _select_reference_image(
        stack,
        reference,
        name="frames",
    )
    corrected: list[np.ndarray] = []
    transforms: list[Any | None] = []
    diagnostics: list[RegistrationDiagnostics] = []
    for index, frame in enumerate(stack):
        if progress_callback is not None:
            progress_callback(index, len(stack))
        if reference_index == index:
            corrected.append(np.asarray(frame))
            transforms.append(None)
            diagnostics.append(_identity_diagnostics("reference frame"))
            continue
        result = register_image(reference_image, frame, **registration_kwargs)
        corrected.append(np.asarray(result.corrected))
        transforms.append(result.transform)
        diagnostics.append(result.diagnostics)
    if progress_callback is not None:
        progress_callback(len(stack), len(stack))
    return FrameStackMotionResult(
        corrected=np.stack(corrected, axis=0),
        reference_image=np.asarray(reference_image),
        reference_index=reference_index,
        transforms=tuple(transforms),
        diagnostics=tuple(diagnostics),
    )


def _clone_dataset(dataset: DataSet, data: np.ndarray) -> DataSet:
    return DataSet(
        data=jnp.asarray(data),
        name=dataset.name,
        noise_level=dataset.noise_level,
        processed=dict(dataset.processed or {}),
        kspace=dataset.kspace,
        trajectory=dataset.trajectory,
        density=dataset.density,
        sampling_mask=dataset.sampling_mask,
        kspace_metadata=dataset.kspace_metadata,
        spatial_geometry=dataset.spatial_geometry,
        acquisition=dataset.acquisition,
    )


def _normalize_registration_dimensions(
    dimensions: Iterable[str] | None,
    available: tuple[str, ...],
) -> tuple[str, ...]:
    if dimensions is None:
        return tuple(name for name in available if name != "slice_package")
    selected = tuple(str(name) for name in dimensions)
    if len(set(selected)) != len(selected):
        raise ValueError("dimensions must not contain duplicates")
    unknown = [name for name in selected if name not in available]
    if unknown:
        raise ValueError(
            f"Registration dimensions {unknown} are not non-spatial experiment dimensions {available}"
        )
    if "slice_package" in selected:
        raise ValueError("Slice packages have separate image grids. Select acquisition dimensions within each package")
    return tuple(name for name in available if name in selected)


def _dim_ordered_data(experiment: Experiment, dataset: DataSet) -> np.ndarray:
    data = np.asarray(reshape_frames_to_dims(np.asarray(dataset.data), experiment.dims))
    if tuple(data.shape) != tuple(experiment.shape):
        raise ValueError(
            f"Dataset data shape {dataset.data.shape} cannot be resolved to experiment shape {experiment.shape}"
        )
    return data


def motion_correct_dataset(
    experiment: Experiment,
    dataset: DataSet,
    *,
    dimensions: Iterable[str] | None = None,
    reference: ReferenceSelector = "first",
    estimation_dimensions: Iterable[str] = (),
    progress_callback: Callable[[int, int], None] | None = None,
    mask: np.ndarray | None = None,
    mask_dims: Iterable[str] | None = None,
    **registration_kwargs,
) -> MotionCorrectionResult:
    """Motion-correct MRax dataset data while preserving experiment dimensions.

    With ``dimensions=None`` all non-spatial experiment dimensions are collapsed
    into one frame stack in experiment order. When ``dimensions`` names a subset
    such as ``("t",)``, each combination of the remaining dimensions receives a
    separate registration stack before the corrected data is expanded back.
    ``estimation_dimensions`` averages magnitude along selected group dimensions
    to estimate motion. The resulting transforms resample each original image.
    Slice packages stay independent.
    """

    dim_data = _dim_ordered_data(experiment, dataset)
    non_spatial = tuple(experiment.non_spatial_dim_names)
    selected = _normalize_registration_dimensions(dimensions, non_spatial)
    group_dimensions = tuple(name for name in non_spatial if name not in selected)
    averaged = _normalize_registration_dimensions(estimation_dimensions, group_dimensions)
    independent = tuple(name for name in group_dimensions if name not in averaged)
    non_spatial_axis = {name: index for index, name in enumerate(experiment.dims)}
    selected_axes = tuple(non_spatial_axis[name] for name in selected)
    group_axes = tuple(non_spatial_axis[name] for name in group_dimensions)
    spatial_axes = tuple(non_spatial_axis[name] for name in experiment.spatial_dim_names)

    permuted_axes = selected_axes + group_axes + spatial_axes
    permuted = np.transpose(dim_data, permuted_axes) if permuted_axes else dim_data
    selected_shape = tuple(int(experiment.dims[name]) for name in selected)
    group_shape = tuple(int(experiment.dims[name]) for name in group_dimensions)
    spatial_shape = tuple(int(size) for size in experiment.spatial_shape)
    selected_frames = int(np.prod(selected_shape, dtype=int)) if selected_shape else 1
    group_count = int(np.prod(group_shape, dtype=int)) if group_shape else 1
    grouped = permuted.reshape((selected_frames, group_count) + spatial_shape)

    estimation_shape = (selected_frames,) + group_shape + spatial_shape
    estimation_data = grouped.reshape(estimation_shape)
    if averaged:
        axes = tuple(1 + group_dimensions.index(name) for name in averaged)
        estimation_data = np.mean(np.abs(estimation_data), axis=axes)
    independent_shape = tuple(experiment.dims[name] for name in independent)
    independent_count = int(np.prod(independent_shape)) if independent_shape else 1
    progress_total = selected_frames * (independent_count + (group_count if averaged else 0))
    estimation_data = estimation_data.reshape((selected_frames, independent_count) + spatial_shape)
    estimation_mask = None
    if mask is not None:
        from mrax.analysis import broadcast_mask

        full_mask = broadcast_mask(
            mask,
            experiment.shape,
            mask_dims=tuple(mask_dims) if mask_dims is not None else experiment.spatial_dim_names,
            target_dims=tuple(experiment.dims),
        )
        estimation_mask = np.transpose(full_mask, permuted_axes).reshape(estimation_shape)
        if averaged:
            estimation_mask = np.any(
                estimation_mask, axis=tuple(1 + group_dimensions.index(name) for name in averaged)
            )
        estimation_mask = estimation_mask.reshape(
            (selected_frames, independent_count) + spatial_shape
        )
    groups: list[FrameStackMotionResult] = []
    for group_index in range(independent_count):

        def report(index, count, group_index=group_index):
            if progress_callback is not None:
                progress_callback(group_index * count + index, progress_total)

        group_options = dict(registration_kwargs)
        geometry = dataset.spatial_geometry
        if geometry is not None:
            coordinates = (
                dict(
                    zip(independent, np.unravel_index(group_index, independent_shape), strict=True)
                )
                if independent
                else {}
            )
            reference_frame = (
                int(reference)
                if isinstance(reference, (int, np.integer))
                and 0 <= int(reference) < selected_frames
                else 0
            )
            if selected_shape:
                coordinates.update(
                    zip(selected, np.unravel_index(reference_frame, selected_shape), strict=True)
                )
            spacing = geometry.for_frame(geometry.frame_index(coordinates)).voxel_sizes
            if spacing is not None and len(spacing) == len(spatial_shape):
                group_options.setdefault("fixed_spacing", tuple(reversed(spacing)))
        if estimation_mask is not None:
            masks = estimation_mask[:, group_index]
            if isinstance(reference, (int, np.integer)):
                fixed = masks[int(reference)]
            elif isinstance(reference, str) and reference.lower() in {"first", "frame0"}:
                fixed = masks[0]
            else:
                fixed = np.any(masks, axis=0)
            group_options["fixed_mask"] = fixed
        groups.append(
            motion_correct_framestack(
                estimation_data[:, group_index, ...],
                reference=reference,
                progress_callback=report,
                **group_options,
            )
        )

    if not averaged:
        corrected_grouped = np.stack([group.corrected for group in groups], axis=1)
    else:
        corrected_grouped = np.empty(grouped.shape, dtype=np.result_type(grouped.dtype, np.float32))
        for group_index in range(group_count):
            coordinates = dict(
                zip(group_dimensions, np.unravel_index(group_index, group_shape), strict=True)
            )
            independent_index = (
                int(
                    np.ravel_multi_index(
                        tuple(coordinates[name] for name in independent), independent_shape
                    )
                )
                if independent
                else 0
            )
            fitted = groups[independent_index]
            for frame, transform in enumerate(fitted.transforms):
                corrected_grouped[frame, group_index] = (
                    grouped[frame, group_index]
                    if transform is None
                    else resample_image(
                        grouped[frame, group_index],
                        fitted.reference_image,
                        transform,
                        interpolator=registration_kwargs.get("interpolator", "linear"),
                        default_value=registration_kwargs.get("default_value", 0.0),
                    )
                )
                if progress_callback is not None:
                    progress_callback(
                        selected_frames * independent_count
                        + group_index * selected_frames
                        + frame
                        + 1,
                        progress_total,
                    )
    corrected_permuted = corrected_grouped.reshape(selected_shape + group_shape + spatial_shape)
    if permuted_axes:
        inverse_axes = tuple(np.argsort(permuted_axes))
        corrected_dim_data = np.transpose(corrected_permuted, inverse_axes)
    else:
        corrected_dim_data = corrected_permuted
    group_indices = tuple(
        tuple(int(value) for value in np.unravel_index(index, independent_shape))
        if independent_shape
        else ()
        for index in range(independent_count)
    )
    return MotionCorrectionResult(
        dataset=_clone_dataset(dataset, corrected_dim_data),
        dimensions=selected,
        group_dimensions=independent,
        group_indices=group_indices,
        groups=tuple(groups),
        estimation_dimensions=averaged,
    )


def _copy_experiment_with_dims(experiment: Experiment, dims: dict[str, int]) -> Experiment:
    updated = Experiment(
        dims=dims,
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


def _registered_spatial_dims(
    reference_experiment: Experiment,
    moving_experiment: Experiment,
) -> dict[str, int]:
    dims = {
        name: int(size)
        for name, size in moving_experiment.dims.items()
        if name not in moving_experiment.spatial_dim_names
    }
    dims.update(
        {
            name: int(reference_experiment.dims[name])
            for name in reference_experiment.spatial_dim_names
        }
    )
    return dims


def register_experiment_to_reference(
    reference_experiment: Experiment,
    reference_dataset: DataSet,
    moving_experiment: Experiment,
    moving_dataset: DataSet,
    *,
    reference_frame: ReferenceSelector = "first",
    moving_frame: ReferenceSelector = "first",
    interpolator: str | Any = "linear",
    default_value: float = 0.0,
    output_pixel_type: Any | None = None,
    **registration_kwargs,
) -> ExperimentRegistrationResult:
    """Register one MRax experiment to a reference experiment image.

    The transform is estimated from ``moving_frame`` to ``reference_frame`` and
    then applied to every moving frame. The returned experiment keeps the moving
    non-spatial dimensions and adopts the reference spatial grid.
    """

    reference_data = _dim_ordered_data(reference_experiment, reference_dataset)
    moving_data = _dim_ordered_data(moving_experiment, moving_dataset)
    reference_frames = np.asarray(
        reshape_dims_to_frames(reference_data, reference_experiment.dims)
    )
    moving_frames = np.asarray(reshape_dims_to_frames(moving_data, moving_experiment.dims))
    fixed, _ = _select_reference_image(
        reference_frames,
        reference_frame,
        name="reference experiment",
    )
    moving, _ = _select_reference_image(
        moving_frames,
        moving_frame,
        name="moving experiment",
    )
    registration = register_image(
        fixed,
        moving,
        interpolator=interpolator,
        default_value=default_value,
        output_pixel_type=output_pixel_type,
        **registration_kwargs,
    )
    reference_image = _sitk_reference_image(fixed)
    registered_frames = np.stack(
        [
            _resample_array_to_reference(
                frame,
                reference_image=reference_image,
                transform=registration.transform,
                interpolator=interpolator,
                default_value=default_value,
                output_pixel_type=output_pixel_type,
            )
            for frame in moving_frames
        ],
        axis=0,
    )
    updated_dims = _registered_spatial_dims(reference_experiment, moving_experiment)
    registered_data = reshape_frames_to_dims(registered_frames, updated_dims)
    return ExperimentRegistrationResult(
        experiment=_copy_experiment_with_dims(moving_experiment, updated_dims),
        dataset=_clone_dataset(moving_dataset, registered_data),
        registration=registration,
    )


__all__ = [
    "ExperimentRegistrationResult",
    "FrameStackMotionResult",
    "ImageRegistrationResult",
    "MotionCorrectionResult",
    "RegistrationDiagnostics",
    "motion_correct_dataset",
    "motion_correct_framestack",
    "register_experiment_to_reference",
    "register_image",
    "resample_image",
]
