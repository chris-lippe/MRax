"""Registration helpers for the MRax GUI.

The functions in this module operate on numerical image data. Lazy SimpleITK
imports keep registration dependencies local to the registration workflow.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

import numpy as np

from mrax.data_struct import DataSet, Experiment
from mrax.preprocessing.motion import register_image, resample_image


@dataclass(frozen=True)
class RegistrationOptions:
    """SimpleITK registration controls exposed by the GUI."""

    transform: str = "rigid"
    metric: str = "mattes_mi"
    optimizer: str = "regular_step_gradient_descent"
    interpolator: str = "linear"
    initializer: str = "geometry"
    sampling_strategy: str = "none"
    sampling_percentage: float = 0.2
    histogram_bins: int = 32
    learning_rate: float = 0.25
    min_step: float = 1e-4
    iterations: int = 200
    convergence_minimum: float = 1e-6
    convergence_window: int = 10
    shrink_factors: tuple[int, ...] = (2, 1)
    smoothing_sigmas: tuple[float, ...] = (1.0, 0.0)
    normalize_intensity: bool = True
    validate_alignment: bool = True
    prealign_translation: bool = False
    initialization_mask_padding: int = 8
    max_displacement_voxels: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DriftCorrectionOptions:
    """MRax linear drift-correction controls exposed by the GUI."""

    transform: str = "translation"
    metric: str = "ncc"
    reference: str | int | np.ndarray = "first"
    dimensions: tuple[str, ...] | None = None
    interpolation: str = "linear"
    optimizer: str = "adam"
    step_size: float = 0.05
    num_steps: int = 200
    fit_intercept: bool = False
    normalize_intensity: bool = True


def registration_choice_sets() -> dict[str, tuple[str, ...]]:
    """Return option values supported by :func:`run_registration`."""

    return {
        "transform": ("translation", "rigid", "similarity", "affine"),
        "metric": ("mattes_mi", "joint_histogram_mi", "mean_squares", "correlation", "ants_correlation"),
        "optimizer": ("gradient_descent", "regular_step_gradient_descent", "lbfgsb", "powell", "amoeba"),
        "interpolator": ("linear", "nearest", "bspline"),
        "initializer": ("geometry", "moments", "identity"),
        "sampling_strategy": ("none", "regular", "random"),
        "channel": ("red", "green", "blue", "cyan", "magenta", "yellow", "gray"),
    }


def registration_presets() -> list[dict[str, Any]]:
    """Conservative settings for common registration tasks."""
    base = RegistrationOptions(prealign_translation=True, max_displacement_voxels=12)
    return [
        {
            "key": "different_contrasts",
            "label": "Different contrasts",
            "description": "Rigid mutual information with translation initialization.",
            "settings": _preset_settings(base),
            "contexts": ["pair", "series"],
        },
        {
            "key": "same_contrast",
            "label": "Similar contrast",
            "description": "Rigid correlation for repeated anatomical images.",
            "settings": _preset_settings(replace(base, metric="correlation")),
            "contexts": ["pair", "series"],
        },
        {
            "key": "small_translation",
            "label": "Small translations",
            "description": "Translation only. Use when rotation is negligible.",
            "settings": _preset_settings(
                replace(
                    base,
                    transform="translation",
                    prealign_translation=False,
                    max_displacement_voxels=6,
                )
            ),
            "contexts": ["pair", "series"],
        },
        {
            "key": "dynamic_contrast",
            "label": "Dynamic contrast series",
            "description": "Average acquisition contrasts to estimate one transform per time point.",
            "settings": _preset_settings(replace(base, max_displacement_voxels=6)),
            "contexts": ["series"],
            "average_contrasts": True,
        },
    ]


def _preset_settings(options: RegistrationOptions) -> dict[str, Any]:
    from dataclasses import asdict

    return asdict(options)


def drift_choice_sets() -> dict[str, tuple[str, ...]]:
    """Return option values supported by :func:`run_drift_correction`."""

    return {
        "transform": ("translation", "rigid", "similarity", "affine"),
        "metric": ("ncc", "negative_mse", "mutual_information"),
        "reference": ("first", "current", "mean", "median"),
        "interpolation": ("linear", "nearest"),
        "optimizer": ("adam", "adamw", "sgd", "lion"),
    }


def run_drift_correction(
    experiment: Experiment,
    dataset: DataSet,
    options: DriftCorrectionOptions,
    *,
    mask: np.ndarray | None = None,
    mask_dims: Sequence[str] | None = None,
):
    """Run MRax linear drift correction with GUI options."""

    from mrax.preprocessing.drift import linear_drift_correct_dataset

    return linear_drift_correct_dataset(
        experiment,
        dataset,
        dimensions=options.dimensions,
        reference=options.reference,
        transform=_drift_transform(options.transform, interpolation=options.interpolation),
        metric=_drift_metric(options.metric),
        mask=mask,
        mask_dims=mask_dims,
        optimizer_name=options.optimizer,
        step_size=float(options.step_size),
        num_steps=int(options.num_steps),
        fit_intercept=bool(options.fit_intercept),
        normalize_intensity=options.normalize_intensity,
    )


def run_registration(
    fixed: np.ndarray,
    moving: np.ndarray,
    options: RegistrationOptions,
    *,
    fixed_mask: np.ndarray | None = None,
    moving_mask: np.ndarray | None = None,
    moving_pre_transform=None,
    fixed_spacing=None,
):
    """Run a configured SimpleITK registration and return MRax's result object."""

    result = register_image(
        fixed,
        moving,
        registration_method=registration_method_factory(options),
        interpolator=options.interpolator,
        normalize_intensity=options.normalize_intensity,
        fixed_mask=fixed_mask,
        moving_mask=moving_mask,
        moving_pre_transform=moving_pre_transform,
        fixed_spacing=fixed_spacing,
        validate_alignment=options.validate_alignment,
        max_displacement_voxels=options.max_displacement_voxels,
    )
    return result


def physical_grid_transform(
    fixed_grid,
    moving_grid,
    *,
    fixed_axes,
    moving_axes,
    fixed_frame=0,
    moving_frame=0,
    fixed_indices=None,
    moving_indices=None,
):
    """Map fixed voxel coordinates into a moving image's physical grid.

    Return a voxel transform and fixed spacing in column-first index order.
    Residual image registration can refine the header-based alignment.
    """
    from mrax.data_struct import SpatialGeometry

    def basis(grid, axes, frame, indices):
        geometry = SpatialGeometry.from_metadata(grid.get("geometry"))
        affine = geometry.affine_for_frame(frame)
        if affine is None:
            return None
        spatial = [name for name in grid["dims"] if name in ("x", "y", "z")]
        vectors = {name: affine[:3, len(spatial) - 1 - index] for index, name in enumerate(spatial)}
        origin = affine[:3, 3].copy()
        for name in spatial:
            if name not in axes:
                origin += vectors[name] * (indices or {}).get(name, 0)
        return np.column_stack([vectors[name] for name in reversed(axes)]), origin

    fixed = basis(fixed_grid, fixed_axes, fixed_frame, fixed_indices)
    moving = basis(moving_grid, moving_axes, moving_frame, moving_indices)
    if fixed is None or moving is None:
        return None, None
    a, origin_a = fixed
    b, origin_b = moving
    spacing = tuple(np.linalg.norm(a, axis=0))
    inverse = np.linalg.pinv(b)
    matrix, offset = inverse @ a, inverse @ (origin_a - origin_b)
    if np.max(np.abs(b @ matrix - a)) > 1e-4 or np.linalg.norm(
        b @ offset - (origin_a - origin_b)
    ) > max(0.5 * np.linalg.norm(b, axis=0).min(), 1e-3):
        raise ValueError(
            "The selected image planes differ in physical orientation or position. Use matching planes or volume registration"
        )
    if not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-10:
        raise ValueError("Registration needs valid spatial geometry")
    if np.allclose(matrix, np.eye(len(fixed_axes)), atol=1e-6) and np.allclose(
        offset, 0, atol=1e-5
    ):
        return None, spacing
    transform = _sitk().AffineTransform(len(fixed_axes))
    transform.SetMatrix(matrix.ravel().tolist())
    transform.SetTranslation(offset.tolist())
    return transform, spacing


def _drift_transform(name: str, *, interpolation: str):
    from mrax.preprocessing.drift import (
        Affine2DTransform,
        Rigid2DTransform,
        Similarity2DTransform,
        TranslationTransform,
    )

    key = str(name or "translation").lower().replace("-", "_")
    kwargs = {"order": str(interpolation or "linear")}
    if key == "translation":
        return TranslationTransform(**kwargs)
    if key == "rigid":
        return Rigid2DTransform(**kwargs)
    if key == "similarity":
        return Similarity2DTransform(**kwargs)
    if key == "affine":
        return Affine2DTransform(**kwargs)
    raise ValueError(f"Unsupported MRax drift transform {name!r}")


def _drift_metric(name: str):
    from mrax.preprocessing.drift import (
        mutual_information,
        negative_mean_squared_error,
        normalized_cross_correlation,
    )

    key = str(name or "ncc").lower().replace("-", "_")
    if key in {"ncc", "normalized_cross_correlation", "correlation"}:
        return normalized_cross_correlation
    if key in {"negative_mse", "neg_mse", "mse"}:
        return negative_mean_squared_error
    if key in {"mutual_information", "mi"}:
        def _gui_mutual_information(fixed, moving, *, mask=None):
            return mutual_information(fixed, moving, mask=mask, bins=24, bandwidth=1.75)

        _gui_mutual_information.__name__ = "mutual_information_24bins"
        return _gui_mutual_information
    raise ValueError(f"Unsupported MRax drift metric {name!r}")


class _TranslationInitializedRegistration:
    """Run translation initialization with the same masks and physical grid."""

    def __init__(self, method, options):
        self.method = method
        self.options = options
        self.masks = {}
        self.initialization_status = ""

    def __getattr__(self, name):
        return getattr(self.method, name)

    def SetMetricFixedMask(self, mask):
        self.masks["fixed"] = mask
        self.method.SetMetricFixedMask(mask)

    def SetMetricMovingMask(self, mask):
        self.masks["moving"] = mask
        self.method.SetMetricMovingMask(mask)

    def Execute(self, fixed, moving):
        warm = registration_method_factory(
            replace(
                self.options,
                transform="translation",
                prealign_translation=False,
                iterations=min(self.options.iterations, 80),
            )
        )(fixed, moving)
        for key, mask in self.masks.items():
            if self.options.initialization_mask_padding:
                mask = _sitk().BinaryDilate(
                    mask, [int(self.options.initialization_mask_padding)] * fixed.GetDimension()
                )
            getattr(warm, "SetMetricFixedMask" if key == "fixed" else "SetMetricMovingMask")(mask)
        initial = self.method.GetInitialTransform().Downcast()
        if initial.GetName() == "CompositeTransform":
            initial = initial.GetBackTransform()
        original = initial.GetParameters()
        try:
            translation = warm.Execute(fixed, moving)
            initial.SetTranslation(translation.TransformPoint((0.0,) * fixed.GetDimension()))
            self.method.SetInitialTransform(initial, inPlace=False)
            self.initialization_status = "Translation initialization applied"
        except RuntimeError:
            initial.SetParameters(original)
            self.method.SetInitialTransform(initial, inPlace=False)
            self.initialization_status = "Original initialization retained"
        return self.method.Execute(fixed, moving)

    def GetOptimizerStopConditionDescription(self):
        return self.initialization_status + ". " + self.method.GetOptimizerStopConditionDescription()


def registration_method_factory(
    options: RegistrationOptions,
    *,
    fixed_mask: np.ndarray | None = None,
    moving_mask: np.ndarray | None = None,
):
    """Create a ``register_image`` method factory from GUI options."""

    padding = options.initialization_mask_padding
    if not np.isfinite(padding) or padding < 0 or int(padding) != padding:
        raise ValueError("Initialization mask padding must be a nonnegative integer")
    if (
        options.iterations < 1
        or not np.isfinite(options.learning_rate)
        or options.learning_rate <= 0
        or not np.isfinite(options.min_step)
        or options.min_step <= 0
        or not 0 < options.sampling_percentage <= 1
        or options.histogram_bins < 2
    ):
        raise ValueError(
            "Choose positive registration iterations, step sizes and valid metric sampling"
        )
    if (
        not options.shrink_factors
        or any(int(value) < 1 for value in options.shrink_factors)
        or len(options.shrink_factors) != len(options.smoothing_sigmas)
        or not np.isfinite(options.smoothing_sigmas).all()
        or any(value < 0 for value in options.smoothing_sigmas)
    ):
        raise ValueError(
            "Registration pyramid levels need positive shrink factors and one nonnegative sigma per level"
        )

    def factory(fixed_image, moving_image):
        sitk = _sitk()
        dimension = int(fixed_image.GetDimension())
        method = sitk.ImageRegistrationMethod()
        _configure_metric(method, sitk, options)
        _configure_metric_masks(method, fixed_image, moving_image, fixed_mask, moving_mask)
        _configure_sampling(method, options)
        method.SetInterpolator(_sitk_interpolator(sitk, options.interpolator))
        spacing = min(fixed_image.GetSpacing())
        _configure_optimizer(
            method,
            replace(
                options,
                learning_rate=options.learning_rate * spacing,
                min_step=options.min_step * spacing,
            ),
        )
        _configure_multiresolution(
            method,
            replace(
                options,
                smoothing_sigmas=tuple(value * spacing for value in options.smoothing_sigmas),
            ),
        )
        minimum_size = min((*fixed_image.GetSize(), *moving_image.GetSize()))
        smallest_level = max(options.shrink_factors, default=1)
        if minimum_size / smallest_level < 4:
            method.SetMetricUseFixedImageGradientFilter(False)
            method.SetMetricUseMovingImageGradientFilter(False)
        if minimum_size < 4:
            # Recursive Gaussian smoothing requires four voxels on every axis.
            method.SetShrinkFactorsPerLevel((1,))
            method.SetSmoothingSigmasPerLevel((0.0,))
        initial = _initial_transform(sitk, fixed_image, moving_image, dimension, options)
        method.SetInitialTransform(initial, inPlace=False)
        if options.prealign_translation and options.transform != "translation":
            wrapped = _TranslationInitializedRegistration(method, options)
            if fixed_mask is not None:
                wrapped.SetMetricFixedMask(_sitk_metric_mask(fixed_mask, fixed_image, "fixed"))
            if moving_mask is not None:
                wrapped.SetMetricMovingMask(_sitk_metric_mask(moving_mask, moving_image, "moving"))
            return wrapped
        return method

    return factory


def manual_affine_transform(
    shape: Sequence[int],
    *,
    shift_x: float = 0.0,
    shift_y: float = 0.0,
    rotation_degrees: float = 0.0,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
):
    """Return a 2D SimpleITK affine transform for manual GUI alignment.

    ``shift_x`` and ``shift_y`` are pixel-space horizontal and vertical shifts.
    The transform rotates and stretches about the image center.
    """

    rows, cols = _require_2d_shape(shape)
    sitk = _sitk()
    angle = np.deg2rad(float(rotation_degrees))
    cos_a = float(np.cos(angle))
    sin_a = float(np.sin(angle))
    sx = float(scale_x)
    sy = float(scale_y)
    if sx <= 0 or sy <= 0 or not np.isfinite([sx, sy, angle, shift_x, shift_y]).all():
        raise ValueError("Manual registration requires finite parameters and positive scales")
    matrix = np.asarray(
        [
            [sx * cos_a, -sy * sin_a],
            [sx * sin_a, sy * cos_a],
        ],
        dtype=float,
    )
    transform = sitk.AffineTransform(2)
    transform.SetCenter(((cols - 1.0) * 0.5, (rows - 1.0) * 0.5))
    transform.SetMatrix(tuple(matrix.reshape(-1)))
    transform.SetTranslation((float(shift_x), float(shift_y)))
    return transform


def apply_transform_to_image(
    image: np.ndarray,
    reference: np.ndarray,
    transform,
    *,
    interpolator: str = "linear",
) -> np.ndarray:
    """Resample ``image`` onto ``reference`` using an existing SimpleITK transform."""

    return resample_image(image, reference, transform, interpolator=interpolator)


def resample_mask_with_transform(
    mask: np.ndarray,
    reference: np.ndarray,
    transform,
    *,
    inverse: bool = False,
    output_dtype: Any | None = None,
) -> np.ndarray:
    """Nearest-neighbor resample a binary or label mask with an existing transform."""

    active_transform = inverse_transform(transform) if inverse else transform
    arr = np.asarray(mask)
    dtype = arr.dtype if output_dtype is None else np.dtype(output_dtype)
    sampled = resample_image(
        arr.astype(np.float32, copy=False),
        np.asarray(reference),
        active_transform,
        interpolator="nearest",
        default_value=0.0,
    )
    if np.issubdtype(dtype, np.bool_):
        return np.asarray(sampled >= 0.5, dtype=bool)
    if np.issubdtype(dtype, np.integer):
        return np.rint(sampled).astype(dtype, copy=False)
    return sampled.astype(dtype, copy=False)


def inverse_transform(transform):
    """Return ``transform.GetInverse()`` with a clearer error message."""

    if not hasattr(transform, "GetInverse"):
        raise ValueError("Selected transform does not support inversion")
    try:
        return transform.GetInverse()
    except RuntimeError as exc:
        raise ValueError("Selected transform is not invertible") from exc


def transform_metadata(transform) -> dict[str, Any]:
    """Return JSON-friendly metadata for a SimpleITK transform."""

    out: dict[str, Any] = {"python_type": type(transform).__name__}
    for attr, key in (
        ("GetName", "name"),
        ("GetDimension", "dimension"),
        ("GetParameters", "parameters"),
        ("GetFixedParameters", "fixed_parameters"),
        ("GetMatrix", "matrix"),
        ("GetTranslation", "translation"),
        ("GetCenter", "center"),
        ("GetAngle", "angle_radians"),
    ):
        if not hasattr(transform, attr):
            continue
        try:
            value = getattr(transform, attr)()
        except Exception:
            continue
        if isinstance(value, tuple):
            value = list(value)
        out[key] = value
    if transform.GetName() == "CompositeTransform":
        out["components"] = [transform_metadata(transform.GetNthTransform(index))
            for index in range(transform.GetNumberOfTransforms())]
    return _jsonable(out)


def alignment_metrics(
    fixed: np.ndarray,
    moving: np.ndarray,
    *,
    mask: np.ndarray | None = None,
) -> dict[str, float | None]:
    """Compute simple pre/post image-alignment analytics for preview text."""

    a, b = _matched_arrays(fixed, moving)
    valid = np.isfinite(a) & np.isfinite(b)
    if mask is not None:
        m, _ = _matched_arrays(mask, a)
        valid &= np.asarray(m, dtype=bool)
    if int(np.count_nonzero(valid)) < 2:
        return {"ncc": None, "mse": None, "mad": None}
    av = a[valid].astype(float)
    bv = b[valid].astype(float)
    mad = float(np.mean(np.abs(av - bv)))
    mse = float(np.mean(np.square(av - bv)))
    a_std = float(np.std(av))
    b_std = float(np.std(bv))
    ncc = None
    if a_std > 0 and b_std > 0:
        ncc = float(np.mean(((av - float(np.mean(av))) / a_std) * ((bv - float(np.mean(bv))) / b_std)))
    return {"ncc": _safe_float(ncc), "mse": _safe_float(mse), "mad": _safe_float(mad)}


def registration_overlay(
    fixed: np.ndarray,
    moving: np.ndarray,
    *,
    fixed_channel: str = "red",
    moving_channel: str = "green",
    fixed_alpha: float = 1.0,
    moving_alpha: float = 1.0,
) -> np.ndarray:
    """Return an RGB overlay of fixed and moving images with configurable channels."""

    a, b = _matched_arrays(fixed, moving)
    overlay = np.zeros(a.shape + (3,), dtype=np.float32)
    overlay += float(np.clip(fixed_alpha, 0.0, 1.0)) * _robust_unit(a)[..., None] * _channel_vector(fixed_channel)
    overlay += float(np.clip(moving_alpha, 0.0, 1.0)) * _robust_unit(b)[..., None] * _channel_vector(moving_channel)
    return np.clip(overlay, 0.0, 1.0).astype(np.float32)


def colorize_image(
    image: np.ndarray,
    *,
    channel: str = "gray",
    alpha: float = 1.0,
) -> np.ndarray:
    """Return one image as an RGB display array in the selected color channel."""

    key = str(channel or "gray").lower()
    unit = _robust_unit(np.abs(image) if np.iscomplexobj(image) else image)
    return np.clip(float(np.clip(alpha, 0.0, 1.0)) * unit[..., None] * _channel_vector(key), 0.0, 1.0).astype(np.float32)


def parse_sequence(text: str, cast=float) -> tuple[Any, ...]:
    """Parse comma/space separated GUI multiresolution values."""

    cleaned = str(text or "").replace(";", ",").replace("x", ",")
    values = [item.strip() for chunk in cleaned.split(",") for item in chunk.split() if item.strip()]
    return tuple(cast(value) for value in values)


def options_from_mapping(values: Mapping[str, Any]) -> RegistrationOptions:
    """Build :class:`RegistrationOptions` from loose GUI text values."""

    return RegistrationOptions(
        transform=str(values.get("transform") or "rigid"),
        metric=str(values.get("metric") or "mattes_mi"),
        optimizer=str(values.get("optimizer") or "regular_step_gradient_descent"),
        interpolator=str(values.get("interpolator") or "linear"),
        initializer=str(values.get("initializer") or "geometry"),
        sampling_strategy=str(values.get("sampling_strategy") or "none"),
        sampling_percentage=float(values.get("sampling_percentage", 0.2)),
        histogram_bins=int(values.get("histogram_bins", 32)),
        learning_rate=float(values.get("learning_rate", 0.25)),
        min_step=float(values.get("min_step", 1e-4)),
        iterations=int(values.get("iterations", 200)),
        convergence_minimum=float(values.get("convergence_minimum", 1e-6)),
        convergence_window=int(values.get("convergence_window", 10)),
        shrink_factors=tuple(int(v) for v in values.get("shrink_factors", (2, 1))),
        smoothing_sigmas=tuple(float(v) for v in values.get("smoothing_sigmas", (1.0, 0.0))),
        normalize_intensity=bool(values.get("normalize_intensity", True)),
        validate_alignment=bool(values.get("validate_alignment", True)),
        prealign_translation=bool(values.get("prealign_translation", False)),
        initialization_mask_padding=int(values.get("initialization_mask_padding", 8)),
        max_displacement_voxels=(
            float(values["max_displacement_voxels"])
            if values.get("max_displacement_voxels") is not None
            else None
        ),
    )


def _configure_metric(method, sitk, options: RegistrationOptions) -> None:
    metric = str(options.metric)
    bins = max(2, int(options.histogram_bins))
    if metric == "mattes_mi":
        method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=bins)
    elif metric == "joint_histogram_mi":
        method.SetMetricAsJointHistogramMutualInformation(numberOfHistogramBins=bins)
    elif metric == "mean_squares":
        method.SetMetricAsMeanSquares()
    elif metric == "correlation":
        method.SetMetricAsCorrelation()
    elif metric == "ants_correlation":
        method.SetMetricAsANTSNeighborhoodCorrelation(radius=2)
    elif metric == "demons":
        raise ValueError(
            "The Demons metric requires displacement-field registration. "
            "Choose a metric supported by the linear transforms"
        )
    else:
        raise ValueError(f"Unsupported SimpleITK metric {metric!r}")


def _configure_metric_masks(
    method,
    fixed_image,
    moving_image,
    fixed_mask: np.ndarray | None,
    moving_mask: np.ndarray | None,
) -> None:
    if fixed_mask is not None:
        method.SetMetricFixedMask(_sitk_metric_mask(fixed_mask, fixed_image, "fixed"))
    if moving_mask is not None:
        method.SetMetricMovingMask(_sitk_metric_mask(moving_mask, moving_image, "moving"))


def _configure_sampling(method, options: RegistrationOptions) -> None:
    strategy = str(options.sampling_strategy)
    if strategy == "none":
        method.SetMetricSamplingStrategy(method.NONE)
        return
    if strategy == "regular":
        method.SetMetricSamplingStrategy(method.REGULAR)
    elif strategy == "random":
        method.SetMetricSamplingStrategy(method.RANDOM)
    else:
        raise ValueError(f"Unsupported metric sampling strategy {strategy!r}")
    method.SetMetricSamplingPercentage(float(np.clip(options.sampling_percentage, 0.0, 1.0)), 1)


def _configure_optimizer(method, options: RegistrationOptions) -> None:
    optimizer = str(options.optimizer)
    iterations = max(0, int(options.iterations))
    if optimizer == "gradient_descent":
        method.SetOptimizerAsGradientDescent(
            learningRate=float(options.learning_rate),
            numberOfIterations=iterations,
            convergenceMinimumValue=float(options.convergence_minimum),
            convergenceWindowSize=max(1, int(options.convergence_window)),
            maximumStepSizeInPhysicalUnits=float(options.learning_rate),
        )
        method.SetOptimizerScalesFromPhysicalShift()
    elif optimizer == "regular_step_gradient_descent":
        method.SetOptimizerAsRegularStepGradientDescent(
            learningRate=float(options.learning_rate),
            minStep=float(options.min_step),
            numberOfIterations=iterations,
            relaxationFactor=0.5,
            gradientMagnitudeTolerance=float(options.convergence_minimum),
        )
        method.SetOptimizerScalesFromPhysicalShift()
    elif optimizer == "lbfgsb":
        method.SetOptimizerAsLBFGSB(
            gradientConvergenceTolerance=float(options.convergence_minimum),
            numberOfIterations=iterations,
        )
    elif optimizer == "powell":
        method.SetOptimizerAsPowell(
            numberOfIterations=iterations,
            maximumLineIterations=max(1, int(options.extra.get("maximum_line_iterations", 100))),
            stepLength=float(options.learning_rate),
            stepTolerance=float(options.min_step),
            valueTolerance=float(options.convergence_minimum),
        )
        method.SetOptimizerScalesFromPhysicalShift()
    elif optimizer == "amoeba":
        method.SetOptimizerAsAmoeba(
            simplexDelta=float(options.learning_rate),
            numberOfIterations=iterations,
            parametersConvergenceTolerance=float(options.min_step),
            functionConvergenceTolerance=float(options.convergence_minimum),
        )
        method.SetOptimizerScalesFromPhysicalShift()
    else:
        raise ValueError(f"Unsupported SimpleITK optimizer {optimizer!r}")


def _configure_multiresolution(method, options: RegistrationOptions) -> None:
    shrink = tuple(max(1, int(value)) for value in options.shrink_factors)
    sigmas = tuple(float(value) for value in options.smoothing_sigmas)
    if shrink:
        method.SetShrinkFactorsPerLevel(shrink)
    if sigmas:
        method.SetSmoothingSigmasPerLevel(sigmas)
        method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()


def _initial_transform(sitk, fixed_image, moving_image, dimension: int, options: RegistrationOptions):
    kind = str(options.transform)
    if kind == "translation":
        transform = sitk.TranslationTransform(dimension)
    elif kind == "rigid":
        transform = sitk.Euler2DTransform() if dimension == 2 else sitk.Euler3DTransform()
    elif kind == "similarity":
        transform = sitk.Similarity2DTransform() if dimension == 2 else sitk.Similarity3DTransform()
    elif kind == "affine":
        transform = sitk.AffineTransform(dimension)
    else:
        raise ValueError(f"Unsupported SimpleITK transform {kind!r}")

    initializer = str(options.initializer)
    if initializer == "identity":
        if hasattr(transform, "SetCenter"):
            transform.SetCenter(fixed_image.TransformContinuousIndexToPhysicalPoint(
                tuple((size - 1.0) * .5 for size in fixed_image.GetSize())))
        return transform
    mode = {
        "geometry": sitk.CenteredTransformInitializerFilter.GEOMETRY,
        "moments": sitk.CenteredTransformInitializerFilter.MOMENTS,
    }.get(initializer)
    if mode is None:
        raise ValueError(f"Unsupported transform initializer {initializer!r}")
    if kind == "translation":
        centered = sitk.CenteredTransformInitializer(fixed_image, moving_image,
            sitk.Euler2DTransform() if dimension == 2 else sitk.Euler3DTransform(), mode)
        transform.SetOffset(centered.GetTranslation())
        return transform
    return sitk.CenteredTransformInitializer(fixed_image, moving_image, transform, mode)


def _sitk_interpolator(sitk, value: str):
    key = str(value).lower().replace("-", "_")
    if key in {"linear", "bilinear", "trilinear"}:
        return sitk.sitkLinear
    if key in {"nearest", "nearest_neighbor", "nn"}:
        return sitk.sitkNearestNeighbor
    if key in {"bspline", "b_spline"}:
        return sitk.sitkBSpline
    raise ValueError(f"Unsupported SimpleITK interpolator {value!r}")


def _sitk():
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise ImportError(
            "SimpleITK is required for GUI registration. Install `pip install 'mrax[gui]'`."
        ) from exc
    return sitk


def _sitk_metric_mask(mask: np.ndarray, reference_image, name: str):
    sitk = _sitk()
    arr = np.asarray(mask, dtype=bool)
    if not np.any(arr):
        raise ValueError(f"{name} registration mask must contain selected pixels")
    if arr.ndim != int(reference_image.GetDimension()):
        raise ValueError(
            f"{name} registration mask has {arr.ndim} dimensions. Image has {reference_image.GetDimension()}"
        )
    image = sitk.GetImageFromArray(arr.astype(np.uint8))
    expected_size = tuple(int(v) for v in reference_image.GetSize())
    if tuple(int(v) for v in image.GetSize()) != expected_size:
        image_shape = tuple(reversed(expected_size))
        raise ValueError(f"{name} registration mask shape {arr.shape} does not match image shape {image_shape}")
    image.CopyInformation(reference_image)
    return image


def _matched_arrays(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left = np.asarray(a)
    right = np.asarray(b)
    if np.iscomplexobj(left):
        left = np.abs(left)
    if np.iscomplexobj(right):
        right = np.abs(right)
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.ndim != right.ndim:
        raise ValueError(f"Images must have matching dimensions, got {left.shape} and {right.shape}")
    slices = tuple(slice(0, min(int(l), int(r))) for l, r in zip(left.shape, right.shape))
    return left[slices], right[slices]


def _robust_unit(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    lo, hi = np.percentile(finite, [1.0, 99.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(finite))
        hi = float(np.nanmax(finite))
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _channel_vector(channel: str) -> np.ndarray:
    key = str(channel or "").lower().replace("-", "_")
    colors = {
        "red": (1.0, 0.0, 0.0),
        "green": (0.0, 1.0, 0.0),
        "blue": (0.0, 0.35, 1.0),
        "cyan": (0.0, 1.0, 1.0),
        "magenta": (1.0, 0.0, 1.0),
        "yellow": (1.0, 1.0, 0.0),
        "gray": (1.0, 1.0, 1.0),
        "grey": (1.0, 1.0, 1.0),
    }
    if key not in colors:
        raise ValueError(f"Unsupported overlay color channel {channel!r}")
    return np.asarray(colors[key], dtype=np.float32)


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if np.isfinite(out) else None


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _require_2d_shape(shape: Sequence[int]) -> tuple[int, int]:
    values = tuple(int(v) for v in shape)
    if len(values) != 2:
        raise ValueError(f"Manual affine alignment currently expects a 2D image, got shape {values}")
    return values[0], values[1]
