"""Registration helpers for the MRax GUI.

The functions in this module operate independently of Qt. Lazy SimpleITK
imports keep registration dependencies local to the registration workflow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from mrax.data_struct import DataSet, Experiment
from mrax.preprocessing.motion import register_image, resample_image


@dataclass(frozen=True)
class RegistrationOptions:
    """SimpleITK registration controls exposed by the GUI."""

    transform: str = "rigid"
    metric: str = "mattes_mi"
    optimizer: str = "gradient_descent"
    interpolator: str = "linear"
    initializer: str = "geometry"
    sampling_strategy: str = "random"
    sampling_percentage: float = 0.2
    histogram_bins: int = 50
    learning_rate: float = 1.0
    min_step: float = 1e-4
    iterations: int = 200
    convergence_minimum: float = 1e-6
    convergence_window: int = 10
    shrink_factors: tuple[int, ...] = (2, 1)
    smoothing_sigmas: tuple[float, ...] = (1.0, 0.0)
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


def registration_choice_sets() -> dict[str, tuple[str, ...]]:
    """Return option values supported by :func:`run_registration`."""

    return {
        "transform": ("translation", "rigid", "similarity", "affine"),
        "metric": ("mattes_mi", "joint_histogram_mi", "mean_squares", "correlation", "ants_correlation", "demons"),
        "optimizer": ("gradient_descent", "regular_step_gradient_descent", "lbfgsb", "powell", "amoeba"),
        "interpolator": ("linear", "nearest", "bspline"),
        "initializer": ("geometry", "moments", "identity"),
        "sampling_strategy": ("none", "regular", "random"),
        "channel": ("red", "green", "blue", "cyan", "magenta", "yellow", "gray"),
    }


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
        optimizer_name=options.optimizer,
        step_size=float(options.step_size),
        num_steps=int(options.num_steps),
        fit_intercept=bool(options.fit_intercept),
    )


def run_registration(
    fixed: np.ndarray,
    moving: np.ndarray,
    options: RegistrationOptions,
    *,
    fixed_mask: np.ndarray | None = None,
    moving_mask: np.ndarray | None = None,
):
    """Run a configured SimpleITK registration and return MRax's result object."""

    return register_image(
        fixed,
        moving,
        registration_method=registration_method_factory(
            options,
            fixed_mask=fixed_mask,
            moving_mask=moving_mask,
        ),
        interpolator=options.interpolator,
    )


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


def registration_method_factory(
    options: RegistrationOptions,
    *,
    fixed_mask: np.ndarray | None = None,
    moving_mask: np.ndarray | None = None,
):
    """Create a ``register_image`` method factory from GUI options."""

    def factory(fixed_image, moving_image):
        sitk = _sitk()
        dimension = int(fixed_image.GetDimension())
        method = sitk.ImageRegistrationMethod()
        _configure_metric(method, sitk, options)
        _configure_metric_masks(method, fixed_image, moving_image, fixed_mask, moving_mask)
        _configure_sampling(method, options)
        method.SetInterpolator(_sitk_interpolator(sitk, options.interpolator))
        _configure_optimizer(method, options)
        _configure_multiresolution(method, options)
        initial = _initial_transform(sitk, fixed_image, moving_image, dimension, options)
        method.SetInitialTransform(initial, inPlace=False)
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
    sx = max(float(scale_x), 1e-6)
    sy = max(float(scale_y), 1e-6)
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
        optimizer=str(values.get("optimizer") or "gradient_descent"),
        interpolator=str(values.get("interpolator") or "linear"),
        initializer=str(values.get("initializer") or "geometry"),
        sampling_strategy=str(values.get("sampling_strategy") or "random"),
        sampling_percentage=float(values.get("sampling_percentage", 0.2)),
        histogram_bins=int(values.get("histogram_bins", 50)),
        learning_rate=float(values.get("learning_rate", 1.0)),
        min_step=float(values.get("min_step", 1e-4)),
        iterations=int(values.get("iterations", 200)),
        convergence_minimum=float(values.get("convergence_minimum", 1e-6)),
        convergence_window=int(values.get("convergence_window", 10)),
        shrink_factors=tuple(int(v) for v in values.get("shrink_factors", (2, 1))),
        smoothing_sigmas=tuple(float(v) for v in values.get("smoothing_sigmas", (1.0, 0.0))),
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
        method.SetMetricAsDemons(10)
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
    elif optimizer == "amoeba":
        method.SetOptimizerAsAmoeba(
            simplexDelta=float(options.learning_rate),
            numberOfIterations=iterations,
            parametersConvergenceTolerance=float(options.min_step),
            functionConvergenceTolerance=float(options.convergence_minimum),
        )
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
        if dimension != 2:
            raise ValueError("SimpleITK Similarity transform is exposed here for 2D images only")
        transform = sitk.Similarity2DTransform()
    elif kind == "affine":
        transform = sitk.AffineTransform(dimension)
    else:
        raise ValueError(f"Unsupported SimpleITK transform {kind!r}")

    initializer = str(options.initializer)
    if initializer == "identity" or kind == "translation":
        return transform
    mode = {
        "geometry": sitk.CenteredTransformInitializerFilter.GEOMETRY,
        "moments": sitk.CenteredTransformInitializerFilter.MOMENTS,
    }.get(initializer)
    if mode is None:
        raise ValueError(f"Unsupported transform initializer {initializer!r}")
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
            "SimpleITK is required for GUI registration. Install MRax with `pip install mrax`."
        ) from exc
    return sitk


def _sitk_metric_mask(mask: np.ndarray, reference_image, name: str):
    sitk = _sitk()
    arr = np.asarray(mask, dtype=bool)
    if arr.ndim != int(reference_image.GetDimension()):
        raise ValueError(
            f"{name} registration mask has {arr.ndim} dimensions; image has {reference_image.GetDimension()}"
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
