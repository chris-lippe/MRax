from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Protocol

import jax
import jax.numpy as jnp
import jax.scipy.ndimage as jnd
import numpy as np
import optax

from mrax.data_struct.dataset import DataSet
from mrax.data_struct.dimensions import reshape_frames_to_dims
from mrax.data_struct.experiment import Experiment
from mrax.devices import DeviceLike, device_function, place_optimizer
from mrax.opt.optimizers import make_optimizer

ReferenceSelector = str | int | np.ndarray
InterpolationOrder = int | str
SimilarityMetric = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray | float]
TransformCallable = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]


class ImageTransform(Protocol):
    """Protocol for image transforms used by linear drift correction."""

    def identity_parameters(self, image: jnp.ndarray) -> jnp.ndarray:
        """Return parameters that leave ``image`` unchanged."""

    def apply(self, image: jnp.ndarray, parameters: jnp.ndarray) -> jnp.ndarray:
        """Apply one parameter vector to one image."""


def _sample_coordinates(
    image: jnp.ndarray,
    coordinates: jnp.ndarray,
    *,
    order: InterpolationOrder,
    mode: str,
    cval: float,
) -> jnp.ndarray:
    resolved_order = _interpolation_order(order)
    arr = jnp.asarray(image)

    def _sample(values: jnp.ndarray) -> jnp.ndarray:
        return jnd.map_coordinates(
            values,
            tuple(coordinates[index] for index in range(int(coordinates.shape[0]))),
            order=resolved_order,
            mode=str(mode),
            cval=float(cval),
        )

    if jnp.iscomplexobj(arr):
        return _sample(jnp.real(arr)) + 1j * _sample(jnp.imag(arr))
    return _sample(arr)


def _interpolation_order(order: InterpolationOrder) -> int:
    if isinstance(order, str):
        key = order.strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "nearest": 0,
            "nearest_neighbor": 0,
            "nn": 0,
            "zero": 0,
            "linear": 1,
            "bilinear": 1,
            "trilinear": 1,
            "first": 1,
        }
        if key not in aliases:
            raise ValueError(
                f"Unknown interpolation {order!r}; use 'nearest' or 'linear'"
            )
        return aliases[key]
    resolved_order = int(order)
    if resolved_order not in (0, 1):
        raise ValueError("JAX image transforms support nearest (0) or linear (1) interpolation")
    return resolved_order


def _centered_grid_2d(shape: tuple[int, int]) -> tuple[jnp.ndarray, jnp.ndarray]:
    grid = jnp.stack(
        jnp.meshgrid(
            jnp.arange(int(shape[0]), dtype=jnp.float32),
            jnp.arange(int(shape[1]), dtype=jnp.float32),
            indexing="ij",
        ),
        axis=0,
    )
    center = jnp.asarray(
        [(shape[0] - 1.0) * 0.5, (shape[1] - 1.0) * 0.5],
        dtype=jnp.float32,
    )
    return grid, center


def _resample_affine_2d(
    image: jnp.ndarray,
    matrix: jnp.ndarray,
    translation: jnp.ndarray,
    *,
    order: InterpolationOrder,
    mode: str,
    cval: float,
) -> jnp.ndarray:
    arr = jnp.asarray(image)
    if arr.ndim != 2:
        raise ValueError(f"2D drift transforms need a 2D image, got shape {arr.shape}")
    grid, center = _centered_grid_2d((int(arr.shape[0]), int(arr.shape[1])))
    flat_grid = (grid - center[:, None, None]).reshape((2, -1))
    source = jnp.linalg.solve(
        jnp.asarray(matrix, dtype=jnp.float32),
        flat_grid - jnp.asarray(translation, dtype=jnp.float32).reshape((2, 1)),
    )
    coordinates = source.reshape(grid.shape) + center[:, None, None]
    return _sample_coordinates(arr, coordinates, order=order, mode=mode, cval=cval)


@dataclass(frozen=True)
class TranslationTransform:
    """N-D JAX subpixel translation in MRax array-axis order.

    Parameters are shifts in array-axis order. For a 2D ``(x, y)`` MRax image,
    a parameter vector ``(dx, dy)`` moves the sampled image along those axes.
    ``order`` accepts ``"linear"``/``1`` or ``"nearest"``/``0``.
    """

    order: InterpolationOrder = "linear"
    mode: str = "constant"
    cval: float = 0.0

    def identity_parameters(self, image: jnp.ndarray) -> jnp.ndarray:
        return jnp.zeros(jnp.asarray(image).ndim, dtype=jnp.float32)

    def apply(self, image: jnp.ndarray, parameters: jnp.ndarray) -> jnp.ndarray:
        image_array = jnp.asarray(image)
        shifts = jnp.asarray(parameters, dtype=jnp.float32).reshape(-1)
        if int(shifts.size) != int(image_array.ndim):
            raise ValueError(
                f"TranslationTransform expected {image_array.ndim} shifts, got {shifts.size}"
            )
        coordinates = jnp.stack(
            jnp.meshgrid(
                *[
                    jnp.arange(int(size), dtype=jnp.float32)
                    for size in image_array.shape
                ],
                indexing="ij",
            ),
            axis=0,
        )
        coordinates -= shifts.reshape((-1,) + (1,) * image_array.ndim)
        return _sample_coordinates(
            image_array,
            coordinates,
            order=self.order,
            mode=self.mode,
            cval=self.cval,
        )


@dataclass(frozen=True)
class Rigid2DTransform:
    """2D translation plus rotation about the image center.

    Parameters are ``(dx, dy, angle_radians)`` in MRax array-axis order.
    ``order`` accepts ``"linear"``/``1`` or ``"nearest"``/``0``.
    """

    order: InterpolationOrder = "linear"
    mode: str = "constant"
    cval: float = 0.0

    def identity_parameters(self, image: jnp.ndarray) -> jnp.ndarray:
        if jnp.asarray(image).ndim != 2:
            raise ValueError("Rigid2DTransform requires 2D images")
        return jnp.zeros(3, dtype=jnp.float32)

    def apply(self, image: jnp.ndarray, parameters: jnp.ndarray) -> jnp.ndarray:
        dx, dy, angle = jnp.asarray(parameters, dtype=jnp.float32).reshape((3,))
        cosine = jnp.cos(angle)
        sine = jnp.sin(angle)
        matrix = jnp.asarray([[cosine, -sine], [sine, cosine]], dtype=jnp.float32)
        return _resample_affine_2d(
            image,
            matrix,
            jnp.asarray([dx, dy], dtype=jnp.float32),
            order=self.order,
            mode=self.mode,
            cval=self.cval,
        )


@dataclass(frozen=True)
class Similarity2DTransform:
    """2D translation, rotation, and isotropic log-scale.

    Parameters are ``(dx, dy, angle_radians, log_scale)``. ``order`` accepts
    ``"linear"``/``1`` or ``"nearest"``/``0``.
    """

    order: InterpolationOrder = "linear"
    mode: str = "constant"
    cval: float = 0.0

    def identity_parameters(self, image: jnp.ndarray) -> jnp.ndarray:
        if jnp.asarray(image).ndim != 2:
            raise ValueError("Similarity2DTransform requires 2D images")
        return jnp.zeros(4, dtype=jnp.float32)

    def apply(self, image: jnp.ndarray, parameters: jnp.ndarray) -> jnp.ndarray:
        dx, dy, angle, log_scale = jnp.asarray(parameters, dtype=jnp.float32).reshape((4,))
        cosine = jnp.cos(angle)
        sine = jnp.sin(angle)
        scale = jnp.exp(log_scale)
        matrix = scale * jnp.asarray([[cosine, -sine], [sine, cosine]], dtype=jnp.float32)
        return _resample_affine_2d(
            image,
            matrix,
            jnp.asarray([dx, dy], dtype=jnp.float32),
            order=self.order,
            mode=self.mode,
            cval=self.cval,
        )


@dataclass(frozen=True)
class Affine2DTransform:
    """2D translation plus a small affine update about the image center.

    Parameters are ``(dx, dy, a00, a01, a10, a11)`` and the affine matrix is
    ``eye(2) + [[a00, a01], [a10, a11]]``. ``order`` accepts
    ``"linear"``/``1`` or ``"nearest"``/``0``.
    """

    order: InterpolationOrder = "linear"
    mode: str = "constant"
    cval: float = 0.0

    def identity_parameters(self, image: jnp.ndarray) -> jnp.ndarray:
        if jnp.asarray(image).ndim != 2:
            raise ValueError("Affine2DTransform requires 2D images")
        return jnp.zeros(6, dtype=jnp.float32)

    def apply(self, image: jnp.ndarray, parameters: jnp.ndarray) -> jnp.ndarray:
        dx, dy, a00, a01, a10, a11 = jnp.asarray(
            parameters,
            dtype=jnp.float32,
        ).reshape((6,))
        matrix = jnp.eye(2, dtype=jnp.float32) + jnp.asarray(
            [[a00, a01], [a10, a11]],
            dtype=jnp.float32,
        )
        return _resample_affine_2d(
            image,
            matrix,
            jnp.asarray([dx, dy], dtype=jnp.float32),
            order=self.order,
            mode=self.mode,
            cval=self.cval,
        )


@dataclass(frozen=True)
class LinearDriftDiagnostics:
    """Optimizer diagnostics for one global linear drift fit."""

    success: bool
    status: int
    message: str
    initial_objective: float
    final_objective: float
    function_evaluations: int
    iterations: int
    objective_history: tuple[float, ...]


@dataclass(frozen=True)
class FrameStackLinearDriftResult:
    """Linear drift correction fitted to one ``(frame, *spatial)`` stack."""

    corrected: np.ndarray
    reference_image: np.ndarray
    reference_index: int | None
    transform: ImageTransform | TransformCallable
    parameters: np.ndarray
    coefficients: np.ndarray
    normalized_times: np.ndarray
    similarities_before: np.ndarray
    similarities_after: np.ndarray
    diagnostics: LinearDriftDiagnostics


@dataclass(frozen=True)
class LinearDriftCorrectionResult:
    """MRax dataset corrected by grouped linear drift fits."""

    dataset: DataSet
    dimensions: tuple[str, ...]
    group_dimensions: tuple[str, ...]
    group_indices: tuple[tuple[int, ...], ...]
    groups: tuple[FrameStackLinearDriftResult, ...]

    @property
    def corrected(self):
        """Corrected dataset values in the original MRax dimension layout."""

        return self.dataset.data


def normalized_cross_correlation(
    fixed: jnp.ndarray,
    moving: jnp.ndarray,
    *,
    mask: jnp.ndarray | np.ndarray | None = None,
) -> jnp.ndarray:
    """Return finite-sample normalized cross-correlation in ``[-1, 1]``."""

    fixed_values, moving_values, selected = _masked_values(fixed, moving, mask=mask)
    weights = selected.astype(fixed_values.dtype)
    sample_count = jnp.sum(weights)
    fixed_mean = jnp.sum(fixed_values) / jnp.maximum(sample_count, 1.0)
    moving_mean = jnp.sum(moving_values) / jnp.maximum(sample_count, 1.0)
    fixed_centered = (fixed_values - fixed_mean) * weights
    moving_centered = (moving_values - moving_mean) * weights
    denominator = jnp.linalg.norm(fixed_centered) * jnp.linalg.norm(moving_centered)
    return jnp.where(
        (sample_count >= 2.0) & (denominator > jnp.finfo(fixed_values.dtype).eps),
        jnp.sum(fixed_centered * moving_centered) / denominator,
        jnp.asarray(-1.0, dtype=fixed_values.dtype),
    )


def negative_mean_squared_error(
    fixed: jnp.ndarray,
    moving: jnp.ndarray,
    *,
    mask: jnp.ndarray | np.ndarray | None = None,
) -> jnp.ndarray:
    """Return negative MSE so larger values mean more similar images."""

    fixed_values, moving_values, selected = _masked_values(fixed, moving, mask=mask)
    weights = selected.astype(fixed_values.dtype)
    sample_count = jnp.sum(weights)
    mse = jnp.sum(jnp.square(fixed_values - moving_values) * weights)
    mse /= jnp.maximum(sample_count, 1.0)
    return jnp.where(sample_count > 0.0, -mse, -jnp.inf)


def mutual_information(
    fixed: jnp.ndarray,
    moving: jnp.ndarray,
    *,
    mask: jnp.ndarray | np.ndarray | None = None,
    bins: int = 48,
    bandwidth: float = 1.5,
) -> jnp.ndarray:
    """Return differentiable soft-histogram mutual information."""

    if int(bins) < 2:
        raise ValueError("bins must be at least 2")
    if float(bandwidth) <= 0.0:
        raise ValueError("bandwidth must be positive")
    fixed_values, moving_values, selected = _masked_values(fixed, moving, mask=mask)
    weights = selected.reshape(-1).astype(fixed_values.dtype)
    fixed_flat = fixed_values.reshape(-1)
    moving_flat = moving_values.reshape(-1)
    selected_count = jnp.sum(weights)
    eps = jnp.finfo(fixed_values.dtype).eps

    def _soft_assign(values: jnp.ndarray) -> jnp.ndarray:
        safe_count = jnp.maximum(selected_count, 1.0)
        mean = jnp.sum(weights * values) / safe_count
        variance = jnp.sum(weights * jnp.square(values - mean)) / safe_count
        sigma = jnp.sqrt(jnp.maximum(variance, jnp.finfo(values.dtype).eps))
        centers = jnp.linspace(
            mean - 3.0 * sigma,
            mean + 3.0 * sigma,
            int(bins),
            dtype=values.dtype,
        )
        spacing = jnp.maximum(
            jnp.abs(centers[1] - centers[0]) * float(bandwidth),
            eps,
        )
        assignments = jnp.exp(-0.5 * jnp.square((values[:, None] - centers) / spacing))
        row_sum = jnp.sum(assignments, axis=1, keepdims=True)
        return weights[:, None] * assignments / jnp.maximum(row_sum, eps)

    fixed_assignments = _soft_assign(fixed_flat)
    moving_assignments = _soft_assign(moving_flat)
    joint = fixed_assignments.T @ moving_assignments
    joint /= jnp.maximum(jnp.sum(joint), eps)
    expected = jnp.sum(joint, axis=1)[:, None] * jnp.sum(joint, axis=0)[None, :]
    valid = (joint > eps) & (expected > eps)
    # Avoid log(0 / 0) even on masked-out branches. JAX traces both sides of
    # jnp.where, and invalid inactive branches can still poison gradients.
    safe_ratio = jnp.maximum(joint, eps) / jnp.maximum(expected, eps)
    information_terms = jnp.where(valid, joint * jnp.log(safe_ratio), 0.0)
    information = jnp.sum(information_terms)
    return jnp.where(selected_count >= 2.0, information, jnp.asarray(0.0, dtype=fixed_values.dtype))


def _masked_values(
    fixed: jnp.ndarray,
    moving: jnp.ndarray,
    *,
    mask: jnp.ndarray | np.ndarray | None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    fixed_array = jnp.abs(fixed) if jnp.iscomplexobj(fixed) else jnp.asarray(fixed)
    moving_array = jnp.abs(moving) if jnp.iscomplexobj(moving) else jnp.asarray(moving)
    if fixed_array.shape != moving_array.shape:
        raise ValueError("fixed and moving images must have the same shape")
    selected = jnp.isfinite(fixed_array) & jnp.isfinite(moving_array)
    if mask is not None:
        mask_array = jnp.asarray(mask, dtype=bool)
        if mask_array.shape != fixed_array.shape:
            raise ValueError(f"mask shape {mask_array.shape} does not match image shape {fixed_array.shape}")
        selected &= mask_array
    zero = jnp.zeros((), dtype=jnp.result_type(fixed_array, moving_array, jnp.float32))
    return jnp.where(selected, fixed_array, zero), jnp.where(selected, moving_array, zero), selected


def _check_framestack(frames: np.ndarray, *, name: str) -> np.ndarray:
    stack = np.asarray(frames)
    if stack.ndim not in (3, 4):
        raise ValueError(f"{name} must have shape (frame, x, y) or (frame, x, y, z)")
    if int(stack.shape[0]) <= 0:
        raise ValueError(f"{name} must contain at least one frame")
    return stack


def _select_reference_image(
    frames: np.ndarray,
    reference: ReferenceSelector,
    *,
    name: str,
) -> tuple[np.ndarray, int | None]:
    stack = _check_framestack(frames, name=name)
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
    selected = np.asarray(reference)
    if selected.ndim != stack.ndim - 1 or tuple(selected.shape) != tuple(stack.shape[1:]):
        raise ValueError(
            f"{name} reference image shape {selected.shape} does not match frame shape {stack.shape[1:]}"
        )
    return selected, None


def _identity_parameters(
    transform: ImageTransform | TransformCallable,
    image: np.ndarray,
    identity_parameters: Iterable[float] | None,
) -> jnp.ndarray:
    if identity_parameters is not None:
        params = jnp.asarray(tuple(identity_parameters), dtype=jnp.float32).reshape(-1)
    elif hasattr(transform, "identity_parameters"):
        params = jnp.asarray(
            transform.identity_parameters(jnp.asarray(image)),
            dtype=jnp.float32,
        ).reshape(-1)
    else:
        raise ValueError(
            "Custom callable transforms need identity_parameters, or an "
            "identity_parameters(image) method."
        )
    if params.size <= 0:
        raise ValueError("identity_parameters must contain at least one parameter")
    return params


def _apply_transform(
    transform: ImageTransform | TransformCallable,
    image: jnp.ndarray,
    parameters: jnp.ndarray,
) -> jnp.ndarray:
    if hasattr(transform, "apply"):
        return jnp.asarray(transform.apply(image, parameters))
    return jnp.asarray(transform(image, parameters))


def _apply_stack_jax(
    frames: jnp.ndarray,
    transform: ImageTransform | TransformCallable,
    parameters: jnp.ndarray,
) -> jnp.ndarray:
    return jax.vmap(lambda frame, params: _apply_transform(transform, frame, params))(
        frames,
        parameters,
    )


def apply_image_transform_stack(
    frames: np.ndarray,
    transform: ImageTransform | TransformCallable,
    parameters: np.ndarray,
) -> np.ndarray:
    """Apply one per-frame image-transform parameter vector to a frame stack."""

    stack = _check_framestack(frames, name="frames")
    parameter_array = np.asarray(parameters, dtype=float)
    if parameter_array.ndim != 2 or int(parameter_array.shape[0]) != int(stack.shape[0]):
        raise ValueError("parameters must have shape (frame, transform_parameter)")
    return np.asarray(
        _apply_stack_jax(jnp.asarray(stack), transform, jnp.asarray(parameter_array))
    )


def _normalized_times(
    times: Iterable[float] | None,
    frame_count: int,
    *,
    anchor_index: int,
) -> np.ndarray:
    if times is None:
        values = np.arange(frame_count, dtype=float)
    else:
        values = np.asarray(tuple(times), dtype=float).reshape(-1)
        if values.size != frame_count:
            raise ValueError(f"times must have {frame_count} values, got {values.size}")
        if not np.isfinite(values).all():
            raise ValueError("times must contain finite values")
    scale = float(np.max(values) - np.min(values))
    if scale <= np.finfo(float).eps:
        return np.zeros(frame_count, dtype=float)
    return (values - float(values[anchor_index])) / scale


def _design_matrix(normalized_times: np.ndarray, *, fit_intercept: bool) -> np.ndarray:
    if fit_intercept:
        return np.column_stack((np.ones_like(normalized_times), normalized_times))
    return normalized_times[:, None]


def _coefficient_bounds(
    max_parameter_change: float | Iterable[float] | None,
    *,
    basis_count: int,
    parameter_count: int,
) -> tuple[jnp.ndarray, jnp.ndarray] | None:
    if max_parameter_change is None:
        return None
    values = np.asarray(max_parameter_change, dtype=float)
    if values.ndim == 0:
        limits = np.full(parameter_count, abs(float(values)), dtype=float)
    else:
        limits = np.abs(values.reshape(-1))
        if limits.size != parameter_count:
            raise ValueError(
                "max_parameter_change must be a scalar or have one value per transform parameter"
            )
    if not np.isfinite(limits).all():
        raise ValueError("max_parameter_change must contain finite values")
    tiled = np.tile(np.asarray(limits, dtype=np.float32), int(basis_count))
    tiled = tiled.reshape((int(basis_count), int(parameter_count)))
    return jnp.asarray(-tiled), jnp.asarray(tiled)


def _evaluate_similarities(
    reference_image: jnp.ndarray,
    frames: jnp.ndarray,
    metric: SimilarityMetric,
    *,
    mask: jnp.ndarray | None = None,
) -> np.ndarray:
    scores = jax.vmap(lambda frame: _call_metric(metric, reference_image, frame, mask=mask))(
        frames
    )
    return np.asarray(scores, dtype=float)


def _call_metric(
    metric: SimilarityMetric,
    fixed: jnp.ndarray,
    moving: jnp.ndarray,
    *,
    mask: jnp.ndarray | None,
) -> jnp.ndarray:
    if mask is None:
        return jnp.asarray(metric(fixed, moving))
    return jnp.asarray(metric(fixed, moving, mask=mask))


@device_function
def linear_drift_correct_framestack(
    frames: np.ndarray,
    *,
    transform: ImageTransform | TransformCallable | None = None,
    metric: SimilarityMetric = normalized_cross_correlation,
    reference: ReferenceSelector = "first",
    times: Iterable[float] | None = None,
    anchor_frame: int | None = None,
    identity_parameters: Iterable[float] | None = None,
    fit_intercept: bool = False,
    frame_weights: Iterable[float] | None = None,
    mask: np.ndarray | jnp.ndarray | None = None,
    max_parameter_change: float | Iterable[float] | None = None,
    optimizer_name: str = "adam",
    step_size: float = 0.05,
    num_steps: int = 200,
    optimizer_factory=None,
    optimizer=None,
    device: DeviceLike = "cpu",
) -> FrameStackLinearDriftResult:
    """Fit and apply a global linear transform-parameter drift to a frame stack.

    The JAX optimizer does not fit one transform per frame. It optimizes the
    coefficients of ``identity + design(time) @ coefficients`` and evaluates
    the chosen image transform and similarity metric for each frame under that
    shared model. By default the line is anchored at identity for the reference
    frame or the first frame; set ``fit_intercept=True`` to fit a free line.
    ``mask`` restricts every objective and similarity evaluation to selected
    reference-grid pixels. Custom metrics and transforms used in this objective
    must be JAX-compatible; when ``mask`` is passed the metric must accept a
    ``mask=`` keyword.
    """

    stack = _check_framestack(frames, name="frames")
    reference_image, reference_index = _select_reference_image(stack, reference, name="frames")
    image_transform = TranslationTransform() if transform is None else transform
    identity = _identity_parameters(image_transform, reference_image, identity_parameters)

    if anchor_frame is None:
        anchor_index = int(reference_index) if reference_index is not None else 0
    else:
        anchor_index = int(anchor_frame)
    if anchor_index < 0 or anchor_index >= int(stack.shape[0]):
        raise IndexError(f"anchor_frame {anchor_index} is out of range")

    normalized_times = _normalized_times(times, int(stack.shape[0]), anchor_index=anchor_index)
    design = jnp.asarray(_design_matrix(normalized_times, fit_intercept=fit_intercept))
    coefficients = jnp.zeros((int(design.shape[1]), int(identity.size)), dtype=jnp.float32)

    if frame_weights is None:
        weights = jnp.ones(int(stack.shape[0]), dtype=jnp.float32)
    else:
        weight_values = np.asarray(tuple(frame_weights), dtype=float).reshape(-1)
        if weight_values.size != int(stack.shape[0]):
            raise ValueError(
                f"frame_weights must have {stack.shape[0]} values, got {weight_values.size}"
            )
        if not np.isfinite(weight_values).all() or np.any(weight_values < 0.0):
            raise ValueError("frame_weights must be finite and non-negative")
        if not np.any(weight_values > 0.0):
            raise ValueError("frame_weights must include at least one positive value")
        weights = jnp.asarray(weight_values, dtype=jnp.float32)

    stack_jax = jnp.asarray(stack)
    reference_jax = jnp.asarray(reference_image)
    if mask is None:
        mask_jax = None
    else:
        mask_array = np.asarray(mask, dtype=bool)
        if tuple(mask_array.shape) != tuple(stack.shape[1:]):
            raise ValueError(
                f"mask shape {mask_array.shape} does not match frame shape {stack.shape[1:]}"
            )
        if not np.any(mask_array):
            raise ValueError("mask must contain at least one selected pixel")
        mask_jax = jnp.asarray(mask_array)

    def _parameters_from_coefficients(coeffs: jnp.ndarray) -> jnp.ndarray:
        return identity[None, :] + design @ coeffs

    def _objective(coeffs: jnp.ndarray) -> jnp.ndarray:
        corrected = _apply_stack_jax(
            stack_jax,
            image_transform,
            _parameters_from_coefficients(coeffs),
        )
        scores = jax.vmap(
            lambda frame: _call_metric(metric, reference_jax, frame, mask=mask_jax)
        )(corrected)
        finite_scores = jnp.where(jnp.isfinite(scores), scores, -1e12)
        return -jnp.sum(weights * finite_scores) / jnp.maximum(jnp.sum(weights), 1.0)

    initial_objective = float(_objective(coefficients))
    bounds = _coefficient_bounds(
        max_parameter_change,
        basis_count=int(coefficients.shape[0]),
        parameter_count=int(coefficients.shape[1]),
    )
    if optimizer is not None:
        opt = optimizer
    elif optimizer_factory is not None:
        opt = optimizer_factory(step_size)
    else:
        opt = make_optimizer(
            optimizer_name,
            step_size,
            box_constraints=bounds,
            device=device,
        )
    opt = place_optimizer(opt, device)
    opt_state = opt.init(coefficients)

    @jax.jit
    def _train_step(coeffs, state):
        loss, gradients = jax.value_and_grad(_objective)(coeffs)
        updates, next_state = opt.update(gradients, state, coeffs)
        return optax.apply_updates(coeffs, updates), next_state, loss

    objective_history = [initial_objective]
    completed_iterations = 0
    stopped_nonfinite = False
    stop_detail = ""
    for step in range(max(0, int(num_steps))):
        next_coefficients, next_state, loss = _train_step(coefficients, opt_state)
        loss_value = float(loss)
        next_coefficients_np = np.asarray(next_coefficients)
        if not np.isfinite(loss_value):
            stopped_nonfinite = True
            stop_detail = f"objective became non-finite at step {step + 1}"
            objective_history.append(loss_value)
            break
        if not np.isfinite(next_coefficients_np).all():
            stopped_nonfinite = True
            stop_detail = f"optimizer proposed non-finite transform coefficients at step {step + 1}"
            objective_history.append(loss_value)
            break
        coefficients, opt_state = next_coefficients, next_state
        objective_history.append(loss_value)
        completed_iterations += 1
    final_objective = float(_objective(coefficients))
    objective_history.append(final_objective)
    parameters = _parameters_from_coefficients(coefficients)
    corrected = _apply_stack_jax(stack_jax, image_transform, parameters)
    similarities_before = _evaluate_similarities(
        reference_jax,
        stack_jax,
        metric,
        mask=mask_jax,
    )
    similarities_after = _evaluate_similarities(
        reference_jax,
        corrected,
        metric,
        mask=mask_jax,
    )
    success = bool(np.isfinite(final_objective) and not stopped_nonfinite)
    if stopped_nonfinite:
        status = 2
        message = (
            f"{optimizer_name} stopped after {completed_iterations} finite step(s): "
            f"{stop_detail}; keeping the last finite transform coefficients"
        )
    elif np.isfinite(final_objective):
        status = 0
        message = f"{optimizer_name} completed {completed_iterations} JAX optimizer steps"
    else:
        status = 1
        message = f"{optimizer_name} ended with a non-finite objective"
    diagnostics = LinearDriftDiagnostics(
        success=success,
        status=status,
        message=message,
        initial_objective=float(initial_objective),
        final_objective=final_objective,
        function_evaluations=len(objective_history),
        iterations=completed_iterations,
        objective_history=tuple(float(value) for value in objective_history),
    )
    return FrameStackLinearDriftResult(
        corrected=np.asarray(corrected),
        reference_image=np.asarray(reference_jax),
        reference_index=reference_index,
        transform=image_transform,
        parameters=np.asarray(parameters),
        coefficients=np.asarray(coefficients),
        normalized_times=normalized_times,
        similarities_before=similarities_before,
        similarities_after=similarities_after,
        diagnostics=diagnostics,
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


def _dim_ordered_data(experiment: Experiment, dataset: DataSet) -> np.ndarray:
    data = np.asarray(reshape_frames_to_dims(np.asarray(dataset.data), experiment.dims))
    if tuple(data.shape) != tuple(experiment.shape):
        raise ValueError(
            f"Dataset data shape {dataset.data.shape} cannot be resolved to experiment shape {experiment.shape}"
        )
    return data


def _normalize_dimensions(
    dimensions: Iterable[str] | None,
    available: tuple[str, ...],
) -> tuple[str, ...]:
    if dimensions is None:
        return available
    selected = tuple(str(name) for name in dimensions)
    if len(set(selected)) != len(selected):
        raise ValueError("dimensions must not contain duplicates")
    unknown = [name for name in selected if name not in available]
    if unknown:
        raise ValueError(f"Drift dimensions {unknown} are not non-spatial dimensions {available}")
    return tuple(name for name in available if name in selected)


@device_function
def linear_drift_correct_dataset(
    experiment: Experiment,
    dataset: DataSet,
    *,
    dimensions: Iterable[str] | None = None,
    reference: ReferenceSelector = "first",
    times: Iterable[float] | None = None,
    device: DeviceLike = "cpu",
    **drift_kwargs,
) -> LinearDriftCorrectionResult:
    """Fit grouped linear drift corrections and restore MRax dataset dimensions."""

    dim_data = _dim_ordered_data(experiment, dataset)
    non_spatial = tuple(experiment.non_spatial_dim_names)
    selected = _normalize_dimensions(dimensions, non_spatial)
    group_dimensions = tuple(name for name in non_spatial if name not in selected)
    non_spatial_axis = {name: index for index, name in enumerate(non_spatial)}
    selected_axes = tuple(non_spatial_axis[name] for name in selected)
    group_axes = tuple(non_spatial_axis[name] for name in group_dimensions)
    spatial_axes = tuple(range(len(non_spatial), dim_data.ndim))

    permuted_axes = selected_axes + group_axes + spatial_axes
    permuted = np.transpose(dim_data, permuted_axes) if permuted_axes else dim_data
    selected_shape = tuple(int(experiment.dims[name]) for name in selected)
    group_shape = tuple(int(experiment.dims[name]) for name in group_dimensions)
    spatial_shape = tuple(int(size) for size in experiment.spatial_shape)
    selected_frames = int(np.prod(selected_shape, dtype=int)) if selected_shape else 1
    group_count = int(np.prod(group_shape, dtype=int)) if group_shape else 1
    grouped = permuted.reshape((selected_frames, group_count) + spatial_shape)

    groups = tuple(
        linear_drift_correct_framestack(
            grouped[:, group_index, ...],
            reference=reference,
            times=times,
            device=device,
            **drift_kwargs,
        )
        for group_index in range(group_count)
    )
    corrected_grouped = np.stack([group.corrected for group in groups], axis=1)
    corrected_permuted = corrected_grouped.reshape(selected_shape + group_shape + spatial_shape)
    if permuted_axes:
        corrected_dim_data = np.transpose(corrected_permuted, tuple(np.argsort(permuted_axes)))
    else:
        corrected_dim_data = corrected_permuted
    group_indices = tuple(
        tuple(int(value) for value in np.unravel_index(index, group_shape))
        if group_shape
        else ()
        for index in range(group_count)
    )
    return LinearDriftCorrectionResult(
        dataset=_clone_dataset(dataset, corrected_dim_data),
        dimensions=selected,
        group_dimensions=group_dimensions,
        group_indices=group_indices,
        groups=groups,
    )


__all__ = [
    "Affine2DTransform",
    "FrameStackLinearDriftResult",
    "ImageTransform",
    "LinearDriftCorrectionResult",
    "LinearDriftDiagnostics",
    "Rigid2DTransform",
    "Similarity2DTransform",
    "TranslationTransform",
    "apply_image_transform_stack",
    "linear_drift_correct_dataset",
    "linear_drift_correct_framestack",
    "mutual_information",
    "negative_mean_squared_error",
    "normalized_cross_correlation",
]
