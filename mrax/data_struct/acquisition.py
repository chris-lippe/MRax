"""Explicit, host-side acquisition contracts shared by import and reconstruction."""
from __future__ import annotations

from dataclasses import dataclass, field
from operator import index
from typing import Mapping

import numpy as np

from .geometry import SpatialGeometry


@dataclass(frozen=True)
class ReconstructionRecipe:
    """Numerical operations expressed through the acquisition contract."""

    image_shift: tuple[float, ...] = ()
    frame_image_shifts: tuple[tuple[float, ...], ...] = ()
    partial_fourier: str = "none"
    partial_fourier_iterations: int = 12
    phase_window: str = "boxcar"
    homodyne_weighting: str = "step"
    partial_fourier_tolerance: float = 0.0
    epi_phase_correction: str = "none"
    epi_phase_prior: tuple[float, float] = (0.0, 0.0)
    coil_combine: str = "rss"
    output_scale: float = 1.0
    encoding_scale: float | None = None
    output_permutation: tuple[int, ...] = ()
    output_flips: tuple[int, ...] = ()
    frame_permutations: tuple[tuple[int, ...], ...] = ()
    stack_slices: int = 1
    output_stack_permutation: tuple[int, ...] = ()
    output_frame_indices: tuple[int, ...] = ()
    output_shape: tuple[int, ...] = ()
    crop_offset: tuple[int, ...] = ()
    gridding_shape: tuple[int, ...] = ()
    gridding_kernel_width: float = 3.0
    gridding_shape_parameter: float = 6.0
    deapodize: bool = False
    deapodization_normalization: str = "relative"
    checkerboard: bool = False
    elliptical_support: bool = False
    profile_inversion: bool = False
    average_repeats: int = 1
    readout_regrid: bool = True
    readout_interpolation: str = "adjoint"
    readout_kernel_radius: int = 8
    output_product: str = "image"
    echo_spacing_seconds: float | None = None
    product_unwrap: bool = True
    product_unwrap_method: str = "quality_guided"
    product_mask_snr: float = 0.
    product_signal_fraction: float = 0.
    product_corner_fraction: float = .2
    kspace_window: tuple[str, ...] = ()
    window_parameters: tuple[float, ...] = ()
    window_powers: tuple[int, ...] = ()

    def __post_init__(self):
        if any(index(power) < 1 for power in self.window_powers):
            raise ValueError("Window powers must be positive integers")
        if len(set(self.output_flips)) != len(self.output_flips) or any(index(axis) < 0 for axis in self.output_flips):
            raise ValueError("Output flip axes must be distinct nonnegative integers")
        if self.partial_fourier not in {"none", "pocs", "homodyne"}:
            raise ValueError("Unknown partial Fourier method")
        if index(self.partial_fourier_iterations) <= 0:
            raise ValueError("partial_fourier_iterations must be positive")
        if self.phase_window not in {"boxcar", "hann"}:
            raise ValueError("Unknown partial Fourier phase window")
        if self.homodyne_weighting not in {"step", "ramp"}:
            raise ValueError("Unknown homodyne weighting")
        if not np.isfinite(self.partial_fourier_tolerance) or self.partial_fourier_tolerance < 0:
            raise ValueError("Partial Fourier tolerance must be finite and nonnegative")
        if self.epi_phase_correction not in {"none", "entropy", "fixed", "navigator"}:
            raise ValueError("Unknown EPI phase correction")
        if np.shape(self.epi_phase_prior) != (2,) or not np.isfinite(self.epi_phase_prior).all():
            raise ValueError("EPI phase prior requires two finite radian coefficients")
        if self.coil_combine not in {"rss", "sum", "sense", "none"}:
            raise ValueError("Unknown coil combination")
        if not np.isfinite(self.output_scale) or self.output_scale <= 0:
            raise ValueError("output_scale must be finite and positive")
        if self.encoding_scale is not None and (not np.isfinite(self.encoding_scale) or self.encoding_scale <= 0):
            raise ValueError("encoding_scale must be finite and positive")
        if self.deapodization_normalization not in {"relative", "inverse_fft"}:
            raise ValueError("Unknown deapodization normalization")
        if self.readout_interpolation not in {"linear", "lanczos", "adjoint"} or index(self.readout_kernel_radius) < 2:
            raise ValueError("Readout interpolation requires a known kernel and a radius of at least two")
        if self.output_product not in {"image", "frequency_hz"}:
            raise ValueError("Unknown reconstructed image product")
        if self.product_unwrap_method not in {"quality_guided", "weighted_laplacian"}:
            raise ValueError("Unknown product phase-unwrapping method")
        if self.output_product == "frequency_hz" and (self.echo_spacing_seconds is None or not np.isfinite(self.echo_spacing_seconds) or self.echo_spacing_seconds <= 0):
            raise ValueError("Frequency products require positive echo spacing in seconds")
        if not np.isfinite(self.product_mask_snr) or self.product_mask_snr < 0 or not 0 <= self.product_signal_fraction < 1 or not 0 < self.product_corner_fraction <= .5:
            raise ValueError("Invalid frequency-product reliability mask")
        if any(kind not in {"none", "sine", "sine_squared", "hann", "hamming"} for kind in self.kspace_window):
            raise ValueError("Unknown spectral window")
        if self.window_parameters and (len(self.window_parameters) != len(self.kspace_window) or not np.isfinite(self.window_parameters).all() or any(value < 0 for value in self.window_parameters)):
            raise ValueError("Window parameters must be finite and match the encoding axes")
        if not all(np.isfinite(self.image_shift)):
            raise ValueError("Image shifts must be finite")
        if index(self.stack_slices) < 1:
            raise ValueError("stack_slices must be positive")
        if self.output_stack_permutation and (self.stack_slices <= 1 or sorted(self.output_stack_permutation) != [0, 1, 2]):
            raise ValueError("Stack output permutation requires three spatial axes and multiple slices")
        if index(self.average_repeats) < 1:
            raise ValueError("average_repeats must be positive")
        if not np.isfinite(np.asarray(self.frame_image_shifts, float)).all():
            raise ValueError("Per-frame shifts must be finite")
        for shape in (self.output_shape, self.gridding_shape):
            if any(index(n) <= 0 for n in shape):
                raise ValueError("Reconstruction grids must contain positive integer dimensions")
        for value in (self.gridding_kernel_width, self.gridding_shape_parameter):
            if not np.isfinite(value) or value <= 0:
                raise ValueError("Gridding parameters must be finite and positive")


@dataclass(frozen=True)
class ReadoutModel:
    """One-dimensional Fourier coordinates for complete readout groups.

    Consecutive samples contain one profile, possibly including both gradient
    polarities. Coordinates are cycles per pixel of ``image_points``. The
    target trajectory explicitly describes the samples synthesized for a
    gridding initialization. Iterative data terms retain the measured samples.
    """

    coordinates: np.ndarray
    image_points: int
    target_trajectory: np.ndarray
    target_points: int
    relative_cutoff: float = 1e-8

    def __post_init__(self):
        coordinates = np.asarray(self.coordinates, float)
        trajectory = np.asarray(self.target_trajectory)
        if coordinates.ndim != 1 or len(coordinates) < 2 or not np.isfinite(coordinates).all():
            raise ValueError("Readout coordinates must be a finite vector")
        if index(self.image_points) < 2 or index(self.target_points) < 2:
            raise ValueError("Readout reconstruction grids must contain at least two points")
        if trajectory.ndim != 2 or trajectory.shape[1] not in (2, 3) or len(trajectory) % self.target_points or not np.isfinite(trajectory).all():
            raise ValueError("Readout targets must contain complete finite spatial trajectories")
        if not np.isfinite(self.relative_cutoff) or not 0 < self.relative_cutoff < 1:
            raise ValueError("Readout inverse cutoff must be between zero and one")
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "target_trajectory", trajectory)


@dataclass(frozen=True)
class AcquisitionData:
    """Measured samples with explicit frame, receiver, and encoding axes.

    Cartesian samples are ``(frame, coil, *encoding_shape)``. Sampled data are
    ``(frame, coil, sample)``; trajectory columns follow ``encoding_axes`` and
    use cycles/pixel of the explicitly declared encoding grid. This normalized
    coordinate view is independent of the source's physical coordinate units.
    Validity is separate from signal values and from quadrature/statistical
    weights. The numerical contract contains normalized arrays and geometry;
    import metadata retains source paths and vendor dictionaries.
    """

    samples: np.ndarray
    encoding_shape: tuple[int, ...]
    encoding_axes: tuple[str, ...]
    kind: str = "cartesian"
    domain: str = "kspace"
    acquired: np.ndarray | None = None
    trajectory: np.ndarray | None = None
    density: np.ndarray | None = None
    statistical_weights: np.ndarray | None = None
    readout_polarity: np.ndarray | None = None
    readout_coordinates: np.ndarray | None = None
    auxiliary_samples: Mapping[str, np.ndarray] = field(default_factory=dict)
    geometry: SpatialGeometry | None = None
    frame_coordinates: Mapping[str, np.ndarray] = field(default_factory=dict)
    recipe: ReconstructionRecipe = field(default_factory=ReconstructionRecipe)
    readout_model: ReadoutModel | None = None
    schema_version: int = 1
    auxiliary_coordinates: Mapping[str, np.ndarray] = field(default_factory=dict)
    auxiliary_frame_axes: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self):
        samples = np.asarray(self.samples)
        shape = tuple(index(n) for n in self.encoding_shape)
        if self.schema_version != 1:
            raise ValueError("Unsupported acquisition schema version")
        if not 1 <= len(shape) <= 3 or any(n <= 0 for n in shape):
            raise ValueError("encoding_shape must contain positive dimensions")
        if len(self.encoding_axes) != len(shape) or len(set(self.encoding_axes)) != len(shape):
            raise ValueError("encoding_axes must uniquely name all encoding dimensions")
        if self.kind not in {"cartesian", "sampled"}:
            raise ValueError("kind must be 'cartesian' or 'sampled'")
        if self.domain not in {"kspace", "adc", "hybrid"}:
            raise ValueError("Unknown sample domain")
        if not np.iscomplexobj(samples):
            raise ValueError("Acquisition samples must have a complex dtype")
        expected_ndim = 2 + len(shape) if self.kind == "cartesian" else 3
        if samples.ndim != expected_ndim or any(n <= 0 for n in samples.shape):
            raise ValueError("Samples must have explicit nonempty frame and coil axes")
        if self.kind == "cartesian" and tuple(samples.shape[2:]) != shape:
            raise ValueError("Sample spatial dimensions do not match encoding_shape")
        if self.kind == "sampled":
            traj = np.asarray(self.trajectory)
            if traj.shape != (samples.shape[-1], len(shape)) or not np.isfinite(traj).all():
                raise ValueError("Sampled acquisition requires a finite matching trajectory")
            object.__setattr__(self, "trajectory", traj)
        elif self.trajectory is not None:
            raise ValueError("Cartesian acquisition must not supply a sampled trajectory")
        if self.readout_model is not None:
            readout = self.readout_model
            profiles = len(readout.target_trajectory) // readout.target_points
            if self.kind != "sampled" or samples.shape[-1] != profiles * len(readout.coordinates) or readout.target_trajectory.shape[1] != len(shape):
                raise ValueError("Readout model must account for every measured sample and encoding axis")
        if self.recipe.profile_inversion and self.readout_model is None:
            raise ValueError("Profile inversion requires an explicit readout model")
        for name in ("acquired", "statistical_weights", "readout_polarity"):
            value = getattr(self, name)
            if value is None:
                continue
            array = np.asarray(value)
            try:
                np.broadcast_to(array, samples.shape)
            except ValueError as exc:
                raise ValueError(f"{name} must broadcast to the complete sample layout") from exc
            if name == "acquired":
                if not np.all((array == 0) | (array == 1)):
                    raise ValueError("acquired must be a binary validity mask")
                array = array.astype(bool)
            elif name == "readout_polarity":
                if self.kind != "cartesian" or not np.isin(array, [-1, 0, 1]).all():
                    raise ValueError("Readout polarity must identify Cartesian negative/positive readouts")
                full = np.broadcast_to(array, samples.shape)
                if not np.array_equal(full, np.broadcast_to(full[..., :1], full.shape)):
                    raise ValueError("Readout polarity must be constant along each readout")
            elif not np.isfinite(array).all() or np.any(array < 0):
                raise ValueError("statistical_weights must be finite and nonnegative")
            object.__setattr__(self, name, array)
        for name, values in self.auxiliary_samples.items():
            array = np.asarray(values)
            if not isinstance(name, str) or not name or not np.iscomplexobj(array) or array.dtype.hasobject:
                raise ValueError("Auxiliary sample roles require names and complex numeric arrays")
        for name, values in self.auxiliary_coordinates.items():
            array = np.asarray(values)
            if not isinstance(name, str) or not name or name in self.auxiliary_samples or array.dtype.kind not in 'biuf' or not np.isfinite(array).all():
                raise ValueError("Auxiliary coordinates require distinct names and finite real numeric arrays")
        for name, axis in self.auxiliary_frame_axes.items():
            values = self.auxiliary_samples.get(name, self.auxiliary_coordinates.get(name))
            if values is None:
                raise ValueError("An auxiliary frame axis must name an existing array")
            axis = index(axis)
            if not -np.ndim(values) <= axis < np.ndim(values) or np.shape(values)[axis] != samples.shape[0]:
                raise ValueError("Auxiliary frame axes must match the acquisition frame count")
        if self.readout_coordinates is not None:
            coordinates = np.asarray(self.readout_coordinates)
            if self.kind != "cartesian" or self.readout_polarity is None or coordinates.shape != (2, shape[-1]) or not np.isfinite(coordinates).all():
                raise ValueError("Nonuniform Cartesian readouts require two explicit coordinate vectors and polarities")
            object.__setattr__(self, "readout_coordinates", coordinates)
        if self.density is not None:
            density = np.asarray(self.density)
            if self.kind != "sampled" or density.shape != (samples.shape[-1],):
                raise ValueError("density must match the non-Cartesian sample count")
            if not np.isfinite(density).all() or np.any(density < 0) or not np.any(density > 0):
                raise ValueError("density must be finite, nonnegative, and nonempty")
            object.__setattr__(self, "density", density)
        for name, values in self.frame_coordinates.items():
            if np.asarray(values).ndim == 0 or len(values) != samples.shape[0]:
                raise ValueError(f"Frame coordinate '{name}' must match frame count")
        ndim = len(shape)
        recipe = self.recipe
        if recipe.image_shift and len(recipe.image_shift) != ndim:
            raise ValueError("Image shift must name every encoding axis")
        if recipe.frame_image_shifts and np.shape(recipe.frame_image_shifts) != (samples.shape[0], ndim):
            raise ValueError("Per-frame shifts must match frames and encoding axes")
        if recipe.frame_permutations and len(recipe.frame_permutations) != samples.shape[0]:
            raise ValueError("Per-frame permutations must match frame count")
        if recipe.output_frame_indices and any(index(i) < 0 or i >= samples.shape[0] for i in recipe.output_frame_indices):
            raise ValueError("Output frame selection exceeds acquisition frames")
        for permutation in (recipe.output_permutation,) + recipe.frame_permutations:
            if permutation and sorted(permutation) != list(range(ndim)):
                raise ValueError("Output permutations must name each encoding axis exactly once")
        for values in (recipe.output_shape, recipe.gridding_shape, recipe.crop_offset, recipe.kspace_window):
            if values and len(values) != ndim:
                raise ValueError("Output grids and offsets must match encoding dimensions")
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "encoding_shape", shape)

    @property
    def frame_count(self) -> int:
        return int(self.samples.shape[0])

    @property
    def coil_count(self) -> int:
        return int(self.samples.shape[1])

    def select_frames(self, indices) -> AcquisitionData:
        """Select reconstructed frames while retaining their raw slice groups.

        Encoding coordinates and calibration streams remain unchanged. Arrays
        with an explicit acquisition-frame axis follow the selection; shared
        masks and weights remain shared. Selection follows the explicitly declared
        acquisition-frame axis.
        """
        from dataclasses import replace

        requested = np.asarray(indices)
        if requested.ndim != 1 or not np.issubdtype(requested.dtype, np.integer) or not len(requested):
            raise ValueError("Frame selection requires a nonempty integer vector")
        recipe = self.recipe
        order = np.asarray(recipe.output_frame_indices or tuple(range(self.frame_count)))
        product_frames = 2 if recipe.output_product == "frequency_hz" else 1
        group_size = recipe.stack_slices * recipe.average_repeats * product_frames
        if len(order) % group_size:
            raise ValueError("Output frames do not contain complete slice groups")
        groups = order.reshape((-1, group_size))
        if np.any(requested < 0) or np.any(requested >= len(groups)):
            raise ValueError("Selected frame exceeds reconstructed frame count")
        selected = groups[requested].ravel()
        options = {}
        for name in ("acquired", "statistical_weights", "readout_polarity"):
            value = getattr(self, name)
            if value is not None and np.ndim(value) == self.samples.ndim and np.shape(value)[0] == self.frame_count:
                options[name] = np.asarray(value)[selected]
        frame_shifts = tuple(recipe.frame_image_shifts[i] for i in selected) if recipe.frame_image_shifts else ()
        permutations = tuple(recipe.frame_permutations[i] for i in selected) if recipe.frame_permutations else ()
        for field_name in ('auxiliary_samples', 'auxiliary_coordinates'):
            options[field_name] = {name: np.take(value, selected, axis=self.auxiliary_frame_axes[name])
                if name in self.auxiliary_frame_axes else value
                for name, value in getattr(self, field_name).items()}
        return replace(self, samples=self.samples[selected],
            frame_coordinates={name: np.asarray(value)[selected] for name, value in self.frame_coordinates.items()},
            recipe=replace(recipe, output_frame_indices=(), frame_image_shifts=frame_shifts,
                frame_permutations=permutations), **options)

    def require_prepared(self) -> None:
        if self.domain != "kspace":
            raise ValueError(f"Samples in {self.domain!r} domain require preparation before reconstruction")
        valid = None if self.acquired is None else np.broadcast_to(self.acquired, self.samples.shape)
        if valid is not None and not np.any(valid):
            raise ValueError("Acquisition has no acquired samples")
        if not np.isfinite(self.samples).all(where=True if valid is None else valid):
            raise ValueError("Acquired samples must be finite")
