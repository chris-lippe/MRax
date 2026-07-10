from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol, Sequence

import jax.numpy as jnp
import numpy as np

from .jax_fft import cartesian_fft_jax, cartesian_ifft_jax


def _is_jax_array(value: object) -> bool:
    return isinstance(value, jnp.ndarray)


def _array_module(*arrays: object):
    for arr in arrays:
        if _is_jax_array(arr):
            return jnp
    return np


def cartesian_fft(array: np.ndarray, axes: Sequence[int] = (-2, -1)) -> np.ndarray:
    """Centered FFT with orthonormal scaling."""

    if _is_jax_array(array):
        return cartesian_fft_jax(array, axes=axes)  # type: ignore[return-value]
    arr = np.asarray(array)
    return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(arr, axes=axes), axes=axes, norm="ortho"), axes=axes)


def cartesian_ifft(array: np.ndarray, axes: Sequence[int] = (-2, -1)) -> np.ndarray:
    """Centered inverse FFT with orthonormal scaling."""

    if _is_jax_array(array):
        return cartesian_ifft_jax(array, axes=axes)  # type: ignore[return-value]
    arr = np.asarray(array)
    return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(arr, axes=axes), axes=axes, norm="ortho"), axes=axes)


def make_sampling_mask(kspace: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """Boolean mask that marks acquired samples (non-zero magnitude by default)."""

    xp = _array_module(kspace)
    return xp.abs(xp.asarray(kspace)) > threshold


@dataclass
class RecoState:
    """State flowing through the reconstruction pipeline."""

    kspace: np.ndarray
    image: np.ndarray | None = None
    trajectory: np.ndarray | None = None  # non-Cartesian k-space trajectory
    density: np.ndarray | None = None  # density compensation factors for non-Cartesian data
    mask: np.ndarray | None = None  # sampling mask for undersampled Cartesian data
    metadata: dict[str, Any] = field(default_factory=dict)

    def copy(self) -> "RecoState":
        def _copy_array(arr: np.ndarray | None):
            if arr is None:
                return None
            if _is_jax_array(arr):
                return jnp.array(arr)  # type: ignore[call-arg]
            return np.array(arr, copy=True)

        return RecoState(
            kspace=_copy_array(self.kspace),
            image=_copy_array(self.image),
            trajectory=_copy_array(self.trajectory),
            density=_copy_array(self.density),
            mask=_copy_array(self.mask),
            metadata=dict(self.metadata),
        )


class RecoStage(Protocol):
    """Minimal interface for reconstruction stages."""

    name: str

    def run(self, state: RecoState) -> RecoState:
        ...


class RecoPipeline:
    """Chain of reconstruction stages."""

    def __init__(self, stages: Iterable[RecoStage] | None = None):
        self.stages: list[RecoStage] = list(stages) if stages is not None else []

    def add(self, stage: RecoStage) -> "RecoPipeline":
        self.stages.append(stage)
        return self

    def run(self, kspace: np.ndarray | RecoState, metadata: dict[str, Any] | None = None) -> RecoState:
        if isinstance(kspace, RecoState):
            state = kspace
            if metadata:
                state.metadata.update(metadata)
        else:
            xp = _array_module(kspace)
            kspace_arr = xp.asarray(kspace)
            mask_val = make_sampling_mask(kspace_arr) if metadata is None else metadata.get(
                "mask", make_sampling_mask(kspace_arr)
            )
            mask_arr = None if mask_val is None else xp.asarray(mask_val)
            state = RecoState(
                kspace=kspace_arr,
                mask=mask_arr,
                metadata=metadata or {},
            )

        for stage in self.stages:
            state = stage.run(state)
        return state


def _grid_non_cartesian(
    samples: np.ndarray,
    trajectory: np.ndarray,
    grid_shape: Sequence[int],
    density: np.ndarray | None = None,
) -> np.ndarray:
    """Very small-footprint gridding (nearest-neighbor) for non-Cartesian data."""

    xp = _array_module(samples, trajectory, density)
    grid_shape = tuple(int(s) for s in grid_shape)
    if xp is np:
        traj = np.asarray(trajectory)
        data = np.asarray(samples)
        if data.shape[0] != traj.shape[0]:
            raise ValueError(f"Trajectory length {traj.shape[0]} does not match samples {data.shape[0]}")
        trailing_shape = data.shape[1:]
        grid = np.zeros(tuple(grid_shape) + trailing_shape, dtype=data.dtype)
        weights = np.zeros(tuple(grid_shape), dtype=np.float32)
        coords = ((traj + 0.5) * np.array(grid_shape)).astype(int)
        coords = np.clip(coords, 0, np.array(grid_shape) - 1)
        density = np.asarray(density) if density is not None else None
        for idx, pt in enumerate(coords):
            w = 1.0 if density is None else float(np.ravel(density)[idx])
            pt_tuple = tuple(int(p) for p in pt)
            grid[pt_tuple] += data[idx] * w
            weights[pt_tuple] += w
        with np.errstate(divide="ignore", invalid="ignore"):
            normed = np.where(
                weights == 0,
                grid,
                grid / np.expand_dims(weights, tuple(range(weights.ndim, grid.ndim))),
            )
        return normed

    traj = jnp.asarray(trajectory, dtype=jnp.float32)
    data = jnp.asarray(samples)
    if data.shape[0] != traj.shape[0]:
        raise ValueError(f"Trajectory length {traj.shape[0]} does not match samples {data.shape[0]}")
    trailing_shape = data.shape[1:]
    ny, nx = int(grid_shape[0]), int(grid_shape[1])
    coords = (traj + 0.5) * jnp.array([ny, nx], dtype=jnp.float32)
    coords = jnp.floor(coords).astype(jnp.int32)
    coords_y = jnp.clip(coords[:, 0], 0, ny - 1)
    coords_x = jnp.clip(coords[:, 1], 0, nx - 1)
    flat_idx = coords_y * nx + coords_x
    weights = jnp.ones((int(traj.shape[0]),), dtype=jnp.float32) if density is None else jnp.asarray(
        density, dtype=jnp.float32
    )
    if weights.shape != (int(traj.shape[0]),):
        raise ValueError("density must have shape (num_samples,)")
    flat_size = ny * nx
    if trailing_shape:
        trailing_size = int(np.prod(trailing_shape))
        data_flat = data.reshape((int(traj.shape[0]), trailing_size))
        grid_flat = jnp.zeros((flat_size, trailing_size), dtype=data.dtype)
        grid_flat = grid_flat.at[flat_idx].add(data_flat * weights[:, None])
        weights_flat = jnp.zeros((flat_size,), dtype=jnp.float32)
        weights_flat = weights_flat.at[flat_idx].add(weights)
        grid = grid_flat.reshape((ny, nx) + trailing_shape)
        weights_grid = weights_flat.reshape((ny, nx))
        denom = jnp.where(weights_grid == 0, 1.0, weights_grid)
        return grid / denom[..., None]
    grid_flat = jnp.zeros((flat_size,), dtype=data.dtype)
    grid_flat = grid_flat.at[flat_idx].add(data * weights)
    weights_flat = jnp.zeros((flat_size,), dtype=jnp.float32)
    weights_flat = weights_flat.at[flat_idx].add(weights)
    grid = grid_flat.reshape((ny, nx))
    weights_grid = weights_flat.reshape((ny, nx))
    denom = jnp.where(weights_grid == 0, 1.0, weights_grid)
    return grid / denom


class FFTStage:
    """Forward or inverse FFT on Cartesian data, with optional non-Cartesian gridding."""

    def __init__(
        self,
        direction: str = "inverse",
        axes: Sequence[int] = (-2, -1),
        grid_shape: Sequence[int] | None = None,
        use_trajectory: bool = True,
    ):
        if direction not in ("inverse", "forward"):
            raise ValueError("direction must be 'inverse' or 'forward'")
        self.direction = direction
        self.axes = tuple(axes)
        self.grid_shape = tuple(grid_shape) if grid_shape is not None else None
        self.use_trajectory = use_trajectory
        self.name = f"fft_{direction}"

    def run(self, state: RecoState) -> RecoState:
        xp = _array_module(state.kspace, state.image, state.mask, state.trajectory, state.density)
        kspace = xp.asarray(state.kspace)
        metadata = dict(state.metadata)
        axes = self.axes

        if self.direction == "inverse":
            if self.use_trajectory and state.trajectory is not None:
                shape = (
                    tuple(self.grid_shape)
                    if self.grid_shape is not None
                    else metadata.get("grid_shape")
                    if metadata.get("grid_shape") is not None
                    else metadata.get("spatial_shape")
                )
                if shape is None:
                    raise ValueError("grid_shape or spatial_shape metadata required for non-Cartesian FFT")
                gridded = _grid_non_cartesian(kspace, state.trajectory, shape, density=state.density)
                metadata["gridded_shape"] = shape
                kspace = gridded
            image = cartesian_ifft(kspace, axes=axes)
            state.image = image
            state.kspace = kspace
            state.metadata = metadata
            return state

        image = state.image if state.image is not None else cartesian_ifft(kspace, axes=axes)
        kspace_out = cartesian_fft(image, axes=axes)
        state.image = image
        state.kspace = kspace_out
        state.metadata = metadata
        return state


class ZeroFillStage:
    """Center-pad k-space to a target shape or by a scale factor."""

    def __init__(
        self,
        target_shape: Sequence[int] | None = None,
        scale: float | None = None,
        axes: Sequence[int] = (-2, -1),
    ):
        if target_shape is None and scale is None:
            raise ValueError("target_shape or scale must be provided for zero filling.")
        self.target_shape = tuple(int(s) for s in target_shape) if target_shape is not None else None
        self.scale = float(scale) if scale is not None else None
        self.axes = tuple(axes)
        self.name = "zero_fill"

    def _target(self, kspace: np.ndarray) -> list[int]:
        shape = list(kspace.shape)
        for i, axis in enumerate(self.axes):
            if self.target_shape:
                desired = self.target_shape[i] if i < len(self.target_shape) else self.target_shape[-1]
            else:
                desired = int(np.ceil(shape[axis] * self.scale)) if self.scale is not None else shape[axis]
            shape[axis] = max(shape[axis], desired)
        return shape

    def run(self, state: RecoState) -> RecoState:
        xp = _array_module(state.kspace, state.mask)
        arr = xp.asarray(state.kspace)
        target = self._target(arr)
        pads: list[tuple[int, int]] = []
        for axis, size in enumerate(arr.shape):
            desired = target[axis]
            pad_total = max(desired - size, 0)
            before = pad_total // 2
            after = pad_total - before
            pads.append((before, after))
        filled = xp.pad(arr, pad_width=pads, mode="constant")
        state.kspace = filled
        state.metadata["zero_filled_to"] = tuple(target)
        if state.mask is not None:
            mask = xp.asarray(state.mask)
            if mask.ndim == arr.ndim:
                state.mask = xp.pad(mask, pad_width=pads, mode="constant")
            elif mask.ndim == len(self.axes):
                spatial_axes = [ax if ax >= 0 else arr.ndim + ax for ax in self.axes]
                state.mask = xp.pad(mask, pad_width=[pads[ax] for ax in spatial_axes], mode="constant")
            else:
                raise ValueError(
                    f"mask ndim {mask.ndim} must match kspace ndim {arr.ndim} or spatial ndim {len(self.axes)}"
                )
        return state


class HomodyneStage:
    """Simple homodyne reconstruction for partial Fourier data."""

    def __init__(self, axis: int = -2, ramp_fraction: float = 0.1):
        self.axis = axis
        self.ramp_fraction = ramp_fraction
        self.name = "homodyne"

    def run(self, state: RecoState) -> RecoState:
        xp = _array_module(state.kspace, state.mask)
        kspace = xp.asarray(state.kspace)
        mask = xp.asarray(state.mask) if state.mask is not None else make_sampling_mask(kspace)
        axis = self.axis if self.axis >= 0 else kspace.ndim + self.axis
        n = int(kspace.shape[axis])
        acquired = xp.any(mask, axis=tuple(i for i in range(mask.ndim) if i != axis))
        idx = xp.where(acquired)[0]
        if len(idx) == 0:
            state.image = cartesian_ifft(kspace, axes=(-2, -1))
            return state
        first, last = int(idx.min()), int(idx.max())
        ramp = max(int(self.ramp_fraction * n), 1)
        weights = xp.zeros(n, dtype=xp.float32)
        if xp is np:
            weights[first : last + 1] = 1.0
        else:
            weights = weights.at[first : last + 1].set(1.0)
        ramp_start = max(first - ramp, 0)
        if ramp_start < first:
            ramp_vals = xp.linspace(0.0, 1.0, first - ramp_start, endpoint=False)
            if xp is np:
                weights[ramp_start:first] = ramp_vals
            else:
                weights = weights.at[ramp_start:first].set(ramp_vals)
        wshape = [1] * kspace.ndim
        wshape[axis] = n
        weighted = kspace * weights.reshape(wshape)
        low_res_mask = xp.zeros_like(weights)
        center = (first + last) // 2
        low_res_width = min(last - first + 1, int(0.25 * n) + 1)
        half = low_res_width // 2
        sl = slice(max(center - half, 0), min(center + half + 1, n))
        if xp is np:
            low_res_mask[sl] = 1.0
        else:
            low_res_mask = low_res_mask.at[sl].set(1.0)
        low_res = cartesian_ifft(kspace * low_res_mask.reshape(wshape), axes=(-2, -1))
        phase = xp.exp(-1j * xp.angle(low_res))
        image = cartesian_ifft(weighted, axes=(-2, -1)) * phase
        state.image = xp.real(image)
        state.kspace = weighted
        return state


class POCSStage:
    """Projection-onto-convex-sets for enforcing data consistency on undersampled k-space."""

    def __init__(self, iterations: int = 10, enforce_real: bool = True, axes: Sequence[int] = (-2, -1)):
        self.iterations = iterations
        self.enforce_real = enforce_real
        self.axes = tuple(axes)
        self.name = "pocs"

    def run(self, state: RecoState) -> RecoState:
        xp = _array_module(state.kspace, state.mask)
        measured = xp.asarray(state.kspace)
        mask = xp.asarray(state.mask) if state.mask is not None else make_sampling_mask(measured)
        estimate = measured.copy()
        fft_axes = tuple(ax if ax >= 0 else measured.ndim + ax for ax in self.axes)
        if not bool(xp.all(mask)):
            conj = measured.copy()
            for ax in fft_axes:
                conj = xp.flip(conj, axis=ax)
            conj = xp.conj(conj)
            estimate = xp.where(mask, measured, conj)
        for _ in range(self.iterations):
            img = cartesian_ifft(estimate, axes=self.axes)
            if self.enforce_real:
                img = xp.real(img)
            estimate = cartesian_fft(img, axes=self.axes)
            estimate = xp.where(mask, measured, estimate)
        state.kspace = estimate
        state.image = cartesian_ifft(estimate, axes=self.axes)
        state.mask = mask
        return state
