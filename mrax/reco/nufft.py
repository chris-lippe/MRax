"""Fixed-trajectory, unitary nonuniform Fourier encoding using only JAX.

The Kaiser--Bessel convolution and analytic deapodization follow Beatty et al.,
IEEE TMI 24 (2005), 799--808, doi:10.1109/TMI.2005.848376. Interpolation and
spreading use the same periodic stencil, making the implemented pair adjoint
even though each approximates a nonuniform DFT. Density weighting is explicit.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import lru_cache, partial
from math import ceil, prod
from operator import index

import jax
import jax.numpy as jnp
import numpy as np
from jax.core import Tracer
from scipy.special import i0e, i1e

from ._nufft_autodiff import apply_nufft


@lru_cache(maxsize=32)
def _kernel_table(beta):
    """Cubic Hermite coefficients for the normalized Kaiser--Bessel kernel.

    A small geometry-only table avoids repeated Bessel/exponential evaluation
    when trajectory stencils are streamed. Values and slopes use double precision
    during preparation, then follow the trajectory dtype on the selected device.
    """
    position = np.linspace(0., 1., 4097)
    argument = beta * np.sqrt(np.maximum(0., 1 - position**2))
    normalization = np.exp(argument - beta) / i0e(beta)
    values = i0e(argument) * normalization
    ratio = np.where(argument > 0, i1e(argument) / np.where(argument > 0, argument, 1), .5)
    slopes = -beta**2 * position * ratio * normalization / (len(position) - 1)
    values.setflags(write=False)
    slopes.setflags(write=False)
    return values, slopes


@partial(jax.jit, static_argnames=("grid_shape", "width"))
def _prepare_positions(coordinates, *, grid_shape, width):
    coordinates = jnp.where((coordinates >= -.5) & (coordinates < .5), coordinates,
                           jnp.remainder(coordinates + .5, 1.) - .5)
    location = coordinates * jnp.asarray(grid_shape, dtype=coordinates.dtype)
    first = jnp.ceil(location - width / 2).astype(jnp.int32)
    return first, first.astype(location.dtype) - location


@partial(jax.jit, static_argnames=("grid_shape", "width", "beta"))
def _separable_stencil(first, offset, *, grid_shape, width, beta):
    neighbour = jnp.arange(width, dtype=jnp.int32)
    indices = first[..., None] + neighbour
    displacement = 2 * (offset[..., None] + neighbour) / width
    table, slopes = (jnp.asarray(value, dtype=offset.dtype) for value in _kernel_table(beta))
    location = jnp.minimum(jnp.abs(displacement), 1.) * (len(table) - 1)
    cell = jnp.minimum(location.astype(jnp.int32), len(table) - 2)
    fraction = location - cell
    left, right = table[cell], table[cell + 1]
    first_slope, second_slope = slopes[cell], slopes[cell + 1]
    difference = right - left
    weights = left + fraction * (first_slope + fraction * (
        3 * difference - 2 * first_slope - second_slope
        + fraction * (first_slope + second_slope - 2 * difference)))
    weights = jnp.where(jnp.abs(displacement) <= 1, weights, 0)
    shape = jnp.asarray(grid_shape, dtype=jnp.int32)[None, :, None]
    return jnp.mod(indices + shape // 2, shape), weights


@dataclass(frozen=True)
class NUFFTPlan:
    """Reusable encoding for fixed coordinates in cycles/pixel, in axis order.

    Prepare outside ``jit``/``grad``; transform calls support both and arbitrary
    leading batch axes. Gradients are defined for images and samples at a fixed
    trajectory. ``eps`` selects interpolation accuracy; floating-point error
    also depends on dtype and accumulation.
    Sample chunks and batches bound temporary stencil and oversampled-grid memory.
    """

    spatial_shape: tuple[int, ...]
    grid_shape: tuple[int, ...]
    sample_count: int
    width: int
    chunk_size: int
    batch_size: int
    indices: jax.Array
    weights: jax.Array
    deapodization: tuple[jax.Array, ...]
    cached_stencil: bool
    shape_parameter: float
    prefer_wide_accumulation: bool

    def _stencil(self, chunk):
        # Cache bounded separable weights, expanding neighbours only per chunk.
        indices, weights = chunk
        if not self.cached_stencil:
            indices, weights = _separable_stencil(indices[..., 0], weights[..., 0],
                grid_shape=self.grid_shape, width=self.width, beta=self.shape_parameter)
        flat = indices[:, 0]
        kernel = weights[:, 0]
        for axis in range(1, len(self.spatial_shape)):
            flat = (flat[..., None] * self.grid_shape[axis]
                    + indices[:, axis].reshape((-1,) + (1,) * axis + (self.width,)))
            kernel = (kernel[..., None]
                      * weights[:, axis].reshape((-1,) + (1,) * axis + (self.width,)))
        return flat.reshape(-1), kernel.reshape(self.chunk_size, -1)

    def _apodize(self, image):
        for axis, factor in enumerate(self.deapodization):
            broadcast = (1,) * (image.ndim - len(self.spatial_shape) + axis)
            broadcast += (self.spatial_shape[axis],) + (1,) * (len(self.spatial_shape) - axis - 1)
            image = image * factor.reshape(broadcast).astype(image.real.dtype)
        return image

    def _forward_batch(self, image):
        ndim = len(self.spatial_shape)
        axes = tuple(range(-ndim, 0))
        padding = [(0, 0)]
        for size, full in zip(self.spatial_shape, self.grid_shape, strict=True):
            start = full // 2 - size // 2
            padding.append((start, full - size - start))
        image = jnp.pad(self._apodize(image), padding)
        grid = jnp.fft.fftshift(jnp.fft.fftn(
            jnp.fft.ifftshift(image, axes=axes), axes=axes, norm="ortho"), axes=axes)
        grid = grid.reshape((image.shape[0], -1))

        def interpolate(_, chunk):
            indices, weights = self._stencil(chunk)
            values = grid[:, indices].reshape(image.shape[0], self.chunk_size, -1)
            return None, jnp.sum(values * weights[None], axis=-1)

        _, values = jax.lax.scan(interpolate, None, (self.indices, self.weights))
        values = values.transpose(1, 0, 2).reshape(image.shape[0], -1)[:, :self.sample_count]
        scale = jnp.asarray(np.sqrt(prod(self.grid_shape) / prod(self.spatial_shape)),
                            dtype=values.real.dtype)
        return values * scale

    def _adjoint_batch(self, samples):
        padded = self.indices.shape[0] * self.chunk_size
        samples = jnp.pad(samples, ((0, 0), (0, padded - self.sample_count)))
        chunks = samples.reshape(samples.shape[0], -1, self.chunk_size).transpose(1, 0, 2)
        grid_size = prod(self.grid_shape)
        long_sum = len(chunks) > 64 and self.sample_count > prod(self.spatial_shape)
        # CPU double accumulation improves long sums and preserves output dtype. Consumer GPUs use partial float32 sums to avoid slow atomics.
        grid_dtype = (jnp.complex128 if long_sum and samples.dtype == jnp.complex64
                      and self.prefer_wide_accumulation and jax.config.values["jax_enable_x64"]
                      else samples.dtype)
        groups = 1
        if grid_dtype == jnp.complex64 and long_sum:
            # Partial sums reduce coherent single-precision accumulation error.
            # Bound replicated-grid workspace; one large FFT grid is unavoidable.
            groups = max(1, min(8, (512 * 1024**2) // (samples.shape[0] * grid_size * samples.dtype.itemsize)))

        def spread(grid, chunk):
            indices, weights = self._stencil(chunk[:2])
            indices = indices + chunk[3] * grid_size
            values = (chunk[2][..., None] * weights[None]).reshape(samples.shape[0], -1)
            return grid.at[:, indices].add(values.astype(grid.dtype)), None

        grid = jnp.zeros((samples.shape[0], groups * grid_size), dtype=grid_dtype)
        grid, _ = jax.lax.scan(spread, grid, (self.indices, self.weights, chunks,
            jnp.arange(len(chunks), dtype=jnp.int32) % groups))
        grid = grid.reshape((samples.shape[0], groups) + self.grid_shape).sum(axis=1).astype(samples.dtype)
        axes = tuple(range(-len(self.spatial_shape), 0))
        image = jnp.fft.fftshift(jnp.fft.ifftn(
            jnp.fft.ifftshift(grid, axes=axes), axes=axes, norm="ortho"), axes=axes)
        crop = (slice(None),) + tuple(
            slice(full // 2 - size // 2, full // 2 - size // 2 + size)
            for size, full in zip(self.spatial_shape, self.grid_shape, strict=True))
        scale = jnp.asarray(np.sqrt(prod(self.grid_shape) / prod(self.spatial_shape)),
                            dtype=image.real.dtype)
        return self._apodize(image[crop]) * scale

    def _apply_raw(self, values, *, adjoint):
        values = jnp.asarray(values)
        input_shape = (self.sample_count,) if adjoint else self.spatial_shape
        if tuple(values.shape[-len(input_shape):]) != input_shape:
            raise ValueError("NUFFT input shape does not match the prepared plan")
        dtype = jnp.result_type(values.dtype, self.weights.dtype, jnp.complex64)
        values = values.astype(dtype)
        leading = values.shape[:-len(input_shape)]
        count = prod(leading)
        if count == 0:
            raise ValueError("NUFFT batch dimensions must be nonempty")
        # Limit each FFT grid to 64 MiB where possible; a single large volume
        # still needs its full grid. FFT workspaces are additional allocations.
        grid_batch = max(1, (64 * 1024**2) // (prod(self.grid_shape) * dtype.itemsize))
        batch = min(count, self.batch_size, grid_batch)
        values = values.reshape((count,) + input_shape)
        values = jnp.pad(values, ((0, (-count) % batch),) + ((0, 0),) * len(input_shape))
        values = values.reshape((-1, batch) + input_shape)
        transform = self._adjoint_batch if adjoint else self._forward_batch
        output = jax.lax.map(transform, values)
        output_shape = self.spatial_shape if adjoint else (self.sample_count,)
        return output.reshape((-1,) + output_shape)[:count].reshape(leading + output_shape)

    def forward(self, image):
        """Return sum(x[n] exp(-2πi k·(n-floor(N/2)))) / sqrt(prod(N))."""
        return apply_nufft(self, image, adjoint=False)

    def adjoint(self, samples):
        """Apply the unweighted conjugate transpose."""
        return apply_nufft(self, samples, adjoint=True)


def prepare_nufft(spatial_shape, trajectory, *, eps=1e-6, chunk_size=None, batch_size=None,
                  cache_budget_bytes=128 * 1024**2):
    """Prepare a portable 1D/2D/3D NUFFT with twofold grid oversampling.

    ``eps`` in [1e-12, 1e-2] selects the Kaiser--Bessel width conservatively.
    Single precision has a coordinate/roundoff floor; use float64 trajectories
    and JAX x64 for tighter accuracy. Preparation validates concrete coordinates
    once. Separable stencils exceeding ``cache_budget_bytes`` are evaluated per
    chunk from compact positions instead of cached. Zero disables stencil caching.
    """
    shape = tuple(int(size) for size in spatial_shape)
    if not 1 <= len(shape) <= 3 or any(size <= 0 for size in shape):
        raise ValueError("NUFFT requires one to three positive spatial dimensions")
    if not np.isfinite(eps) or not 1e-12 <= eps <= 1e-2:
        raise ValueError("NUFFT eps must be between 1e-12 and 1e-2")
    coordinates = jnp.asarray(trajectory)
    if isinstance(coordinates, Tracer):
        raise ValueError("Prepare a fixed-trajectory NUFFT outside jit/grad; trajectory gradients are unsupported")
    if jnp.issubdtype(coordinates.dtype, jnp.integer) or coordinates.dtype == jnp.float16:
        coordinates = coordinates.astype(jnp.float32)
    if (coordinates.ndim != 2 or coordinates.shape[1] != len(shape)
            or coordinates.shape[0] == 0 or not np.isfinite(np.asarray(coordinates)).all()
            or not jnp.issubdtype(coordinates.dtype, jnp.floating)):
        raise ValueError("NUFFT trajectory must be a nonempty finite real (samples, ndim) array")
    if eps < 1e-6 and coordinates.dtype == jnp.float32:
        warnings.warn("NUFFT eps below 1e-6 needs float64 coordinates and JAX x64; "
                      "single-precision roundoff may dominate", RuntimeWarning, stacklevel=2)
    width = max(4, ceil(-np.log10(eps)) + 1)
    grid_shape = tuple(2 * size for size in shape)
    if prod(grid_shape) > np.iinfo(np.int32).max:
        raise ValueError("NUFFT oversampled grid exceeds supported index range")
    # Limit each expanded stencil to at most 2**20 neighbours.
    neighbour_budget = max(1, (1 << 20) // width**len(shape))
    default_chunk = min(2048, 1 << (neighbour_budget.bit_length() - 1))
    chunk_size = min(coordinates.shape[0], default_chunk if chunk_size is None else int(chunk_size))
    cpu = all(device.platform == "cpu" for device in coordinates.devices())
    if batch_size is None:
        batch_size = 4 if cpu else 32
    if chunk_size <= 0 or int(batch_size) < 1:
        raise ValueError("NUFFT chunk and batch sizes must be positive")
    cache_budget_bytes = index(cache_budget_bytes)
    if cache_budget_bytes < 0:
        raise ValueError("NUFFT cache budget must be nonnegative")
    beta = float(np.pi * np.sqrt((.75 * width)**2 - .8))
    dtype = coordinates.dtype
    padding = (-coordinates.shape[0]) % chunk_size
    first, offset = _prepare_positions(coordinates, grid_shape=grid_shape, width=width)
    first = jnp.pad(first, ((0, padding), (0, 0)))
    offset = jnp.pad(offset, ((0, padding), (0, 0)))
    cached = (coordinates.shape[0] + padding) * len(shape) * width * (4 + dtype.itemsize)
    cached_stencil = cached <= cache_budget_bytes
    if cached_stencil:
        indices, weights = _separable_stencil(first, offset,
            grid_shape=grid_shape, width=width, beta=beta)
        stored_width = width
    else:
        indices, weights = first[..., None], offset[..., None]
        stored_width = 1
    deapodization = []
    # Compute small geometry-only factors in double precision before device cast.
    for size, full in zip(shape, grid_shape, strict=True):
        position = (np.arange(size, dtype=np.float64) - size // 2) / full
        root = np.sqrt(beta**2 - (np.pi * width * position)**2)
        response = width * np.sinh(root) / (root * np.i0(beta))
        deapodization.append(jnp.asarray(1 / response, dtype=dtype))
    return NUFFTPlan(shape, grid_shape, coordinates.shape[0], width, chunk_size,
        int(batch_size), indices.reshape((-1, chunk_size, len(shape), stored_width)),
        weights.reshape((-1, chunk_size, len(shape), stored_width)), tuple(deapodization),
        cached_stencil, beta, cpu)
