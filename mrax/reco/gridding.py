"""Convolutional gridding on numerical sampling geometry."""
from __future__ import annotations

from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import i0e


def grid_kaiser_bessel(samples, trajectory, grid_shape, *, density=None,
                       kernel_width=3., shape_parameter=6., chunk_size=16384):
    """Accumulate density-weighted samples with a separable Kaiser-Bessel kernel.

    Coordinates use cycles per grid pixel. Grid cells accumulate the quadrature-
    weighted sample contributions.
    Bounded chunks avoid allocating the sample-by-neighbour expansion.
    """
    shape = tuple(int(v) for v in grid_shape)
    if len(shape) not in (2, 3) or any(v <= 0 for v in shape):
        raise ValueError("Gridding requires a positive two- or three-dimensional grid")
    if not np.isfinite(kernel_width) or kernel_width <= 0 or not np.isfinite(shape_parameter) or shape_parameter <= 0:
        raise ValueError("Kernel width and shape parameter must be finite and positive")
    values = jnp.asarray(samples)
    points = jnp.asarray(trajectory)
    if points.shape != (values.shape[-1], len(shape)):
        raise ValueError("Trajectory must match sample count and grid dimensions")
    count = len(points)
    if count == 0 or chunk_size < 1:
        raise ValueError("Sample count and chunk size must be positive")
    weights = jnp.ones(count, dtype=points.dtype) if density is None else jnp.asarray(density)
    if weights.shape != (count,):
        raise ValueError("Density must match sample count")
    leading = values.shape[:-1]
    batch = int(np.prod(leading))
    values = values.reshape((batch, count))
    padded = (-count) % chunk_size
    points = jnp.pad(points, ((0, padded), (0, 0)))
    weights = jnp.pad(weights, (0, padded))
    values = jnp.pad(values, ((0, 0), (0, padded)))
    chunks = points.shape[0] // chunk_size
    offsets = jnp.asarray(list(product(range(int(np.ceil(kernel_width)) + 1), repeat=len(shape))))
    strides = jnp.asarray([int(np.prod(shape[i + 1:])) for i in range(len(shape))])
    normalizer = i0e(shape_parameter)

    def add_chunk(grid, chunk):
        coordinates, quadrature, signal = chunk
        location = coordinates * jnp.asarray(shape) + jnp.asarray(shape) // 2
        first = jnp.ceil(location - kernel_width / 2).astype(jnp.int32)
        def neighbour(grid, offset):
            index = first + offset
            displacement = 2 * (index - location) / kernel_width
            argument = shape_parameter * jnp.sqrt(jnp.maximum(0., 1 - displacement**2))
            kernel = jnp.prod(jnp.where(jnp.abs(displacement) <= 1,
                i0e(argument) * jnp.exp(argument - shape_parameter) / normalizer, 0), axis=-1)
            # Fourier interpolation is periodic. Discarding neighbours crossing
            # a Nyquist boundary introduces an artificial density taper there.
            wrapped = jnp.mod(index, jnp.asarray(shape))
            flat_index = jnp.sum(wrapped * strides, axis=-1)
            return grid.at[:, flat_index].add(signal * (kernel * quadrature)[None]), None
        return jax.lax.scan(neighbour, grid, offsets)[0], None

    grid = jnp.zeros((batch, int(np.prod(shape))), dtype=values.dtype)
    grid = jax.lax.scan(add_chunk, grid, (points.reshape((chunks, chunk_size, len(shape))),
        weights.reshape((chunks, chunk_size)), values.reshape((batch, chunks, chunk_size)).transpose(1, 0, 2)))[0]
    return grid.reshape(leading + shape)
