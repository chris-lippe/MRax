"""Explicit Cartesian spectral windows for reconstruction initialization/output."""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np


def spectral_window(shape, kinds, parameters=(), powers=()):
    """Return separable windows on the centered encoding grid.

    Sine-bell parameters below two specify a full lobe; two specifies a cosine
    half-lobe; larger values begin at phase pi/parameter and end at pi. Windows
    are periodic-grid samples with t=n/N, so a full sine lobe peaks at DC.
    """
    shape, kinds = tuple(shape), tuple(kinds)
    if len(shape) != len(kinds) or parameters and len(parameters) != len(shape) or powers and len(powers) != len(shape):
        raise ValueError("Spectral windows must describe every encoding axis")
    parameters = tuple(parameters) or (0.,) * len(shape)
    powers = tuple(powers) or (1,) * len(shape)
    window = jnp.ones(shape)
    for axis, (size, kind, parameter) in enumerate(zip(shape, kinds, parameters, strict=True)):
        if not np.isfinite(parameter) or parameter < 0:
            raise ValueError("Window parameters must be finite and nonnegative")
        t = jnp.arange(size) / size
        if kind == "none":
            values = jnp.ones(size)
        elif kind in {"sine", "sine_squared"}:
            fraction = 0. if parameter < 2 else 1 / parameter
            values = jnp.sin(jnp.pi * ((1 - fraction) * t + fraction))
            if kind == "sine_squared":
                values = values**2
        elif kind == "hann":
            values = .5 - .5 * jnp.cos(2 * jnp.pi * t)
        elif kind == "hamming":
            values = .54 - .46 * jnp.cos(2 * jnp.pi * t)
        else:
            raise ValueError(f"Unknown spectral window: {kind}")
        dimensions = [1] * len(shape)
        dimensions[axis] = size
        if not isinstance(powers[axis], (int, np.integer)) or powers[axis] < 1:
            raise ValueError("Window powers must be positive integers")
        window *= values.reshape(dimensions) ** powers[axis]
    return window
