"""Affine physical coordinates for convex map regularization."""
from __future__ import annotations

from dataclasses import replace
from math import prod

import jax
import jax.numpy as jnp
import numpy as np


def physical_parametrization(model, raw_params, raw_unpack):
    specs = model.parameter_specs or getattr(model.parametrization_factory, "parameter_specs", None)
    if specs is None:
        raise ValueError("True TV proximal fitting requires Model.parameter_specs or a factory with parameter_specs")
    fields = raw_unpack(raw_params)
    if set(fields) != set(specs):
        raise ValueError("Physical parameter specs must describe every unpacked field")
    scales, bounds, params, fixed = {}, {}, {}, {}
    for key, field in fields.items():
        cfg = specs[key]
        if not cfg.get("trainable", True):
            fixed[key] = field
            continue
        field = jnp.asarray(field)
        if jnp.iscomplexobj(field):
            raise ValueError("Physical TV fitting requires real parameter maps")
        transform = cfg.get("transform", "softplus")
        lower, upper = -np.inf, np.inf
        if transform == "bounded":
            lower, upper = cfg["bounds"]
            base = .5 * (upper - lower)
        elif transform == "tanh":
            base = abs(cfg.get("scale", 1.0))
            lower, upper = -base, base
        elif transform == "softplus":
            offset, scale = cfg.get("offset", 0.0), cfg.get("scale", 1.0)
            if scale > 0:
                lower = offset
            elif scale < 0:
                upper = offset
            else:
                lower = upper = offset
            base = max(float(np.max(np.abs(np.asarray(field) - offset))), abs(scale))
        elif transform in ("identity", "linear"):
            base = max(float(np.max(np.abs(np.asarray(field)))), abs(cfg.get("scale", 1.0)))
        else:
            raise ValueError(f"Unsupported physical parameter transform '{transform}'")
        if any(np.asarray(value).ndim != 0 for value in (base, lower, upper)):
            raise ValueError("Physical proximal fitting requires scalar transform bounds and scales")
        base = cfg.get("optimization_scale", max(float(base), 1e-6))
        if not np.isfinite(base) or base <= 0 or lower > upper:
            raise ValueError("Physical parameter scales must be positive and bounds ordered")
        scale = float(base) * prod(field.shape)**.5
        scales[key] = scale
        bounds[key] = (lower / scale, upper / scale)
        params[key] = field / scale

    def unpack(values):
        return {**fixed, **{key: value * scales[key] for key, value in values.items()}}

    def from_raw(values):
        physical = raw_unpack(values)
        return {key: physical[key] / scale for key, scale in scales.items()}

    unpack.__dict__.update(raw_unpack.__dict__)
    unpack._mrax_physical_scales = scales
    unpack._mrax_physical_bounds = bounds
    unpack._mrax_from_raw = from_raw
    return params, unpack


def balance_physical_coordinates(smooth_loss, params, unpack, operators, *, stored_scales=None, balance=True, metric="curvature"):
    """Choose a fixed block metric from initial smooth curvature, then backtrack.

    The metric stays fixed throughout fitting. It rescales physical coordinates
    and the TV weights together, preserving the objective and proximal geometry.
    """
    @jax.jit
    def curvature(point, direction):
        return jax.jvp(jax.grad(smooth_loss), (point,), (direction,))[1]

    factors = {}
    if metric not in ("curvature", "initial_gradient"):
        raise ValueError("Unknown fixed physical coordinate metric")
    if stored_scales is not None:
        if set(stored_scales) != set(params) or any(not np.isfinite(value) or value <= 0 for value in stored_scales.values()):
            raise ValueError("Stored physical coordinate scales do not match this model")
        factors = {key: stored_scales[key] / unpack._mrax_physical_scales[key] for key in params}
    elif not balance:
        factors = {key: 1.0 for key in params}
    if not factors and metric == "initial_gradient":
        gradient = jax.grad(smooth_loss)(params)
        # A fixed block metric with an initial RMS displacement equal to that
        # of a midpoint bounded transform. Select it once, then use the
        # certified TV prox in these fixed coordinates.
        for key, value in params.items():
            current = unpack._mrax_physical_scales[key]
            physical_rms = float(jnp.sqrt(jnp.mean(jnp.abs(gradient[key] / current)**2)))
            displacement = current / (2 * prod(value.shape)**.5)
            scale = (displacement / max(physical_rms, 1e-20))**.5
            factors[key] = min(max(scale / current, 1e-3), 1e3)
    for key, value in (() if factors else params.items()):
        parity = jnp.zeros(value.shape, dtype=jnp.int32)
        for axis, size in enumerate(value.shape):
            shape = [1] * value.ndim
            shape[axis] = size
            parity = parity + jnp.arange(size).reshape(shape)
        direction = {name: jnp.zeros_like(field) for name, field in params.items()}
        direction[key] = (1 - 2 * (parity % 2)).astype(value.dtype) / prod(value.shape)**.5
        block = curvature(params, direction)[key]
        magnitude = float(jnp.sqrt(jnp.sum(block**2)))
        factor = 1 / max(magnitude, 1e-6)**.5 if np.isfinite(magnitude) else 1.0
        factors[key] = min(max(factor, 1e-3), 1e3)
    # Mutate only the shared affine-coordinate metadata, before loss compilation.
    for key, factor in factors.items():
        unpack._mrax_physical_scales[key] *= factor
        lower, upper = unpack._mrax_physical_bounds[key]
        unpack._mrax_physical_bounds[key] = (lower / factor, upper / factor)
        operators[key] = tuple(replace(op, weights=op.weights * factor) for op in operators[key])
    return {key: value / factors[key] for key, value in params.items()}
