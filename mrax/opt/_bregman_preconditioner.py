"""Deterministic two-level SPD preconditioning for the primal CG solve.

Uniform field modes form a small coupled coarse space. Cosine probes estimate
separable curvature for each mean-free field. This approximates an inverse
Hessian only inside CG: the Hessian products and objective remain unchanged.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.fft import dct, idct


def _metadata(point):
    modes = []
    for key in sorted(point):
        value = point[key]
        if value.size:
            modes.append((key, False))
            if jnp.iscomplexobj(value):
                modes.append((key, True))
    probes = [(i, -1) for i in range(len(modes))]
    probes += [(i, axis) for i, (key, _) in enumerate(modes)
               for axis, size in enumerate(point[key].shape) if size > 1]
    return modes, probes


def _coordinates(value, modes):
    parts = []
    for key, imaginary in modes:
        mean = jnp.sum(value[key]) / jnp.sqrt(float(value[key].size))
        parts.append(jnp.imag(mean) if imaginary else jnp.real(mean))
    return jnp.stack(parts) if parts else jnp.zeros((0,), jnp.float32)


def initial_preconditioner(point):
    modes, _ = _metadata(point)
    dtype = jnp.result_type(*[jnp.real(x).dtype for x in point.values()], jnp.float32)
    return dict(coefficients={key: jnp.concatenate((jnp.ones(1, dtype),
                      jnp.zeros(value.ndim, dtype))) for key, value in point.items()},
                coarse_inverse=jnp.eye(len(modes), dtype=dtype), ready=jnp.asarray(False),
                local_scale=jax.tree.map(lambda x: jnp.ones(x.shape, jnp.real(x).dtype), point),
                coarse_shift=jnp.ones((len(modes),), dtype),
                gradient_reference=jnp.asarray(0., dtype),
                valid=jnp.asarray(True), builds=jnp.asarray(0, jnp.int32))


def build_preconditioner(point, product, previous):
    modes, probes = _metadata(point)
    if not modes:
        return {**previous, "ready": jnp.asarray(True)}
    keys = tuple(sorted(point))
    dtype = previous["coarse_inverse"].dtype
    mode_keys = jnp.asarray([keys.index(key) for key, _ in modes], jnp.int32)
    phases = jnp.asarray([imaginary for _, imaginary in modes])
    probe_modes = jnp.asarray([mode for mode, _ in probes], jnp.int32)
    probe_axes = jnp.asarray([axis for _, axis in probes], jnp.int32)

    def probe(mode, axis):
        result = {}
        for index, key in enumerate(keys):
            x = point[key]
            if not x.size:
                result[key] = jnp.zeros_like(x)
                continue
            value = jnp.ones_like(x) / jnp.sqrt(jnp.asarray(x.size, jnp.real(x).dtype))
            if jnp.iscomplexobj(x):
                value = value * jnp.where(phases[mode], jnp.asarray(1j, x.dtype),
                                         jnp.asarray(1., x.dtype))
            for dimension, size in enumerate(x.shape):
                if size > 1:
                    shape = [1] * x.ndim
                    shape[dimension] = size
                    positions = jnp.arange(size, dtype=jnp.real(x).dtype)
                    wave = jnp.sqrt(2.) * jnp.cos(jnp.pi * (size - 1) * (positions + .5) / size)
                    value = value * jnp.where(axis == dimension, wave.reshape(shape), 1.)
            result[key] = jnp.where(mode_keys[mode] == index, value, 0.).astype(x.dtype)
        return result

    # Matrix-free uniform-field products estimate local diagonal curvature. Symmetric equilibration prevents an isolated stiff location from
    # controlling every mean-free field mode. This is a numerical coordinate
    # change confined to the inverse approximation.
    zero = {key: jnp.zeros(x.shape, jnp.real(x).dtype) for key, x in point.items()}

    def diagonal_body(mode, diagonal):
        vector = probe(mode, jnp.asarray(-1, jnp.int32))
        image = product(vector)
        updated = {}
        for which, key in enumerate(keys):
            value = image[key]
            component = (jnp.where(phases[mode], jnp.imag(value), jnp.real(value))
                         if jnp.iscomplexobj(value) else value)
            curvature = jnp.abs(component) * jnp.sqrt(float(value.size))
            updated[key] = jnp.where(mode_keys[mode] == which,
                jnp.maximum(diagonal[key], curvature), diagonal[key])
        return updated

    diagonal = jax.lax.fori_loop(0, len(modes), diagonal_body, zero)
    local_scale = {}
    for key, values in diagonal.items():
        positive = values > 0
        tiny = jnp.finfo(values.dtype).tiny
        average_log = jnp.sum(jnp.where(positive, jnp.log(jnp.maximum(values, tiny)), 0.))
        geometric = jnp.exp(average_log / jnp.maximum(jnp.sum(positive), 1))
        floor = jnp.maximum(32 * jnp.finfo(values.dtype).eps * geometric, 32 * tiny)
        local_scale[key] = jnp.where(positive, jax.lax.rsqrt(jnp.maximum(values, floor)), 1.)

    def equilibrated_product(value):
        scaled = jax.tree.map(lambda x, s: x * s, value, local_scale)
        return jax.tree.map(lambda x, s: x * s, product(scaled), local_scale)

    def body(index, carry):
        coarse, coefficients = carry
        mode, axis = probe_modes[index], probe_axes[index]
        vector = probe(mode, axis)
        image = equilibrated_product(vector)
        curvature = sum(jnp.real(jnp.vdot(vector[key], image[key])) for key in keys)
        coarse = jax.lax.cond(axis < 0,
            lambda value: value.at[:, mode].set(_coordinates(image, modes)), lambda value: value, coarse)
        updated = {}
        for which, key in enumerate(keys):
            field = coefficients[key]
            active = mode_keys[mode] == which
            base = jnp.where(active & (axis < 0), jnp.maximum(field[0], jnp.abs(curvature)), field[0])
            field = field.at[0].set(base)
            for dimension, size in enumerate(point[key].shape):
                if size > 1:
                    eigenvalue = 4 * jnp.sin(jnp.pi * (size - 1) / (2 * size)) ** 2
                    estimate = jnp.maximum((curvature - base) / eigenvalue, 0.)
                    field = field.at[dimension + 1].set(jnp.where(active & (axis == dimension),
                        jnp.maximum(field[dimension + 1], estimate), field[dimension + 1]))
            updated[key] = field
        return coarse, updated

    zero = {key: jnp.zeros_like(value) for key, value in previous["coefficients"].items()}
    coarse, coefficients = jax.lax.fori_loop(0, len(probes), body,
        (jnp.zeros_like(previous["coarse_inverse"]), zero))
    finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(x))
                               for x in jax.tree.leaves((coarse, coefficients, local_scale))]))
    coarse = jnp.where(finite, .5 * (coarse + coarse.T), jnp.eye(len(modes), dtype=dtype))
    values, vectors = jnp.linalg.eigh(coarse)
    epsilon = max([jnp.finfo(dtype).eps] +
                  [jnp.finfo(jnp.real(x).dtype).eps for x in point.values()])
    floor = jnp.maximum(32 * epsilon * jnp.max(jnp.abs(values)), 32 * jnp.finfo(dtype).tiny)
    inverse = (vectors / jnp.maximum(jnp.abs(values), floor)[None, :]) @ vectors.T
    coefficients = {key: value.at[0].set(jnp.maximum(value[0], floor))
                    for key, value in coefficients.items()}
    finite = finite & jnp.all(jnp.isfinite(inverse)) & jnp.all(jnp.stack([
        jnp.all(jnp.isfinite(x)) for x in coefficients.values()]))
    fallback = initial_preconditioner(point)
    return dict(coefficients=jax.tree.map(lambda x, y: jnp.where(finite, x, y),
                                           coefficients, fallback["coefficients"]),
                coarse_inverse=jnp.where(finite, inverse, fallback["coarse_inverse"]),
                local_scale=jax.tree.map(lambda x, y: jnp.where(finite, x, y),
                                         local_scale, fallback["local_scale"]),
                coarse_shift=jnp.where(finite,
                    jnp.stack([jnp.mean(local_scale[key] ** 2) for key, _ in modes]), fallback["coarse_shift"]),
                gradient_reference=previous["gradient_reference"],
                ready=jnp.asarray(True), valid=finite,
                builds=previous["builds"] + 1)


def apply_preconditioner(value, state):
    original = value
    value = jax.tree.map(lambda x, s: x * s, value, state["local_scale"])
    modes, _ = _metadata(value)
    coarse = state["coarse_inverse"] @ _coordinates(value, modes)
    result = {}
    for key, x in value.items():
        if not x.size:
            result[key] = x
            continue
        coefficients = state["coefficients"][key].astype(jnp.real(x).dtype)
        transformed = x - jnp.mean(x)
        denominator = jnp.full(x.shape, coefficients[0], jnp.real(x).dtype)
        for axis, size in enumerate(x.shape):
            if size > 1:
                if jnp.iscomplexobj(x):
                    transformed = dct(jnp.real(transformed), axis=axis, norm="ortho") + 1j * dct(
                        jnp.imag(transformed), axis=axis, norm="ortho")
                else:
                    transformed = dct(transformed, axis=axis, norm="ortho")
                shape = [1] * x.ndim
                shape[axis] = size
                frequency = jnp.arange(size, dtype=jnp.real(x).dtype)
                eigenvalue = 4 * jnp.sin(jnp.pi * frequency / (2 * size)) ** 2
                denominator = denominator + coefficients[axis + 1] * eigenvalue.reshape(shape)
        transformed = transformed / denominator
        for axis in reversed(range(x.ndim)):
            if x.shape[axis] > 1:
                if jnp.iscomplexobj(x):
                    transformed = idct(jnp.real(transformed), axis=axis, norm="ortho") + 1j * idct(
                        jnp.imag(transformed), axis=axis, norm="ortho")
                else:
                    transformed = idct(transformed, axis=axis, norm="ortho")
        correction = sum((coarse[i] * (1j if imaginary else 1.) / jnp.sqrt(float(x.size))
                          for i, (name, imaginary) in enumerate(modes) if name == key),
                         jnp.asarray(0., x.dtype))
        result[key] = (transformed - jnp.mean(transformed) + correction).astype(x.dtype)
    result = jax.tree.map(lambda x, s: x * s, result, state["local_scale"])
    return jax.tree.map(lambda preconditioned, raw: jnp.where(state["valid"], preconditioned, raw),
                        result, original)


def shifted_preconditioner(state, shift):
    """Apply the same positive Newton shift to the inverse approximation."""
    values, vectors = jnp.linalg.eigh(state["coarse_inverse"])
    tiny = jnp.finfo(values.dtype).tiny
    precision = (vectors / jnp.maximum(values, tiny)[None, :]) @ vectors.T
    precision = precision + jnp.diag(shift * state["coarse_shift"])
    values, vectors = jnp.linalg.eigh(.5 * (precision + precision.T))
    inverse = (vectors / jnp.maximum(values, tiny)[None, :]) @ vectors.T
    return {**state, "coarse_inverse": inverse,
            "coefficients": {key: x.at[0].add((shift * jnp.sum(state["local_scale"][key] ** 2) /
                                             max(1, state["local_scale"][key].size)).astype(x.dtype))
                             for key, x in state["coefficients"].items()}}
