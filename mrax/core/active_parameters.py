"""Optional active-ROI coordinates with original-grid TV topology.

Only parameter/state buffers are packed. Arbitrary model forward evaluations
retain their original geometry (including nonlocal models); this does not claim
to skip all forward-model arithmetic outside an ROI.
"""
from dataclasses import dataclass
from hashlib import sha256
from math import comb

import jax.numpy as jnp
import numpy as np

from mrax.opt.tv import vector_norm


@dataclass(frozen=True)
class PackedTVOperator:
    axes: tuple
    order: int
    weights: object
    indices: tuple
    coefficients: tuple
    size: int
    norm_bound_squared: float

    def gradient(self, field):
        return jnp.stack([jnp.diff(field[index], n=self.order, axis=-1)[..., 0]
                          * jnp.any(coefficient != 0, axis=-1)
                          for index, coefficient in zip(self.indices, self.coefficients, strict=True)], axis=-1)

    def adjoint(self, dual):
        result = jnp.zeros((self.size,), dual.dtype)
        for axis, (index, coefficient) in enumerate(zip(self.indices, self.coefficients, strict=True)):
            result = result.at[index.reshape(-1)].add((dual[..., axis, None] * coefficient).reshape(-1))
        return result

    def value(self, field, smoothed=False):
        gradient = self.gradient(field)
        norm = (jnp.sqrt(jnp.sum(jnp.abs(gradient)**2, axis=-1) + 1e-12) - 1e-6
                if smoothed else vector_norm(gradient))
        return jnp.sum(self.weights * norm)

    def project_dual(self, dual, multiplier=1.):
        norm = vector_norm(dual)
        return dual * jnp.minimum(1., multiplier * self.weights / jnp.where(norm > 0, norm, 1))[..., None]


def _pack_operator(op, active, shape):
    mapping = np.full(np.prod(shape), -1, np.int32)
    mapping[active] = np.arange(len(active))
    weights = np.asarray(op.weights)
    anchors = np.flatnonzero(weights.reshape(-1) > 0)
    indices, coefficients = [], []
    for axis, valid in zip(op.axes, op.valid, strict=True):
        valid = np.asarray(valid).reshape(-1)[anchors]
        stride = int(np.prod(shape[axis + 1:]))
        stencil = anchors[:, None] + stride * np.arange(op.order + 1)[None, :]
        stencil = np.where(valid[:, None], stencil, active[0])
        index = mapping[stencil]
        if np.any(index < 0):
            raise ValueError("Active TV stencil reaches an inactive coordinate")
        coefficient = np.array([(-1)**(op.order-i) * comb(op.order, i) for i in range(op.order+1)])
        indices.append(jnp.asarray(index))
        coefficients.append(jnp.asarray(valid[:, None] * coefficient, op.weights.dtype))
    return PackedTVOperator(op.axes, op.order, jnp.asarray(weights.reshape(-1)[anchors]),
                            tuple(indices), tuple(coefficients), len(active), op.norm_bound_squared)


def pack_problem(loss, params, unpack, support):
    """Pack every nonconstant leaf; collapse support along shared dimensions."""
    defaults, indices, packed = {}, {}, {}
    digest = sha256()
    support = np.asarray(support) > 0
    for key, value in params.items():
        shape = tuple(value.shape)
        if len(shape) != support.ndim:
            raise ValueError("Active packing requires field-shaped parameter leaves")
        axes = tuple(i for i, (n, m) in enumerate(zip(shape, support.shape, strict=True)) if n == 1 and m != 1)
        selected = np.any(support, axis=axes, keepdims=True) if axes else support
        selected = np.broadcast_to(selected, shape)
        active = np.flatnonzero(selected.reshape(-1))
        if len(active) == 0:
            raise ValueError("A fitted parameter has no active coordinates")
        defaults[key], indices[key] = value, jnp.asarray(active)
        packed[key] = value.reshape(-1)[active]
        digest.update(key.encode())
        digest.update(str(shape).encode())
        digest.update(active.tobytes())

    def expand(p):
        return {key: value.reshape(-1).at[indices[key]].set(p[key]).reshape(value.shape)
                for key, value in defaults.items()}

    def packed_unpack(p):
        return unpack(expand(p))
    packed_unpack.__dict__.update(unpack.__dict__)
    if hasattr(unpack, '_mrax_smooth_loss'):
        packed_unpack._mrax_smooth_loss = lambda p: unpack._mrax_smooth_loss(expand(p))
    if hasattr(unpack, '_mrax_tv_operators'):
        packed_unpack._mrax_tv_operators = {key: tuple(_pack_operator(op, np.asarray(indices[key]), defaults[key].shape)
            for op in operators) for key, operators in unpack._mrax_tv_operators.items()}
    packed_unpack._mrax_active_layout = digest.hexdigest()
    packed_unpack._mrax_expand_active = expand
    packed_unpack._mrax_pack_active = lambda p: {key: value.reshape(-1)[indices[key]] for key, value in p.items()}
    return lambda p: loss(expand(p)), packed, packed_unpack
