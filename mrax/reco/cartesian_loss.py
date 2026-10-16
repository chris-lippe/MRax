"""Bounded-memory Cartesian sensitivity-encoded data terms."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def cartesian_sense_loss(samples, weights, sensitivities, spatial_shape, *, homogeneous=False):
    """Construct the exact weighted loss, streaming receivers through its gradient.

    The custom JVP contracts the explicitly computed Fourier adjoint gradient;
    reverse-mode differentiation therefore retains an image-sized residual rather
    than a full stack of receiver images. Forward and higher-order derivatives
    remain available. Samples and sensitivities retain their existing layouts.
    """
    from .variational import fftnc, ifftnc

    shape = tuple(spatial_shape)
    ndim = len(shape)
    has_frames = samples.ndim == ndim + 2
    y = samples if has_frames else samples[None]
    coil_count = y.shape[1]
    total_weight = jnp.sum(weights) * (int(np.prod(samples.shape)) // int(np.prod(weights.shape)))
    image_shape = y.shape[:1] + shape
    axes = tuple(range(-ndim, 0))

    def evaluate(image, *, gradient):
        field = image if has_frames else image[None]
        if field.shape != image_shape:
            raise ValueError("Encoded image layout does not match measured samples")

        def receiver(coil, carry):
            value, derivative = carry
            sens = jnp.broadcast_to(sensitivities, y.shape)[:, coil]
            weight = jnp.broadcast_to(weights, samples.shape)
            weight = (weight if has_frames else weight[None])[:, coil]
            prediction = fftnc(field * sens, axes=axes)
            residual = prediction if homogeneous else prediction - y[:, coil]
            residual = jnp.where(weight > 0, residual, 0)
            value += .5 * jnp.sum(weight * jnp.abs(residual)**2)
            if gradient:
                contribution = jnp.conj(sens) * ifftnc(weight * residual, axes=axes)
                derivative += contribution if jnp.iscomplexobj(field) else jnp.real(contribution)
            return value, derivative

        value, derivative = jax.lax.fori_loop(0, coil_count, receiver,
            (jnp.asarray(0., dtype=jnp.real(field).dtype), jnp.zeros_like(field)))
        derivative = derivative if has_frames else derivative[0]
        return value / total_weight, derivative / total_weight

    @jax.custom_jvp
    def loss(image):
        return evaluate(image, gradient=False)[0]

    @loss.defjvp
    def loss_jvp(primals, tangents):
        (image,), (tangent,) = primals, tangents
        value, gradient = evaluate(image, gradient=True)
        return value, jnp.real(jnp.vdot(gradient, tangent))

    return loss
