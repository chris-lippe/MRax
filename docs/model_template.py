"""
Template for defining a custom Model.

Copy this file, rename it, and adapt the pieces you need. The only hard requirements
are:
  - implement `model_apply(fields, model_axis, observables)` returning a predicted cube.
  - implement `parametrization_factory(spatial_shape)` returning (params, unpack_fn).
  - provide `parameter_units` for every unpacked field.

Optional: declare default regularizers via `regularizer_configs` / `reg_weights` /
`regularizer_type` (overridable in run_fit or when executing a planned study).
Optional: set `parameter_constant_dims` to collapse parameters known to be constant
along certain spatial dimensions (e.g., {"amplitude": ("x",)}).
Optional: set `required_observables` (e.g., ("baseline_map",)) so the study planner
can pull upstream experiments that provide them.
Optional: pass a spatial `mask` to `run_fit` to ignore pixels
outside the mask when computing the loss/regularization.
"""

from __future__ import annotations

import jax.nn as jnn
import jax.numpy as jnp

from mrax.data_struct.model import Model


def make_custom_model() -> Model:
    # Example: exponential response with a bias term
    def model_apply(fields, model_axis, observables=None):
        model_axis = model_axis.reshape((-1, 1, 1))
        amp = fields["amplitude"][None, ...]
        tau = fields["tau"][None, ...]
        bias = fields["bias"][None, ...]
        return bias + amp * jnp.exp(-model_axis / tau)

    def parametrization_factory(spatial_shape):
        params = {
            "amplitude_raw": jnp.zeros(spatial_shape),
            "tau_raw": jnp.zeros(spatial_shape),
            "bias_raw": jnp.zeros(spatial_shape),
        }

        def unpack(p):
            return {
                "amplitude": 0.1 + 2.0 * jnn.softplus(p["amplitude_raw"]),
                "tau": 1e-3 + 5.0 * jnn.softplus(p["tau_raw"]),
                "bias": p["bias_raw"],
            }

        return params, unpack

    return Model(
        name="custom_exponential",
        model_apply=model_apply,
        parametrization_factory=parametrization_factory,
        parameter_units={
            "amplitude": "1",
            "tau": "a.u.",
            "bias": "1",
        },
        # parameter_constant_dims={"amplitude": ("x",)},  # Example: shared across rows
        regularizer_configs={
            "amplitude": [{"type": "smoothness", "weight": 1e-3}],
            "tau": [{"type": "tv", "axes": (0, 1), "weight": 5e-4}],
        },
    )


__all__ = ["make_custom_model"]
