.. _api-optim:

Optimizers and regularizers
===========================

Core primitives for solvers and penalties. All regularizer functions operate on ndarrays
and return scalar costs.

- ``make_optimizer(optimizer_name, step_size, regularizer_weight, regularizer_prox, box_constraints)``
  resolves Optax optimizers plus proximal ``split_bregman``/``fista`` variants.
- ``spatial_smoothness_penalty`` and ``total_variation_penalty`` operate along configurable axes.
  ``higher_order_total_variation_penalty`` adds finite-difference TV of configurable order.
- Low-level ops like ``soft_threshold``, ``isotropic_shrinkage``, gradient/divergence
  helpers, and ``conjugate_gradient`` are available for custom solvers.

.. autofunction:: mrax.opt.optimizers.make_optimizer

.. autofunction:: mrax.opt.regularizers.spatial_smoothness_penalty
.. autofunction:: mrax.opt.regularizers.total_variation_penalty
.. autofunction:: mrax.opt.regularizers.higher_order_total_variation_penalty
.. autofunction:: mrax.opt.regularizers.make_regularizer_fn
.. autofunction:: mrax.opt.regularizers.estimate_field_scales
.. autofunction:: mrax.opt.regularizers.estimate_param_scales
.. autofunction:: mrax.opt.regularizers.autoscale_regularizer_configs

.. autofunction:: mrax.opt.ops.soft_threshold
.. autofunction:: mrax.opt.ops.isotropic_shrinkage
.. autofunction:: mrax.opt.ops.forward_gradient2d
.. autofunction:: mrax.opt.ops.divergence2d
.. autofunction:: mrax.opt.ops.conjugate_gradient
