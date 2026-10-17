Custom models
=============

Use the provided template to add a custom forward model without touching the
built-in library:

1. Copy ``docs/model_template.py`` next to your code (e.g., ``my_models.py``).
2. Edit ``model_apply(fields, offsets, observables=None)`` to return a predicted
   data cube shaped like ``(frames, *spatial)``.
3. Edit ``parametrization_factory(spatial_shape)`` to return
   ``(params, unpack_fn)`` so the optimizer can map raw parameters to physical
   fields.
4. Fill in ``parameter_units`` for every field returned by ``unpack_fn``.

Optional helpers:

- ``required_observables``: declare extra maps that must be provided via
  ``experiment.insert_observables(...)`` (e.g., ``("t1_map",)``).
- ``regularizer_configs`` / ``reg_weights`` / ``regularizer_type``: define
  default regularizers (override in ``run_fit`` or study plans).
- Use ``estimate_param_scales`` + ``autoscale_regularizer_configs`` (in
  ``mrax.opt.regularizers``) to normalize per-parameter regularizer weights by
  parameter magnitude when scales differ widely.
- ``parameter_constant_dims``: collapse parameters along spatial axes when you
  know they are constant.

Once defined, construct the model from your module and pass it to ``run_fit`` or attach
it to an experiment/study.

Shared parameters and masks
---------------------------

Optionally set default ``regularizer_configs``/``regularizer_type``/``reg_weights``.
Observables can include timing, encoding, calibration, or sequence metadata using
``Quantity`` objects to track units.

To shrink the parameter space when a map is known to be constant along an axis, set
``parameter_constant_dims`` on your ``Model`` (e.g., ``{"amp": ("x",)}``). The fitter
optimizes a collapsed parameter array and broadcasts the result back to the full spatial
shape when returning fitted fields.

You can also provide a spatial ``mask`` (e.g., a 2D ``x,y`` array) to ``run_fit``.
Pixels outside the mask are ignored in the misfit and regularization terms, so
constant-dimension constraints only matter inside the masked region. Fitted maps are set
to ``NaN`` outside the mask, which makes downstream summaries and plots easier to filter.

See :doc:`custom_model_example` for a complete factory and fitting example.
