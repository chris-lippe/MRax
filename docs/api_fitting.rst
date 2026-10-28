.. _api-fitting:

Fitting
=======

Build differentiable losses, fit individual experiments and execute study plans.
See :doc:`model_fitting` for the workflow and :ref:`api-optim` for solver controls.

Objective and inputs
--------------------

``build_loss`` returns the scalar loss, initial parameter tree and field unpacker.
Weighted least squares is the default. Masks combine spatial confidence, frames
and observable support; ``mask_dims`` specifies their axis order.
Each fit requires usable finite observations.

``data_normalization="roi"`` computes RMS over retained finite observations;
``"full"`` uses all finite image data. To preserve regularization balance when
switching from full-image to ROI normalization, multiply each regularizer weight
by ``(RMS_full / RMS_roi)**2``. Double-precision fits use
``JAX_ENABLE_X64=true`` with float64 data, observables and initial parameters.

``reference_image`` fits signal intensities as the fixed spatial reference
times the relative model prediction. Returned predictions include that scale,
and measurement uncertainty in the reference remains relevant. Signal-domain
and Z-spectrum weighting define distinct statistical objectives.

``data_variance`` specifies observation variances, with uniform variance as the
default and ``"auto"`` for robust estimation before fitting. The optional
``variance_estimation_mask`` identifies a noise region separately from the
fitting mask. See :doc:`variance_weighting`.

A custom differentiable ``data_loss(residual, weights)`` receives RMS-normalized
residuals and the combined mask/precision weights and returns a real scalar.
All optimizer interfaces use that callable. Grouped pseudo-Huber loss applies
one bounded influence to the RMS of consecutive frames at each spatial position;
``delta`` is in normalized residual units. Calibrate it from the training noise
for the intended statistical objective.

.. autofunction:: mrax.core.loss_builder.build_loss

.. autofunction:: mrax.core.loss_builder.make_model_callable

.. autofunction:: mrax.core.data_losses.pseudo_huber_loss

Execution and continuation
--------------------------

``run_fit`` returns fitted fields with units and a loss history.
``return_model_callable=True`` adds the prediction callable;
``return_fit_info=True`` adds ``FitInfo`` after it. Study globals and
observables are merged by ``make_model_callable``.

``step_size`` accepts a constant or Optax schedule. ``chunk_size`` sets the
interval between host progress and finiteness checks; a progress callback may
cancel by raising an exception. Early stopping uses ``convergence_tolerance``,
``parameter_tolerance``, ``convergence_patience`` and ``min_steps``, with
``num_steps`` as the maximum.

History entries precede updates. ``info.final_loss`` evaluates the returned
fields; ``info`` also records completed updates, termination and the complete
checkpoint. Continue with ``initial_state=info.state`` and matching model,
data, regularizers and optimizer. ``initial_params`` instead starts a fresh
optimizer. Solver-family and state-layout changes require fresh optimizer state.

``FitError`` reports nonfinite updates and failed proximal/backtracking
certificates. Its ``last_valid_state`` and ``loss_history`` preserve the latest
verified checkpoint when available.

.. autofunction:: mrax.core.loss_builder.run_fit

Study plans and diagnostics
---------------------------

Study plans order experiments by their observable dependencies and may reuse
precomputed fields. Shared options use ``run_fit_kwargs``; per-experiment
options use ``run_fit_overrides``. By default, experiments requiring additional
observables receive ``skipped=True``.

.. autofunction:: mrax.core.loss_builder.build_study_losses

.. autofunction:: mrax.core.study_runner.plan_study_fits

.. autofunction:: mrax.core.study_runner.execute_study_plan

.. autofunction:: mrax.core.fit_sanity.check_fit_sanity

.. autoclass:: mrax.core.fit_sanity.FitSanityReport

.. autofunction:: mrax.core.variance.estimate_data_variance

.. autoclass:: mrax.core.variance.VarianceEstimate
   :members:
