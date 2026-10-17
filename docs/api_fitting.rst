.. _api-fitting:

Fitting
=======

Build and run differentiable losses with Optax/Proximal variants. All functions accept
per-parameter regularizer configs/weights that override model defaults.

- ``build_loss(experiment, model, dataset, ...)`` → ``(loss_fn, init_params, unpack)``
  where ``loss_fn(params)`` returns a scalar data+regularization loss. Pass known
  observation variances with ``data_variance`` for precision weighting; uniform
  variance is the default, ``data_variance="auto"`` estimates robust per-frame
  variances before fitting, and the RMS loss normalization is retained.
- ``run_fit(experiment, model, dataset, optimizer_name="adam", step_size=0.02, num_steps=200, ...)``
  fits a single experiment; returns ``(fitted_fields_with_units, loss_history)`` or
  ``(fitted_fields_with_units, loss_history, model_fn)`` when
  ``return_model_callable=True`` (``model_fn(fields, offsets=None, observables=None)`` applies
  study globals + observables).
  Supports Optax optimizers plus ``split_bregman``/``fista`` wrappers, box constraints,
  proximal regularizers, and explicit initial parameters.
- ``make_model_callable(experiment, model, study_globals=None)`` builds a callable that
  merges study globals/observables before invoking ``model.model_apply``. Use it in
  notebooks to evaluate predictions without manually attaching ``field_strength_mhz``.
- ``plan_study_fits(study, target_names=None, precomputed_fields=None)`` returns an
  ordered list of :class:`mrax.core.study_runner.PlannedFit` entries for sequential
  fitting, optionally reusing cached fields.
- ``execute_study_plan(plan, run_fit_kwargs=None, run_fit_overrides=None, ...)`` executes
  the plan via :func:`run_fit` and returns result entries plus the
  fitted field cache. Experiments missing required observables are skipped by default
  and marked with ``skipped=True`` in the results. Put shared ``run_fit`` options in
  ``run_fit_kwargs`` and per-experiment options in ``run_fit_overrides``; dict-valued
  ``run_fit`` options such as ``regularizer_weights`` are passed through unchanged.
- ``check_fit_sanity(experiment, model, dataset, study_globals=None)`` runs a fast
  pre-fit check and returns a :class:`mrax.core.fit_sanity.FitSanityReport`.
- ``estimate_data_variance(experiment, dataset, mask=None, ...)`` estimates robust
  per-frame observation variances from background noise, with a spatial
  wavelet-MAD fallback, and returns diagnostics in a
  :class:`mrax.core.variance.VarianceEstimate`. The optional mask must identify a
  noise-only region; fitting masks are separate.

.. autofunction:: mrax.core.loss_builder.build_loss
.. autofunction:: mrax.core.loss_builder.run_fit
.. autofunction:: mrax.core.loss_builder.make_model_callable
.. autofunction:: mrax.core.variance.estimate_data_variance
.. autoclass:: mrax.core.variance.VarianceEstimate
   :members:
.. autofunction:: mrax.core.loss_builder.build_study_losses
.. autofunction:: mrax.core.study_runner.plan_study_fits
.. autofunction:: mrax.core.study_runner.execute_study_plan
.. autofunction:: mrax.core.fit_sanity.check_fit_sanity
.. autoclass:: mrax.core.fit_sanity.FitSanityReport
