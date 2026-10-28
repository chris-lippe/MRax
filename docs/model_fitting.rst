Fitting controls
================

Use :func:`mrax.core.loss_builder.run_fit` for one experiment, or
:doc:`studies` for a sequence of dependent fits. The same fitting interface
supports Optax, FISTA and split Bregman.

Choose an optimizer
-------------------

.. list-table::
   :header-rows: 1
   :widths: 1 3

   * - Optimizer
     - Typical use
   * - Optax, such as Adam
     - Differentiable objectives and smoothed regularization.
   * - FISTA
     - Accelerated proximal-gradient fitting with physical bounds and native TV.
   * - Split Bregman
     - Augmented-primal, shrinkage and multiplier updates for native TV.

Automatic selection uses FISTA for enabled TV terms and Adam otherwise.
Explicit Adam uses smoothed TV and emits a warning. FISTA and split Bregman
use native TV with physical field bounds; temporal smoothness
remains a squared-difference penalty. Model parameter specifications define
bounds, constant dimensions and regularizer weights.

.. code-block:: python

   from mrax.core import run_fit

   fields, history, info = run_fit(
       experiment, model, dataset,
       optimizer_name="fista",
       step_size=0.02,
       num_steps=200,
       return_fit_info=True,
   )
   print(info.optimizer_name, info.termination_reason, info.final_loss)

Prepare the objective
---------------------

Check the model's units, observables, parameter bounds and initialization.
Use Boolean masks for equal-weight regions or numeric masks for confidence
weights. ROI normalization uses the retained finite observations; full-image
normalization uses all finite image data. Adjust regularizer weights consistently
when changing normalization. See :doc:`variance_weighting` for noise weighting.

Weighted least squares is the default data term. The optional ``data_loss``
callable supports alternatives such as grouped pseudo-Huber loss.
:ref:`api-fitting` documents normalization, masks, reference-image fitting and
custom losses.

Run and inspect
----------------

Fits default to CPU. Select a local device with ``device``; use
``mrax.get_available_devices()`` to list available devices. GPU setup is covered
in :doc:`docker_setup` and :doc:`windows_setup`.

Inspect the loss history, fitted fields and ``FitInfo`` termination reason.
For nonlinear models, assess stationarity and sensitivity to initialization.
Early-stopping tolerances, schedules and host-check intervals are described in
:ref:`api-fitting`.

Continue a fit with ``initial_state=info.state`` and the same model, data,
regularizers and optimizer. Use ``initial_params`` for a fresh optimizer.

Further details
----------------

- :ref:`api-optim`: algorithm options, proximal accuracy, direct Optax loops and
  checkpoint compatibility.
- :ref:`api-fitting`: complete fitting arguments and return values.
- :doc:`model_parameters`: parameter coordinates and model specifications.
- :doc:`references`: publications underlying the models and optimizers.
