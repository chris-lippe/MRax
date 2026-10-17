Fitting controls
================

Optimization defaults to ``device="cpu"``. Choose ``"gpu"``, ``"gpu:0"`` or a
local JAX device; unavailable selections raise an error.
``mrax.get_available_devices()`` lists local devices, CPU first; pass ``"gpu"``
to list only GPUs. Fitting, reconstruction, denoising, drift correction and
batch entry points accept ``device``. Per-experiment study-plan overrides take
precedence over the plan default.

Selecting ``device="gpu"`` requires a Linux environment with GPU-enabled JAX.
On Windows, run MRax inside WSL 2; native Windows installations are CPU-only due
to JAX's platform restrictions. See :doc:`windows_setup` and :doc:`docker_setup`.

.. code-block:: python

   from mrax import get_available_devices
   from mrax.core import run_fit

   print(get_available_devices())
   fields, history, info = run_fit(
       experiment, model, dataset, device="gpu", return_fit_info=True,
   )

Normalization and masks
~~~~~~~~~~~~~~~~~~~~~~~

``data_normalization="roi"`` computes RMS over finite observations retained by
ROI, frame and observable masks. ``"full"`` uses all finite image data. When
switching from full-image to ROI normalization, multiply every regularizer weight
by ``(RMS_full / RMS_roi)**2`` to preserve the relative data/regularization balance.

Numeric masks are nonnegative confidence weights. Use Boolean masks for equal
weight across segmentation labels; ``mask_dims`` specifies their axis order.
Shared parameters remain visible wherever any included observation supports them.
A fit with no usable observations raises ``ValueError``.

``reference_image=reference`` in ``run_fit``, ``build_loss`` or
``make_model_callable`` fits signal intensities as
:math:`S_{\mathrm{ref}} M(\theta)` without dividing the measurements. Supply a
finite spatial reference; returned predictions include its scale. The reference
is fixed, so its measurement uncertainty remains. Uniform signal-domain weights
and uniform Z-spectrum weights generally give different fits.

For double precision, set ``JAX_ENABLE_X64=true`` before starting Python and use
float64 data, observables and initial parameters.

Regularization
~~~~~~~~~~~~~~

``optimizer_name=None`` or ``"auto"`` selects FISTA with enabled TV terms and
Adam otherwise. Explicit Adam with TV warns and uses a smoothed penalty.
``FitInfo.optimizer_name`` and ``regularization_mode`` record the selection.

FISTA and split Bregman use true proximal TV on physical parameter maps, respect
bounds and constant dimensions, and avoid coupling across mask boundaries.
Temporal ``smoothness`` remains a squared-difference penalty. Custom TV models
must declare separable physical fields and bounds in ``Model.parameter_specs``.
The proximal optimizers do not accept Adam base factories.

``proximal_tolerance`` and ``proximal_max_steps`` control inner TV accuracy and
work; ``bregman_penalty`` and ``bregman_inner_steps`` control split Bregman.
Invalid weights, parameter names, rates or bounds raise errors.
Optimizer-level ``regularizer_weight`` supplies gradient decay for ordinary
Optax optimizers and proximal penalties for FISTA/split Bregman. It accepts a
scalar or matching parameter tree. AdamW also applies its own decoupled decay.

Stopping and resuming
~~~~~~~~~~~~~~~~~~~~~

``step_size`` accepts a constant or Optax schedule; true-TV solvers adapt it by
backtracking. ``chunk_size`` sets the interval between host progress and
finiteness checks. A progress callback can cancel by raising an exception.

Enable early stopping with ``convergence_tolerance`` and ``parameter_tolerance``;
``convergence_patience`` and ``min_steps`` control when it takes effect.
``num_steps`` is the maximum. With ``return_fit_info=True``, the result includes
the termination reason, completed updates, final objective and ``info.state``.
History entries precede updates; ``info.final_loss`` evaluates the returned fit.
With ``return_model_callable=True``, fit information follows the callable.

Resume with ``initial_state=info.state`` and the same model, data, regularizers
and optimizer. State and schedule counters continue; stopping counters restart.
``initial_params`` starts a fresh optimizer and cannot accompany ``initial_state``.
Optimizer states cannot be exchanged between Adam, FISTA and split Bregman.

Nonfinite updates or failed proximal/backtracking steps raise ``FitError``.
Its ``last_valid_state`` and ``loss_history`` retain the last verified checkpoint
when one exists.
