.. _api-optim:

Optimizers and regularizers
===========================

:func:`mrax.opt.optimizers.make_optimizer` provides Optax transformations,
FISTA and split Bregman through one factory. Fitting, reconstruction and
diagnostics use the shared executor. See :doc:`model_fitting` for a workflow
overview.

Proximal solvers
----------------

FISTA follows the accelerated proximal-gradient recurrence
[BeckTeboulle2009]_. Full backtracking updates the trial step, momentum and
extrapolated gradient point together [Scheinberg2014]_; ``"classic"`` selects
the original nonincreasing-step rule. The optional gradient restart follows
[ODonoghue2015]_. A restart retains the last certified step scale.

The native TV/box prox uses accelerated dual projected gradients
[BeckTeboulleTV2009]_, with masked finite differences, higher orders and warm
starts. Its primal-dual gap controls accuracy. Accepted parameters satisfy the
boxes, while extrapolation follows the FISTA recurrence.

Split Bregman follows the augmented-primal, split-shrinkage and multiplier
recurrence [Goldstein2009]_. It initializes splits from the current field
differences and multipliers at zero. TV weights enter shrinkage;
``bregman_penalty`` fixes the augmented constraint penalty.

The primal solve uses projected Armijo steps: matrix-free Newton/CG for at most
32 real unknowns and an eight-pair L-BFGS inverse for larger systems.
Positive-curvature secants use gradients with matching splits and multipliers.
A two-level local/coarse preconditioner conditions CG; curvature shifts damp
Newton directions, with acceptance evaluated on the augmented objective.
One bounded inverse/line-search restart handles a failed large-system primal
step. Shrinkage and multiplier updates follow a certified primal step.

Smooth-model intermediates are recomputed during differentiation so curvature
probes do not retain a full reverse-mode tape for an image series. This reduces
memory use.

Convex convergence results apply under the assumptions in the cited papers.
For nonlinear MRI objectives, assess the projected KKT residual, constraint
residuals and sensitivity to starting values.

Controls
~~~~~~~~

Both solvers default to ``proximal_tolerance=1e-8`` and
``proximal_max_steps=2048``. Named, injected and factory-based fits share these
defaults; injected transformations retain their explicit constructor settings.

.. list-table::
   :header-rows: 1
   :widths: 2 3

   * - Option
     - Meaning
   * - ``fista_backtracking``
     - ``"full"`` or ``"classic"`` step/momentum rule.
   * - ``fista_restart``
     - ``"gradient"`` or ``"none"`` momentum restart.
   * - ``proximal_tolerance`` / ``proximal_max_steps``
     - Native-TV gap target and inner work limit; Newton/CG residual target
       and total CG work limit, respectively.
   * - ``bregman_penalty``
     - Fixed augmented penalty, separate from TV strength.
   * - ``bregman_inner_steps``
     - Primal iterations before shrinkage and multiplier updates; default one.
   * - ``proximal_max_step_size``
     - Upper limit for the trial step.
   * - ``grad_clip_norm``
     - Trial-step cap with consistent gradient/proximal scaling.

Newton/CG uses a relative residual target based on
``sqrt(proximal_tolerance)`` and floating-point resolution. Larger subproblems
use inexact forcing that tightens toward stationarity. The residual uses the
positive-preconditioner energy norm. L-BFGS work follows
``bregman_inner_steps``; its split shrinkage is analytic.

Native field TV uses affine physical coordinates. Fixed block scales transform
coordinates, bounds and TV operators together. Use ``parameter_space="auto"``
or ``"physical"`` for this fitting path; smooth FISTA and explicit
raw-coordinate proximal penalties also support raw coordinates.

Optimizer-level ``regularizer_weight`` supplies gradient decay for ordinary
Optax transformations and proximal penalties for the two proximal solvers.
It accepts a scalar or matching parameter tree. AdamW additionally applies its
decoupled decay.

Injected and direct Optax use
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fitting binds the physical objective, bounds and TV operators for named,
injected and factory-provided transformations::

    run_fit(experiment, model, data, optimizer_name="fista", **options)
    run_fit(experiment, model, data,
            optimizer=make_optimizer("fista", 0.02), **options)
    run_fit(experiment, model, data,
            optimizer_factory=lambda rate: make_optimizer("fista", rate),
            **options)

For a direct Optax loop, supply the smooth ``value_fn`` to ``update`` and use
``optax.apply_updates``. FISTA reevaluates gradients at extrapolated points;
split Bregman solves its augmented primal objective. Configure gradient clipping,
bounds and proximal terms on the solver transformation.

Fixed-step classic FISTA also supports caller-supplied extrapolated gradients:
set ``gradient_at_extrapolated=True`` and evaluate at
:func:`mrax.opt.optimizers.fista_extrapolated_params`. Generic ``optax.chain``
composition belongs to direct loops with explicit objective and proximal terms;
model-fitting field-TV binding uses the transformation directly.

State and termination
~~~~~~~~~~~~~~~~~~~~~

Zero schedule entries pause optimizer, split and multiplier updates; positive
entries resume them. Convergence uses the configured stationarity checks.
Split-Bregman state also records primal/dual residuals, inner work,
preconditioner rebuilds and quasi-Newton history diagnostics.

Status 3 reports an uncertified proximal or line-search step. Status 4 reports
a precision-limited rounded step with a nonzero composite KKT residual. Both
retain the last accepted state for inspection. Higher precision or an explicit
model-domain decision may be appropriate.

Resume using the same implementation and its complete state. Start a fresh
optimizer when changing solver family or upgrading from the released FISTA or
split-Bregman state layouts. :ref:`api-fitting` describes ``FitError`` and
checkpoint continuation.

Factory
-------

.. autofunction:: mrax.opt.optimizers.make_optimizer

.. autofunction:: mrax.opt.optimizers.fista_extrapolated_params

Regularizers
------------

.. autofunction:: mrax.opt.regularizers.spatial_smoothness_penalty

.. autofunction:: mrax.opt.regularizers.total_variation_penalty

.. autofunction:: mrax.opt.regularizers.higher_order_total_variation_penalty

.. autofunction:: mrax.opt.regularizers.make_regularizer_fn

.. autofunction:: mrax.opt.regularizers.estimate_field_scales

.. autofunction:: mrax.opt.regularizers.estimate_param_scales

.. autofunction:: mrax.opt.regularizers.autoscale_regularizer_configs

Low-level operations
--------------------

.. autofunction:: mrax.opt.ops.soft_threshold

.. autofunction:: mrax.opt.ops.isotropic_shrinkage

.. autofunction:: mrax.opt.ops.forward_gradient2d

.. autofunction:: mrax.opt.ops.divergence2d

.. autofunction:: mrax.opt.ops.conjugate_gradient
