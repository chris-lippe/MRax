Models
======

Built-in forward models and parameterizations used throughout MRax live in
``mrax.library.default_models`` and return :class:`mrax.data_struct.model.Model` instances with
parametrizations, regularizer hooks, and unit metadata.
Method references are collected in :doc:`references`; the sections below cite the
non-trivial model families directly where they are introduced.

All equations below define the mathematical forward model, evaluated independently
at each spatial location and acquisition frame.

.. contents::
   :local:
   :depth: 2

Parameter library
-----------------

Default bounds, transforms, units, and initial values are defined in
``mrax/library/parameter_library.toml``. Each ``make_*_model`` factory can select a named
parameter set via ``param_key`` (default ``"default"``) and apply targeted overrides via
``param_overrides``:

.. code-block:: python

   from mrax.library.default_models import make_t1_ir_model

   model = make_t1_ir_model(
       param_key="default",
       param_overrides={
           "t1": {"bounds": (100.0, 4000.0)},
           "amplitude": {"init_raw": 0.5},
       },
   )

Use ``init`` when you want to specify the starting value in constrained physical units;
it takes precedence if an inherited default also defines ``init_raw``. Built-in defaults
use physical ``init`` values throughout. Scale-relative softplus amplitudes use
``init_scale_key`` to keep the physical initialization tied to the model scale. Use
``init_raw`` only when you deliberately want to initialize the unconstrained optimizer
coordinate directly.

The parameter library is plain TOML. You can point to another file with
``MRAX_PARAMETER_LIBRARY``. For local pool definitions outside the public defaults, set
``MRAX_PRIVATE_PARAMETER_LIBRARY`` to an overlay TOML file; its contents are merged on
top of the public library.

Fitting controls and optimizer state
------------------------------------

``run_fit`` uses device-side optimizer chunks on CPU and CUDA. ``chunk_size``
limits work between host-side progress and finiteness checks; the complete loss
history still contains one entry before each accepted update. Progress callbacks
retain their existing cadence and can cancel a fit by raising an exception.
Existing return tuples are preserved unless ``return_fit_info=True`` is requested.

Optimization entry points default to ``device="cpu"``, independently of JAX's
ambient default backend or the device storing the input arrays. Select
``device="gpu"``, ``device="gpu:0"``, or pass a local JAX device object. Invalid
or unavailable selections raise an error without falling back to another device.
The selection applies to parameters, optimizer state, resumed moments, data,
observables and compiled update kernels. Inputs and the caller's JAX device
context are preserved; MRax does not change global precision settings.

``mrax.get_available_devices()`` returns local device objects, CPU first.
Pass ``"gpu"`` to list only GPUs; an unavailable platform returns an empty list.
The same helper is exported by ``mrax.core``. For example:

.. code-block:: python

   from mrax import get_available_devices
   from mrax.core import run_fit

   devices = get_available_devices("gpu")
   selected = devices[0] if devices else "cpu"
   fields, history = run_fit(experiment, model, dataset, device=selected)

Device selection is also explicit in ``build_loss``, ``execute_optimizer``,
``make_optimizer``, ``make_tv_steps``, ``debug_fit_prototype``, variational and
model-based reconstruction, regularized denoising, and linear drift correction.
Study-plan and batch runners and the GUI session's current/preview fit methods
also accept ``device``. Per-experiment device overrides in a study plan take
precedence over its default. Custom batch analysis callbacks retain their own
keyword contract and must forward device selection to the optimization they run.
Nested MRax optimization calls forward the same selection. When using an Optax
transform returned by ``make_optimizer`` directly, place the parameter tree on
that device before compiling or applying updates; an enclosing external JAX
compilation must use the same device. A resumed fit may select a different
available device; its parameter and optimizer arrays are transferred together.

Data RMS normalization defaults to ``data_normalization="roi"``. Its sample set
contains finite observations retained by the ROI mask, frame mask and automatic
observable masks. Excluded pixels and frames therefore do not change the balance
between the data term and regularization. Select ``data_normalization="full"``
in ``build_loss`` or ``run_fit`` to use the previous finite-image RMS convention.
This default changes the objective for masked fits. To preserve previous tuning
when switching scopes, multiply every regularizer weight, including temporal
terms, by ``(RMS_full / RMS_roi)**2``; the objective then differs only by an
overall positive multiplier. Merely changing the optimizer does not require
this compensation.

Numeric fitting and frame masks are nonnegative residual confidence weights;
their positive support defines the RMS sample set. Convert segmentation labels
to a Boolean mask before fitting when all regions should contribute equally.
``mask_dims`` names the axes of the supplied array, including their order.
Shared parameter support is the union of its included observations, so one
missing frame does not hide a parameter fitted from other usable frames.
No remaining observations raises ``ValueError``. Negative or nonfinite
regularizer weights, unknown configured parameter names, invalid learning
rates and invalid box bounds are rejected rather than silently ignored.

Optimizer-level ``regularizer_weight`` supplies coupled gradient decay for
ordinary Optax optimizers and proximal penalties for FISTA and split Bregman.
Scalar decay weights or matching per-parameter trees are supported. AdamW also
retains its constructor's decoupled weight decay. Only supported optimizer
constructors are accepted; standalone Optax transforms and methods needing
extra loss callbacks cannot be selected as optimizer names. Complex gradients
are conjugated before Euclidean descent updates in fitting and reconstruction.
Explicit study globals take precedence, including an empty mapping; otherwise
the current experiment's globals take precedence over model fallback globals.

``optimizer_name=None`` or ``"auto"`` selects FISTA when enabled TV terms are
present, and Adam otherwise. This policy also applies to new GUI fit settings.
Explicit ``"adam"`` with TV emits a warning and uses an explicitly smoothed TV
gradient. The GUI also includes this warning in its fit notes. The effective
optimizer and ``regularization_mode`` are recorded in ``FitInfo``; selecting
``"fista"`` or ``"split_bregman"`` never silently selects Adam.

True TV is evaluated on physical parameter maps. FISTA applies a joint TV and
box-constrained proximal map with a checked primal-dual gap, smooth-objective
backtracking, and acceleration restart. Split Bregman includes the finite-
difference constraint, the augmented-Lagrangian coupling in its primal solve,
isotropic shrinkage, and a dual update. Its primal subproblem uses backtracked
projected gradient steps. Temporal ``smoothness`` remains a squared-difference
penalty in the smooth objective. Invalid mask edges, including complete
higher-order stencils, do not couple across the ROI boundary.

Physical coordinates preserve parameter bounds and constant dimensions. They
use fixed initial block-curvature scaling; this changes coordinates and their
corresponding TV weights together, preserving the objective. Built-in factories
provide parameter specs from the parameter library. Custom models using true TV
must declare ``Model.parameter_specs`` describing their separable physical
fields and scalar transform bounds. Unsupported parametrizations raise an error
instead of selecting a different optimizer.

``proximal_tolerance`` controls the inner TV dual-gap accuracy (default
``1e-10``), and ``proximal_max_steps`` bounds its work (default ``4096``).
``bregman_penalty`` and ``bregman_inner_steps`` control split-Bregman coupling
and primal work. Failure to certify a proximal step or find a valid backtracking
step raises ``FitError`` and preserves the last checked state. An inner convex
TV certificate does not establish a global optimum of a nonlinear model fit.

Classical FISTA uses Euclidean gradient steps; attaching Adam's adaptive metric
and momentum does not implement that algorithm. The low-level proximal
optimizers default to gradient steps and reject Adam base factories. Adam can
in principle solve a properly specified split-Bregman primal subproblem, but
the provided solver uses backtracked gradient steps for that subproblem.

In variational reconstruction, the L1 and TV penalties use a mean over image
elements. Their proximal weights include the same element-count normalization,
so the declared regularization weight does not acquire an extra image-size
factor when a proximal solver is selected.

``step_size`` accepts a constant rate or an Optax-compatible schedule. For true
TV it seeds the backtracking rate; the accepted rate can change during fitting.
Schedules scale the adaptive rate, and their counters persist when resuming.
For example, an application can choose a schedule without changing model defaults:

.. code-block:: python

   import optax
   from mrax.core import run_fit

   rate = optax.exponential_decay(0.03, transition_steps=64, decay_rate=0.8)
   fields, history, info = run_fit(
       experiment, model, dataset,
       step_size=rate,
       num_steps=512,
       convergence_tolerance=1e-7,
       parameter_tolerance=1e-5,
       convergence_patience=5,
       min_steps=16,
       return_fit_info=True,
   )

Convergence is opt-in and requires consecutive small scaled objective changes and
relative optimizer-coordinate updates and, for TV, stationarity and splitting
checks. ``num_steps`` is then a maximum, not a guarantee
that the optimizer found a unique optimum. ``info`` records the termination reason,
completed updates, actual parameter devices, final objective, and a ``FitState``.
Raw-coordinate fits store raw parameters. True-TV fits store scaled physical
coordinates and their fixed scales, along with the primal and dual solver
state. Resume with the same model, inputs, regularizers and optimizer. Raw
``initial_params`` from the model factory can initialize a new true-TV fit;
optimizer states cannot be exchanged between Adam, FISTA and split Bregman.
``info.final_loss`` evaluates the returned parameters, whereas history entries
precede their updates. A zero-step fit returns initialization without evaluating
the model and has ``info.final_loss=None``.

Pass ``initial_state=info.state`` to continue with raw parameters and optimizer
moments preserved. Use the same problem and compatible optimizer state structure;
``initial_state`` and ``initial_params`` are mutually exclusive. State step counts
and schedule counts continue, while stopping counters start afresh. Supplying
``initial_params`` alone initializes a new optimizer. With
``return_model_callable=True``, fit information is appended after the callable.

Nonfinite initial objectives, gradients, optimizer states, proposals, or final
chunk objectives raise ``FitError``. Its ``last_valid_state`` and ``loss_history``
refer to the last verified chunk boundary, or are empty when initialization is
invalid. Applications can use that checkpoint for recovery with compatible
optimizer settings. Library calls do not change global JAX precision, device
selection, or memory-allocation configuration.

Symmetric bounded parameter transforms use the centered, algebraically equivalent
form ``h * tanh(raw / 2)`` for bounds ``[-h, h]``. The inverse preserves small
nonzero initial values and retains boundary clipping. Bounds, default metadata,
units, and the forward-model equations are unchanged.

Define your own model
---------------------

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

Relaxometry and perfusion models
--------------------------------

T1 inversion recovery (``make_t1_ir_model``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Defining equation:

  .. math::

     S(TI) = \left|A - 2A\exp\left(-\frac{TI}{T_1}\right)\right|.

- Parameters: ``amplitude`` (a.u.), ``t1`` (ms).
- Observables: inversion times ``TI`` (ms) provided either in ``offsets`` or ``experiment.observables``.
- Default parameter bounds: ``t1`` in ``[50, 5000]`` ms via a sigmoid mapping; amplitude constrained positive.
  (See ``parameter_library.toml`` for the editable defaults.)

FAIR ASL (``make_fair_asl_model``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- References: FAIR was introduced for relative cerebral blood-flow and perfusion
  imaging by Kim [Kim1995]_ and Kim/Tsekos [KimTsekos1997]_.
- Defining equation: let :math:`\ell` be ``fair_label`` and define
  :math:`(A_\ell,T_{1,\ell})=(A_{sel},T_{1,sel})` for :math:`\ell<0.5`, otherwise
  :math:`(A_{non},T_{1,non})`.  The frame signal is

  .. math::

     S(TI,\ell) = \left|A_\ell - 2A_\ell\exp\left(-\frac{TI}{T_{1,\ell}}\right)\right|.

- Parameters: ``amplitude_sel``/``t1_sel`` and ``amplitude_non``/``t1_non``.
- Observables: ``TI`` (ms) and ``fair_label`` (0=selective, 1=non-selective).
- Derived maps: ``t1_sel``, ``t1_non``, and ``perfusion_map`` (ml/100g/min).
- Defining derived-perfusion equation:

  .. math::

     f = \lambda\,60000\,\frac{T_{1,non}}{T_{1,blood}}
     \left(\frac{1}{T_{1,sel}}-\frac{1}{T_{1,non}}\right).

  Defaults to ``t1_blood=2430`` ms and ``partition_coeff=90``; override via observables or study globals with keys
  ``t1_blood``/``t1_blood_ms`` and ``partition_coeff``/``partition_coefficient``/``lambda``.
- FAIR scans default to experiment type ``fair_asl`` and use this model.

T1 variable TR (``make_t1_vtr_model``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Defining equation: :math:`S(TR) = A \left(1 - \exp(-TR/T_1)\right) + c`.
- Parameters: ``amplitude`` (a.u.), ``t1`` (ms), ``bias`` (a.u.).
- Observables: repetition times ``TR`` (ms).
- Default parameter bounds: ``t1`` in ``[50, 5000]`` ms, amplitudes/bias scaled positive.

T1 variable flip angle (``make_t1_vfa_model``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Reference: the spoiled-GRE variable-flip-angle formulation follows the
  DESPOT-style T1 mapping literature [Deoni2003]_.
- Defining equation (with :math:`\alpha` converted from degrees to radians):

  .. math::

     E_1 &= \exp(-TR/T_1), \\
     S(TR,\alpha) &= A\frac{(1-E_1)\sin\alpha}{1-E_1\cos\alpha}+c.
- Parameters: ``amplitude`` (a.u.), ``t1`` (ms), ``bias`` (a.u.).
- Observables: repetition time ``TR`` (ms) plus flip angles (deg) provided via
  ``flip_angle``/``flip_angles``/``FA`` in observables or directly as ``offsets``.
- Default parameter bounds: ``t1`` in ``[50, 5000]`` ms, amplitudes/bias scaled positive.

Exponential decay, T2, and T2* (``make_exp_decay_model``, ``make_t2_se_model``, ``make_t2star_se_model``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Shared defining equation: :math:`S(TE) = A\exp(-TE/\tau)+c`.
- The generic ``make_exp_decay_model`` exposes ``decay_time`` (ms).
- ``make_t2_se_model`` uses the same model with the decay parameter named ``t2``;
  multi-spin-echo (``mse``) experiments default to this model.
- Defining T2 equation: :math:`S(TE) = A\exp(-TE/T_2)+c`.
- Parameters: ``amplitude`` (a.u.), ``t2`` (ms), ``bias`` (a.u.).
- Observables: echo times ``TE`` (ms) via ``offsets`` or ``experiment.observables``.
- Default parameter bounds: ``t2`` in ``[5, 200]`` ms, amplitudes positive.
- ``make_t2star_se_model`` uses the same model with the decay parameter named
  ``t2star``; multi-gradient-echo (``mge``) experiments default to this model.
- Defining T2* equation: :math:`S(TE) = A\exp(-TE/T_2^*)+c`.
- Parameters: ``amplitude`` (a.u.), ``t2star`` (ms), ``bias`` (a.u.).
- Observables: echo times ``TE`` (ms).
- Default parameter bounds: ``t2star`` in ``[1, 200]`` ms, amplitudes positive.

Diffusion models
----------------

Diffusion tensor imaging (``make_dti_tensor_model``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Reference: diffusion tensor MRI follows the tensor formulation introduced by
  Basser, Mattiello, and LeBihan [Basser1994]_.
- Defining equation:

  .. math::

     \mathbf D &=
     \begin{bmatrix}
     d_{xx} & d_{xy} & d_{xz} \\
     d_{xy} & d_{yy} & d_{yz} \\
     d_{xz} & d_{yz} & d_{zz}
     \end{bmatrix}, \qquad
     \mathbf g = \begin{bmatrix}g_x & g_y & g_z\end{bmatrix}^{\mathsf T}, \\
     D_{app}(\mathbf g) &= \mathbf g^{\mathsf T}\mathbf D\mathbf g, \\
     S(b,\mathbf g) &= S_0\exp\left[-bD_{app}(\mathbf g)\right].
- Parameters: ``s0`` (a.u.) plus tensor components ``dxx``, ``dyy``, ``dzz``,
  ``dxy``, ``dxz``, ``dyz`` (mm\ :sup:`2`/s).
- Observables: ``bvals`` (s/mm\ :sup:`2`) and ``bvecs`` (unitless, shape ``(N, 3)``).
- Derived maps: ``adc``/``md`` (mean diffusivity), ``fa`` (fractional anisotropy), ``ad`` (axial diffusivity),
  ``rd`` (radial diffusivity), and ``trace`` (tensor trace).

Diffusion kurtosis (``make_dki_kurtosis_model``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- References: the kurtosis extension follows diffusional kurtosis imaging
  [Jensen2005]_, with MRI context summarized in [JensenHelpern2010]_.
- Defining equation, using the same :math:`D_{app}(\mathbf g)` tensor contraction
  as DTI and a scalar (isotropic) kurtosis field :math:`K`:

  .. math::

     S(b,\mathbf g) = S_0\exp\left[-bD_{app}(\mathbf g)
     +\frac{b^2D_{app}(\mathbf g)^2K}{6}\right].
- Parameters: ``s0`` (a.u.), tensor components ``dxx``...``dyz`` (mm\ :sup:`2`/s), and ``kurtosis`` (unitless).
- Observables: ``bvals`` (s/mm\ :sup:`2`) and ``bvecs`` (unitless, shape ``(N, 3)``).
- Derived maps: DTI metrics plus ``mk`` (mean kurtosis).

Saturation-transfer / CEST models
---------------------------------

WASABI / B0-B1 mapping (``make_wasabi_model``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Reference: the WASABI signal model follows Schuenke et al. [Schuenke2017]_.
- Defining equation, with :math:`\Delta\omega=2\pi B_0(\Delta_{ppm}-\delta_{ppm})`
  in rad/s and :math:`\tau=t_{sat}` in seconds:

  .. math::

     \theta &= \arctan\left(\frac{\gamma B_1}{\Delta\omega}\right), \\
     \phi &= \frac{\tau}{2}\sqrt{(\gamma B_1)^2+\Delta\omega^2}, \\
     S &= \left|c-d\sin^2\theta\sin^2\phi\right|.
- Parameters: ``c`` and ``d`` (scaling), ``b1`` (uT), ``delta`` (Hz, B0 offset). Derived maps return B0 (ppm) and
  B1 deviation (% of nominal).
- Observables: frequency ``offsets`` in ppm (converted using the study field strength), optional ``sat_time`` (ms)
  and nominal ``sat_b1`` (uT). ``field_strength_mhz`` must be provided once per study via ``Study.study_globals``.
- Default parameter bounds: ``b1`` around ``[3, 5]`` uT and ``delta`` around
  ``[-100, 100]`` Hz, converted to rad/s internally.

Multi-pool Lorentzian (``make_multipool_lorentzian_model``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- References: the model is an empirical Lorentzian Z-spectrum decomposition in
  the CEST/MR Z-spectroscopy tradition [Liu2013]_ [ZaissBachert2013]_,
  including Lorentzian-line-fit separation of pool contributions [Zaiss2011]_.
- Defining equation: with :math:`\omega=2\pi(\Delta_{ppm}-B_{0,ppm})`,
  :math:`\omega_i=2\pi\delta_{i,ppm}`, and :math:`h_i=G_i/2`,

  .. math::

     L_i(\omega) = a_i\frac{h_i^2}{(\omega-\omega_i)^2+h_i^2},
     \qquad Z(\omega)=Z_{base}-\sum_iL_i(\omega).
- Observables: ``offsets`` in ppm. Saturation-transfer offsets and pool frequency
  shifts are handled as angular ppm by multiplying those ppm values by
  :math:`2\pi`. Empirical linewidth and rate parameters stay in the units declared
  by the parameter library.
- Default pools and bounds (see ``parameter_library.toml`` for the editable source):
  - ``water``: :math:`a \in [0.3, 1.0]`, :math:`G \in [1, 100]`, fixed :math:`\delta = 0`.
  - ``mt``: :math:`a \in [0.02, 0.4]`, :math:`G \in [50, 500]`, :math:`\delta \in [-4, -2]`.
  - ``glu``: :math:`a \in [0.01, 0.3]`, :math:`G \in [1, 20]`, :math:`\delta \in [2.5, 3.5]`.
  - ``glc``: :math:`a \in [0.01, 0.3]`, :math:`G \in [2, 50]`, :math:`\delta \in [1.5, 2.5]`.
- Pools can be reordered/overridden via ``pools=`` and ``pool_overrides=``. Unknown pools can be
  provided by passing a param-spec dict for that pool or by loading a private parameter-library
  overlay.

Zaiss full analytical (``make_zaiss_full_model``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Reference: implements the approximate continuous-wave analytical Z-spectrum
  model of Zaiss/Trott-Palmer used as the starting point for AB-MT+n
  [ZaissBachert2013]_.
- Defining equations: after B0/residual-water correction, let
  :math:`\omega=2\pi\Delta_{ppm}`, :math:`\omega_1=\gamma B_{1,eff}/B_0`,
  :math:`c_\theta=|\omega|/\sqrt{\omega^2+\omega_1^2}`, and
  :math:`P_z=c_\theta` (or :math:`P_z=1` when ``spin_lock`` is enabled).  For each
  pool :math:`i`, with :math:`\omega_i=2\pi\delta_i`, the implemented exchange rate is

  .. math::

     \Gamma_i^2/4 &= \frac{R_{2i}+k_i}{k_i}\omega_1^2+(R_{2i}+k_i)^2, \\
     q_i &= R_{2i}+k_i\frac{\omega_i^2+R_{2i}(R_{2i}+k_i)}
     {\omega^2+\omega_1^2}, \\
     R_{ex,i} &= \frac{f_i\omega_1^2q_i}
     {\Gamma_i^2/4+(\omega-\omega_i)^2}.

  Water relaxation and the steady state are

  .. math::

     R_{eff} &= \frac{R_{1w}\omega^2+R_{2w}\omega_1^2}{\omega^2+\omega_1^2}, \\
     R_{1\rho} &= R_{eff}+R_{ex,mt}+\sum_{i\ne mt}\frac{R_{ex,i}}{1+f_{mt}}, \\
     Z_{ss} &= \frac{R_{1,obs}P_zc_\theta}{R_{1\rho}}, \\
     Z &= (P_z^2Z_i-Z_{ss})\exp(-R_{1\rho}t_{sat})+Z_{ss}.

  :math:`R_{1,obs}` is an inserted ``r1_map`` converted to ppm, or its
  model-based fallback :math:`(R_{1w}+f_{mt}R_{1,mt})/(1+f_{mt})` when the MT
  pool is present (:math:`R_{1w}` otherwise).

  The shipped forward model fixes :math:`Z_i=1`; the factor is shown explicitly
  because it defines the initial longitudinal magnetization term.
- Observables: ``freq_offsets`` (ppm), ``sat_b1`` (uT), ``sat_time`` (ms), plus
  study-wide ``field_strength_mhz``. Optional ``t1_map``/``r1_map``,
  ``b0_map`` and ``b1_map`` are consumed with the same units as AB-MT+n.
- Unit convention: frequency offsets and chemical shifts are converted to
  angular ppm via :math:`2\pi`; saturation amplitude uses
  ``SAT_GAMMA * sat_b1_uT / field_strength_mhz``. Pool rate parameters remain
  in the parameter-library ``ppm`` convention used by AB-MT+n.

AB-MT+n (``make_AB_MT_n_model``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Reference: AB-MT+n is the saturation-transfer model introduced by Lippe and Hoerr
  [Lippe2026]_. Semisolid MT line-shape handling follows the
  super-Lorentzian MT modeling context of Morrison, Stanisz, and Henkelman
  [Morrison1995]_.
- Defining equations: use the corrected angular offset :math:`\omega=2\pi\Delta_{ppm}`
  and :math:`\omega_1=\gamma B_{1,eff}/B_0`.  Let
  :math:`P=|\omega|/\sqrt{\omega^2+\omega_1^2}` (or :math:`P=1` for
  ``spin_lock``).  For a non-MT pool with fraction :math:`f`, rate :math:`k`, and
  offset :math:`\delta`, the off-resonance contribution is

  .. math::

     D &= 4\omega_1^2\delta^2+(k^2+\delta^2)^2, \\
     a &= \frac{2\delta^3fk}{D},\qquad
     b = \frac{\delta^2fk(\delta^2+k^2)}{D},\qquad
     d = \frac{\delta^2fk(3\delta^2-k^2)}{D}, \\
     F(\omega) &= \frac{d-a\omega}{(\omega-\delta)^2+k^2+\omega_1^2}
     +\frac{b+a\omega}{\omega^2+\omega_1^2}.

  For the MT pool, with :math:`u=\omega-\delta_{mt}`, the selected RF rate is

  .. math::

     R_{rf}^{Lor}(u) &= \frac{\omega_1^2R_{2,mt}}{R_{2,mt}^2+u^2}, \\
     R_{rf}^{Gau}(u) &= \frac{\omega_1^2\sqrt{\pi/2}}{R_{2,mt}}
     \exp\left[-\left(\frac{u}{2R_{2,mt}}\right)^2\right], \\
     R_{rf}^{SL}(u) &= \omega_1^2\frac{L_{SL}(|u|/R_{2,safe})}{R_{2,safe}},
     \qquad R_{2,safe}=|R_{2,mt}|.

  :math:`L_{SL}` is the package's tabulated super-Lorentzian kernel, linearly
  interpolated over its stored domain and set to zero outside it.  The complete
  rate and signal equations are

  .. math::

     R_{ex,mt} &= R_{rf}(\omega-\delta_{mt},\omega_1,R_{2,mt})
     \frac{f_{mt}}{1+f_{mt}}\frac{\omega^2}{\omega^2+\omega_1^2}, \\
     R_{1\rho} &= \frac{R_{1,obs}\omega^2+b_0\omega_1^2+a_0\omega_1^2\omega}
     {\omega^2+\omega_1^2}+R_{ex,mt}
     +\sum_{i\ne mt}\frac{\omega_1^2F_i(\omega)}{1+f_{mt}}, \\
     Z_{ss} &= \frac{R_{1,obs}P|\omega|/\sqrt{\omega^2+\omega_1^2}}
     {R_{1\rho}}, \\
     Z_{rec} &= 1+\left(\sqrt{\max(\sin^{-2}\alpha_{eff}-1,0)}\,Z_{prev}-1\right)
     \exp(-R_{1,obs}t_{rec}), \\
     Z &= 1-C+C\left[(Z_{rec}P^2-Z_{ss})\exp(-R_{1\rho}t_{sat})+Z_{ss}\right],
     \qquad C=\exp(-R_{1,obs}t_{delay}).

  Here :math:`t_{sat}`, :math:`t_{delay}`, and :math:`t_{rec}=TR-t_{sat}-t_{delay}`
  are converted to the implementation's ppm-time convention by multiplying
  seconds by the study field strength.  :math:`a_0`, :math:`b_0`, and
  :math:`\sin\alpha_{eff}` are fitted AB-MT+n parameters; :math:`Z_{prev}=1`
  unless supplied as an observable.
- Observables: ``freq_offsets`` (ppm), ``sat_b1`` (uT), ``sat_time`` (ms), ``TR`` (ms),
  plus a study-wide ``field_strength_mhz``.
- Optional maps: ``r1_map`` or ``t1_map`` (required), ``b0_map``/``b1_map`` for B0/B1 corrections.
- Default ``t_delay`` bounds are ``[12.5, 100]`` ms with a 56.25 ms physical-unit initialization.
- Use :func:`mrax.data_struct.join_experiments` to stack multiple sat_trans scans with
  different ``sat_b1``/``sat_time`` into a single fit. Repeated acquisitions (``t``)
  expand parameter maps, so a single ``run_fit`` returns ``(t, x, y)`` fields
  without splitting; you can regularize along ``t`` or set ``parameter_constant_dims`` to
  share parameters. If you want the repeat axis explained by a side model, insert the
  derived observable directly into the experiment (``Experiment.insert_observables``)
  and map the repeat dimension to it.

Identity model (``make_identity_model``)
-----------------------------------------
- Defining equation: :math:`S_j(\mathbf r)=A(\mathbf r)` for every acquisition
  frame :math:`j`.  The amplitude map is broadcast across frames; when its leading
  axis already has the frame length, it is returned unchanged.
- Parameters: ``amplitude`` (unitless).  This is the default placeholder for
  unsupported experiment types and does not encode acquisition dependence.

Using the models with importers
-------------------------------

- ``sat_trans`` scans use the identity model unless you explicitly pick one. Importing them
  is quiet; fitting the default identity model warns because it only estimates an amplitude
  image. Provide a ``model_map_by_type`` when calling
  :func:`mrax.io.bruker_import.load_bruker_study` to select a saturation-transfer model:

  .. code-block:: python

     from mrax.library.default_models import make_multipool_lorentzian_model
     model_map = {"sat_trans": make_multipool_lorentzian_model}
     study, meta = load_bruker_study(bruker_root, model_map_by_type=model_map)

- Pool bounds and observable fallbacks can be customized via the factory arguments shown above.
- For bespoke signal equations, copy ``docs/model_template.py`` and adapt ``model_apply`` / ``parametrization_factory``.

Model-based reconstruction
--------------------------

Every :class:`mrax.data_struct.model.Model` can also be used as a non-linear “model operator”
inside the reconstruction objective via :func:`mrax.reco.variational.reconstruct_model_based`,
which reconstructs parameter maps directly from raw k-space (instead of fitting after a separate
image reconstruction step).
