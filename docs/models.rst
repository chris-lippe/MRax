Models
======

Built-in forward models and parameterizations used throughout MRax live in
``mrax.library.default_models`` and return :class:`mrax.data_struct.model.Model` instances with
parametrizations, regularizer hooks, and unit metadata.
Method references are collected in :doc:`references`; the sections below cite the
non-trivial model families directly where they are introduced.

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

Core relaxometry
----------------

T1 inversion recovery (``make_t1_ir_model``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Signal: :math:`S = |A - 2 A e^{-TI / T1}|` (small bias added in code for stability).
- Parameters: ``amplitude`` (a.u.), ``t1`` (ms).
- Observables: inversion times ``TI`` (ms) provided either in ``offsets`` or ``experiment.observables``.
- Default parameter bounds: ``t1`` in ``[50, 5000]`` ms via a sigmoid mapping; amplitude constrained positive.
  (See ``parameter_library.toml`` for the editable defaults.)

FAIR ASL (``make_fair_asl_model``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- References: FAIR was introduced for relative cerebral blood-flow and perfusion
  imaging by Kim [Kim1995]_ and Kim/Tsekos [KimTsekos1997]_.
- Signal: inversion recovery with selective/non-selective parameter maps chosen by ``fair_label``.
- Parameters: ``amplitude_sel``/``t1_sel`` and ``amplitude_non``/``t1_non``.
- Observables: ``TI`` (ms) and ``fair_label`` (0=selective, 1=non-selective).
- Derived maps: ``t1_sel``, ``t1_non``, and ``perfusion_map`` (ml/100g/min).
- Perfusion formula uses :math:`\lambda \cdot 60000 \cdot (T1_{non}/T1_{blood}) \cdot (1/T1_{sel} - 1/T1_{non})`.
  Defaults to ``t1_blood=2430`` ms and ``partition_coeff=90``; override via observables or study globals with keys
  ``t1_blood``/``t1_blood_ms`` and ``partition_coeff``/``partition_coefficient``/``lambda``.
- FAIR scans default to experiment type ``fair_asl`` and use this model.

T1 variable TR (``make_t1_vtr_model``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Signal: :math:`S = A (1 - e^{-TR / T1}) + c`.
- Parameters: ``amplitude`` (a.u.), ``t1`` (ms), ``bias`` (a.u.).
- Observables: repetition times ``TR`` (ms).
- Default parameter bounds: ``t1`` in ``[50, 5000]`` ms, amplitudes/bias scaled positive.

T1 variable flip angle (``make_t1_vfa_model``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Reference: the spoiled-GRE variable-flip-angle formulation follows the
  DESPOT-style T1 mapping literature [Deoni2003]_.
- Signal: spoiled GRE steady state :math:`S = A \tfrac{(1-e^{-TR/T1})\sin(\alpha)}{1-\cos(\alpha)e^{-TR/T1}} + c`.
- Parameters: ``amplitude`` (a.u.), ``t1`` (ms), ``bias`` (a.u.).
- Observables: repetition time ``TR`` (ms) plus flip angles (deg) provided via
  ``flip_angle``/``flip_angles``/``FA`` in observables or directly as ``offsets``.
- Default parameter bounds: ``t1`` in ``[50, 5000]`` ms, amplitudes/bias scaled positive.

Exponential decay, T2, and T2* (``make_exp_decay_model``, ``make_t2_se_model``, ``make_t2star_se_model``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Shared base signal: :math:`S = A \exp(-TE / \tau) + c`.
- The generic ``make_exp_decay_model`` exposes ``decay_time`` (ms).
- ``make_t2_se_model`` uses the same model with the decay parameter named ``t2``;
  multi-spin-echo (``mse``) experiments default to this model.
- Signal: :math:`S = A \exp(-TE / T2) + c`.
- Parameters: ``amplitude`` (a.u.), ``t2`` (ms), ``bias`` (a.u.).
- Observables: echo times ``TE`` (ms) via ``offsets`` or ``experiment.observables``.
- Default parameter bounds: ``t2`` in ``[5, 200]`` ms, amplitudes positive.
- ``make_t2star_se_model`` uses the same model with the decay parameter named
  ``t2star``; multi-gradient-echo (``mge``) experiments default to this model.
- Signal: :math:`S = A \exp(-TE / T2^*) + c`.
- Parameters: ``amplitude`` (a.u.), ``t2star`` (ms), ``bias`` (a.u.).
- Observables: echo times ``TE`` (ms).
- Default parameter bounds: ``t2star`` in ``[1, 200]`` ms, amplitudes positive.

Diffusion tensor imaging (``make_dti_tensor_model``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Reference: diffusion tensor MRI follows the tensor formulation introduced by
  Basser, Mattiello, and LeBihan [Basser1994]_.
- Signal: :math:`S = S_0 \exp(-b \cdot \mathbf{g}^T \mathbf{D} \mathbf{g})`.
- Parameters: ``s0`` (a.u.) plus tensor components ``dxx``, ``dyy``, ``dzz``,
  ``dxy``, ``dxz``, ``dyz`` (mm\ :sup:`2`/s).
- Observables: ``bvals`` (s/mm\ :sup:`2`) and ``bvecs`` (unitless, shape ``(N, 3)``).
- Derived maps: ``adc``/``md`` (mean diffusivity), ``fa`` (fractional anisotropy), ``ad`` (axial diffusivity),
  ``rd`` (radial diffusivity), and ``trace`` (tensor trace).

Diffusion kurtosis (``make_dki_kurtosis_model``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- References: the kurtosis extension follows diffusional kurtosis imaging
  [Jensen2005]_, with MRI context summarized in [JensenHelpern2010]_.
- Signal: :math:`S = S_0 \exp(-b D_{app} + \tfrac{1}{6} b^2 D_{app}^2 K)` with isotropic kurtosis :math:`K`.
- Parameters: ``s0`` (a.u.), tensor components ``dxx``...``dyz`` (mm\ :sup:`2`/s), and ``kurtosis`` (unitless).
- Observables: ``bvals`` (s/mm\ :sup:`2`) and ``bvecs`` (unitless, shape ``(N, 3)``).
- Derived maps: DTI metrics plus ``mk`` (mean kurtosis).

Saturation-transfer / CEST models
---------------------------------

WASABI / B0-B1 mapping (``make_wasabi_model``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Reference: the WASABI signal model follows Schuenke et al. [Schuenke2017]_.
- Signal: :math:`|c - d \sin^2(\theta) \sin^2(\phi)|` with :math:`\theta = \arctan(\gamma B1 / \Delta\omega)`
  and :math:`\phi = \tfrac{1}{2} \tau \sqrt{(\gamma B1)^2 + \Delta\omega^2}` (``tau`` is the pulse duration).
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
- Signal: :math:`Z(\Delta\omega) = Z_{base} - \sum_i a_i \tfrac{(G_i/2)^2}{(\Delta\omega - \delta_i)^2 + (G_i/2)^2}`.
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
- Signal: :math:`Z = (P_z^2 - Z_{ss}) e^{-R_{1\rho} t_{sat}} + Z_{ss}` with
  :math:`R_{1\rho} = R_{eff} + \sum_i R_{ex,i}` and explicit exchange pools.
- Observables: ``freq_offsets`` (ppm), ``sat_b1`` (uT), ``sat_time`` (ms), plus
  study-wide ``field_strength_mhz``. Optional ``t1_map``/``r1_map``,
  ``b0_map`` and ``b1_map`` are consumed with the same units as AB-MT+n.
- Unit convention: frequency offsets and chemical shifts are converted to
  angular ppm via :math:`2\pi`; saturation amplitude uses
  ``SAT_GAMMA * sat_b1_uT / field_strength_mhz``. Pool rate parameters remain
  in the parameter-library ``ppm`` convention used by AB-MT+n.

AB-MT+n (``make_AB_MT_n_model``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Reference: AB-MT+n is an MRax model described by Lippe and Hoerr in a
  manuscript currently under review at *Magnetic Resonance in Medicine*
  [Lippe2026]_. Semisolid MT line-shape handling follows the
  super-Lorentzian MT modeling context of Morrison, Stanisz, and Henkelman
  [Morrison1995]_.
- Signal: analytic approximation to multi-pool Bloch–McConnell dynamics with explicit
  saturation power and duration dependence.
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
