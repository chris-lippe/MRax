Quantitative Mapping Recipes
============================

These short recipes show the minimum steps required to fit common models. Replace
the ``exp``/``ds`` selections with your own ``Study`` entries (from Bruker import,
synthetic builders, or custom data loaders). See :doc:`models` for model equations
and :doc:`references` for method citations.

T1 mapping (inversion recovery)
-------------------------------

.. code-block:: python

   from mrax.core.loss_builder import run_fit
   from mrax.library.default_models import make_t1_ir_model

   exp = study.experiments[idx]     # exp_type "ir"
   ds = study.datasets[idx]
   model = make_t1_ir_model(scale=1e6)
   fields, losses = run_fit(exp, model, ds, num_steps=2000, step_size=0.02)

   t1_map = fields["t1"]            # Quantity with units

T2 mapping (multi-echo spin echo)
---------------------------------

.. code-block:: python

   from mrax.core.loss_builder import run_fit
   from mrax.library.default_models import make_t2_se_model

   exp = study.experiments[idx]     # exp_type "mse"
   ds = study.datasets[idx]
   model = make_t2_se_model(scale=1e6)
   fields, losses = run_fit(exp, model, ds, num_steps=2000, step_size=0.02)

   t2_map = fields["t2"]

FAIR perfusion (selective + non-selective)
------------------------------------------

.. code-block:: python

   from mrax.core.loss_builder import run_fit
   from mrax.library.default_models import make_fair_asl_model

   exp = study.experiments[idx]     # exp_type "fair_asl"
   ds = study.datasets[idx]
   model = make_fair_asl_model(scale=1e6, t1_blood_ms=2430.0, partition_coeff=90.0)

   fields, losses = run_fit(exp, model, ds, num_steps=2000, step_size=0.02)
   derived = model.derived_maps(fields, observables=exp.merged_observables())
   perfusion_map = derived["perfusion_map"]

Saturation transfer (WASABI + AB-MT)
------------------------------------

Fit WASABI first to estimate B0/B1, then insert those maps into the AB-MT experiment.

.. code-block:: python

   from mrax.core.loss_builder import run_fit
   from mrax.data_struct import Quantity
   from mrax.library.default_models import make_wasabi_model, make_AB_MT_n_model

   # WASABI / B0/B1
   wasabi_exp = study.experiments[wasabi_idx]
   wasabi_ds = study.datasets[wasabi_idx]
   wasabi_model = make_wasabi_model()
   wasabi_fields, _ = run_fit(wasabi_exp, wasabi_model, wasabi_ds, num_steps=1000, step_size=0.02)
   wasabi_maps = wasabi_model.derived_maps(wasabi_fields, observables=wasabi_exp.merged_observables())

   # AB-MT (requires r1_map or t1_map + sat_b1/sat_time/TR observables)
   abmt_exp = study.experiments[abmt_idx]
   abmt_ds = study.datasets[abmt_idx]
   abmt_exp = abmt_exp.insert_observables(
       {
           "b0_map": Quantity(wasabi_maps["B0_map"], "ppm"),
           "b1_map": Quantity(wasabi_maps["B1_map"], "%"),
           "t1_map": Quantity(t1_map_ms, "ms"),
       }
   )
   abmt_model = make_AB_MT_n_model()
   abmt_fields, _ = run_fit(
       abmt_exp,
       abmt_model,
       abmt_ds,
       num_steps=500,
       step_size=0.02,
       auto_mask_nan_observables=True,
   )

DTI (tensor model)
------------------

.. code-block:: python

   from mrax.core.loss_builder import run_fit
   from mrax.library.default_models import make_dti_tensor_model
   from mrax.analysis import compute_dti_maps

   exp = study.experiments[idx]     # exp_type "dti"
   ds = study.datasets[idx]
   model = make_dti_tensor_model()
   fields, losses = run_fit(exp, model, ds, num_steps=3000, step_size=0.02)

   maps = compute_dti_maps(fields)
   adc_map = maps.adc
   fa_map = maps.fa
   md_map = maps.md
   ad_map = maps.ad
   rd_map = maps.rd

   # run_fit attaches diffusion-derived maps (adc/md/fa/ad/rd/trace) to ``fields``

DKI (kurtosis model)
--------------------

.. code-block:: python

   from mrax.core.loss_builder import run_fit
   from mrax.library.default_models import make_dki_kurtosis_model

   exp = study.experiments[idx]     # exp_type "dki" or "dti"
   ds = study.datasets[idx]
   model = make_dki_kurtosis_model()
   fields, losses = run_fit(exp, model, ds, num_steps=3000, step_size=0.02)

   mk_map = fields["mk"]            # mean kurtosis

Notes on observables
--------------------

- FAIR models require ``TI`` and ``fair_label`` observables (Bruker import supplies them).
- DTI/DKI models require ``bvals`` and ``bvecs`` observables.
- AB-MT requires ``sat_b1``, ``sat_time``, and ``TR`` per-frame observables plus a
  ``t1_map`` (or ``r1_map``) and optional ``b0_map``/``b1_map``.
- Use :func:`mrax.data_struct.pick_observable_for_frames` to pick an observable that
  matches the frame axis when plotting spectra.
