Overview
========

MRax is a JAX-based toolkit for quantitative MRI research. It provides:

- **Experiments** to describe acquisition geometry, observables, and fitted maps.
- **Models** to encode signal behavior and parameterizations.
- **Studies** to coordinate multiple experiments that may depend on one another.
- **Optimization utilities** for fitting with Optax-based solvers.
- **Storage helpers** to persist MR images, k-space, and fitted maps in ISMRMRD files.

Getting Started
---------------

1. Ensure Python >= 3.12.

2. Install the package from PyPI or from a clone (ideally in a virtual environment)::

     pip install mrax
     # or, from a clone:
     pip install -e .

3. Install dev tooling for tests/docs/notebooks when working from a clone::

     pip install -e ".[all,dev,docs,notebooks]"

4. Run the smoke tests::

     python -m pytest

5. Open ``notebooks/getting_started.ipynb`` for a synthetic fitting walkthrough.

Key Modules
-----------

- ``mrax.data_struct``: data containers for experiments, datasets, models, studies, and storage.
- ``mrax.core.loss_builder``: utilities to build and run fitting loops over experiments and studies.
- ``mrax.opt``: optimization primitives, regularizers, and parameterizations.
- Built-in model equations and bounds: :doc:`models`.
- Method citations for non-trivial model families, regularizers, reconstruction
  methods, and denoisers: :doc:`references`.

Fitting Options
---------------

``run_fit`` accepts Optax optimizers plus ``split_bregman`` and ``fista`` variants.
It supports proximal regularizers, parameter bounds, explicit initial values, masks,
and per-parameter regularizer settings. Use ``plan_study_fits`` and
``execute_study_plan`` for multi-experiment studies.

Unified Pipeline
----------------

Synthetic examples and imported studies use the same ``Study`` abstraction. Build
synthetic datasets with :func:`mrax.core.synthetic.make_synthetic_study` or import
ParaVision scans with :func:`mrax.io.bruker_import.load_bruker_study`. Importers map
vendor metadata into a shared schema (see :mod:`mrax.io.import_definitions`).

Dimension names follow a consistent convention: ``x/y/z`` are spatial axes and ``t``
is temporal. Other non-spatial axes are inferred from observables when the metadata is
unambiguous. Otherwise, MRax keeps a single ``frame`` axis.

For multi-scan datasets collected with related acquisition settings, use
:func:`mrax.data_struct.join_experiments` to stack them into a single frame-major
experiment. The helper keeps a per-source frame index so you can fit subsets as
needed.

For multiple subjects, :func:`mrax.core.batch.run_batch_studies` loops over study
folders, runs a shared fitting or analysis function, and exports one manifest per run.

Custom Models
-------------

Define new models in separate files using the template in ``docs/model_template.py``. Minimum requirements:

- ``model_apply(fields, model_axis, observables)`` returning a predicted data cube.
- ``parametrization_factory(spatial_shape)`` returning (params, unpack_fn).
- ``parameter_units`` covering every unpacked field.

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

Quick tutorial (custom model in a separate file)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Copy ``docs/model_template.py`` next to your code (e.g., ``my_models.py``) and edit the two required callables:

   .. code-block:: python

      # my_models.py
      import jax.numpy as jnp
      from mrax.data_struct.model import Model

      def make_my_model():
          def model_apply(fields, model_axis, observables=None):
              # model_axis/observables can be None for purely spatial models
              x = model_axis[:, None, None]
              return fields["amp"][None, ...] * jnp.exp(-x / fields["tau"][None, ...])

          def parametrization_factory(spatial_shape):
              params = {"amp_raw": jnp.zeros(spatial_shape), "tau_raw": jnp.zeros(spatial_shape)}

              def unpack(p):
                  amp = jnp.exp(p["amp_raw"])
                  tau = 1e-3 + 5.0 * jnp.exp(p["tau_raw"])
                  return {"amp": amp, "tau": tau}

              return params, unpack

          return Model(
              name="my_decay",
              model_apply=model_apply,
              parametrization_factory=parametrization_factory,
              parameter_units={"amp": "a.u.", "tau": "s"},
          )

2. Use it in code or notebooks:

   .. code-block:: python

     from my_models import make_my_model
     from mrax.core.loss_builder import run_fit, make_model_callable

     model = make_my_model()
     fields, losses, model_fn = run_fit(
         experiment,
         model,
         dataset,
         num_steps=200,
         step_size=0.02,
         return_model_callable=True,
     )
     preds = model_fn(field_arrays)  # merges study globals + observables

3. Hook it into a Study or Bruker import:

   .. code-block:: python

     custom_map = {"my_experiment_type": make_my_model}
     study, meta = load_bruker_study(root, model_map_by_type=custom_map)

The same pattern applies to synthetic studies: construct your ``Model`` first, then
pass it into ``Study`` entries or ``run_fit`` directly. For imported data, use
``model_map_by_type`` or ``recipe=`` when the automatically selected model is not the
one you want.

Documentation Build
-------------------

To build the docs locally::

   cd docs
   sphinx-build -E -b html . _build/html

Open ``_build/html/index.html`` in your browser to view the rendered pages.
