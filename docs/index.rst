MRax
====

.. raw:: html

   <section class="mrax-hero">
     <div class="mrax-hero-inner">
       <div class="mrax-hero-mark">
         <img src="_static/MRax_logo_square.png" alt="MRax logo">
       </div>
       <div class="mrax-hero-content">
         <p class="mrax-hero-title">A Python library for quantitative MRI research.</p>
         <p class="mrax-hero-copy">
           MRax provides data structures and optimization utilities for quantitative MRI
           research. It includes unit-aware observables, common signal models,
           reconstruction helpers, and portable <code>.mrax</code> study bundles.
         </p>
         <p class="mrax-hero-copy">
           MRax is intended for research use only.
         </p>
         <div class="mrax-hero-actions">
           <a class="mrax-button mrax-button-primary" href="overview.html">Start with the overview</a>
           <a class="mrax-button" href="notebooks.html">Open the notebooks</a>
           <a class="mrax-button mrax-button-github" href="https://github.com/chris-lippe/MRax">
             <i class="fa-brands fa-github" aria-hidden="true"></i>
             <span>View on GitHub</span>
           </a>
         </div>
       </div>
     </div>
   </section>

Main components
---------------

.. list-table::
   :widths: 1 3
   :header-rows: 0

   * - **Signal models**
     - Built-in model factories for common quantitative MRI equations, plus a
       template for custom models.
   * - **Named experiments**
     - ``Experiment`` and ``Quantity`` objects keep dimensions and units attached to arrays.
   * - **JAX optimization**
     - Differentiable fitting with masks, constraints and regularization, using Optax and proximal solvers.
   * - **MRI reconstruction**
     - Cartesian and non-Cartesian helpers, k-space processing blocks, variational
       solvers, and model-based parameter reconstruction.
   * - **Storage**
     - Portable ``.mrax`` study bundles for images, k-space, fitted maps and metadata,
       with NIfTI and DICOM export.

Core workflow
-------------

1. Build or import a ``Study`` from synthetic data, Pulseq, Bruker ParaVision, or a
   custom loader.
2. Pick a model for each experiment and insert any required observables from study
   globals or prior fits.
3. Plan and execute fits with :func:`mrax.core.study_runner.plan_study_fits` and
   :func:`mrax.core.study_runner.execute_study_plan`, or call
   :func:`mrax.core.loss_builder.run_fit` directly for a single experiment.
4. Inspect fitted fields, derived maps, and convergence history.
5. Save the study and results in a ``.mrax`` bundle; see :doc:`storage`. Use
   :doc:`analysis` for ROI or group summaries.

Start Here
----------

.. toctree::
   :maxdepth: 1

   overview
   guides
   models
   gui
   api
   notebooks
   references
   release_notes
   roadmap

Quick API Jump
--------------

- Data containers and studies: :ref:`api-data`
- Run single- or multi-experiment fits: :ref:`api-fitting`
- Batch processing on disk folders: :ref:`api-batch`
- Preprocessing toolbox: :ref:`api-preprocessing`
- Post-fit analysis toolbox: :ref:`api-analysis`
- Optimizers and regularizers: :ref:`api-optim`
- Reconstruction operators and solvers: :ref:`api-reco`
- Import helpers: :ref:`api-importers`

Useful Links
------------
- Install docs/test/notebook tooling: ``pip install -r requirements-dev.txt``
- Build docs locally: ``sphinx-build -E -b html docs docs/_build/html``
- Launch notebooks locally: ``jupyter lab notebooks``
