Overview
========

MRax is a JAX-based toolkit for quantitative MRI research. It provides:

- **Experiments** to describe acquisition geometry, observables, and fitted maps.
- **Models** to encode signal behavior and parameterizations.
- **Studies** to coordinate multiple experiments that may depend on one another.
- **Optimization utilities** for voxelwise and spatially regularized fitting.
- **Study bundles** to save images, prepared k-space, fitted maps and metadata in
  portable ``.mrax`` files.

.. toctree::
   :maxdepth: 1

   setup
   docker_setup
   windows_setup
   contributing
   importer_development

Getting Started
---------------

1. Install MRax with Python 3.12 or newer, preferably in a virtual environment::

     pip install mrax

2. Launch the desktop GUI to import measurements, reconstruct images and fit maps::

     mrax-gui

3. Follow :doc:`gui` for interactive analysis, or :doc:`guides` for Python workflows.
   The :doc:`notebooks` provide worked examples with synthetic data.

.. note::

   **GPU acceleration requires a Linux environment** with CUDA-enabled JAX.
   Windows users can run the Linux installation inside WSL 2 or a Linux container
   through WSL 2. Native Windows installations run calculations on CPU because
   JAX has no native Windows GPU backend. See :doc:`windows_setup` and
   :doc:`docker_setup`.

For editable installations, containers and contributor checks, see :doc:`setup`.

Key Modules
-----------

- ``mrax.data_struct``: data containers for experiments, datasets, models, studies, and storage.
- ``mrax.core.loss_builder``: utilities to build and run fitting loops over experiments and studies.
- ``mrax.opt``: optimization primitives, regularizers, and parameterizations.
- Built-in model equations and bounds: :doc:`models`.

Fitting Options
---------------

``run_fit`` accepts Optax optimizers plus ``split_bregman`` and ``fista`` variants.
It supports proximal regularizers, parameter bounds, explicit initial values, masks,
and per-parameter regularizer settings. Use ``plan_study_fits`` and
``execute_study_plan`` for multi-experiment studies.

Unified Pipeline
----------------

Synthetic examples and imported studies use the same ``Study`` abstraction. Build
synthetic datasets with :func:`mrax.core.synthetic.make_synthetic_study`, import
standards-based DICOM series or folders with :func:`mrax.io.dicom.load_dicom_study`,
or import ParaVision scans with :func:`mrax.io.bruker_import.import_bruker_study`.
Importers map metadata into a shared schema (see :mod:`mrax.io.import_definitions`).

Dimension names follow a consistent convention: ``x/y/z`` are spatial axes and ``t``
is temporal. Anatomical orientation and voxel spacing are separate geometry metadata,
so they do not change reconstruction or fitting array order. Other non-spatial axes are
inferred from observables when the metadata is unambiguous. Otherwise, MRax keeps a
single ``frame`` axis.

For multi-scan datasets collected with related acquisition settings, use
:func:`mrax.data_struct.join_experiments` to stack them into a single frame-major
experiment. The helper keeps a per-source frame index so you can fit subsets as
needed.

For multiple subjects, :func:`mrax.core.batch.run_batch_studies` loops over study
folders, runs a shared fitting or analysis function, and exports one manifest per run.

For custom forward models and a worked example, see :doc:`custom_models`.
