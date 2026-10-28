Release Notes
=============

Version 0.3.2
-------------

Highlights
~~~~~~~~~~

- Unified Optax, FISTA and split-Bregman dispatch through the shared fitting
  executor. FISTA and split Bregman support native TV/box proximal steps.
- Added certified FISTA backtracking and restart, and augmented-primal split
  Bregman with safeguarded Newton/CG and L-BFGS subproblem solves.
- Added optimizer-independent custom data losses and grouped pseudo-Huber loss.
- Fixed GUI warnings.


Version 0.3.1
-------------

Highlights
~~~~~~~~~~

- Improved FISTA and split Bregman fitting and reconstruction stability.
- Added optional active-ROI parameter packing,
  gradient clipping and an explicit ``masked_mean_smoothed`` TV formulation.
- Improved custom proximal callbacks with box constraints, mixed-precision
  fitting and checkpoint continuation.

Version 0.3.0
-------------

Highlights
~~~~~~~~~~

- Unified Bruker and Pulseq import with editable plans for FFT, variational and
  model-based reconstruction.
- Bruker import now includes PV 360 3.x and PV 6.0.1 support.
- Expanded Pulseq support for MRD, Siemens, NIfTI and DICOM measurements, with
  improved quantitative timing and frame alignment.
- Built-in CPU/GPU JAX NUFFT replaces ``jax-finufft`` and the ``nufft`` extra,
  with bounded workspace memory.
- Improved FISTA and split Bregman optimizers. TV regularization selects FISTA
  automatically; explicit Adam uses smoothed TV with a warning.
- Fitting defaults to CPU and ROI-based normalization, with explicit device
  selection. See :doc:`models` for regularizer-weight adjustments.
- Reference-image forward fitting supports conventional and model-based fits,
  with consistent loss scaling and reduced model-based memory use.
- Portable ``.mrax`` bundles are the default study format. GUI saves show progress;
  exports support multi-frame NIfTI and DICOM.
- GUI import detects Bruker or Pulseq folders automatically. Improved plot axes,
  elapsed-time display, ROI fitting and drawing, and GPU startup memory use.

API changes: replace ``load_bruker_*`` with ``import_bruker_*``. Pulseq raw import
uses ``read_pulseq_acquisition``. Reconstruct explicitly with
``reconstruct_acquisition`` or ``ReconstructionPipeline``; these replace
``RecoPipeline``/``RecoState`` and the separate FFT/regridding helpers.
Configure reconstruction and cropping in the plan; import options
``prefer_raw_image``, ``reco_pipeline`` and ``slice_crop_mode`` are removed.
Low-level solvers are ``solve_variational`` and ``solve_model_based``.

Version 0.2.1
-------------

Bug fixes
~~~~~~~~~

- Fixed Qt popup and transient-window handling when MRax runs through VS Code's
  forwarded Wayland socket. MRax now uses the accompanying X11/XWayland display
  in that environment, allowing combo-box menus to close normally, dialogs to be
  dismissed, and parameter-table values to be edited reliably.

Version 0.2.0
-------------

Highlights
~~~~~~~~~~

- Added standards-based single-series and recursive-folder DICOM import, with a
  vendor-neutral core and an isolated Bruker metadata adapter. Quantitative frame
  observables and spatial/non-spatial dimensions are inferred from standard and
  enhanced DICOM metadata when available.
- Added explicit spatial geometry to MRax experiments. Array axes retain fixed
  ``x/y/z`` meaning while affine orientation and voxel spacing remain separate for
  anatomical plane display and NIfTI export.
- Added axial, coronal, and sagittal GUI views with geometry-aware aspect ratios,
  plus reference-aware NIfTI ROI export.
- Added combined and per-coil magnitude/phase reconstruction views and exports.
  Phase images use a fast JAX-native, magnitude-weighted unwrapping implementation;
  wrapped phase remains available.
- Added variance-aware fitting and diagnostics for quantitative experiments.
- Added a global package/data version source. MRax manifests, bundles, ROI sets, and
  GUI state now record their producer version and spatial geometry, while MRD image
  payloads remain unchanged. Untagged legacy files are still loaded with a warning.
- Improved GUI DICOM folder import, window sizing, view state, and reconstruction
  controls.

Compatibility notes
~~~~~~~~~~~~~~~~~~~

- DICOM import requires ``pydicom``, which is now a base dependency.
- Producer-version checks diagnose tagged data-format compatibility and attempt
  loading untagged legacy MRax files.
- The DICOM layer imports the exported metadata. Additional vendor acquisition
  fields require a separate source.

Version 0.1.0
-------------

Initial MRax release.

Highlights
~~~~~~~~~~

- JAX-based fitting utilities for quantitative MRI experiments.
- Unit-aware ``Experiment``, ``DataSet``, ``Quantity``, and ``Study`` containers.
- Built-in model factories for common quantitative MRI workflows.
- ISMRMRD-backed manifests for images, k-space, fitted maps, units, metadata, and optional NIfTI exports.
- Bruker ParaVision and Pulseq import helpers.
- Cartesian and non-Cartesian reconstruction utilities.
- ROI, timecourse, group-summary, and batch-analysis helpers.
- Desktop GUI for inspecting studies, managing ROIs, and running interactive fits.
- A portable base installation, with NUFFT and non-commercial BM3D/BM4D backends in
  explicit optional extras.

Notes
~~~~~

- Bruker import has been tested on ParaVision 360 3.6 data.
- Initial Pulseq coverage used demo workflows.
- Initial NUFFT support used the optional ``jax-finufft`` backend on Linux;
  :ref:`api-nufft-plans` describes the current built-in implementation.
- The upstream BM3D/BM4D packages retain their non-commercial licensing terms.
