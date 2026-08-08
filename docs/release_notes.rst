Release Notes
=============

Version 0.2.0
-------------

Released 2026-08-08.

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
- Producer-version checks diagnose unsupported tagged data-format lines; they do not
  reject untagged legacy MRax files without first attempting to load them.
- The DICOM layer imports metadata that was actually exported. Vendor-only acquisition
  fields absent from a DICOM object cannot be reconstructed from it.

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
- The Pulseq importer is included for demo workflows and has not been validated on
  measured scanner data.
- ``jax-finufft`` currently has no Windows wheel. Install ``mrax[nufft]`` only on a
  supported platform or in the documented Linux development container.
- The upstream BM3D/BM4D packages in ``mrax[denoise-external]`` permit
  non-commercial use only and are not covered by the MRax MIT license.
