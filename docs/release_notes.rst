Release Notes
=============

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
