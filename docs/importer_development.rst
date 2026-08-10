Importer Development
====================

MRax reconstruction code is vendor-neutral. Importers are responsible for
turning scanner-specific files into a small, stable contract:

- a frame-major image array on ``DataSet.data``
- an optional raw k-space array on ``DataSet.kspace``, sorted and corrected according
  to the importer contract
- an optional Cartesian sampling mask on ``DataSet.sampling_mask``
- metadata on ``DataSet.kspace_metadata`` that explains how the raw data was sorted and
  which vendor reconstruction steps were applied before the data entered MRax

This separation keeps :mod:`mrax.reco.pipeline` and :mod:`mrax.reco.variational`
independent from vendor-specific file formats.

Validation status
-----------------

Importer documentation includes validation scope. The Bruker path has been tested on
ParaVision 360 3.5–3.7 data. The Pulseq path supports synthetic demos and has not been
validated on measured scanner data. Tests for new import paths compare array ordering,
observables, and reconstruction previews against a known sequence or vendor output.

.. contents::
   :local:
   :depth: 2

What "Reconstruction-Ready K-Space" Means
-----------------------------------------

In MRax, reconstruction-ready k-space means:

- the raw samples have already been sorted into the correct logical acquisition order
  for the sequence
- the array uses MRax's internal axis conventions (for Cartesian 2D data:
  ``(frame, coil?, ky, kx)``)
- vendor reco metadata that acts directly in k-space has already been applied
- missing samples remain zero so the sampling pattern is still explicit

This is different from a vendor's final display image. A vendor may still apply
additional image-domain operations such as cropping, transposition, sum-of-squares coil
combination, magnitude conversion, or integer storage mapping after the FFT.

The practical consequence is:

- ``DataSet.kspace`` is suitable for direct use by FFT- or model-based reconstruction
  code
- ``DataSet.data`` can remain the vendor image, or optionally a preview reconstruction,
  as long as the raw k-space path is preserved separately

Three Separate Problems
-----------------------

When implementing a new importer, keep these three layers separate:

Sequence ordering
~~~~~~~~~~~~~~~~~

This answers: "Which sample belongs to which frame, slice, echo, or phase-encode line?"

Examples:

- Pulseq RARE/TurboSE: one shot acquires ``rare_factor`` lines from the same slice,
  then the acquisition moves to the next slice
- Bruker multi-slice scans: ``PVM_ObjOrderList`` may interleave slices or packages
- EPI: the readout polarity and line order can alternate

If this step is wrong, the k-space geometry itself is wrong and no later correction can
repair it.

Vendor k-space corrections
~~~~~~~~~~~~~~~~~~~~~~~~~~

This answers: "What metadata must be applied before the FFT so the stored raw data
represents the same k-space grid the vendor reconstructs from?"

Typical examples:

- quadrature or conjugation flags
- readout placement / partial-Fourier zero-filling
- phase ramps that implement image shifts
- per-channel complex scale factors

These corrections belong in ``DataSet.kspace`` because they change the encoded data,
not only the display image.

Image-domain formatting
~~~~~~~~~~~~~~~~~~~~~~~

This answers: "How does the vendor turn FFT output into the final image written to disk?"

Typical examples:

- cropping from FT size to output size
- per-frame transposition
- sum-of-squares coil combination
- magnitude conversion
- mapping float images into integer storage ranges

These steps are for vendor comparison and preview images; they are not part of the raw
k-space contract.

MRax Import Pipeline
--------------------

The usual implementation pattern is:

1. Parse vendor metadata into a normalized description.
2. Infer the experiment type and observables.
3. Read the vendor image product if it exists (for example Bruker ``2dseq``).
4. Read raw samples if present.
5. Sort the raw samples into a vendor-neutral k-space array.
6. Apply vendor k-space corrections.
7. Attach optional preview reconstructions and sampling masks.
8. Record the applied correction stages in ``kspace_metadata``.

Shared helpers in :mod:`mrax.io.import_definitions` already cover the experiment typing,
observable extraction, and generic k-space definition objects once the vendor metadata
has been normalized.

Bruker Case Study
-----------------

The Bruker importer is the current reference implementation. ParaVision metadata
separates sample ordering, reconstruction parameters, and final image storage.

Sample sorting metadata
~~~~~~~~~~~~~~~~~~~~~~~

For the currently supported Cartesian Bruker raw import path, the main ordering fields
are:

- ``PVM_EncSteps1``: phase-encode step order
- ``PVM_ObjOrderList``: acquisition-to-output frame order
- ``PVM_SPackArrNSlices``: package structure for multi-package localizers
- ``PVM_RareFactor``: echo train length for RARE/TurboSE

Those fields determine how MRax fills the Cartesian ``ky`` dimension and how it maps
acquisition-order slices back into natural output order.

Vendor k-space corrections applied by MRax
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For Bruker Cartesian raw data, MRax exposes public helpers in
:mod:`mrax.io.bruker_import`:

- :func:`mrax.io.bruker_import.prepare_bruker_cartesian_kspace`
- :func:`mrax.io.bruker_import.prepare_bruker_non_cartesian_kspace`
- :func:`mrax.io.bruker_import.bruker_vendor_regrid_options`
- :func:`mrax.io.bruker_import.read_bruker_regrid_density`
- :func:`mrax.io.bruker_import.read_bruker_regrid_trajectory`
- :func:`mrax.io.bruker_import.format_bruker_cartesian_image`
- :func:`mrax.io.bruker_import.reconstruct_bruker_cartesian_kspace`

The importer uses these helpers internally, so ``load_bruker_experiment(...,
load_raw_kspace=True)`` stores vendor-corrected, reconstruction-ready Cartesian
k-space on ``DataSet.kspace``.

The Bruker k-space helper applies:

- ``RECO_qopts``
- readout placement / zero-filling implied by ``RECO_ft_size``
- ``RECO_rotate`` as a phase ramp in k-space
- ``RecoScaleChan`` and ``RecoPhaseChan``

The Bruker image formatter then applies:

- FFT normalization adjustment
- ``RECO_size`` / ``RECO_offset`` cutoff
- sum-of-squares coil combination
- ``RECO_transposition`` or ``VisuCoreTransposition``

Metadata fields that are relevant for vendor comparison but do not alter
``DataSet.kspace`` are recorded in ``kspace_metadata``. This includes active
filters such as ``RECO_pc_mode`` or ``RECO_wdw_mode``.

For Bruker 2D and 3D non-Cartesian scans, the importer uses ParaVision ``RecoRegridN``
metadata and local ``traj`` files to keep samples in the same contract as Pulseq
non-Cartesian imports. PV360 ``rawdata.job*`` files are decoded using the
``ACQ_jobs`` block width, receiver count, and ``RecoSortSize``/``RecoSortMaps`` so
the final sample axis is ordered by trajectory interleave and frame. Spiral
navigator/pre/post points are discarded so the last axis of ``DataSet.kspace``
matches the stored trajectory exactly. If ParaVision records a per-sample ``b0``
correction in the reconstruction graph, MRax multiplies it into the retained sample
stream before applying receiver scale/phase corrections. Bruker's density
compensation is derived from ``RecoRegridDensCorrFilter`` for the vendor-comparison
preview (radial ``rho`` weights, spiral ``trajDC`` Jacobian path when available),
while ``DataSet.density`` remains the generic trajectory-based weighting used by the
Pulseq-style NUFFT operators. The resulting metadata records the regridding
trajectory type, MRax density source, vendor density source, vendor-style regrid
options, B0 correction, points per interleave, interleave count, trimming summary,
raw storage layout, sort maps, and any unsupported vendor-only processing.
For 3D radial scans, density estimation uses the radial ``rho^2`` convention. ZTE
scans can include ParaVision ``RecoInverseFidFilter`` profile synthesis before
regridding; MRax approximates that step by transforming the shuffled FID profiles into
the readout-frequency domain and applying the numeric readout shift before matching the
profiles to the stored trajectory. The proprietary inverse-FID regularization and
remaining FT/phase-correction details remain vendor-only metadata.

Separate scanner parsing from array operations. When a vendor step is just a numeric
k-space/readout/image operation, map the vendor
metadata to the public helpers in :mod:`mrax.reco.processing` or
:mod:`mrax.reco.variational` rather than adding importer-only array code. That keeps the
same step available to users working with Pulseq, external raw-data readers, or
synthetic arrays.

Mixed slice packages
~~~~~~~~~~~~~~~~~~~~

Not every Bruker 2D multi-slice scan is an ``(x, y, z)`` stack. Localizer-style
scans can contain multiple slice packages with different orientations. In that case,
collapsing frames into a single 3D volume destroys the geometry.

MRax therefore keeps those scans frame-major when:

- ``VisuFGOrderDesc`` indicates ``FG_SLICE``
- frame orientations differ across frames, or
- frame transposition differs across frames

This is why some Bruker datasets remain ``(frame, x, y)`` instead of being promoted to
``(1, x, y, z)``.

Using the Bruker helpers directly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mrax.io.bruker_import import (
       prepare_bruker_cartesian_kspace,
       reconstruct_bruker_cartesian_kspace,
       parse_jcamp,
   )

   reco = parse_jcamp(scan_dir / "pdata" / "1" / "reco")
   corrected_kspace, correction_meta = prepare_bruker_cartesian_kspace(raw_kspace, reco)
   vendor_like_image = reconstruct_bruker_cartesian_kspace(
       corrected_kspace,
       reco,
       assume_corrections_applied=True,
   )

Implementing a New Vendor or Sequence
-------------------------------------

When adding an importer, follow this checklist:

1. Normalize the vendor metadata into explicit experiment descriptors and observables.
2. Decide whether the sequence is standard Cartesian, RARE/TurboSE, EPI, or
   non-Cartesian, and sort raw samples accordingly.
3. Apply vendor metadata that changes the sampled signal before writing
   ``DataSet.kspace``.
4. Separate image-domain display/storage operations from the raw k-space contract.
5. Store a concise summary of applied steps in ``DataSet.kspace_metadata``.
6. Add a synthetic test for the sorting logic and validate real scanner data locally
   against an appropriate vendor image product.

When vendor reference code or scanner manuals are needed, keep those resources outside
the public MRax repository unless their licenses permit redistribution.
