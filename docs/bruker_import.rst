Bruker Import
=============

``mrax.io.bruker_import`` reads ParaVision study folders without Bruker APIs. It
parses ``method``, ``acqp``, ``visu_pars``, and ``reco`` files; loads
``pdata/<idx>/2dseq`` images; and, when raw files are present, attaches sorted
k-space to the resulting :class:`mrax.data_struct.study.Study`.

Validation scope
----------------

The importer has been tested on ParaVision 360 3.5–3.7 data. Other ParaVision versions, local reconstruction settings, and custom
sequences can use different metadata fields or ordering rules and have not been
validated in MRax. The importer records metadata it cannot emulate under ``metadata_only`` entries when
possible.

Quick Usage
-----------

.. code-block:: python

   from pathlib import Path
   from mrax.core.loss_builder import make_model_callable, run_fit
   from mrax.io.bruker_import import load_bruker_study

   root = Path("path/to/bruker_study")
   study, meta = load_bruker_study(root, field_strength_mhz=400.325)
   print(study.describe())

   idx = study.names.index("5")
   fields, losses = run_fit(
       study.experiments[idx],
       study.models[idx],
       study.datasets[idx],
       study_globals=study.study_globals,
   )
   model_fn = make_model_callable(study.experiments[idx], study.models[idx], study.study_globals)

Pass ``pdata_index=...`` if the reconstruction is not under ``pdata/1``. A dict can
map scan numbers to different ``pdata`` indices; use ``"default"`` for the fallback
entry.

Study-wide constants such as ``field_strength_mhz`` must be provided at load time via
``field_strength_mhz=`` or ``study_globals=``. Saturation-transfer models require field
strength to convert ppm offsets.

Shared Import Schema
--------------------

Bruker parsing is limited to ParaVision files and metadata. Experiment typing,
observable extraction, k-space ordering, and construction of ``Experiment`` /
``DataSet`` objects are handled by :mod:`mrax.io.import_definitions`. This keeps the
internal data contract shared with other importers.

Common inferred experiment types are ``ir``, ``vtr``, ``vfa``, ``mse``, ``mge``,
``dti``, ``sat_trans``, ``fair_asl``, and ``anatomical``. Unknown scans load as
``anatomical`` with the identity model. Override typing with ``type_overrides``,
``model_map_by_type``, ``model_map_by_name``, or ``recipe=`` when metadata is
insufficient.

Intensity Scaling
-----------------

ParaVision ``2dseq`` images are often stored as integer arrays. ``visu_pars`` defines
the conversion to floating-point image values:

.. code-block:: text

   image_float = image_2dseq_raw * VisuCoreDataSlope + VisuCoreDataOffs

MRax applies this conversion in :func:`mrax.io.bruker_import.load_2dseq`.
``DataSet.data`` therefore contains scaled floating-point values. Applying
``VisuCoreDataSlope`` again downstream double-scales the image.

``RECO_map_slope`` and ``RECO_map_offset`` describe the vendor mapping to stored
``2dseq`` values. Integer quantization and clipping can still produce pixel-wise
differences from a raw-pipeline reconstruction.

Raw Cartesian K-Space
---------------------

When ``rawdata.job`` or ``rawdat.job`` is present with a matching ``reco`` file, MRax
reads the interleaved real/imaginary samples, sorts them, and stores reconstruction
input on ``DataSet.kspace``. For Cartesian scans this is intended to be a corrected
k-space grid, not only a reshaped raw buffer.

The importer applies k-space operations described by Bruker metadata when supported:

- ``RECO_qopts``,
- readout placement and zero filling implied by ``RECO_ft_size``,
- ``RECO_rotate`` as a phase ramp,
- ``RecoScaleChan`` and ``RecoPhaseChan``.

Image-domain formatting is kept separate and is used for the preview image in
``DataSet.processed["raw_reconstruction"]``:

- FFT normalization,
- ``RECO_size`` / ``RECO_offset`` cropping,
- sum-of-squares coil combination,
- ``RECO_transposition`` / ``VisuCoreTransposition``.

This distinction matters. Variational and model-based reconstruction starts from
corrected k-space. Vendor-image comparison may also need the image-domain formatting
steps.

Direct helpers are available for raw Cartesian data:

.. code-block:: python

   from mrax.io.bruker_import import (
       parse_jcamp,
       prepare_bruker_cartesian_kspace,
       reconstruct_bruker_cartesian_kspace,
   )

   reco = parse_jcamp(scan_dir / "pdata" / "1" / "reco")
   corrected, correction_meta = prepare_bruker_cartesian_kspace(raw_kspace, reco)
   preview = reconstruct_bruker_cartesian_kspace(
       corrected,
       reco,
       assume_corrections_applied=True,
   )

EPI K-Space
-----------

For ParaVision EPI scans, MRax stores measured pre-reconstruction k-space after
raw-space sorting and supported corrections. It does not replace missing samples with
phase half-Fourier estimates in ``DataSet.kspace``. Half-Fourier completion is used only
for preview image generation when enabled by the pipeline.

``DataSet.kspace_metadata`` records the EPI status, selected reconstruction candidate,
candidate errors, and notes about applied corrections. The path handles normal EPI,
multi-object EPI, and DTI-EPI through the same raw-space pipeline. Supported steps
include receiver unpacking, navigator/stability-sample skipping, readout reversal,
Bruker sort maps, sample-wise B0 phase terms, recorded phase corrections, channel
scale/phase correction, and selected ghost-correction approximations.

Some Bruker EPI details remain approximate or metadata-only, especially proprietary
ramp-sampling kernels, navigator-based stability correction internals, and ghost filters
outside the implemented cases. Check EPI raw reconstruction against local data.

Non-Cartesian Raw Data
----------------------

ParaVision spiral, radial UTE, 3D UTE/ZTE, and related ``RecoRegridN`` scans can store
raw samples plus normalized trajectories in files such as ``traj`` and ``trajDC``.
MRax maps those scans to the same contract used by Pulseq imports:

- ``DataSet.kspace`` has shape ``(frame, coil, sample)``,
- ``DataSet.trajectory`` has shape ``(sample, D)`` with ``D=2`` or ``D=3``,
- ``DataSet.density`` contains MRax density weights for reconstruction,
- ``DataSet.kspace_metadata["kind"]`` is ``"non_cartesian"``.

The preview uses nearest-neighbor regridding and is meant for inspection. Use
:mod:`mrax.reco.variational` for iterative reconstruction. Vendor density mode and
remaining image formatting choices are recorded in metadata where available.

Frame Geometry
--------------

Non-spatial dimensions are inferred from observables such as ``TI``, ``TE``,
``freq_offsets``, and ``sat_b1``. If the inferred product matches the frame count,
``DataSet.data`` is reshaped to ``(dims..., x, y[, z])``. Otherwise MRax keeps a single
``frame`` axis.

ParaVision 2D scans are not always slice stacks. Multi-package localizers can contain
groups of slices with different orientations. When orientation, transposition, or
``VisuFGOrderDesc`` indicates mixed geometry, MRax keeps the scan frame-major instead of
forcing ``(x, y, z)``.

TimeStamps TR
-------------

If ``TimeStamps.txt`` is present, MRax can derive per-repetition ``TR`` values from
ParaVision ADC event timing. Values are stored in milliseconds. For saturation-transfer
scans this can create one ``TR`` value per offset/time frame.

Diffusion and Saturation Transfer
---------------------------------

Diffusion scans attach ``bvals`` and ``bvecs`` when available from
``VisuAcqDiffusionBMatrix``, ``VisuAcqDiffusionGradOrient``, or ``PVM_Dw*`` metadata.
By default, b-vectors are rotated into image/voxel coordinates using
``VisuCoreOrientation``. Pass ``rotate_bvecs_to_image=False`` to keep scanner
coordinates.

Continuous-wave CEST/CESL scans are tagged as ``sat_trans``. The importer attaches
available ``freq_offsets`` (ppm), ``sat_b1`` (uT), and ``sat_time`` (ms). Choose a model
with ``model_map_by_type`` or ``recipe=`` when loading. Multiple ``sat_trans`` scans can
be combined with :func:`mrax.data_struct.join_experiments`.

Storage
-------

After import, the study is vendor-neutral. Persist it with
:func:`mrax.data_struct.storage.save_study_results`. The metadata returned by
``load_bruker_study`` can be saved for inspection. Downstream analysis uses MRax
``Study`` objects and manifests rather than Bruker-specific files.

Bruker example data are not bundled because they require local scanner studies.
