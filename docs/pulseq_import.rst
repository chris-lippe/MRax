Pulseq import
=============

MRax reads extracted ISMRMRD/HDF5 (including ``.mrd``), Siemens ``.dat``, NIfTI and DICOM
measurements. A Pulseq ``.seq`` describes the executed sequence, not the measured
signal. Pair it with the measurement using ``sequence_path`` or a matching
filename. Archives must be extracted first.

Read, reconstruct, fit
----------------------

Import prepares samples; reconstruction is a separate, vendor-independent step.

.. code-block:: python

   import numpy as np
   from mrax.io import read_pulseq_measurement, build_measurement_experiment
   from mrax.reco import reconstruct_acquisition
   from mrax.data_struct import Study

   measured = read_pulseq_measurement("raw.h5", sequence_path="sequence.seq")
   result = reconstruct_acquisition(measured.acquisition, method="fft")
   experiment, dataset, model, metadata = build_measurement_experiment(
       measured, image_data=np.abs(result.image), override_type="mse",
   )
   study = Study([experiment], [dataset], [model])
   study.save("study.mrax")

For files containing reconstructed images, call
``import_pulseq_experiment("images.h5")`` directly. Select ``image_series`` or
``group`` through ``reader_options`` when the file contains several products.
Use ``import_pulseq_study(directory)`` to import all discoverable image series.
Its metadata contains ``raw_measurements`` for recordings awaiting reconstruction
and ``import_issues`` for skipped inputs. Invalid files warn without blocking
other experiments. ``sequence_map`` and per-file options use relative paths.

``read_pulseq_acquisition(sequence, samples, coil_axis=0)`` accepts serial arrays
with an explicit receiver axis. Without ``coil_axis``, leading axes are frames.
For DICOM-only workflows, ``load_dicom_series`` and ``load_dicom_study`` are also
available. Compressed DICOM transfer syntaxes require ``pip install "mrax[dicom-codecs]"``.
For sweeps stored as separate DICOM series, pass
``series_groups={"relaxation": [series_uid_1, series_uid_2, ...]}`` to
``load_dicom_study`` (``dicom_series_groups`` in ``import_pulseq_study``).
Selected series must share physical geometry and study; relaxation frames are
sorted by their timing. Unlisted series remain separate.
MATLAB input uses ``read_measurement(..., image_key=..., axes=...)``
or explicit raw-array keys and encoding geometry; variable names are not guessed.

Metadata and units
------------------

Pulseq timing definitions use seconds; MRax quantitative observables use
milliseconds. FOV uses metres; ``FOVUnit = mm`` declares millimetres. Large
undeclared FOV values are interpreted as millimetres with a warning. NIfTI
spatial and temporal units are respected. Image pixels retain their stored
scaling. Model amplitude initialization follows image scale and survives bundle
reloads.
Magnitude inversion-recovery imports estimate the initial T1 from the recovery
curve to avoid starting on the wrong side of the signal null.

Missing acquisition parameters must be supplied before quantitative fitting.
An anatomical image alone cannot determine a T1, T2 or diffusion model.
``metadata_overrides`` accepts :class:`mrax.io.import_definitions.NormalizedMetadata`
fields, with times in milliseconds and flip angles in degrees.
For DICOM preparation sweeps, ``import_pulseq_study`` accepts
``sequence_map={"dicom:" + series_uid: "preparation.seq"}`` and
``metadata_overrides`` with the same keys. DICOM ``.IMA`` files are discovered
alongside ``.dcm`` files. Preparation offsets must follow decoded volume order;
pair sequences explicitly and verify their acquisition parameters. A single TE
does not define a multi-echo timing vector: supply the measured echo times.

A JSON sidecar alongside the measurement can provide ``sequence_path``,
``reader_options`` and ``normalized_metadata``. Alongside a sequence,
``sequence_definitions`` can declare missing matrix, FOV or unit conventions
without editing the sequence file. Unknown normalized fields raise an error.

Prepared samples
----------------

Acquisitions retain explicit frame and receiver axes, measured-zero masks,
calibration records and geometry. Sequence coordinates determine Cartesian
ordering; ramp and fractional-grid samples keep their actual trajectories.
Recordings without a proven spatial encoding remain ADC data. Matched three-echo
EPI navigators retain polarity and precise timing for a generic phase-calibration
stage. When that calibration is unavailable, the suggested entropy stage remains
editable in the reconstruction plan.

MRD trajectory units can be selected explicitly with
``reader_options={"trajectory_units": "cycles/fov"}``; other choices are
``cycles/pixel``, ``cycles/m`` and ``radians/pixel``. Automatic interpretation
warns when assuming cycles/FOV. Use ``density_column`` for a declared quadrature
column, or supply ``density`` explicitly. The default sampled preview estimates
quadrature on a sampling mesh; these weights do not alter an iterative solver's
statistical data term. ``frame_selection`` limits raw/image loading. FFT
reconstruction automatically batches large recordings before device placement;
``frame_batch_size`` overrides the batch size while retaining complete slice and
repeat groups. Batched coil images stay in host memory; the combined image uses
the requested device. Set the batch size to zero to disable batching.

When conversion leaves placeholder acquisition counters, supply a matching MRD
metadata file through ``reader_options={"metadata_template": "metadata.h5"}``.
Its records must follow exactly the raw acquisition order. Relative paths resolve
beside the measurement. The template supplies encoding, roles and geometry;
raw sample sizes, receiver layout and discard windows are retained.
For incomplete conversions, explicit MRD reader options ``noise_records`` and
``exclude_records`` select zero-based record indices. ``sample_window=[start,
stop]`` trims imaging readouts and ``center_sample`` sets their zero-frequency
sample in original readout coordinates. ``trajectory_mode="cartesian"`` uses
declared Cartesian counters instead of stored trajectories, with a warning;
the default retains supplied trajectories. These controls require knowledge of
the conversion and never modify input files.

``prewhiten_acquisition`` in ``mrax.reco`` uses attached or explicit noise samples
before any reconstruction method and returns the transformed acquisition and
receiver matrix. Calibrate sensitivities afterwards, or apply that matrix to
external sensitivity maps with ``apply_receiver_transform``. Whitening expresses
amplitudes in noise units; use ``covariance_scale`` for differing calibration bandwidths.

FFT gives a zero-filled preview for undersampled Cartesian data. For a complete
centred calibration block, ``estimate_cartesian_sensitivities(acquisition)``
provides smooth normalized receiver maps for iterative unfolding. External maps
can also be supplied. Non-Cartesian iterative methods require the optional NUFFT
backend. Reconstruction plans remain editable; see :doc:`acquisition_reconstruction`.

Scanner planes supply image affines when the mapping to sequence axes is known.
For a nonstandard Siemens ``OrientationMapping``, provide ``gradient_directions``
through ``reader_options``: a 3-by-3 matrix whose columns map sequence gradient
axes to patient LPS directions. Otherwise orientation stays unknown with a warning.
Complete, uniformly spaced parallel slice sets become volumes; incomplete sets
retain separate frame affines. If an interpreter repeats one prototype plane,
explicit sequence ``SlicePositions`` resolve slice offsets in metres
(``SlicePositionsUnit = mm`` declares millimetres). Without them, slice positions
remain uncertain and geometry is marked partial.

Extension points
----------------

Register another extracted-file adapter with ``register_measurement_reader``.
It returns ``MeasurementData`` containing images or ``AcquisitionData`` and
normalized metadata; it must not reconstruct images or change source files.
An optional inspector lists groups or image series without reading pixels.

Observable aliases and unit scales live in ``mrax/io/definitions/pulseq.toml``.
Pass ``definition_paths`` to extend this library with additional TOML mappings.
See :doc:`notebooks/pulseq_import_demo` for an array-based example.
