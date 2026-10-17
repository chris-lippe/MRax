Pulseq and measurement readers
==============================

Read Pulseq ``.seq`` files and pair their timing and trajectories with extracted
measurements or supplied arrays. See :doc:`pulseq_import` for file formats,
metadata overrides and reconstruction examples.

.. autoclass:: mrax.io.MeasurementData
.. autofunction:: mrax.io.read_pulseq_measurement
.. autofunction:: mrax.io.import_pulseq_study
.. autofunction:: mrax.io.read_measurement
.. autofunction:: mrax.io.register_measurement_reader

.. autoclass:: mrax.io.pulseq_import.PulseqSequenceInfo
.. autofunction:: mrax.io.pulseq_import.read_pulseq
.. autofunction:: mrax.io.pulseq_import.read_pulseq_acquisition
.. autofunction:: mrax.io.pulseq_import.read_pulseq_sidecar_metadata
.. autofunction:: mrax.io.pulseq_import.import_pulseq_experiment
.. autofunction:: mrax.io.pulseq_import.detect_pulseq_kspace_kind
