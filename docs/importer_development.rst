Importer architecture
=====================

This page describes extension points for contributors. For loading measurements,
see the :doc:`bruker_import`, :doc:`pulseq_import` and :doc:`dicom_import` guides.

Importers translate file records and scanner metadata into
:class:`mrax.data_struct.AcquisitionData`, images and experiment observables.
They resolve record order, receiver calibration, Fourier signs, units and geometry
before reconstruction. Scanner images are decoded independently of raw samples.
Numerical reconstruction then uses the :ref:`api-acquisition-conventions` without
reading scanner metadata.

Extending Bruker definitions
----------------------------

``mrax/io/definitions/bruker.toml`` maps acquisition settings to storage, event,
readout and product resolvers. Its predicates support values, lists, existence,
prefixes, numeric ranges and version bounds. ``definition_paths`` loads additional
TOML files; ``profiles`` selects a named definition. Ambiguous matches raise an error.

A resolver implements an encoding representation, so several sequences can share
one. A new representation, such as a coupled phase/partition event table, needs
different decoding from independent nested phase tables.

The implementation separates parsing (``bruker_parameters``), images/geometry
(``bruker_images``), experiment metadata (``bruker_metadata``), sampling hints
(``bruker_sampling``) and raw records/events (``bruker_storage``).
``bruker_acquisition`` assembles acquisitions; ``bruker_graph`` resolves declared
window applications; ``bruker_import`` builds experiments and studies.
Numerical algorithms are implemented in ``mrax.reco``.

Extracted-file readers and Pulseq metadata
------------------------------------------

``register_measurement_reader(name, extensions, reader, inspector=...)`` adds an
extracted-file reader. The reader returns ``MeasurementData`` with images or an
``AcquisitionData`` object and normalized metadata. An optional inspector lists
groups or image series without loading pixels. Image reconstruction remains a
separate operation.

``mrax/io/definitions/pulseq.toml`` maps observable aliases and unit scales.
``definition_paths`` adds TOML mappings for additional sequence metadata.

Executed Pulseq coordinates establish physical trajectory units and gradient-channel
order. Automatic Cartesian assignment cross-checks sequence trajectories, stored
coordinates and encoding counters; fractional samples retain their trajectories.
EPI navigator records retain polarity and ADC timing for reconstruction calibration.

DICOM metadata adapters
-----------------------

``mrax.io.dicom`` handles discovery, pixel decoding, scaling, geometry and standard
MR observables. ``mrax.io.dicom_bruker`` supplements these with Bruker-specific
metadata. Other origins can implement ``DicomMetadataAdapter`` and return a
``DicomMetadataPatch``; adapters are passed through ``metadata_adapters`` in
``load_dicom_series`` or ``load_dicom_study``.

Import failures
---------------

Incomplete records, inconsistent counts, ambiguous definitions and unknown
coordinate conventions raise import errors. Study import records failed scans
in ``metadata['import_issues']`` and warns while retaining valid experiments.
