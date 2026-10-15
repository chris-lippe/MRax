Importer development
====================

An importer prepares :class:`mrax.data_struct.AcquisitionData` and optional image
and experiment containers. Reconstruction uses the conventions in
:doc:`acquisition_reconstruction` without reading scanner metadata.

Import steps
------------

1. Decode numeric types, records, padding and auxiliary ADC windows.
2. Resolve frame, receiver and encoding-event order.
3. Normalize Fourier signs, polarity, units and calibration.
4. Attach measured support, geometry and reconstruction settings.
5. Read existing scanner images independently of raw reconstruction.

``DataSet.kspace``, trajectory, density and sampling masks mirror the acquisition.
Use ``AcquisitionData.select_frames`` to select output frames and their associated
native measurements.

Extending Bruker definitions
----------------------------

``mrax/io/definitions/bruker.toml`` selects storage, event, readout and product
resolvers. Predicates support values, lists, existence, prefixes, numeric ranges
and version bounds. Multiple matches raise an error; select a compatible profile
explicitly when needed.

Reuse resolvers for the same encoding representation. Add one only for a new
representation: coupled phase/partition event tables, for example, differ from
two independent nested phase tables. Preserve declared dimensions and units;
broadcast only over the declared repetition/object axes.

Modules separate parsing (``bruker_parameters``), images/geometry
(``bruker_images``), experiment metadata (``bruker_metadata``), sampling hints
(``bruker_sampling``) and raw records/events (``bruker_storage``).
``bruker_acquisition`` assembles acquisitions; ``bruker_graph`` resolves declared
window applications; ``bruker_import`` builds experiments and studies.
Numerical algorithms belong in ``mrax.reco``.

Validation
----------

Reject incomplete records, inconsistent counts, ambiguous definitions and unknown
coordinate conventions. Study import warns and skips failed scans while retaining
valid experiments and failure status.

Use synthetic public tests for byte accounting, event order, measured zeros,
geometry, absolute scaling, complex adjoints, archive round trips and model layout
inversion. Compare artifacts, contrast and noise when algorithms differ. Keep
private measurements and scanner resources outside public fixtures and docs.
