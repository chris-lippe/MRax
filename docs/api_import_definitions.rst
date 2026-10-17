Import metadata and definitions
===============================

Normalize vendor metadata, infer experiment types, extract observables, and materialize
vendor-neutral experiments/datasets from importer payloads.

.. autoclass:: mrax.io.import_definitions.NormalizedMetadata
.. autoclass:: mrax.io.import_definitions.ImportDefinition

.. autofunction:: mrax.io.import_definitions.build_imported_experiment_dataset
.. autofunction:: mrax.io.import_definitions.detect_experiment_type
.. autofunction:: mrax.io.import_definitions.extract_observables_from_metadata
.. autofunction:: mrax.io.import_definitions.reshape_epi_samples
