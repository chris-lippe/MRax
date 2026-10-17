.. _api-batch:

Batch execution and runtime
===========================

Utilities for looping over many studies on disk and exporting consistent manifests.

- ``discover_study_dirs(root, selected=None)`` returns study folder paths (optionally
  filtered by name) with helpful error messages when nothing is found.
- ``run_batch_studies(studies_root, study_loader, output_root, ...)`` iterates through
  discovered study folders, converts each to a ``Study`` via ``study_loader``, runs
  analysis (default: study planner + executor), then calls ``save_study_results``.

.. autofunction:: mrax.core.batch.discover_study_dirs
.. autofunction:: mrax.core.batch.run_batch_studies

Runtime helpers
---------------

.. autofunction:: mrax.core.runtime.release_memory
