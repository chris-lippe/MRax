Guides and Workflows
====================

Choose a workflow below. Section Navigation links to pages in the current
section; notebooks accompany the relevant workflows.

Core pipeline
-------------

.. toctree::
   :maxdepth: 1

   acquisition_reconstruction
   reconstruction
   preprocessing
   variance_weighting
   studies
   batch
   storage
   analysis

Import Workflows
----------------

.. toctree::
   :maxdepth: 1

   dicom_import
   bruker_import
   pulseq_import

- DICOM import supports individual series and recursive study folders, with
  identifiers excluded from returned metadata by default.
- Bruker import reads ParaVision study folders and standalone experiment exports.
- Pulseq walkthrough notebook: :doc:`notebooks/pulseq_import_demo`
- Pulseq notebook download: :download:`pulseq_import_demo.ipynb <notebooks/pulseq_import_demo.ipynb>`

Quantitative mapping
--------------------

.. toctree::
   :maxdepth: 1

   quantitative_fits

Interactive Notebooks
---------------------

- Rendered notebook: :doc:`notebooks/getting_started`
- Notebook (download): :download:`getting_started.ipynb <notebooks/getting_started.ipynb>`
