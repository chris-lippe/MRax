Notebook Gallery
================

The demo notebooks synthesize their own inputs and write temporary outputs under
``tmp/`` or notebook-local demo folders. They do not require bundled datasets.

Notebooks are rendered inline via ``nbsphinx`` with execution disabled during docs
builds. The downloads import the installed ``mrax`` package and therefore work outside
a source checkout. Install notebook tooling and the optional example backends, then
launch the directory containing the downloaded notebook:

.. code-block:: bash

   pip install "mrax[notebooks,nufft,denoise-external]"
   jupyter lab

The upstream BM3D/BM4D packages permit non-commercial use only. Method references for
the models, regularizers, reconstruction methods, and denoisers used in the notebooks
are collected in :doc:`references`.

.. toctree::
   :maxdepth: 1
   :caption: Rendered notebooks

   Getting Started <notebooks/getting_started>
   Pulseq Import Demo <notebooks/pulseq_import_demo>
   Reconstruction Pipeline <notebooks/reconstruction_pipeline>
   Variational Reconstruction <notebooks/variational_reconstruction>
   Analysis Toolbox <notebooks/analysis_toolbox>
   Preprocessing Denoiser Benchmarks <notebooks/preprocessing_denoiser_benchmarks>

Downloads
---------
- :download:`getting_started.ipynb <notebooks/getting_started.ipynb>` — synthetic pipeline
- :download:`pulseq_import_demo.ipynb <notebooks/pulseq_import_demo.ipynb>` — Pulseq Cartesian import and fitting walkthrough
- :download:`reconstruction_pipeline.ipynb <notebooks/reconstruction_pipeline.ipynb>` — raw k-space and reconstruction blocks
- :download:`variational_reconstruction.ipynb <notebooks/variational_reconstruction.ipynb>` — variational MRI reconstruction (TV + NUFFT)
- :download:`analysis_toolbox.ipynb <notebooks/analysis_toolbox.ipynb>` — post-fit analysis demo
- :download:`preprocessing_denoiser_benchmarks.ipynb <notebooks/preprocessing_denoiser_benchmarks.ipynb>` — denoiser benchmarks (metrics + images)
