Pulseq Import
=============

The module ``mrax.io.pulseq_import`` reads Pulseq ``.seq`` files and exposes their
definitions, ADC sample timing, and calculated k-space trajectory. Pulseq files do
not contain measured signal data, so :func:`mrax.io.pulseq_import.import_pulseq_experiment`
expects either measured ``kspace`` samples or reconstructed ``image_data``.

Validation scope
----------------

The Pulseq importer is implemented for synthetic demos and importer-development
experiments. It has not been validated on measured scanner data. The generated
trajectory, k-space reshaping, and density compensation are development outputs.

Pulseq support is included in the release package:

.. code-block:: bash

   pip install mrax

Quick usage
-----------

.. code-block:: python

   import numpy as np
   from mrax.io import read_pulseq, read_pulseq_acquisition, import_pulseq_experiment
   from mrax.reco import reconstruct_acquisition

   info = read_pulseq("sequence.seq")
   print(info.sequence_kind, info.spatial_shape, info.adc_counts)

   kspace = acquire_or_simulate_kspace(info)
   acquisition = read_pulseq_acquisition("sequence.seq", kspace, spatial_shape=info.spatial_shape)
   image = reconstruct_acquisition(acquisition).image
   experiment, dataset, model, metadata = import_pulseq_experiment(
       "sequence.seq",
       kspace=kspace, image_data=image,
       spatial_shape=info.spatial_shape,
   )

   if dataset.kspace_metadata["kind"] == "non_cartesian":
       print(dataset.trajectory.shape, dataset.density.shape)

Notebook walkthrough
--------------------

For an end-to-end example that generates a Cartesian Pulseq sequence, simulates
k-space, imports it through :func:`mrax.io.pulseq_import.import_pulseq_experiment`,
and fits a T2 model, see:

- Rendered notebook: :doc:`notebooks/pulseq_import_demo`
- Notebook download: :download:`pulseq_import_demo.ipynb <notebooks/pulseq_import_demo.ipynb>`

K-space conventions
-------------------

Pulseq trajectories are stored on ``DataSet.trajectory`` as ``(num_samples, 2)``
with columns ``(kx, ky)`` in cycles/pixel. The unnormalized trajectory returned by
``pypulseq`` is kept in the loader metadata under ``metadata["pulseq"]["trajectory_physical"]``.

EPI acquisitions are detected from alternating readout directions. Serial EPI ADC
samples are reshaped to ``(..., ky, kx)`` and odd readout lines are reversed so
``DataSet.kspace`` has consistent kx ordering. Non-Cartesian acquisitions keep
the sample axis last, with matching density compensation on ``DataSet.density``.

Non-Cartesian matching
----------------------

For radial, spiral, and other non-Cartesian trajectories, the importer keeps
``DataSet.kspace`` as ``(..., num_samples)`` and stores the matching trajectory on
``DataSet.trajectory``. The sample stream can be supplied in several forms:

- One trajectory pass per frame, already shaped as ``(..., num_samples)``.
- A flattened serial stream that concatenates multiple frames with the same
  trajectory.
- A flattened serial stream where each frame contains several repeated trajectory
  passes, for example interleave groups or repeated shot blocks.

``import_pulseq_experiment`` resolves these cases by combining the Pulseq ADC
trajectory with frame-count hints from:

- Pulseq definitions such as ``TE``, ``TR``, ``TI``, or ``FlipAngle``.
- Optional JSON sidecars next to the ``.seq`` file, read with
  :func:`mrax.io.pulseq_import.read_pulseq_sidecar_metadata`.
- Explicit arguments such as ``sample_axis`` when the sample dimension is not last.

Useful sidecar keys for non-Cartesian imports are:

- ``frame_count`` / ``num_frames`` / ``frames`` for flattened multi-frame streams.
- ``trajectory_repetitions_per_frame`` for repeated trajectory passes per frame.

``read_pulseq_acquisition`` prepares measured samples independently of images.
Call :func:`mrax.reco.reconstruct_acquisition` with ``method="fft"`` for a gridding
preview or ``method="variational"`` for iterative reconstruction. Supply the
resulting image explicitly when constructing an experiment. ``DataSet.data``
contains the supplied image and ``DataSet.acquisition`` retains raw measurements.

Prepared acquisitions use the frame, receiver and encoding-axis conventions in
:doc:`acquisition_reconstruction`. The importer converts the sequence descriptor's
``(kx, ky)`` coordinates to that order and preserves measured zeros.
