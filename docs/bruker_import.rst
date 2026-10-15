Bruker import
=============

Import ParaVision images, raw acquisitions, geometry and experiment observables
without scanner software. Raw reconstruction is a separate operation.

.. code-block:: python

   from mrax.io import import_bruker_study
   from mrax.reco import reconstruct_acquisition

   study, metadata = import_bruker_study(study_directory)
   dataset = study.datasets[0]
   result = reconstruct_acquisition(dataset.acquisition, device="cpu")

``import_bruker_study`` returns ``(study, metadata)``;
``import_bruker_experiment`` returns ``(experiment, dataset, model, metadata)``.
Study folders and standalone exports are accepted, including nonnumeric experiment
names. ``dataset.data`` holds the scanner image; ``dataset.acquisition`` holds
prepared measurements. Set ``load_raw_kspace=False`` for image-only import.

Study import warns and skips incomplete or unsupported experiments while retaining
valid scans. Images remain available when raw data are missing; metadata reports
raw availability. Unknown layouts are rejected instead of guessed. Supply
``field_strength_mhz`` for models that require the scanner field strength.

Geometry and raw data
---------------------

Image import applies byte order, slope and offset, and retains frame grouping,
voxel spacing and physical affines. Differently oriented slice packages keep
per-frame geometry. Reconstruction uses the declared image layout and slice order.

Raw import resolves FID/job records, receiver calibration, encoding events,
polarity and nonuniform coordinates. Measured support includes measured zeros;
auxiliary ADC windows are retained. ADC-only data require spatial encoding before
MRI reconstruction. See :doc:`acquisition_reconstruction` for conventions.

Custom definitions
------------------

The TOML library selects storage, event, coordinate and product resolvers from
acquisition settings. ``definition_paths`` adds definitions; ``profiles`` selects
a compatible named definition. See :doc:`importer_development` to extend it.

The GUI reconstructs when a reconstruction view is requested. Scripted and GUI
workflows both use ``reconstruct_acquisition``.
