.. _api-storage:

Study storage
=============

``Study.save(path)`` saves inputs to a portable ``.mrax`` bundle.
``save_study_results(study, results, path)`` includes fitted outputs and also
defaults to a bundle. Pass ``bundle=False`` for a manifest folder.
``load_study_results`` and ``load_precomputed_fields`` accept either format.
See :doc:`storage` for examples and image exports.

.. autofunction:: mrax.data_struct.storage.save_study_results
.. autofunction:: mrax.data_struct.storage.load_study_results

.. autofunction:: mrax.io.export_image_files
