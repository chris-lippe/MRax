MRax Study Storage
==================

A ``.mrax`` bundle is the standard portable study file. It contains images,
prepared raw acquisitions, fitted maps, units and geometry. GUI saves may also
include view state; Python API saves contain the study data.

Saving and loading
------------------

.. code-block:: python

   from mrax.data_struct import save_study_results, load_study_results

   inputs = study.save("out/inputs.mrax")
   fitted = save_study_results(study, results, "out/fitted.mrax")
   data = load_study_results(fitted)

A directory passed to ``save_study_results`` receives ``<study_name>.mrax``.
``progress_callback(current, total, message)`` reports writing and compression.
``load_precomputed_fields`` also accepts bundles for cached-fit reuse.

For an editable manifest/ISMRMRD folder, pass ``bundle=False``. The loader accepts
both formats. Bundles keep the same relative metadata and array layout internally;
geometry stays separate from pixel values. ``save_nifti=True`` includes optional
NIfTI exports in the saved study.

Image export
------------

.. code-block:: python

   from mrax.io import export_image_files

   paths = export_image_files(
       "out/image.nii", dataset.data, dims=experiment.dims,
       geometry=dataset.spatial_geometry,
   )
   dicom_paths = export_image_files(
       "out/image.dcm", dataset.data, dims=experiment.dims,
       geometry=dataset.spatial_geometry, format="dicom",
   )

NIfTI export writes one spatial image per frame, with dimension indices in filenames.
Singleton non-spatial axes produce a single file; singleton spatial axes are retained.
DICOM export writes one derived MR image series per frame, with geometry and 16-bit pixels plus
rescale slope/intercept. Floating-point values are quantized; patient identifiers
are empty. Select a real component before exporting complex images.
