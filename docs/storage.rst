Storage and ISMRMRD Export
==========================

MRax stores MR images, raw k-space, and fitted maps using the ISMRMRD
format. Because the format is single-image oriented, each dataset or map is written to its
own file and linked through a manifest.

Saving Results
--------------

Use :func:`mrax.data_struct.storage.save_study_results` after fitting:

.. code-block:: python

   from mrax.data_struct import save_study_results

   manifest = save_study_results(
       study,
       results,
       "out/study_run",
       study_name="demo",
       save_nifti=True,  # optional NIfTI export alongside MRD
   )

This writes:

- ``study_manifest.json``: compact manifest with study-level info (structured summary,
  study-wide globals such as ``field_strength_mhz``) plus pointers to per-experiment
  metadata.
- ``experiments/<name>/``: per-observable ISMRMRD slices for MR images (one file per set
  of observables, containing only ``x``, ``y`` and optional ``z`` dimensions), optional
  raw k-space slices, fitted maps, and metadata.
- ``experiments/<name>/nifti/``: optional NIfTI exports for MR images and fitted maps when
  ``save_nifti=True``.

The manifest stores plain paths and metadata, so downstream plotting or reporting code
can load the exported arrays without depending on private project scripts.

Loading and Reuse
-----------------

To load persisted data:

.. code-block:: python

   from mrax.data_struct import load_study_results, load_precomputed_fields
   from mrax.core.study_runner import execute_study_plan, plan_study_fits

   data = load_study_results(manifest)
   field_strength = data["study_globals"].get("field_strength_mhz")
   cached = load_precomputed_fields(manifest)
   plan = plan_study_fits(study, precomputed_fields=cached, target_names=["exp_b"])
   results, _ = execute_study_plan(plan, run_fit_kwargs={"num_steps": 20})

The manifest points to per-experiment ``metadata.json`` files and embeds a structured
summary inline. ``load_study_results`` reloads the arrays and metadata referenced by
the manifest.

Folder Layout
-------------

::

   study_run/
     study_manifest.json
     experiments/
       exp_a/
         mr_images/           # MRD files contain both image (group='image') and k-space (group='kspace')
           mr_image_offset00.mrd
           mr_image_offset01.mrd
           ...
         nifti/               # optional if save_nifti=True
           mr_images/
             mr_image_1.nii
             ...
           fitted_maps/
             amplitude.nii
         fitted_maps/
           amplitude.mrd
           gamma.mrd
         metadata.json
       exp_b/
         ...
