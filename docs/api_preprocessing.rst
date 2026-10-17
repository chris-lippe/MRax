.. _api-preprocessing:

Preprocessing
=============

Denoising and motion-correction helpers for image stacks.

- ``denoise_filter`` applies Gaussian/median/uniform filtering on N-D arrays.
- ``regularized_denoise`` runs TV, higher-order TV, and smoothness regularizers using
  project optimizers; see [Rudin1992]_ for the TV background.
- ``denoise_bm3d`` and ``denoise_bm4d`` wrap the bundled external denoiser packages
  [Dabov2007]_ [Maggioni2013]_.
- ``linear_drift_correct_framestack`` and ``linear_drift_correct_dataset`` fit
  one linear transform-parameter model over a frame stack for long-term drift.
- ``TranslationTransform``, ``Rigid2DTransform``, ``Similarity2DTransform``, and
  ``Affine2DTransform`` provide differentiable JAX transforms for that drift fit.
- ``register_image`` and ``motion_correct_framestack`` provide SimpleITK-backed
  image registration.
- ``motion_correct_dataset`` handles MRax experiment dimensions; use
  ``register_experiment_to_reference`` to align an experiment to a reference
  experiment image dataset.
- ``resample_image`` applies a SimpleITK transform to another image on a chosen
  reference image grid.

.. autofunction:: mrax.preprocessing.denoise.denoise_filter
.. autofunction:: mrax.preprocessing.denoise.regularized_denoise
.. autofunction:: mrax.preprocessing.denoise.denoise_bm3d
.. autofunction:: mrax.preprocessing.denoise.denoise_bm4d
.. autofunction:: mrax.preprocessing.drift.linear_drift_correct_framestack
.. autofunction:: mrax.preprocessing.drift.linear_drift_correct_dataset
.. autofunction:: mrax.preprocessing.drift.apply_image_transform_stack
.. autoclass:: mrax.preprocessing.drift.TranslationTransform
.. autoclass:: mrax.preprocessing.drift.Rigid2DTransform
.. autoclass:: mrax.preprocessing.drift.Similarity2DTransform
.. autoclass:: mrax.preprocessing.drift.Affine2DTransform
.. autofunction:: mrax.preprocessing.motion.register_image
.. autofunction:: mrax.preprocessing.motion.motion_correct_framestack
.. autofunction:: mrax.preprocessing.motion.motion_correct_dataset
.. autofunction:: mrax.preprocessing.motion.register_experiment_to_reference
.. autofunction:: mrax.preprocessing.motion.resample_image
