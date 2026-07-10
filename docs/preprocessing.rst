Preprocessing Toolbox
=====================

Preprocessing functions live in :mod:`mrax.preprocessing`. They cover filtering,
regularized denoising, frame-stack drift correction, and SimpleITK-backed
registration.

Filtering
---------

``denoise_filter`` applies Gaussian, median, or uniform filters to N-D arrays:

.. code-block:: python

   import numpy as np
   from mrax.preprocessing import denoise_filter

   noisy = np.random.default_rng(0).normal(size=(32, 32))
   smooth = denoise_filter(noisy, method="gaussian", sigma=1.2)

Regularized Denoising
---------------------

``regularized_denoise`` solves
:math:`\tfrac12 \|x - y\|^2 + \lambda R(x)` with MRax optimizers. Available
regularizers include TV, higher-order TV, and smoothness penalties.

TV follows Rudin, Osher, and Fatemi [Rudin1992]_.

.. code-block:: python

   from mrax.preprocessing import regularized_denoise

   denoised = regularized_denoise(
       noisy,
       regularizer="tv",
       weight=0.15,
       num_steps=60,
       step_size=0.2,
       optimizer_name="adam",
   )

Pass ``axes`` to choose regularized dimensions and ``mask`` to restrict the penalty.

BM3D and BM4D (optional, non-commercial license)
-------------------------------------------------

``denoise_bm3d`` and ``denoise_bm4d`` call the ``bm3d`` and ``bm4d`` packages included
in the optional ``denoise-external`` extra. Install it with
``pip install "mrax[denoise-external]"``. The upstream packages allow non-commercial
use only; they are not covered by MRax's MIT license. BM3D follows Dabov et al.
[Dabov2007]_; BM4D follows Maggioni et al. [Maggioni2013]_.

Drift Correction
----------------

The MRax drift optimizer fits one low-dimensional transform-parameter curve over a
frame stack. It is intended for slow frame-to-frame drift, not arbitrary motion.

.. code-block:: python

   from mrax.preprocessing import linear_drift_correct_dataset

   result = linear_drift_correct_dataset(
       experiment,
       dataset,
       dimensions=("t",),
       max_parameter_change=8.0,
   )

The default transform is N-D translation. ``Rigid2DTransform``,
``Similarity2DTransform``, and ``Affine2DTransform`` are also available. Metrics include
normalized cross-correlation, mutual information, and negative mean squared error.
Transforms and metrics used by the drift objective must be JAX-compatible.

Use ``order="linear"`` for differentiable fitting. Use ``order="nearest"`` when
applying fitted transforms to label-like or intensity-sensitive data.

SimpleITK Registration
----------------------

SimpleITK-backed registration is available through the default MRax install. The
high-level wrapper returns a new dataset in the original experiment layout and stores
the transform plus optimizer diagnostics on the result object:

.. code-block:: python

   from mrax.preprocessing import motion_correct_dataset

   result = motion_correct_dataset(
       experiment,
       dataset,
       dimensions=("t",),
   )
   corrected_dataset = result.dataset

When ``dimensions`` is omitted, all non-spatial dimensions are collapsed into one frame
stack in experiment order. Passing a subset, such as ``("t",)`` for an experiment with
temporal repeats and additional acquisition axes, runs separate registrations for each
remaining dimension combination and expands the corrected data back to the original
layout.

``register_experiment_to_reference`` estimates one transform between selected images
from two experiments and resamples all moving frames onto the reference grid.

``register_image`` uses a rigid Mattes mutual-information setup by default. To use a
different setup, pass a configured ``SimpleITK.ImageRegistrationMethod`` or a factory
that receives fixed and moving SimpleITK images and returns the configured method.

Notebook
--------

- :doc:`notebooks/preprocessing_denoiser_benchmarks`
