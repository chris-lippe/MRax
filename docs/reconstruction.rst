Reconstruction
==============

MRax reconstruction functions consume generic arrays: k-space, sampling masks,
trajectories, density weights, and coil sensitivity maps. Importers are responsible for
parsing vendor metadata and sorting raw data into that contract. See
:doc:`importer_development` for the importer contract.

Use reconstruction outputs according to how they were produced. Quick regridding and
zero-filled FFT outputs are for inspection and debugging. For quantitative analysis, use
reconstruction settings checked against the acquisition and site.

Inputs
------

Reconstruction code assumes that geometry already matches the data:

- Cartesian ``DataSet.kspace`` is sorted on the intended ``ky``/``kx`` grid.
- Non-Cartesian ``DataSet.trajectory`` is in the same sample order as
  ``DataSet.kspace``.
- Corrections that act in k-space are applied before image or
  parameter map reconstruction.

Image-domain operations such as output cropping, magnitude conversion, coil
combination, and transposition are separate. They support display or vendor comparison;
they are not corrected raw k-space.

Phase images
------------

``unwrap_phase_image(phase, spatial_ndim=...)`` unwraps one to three trailing spatial
axes independently for every leading frame, echo, coil, or observable index. Supplying
``spatial_ndim`` is important whenever the array contains non-spatial leading axes.
MRax reconstruction views use this helper automatically, while wrapped phase remains
available for diagnostics and export. Two- and three-dimensional data use a JAX-native
least-squares cosine-transform solution. Reconstruction views additionally supply image
magnitude as reliability weights to a bounded preconditioned conjugate-gradient
refinement, concentrating accuracy in supported anatomy without letting noisy
background phase create line artifacts [GhigliaRomero1994]_. One-dimensional data use
direct Itoh integration [Itoh1982]_. The returned phase always remains exactly
equivalent to its input modulo ``2*pi``.

Callers with a complex image can use ``weights=abs(image)``. Without weights, the fast
unweighted cosine-transform solution is returned.

.. [GhigliaRomero1994] D. C. Ghiglia and L. A. Romero, "Robust two-dimensional
   weighted and unweighted phase unwrapping that uses fast transforms and iterative
   methods," *JOSA A* 11(1), 107-117 (1994).
   https://doi.org/10.1364/JOSAA.11.000107

.. [Itoh1982] K. Itoh, "Analysis of the phase unwrapping algorithm," *Applied
   Optics* 21(14), 2470 (1982). https://doi.org/10.1364/AO.21.002470

Processing Blocks
-----------------

Low-level helpers in :mod:`mrax.reco.processing` are metadata-free. The caller is
responsible for supplying sequence or vendor metadata.

``apply_channel_scale_phase(data, scale=..., phase_degrees=..., coil_axis=...)``
  Applies receiver-channel scale and phase factors.

``apply_samplewise_correction(data, correction, sample_axis=-1)``
  Multiplies a k-space stream by one complex value per acquired sample.

``estimate_radial_density(trajectory, spatial_shape=..., power=None)``
  Returns radial density weights for previews and synthetic examples. Use sequence- or
  vendor-provided density weights for measured data when available.

``fid_profiles_to_kspace(profiles, transform=..., output_points=..., readout_shift=...)``
  Converts time-domain FID profiles into frequency-domain profiles for ZTE-like or
  custom readouts.

``crop_along_axis`` / ``readout_crop_bounds``
  Crop oversampled readout axes.

``apply_readout_phase_shift``
  Applies a fractional readout phase ramp.

Example:

.. code-block:: python

   from mrax.reco import apply_channel_scale_phase, apply_samplewise_correction, estimate_radial_density

   kspace = apply_channel_scale_phase(kspace, scale=receiver_scale, phase_degrees=receiver_phase, coil_axis=1)
   kspace = apply_samplewise_correction(kspace, b0_vector, sample_axis=-1)
   density = estimate_radial_density(trajectory, spatial_shape=(128, 128))

Regridding
----------

:func:`mrax.reco.variational.reconstruct_regrid` provides a non-Cartesian preview path
for 2D and 3D trajectories in cycles/pixel. By default it applies density
compensation, nearest-neighbor gridding, centered IFFT, and coil combination. Vendor
image-formatting steps are opt-in:

``grid_shape``
  Reconstruct on an oversampled grid before cropping.

``crop_offset``
  Select the crop origin inside the gridded image.

``phase_correction="checkerboard"``
  Apply a parity phase after the IFFT.

``apodization_correction=RegridApodizationCorrection(...)``
  Apply separable Kaiser-Bessel deapodization after cropping. Use only with a matching
  gridding kernel/correction pair.

``image_mask="ellipse"`` or an explicit mask
  Apply a support mask before coil combination or magnitude output.

``coil_combine="sum" | "rss" | "sense"``
  Select complex sum, root-sum-of-squares, or sensitivity-weighted combination.

``magnitude=True``
  Convert output to magnitude. Use complex output when phase is part of the model.

Cartesian and Non-Cartesian Sketches
------------------------------------

Cartesian scans:

1. Sort samples to ``(frame, coil, ky, kx)``.
2. Apply k-space corrections such as receiver scale/phase and readout phase terms.
3. Use :func:`mrax.reco.pipeline.cartesian_ifft` or
   :func:`mrax.reco.variational.reconstruct_ifft` for a zero-filled preview.
4. Estimate sensitivity maps when needed.
5. Use :func:`mrax.reco.variational.reconstruct_variational` or
   :func:`mrax.reco.variational.reconstruct_model_based` with the sampling mask.

Radial, spiral, UTE, and ZTE scans:

1. Sort raw samples to ``(frame, coil, sample)``.
2. Make sure ``trajectory[sample]`` follows the same order.
3. Apply receiver and sample-wise corrections.
4. Supply density weights from the sequence/vendor, or use MRax radial weights for
   preview only.
5. Use :func:`mrax.reco.variational.reconstruct_regrid` for inspection or the NUFFT
   operator in :func:`mrax.reco.variational.reconstruct_variational` for iterative
   reconstruction.

Variational Solver
------------------

The main entry point is :func:`mrax.reco.variational.reconstruct_variational`:

.. math::

   \min_x \; \tfrac{1}{2}\|A x - y\|_2^2 + \sum_i \lambda_i R_i(x)

For Cartesian data, ``A = M F S``: coil sensitivities, centered FFT, and sampling mask.
For non-Cartesian data, install ``mrax[nufft]``; the forward/adjoint then use
``jax-finufft`` with the supplied
trajectory. The NUFFT path is normalized to match the orthonormal FFT convention used by
:func:`mrax.reco.variational.fftnc` and :func:`mrax.reco.variational.ifftnc`.

The SENSE-style multi-coil encoding follows sensitivity encoding [Pruessmann1999]_. The
NUFFT path follows nonuniform FFT interpolation ideas [Fessler2003]_ and the FINUFFT
implementation [Barnett2019]_. TV penalties follow ROF/TV literature [Rudin1992]_.
Split Bregman and FISTA solver options follow [Goldstein2009]_ and [BeckTeboulle2009]_.
Partial-Fourier and homodyne stages follow [Haacke1991]_ and [Noll1991]_.

Example:

.. code-block:: python

   from mrax.reco import reconstruct_variational

   img = reconstruct_variational(
       kspace,
       mask=mask,
       sensitivities=sens_maps,
       optimizer_name="fista",
       regularizer="tv",
       regularizer_weight=0.02,
   )

Model-Based Reconstruction
--------------------------

When a quantitative model is available, MRax can reconstruct parameter maps directly
from k-space:

.. math::

   \min_{\theta}\; \tfrac{1}{2}\|A\,M(\theta) - y\|_2^2 + \sum_j \lambda_j R_j(\theta_j)

Here ``M`` is a :class:`mrax.data_struct.model.Model` mapping parameter fields to an
image series, and ``A`` is the MRI encoding operator.

.. code-block:: python

   from mrax.reco import reconstruct_model_based

   fields, losses = reconstruct_model_based(
       kspace,
       experiment=experiment,
       model=model,
       sensitivities=sens_maps,
       mask=sampling_mask,
       frame_phase_correction="align",
       num_steps=500,
       step_size=0.02,
       return_history=True,
   )

For relative signal models, provide a reference image or reconstruct the reference
jointly with ``reference_kspace``. Vendor
``2dseq`` reference images may not share raw k-space scaling; for model-based work,
prefer a reference built from the same k-space convention as the encoding operator.

For partial Fourier, Bruker imports store unfilled k-space and expose the acquired mask
as ``dataset.processed["sampling_mask"]``. Pass that mask to enforce only acquired
samples, or pre-complete k-space with an explicit POCS/homodyne stage when that is the
chosen reconstruction model.

Common Checks
-------------

- Missing sensitivity maps: estimate them with
  :func:`mrax.reco.estimate_sensitivities_cartesian` and
  :func:`mrax.reco.pick_sensitivity_frame`.
- Partial Fourier / undersampling: inspect
  :func:`mrax.reco.make_sampling_mask`, :func:`mrax.reco.pocs_complete_kspace`, and the
  ``completion_kspace`` argument in :func:`mrax.reco.reconstruct_model_based`.
- Frame/observable mismatches: use
  :func:`mrax.data_struct.reshape_frames_to_dims` and
  :func:`mrax.data_struct.reshape_dims_to_frames`.
- Long batch runs: call :func:`mrax.core.runtime.release_memory` between studies when
  Python/JAX caches become a memory issue.

See :doc:`notebooks/variational_reconstruction` for a runnable example with synthetic
data.
