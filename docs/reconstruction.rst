Reconstruction
==============

Use ``reconstruct_acquisition`` or ``ReconstructionPipeline`` to reconstruct imported
raw data with Cartesian FFT, non-Cartesian gridding, variational or model-based
methods. The imported acquisition supplies the sampling and geometry settings.

.. code-block:: python

   from mrax.reco import ReconstructionPipeline

   plan = ReconstructionPipeline.from_acquisition(acquisition)
   preview = plan.run(acquisition, device="cpu")
   tv_plan = plan.replace_stage("fft", replacement="variational",
       regularizer="tv", regularizer_weight=0.001, num_steps=100)
   result = tv_plan.run(acquisition, device="cpu")

See :doc:`acquisition_reconstruction` for editable plans,
normalization conventions, partial Fourier, readout correction, gridding and
frequency products. All numerical entry points default to CPU and accept an
explicit ``device``. ``mrax.get_available_devices()`` lists available devices.

Encoding and optimization
-------------------------

Variational reconstruction minimizes

.. math::

   \tfrac12 \| W^{1/2}(A x-y) \|_2^2 + \sum_i \lambda_i R_i(x).

``A`` encodes measured support, receiver sensitivities and Fourier coordinates;
``W`` holds statistical weights. Density weights initialize reconstruction;
completed samples serve initialization. L1/TV penalties and their
proximal weights use the same mean-over-image normalization.

Model-based reconstruction applies this encoding to a signal model. Supply
``experiment``, ``model`` and compatible receiver sensitivities. Regularizer axes
and masks follow parameter fields. FISTA/split Bregman use proximal TV; explicit
Adam uses smoothed TV with a warning. See :doc:`model_fitting` for fitting controls.

Model-based fitting uses RMS-normalized mean squared error, with the same
prefactor as image fitting. ``reference_image`` multiplies a relative signal
model by a fixed spatial reference. ``roi_mask`` controls regularizers and output;
Fourier encoding retains the full signal, including exterior contributions. ``object_support_mask`` instead
declares physical object support. Comparisons require matching likelihoods,
support, references and regularizer weights.
Model-based encoding uses batches of up to 32 frames with rematerialized
gradients, retaining the complete time model and temporal regularizers.
``encoding_batch_size`` changes this bound; ``None`` uses a single batch.

Paired-echo frequency maps are ambiguous by integer multiples of ``1 / delta_TE``
Hz, with ``delta_TE`` in seconds. An absolute frequency reference requires
additional calibration. Reliability masks can also differ between reconstruction methods.

``solve_variational`` and ``solve_model_based`` expose low-level solvers;
``build_operator`` exposes forward/adjoint encoding. Non-Cartesian encoding uses
the built-in JAX NUFFT on the selected CPU or GPU.
Its fixed trajectory uses cycles per pixel in spatial-axis order. Reusable
``prepare_nufft(shape, trajectory)`` plans support batches, JIT and image/model
gradients; trajectory gradients are unsupported. Prepare plans outside JIT.

``nufft_eps`` (default ``1e-6``, range ``1e-12`` to ``1e-2``)
selects kernel width; floating-point roundoff can exceed this interpolation target.
Tighter precision requires JAX x64 and float64 trajectories and images. Sample
and batch chunks bound intermediate memory, but large 3D FFT grids still require
substantial memory. Windows GPU execution requires a JAX-supported environment
such as WSL 2; native Windows uses CPU.

For NUFFT caching, batching and numerical implementation details, see
:ref:`api-nufft-plans`.

Sensitivity encoding follows [Pruessmann1999]_. Nonuniform Fourier encoding uses
the interpolation framework in [Fessler2003]_ and Kaiser--Bessel gridding
in [Beatty2005]_.
Partial-Fourier phase projection and homodyne follow [Haacke1991]_ and [Noll1991]_.
Proximal solvers follow split Bregman [Goldstein2009]_ and FISTA
[BeckTeboulle2009]_. Phase integration and weighted least-squares unwrapping follow
[Itoh1982]_ and [GhigliaRomero1994]_.

Image products and processing
-----------------------------

Results retain complex coil images and combined products. Coil combination,
permutations, flips and cropping are configurable; frequency maps retain signed Hz.
``unwrap_phase_image`` supports 2D, stacks and volumes with magnitude-weighted
least squares. Quantitative two-echo products also offer quality-guided or
weighted Laplacian unwrapping; low-signal and disconnected regions can be ambiguous.

``mrax.reco.processing`` provides gradients, calibration, sample corrections,
readout shifts, density estimation and centered Fourier padding.
``pad_centered_kspace`` preserves DC across odd/even sizes and supports scaling
that preserves mean image intensity.

Check geometry and absolute scaling before comparing methods. For different
correction or regularization choices, also compare resolution, contrast, noise,
phase and artifacts.
