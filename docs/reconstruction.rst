Reconstruction
==============

``reconstruct_acquisition`` and ``ReconstructionPipeline`` consume the same explicit
acquisition contract for Cartesian FFT, gridding, variational and model-based
reconstruction. Importers prepare samples and numerical hints; reconstruction
functions do not read source-format metadata.

.. code-block:: python

   from mrax.reco import ReconstructionPipeline

   plan = ReconstructionPipeline.from_acquisition(acquisition)
   preview = plan.run(acquisition, device="cpu")
   tv_plan = plan.replace_stage("fft", replacement="variational",
       regularizer="tv", regularizer_weight=0.001, num_steps=100)
   result = tv_plan.run(acquisition, device="cpu")

See :doc:`acquisition_reconstruction` for contract details, editable TOML plans,
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
completed samples do not count as new measurements. L1/TV penalties and their
proximal weights use the same mean-over-image normalization.

Model-based reconstruction applies this encoding to a signal model. Supply
``experiment``, ``model`` and compatible receiver sensitivities. Regularizer axes
and masks follow parameter fields. FISTA/split Bregman use proximal TV; explicit
Adam uses smoothed TV with a warning. See :doc:`models` for fitting controls.

Model-based fitting uses RMS-normalized mean squared error, with the same
prefactor as image fitting. ``reference_image`` multiplies a relative signal
model by a fixed spatial reference. ``roi_mask`` controls regularizers and output;
it does not discard exterior k-space signal. ``object_support_mask`` instead
declares physical object support. Comparisons require matching likelihoods,
support, references and regularizer weights.
Model-based encoding uses batches of up to 32 frames with rematerialized
gradients, retaining the complete time model and temporal regularizers.
``encoding_batch_size`` changes this bound; ``None`` uses a single batch.

``solve_variational`` and ``solve_model_based`` expose low-level solvers;
``build_operator`` exposes forward/adjoint encoding. Non-Cartesian encoding uses
the built-in JAX NUFFT on the selected CPU or GPU, without an additional backend.
Its fixed trajectory uses cycles per pixel in spatial-axis order. Reusable
``prepare_nufft(shape, trajectory)`` plans support batches, JIT and image/model
gradients; trajectory gradients are unsupported. Prepare plans outside JIT.

The NUFFT uses twofold FFT oversampling and matched Kaiser--Bessel interpolation
and spreading. ``nufft_eps`` (default ``1e-6``, range ``1e-12`` to ``1e-2``)
selects kernel width; floating-point roundoff can exceed this interpolation target.
Tighter precision requires JAX x64 and float64 trajectories and images. Sample
and batch chunks bound intermediate memory, but large 3D FFT grids still require
substantial memory. Windows GPU execution requires a JAX-supported environment
such as WSL 2; native Windows uses CPU.

``prepare_nufft`` limits stencil caching with ``cache_budget_bytes`` (128 MiB
by default). Larger trajectories retain compact positions and evaluate tabulated
kernel weights per chunk. Explicit adjoint derivative rules avoid storing expanded
stencils during differentiation. Long single-precision adjoints use bounded partial
sums, or CPU double-precision accumulation when JAX x64 is enabled; outputs retain the
plan's single precision.

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
