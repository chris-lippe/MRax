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

``solve_variational`` and ``solve_model_based`` expose low-level solvers;
``build_operator`` exposes forward/adjoint encoding. Non-Cartesian iterative
encoding requires the optional ``mrax[nufft]`` backend.

Sensitivity encoding follows [Pruessmann1999]_. Nonuniform Fourier encoding uses
the interpolation framework in [Fessler2003]_ and FINUFFT [Barnett2019]_.
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
