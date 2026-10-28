Acquisitions and reconstruction plans
=====================================

Reconstruction plans let you compare methods and corrections on the same imported
measurements. The workflow is the same for Bruker and Pulseq data.

Run the suggested plan
----------------------

After importing a study with raw data, choose a dataset and build its suggested
reconstruction plan:

.. code-block:: python

   from mrax.reco import ReconstructionPipeline

   dataset = study.datasets[0]
   acquisition = dataset.acquisition
   plan = ReconstructionPipeline.from_acquisition(acquisition)
   result = plan.run(acquisition, device="cpu")
   image = result.image

The plan includes the corrections, output geometry and scaling supplied by the
importer. ``result.image`` holds the reconstructed product; the imported scanner
image remains in ``dataset.data``. See :doc:`bruker_import` and
:doc:`pulseq_import` for loading a study.

Editing a plan
--------------

Replace a stage to compare reconstruction methods while keeping the other settings:

.. code-block:: python

   tv_plan = plan.replace_stage(
       "fft", replacement="variational",
       regularizer="tv", regularizer_weight=0.001, num_steps=100,
   )
   tv_result = tv_plan.run(acquisition, device="cpu")

Plans support EPI correction, FOV shifts, partial Fourier or gridding, a solver
(``fft``, ``variational`` or ``model_based``), output formatting and coil combination.
See :doc:`reconstruction` for fitting models directly to raw measurements.

TOML plans use ``schema_version = 1`` and ``[[stages]]`` with ``kind`` and optional
``[stages.options]``. Include ``format`` and ``combine`` explicitly. Omitting a
correction stage disables it; empty output options retain imported geometry and
scaling. Invalid stages, options or ordering raise errors.

Algorithm choices
-----------------

**Partial Fourier:** homodyne uses central phase estimation with step/ramp
weighting; POCS restores measured samples between phase projections. The phase
window and convergence tolerance are configurable. See
[Noll1991]_ and [Haacke1991]_.

**Gridding:** Kaiser--Bessel width, shape, oversampling and deapodization are
editable. ``deapodization_normalization="relative"`` gives unit correction at DC;
``"inverse_fft"`` inverts the absolute kernel response, including its ``1/N``
factor per axis. ``output_scale`` calibrates FFT images; ``encoding_scale`` sets
image units for iterative encoding. See
[Beatty2005]_.

``radial_cell_density`` integrates radial cells and solid angles, assuming equal
angular coverage unless weights are supplied; ``target_sum`` sets normalization.
``radial_profile_density`` integrates profiles relative to a Nyquist radius
with geometric solid-angle quadrature supplied separately. These conventions have different image units.
For iterative density design, see
[PipeMenon1999]_.

**EPI:** timed three-echo navigators interpolate the outer echoes to the middle
TE before fitting a linear phase in readout hybrid space. Entropy fitting
searches phase offset and slope. The entropy method needs a
phase prior or navigators to resolve half-FOV ambiguity. Calibration uses a
common phase across receivers. See [Skare2006]_ for entropy-based correction
and [Poser2013]_ for EPI acquisition context.

**Windows:** sine, squared-sine, Hann and Hamming windows use the centered Fourier
grid; per-axis powers compose serial applications. Iterative data terms retain
the original samples.

**Frequency maps:** receiver-summed ``conj(echo_1)*echo_2`` gives phase divided by
``2*pi*delta_TE`` in seconds, preserving signed Hz. Reliability thresholds and
quality-guided or weighted Laplacian unwrapping are configurable. Quality-guided
unwrapping uses the CPU. Each disconnected region has an independent integer
phase-period ambiguity.

Array shapes, encoding axes and low-level readout conventions are documented in
the :ref:`api-acquisition-conventions`.
