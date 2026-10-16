Acquisitions and reconstruction plans
=====================================

Importers prepare measurements and geometry; numerical reconstruction is vendor
independent. Existing scanner images can be imported separately. For a runnable
example, see :doc:`reconstruction`.

Acquisition contract
--------------------

``AcquisitionData`` uses these conventions:

- Complex samples: ``(frame, coil, *encoding_shape)`` for Cartesian data or
  ``(frame, coil, sample)`` for trajectories.
- Auxiliary samples hold complex calibration data; auxiliary coordinates hold
  real values such as navigator times. ``auxiliary_frame_axes`` declares their
  frame-dependent axes for selection, batching and saving.
- Trajectory columns follow ``encoding_axes``, in cycles per pixel of the encoding
  grid. Cartesian DC is at ``size // 2``; the inverse FFT is centered and orthonormal.
- ``acquired`` marks measured support, including measured zeros. Interpolated or
  completed samples do not enter iterative objectives as new measurements.
- ``density`` supplies quadrature weights for initialization;
  ``statistical_weights`` supplies confidence in the likelihood. Sampling masks
  are binary.
- ``domain="adc"`` requires spatial encoding before MRI reconstruction.
  ``auxiliary_samples`` retains additional navigators or unresolved ADC regions.

Importers normalize storage order, polarity, Fourier signs, units and receiver
calibration. Recipes specify image permutations, flips and cropping.
``stack_slices`` and ``average_repeats`` map native readouts to output frames;
model-based encoding reverses this layout. Sensitivities may be shared or vary
by frame, with an explicit receiver axis. Acquisition archives use arrays and
JSON without pickle.

Readouts can carry nonuniform coordinates and polarity. FFT initialization uses
direct Fourier quadrature by default; linear and windowed-sinc interpolation are
alternatives. Coordinates retain their declared span. Iterative solvers encode
the original measurements. ``ReadoutModel`` optionally initializes delayed or
paired profiles with a truncated-SVD inverse and reports rank and conditioning.

Editing a plan
--------------

``ReconstructionPipeline.from_acquisition(acquisition)`` supplies the suggested
plan. Replace stages to compare methods. Plans allow EPI correction, FOV shifts,
partial Fourier or gridding, one solver (``fft``, ``variational`` or
``model_based``), output formatting and coil combination.

TOML plans use ``schema_version = 1`` and ``[[stages]]`` with ``kind`` and optional
``[stages.options]``. Include ``format`` and ``combine`` explicitly. Omitting a
correction stage disables it; empty output options retain imported geometry and
scaling. Invalid stages, options or ordering raise errors.

Algorithm choices
-----------------

**Partial Fourier:** homodyne uses central phase estimation with step/ramp
weighting; POCS restores measured samples between phase projections. The phase
window and convergence tolerance are configurable. See
`Noll (1991) <https://doi.org/10.1109/42.79473>`_ and
`Haacke (1991) <https://doi.org/10.1016/0022-2364(91)90253-P>`_.

**Gridding:** Kaiser--Bessel width, shape, oversampling and deapodization are
editable. ``deapodization_normalization="relative"`` gives unit correction at DC;
``"inverse_fft"`` inverts the absolute kernel response, including its ``1/N``
factor per axis. ``output_scale`` calibrates FFT images; ``encoding_scale`` sets
image units for iterative encoding. See
`Beatty (2005) <https://doi.org/10.1109/TMI.2005.848376>`_.

``radial_cell_density`` integrates radial cells and solid angles, assuming equal
angular coverage unless weights are supplied; ``target_sum`` sets normalization.
``radial_profile_density`` integrates profiles relative to a Nyquist radius
without angular averaging. These conventions have different image units.
For iterative density design, see
`Pipe and Menon (1999) <https://pubmed.ncbi.nlm.nih.gov/10025627/>`_.

**EPI:** timed three-echo navigators interpolate the outer echoes to the middle
TE before fitting a linear phase in readout hybrid space. Entropy fitting
searches phase offset and slope. The entropy method needs a
phase prior or navigators to resolve half-FOV ambiguity. Calibration uses a
common phase across receivers. See
`Poser (2013) <https://doi.org/10.1002/mrm.24222>`_ and
`Skare (2006) <https://cds.ismrm.org/protected/06MProceedings/PDFfiles/02349.pdf>`_.

**Windows:** sine, squared-sine, Hann and Hamming windows use the centered Fourier
grid; per-axis powers compose serial applications. Iterative data terms retain
the original samples.

**Frequency maps:** receiver-summed ``conj(echo_1)*echo_2`` gives phase divided by
``2*pi*delta_TE`` in seconds, preserving signed Hz. Reliability thresholds and
quality-guided or weighted Laplacian unwrapping are configurable. Quality-guided
unwrapping uses the CPU. Each disconnected region has an independent integer
phase-period ambiguity.

For source definitions and extension points, see :doc:`bruker_import` and
:doc:`importer_development`.
