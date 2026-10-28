.. _api-acquisition-conventions:

Acquisition conventions
=======================

``AcquisitionData`` stores complex samples as ``(frame, coil, *encoding_shape)``
for Cartesian data or ``(frame, coil, sample)`` for trajectories.

- Trajectory columns follow ``encoding_axes``, in cycles per pixel of the encoding
  grid. Cartesian DC is at ``size // 2``; the inverse FFT is centered and orthonormal.
- ``acquired`` marks measured support, including measured zeros. Iterative
  objectives use this measured support; completed samples serve initialization.
- ``density`` supplies quadrature weights for initialization;
  ``statistical_weights`` supplies confidence in the likelihood. Sampling masks
  are binary.
- ``domain="adc"`` requires spatial encoding before MRI reconstruction.
- Auxiliary samples hold complex calibration data; auxiliary coordinates hold
  real values such as navigator times. ``auxiliary_frame_axes`` declares their
  frame-dependent axes for selection, batching and saving.

``DataSet.kspace``, trajectory, density and sampling masks mirror the acquisition.
``AcquisitionData.select_frames`` selects output frames and their associated
native measurements.

Importers normalize storage order, polarity, Fourier signs, units and receiver
calibration. Recipes specify image permutations, flips and cropping.
``stack_slices`` and ``average_repeats`` map native readouts to output frames;
model-based encoding reverses this layout. Sensitivities may be shared or vary
by frame, with an explicit receiver axis. Acquisition archives use arrays and
JSON with explicit numeric arrays.

Readouts can carry nonuniform coordinates and polarity. FFT initialization uses
direct Fourier quadrature by default; linear and windowed-sinc interpolation are
alternatives. Coordinates retain their declared span. Iterative solvers encode
the original measurements. ``ReadoutModel`` optionally initializes delayed or
paired profiles with a truncated-SVD inverse and reports rank and conditioning.

.. autoclass:: mrax.data_struct.AcquisitionData
   :members: select_frames, require_prepared

.. autoclass:: mrax.data_struct.ReconstructionRecipe

.. autoclass:: mrax.reco.ReconstructionPipeline
   :members: from_acquisition, from_toml, replace_stage, run

.. autoclass:: mrax.reco.ReconstructionStage
