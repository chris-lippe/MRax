.. _api-reco:

Reconstruction functions
========================

.. autofunction:: mrax.reco.reconstruct_acquisition
.. autofunction:: mrax.reco.processing.pad_centered_kspace
.. autofunction:: mrax.reco.utils.estimate_sensitivities_cartesian
.. autofunction:: mrax.reco.utils.pick_sensitivity_frame
.. autofunction:: mrax.reco.utils.pocs_complete_kspace

.. autofunction:: mrax.reco.processing.apply_channel_scale_phase
.. autofunction:: mrax.reco.processing.apply_samplewise_correction
.. autofunction:: mrax.reco.processing.estimate_radial_density
.. autofunction:: mrax.reco.processing.fid_profiles_to_kspace
.. autofunction:: mrax.reco.processing.apply_readout_phase_shift
.. autofunction:: mrax.reco.processing.crop_along_axis
.. autofunction:: mrax.reco.processing.readout_crop_bounds
.. autofunction:: mrax.reco.phase.unwrap_phase_image
.. autofunction:: mrax.reco.components.reconstruction_variant

.. autoclass:: mrax.reco.variational.MRIEncodingOperator
   :members: forward, adjoint, coil_count, is_non_cartesian

.. autoclass:: mrax.reco.variational.RegridApodizationCorrection

.. autofunction:: mrax.reco.variational.build_operator
.. autofunction:: mrax.reco.variational.center_crop_spatial
.. autofunction:: mrax.reco.variational.checkerboard_phase
.. autofunction:: mrax.reco.variational.apply_regrid_apodization_correction
.. autofunction:: mrax.reco.variational.elliptical_support_mask
.. autofunction:: mrax.reco.variational.apply_image_support_mask
.. autofunction:: mrax.reco.variational.combine_coil_images
.. autofunction:: mrax.reco.variational.solve_variational
.. autofunction:: mrax.reco.variational.solve_model_based
