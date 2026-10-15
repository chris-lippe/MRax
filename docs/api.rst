API Reference
=============

Use this page as a map of the public API. Narrative explanations are in the guide
pages; this page lists the main entry points.

.. contents::
   :local:
   :depth: 2

.. _api-data:

Data and Study Containers
-------------------------

Data structures that keep arrays, units, and experiment context together.

- ``DataSet`` bundles magnitude/k-space arrays plus optional noise level and derived
  results; ``shape`` and ``spatial_shape`` expose the spatial dimensions expected by
  models.
- ``Quantity`` pairs any ndarray with an SI unit string so observables are explicitly
  typed.
- ``Experiment`` declares dimension names and observables; ``validate`` checks dataset
  compatibility and observable types.
- ``JointExperiment`` and ``join_experiments`` stack compatible experiments into a
  frame-major dataset for joint fitting or reconstruction; ``select_experiment_frames``
  extracts sub-experiments by frame indices.
- Dimension helpers in :mod:`mrax.data_struct.dimensions` infer non-spatial axis splits,
  reshape frame-major arrays, and expand observables to per-frame arrays.
- ``Model`` ties a forward model, parametrization factory, and per-parameter units; it
  can emit regularizer terms via ``make_regularizer_terms``.
- ``Study`` keeps multiple experiments/datasets/models together and attaches shared
  study globals (e.g., ``field_strength_mhz``) to each experiment/model.

.. autoclass:: mrax.data_struct.dataset.DataSet
   :members: shape, spatial_shape

.. autoclass:: mrax.data_struct.quantity.Quantity
   :members: shape, as_array, with_values

.. autofunction:: mrax.data_struct.quantity.attach_units
.. autofunction:: mrax.data_struct.quantity.unwrap_values
.. autofunction:: mrax.data_struct.quantity.require_quantity_dict

.. autoclass:: mrax.data_struct.experiment.Experiment
   :members: shape, spatial_shape, frame_count, describe_dims, merged_observables, insert_observables, sanity_check, explain_sanity, validate

.. autoclass:: mrax.data_struct.joint.JointExperiment

.. autofunction:: mrax.data_struct.joint.join_experiments
.. autofunction:: mrax.data_struct.joint.select_experiment_frames

.. autoclass:: mrax.data_struct.dimensions.DimensionSpec

.. autofunction:: mrax.data_struct.dimensions.build_experiment_dims
.. autofunction:: mrax.data_struct.dimensions.reshape_frames_to_dims
.. autofunction:: mrax.data_struct.dimensions.reshape_dims_to_frames
.. autofunction:: mrax.data_struct.dimensions.reshape_observables_to_dims
.. autofunction:: mrax.data_struct.dimensions.expand_observables_to_frames
.. autofunction:: mrax.data_struct.dimensions.pick_observable_for_frames

.. autoclass:: mrax.data_struct.model.Model
   :members: make_regularizer_terms

.. autoclass:: mrax.data_struct.study.Study
   :members: iter_entries, entries_by_name, execution_order, describe, sanity_check, explain_sanity, from_examples

.. _api-storage:

Storage and Manifests
---------------------

Persist inputs/outputs for one or more experiments in an ISMRMRD-backed manifest. All
files referenced in the manifest are created relative to the provided ``output_dir``.

- ``save_study_results(study, results, output_dir, ...)`` writes MR images, optional
  k-space, fitted parameter maps with units, plots when requested by the result payload,
  and optional NIfTI exports when ``save_nifti=True``. Returns ``study_manifest.json``.
- ``load_study_results(manifest_path)`` reads the manifest plus all referenced arrays
  and reconstructs per-experiment dictionaries containing fitted maps, metadata, units,
  and optional plot paths.

.. autofunction:: mrax.data_struct.storage.save_study_results
.. autofunction:: mrax.data_struct.storage.load_study_results

.. _api-fitting:

Fitting Pipeline
----------------

Build and run differentiable losses with Optax/Proximal variants. All functions accept
per-parameter regularizer configs/weights that override model defaults.

- ``build_loss(experiment, model, dataset, ...)`` → ``(loss_fn, init_params, unpack)``
  where ``loss_fn(params)`` returns a scalar data+regularization loss. Pass known
  observation variances with ``data_variance`` for precision weighting; uniform
  variance is the default, ``data_variance="auto"`` estimates robust per-frame
  variances before fitting, and the RMS loss normalization is retained.
- ``run_fit(experiment, model, dataset, optimizer_name="adam", step_size=0.02, num_steps=200, ...)``
  fits a single experiment; returns ``(fitted_fields_with_units, loss_history)`` or
  ``(fitted_fields_with_units, loss_history, model_fn)`` when
  ``return_model_callable=True`` (``model_fn(fields, offsets=None, observables=None)`` applies
  study globals + observables).
  Supports Optax optimizers plus ``split_bregman``/``fista`` wrappers, box constraints,
  proximal regularizers, and explicit initial parameters.
- ``make_model_callable(experiment, model, study_globals=None)`` builds a callable that
  merges study globals/observables before invoking ``model.model_apply``. Use it in
  notebooks to evaluate predictions without manually attaching ``field_strength_mhz``.
- ``plan_study_fits(study, target_names=None, precomputed_fields=None)`` returns an
  ordered list of :class:`mrax.core.study_runner.PlannedFit` entries for sequential
  fitting, optionally reusing cached fields.
- ``execute_study_plan(plan, run_fit_kwargs=None, run_fit_overrides=None, ...)`` executes
  the plan via :func:`run_fit` and returns result entries plus the
  fitted field cache. Experiments missing required observables are skipped by default
  and marked with ``skipped=True`` in the results. Put shared ``run_fit`` options in
  ``run_fit_kwargs`` and per-experiment options in ``run_fit_overrides``; dict-valued
  ``run_fit`` options such as ``regularizer_weights`` are passed through unchanged.
- ``check_fit_sanity(experiment, model, dataset, study_globals=None)`` runs a fast
  pre-fit check and returns a :class:`mrax.core.fit_sanity.FitSanityReport`.
- ``estimate_data_variance(experiment, dataset, mask=None, ...)`` estimates robust
  per-frame observation variances from background noise, with a spatial
  wavelet-MAD fallback, and returns diagnostics in a
  :class:`mrax.core.variance.VarianceEstimate`. The optional mask must identify a
  noise-only region; fitting masks are separate.

.. autofunction:: mrax.core.loss_builder.build_loss
.. autofunction:: mrax.core.loss_builder.run_fit
.. autofunction:: mrax.core.loss_builder.make_model_callable
.. autofunction:: mrax.core.variance.estimate_data_variance
.. autoclass:: mrax.core.variance.VarianceEstimate
   :members:
.. autofunction:: mrax.core.loss_builder.build_study_losses
.. autofunction:: mrax.core.study_runner.plan_study_fits
.. autofunction:: mrax.core.study_runner.execute_study_plan
.. autofunction:: mrax.core.fit_sanity.check_fit_sanity
.. autoclass:: mrax.core.fit_sanity.FitSanityReport

.. _api-batch:

Batch Execution
---------------

Utilities for looping over many studies on disk and exporting consistent manifests.

- ``discover_study_dirs(root, selected=None)`` returns study folder paths (optionally
  filtered by name) with helpful error messages when nothing is found.
- ``run_batch_studies(studies_root, study_loader, output_root, ...)`` iterates through
  discovered study folders, converts each to a ``Study`` via ``study_loader``, runs
  analysis (default: study planner + executor), then calls ``save_study_results``.

.. autofunction:: mrax.core.batch.discover_study_dirs
.. autofunction:: mrax.core.batch.run_batch_studies

Runtime Helpers
---------------

.. autofunction:: mrax.core.runtime.release_memory

.. _api-preprocessing:

Preprocessing
-------------

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

.. _api-analysis:

Post-fit Analysis
-----------------

Utilities for extracting ROI statistics, ROI timecourses, and performing group-level
comparisons on batch outputs.

- ``ROI`` defines an ROI mask with optional dimension names for broadcasting.
- ``collect_roi_stats`` produces a long-form table of ROI summaries (mean/variance/etc.).
- ``extract_timecourse`` reduces an image stack to an ROI timecourse.
- ``analyze_groups`` builds subject-level metrics, group summaries, and pairwise statistics.

.. autoclass:: mrax.analysis.roi.ROI
   :members: name, mask, dims

.. autofunction:: mrax.analysis.roi.collect_roi_stats
.. autofunction:: mrax.analysis.roi.match_spatial_mask
.. autofunction:: mrax.analysis.timecourse.extract_timecourse
.. autofunction:: mrax.analysis.timecourse.extract_timecourse_from_experiment
.. autofunction:: mrax.analysis.group.analyze_groups
.. autofunction:: mrax.analysis.group.collect_subject_roi_stats
.. autofunction:: mrax.analysis.group.summarize_group_stats
.. autofunction:: mrax.analysis.group.compare_group_stats

.. _api-synthetic:

Synthetic Examples
------------------

Synthetic datasets that use the same containers as imported studies.

- ``make_example_suite(noise_level=0.05, seed=0, model_names=("t1_ir","t2_se"))``
  → list of :class:`mrax.core.synthetic.Example` objects.
- ``make_synthetic_study(...)`` → ``(Study, metadata)``; the metadata includes ground
  truth fields for benchmarking.

.. autoclass:: mrax.core.synthetic.Example

.. autofunction:: mrax.core.synthetic.make_example_suite
.. autofunction:: mrax.core.synthetic.make_synthetic_study

.. _api-optim:

Optimization and Regularization
-------------------------------

Core primitives for solvers and penalties. All regularizer functions operate on ndarrays
and return scalar costs.

- ``make_optimizer(optimizer_name, step_size, regularizer_weight, regularizer_prox, box_constraints)``
  resolves Optax optimizers plus proximal ``split_bregman``/``fista`` variants.
- ``spatial_smoothness_penalty`` and ``total_variation_penalty`` operate along configurable axes.
  ``higher_order_total_variation_penalty`` adds finite-difference TV of configurable order.
- Low-level ops like ``soft_threshold``, ``isotropic_shrinkage``, gradient/divergence
  helpers, and ``conjugate_gradient`` are available for custom solvers.

.. autofunction:: mrax.opt.optimizers.make_optimizer

.. autofunction:: mrax.opt.regularizers.spatial_smoothness_penalty
.. autofunction:: mrax.opt.regularizers.total_variation_penalty
.. autofunction:: mrax.opt.regularizers.higher_order_total_variation_penalty
.. autofunction:: mrax.opt.regularizers.make_regularizer_fn
.. autofunction:: mrax.opt.regularizers.estimate_field_scales
.. autofunction:: mrax.opt.regularizers.estimate_param_scales
.. autofunction:: mrax.opt.regularizers.autoscale_regularizer_configs

.. autofunction:: mrax.opt.ops.soft_threshold
.. autofunction:: mrax.opt.ops.isotropic_shrinkage
.. autofunction:: mrax.opt.ops.forward_gradient2d
.. autofunction:: mrax.opt.ops.divergence2d
.. autofunction:: mrax.opt.ops.conjugate_gradient

.. _api-reco:

Reconstruction
--------------

Reconstruct prepared acquisitions with one vendor-independent entry point.
Plans select FFT, variational or model-based algorithms and numerical stages.
See :doc:`acquisition_reconstruction` for the acquisition contract and recipes.

.. autoclass:: mrax.data_struct.AcquisitionData
   :members: select_frames, require_prepared

.. autoclass:: mrax.data_struct.ReconstructionRecipe

.. autoclass:: mrax.reco.ReconstructionPipeline
   :members: from_acquisition, from_toml, replace_stage, run

.. autoclass:: mrax.reco.ReconstructionStage

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

.. _api-importers:

Import Helpers
--------------

DICOM Importers
~~~~~~~~~~~~~~~

Discover and load classic or enhanced DICOM image series. The general layer handles
pixel scaling, functional groups, geometry, slice/frame ordering, dimensions, and
standard MR observables. Vendor adapters only add origin-specific metadata; the
built-in Bruker adapter does not own pixel decoding or geometry.

.. autoclass:: mrax.io.dicom.DicomSeriesInfo
.. autoclass:: mrax.io.dicom.DicomMetadataPatch
.. autoclass:: mrax.io.dicom.DicomMetadataAdapter
.. autoclass:: mrax.io.dicom_bruker.BrukerDicomMetadataAdapter
.. autofunction:: mrax.io.dicom.discover_dicom_series
.. autofunction:: mrax.io.dicom.load_dicom_series
.. autofunction:: mrax.io.dicom.load_dicom_study

Import definitions
~~~~~~~~~~~~~~~~~~

Normalize vendor metadata, infer experiment types, extract observables, and materialize
vendor-neutral experiments/datasets from importer payloads.

.. autoclass:: mrax.io.import_definitions.NormalizedMetadata
.. autoclass:: mrax.io.import_definitions.ImportDefinition

.. autofunction:: mrax.io.import_definitions.build_imported_experiment_dataset
.. autofunction:: mrax.io.import_definitions.detect_experiment_type
.. autofunction:: mrax.io.import_definitions.extract_observables_from_metadata
.. autofunction:: mrax.io.import_definitions.reshape_epi_samples

Pulseq Importers
~~~~~~~~~~~~~~~~

Read Pulseq ``.seq`` files, calculate ADC trajectories when ``pypulseq`` is installed,
and combine sequence definitions with supplied k-space or image arrays. This importer is
intended for synthetic demos and importer-development experiments; it has not been
validated on measured scanner data.

.. autoclass:: mrax.io.pulseq_import.PulseqSequenceInfo
.. autofunction:: mrax.io.pulseq_import.read_pulseq
.. autofunction:: mrax.io.pulseq_import.read_pulseq_acquisition
.. autofunction:: mrax.io.pulseq_import.read_pulseq_sidecar_metadata
.. autofunction:: mrax.io.pulseq_import.import_pulseq_experiment
.. autofunction:: mrax.io.pulseq_import.detect_pulseq_kspace_kind

Bruker Importers and Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Import ParaVision images, raw acquisitions, geometry and model observables.
Study import warns and skips incomplete or unsupported experiments while retaining
valid experiments. Import and numerical reconstruction are separate operations.
See :doc:`bruker_import` and :doc:`importer_development` for definitions and options.

.. autofunction:: mrax.io.import_bruker_experiment
.. autofunction:: mrax.io.import_bruker_study
.. autofunction:: mrax.io.read_bruker_acquisition
.. autofunction:: mrax.io.load_acquisition
.. autofunction:: mrax.io.save_acquisition
.. autoclass:: mrax.io.DefinitionLibrary
   :members: select

.. autofunction:: mrax.io.study_selection.auto_select_scans
.. autofunction:: mrax.io.study_selection.pick_scan_by_type

.. autofunction:: mrax.library.default_models.make_t1_ir_model
.. autofunction:: mrax.library.default_models.make_t1_vtr_model
.. autofunction:: mrax.library.default_models.make_t1_vfa_model
.. autofunction:: mrax.library.default_models.make_exp_decay_model
.. autofunction:: mrax.library.default_models.make_t2_se_model
.. autofunction:: mrax.library.default_models.make_t2star_se_model
.. autofunction:: mrax.library.default_models.make_wasabi_model
.. autofunction:: mrax.library.default_models.make_multipool_lorentzian_model
.. autofunction:: mrax.library.default_models.make_AB_MT_n_model
.. autofunction:: mrax.library.default_models.make_zaiss_full_model
