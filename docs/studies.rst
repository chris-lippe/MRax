Studies and Observables
=======================

The :class:`mrax.data_struct.study.Study` class coordinates multiple experiments, models,
and datasets. Observables are explicit: if a model needs a map from another fit (for
example a calibration or baseline map), insert it into the target experiment before
calling ``run_fit``.

Dimension handling
------------------

Experiments use named dimensions to describe how non-spatial axes map to observables.
Spatial axes are always ``x/y/z``. Non-spatial axes are inferred from observable
lengths when importing data. Ambiguous observable layouts use a single ``frame``
axis.

You can inspect the current layout via ``experiment.describe_dims()`` and reshape arrays
manually using :func:`mrax.data_struct.reshape_frames_to_dims` or
:func:`mrax.data_struct.reshape_observables_to_dims`. To go back to frame-major layouts,
use :func:`mrax.data_struct.reshape_dims_to_frames`.
To choose an observable that matches the frame axis, use
:func:`mrax.data_struct.pick_observable_for_frames`.

ROI masks from external tools sometimes arrive transposed relative to MRax spatial
ordering. Use :func:`mrax.analysis.match_spatial_mask` to validate or transpose masks
before ROI extraction.

By default, each non-spatial dimension lacking a mapped observable is treated like an
additional parameter axis. Fits will return fields with that dimension up front, for
example ``t, x, y`` or ``frame, x, y``. This enables per-repeat regularizers and
``parameter_constant_dims`` along the repeat axis within one dataset. If you
want parameters shared across a repeat dimension, map it to an explicit observable or set
``parameter_constant_dims`` to collapse that axis.

Multi-axis editing example
--------------------------

When an imported scan has multiple acquisition axes and temporal repeats, you can
override the inferred layout and reshape the data explicitly:

.. code-block:: python

   from mrax.data_struct import reshape_frames_to_dims, reshape_observables_to_dims

   exp = study.experiments[idx]
   ds = study.datasets[idx]
   print(exp.describe_dims())

   # Override dims: 12 settings along axis_a, 3 along axis_b, 2 repeats.
   exp.dims = {
       "axis_a": 12,
       "axis_b": 3,
       "t": 2,
       "x": exp.dims["x"],
       "y": exp.dims["y"],
   }
   dim_map = getattr(exp, "_mrax_dim_observables", None)
   exp.observables = reshape_observables_to_dims(exp.observables, exp.dims, dim_map)
   ds.data = reshape_frames_to_dims(ds.data, exp.dims)

Joint experiments and partial fits
----------------------------------

When multiple scans measure the same subject or phantom with related acquisition
settings, you can combine them into a single frame-major experiment for a joint fit:

.. code-block:: python

   from mrax.data_struct import join_experiments
   from mrax.core.loss_builder import run_fit

   joint = join_experiments(
       [exp_15, exp_16, exp_17],
       [ds_15, ds_16, ds_17],
       names=["15", "16", "17"],
       joint_name="joint_protocol",
   )
   joint_exp, joint_ds = joint.experiment, joint.dataset
   fields, losses = run_fit(joint_exp, model, joint_ds, auto_mask_nan_observables=True)

Missing observables are filled with ``NaN`` by default; ``auto_mask_nan_observables=True``
restricts the loss to frames with valid observables.
By default the join helper adds a ``joint_source`` observable (integer index per frame) so
side models can condition on scan/timepoint membership.

To fit only a subset (e.g., a single scan or timepoint), select frames by source name:

.. code-block:: python

   sub_exp, sub_ds = joint.select("16")
   fields_sub, _ = run_fit(sub_exp, model, sub_ds)

Side models that explain parameter drift or time dependence can be expressed by fitting
a separate experiment that outputs the needed map, then inserting it into the joint
experiment with ``Experiment.insert_observables``.

Defining a Study
----------------

.. code-block:: python

   from mrax.data_struct import Quantity, Study
   from mrax.core.synthetic import make_example_suite

   examples = make_example_suite(noise_level=0.05, seed=0)
   study_globals = {"field_strength_mhz": Quantity(400.0, "MHz")}
   study = Study.from_examples(examples, study_globals=study_globals)

Synthetic builders (:func:`mrax.core.synthetic.make_synthetic_study`) and importers
(:func:`mrax.io.bruker_import.import_bruker_study`) both emit ``Study`` objects,
so the downstream fitting and storage flows are identical regardless of the data source.
Study-wide globals, such as scanner or protocol constants, belong in ``study_globals`` and are
merged into every experiment when fitting.

Inserting observables from prior fits
--------------------------------------------------

If a downstream model needs maps produced elsewhere, insert them explicitly:

.. code-block:: python

   from mrax.data_struct import Quantity

   target_exp = study.experiments[target_idx]
   target_exp = target_exp.insert_observables(
       {
           "baseline_map": Quantity(baseline_vals, "a.u."),
           "calibration_map": Quantity(calibration_vals, "1"),
       }
   )

Sanity checks
-------------

Use the built-in sanity checks to confirm that ``run_fit`` has everything it needs:

.. code-block:: python

   report = study.experiments[idx].sanity_check(model, dataset)
   print(report.explain())

Study-level helpers are also available:

.. code-block:: python

   print(study.explain_sanity())

Planning and Execution
----------------------

Use :func:`mrax.core.study_runner.plan_study_fits` to run fits in the order provided by
the study, optionally filtering with ``target_names`` or reusing cached results via
``precomputed_fields``.

If an experiment is missing required observables, ``execute_study_plan`` skips it by
default and returns a result entry marked ``skipped``. Insert the missing observables
and re-run to fit the skipped experiments.

Shared ``run_fit`` options belong in ``run_fit_kwargs``. Use ``run_fit_overrides`` for
per-experiment settings; dict-valued ``run_fit`` options such as
``regularizer_weights`` are passed through unchanged.

To re-fit only the skipped experiments, collect them from the plan/results:

.. code-block:: python

   from mrax.core.study_runner import collect_skipped_experiments

   plan = plan_study_fits(study)
   results, _ = execute_study_plan(plan, run_fit_kwargs={"num_steps": 50})
   skipped = collect_skipped_experiments(plan, results)
   refit_plan = list(skipped.values())
   results_refit, _ = execute_study_plan(refit_plan, run_fit_kwargs={"num_steps": 50})

Visualization
-------------

Human-readable descriptions are available:

.. code-block:: python

   print(study.describe())          # text summary

Add ``save_nifti=True`` to ``save_study_results`` to export NIfTI volumes alongside
the MRD outputs.
