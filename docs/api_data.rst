.. _api-data:

Data and study containers
=========================

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
