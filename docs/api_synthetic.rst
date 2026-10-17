.. _api-synthetic:

Synthetic examples
==================

Synthetic datasets that use the same containers as imported studies.

- ``make_example_suite(noise_level=0.05, seed=0, model_names=("t1_ir","t2_se"))``
  → list of :class:`mrax.core.synthetic.Example` objects.
- ``make_synthetic_study(...)`` → ``(Study, metadata)``; the metadata includes ground
  truth fields for benchmarking.

.. autoclass:: mrax.core.synthetic.Example

.. autofunction:: mrax.core.synthetic.make_example_suite
.. autofunction:: mrax.core.synthetic.make_synthetic_study
