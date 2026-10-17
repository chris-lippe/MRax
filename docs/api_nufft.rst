.. _api-nufft-plans:

NUFFT plans
===========

``prepare_nufft`` uses twofold FFT oversampling with matched Kaiser--Bessel
interpolation and spreading. ``cache_budget_bytes`` (128 MiB by default) bounds
stencil caching. Larger trajectories retain compact positions and evaluate
tabulated kernel weights per chunk.

Explicit adjoint derivative rules avoid storing expanded stencils during
differentiation. Long single-precision adjoints use bounded partial sums, or CPU
double-precision accumulation when JAX x64 is enabled; outputs retain the plan's
single precision.

.. autofunction:: mrax.reco.prepare_nufft

.. autoclass:: mrax.reco.NUFFTPlan
   :members: forward, adjoint
