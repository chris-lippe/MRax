Variance Estimation and Weighted Fitting
========================================

MRax can combine inverse-variance statistical weighting with its existing RMS
loss normalization. Use uniform variance by default, supply known variances, or
request a robust pre-fit estimate:

.. code-block:: python

   from mrax.core.loss_builder import run_fit

   fields, losses = run_fit(
       experiment,
       model,
       dataset,
       data_variance="auto",
   )

``data_variance="auto"`` estimates one variance per acquisition frame before
the model optimizer is initialized. The estimate is then fixed throughout the
fit; variance and model parameters are not jointly optimized.

Weighted and scaled objective
-----------------------------

For observation variance :math:`v_i`, MRax assigns precision
:math:`p_i=v_i^{-1}` and minimizes

.. math::

   L_{data} = \frac{\sum_i p_i |f_i-y_i|^2}
                   {s_{RMS}^2\sum_i p_i}.

Here, :math:`s_{RMS}` is the RMS magnitude of the finite measured data. The
precision denominator and RMS factor are constants with respect to the fitted
parameters. The objective therefore has the same optimum as Gaussian
inverse-variance weighted least squares, while retaining an optimizer-friendly
scale across datasets.

The available modes are:

- ``data_variance=None``: uniform variance and the ordinary RMS-scaled mean
  squared residual.
- A positive scalar: uniform explicit variance, equivalent to ``None`` after
  precision normalization.
- A positive per-frame, spatial, or data-broadcastable array: explicit
  heteroscedastic variance.
- ``data_variance="auto"``: robust per-frame estimation before fitting.

Multiplying every supplied variance by a common constant leaves the objective
unchanged. Only relative precision affects the fitted optimum. Consequently,
the normalized loss value is not an absolute Gaussian negative log-likelihood
and should not be used directly for likelihood-curvature uncertainty estimates.

Automatic background selection
------------------------------

Automatic mode first attempts to find low-signal background. For each frame
:math:`t`, selection uses all *other* frames to calculate

.. math::

   R_{-t}(x) = \sqrt{\operatorname{mean}_{j\ne t}|y_j(x)|^2}.

This leave-one-frame-out construction prevents a frame from selecting pixels
simply because their noise happened to be unusually small in that same frame.
MRax then:

1. keeps spatial locations with finite, nonzero support;
2. selects approximately the lowest one percent of :math:`R_{-t}`, enlarged as
   needed to provide enough samples; and
3. accepts the selection only when its median is below 25 percent of the field's
   75th-percentile signal.

If these conditions are not met, automatic mode uses the spatial high-frequency
fallback described below.

Magnitude, signed, and complex data
-----------------------------------

MRax treats data as magnitude MRI when at least 99.5 percent of valid values are
nonnegative. For noise-only single-coil magnitude data, :math:`M` has a Rayleigh
distribution and

.. math::

   \operatorname{median}(M^2) = 2\sigma^2\log 2.

The per-frame estimate is therefore

.. math::

   \hat\sigma_t^2 =
   \frac{\operatorname{median}(M_t^2)}{2\log 2}.

This estimates the underlying quadrature variance, which also approximates the
magnitude-signal variance at sufficiently high SNR.

For signed real data, MRax uses the normal median-absolute-deviation estimator

.. math::

   \hat\sigma_t =
   \frac{\operatorname{median}|x-\operatorname{median}(x)|}
        {0.6744897502}.

For complex data, the real and imaginary channel variances are estimated
separately and added, giving an estimate of :math:`E[|n|^2]`.

Spatial high-frequency fallback
-------------------------------

When no convincing background exists, MRax calculates normalized diagonal Haar
coefficients over non-overlapping :math:`2\times2` spatial blocks:

.. math::

   h = \frac{x_{00}-x_{01}-x_{10}+x_{11}}{2}.

This cancels constant and linear spatial structure. For independent pixels with
equal noise variance, the normalization gives
:math:`\operatorname{Var}(h)=\sigma^2`. MRax applies the MAD estimator to these
coefficients. If only one spatial axis is available, it instead uses normalized
first differences

.. math::

   h_i = \frac{x_{i+1}-x_i}{\sqrt{2}}.

Exactly constant blocks are excluded to reduce zero-padding bias.

Minimum support and shrinkage
-----------------------------

Each frame requires at least ``min_samples`` valid samples or coefficients; the
default is 32. Raw estimates are stabilized toward the median usable frame
variance:

.. math::

   \widetilde v_t = \alpha_t\widehat v_t
       +(1-\alpha_t)v_{pooled},
   \qquad
   \alpha_t = \frac{n_t}{n_t+\lambda},

where :math:`n_t` is the sample count and the default shrinkage strength
:math:`\lambda` is 32. Frames without enough usable support receive the pooled
variance and report a warning. A small positive floor prevents infinite
precision.

Noise masks and diagnostics
---------------------------

The fitting ``mask`` identifies pixels whose model residuals should contribute
to the objective. It is deliberately not interpreted as a noise mask. If a
known background or noise-only region is available, provide it separately:

.. code-block:: python

   fields, losses = run_fit(
       experiment,
       model,
       dataset,
       mask=tissue_mask,
       mask_dims=("x", "y"),
       data_variance="auto",
       variance_estimation_mask=background_mask,
       variance_estimation_mask_dims=("x", "y"),
   )

Call the estimator directly to inspect its result before fitting:

.. code-block:: python

   from mrax.core import estimate_data_variance

   estimate = estimate_data_variance(experiment, dataset)
   print(estimate.method)
   print(estimate.variance)
   print(estimate.relative_variance)
   print(estimate.sample_count)
   print(estimate.warnings)

:class:`mrax.core.variance.VarianceEstimate` also exposes
``standard_deviation`` and ``pooled_variance``. Available direct estimator
methods are ``"auto"``, ``"background"``, and ``"wavelet_mad"``.

Assumptions and limitations
---------------------------

The weighted fitting objective assumes independent observations with fixed
variances. Automatic estimation assumes additive Gaussian noise, or the usual
Gaussian approximation to magnitude MRI in the fitted tissue at sufficiently
high SNR.

Use acquisition-specific explicit variances when working with:

- low-SNR magnitude signals where Rician bias materially affects the model;
- multi-coil noncentral-chi magnitude noise;
- parallel-imaging or compressed-sensing spatial noise fields;
- correlated reconstruction noise; or
- repeat measurements that provide a direct experimental variance estimate.

In these cases, pass the resulting positive variance vector or map through
``data_variance``. MRax will retain its relative weighting while applying the
same RMS scale normalization.
