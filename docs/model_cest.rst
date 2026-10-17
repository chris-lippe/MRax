Field mapping and Lorentzian models
===================================

WASABI / B0-B1 mapping (``make_wasabi_model``)
----------------------------------------------

- Reference: the WASABI signal model follows Schuenke et al. [Schuenke2017]_.
- Defining equation, with :math:`\Delta\omega=2\pi B_0(\Delta_{ppm}-\delta_{ppm})`
  in rad/s and :math:`\tau=t_{sat}` in seconds:

  .. math::

     \theta &= \arctan\left(\frac{\gamma B_1}{\Delta\omega}\right), \\
     \phi &= \frac{\tau}{2}\sqrt{(\gamma B_1)^2+\Delta\omega^2}, \\
     S &= \left|c-d\sin^2\theta\sin^2\phi\right|.
- Parameters: ``c`` and ``d`` (scaling), ``b1`` (uT), ``delta`` (Hz, B0 offset). Derived maps return B0 (ppm) and
  B1 deviation (% of nominal).
- Observables: frequency ``offsets`` in ppm (converted using the study field strength), optional ``sat_time`` (ms)
  and nominal ``sat_b1`` (uT). ``field_strength_mhz`` must be provided once per study via ``Study.study_globals``.
- Default parameter bounds: ``b1`` around ``[3, 5]`` uT and ``delta`` around
  ``[-100, 100]`` Hz, converted to rad/s internally.

Multi-pool Lorentzian (``make_multipool_lorentzian_model``)
-----------------------------------------------------------

- References: the model is an empirical Lorentzian Z-spectrum decomposition in
  the CEST/MR Z-spectroscopy tradition [Liu2013]_ [ZaissBachert2013]_,
  including Lorentzian-line-fit separation of pool contributions [Zaiss2011]_.
- Defining equation: with :math:`\omega=2\pi(\Delta_{ppm}-B_{0,ppm})`,
  :math:`\omega_i=2\pi\delta_{i,ppm}`, and :math:`h_i=G_i/2`,

  .. math::

     L_i(\omega) = a_i\frac{h_i^2}{(\omega-\omega_i)^2+h_i^2},
     \qquad Z(\omega)=Z_{base}-\sum_iL_i(\omega).
- Observables: ``offsets`` in ppm. Saturation-transfer offsets and pool frequency
  shifts are handled as angular ppm by multiplying those ppm values by
  :math:`2\pi`. Empirical linewidth and rate parameters stay in the units declared
  by the parameter library.
- Default pools and bounds (see ``parameter_library.toml`` for the editable source):
  - ``water``: :math:`a \in [0.3, 1.0]`, :math:`G \in [1, 100]`, fixed :math:`\delta = 0`.
  - ``mt``: :math:`a \in [0.02, 0.4]`, :math:`G \in [50, 500]`, :math:`\delta \in [-4, -2]`.
  - ``glu``: :math:`a \in [0.01, 0.3]`, :math:`G \in [1, 20]`, :math:`\delta \in [2.5, 3.5]`.
  - ``glc``: :math:`a \in [0.01, 0.3]`, :math:`G \in [2, 50]`, :math:`\delta \in [1.5, 2.5]`.
- Pools can be reordered/overridden via ``pools=`` and ``pool_overrides=``. Unknown pools can be
  provided by passing a param-spec dict for that pool or by loading a private parameter-library
  overlay.
