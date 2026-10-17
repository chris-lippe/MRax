Zaiss analytical model
======================

- Reference: implements the approximate continuous-wave analytical Z-spectrum
  model of Zaiss/Trott-Palmer used as the starting point for AB-MT+n
  [ZaissBachert2013]_.
- Defining equations: after B0/residual-water correction, let
  :math:`\omega=2\pi\Delta_{ppm}`, :math:`\omega_1=\gamma B_{1,eff}/B_0`,
  :math:`c_\theta=|\omega|/\sqrt{\omega^2+\omega_1^2}`, and
  :math:`P_z=c_\theta` (or :math:`P_z=1` when ``spin_lock`` is enabled).  For each
  pool :math:`i`, with :math:`\omega_i=2\pi\delta_i`, the implemented exchange rate is

  .. math::

     \Gamma_i^2/4 &= \frac{R_{2i}+k_i}{k_i}\omega_1^2+(R_{2i}+k_i)^2, \\
     q_i &= R_{2i}+k_i\frac{\omega_i^2+R_{2i}(R_{2i}+k_i)}
     {\omega^2+\omega_1^2}, \\
     R_{ex,i} &= \frac{f_i\omega_1^2q_i}
     {\Gamma_i^2/4+(\omega-\omega_i)^2}.

  Water relaxation and the steady state are

  .. math::

     R_{eff} &= \frac{R_{1w}\omega^2+R_{2w}\omega_1^2}{\omega^2+\omega_1^2}, \\
     R_{1\rho} &= R_{eff}+R_{ex,mt}+\sum_{i\ne mt}\frac{R_{ex,i}}{1+f_{mt}}, \\
     Z_{ss} &= \frac{R_{1,obs}P_zc_\theta}{R_{1\rho}}, \\
     Z &= (P_z^2Z_i-Z_{ss})\exp(-R_{1\rho}t_{sat})+Z_{ss}.

  :math:`R_{1,obs}` is an inserted ``r1_map`` converted to ppm, or its
  model-based fallback :math:`(R_{1w}+f_{mt}R_{1,mt})/(1+f_{mt})` when the MT
  pool is present (:math:`R_{1w}` otherwise).

  The shipped forward model fixes :math:`Z_i=1`; the factor is shown explicitly
  because it defines the initial longitudinal magnetization term.
- Observables: ``freq_offsets`` (ppm), ``sat_b1`` (uT), ``sat_time`` (ms), plus
  study-wide ``field_strength_mhz``. Optional ``t1_map``/``r1_map``,
  ``b0_map`` and ``b1_map`` are consumed with the same units as AB-MT+n.
- Unit convention: frequency offsets and chemical shifts are converted to
  angular ppm via :math:`2\pi`; saturation amplitude uses
  ``SAT_GAMMA * sat_b1_uT / field_strength_mhz``. Pool rate parameters remain
  in the parameter-library ``ppm`` convention used by AB-MT+n.
