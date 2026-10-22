AB-MT+n model
=============

- Reference: AB-MT+n is the saturation-transfer model introduced by Lippe and Hoerr
  [Lippe2026]_. Semisolid MT line-shape handling follows the
  super-Lorentzian MT modeling context of Morrison, Stanisz, and Henkelman
  [Morrison1995]_.
- Defining equations: use the corrected angular offset :math:`\omega=2\pi\Delta_{ppm}`
  and :math:`\omega_1=\gamma B_{1,eff}/B_0`.  Let
  :math:`P=|\omega|/\sqrt{\omega^2+\omega_1^2}` (or :math:`P=1` for
  ``spin_lock``).  For a non-MT pool with fraction :math:`f`, rate :math:`k`, and
  offset :math:`\delta`, the off-resonance contribution is

  .. math::

     D &= 4\omega_1^2\delta^2+(k^2+\delta^2)^2, \\
     a &= \frac{2\delta^3fk}{D},\qquad
     b = \frac{\delta^2fk(\delta^2+k^2)}{D},\qquad
     d = \frac{\delta^2fk(3\delta^2-k^2)}{D}, \\
     F(\omega) &= \frac{d-a\omega}{(\omega-\delta)^2+k^2+\omega_1^2}
     +\frac{b+a\omega}{\omega^2+\omega_1^2}.

  For the MT pool, with :math:`u=\omega-\delta_{mt}`, the selected RF rate is

  .. math::

     R_{rf}^{Lor}(u) &= \frac{\omega_1^2R_{2,mt}}{R_{2,mt}^2+u^2}, \\
     R_{rf}^{Gau}(u) &= \frac{\omega_1^2\sqrt{\pi/2}}{R_{2,mt}}
     \exp\left[-\left(\frac{u}{2R_{2,mt}}\right)^2\right], \\
     R_{rf}^{SL}(u) &= \omega_1^2\frac{L_{SL}(|u|/R_{2,safe})}{R_{2,safe}},
     \qquad R_{2,safe}=|R_{2,mt}|.

  :math:`L_{SL}` is the package's tabulated super-Lorentzian kernel, linearly
  interpolated over its stored domain and set to zero outside it.  The complete
  rate and signal equations are

  .. math::

     R_{ex,mt} &= R_{rf}(\omega-\delta_{mt},\omega_1,R_{2,mt})
     \frac{f_{mt}}{1+f_{mt}}\frac{\omega^2}{\omega^2+\omega_1^2}, \\
     R_{1\rho} &= \frac{R_{1,obs}\omega^2+b_0\omega_1^2+a_0\omega_1^2\omega}
     {\omega^2+\omega_1^2}+R_{ex,mt}
     +\sum_{i\ne mt}\frac{\omega_1^2F_i(\omega)}{1+f_{mt}}, \\
     Z_{ss} &= \frac{R_{1,obs}P|\omega|/\sqrt{\omega^2+\omega_1^2}}
     {R_{1\rho}}, \\
     Z_{rec} &= 1+\left(\sqrt{\max(\sin^{-2}\alpha_{eff}-1,0)}\,Z_{prev}-1\right)
     \exp(-R_{1,obs}t_{rec}), \\
     Z &= 1-C+C\left[(Z_{rec}P^2-Z_{ss})\exp(-R_{1\rho}t_{sat})+Z_{ss}\right],
     \qquad C=\exp(-R_{1,obs}t_{delay}).

  Here :math:`t_{sat}`, :math:`t_{delay}`, and :math:`t_{rec}=TR-t_{sat}-t_{delay}`
  are converted to the implementation's ppm-time convention by multiplying
  seconds by the study field strength.  :math:`a_0`, :math:`b_0`, and
  :math:`\sin\alpha_{eff}` are fitted AB-MT+n parameters; :math:`Z_{prev}=1`
  unless supplied as an observable.
- Observables: ``freq_offsets`` (ppm), ``sat_b1`` (uT), ``sat_time`` (ms), ``TR`` (ms),
  plus a study-wide ``field_strength_mhz``.
- Optional maps: ``r1_map`` or ``t1_map`` (required), ``b0_map``/``b1_map`` for B0/B1 corrections.
- Use :func:`mrax.data_struct.join_experiments` to stack multiple sat_trans scans with
  different ``sat_b1``/``sat_time`` into a single fit. Repeated acquisitions (``t``)
  expand parameter maps, so a single ``run_fit`` returns ``(t, x, y)`` fields
  without splitting; you can regularize along ``t`` or set ``parameter_constant_dims`` to
  share parameters. If you want the repeat axis explained by a side model, insert the
  derived observable directly into the experiment (``Experiment.insert_observables``)
  and map the repeat dimension to it.
