Relaxometry and perfusion models
================================

T1 inversion recovery (``make_t1_ir_model``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Defining equation:

  .. math::

     S(TI) = \left|A - 2A\exp\left(-\frac{TI}{T_1}\right)\right|.

- Parameters: ``amplitude`` (a.u.), ``t1`` (ms).
- Observables: inversion times ``TI`` (ms) provided either in ``offsets`` or ``experiment.observables``.
- Default parameter bounds: ``t1`` in ``[50, 5000]`` ms via a sigmoid mapping; amplitude constrained positive.
  (See ``parameter_library.toml`` for the editable defaults.)

FAIR ASL (``make_fair_asl_model``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- References: FAIR was introduced for relative cerebral blood-flow and perfusion
  imaging by Kim [Kim1995]_ and Kim/Tsekos [KimTsekos1997]_.
- Defining equation: let :math:`\ell` be ``fair_label`` and define
  :math:`(A_\ell,T_{1,\ell})=(A_{sel},T_{1,sel})` for :math:`\ell<0.5`, otherwise
  :math:`(A_{non},T_{1,non})`.  The frame signal is

  .. math::

     S(TI,\ell) = \left|A_\ell - 2A_\ell\exp\left(-\frac{TI}{T_{1,\ell}}\right)\right|.

- Parameters: ``amplitude_sel``/``t1_sel`` and ``amplitude_non``/``t1_non``.
- Observables: ``TI`` (ms) and ``fair_label`` (0=selective, 1=non-selective).
- Derived maps: ``t1_sel``, ``t1_non``, and ``perfusion_map`` (ml/100g/min).
- Defining derived-perfusion equation:

  .. math::

     f = \lambda\,60000\,\frac{T_{1,non}}{T_{1,blood}}
     \left(\frac{1}{T_{1,sel}}-\frac{1}{T_{1,non}}\right).

  Defaults to ``t1_blood=2430`` ms and ``partition_coeff=90``; override via observables or study globals with keys
  ``t1_blood``/``t1_blood_ms`` and ``partition_coeff``/``partition_coefficient``/``lambda``.
- FAIR scans default to experiment type ``fair_asl`` and use this model.

T1 variable TR (``make_t1_vtr_model``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Defining equation: :math:`S(TR) = A \left(1 - \exp(-TR/T_1)\right) + c`.
- Parameters: ``amplitude`` (a.u.), ``t1`` (ms), ``bias`` (a.u.).
- Observables: repetition times ``TR`` (ms).
- Default parameter bounds: ``t1`` in ``[50, 5000]`` ms, amplitudes/bias scaled positive.

T1 variable flip angle (``make_t1_vfa_model``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Reference: the spoiled-GRE variable-flip-angle formulation follows the
  DESPOT-style T1 mapping literature [Deoni2003]_.
- Defining equation (with :math:`\alpha` converted from degrees to radians):

  .. math::

     E_1 &= \exp(-TR/T_1), \\
     S(TR,\alpha) &= A\frac{(1-E_1)\sin\alpha}{1-E_1\cos\alpha}+c.
- Parameters: ``amplitude`` (a.u.), ``t1`` (ms), ``bias`` (a.u.).
- Observables: repetition time ``TR`` (ms) plus flip angles (deg) provided via
  ``flip_angle``/``flip_angles``/``FA`` in observables or directly as ``offsets``.
- Default parameter bounds: ``t1`` in ``[50, 5000]`` ms, amplitudes/bias scaled positive.

Exponential decay, T2, and T2* (``make_exp_decay_model``, ``make_t2_se_model``, ``make_t2star_se_model``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Shared defining equation: :math:`S(TE) = A\exp(-TE/\tau)+c`.
- The generic ``make_exp_decay_model`` exposes ``decay_time`` (ms).
- ``make_t2_se_model`` uses the same model with the decay parameter named ``t2``;
  multi-spin-echo (``mse``) experiments default to this model.
- Defining T2 equation: :math:`S(TE) = A\exp(-TE/T_2)+c`.
- Parameters: ``amplitude`` (a.u.), ``t2`` (ms), ``bias`` (a.u.).
- Observables: echo times ``TE`` (ms) via ``offsets`` or ``experiment.observables``.
- Default parameter bounds: ``t2`` in ``[5, 200]`` ms, amplitudes positive.
- ``make_t2star_se_model`` uses the same model with the decay parameter named
  ``t2star``; multi-gradient-echo (``mge``) experiments default to this model.
- Defining T2* equation: :math:`S(TE) = A\exp(-TE/T_2^*)+c`.
- Parameters: ``amplitude`` (a.u.), ``t2star`` (ms), ``bias`` (a.u.).
- Observables: echo times ``TE`` (ms).
- Default parameter bounds: ``t2star`` in ``[1, 200]`` ms, amplitudes positive.
