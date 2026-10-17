Diffusion models
================

Diffusion tensor imaging (``make_dti_tensor_model``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Reference: diffusion tensor MRI follows the tensor formulation introduced by
  Basser, Mattiello, and LeBihan [Basser1994]_.
- Defining equation:

  .. math::

     \mathbf D &=
     \begin{bmatrix}
     d_{xx} & d_{xy} & d_{xz} \\
     d_{xy} & d_{yy} & d_{yz} \\
     d_{xz} & d_{yz} & d_{zz}
     \end{bmatrix}, \qquad
     \mathbf g = \begin{bmatrix}g_x & g_y & g_z\end{bmatrix}^{\mathsf T}, \\
     D_{app}(\mathbf g) &= \mathbf g^{\mathsf T}\mathbf D\mathbf g, \\
     S(b,\mathbf g) &= S_0\exp\left[-bD_{app}(\mathbf g)\right].
- Parameters: ``s0`` (a.u.) plus tensor components ``dxx``, ``dyy``, ``dzz``,
  ``dxy``, ``dxz``, ``dyz`` (mm\ :sup:`2`/s).
- Observables: ``bvals`` (s/mm\ :sup:`2`) and ``bvecs`` (unitless, shape ``(N, 3)``).
- Derived maps: ``adc``/``md`` (mean diffusivity), ``fa`` (fractional anisotropy), ``ad`` (axial diffusivity),
  ``rd`` (radial diffusivity), and ``trace`` (tensor trace).

Diffusion kurtosis (``make_dki_kurtosis_model``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- References: the kurtosis extension follows diffusional kurtosis imaging
  [Jensen2005]_, with MRI context summarized in [JensenHelpern2010]_.
- Defining equation, using the same :math:`D_{app}(\mathbf g)` tensor contraction
  as DTI and a scalar (isotropic) kurtosis field :math:`K`:

  .. math::

     S(b,\mathbf g) = S_0\exp\left[-bD_{app}(\mathbf g)
     +\frac{b^2D_{app}(\mathbf g)^2K}{6}\right].
- Parameters: ``s0`` (a.u.), tensor components ``dxx``...``dyz`` (mm\ :sup:`2`/s), and ``kurtosis`` (unitless).
- Observables: ``bvals`` (s/mm\ :sup:`2`) and ``bvecs`` (unitless, shape ``(N, 3)``).
- Derived maps: DTI metrics plus ``mk`` (mean kurtosis).
