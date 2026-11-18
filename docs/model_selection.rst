Model selection
===============

Identity model
--------------

- Defining equation: :math:`S_j(\mathbf r)=A(\mathbf r)` for every acquisition
  frame :math:`j`.  The amplitude map is broadcast across frames. When its leading
  axis already has the frame length, it is returned unchanged.
- Parameters: ``amplitude`` (unitless).  This is the default placeholder for
  other experiment types, with a constant signal across acquisition settings.

Using models with importers
---------------------------

- ``sat_trans`` scans use the identity model unless you explicitly pick one. Importing them
  is quiet. Fitting the default identity model warns because it only estimates an amplitude
  image. Provide a ``model_map_by_type`` when calling
  :func:`mrax.io.bruker_import.import_bruker_study` to select a saturation-transfer model:

  .. code-block:: python

     from mrax.library.default_models import make_multipool_lorentzian_model
     model_map = {"sat_trans": make_multipool_lorentzian_model}
     study, meta = import_bruker_study(bruker_root, model_map_by_type=model_map)

- Pool bounds and observable fallbacks can be customized through the model factory.
  See :doc:`model_parameters` and the chosen model's page.
- For bespoke signal equations, copy ``docs/model_template.py`` and adapt ``model_apply`` / ``parametrization_factory``.

Model-based reconstruction
--------------------------

Use any :class:`mrax.data_struct.model.Model` as a non-linear model operator in
:func:`mrax.reco.variational.solve_model_based`. It reconstructs parameter maps
directly from raw k-space.
