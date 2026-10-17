Bruker import functions
=======================

Import ParaVision images, raw acquisitions, geometry and model observables.
Study import warns and skips incomplete or unsupported experiments while retaining
valid experiments. Import and numerical reconstruction are separate operations.
See :doc:`bruker_import` and :doc:`importer_development` for definitions and options.

.. autofunction:: mrax.io.import_bruker_experiment
.. autofunction:: mrax.io.import_bruker_study
.. autofunction:: mrax.io.read_bruker_acquisition
.. autofunction:: mrax.io.load_acquisition
.. autofunction:: mrax.io.save_acquisition
.. autoclass:: mrax.io.DefinitionLibrary
   :members: select

.. autofunction:: mrax.io.study_selection.auto_select_scans
.. autofunction:: mrax.io.study_selection.pick_scan_by_type
