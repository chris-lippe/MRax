Desktop GUI
===========

MRax includes a Qt desktop GUI for inspecting saved MRax studies, drawing ROIs,
running fits, and exporting maps or masks. It uses the same ``Study``,
``Experiment``, ``DataSet`` and ``Model`` objects as the Python API. Imported
source Bruker files remain intact.

.. toctree::
   :maxdepth: 1

   gui_loading
   gui_rois
   gui_fitting
   gui_registration
   gui_export

Installation
------------

Install and launch:

.. code-block:: bash

   pip install mrax
   mrax-gui
   mrax-gui path/to/study_manifest.json

The Python package includes the GUI dependencies. On minimal Linux systems, Qt may
still need native libraries such as ``libglib-2.0.so.0`` or ``libGL.so.1``:

.. code-block:: bash

   apt-get update
   apt-get install -y libglib2.0-0 libgl1

For a user-local installation, place extracted native libraries in
``~/.local/mrax-gui-libs`` or set ``MRAX_GUI_LOCAL_LIB_DIR``.

The GUI can also be launched from Python:

.. code-block:: python

   import mrax.gui
   mrax.gui.launch("path/to/study_manifest.json")

The GUI defaults ``XLA_PYTHON_CLIENT_PREALLOCATE=false`` to avoid reserving GPU
memory at startup. Set overrides before starting Python.

Plot axes match the image sliders. Selecting one sweeps that dimension while
other dimensions stay at their slider positions. Axes include observable units;
``t`` uses elapsed seconds.
