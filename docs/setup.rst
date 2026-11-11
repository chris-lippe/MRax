Installation
============

Python toolbox
--------------

Install into a Python 3.12 or newer virtual environment:

.. code-block:: bash

   python -m pip install mrax

This includes CPU fitting, reconstruction, analysis, common MRI I/O and MRax
storage and workflow files. Use ``mrax[full]`` for specialized raw importers,
SimpleITK registration, DIPY interoperability and plotting. Use ``mrax[gui]``
for those scientific tools and the browser GUI, started with ``mrax-gui``.
Built wheels include the frontend. Node is needed only for frontend development.

Compressed DICOM codecs are available through ``mrax[dicom-codecs]``.
The optional ``mrax[denoise-external]`` denoisers have their own upstream
non-commercial licenses.

Windows desktop distribution
----------------------------

The standalone CPU installer includes private Python and browser runtimes.
Install it into your user folder and start **MRax.exe**, or its shortcut.
It does not require a separate Python installation or use an existing Python
environment. Save workspaces and exports in your chosen data folders.
GPU acceleration is not included in this installer.

Contributors
------------

Choose the contributor topic you need:

- :doc:`docker_setup`: development images, VS Code containers and NVIDIA GPUs.
- :doc:`windows_setup`: WSL 2, Docker Desktop and native Windows installs.
- :doc:`contributing`: editable installs, checks and documentation builds.
- :doc:`importer_development`: importer architecture and extension points.

These pages are also available directly from the Overview navigation.
