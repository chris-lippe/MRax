Docker development setup
========================

Docker Image
------------

The repository provides one development Dockerfile under ``docker/``:

.. list-table::
   :widths: 1 3
   :header-rows: 0

   * - ``Dockerfile.dev``
     - Python 3.12 development image with all MRax runtime feature dependencies
       including the GUI stack, plus test, documentation, notebook, Git, Git LFS,
       and native system packages commonly needed by development workflows.

For day-to-day development, use ``docker/Dockerfile.dev``.

Build the Development Container
-------------------------------

From the repository root, build the development image:

.. code-block:: bash

   docker build -f docker/Dockerfile.dev -t mrax-dev .

Run an interactive shell with the current checkout mounted at ``/workspace``:

.. code-block:: bash

   docker run --rm -it -v "$PWD":/workspace -w /workspace mrax-dev

Inside the container, install the mounted checkout in editable mode:

.. code-block:: bash

   python -m pip install -e ".[all,dev,docs,notebooks]"

Run the standard checks:

.. code-block:: bash

   python -m pytest
   sphinx-build -E -b html docs docs/_build/html

To start Jupyter from the container and access it on the host:

.. code-block:: bash

   docker run --rm -it -p 8888:8888 -v "$PWD":/workspace -w /workspace mrax-dev
   python -m pip install -e ".[all,dev,docs,notebooks]"
   jupyter lab --ip=0.0.0.0 --no-browser --NotebookApp.token=''

Then open ``http://localhost:8888`` on the host.

VS Code Dev Container
---------------------

Choose a dev container configuration, both using ``docker/Dockerfile.dev``:

- **CPU (default):** ``.devcontainer/devcontainer.json``; standard CPU JAX.
- **NVIDIA GPU:** ``.devcontainer/gpu/devcontainer.json``; GPU passthrough and
  CUDA 12-enabled JAX.

With Docker running, use ``Dev Containers: Reopen in Container`` and select a
configuration. The post-create step installs MRax with runtime and development
extras:

.. code-block:: bash

   python -m pip install -e ".[all,dev,docs,notebooks]"

NVIDIA GPU Development
~~~~~~~~~~~~~~~~~~~~~~

The GPU configuration runs MRax in a Linux container, including on Windows through
WSL 2. It enables ``--gpus=all`` and installs CUDA-enabled JAX:

.. code-block:: bash

   python -m pip install -e '.[all,dev,docs,notebooks]' 'jax[cuda12]>=0.4.35,<0.11'

The host needs an NVIDIA GPU and compatible driver. Configure the
`NVIDIA Container Toolkit
<https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>`_
on Linux or `Docker Desktop WSL 2 GPU support
<https://docs.docker.com/desktop/features/gpu/>`_ on Windows. JAX's pip extra
supplies CUDA libraries; see `JAX requirements
<https://docs.jax.dev/en/latest/installation.html#nvidia-gpu>`_.

After switching configurations, run ``Dev Containers: Rebuild Container``. Verify
GPU access inside the container:

.. code-block:: bash

   python -c "import jax; print(jax.devices())"
