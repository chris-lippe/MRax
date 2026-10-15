Docker Setup and Contribution Guide
===================================

This chapter covers the Docker-based development path for MRax, a minimal Windows
setup using WSL 2 and Docker Desktop, and the contribution workflow.

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

- **CPU (default):** ``.devcontainer/devcontainer.json``; no GPU required.
- **NVIDIA GPU:** ``.devcontainer/gpu/devcontainer.json``; GPU passthrough and
  CUDA 12-enabled JAX.

With Docker running, use ``Dev Containers: Reopen in Container`` and select a
configuration. The post-create step installs MRax with runtime and development
extras:

.. code-block:: bash

   python -m pip install -e ".[all,dev,docs,notebooks]"

NVIDIA GPU Development
~~~~~~~~~~~~~~~~~~~~~~

The GPU configuration enables ``--gpus=all`` and installs CUDA-enabled JAX:

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

Contribution Guide
------------------

Development workflow:

1. Create a branch from the current development branch.
2. Use the dev container or install the editable checkout locally:

   .. code-block:: bash

      python -m pip install -e ".[all,dev,docs,notebooks]"

3. Add tests for public APIs, importer output, storage contracts, or numerical
   conventions. Use synthetic inputs, not local scanner data or private files.
4. Update the affected docs page when units, shapes, model assumptions, dependencies,
   or validation status change.

Run these checks from the repository root before opening a pull request:

.. code-block:: bash

   python -m ruff check mrax tests
   python -m isort --check-only mrax tests
   python -m pyright
   python -m coverage run -m pytest
   python -m coverage report
   python -m sphinx -W --keep-going -E -b html docs docs/_build/html
   git diff --check

For documentation-only changes, run the Sphinx build and ``git diff --check``. For code
changes, run the test suite. If a check cannot run in your environment, mention it in
the pull request.

Docs changes:

- Add or update method citations in :doc:`references`.
- Use generated data in examples unless the data are included in the repository.

Dependency and release changes:

- Runtime dependencies live in ``pyproject.toml`` when required by ``pip install mrax``.
- Development, documentation, and notebook authoring tools belong in contributor extras.
- Release notes list user-facing changes.

Minimal WSL 2 and Docker Setup
------------------------------

On Windows, the most direct setup is WSL 2 plus Docker Desktop with WSL
integration enabled.

1. Install WSL 2 from PowerShell:

   .. code-block:: powershell

      wsl --install

2. Restart Windows if prompted, then open the installed Linux distribution and
   confirm it is using WSL 2:

   .. code-block:: powershell

      wsl --status
      wsl --list --verbose

3. Install Docker Desktop for Windows and enable WSL integration:

   - Open Docker Desktop settings.
   - Enable ``Use the WSL 2 based engine``.
   - Under ``Resources -> WSL Integration``, enable integration for your Linux
     distribution.

4. In the WSL terminal, install basic development tools and clone the repository:

   .. code-block:: bash

      sudo apt update
      sudo apt install -y git build-essential
      git clone <repository-url>
      cd <repository-name>

5. Confirm Docker is available from WSL:

   .. code-block:: bash

      docker version
      docker run --rm hello-world

6. Build and enter the MRax development container:

   .. code-block:: bash

      docker build -f docker/Dockerfile.dev -t mrax-dev .
      docker run --rm -it -v "$PWD":/workspace -w /workspace mrax-dev

For best filesystem performance, keep the repository inside the WSL Linux
filesystem, for example under ``~/code/``, instead of editing it from
``/mnt/c/``.

Native Windows Pip Installs
---------------------------

Developing MRax in a Linux container does not make the Python package
Linux-only. The base release package is tested with native ``pip`` installations when
its dependencies provide Windows wheels:

.. code-block:: powershell

   python -m pip install --upgrade pip
   python -m pip install mrax

GUI, registration, reconstruction, denoising, and importer workflows may have
platform-specific requirements. WSL 2 plus Docker is the documented Windows
development path for those workflows.

The optional ``mrax[nufft]`` extra is not supported by a Windows wheel from
``jax-finufft``. Use WSL 2 or the development container for that backend. The optional
``mrax[denoise-external]`` extra installs upstream BM3D/BM4D packages whose licenses
permit non-commercial use only.
