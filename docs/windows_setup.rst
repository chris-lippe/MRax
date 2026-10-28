Windows setup
=============

.. note::

   Native Windows installations use CPU JAX. GPU calculations require
   the Linux version of MRax inside WSL 2, optionally in a Linux Docker container,
   with CUDA-enabled JAX. JAX lists NVIDIA GPU support under WSL 2 as experimental;
   see `JAX platform support
   <https://docs.jax.dev/en/latest/installation.html#supported-platforms>`_.

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

MRax supports Linux development containers and native Windows installations.
The base release package is tested with native ``pip`` installations when
its dependencies provide Windows wheels:

.. code-block:: powershell

   python -m pip install --upgrade pip
   python -m pip install mrax

GUI, registration, reconstruction, denoising, and importer workflows may have
platform-specific requirements. WSL 2 plus Docker is the documented Windows
development path for those workflows.

The optional ``mrax[denoise-external]`` extra installs upstream BM3D/BM4D packages
whose licenses permit non-commercial use only.
