![MRax logo](MRax_logo.png)

# MRax

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21723327.svg)](https://doi.org/10.5281/zenodo.21723327)

MRax is a JAX-based Python library for quantitative MRI research. It provides
data structures for experiments and studies, differentiable signal models,
optimization utilities, reconstruction helpers, analysis tools, and ISMRMRD-backed
storage for images, k-space, and fitted maps.

Disclaimer: MRax is intended for research use only. This is an early release, and bugs may occur.
Coding agents were used to assist in development. Code was manually reviewed and validated on Bruker ParaVision 360 3.5–3.7 data.

## Installation

Install the release package from PyPI:

```bash
pip install mrax
```

The default install includes fitting, Cartesian and non-Cartesian reconstruction,
import, built-in denoising, registration, analysis, storage, and desktop GUI support.
External denoising backends are installed explicitly:

```bash
pip install "mrax[denoise-external]"
```

The upstream `bm3d` and `bm4d` packages are licensed for non-commercial use;
review their licenses before installing
the `denoise-external` extra. Development, documentation, and notebook authoring tools
remain outside the base runtime package.

MRax uses the `mrax.*` import namespace:

```python
from mrax.core.loss_builder import run_fit
from mrax.library import make_t2_se_model
```

Contributor extras are available for notebook and documentation workflows:

```bash
pip install "mrax[notebooks]"
pip install "mrax[docs]"
```

For local development from a clone:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[all,dev,docs,notebooks]"
pytest
```

## Dev Container

The repository includes two VS Code compatible dev container configurations that
share `docker/Dockerfile.dev`:

| Configuration | File | GPU support |
| --- | --- | --- |
| MRax development (CPU), default | `.devcontainer/devcontainer.json` | No GPU required; standard JAX installation |
| MRax development (NVIDIA GPU), opt-in | `.devcontainer/gpu/devcontainer.json` | NVIDIA GPU passthrough and CUDA 12-enabled JAX |

Open the public clone in VS Code and choose **Dev Containers: Reopen in Container**.
Select **MRax development (CPU)** for the default setup, which works on machines
without a GPU. Both configurations install all runtime features plus development,
documentation, and notebook tooling, then install MRax in editable mode. The CPU
configuration uses:

```bash
python -m pip install -e ".[all,dev,docs,notebooks]"
```

For NVIDIA GPU development, select **MRax development (NVIDIA GPU)** instead. From
an open dev container, use **Dev Containers: Switch Container** to change the
configuration. The GPU option also installs `jax[cuda12]>=0.4.35,<0.11` and requires
an NVIDIA GPU and compatible driver on the Docker host. On Linux, install and
configure the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
for Docker. On Windows, follow [Docker Desktop's WSL 2 GPU setup](https://docs.docker.com/desktop/features/gpu/).

After changing a configuration, run **Dev Containers: Rebuild Container** to apply
the startup options and dependency installation. Inside the GPU container, verify
that JAX lists a CUDA device:

```bash
python -c "import jax; print(jax.devices())"
```

Inside the container, common checks are:

```bash
pytest
sphinx-build -E -b html docs docs/_build/html
jupyter lab notebooks
```

## Documentation Website

Documentation is built with Sphinx:

```bash
python -m pip install -e ".[docs]"
sphinx-build -E -b html docs docs/_build/html
```

Visit the GitHub-hosted documentation at [https://chris-lippe.github.io/MRax/](https://chris-lippe.github.io/MRax/).

## PyPI Release

Build and inspect a local distribution with:

```bash
python -m pip install -e ".[all,dev,docs,notebooks]"
python -m build
twine check dist/*
```
## Quick Example

```python
import jax.numpy as jnp

from mrax.core.loss_builder import run_fit
from mrax.data_struct import DataSet, Quantity
from mrax.library import build_default_experiment, make_t2_se_model

te = jnp.asarray([8.0, 24.0, 48.0, 96.0])
experiment = build_default_experiment(
    "mse",
    frame_count=te.size,
    spatial_shape=(2, 2),
    observables={"TE": Quantity(te, "ms")},
)
model = make_t2_se_model(scale=1.0)
dataset = DataSet(data=jnp.ones((te.size, 2, 2)), name="demo")

fields, losses = run_fit(experiment, model, dataset, num_steps=20, step_size=0.03)
print(fields["t2"].si_unit, losses[-1])
```
