![MRax logo](MRax_logo.png)

# MRax

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21723327.svg)](https://doi.org/10.5281/zenodo.21723327)
[![Documentation](https://img.shields.io/badge/Documentation-1f77b4?logo=readthedocs&logoColor=white)](https://chris-lippe.github.io/MRax/)

MRax is a JAX-based Python library for quantitative MRI research. It combines
unit-aware experiments and studies, differentiable signal models, Optax and
proximal optimizers, MRI reconstruction, analysis, and portable `.mrax` bundles.

MRax is intended for research use. Coding agents assisted development.

## Install

```bash
pip install mrax
```

The base package includes fitting, reconstruction, import, preprocessing,
registration, analysis, storage and the desktop GUI. Optional BM3D/BM4D
denoisers use `pip install "mrax[denoise-external]"` under their upstream
non-commercial licenses.
See [installation](https://chris-lippe.github.io/MRax/setup.html) and
[development containers](https://chris-lippe.github.io/MRax/docker_setup.html)
for platform and GPU setup.

## Quick example

```python
import jax.numpy as jnp

from mrax.core import run_fit
from mrax.data_struct import DataSet, Quantity
from mrax.library import build_default_experiment, make_t2_se_model

te = jnp.asarray([8.0, 24.0, 48.0, 96.0])
experiment = build_default_experiment(
    "mse", frame_count=te.size, spatial_shape=(2, 2),
    observables={"TE": Quantity(te, "ms")},
)
dataset = DataSet(data=jnp.ones((te.size, 2, 2)), name="demo")
fields, history = run_fit(
    experiment, make_t2_se_model(scale=1.0), dataset,
    num_steps=20, step_size=0.03,
)
```

## Documentation

- [Overview and workflows](https://chris-lippe.github.io/MRax/overview.html)
- [Fitting guide](https://chris-lippe.github.io/MRax/model_fitting.html)
- [API reference](https://chris-lippe.github.io/MRax/api.html)
- [Tutorial notebooks](https://chris-lippe.github.io/MRax/notebooks.html)
- [Contributing and local checks](https://chris-lippe.github.io/MRax/contributing.html)


## How to cite

If you use MRax in your research, please cite the specific release you used. Select the release used from [MRax on Zenodo](https://doi.org/10.5281/zenodo.21723327)
and copy its version-specific citation.

For a general reference to the project, use:

> Lippe, C. *MRax: JAX-based tools for quantitative MRI research* [Computer software].
> Zenodo. [https://doi.org/10.5281/zenodo.21723327](https://doi.org/10.5281/zenodo.21723327)
