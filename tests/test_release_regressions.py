import json
import tomllib
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from mrax import MRAX_VERSION, MRAX_VERSION_TAG, __version__
from mrax.core.loss_builder import run_fit
from mrax.data_struct import DataSet, Quantity
from mrax.data_struct.storage import _resolve_manifest_path
from mrax.gui.app import _mrax_package_entries
from mrax.library import (
    build_default_experiment,
    make_dki_kurtosis_model,
    make_dti_tensor_model,
    make_t2_se_model,
)
from mrax.versioning import versioned_payload


def test_global_mrax_version_is_the_packaging_and_file_tag_source():
    project = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())

    assert MRAX_VERSION == __version__ == "0.3.0"
    assert MRAX_VERSION_TAG == "mrax_version"
    assert project["project"]["dynamic"] == ["version"]
    assert project["tool"]["setuptools"]["dynamic"]["version"] == {
        "attr": "mrax._version.MRAX_VERSION"
    }
    assert versioned_payload({}) == {MRAX_VERSION_TAG: MRAX_VERSION}


def test_constant_data_fit_uses_bounded_rms_scaling():
    echo_times = jnp.asarray([8.0, 24.0, 48.0, 96.0])
    experiment = build_default_experiment(
        "mse",
        frame_count=4,
        spatial_shape=(2, 2),
        observables={"TE": Quantity(echo_times, "ms")},
    )
    fields, losses = run_fit(
        experiment,
        make_t2_se_model(scale=1.0),
        DataSet(jnp.ones((4, 2, 2)), name="constant"),
        num_steps=20,
        step_size=0.03,
    )

    assert losses[0] < 1.0
    assert losses[-1] < losses[0]
    assert np.isfinite(losses).all()
    assert np.isfinite(np.asarray(fields["t2"].values)).all()


def test_experiment_help_is_multiline():
    experiment = build_default_experiment("mse", 1, (2, 2), {})
    help_text = experiment.help()

    assert "\\n" not in help_text
    assert len(help_text.splitlines()) > 3


def test_diffusion_models_are_exported_from_public_library_namespace():
    assert make_dti_tensor_model().name == "dti"
    assert make_dki_kurtosis_model().name == "dki"


def test_manifest_paths_must_remain_below_manifest_directory(tmp_path):
    with pytest.raises(ValueError, match="outside manifest directory"):
        _resolve_manifest_path(tmp_path / "study", "../secret.npy", label="image")


def test_package_entries_must_remain_below_package_directory(tmp_path):
    package = tmp_path / "package"
    package.mkdir()
    (package / "mrax_package.json").write_text(
        json.dumps(
            {
                "format": "mrax.gui.package",
                "version": 1,
                "mrax_version": __version__,
                "entries": {"gui_state": "../outside.json"},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="outside package directory"):
        _mrax_package_entries(package)
