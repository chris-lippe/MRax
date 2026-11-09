"""Exercise a real installed wheel without inheriting developer packages."""

import argparse
import importlib.metadata
import importlib.util
import json
import tempfile
from pathlib import Path


def qualify(profile):
    import jax.numpy as jnp
    import numpy as np

    import mrax
    from mrax.core import run_fit
    from mrax.data_struct import DataSet, Quantity
    from mrax.library import build_default_experiment, make_t2_se_model
    from mrax.reco import fftnc, ifftnc
    from mrax.workflows import WorkflowDocument

    assert "site-packages" in Path(mrax.__file__).resolve().parts
    if profile == "base":
        for package in ("fastapi", "uvicorn", "SimpleITK", "dipy", "pypulseq", "matplotlib"):
            assert importlib.util.find_spec(package) is None, package
    if profile in {"full", "gui"}:
        for package in ("SimpleITK", "dipy", "pypulseq", "mapvbvd", "matplotlib"):
            assert importlib.util.find_spec(package) is not None, package
    if profile == "full":
        assert importlib.util.find_spec("fastapi") is None
        assert importlib.util.find_spec("uvicorn") is None
    if profile == "gui":
        from mrax.web.app import create_app

        with tempfile.TemporaryDirectory() as directory:
            assert create_app(runtime_dir=Path(directory)) is not None
    te = jnp.asarray([8.0, 24.0, 48.0, 96.0])
    experiment = build_default_experiment(
        "mse", frame_count=4, spatial_shape=(2, 2), observables={"TE": Quantity(te, "ms")}
    )
    values = jnp.broadcast_to(jnp.exp(-te[:, None, None] / 70), (4, 2, 2))
    fields, history = run_fit(
        experiment, make_t2_se_model(scale=1.0), DataSet(values),
        num_steps=4, step_size=0.03, device="cpu",
    )
    assert fields and history is not None
    assert all(np.isfinite(np.asarray(value)).all() for value in fields.values())
    array = np.arange(24, dtype=np.float32).reshape(4, 6)
    np.testing.assert_allclose(ifftnc(fftnc(array)), array, atol=1e-5)
    # Importing the shared workflow contract must not require the web server.
    assert WorkflowDocument.model_json_schema()["properties"]
    import mrax.io
    from mrax.preprocessing import denoise_filter

    assert mrax.io.import_bruker_study
    np.testing.assert_allclose(denoise_filter(array, method="gaussian", sigma=0), array)
    return {"profile": profile, "version": importlib.metadata.version("mrax"), "fit": "cpu"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("base", "full", "gui"), required=True)
    print(json.dumps(qualify(parser.parse_args().profile)))
