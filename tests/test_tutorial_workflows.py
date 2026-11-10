"""Tutorial practice data is independent, reconstructable and quantitative."""

import numpy as np
from fastapi.testclient import TestClient

from mrax.gui.services.provenance import pipeline_sources
from mrax.gui.services.synthetic import SYNTHETIC_T2_MS, synthetic_tutorial_session
from mrax.gui.services.workspace import StudyWorkspace
from mrax.reco import ifftnc
from mrax.web.app import create_app


def test_tutorial_cartesian_samples_reconstruct_the_reference_images():
    session = synthetic_tutorial_session()
    assert session.study_name == "Tutorial — synthetic T2"
    assert session.frame_count == 8
    assert session.experiment_dims == {"frame": 8, "x": 24, "y": 32, "z": 3}
    acquisition = session.current_experiment["acquisition"]
    acquisition.require_prepared()
    assert acquisition is session.study.datasets[0].acquisition
    reconstructed = np.asarray(ifftnc(acquisition.samples, axes=(-3, -2, -1)))[:, 0]
    np.testing.assert_allclose(reconstructed.real, session.image_array, rtol=3e-6, atol=0.2)
    np.testing.assert_allclose(reconstructed.imag, 0, atol=0.2)
    times = np.asarray(session.build_experiment().observables["TE"].values)
    np.testing.assert_allclose(
        session.image_array[:, 12, 16, 1] / session.image_array[0, 12, 16, 1],
        np.exp(-(times - times[0]) / SYNTHETIC_T2_MS),
        rtol=1e-6,
    )
    assert session.current_experiment["extra_metadata"]["synthetic_reference"]["T2_ms"] == 80
    sources = pipeline_sources(StudyWorkspace("tutorial", session.study_name, session))
    assert all(row["format"] == "Synthetic" and row["path"] is None for row in sources.values())


def test_starting_tutorial_retains_existing_studies_and_never_saves_files(tmp_path):
    with TestClient(create_app(runtime_dir=tmp_path), base_url="http://localhost") as client:
        client.headers["x-mrax-token"] = client.get("/api/session").json()["token"]
        previous = client.post("/api/workspaces/synthetic").json()
        record = client.app.state.records[previous["id"]]
        before = record.workspace.session.image_array.copy()
        response = client.post("/api/workspaces/synthetic?kind=tutorial")
        assert response.status_code == 200, response.text
        practice = response.json()
        assert practice["can_reconstruct"] and practice["has_kspace"]
        assert not practice["has_reconstruction"] and practice["maps"] == []
        assert practice["pipeline"]["steps"] == []
        assert practice["label"] == "Tutorial — synthetic T2"
        assert len(client.get("/api/workspaces").json()) == 2
        assert client.get(f"/api/workspaces/{previous['id']}").json() == previous
        np.testing.assert_array_equal(record.workspace.session.image_array, before)
        assert not list(tmp_path.rglob("*.mrax"))
        assert not list(tmp_path.rglob("*.mpip"))
