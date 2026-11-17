"""Synthetic browser contracts and numerical integration."""

from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pytest
from fastapi.testclient import TestClient

from mrax.core.loss_builder import make_model_callable
from mrax.data_struct import Experiment, Quantity
from mrax.gui.models import make_model
from mrax.gui.services.synthetic import synthetic_t2_session
from mrax.gui.session import GuiSession
from mrax.web.app import create_app


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(runtime_dir=tmp_path), base_url="http://localhost") as connection:
        connection.headers["x-mrax-token"] = connection.get("/api/session").json()["token"]
        yield connection


def new_workspace(client):
    response = client.post("/api/workspaces/synthetic")
    assert response.status_code == 200, response.text
    return response.json()


def add_roi(client, workspace, **overrides):
    command = {
        "name": "ROI 1",
        "bounds": [4, 3, 8, 6],
        "revision": workspace["revision"],
        "view_revision": workspace["view_revision"],
    }
    command.update(overrides)
    return client.post(f"/api/workspaces/{workspace['id']}/rois", json=command)


def test_native_roi_and_binary_slice_are_numerically_correct(client):
    workspace = new_workspace(client)
    previous = client.app.state.records[workspace["id"]].workspace
    assert workspace["experiment_catalog"][0]["exp_type"] == "t2_se"
    response = client.get(f"/api/workspaces/{workspace['id']}/slice")
    image = np.frombuffer(response.content, dtype="<f4").reshape((24, 32))
    reference = synthetic_t2_session().frame_image()
    np.testing.assert_array_equal(image, reference)
    assert response.headers["x-dtype"] == "float32-le"
    assert response.headers["x-view-revision"] == "0"
    response = add_roi(client, workspace)
    assert response.status_code == 200
    loaded = response.json()
    expected = np.zeros((24, 32), dtype=int)
    expected[3:9, 4:12] = 1
    np.testing.assert_array_equal(loaded["rois"][0]["mask"], expected)
    assert loaded["rois"][0]["pixels"] == 48
    current = client.app.state.records[workspace["id"]].workspace
    assert current.session.data is previous.session.data
    assert current.session.study is previous.session.study
    assert previous.session.shared_rois == []
    assert current.session.shared_rois is not previous.session.shared_rois
    assert loaded["timecourse"]["x_unit"] == "ms"
    observed = loaded["timecourse"]["observed"]
    assert observed[4] / observed[0] == pytest.approx(np.exp(-70 / 80), rel=1e-6)


def test_multipool_edits_preserve_parameters_predictions_and_saved_settings(client):
    workspace = new_workspace(client)
    path = "/api/workspaces/" + workspace["id"]
    original = np.asarray(
        client.app.state.records[workspace["id"]].workspace.session.image_array
    ).copy()

    def edit(action_name, **options):
        nonlocal workspace
        response = client.post(
            path + "/actions",
            json={
                "revision": workspace["revision"],
                "action": action_name,
                "options": options,
            },
        )
        assert response.status_code == 200, response.text
        workspace = response.json()

    def prediction(settings):
        model = make_model(
            settings["model_name"],
            pools=settings["selected_pools"],
            pool_overrides=settings["pool_overrides"],
        )
        parameters, unpack = model.parametrization_factory((1, 1))
        experiment = Experiment(
            dims={"frame": 9, "x": 1, "y": 1},
            exp_type="sat_trans",
            observables={
                "freq_offsets": Quantity(np.linspace(-4, 4, 9), "ppm"),
                "sat_time": Quantity(1000.0, "ms"),
                "sat_b1": Quantity(1.0, "uT"),
                "TR": Quantity(2000.0, "ms"),
                "t1_map": Quantity(np.full((1, 1), 1000.0), "ms"),
                "b0_map": Quantity(np.zeros((1, 1)), "ppm"),
                "b1_map": Quantity(np.zeros((1, 1)), "%"),
            },
        )
        predict = make_model_callable(
            experiment,
            model,
            study_globals={"field_strength_mhz": Quantity(100.0, "MHz")},
        )
        values = predict(unpack(parameters))
        assert np.isfinite(values).all()
        return np.asarray(values)

    for model in ("multipool_lorentzian", "ab_mt_n", "zaiss_full"):
        edit(
            "fit.settings",
            model_name=model,
            selected_pools=[],
            param_overrides={},
            pool_overrides={},
            regularizer_configs={},
            parameter_constant_dims={},
        )
        edit("fit.pool", operation="add", name="glu", template="glu")
        edit("fit.pool", operation="add", name="mobile_glu", template="glu")
        specs = client.get(path + "/parameters").json()
        if model == "multipool_lorentzian":
            assert specs["dw_water"]["trainable"] is False
            assert specs["amp_mobile_glu"]["trainable"] is True
        elif model == "zaiss_full":
            assert specs["r2_mobile_glu"]["trainable"] is False
        pools = workspace["fit_settings"]["pool_overrides"]
        parameter = "amp" if model == "multipool_lorentzian" else "f"
        pools["mobile_glu"][parameter]["init"] *= 1.05
        edit(
            "fit.settings",
            pool_overrides=pools,
            regularizer_configs={
                parameter + "_mobile_glu": [
                    {
                        "type": "smoothness",
                        "weight": 0.03,
                        "axes": ["x"],
                    }
                ]
            },
            parameter_constant_dims={parameter + "_mobile_glu": ["y"]},
        )
        before = prediction(workspace["fit_settings"])
        edit("fit.pool", operation="rename", name="mobile_glu", new_name="amide")
        settings = workspace["fit_settings"]
        assert "mobile_glu" not in settings["pool_overrides"]
        assert settings["pool_overrides"]["amide"] == pools["mobile_glu"]
        assert parameter + "_amide" in settings["regularizer_configs"]
        assert settings["parameter_constant_dims"] == {parameter + "_amide": ["y"]}
        np.testing.assert_allclose(prediction(settings), before, rtol=1e-6)
        specs = client.get(path + "/parameters").json()
        assert specs[parameter + "_amide"]["initial"] == pytest.approx(
            pools["mobile_glu"][parameter]["init"]
        )
        saved = client.post(path + "/save")
        restored = client.post("/api/workspaces/upload", content=saved.content)
        assert restored.status_code == 200, restored.text
        assert restored.json()["fit_settings"]["pool_overrides"] == settings["pool_overrides"]
        captured = client.post(
            path + "/workflows",
            json={
                "revision": workspace["revision"],
                "kind": "fitting",
                "name": "Pool fit",
                "draft": {"use_roi": False},
            },
        )
        assert captured.status_code == 200, captured.text
        assert '"amide"' in json.dumps(captured.json())
        edit("fit.pool", operation="remove", name="amide")
        assert "amide" not in workspace["fit_settings"]["pool_overrides"]
        assert parameter + "_amide" not in workspace["fit_settings"]["regularizer_configs"]
        edit("fit.pool", operation="remove", name="glu")

    # Proposed model and pool edits leave the applied workspace unchanged.
    unchanged = client.get(path).json()
    proposed = {**workspace["fit_settings"], "model_name": "multipool_lorentzian", "num_steps": 123}
    response = client.post(path + "/fit-options", json={
        "revision": workspace["revision"], "settings": proposed,
        "pool_edit": {"operation": "add", "name": "test_pool", "template": "glu"},
    })
    assert response.status_code == 200, response.text
    assert response.json()["choices"]["model"] == "multipool_lorentzian"
    assert "amp_test_pool" in response.json()["parameters"]
    assert response.json()["settings"]["num_steps"] == 123
    assert client.get(path).json() == unchanged
    captured = client.post(path + "/workflows", json={
        "revision": workspace["revision"], "kind": "fitting",
    }).json()
    resolved = client.post(path + "/pipeline-editor", json={
        "revision": workspace["revision"], "kind": "fit",
        "options": {"workflow": captured["document"], "bindings": {}},
    })
    assert resolved.status_code == 200, resolved.text
    assert resolved.json()["options"]["settings"]["model_name"] == workspace["model"]
    assert resolved.json()["options"]["use_roi"] is False
    assert client.get(path).json() == unchanged

    revision = workspace["revision"]
    for options in (
        {"operation": "add", "name": "bad name", "template": "glu"},
        {"operation": "rename", "name": "mt", "new_name": "other"},
        {"operation": "remove", "name": "mt"},
    ):
        response = client.post(
            path + "/actions",
            json={
                "revision": revision,
                "action": "fit.pool",
                "options": options,
            },
        )
        assert response.status_code == 422, response.text
        assert client.get(path).json()["revision"] == revision
    session = client.app.state.records[workspace["id"]].workspace.session
    np.testing.assert_array_equal(session.image_array, original)
    np.testing.assert_array_equal(session.original_image_array, original)

    pooled = GuiSession()
    pooled.create_empty_study("Pool selection", field_strength_mhz=400)
    pooled.add_experiment("Reference", np.ones((2, 3)), dims={"x": 2, "y": 3},
                          exp_type="sat_trans", observables={"sat_b1": Quantity(0, "uT")})
    pooled.add_experiment("Spectra", np.ones((3, 2, 3)), dims={"offset": 3, "x": 2, "y": 3},
                          exp_type="sat_trans", model_name="multipool_lorentzian",
                          observables={"freq_offsets": Quantity(np.asarray([-1, 0, 1]), "ppm"),
                                       "sat_b1": Quantity(2, "uT")})
    pooled.fit_settings.selected_pools = ("water",)
    assert set(pooled.build_fit_inputs()[1].parameter_units) == {
        "baseline", "amp_water", "gamma_water", "dw_water",
    }
    assert "amp_mt" not in pooled.run_preview_fit(use_roi=False, num_steps=1)["fields"]


def test_origins_tokens_hosts_and_archives_are_checked(client):
    assert (
        client.post("/api/workspaces/synthetic", headers={"x-mrax-token": "wrong"}).status_code
        == 403
    )
    assert (
        client.post(
            "/api/workspaces/synthetic", headers={"origin": "https://example.invalid"}
        ).status_code
        == 403
    )
    assert client.get("/api/health", headers={"host": "example.invalid"}).status_code == 400
    assert client.post("/api/workspaces/upload", content=b"invalid").status_code == 422
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as target:
        target.writestr("../outside.txt", "untrusted")
    assert client.post("/api/workspaces/upload", content=archive.getvalue()).status_code == 422
    assert client.get("/api/workspaces").json() == []
    assert client.get("/api/workspaces/missing").status_code == 404
    assert client.get("/api/jobs/missing").status_code == 404


def test_edited_observables_rebuild_the_study_and_survive_save_restore(client, monkeypatch):
    workspace = new_workspace(client)
    key = workspace["id"]
    original = client.get(f"/api/workspaces/{key}/inspect").json()["metadata"]
    options = json.loads(json.dumps(original))
    options["observables"]["TE"]["values"][0] = 12.5
    options["metadata"]["note"] = "Edited synthetic coordinate"
    image = client.get(f"/api/workspaces/{key}/slice").content
    response = client.post(
        f"/api/workspaces/{key}/actions",
        json={"revision": workspace["revision"], "action": "metadata.update", "options": options},
    )
    assert response.status_code == 200, response.text
    updated = response.json()
    inspected = client.get(f"/api/workspaces/{key}/inspect").json()["metadata"]
    assert inspected["observables"]["TE"]["values"][0] == pytest.approx(12.5)
    assert inspected["observables"]["TE"]["unit"] == original["observables"]["TE"]["unit"]
    assert (
        inspected["observables"]["TE"]["values"][1:] == original["observables"]["TE"]["values"][1:]
    )
    assert client.get(f"/api/workspaces/{key}/slice").content == image

    invalid = json.loads(json.dumps(options))
    invalid["observables"]["TE"]["values"] = [1, 2]
    rejected = client.post(
        f"/api/workspaces/{key}/actions",
        json={"revision": updated["revision"], "action": "metadata.update", "options": invalid},
    )
    assert rejected.status_code == 422
    assert client.get(f"/api/workspaces/{key}/inspect").json()["metadata"] == inspected
    assert client.get(f"/api/workspaces/{key}").json()["revision"] == updated["revision"]

    summary = client.get(f"/api/workspaces/{key}/inspect").json()["summary"]
    invalid["observables"] = options["observables"]
    invalid["exp_type"] = "dwi"
    fit_study = GuiSession.fit_study

    def reject_candidate(session):
        if session.study is None:
            raise ValueError("Invalid synthetic experiment metadata")
        return fit_study(session)

    with monkeypatch.context() as patch:
        patch.setattr(GuiSession, "fit_study", reject_candidate)
        rejected = client.post(
            f"/api/workspaces/{key}/actions",
            json={"revision": updated["revision"], "action": "metadata.update", "options": invalid},
        )
    assert rejected.status_code == 422
    after = client.get(f"/api/workspaces/{key}/inspect").json()
    assert after["metadata"] == inspected
    assert after["summary"] == summary

    saved = client.post(f"/api/workspaces/{key}/save")
    assert saved.status_code == 200
    restored = client.post("/api/workspaces/upload", content=saved.content)
    assert restored.status_code == 200, restored.text
    restored_metadata = client.get(f"/api/workspaces/{restored.json()['id']}/inspect").json()[
        "metadata"
    ]
    assert restored_metadata["observables"] == inspected["observables"]
    assert restored_metadata["metadata"]["note"] == "Edited synthetic coordinate"
