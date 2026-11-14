"""Synthetic browser contracts and numerical integration."""

from __future__ import annotations

import io
import json
import time
import zipfile
from dataclasses import replace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from mrax.core.loss_builder import make_model_callable
from mrax.data_struct import Experiment, Quantity
from mrax.gui.models import make_model
from mrax.gui.services.synthetic import synthetic_t2_session
from mrax.gui.session import GuiSession
from mrax.web.app import create_app
from mrax.web.job_runtime import _workflow_worker


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


def wait_job(client, key):
    deadline = time.monotonic() + 40
    while time.monotonic() < deadline:
        response = client.get(f"/api/jobs/{key}")
        assert response.status_code == 200
        job = response.json()
        if job["status"] not in {"queued", "running", "cancelling"}:
            return job
        time.sleep(0.05)
    pytest.fail("The synthetic fit worker did not finish")


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


def test_revision_checks_prevent_stale_drawing_and_cross_workspace_edits(client):
    first, second = new_workspace(client), new_workspace(client)
    path = f"/api/workspaces/{first['id']}/view"
    assert client.patch(path, json={"revision": 0, "frame": 1, "z": 1}).status_code == 200
    assert add_roi(client, first).status_code == 409
    current = client.get(f"/api/workspaces/{first['id']}").json()
    assert add_roi(client, current).status_code == 200
    assert add_roi(client, current, name="Other ROI").status_code == 409
    assert client.get(f"/api/workspaces/{second['id']}").json()["rois"] == []
    assert client.patch(path, json={"revision": 1, "frame": 99, "z": 0}).status_code == 422



def test_fitting_inputs_follow_the_model_and_reference_normalization_preserves_sources(client):
    workspace = new_workspace(client)
    key = workspace["id"]
    session = client.app.state.records[key].workspace.session
    options = client.get(f"/api/workspaces/{key}/fit-options").json()
    assert options["inputs"] == {
        "reference_scan": False, "prior_maps": [], "derived_maps": False,
    }
    assert options["reference_scans"] == [] and options["prior_maps"] == {}

    reference = np.full((1, 2, 3), 100, dtype=np.float32)
    signal = np.linspace(40, 90, 5, dtype=np.float32)[:, None, None] * np.ones((1, 2, 3))
    session.add_experiment(
        "Reference", reference, dims={"frame": 1, "x": 2, "y": 3},
        exp_type="sat_trans", model_name="wasabi",
        observables={"freq_offsets": Quantity([100], "ppm")},
    )
    session.add_experiment(
        "Z spectrum", signal, dims={"frame": 5, "x": 2, "y": 3},
        exp_type="sat_trans", model_name="wasabi",
        observables={
            "freq_offsets": Quantity(np.linspace(-2, 2, 5), "ppm"),
            "sat_time": Quantity(2, "s"), "sat_b1": Quantity(1, "uT"),
        },
    )
    for model, priors, derived in (
        ("wasabi", [], True),
        ("multipool_lorentzian", [], False),
        ("ab_mt_n", ["t1", "b1", "b0"], False),
        ("zaiss_full", ["t1", "b1", "b0"], False),
        ("identity", [], False),
    ):
        session.fit_settings.model_name = model
        options = client.get(f"/api/workspaces/{key}/fit-options").json()
        assert options["model"] == model
        assert options["inputs"] == {
            "reference_scan": model != "identity", "prior_maps": priors, "derived_maps": derived,
        }
        assert list(options["prior_maps"]) == priors
        assert options["reference_scans"] == ([] if model == "identity" else ["Reference"])

    session.fit_settings.model_name = "wasabi"
    session.fit_settings.sat_trans_reference_scan = "Reference"
    _experiment, _model, dataset, _study = session.build_fit_inputs()
    np.testing.assert_allclose(np.asarray(dataset.data), signal / reference, rtol=1e-6)
    np.testing.assert_array_equal(session.image_array, signal)
    np.testing.assert_array_equal(session.original_image_array, signal)
    study = session.fit_study()
    np.testing.assert_array_equal(study.datasets[study.names.index("Reference")].data, reference)

    session.fit_settings.model_name = "identity"
    session.fit_settings.sat_trans_reference_scan = "Unused selection"
    _experiment, _model, dataset, _study = session.build_fit_inputs()
    np.testing.assert_array_equal(np.asarray(dataset.data), signal)


def test_multipool_edits_preserve_parameters_predictions_and_saved_settings(client):
    workspace = new_workspace(client)
    path = "/api/workspaces/" + workspace["id"]
    original = np.asarray(client.app.state.records[workspace["id"]].workspace.session.image_array).copy()

    def edit(action_name, **options):
        nonlocal workspace
        response = client.post(path + "/actions", json={
            "revision": workspace["revision"], "action": action_name, "options": options,
        })
        assert response.status_code == 200, response.text
        workspace = response.json()

    def prediction(settings):
        model = make_model(
            settings["model_name"], pools=settings["selected_pools"],
            pool_overrides=settings["pool_overrides"],
        )
        parameters, unpack = model.parametrization_factory((1, 1))
        experiment = Experiment(
            dims={"frame": 9, "x": 1, "y": 1}, exp_type="sat_trans",
            observables={
                "freq_offsets": Quantity(np.linspace(-4, 4, 9), "ppm"),
                "sat_time": Quantity(1000.0, "ms"), "sat_b1": Quantity(1.0, "uT"),
                "TR": Quantity(2000.0, "ms"), "t1_map": Quantity(np.full((1, 1), 1000.0), "ms"),
                "b0_map": Quantity(np.zeros((1, 1)), "ppm"),
                "b1_map": Quantity(np.zeros((1, 1)), "%"),
            },
        )
        predict = make_model_callable(
            experiment, model, study_globals={"field_strength_mhz": Quantity(100.0, "MHz")},
        )
        values = predict(unpack(parameters))
        assert np.isfinite(values).all()
        return np.asarray(values)

    for model in ("multipool_lorentzian", "ab_mt_n", "zaiss_full"):
        edit("fit.settings", model_name=model, selected_pools=[], param_overrides={},
             pool_overrides={}, regularizer_configs={}, parameter_constant_dims={})
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
        edit("fit.settings", pool_overrides=pools,
             regularizer_configs={parameter + "_mobile_glu": [{
                 "type": "smoothness", "weight": 0.03, "axes": ["x"],
             }]},
             parameter_constant_dims={parameter + "_mobile_glu": ["y"]})
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
        captured = client.post(path + "/workflows", json={
            "revision": workspace["revision"], "kind": "fitting",
            "name": "Pool fit", "draft": {"use_roi": False},
        })
        assert captured.status_code == 200, captured.text
        assert '"amide"' in json.dumps(captured.json())
        edit("fit.pool", operation="remove", name="amide")
        assert "amide" not in workspace["fit_settings"]["pool_overrides"]
        assert parameter + "_amide" not in workspace["fit_settings"]["regularizer_configs"]
        edit("fit.pool", operation="remove", name="glu")

    revision = workspace["revision"]
    for options in (
        {"operation": "add", "name": "bad name", "template": "glu"},
        {"operation": "rename", "name": "mt", "new_name": "other"},
        {"operation": "remove", "name": "mt"},
    ):
        response = client.post(path + "/actions", json={
            "revision": revision, "action": "fit.pool", "options": options,
        })
        assert response.status_code == 422, response.text
        assert client.get(path).json()["revision"] == revision
    session = client.app.state.records[workspace["id"]].workspace.session
    np.testing.assert_array_equal(session.image_array, original)
    np.testing.assert_array_equal(session.original_image_array, original)


def test_fit_readiness_checks_drafts_rois_frames_and_inputs_without_changing_state(client, monkeypatch):
    workspace = new_workspace(client)
    path = "/api/workspaces/" + workspace["id"]
    record = client.app.state.records[workspace["id"]]
    original = record.workspace
    images = np.asarray(original.session.image_array).copy()

    def check(**values):
        response = client.post(path + "/fit-check", json={
            "revision": workspace["revision"], "view_revision": workspace["view_revision"],
            "use_roi": False, **values,
        })
        assert response.status_code == 200, response.text
        return response.json()

    ready = check(settings={"num_steps": 17}, excluded_frames=[0])
    assert ready["ok"] and ready["errors"] == []
    assert "Included frames: 7 of 8." in ready["notes"]
    frames = client.get(path + "/fit-frames?offset=2&limit=2").json()
    assert frames["revision"] == workspace["revision"] and frames["frame_count"] == 8
    assert [frame["index"] for frame in frames["frames"]] == [2, 3]
    assert all("TE:" in frame["label"] and "ms" in frame["label"] for frame in frames["frames"])
    assert client.get(path + "/fit-frames?limit=101").status_code == 422
    assert not check(use_roi=True, roi_names=[])["ok"]
    assert not check(settings={"num_steps": 0})["ok"]
    assert not check(excluded_frames=list(range(8)))["ok"]
    assert not check(settings={"parameter_constant_dims": {"t2": ["unknown_axis"]}})["ok"]
    missing = check(settings={"model_name": "wasabi"})
    assert not missing["ok"] and any("freq_offsets" in error for error in missing["errors"])
    build_inputs = GuiSession.build_fit_inputs
    invalid_te = np.full(8, np.nan)

    def with_invalid_observables(session, model_name=None):
        experiment, model, dataset, study = build_inputs(session, model_name)
        observables = {**experiment.observables, "TE": Quantity(invalid_te, "ms")}
        return replace(experiment, observables=observables), model, dataset, study

    with monkeypatch.context() as patch:
        patch.setattr(GuiSession, "build_fit_inputs", with_invalid_observables)
        assert not check()["ok"]
        invalid_te[1:] = np.arange(1, 8) * 10.0
        masked = check()
        assert masked["ok"] and "Included frames: 7 of 8." in masked["notes"]
        assert masked["warnings"]
        assert not check(settings={"auto_mask_nan_observables": False})["ok"]
    assert record.workspace is original
    assert client.get(path).json() == workspace
    assert original.session.frame_mask is None or original.session.frame_mask.excluded_indices == []
    np.testing.assert_array_equal(original.session.image_array, images)
    np.testing.assert_array_equal(original.session.original_image_array, images)
    workspace = add_roi(client, workspace).json()
    assert check(use_roi=True, roi_names=["ROI 1"])["ok"]
    assert not check(use_roi=True, roi_names=["Unavailable ROI"])["ok"]
    assert client.post(path + "/fit-check", json={"revision": 0, "use_roi": False}).status_code == 409


def test_real_spawned_fit_matches_known_t2_and_round_trips(client):
    workspace = new_workspace(client)
    workspace = add_roi(client, workspace).json()
    path = f"/api/workspaces/{workspace['id']}"
    dimensions = client.get(path + "/fit-options").json()["field_dimensions"]
    assert {dimension["name"] for dimension in dimensions} == {"x", "y", "z"}
    configured = client.post(path + "/actions", json={
        "revision": workspace["revision"], "action": "fit.settings", "options": {
            "parameter_constant_dims": {"t2": ["z"]},
            "regularizer_configs": {"t2": [{"type": "smoothness", "weight": 0.01, "axes": ["y"]}]},
            "excluded_frames": [0],
        },
    })
    assert configured.status_code == 200, configured.text
    workspace = configured.json()
    assert workspace["frame_exclusions"] == [0]
    expected_axis = next(dimension["axis"] for dimension in dimensions if dimension["name"] == "y")
    assert workspace["fit_settings"]["regularizer_configs"]["t2"][0]["axes"] == [expected_axis]
    rejected = client.post(path + "/actions", json={
        "revision": workspace["revision"], "action": "fit.settings", "options": {
            "num_steps": 17, "excluded_frames": list(range(8)),
        },
    })
    assert rejected.status_code == 422 and client.get(path).json() == workspace
    response = client.post(
        path + "/preview-fit",
        json={"revision": workspace["revision"], "roi_name": "ROI 1", "steps": 600},
    )
    assert response.status_code == 200, response.text
    # Serving requests while JAX compiles/fits is part of this contract.
    assert client.get("/api/health").json()["status"] == "ok"
    job = wait_job(client, response.json()["id"])
    assert job["status"] == "completed", job
    assert job["result"]["parameters"]["t2"] == pytest.approx(80, abs=0.05)
    assert job["result"]["relative_rmse"] < 1e-4
    assert job["result"]["x_unit"] == "ms"
    preview = client.get(path).json()["preview"]
    assert preview["roi_name"] == "ROI 1"
    saved = client.post(path + "/save")
    assert saved.status_code == 200
    with zipfile.ZipFile(io.BytesIO(saved.content)) as archive:
        assert "mrax_gui_state.json" in archive.namelist()
        payload = json.loads(archive.read("mrax_gui_state.json"))
        assert payload["format"] == "mrax.gui.state" and payload["version"] == 3
        assert "theme" not in payload
    response = client.post("/api/workspaces/upload", content=saved.content)
    assert response.status_code == 200, response.text
    restored = response.json()
    assert restored["rois"] == workspace["rois"]
    assert restored["spacing"] == workspace["spacing"]
    assert restored["preview"]["parameters"]["t2"] == job["result"]["parameters"]["t2"]
    assert restored["preview"]["roi_name"] == "ROI 1"
    assert restored["id"] != workspace["id"]
    assert restored["label"] == workspace["label"]


def test_cancelled_worker_does_not_start_an_operation(tmp_path):
    def cancelled_progress(*_args):
        raise InterruptedError("Cancelled")

    assert _workflow_worker(None, tmp_path, "preview", {}, cancelled_progress) == {
        "cancelled": True
    }
    assert not list(tmp_path.iterdir())


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
