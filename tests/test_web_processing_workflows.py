"""Toolbox processing, shared masks, quantitative outputs and persistent workflows."""

import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from mrax.web.app import create_app


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(runtime_dir=tmp_path), base_url="http://localhost") as browser:
        browser.headers["x-mrax-token"] = browser.get("/api/session").json()["token"]
        yield browser


def create(client):
    response = client.post("/api/workspaces/synthetic")
    assert response.status_code == 200, response.text
    return response.json()


def latest(client, w):
    return client.get(f"/api/workspaces/{w['id']}").json()


def action(client, w, operation, **options):
    response = client.post(
        f"/api/workspaces/{w['id']}/actions",
        json={
            "revision": w["revision"],
            "view_revision": w["view_revision"],
            "action": operation,
            "options": options,
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def view(client, w, **options):
    response = client.patch(
        f"/api/workspaces/{w['id']}/view", json={"revision": w["revision"], **options}
    )
    assert response.status_code == 200, response.text
    return response.json()


def finish(client, w, job):
    deadline = time.monotonic() + 80
    while time.monotonic() < deadline:
        state = client.get(f"/api/jobs/{job['id']}").json()
        if state["status"] not in {"queued", "running", "cancelling"}:
            assert state["status"] == "completed", state["detail"]
            return latest(client, w), state
        time.sleep(0.05)
    pytest.fail("Synthetic processing exceeded its bounded timeout")


def job(client, w, operation, **options):
    response = client.post(
        f"/api/workspaces/{w['id']}/tasks",
        json={"revision": w["revision"], "operation": operation, "options": options},
    )
    assert response.status_code == 200, response.text
    return finish(client, w, response.json())


def test_denoise_fit_statistics_and_save_form_one_workflow(client):
    w = create(client)
    workspace = client.app.state.records[w["id"]].workspace
    truth = np.asarray(workspace.session.image_array).copy()
    source = workspace.session.build_experiment()
    geometry = workspace.session.build_dataset().spatial_geometry
    noisy = truth + np.random.default_rng(12).normal(0, 20000, truth.shape)
    workspace.session.add_experiment(
        "Noisy",
        noisy,
        dims=source.dims,
        observables=source.observables,
        model_name=workspace.session.current_model_name,
        extra_metadata={"spatial_geometry": geometry.to_metadata()},
    )
    w = latest(client, w)
    w = action(client, w, "roi.create", name="Tissue", tool="rect", bounds=[7, 5, 12, 10], label=7)
    w = action(client, w, "frames.exclude", indices=[7])
    w = view(client, w, scene={"hidden_rois": ["Tissue"], "fit_rois": ["Tissue"]})
    w, result = job(
        client,
        w,
        "process",
        method="gaussian",
        input="image",
        axes=["x", "y"],
        parameters={"sigma": 1.0},
        name="Denoised",
    )
    assert result["summary"]["experiment"] == "Noisy" and w["experiment"] == "Noisy"
    assert w["experiments"] == ["T2 phantom", "Noisy"]
    workspace = client.app.state.records[w["id"]].workspace
    denoised = np.asarray(workspace.session.image_array)
    assert np.mean((denoised - truth) ** 2) < 0.3 * np.mean((noisy - truth) ** 2)
    np.testing.assert_array_equal(workspace.session.original_image_array, noisy.astype(np.float32))
    assert w["rois"][0]["enabled"] and not w["rois"][0]["visible"]
    assert w["rois"][0]["labels"] == [7] and w["frame_exclusions"] == [7]
    assert w["scene"]["fit_rois"] == ["Tissue"]
    assert w["scene"]["layers"] == []
    w, _ = job(
        client, w, "full_fit", roi_names=["Tissue"], use_roi=True, settings={"num_steps": 600}
    )
    statistics = client.get(f"/api/workspaces/{w['id']}/roi-statistics", params={"map_name": "t2"})
    assert statistics.status_code == 200, statistics.text
    rows = {row["stat"]: row for row in statistics.json()}
    assert rows["median"]["value"] == pytest.approx(80, abs=2)
    assert rows["count"]["value"] == 120 and rows["mean"]["unit"] == "ms"
    w, _ = job(
        client,
        w,
        "process",
        method="baseline",
        input="image",
        axis="frame",
        parameters={"baseline_frames": 2, "mode": "divide"},
        name="Normalized",
    )
    assert w["model"] == "identity" and w["unit"] == "1"
    catalog = client.get(f"/api/workspaces/{w['id']}/processing").json()
    assert [row["method"] for row in catalog["history"]] == ["gaussian", "baseline"]
    bundle = client.post(f"/api/workspaces/{w['id']}/save")
    restored = client.post("/api/workspaces/upload", content=bundle.content)
    assert restored.status_code == 200, restored.text
    r = restored.json()
    assert (
        client.get(f"/api/workspaces/{r['id']}/processing").json()["history"] == catalog["history"]
    )
    assert r["scene"]["fit_rois"] == ["Tissue"] and not r["rois"][0]["visible"]
    r = view(client, r, experiment="Noisy", view="Map", map_name="t2")
    assert r["fit_info"]["roi_names"] == ["Tissue"]
