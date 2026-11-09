"""Atomic chain editing and truthful progress on independently generated images."""

import copy
import threading
import time

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter, median_filter

from mrax.analysis import ROI
from mrax.gui.services import processing
from mrax.gui.services.pipeline import change_pipeline, pipeline_info
from tests import test_web_analysis_workflows as web
from tests.test_web_pipeline_workflows import process, workspace

browser = web.browser


def draft_step(step, *, client_id=None):
    return {
        "id": client_id or step["id"],
        "source_id": step["id"],
        "source_version": step["version"],
        "kind": step["kind"],
        "options": copy.deepcopy(step["options"]),
        "enabled": step["enabled"],
    }


def test_insert_copy_reorder_and_remove_apply_once_from_protected_source():
    w, original = workspace()
    process(w, "gaussian", axes=["x", "y"], parameters={"sigma": 0.6})
    before = pipeline_info(w)
    first = before["steps"][0]
    median = {
        "id": "inserted-median",
        "kind": "process",
        "enabled": True,
        "options": {
            "method": "median",
            "input": "image",
            "axes": ["x", "y"],
            "parameters": {"size": 3},
        },
    }
    proposal = [median, draft_step(first), draft_step(first, client_id="copied-gaussian")]
    events = []

    def progress(*_):
        pass

    progress.pipeline_event = events.append
    change_pipeline(
        w,
        {"action": "replace", "base_generation": before["generation"], "steps": proposal},
        progress,
    )
    result = pipeline_info(w)
    assert result["generation"] == before["generation"] + 1
    assert [s["method"] for s in result["steps"]] == ["median", "gaussian", "gaussian"]
    assert result["steps"][1]["id"] == first["id"]
    assert len({s["id"] for s in result["steps"]}) == 3
    expected = median_filter(original, size=(1, 3, 3), mode="mirror")
    expected = gaussian_filter(
        gaussian_filter(expected, sigma=(0, 0.6, 0.6), mode="mirror"),
        sigma=(0, 0.6, 0.6),
        mode="mirror",
    )
    np.testing.assert_allclose(w.session.image_array, expected, rtol=1e-6)
    np.testing.assert_array_equal(w.session.original_image_array, original)
    assert events[-1]["status"] == "completed"
    assert [s["client_id"] for s in events[-1]["steps"]] == [s["id"] for s in proposal]
    assert all(s["status"] == "completed" for s in events[-1]["steps"])
    change_pipeline(
        w,
        {
            "action": "replace",
            "base_generation": result["generation"],
            "steps": [draft_step(result["steps"][1])],
        },
        lambda *_: None,
    )
    np.testing.assert_allclose(
        w.session.image_array,
        gaussian_filter(original, sigma=(0, 0.6, 0.6), mode="mirror"),
        rtol=1e-6,
    )


@pytest.mark.parametrize("failure", ["generation", "version", "duplicate", "late", "cancel"])
def test_failed_stale_and_cancelled_proposals_leave_pipeline_and_images_untouched(failure):
    w, _ = workspace()
    process(w, "gaussian", axes=["x", "y"], parameters={"sigma": 0.6})
    before = pipeline_info(w)
    images = w.session.image_array.copy()
    rows = [draft_step(before["steps"][0]), draft_step(before["steps"][0], client_id="copy")]
    generation = before["generation"]
    if failure == "generation":
        generation -= 1
    elif failure == "version":
        rows[0]["source_version"] = "outdated"
    elif failure == "duplicate":
        rows[1]["id"] = rows[0]["id"]
    elif failure == "late":
        rows[1]["options"]["parameters"]["sigma"] = -1
    events = []

    def progress(done, _total, _detail):
        if failure == "cancel" and events and events[-1]["steps"][1]["status"] == "running":
            raise InterruptedError("Cancel synthetic replay")

    progress.pipeline_event = events.append
    with pytest.raises((ValueError, InterruptedError)):
        change_pipeline(
            w, {"action": "replace", "base_generation": generation, "steps": rows}, progress
        )
    assert pipeline_info(w) == before
    np.testing.assert_array_equal(w.session.image_array, images)
    if failure in {"late", "cancel"}:
        assert events[-1]["status"] == ("cancelled" if failure == "cancel" else "failed")
        assert events[-1]["steps"][0]["status"] == "completed"
        assert events[-1]["steps"][1]["status"] in {"failed", "cancelled"}


def test_job_reports_progress_for_the_confirmed_chain(browser):
    client, _ = browser
    w = web.new(client)
    w, _ = web.job(
        client, w, "process", method="gaussian", axes=["x", "y"], parameters={"sigma": 0.5}
    )
    first = w["pipeline"]["steps"][0]
    rows = [draft_step(first), draft_step(first, client_id="copied-step")]
    w, result = web.job(
        client,
        w,
        "pipeline",
        action="replace",
        base_generation=w["pipeline"]["generation"],
        steps=rows,
    )
    assert result["pipeline"]["experiment"] == w["experiment"]
    assert result["pipeline"]["status"] == "completed"
    assert [s["status"] for s in result["pipeline"]["steps"]] == ["completed", "completed"]
    assert [s["client_id"] for s in result["pipeline"]["steps"]] == [r["id"] for r in rows]


def test_copied_steps_reuse_frozen_references_and_masks_after_the_sources_change():
    w, original = workspace()
    w.session.current_experiment.setdefault("extra_metadata", {})["image_unit"] = "1"
    w.session.add_experiment(
        "Reference",
        np.full((1, 9, 11), 2.0),
        dims={"frame": 1, "x": 9, "y": 11},
        extra_metadata={"image_unit": "1"},
    )
    w.session.select_experiment("Experiment")
    mask = np.zeros((9, 11), dtype=np.uint16)
    mask[2:7, 3:8] = 1
    w.session.add_roi(ROI("Tissue", mask, dims=("x", "y")))
    process(w, "normalize", reference_experiment="Reference", roi_names=["Tissue"])
    w.session.select_experiment("Reference")
    w.session.replace_image_data(np.full((1, 9, 11), 20.0))
    w.session.select_experiment("Experiment")
    w.session.remove_shared_roi("Tissue")
    before = pipeline_info(w)
    step = before["steps"][0]
    change_pipeline(
        w,
        {
            "action": "replace",
            "base_generation": before["generation"],
            "steps": [draft_step(step), draft_step(step, client_id="copy")],
        },
        lambda *_: None,
    )
    np.testing.assert_allclose(w.session.image_array[:, 3, 4], original[:, 3, 4] / 4)
    assert np.isnan(w.session.image_array[:, 0, 0]).all()


def test_cancel_endpoint_reports_chain_progress_without_applying_partial_results(
    browser, monkeypatch
):
    client, _ = browser
    w = web.new(client)
    w, _ = web.job(
        client, w, "process", method="gaussian", axes=["x", "y"], parameters={"sigma": 0.5}
    )
    before_pixels = web.pixels(client, w).copy()
    first = w["pipeline"]["steps"][0]
    rows = [draft_step(first), draft_step(first, client_id="cancel-copy")]
    entered = threading.Event()
    release = threading.Event()
    original = processing.apply_processing
    calls = 0

    def controlled(workspace, options, progress):
        nonlocal calls
        calls += 1
        if calls == 2:
            entered.set()
            deadline = time.monotonic() + 5
            while not release.wait(0.01) and time.monotonic() < deadline:
                progress(0, 1, "Synthetic cancellation checkpoint")
        return original(workspace, options, progress)

    monkeypatch.setattr(processing, "apply_processing", controlled)
    response = client.post(
        f"/api/workspaces/{w['id']}/tasks",
        json={
            "revision": w["revision"],
            "operation": "pipeline",
            "options": {
                "action": "replace",
                "base_generation": w["pipeline"]["generation"],
                "steps": rows,
            },
        },
    )
    assert response.status_code == 200
    key = response.json()["id"]
    try:
        assert entered.wait(5)
        running = client.get(f"/api/jobs/{key}").json()
        assert [s["status"] for s in running["pipeline"]["steps"]] == [
            "completed",
            "running",
        ]
        cancel = client.post(f"/api/jobs/{key}/cancel")
        assert cancel.status_code == 200
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            result = client.get(f"/api/jobs/{key}").json()
            if result["status"] not in {"queued", "running", "cancelling"}:
                break
            time.sleep(0.01)
        assert result["status"] == "cancelled"
        assert result["pipeline"]["status"] == "cancelled"
        assert [s["status"] for s in result["pipeline"]["steps"]] == [
            "completed",
            "cancelled",
        ]
        after = client.get(f"/api/workspaces/{w['id']}").json()
        assert after["revision"] == w["revision"]
        assert after["pipeline"] == w["pipeline"]
        np.testing.assert_array_equal(web.pixels(client, after), before_pixels)
    finally:
        release.set()
