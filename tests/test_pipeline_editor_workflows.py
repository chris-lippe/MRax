"""Atomic chain editing and truthful progress on independently generated images."""

import copy

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter, median_filter

from mrax.gui.services.pipeline import change_pipeline, pipeline_info
from tests.test_web_pipeline_workflows import process, workspace


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
