"""Performance invariants and data safety on independently synthetic studies."""

import shutil
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from _synthetic import _write_minimal_bruker_scan
from fastapi.testclient import TestClient

from mrax.gui.services.cloning import clone_workspace
from mrax.gui.services.synthetic import synthetic_t2_session
from mrax.gui.services.workspace import StudyWorkspace
from mrax.io.bruker_import import import_bruker_study
from mrax.web.app import create_app
from mrax.web.job_runtime import Jobs


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(runtime_dir=tmp_path), base_url="http://localhost") as browser:
        browser.headers["x-mrax-token"] = browser.get("/api/session").json()["token"]
        yield browser


def new(client):
    response = client.post("/api/workspaces/synthetic")
    assert response.status_code == 200, response.text
    return response.json()


def finish(client, response):
    assert response.status_code == 200, response.text
    for _ in range(600):
        job = client.get("/api/jobs/" + response.json()["id"]).json()
        if job["status"] not in {"queued", "running", "cancelling"}:
            return job
        time.sleep(0.02)
    pytest.fail("Operation did not finish")


def test_processing_never_serializes_a_workspace_and_releases_finished_futures(client, monkeypatch):
    workspace = new(client)
    original = client.app.state.records[workspace["id"]].workspace
    before = np.asarray(original.session.image_array).copy()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Routine operations must not save or compress a workspace")

    monkeypatch.setattr("mrax.web.app.save_workspace_bundle", forbidden)
    monkeypatch.setattr("mrax.gui.services.workspace_io.save_workspace_bundle", forbidden)
    job = finish(
        client,
        client.post(
            "/api/workspaces/" + workspace["id"] + "/tasks",
            json={
                "revision": workspace["revision"],
                "operation": "process",
                "options": {"method": "gaussian", "parameters": {"sigma": 1.0}, "name": "Filtered"},
            },
        ),
    )
    assert job["status"] == "completed", job
    assert not list(client.app.state.root.rglob("*.mrax"))
    assert all("future" not in row for row in client.app.state.jobs.items.values())
    np.testing.assert_array_equal(original.session.image_array, before)


def test_worker_inputs_share_read_only_buffers_but_own_their_metadata():
    workspace = StudyWorkspace(key="test", label="Synthetic", session=synthetic_t2_session())
    workspace.session.current_experiment["original_mr_image"] = np.asarray(
        workspace.session.original_image_array
    ).copy()
    copied = clone_workspace(workspace, readonly_arrays=True)
    original = workspace.session.current_experiment["original_mr_image"]
    duplicate = copied.session.current_experiment["original_mr_image"]
    assert np.shares_memory(original, duplicate)
    assert not duplicate.flags.writeable
    copied.session.current_experiment.setdefault("extra_metadata", {})["note"] = "new"
    assert "note" not in workspace.session.current_experiment.get("extra_metadata", {})
    with pytest.raises(ValueError):
        duplicate.flat[0] = 0


def test_cancelled_operation_never_attaches_partial_results(tmp_path, monkeypatch):
    started, release = threading.Event(), threading.Event()
    attached = []
    workspace = StudyWorkspace(key="test", label="Synthetic", session=synthetic_t2_session())
    snapshot = clone_workspace(workspace, readonly_arrays=True)

    def operation(candidate, _operation, _options, progress):
        candidate.label = "Partial"
        started.set()
        assert release.wait(3)
        progress(1, 1, "Ready")
        return candidate, {}

    monkeypatch.setattr("mrax.gui.services.analysis_workflows.run_operation", operation)
    jobs = Jobs(tmp_path, lambda *args: attached.append(args) or True)
    try:
        job = jobs.submit("test", 0, snapshot, operation="process")
        assert started.wait(3)
        jobs.cancel(job["id"])
        release.set()
        for _ in range(100):
            result = jobs.get(job["id"])
            if result["status"] == "cancelled":
                break
            time.sleep(0.01)
        assert result["status"] == "cancelled"
        assert not attached
        assert workspace.label == "Synthetic"
    finally:
        release.set()
        jobs.close()


def test_navigation_reuses_catalog_and_rendered_plane_without_cloning_study(client, monkeypatch):
    workspace = new(client)
    key = workspace["id"]
    client.get("/api/workspaces/" + key)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Index navigation must not clone the study or rebuild its catalog")

    monkeypatch.setattr("mrax.web.app.clone_workspace", forbidden)
    monkeypatch.setattr("mrax.web.info.experiment_catalog", forbidden)
    from mrax.web import info

    render = info.render_image
    calls = []

    def counted(value):
        calls.append(True)
        return render(value)

    monkeypatch.setattr(info, "render_image", counted)
    for index in (1, 2, 3):
        changed = client.patch(
            "/api/workspaces/" + key + "/view",
            json={
                "revision": workspace["revision"],
                "indices": {"frame": index},
            },
        )
        assert changed.status_code == 200, changed.text
        assert changed.json()["frame"] == index
        image = client.get("/api/workspaces/" + key + "/slice")
        np.testing.assert_array_equal(
            np.frombuffer(image.content, dtype="<f4").reshape((24, 32)),
            client.app.state.records[key].workspace.session.frame_image(),
        )
    assert len(calls) == 3


def test_invalid_navigation_does_not_change_the_current_view(client):
    workspace = new(client)
    response = client.patch(
        "/api/workspaces/" + workspace["id"] + "/view",
        json={
            "revision": workspace["revision"],
            "indices": {"frame": 3, "z": 9999},
        },
    )
    assert response.status_code == 422
    actual = client.get("/api/workspaces/" + workspace["id"]).json()
    assert actual["indices"] == workspace["indices"]
    assert actual["view_revision"] == workspace["view_revision"]


def test_image_and_map_navigation_transfer_only_the_selected_plane():
    class DeviceVolume:
        def __init__(self, values):
            self.values_array = values
            self.shape, self.ndim = values.shape, values.ndim
            self.reads = []

        def __array__(self, *_args, **_kwargs):
            raise AssertionError("The full device volume must not be transferred for display")

        def __getitem__(self, key):
            self.reads.append(key)
            return self.values_array[key]

    session = synthetic_t2_session()
    image = DeviceVolume(np.asarray(session.image_array))
    session.current_experiment["mr_image"] = image
    assert session.dimension_controls("image")
    assert session.frame_image().shape == (24, 32)
    assert len(image.reads) == 1
    mapped = DeviceVolume(np.ones((24, 32, 3), dtype=np.float32))
    session.current_experiment["fitted_maps"] = {"test": mapped}
    session.active_map = "test"
    session._slice_cache.clear()
    assert session.dimension_controls("map")
    assert session.map_image().shape == (24, 32)
    assert len(mapped.reads) == 1
    session.current_experiment["processed"] = {
        "raw_reconstruction_coils": DeviceVolume(np.zeros((8, 2, 24, 32, 3)))
    }
    assert session.has_reconstruction
    assert session.reconstruction_coil_count == 2


def test_numpy_archive_preview_reads_headers_without_loading_arrays(client, tmp_path, monkeypatch):
    path = tmp_path / "large-array.npz"
    np.savez_compressed(path, image=np.zeros((256, 1024), dtype=np.float32))

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Preview must not decompress an array")

    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", forbidden)
    response = client.post(
        "/api/files/preview", json={"root": "home", "path": str(path), "options": {"kind": "auto"}}
    )
    assert response.status_code == 200, response.text
    assert response.json()["arrays"] == [{"key": "image", "shape": [256, 1024], "dtype": "float32"}]


def test_bruker_review_and_selection_skip_unselected_experiment_data(client, tmp_path, monkeypatch):
    root = tmp_path / "Synthetic acquisition"
    _write_minimal_bruker_scan(root)
    shutil.copytree(root / "1", root / "2")
    shutil.copytree(root / "2" / "pdata" / "1", root / "2" / "pdata" / "2")
    (np.arange(4, dtype=np.int16) + 100).tofile(root / "2" / "pdata" / "2" / "2dseq")
    from mrax.io import bruker_import

    load = bruker_import.import_bruker_experiment
    calls = []

    def selected_only(path, *args, **kwargs):
        assert Path(path).name == "2"
        calls.append(Path(path).name)
        return load(path, *args, **kwargs)

    monkeypatch.setattr(bruker_import, "import_bruker_experiment", selected_only)
    preview = client.post(
        "/api/files/preview",
        json={"root": "home", "path": str(root), "options": {"kind": "bruker"}},
    )
    assert preview.status_code == 200, preview.text
    assert not calls
    assert preview.json()["experiments"][1]["reconstructions"] == ["1", "2"]
    assert preview.json()["experiments"][1]["detected_model"] == "t2_se"
    result = finish(
        client,
        client.post(
            "/api/import",
            json={
                "root": "home",
                "path": str(root),
                "options": {
                    "kind": "bruker",
                    "selected_experiments": ["2"],
                    "pdata_index": {"2": "2"},
                    "field_strength_mhz": 400,
                    "load_raw_kspace": False,
                },
            },
        ),
    )
    assert result["status"] == "completed", result
    assert calls == ["2"]
    key = result["workspace_id"]
    workspace = client.get("/api/workspaces/" + key).json()
    assert workspace["experiments"] == ["2"]
    pixels = np.frombuffer(client.get("/api/workspaces/" + key + "/slice").content, dtype="<f4")
    np.testing.assert_array_equal(np.sort(pixels), [210, 212, 214, 216])
    loaded = client.app.state.records[key].workspace.session
    assert np.shares_memory(loaded.image_array, loaded.original_image_array)
    assert not list(client.app.state.root.rglob("*.mrax"))
    with pytest.raises(ValueError, match="at least one"):
        import_bruker_study(root, selected_experiments=[], field_strength_mhz=400)


def test_echo_navigation_reuses_roi_geometry_and_timecourse(client, monkeypatch):
    workspace = new(client)
    key = workspace["id"]
    response = client.post(
        "/api/workspaces/" + key + "/actions",
        json={
            "revision": workspace["revision"],
            "view_revision": workspace["view_revision"],
            "action": "roi.create",
            "options": {"tool": "rect", "name": "Region", "bounds": [2, 3, 8, 9]},
        },
    )
    assert response.status_code == 200, response.text
    workspace = response.json()
    response = client.post(
        "/api/workspaces/" + key + "/actions",
        json={
            "revision": workspace["revision"],
            "action": "roi.select",
            "options": {"name": "Region"},
        },
    )
    assert response.status_code == 200, response.text
    workspace = client.get("/api/workspaces/" + key).json()
    assert workspace["timecourse"]

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Changing an echo must not regenerate a native ROI or its timecourse")

    monkeypatch.setattr("mrax.web.info.curve_for_workspace", forbidden)
    monkeypatch.setattr("mrax.web.info.roi_runs", forbidden)
    for index in (1, 2):
        response = client.patch(
            "/api/workspaces/" + key + "/view",
            json={
                "revision": workspace["revision"],
                "indices": {"frame": index},
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["timecourse"] == workspace["timecourse"]
        assert response.json()["rois"] == workspace["rois"]


@pytest.mark.parametrize("version", [(1, 0), (2, 0), (3, 0)])
def test_array_header_preview_preserves_numpy_layouts(version):
    import io

    from mrax.web.array_headers import read_array_header

    dtype = np.dtype([("signal" if version != (3, 0) else "δ", "<f4")])
    array = np.zeros((3, 4), dtype=dtype)
    stream = io.BytesIO()
    np.lib.format.write_array(stream, array, version=version, allow_pickle=False)
    size = stream.tell()
    stream.seek(0)
    shape, order, actual_dtype = read_array_header(stream)
    assert shape == array.shape
    assert actual_dtype == dtype
    assert order is False
    assert stream.tell() < size
