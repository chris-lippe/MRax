"""Reusable recipes tested only with independently generated numerical data."""

import copy
import json
import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from scipy.ndimage import gaussian_filter

from mrax.analysis import ROI
from mrax.data_struct import Quantity
from mrax.gui.services.analysis_workflows import run_operation
from mrax.gui.services.pipeline import pipeline_info
from mrax.gui.services.reusable_workflows import (
    apply_workflow,
    capture_workflow,
    resource_choices,
    validate_workflow_targets,
)
from mrax.gui.services.synthetic import synthetic_multimodal_session, synthetic_t2_session
from mrax.gui.services.workspace import StudyWorkspace
from mrax.gui.services.workspace_io import load_workspace_bundle, save_workspace_bundle
from mrax.gui.session import GuiSession
from mrax.web.app import create_app
from mrax.workflows import (
    WorkflowDocument,
    dumps_workflow,
    load_workflow,
    loads_workflow,
    new_workflow,
    save_workflow,
)


def phantom(*, second=False):
    session = GuiSession()
    session.create_empty_study("Independent reusable-workflow study")
    image = np.random.default_rng(31).normal(8, 1, (4, 9, 11)).astype(np.float32)
    session.add_experiment("A", image, dims={"frame": 4, "x": 9, "y": 11})
    if second:
        session.add_experiment("B", image * 2, dims={"frame": 4, "x": 9, "y": 11})
        session.select_experiment("A")
    return StudyWorkspace(key="synthetic", label=session.study_name, session=session), image


def processing_document(*steps):
    return new_workflow(
        "processing",
        [
            {"id": "step_" + str(index), "operation": "process", "version": 1, "settings": settings}
            for index, settings in enumerate(
                steps
                or [
                    {
                        "method": "gaussian",
                        "input": "image",
                        "axes": ["x", "y"],
                        "parameters": {"sigma": 1},
                    },
                ]
            )
        ],
    )


def apply(workspace, document, experiments=None, **kwargs):
    return apply_workflow(
        workspace,
        document.model_dump(),
        experiments or [workspace.session.selected_experiment],
        kwargs.pop("bindings", {}),
        kwargs.pop("progress", lambda *_: None),
        **kwargs,
    )


def test_utf8_roundtrip_required_versions_extensions_and_atomic_files(tmp_path):
    document = processing_document()
    values = document.model_dump()
    values["name"] = "Denoising α"
    values["extensions"] = {"org.example.review": {"notes": "Keep for later"}}
    document = WorkflowDocument.model_validate(values)
    path = tmp_path / "Unicode ä α" / "denoise.mpip"
    path.parent.mkdir()
    save_workflow(document, path)
    assert path.read_text().startswith('{\n  "format": "mrax.workflow",')
    assert load_workflow(path) == document
    assert loads_workflow(dumps_workflow(document).encode("utf-8")) == document
    with pytest.raises((ValueError, FileExistsError)):
        save_workflow(document, path)
    assert load_workflow(path) == document
    save_workflow(document, path, overwrite=True)
    with pytest.raises(ValueError, match="extension"):
        save_workflow(document, path.with_suffix(".mfit"))
    with pytest.raises(ValueError, match="extension"):
        loads_workflow(dumps_workflow(document), suffix=".mrec")


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d.pop("format_version"),
        lambda d: d.update(format_version=2),
        lambda d: d.update(unrecognized=True),
        lambda d: d.update(mrax_version=None),
        lambda d: d.update(mrax_version="unknown"),
        lambda d: d["steps"][0].update(version=2),
        lambda d: d["steps"][0].update(operation="python"),
        lambda d: d["steps"][0]["settings"].update(path="/not-a-workflow-input"),
        lambda d: d["steps"][0]["settings"]["parameters"].update(sigma=float("nan")),
        lambda d: d["steps"][0]["settings"].update(roi_names=["embedded-resource-name"]),
        lambda d: d["steps"][0]["settings"].update(input="map:unbound-resource"),
        lambda d: d["steps"][0]["settings"].update(input=[]),
        lambda d: d["steps"][0]["settings"].update(axes=[["x"]]),
        lambda d: d["steps"][0]["settings"].update(roi_names=[{"$binding": "undeclared"}]),
        lambda d: d.update(extensions={"unqualified": {}}),
        lambda d: d["steps"].append(copy.deepcopy(d["steps"][0])),
        lambda d: d["steps"][0].update(enabled=False),
    ],
)
def test_invalid_or_future_documents_fail_closed(change):
    data = processing_document().model_dump()
    change(data)
    with pytest.raises((ValueError, TypeError)):
        loads_workflow(json.dumps(data))


def test_duplicate_keys_bounded_input_and_no_code_execution(tmp_path):
    text = dumps_workflow(processing_document())
    with pytest.raises(ValueError, match="Duplicate"):
        loads_workflow(
            text.replace('"format_version": 1', '"format_version": 1, "format_version": 2')
        )
    with pytest.raises(ValueError, match="1 MiB"):
        loads_workflow(" " * (1024 * 1024 + 1))
    with pytest.raises(ValueError):
        loads_workflow(b"\xff")
    data = processing_document().model_dump()
    data["steps"][0]["settings"]["method"] = "__import__('os').system('false')"
    workflow = loads_workflow(json.dumps(data))
    workspace, _ = phantom()
    checks = validate_workflow_targets(workspace, workflow, ["A"], {})
    assert not checks[0]["ready"]
    assert "Unknown processing" in checks[0]["errors"][0]
    assert not list(tmp_path.iterdir())


def test_unsupported_application_modes_are_rejected_before_running():
    workspace, image = phantom()
    with pytest.raises(ValueError, match="only for fitting"):
        apply(workspace, processing_document(), run=False)
    reconstruction = new_workflow(
        "reconstruction",
        [
            {
                "id": "reco",
                "version": 1,
                "operation": "reconstruct",
                "settings": {"component": "magnitude"},
            }
        ],
    )
    with pytest.raises(ValueError, match="only for processing"):
        apply(workspace, reconstruction, mode="append")
    assert not workspace.pipelines
    np.testing.assert_array_equal(workspace.session.image_array, image)


def test_processing_replay_is_numerically_correct_repeatable_and_preserves_maps(tmp_path):
    workspace, image = phantom(second=True)
    workspace.session.replace_fitted_maps({"old": Quantity(np.ones((9, 11)) * 42, "ms")})
    workspace.fit_results_by_experiment["A"] = {
        "fields": workspace.session.fitted_maps,
        "pipeline_signature": "original",
    }
    document = processing_document(
        {"method": "gaussian", "input": "image", "axes": ["x", "y"], "parameters": {"sigma": 1}},
        {
            "method": "baseline",
            "input": "image",
            "axis": "frame",
            "parameters": {"baseline_frames": 1, "mode": "subtract"},
        },
    )
    apply(workspace, document, ["A", "B"])
    assert workspace.session.selected_experiment == "A"
    expected = gaussian_filter(image, sigma=(0, 1, 1), mode="mirror")
    expected = expected - expected[:1]
    np.testing.assert_allclose(workspace.session.image_array, expected, atol=1e-5)
    np.testing.assert_array_equal(workspace.session.original_image_array, image)
    np.testing.assert_array_equal(workspace.session.map_array("old"), 42)
    assert pipeline_info(workspace)["fits_stale"]
    assert workspace.session.experiment_names == ["A", "B"]
    apply(workspace, document)
    np.testing.assert_allclose(workspace.session.image_array, expected, atol=1e-5)
    workspace.session.select_experiment("B")
    np.testing.assert_allclose(workspace.session.image_array, expected * 2, atol=2e-5)
    path = tmp_path / "replayed.mrax"
    save_workspace_bundle(workspace, path)
    restored = load_workspace_bundle(path, tmp_path / "restored", key="reopened")
    np.testing.assert_allclose(restored.session.image_array, expected * 2, atol=2e-5)
    np.testing.assert_array_equal(restored.session.original_image_array, image * 2)


def test_append_repeated_denoisers_and_disabled_step_remain_editable():
    workspace, image = phantom()
    document = processing_document()
    apply(workspace, document)
    apply(workspace, document, mode="append")
    assert len(pipeline_info(workspace)["steps"]) == 2
    once = gaussian_filter(image, (0, 1, 1), mode="mirror")
    np.testing.assert_allclose(
        workspace.session.image_array, gaussian_filter(once, (0, 1, 1), mode="mirror"), atol=1e-5
    )
    values = document.model_dump()
    values["steps"].append(
        {
            "id": "disabled",
            "operation": "process",
            "version": 1,
            "enabled": False,
            "settings": {"method": "median", "input": "image", "axes": ["x", "y"]},
        }
    )
    apply(workspace, WorkflowDocument.model_validate(values))
    np.testing.assert_allclose(workspace.session.image_array, once, atol=1e-5)
    assert not pipeline_info(workspace)["steps"][1]["enabled"]


def test_capture_keeps_settings_and_bindings_but_no_image_buffers_or_resource_names():
    workspace, image = phantom()
    mask = np.zeros((9, 11), dtype=bool)
    mask[2:7, 3:8] = True
    workspace.session.add_roi(ROI("Source-specific mask", mask, dims=("x", "y")))
    run_operation(
        workspace,
        "process",
        {
            "method": "gaussian",
            "input": "image",
            "axes": ["x", "y"],
            "parameters": {"sigma": 1},
            "roi_names": ["Source-specific mask"],
        },
        lambda *_: None,
    )
    document, suggested = capture_workflow(workspace, "processing")
    text = dumps_workflow(document)
    assert "Source-specific mask" not in text
    assert "Independent reusable-workflow study" not in text
    assert "input_grid" not in text and "references" not in text
    assert len(document.bindings) == 1 and list(suggested.values()) == ["Source-specific mask"]
    target, _ = phantom()
    target.session.add_roi(ROI("Target mask", mask, dims=("x", "y")))
    binding = {document.bindings[0].id: "Target mask"}
    apply(target, document, bindings={"A": binding})
    expected = image.copy()
    filtered = gaussian_filter(image, (0, 1, 1), mode="mirror")
    expected[:, mask] = filtered[:, mask]
    np.testing.assert_allclose(target.session.image_array, expected, atol=1e-5)


def test_missing_bindings_and_late_failure_or_cancel_leave_every_target_untouched():
    workspace, image = phantom(second=True)
    document = processing_document({"method": "median", "input": "image", "axes": ["x", "y"]})
    # A passes and is evaluated; a second target fails during filtering, after preflight.
    broken = np.full_like(image, np.nan)
    workspace.session.add_experiment("Invalid", broken, dims={"frame": 4, "x": 9, "y": 11})
    workspace.session.select_experiment("A")
    with pytest.raises(ValueError, match="finite"):
        apply(workspace, document, ["A", "Invalid"])
    np.testing.assert_array_equal(workspace.session.image_array, image)
    assert not workspace.pipelines
    assert workspace.session.selected_experiment == "A"

    def cancel(done, total, _detail):
        if done >= 1:
            raise InterruptedError("Synthetic cancellation")

    with pytest.raises(InterruptedError):
        apply(workspace, document, progress=cancel)
    np.testing.assert_array_equal(workspace.session.image_array, image)
    assert not workspace.pipelines


def test_reconstruction_recipe_reuses_target_raw_data_and_retains_processing(tmp_path):
    workspace = StudyWorkspace(
        key="reco", label="Synthetic raw", session=synthetic_multimodal_session()
    )
    workspace.session.select_experiment("Complex coils")
    document, _ = capture_workflow(workspace, "reconstruction", draft={"component": "magnitude"})
    path = tmp_path / "reconstruct.mrec"
    save_workflow(document, path)
    apply(workspace, load_workflow(path))
    original = np.asarray(workspace.session.original_image_array).copy()
    apply(workspace, processing_document())
    assert [row["kind"] for row in pipeline_info(workspace)["steps"]] == ["reconstruct", "process"]
    values = document.model_dump()
    values["steps"][0]["settings"]["component"] = "phase"
    apply(workspace, WorkflowDocument.model_validate(values))
    assert [row["kind"] for row in pipeline_info(workspace)["steps"]] == ["reconstruct", "process"]
    assert np.all(np.isfinite(workspace.session.image_array))
    np.testing.assert_array_equal(workspace.session.original_image_array, original)
    no_raw, _ = phantom()
    with pytest.raises(ValueError, match="raw data"):
        apply(no_raw, document)


def test_fitting_workflow_configures_then_runs_on_current_processed_state(tmp_path):
    source = StudyWorkspace(key="fit-source", label="Synthetic T2", session=synthetic_t2_session())
    mask = np.zeros(source.session.spatial_shape, dtype=bool)
    mask[7:17, 10:22, :] = True
    source.session.add_roi(ROI("Source region", mask, dims=tuple(source.session.spatial_dim_names)))
    document, _ = capture_workflow(
        source,
        "fitting",
        draft={
            "settings": {"num_steps": 600, "step_size": 0.02},
            "use_roi": True,
            "roi_names": ["Source region"],
            "excluded_frames": [0],
        },
    )
    path = tmp_path / "T2.mfit"
    save_workflow(document, path)
    assert "Source region" not in path.read_text()
    target = StudyWorkspace(key="fit-target", label="Another study", session=synthetic_t2_session())
    target.session.add_roi(ROI("Target region", mask, dims=tuple(target.session.spatial_dim_names)))
    target.session.replace_fitted_maps(
        {"retained": Quantity(np.ones(target.session.spatial_shape), "ms")}
    )
    apply(target, processing_document())
    selection = {target.session.selected_experiment: {document.bindings[0].id: "Target region"}}
    apply(target, load_workflow(path), bindings=selection, run=False)
    assert target.session.fit_settings.num_steps == 600
    assert target.session.frame_mask.excluded_indices == [0]
    assert target.session.map_names == ["retained"]
    apply(target, document, bindings=selection, run=True)
    assert "t2" in target.session.map_names and "retained" not in target.session.map_names
    np.testing.assert_allclose(target.session.map_array("t2")[mask], 80, rtol=0.02)
    result = target.fit_results_by_experiment[target.session.selected_experiment]
    assert result["workflow"]["kind"] == "fitting"
    assert result["pipeline_signature"] == pipeline_info(target)["signature"]


def test_target_catalog_and_validation_do_not_mutate_active_state():
    workspace, image = phantom(second=True)
    before = copy.deepcopy(workspace.session.view_states)
    assert len(resource_choices(workspace)) == 2
    invalid = processing_document({"method": "gaussian", "axes": ["absent"]})
    assert not validate_workflow_targets(workspace, invalid, ["A", "B"], {})[0]["ready"]
    assert workspace.session.selected_experiment == "A"
    assert workspace.session.view_states.keys() == before.keys()
    assert not workspace.pipelines and not workspace.dirty
    np.testing.assert_array_equal(workspace.session.image_array, image)


def test_recipe_reconstructs_new_cartesian_samples_instead_of_reusing_saved_output():
    import jax.numpy as jnp

    from mrax.data_struct import AcquisitionData
    from mrax.reco import fftnc

    data = np.random.default_rng(18).uniform(1, 4, (2, 9, 11)).astype(np.float32)
    document = new_workflow(
        "reconstruction",
        [
            {
                "id": "reco",
                "operation": "reconstruct",
                "version": 1,
                "settings": {"component": "magnitude"},
            }
        ],
    )
    for scale in (1, 2):
        session = GuiSession()
        session.create_empty_study("Independent Cartesian samples")
        raw = AcquisitionData(
            np.asarray(fftnc(jnp.asarray(data[:, None] * scale))),
            (9, 11),
            ("phase", "read"),
        )
        vendor = data * 0.25
        session.add_experiment(
            "Target", vendor, dims={"frame": 2, "x": 9, "y": 11}, kspace=raw.samples
        )
        session.current_experiment["acquisition"] = raw
        session.study.datasets[0].acquisition = raw
        workspace = StudyWorkspace(key="new-raw", label=session.study_name, session=session)
        apply(workspace, document)
        np.testing.assert_allclose(workspace.session.image_array, data * scale, rtol=1e-5)
        np.testing.assert_array_equal(workspace.session.original_image_array, vendor)


def test_map_inputs_and_reference_experiments_are_rebound_on_the_target():
    source, image = phantom(second=True)
    source.session.replace_fitted_maps({"Source map": Quantity(image[0], "ms")})
    run_operation(
        source,
        "process",
        {
            "method": "gaussian",
            "input": "map:Source map",
            "axes": ["x", "y"],
            "parameters": {"sigma": 1},
        },
        lambda *_: None,
    )
    document, _ = capture_workflow(source, "processing")
    assert "Source map" not in dumps_workflow(document)
    target, _ = phantom()
    target.session.replace_fitted_maps({"Different map": Quantity(image[0] * 3, "ms")})
    apply(target, document, bindings={"A": {document.bindings[0].id: "Different map"}})
    np.testing.assert_allclose(
        target.session.image_array, gaussian_filter(image[0] * 3, 1, mode="mirror"), atol=1e-5
    )
    source, image = phantom(second=True)
    run_operation(
        source,
        "register",
        {
            "reference_experiment": "B",
            "moving_experiment": "A",
            "reference_frame": 0,
            "moving_frame": 0,
            "scope": "plane",
            "manual": True,
            "manual_settings": {"shift_x": 1, "shift_y": 0},
        },
        lambda *_: None,
    )
    recipe, _ = capture_workflow(source, "processing")
    assert '"B"' not in dumps_workflow(recipe)
    target, _ = phantom(second=True)
    target.session.rename_experiment("B", "New reference")
    binding = {recipe.bindings[0].id: "New reference"}
    apply(target, recipe, bindings={"A": binding})
    assert pipeline_info(target)["steps"][0]["kind"] == "register"
    assert np.all(np.isfinite(target.session.image_array))
    np.testing.assert_array_equal(target.session.original_image_array, image)


def test_all_target_rois_are_dynamic_and_invalid_fit_settings_cannot_replace_maps():
    source = StudyWorkspace(key="all", label="Synthetic T2", session=synthetic_t2_session())
    document, _ = capture_workflow(source, "fitting", draft={"use_roi": True, "roi_names": None})
    assert not document.bindings
    target = StudyWorkspace(
        key="all-target", label="Another synthetic", session=synthetic_t2_session()
    )
    mask = np.zeros(target.session.spatial_shape, dtype=bool)
    mask[8:12, 10:15, :] = True
    target.session.add_roi(
        ROI("Any target name", mask, dims=tuple(target.session.spatial_dim_names))
    )
    apply(target, document, run=False)
    assert target.display_settings[target.session.selected_experiment]["scene"]["fit_rois"] is None
    bad = document.model_dump()
    bad["steps"][0]["settings"]["excluded_frames"] = list(range(target.session.frame_count))
    before = target.session.fit_settings
    with pytest.raises(ValueError, match="at least one"):
        apply(target, WorkflowDocument.model_validate(bad), run=False)
    assert target.session.fit_settings == before
    bad["steps"][0]["settings"]["excluded_frames"] = []
    bad["steps"][0]["settings"]["settings"]["num_steps"] = True
    with pytest.raises(ValueError, match="integer"):
        WorkflowDocument.model_validate(bad)


def wait_job(client, job):
    deadline = time.monotonic() + 20
    while job["status"] in {"queued", "running", "cancelling"}:
        assert time.monotonic() < deadline
        time.sleep(0.01)
        job = client.get("/api/jobs/" + job["id"]).json()
    return job


def test_api_session_sharing_edit_conflicts_files_and_fresh_session(tmp_path):
    app = create_app(read_roots=[tmp_path], write_roots=[tmp_path], filesystem_scope="shared")
    with TestClient(app, base_url="http://127.0.0.1") as client:
        headers = {"x-mrax-token": client.get("/api/session").json()["token"]}
        first = client.post("/api/workspaces/synthetic", headers=headers).json()
        second = client.post("/api/workspaces/synthetic", headers=headers).json()
        created = client.post(
            "/api/workflows", headers=headers, json={"document": processing_document().model_dump()}
        )
        assert created.status_code == 200
        recipe = created.json()
        root = next(row["id"] for row in client.get("/api/files/roots").json() if row["writable"])
        saved = client.post(
            "/api/workflows/" + recipe["id"] + "/save-file",
            headers=headers,
            json={
                "revision": 0,
                "root": root,
                "path": "shared.mpip",
            },
        )
        assert saved.status_code == 200
        for workspace in [first, second]:
            response = client.post(
                "/api/workspaces/" + workspace["id"] + "/workflows/" + recipe["id"] + "/apply",
                headers=headers,
                json={
                    "revision": workspace["revision"],
                    "workflow_revision": 0,
                    "experiments": [workspace["experiment"]],
                },
            )
            assert response.status_code == 200, response.text
            assert wait_job(client, response.json())["status"] == "completed"
        assert len(client.get("/api/workspaces").json()) == 2
        changed = copy.deepcopy(recipe["document"])
        changed["name"] = "Edited Gaussian"
        assert (
            client.put(
                "/api/workflows/" + recipe["id"],
                headers=headers,
                json={
                    "revision": 0,
                    "document": changed,
                },
            ).status_code
            == 200
        )
        assert (
            client.put(
                "/api/workflows/" + recipe["id"],
                headers=headers,
                json={
                    "revision": 0,
                    "document": changed,
                },
            ).status_code
            == 409
        )
        bad = dumps_workflow(processing_document()).replace(
            '"format_version": 1', '"format_version": 1, "format_version": 2'
        )
        assert (
            client.put(
                "/api/workflows/" + recipe["id"],
                headers=headers,
                json={
                    "revision": 1,
                    "content": bad,
                },
            ).status_code
            == 422
        )
        assert client.get("/api/workflows").json()[0]["revision"] == 1
    with TestClient(
        create_app(read_roots=[tmp_path], filesystem_scope="shared"), base_url="http://127.0.0.1"
    ) as client:
        assert client.get("/api/workflows").json() == []
        headers = {"x-mrax-token": client.get("/api/session").json()["token"]}
        root = client.get("/api/files/roots").json()[0]["id"]
        reopened = client.post(
            "/api/workflows/open-file", headers=headers, json={"root": root, "path": "shared.mpip"}
        )
        assert reopened.status_code == 200
        assert reopened.json()["document"] == processing_document().model_dump()


def test_running_job_uses_snapshot_and_stale_results_do_not_overwrite_edits(monkeypatch):
    from mrax.gui.services import reusable_workflows

    entered, release = threading.Event(), threading.Event()
    original = reusable_workflows.apply_workflow
    captured = {}

    def delayed(*args, **kwargs):
        captured["document"] = copy.deepcopy(args[1])
        entered.set()
        assert release.wait(10)
        return original(*args, **kwargs)

    monkeypatch.setattr(reusable_workflows, "apply_workflow", delayed)
    with TestClient(create_app(), base_url="http://127.0.0.1") as client:
        headers = {"x-mrax-token": client.get("/api/session").json()["token"]}
        workspace = client.post("/api/workspaces/synthetic", headers=headers).json()
        recipe = client.post(
            "/api/workflows", headers=headers, json={"document": processing_document().model_dump()}
        ).json()
        job = client.post(
            "/api/workspaces/" + workspace["id"] + "/workflows/" + recipe["id"] + "/apply",
            headers=headers,
            json={
                "revision": workspace["revision"],
                "workflow_revision": 0,
                "experiments": [workspace["experiment"]],
            },
        ).json()
        try:
            assert entered.wait(5)
            changed = copy.deepcopy(recipe["document"])
            changed["steps"][0]["settings"]["parameters"]["sigma"] = 3
            assert (
                client.put(
                    "/api/workflows/" + recipe["id"],
                    headers=headers,
                    json={
                        "revision": 0,
                        "document": changed,
                    },
                ).status_code
                == 200
            )
            edited = client.post(
                "/api/workspaces/" + workspace["id"] + "/actions",
                headers=headers,
                json={
                    "revision": workspace["revision"],
                    "action": "study.rename",
                    "options": {"name": "User edit"},
                },
            )
            assert edited.status_code == 200
        finally:
            release.set()
        assert wait_job(client, job)["status"] == "stale"
        result = client.get("/api/workspaces/" + workspace["id"]).json()
        assert result["label"] == "User edit"
        assert not result["pipeline"]["steps"]
        assert captured["document"]["steps"][0]["settings"]["parameters"]["sigma"] == 1
