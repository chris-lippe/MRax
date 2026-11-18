"""Composite pipelines use only independently generated acquisitions and signals."""

import copy

import numpy as np
import pytest

from mrax.analysis import ROI
from mrax.data_struct import AcquisitionData, Quantity
from mrax.gui import session as session_module
from mrax.gui.services.analysis_workflows import run_operation
from mrax.gui.services.comparison import render_comparison
from mrax.gui.services.pipeline import change_pipeline, pipeline_info
from mrax.gui.services.pipeline_blocks import (
    block_options,
)
from mrax.gui.services.pipeline_editing import apply_pipeline_draft
from mrax.gui.services.reusable_workflows import (
    apply_workflow,
    capture_workflow,
    validate_workflow_targets,
)
from mrax.gui.services.scene import scene_settings, update_scene
from mrax.gui.services.workspace import StudyWorkspace
from mrax.gui.services.workspace_io import load_workspace_bundle, save_workspace_bundle
from mrax.gui.session import GuiSession
from mrax.reco import fftnc
from mrax.workflows import WorkflowDocument, dumps_workflow, loads_workflow, new_workflow


def workspace(*, raw=True):
    times = np.linspace(10, 120, 8, dtype=np.float32)
    x, y = np.meshgrid(np.linspace(-1, 1, 4), np.linspace(-1, 1, 5), indexing="ij")
    amplitude = 1e6 * (0.6 + 0.4 * np.exp(-3 * (x * x + y * y)))
    image = (amplitude[None] * np.exp(-times[:, None, None] / 80)).astype(np.float32)
    session = GuiSession()
    session.create_empty_study("Independent composite pipeline", field_strength_mhz=100)
    session.add_experiment(
        "A",
        image,
        dims={"frame": 8, "x": 4, "y": 5},
        model_name="t2_se",
        observables={"TE": Quantity(times, "ms")},
    )
    session.fit_settings.num_steps = 600
    session.fit_settings.param_overrides = {"bias": {"trainable": False, "value": 0}}
    if raw:
        samples = np.asarray(fftnc(image[:, None].astype(np.complex64) * 2, axes=(-2, -1)))
        acquisition = AcquisitionData(samples, (4, 5), ("x", "y"))
        session.current_experiment.update(acquisition=acquisition, raw_kspace=samples)
        session.study.datasets[0].acquisition = acquisition
        session.study.datasets[0].kspace = samples
    return StudyWorkspace("composite", session.study_name, session), image


def row(kind, options, name):
    return {"id": name, "kind": kind, "options": copy.deepcopy(options), "enabled": True}


def apply(work, rows, progress=lambda *_: None):
    return apply_pipeline_draft(
        work, {"base_generation": pipeline_info(work)["generation"], "steps": rows}, progress
    )


def gaussian():
    return {
        "method": "gaussian",
        "input": "image",
        "axes": ["x", "y"],
        "parameters": {"sigma": 0.7},
    }


def editable(step, *, identifier=None):
    return {
        **row(step["kind"], step["options"], identifier or step["id"]),
        "source_id": step["id"],
        "source_version": step["version"],
    }


def test_common_workflow_replays_flat_reconstruction_processing_and_fitting_blocks(tmp_path, monkeypatch):
    fit_inputs = []
    original_fit = session_module.run_fit

    def record_fit(experiment, model, dataset, **kwargs):
        fit_inputs.append(np.asarray(dataset.data).copy())
        return original_fit(experiment, model, dataset, **kwargs)

    monkeypatch.setattr(session_module, "run_fit", record_fit)
    work, source = workspace()
    apply(
        work,
        [
            row("reconstruct", block_options(work, "reconstruct", {}), "reco"),
            row("process", gaussian(), "smooth"),
            row("fit", block_options(work, "fit", {"use_roi": False}), "fit"),
        ],
    )
    np.testing.assert_array_equal(fit_inputs[-1], work.session.image_array)
    smooth = work.pipelines["A"]["steps"][1]
    first_image = work.session.image_array.copy()
    run_operation(work, "process", {**gaussian(), "step_id": smooth["id"],
                  "parameters": {"sigma": 1.1}}, lambda *_: None)
    assert len(fit_inputs) == 2
    assert not np.allclose(first_image, work.session.image_array)
    np.testing.assert_array_equal(fit_inputs[-1], work.session.image_array)
    proposal = [editable(step) for step in work.pipelines["A"]["steps"]]
    proposal[1]["options"]["parameters"]["sigma"] = 0.4
    apply(work, proposal)
    assert len(fit_inputs) == 3
    np.testing.assert_array_equal(fit_inputs[-1], work.session.image_array)
    apply(work, [editable(step) for step in work.pipelines["A"]["steps"]])
    assert len(fit_inputs) == 3
    for count, enabled in ((4, False), (5, True)):
        change_pipeline(work, {"action": "toggle", "step_id": smooth["id"], "enabled": enabled}, lambda *_: None)
        assert len(fit_inputs) == count
        np.testing.assert_array_equal(fit_inputs[-1], work.session.image_array)
    np.testing.assert_array_equal(work.session.original_image_array, source)
    document, bindings = capture_workflow(work, "processing")
    restored = loads_workflow(dumps_workflow(document), suffix=".mflow")
    assert [step.operation for step in restored.steps] == ["reconstruct", "process", "fit"]
    assert "pipeline" in restored.steps[0].settings
    assert "model_name" in restored.steps[2].settings["settings"]
    assert all("workflow" not in step.settings for step in restored.steps)
    target, original = workspace()
    assert validate_workflow_targets(target, restored, ["A"], {"A": bindings})[0]["ready"]
    apply_workflow(target, restored.model_dump(), ["A"], {"A": bindings}, lambda *_: None)
    np.testing.assert_allclose(target.session.image_array, work.session.image_array)
    np.testing.assert_array_equal(target.session.original_image_array, original)
    assert len(target.session.map_names) == 3
    work.fit_results_by_experiment["A"]["workflow"] = restored.model_dump()
    saved = save_workspace_bundle(work, tmp_path / "combined.mrax")
    original_bundle = saved.read_bytes()
    reopened = load_workspace_bundle(saved, tmp_path / "reopened", key="reopened")
    assert saved.read_bytes() == original_bundle
    assert reopened.fit_results_by_experiment["A"]["workflow"] == restored.model_dump()
    for saved_step, opened_step in zip(
        work.pipelines["A"]["steps"], reopened.pipelines["A"]["steps"], strict=True
    ):
        if saved_step["kind"] in {"reconstruct", "fit"}:
            assert opened_step["options"]["workflow"] == saved_step["options"]["workflow"]
    np.testing.assert_array_equal(reopened.session.image_array, work.session.image_array)

    # Presets and action buttons refresh the same block, including after a method change.
    actions, source = workspace()
    fit, bindings = capture_workflow(actions, "fitting", draft={"use_roi": False})
    apply_workflow(actions, fit.model_dump(), ["A"], {"A": bindings}, lambda *_: None, run=False)
    configured = pipeline_info(actions)["steps"][0]
    assert configured["pending"]
    assert not actions.session.map_names
    np.testing.assert_array_equal(actions.session.image_array, source)
    for _ in range(2):
        run_operation(actions, "full_fit", {"use_roi": False}, lambda *_: None)
        assert pipeline_info(actions)["steps"][0]["id"] == configured["id"]
        assert len(pipeline_info(actions)["steps"]) == 1
        assert not pipeline_info(actions)["steps"][0]["pending"]
    fitted = {name: actions.session.map_array(name).copy() for name in actions.session.map_names}
    fit_version = actions.pipelines["A"]["fits"][configured["id"]]["result"]["pipeline_signature"]
    for sigma in (0.5, 0.9):
        run_operation(actions, "process", {**gaussian(), "parameters": {"sigma": sigma}}, lambda *_: None)
    assert len(pipeline_info(actions)["steps"]) == 2
    for name, values in fitted.items():
        np.testing.assert_array_equal(actions.session.map_array(name), values)
    assert actions.pipelines["A"]["fits"][configured["id"]]["result"]["pipeline_signature"] == fit_version
    before = actions.session.image_array.copy()
    run_operation(actions, "pipeline", {"action": "enabled", "enabled": False}, lambda *_: None)
    run_operation(actions, "full_fit", {"use_roi": False}, lambda *_: None)
    np.testing.assert_array_equal(actions.session.image_array, source)
    assert not pipeline_info(actions)["enabled"]
    assert actions.fit_results_by_experiment["A"]["pipeline_signature"] == "original"
    run_operation(actions, "pipeline", {"action": "enabled", "enabled": True}, lambda *_: None)
    np.testing.assert_array_equal(actions.session.image_array, before)
    reco, bindings = capture_workflow(actions, "reconstruction")
    apply_workflow(actions, reco.model_dump(), ["A"], {"A": bindings}, lambda *_: None, run=False)
    np.testing.assert_array_equal(actions.session.image_array, before)
    reco_id = pipeline_info(actions)["steps"][0]["id"]
    for _ in range(2):
        run_operation(actions, "reconstruct", {"view": "mrax_reconstruction"}, lambda *_: None)
        assert pipeline_info(actions)["steps"][0]["id"] == reco_id
        assert len(pipeline_info(actions)["steps"]) == 3
    # Rerunning fitting uses all current image processing and retains its identity.
    run_operation(actions, "full_fit", {"use_roi": False}, lambda *_: None)
    chain = pipeline_info(actions)["steps"]
    assert chain[-1]["id"] == configured["id"]
    assert not pipeline_info(actions)["fits_stale"]

    current_image = actions.session.image_array.copy()
    baseline = pipeline_info(actions)
    for kind, label in (("reconstruct", "reconstruction"), ("fit", "fitting")):
        source_step = next(step for step in chain if step["kind"] == kind)
        proposal = [editable(step) for step in chain]
        proposal.insert(1 if kind == "reconstruct" else len(proposal), editable(source_step, identifier="copy"))
        with pytest.raises(ValueError, match=f"one {label} block"):
            apply(actions, proposal)
        proposal[1 if kind == "reconstruct" else -1]["enabled"] = False
        with pytest.raises(ValueError, match=f"one {label} block"):
            apply(actions, proposal)
        assert pipeline_info(actions) == baseline
        np.testing.assert_array_equal(actions.session.image_array, current_image)
        invalid = copy.deepcopy(document.model_dump())
        duplicate = copy.deepcopy(next(step for step in invalid["steps"] if step["operation"] == kind))
        duplicate["id"] = "copy"
        invalid["steps"].append(duplicate)
        with pytest.raises(ValueError, match=f"one {label} block"):
            WorkflowDocument.model_validate(invalid)


def test_named_outputs_cross_experiment_targets_and_late_failures_are_transactional(tmp_path):
    work, original = workspace(raw=False)
    work.session.add_experiment(
        "B",
        original * 2,
        dims={"frame": 8, "x": 4, "y": 5},
        model_name="t2_se",
        observables={"TE": Quantity(np.linspace(10, 120, 8), "ms")},
    )
    work.session.select_experiment("A")
    fit, _ = capture_workflow(work, "fitting", draft={"use_roi": False})
    steps = [
        {"id": "anatomy", "operation": "process", "version": 1, "settings": gaussian()},
        {"id": "relaxation", "operation": "fit", "version": 1, "settings": fit.steps[0].settings},
        {
            "id": "map_smoothing",
            "operation": "process",
            "version": 1,
            "settings": {
                **gaussian(),
                "input": {"$output": "relaxation", "port": "map", "name": "t2"},
            },
        },
        {
            "id": "align",
            "operation": "register",
            "version": 1,
            "target": {"$binding": "moving"},
            "settings": {
                "reference_experiment": {"$output": "anatomy", "port": "image"},
                "manual": True,
            },
        },
    ]
    document = new_workflow(
        "processing",
        steps,
        bindings=[{"id": "moving", "kind": "experiment", "label": "Second acquisition"}],
    )
    assignments = {"A": {"moving": "B"}}
    assert validate_workflow_targets(work, document, ["A"], assignments)[0]["ready"]
    apply_workflow(work, document.model_dump(), ["A"], assignments, lambda *_: None)
    np.testing.assert_array_equal(work.session.original_image_array, original)
    assert "relaxation / t2" in work.session.map_names
    a = work.session.image_array.copy()
    assert work.pipelines["A"]["steps"][-1]["options"]["input"] == "map:relaxation / t2"
    assert work.pipelines["B"]["steps"][0]["options"]["reference_experiment"] == "A"
    from scipy.ndimage import gaussian_filter

    np.testing.assert_allclose(
        work.pipelines["B"]["steps"][0]["references"]["A"]["data"],
        gaussian_filter(original, sigma=(0, 0.7, 0.7), mode="mirror"),
        rtol=1e-5,
    )
    assert a.shape == (4, 5)
    recaptured, _ = capture_workflow(work, "processing")
    assert recaptured.steps[2].settings["input"] == {
        "$output": "step_2",
        "port": "map",
        "name": "t2",
    }
    saved = save_workspace_bundle(work, tmp_path / "named.mrax")
    restored = load_workspace_bundle(saved, tmp_path / "opened", key="restored")
    np.testing.assert_array_equal(restored.session.image_array, a)
    maps = {name: work.session.map_array(name).copy() for name in work.session.map_names}
    bad = document.model_dump()
    bad["steps"][2]["settings"]["input"]["name"] = "missing"
    with pytest.raises(ValueError, match="did not produce"):
        apply_workflow(work, bad, ["A"], assignments, lambda *_: None)
    np.testing.assert_array_equal(work.session.image_array, a)
    for name, values in maps.items():
        np.testing.assert_array_equal(work.session.map_array(name), values)
    future = document.model_dump()
    future["steps"][0]["settings"]["input"] = {"$output": "relaxation", "port": "map", "name": "t2"}
    with pytest.raises(ValueError, match="earlier"):
        WorkflowDocument.model_validate(future)
    targeted_fit = new_workflow(
        "fitting",
        [{**steps[1], "target": {"$binding": "moving"}}],
        bindings=[{"id": "moving", "kind": "experiment", "label": "Second acquisition"}],
    )
    with pytest.raises(ValueError, match="another experiment"):
        block_options(
            work, "fit", {"workflow": targeted_fit.model_dump(), "bindings": {"moving": "B"}}
        )


def test_experiment_pipelines_restore_cached_images_maps_and_settings(tmp_path, monkeypatch):
    work, original = workspace()
    mask = np.zeros((4, 5), dtype=bool)
    mask[1:3, 1:4] = True
    work.session.add_roi(ROI("First", mask, dims=("x", "y")))
    work.session.add_roi(ROI("Second", ~mask, dims=("x", "y")))
    run_operation(work, "reconstruct", {"view": "mrax_reconstruction"}, lambda *_: None)
    run_operation(work, "full_fit", {"use_roi": False, "settings": {"num_steps": 80}}, lambda *_: None)
    first = pipeline_info(work)
    image = work.session.image_array.copy()
    maps = {name: work.session.map_array(name).copy() for name in work.session.map_names}
    def change(**options):
        return change_pipeline(work, options, lambda *_: None)
    change(action="rename", name="Fourier")
    change(action="create")
    second_id = pipeline_info(work)["id"]
    assert second_id != first["id"]
    assert work.session.map_names == []
    assert pipeline_info(work)["steps"] == []
    np.testing.assert_array_equal(work.session.image_array, original)
    run_operation(work, "process", gaussian(), lambda *_: None)
    second_image = work.session.image_array.copy()
    run_operation(work, "full_fit", {"use_roi": False, "settings": {"num_steps": 90}}, lambda *_: None)
    second_maps = {name: work.session.map_array(name).copy() for name in work.session.map_names}
    second = pipeline_info(work)
    update_scene(work, {"fit_region": "rois", "fit_rois": ["First"]})
    change(action="select", pipeline_id=first["id"])
    update_scene(work, {"fit_region": "rois", "fit_rois": None})
    change(action="select", pipeline_id=second_id)
    assert scene_settings(work)["fit_rois"] == ["First"]
    change(action="select", pipeline_id=first["id"])
    assert scene_settings(work)["fit_rois"] is None
    assert {roi.name for roi in work.session.shared_rois} == {"First", "Second"}
    change(action="select", pipeline_id=second_id)

    def no_calculation(*args, **kwargs):
        raise AssertionError("Switching a pipeline must restore its cache")

    monkeypatch.setattr("mrax.gui.services.pipeline._evaluate", no_calculation)
    for _ in range(2):
        change(action="select", pipeline_id=first["id"])
        assert work.session.fit_settings.num_steps == 80
        assert pipeline_info(work)["generation"] == first["generation"]
        np.testing.assert_array_equal(work.session.image_array, image)
        for name, values in maps.items():
            np.testing.assert_array_equal(work.session.map_array(name), values)
        change(action="select", pipeline_id=second_id)
        assert work.session.fit_settings.num_steps == 90
        np.testing.assert_array_equal(work.session.image_array, second_image)
        for name, values in second_maps.items():
            np.testing.assert_array_equal(work.session.map_array(name), values)
    with pytest.raises(ValueError, match="Select this pipeline"):
        change(action="replace", pipeline_id=first["id"], base_generation=second["generation"], steps=[])
    baseline = pipeline_info(work)
    for kind, map_name in (("processed", None), ("map", "t2")):
        settings = scene_settings(work)["comparison"]
        settings.update(enabled=True, mode="split")
        for side, identifier in (("left", first["id"]), ("right", second_id)):
            settings[side] = {"experiment": "A", "pipeline_id": identifier, "kind": kind, "map_name": map_name}
        update_scene(work, {"comparison": settings})
        arrays, information = render_comparison(work)
        assert information["geometry_compatible"]
        if kind == "map":
            np.testing.assert_array_equal(arrays[0], maps["t2"])
            np.testing.assert_array_equal(arrays[1], second_maps["t2"])
        assert pipeline_info(work) == baseline
        np.testing.assert_array_equal(work.session.image_array, second_image)
    saved = save_workspace_bundle(work, tmp_path / "pipelines.mrax")
    restored = load_workspace_bundle(saved, tmp_path / "reopen-pipelines", key="restored")
    assert pipeline_info(restored) == baseline
    change_pipeline(restored, {"action": "select", "pipeline_id": first["id"]}, lambda *_: None)
    np.testing.assert_array_equal(restored.session.image_array, image)
    assert restored.session.fit_settings.num_steps == 80
    change(action="duplicate")
    third = pipeline_info(work)["id"]
    assert third != second_id
    np.testing.assert_array_equal(work.session.image_array, second_image)
    change(action="delete")
    assert third not in {choice["id"] for choice in pipeline_info(work)["choices"]}
    assert len(work.session.experiment_names) == 1
    np.testing.assert_array_equal(work.session.original_image_array, original)
