"""Composite pipelines use only independently generated acquisitions and signals."""

import copy

import numpy as np
import pytest

from mrax.data_struct import AcquisitionData, Quantity
from mrax.gui.services.pipeline import pipeline_info
from mrax.gui.services.pipeline_blocks import (
    block_options,
)
from mrax.gui.services.pipeline_editing import apply_pipeline_draft
from mrax.gui.services.reusable_workflows import (
    apply_workflow,
    capture_workflow,
    validate_workflow_targets,
)
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


def test_common_workflow_replays_flat_reconstruction_processing_and_fitting_blocks(tmp_path):
    work, _ = workspace()
    apply(
        work,
        [
            row("reconstruct", block_options(work, "reconstruct", {}), "reco"),
            row("process", gaussian(), "smooth"),
            row("fit", block_options(work, "fit", {"use_roi": False}), "fit"),
        ],
    )
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
