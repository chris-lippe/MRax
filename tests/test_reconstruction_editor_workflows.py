"""Editable reconstruction plans on independently generated acquisitions."""

import copy

import numpy as np
import pytest

from mrax.data_struct.acquisition import AcquisitionData, ReconstructionRecipe
from mrax.gui.services.analysis_workflows import run_operation
from mrax.gui.services.reconstruction import reconstruction_label
from mrax.gui.services.workspace import StudyWorkspace
from mrax.gui.session import GuiSession
from mrax.reco import fftnc, reconstruct_acquisition
from mrax.reco.plans import compile_plan, reconstruction_presets
from mrax.workflows import WorkflowDocument, dumps_workflow, loads_workflow


def acquisition(*, sampled=False):
    rng = np.random.default_rng(43)
    coils = (rng.normal(size=(2, 2, 6, 8)) + 1j * rng.normal(size=(2, 2, 6, 8))).astype(
        np.complex64
    )
    samples = np.asarray(fftnc(coils, axes=(-2, -1)))
    coordinates = (
        np.stack(np.meshgrid((np.arange(6) - 3) / 6, (np.arange(8) - 4) / 8, indexing="ij"), -1)
        .reshape(-1, 2)
        .astype(np.float32)
    )
    raw = AcquisitionData(
        samples.reshape(2, 2, -1) if sampled else samples,
        (6, 8),
        ("x", "y"),
        kind="sampled" if sampled else "cartesian",
        trajectory=coordinates if sampled else None,
        recipe=ReconstructionRecipe(coil_combine="sum"),
    )
    return raw, coils


def workspace(raw):
    session = GuiSession()
    session.create_empty_study("Independent reconstruction stages")
    session.add_experiment(
        "Acquisition",
        np.ones((2, 6, 8), np.float32),
        dims={"frame": 2, "x": 6, "y": 8},
        model_name="identity",
        kspace=raw.samples,
    )
    session.current_experiment["acquisition"] = raw
    session.study.datasets[0].acquisition = raw
    return StudyWorkspace("reconstruction", session.study_name, session)


def preset(name):
    return copy.deepcopy(next(p["plan"] for p in reconstruction_presets() if p["id"] == name))


def test_shipped_presets_are_versioned_mrec_documents_and_have_distinct_algorithms():
    rows = reconstruction_presets()
    assert len(rows) >= 6
    assert {"fft", "variational", "nufft"} <= {s["kind"] for p in rows for s in p["plan"]["stages"]}
    for row in rows:
        doc = loads_workflow(dumps_workflow(WorkflowDocument.model_validate(row["document"])))
        assert doc.kind == "reconstruction"
        assert doc.steps[0].settings["pipeline"]["name"] == row["name"]
        assert row["filename"].endswith(".mrec")


def test_reconstruction_edits_rerun_cached_images_preserve_source_and_keep_downstream_processing():
    raw, _ = acquisition()
    work = workspace(raw)
    original = work.session.image_array.copy()
    settings = preset("cartesian-fourier")
    run_operation(
        work, "reconstruct", {"pipeline": settings, "component": "magnitude"}, lambda *_: None
    )
    run_operation(
        work,
        "process",
        {"method": "gaussian", "input": "image", "axes": ["x", "y"], "parameters": {"sigma": 0.4}},
        lambda *_: None,
    )
    first = work.session.image_array.copy()
    settings["name"] = "Windowed Fourier"
    window = next(s for s in settings["stages"] if s["kind"] == "window")
    window["options"] = {"kspace_window": ["hann", "hann"]}
    run_operation(
        work, "reconstruct", {"pipeline": settings, "component": "magnitude"}, lambda *_: None
    )
    assert not np.array_equal(work.session.image_array, first)
    assert reconstruction_label(work) == "Windowed Fourier"
    assert [step["kind"] for step in work.pipelines["Acquisition"]["steps"]] == [
        "reconstruct",
        "process",
    ]
    np.testing.assert_array_equal(work.session.original_image_array, original)
    np.testing.assert_array_equal(
        work.session.current_experiment["acquisition"].samples, raw.samples
    )


def test_variational_preset_runs_its_solver_and_invalid_plans_fail_before_modifying_workspace():
    raw, _ = acquisition()
    plan = preset("variational")
    solver = next(s for s in plan["stages"] if s["kind"] == "variational")
    solver["options"].update(num_steps=2, regularizer="none", regularizer_weight=0)
    result = compile_plan(raw, plan).run(raw)
    assert result.method == "variational"
    np.testing.assert_allclose(
        result.image, reconstruct_acquisition(raw).image, rtol=1e-5, atol=1e-5
    )
    work = workspace(raw)
    original = work.session.image_array.copy()
    invalid = preset("nufft-adjoint")
    with pytest.raises(ValueError, match="sampled"):
        run_operation(
            work, "reconstruct", {"pipeline": invalid, "component": "magnitude"}, lambda *_: None
        )
    np.testing.assert_array_equal(work.session.image_array, original)
    assert not work.pipelines
