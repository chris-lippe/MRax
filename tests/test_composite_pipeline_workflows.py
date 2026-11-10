"""Composite pipelines use only independently generated acquisitions and signals."""

import copy
from dataclasses import replace

import numpy as np
import pytest

from mrax.analysis import ROI
from mrax.data_struct import AcquisitionData, Quantity
from mrax.gui.services.pipeline import change_pipeline, pipeline_info, signature
from mrax.gui.services.pipeline_blocks import (
    block_options,
    fit_result_for_map,
    reconstruction_available,
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
from mrax.reco.plans import reconstruction_presets
from mrax.workflows import WorkflowDocument, dumps_workflow, loads_workflow


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


def test_fit_blocks_pass_images_through_and_store_distinct_maps_at_their_chain_positions(tmp_path):
    work, original = workspace()
    reco = block_options(work, "reconstruct", {})
    fit = block_options(work, "fit", {"use_roi": False})
    fit["workflow"]["name"] = "Before smoothing"
    second = copy.deepcopy(fit)
    second["workflow"]["name"] = "After smoothing"
    apply(
        work,
        [
            row("reconstruct", reco, "reco"),
            row("fit", fit, "fit_first"),
            row("process", gaussian(), "smooth"),
            row("fit", second, "fit_second"),
        ],
    )
    from scipy.ndimage import gaussian_filter

    np.testing.assert_allclose(
        work.session.image_array,
        gaussian_filter(original * 2, (0, 0.7, 0.7), mode="mirror"),
        rtol=1e-5,
    )
    np.testing.assert_array_equal(work.session.original_image_array, original)
    assert work.session.experiment_names == ["A"]
    first = work.session.map_array("Before smoothing / t2")
    np.testing.assert_allclose(first, 80, rtol=0.02)
    np.testing.assert_allclose(work.session.map_array("After smoothing / t2"), 80, rtol=0.02)
    assert not np.allclose(
        work.session.map_array("Before smoothing / amplitude"),
        work.session.map_array("After smoothing / amplitude"),
    )
    result = fit_result_for_map(work, "Before smoothing / t2")
    assert [step["kind"] for step in result["pipeline_steps"]] == ["reconstruct"]
    assert len(work.pipelines["A"]["fits"]) == 2
    save_workspace_bundle(work, tmp_path / "chain.mrax")
    loaded = load_workspace_bundle(tmp_path / "chain.mrax", tmp_path / "unpacked", key="loaded")
    np.testing.assert_allclose(loaded.session.map_array("Before smoothing / t2"), first)
    assert fit_result_for_map(loaded, "Before smoothing / t2")["block_id"] == result["block_id"]


def test_mpip_wraps_versioned_leaf_documents_and_reuses_target_kspace():
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
    restored = loads_workflow(dumps_workflow(document), suffix=".mpip")
    assert restored.format_version == 2
    assert [step.operation for step in restored.steps] == ["reconstruct", "process", "fit"]
    assert restored.steps[0].settings["workflow"]["kind"] == "reconstruction"
    assert restored.steps[2].settings["workflow"]["kind"] == "fitting"
    target, original = workspace()
    assert validate_workflow_targets(target, restored, ["A"], {"A": bindings})[0]["ready"]
    apply_workflow(target, restored.model_dump(), ["A"], {"A": bindings}, lambda *_: None)
    np.testing.assert_allclose(target.session.image_array, work.session.image_array)
    np.testing.assert_array_equal(target.session.original_image_array, original)
    assert len(target.session.map_names) == 3


def test_nested_roi_binding_namespaces_are_portable_and_assigned_on_the_target():
    work, _ = workspace(raw=False)
    mask = np.zeros((4, 5), np.uint8)
    mask[1:3, 1:4] = 1
    work.session.add_roi(ROI("Source region", mask, dims=("x", "y")))
    fit = block_options(work, "fit", {"use_roi": True, "roi_names": ["Source region"]})
    apply(work, [row("fit", fit, "fit")])
    doc, _ = capture_workflow(work, "processing")
    assert len(doc.bindings) == 1
    target, original = workspace(raw=False)
    target.session.add_roi(ROI("Target region", mask, dims=("x", "y")))
    mapping = {doc.bindings[0].id: "Target region"}
    apply_workflow(target, doc.model_dump(), ["A"], {"A": mapping}, lambda *_: None)
    np.testing.assert_array_equal(target.session.image_array, original)
    np.testing.assert_allclose(
        target.session.map_array("Fitting workflow / t2")[mask.astype(bool)], 80, rtol=0.02
    )
    assert signature(target) == "original"
    assert not pipeline_info(target)["fits_stale"]


@pytest.mark.parametrize(
    "invalid", ["missing", "cached_only", "no_samples", "nonfinite", "nufft_on_cartesian"]
)
def test_invalid_reconstruction_blocks_are_rejected_atomically(invalid):
    work, original = workspace(raw=invalid not in {"missing", "cached_only"})
    if invalid == "cached_only":
        work.session.current_experiment["processed"] = {"raw_reconstruction": original * 2}
    if invalid in {"no_samples", "nonfinite"}:
        raw = work.session.current_experiment["acquisition"]
        if invalid == "no_samples":
            raw = replace(raw, acquired=np.zeros_like(raw.samples, dtype=bool))
        else:
            values = raw.samples.copy()
            values[0, 0, 0, 0] = np.nan
            raw = replace(raw, samples=values)
        work.session.current_experiment["acquisition"] = raw
    options = block_options(work, "reconstruct", {})
    if invalid == "nufft_on_cartesian":
        options["workflow"] = next(
            p["document"] for p in reconstruction_presets() if p["id"] == "nufft-adjoint"
        )
    before = pipeline_info(work)
    with pytest.raises(ValueError):
        apply(work, [row("reconstruct", options, "invalid")])
    np.testing.assert_array_equal(work.session.image_array, original)
    assert pipeline_info(work) == before
    if invalid != "nufft_on_cartesian":
        assert reconstruction_available(work.session)[0] is False


def test_reconstruction_must_be_unique_and_first_and_no_block_uses_image_data():
    work, original = workspace()
    reco = row("reconstruct", block_options(work, "reconstruct", {}), "reco")
    for rows in ([row("process", gaussian(), "process"), reco], [reco, {**reco, "id": "second"}]):
        with pytest.raises(ValueError, match="first"):
            apply(work, rows)
    apply(work, [reco])
    assert np.max(work.session.image_array) > np.max(original)
    change_pipeline(work, {"action": "clear"}, lambda *_: None)
    np.testing.assert_array_equal(work.session.image_array, original)
    assert pipeline_info(work)["steps"] == []


def test_fitting_failure_rolls_back_processing_and_keeps_preexisting_maps():
    work, original = workspace(raw=False)
    work.session.replace_fitted_maps({"Earlier map": np.full((4, 5), 12.0)})
    fit = block_options(work, "fit", {"use_roi": False})
    fit["workflow"]["name"] = "Stop this fit"

    def cancel(done, total, message):
        if message == "Stop this fit":
            raise InterruptedError("Independent cancellation")

    with pytest.raises(InterruptedError):
        apply(work, [row("process", gaussian(), "smooth"), row("fit", fit, "fit")], cancel)
    np.testing.assert_array_equal(work.session.image_array, original)
    np.testing.assert_array_equal(work.session.map_array("Earlier map"), 12)
    assert not work.pipelines


def test_format_versions_and_embedded_document_kinds_are_strict():
    work, _ = workspace()
    apply(work, [row("fit", block_options(work, "fit", {"use_roi": False}), "fit")])
    document, _ = capture_workflow(work, "processing")
    malformed = document.model_dump()
    malformed["format_version"] = 1
    with pytest.raises(ValueError):
        WorkflowDocument.model_validate(malformed)
    malformed = document.model_dump()
    malformed["steps"][0]["settings"]["workflow"] = document.model_dump()
    with pytest.raises(ValueError, match="kind"):
        WorkflowDocument.model_validate(malformed)


def test_copied_and_reinserted_fit_blocks_do_not_overwrite_earlier_map_sets():
    work, _ = workspace(raw=False)
    fit = block_options(work, "fit", {"use_roi": False})
    apply(work, [row("fit", fit, "one"), row("fit", fit, "two")])
    assert len(work.session.map_names) == 6
    assert len({step["map_prefix"] for step in work.pipelines["A"]["steps"]}) == 2
    previous_maps = set(work.session.map_names)
    change_pipeline(work, {"action": "clear"}, lambda *_: None)
    apply(work, [row("fit", fit, "three")])
    assert previous_maps < set(work.session.map_names)
    assert len(work.session.map_names) == 9
    stored = pipeline_info(work)["steps"][0]
    renamed = copy.deepcopy(stored["options"])
    renamed["workflow"]["name"] = "Renamed fit"
    apply(
        work,
        [
            {
                "id": stored["id"],
                "source_id": stored["id"],
                "source_version": stored["version"],
                "kind": "fit",
                "enabled": True,
                "options": renamed,
            }
        ],
    )
    assert "Renamed fit / t2" in work.session.map_names
    assert "Fitting workflow (3) / t2" not in work.session.map_names
    assert previous_maps < set(work.session.map_names)


def test_incompatible_reconstruction_geometry_is_transactional_and_duplicate_append_is_rejected():
    work, original = workspace()
    reco = block_options(work, "reconstruct", {})
    geometry = next(
        stage
        for stage in reco["workflow"]["steps"][0]["settings"]["pipeline"]["stages"]
        if stage["kind"] == "format"
    )
    geometry["options"]["output_shape"] = [3, 3]
    with pytest.raises(ValueError):
        apply(work, [row("reconstruct", reco, "crop")])
    np.testing.assert_array_equal(work.session.image_array, original)
    assert not work.pipelines
    apply(work, [row("reconstruct", block_options(work, "reconstruct", {}), "reco")])
    document, bindings = capture_workflow(work, "processing")
    check = validate_workflow_targets(work, document, ["A"], {"A": bindings}, mode="append")
    assert not check[0]["ready"]
    before = pipeline_info(work)
    with pytest.raises(ValueError, match="first"):
        apply_workflow(
            work, document.model_dump(), ["A"], {"A": bindings}, lambda *_: None, mode="append"
        )
    assert pipeline_info(work) == before
