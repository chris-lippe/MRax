"""Reusable recipes tested only with independently generated numerical data."""

import copy
import json

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from mrax.analysis import ROI
from mrax.data_struct import Quantity
from mrax.gui.services.pipeline import pipeline_info
from mrax.gui.services.reusable_workflows import (
    apply_workflow,
    capture_workflow,
)
from mrax.gui.services.synthetic import synthetic_multimodal_session, synthetic_t2_session
from mrax.gui.services.workspace import StudyWorkspace
from mrax.gui.services.workspace_io import load_workspace_bundle, save_workspace_bundle
from mrax.gui.session import GuiSession
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
    assert [row["kind"] for row in pipeline_info(workspace)["steps"]] == ["process"]
    np.testing.assert_allclose(
        workspace.session.image_array,
        gaussian_filter(original, (0, 1, 1, 0), mode="mirror"),
        rtol=1e-5,
        atol=1e-6,
    )
    values = document.model_dump()
    values["steps"][0]["settings"]["component"] = "phase"
    apply(workspace, WorkflowDocument.model_validate(values))
    assert [row["kind"] for row in pipeline_info(workspace)["steps"]] == ["reconstruct", "process"]
    assert np.all(np.isfinite(workspace.session.image_array))
    np.testing.assert_array_equal(workspace.session.original_image_array, original)
    no_raw, _ = phantom()
    with pytest.raises(ValueError, match="k-space"):
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
