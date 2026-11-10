"""Editable reconstruction plans on independently generated acquisitions."""

import copy
from dataclasses import replace

import numpy as np
import pytest

from mrax.data_struct.acquisition import AcquisitionData, ReconstructionRecipe
from mrax.gui.services.analysis_workflows import run_operation
from mrax.gui.services.reconstruction import reconstruction_catalog, reconstruction_label
from mrax.gui.services.workspace import StudyWorkspace
from mrax.gui.session import GuiSession
from mrax.reco import fftnc, reconstruct_acquisition
from mrax.reco.plans import compile_plan, normalize_plan, reconstruction_presets
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


@pytest.mark.parametrize("sampled,name", [(False, "cartesian-fourier"), (True, "nufft-adjoint")])
def test_fft_and_nufft_presets_reproduce_known_complex_images_without_changing_samples(
    sampled, name
):
    raw, truth = acquisition(sampled=sampled)
    original = raw.samples.copy()
    result = compile_plan(raw, preset(name)).run(raw)
    np.testing.assert_allclose(result.coil_images, truth, atol=2e-4, rtol=2e-4)
    np.testing.assert_allclose(result.image, truth.sum(axis=1), atol=3e-4, rtol=3e-4)
    np.testing.assert_array_equal(raw.samples, original)
    assert result.method == ("nufft" if sampled else "fft")


def test_acquisition_defaults_match_existing_fourier_reconstruction_with_geometry_and_shift():
    raw, _ = acquisition()
    raw = replace(
        raw,
        recipe=replace(raw.recipe, image_shift=(0.25, -0.5), output_scale=2.5, output_flips=(1,)),
    )
    result = compile_plan(raw, preset("acquisition-fourier")).run(raw)
    expected = reconstruct_acquisition(raw)
    np.testing.assert_allclose(result.coil_images, expected.coil_images, atol=1e-6)


@pytest.mark.parametrize("compensate", [False, True])
def test_nufft_nonuniform_samples_match_direct_fourier_sum_and_density_weights(compensate):
    rng = np.random.default_rng(82)
    shape = (6, 8)
    trajectory = rng.uniform(-0.49, 0.49, (37, 2)).astype(np.float32)
    samples = (rng.normal(size=(2, 2, 37)) + 1j * rng.normal(size=(2, 2, 37))).astype(np.complex64)
    density = np.linspace(0.2, 1.3, 37, dtype=np.float32)
    raw = AcquisitionData(
        samples,
        shape,
        ("x", "y"),
        kind="sampled",
        trajectory=trajectory,
        density=density,
        recipe=ReconstructionRecipe(coil_combine="sum"),
    )
    settings = preset("nufft-adjoint")
    next(s for s in settings["stages"] if s["kind"] == "nufft")["options"][
        "density_compensation"
    ] = compensate
    coordinates = np.stack(
        np.meshgrid(np.arange(6) - 3, np.arange(8) - 4, indexing="ij"), -1
    ).reshape(-1, 2)
    adjoint = np.exp(2j * np.pi * (coordinates @ trajectory.T)) / np.sqrt(np.prod(shape))
    expected = np.einsum("nm,fcm->fcn", adjoint, samples * (density if compensate else 1)).reshape(
        (2, 2) + shape
    )
    result = compile_plan(raw, settings).run(raw)
    np.testing.assert_allclose(result.coil_images, expected, atol=3e-5, rtol=3e-5)
    np.testing.assert_array_equal(raw.samples, samples)
    np.testing.assert_array_equal(raw.density, density)


def test_gridded_fourier_applies_inserted_windows_and_does_not_change_acquisition_samples():
    raw, _ = acquisition(sampled=True)
    original = raw.samples.copy()
    settings = preset("gridded-fourier")
    before = compile_plan(raw, settings).run(raw).image
    next(s for s in settings["stages"] if s["kind"] == "window")["options"] = {
        "kspace_window": ["hann", "hann"]
    }
    after = compile_plan(raw, settings).run(raw).image
    assert not np.allclose(before, after)
    assert np.linalg.norm(after) < np.linalg.norm(before)
    np.testing.assert_array_equal(raw.samples, original)


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


@pytest.mark.parametrize("component", ["magnitude", "phase"])
def test_pipeline_output_uses_the_configured_coil_combination_independently_of_view_settings(
    component,
):
    raw, truth = acquisition()
    work = workspace(raw)
    work.session.active_reconstruction_coil = 0
    work.session.active_reconstruction_unwrap_phase = True
    run_operation(
        work,
        "reconstruct",
        {"pipeline": preset("cartesian-fourier"), "component": component},
        lambda *_: None,
    )
    combined = truth.sum(axis=1)
    expected = np.abs(combined) if component == "magnitude" else np.angle(combined)
    np.testing.assert_allclose(work.session.image_array, expected, rtol=1e-5, atol=1e-5)


def test_mrec_and_workspace_roundtrip_retain_the_plan_and_cached_source_settings(
    tmp_path, monkeypatch
):
    from mrax.gui.services.persistence import load_workspace_bundle, save_workspace_bundle
    from mrax.gui.services.pipeline import change_pipeline
    from mrax.gui.services.reusable_workflows import capture_workflow, validate_workflow_targets

    raw, _ = acquisition()
    work = workspace(raw)
    plan = preset("cartesian-pocs")
    plan["name"] = "Reusable POCS"
    run_operation(
        work, "reconstruct", {"pipeline": plan, "component": "magnitude"}, lambda *_: None
    )
    document, bindings = capture_workflow(work, "reconstruction")
    restored_document = loads_workflow(dumps_workflow(document))
    assert restored_document.steps[0].settings["pipeline"] == plan
    validate_workflow_targets(work, restored_document, ["Acquisition"], {"Acquisition": bindings})
    save_workspace_bundle(work, tmp_path / "reconstruction.mrax")
    restored = load_workspace_bundle(
        tmp_path / "reconstruction.mrax", tmp_path / "unpacked", key="restored"
    )
    assert reconstruction_catalog(restored)["plan"] == plan
    step = restored.pipelines["Acquisition"]["steps"][0]
    change_pipeline(restored, {"action": "remove", "step_id": step["id"]}, lambda *_: None)
    assert reconstruction_catalog(restored)["plan"] == plan

    def should_not_recompute(*_args, **_kwargs):
        raise AssertionError("Switching back to a confirmed reconstruction should use its cache")

    monkeypatch.setattr(GuiSession, "reconstruct_raw_data_for_experiment", should_not_recompute)
    run_operation(restored, "reconstruct", {"component": "magnitude"}, lambda *_: None)
    assert restored.pipelines["Acquisition"]["steps"][0]["options"]["workflow"]["steps"][0]["settings"]["pipeline"] == plan


def test_cached_images_do_not_prevent_loading_raw_samples_for_an_edited_plan(tmp_path, monkeypatch):
    raw, _ = acquisition()
    work = workspace(raw)
    session = work.session
    session.source_kind = "bruker"
    session.current_experiment.pop("acquisition", None)
    session.current_experiment.pop("raw_kspace", None)
    session.current_experiment["processed"] = {"raw_reconstruction": np.ones((2, 6, 8))}
    session.current_experiment["extra_metadata"]["path"] = str(tmp_path)
    session.current_experiment["kspace_metadata"] = {"status": "loaded"}
    assert session.raw_data_loaded()
    assert session.can_lazy_load_raw_data(require_acquisition=True)
    calls = []

    def load(self, *, progress_callback=None, require_acquisition=False):
        calls.append(require_acquisition)
        self.current_experiment["acquisition"] = raw
        self.current_experiment["raw_kspace"] = raw.samples
        return True

    monkeypatch.setattr(GuiSession, "load_raw_data_for_experiment", load)
    run_operation(
        work,
        "reconstruct",
        {"pipeline": preset("cartesian-fourier"), "component": "magnitude"},
        lambda *_: None,
    )
    assert calls == [True]


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


@pytest.mark.parametrize(
    "change",
    [
        lambda p: p.update(schema_version=2),
        lambda p: p.update(encoding=[]),
        lambda p: p["stages"].append(copy.deepcopy(p["stages"][0])),
        lambda p: p["stages"].reverse(),
        lambda p: p["stages"].append({"kind": "python", "options": {}}),
        lambda p: p["stages"].__setitem__(0, {"kind": "fft", "options": {"code": "no"}}),
        lambda p: p["stages"].__setitem__(0, {"kind": "fft", "options": {"frame_batch_size": 1.5}}),
        lambda p: p["stages"].__setitem__(0, {"kind": []}),
        lambda p: p["stages"][0].update(when=[]),
        lambda p: p["stages"].__setitem__(
            0, {"kind": "variational", "options": {"num_steps": True}}
        ),
        lambda p: p["stages"].__setitem__(
            0, {"kind": "format", "options": {"output_scale": "invalid"}}
        ),
    ],
)
def test_reconstruction_schema_rejects_unknown_unsafe_or_ambiguous_stages(change):
    value = preset("cartesian-fourier")
    change(value)
    with pytest.raises(ValueError):
        normalize_plan(value)


def test_catalog_reports_current_encoding_presets_and_editable_solver_settings():
    work = workspace(acquisition()[0])
    catalog = reconstruction_catalog(work)
    assert catalog["encoding"] == "cartesian"
    assert next(p for p in catalog["presets"] if p["id"] == "nufft-adjoint")["compatible"] is False
    fields = next(s for s in catalog["stages"] if s["kind"] == "variational")["fields"]
    assert {"regularizer", "regularizer_weight", "num_steps"} <= {f["key"] for f in fields}
    readout = next(s for s in catalog["stages"] if s["kind"] == "readout_regrid")
    for choice in next(f for f in readout["fields"] if f["key"] == "readout_interpolation")[
        "choices"
    ]:
        replace(ReconstructionRecipe(), readout_interpolation=choice)


def test_partial_stage_options_inherit_calibration_and_validate_against_the_target_acquisition():
    raw, _ = acquisition()
    calibrated = replace(
        raw, recipe=replace(raw.recipe, echo_spacing_seconds=0.003, kspace_window=("hann", "hann"))
    )
    settings = preset("cartesian-fourier")
    next(s for s in settings["stages"] if s["kind"] == "product")["options"] = {
        "output_product": "frequency_hz"
    }
    next(s for s in settings["stages"] if s["kind"] == "window")["options"] = {
        "window_parameters": [2.0, 2.0]
    }
    plan = compile_plan(calibrated, settings)
    product = next(s for s in plan.stages if s.kind == "product")
    assert product.options["echo_spacing_seconds"] == 0.003
    assert next(s for s in plan.stages if s.kind == "window").options["kspace_window"] == (
        "hann",
        "hann",
    )
    with pytest.raises(ValueError, match="echo spacing"):
        compile_plan(
            replace(calibrated, recipe=replace(calibrated.recipe, echo_spacing_seconds=None)),
            settings,
        )
