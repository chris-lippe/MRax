"""Reusable recipes tested only with independently generated numerical data."""

import copy
import json

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter, median_filter

from mrax._version import MRAX_VERSION
from mrax.data_struct import Quantity
from mrax.gui.services.reusable_workflows import (
    apply_workflow,
    capture_workflow,
    resolve_settings,
    resource_choices,
    validate_workflow_targets,
)
from mrax.gui.services.synthetic import synthetic_t2_session
from mrax.gui.services.workspace import StudyWorkspace
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


def test_bundled_presets_adapt_to_spatial_axes_and_open_as_independent_session_copies(tmp_path):
    from fastapi.testclient import TestClient

    from mrax.web.app import create_app
    from mrax.workflow_presets import workflow_presets

    presets = {row["id"]: row for row in workflow_presets()}
    assert {row["document"]["kind"] for row in presets.values()} == {
        "denoising",
        "registration",
        "fitting",
        "reconstruction",
        "processing",
    }
    for row in presets.values():
        document = loads_workflow(json.dumps(row["document"]))
        assert document == loads_workflow(dumps_workflow(document))
        assert row["requirements"]

    workspace, image = phantom(second=True)
    smooth = WorkflowDocument.model_validate(presets["spatial-gaussian"]["document"])
    apply(workspace, smooth, ["A", "B"])
    expected = gaussian_filter(image, sigma=(0, 0.6, 0.6), mode="mirror")
    np.testing.assert_allclose(workspace.session.image_array, expected, rtol=1e-5)
    np.testing.assert_array_equal(workspace.session.original_image_array, image)
    apply(workspace, WorkflowDocument.model_validate(presets["spatial-median"]["document"]))
    np.testing.assert_allclose(
        workspace.session.image_array, median_filter(image, size=(1, 3, 3), mode="mirror")
    )
    volume = StudyWorkspace(key="volume", label="Volume", session=synthetic_t2_session())
    source = volume.session.image_array.copy()
    apply(volume, smooth)
    np.testing.assert_allclose(
        volume.session.image_array,
        gaussian_filter(source, sigma=(0, 0.6, 0.6, 0.6), mode="mirror"),
        rtol=1e-5,
    )
    pair = WorkflowDocument.model_validate(presets["align-similar-contrast"]["document"])
    checks = validate_workflow_targets(workspace, pair, ["A"], {"A": {"reference": "A"}})
    assert not checks[0]["ready"] and "different reference" in checks[0]["errors"][0]
    workspace.session.set_active_frame(1)
    pair.steps[0].settings.pop("moving_frame")
    assert validate_workflow_targets(workspace, pair, ["A"], {"A": {"reference": "A"}})[0]["ready"]
    still = StudyWorkspace(key="still", label="Still", session=GuiSession())
    still.session.create_empty_study("Independent static image", field_strength_mhz=100)
    still.session.add_experiment("Image", image[0], dims={"x": 9, "y": 11})
    motion = WorkflowDocument.model_validate(presets["align-repeated-frames"]["document"])
    checks = validate_workflow_targets(still, motion, ["Image"], {})
    assert not checks[0]["ready"] and "two acquisition frames" in checks[0]["errors"][0]
    fitting = WorkflowDocument.model_validate(presets["fit-t2"]["document"])
    checks = validate_workflow_targets(still, fitting, ["Image"], {})
    assert not checks[0]["ready"] and "Missing observable: TE" in checks[0]["errors"][0]

    with TestClient(create_app(runtime_dir=tmp_path), base_url="http://localhost") as client:
        headers = {"x-mrax-token": client.get("/api/session").json()["token"]}
        before = client.get("/api/workflow-presets").json()
        assert not client.get("/api/workflows").json()
        first = client.post(
            "/api/workflow-presets/spatial-gaussian/open", headers=headers, json={}
        ).json()
        second = client.post(
            "/api/workflow-presets/spatial-gaussian/open", headers=headers, json={}
        ).json()
        assert first["id"] != second["id"]
        changed = copy.deepcopy(first["document"])
        changed["steps"][0]["settings"]["parameters"]["sigma"] = 2
        response = client.put(
            "/api/workflows/" + first["id"],
            headers=headers,
            json={"revision": first["revision"], "document": changed},
        )
        assert response.status_code == 200
        assert client.get("/api/workflow-presets").json() == before
        assert client.get("/api/workflows").json()[1]["document"] == second["document"]
        assert (
            client.post(
                "/api/workflow-presets/unavailable/open", headers=headers, json={}
            ).status_code
            == 404
        )


def test_utf8_roundtrip_required_versions_extensions_and_atomic_files(tmp_path, monkeypatch):
    document = processing_document()
    values = document.model_dump()
    values["name"] = "Denoising α"
    values["extensions"] = {"org.example.review": {"notes": "Keep for later"}}
    document = WorkflowDocument.model_validate(values)
    path = tmp_path / "Unicode ä α" / "denoise.mflow"
    path.parent.mkdir()
    save_workflow(document, path)
    assert path.read_text().startswith('{\n  "format": "mrax.workflow",')
    assert load_workflow(path) == document
    assert loads_workflow(dumps_workflow(document).encode("utf-8")) == document
    assert document.mrax_version == MRAX_VERSION
    major, minor, _patch = MRAX_VERSION.split(".", 2)
    compatible = f"{major}.{minor}.9999"
    assert (
        loads_workflow(json.dumps({**values, "mrax_version": compatible})).mrax_version
        == compatible
    )
    import mrax.workflows as workflows

    with monkeypatch.context() as scope:
        scope.setattr(workflows, "MRAX_VERSION", compatible)
        assert processing_document().mrax_version == compatible
        assert json.loads(dumps_workflow(document))["mrax_version"] == compatible
    assert document.mrax_version == MRAX_VERSION
    with pytest.raises((ValueError, FileExistsError)):
        save_workflow(document, path)
    assert load_workflow(path) == document
    save_workflow(document, path, overwrite=True)
    with pytest.raises(ValueError, match="extension"):
        save_workflow(document, path.with_suffix(".json"))
    with pytest.raises(ValueError, match="extension"):
        loads_workflow(dumps_workflow(document), suffix=".json")
    with pytest.raises(ValueError, match="version"):
        loads_workflow(json.dumps({**values, "mrax_version": "9999.0.0"}))
    invalid = copy.deepcopy(values)
    invalid["steps"][0]["operation"] = "python"
    with pytest.raises(ValueError, match="operation"):
        loads_workflow(json.dumps(invalid))


def test_fitting_resources_rebind_across_studies(tmp_path):
    source, _ = phantom(second=True)
    source.session.select_experiment("B")
    maps = {
        "relaxation": Quantity(np.full((9, 11), 1200), "ms"),
        "transmit": Quantity(np.full((9, 11), 100), "%"),
        "frequency": Quantity(np.zeros((9, 11)), "ppm"),
    }
    source.session.replace_fitted_maps(maps)
    source.session.select_experiment("A")
    settings = source.session.fit_settings
    settings.model_name = "ab_mt_n"
    settings.sat_trans_reference_scan = "B"
    settings.ab_mt_t1_map_ref = "B::relaxation"
    settings.ab_mt_b1_map_ref = "B::transmit"
    settings.ab_mt_b0_map_ref = "B::frequency"
    document, suggested = capture_workflow(source, "fitting", draft={"use_roi": False})
    assert {binding.kind for binding in document.bindings} == {"experiment", "image", "map"}
    assert set(document.steps[0].settings["inputs"]) == {"reference", "t1", "b1", "b0"}
    payload = document.steps[0].settings["settings"]
    assert (
        not {key for key in vars(settings) if key.endswith("_ref") or key.endswith("_scan")}
        & payload.keys()
    )
    path = tmp_path / "resources.mflow"
    save_workflow(document, path)
    document = load_workflow(path)
    assert "B::" not in path.read_text()

    image = source.session.image_array.copy()
    session = GuiSession()
    session.create_empty_study("Independent workflow target", field_strength_mhz=400)
    session.add_experiment(
        "A",
        image,
        dims={"frame": 4, "x": 9, "y": 11},
        exp_type="sat_trans",
        model_name="ab_mt_n",
        observables={
            "freq_offsets": Quantity(np.array([-3, -1, 1, 3]), "ppm"),
            "sat_time": Quantity(100, "ms"),
            "sat_b1": Quantity(1, "uT"),
            "TR": Quantity(2000, "ms"),
        },
    )
    session.add_experiment(
        "Priors",
        image[:1],
        dims={"frame": 1, "x": 9, "y": 11},
        exp_type="sat_trans",
    )
    names = {"relaxation": "new_t1", "transmit": "new_b1", "frequency": "new_b0"}
    session.replace_fitted_maps({names[key]: value for key, value in maps.items()})
    session.select_experiment("A")
    target = StudyWorkspace(key="target", label=session.study_name, session=session)
    assignments = {
        binding.id: "Priors"
        if binding.kind in {"experiment", "image"}
        else names[suggested[binding.id]]
        for binding in document.bindings
    }
    assert all(
        set(row["resources"]) == {"experiment", "image", "roi", "map"}
        for row in resource_choices(target)
    )
    resolved = resolve_settings(document, target, assignments)[0][1]["settings"]
    assert resolved["sat_trans_reference_scan"] == "Priors"
    assert resolved["ab_mt_t1_map_ref"] == "Priors::new_t1"
    assert resolved["ab_mt_b1_map_ref"] == "Priors::new_b1"
    assert resolved["ab_mt_b0_map_ref"] == "Priors::new_b0"
    apply(target, document, bindings={"A": assignments}, run=False)
    experiment, _model, _dataset, _study = target.session.build_fit_inputs()
    assert {"t1_map", "b1_map", "b0_map"} <= experiment.observables.keys()

    values = document.model_dump()
    block = values["steps"][0]
    reference = block["settings"]["inputs"]["reference"]["$binding"]
    values["bindings"] = [b for b in values["bindings"] if b["id"] != reference]
    block["settings"]["inputs"]["reference"] = block["settings"]["inputs"]["t1"]
    invalid = WorkflowDocument.model_validate(values)
    with pytest.raises(ValueError, match="Reference scan requires"):
        resolve_settings(invalid, target, {k: v for k, v in assignments.items() if k != reference})
    np.testing.assert_array_equal(target.session.image_array, image)
