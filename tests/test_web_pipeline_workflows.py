"""Independent synthetic data for non-destructive experiment pipelines."""

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter, shift

from mrax.analysis import ROI
from mrax.gui.services.pipeline import (
    change_pipeline,
    pipeline_info,
)
from mrax.gui.services.workspace import StudyWorkspace
from mrax.gui.session import GuiSession


def test_series_alignment_preserves_masks_dimensions_and_bundle_state(tmp_path):
    pytest.importorskip("SimpleITK")
    from mrax.gui.services.analysis_workflows import run_alignment, run_operation
    from mrax.gui.services.persistence import load_workspace_bundle, save_workspace_bundle

    x, y = np.indices((28, 32))
    image = (
        np.exp(-((x - 11) ** 2 / 17 + (y - 17) ** 2 / 29))
        + 0.4 * np.exp(-((x - 20) ** 2 / 8 + (y - 8) ** 2 / 11))
    ).astype(np.float32)
    source = np.stack(
        [
            np.stack([shift(image * scale, (t, -0.5 * t), order=1) for scale in (1.0, 0.2)])
            for t in range(3)
        ]
    )
    session = GuiSession()
    session.create_empty_study("Independent synthetic series")
    session.add_experiment(
        "Series",
        source,
        dims={"t": 3, "offset": 2, "x": 28, "y": 32},
        extra_metadata={
            "spatial_geometry": {
                "affine": np.eye(4).tolist(),
                "spatial_shape": [28, 32],
                "axis_anatomical_labels": ["x", "y"],
            }
        },
    )
    assert session.image_plane_dims == ("y", "x")
    masks = np.zeros(source.shape, dtype=bool)
    masks[:, :, 8:20, 8:22] = True
    masks[2, 1] = False
    masks[2, 1, 10:18, 10:20] = True
    session.add_roi(ROI("Region", masks, dims=("t", "offset", "x", "y")))
    workspace = StudyWorkspace(key="synthetic", label=session.study_name, session=session)

    def progress(*_):
        return None

    result = run_alignment(
        workspace,
        {
            "reference_frame": 5,
            "moving_frame": 0,
            "reference_roi": "Region",
            "moving_roi": "Region",
            "manual": True,
        },
        progress,
    )
    assert result["diagnostics"]["reference_mask_pixels"] == 80
    assert result["diagnostics"]["moving_mask_pixels"] == 168
    assert session.active_frame == 0
    from mrax.gui.services.analysis_workflows import transform_roi
    from mrax.gui.services.scene import render_overlay, scene_settings

    transform_roi(
        workspace, {"transform": result["key"], "name": "Region", "new_name": "Transferred"}
    )
    np.testing.assert_array_equal(session.roi("Transferred").mask, masks[0, 0])
    run_operation(
        workspace,
        "register",
        {
            "reference_frame": 0,
            "moving_frame": 0,
            "manual": True,
            "manual_settings": {"shift_x": 2},
        },
        progress,
    )
    np.testing.assert_allclose(
        workspace.session.image_array,
        shift(source, (0, 0, -2, 0), order=1, mode="constant"),
        atol=1e-7,
    )
    layer = [
        layer for layer in scene_settings(workspace)["layers"] if layer["kind"] == "registration"
    ][-1]
    np.testing.assert_allclose(render_overlay(workspace, layer), workspace.session.frame_image())
    change_pipeline(workspace, {"action": "clear"}, progress)
    session = workspace.session
    volume = np.stack([image * (1 + 0.1 * np.cos(z)) for z in range(8)], axis=-1)
    session.add_experiment(
        "Volume", np.stack([volume, volume]), dims={"frame": 2, "x": 28, "y": 32, "z": 8}
    )
    volume_masks = np.zeros((2, 28, 32, 8), dtype=bool)
    volume_masks[0, 8:20, 8:22, 2:6] = True
    volume_masks[1, 10:18, 10:20, 2:6] = True
    session.add_roi(ROI("Volume region", volume_masks, dims=("frame", "x", "y", "z")))
    aligned = run_alignment(
        workspace,
        {
            "reference_frame": 0,
            "moving_frame": 1,
            "scope": "volume",
            "reference_roi": "Volume region",
            "moving_roi": "Volume region",
            "settings": {"transform": "translation", "metric": "mean_squares"},
        },
        progress,
    )
    assert aligned["diagnostics"]["reference_mask_pixels"] == 672
    assert aligned["diagnostics"]["moving_mask_pixels"] == 320
    session.select_experiment("Series")
    session.set_active_frame(5)
    _, result = run_operation(
        workspace,
        "_drift_step",
        {
            "roi_name": "Region",
            "settings": {
                "dimensions": ["t"],
                "reference": "current",
                "num_steps": 10,
                "interpolation": "nearest",
            },
        },
        progress,
    )
    assert [group["reference_index"] for group in result["summary"]["groups"]] == [2, 2]
    session.rois = [ROI("Region", masks[0, 0].copy(), dims=("x", "y"))]
    session._save_current_view_state()
    run_operation(
        workspace,
        "drift",
        {
            "roi_name": "Region",
            "settings": {
                "method": "framewise",
                "dimensions": ["t"],
                "estimation_dimensions": ["offset"],
                "reference": "first",
                "registration": {
                    "transform": "translation",
                    "metric": "mean_squares",
                    "iterations": 120,
                },
            },
        },
        progress,
    )
    corrected = np.asarray(workspace.session.image_array)
    assert corrected.shape == source.shape
    assert np.mean((corrected[-1, 0] - image) ** 2) < np.mean((source[-1, 0] - image) ** 2) / 5
    np.testing.assert_array_equal(workspace.session.original_image_array, source)
    assert pipeline_info(workspace)["steps"][0]["label"] == "Series alignment"
    saved = save_workspace_bundle(workspace, tmp_path / "series.mrax")
    restored = load_workspace_bundle(saved, tmp_path / "restored", key="restored")
    np.testing.assert_allclose(restored.session.image_array, corrected)
    np.testing.assert_array_equal(restored.session.original_image_array, source)
    assert any(row["kind"] == "mrax_framewise_motion" for row in restored.drift_results.values())
    assert pipeline_info(restored)["steps"][0]["options"]["settings"]["estimation_dimensions"] == [
        "offset"
    ]


def test_cross_experiment_labels_and_reference_normalization_share_the_registered_grid(tmp_path):
    pytest.importorskip("SimpleITK")
    from scipy.ndimage import map_coordinates

    from mrax.analysis.saturation_transfer import get_z_spectra
    from mrax.data_struct import Quantity
    from mrax.gui.services.analysis_workflows import run_operation, transform_roi
    from mrax.gui.services.persistence import load_workspace_bundle, save_workspace_bundle
    from mrax.gui.services.pipeline import project_roi
    from mrax.gui.services.scene import fitting_roi, scene_settings

    def signal(x, y):
        return (
            100
            + 1800 * np.exp(-((x - 13) ** 2 / 18 + (y - 9) ** 2 / 22))
            + 1200 * np.exp(-((x - 26) ** 2 / 12 + (y - 22) ** 2 / 16))
        ).astype(np.float32)

    x, y = np.indices((32, 40))
    anatomy = signal(y, x)
    labels = np.zeros(anatomy.shape, dtype=np.uint16)
    labels[(x - 9) ** 2 + (y - 13) ** 2 < 12] = 1
    labels[(x - 22) ** 2 + (y - 26) ** 2 < 10] = 2
    a = np.eye(4)
    b = np.array([[0, 1.2, 0, 0], [-1.4, 0, 0, 35], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
    u, v = np.indices((24, 28))
    baseline = signal(1.2 * u, 35 - 1.4 * v)
    spectra = np.broadcast_to(
        np.array([0.8, 0.4, 0.6], dtype=np.float32)[None, :, None, None] * baseline[None, None],
        (2, 3, 24, 28),
    ).copy()

    def metadata(affine, shape, axes):
        return {
            "spatial_geometry": {
                "affine": affine.tolist(),
                "spatial_shape": list(shape),
                "axis_anatomical_labels": axes,
            }
        }

    session = GuiSession()
    session.create_empty_study("Independent synthetic multimodal registration")
    session.add_experiment(
        "Anatomy",
        np.stack([anatomy, anatomy * 0.7]),
        dims={"echo": 2, "x": 32, "y": 40},
        extra_metadata=metadata(a, anatomy.shape, ["y", "x"]),
    )
    session.replace_fitted_maps({"t2": Quantity(np.sqrt(anatomy), "s")})
    session.add_roi(
        ROI(
            "Labels",
            labels,
            dims=("x", "y"),
            metadata={
                "label_names": {"1": "First", "2": "Second"},
                "label_colors": {"1": "#b04444"},
            },
        )
    )
    session.add_experiment(
        "Spectra",
        spectra,
        dims={"t": 2, "offset": 3, "x": 24, "y": 28},
        exp_type="sat_trans",
        extra_metadata=metadata(b, baseline.shape, ["x", "y"]),
    )
    session.add_experiment(
        "Reference",
        baseline[None],
        dims={"frame": 1, "x": 24, "y": 28},
        exp_type="sat_trans",
        extra_metadata=metadata(b, baseline.shape, ["x", "y"]),
    )
    workspace = StudyWorkspace(key="synthetic", label=session.study_name, session=session)
    source_style = {
        "color": "#22aaff",
        "opacity": 0.4,
        "hidden_labels": [2],
        "label_colors": {"1": "#bb2244"},
        "label_opacities": {"1": 0.7},
    }
    scene = scene_settings(workspace, "Anatomy")
    scene["roi_styles"]["Labels"] = source_style
    workspace.display_settings.setdefault("Anatomy", {})["scene"] = scene
    run_operation(
        workspace,
        "register",
        {
            "reference_experiment": "Anatomy",
            "reference_source": "map",
            "reference_map": "t2",
            "moving_experiment": "Spectra",
            "moving_aggregation": "mean",
            "manual": True,
        },
        lambda *_: None,
    )
    session = workspace.session
    np.testing.assert_array_equal(session.original_image_array, spectra)
    assert session.experiment_dims == {"t": 2, "offset": 3, "x": 32, "y": 40}
    assert session.enable_shared_roi("Labels")
    np.testing.assert_array_equal(project_roi(workspace, session.roi("Labels")).mask, labels)
    np.testing.assert_array_equal(fitting_roi(workspace, ["Labels"]).mask, labels != 0)
    key = workspace.pipelines["Spectra"]["steps"][0]["result_key"]
    transform_roi(
        workspace, {"transform": key, "direction": "reference_to_moving", "name": "Labels"}
    )
    transferred = session.roi()
    assert transferred.mask.shape == baseline.shape and transferred.mask.dtype == labels.dtype
    np.testing.assert_array_equal(np.unique(transferred.mask), [0, 1, 2])
    assert transferred.metadata["label_names"] == {"1": "First", "2": "Second"}
    assert scene_settings(workspace)["roi_styles"][transferred.name] == source_style
    projected = project_roi(workspace, transferred).mask
    assert projected.shape == labels.shape
    expected_native = map_coordinates(
        labels, [35 - 1.4 * v, 1.2 * u], order=0, mode="grid-constant"
    )
    np.testing.assert_array_equal(transferred.mask, expected_native)
    expected_projection = map_coordinates(
        expected_native, [y / 1.2, (35 - x) / 1.4], order=0, mode="grid-constant"
    )
    np.testing.assert_array_equal(projected, expected_projection)
    from mrax.gui.services.export import (
        _export_object_records_for_session,
        _write_single_export_object,
    )
    from mrax.io.nifti import load_nifti

    record = next(
        value
        for value in _export_object_records_for_session(workspace.session)
        if value["kind"] == "roi" and value["name"] == transferred.name
    )
    exported = _write_single_export_object(record, tmp_path / "transferred.nii", "nifti")[0]
    exported_mask, exported_affine = load_nifti(exported)
    np.testing.assert_array_equal(exported_mask, transferred.mask)
    np.testing.assert_allclose(exported_affine, b, atol=1e-6)
    before_filter = np.asarray(workspace.session.image_array).copy()
    run_operation(
        workspace,
        "process",
        {
            "method": "gaussian",
            "input": "image",
            "axes": ["x", "y"],
            "parameters": {"sigma": 0.6},
            "use_roi": True,
            "roi_names": ["Labels"],
        },
        lambda *_: None,
    )
    expected_filter = np.where(
        labels != 0,
        gaussian_filter(before_filter, sigma=(0, 0, 0.6, 0.6), mode="mirror"),
        before_filter,
    )
    np.testing.assert_allclose(workspace.session.image_array, expected_filter, rtol=1e-6, atol=1e-5)
    step = pipeline_info(workspace)["steps"][-1]["id"]
    change_pipeline(workspace, {"action": "remove", "step_id": step}, lambda *_: None)
    session = workspace.session
    np.testing.assert_allclose(session.image_array, before_filter)
    with pytest.raises(ValueError, match="Register"):
        get_z_spectra(session.fit_study(), "Spectra", reference="Reference")
    run_operation(
        workspace,
        "register",
        {
            "reference_experiment": "Spectra",
            "reference_aggregation": "mean",
            "moving_experiment": "Reference",
            "manual": True,
        },
        lambda *_: None,
    )
    workspace.session.select_experiment("Spectra")
    study = workspace.session.fit_study()
    normalized = get_z_spectra(study, "Spectra", reference="Reference")
    valid = np.asarray(normalized.reference_image) > 1
    expected = np.broadcast_to(
        np.array([0.8, 0.4, 0.6])[None, :, None], (2, 3, np.count_nonzero(valid))
    )
    np.testing.assert_allclose(np.asarray(normalized.values)[:, :, valid], expected, rtol=2e-6)
    np.testing.assert_array_equal(workspace.session.original_image_array, spectra)
    saved = save_workspace_bundle(workspace, tmp_path / "cross-experiment.mrax")
    restored = load_workspace_bundle(saved, tmp_path / "cross-restored", key="restored")
    actual = get_z_spectra(restored.session.fit_study(), "Spectra", reference="Reference")
    np.testing.assert_allclose(actual.values, normalized.values)
    np.testing.assert_array_equal(restored.session.original_image_array, spectra)
    from mrax.gui.services.processing import rename_processing_sources

    restored.session.rename_experiment("Anatomy", "Structural")
    rename_processing_sources(restored, "Anatomy", "Structural")
    renamed = get_z_spectra(restored.session.fit_study(), "Spectra", reference="Reference")
    np.testing.assert_allclose(renamed.values, normalized.values)
    assert restored.pipelines["Spectra"]["output"]["space"] == "Structural"
