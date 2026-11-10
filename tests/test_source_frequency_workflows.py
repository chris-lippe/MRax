"""Source descriptions and Bruker frequencies from independent synthetic headers."""

import shutil

import numpy as np
import pytest
from _synthetic import _write_minimal_bruker_scan

from mrax.data_struct import Quantity
from mrax.gui.services.importing import _detect_bruker_import_summary
from mrax.gui.services.pipeline import ensure_pipeline, pipeline_info
from mrax.gui.services.provenance import pipeline_sources
from mrax.gui.services.workspace import StudyWorkspace
from mrax.gui.services.workspace_io import load_workspace_bundle, save_workspace_bundle
from mrax.gui.session import GuiSession
from mrax.io import import_bruker_study
from mrax.io.bruker_frequency import bruker_imaging_frequency, consistent_bruker_frequency


def headers(root):
    _write_minimal_bruker_scan(root)
    shutil.copytree(root / "1", root / "2")
    for scan, value in (("1", 400.123), ("2", 400.125)):
        path = root / scan / "method"
        path.write_text(
            path.read_text().replace("##END=", f"##$PVM_FrqWork=( 1 )\n{value}\n##END=")
        )


def test_frequency_uses_nominal_headers_and_supports_visu_only_metadata():
    assert bruker_imaging_frequency({"PVM_FrqWork": [400.15]}, {}, {"BF1": 400.12}) == 400.12
    assert bruker_imaging_frequency({"PVM_FrqWork": [400.15], "PVM_FrqRef": [400.12]}) == 400.12
    assert bruker_imaging_frequency({}, {"VisuAcqImagingFrequency": 400.12}) == 400.12
    assert bruker_imaging_frequency({"PVM_FrqWork": [float("nan"), 400.12]}) == 400.12
    assert bruker_imaging_frequency({"PVM_FrqWork": [False, -1, 0, float("inf")]}) is None


def test_small_scan_offsets_are_consistent_but_different_fields_are_not():
    assert consistent_bruker_frequency([400.123, None, 400.125]) == pytest.approx(400.124)
    assert consistent_bruker_frequency([400.123, 300.2]) is None
    assert consistent_bruker_frequency([None, float("nan")]) is None


def test_empty_workspace_has_no_source_before_import_finishes():
    session = GuiSession()
    workspace = StudyWorkspace("empty", "Importing", session)
    assert pipeline_sources(workspace) == {}
    assert pipeline_info(workspace)["sources"] == {}


def test_preview_and_direct_import_detect_frequency_without_a_gui_override(tmp_path):
    root = tmp_path / "Independent headers"
    headers(root)
    rows = _detect_bruker_import_summary(root)
    assert [row["field_strength_mhz"] for row in rows] == [400.123, 400.125]
    study, _ = import_bruker_study(root, load_raw_kspace=False)
    value = study.study_globals["field_strength_mhz"]
    assert value.si_unit == "MHz"
    assert float(value.values) == pytest.approx(400.124)
    selected, _ = import_bruker_study(root, load_raw_kspace=False, selected_experiments=["2"])
    assert float(selected.study_globals["field_strength_mhz"].values) == pytest.approx(400.125)
    overridden, _ = import_bruker_study(root, load_raw_kspace=False, field_strength_mhz=425)
    assert float(overridden.study_globals["field_strength_mhz"].values) == 425
    supplied, _ = import_bruker_study(
        root, load_raw_kspace=False, study_globals={"field_strength_mhz": Quantity(426, "MHz")}
    )
    assert float(supplied.study_globals["field_strength_mhz"].values) == 426


def test_original_source_paths_survive_workspace_save_without_using_extraction_paths(tmp_path):
    root = tmp_path / "Independent provenance"
    headers(root)
    (root / "1" / "fid").write_bytes(b"independent placeholder; raw import disabled")
    session = GuiSession()
    session.import_bruker_study(root, load_raw_kspace=False)
    work = StudyWorkspace("source", session.study_name, session)
    work.session.select_experiment("1")
    descriptions = pipeline_sources(work)
    assert descriptions["image"]["format"] == "Bruker"
    assert descriptions["image"]["location"] == "Independent provenance/1/pdata/1/2dseq"
    assert descriptions["kspace"]["path"] == str(root / "1" / "fid")
    ensure_pipeline(work)
    assert pipeline_info(work)["sources"] == descriptions
    save_workspace_bundle(work, tmp_path / "study.mrax")
    loaded = load_workspace_bundle(tmp_path / "study.mrax", tmp_path / "unpacked", key="loaded")
    assert pipeline_info(loaded)["sources"] == descriptions


def test_source_overview_handles_cross_platform_file_paths_without_reading_them():
    session = GuiSession()
    session.create_empty_study("Independent file source", field_strength_mhz=100)
    session.add_experiment(
        "A",
        np.ones((1, 2, 3)),
        dims={"frame": 1, "x": 2, "y": 3},
        extra_metadata={
            "import_format": "nifti",
            "source_path": r"C:\Independent examples\Images\image.nii.gz",
        },
    )
    sources = pipeline_sources(StudyWorkspace("paths", session.study_name, session))
    assert sources["image"]["format"] == "NIfTI"
    assert sources["image"]["location"] == "Images/image.nii.gz"
