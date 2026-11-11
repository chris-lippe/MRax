"""Synthetic save export contracts and numerical references."""

import json
import zipfile

import jax.numpy as jnp
import numpy as np
import pytest

from mrax import __version__
from mrax.analysis.group import discover_study_manifests
from mrax.data_struct import (
    DataSet,
    Experiment,
    Quantity,
    Study,
    load_study_results,
)
from mrax.data_struct.storage import _resolve_manifest_path
from mrax.gui import GuiSession
from mrax.gui.services.package import _mrax_package_entries
from mrax.io.bundle import extract_bundle, write_bundle
from mrax.io.image_export import export_image_files
from mrax.io.nifti import (
    export_nifti_4d,
    export_nifti_series,
    load_nifti,
    mrax_to_nifti_volume,
    nifti_spatial_geometry,
    save_nifti,
)
from mrax.library import make_t2_se_model


def small_study():
    experiment = Experiment(
        dims={"te": 3, "x": 2, "y": 4},
        exp_type="mse",
        observables={"TE": Quantity([10.0, 30.0, 70.0], "ms")},
    )
    data = np.arange(24, dtype=np.float32).reshape(3, 2, 4)
    return Study(
        [experiment],
        [DataSet(data)],
        [make_t2_se_model()],
        names=["synthetic"],
        study_globals={"field_strength_mhz": Quantity(100.0, "MHz")},
    )


def test_storage_rejects_path_traversal(tmp_path):
    with pytest.raises(ValueError, match="outside manifest directory"):
        _resolve_manifest_path(tmp_path / "study", "../secret.npy", label="image")
    package = tmp_path / "package"
    package.mkdir()
    (package / "mrax_package.json").write_text(
        json.dumps(
            {
                "format": "mrax.gui.package",
                "version": 1,
                "mrax_version": __version__,
                "entries": {"gui_state": "../outside.json"},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="outside package directory"):
        _mrax_package_entries(package)


def test_nifti_single_volume_and_series_round_trip(tmp_path):
    image = np.arange(12, dtype=np.float32).reshape(3, 4)
    affine = np.diag([1.5, 2.0, 1.0, 1.0])
    path = tmp_path / "image.nii.gz"
    save_nifti(path, image, affine)
    loaded, loaded_affine = load_nifti(path)
    np.testing.assert_allclose(loaded, image)
    np.testing.assert_allclose(loaded_affine, affine, atol=1e-6)
    geometry = nifti_spatial_geometry(loaded_affine, spatial_shape=image.shape)
    assert geometry.voxel_sizes == (2.0, 1.5)
    assert geometry.axis_anatomical_labels == ("y", "x")

    experiment = Experiment(
        dims={"echo": 2, "x": 3, "y": 4},
        observables={"TE": Quantity(jnp.asarray([10.0, 20.0]), "ms")},
        exp_type="mse",
    )
    dataset = DataSet(jnp.stack([jnp.asarray(image), jnp.asarray(image + 1)]), name="nifti")
    outputs = export_nifti_series(experiment, dataset, tmp_path / "series", affine=affine)
    output_4d = export_nifti_4d(experiment, dataset, tmp_path / "series_4d.nii.gz", affine=affine)

    assert len(outputs) == 2
    assert all(path.is_file() for path in outputs)
    assert output_4d.is_file()


def test_default_study_save_is_self_contained_bundle_without_gui(tmp_path):
    study = small_study()
    events = []
    output = study.save(
        tmp_path / "synthetic.mrax", progress_callback=lambda *args: events.append(args)
    )
    assert output.suffix == ".mrax"
    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
        assert "study_manifest.json" in names
        assert all("gui" not in name for name in names)
    restored = load_study_results(output)
    np.testing.assert_array_equal(
        restored["experiments"]["synthetic"]["mr_image"], study.datasets[0].data
    )
    assert GuiSession.from_manifest(output).frame_count == 3
    assert discover_study_manifests(tmp_path)["synthetic"] == output
    assert len(events) > 3 and events[-1][0] == events[-1][1]
    fractions = [current / total for current, total, _ in events]
    assert np.all(np.diff(fractions) >= 0)


def test_bundle_failure_preserves_previous_file_and_rejects_traversal(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "array.bin").write_bytes(b"x" * 2048)
    target = tmp_path / "study.mrax"
    target.write_bytes(b"previous contents")

    def cancel(*_args):
        raise RuntimeError("cancel")

    with pytest.raises(RuntimeError, match="cancel"):
        write_bundle(target, source, progress_callback=cancel, overwrite=True)
    assert target.read_bytes() == b"previous contents"
    assert not list(tmp_path.glob("*.partial"))
    with zipfile.ZipFile(target, "w") as archive:
        archive.writestr("../outside", b"bad")
    with pytest.raises(ValueError, match="Unsafe"):
        extract_bundle(target, tmp_path / "extract")
    assert not (tmp_path / "outside").exists()


def test_dicom_export_preserves_calibrated_pixels_units_and_geometry(tmp_path):
    import pydicom

    data = np.linspace(-12.5, 93.7, 2 * 3 * 4 * 5).reshape((2, 3, 4, 5))
    affine = np.diag([1.2, 2.3, 3.4, 1.0])
    affine[:3, 3] = [10.0, -20.0, 30.0]
    paths = export_image_files(
        tmp_path / "image.dcm",
        data,
        dims={"t": 2, "x": 3, "y": 4, "z": 5},
        affine=affine,
        format="dicom",
        unit="Hz",
    )
    assert len(paths) == 6
    series = [pydicom.dcmread(path) for path in paths]
    assert len({ds.StudyInstanceUID for ds in series}) == 1
    assert len({ds.FrameOfReferenceUID for ds in series}) == 1
    assert len({ds.SeriesInstanceUID for ds in series}) == 2
    for index in range(2):
        expected = mrax_to_nifti_volume(data[index])
        for z, ds in enumerate(series[index * 3 : (index + 1) * 3]):
            pixels = ds.pixel_array.astype(float) * float(ds.RescaleSlope) + float(
                ds.RescaleIntercept
            )
            np.testing.assert_allclose(
                pixels, expected[:, :, z].T, atol=float(ds.RescaleSlope) * 0.51
            )
            np.testing.assert_allclose(ds.ImagePositionPatient, [-10.0, 20.0, 30.0 + z * 3.4])
            np.testing.assert_allclose(ds.PixelSpacing, [2.3, 1.2])
            assert ds.RescaleType == "Hz" and not ds.PatientID
