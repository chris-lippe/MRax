"""Synthetic import contracts and numerical references."""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _synthetic import (
    _base_dicom_dataset,
    _write_classic_dicom_slice,
    _write_minimal_bruker_scan,
)
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence
from pydicom.tag import Tag
from pydicom.uid import EnhancedMRImageStorage, generate_uid

from mrax.core import make_model_callable
from mrax.data_struct import Quantity
from mrax.gui import GuiSession
from mrax.gui.app import _detect_import_vendor, _load_dicom_experiment_from_dialog
from mrax.io import (
    discover_dicom_series,
    import_bruker_experiment,
    import_bruker_study,
    import_pulseq_experiment,
    load_dicom_series,
    load_dicom_study,
)
from mrax.io.bruker_images import bruker_spatial_geometry
from mrax.io.bruker_metadata import extract_observables
from mrax.library.default_models import make_AB_MT_n_model, make_zaiss_full_model


def _write_enhanced_bruker_dicom(path, *, series_uid):
    dataset = _base_dicom_dataset(
        path,
        series_uid=series_uid,
        sop_class_uid=EnhancedMRImageStorage,
    )
    dataset.Manufacturer = "Bruker BioSpin MRI GmbH"
    dataset.PulseSequenceName = "MSME"
    dataset.MagneticFieldStrength = 9.4
    dataset.NumberOfFrames = 4

    pixel_measures = Dataset()
    pixel_measures.PixelSpacing = [1.5, 2.0]
    pixel_measures.SliceThickness = 3.0
    pixel_measures.SpacingBetweenSlices = 3.0
    orientation = Dataset()
    orientation.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    timing = Dataset()
    timing.RepetitionTime = 1200.0
    timing.FlipAngle = 90.0
    shared = Dataset()
    shared.PixelMeasuresSequence = Sequence([pixel_measures])
    shared.PlaneOrientationSequence = Sequence([orientation])
    shared.MRTimingAndRelatedParametersSequence = Sequence([timing])
    dataset.SharedFunctionalGroupsSequence = Sequence([shared])

    echo_dimension = Dataset()
    echo_dimension.DimensionIndexPointer = Tag(0x0018, 0x9082)
    echo_dimension.FunctionalGroupPointer = Tag(0x0018, 0x9114)
    echo_dimension.DimensionDescriptionLabel = "Echo"
    slice_dimension = Dataset()
    slice_dimension.DimensionIndexPointer = Tag(0x0020, 0x9057)
    slice_dimension.FunctionalGroupPointer = Tag(0x0020, 0x9111)
    slice_dimension.DimensionDescriptionLabel = "Slice"
    dataset.DimensionIndexSequence = Sequence([echo_dimension, slice_dimension])

    per_frame = []
    frames = []
    for echo_index in range(2):
        for slice_index in range(2):
            content = Dataset()
            content.DimensionIndexValues = [echo_index + 1, slice_index + 1]
            content.InStackPositionNumber = slice_index + 1
            content.FrameLabel = f"Echo {echo_index + 1}"
            position = Dataset()
            position.ImagePositionPatient = [0.0, 0.0, float(slice_index) * 3.0]
            echo = Dataset()
            echo.EffectiveEchoTime = float((echo_index + 1) * 12)
            transform = Dataset()
            transform.RescaleSlope = 1.0
            transform.RescaleIntercept = 0.0
            group = Dataset()
            group.FrameContentSequence = Sequence([content])
            group.PlanePositionSequence = Sequence([position])
            group.MREchoSequence = Sequence([echo])
            group.PixelValueTransformationSequence = Sequence([transform])
            per_frame.append(group)
            frames.append(
                np.arange(6, dtype=np.int16).reshape((2, 3))
                + np.int16(echo_index * 100 + slice_index * 10)
            )
    dataset.PerFrameFunctionalGroupsSequence = Sequence(per_frame)
    dataset.PixelData = np.stack(frames).tobytes()
    dataset.save_as(path, enforce_file_format=True)
    return np.stack(frames)


def _write_enhanced_bruker_quantitative_dicom(
    path,
    *,
    protocol,
    frame_specs,
    series_uid=None,
    image_type=None,
):
    dataset = _base_dicom_dataset(
        path,
        series_uid=series_uid or generate_uid(),
        sop_class_uid=EnhancedMRImageStorage,
    )
    dataset.Manufacturer = "Bruker BioSpin MRI GmbH"
    dataset.ProtocolName = protocol
    dataset.SeriesDescription = protocol
    dataset.PulseSequenceName = protocol
    dataset.MagneticFieldStrength = 9.4
    dataset.NumberOfFrames = len(frame_specs)
    if image_type is not None:
        dataset.ImageType = list(image_type)
    if "cest" in protocol.lower() or "wasabi" in protocol.lower():
        dataset.MagnetizationTransfer = "YES"

    pixel_measures = Dataset()
    pixel_measures.PixelSpacing = [1.0, 1.0]
    pixel_measures.SliceThickness = 2.0
    orientation = Dataset()
    orientation.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    shared = Dataset()
    shared.PixelMeasuresSequence = Sequence([pixel_measures])
    shared.PlaneOrientationSequence = Sequence([orientation])
    dataset.SharedFunctionalGroupsSequence = Sequence([shared])

    temporal_dimension = Dataset()
    temporal_dimension.DimensionIndexPointer = Tag(0x0020, 0x9128)
    temporal_dimension.FunctionalGroupPointer = Tag(0x0020, 0x9111)
    temporal_dimension.DimensionDescriptionLabel = "Cycle"
    dataset.DimensionIndexSequence = Sequence([temporal_dimension])

    frames = []
    per_frame = []
    for frame_index, spec in enumerate(frame_specs):
        content = Dataset()
        content.DimensionIndexValues = [int(spec.get("cycle", frame_index)) + 1]
        content.TemporalPositionIndex = int(spec.get("cycle", frame_index)) + 1
        content.FrameComments = str(spec.get("comment") or "")
        content.FrameLabel = str(spec.get("label") or "")
        position = Dataset()
        position.ImagePositionPatient = [0.0, 0.0, 0.0]
        transform = Dataset()
        transform.RescaleSlope = 1.0
        transform.RescaleIntercept = 0.0
        group = Dataset()
        group.FrameContentSequence = Sequence([content])
        group.PlanePositionSequence = Sequence([position])
        group.PixelValueTransformationSequence = Sequence([transform])
        if spec.get("inversion_time") is not None:
            modifier = Dataset()
            modifier.InversionTimes = float(spec["inversion_time"])
            group.MRModifierSequence = Sequence([modifier])
        per_frame.append(group)
        frames.append(
            np.arange(6, dtype=np.int16).reshape((2, 3))
            + np.int16(frame_index * 10)
        )
    dataset.PerFrameFunctionalGroupsSequence = Sequence(per_frame)
    dataset.PixelData = np.stack(frames).tobytes()
    dataset.save_as(path, enforce_file_format=True)
    return np.stack(frames)




def _write_multicycle_bruker_slice_scan(root):
    scan = root / "1"
    pdata = scan / "pdata" / "1"
    pdata.mkdir(parents=True)
    frame_count, slice_count, x, y, coil_count = 3, 2, 3, 2, 2
    object_order = np.asarray([1, 0], dtype=int)

    images = np.empty((frame_count, x, y, slice_count), dtype=np.float32)
    base = np.arange(x * y, dtype=np.float32).reshape((x, y)) + 1.0
    for frame_index in range(frame_count):
        for slice_index in range(slice_count):
            images[frame_index, :, :, slice_index] = (
                base + 10.0 * frame_index + 100.0 * slice_index
            )

    (scan / "method").write_text(
        "\n".join(
            [
                "##TITLE=method",
                "##$Method=<Bruker:MGE>",
                "##$PVM_SpatDimEnum=<2D>",
                "##$PVM_EncMatrix=( 2 )",
                f"{x} {y}",
                f"##$PVM_EncNReceivers={coil_count}",
                f"##$PVM_NRepetitions={frame_count}",
                "##$PVM_NSPacks=1",
                "##$PVM_SPackArrNSlices=( 1 )",
                f"{slice_count}",
                f"##$PVM_ObjOrderList=( {slice_count} )",
                " ".join(str(value) for value in object_order.tolist()),
                f"##$PVM_EncSteps1=( {y} )",
                "-1 0",
                "##$PVM_EncUseMultiRec=<Yes>",
                "##END=",
            ]
        ),
        encoding="utf-8",
    )
    (scan / "acqp").write_text(
        "\n".join(
            [
                "##TITLE=acqp",
                "##$ACQ_dim=2",
                f"##$NI={slice_count}",
                f"##$NR={frame_count}",
                f"##$NSLICES={slice_count}",
                "##$GO_raw_data_format=GO_32BIT_FLOAT",
                "##$ACQ_jobs=( 9 )",
                f"{2 * x} 0 0 0 0 0 {frame_count * slice_count * y} 1 <job0>",
                "##END=",
            ]
        ),
        encoding="utf-8",
    )
    (pdata / "reco").write_text(
        "\n".join(
            [
                "##TITLE=reco",
                f"##$RECO_inp_size=( 2 )\n{x} {y}",
                f"##$RECO_size=( 2 )\n{x} {y}",
                f"##$RECO_ft_size=( 2 )\n{x} {y}",
                f"##$RecoSortRange=( 2 )\n{y} {slice_count}",
                f"##$RecoSortSize=( 2 )\n{y} {slice_count}",
                f"##$RecoSortMaps=( {y + slice_count} )",
                "0 1 1 0",
                f"##$RecoObjectsPerRepetition={slice_count}",
                f"##$RECO_transposition=( {slice_count} )\n0 0",
                "##END=",
            ]
        ),
        encoding="utf-8",
    )
    (pdata / "visu_pars").write_text(
        "\n".join(
            [
                "##TITLE=visu_pars",
                f"##$VisuCoreFrameCount={frame_count * slice_count}",
                "##$VisuCoreDim=2",
                f"##$VisuCoreSize=( 2 )\n{x} {y}",
                "##$VisuFGOrderDesc=( 2 )",
                f"({slice_count}, <FG_SLICE>, <>, 0, 2)",
                f"({frame_count}, <FG_CYCLE>, <>, 2, 0)",
                "##$VisuCoreDataSlope=( 1 )\n1",
                "##$VisuCoreDataOffs=( 1 )\n0",
                "##$VisuCoreFrameType=( 1 )\nMAGNITUDE_IMAGE",
                "##$VisuCoreWordType=_16BIT_SGN_INT",
                "##$VisuCoreByteOrder=littleEndian",
                f"##$VisuCoreTransposition=( {slice_count} )\n0 0",
                "##END=",
            ]
        ),
        encoding="utf-8",
    )

    stored_frames = np.moveaxis(images, -1, 1).reshape((frame_count * slice_count, x, y))
    np.concatenate([frame.reshape((-1,), order="F") for frame in stored_frames]).astype(np.int16).tofile(
        pdata / "2dseq"
    )

    scaled_images = images * np.sqrt(float(x * y))
    image_objects = np.moveaxis(scaled_images, -1, 1)
    kspace = np.fft.fftshift(
        np.fft.fftn(
            np.fft.ifftshift(image_objects, axes=(-2, -1)),
            axes=(-2, -1),
            norm="ortho",
        ),
        axes=(-2, -1),
    ).transpose((0, 1, 3, 2))
    coil_kspace = np.zeros((frame_count, slice_count, coil_count, y, x), dtype=np.complex64)
    coil_kspace[:, :, 0] = kspace
    acquired = np.empty_like(coil_kspace)
    for acquired_slice, output_slice in enumerate(object_order.tolist()):
        acquired[:, acquired_slice] = coil_kspace[:, output_slice]
    blocks = np.transpose(acquired, (0, 3, 1, 2, 4))
    interleaved = np.empty(blocks.shape + (2,), dtype=np.float32)
    interleaved[..., 0] = blocks.real
    interleaved[..., 1] = blocks.imag
    interleaved.reshape((-1,)).tofile(scan / "rawdata.job0")
    return scan, images


def test_bruker_experiment_accepts_string_paths(tmp_path):
    _write_minimal_bruker_scan(tmp_path)
    _, dataset, _, _ = import_bruker_experiment(str(tmp_path / "1"))
    assert dataset.shape == (1, 2, 2)


def test_bruker_study_discovers_standalone_exports_and_nonnumeric_experiment_names(tmp_path):
    _write_minimal_bruker_scan(tmp_path)
    exported = tmp_path / "exported_scan"
    (tmp_path / "1").rename(exported)
    for root in [tmp_path, exported]:
        study, metadata = import_bruker_study(root, field_strength_mhz=100)
        assert study.names == ["exported_scan"]
        assert study.datasets[0].shape == (1, 2, 2)
        assert not metadata["skipped_experiments"]


def test_bruker_study_skips_missing_and_unsupported_acquisitions(tmp_path):
    import shutil

    _write_minimal_bruker_scan(tmp_path)
    shutil.copytree(tmp_path / "1", tmp_path / "2")
    (tmp_path / "2/pdata/1/visu_pars").unlink()
    shutil.copytree(tmp_path / "1", tmp_path / "3")
    (tmp_path / "3/acqp").write_text('##$ACQ_dim=2\n##$ACQ_size=( 2 )\n4 2\n##$NI=1\n##$NR=1\n##$GO_raw_data_format=UNSUPPORTED_SCALAR\n')
    (tmp_path / "3/fid").write_bytes(bytes(16))
    with pytest.warns(RuntimeWarning, match="skipped Bruker scan"):
        study, metadata = import_bruker_study(tmp_path, field_strength_mhz=100)
    assert study.names == ["1"]
    assert set(metadata["skipped_experiments"]) == {"2", "3"}
    assert metadata["skipped_experiments"]["2"]["reason"] == "missing_file"
    assert metadata["skipped_experiments"]["3"]["reason"] == "import_failed"


def test_synthetic_bruker_study_import_workflow(tmp_path):
    root = tmp_path / "bruker"
    _write_minimal_bruker_scan(root)

    study, metadata = import_bruker_study(
        root,
        study_globals={"field_strength_mhz": Quantity(jnp.asarray(400.0), "MHz")},
        load_raw_kspace=False,
    )

    assert study.names == ["1"]
    assert study.experiments[0].exp_type == "mse"
    assert study.datasets[0].shape == (1, 2, 2)
    np.testing.assert_allclose(np.asarray(study.datasets[0].data[0]), [[10.0, 12.0], [14.0, 16.0]])

    exp_meta = metadata["experiments"]["1"]
    assert exp_meta["method"]["Method"] == "MSME"
    assert exp_meta["acqp"]["ACQ_dim"] == 2
    assert exp_meta["image_scaling"]["applied_to"] == "DataSet.data"


@pytest.mark.parametrize("flag, expected", [
    ("On", True), ("Off", False), ("<On>", True), ("<Off>", False),
    ("true", True), ("false", False), ("YES", True), ("NO", False),
    (True, True), (False, False), (1, True), (0, False),
    (["Off"], False), (["On", "On"], True),
])
def test_bruker_spin_lock_metadata_is_preserved(flag, expected):
    observables = extract_observables("sat_trans", {}, 1, {"SpinLock": flag})
    spin_lock = observables["spin_lock"]
    assert spin_lock.si_unit == "1"
    assert spin_lock.shape == ()
    assert spin_lock.values.dtype == jnp.bool_
    assert bool(spin_lock.values) is expected


@pytest.mark.parametrize("flag", ["unknown", 2, [], ["On", "Off"]])
def test_bruker_invalid_spin_lock_metadata_is_rejected(flag):
    with pytest.raises(ValueError, match="SpinLock.*Boolean"):
        extract_observables("sat_trans", {}, 1, {"SpinLock": flag})


def test_bruker_missing_spin_lock_metadata_keeps_model_default():
    assert "spin_lock" not in extract_observables("sat_trans", {}, 1, {})
    assert "spin_lock" not in extract_observables("mse", {}, 1, {"SpinLock": "On"})


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
@pytest.mark.parametrize("factory", [make_AB_MT_n_model, make_zaiss_full_model])
@pytest.mark.parametrize("flag, enabled", [("On", True), ("Off", False)])
def test_imported_bruker_spin_lock_selects_model_projection(tmp_path, backend, factory, flag, enabled):
    try:
        device = jax.devices(backend)[0]
    except RuntimeError:
        pytest.skip(f"{backend} backend is unavailable")
    root = tmp_path / "bruker"
    _write_minimal_bruker_scan(root)
    (root / "1" / "method").write_text(
        f"##TITLE=method\n##$Method=<CEST>\n##$SpinLock=<{flag}>\n##END=\n",
        encoding="utf-8",
    )
    model = factory(pools=[])
    experiment, _, _, _ = import_bruker_experiment(root / "1", model_map_by_type={"sat_trans": model})
    assert experiment.exp_type == "sat_trans"
    assert bool(experiment.observables["spin_lock"].values) is enabled
    experiment = experiment.insert_observables({
        "field_strength_mhz": Quantity(400.0, "MHz"),
        "sat_b1": Quantity(2.0, "uT"),
        "sat_time": Quantity(0.0, "ms"),
        "TR": Quantity(1000.0, "ms"),
        "r1_map": Quantity(jnp.ones(experiment.spatial_shape), "1/s"),
    })
    params, unpack = model.parametrization_factory(experiment.spatial_shape)
    fields = unpack(params)
    if model.name == "ab_mt_n":
        fields["t_delay"] = jnp.zeros(experiment.spatial_shape)
        fields["sin_a_eff"] = jnp.full(experiment.spatial_shape, np.sqrt(0.5))
    model_fn = jax.jit(make_model_callable(experiment, model))
    fields = jax.device_put(fields, device)
    for offset in (0.0, 1.0):
        offsets = jax.device_put(jnp.asarray([offset]), device)
        result = model_fn(fields, offsets)
        assert result.device == device
        omega1 = 2.0 * np.pi * 42.58 * 2.0 / 400.0
        expected = 1.0 if enabled else (2.0 * np.pi * offset)**2 / (
            (2.0 * np.pi * offset)**2 + omega1**2 + 1e-8)
        np.testing.assert_allclose(result, expected, rtol=2e-6, atol=2e-7)


def test_bruker_multicycle_slice_stack_sorts_vendor_images_and_raw_kspace(tmp_path):
    scan, expected_images = _write_multicycle_bruker_slice_scan(tmp_path / "bruker")

    experiment, dataset, _, metadata = import_bruker_experiment(scan, load_raw_kspace=True)

    assert experiment.dims == {"frame": 3, "x": 3, "y": 2, "z": 2}
    np.testing.assert_allclose(np.asarray(dataset.data), expected_images)
    assert dataset.kspace.shape == (6, 2, 2, 3)
    assert dataset.acquisition.samples.shape == dataset.kspace.shape
    assert metadata["raw_import"]["reconstruction_status"] == "not_run"
    assert "raw_reconstruction" not in (dataset.processed or {})
    from mrax.reco import reconstruct_acquisition

    result = reconstruct_acquisition(dataset.acquisition)
    np.testing.assert_allclose(
        np.asarray(result.image),
        expected_images,
        rtol=1e-5,
        atol=1e-5,
    )
    coil_images = np.asarray(result.coil_images)
    assert coil_images.shape == (3, 2, 3, 2, 2)
    assert np.iscomplexobj(coil_images)
    np.testing.assert_allclose(np.abs(coil_images[:, 0]), expected_images, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(coil_images[:, 1], 0.0, atol=1e-6)


def test_explicit_dicom_contrast_groups_sort_timings_and_reject_misalignment(tmp_path):
    from pydicom import dcmread

    from mrax.io import import_pulseq_study, load_dicom_study

    uids = [generate_uid() for _ in range(3)]
    expected = []
    for echo in range(3):
        slices = []
        for plane in range(2):
            pixels = _write_classic_dicom_slice(tmp_path / f'{2-echo}_{plane}.dcm',
                series_uid=uids[echo], echo_index=echo, slice_index=plane)
            slices.append(pixels * 2 + 5)
        expected.append(np.stack(slices, axis=-1))
    groups = {'decay': list(reversed(uids))}
    study, metadata = load_dicom_study(tmp_path, series_groups=groups)
    name, experiment, _model, dataset = next(study.iter_entries())
    assert name == 'decay'
    assert experiment.exp_type == 'mse'
    assert dataset.spatial_geometry.status == 'known'
    np.testing.assert_array_equal(experiment.observables['TE'].values, [10., 20., 30.])
    np.testing.assert_array_equal(dataset.data, expected)
    assert metadata['source_series_count'] == 3
    combined, _ = import_pulseq_study(tmp_path, dicom_series_groups=groups)
    assert len(combined.names) == 1
    np.testing.assert_array_equal(next(combined.iter_entries())[-1].data, expected)
    with pytest.raises(ValueError, match='selected more than once'):
        load_dicom_study(tmp_path, series_groups={'a': uids, 'b': [uids[0]]})
    changed = tmp_path / '0_0.dcm'
    header = dcmread(changed)
    header.ImagePositionPatient = [5., 0., 0.]
    header.save_as(changed, enforce_file_format=True)
    with pytest.raises(ValueError, match='geometry'):
        load_dicom_study(tmp_path, series_groups=groups)


def test_dicom_preparation_metadata_and_ima_discovery(tmp_path):
    from pydicom.uid import generate_uid

    from mrax.io import import_pulseq_study, load_dicom_series

    uid = generate_uid()
    for frame in range(3):
        _write_classic_dicom_slice(tmp_path / f'frame_{frame}.IMA', series_uid=uid,
            echo_index=frame, slice_index=0, description='Synthetic preparation')
    sequence = tmp_path / 'preparation.seq'
    sequence.write_text('[DEFINITIONS]\nName saturation\noffsets_ppm -2 0 2\nb1pa 2.5\ntsat 0.004\n')
    study, metadata = import_pulseq_study(tmp_path,
        sequence_map={'dicom:' + uid: sequence},
        type_overrides={'dicom:' + uid: 'sat_trans'},
        metadata_overrides={'dicom:' + uid: {'sat_b1_uT': [3.]}})
    assert len(study.names) == 1 and not metadata['import_issues']
    experiment = study.experiments[0]
    experiment.validate(study.datasets[0])
    np.testing.assert_array_equal(experiment.observables['freq_offsets'].values, [-2., 0., 2.])
    np.testing.assert_array_equal(experiment.observables['sat_b1'].values, [3.])
    np.testing.assert_array_equal(experiment.observables['sat_time'].values, [4.])
    with pytest.raises(ValueError, match='Unknown normalized'):
        load_dicom_series(tmp_path, metadata_overrides={'misspelled_time': [1.]})
    with pytest.raises(ValueError, match='frame_count'):
        load_dicom_series(tmp_path, metadata_overrides={'frame_count': 7})


def test_standard_dicom_series_import_builds_mrax_experiment(tmp_path):
    series_uid = "1.2.826.0.1.3680043.10.1000.10"
    raw_frames = {}
    for echo_index in range(2):
        for slice_index in range(2):
            raw_frames[(echo_index, slice_index)] = _write_classic_dicom_slice(
                tmp_path / f"image_{echo_index}_{slice_index}.dcm",
                series_uid=series_uid,
                echo_index=echo_index,
                slice_index=slice_index,
            )

    discovered = discover_dicom_series(tmp_path)
    experiment, dataset, metadata = load_dicom_series(tmp_path)

    assert len(discovered) == 1
    assert discovered[0].file_count == 4
    assert experiment.exp_type == "mse"
    assert experiment.dims == {"te": 2, "x": 2, "y": 3, "z": 2}
    assert dataset.shape == (2, 2, 3, 2)
    np.testing.assert_allclose(np.asarray(experiment.observables["TE"].values), [10.0, 20.0])
    expected = raw_frames[(0, 0)].astype(float) * 2.0 + 5.0
    np.testing.assert_allclose(np.asarray(dataset.data)[0, ..., 0], expected)
    np.testing.assert_allclose(
        np.asarray(metadata["affine"]),
        np.asarray(
            [
                [0.0, -3.0, 0.0, 0.0],
                [0.0, 0.0, -2.0, 0.0],
                [2.5, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    )
    assert metadata["adapter"] == "standard"
    assert dataset.spatial_geometry is not None
    assert dataset.spatial_geometry.voxel_sizes == (2.0, 3.0, 2.5)
    assert dataset.spatial_geometry.axis_anatomical_labels == ("y", "x", "z")
    assert "study_instance_uid" not in metadata
    assert experiment._mrax_dicom_metadata is metadata


def test_bruker_geometry_uses_slice_positions_and_anatomical_axis_order():
    orientation = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=float,
    )
    visu = {
        "VisuCoreDim": 2,
        "VisuCoreSize": [256, 256],
        "VisuCoreExtent": [40.0, 32.0],
        "VisuCoreFrameCount": 3,
        "VisuFGOrderDesc": "FG_SLICE",
        "VisuCoreFrameThickness": 0.5,
        "VisuCoreOrientation": orientation.T.reshape((-1,)).tolist(),
        "VisuCorePosition": [
            0.0,
            0.0,
            0.0,
            0.0,
            0.75,
            0.0,
            0.0,
            1.5,
            0.0,
        ],
    }

    geometry = bruker_spatial_geometry(visu, spatial_shape=(256, 256, 3))

    assert geometry.voxel_sizes == (0.125, 0.15625, 0.75)
    assert geometry.axis_anatomical_labels == ("z", "x", "y")


def test_enhanced_bruker_dicom_uses_separate_adapter(tmp_path):
    path = tmp_path / "enhanced.dcm"
    source_frames = _write_enhanced_bruker_dicom(
        path,
        series_uid="1.2.826.0.1.3680043.10.1000.20",
    )

    experiment, dataset, metadata = load_dicom_series(path)

    assert experiment.exp_type == "mse"
    assert experiment.dims == {"te": 2, "x": 2, "y": 3, "z": 2}
    assert dataset.shape == (2, 2, 3, 2)
    np.testing.assert_array_equal(np.asarray(dataset.data)[1, ..., 1], source_frames[3])
    np.testing.assert_allclose(np.asarray(experiment.observables["TE"].values), [12.0, 24.0])
    assert metadata["adapter"] == "bruker"
    assert metadata["vendor"]["profile"] == "enhanced_mr"
    assert metadata["vendor"]["standard_functional_groups"] is True
    assert metadata["magnetic_field_strength_t"] == 9.4

    study, study_metadata = load_dicom_study(path)
    assert study.names == ["synthetic_multi_echo"]
    assert study.experiments[0].exp_type == "mse"
    assert study.study_globals["field_strength_mhz"].si_unit == "MHz"
    assert study_metadata["series_count"] == 1


def test_recursive_dicom_folder_import_builds_short_named_gui_study(tmp_path):
    descriptions = (
        "Quantitative Multi Echo Mapping Acquisition Long Description",
        "Reference Anatomy",
    )
    series = (
        (generate_uid(), tmp_path / "first", 7, descriptions[0], 2),
        (generate_uid(), tmp_path / "nested" / "second", 8, descriptions[1], 1),
    )
    for uid, folder, number, description, echo_count in series:
        folder.mkdir(parents=True)
        for echo_index in range(echo_count):
            for slice_index in range(2):
                _write_classic_dicom_slice(
                    folder / f"image_{echo_index}_{slice_index}.dcm",
                    series_uid=uid,
                    echo_index=echo_index,
                    slice_index=slice_index,
                    description=description,
                    series_number=number,
                )

    study, metadata = load_dicom_study(tmp_path)
    session = GuiSession()
    session.load_dicom_study(tmp_path)

    assert len(study.names) == 2
    assert all(len(name) <= 40 for name in study.names)
    assert study.names[0].startswith("7_quantitative_multi_echo")
    assert study.names[1] == "8_reference_anatomy"
    assert list(metadata["experiments"]) == list(study.names)
    assert session.experiment_names == list(study.names)
    assert session.source_kind == "dicom"
    assert session.data["manifest"]["source"]["kind"] == "dicom"
    assert _detect_import_vendor(tmp_path) == "DICOM"


def test_gui_single_dicom_file_payload_preserves_inferred_experiment(tmp_path):
    path = tmp_path / "single.dcm"
    _write_enhanced_bruker_dicom(path, series_uid=generate_uid())

    payload = _load_dicom_experiment_from_dialog(path)
    session = GuiSession()
    session.create_empty_study("dicom_single")
    added_name = session.add_experiment(
        payload["name"],
        payload["data"],
        dims=payload["dims"],
        exp_type=payload["exp_type"],
        model_name=payload["model_name"],
        observables=payload["observables"],
        extra_metadata=payload["extra_metadata"],
    )

    assert payload["source"] == "dicom"
    assert payload["name"] == "synthetic_multi_echo"
    assert payload["dims"] == {"te": 2, "x": 2, "y": 3, "z": 2}
    np.testing.assert_allclose(np.asarray(payload["observables"]["TE"].values), [12.0, 24.0])
    assert added_name == "synthetic_multi_echo"
    assert session.current_experiment["extra_metadata"]["adapter"] == "bruker"


def test_enhanced_bruker_dynamic_cest_infers_time_and_offset_dimensions(tmp_path):
    offsets = (-3.0, 0.0, 3.0)
    frame_specs = [
        {"cycle": cycle, "comment": f"{offset} ppm"}
        for cycle in range(2)
        for offset in offsets
    ]
    source_frames = _write_enhanced_bruker_quantitative_dicom(
        tmp_path / "dynamic_cest.dcm",
        protocol="Dynamic CEST",
        frame_specs=frame_specs,
    )

    experiment, dataset, metadata = load_dicom_series(tmp_path / "dynamic_cest.dcm")

    assert experiment.exp_type == "sat_trans"
    assert experiment.dims == {"t": 2, "b1": 1, "offset": 3, "x": 2, "y": 3}
    assert dataset.shape == (2, 1, 3, 2, 3)
    np.testing.assert_array_equal(np.asarray(dataset.data)[1, 0, 1], source_frames[4])
    np.testing.assert_allclose(
        np.asarray(experiment.observables["freq_offsets"].values),
        offsets,
    )
    assert experiment.observables["freq_offsets"].si_unit == "ppm"
    assert metadata["vendor"]["unavailable_observables"] == ["sat_b1", "sat_time"]

    _write_enhanced_bruker_quantitative_dicom(
        tmp_path / "wasabi.dcm",
        protocol="WASABI",
        frame_specs=[{"cycle": 0, "comment": f"{offset} ppm"} for offset in offsets],
    )
    wasabi_experiment, _wasabi_dataset, wasabi_metadata = load_dicom_series(
        tmp_path / "wasabi.dcm"
    )
    assert wasabi_experiment.exp_type == "sat_trans"
    assert wasabi_experiment.dims == {"t": 1, "b1": 1, "offset": 3, "x": 2, "y": 3}
    np.testing.assert_allclose(
        np.asarray(wasabi_experiment.observables["freq_offsets"].values),
        offsets,
    )
    assert wasabi_metadata["adapter"] == "bruker"


def test_enhanced_bruker_fair_t1_infers_ti_and_label_dimensions(tmp_path):
    frame_specs = [
        {"cycle": 0, "label": "Selective", "inversion_time": 100.0},
        {"cycle": 0, "label": "Non-selective", "inversion_time": 100.0},
        {"cycle": 1, "label": "Selective", "inversion_time": 500.0},
        {"cycle": 1, "label": "Non-selective", "inversion_time": 500.0},
    ]
    _write_enhanced_bruker_quantitative_dicom(
        tmp_path / "fair_t1.dcm",
        protocol="FAIR T1map",
        frame_specs=frame_specs,
    )

    experiment, dataset, _metadata = load_dicom_series(tmp_path / "fair_t1.dcm")

    assert experiment.exp_type == "fair_asl"
    assert experiment.dims == {"ti": 2, "fair": 2, "x": 2, "y": 3}
    assert dataset.shape == (2, 2, 2, 3)
    np.testing.assert_allclose(np.asarray(experiment.observables["TI"].values), [100.0, 500.0])
    np.testing.assert_array_equal(np.asarray(experiment.observables["fair_label"].values), [0, 1])
    assert experiment.observables["TI"].si_unit == "ms"
    assert experiment.observables["fair_label"].si_unit == "1"


def test_bruker_adapter_selects_quantitative_stack_over_auxiliary_objects(tmp_path):
    series_uid = generate_uid()
    offsets = (-2.0, 0.0, 2.0)
    source_frames = _write_enhanced_bruker_quantitative_dicom(
        tmp_path / "quantitative.dcm",
        protocol="Dynamic CEST",
        frame_specs=[
            {"cycle": cycle, "comment": f"{offset} ppm"}
            for cycle in range(2)
            for offset in offsets
        ],
        series_uid=series_uid,
        image_type=("ORIGINAL", "PRIMARY", "NON_PARALLEL", "NONE"),
    )
    _write_enhanced_bruker_quantitative_dicom(
        tmp_path / "auxiliary.dcm",
        protocol="Dynamic CEST",
        frame_specs=[
            {"cycle": cycle, "comment": "0 ppm"}
            for cycle in range(2)
        ],
        series_uid=series_uid,
        image_type=("ORIGINAL", "PRIMARY", "PARALLEL", "NONE"),
    )

    experiment, dataset, metadata = load_dicom_series(tmp_path)

    assert experiment.dims == {"t": 2, "b1": 1, "offset": 3, "x": 2, "y": 3}
    np.testing.assert_array_equal(np.asarray(dataset.data).reshape(source_frames.shape), source_frames)
    assert metadata["source_file_count"] == 1
    assert metadata["auxiliary_source_file_count"] == 1


def test_synthetic_pulseq_import_workflow(tmp_path):
    seq = tmp_path / "demo.seq"
    seq.write_text(
        "\n".join(
            [
                "[DEFINITIONS]",
                "Name demo_mse",
                "TE 8 24",
                "Matrix 2 2",
                "[ADC]",
                "1 2 0 0 0 0 0",
                "[BLOCKS]",
                "1 0 0 0 0 0 1",
                "2 0 0 0 0 0 1",
            ]
        ),
        encoding="utf-8",
    )
    image_data = np.arange(8, dtype=np.float32).reshape((2, 2, 2))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        experiment, dataset, model, metadata = import_pulseq_experiment(
            seq,
            image_data=image_data,
            spatial_shape=(2, 2),
            override_type="mse",
            name="pulseq_demo",
        )

    assert experiment.exp_type == "mse"
    assert dataset.name == "pulseq_demo"
    assert dataset.shape == (2, 2, 2)
    assert model.name == "t2_se"
    assert metadata["path"] == seq
    assert metadata["pulseq"]["adc_counts"] == (2, 2)
    np.testing.assert_array_equal(np.asarray(dataset.data), image_data)
