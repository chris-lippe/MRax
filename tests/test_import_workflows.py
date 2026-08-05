import warnings

import jax.numpy as jnp
import numpy as np
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.tag import Tag
from pydicom.uid import EnhancedMRImageStorage, ExplicitVRLittleEndian, MRImageStorage, generate_uid

from mrax.data_struct import Quantity
from mrax.gui import GuiSession
from mrax.gui.app import _detect_import_vendor, _load_dicom_experiment_from_dialog
from mrax.io import (
    discover_dicom_series,
    load_bruker_study,
    load_dicom_series,
    load_dicom_study,
    load_pulseq_experiment,
)


def _base_dicom_dataset(path, *, series_uid, sop_class_uid, instance_number=1):
    file_meta = FileMetaDataset()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.MediaStorageSOPClassUID = sop_class_uid
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.ImplementationClassUID = generate_uid()
    dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = sop_class_uid
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.StudyInstanceUID = generate_uid()
    dataset.SeriesInstanceUID = series_uid
    dataset.FrameOfReferenceUID = generate_uid()
    dataset.Modality = "MR"
    dataset.Manufacturer = "Synthetic"
    dataset.SeriesDescription = "Synthetic multi-echo"
    dataset.ProtocolName = "Synthetic MSME"
    dataset.InstanceNumber = instance_number
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.Rows = 2
    dataset.Columns = 3
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 1
    return dataset


def _write_classic_dicom_slice(
    path,
    *,
    series_uid,
    echo_index,
    slice_index,
    description="Synthetic multi-echo",
    series_number=None,
):
    instance_number = echo_index * 10 + slice_index + 1
    dataset = _base_dicom_dataset(
        path,
        series_uid=series_uid,
        sop_class_uid=MRImageStorage,
        instance_number=instance_number,
    )
    dataset.StudyInstanceUID = "1.2.826.0.1.3680043.10.1000.1"
    dataset.FrameOfReferenceUID = "1.2.826.0.1.3680043.10.1000.2"
    dataset.SeriesDescription = description
    if series_number is not None:
        dataset.SeriesNumber = int(series_number)
    dataset.ScanningSequence = "SE"
    dataset.MagneticFieldStrength = 9.4
    dataset.EchoTime = float((echo_index + 1) * 10)
    dataset.RepetitionTime = 1000.0
    dataset.EchoNumbers = echo_index + 1
    dataset.PixelSpacing = [2.0, 3.0]
    dataset.SliceThickness = 2.5
    dataset.SpacingBetweenSlices = 2.5
    dataset.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    dataset.ImagePositionPatient = [0.0, 0.0, float(slice_index) * 2.5]
    dataset.RescaleSlope = 2.0
    dataset.RescaleIntercept = 5.0
    pixels = (
        np.arange(6, dtype=np.int16).reshape((2, 3))
        + np.int16(echo_index * 100 + slice_index * 10)
    )
    dataset.PixelData = pixels.tobytes()
    dataset.save_as(path, enforce_file_format=True)
    return pixels


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


def _write_minimal_bruker_scan(root):
    scan = root / "1"
    pdata = scan / "pdata" / "1"
    pdata.mkdir(parents=True)
    (scan / "method").write_text("##TITLE=method\n##$Method=<MSME>\n##END=\n", encoding="utf-8")
    (scan / "acqp").write_text("##TITLE=acqp\n##$ACQ_dim=2\n##END=\n", encoding="utf-8")
    (pdata / "reco").write_text("##TITLE=reco\n##$RECO_mode=<USER>\n##END=\n", encoding="utf-8")
    (pdata / "visu_pars").write_text(
        "\n".join(
            [
                "##TITLE=visu_pars",
                "##$VisuCoreFrameCount=1",
                "##$VisuCoreDim=2",
                "##$VisuCoreSize=( 2 )",
                "2 2",
                "##$VisuCoreDataSlope=( 1 )",
                "2",
                "##$VisuCoreDataOffs=( 1 )",
                "10",
                "##$VisuCoreFrameType=( 1 )",
                "MAGNITUDE_IMAGE",
                "##$VisuCoreWordType=_16BIT_SGN_INT",
                "##$VisuCoreByteOrder=littleEndian",
                "##END=",
            ]
        ),
        encoding="utf-8",
    )
    np.arange(4, dtype=np.int16).tofile(pdata / "2dseq")


def test_synthetic_bruker_study_import_workflow(tmp_path):
    root = tmp_path / "bruker"
    _write_minimal_bruker_scan(root)

    study, metadata = load_bruker_study(
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
                [-3.0, 0.0, 0.0, 0.0],
                [0.0, -2.0, 0.0, 0.0],
                [0.0, 0.0, 2.5, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    )
    assert metadata["adapter"] == "standard"
    assert "study_instance_uid" not in metadata
    assert experiment._mrax_dicom_metadata is metadata


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
        experiment, dataset, model, metadata = load_pulseq_experiment(
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
