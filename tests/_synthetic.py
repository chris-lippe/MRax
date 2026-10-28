"""Small file-format fixtures generated independently from NumPy arrays and headers."""


import ismrmrd
import numpy as np
from ismrmrd import xsd
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid


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


def _header(nx=8, ny=6, trajectory='cartesian'):
    header = xsd.ismrmrdHeader(experimentalConditions=xsd.experimentalConditionsType(H1resonanceFrequency_Hz=128000000))
    header.encoding = [xsd.encodingType(
        encodedSpace=xsd.encodingSpaceType(matrixSize=xsd.matrixSizeType(x=nx, y=ny, z=1), fieldOfView_mm=xsd.fieldOfViewMm(x=80, y=60, z=4)),
        reconSpace=xsd.encodingSpaceType(matrixSize=xsd.matrixSizeType(x=nx, y=ny, z=1), fieldOfView_mm=xsd.fieldOfViewMm(x=80, y=60, z=4)),
        encodingLimits=xsd.encodingLimitsType(kspace_encoding_step_1=xsd.limitType(minimum=0, maximum=ny - 1, center=ny // 2)),
        trajectory=xsd.trajectoryType(trajectory))]
    header.sequenceParameters = xsd.sequenceParametersType(TE=[12., 36.], TR=[1000.])
    return header


def _write_mrd(path, *, sampled=False, bad_trajectory=False, empty=False):
    header = _header(trajectory='radial' if sampled else 'cartesian')
    dataset = ismrmrd.Dataset(str(path), 'dataset')
    dataset.write_xml_header(xsd.ToXML(header))
    if empty:
        dataset.close()
        return
    for contrast in range(2):
        for line in [3, 0, 5, 1, 4, 2]:
            acq = ismrmrd.Acquisition()
            acq.resize(8, 2, 2 if sampled else 0)
            acq.idx.kspace_encode_step_1 = line
            acq.idx.contrast = contrast
            acq.center_sample = 4
            acq.read_dir[:] = [1, 0, 0]
            acq.phase_dir[:] = [0, 1, 0]
            acq.slice_dir[:] = [0, 0, 1]
            acq.data[:] = (contrast + 1) * (line + 1)
            if line == 0:
                acq.data[0] = 0  # a measured zero remains acquired
            if sampled:
                phi = line * np.pi / 6
                radius = (np.arange(8) - 4).astype(np.float32)
                acq.traj[:] = np.column_stack([radius * np.cos(phi), radius * np.sin(phi)])
                if bad_trajectory and contrast:
                    acq.traj[:] += .1
            dataset.append_acquisition(acq)
    dataset.close()
