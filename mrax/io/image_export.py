"""Dimension-aware image exports shared by scripts and the desktop GUI."""

from __future__ import annotations

from collections.abc import Mapping
from itertools import combinations
from pathlib import Path

import numpy as np

from mrax.data_struct.geometry import SpatialGeometry
from mrax.io.nifti import mrax_to_nifti_volume, save_nifti


def image_frames(array, dims=None):
    """Yield (index, suffix, spatial volume), retaining singleton spatial axes."""
    arr = np.asarray(array)
    if np.iscomplexobj(arr):
        raise ValueError("Choose a real image component before image export")
    if isinstance(dims, Mapping):
        names = list(dims)
        spatial = [name for name in names if name in ("x", "y", "z")]
        spatial_shape = tuple(int(dims[name]) for name in spatial)
        if tuple(arr.shape) == tuple(dims.values()):
            pass
        elif spatial_shape == arr.shape:
            names = spatial
        elif spatial and arr.shape == (int(np.prod([dims[n] for n in names if n not in spatial])),) + spatial_shape:
            arr = arr.reshape(tuple(dims.values()))
        elif spatial_shape and arr.shape[-len(spatial_shape):] == spatial_shape:
            leading = arr.shape[:-len(spatial_shape)]
            non_spatial = [name for name in names if name not in spatial]
            candidates = [axes for axes in combinations(non_spatial, len(leading))
                          if tuple(dims[name] for name in axes) == leading]
            if len(candidates) != 1:
                repeats = [axes for axes in candidates if all(name in ("t", "frame") for name in axes)]
                candidates = repeats if len(repeats) == 1 else candidates
            if len(candidates) != 1:
                raise ValueError("Ambiguous map dimensions; provide the map's explicit dimension names")
            names = list(candidates[0]) + spatial
        else:
            raise ValueError("Image dimensions do not match the exported array")
    else:
        names = list(dims or ())
        if not names and arr.ndim in (2, 3):
            names = list(("x", "y", "z")[:arr.ndim])
        if len(names) != arr.ndim:
            raise ValueError("Name spatial and frame dimensions before image export")
    spatial_axes = [i for i, name in enumerate(names) if name in ("x", "y", "z")]
    if len(spatial_axes) not in (2, 3):
        raise ValueError("Image export requires two or three spatial dimensions")
    frame_axes = [i for i in range(arr.ndim) if i not in spatial_axes]
    frame_shape = tuple(arr.shape[i] for i in frame_axes)
    for flat, indices in enumerate(np.ndindex(frame_shape)):
        selection = [slice(None)] * arr.ndim
        labels = []
        for axis, index in zip(frame_axes, indices):
            selection[axis] = index
            if arr.shape[axis] > 1:
                labels.append(f"{names[axis]}-{index:04d}")
        yield flat, "_".join(labels), np.asarray(arr[tuple(selection)])


def export_image_files(path, array, *, dims=None, affine=None, geometry=None, format="nifti", unit="", description="MRax image"):
    """Write one NIfTI per spatial frame, or one DICOM series per frame."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(geometry, Mapping):
        geometry = SpatialGeometry.from_metadata(geometry)
    default_affine = np.eye(4) if affine is None else np.asarray(affine, dtype=float)
    outputs = []
    study_uid = reference_uid = None
    for index, suffix, volume in image_frames(array, dims):
        frame_affine = geometry.affine_for_frame(index) if geometry is not None else None
        matrix = default_affine if frame_affine is None else frame_affine
        stem = path.name[:-7] if path.name.endswith(".nii.gz") else path.stem
        stem += "_" + suffix if suffix else ""
        if format == "nifti":
            extension = ".nii.gz" if path.name.endswith(".nii.gz") else ".nii"
            output = path.with_name(stem + extension)
            exported = volume.astype(np.uint8) if volume.dtype == bool else volume
            save_nifti(output, exported, matrix, dtype=exported.dtype)
            outputs.append(output)
        elif format == "dicom":
            files, study_uid, reference_uid = _export_dicom_frame(path.with_name(stem + ".dcm"), volume, matrix,
                                                                 unit, description, study_uid, index, reference_uid)
            outputs.extend(files)
        else:
            raise ValueError("Image format must be 'nifti' or 'dicom'")
    return outputs


def _export_dicom_frame(path, volume, affine, unit, description, study_uid, frame_index, reference_uid):
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid
    from pydicom.valuerep import format_number_as_ds

    if not np.isfinite(volume).all():
        raise ValueError("DICOM export requires finite image values")
    # NIfTI XYZ/RAS -> DICOM columns, rows, slices / LPS.
    xyz = mrax_to_nifti_volume(volume)
    lps = np.diag([-1., -1., 1., 1.]) @ np.asarray(affine, dtype=float)
    spacing = np.linalg.norm(lps[:3, :3], axis=0)
    if not np.all(np.isfinite(lps)) or np.any(spacing <= 0):
        raise ValueError("DICOM export requires a finite, nondegenerate affine")
    directions = lps[:3, :3] / spacing
    if not np.allclose(directions.T @ directions, np.eye(3), atol=1e-5):
        raise ValueError("DICOM export requires orthogonal image axes; resample sheared geometry first")
    lo, hi = float(np.min(xyz)), float(np.max(xyz))
    integer = np.all(xyz == np.rint(xyz))
    unsigned = integer and lo >= 0 and hi > 32767 and hi <= 65535
    if integer and ((lo >= -32768 and hi <= 32767) or unsigned):
        slope, intercept = 1., 0.
    else:
        slope = (hi - lo) / 65534 if hi > lo else 1.
        intercept = (hi + lo) / 2
    encoded = np.rint((xyz - intercept) / slope).astype("<u2" if unsigned else "<i2")
    study_uid = study_uid or generate_uid()
    series_uid, frame_uid = generate_uid(), reference_uid or generate_uid()
    paths = []
    for z in range(xyz.shape[2]):
        output = path if xyz.shape[2] == 1 else path.with_name(f"{path.stem}_slice-{z:04d}.dcm")
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = MRImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = FileDataset(str(output), {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SOPClassUID, ds.SOPInstanceUID = meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID
        ds.StudyInstanceUID, ds.SeriesInstanceUID, ds.FrameOfReferenceUID = study_uid, series_uid, frame_uid
        ds.PatientName, ds.PatientID = "", ""
        ds.PatientBirthDate, ds.PatientSex = "", ""
        ds.StudyDate, ds.StudyTime, ds.AccessionNumber, ds.StudyID = "", "", "", ""
        ds.Modality, ds.ImageType = "MR", ["DERIVED", "SECONDARY"]
        ds.SeriesDescription = description[:64]
        ds.SeriesNumber, ds.InstanceNumber = frame_index + 1, z + 1
        ds.ImageOrientationPatient = [format_number_as_ds(float(value)) for value in np.concatenate([directions[:, 0], directions[:, 1]])]
        ds.ImagePositionPatient = [format_number_as_ds(float(value)) for value in (lps @ [0, 0, z, 1])[:3]]
        ds.PixelSpacing = [format_number_as_ds(float(spacing[1])), format_number_as_ds(float(spacing[0]))]
        ds.SliceThickness = format_number_as_ds(float(spacing[2]))
        ds.ScanningSequence, ds.SequenceVariant, ds.ScanOptions = "RM", "NONE", ""
        ds.MRAcquisitionType, ds.RepetitionTime, ds.EchoTime, ds.EchoTrainLength = "", "", "", ""
        ds.Rows, ds.Columns = xyz.shape[1], xyz.shape[0]
        ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0 if unsigned else 1
        ds.RescaleSlope, ds.RescaleIntercept, ds.RescaleType = format_number_as_ds(slope), format_number_as_ds(intercept), unit or "US"
        ds.PixelData = encoded[:, :, z].T.copy().tobytes()
        ds.save_as(output, enforce_file_format=True)
        paths.append(output)
    return paths, study_uid, frame_uid
