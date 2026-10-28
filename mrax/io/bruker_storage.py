"""Byte-accounted Bruker storage decoding, independent of image reconstruction."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .definitions import DefinitionLibrary


@dataclass(frozen=True)
class DecodedRecords:
    samples: np.ndarray  # record, receiver, complex readout point
    profile: str
    source_kind: str
    padding_bytes: int
    auxiliary_samples: dict[str, np.ndarray] = field(default_factory=dict)


def _flat(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        return [v for item in value for v in _flat(item)]
    return [value]


def scalar_dtype(acqp, *, byte_order=None):
    token = acqp.get("GO_raw_data_format")
    types = {"GO_16BIT_SGN_INT": "i2", "GO_32BIT_SGN_INT": "i4", "GO_32BIT_FLOAT": "f4"}
    if token is None:
        width = acqp.get("ACQ_word_size")
        types.update({"_8_BIT": "i1", "_16_BIT": "i2", "_32_BIT": "i4", "_64_BIT": "i8"})
        token = width
    if token not in types:
        raise ValueError("Raw scalar type is absent or unsupported; image types are not raw types")
    order = str(byte_order or acqp.get("ACQ_byte_order") or acqp.get("BYTORDA") or "little").lower()
    return np.dtype(types[token]).newbyteorder(">" if order.startswith("big") else "<")


def job_descriptor(acqp, index=0):
    value = acqp.get("ACQ_jobs")
    if not isinstance(value, (list, tuple)) or not value:
        return None
    if isinstance(value[0], (list, tuple)):
        return value[index] if index < len(value) else None
    return value if index == 0 else None


def discover_raw_stream(folder, acqp):
    """Select a declared imaging job or a declared FID stream."""
    folder = Path(folder)
    descriptor = job_descriptor(acqp)
    if descriptor and isinstance(descriptor[0], (int, float)) and descriptor[0] > 0:
        names = [str(descriptor[-1])] if isinstance(descriptor[-1], str) else []
        names = [f"rawdata.{name}" for name in names] + ["rawdata.job0", "rawdat.job0"]
        candidates = list(dict.fromkeys(folder / name for name in names if (folder / name).is_file()))
        if len(candidates) > 1:
            raise ValueError("Multiple declared primary job streams require explicit selection")
        if candidates:
            return candidates[0], "job"
    if (folder / "fid").is_file():
        return folder / "fid", "fid"
    return None, None


def decode_records(folder, method, acqp, *, library=None, profile=None, retain_record=False):
    """Decode complete records and validate scalar and record sizes."""
    library = library or DefinitionLibrary()
    path, kind = discover_raw_stream(folder, acqp)
    if path is None:
        raise FileNotFoundError("No declared primary raw stream is available")
    receivers = int(method.get("PVM_EncNReceivers") or method.get("PVM_NReceivers") or 1)
    if receivers <= 0:
        raise ValueError("Receiver count must be positive")
    descriptor = job_descriptor(acqp)
    features = {**method, **acqp, "stream": kind, "descriptor_valid": bool(descriptor and descriptor[0] > 0),
        "block_format": acqp.get("GO_block_size", "Standard_KBlock_Format")}
    profile = library.select("storage", features, profile=profile)
    if set(profile.settings) != {"reader", "alignment_bytes"} or profile.settings["reader"] not in {"fid_records", "job_records"}:
        raise ValueError("Unsupported raw storage reader or settings")
    dtype = scalar_dtype(acqp)
    if path.stat().st_size % dtype.itemsize:
        raise ValueError("Raw byte count is not a whole number of scalar words")
    if kind == "job":
        scalar_points = int(descriptor[0])
    else:
        scalar_points = int(_flat(acqp.get("ACQ_size", []))[0])
    if scalar_points <= 0 or scalar_points % 2:
        raise ValueError("Complex readouts require a positive even scalar count")
    record_bytes = scalar_points * receivers * dtype.itemsize
    alignment = int(profile.settings["alignment_bytes"])
    if alignment < 1 or alignment % dtype.itemsize and alignment != 1:
        raise ValueError("Raw alignment must be positive and compatible with scalar words")
    stride = ((record_bytes + alignment - 1) // alignment) * alignment
    total_bytes = path.stat().st_size
    if total_bytes % stride:
        raise ValueError("Raw file contains an incomplete or inconsistent record")
    raw = np.memmap(path, dtype=dtype, mode="r").reshape((-1, stride // dtype.itemsize))
    padding = raw[:, scalar_points * receivers:]
    if padding.size and np.any(padding):
        raise ValueError("Declared record padding contains nonzero samples")
    active_points = scalar_points
    active_start = 0
    if not retain_record and kind == "fid" and features["block_format"] == "continuous" and method.get("PVM_DigNp"):
        active_points = 2 * int(method["PVM_DigNp"])
        if active_points > scalar_points or active_points <= 0:
            raise ValueError("Digitizer length exceeds the declared record")
        if str(acqp.get("ACQ_scan_size", "")).lower() == "acq_phase_factor_scans" and int(acqp.get("ACQ_scan_shift", 0)) < 0:
            active_start = scalar_points - active_points
    full_words = raw[:, :scalar_points * receivers].reshape((-1, receivers, scalar_points))
    words = full_words[..., active_start:active_start + active_points]
    # Integer measurements need a double-precision host view to preserve counts.
    floating = np.float64 if dtype.kind in "iu" and dtype.itemsize >= 4 else np.float32
    samples = words[..., 0::2].astype(floating) + 1j * words[..., 1::2].astype(floating)
    auxiliary = {}
    if active_points != scalar_points:
        excluded = np.ones(scalar_points // 2, bool)
        excluded[active_start // 2:(active_start + active_points) // 2] = False
        inactive = full_words.reshape(full_words.shape[:-1] + (-1, 2))[..., excluded, :]
        auxiliary["inactive_adc"] = inactive[..., 0].astype(floating) + 1j * inactive[..., 1].astype(floating)
    return DecodedRecords(samples, profile.name, kind, stride - record_bytes, auxiliary)


def coordinate_event_map(records, method, acqp, *, event_count, profile):
    """Map explicit encoding-coordinate events in their declared acquisition order."""
    objects, repeats = int(acqp.get("NI", 1)), int(acqp.get("NR", 1))
    group = int(acqp.get("ACQ_phase_factor", 1))
    if records.source_kind == "job":
        group = max(group, int(acqp.get("NECHOES") or 1) // int(method.get("PVM_NEchoImages") or 1))
    if group <= 0 or event_count % group:
        raise ValueError("Encoding events must form complete acquisition groups")
    canonical = ("repeat", "event_group", "object", "within_group", "coil", "read")
    order = tuple(profile.settings["loop_order"])
    if len(order) != len(canonical) or set(order) != set(canonical):
        raise ValueError("Coordinate-event loops must name every acquisition axis exactly once")
    sizes = dict(zip(canonical, (repeats, event_count // group, objects, group) + records.samples.shape[1:], strict=True))
    shape = tuple(sizes[key] for key in order)
    if np.prod(shape) != records.samples.size:
        raise ValueError("Encoding events do not account for the complete raw stream")
    data = records.samples.reshape(shape).transpose(tuple(order.index(key) for key in canonical))
    data = data.transpose(0, 2, 4, 1, 3, 5).reshape((repeats, objects, records.samples.shape[1], event_count, records.samples.shape[-1]))
    object_order = np.asarray(_flat(acqp.get("ACQ_obj_order", list(range(objects)))), int)
    if not np.array_equal(np.sort(object_order), np.arange(objects)):
        raise ValueError("Acquisition object order must be a complete permutation")
    return data[:, np.argsort(object_order)].reshape((repeats * objects,) + data.shape[2:])


def cartesian_event_map(records, method, acqp, *, phase_count, partition_count=1, profile=None):
    """Resolve default object/phase-group loops from acquisition declarations."""
    samples = records.samples
    objects = int(acqp.get("NI", 1))
    repeats = int(acqp.get("NR", 1))
    group = int(acqp.get("ACQ_phase_factor", 1))
    if records.source_kind == "job":
        echo_images = int(method.get("PVM_NEchoImages") or 1)
        echoes = int(acqp.get("NECHOES") or 1)
        group = max(group, echoes // echo_images)
    if phase_count % group:
        raise ValueError("Phase-group length does not divide the encoding table")
    canonical = ("repeat", "partition", "phase_group", "object", "within_group", "coil", "read")
    loop_order = canonical if profile is None else tuple(profile.settings["loop_order"])
    if len(loop_order) != len(canonical) or set(loop_order) != set(canonical):
        raise ValueError("Event loop definition must identify every acquisition axis exactly once")
    dimensions = dict(zip(canonical, (repeats, partition_count, phase_count // group, objects, group) + samples.shape[1:], strict=True))
    shape = tuple(dimensions[key] for key in loop_order)
    if int(np.prod(shape)) != samples.size:
        raise ValueError("Acquisition loop counts do not account for the complete raw stream")
    data = samples.reshape(shape).transpose(tuple(loop_order.index(key) for key in canonical))
    # repetition, object, receiver, partition, phase group, within-group, read
    data = np.transpose(data, (0, 3, 5, 1, 2, 4, 6))
    data = data.reshape((repeats, objects, samples.shape[1], partition_count, phase_count, samples.shape[-1]))
    fallback_order = _flat(method.get("PVM_ObjOrderList", []))
    if len(fallback_order) != objects:
        fallback_order = list(range(objects))
    order = np.asarray(_flat(acqp.get("ACQ_obj_order", fallback_order)), int)
    if order.size != objects or not np.array_equal(np.sort(order), np.arange(objects)):
        raise ValueError("Acquisition object order is not a complete permutation")
    return data[:, np.argsort(order)].reshape((repeats * objects, samples.shape[1], partition_count, phase_count, samples.shape[-1]))
