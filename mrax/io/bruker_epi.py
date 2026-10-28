"""EPI event decoding and measured calibration for explicit reconstruction."""
from __future__ import annotations

import numpy as np

from mrax.data_struct.acquisition import AcquisitionData, ReconstructionRecipe

from .bruker_storage import decode_records
from .definitions import DefinitionLibrary


def read_epi_acquisition(folder, method, acqp, reco, visu, *, geometry=None, library=None, storage_profile=None, event_profile=None, readout_profile=None):
    from .bruker_images import _bruker_decoded_in_plane_axis_order, _visu_fg_order_entries
    from .bruker_sampling import (
        _apply_bruker_epi_alternating_readout_reversal,
        _bruker_epi_prefix_nav_points,
    )

    records = decode_records(folder, method, acqp, library=library, profile=storage_profile, retain_record=True)
    matrix = np.asarray(method["PVM_EpiMatrix"], int).ravel()
    read, phase = matrix[:2]
    partitions = int(matrix[2]) if len(matrix) > 2 and matrix[2] > 0 else 1
    ndim = int(acqp.get("ACQ_dim", 2))
    shots = int(method.get("PVM_EpiNShots", 1))
    if phase % shots:
        raise ValueError("EPI shots do not divide acquired phase lines")
    echoes = int(phase // shots)
    frames = int(acqp.get("NI", 1)) * int(acqp.get("NR", 1))
    if len(records.samples) != frames * partitions * shots:
        raise ValueError("EPI loops do not account for complete raw records")
    prefix, _ = _bruker_epi_prefix_nav_points(method, int(read), reco=reco)
    usable = records.samples.shape[-1] - prefix
    declared = int(method.get("PVM_EpiNEchoes", echoes))
    raw_echoes = declared
    used = raw_echoes * int(read)
    if usable < used or raw_echoes % echoes:
        raise ValueError("EPI imaging and calibration echo counts are inconsistent")
    object_count = raw_echoes // echoes
    library = library or DefinitionLibrary()
    profile = library.select("events", {**method, **acqp, "encoding": "epi",
        "dimensions": ndim, "echo_objects": object_count}, profile=event_profile)
    settings = profile.settings
    if set(settings) != {"resolver", "loop_order", "first_negative_echo", "image_object", "phase_correction", "object_combination"} or settings["resolver"] != "segmented_epi":
        raise ValueError("Unsupported EPI event resolver or settings")
    canonical = ("repeat", "partition", "shot", "coil", "echo", "read")
    loop_order = tuple(settings["loop_order"])
    if len(loop_order) != len(canonical) or set(loop_order) != set(canonical):
        raise ValueError("EPI event loops must name every acquisition axis exactly once")
    sizes = dict(zip(canonical, (frames, partitions, shots, records.samples.shape[1], raw_echoes, read), strict=True))
    data = records.samples[..., prefix:prefix + used].reshape(tuple(sizes[key] for key in loop_order)).transpose(tuple(loop_order.index(key) for key in canonical))
    image_object = int(settings["image_object"])
    combination = settings["object_combination"]
    negative_echo = int(settings["first_negative_echo"])
    if not 0 <= image_object < object_count or negative_echo not in (0, 1) or settings["phase_correction"] not in {"none", "entropy", "entropy_if_requested"}:
        raise ValueError("Invalid EPI readout object, polarity or phase correction")
    if combination not in {"selected", "magnitude_mean"}:
        raise ValueError("Unknown EPI object combination")
    selected_objects = range(object_count) if combination == "magnitude_mean" else (image_object,)
    repeats = len(selected_objects)
    auxiliary = dict(records.auxiliary_samples)
    if prefix:
        auxiliary["prefix_adc"] = records.samples[..., :prefix]
    if usable > used:
        auxiliary["trailing_adc"] = records.samples[..., prefix + used:]
    if records.source_kind == "job":
        data = data[:, ::-1]
    objects = int(acqp.get("NI", 1))
    order = np.asarray(acqp.get("ACQ_obj_order", list(range(objects))), int).ravel()
    if not np.array_equal(np.sort(order), np.arange(objects)):
        raise ValueError("EPI object order must be a complete permutation")
    data = data.reshape((-1, objects) + data.shape[1:])[:, np.argsort(order)].reshape(data.shape)
    corrected = []
    for item in data:
        item = _apply_bruker_epi_alternating_readout_reversal(item, first_echo=negative_echo)
        for object_index in selected_objects:
            corrected.append(item[..., object_index::object_count, :])
    original_frames = frames
    frames *= repeats
    data = np.stack(corrected)
    grid = tuple(reversed(tuple(int(v) for v in np.asarray(reco.get("RECO_ft_size", reco["RECO_size"])).ravel()[:ndim])))
    samples = np.zeros((frames, records.samples.shape[1]) + grid, dtype=data.dtype)
    mask = np.zeros(grid, bool)
    polarity = np.zeros((frames, 1) + grid, np.int8)
    read_start = grid[-1] // 2 - read // 2
    line_count = int(partitions * phase)
    sort_map = np.asarray(reco.get("RecoSortMaps", []), int).ravel()
    if records.source_kind == "job" and len(sort_map) >= line_count:
        destination = sort_map[:line_count]
    else:
        phase_steps = np.asarray(method["PVM_EncSteps1"], int).ravel() + grid[-2] // 2
        partition_steps = (np.asarray(method.get("PVM_EncSteps2", np.arange(partitions) - partitions // 2), int).ravel()
            + (grid[-3] // 2 if ndim == 3 else 0))
        destination = (partition_steps[:, None] * grid[-2] + phase_steps[None, :]).ravel()
    if len(destination) != line_count or np.any(destination < 0) or np.any(destination >= int(np.prod(grid[:-1]))):
        raise ValueError("EPI encoding table does not match the spatial grid")
    lines = data.transpose(0, 3, 1, 2, 4, 5).reshape((frames, records.samples.shape[1], line_count, read))
    flat = samples.reshape((frames, records.samples.shape[1], -1, grid[-1]))
    flat[:, :, destination, read_start:read_start + read] = lines
    mask.reshape((-1, grid[-1]))[destination, read_start:read_start + read] = True
    for repeat, object_index in enumerate(selected_objects):
        selected_echoes = np.arange(object_index, raw_echoes, object_count)
        signs = np.where(selected_echoes % 2 == negative_echo, -1, 1)
        polarity.reshape((frames, -1, grid[-1]))[repeat::repeats, destination] = np.tile(signs, partitions * shots)[None, :, None]
    # EPI's readout transform uses the opposite sign from the phase encodings.
    # Express that source convention as a DC-preserving coordinate permutation,
    # leaving a standard inverse Fourier transform for every internal axis.
    read_indices = (2 * (grid[-1] // 2) - np.arange(grid[-1])) % grid[-1]
    samples = samples[..., read_indices]
    mask = mask[..., read_indices]
    polarity = polarity[..., read_indices]
    readout_coordinates = None
    readout_definition = None
    if "PVM_EpiTrajAdjkx" in method:
        integrated = np.asarray(method["PVM_EpiTrajAdjkx"], float).ravel()
        if integrated.shape != (read,) or read != grid[-1] or not np.isfinite(integrated).all():
            raise ValueError("Measured EPI readout coordinates do not match the acquired ADC grid")
        fraction = float(method.get("PVM_EpiEchoPosition", 50.)) / 100
        if not 0 < fraction < 1:
            raise ValueError("EPI echo position must be a percentage strictly inside the acquisition")
        readout_definition = library.select("readout", {**method, **acqp,
            "encoding": "epi", "partial_echo": not np.isclose(fraction, .5)}, profile=readout_profile)
        settings_readout = readout_definition.settings
        if set(settings_readout) != {"resolver", "center"} or settings_readout["resolver"] != "integrated_gradient":
            raise ValueError("Unsupported EPI readout coordinate resolver")
        mode = settings_readout["center"]
        if mode == "trajectory_midpoint":
            center = (integrated.min() + integrated.max()) / 2
        elif mode == "nominal_echo_fraction":
            center = fraction * read
        else:
            raise ValueError("Unknown EPI readout coordinate origin")
        # Canonicalize the source's opposite readout FT sign along with the
        # samples. The two stored polarities have different coordinate orders.
        positive = integrated - center
        negative = center - integrated[::-1]
        readout_coordinates = -np.stack((positive, negative))[:, read_indices] / grid[-1]
        # +Nyquist and -Nyquist represent the same discrete Fourier coordinate.
        readout_coordinates = np.where(np.isclose(readout_coordinates, .5, atol=1e-12, rtol=0), -.5, readout_coordinates)
        if "PVM_EpiTrajAdjb0" in method:
            phase_trajectory = np.asarray(method["PVM_EpiTrajAdjb0"], float)
            if phase_trajectory.shape != (read,) or not np.isfinite(phase_trajectory).all():
                raise ValueError("EPI trajectory phase does not match the readout")
            auxiliary["readout_phase_reference"] = np.exp(1j * phase_trajectory)
    from .bruker_acquisition import _numbers
    scale = np.asarray(_numbers(reco.get("RecoScaleChan")), float)
    if scale.size:
        if len(scale) != samples.shape[1]:
            raise ValueError("EPI receiver scale count does not match samples")
        samples *= scale.reshape((1, len(scale)) + (1,) * ndim)
    rotate = np.asarray(_numbers(reco.get("RECO_rotate")), float)
    if rotate.size == ndim:
        rotate = rotate[:, None]
    elif rotate.size == ndim * original_frames:
        rotate = rotate.reshape((ndim, original_frames))
    else:
        raise ValueError("EPI rotation metadata does not match frames")
    shift = (rotate[::-1].T - .5) * np.asarray(grid)
    shift = np.repeat(np.broadcast_to(shift, (original_frames, ndim)), repeats, axis=0)
    slices = int(np.asarray(method.get("PVM_SPackArrNSlices", 1)).sum()) if ndim == 2 else 1
    transpose = np.asarray(reco.get("RECO_transposition", 0)).ravel()
    decoded_axes = _bruker_decoded_in_plane_axis_order(visu)
    permutations = tuple(tuple(1 - axis for axis in decoded_axes)[::(-1 if transpose[(i // repeats) % len(transpose)] else 1)] for i in range(frames)) if ndim == 2 else ()
    frame_indices = ()
    groups = _visu_fg_order_entries(visu)
    counts = [int(group[0]) for group in groups]
    slice_group = next((i for i, group in enumerate(groups) if len(group) > 1 and "FG_SLICE" in str(group[1]).upper()), None)
    indices = np.arange(original_frames)
    if slices > 1 and slice_group is not None and np.prod(counts) == original_frames:
        indices = indices.reshape(tuple(reversed(counts)))
        indices = np.moveaxis(indices, len(counts) - 1 - slice_group, -1).ravel()
    if repeats > 1 or slices > 1:
        grouped = indices.reshape((-1, slices))
        frame_indices = tuple((grouped[:, None, :] * repeats + np.arange(repeats)[None, :, None]).ravel().tolist())
    correction = settings["phase_correction"]
    if correction == "entropy_if_requested":
        correction = "entropy" if "RecoEpiGhostFilter" in str(reco.get("RecoStageNodes", "")) else "none"
    recipe = ReconstructionRecipe(frame_image_shifts=tuple(tuple(float(x) for x in row) for row in np.broadcast_to(shift, (frames, ndim))),
        output_shape=tuple(reversed(tuple(int(v) for v in np.asarray(reco["RECO_size"]).ravel()[:ndim]))),
        output_permutation=(2, 1, 0) if ndim == 3 else (), frame_permutations=permutations,
        stack_slices=slices, output_frame_indices=frame_indices,
        average_repeats=repeats,
        output_scale=1/np.sqrt(np.prod(grid)), epi_phase_correction=correction)
    return AcquisitionData(samples, grid, tuple(f"encoding_{i}" for i in range(ndim)),
        acquired=mask, geometry=geometry, recipe=recipe, readout_polarity=polarity,
        readout_coordinates=readout_coordinates, auxiliary_samples=auxiliary,
        frame_coordinates={"readout_repeat": np.tile(np.asarray(tuple(selected_objects)), original_frames)}), {
        "raw_status": "prepared", "storage_profile": records.profile,
        "event_profile": profile.name, "alternate_readout_count": object_count - 1,
        "object_combination": combination, "reconstructed_repeats": repeats,
        "trailing_complex_points": int(usable - used),
        "reconstruction_hints": {"epi_phase_correction": correction},
        "readout_profile": None if readout_definition is None else readout_definition.name,
        "unresolved_operations": ["navigator_stability", "trajectory_phase"],
        "auxiliary_roles": list(auxiliary)}
