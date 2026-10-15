"""Pure translation of Bruker storage and metadata into the acquisition contract."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from mrax.data_struct.acquisition import AcquisitionData, ReconstructionRecipe

from .bruker_storage import cartesian_event_map, coordinate_event_map, decode_records
from .definitions import DefinitionLibrary


def _numbers(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple, np.ndarray)):
        return [x for item in value for x in _numbers(item)]
    return [float(value)]


def _series(value, count, default=0):
    data = np.asarray(_numbers(value), float)
    if data.size == 0:
        return np.full(count, default)
    if count % len(data):
        raise ValueError("Metadata series does not match acquisition frames")
    return np.tile(data, count // len(data))


def read_bruker_acquisition(folder, *, pdata_index="1", definition_paths=(), profiles=None):
    """Read prepared samples and geometry without reconstructing images.

    Calibration and source loop interpretation are resolved here; image reconstruction only
    consumes the returned vendor-independent acquisition contract.
    """
    from .bruker_images import (
        _bruker_decoded_in_plane_axis_order,
        _bruker_has_mixed_slice_geometry,
        _visu_fg_count,
        bruker_spatial_geometry,
    )
    from .bruker_parameters import parse_jcamp
    from .bruker_sampling import (
        _apply_bruker_b0_correction,
        _apply_bruker_channel_corrections,
        _bruker_inverse_fid_active,
        _bruker_regrid_metadata,
        _non_cartesian_segment_keep_indices,
        _reshape_bruker_non_cartesian_blocks,
        read_bruker_regrid_density,
        read_bruker_regrid_trajectory,
    )

    folder = Path(folder)
    profiles = profiles or {}
    if set(profiles) - {"storage", "events", "readout"}:
        raise ValueError("Unknown import profile role")
    method, acqp = parse_jcamp(folder / "method"), parse_jcamp(folder / "acqp")
    pdata = folder / "pdata" / str(pdata_index)
    reco, visu = parse_jcamp(pdata / "reco"), parse_jcamp(pdata / "visu_pars")
    image_frames = int(visu.get("VisuCoreFrameCount", 1))
    visu["_mrax_mixed_slice_geometry"] = _bruker_has_mixed_slice_geometry(visu, image_frames)
    sizes = tuple(int(v) for v in _numbers(visu.get("VisuCoreSize")))
    if len(sizes) == 2:
        sizes = tuple(sizes[i] for i in _bruker_decoded_in_plane_axis_order(visu))
        slices = _visu_fg_count(visu, "FG_SLICE")
        if slices and not _bruker_has_mixed_slice_geometry(visu, image_frames):
            sizes += (int(slices),)
    geometry = bruker_spatial_geometry(visu, spatial_shape=sizes)
    regrid = _bruker_regrid_metadata(method, reco)
    if method.get("PVM_EpiMatrix"):
        from .bruker_epi import read_epi_acquisition
        return read_epi_acquisition(folder, method, acqp, reco, visu, geometry=geometry,
            library=DefinitionLibrary(definition_paths), storage_profile=profiles.get("storage"),
            event_profile=profiles.get("events"), readout_profile=profiles.get("readout"))
    if regrid is None:
        return read_cartesian_acquisition(folder, method, acqp, reco, visu,
            geometry=geometry, definition_paths=definition_paths, profiles=profiles)
    library = DefinitionLibrary(definition_paths)
    delayed = _bruker_inverse_fid_active(reco)
    events = library.select("events", {**method, **acqp, "encoding": "sampled", "delayed_readout": delayed}, profile=profiles.get("events"))
    if set(events.settings) != {"resolver", "loop_order"}:
        raise ValueError("Unsupported sampled event settings")
    records = decode_records(folder, method, acqp, library=library, profile=profiles.get("storage"))
    frames = int(acqp.get("NI", 1)) * int(acqp.get("NR", 1))
    trajectory, _ = read_bruker_regrid_trajectory(folder, method=method, reco=reco)
    readout_model, auxiliary = None, dict(records.auxiliary_samples)
    if delayed:
        from .bruker_readout import prepare_delayed_readouts

        if events.settings["resolver"] != "delayed_profile_pairs":
            raise ValueError("Delayed ADC readouts require a compatible profile resolver")
        samples, trajectory, readout_model, auxiliary = prepare_delayed_readouts(records,
            trajectory, folder=folder, method=method, acqp=acqp, reco={**reco, "_pdata_index": str(pdata_index)},
            loop_order=events.settings["loop_order"])
        sorting = {"ignored_scan_count": 0}
    else:
        if events.settings["resolver"] != "trajectory_profiles":
            raise ValueError("Sampled trajectories require a compatible profile resolver")
        samples, sorting = _reshape_bruker_non_cartesian_blocks(records.samples,
            frame_count=frames, trajectory_samples=len(trajectory), method=method,
            reco=reco, regrid_meta=regrid, loop_order=events.settings["loop_order"])
        if records.samples.shape[-1] != int(regrid["points_per_interleave"]):
            kept = _non_cartesian_segment_keep_indices(raw_points_per_interleave=records.samples.shape[-1],
                kept_points_per_interleave=int(regrid["points_per_interleave"]), interleaves=1, method=method, reco=reco)
            excluded = np.ones(records.samples.shape[-1], bool)
            excluded[kept] = False
            auxiliary["unmapped_adc"] = records.samples[..., excluded]
    if sorting["ignored_scan_count"]:
        # A reversible raw import is still possible when a reconstruction graph
        # describes additional profile groups whose roles are not established.
        # Preserve every ADC record rather than discarding it or labelling an
        # approximate inverse-FID transform as measured Fourier data.
        return AcquisitionData(records.samples, (records.samples.shape[-1],),
            ("readout_sample",), domain="adc", geometry=geometry, auxiliary_samples=auxiliary), {
            "raw_status": "decoded", "storage_profile": records.profile,
            "event_profile": "unresolved_sampled_profiles", "raw_records_preserved": len(records.samples),
            "preparation_required": ["profile_roles", "inverse_fid_encoding"]}
    inverse = sorting.get("profile_preprocessing", {}).get("inverse_fid", {})
    if readout_model is None:
        samples, _ = _apply_bruker_b0_correction(samples, exp_dir=folder, reco=reco, regrid_meta=regrid)
    samples, _ = _apply_bruker_channel_corrections(samples, reco, min_ndim=3)
    ndim = int(regrid["dim"])
    output = tuple(int(v) for v in _numbers(reco.get("RECO_size"))[:ndim])
    grid = tuple(int(v) for v in _numbers(reco.get("RecoRegridNFtSize"))[:ndim]) or output
    if readout_model is None:
        density, _ = read_bruker_regrid_density(folder, trajectory, spatial_shape=output, method=method, reco=reco)
    else:
        from mrax.reco.processing import estimate_radial_density

        density = estimate_radial_density(trajectory, spatial_shape=output)
    interleaves = int(regrid["interleaves"])
    if readout_model is not None:
        angular_scale = 2 * interleaves * ndim * 2**ndim / (4 * np.pi if ndim == 3 else 2 * np.pi)
    elif str(reco.get("RecoRegridNTrajType", "")).lower().endswith("radial"):
        angular_scale = interleaves * ndim * 2**ndim / (4 * np.pi if ndim == 3 else 2 * np.pi)
    else:
        angular_scale = interleaves / (2 * np.pi)
    stages = str(reco.get("RecoStageNodes", ""))
    slice_sizes = _numbers(method.get("PVM_SPackArrNSlices"))
    slices = int(slice_sizes[0]) if ndim == 2 and len(slice_sizes) == 1 and slice_sizes[0] > 1 else 1
    rotate = _numbers(reco.get("RECO_rotate"))
    shift = tuple((value - .5) * size for value, size in zip(rotate, grid, strict=True)) if len(rotate) == ndim else ()
    recipe = ReconstructionRecipe(gridding_shape=grid, output_shape=output,
        profile_inversion=readout_model is not None,
        image_shift=shift,
        output_scale=1 / float(np.sqrt(np.prod(grid))),
        encoding_scale=angular_scale / float(np.sqrt(np.prod(output))),
        gridding_kernel_width=float(reco.get("RecoRegridNKernelWidth", 3.)),
        gridding_shape_parameter=float(reco.get("RecoRegridNKernelShapeParameter", 6.)),
        deapodize="RecoRegridApodCorrFilter" in stages,
        deapodization_normalization="inverse_fft",
        checkerboard="RecoPhasCorrFilter" in stages,
        elliptical_support="RecoMaskFilter" in stages,
        coil_combine="rss" if "RecoSumOfSquaresFilter" in stages else "sum",
        stack_slices=slices, output_permutation=(1, 0) if ndim == 2 else ())
    return AcquisitionData(samples, output, tuple(f"encoding_{i}" for i in range(ndim)),
        kind="sampled", trajectory=trajectory, density=density, geometry=geometry,
        readout_model=readout_model, auxiliary_samples=auxiliary,
        recipe=recipe), {"storage_profile": records.profile, "event_profile": events.name,
        "raw_status": "prepared", "approximate_preprocessing": bool(inverse.get("applied"))}


def read_cartesian_acquisition(folder, method, acqp, reco, visu, *, geometry=None, definition_paths=(), profiles=None):
    """Decode Cartesian imaging without Fourier transforms or image reconstruction."""
    library = DefinitionLibrary(definition_paths)
    ndim = int(acqp.get("ACQ_dim") or len(_numbers(method.get("PVM_EncMatrix"))))
    if ndim not in (1, 2, 3):
        raise ValueError("Only one, two or three encoding dimensions are supported")
    profiles = profiles or {}
    event_phase = np.asarray(_numbers(method.get("PVM_EncGenSteps1")), int)
    event_partition = np.asarray(_numbers(method.get("PVM_EncGenSteps2")), int)
    explicit_events = bool(event_phase.size > 1 and int(method.get("PVM_EncGenTotalSteps") or 0) == event_phase.size)
    events = library.select("events", {**method, **acqp, "encoding": "cartesian", "dimensions": ndim,
        "explicit_events": explicit_events}, profile=profiles.get("events"))
    if set(events.settings) != {"resolver", "loop_order"} or events.settings["resolver"] not in {"object_phase_groups", "coordinate_events"}:
        raise ValueError("Unsupported event resolver or settings")
    records = decode_records(Path(folder), method, acqp, library=library, profile=profiles.get("storage"))
    matrix = [int(x) for x in _numbers(method.get("PVM_EncMatrix"))]
    ft_size = [int(x) for x in _numbers(reco.get("RECO_ft_size") or reco.get("RECO_size"))]
    if not ft_size:
        ft_size = matrix or [records.samples.shape[-1]]
    grid = tuple(reversed(ft_size[:ndim]))
    if ndim == 1:
        samples = records.samples
        return AcquisitionData(samples, (samples.shape[-1],), ("spectral",), domain="adc", geometry=geometry, auxiliary_samples=records.auxiliary_samples), {
            "storage_profile": records.profile, "domain": "adc", "raw_status": "decoded"}
    phase_steps = np.asarray(_numbers(method.get("PVM_EncSteps1")), int)
    if not phase_steps.size:
        phase_steps = np.arange(matrix[1]) - matrix[1] // 2
    partition_steps = np.asarray(_numbers(method.get("PVM_EncSteps2")), int) if ndim == 3 else np.asarray([0])
    if not partition_steps.size:
        partition_steps = np.arange(matrix[2]) - matrix[2] // 2
    if explicit_events:
        if ndim == 3 and event_partition.size != event_phase.size:
            raise ValueError("Explicit encoding coordinates must have equal event counts")
        phase_steps = event_phase
        partition_steps = event_partition if ndim == 3 else np.zeros(len(event_phase), int)
        ordered = coordinate_event_map(records, method, acqp, event_count=len(event_phase), profile=events)
    else:
        ordered = cartesian_event_map(records, method, acqp, phase_count=len(phase_steps), partition_count=len(partition_steps), profile=events)
    echo_images = int(method.get("PVM_NEchoImages") or 1)
    if str(method.get("EchoAcqMode", "")) == "allEchoes" and echo_images > 1:
        negative_readouts = np.arange(ordered.shape[0]) % echo_images % 2 == 1
        ordered[negative_readouts] = ordered[negative_readouts, ..., ::-1]
    receivers = ordered.shape[1]
    channel_scale = np.asarray(_numbers(reco.get("RecoScaleChan")), float)
    channel_phase = np.asarray(_numbers(reco.get("RecoPhaseChan")), float)
    for values, name in ((channel_scale, "channel scale"), (channel_phase, "channel phase")):
        if values.size not in (0, receivers):
            raise ValueError(f"Receiver count does not match {name}")
    calibration = ((channel_scale if channel_scale.size else np.ones(receivers)) *
        np.exp(1j * np.deg2rad(channel_phase if channel_phase.size else np.zeros(receivers))))
    ordered = ordered * calibration.reshape((1, receivers) + (1,) * (ordered.ndim - 2))
    read = ordered.shape[-1]
    if read > grid[-1]:
        raise ValueError("Readout exceeds the declared encoding grid")
    read_start = grid[-1] // 2 - read // 2
    pft = _numbers(method.get("PVM_EncPft"))
    if pft and pft[0] < 1:
        read_start = grid[-1] - read
    destination_phase = phase_steps + grid[-2] // 2
    destination_partition = partition_steps + grid[-3] // 2 if ndim == 3 else np.asarray([0])
    if np.any(destination_phase < 0) or np.any(destination_phase >= grid[-2]):
        raise ValueError("Phase encoding indices exceed the declared grid")
    if ndim == 3 and (np.any(destination_partition < 0) or np.any(destination_partition >= grid[-3])):
        raise ValueError("Partition encoding indices exceed the declared grid")
    samples = np.zeros(ordered.shape[:2] + grid, dtype=ordered.dtype)
    acquired = np.zeros(grid, bool)
    read_slice = slice(read_start, read_start + read)
    if explicit_events:
        coordinates = (destination_partition, destination_phase) if ndim == 3 else (destination_phase,)
        if len(np.unique(np.stack(coordinates, axis=1), axis=0)) != len(phase_steps):
            raise ValueError("Repeated encoding events need an explicit averaging or independent-frame definition")
        samples[(slice(None), slice(None)) + coordinates + (read_slice,)] = ordered
        acquired[coordinates + (read_slice,)] = True
    elif ndim == 2:
        samples[:, :, destination_phase, read_slice] = ordered[:, :, 0]
        acquired[destination_phase, read_slice] = True
    else:
        for source, destination in enumerate(destination_partition):
            samples[:, :, destination, destination_phase, read_slice] = ordered[:, :, source]
            acquired[destination, destination_phase, read_slice] = True
    count = samples.shape[0]
    # Both file image formatting and reconstruction presentation are explicit
    # transformations. Their composition, not either setting in isolation,
    # maps reconstruction coordinates into the existing image-container layout.
    reconstruction_transpose = _series(reco.get("RECO_transposition"), count).astype(int)
    image_transpose = _series(visu.get("VisuCoreTransposition"), count).astype(int)
    from .bruker_images import _bruker_decoded_in_plane_axis_order

    decoded_base = _bruker_decoded_in_plane_axis_order({**visu,
        "VisuCoreTransposition": 0, "_mrax_frame_transposition": 0})
    base_permutation = tuple(1 - axis for axis in decoded_base)
    permutations = tuple(base_permutation[::-1] if bool(a) != bool(b) else base_permutation
        for a, b in zip(reconstruction_transpose, image_transpose)) if ndim == 2 else ()
    if ndim == 3:
        volume_permutations = []
        for code in reconstruction_transpose:
            if not 0 <= code <= ndim:
                raise ValueError("Volume transposition code exceeds the encoding dimensions")
            permutation = list(range(ndim))
            if code:
                first, second = int(code - 1), int(code % ndim)
                permutation[first], permutation[second] = permutation[second], permutation[first]
            volume_permutations.append(tuple(permutation))
        permutations = tuple(volume_permutations)
    rotate = np.asarray(_numbers(reco.get("RECO_rotate")), float)
    if rotate.size:
        if rotate.size == ndim:
            rotate = rotate[:, None]
        elif rotate.size % ndim == 0 and count % (rotate.size // ndim) == 0:
            rotate = np.tile(rotate.reshape((ndim, -1)), (1, count // (rotate.size // ndim)))
        else:
            raise ValueError("Rotation metadata does not match encoding dimensions/frames")
        shifts = (rotate[::-1].T - .5) * np.asarray(grid)[None, :]
        frame_shifts = tuple(tuple(float(v) for v in row) for row in shifts) if shifts.shape[0] > 1 else ()
        shift = () if frame_shifts else tuple(float(v) for v in shifts[0])
    else:
        shift = (0.,) * ndim
        frame_shifts = ()
    target_size = tuple(reversed(tuple(int(x) for x in _numbers(reco.get("RECO_size")))[:ndim]))
    if not target_size:
        target_size = grid
    # A recommended image recipe is optional; no completion occurs during import.
    partial = "none"
    half_flags = reco.get("RecoHalfFT", [])
    if isinstance(half_flags, str):
        half_flags = [half_flags]
    if np.any(~acquired) and any(str(flag).lower() in {"yes", "on", "true", "1"} for flag in half_flags):
        algorithm = str(reco.get("RecoHalfFTAlgorithm", "POCS")).lower()
        if algorithm not in {"pocs", "homodyne", "zerofilling"}:
            raise ValueError("Unknown partial Fourier reconstruction algorithm")
        partial = "none" if algorithm == "zerofilling" else algorithm
    slices = 1
    if ndim == 2:
        image_dimensions = int(visu.get("VisuCoreDim", 2))
        slice_sizes = _numbers(method.get("PVM_SPackArrNSlices"))
        if len(slice_sizes) == 1 and slice_sizes[0] > 1 and image_dimensions == 2:
            slices = int(slice_sizes[0])
    echo_images = int(method.get("PVM_NEchoImages") or 1)
    frame_indices = ()
    if slices > 1 and echo_images > 1:
        repeats = int(acqp.get("NR", 1))
        if count != repeats * slices * echo_images:
            raise ValueError("Slice and echo dimensions do not account for acquisition frames")
        frame_indices = tuple(np.arange(count).reshape((repeats, slices, echo_images)).transpose(0, 2, 1).ravel().tolist())
    from .bruker_products import product_recipe

    window_modes = reco.get("RECO_wdw_mode", [])
    if isinstance(window_modes, str):
        window_modes = [window_modes]
    window_lookup = {"NO_WDW": "none", "SINE": "sine", "QSINE": "sine_squared",
        "HANN": "hann", "HAMMING": "hamming"}
    if any(mode not in window_lookup for mode in window_modes):
        raise ValueError("A declared spectral window requires an explicit reconstruction definition")
    window = tuple(reversed(tuple(window_lookup[mode] for mode in window_modes)))
    bell_shift = tuple(reversed(_numbers(reco.get("RECO_sbs")))) if window else ()
    from .bruker_graph import spectral_window_powers

    powers = tuple(reversed(spectral_window_powers(reco, ndim))) if window else ()
    recipe = ReconstructionRecipe(image_shift=shift, frame_image_shifts=frame_shifts, partial_fourier=partial,
        kspace_window=window, window_parameters=bell_shift, window_powers=powers,
        output_scale=1 / float(np.sqrt(np.prod(grid))),
        output_permutation=(2, 1, 0) if ndim == 3 else (),
        output_flips=(2,) if str(visu.get("VisuCoreDiskSliceOrder", "")).lower() == "disk_reverse_slice_order" and (ndim == 3 or slices > 1) else (),
        frame_permutations=permutations, stack_slices=slices, output_frame_indices=frame_indices,
        output_shape=target_size, **product_recipe(method, reco, visu, library=library))
    acquisition = AcquisitionData(samples, grid, tuple(f"encoding_{i}" for i in range(ndim)),
        acquired=acquired, geometry=geometry, recipe=recipe, auxiliary_samples=records.auxiliary_samples)
    return acquisition, {"storage_profile": records.profile, "event_profile": events.name,
        "raw_status": "prepared", "padding_bytes_per_record": records.padding_bytes}
