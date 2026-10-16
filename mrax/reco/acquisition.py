"""Vendor-independent execution of validated acquisition reconstruction recipes."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from mrax.data_struct.acquisition import AcquisitionData, ReconstructionRecipe
from mrax.devices import DeviceLike, device_function

from .variational import fftnc, ifftnc


@dataclass(frozen=True)
class ReconstructionResult:
    image: Any
    coil_images: Any
    method: str
    history: tuple[float, ...] = ()
    parameters: dict | None = None
    diagnostics: tuple[dict, ...] = ()


def apply_image_shift(kspace, shifts, *, spatial_ndim: int, trajectory=None,
                      readout_coordinates=None, readout_polarity=None):
    """Translate a spatial image using signed frequencies, retaining subpixels."""
    array = jnp.asarray(kspace)
    shifts = jnp.asarray(shifts)
    if shifts.shape == (array.shape[0], spatial_ndim):
        frame_shifts = True
    elif shifts.shape == (spatial_ndim,):
        frame_shifts = False
    else:
        raise ValueError("A shift is required for each encoding axis")
    if trajectory is not None:
        points = jnp.asarray(trajectory)
        if array.ndim != 3 or points.shape != (array.shape[-1], spatial_ndim):
            raise ValueError("Sampled shifts require matching explicit trajectory coordinates")
        phase = (jnp.einsum("md,fd->fm", points, shifts)[:, None, :]
                 if frame_shifts else jnp.einsum("md,d->m", points, shifts))
        return array * jnp.exp(-2j * jnp.pi * phase)
    phase = 0.0
    for i, size in enumerate(array.shape[-spatial_ndim:]):
        shape = [1] * array.ndim
        shape[array.ndim - spatial_ndim + i] = size
        if i == spatial_ndim - 1 and readout_coordinates is not None:
            frequency = jnp.where(jnp.asarray(readout_polarity) < 0,
                jnp.asarray(readout_coordinates)[1], jnp.asarray(readout_coordinates)[0])
        else:
            frequency = jnp.fft.fftshift(jnp.fft.fftfreq(size)).reshape(shape)
        shift = shifts[:, i].reshape((array.shape[0],) + (1,) * (array.ndim - 1)) if frame_shifts else shifts[i]
        phase = phase + frequency * shift
    return array * jnp.exp(-2j * jnp.pi * phase)


def complete_partial_fourier(kspace, acquired, *, spatial_ndim=2, method="pocs", iterations=12,
                             phase_window="boxcar", weighting="step", tolerance=0.0):
    """Phase-constrained POCS or homodyne from symmetric measured support.

    The phase reference uses paired frequencies around the *encoding* DC, not
    the midpoint of acquired indices. POCS restores every measured value after
    each projection. Homodyne is a different estimator and does not impose that
    constraint. See Noll et al. (1991), DOI 10.1109/42.79473, and Haacke et al.
    (1991), DOI 10.1016/0022-2364(91)90253-P.
    """
    if method not in {"pocs", "homodyne"}:
        raise ValueError("Unknown partial Fourier method")
    if iterations <= 0:
        raise ValueError("Partial Fourier iteration count must be positive")
    if phase_window not in {"boxcar", "hann"} or weighting not in {"step", "ramp"}:
        raise ValueError("Unknown partial Fourier window or weighting")
    if tolerance < 0:
        raise ValueError("Partial Fourier tolerance must be nonnegative")
    measured = jnp.asarray(kspace)
    mask = jnp.broadcast_to(jnp.asarray(acquired, bool), measured.shape)
    axes = tuple(range(-spatial_ndim, 0))
    mirror = mask
    for axis in axes:
        size = measured.shape[axis]
        indices = (2 * (size // 2) - jnp.arange(size)) % size
        mirror = jnp.take(mirror, indices, axis=axis)
    symmetric = mask & mirror
    phase_weights = symmetric.astype(measured.real.dtype)
    ramp_weights = jnp.ones_like(phase_weights)
    for axis in axes:
        size = measured.shape[axis]
        shape = [1] * measured.ndim
        shape[axis] = size
        coordinate = (jnp.arange(size) - size // 2).reshape(shape)
        other_axes = tuple(other for other in axes if other != axis)
        paired = jnp.any(symmetric, axis=other_axes, keepdims=True)
        radius = jnp.min(jnp.where(paired, size, jnp.abs(coordinate)), axis=axis, keepdims=True)
        if phase_window == "hann":
            phase_weights *= jnp.where(jnp.abs(coordinate) < radius,
                .5 + .5 * jnp.cos(jnp.pi * coordinate / jnp.maximum(radius, 1)), 0)
        active = jnp.any(mask, axis=other_axes, keepdims=True)
        first = jnp.min(jnp.where(active, coordinate, size), axis=axis, keepdims=True)
        last = jnp.max(jnp.where(active, coordinate, -size), axis=axis, keepdims=True)
        direction = jnp.sign(first + last)
        ramp_weights *= jnp.clip(1 + direction * coordinate / jnp.maximum(radius, 1), 0, 2)
    low = ifftnc(jnp.where(symmetric, measured, 0) * phase_weights, axes=axes)
    phase = jnp.exp(1j * jnp.angle(low))
    if method == "homodyne":
        weights = mask * (2.0 - mirror) if weighting == "step" else mask * ramp_weights
        image = ifftnc(measured * weights, axes=axes)
        return fftnc(jnp.real(image * jnp.conj(phase)) * phase, axes=axes)

    def body(_, estimate):
        image = ifftnc(estimate, axes=axes)
        projected = jnp.real(image * jnp.conj(phase)) * phase
        return jnp.where(mask, measured, fftnc(projected, axes=axes))

    initial = jnp.where(mask, measured, 0)
    if tolerance == 0:
        return jax.lax.fori_loop(0, int(iterations), body, initial)

    def converged_step(state):
        count, estimate, _ = state
        updated = body(count, estimate)
        change = jnp.linalg.norm(updated - estimate) / jnp.maximum(jnp.linalg.norm(estimate), 1e-20)
        return count + 1, updated, change

    return jax.lax.while_loop(lambda state: (state[0] < iterations) & (state[2] > tolerance),
        converged_step, (0, initial, jnp.asarray(jnp.inf, measured.real.dtype)))[1]


def format_coil_images(coils, recipe: ReconstructionRecipe, *, spatial_ndim: int):
    array = jnp.asarray(coils)
    full_shape = array.shape[-spatial_ndim:]
    if recipe.checkerboard:
        from .variational import checkerboard_phase
        array = array * checkerboard_phase(full_shape)
    if recipe.output_shape:
        if len(recipe.output_shape) != spatial_ndim:
            raise ValueError("Output shape must name every encoding dimension")
        slices = [slice(None)] * (array.ndim - spatial_ndim)
        offset = recipe.crop_offset or (0,) * spatial_ndim
        for size, target, displacement in zip(array.shape[-spatial_ndim:], recipe.output_shape, offset):
            start = size // 2 - target // 2 + displacement
            if start < 0 or start + target > size:
                raise ValueError("Output crop extends beyond the reconstructed image")
            slices.append(slice(start, start + target))
        array = array[tuple(slices)]
    if recipe.deapodize:
        from .variational import RegridApodizationCorrection, apply_regrid_apodization_correction
        array = apply_regrid_apodization_correction(array, RegridApodizationCorrection(
            recipe.gridding_kernel_width, recipe.gridding_shape_parameter,
            full_shape=full_shape, normalization=recipe.deapodization_normalization), array.shape[-spatial_ndim:])
    if recipe.elliptical_support:
        from .variational import elliptical_support_mask
        array = array * elliptical_support_mask(array.shape[-spatial_ndim:])
    if recipe.output_permutation:
        if sorted(recipe.output_permutation) != list(range(spatial_ndim)):
            raise ValueError("Invalid output spatial permutation")
        prefix = tuple(range(array.ndim - spatial_ndim))
        array = jnp.transpose(array, prefix + tuple(array.ndim - spatial_ndim + i for i in recipe.output_permutation))
    if recipe.frame_permutations:
        if len(recipe.frame_permutations) != array.shape[0]:
            raise ValueError("Per-frame output layout must match frame count")
        frames = []
        for frame, permutation in zip(array, recipe.frame_permutations):
            if sorted(permutation) != list(range(spatial_ndim)):
                raise ValueError("Invalid per-frame spatial permutation")
            frames.append(jnp.transpose(frame, (0,) + tuple(i + 1 for i in permutation)))
        if len({frame.shape for frame in frames}) != 1:
            raise ValueError("Mixed frame permutations cannot form a rectangular image stack")
        array = jnp.stack(frames)
    if recipe.stack_slices > 1:
        if recipe.output_frame_indices:
            array = array[jnp.asarray(recipe.output_frame_indices)]
        if spatial_ndim != 2 or array.shape[0] % recipe.stack_slices:
            raise ValueError("Slice grouping must exactly account for two-dimensional frames")
        array = array.reshape((array.shape[0] // recipe.stack_slices, recipe.stack_slices) + array.shape[1:])
        array = jnp.transpose(array, (0, 2, 3, 4, 1))
        if recipe.output_stack_permutation:
            array = jnp.transpose(array, (0, 1) + tuple(i + 2 for i in recipe.output_stack_permutation))
    elif recipe.output_frame_indices:
        array = array[jnp.asarray(recipe.output_frame_indices)]
    if recipe.output_flips:
        if any(axis >= array.ndim - 2 for axis in recipe.output_flips):
            raise ValueError("Output flip axis exceeds the formatted spatial dimensions")
        array = jnp.flip(array, axis=tuple(axis + 2 for axis in recipe.output_flips))
    return array * recipe.output_scale


@device_function
def reconstruct_acquisition(
    acquisition: AcquisitionData,
    *,
    method: str = "fft",
    sensitivities=None,
    experiment=None,
    model=None,
    device: DeviceLike = "cpu",
    frame_batch_size: int | None = None,
    **options,
) -> ReconstructionResult:
    """Reconstruct explicit samples without consulting source-format metadata.

    FFT, variational and signal-model reconstruction share the same measured
    samples, encoding grid, calibration convention and output recipe. Completion
    is an initialization for iterative algorithms, never additional measured data.
    """
    acquisition.require_prepared()
    name = method.lower().replace("-", "_")
    if name not in {"fft", "variational", "model_based"}:
        raise ValueError("method must be fft, variational, or model_based")
    if frame_batch_size is not None:
        from operator import index

        frame_batch_size = index(frame_batch_size)
        if frame_batch_size < 0:
            raise ValueError("frame_batch_size must be nonnegative or None")
        if name != "fft" and frame_batch_size:
            raise ValueError("Frame batching is available for FFT reconstruction")
    if name == "fft" and frame_batch_size != 0 and acquisition.recipe.output_product == "image":
        batched = _batched_fft(acquisition, sensitivities, device, frame_batch_size)
        if batched is not None:
            return batched
    ndim = len(acquisition.encoding_shape)
    axes = tuple(range(-ndim, 0))
    samples = jax.device_put(acquisition.samples, device)
    mask_shape = samples.shape[-ndim:] if acquisition.kind == "cartesian" else samples.shape[-1:]
    mask = jnp.ones(mask_shape, bool) if acquisition.acquired is None else jnp.asarray(acquisition.acquired)
    recipe = acquisition.recipe
    encoding_scale = recipe.output_scale if recipe.encoding_scale is None else recipe.encoding_scale
    samples = jnp.where(mask, samples, 0)
    diagnostics = []
    unshifted_samples = samples
    shift_options = {"trajectory": acquisition.trajectory,
        "readout_coordinates": acquisition.readout_coordinates, "readout_polarity": acquisition.readout_polarity}
    if recipe.image_shift:
        samples = apply_image_shift(samples, recipe.image_shift, spatial_ndim=ndim, **shift_options)
    if recipe.frame_image_shifts:
        samples = apply_image_shift(samples, recipe.frame_image_shifts, spatial_ndim=ndim, **shift_options)
    preview, preview_mask = samples, mask
    if acquisition.readout_coordinates is not None and recipe.readout_regrid:
        from .epi import regrid_readouts

        preview, preview_mask = regrid_readouts(samples, acquisition.readout_coordinates,
            acquisition.readout_polarity, mask, kernel=recipe.readout_interpolation, radius=recipe.readout_kernel_radius)
        diagnostics.append({"stage": "readout_regrid", "method": recipe.readout_interpolation, "usage": "initialization"})
    readout_phase = None
    if recipe.epi_phase_correction != "none":
        from .epi import correct_epi_entropy, correct_readout_phase
        if acquisition.kind != "cartesian" or acquisition.readout_polarity is None:
            raise ValueError("EPI phase correction requires explicit Cartesian readout polarities")
        polarity = jnp.broadcast_to(jnp.asarray(acquisition.readout_polarity), samples.shape)
        corrected = []
        for frame in range(acquisition.frame_count):
            if recipe.epi_phase_correction == "fixed":
                values = correct_readout_phase(preview[frame], polarity[frame] < 0, recipe.epi_phase_prior)
                report = {"coefficients_radians": list(recipe.epi_phase_prior), "method": "fixed"}
            elif recipe.epi_phase_correction == "navigator":
                from .epi import calibrate_epi_navigators

                coefficients = calibrate_epi_navigators(acquisition, frame)
                values = correct_readout_phase(preview[frame], polarity[frame] < 0, coefficients)
                report = {"coefficients_radians": coefficients.tolist(), "method": "navigator"}
            else:
                values, report = correct_epi_entropy(preview[frame], polarity[frame] < 0,
                    spatial_ndim=ndim, phase_prior=recipe.epi_phase_prior)
            corrected.append(values)
            diagnostics.append({"stage": "epi_phase", "frame": frame, **report})
        preview = jnp.stack(corrected)
        readout_phase = jnp.asarray([report["coefficients_radians"] for report in diagnostics if report["stage"] == "epi_phase"])
    prepared = preview
    if recipe.kspace_window and acquisition.kind == "cartesian":
        from .windows import spectral_window

        prepared = preview * spectral_window(acquisition.encoding_shape, recipe.kspace_window, recipe.window_parameters, recipe.window_powers)
    if acquisition.kind == "cartesian":
        if recipe.partial_fourier != "none":
            prepared = complete_partial_fourier(prepared, preview_mask, spatial_ndim=ndim,
                method=recipe.partial_fourier, iterations=recipe.partial_fourier_iterations,
                phase_window=recipe.phase_window, weighting=recipe.homodyne_weighting,
                tolerance=recipe.partial_fourier_tolerance)
        from .variational import build_operator
        coils = build_operator(spatial_shape=acquisition.encoding_shape,
            independent_coils=True).adjoint(prepared, kspace_has_coils=True)
    else:
        from .gridding import grid_kaiser_bessel
        gridding_samples, trajectory, density = samples, acquisition.trajectory, acquisition.density
        if recipe.profile_inversion:
            from .density import radial_profile_density
            from .readout import invert_readout_profiles

            assert acquisition.readout_model is not None
            gridding_samples, report = invert_readout_profiles(unshifted_samples, acquisition.readout_model)
            trajectory = acquisition.readout_model.target_trajectory
            if recipe.image_shift:
                gridding_samples = apply_image_shift(gridding_samples, recipe.image_shift, spatial_ndim=ndim, trajectory=trajectory)
            if recipe.frame_image_shifts:
                gridding_samples = apply_image_shift(gridding_samples, recipe.frame_image_shifts, spatial_ndim=ndim, trajectory=trajectory)
            density = radial_profile_density(trajectory,
                points_per_readout=acquisition.readout_model.target_points, center_out=False)
            diagnostics.append(report)
        gridded = grid_kaiser_bessel(gridding_samples, trajectory,
            recipe.gridding_shape or acquisition.encoding_shape, density=density,
            kernel_width=recipe.gridding_kernel_width, shape_parameter=recipe.gridding_shape_parameter)
        coils = ifftnc(gridded, axes=axes)
    history = ()
    parameters = None
    if name == "variational":
        from .variational import solve_variational

        # Gridding deapodization is an initialization correction. Applying it
        # after a true Fourier encoding solve would distort the fitted image.
        initial = coils
        if acquisition.kind == "sampled":
            initial = format_coil_images(coils, replace(recipe,
                output_shape=acquisition.encoding_shape, output_permutation=(),
                frame_permutations=(), stack_slices=1, output_stack_permutation=(), output_frame_indices=(),
                checkerboard=False, elliptical_support=False), spatial_ndim=ndim)
        else:
            initial = initial * recipe.output_scale
        if sensitivities is not None:
            sens = jnp.asarray(sensitivities)
            if sens.shape not in ((acquisition.coil_count,) + acquisition.encoding_shape,
                                  (acquisition.frame_count, acquisition.coil_count) + acquisition.encoding_shape):
                raise ValueError("Sensitivities must match the receivers and encoding grid")
            initial = jnp.sum(initial * jnp.conj(sens), axis=1) / jnp.maximum(jnp.sum(jnp.abs(sens)**2, axis=-ndim - 1), 1e-12)
        del coils
        solved, losses = solve_variational(samples * encoding_scale,
            spatial_shape=acquisition.encoding_shape,
            mask=mask, data_weights=acquisition.statistical_weights,
            sensitivities=sensitivities, independent_coils=sensitivities is None,
            trajectory=acquisition.trajectory,
            density=acquisition.density, initial_image=initial,
            readout_coordinates=acquisition.readout_coordinates,
            readout_polarity=acquisition.readout_polarity, readout_phase=readout_phase,
            return_history=True, device=device, **options)
        history = tuple(losses)
        coils = solved if sensitivities is None else solved[:, None] * jnp.asarray(sensitivities)
        coils = coils / recipe.output_scale
        if acquisition.kind == "sampled":
            recipe = replace(recipe, deapodize=False, crop_offset=())
    elif name == "model_based":
        from .layout import encoding_frame_map, image_to_encoding
        from .variational import solve_model_based
        if experiment is None or model is None:
            raise ValueError("Model-based reconstruction requires experiment and model")
        if sensitivities is None and acquisition.coil_count > 1:
            raise ValueError("Multi-coil model-based reconstruction requires sensitivity maps")
        parameters, losses, predict, image = solve_model_based(samples * encoding_scale,
            experiment=experiment, model=model, spatial_shape=acquisition.encoding_shape,
            sensitivities=sensitivities, mask=mask,
            data_weights=acquisition.statistical_weights,
            trajectory=acquisition.trajectory, density=acquisition.density,
            image_to_encoding=lambda value: image_to_encoding(value, acquisition),
            encoding_frame_indices=encoding_frame_map(acquisition),
            readout_coordinates=acquisition.readout_coordinates,
            readout_polarity=acquisition.readout_polarity, readout_phase=readout_phase,
            return_history=True, return_model_callable=True,
            return_reconstruction_image=True, device=device, **options)
        image = image_to_encoding(jnp.asarray(image), acquisition) / recipe.output_scale
        coils = image[:, None, ...] if sensitivities is None else image[:, None, ...] * jnp.asarray(sensitivities)
        history = tuple(losses)
        if acquisition.kind == "sampled":
            recipe = replace(recipe, deapodize=False, crop_offset=())
    if name != "fft" and recipe.kspace_window:
        from .windows import spectral_window

        coils = ifftnc(fftnc(coils, axes=axes) * spectral_window(coils.shape[-ndim:],
            recipe.kspace_window, recipe.window_parameters, recipe.window_powers), axes=axes)
    coils = format_coil_images(coils, recipe, spatial_ndim=ndim)
    if recipe.coil_combine == "rss":
        image = jnp.sqrt(jnp.sum(jnp.abs(coils) ** 2, axis=1))
    elif recipe.coil_combine == "none":
        image = coils
    elif recipe.coil_combine == "sense":
        if sensitivities is None:
            raise ValueError("SENSE combination requires sensitivities")
        sens = jnp.broadcast_to(jnp.asarray(sensitivities),
            (acquisition.frame_count, acquisition.coil_count) + acquisition.encoding_shape)
        sens = format_coil_images(sens, replace(recipe, output_scale=1., deapodize=False,
            elliptical_support=False), spatial_ndim=ndim)
        image = jnp.sum(coils * jnp.conj(sens), axis=1) / jnp.maximum(jnp.sum(jnp.abs(sens) ** 2, axis=1), 1e-12)
    else:
        image = jnp.sum(coils, axis=1)
    if recipe.average_repeats > 1:
        repeats = recipe.average_repeats
        if image.shape[0] % repeats:
            raise ValueError("Magnitude averaging requires complete reconstruction repeats")
        image = jnp.mean(jnp.abs(image).reshape((-1, repeats) + image.shape[1:]), axis=1)
        coils = jnp.mean(coils.reshape((-1, repeats) + coils.shape[1:]), axis=1)
        diagnostics.append({"stage": "magnitude_average", "repeats": repeats})
    if recipe.output_product == "frequency_hz":
        from .products import frequency_from_echoes

        image, reliable = frequency_from_echoes(coils, recipe.echo_spacing_seconds,
            spatial_ndim=coils.ndim - 2, unwrap=recipe.product_unwrap,
            unwrap_method=recipe.product_unwrap_method,
            mask_snr=recipe.product_mask_snr, signal_fraction=recipe.product_signal_fraction,
            corner_fraction=recipe.product_corner_fraction)
        diagnostics.append({"stage": "frequency_product", "units": "Hz",
            "reliable_fraction": float(jnp.mean(reliable))})
    return ReconstructionResult(image, coils, name, history, parameters, tuple(diagnostics))


def _batched_fft(acquisition, sensitivities, device, batch_size):
    """Execute independent frame groups before any whole-array device placement."""
    recipe = acquisition.recipe
    frames = acquisition.frame_count
    group_size = recipe.stack_slices * recipe.average_repeats
    order = np.asarray(recipe.output_frame_indices or tuple(range(frames)), dtype=int)
    if batch_size is None:
        bytes_per_frame = acquisition.samples.nbytes // frames
        batch_size = max(1, (256 * 1024**2) // max(bytes_per_frame, 1))
    batch_size = max(group_size, batch_size // group_size * group_size)
    if batch_size >= len(order):
        return None
    if len(order) % group_size:
        raise ValueError("Frame batches require complete slice and repeat groups")

    def select(value, indices):
        if value is not None and np.ndim(value) == acquisition.samples.ndim and np.shape(value)[0] == frames:
            return value[indices]
        return value

    images = coils = None
    diagnostics = [{"stage": "frame_batches", "batch_size": batch_size}]
    output_index = 0
    for start in range(0, len(order), batch_size):
        indices = order[start:start + batch_size]
        part_recipe = replace(recipe, output_frame_indices=(),
            frame_permutations=tuple(recipe.frame_permutations[i] for i in indices) if recipe.frame_permutations else (),
            frame_image_shifts=tuple(recipe.frame_image_shifts[i] for i in indices) if recipe.frame_image_shifts else ())
        part = replace(acquisition, samples=acquisition.samples[indices],
            acquired=select(acquisition.acquired, indices),
            statistical_weights=select(acquisition.statistical_weights, indices),
            readout_polarity=select(acquisition.readout_polarity, indices),
            frame_coordinates={key: np.asarray(value)[indices] for key, value in acquisition.frame_coordinates.items()},
            auxiliary_samples={key: np.take(value, indices, axis=acquisition.auxiliary_frame_axes[key])
                if key in acquisition.auxiliary_frame_axes else value for key, value in acquisition.auxiliary_samples.items()},
            auxiliary_coordinates={key: np.take(value, indices, axis=acquisition.auxiliary_frame_axes[key])
                if key in acquisition.auxiliary_frame_axes else value for key, value in acquisition.auxiliary_coordinates.items()},
            geometry=None, recipe=part_recipe)
        result = reconstruct_acquisition(part, method="fft", sensitivities=select(sensitivities, indices),
            device=device, frame_batch_size=0)
        image_part, coil_part = np.asarray(result.image), np.asarray(result.coil_images)
        if images is None:
            output_frames = len(order) // group_size
            images = np.empty((output_frames,) + image_part.shape[1:], image_part.dtype)
            coils = np.empty((output_frames,) + coil_part.shape[1:], coil_part.dtype)
        end = output_index + len(image_part)
        images[output_index:end] = image_part
        coils[output_index:end] = coil_part
        output_index = end
        for report in result.diagnostics:
            report = dict(report)
            if 'frame' in report:
                report['frame'] = int(indices[report['frame']])
            diagnostics.append(report)
    return ReconstructionResult(jax.device_put(images, device), coils, "fft", diagnostics=tuple(diagnostics))
