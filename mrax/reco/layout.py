"""Invert explicit image layouts for encoding signal-model predictions."""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from mrax.data_struct.acquisition import AcquisitionData


def encoding_frame_map(acquisition: AcquisitionData):
    """Map native acquisition frames to reconstructed signal-model frames."""
    recipe = acquisition.recipe
    order = np.asarray(recipe.output_frame_indices or tuple(range(acquisition.frame_count)))
    if len(order) != acquisition.frame_count or not np.array_equal(np.sort(order), np.arange(acquisition.frame_count)):
        raise ValueError("Model-based encoding requires every acquired frame exactly once; subset the acquisition first")
    group = recipe.stack_slices * recipe.average_repeats
    if len(order) % group:
        raise ValueError("Acquisition frames do not contain complete output groups")
    result = np.empty(len(order), int)
    result[order] = np.arange(len(order)) // group
    return result


def image_to_encoding(image, acquisition: AcquisitionData):
    """Undo slice grouping, permutations and cropping, retaining native axes.

    The reverse of a crop pads with zeros: a model defined on the output FOV
    explicitly assumes no object outside that FOV. Gridding deapodization,
    visualization support and intensity calibration are not encoding operators.
    """
    recipe = acquisition.recipe
    ndim = len(acquisition.encoding_shape)
    array = jnp.asarray(image)
    if recipe.output_flips:
        array = jnp.flip(array, axis=tuple(axis + 1 for axis in recipe.output_flips))
    if recipe.output_stack_permutation:
        array = jnp.transpose(array, (0,) + tuple(i + 1 for i in np.argsort(recipe.output_stack_permutation)))
    frame_map = encoding_frame_map(acquisition)
    output_frames = int(frame_map.max()) + 1
    if array.shape[0] != output_frames:
        raise ValueError("Predicted image frame count does not match the acquisition layout")
    if recipe.average_repeats > 1:
        array = jnp.repeat(array, recipe.average_repeats, axis=0)
    array = array[:, None]
    if recipe.stack_slices > 1:
        if ndim != 2 or array.shape[-1] != recipe.stack_slices:
            raise ValueError("Predicted image slice axis does not match the acquisition layout")
        array = jnp.transpose(array, (0, 4, 1, 2, 3))
        array = array.reshape((-1, 1) + array.shape[-2:])
    if recipe.output_frame_indices:
        array = jnp.zeros_like(array).at[jnp.asarray(recipe.output_frame_indices)].set(array)
    if recipe.frame_permutations:
        array = jnp.stack([jnp.transpose(frame, (0,) + tuple(i + 1 for i in np.argsort(permutation)))
            for frame, permutation in zip(array, recipe.frame_permutations, strict=True)])
    if recipe.output_permutation:
        array = jnp.transpose(array, (0, 1) + tuple(i + 2 for i in np.argsort(recipe.output_permutation)))
    target = acquisition.encoding_shape
    offset = recipe.crop_offset or (0,) * ndim
    pads = [(0, 0), (0, 0)]
    for full, current, displacement in zip(target, array.shape[-ndim:], offset, strict=True):
        before = full // 2 - current // 2 + displacement
        if before < 0 or before + current > full:
            raise ValueError("Predicted image crop exceeds the encoding grid")
        pads.append((before, full - current - before))
    return jnp.pad(array, pads)[:, 0]
