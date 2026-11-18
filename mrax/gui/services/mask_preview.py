"""Read-only slices of the same native ROI arrays used by file exports."""

import numpy as np

from mrax.gui.masks import mask_slice_for_display

from .roi_tools import roi_runs


def native_mask_preview(session, name, axes=None, slice_index=0):
    roi = next((r for r in session.shared_rois if r.name == name), None)
    if roi is None:
        raise ValueError("Choose an existing ROI")
    mask = np.asarray(roi.mask)
    dims = tuple(roi.dims or session.spatial_dim_names)
    if "slice_package" in dims:
        axis = dims.index("slice_package")
        mask = np.take(mask, session.active_indices.get("slice_package", 0), axis=axis)
        dims = tuple(dim for dim in dims if dim != "slice_package")
        mask = mask[tuple(slice(0, session.view_dimensions[dim]) for dim in dims)]
    if mask.ndim not in {2, 3} or len(dims) != mask.ndim:
        raise ValueError("A mask preview requires two or three named spatial dimensions")
    plane_dims = tuple(axes.split(",")) if axes else dims[:2]
    if (
        len(plane_dims) != 2
        or len(set(plane_dims)) != 2
        or any(dim not in dims for dim in plane_dims)
        or tuple(sorted(plane_dims, key=dims.index)) != plane_dims
    ):
        raise ValueError("Choose two mask axes in their stored order")
    slice_dim = next((dim for dim in dims if dim not in plane_dims), None)
    count = mask.shape[dims.index(slice_dim)] if slice_dim else 1
    if not 0 <= slice_index < count:
        raise ValueError("Mask slice is out of range")
    plane = mask_slice_for_display(
        mask,
        dims=dims,
        plane_dims=plane_dims,
        indices={slice_dim: slice_index} if slice_dim else {},
    )
    if (
        not plane.size
        or plane.dtype.kind not in "buif"
        or not np.all(np.isfinite(plane))
        or np.any(plane < 0)
        or np.any(plane > 65535)
        or np.any(plane != np.floor(plane))
    ):
        raise ValueError("Mask preview requires integer label values between 0 and 65535")
    return {
        "name": roi.name,
        "dims": list(dims),
        "shape": list(mask.shape),
        "plane_dims": list(plane_dims),
        "plane_shape": list(plane.shape),
        "slice_dim": slice_dim,
        "slice_index": slice_index,
        "slice_count": count,
        "dtype": str(mask.dtype),
        "labels": [int(value) for value in np.unique(plane) if value != 0],
        "pixels": int(np.count_nonzero(plane)),
        "runs": roi_runs(plane),
    }
