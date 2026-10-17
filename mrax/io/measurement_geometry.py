"""Physical planes converted to MRax native-array geometry at the I/O boundary."""
from __future__ import annotations

import numpy as np

from mrax.data_struct import SpatialGeometry


def plane_geometry(positions, directions, shape, fov_mm, *, source):
    """Build RAS affines from LPS centres and XYZ direction columns.

    Native arrays have reversed XYZ spatial axes. Each supplied centre and
    orthonormal direction matrix describes one output frame.
    """
    positions = np.atleast_2d(np.asarray(positions, float))
    directions = np.asarray(directions, float).reshape((-1, 3, 3))
    fov = np.asarray(fov_mm, float)
    if (positions.shape != (len(directions), 3) or not np.isfinite(positions).all() or
            not np.isfinite(directions).all() or fov.shape != (3,) or not np.isfinite(fov).all() or
            np.any(fov <= 0) or not np.allclose(np.swapaxes(directions, -1, -2) @ directions, np.eye(3), atol=1e-3)):
        return SpatialGeometry(spatial_shape=shape, source=source)
    xyz_shape = np.asarray(tuple(reversed(shape)) + (1,) * (3 - len(shape)))
    spacing = fov / xyz_shape
    affines = []
    for position, direction in zip(positions, directions):
        affine = np.eye(4)
        affine[:3, :3] = np.diag([-1., -1., 1.]) @ direction @ np.diag(spacing)
        affine[:3, 3] = np.diag([-1., -1., 1.]) @ position - affine[:3, :3] @ ((xyz_shape - 1) / 2)
        affines.append(affine)
    return SpatialGeometry(affine=affines[0], frame_affines=affines if len(affines) > 1 else (),
        spatial_shape=shape, voxel_sizes=tuple(reversed(spacing[:len(shape)])),
        axis_anatomical_labels=tuple(reversed("xyz"[:len(shape)])), source=source, status="known")
