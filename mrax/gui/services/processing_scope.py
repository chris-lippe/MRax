"""Named, replayable denoising targets on the experiment's stored image grid."""

from __future__ import annotations

import numpy as np

from mrax.data_struct.geometry import SpatialGeometry


def freeze_scope(session, options, dims=None):
    """Capture the cursor once; legacy steps without a scope keep their meaning."""
    if "scope" not in options:
        if {"indices", "plane_dims"} & options.keys():
            raise ValueError("Denoising target coordinates require a scope")
        return options
    dims = dict(session.experiment_dims if dims is None else dims)
    scope = options["scope"]
    if not isinstance(scope, str) or scope not in {"image", "package", "dataset"}:
        raise ValueError("Choose the viewed image, active slice package or whole dataset")
    if scope == "package" and "slice_package" not in dims:
        raise ValueError("This experiment has no slice packages; choose the whole dataset")
    plane = options.get("plane_dims", list(session.image_plane_dims))
    if scope == "image" and (
        not isinstance(plane, list)
        or len(plane) != 2
        or any(
            not isinstance(name, str) or name not in dims or name not in {"x", "y", "z"}
            for name in plane
        )
        or len(set(plane)) != 2
    ):
        raise ValueError("The viewed image must have two spatial dimensions")
    fixed = (
        [name for name in dims if name not in plane]
        if scope == "image"
        else ["slice_package"]
        if scope == "package"
        else []
    )
    indices = options.get("indices")
    if indices is None:
        indices = {name: int(session.active_indices.get(name, 0)) for name in fixed}
    if (
        not isinstance(indices, dict)
        or set(indices) != set(fixed)
        or any(
            type(index) is not int or not 0 <= index < dims[name] for name, index in indices.items()
        )
    ):
        raise ValueError("The denoising target must identify valid indices for its scope")
    options["indices"] = dict(indices)
    if scope == "image":
        options["plane_dims"] = list(plane)
    else:
        options.pop("plane_dims", None)
    return options


def target_key(options):
    return (
        options.get("input", "image"),
        options.get("scope", "dataset"),
        tuple(sorted(options.get("indices", {}).items())),
        tuple(options.get("plane_dims", ())) if options.get("scope") == "image" else (),
    )


def region(session, options, dims, geometry):
    """Return a rectangular view, keeping fixed dimensions as singleton axes."""
    options = freeze_scope(session, dict(options), dims)
    if options.get("scope", "dataset") == "dataset":
        return tuple(slice(None) for _ in dims)
    indices = options["indices"]
    if options["scope"] == "image" and any(
        name not in options["plane_dims"] for name in options.get("axes", ())
    ):
        raise ValueError("Viewed-image denoising can only filter dimensions in that image plane")
    spatial = [name for name in dims if name in {"x", "y", "z"}]
    sizes = {name: dims[name] for name in spatial}
    if geometry is not None and geometry.spatial_shape is not None:
        shape = geometry.shape_for_frame(
            geometry.frame_index(indices, default=session.active_frame)
        )
        if options["scope"] == "package" and geometry.frame_shapes:
            frame_dims = geometry.frame_dims
            if frame_dims and "slice_package" in dict(frame_dims):
                axis = [name for name, _ in frame_dims].index("slice_package")
                frames = [
                    index
                    for index in range(len(geometry.frame_shapes))
                    if np.unravel_index(index, tuple(size for _, size in frame_dims))[axis]
                    == indices["slice_package"]
                ]
                shape = tuple(
                    max(geometry.frame_shapes[index][axis] for index in frames)
                    for axis in range(len(spatial))
                )
        if shape is not None and len(shape) == len(spatial):
            sizes.update(zip(spatial, shape))
    result = []
    for name, size in dims.items():
        measured = sizes.get(name, size)
        if name in indices:
            if indices[name] >= measured:
                raise ValueError("The saved denoising target is outside the acquired image")
            result.append(slice(indices[name], indices[name] + 1))
        else:
            result.append(slice(0, measured))
    return tuple(result)


def merge_component(original, filtered, component, parameters):
    """Keep the complementary complex component when filtering a partial target."""
    if component == "phase":
        return np.abs(original) * np.exp(1j * filtered)
    if np.iscomplexobj(original):
        if component == "real":
            return filtered + 1j * original.imag
        if component == "imag":
            return original.real + 1j * filtered
        if component == "magnitude" or (
            component == "auto" and parameters.get("complex_mode") == "magnitude"
        ):
            return filtered * np.exp(1j * np.angle(original))
    return filtered


def axis_info(workspace, source="image", package=None):
    """Describe the selected package without materializing image buffers."""
    from .cloning import clone_workspace
    from .pipeline import display_session

    session = workspace.session
    if package is not None and package != session.active_indices.get("slice_package"):
        if (
            "slice_package" not in session.experiment_dims
            or not 0 <= package < session.experiment_dims["slice_package"]
        ):
            raise ValueError("Choose an available slice package")
        work = clone_workspace(workspace, readonly_arrays=True)
        work.session.set_active_index("slice_package", package)
        session = work.session
    if source.startswith("map:"):
        session = display_session(workspace, fitted=True, map_name=source.removeprefix("map:"))
        if package is not None and "slice_package" in session.experiment_dims:
            session.set_active_index("slice_package", package)
    dims = session.view_dimensions
    geometry = session.image_geometry()
    pipeline = workspace.pipelines.get(session.selected_experiment)
    if source in {"original", "reconstruction"} and pipeline:
        state = pipeline["source"]
        dims = dict(state["dims"])
        geometry = SpatialGeometry.from_metadata(state.get("geometry"))
        geometry = geometry.for_frame(
            geometry.frame_index(session.active_indices, default=session.active_frame)
        )
        if geometry.spatial_shape:
            dims.update(
                zip([name for name in dims if name in {"x", "y", "z"}], geometry.spatial_shape)
            )
    spatial = [name for name in dims if name in {"x", "y", "z"}]
    labels = dict(zip(spatial, geometry.axis_anatomical_labels or spatial))
    if source.startswith("map:"):
        dims = {name: size for name, size in dims.items() if name in session.roi_dim_names}
    return [
        {
            "name": name,
            "label": labels.get(name, name),
            "size": int(size),
            "spatial": name in spatial,
        }
        for name, size in dims.items()
        if name != "slice_package"
    ]
