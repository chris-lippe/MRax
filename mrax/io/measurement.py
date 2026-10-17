"""Reader boundary for extracted measurement files and vendor-neutral containers."""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

from mrax.data_struct import AcquisitionData, SpatialGeometry
from mrax.devices import device_function
from mrax.io.import_definitions import (
    ImportDefinition,
    NormalizedMetadata,
    build_imported_experiment_dataset,
    detect_experiment_type,
    extract_observables_from_metadata,
)
from mrax.library.default_experiments import default_model_for_type


@dataclass(frozen=True)
class MeasurementData:
    """Images and/or prepared measurements, without running reconstruction."""

    image_data: np.ndarray | None = None
    acquisition: AcquisitionData | None = None
    spatial_geometry: SpatialGeometry | None = None
    normalized_metadata: NormalizedMetadata = field(default_factory=NormalizedMetadata)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self):
        if self.image_data is None and self.acquisition is None:
            raise ValueError("Measurement contains neither image pixels nor acquired samples")


_READERS: dict[str, tuple[tuple[str, ...], Callable[..., MeasurementData], Callable | None]] = {}


def register_measurement_reader(name: str, extensions, reader, *, inspector=None, replace=False):
    """Register a file adapter; adapters must not reconstruct or change source files."""
    if name in _READERS and not replace:
        raise ValueError(f"Measurement reader already registered: {name}")
    _READERS[name] = (tuple(str(x).lower() for x in extensions), reader, inspector)


def read_measurement(path: str | Path, *, format: str | None = None, **options) -> MeasurementData:
    """Read an extracted file using a named adapter or an unambiguous extension."""
    path = Path(path)
    if not path.is_file():
        raise ValueError("A measurement file is required; extract archives before importing")
    if format is None:
        matches = [name for name, (exts, _, _) in _READERS.items() if any(path.name.lower().endswith(x) for x in exts)]
        if len(matches) != 1:
            raise ValueError("Unknown or ambiguous measurement format; choose format explicitly")
        format = matches[0]
    if format not in _READERS:
        raise ValueError(f"Unknown measurement reader: {format}")
    return _READERS[format][1](path, **options)


def measurement_entries(path: str | Path, *, format=None):
    """Inspect groups/series without loading pixels; each entry is read separately."""
    path = Path(path)
    matches = [name for name, (extensions, _, _) in _READERS.items()
               if any(path.name.lower().endswith(ext) for ext in extensions)]
    if format is None:
        if len(matches) != 1:
            raise ValueError("Unknown or ambiguous measurement format")
        format = matches[0]
    inspector = _READERS[format][2]
    entries = [{}] if inspector is None else inspector(path)
    return [{'format': format, 'reader_options': entry} for entry in entries]


def _read_nifti(path, *, frame_selection=None, spatial_ndim=None, spatial_unit=None, frame_order=None):
    import nibabel as nib

    from mrax.io.nifti import nifti_spatial_geometry, nifti_to_mrax_volume

    image = nib.load(path)
    if spatial_ndim is None:
        spatial_ndim = min(len(image.shape), 3)
    if spatial_ndim not in (2, 3) or spatial_ndim > len(image.shape):
        raise ValueError("NIfTI spatial_ndim must be 2 or 3 and fit the image shape")
    frame_shape = image.shape[spatial_ndim:]
    frame_count = int(np.prod(frame_shape))
    selected = np.arange(frame_count) if frame_selection is None else np.atleast_1d(np.arange(frame_count)[frame_selection])
    if not len(selected):
        raise ValueError("Frame selection is empty")
    frame_indices = selected.copy()
    if frame_order is not None:
        order = np.asarray(frame_order)
        if (order.shape != (frame_count,) or not np.issubdtype(order.dtype, np.integer)
                or not np.array_equal(np.sort(order), np.arange(frame_count))):
            raise ValueError("NIfTI frame_order must be a permutation of all source frames")
        selected = order[selected]
    if frame_selection is None and frame_order is None:
        array = np.asanyarray(image.dataobj)
        leading = tuple(range(spatial_ndim, array.ndim))
        native = np.transpose(array, leading + tuple(reversed(range(spatial_ndim))))
        if spatial_ndim == 3 and image.shape[2] == 1:
            native = np.squeeze(native, axis=len(leading))
        shape = tuple(reversed(image.shape[:2] if spatial_ndim == 3 and image.shape[2] == 1 else image.shape[:spatial_ndim]))
        native = native.reshape((-1,) + shape)
    else:
        # Proxy indexing reads selected volumes, including compressed NIfTI input.
        volumes = []
        for frame in selected:
            index = (slice(None),) * spatial_ndim + np.unravel_index(frame, frame_shape)
            volumes.append(nifti_to_mrax_volume(np.asanyarray(image.dataobj[index])))
        native = np.stack(volumes)
        shape = native.shape[1:]
    header_unit, time_unit = image.header.get_xyzt_units()
    spatial_unit = spatial_unit or header_unit
    if spatial_unit not in ('unknown', 'mm', 'meter', 'micron'):
        raise ValueError("Unsupported NIfTI spatial_unit")
    affine = image.affine.copy()
    affine[:3] *= {'meter': 1000., 'micron': .001}.get(spatial_unit, 1.)
    geometry = nifti_spatial_geometry(affine, spatial_shape=shape)
    metadata: dict[str, object] = {"format": "nifti", "path": str(path), "spatial_unit": spatial_unit,
        "time_unit": time_unit, "source_frame_indices": selected,
        "frame_indices": frame_indices, "source_frame_count": frame_count}
    normalized = NormalizedMetadata(frame_count=len(native))
    if len(image.shape) >= 4 and time_unit in ('sec', 'msec', 'usec'):
        factor = {'sec': 1., 'msec': 1e-3, 'usec': 1e-6}[time_unit]
        dt = float(image.header.get_zooms()[3]) * factor
        if dt > 0:
            times = np.repeat(np.arange(image.shape[3]) * dt, int(np.prod(image.shape[4:])))
            metadata['frame_coordinates'] = {'time': times[selected] + float(image.header['toffset']) * factor}
            metadata['frame_coordinate_units'] = {'time': 's'}
            normalized = NormalizedMetadata(frame_count=len(native), repetition_times=[dt * 1000.], scan_time_ms=image.shape[3] * dt * 1000.)
    return MeasurementData(native, spatial_geometry=geometry, normalized_metadata=normalized, metadata=metadata)


def _read_matlab(path, *, image_key=None, sample_key=None, axes=None,
                 encoding_shape=None, trajectory_key=None, trajectory_units=None):
    from scipy.io import loadmat

    if (image_key is None) == (sample_key is None):
        raise ValueError("MATLAB input requires exactly one of image_key or sample_key; variables are never guessed")
    if axes is None:
        raise ValueError("MATLAB arrays require explicit axes, e.g. ('y', 'x', 'frame')")
    key = image_key or sample_key
    values = loadmat(path, variable_names=[key, trajectory_key] if trajectory_key else [key])
    array = np.asarray(values[key])
    axes = tuple(axes)
    if len(axes) != array.ndim or len(set(axes)) != len(axes):
        raise ValueError("MATLAB axes must uniquely name every array dimension")
    if image_key:
        spatial = [x for x in axes if x in {"x", "y", "z"}]
        leading = [x for x in axes if x not in spatial]
        native = array.transpose([axes.index(x) for x in leading + spatial])
        native = native.reshape((-1,) + tuple(array.shape[axes.index(x)] for x in spatial))
        geometry = SpatialGeometry(spatial_shape=native.shape[1:], axis_anatomical_labels=spatial)
        return MeasurementData(native, spatial_geometry=geometry, metadata={"format": "matlab", "path": str(path)})
    if encoding_shape is None:
        raise ValueError("MATLAB raw arrays require encoding_shape")
    target = ("frame", "coil", "sample") if trajectory_key else ("frame", "coil") + tuple(x for x in axes if x not in {"frame", "coil"})
    if set(target) != set(axes):
        raise ValueError("Raw MATLAB axes must explicitly include frame and coil")
    samples = array.transpose([axes.index(x) for x in target])
    trajectory = None
    if trajectory_key:
        if trajectory_units != "cycles/pixel":
            raise ValueError("MATLAB trajectories require explicit cycles/pixel coordinates in encoding-axis order")
        trajectory = np.asarray(values[trajectory_key])
    acquisition = AcquisitionData(samples, tuple(encoding_shape), tuple(f"encoding_{i}" for i in range(len(encoding_shape))),
        kind="sampled" if trajectory is not None else "cartesian", trajectory=trajectory)
    return MeasurementData(acquisition=acquisition, metadata={"format": "matlab", "path": str(path)})


register_measurement_reader("nifti", (".nii", ".nii.gz"), _read_nifti)
register_measurement_reader("matlab", (".mat",), _read_matlab)

def _frame_metadata(meta, coordinates, frame_count):
    values = {}
    for name in ('echo_times', 'inversion_times', 'repetition_times', 'flip_angles'):
        value = getattr(meta, name)
        if value is not None:
            array = np.asarray(value).reshape(-1)
            if len(array) not in (1, frame_count):
                contrast = coordinates.get('contrast')
                if contrast is None or np.max(contrast) >= len(array):
                    raise ValueError(f"{name} does not match frames; supply explicit normalized metadata")
                array = array[np.asarray(contrast, int)]
            values[name] = array.tolist()
    return replace(meta, frame_count=frame_count, **values)


def _output_groups(acquisition):
    order = np.asarray(acquisition.recipe.output_frame_indices or tuple(range(acquisition.frame_count)))
    group = acquisition.recipe.stack_slices * acquisition.recipe.average_repeats
    return order.reshape((-1, group))


def _output_metadata(meta, acquisition, groups):
    values = {}
    for item in fields(NormalizedMetadata):
        value = getattr(meta, item.name)
        if value is None:
            continue
        array = np.asarray(value)
        if array.ndim and array.shape[0] == acquisition.frame_count:
            grouped = array[groups]
            reference = grouped[:, :1]
            equal = np.allclose(grouped, reference, equal_nan=True) if np.issubdtype(array.dtype, np.number) else np.all(grouped == reference)
            if not equal:
                raise ValueError(f"{item.name} varies within a slice/repeat group; select slices separately or supply voxel-specific observables")
            values[item.name] = grouped[:, 0].tolist()
    return replace(meta, **values)


def _sort_contrasts(data, acquisition, geometry, meta, coordinates, exp_type):
    from mrax.data_struct.dimensions import get_default_dim_specs

    observables = extract_observables_from_metadata(exp_type, meta, len(data))
    columns = []
    for spec in get_default_dim_specs(exp_type):
        if spec.observable in observables:
            values = np.asarray(observables[spec.observable].values)
            if values.size == len(data):
                columns.append(values.reshape(-1))
    order = np.lexsort(tuple(reversed(columns))) if columns else np.arange(len(data))
    if np.array_equal(order, np.arange(len(data))):
        return data, acquisition, geometry, meta, coordinates
    data = data[order]
    if geometry is not None and len(geometry.frame_affines) == len(data):
        affines = tuple(geometry.frame_affines[i] for i in order)
        geometry = replace(geometry, affine=affines[0], frame_affines=affines)
    if acquisition is not None:
        groups = _output_groups(acquisition)[order]
        acquisition = replace(acquisition, geometry=geometry,
            recipe=replace(acquisition.recipe, output_frame_indices=tuple(int(i) for i in groups.ravel())))
    values = {}
    for item in fields(NormalizedMetadata):
        value = getattr(meta, item.name)
        if value is not None and np.asarray(value).ndim and np.asarray(value).shape[0] == len(data):
            values[item.name] = np.asarray(value)[order].tolist()
    coordinates = {key: np.asarray(value)[order] for key, value in coordinates.items()}
    return data, acquisition, geometry, replace(meta, **values), coordinates


def _axis_observables(observables, exp_type, frame_count):
    """Use axis vectors only for a rectangular contrast/repeat table."""
    from mrax.data_struct.dimensions import get_default_dim_specs

    result = dict(observables)
    product = 1
    candidates, columns = {}, []
    compact_axes = False
    for spec in get_default_dim_specs(exp_type):
        if spec.observable not in result:
            continue
        quantity = result[spec.observable]
        values = np.asarray(quantity.values)
        if values.size != frame_count:
            product *= max(1, values.size)
            compact_axes |= values.size > 1
            continue
        unique, counts = np.unique(values, return_counts=True)
        if len(set(counts)) != 1:
            return result, False
        product *= len(unique)
        columns.append(values.reshape(-1))
        candidates[spec.observable] = quantity.with_values(unique)
    rectangular = bool(product and frame_count % product == 0)
    if columns and not compact_axes:
        rectangular &= len(np.unique(np.column_stack(columns), axis=0)) == product
    if rectangular:
        result.update(candidates)
    return result, rectangular


@device_function
def build_measurement_experiment(measurement: MeasurementData, *, image_data=None,
                                         override_type=None, name=None, model_map_by_type=None,
                                         device='cpu'):
    """Materialize an image experiment; reconstruction is an explicit prior step."""
    data = measurement.image_data if image_data is None else np.asarray(image_data)
    if data is None:
        raise ValueError("Raw-only measurement: call reconstruct_acquisition first and supply image_data")
    acquisition = measurement.acquisition
    geometry = measurement.spatial_geometry
    if geometry is not None and geometry.spatial_shape:
        shape = tuple(geometry.spatial_shape)
    elif acquisition is not None:
        shape = tuple(acquisition.recipe.output_shape or acquisition.encoding_shape)
    else:
        shape = tuple(data.shape[-2:])
    if tuple(data.shape[-len(shape):]) != shape:
        raise ValueError("Image shape does not match imported geometry")
    data = np.asarray(data).reshape((-1,) + shape)
    coordinates = measurement.metadata.get('frame_coordinates', {}) if acquisition is None else acquisition.frame_coordinates
    meta = measurement.normalized_metadata
    if acquisition is not None and acquisition.domain == 'kspace':
        groups = _output_groups(acquisition)
        if len(data) != len(groups):
            raise ValueError("Reconstructed image frames do not match the prepared acquisition")
        coordinates = {key: np.asarray(value)[groups[:, 0]] for key, value in coordinates.items()}
        meta = _output_metadata(meta, acquisition, groups)
    meta = _frame_metadata(meta, coordinates, len(data))
    derived_image = measurement.metadata.get('image_component') in ('phase', 'parameter_map')
    exp_type = override_type or ('anatomical' if derived_image else detect_experiment_type(meta))
    data, acquisition, geometry, meta, coordinates = _sort_contrasts(data, acquisition, geometry, meta, coordinates, exp_type)
    observables, rectangular = _axis_observables(extract_observables_from_metadata(exp_type, meta, len(data)), exp_type, len(data))
    dims = None if rectangular else {"frame": len(data), **dict(zip(("x", "y", "z")[:len(shape)], shape))}
    experiment, dataset = build_imported_experiment_dataset(ImportDefinition(
        name=name or 'measurement', data=data, frame_count=len(data), spatial_shape=shape,
        exp_type=exp_type, observables=observables, acquisition=acquisition, dims=dims,
        spatial_geometry=geometry, kspace_metadata=measurement.metadata))
    from mrax.data_struct import Quantity

    frame_units = measurement.metadata.get('frame_coordinate_units', {})
    for key, values in coordinates.items():
        if key in frame_units and np.asarray(values).shape == (len(data),):
            experiment.observables[key] = Quantity(np.asarray(values).reshape((len(data),) + (1,) * len(shape)), frame_units[key])
    candidate = (model_map_by_type or {}).get(exp_type)
    model = candidate() if callable(candidate) else candidate
    model = model if model is not None else default_model_for_type(exp_type, scale=image_signal_scale(data),
        observables=experiment.observables, image_data=dataset.data)
    if str(model.name).lower() == 'identity' and not derived_image:
        message = "No quantitative model was inferred. Supply override_type and required observables before quantitative fitting"
        experiment._mrax_default_identity_fit_warning = message
        model._mrax_default_identity_fit_warning = message
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    return experiment, dataset, model, {**measurement.metadata, 'type': exp_type}



def sampled_reconstruction_recipe(*, output_shape=()):
    """Unitary Fourier normalization for the default Kaiser–Bessel preview."""
    from scipy.special import i0e

    from mrax.data_struct import ReconstructionRecipe

    recipe = ReconstructionRecipe(output_shape=tuple(output_shape), deapodize=True, encoding_scale=1.)
    beta, width = recipe.gridding_shape_parameter, recipe.gridding_kernel_width
    kernel_integral = width * (-np.expm1(-2 * beta)) / (2 * beta * i0e(beta))
    return replace(recipe, output_scale=float(kernel_integral ** -len(output_shape)))


def image_signal_scale(data):
    """A bounded-memory RMS estimate for amplitude parametrization, in image units."""
    values = np.asarray(data).reshape(-1)
    values = values[::max(1, int(np.ceil(values.size / 262144)))]
    values = np.abs(values.astype(np.complex128 if np.iscomplexobj(values) else np.float64))
    values = values[np.isfinite(values)]
    peak = float(np.max(values, initial=0.))
    rms = peak * float(np.sqrt(np.mean((values / peak)**2))) if peak else 0.
    return rms if rms > 0 else 1.
