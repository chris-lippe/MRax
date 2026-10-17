from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import jax.numpy as jnp
import numpy as np

from .dataset import DataSet
from .dimensions import dim_observable_map, expand_observables_to_frames
from .experiment import Experiment
from .quantity import Quantity, require_quantity_dict


@dataclass(frozen=True)
class JointSource:
    """Reference to a source experiment included in a joint stack."""

    name: str
    experiment: Experiment
    dataset: DataSet
    frame_indices: np.ndarray


@dataclass(frozen=True)
class JointExperiment:
    """Joined experiment with frame indices for each source."""

    experiment: Experiment
    dataset: DataSet
    sources: tuple[JointSource, ...]
    frame_map: Mapping[str, np.ndarray]
    source_observable: str | None = None

    def select(self, name: str | Sequence[str]) -> tuple[Experiment, DataSet]:
        """Return a sub-experiment by source name(s)."""
        if isinstance(name, (list, tuple)):
            indices = np.concatenate([self.frame_map[n] for n in name], axis=0)
        else:
            indices = self.frame_map[name]
        return select_experiment_frames(
            self.experiment,
            self.dataset,
            frame_indices=np.asarray(indices, dtype=int),
        )

    def iter_subexperiments(self):
        """Yield (name, experiment, dataset) for each source."""
        for src in self.sources:
            exp, ds = select_experiment_frames(
                self.experiment,
                self.dataset,
                frame_indices=np.asarray(src.frame_indices, dtype=int),
            )
            yield src.name, exp, ds


def _dim_map_for_experiment(experiment: Experiment) -> dict[str, str | None]:
    dim_map = getattr(experiment, "_mrax_dim_observables", None)
    if dim_map is None:
        dim_map = dim_observable_map(experiment.exp_type, experiment.exp_type)
    return dim_map or {}


def _flatten_to_frames(arr: np.ndarray, experiment: Experiment) -> np.ndarray:
    spatial_shape = experiment.spatial_shape
    frame_count = experiment.frame_count
    if tuple(arr.shape) == tuple(experiment.shape):
        dim_names = list(experiment.dims.keys())
        non_spatial = experiment.non_spatial_dim_names
        spatial = experiment.spatial_dim_names
        axis_order = [dim_names.index(name) for name in non_spatial + spatial]
        if axis_order != list(range(arr.ndim)):
            arr = np.transpose(arr, axis_order)
        return arr.reshape((frame_count,) + spatial_shape)
    if arr.ndim == len(spatial_shape):
        return np.broadcast_to(arr, (frame_count,) + spatial_shape)
    if arr.ndim == len(spatial_shape) + 1 and int(arr.shape[0]) == frame_count:
        return arr
    return arr


def _slice_kspace(kspace: np.ndarray, frame_count: int, indices: np.ndarray) -> np.ndarray:
    arr = np.asarray(kspace)
    if arr.ndim == 0:
        raise ValueError("kspace must be at least 1D when selecting frames")
    if arr.shape[0] == frame_count:
        return arr[indices]
    if arr.ndim == 2 and frame_count == 1:
        if len(indices) != 1:
            raise ValueError("Cannot select multiple frames from single-frame kspace")
        return arr.reshape((1,) + arr.shape)[indices]
    raise ValueError(
        f"kspace shape {tuple(arr.shape)} does not match expected frame axis {frame_count}"
    )


def _coerce_spatial_array(arr: np.ndarray, spatial_shape: tuple[int, ...]) -> np.ndarray | None:
    """Return a spatial-only array if `arr` can be reshaped to spatial_shape."""
    arr = np.asarray(arr)
    if arr.shape == spatial_shape:
        return arr
    if len(spatial_shape) >= 1 and arr.ndim == len(spatial_shape) - 1 and spatial_shape[-1] == 1:
        if tuple(arr.shape) == tuple(spatial_shape[:-1]):
            return arr.reshape(spatial_shape)
    if arr.ndim == len(spatial_shape) + 1 and int(arr.shape[0]) == 1:
        if tuple(arr.shape[1:]) == tuple(spatial_shape):
            return arr.reshape(spatial_shape)
    if arr.ndim > len(spatial_shape) and tuple(arr.shape[-len(spatial_shape) :]) == tuple(spatial_shape):
        leading = arr.shape[:-len(spatial_shape)]
        if all(int(s) == 1 for s in leading):
            return arr.reshape(spatial_shape)
    return None


def select_experiment_frames(
    experiment: Experiment,
    dataset: DataSet | None = None,
    *,
    frame_indices: Sequence[int] | np.ndarray | None = None,
    frame_mask: Sequence[bool] | np.ndarray | None = None,
    frame_dim: str = "frame",
    name: str | None = None,
) -> tuple[Experiment, DataSet | None]:
    """Return a frame-subset experiment (and optional dataset)."""

    if frame_indices is None:
        if frame_mask is None:
            raise ValueError("frame_indices or frame_mask is required")
        mask = np.asarray(frame_mask)
        if mask.ndim != 1:
            mask = mask.reshape((-1,))
        frame_indices = np.nonzero(mask != 0)[0]
    idx = np.asarray(frame_indices, dtype=int)

    frame_count = int(experiment.frame_count)
    if idx.ndim != 1:
        raise ValueError("frame_indices must be 1D")
    if idx.size == 0:
        raise ValueError("frame_indices must select at least one frame")
    if np.any(idx < 0) or np.any(idx >= frame_count):
        raise IndexError("frame_indices contain out-of-range entries")

    dim_map = _dim_map_for_experiment(experiment)
    expanded = expand_observables_to_frames(experiment.observables, experiment.dims, dim_map)

    obs_out: dict[str, Quantity] = {}
    spatial_shape = experiment.spatial_shape
    for key, qty in require_quantity_dict(dict(experiment.observables), "Observable").items():
        arr = np.asarray(expanded.get(key, qty.values))
        if tuple(arr.shape) == tuple(spatial_shape):
            obs_out[key] = qty
            continue
        if arr.ndim >= 1 and int(arr.shape[0]) == frame_count:
            obs_out[key] = qty.with_values(arr[idx])
        else:
            obs_out[key] = qty

    spatial_dims = {name: int(experiment.dims[name]) for name in experiment.spatial_dim_names}
    dims = {frame_dim: int(idx.size), **spatial_dims}
    new_exp = Experiment(dims=dims, observables=obs_out, exp_type=experiment.exp_type)
    if hasattr(experiment, "study_globals"):
        new_exp.study_globals = experiment.study_globals
    new_exp._mrax_dim_observables = {}

    if dataset is None:
        return new_exp, None

    data = np.asarray(dataset.data)
    data_flat = _flatten_to_frames(data, experiment)
    data_sel = data_flat[idx]

    acquisition_sel = dataset.acquisition.select_frames(idx) if dataset.acquisition is not None else None
    kspace_sel = acquisition_sel.samples if acquisition_sel is not None else None
    if dataset.kspace is not None and acquisition_sel is None:
        kspace_sel = _slice_kspace(np.asarray(dataset.kspace), frame_count, idx)

    sampling_mask_sel = acquisition_sel.acquired if acquisition_sel is not None else None
    if dataset.sampling_mask is not None and acquisition_sel is None:
        sampling_mask_arr = np.asarray(dataset.sampling_mask)
        if sampling_mask_arr.ndim >= 1 and int(sampling_mask_arr.shape[0]) == frame_count:
            sampling_mask_sel = sampling_mask_arr[idx]
        else:
            sampling_mask_sel = sampling_mask_arr

    new_ds = DataSet(
        data=jnp.asarray(data_sel),
        name=name or dataset.name,
        noise_level=dataset.noise_level,
        processed=dataset.processed,
        kspace=jnp.asarray(kspace_sel) if kspace_sel is not None else None,
        trajectory=dataset.trajectory,
        density=dataset.density,
        sampling_mask=jnp.asarray(sampling_mask_sel) if sampling_mask_sel is not None else None,
        kspace_metadata=dataset.kspace_metadata,
        spatial_geometry=dataset.spatial_geometry,
        acquisition=acquisition_sel,
    )
    return new_exp, new_ds


def join_experiments(
    experiments: Sequence[Experiment],
    datasets: Sequence[DataSet],
    *,
    names: Sequence[str] | None = None,
    exp_type: str | None = None,
    joint_name: str = "joint",
    frame_dim: str = "frame",
    source_observable: str | None = "joint_source",
    source_unit: str = "1",
    fill_value: float = float("nan"),
    allow_missing_observables: bool = True,
    spatial_mode: str = "require_equal",
    keep_kspace: bool = False,
) -> JointExperiment:
    """Join multiple experiments into a frame-major stack.

    Parameters
    ----------
    spatial_mode : {"require_equal", "broadcast"}
        How to handle spatial observables (e.g., B0/B1 maps) that differ across inputs.
        "require_equal" raises if multiple distinct maps are present. "broadcast"
        converts spatial maps into per-frame observables by repeating them per source.
    """

    if len(experiments) != len(datasets):
        raise ValueError("join_experiments expects the same number of experiments and datasets")
    if not experiments:
        raise ValueError("join_experiments requires at least one experiment")

    name_list = list(names) if names is not None else [ds.name or f"exp_{i}" for i, ds in enumerate(datasets)]
    if len(name_list) != len(experiments):
        raise ValueError("names length must match experiments length")
    if len(set(name_list)) != len(name_list):
        raise ValueError("names must be unique")

    spatial_shape = experiments[0].spatial_shape
    spatial_dims = {name: int(experiments[0].dims[name]) for name in experiments[0].spatial_dim_names}
    exp_types = {exp.exp_type for exp in experiments}
    if exp_type is None:
        if len(exp_types) > 1:
            raise ValueError(f"Experiments have multiple exp_type values: {sorted(exp_types)}")
        exp_type = next(iter(exp_types))

    study_globals = getattr(experiments[0], "study_globals", None)
    for exp in experiments:
        if exp.spatial_shape != spatial_shape:
            raise ValueError("All experiments must share the same spatial shape to join")
        if study_globals is not None and getattr(exp, "study_globals", None) != study_globals:
            raise ValueError("All experiments must share the same study_globals to join")

    data_blocks: list[np.ndarray] = []
    kspace_blocks: list[np.ndarray] = []
    expanded_obs: list[dict[str, np.ndarray]] = []
    frame_offsets: dict[str, np.ndarray] = {}
    sources: list[JointSource] = []

    total_frames = 0
    for name, exp, ds in zip(name_list, experiments, datasets):
        data = np.asarray(ds.data)
        data_flat = _flatten_to_frames(data, exp)
        frame_count = int(exp.frame_count)
        if int(data_flat.shape[0]) != frame_count:
            raise ValueError("Flattened data does not match experiment frame_count")
        data_blocks.append(data_flat)

        if keep_kspace:
            if ds.kspace is None:
                raise ValueError("keep_kspace=True requires all datasets to have kspace")
            kspace_arr = np.asarray(ds.kspace)
            if kspace_arr.shape[0] != frame_count:
                raise ValueError("kspace must have frame axis as the first dimension")
            kspace_blocks.append(kspace_arr)

        dim_map = _dim_map_for_experiment(exp)
        expanded_obs.append(expand_observables_to_frames(exp.observables, exp.dims, dim_map))

        frame_indices = np.arange(frame_count, dtype=int) + total_frames
        frame_offsets[name] = frame_indices
        sources.append(JointSource(name=name, experiment=exp, dataset=ds, frame_indices=frame_indices))
        total_frames += frame_count

    joined_data = np.concatenate(data_blocks, axis=0)
    joined_kspace = np.concatenate(kspace_blocks, axis=0) if keep_kspace else None

    joined_obs: dict[str, Quantity] = {}
    all_keys = set()
    for exp in experiments:
        all_keys.update(exp.observables.keys())

    for key in sorted(all_keys):
        unit = None
        for exp in experiments:
            qty = exp.observables.get(key)
            if qty is None:
                continue
            if unit is None:
                unit = qty.si_unit
            elif unit != qty.si_unit:
                raise ValueError(f"Observable '{key}' has mismatched units: {unit} vs {qty.si_unit}")
        if unit is None:
            continue

        per_frame_arrays: list[np.ndarray] = []
        spatial_arrays: list[np.ndarray] = []
        for obs_map in expanded_obs:
            if key not in obs_map:
                per_frame_arrays.append(None)  # type: ignore[arg-type]
                spatial_arrays.append(None)  # type: ignore[arg-type]
                continue
            arr = np.asarray(obs_map[key])
            spatial_arr = _coerce_spatial_array(arr, spatial_shape)
            if spatial_arr is not None:
                per_frame_arrays.append(None)  # type: ignore[arg-type]
                spatial_arrays.append(spatial_arr)
            else:
                per_frame_arrays.append(arr)
                spatial_arrays.append(None)  # type: ignore[arg-type]

        has_per_frame = any(arr is not None for arr in per_frame_arrays)
        has_spatial = any(arr is not None for arr in spatial_arrays)

        if has_spatial and not has_per_frame and spatial_mode != "broadcast":
            base = None
            for arr in spatial_arrays:
                if arr is None:
                    continue
                if base is None:
                    base = arr
                    continue
                if not np.allclose(base, arr, rtol=0.0, atol=1e-6):
                    if spatial_mode != "require_equal":
                        base = arr
                        break
                    raise ValueError(
                        f"Spatial observable '{key}' differs across inputs; "
                        "use spatial_mode='broadcast' to keep per-source maps"
                    )
            if base is None:
                if not allow_missing_observables:
                    raise ValueError(f"Observable '{key}' missing for all sources")
                continue
            joined_obs[key] = Quantity(jnp.asarray(base), unit)
            continue

        # Per-frame or mixed (broadcast spatial if requested).
        if has_spatial and spatial_mode != "broadcast":
            raise ValueError(
                f"Observable '{key}' mixes spatial and per-frame values; "
                "use spatial_mode='broadcast' to resolve"
            )

        tail_shape = None
        per_frame_chunks: list[np.ndarray] = []
        for name, exp, arr_pf, arr_sp in zip(name_list, experiments, per_frame_arrays, spatial_arrays):
            frame_count = int(exp.frame_count)
            if arr_pf is None and arr_sp is not None and spatial_mode == "broadcast":
                arr_pf = np.broadcast_to(arr_sp, (frame_count,) + arr_sp.shape)
            if arr_pf is None:
                if not allow_missing_observables:
                    raise ValueError(f"Observable '{key}' missing for source '{name}'")
                if tail_shape is None:
                    # Defer fill until tail_shape known.
                    per_frame_chunks.append(None)  # type: ignore[arg-type]
                else:
                    per_frame_chunks.append(
                        np.full((frame_count,) + tail_shape, fill_value, dtype=float)
                    )
                continue
            if arr_pf.ndim == 0:
                arr_pf = np.full((frame_count,), arr_pf, dtype=float)
            if int(arr_pf.shape[0]) != frame_count:
                raise ValueError(f"Observable '{key}' does not match frame_count for '{name}'")
            if tail_shape is None:
                tail_shape = tuple(arr_pf.shape[1:])
            elif tuple(arr_pf.shape[1:]) != tail_shape:
                raise ValueError(f"Observable '{key}' trailing shape mismatch across sources")
            per_frame_chunks.append(np.asarray(arr_pf))

        if tail_shape is None:
            continue
        # Fill deferred missing chunks now that tail_shape is known.
        for idx, chunk in enumerate(per_frame_chunks):
            if chunk is None:
                frame_count = int(experiments[idx].frame_count)
                per_frame_chunks[idx] = np.full(
                    (frame_count,) + tail_shape, fill_value, dtype=float
                )

        merged = np.concatenate(per_frame_chunks, axis=0)
        joined_obs[key] = Quantity(jnp.asarray(merged), unit)

    if source_observable:
        source_idx = np.zeros((total_frames,), dtype=float)
        for idx, name in enumerate(name_list):
            source_idx[frame_offsets[name]] = float(idx)
        joined_obs[source_observable] = Quantity(jnp.asarray(source_idx), source_unit)

    dims = {frame_dim: int(total_frames), **spatial_dims}
    joint_exp = Experiment(dims=dims, observables=joined_obs, exp_type=exp_type)
    if study_globals is not None:
        joint_exp.study_globals = study_globals
    joint_exp._mrax_dim_observables = {}

    joint_ds = DataSet(
        data=jnp.asarray(joined_data),
        name=joint_name,
        noise_level=None,
        processed=None,
        kspace=jnp.asarray(joined_kspace) if joined_kspace is not None else None,
        spatial_geometry=(
            datasets[0].spatial_geometry
            if datasets
            and all(dataset.spatial_geometry == datasets[0].spatial_geometry for dataset in datasets[1:])
            else None
        ),
    )

    frame_map = {name: np.asarray(idxs, dtype=int) for name, idxs in frame_offsets.items()}
    return JointExperiment(
        experiment=joint_exp,
        dataset=joint_ds,
        sources=tuple(sources),
        frame_map=frame_map,
        source_observable=source_observable,
    )


__all__ = [
    "JointExperiment",
    "JointSource",
    "join_experiments",
    "select_experiment_frames",
]
