"""Non-Qt session state for the MRax GUI."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence

import jax.numpy as jnp
import numpy as np

from mrax.analysis import ROI, broadcast_mask, collect_roi_stats, prepare_ab_mt_z_spectra
from mrax.analysis.dti import dti_design_rank, fit_dti_log_linear
from mrax.analysis.timecourse import extract_timecourse as extract_timecourse_array
from mrax.analysis.timecourse import extract_timecourse_from_experiment
from mrax.core.loss_builder import run_fit
from mrax.data_struct import (
    DataSet,
    Experiment,
    Quantity,
    SpatialGeometry,
    Study,
    save_study_results,
)
from mrax.data_struct.dimensions import (
    frame_count_from_dims,
    non_spatial_names,
    reshape_dims_to_frames,
    spatial_names,
    spatial_shape_from_dims,
)
from mrax.data_struct.quantity import require_quantity_dict
from mrax.data_struct.storage import load_study_results
from mrax.devices import DeviceLike
from mrax.reco.components import reconstruction_coil_count, reconstruction_variant

from .models import make_model, normalize_model_key

_LOG = logging.getLogger(__name__)

IMAGE_PLANE_DIMS: dict[str, tuple[str, str]] = {
    "axial": ("x", "y"),
    "coronal": ("x", "z"),
    "sagittal": ("y", "z"),
}


@dataclass
class FrameMask:
    """GUI-managed include/exclude mask for model-fitting frames."""

    included: np.ndarray

    def __post_init__(self):
        self.included = np.asarray(self.included, dtype=bool).reshape((-1,))

    @classmethod
    def all(cls, length: int) -> "FrameMask":
        return cls(np.ones((int(length),), dtype=bool))

    @classmethod
    def from_excluded(cls, length: int, excluded: Iterable[int]) -> "FrameMask":
        mask = np.ones((int(length),), dtype=bool)
        for idx in excluded:
            if 0 <= int(idx) < mask.size:
                mask[int(idx)] = False
        return cls(mask)

    def exclude(self, *indices: int) -> None:
        for idx in indices:
            if 0 <= int(idx) < self.included.size:
                self.included[int(idx)] = False

    def include(self, *indices: int) -> None:
        for idx in indices:
            if 0 <= int(idx) < self.included.size:
                self.included[int(idx)] = True

    def toggle(self, index: int) -> None:
        if 0 <= int(index) < self.included.size:
            self.included[int(index)] = ~self.included[int(index)]

    def resize(self, length: int) -> None:
        length = int(length)
        if length == self.included.size:
            return
        resized = np.ones((length,), dtype=bool)
        n = min(length, self.included.size)
        resized[:n] = self.included[:n]
        self.included = resized

    def as_array(self) -> np.ndarray:
        return np.asarray(self.included, dtype=bool)

    @property
    def excluded_indices(self) -> list[int]:
        return [int(idx) for idx in np.flatnonzero(~self.included)]


@dataclass
class FitSettings:
    """Fitting settings controlled by the GUI."""

    model_name: str | None = None
    optimizer_name: str = "auto"
    num_steps: int = 200
    step_size: float = 0.02
    selected_roi: str | None = None
    output_dir: Path | None = None
    mask_fill_value: float | None = 0.0
    auto_mask_nan_observables: bool = True
    use_view_state_data: bool = False
    compute_derived_maps: bool | None = None
    sat_trans_reference_scan: str | None = None
    ab_mt_t1_map_ref: str | None = None
    ab_mt_b1_map_ref: str | None = None
    ab_mt_b0_map_ref: str | None = None
    selected_pools: tuple[str, ...] = ()
    param_overrides: dict[str, dict[str, object]] = field(default_factory=dict)
    pool_overrides: dict[str, dict[str, object]] = field(default_factory=dict)
    regularizer_configs: dict[str, list[dict[str, object]]] = field(default_factory=dict)
    parameter_constant_dims: dict[str, tuple[int | str, ...]] = field(default_factory=dict)


@dataclass
class ExperimentViewState:
    """GUI state that should follow one selected experiment."""

    active_frame: int = 0
    active_z: int = 0
    active_indices: dict[str, int] = field(default_factory=dict)
    active_image_plane: str = "axial"
    active_map: str | None = None
    active_kspace_mode: str = "log_magnitude"
    active_kspace_centering: str = "auto"
    active_reconstruction_component: str = "magnitude"
    active_reconstruction_coil: int | None = None
    active_reconstruction_unwrap_phase: bool = True
    rois: list[ROI] = field(default_factory=list)
    fit_settings: FitSettings = field(default_factory=FitSettings)
    frame_mask: FrameMask | None = None
    visual_state: "VisualViewState" = field(default_factory=lambda: VisualViewState())


@dataclass
class VisualViewState:
    """Non-destructive display modifiers for one experiment."""

    show_original: bool = True
    display_mode: str = "current"
    mask_enabled: bool = False
    mask_roi: str | None = None
    drift_key: str | None = None
    transform_key: str | None = None
    transform_direction: str = "moving_to_reference"
    color_channel: str = "gray"
    color_alpha: float = 1.0
    reference_channel: str = "red"
    moving_channel: str = "green"
    reference_alpha: float = 1.0
    moving_alpha: float = 1.0


@dataclass(frozen=True)
class Diagnostic:
    """A user-facing diagnostic message."""

    level: str
    message: str


@dataclass
class GuiSession:
    """Qt-independent, manifest-backed GUI session state."""

    manifest_path: Path | None = None
    data: dict[str, Any] | None = None
    study: Study | None = None
    import_metadata: dict[str, Any] | None = None
    source_kind: str | None = None
    selected_experiment: str | None = None
    active_frame: int = 0
    active_z: int = 0
    active_indices: dict[str, int] = field(default_factory=dict)
    active_image_plane: str = "axial"
    active_map: str | None = None
    active_kspace_mode: str = "log_magnitude"
    active_kspace_centering: str = "auto"
    active_reconstruction_component: str = "magnitude"
    active_reconstruction_coil: int | None = None
    active_reconstruction_unwrap_phase: bool = True
    rois: list[ROI] = field(default_factory=list)
    shared_rois: list[ROI] = field(default_factory=list)
    fit_settings: FitSettings = field(default_factory=FitSettings)
    frame_mask: FrameMask | None = None
    visual_state: VisualViewState = field(default_factory=VisualViewState)
    view_states: dict[str, ExperimentViewState] = field(default_factory=dict)
    _slice_cache: dict[tuple[Any, ...], np.ndarray] = field(default_factory=dict, init=False, repr=False)

    @classmethod
    def from_manifest(cls, manifest_path: str | Path) -> "GuiSession":
        session = cls()
        session.load_manifest(manifest_path)
        return session

    def load_manifest(self, manifest_path: str | Path) -> None:
        path = Path(manifest_path)
        self.data = load_study_results(path)
        self._ensure_original_image_snapshots()
        self.manifest_path = path
        self.study = None
        self.import_metadata = None
        self.source_kind = "manifest"
        self.view_states.clear()
        self.shared_rois.clear()
        self._slice_cache.clear()
        names = self.experiment_names
        self.selected_experiment = names[0] if names else None
        self._reset_current_view_state()

    def create_empty_study(
        self,
        name: str = "Untitled Study",
        *,
        field_strength_mhz: float | Quantity | None = None,
    ) -> None:
        """Start a writable in-memory study that can receive experiments later."""

        study_name = str(name or "").strip() or "Untitled Study"
        globals_map: dict[str, Quantity] = {}
        if field_strength_mhz is not None:
            globals_map["field_strength_mhz"] = (
                field_strength_mhz
                if isinstance(field_strength_mhz, Quantity)
                else Quantity(jnp.asarray(float(field_strength_mhz)), "MHz")
            )
        self.data = _empty_session_data(study_name, study_globals=globals_map)
        self.study = Study([], [], [], names=[], study_globals=globals_map)
        self.import_metadata = {"subject": study_name, "experiments": {}}
        self.manifest_path = None
        self.source_kind = "memory"
        self.selected_experiment = None
        self.view_states.clear()
        self.shared_rois.clear()
        self._slice_cache.clear()
        self._reset_current_view_state()

    @property
    def study_name(self) -> str:
        manifest = (self.data or {}).get("manifest") or {}
        value = manifest.get("study_name")
        return str(value or "Untitled Study")

    def rename_study(self, name: str) -> None:
        study_name = str(name or "").strip()
        if not study_name:
            raise ValueError("Study name cannot be empty")
        if self.data is None:
            self.create_empty_study(study_name)
            return
        manifest = self.data.setdefault("manifest", {})
        manifest["study_name"] = study_name
        if self.import_metadata is not None:
            self.import_metadata["subject"] = study_name
        if manifest.get("source", {}).get("kind") == "memory":
            manifest["source"] = {**dict(manifest.get("source") or {}), "name": study_name}
        self._rebuild_study_from_session()
        self.manifest_path = None
        self.source_kind = "memory"

    def import_bruker_study(
        self,
        study_root: str | Path,
        *,
        field_strength_mhz: float | Quantity | None = None,
        pdata_index: str | Mapping[str, str] = "1",
        load_raw_kspace: bool = False,
        type_overrides: Mapping[str, str] | None = None,
        model_overrides_by_name: Mapping[str, str] | None = None,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> None:
        """Import a Bruker study into MRax internal structures for GUI use.

        The Bruker source folder is only read by ``mrax.io.bruker_import``. After
        this method returns, GUI operations use the in-memory ``Study`` and arrays.
        """

        from mrax.io.bruker_import import import_bruker_study

        root = Path(study_root)
        model_map_by_name = None
        if model_overrides_by_name:
            model_map_by_name = {
                str(name): make_model(model_key)
                for name, model_key in model_overrides_by_name.items()
                if model_key
            }
        study, metadata = import_bruker_study(
            root,
            type_overrides=dict(type_overrides or {}),
            model_map_by_name=model_map_by_name,
            field_strength_mhz=field_strength_mhz,
            pdata_index=pdata_index,
            load_raw_kspace=load_raw_kspace,
            progress_callback=progress_callback,
        )
        self.study = study
        self.import_metadata = metadata
        self.data = _study_to_session_data(study, metadata, source_root=root)
        self._ensure_original_image_snapshots()
        self.manifest_path = None
        self.source_kind = "bruker"
        self.view_states.clear()
        self.shared_rois.clear()
        self._slice_cache.clear()
        names = self.experiment_names
        self.selected_experiment = names[0] if names else None
        self._reset_current_view_state()

    def import_pulseq_study(self, source, *, options=None, progress_callback=None):
        """Read a prepared GUI import plan into the same containers as other vendors."""
        from mrax.gui.pulseq_import import (
            PulseqImportPlan,
            load_pulseq_import_plan,
            plan_pulseq_import,
        )

        plan = source if isinstance(source, PulseqImportPlan) else plan_pulseq_import(source)
        study, metadata = load_pulseq_import_plan(plan, dict(options or {}), progress_callback)
        self.study = study
        self.import_metadata = metadata
        self.data = _study_to_session_data(study, metadata, source_root=plan.root, source_kind="pulseq")
        self._ensure_original_image_snapshots()
        self.manifest_path = None
        self.source_kind = "pulseq"
        self.view_states.clear()
        self.shared_rois.clear()
        self._slice_cache.clear()
        self.selected_experiment = self.experiment_names[0] if self.experiment_names else None
        self._reset_current_view_state()

    def load_dicom_study(
        self,
        study_root: str | Path,
        *,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> None:
        """Recursively import a DICOM folder as an in-memory MRax study."""

        from mrax.io.dicom import load_dicom_study

        root = Path(study_root)
        if progress_callback is not None:
            progress_callback(0, 1, "Reading DICOM series")
        study, metadata = load_dicom_study(root, recursive=True)
        if progress_callback is not None:
            progress_callback(1, 1, "DICOM import complete")
        self.study = study
        self.import_metadata = metadata
        self.data = _study_to_session_data(
            study,
            metadata,
            source_root=root,
            source_kind="dicom",
        )
        self._ensure_original_image_snapshots()
        self.manifest_path = None
        self.source_kind = "dicom"
        self.view_states.clear()
        self.shared_rois.clear()
        self._slice_cache.clear()
        names = self.experiment_names
        self.selected_experiment = names[0] if names else None
        self._reset_current_view_state()

    def add_experiment(
        self,
        name: str,
        data: Any,
        *,
        dims: Mapping[str, int],
        exp_type: str | None = "anatomical",
        model_name: str | None = "identity",
        observables: Mapping[str, Quantity] | None = None,
        kspace: Any | None = None,
        processed: Mapping[str, Any] | None = None,
        kspace_metadata: Mapping[str, Any] | None = None,
        extra_metadata: Mapping[str, Any] | None = None,
        data_role: str | None = None,
        non_imaging: bool = False,
        select: bool = True,
    ) -> str:
        """Append an experiment to the current in-memory study."""

        if self.data is None:
            self.create_empty_study()
        if not self.data:
            raise RuntimeError("No study is available")
        exp_name = _validated_new_experiment_name(name, self.experiment_names)
        dims_map = _validated_dims(dims)
        arr = jnp.asarray(data)
        if tuple(int(v) for v in arr.shape[: len(dims_map)]) != tuple(dims_map.values()):
            raise ValueError(
                f"Data shape {tuple(arr.shape)} does not match experiment dims {tuple(dims_map.values())}."
            )
        observables_map = require_quantity_dict(dict(observables or {}), "Observable")
        experiment = Experiment(dims=dims_map, observables=observables_map, exp_type=exp_type or "anatomical")
        metadata = dict(extra_metadata or {})
        spatial_shape = tuple(dims_map[name] for name in spatial_names(dims_map))
        geometry = _spatial_geometry_from_metadata(metadata, spatial_shape=spatial_shape)
        metadata["spatial_geometry"] = geometry.to_metadata()
        dataset = DataSet(
            data=arr,
            name=exp_name,
            processed=dict(processed or {}),
            kspace=None if kspace is None else jnp.asarray(kspace),
            kspace_metadata=dict(kspace_metadata or {}),
            spatial_geometry=geometry,
        )
        model = make_model(model_name or "identity")
        experiment.validate(dataset)

        if self.selected_experiment:
            self._save_current_view_state()
        study = self.fit_study()
        if study is None:
            study = Study(
                [experiment],
                [dataset],
                [model],
                names=[exp_name],
                study_globals=dict((self.data or {}).get("study_globals") or {}),
            )
        else:
            study = Study(
                list(study.experiments) + [experiment],
                list(study.datasets) + [dataset],
                list(study.models) + [model],
                names=list(study.names or []) + [exp_name],
                study_globals=dict(study.study_globals or {}),
            )
        previous_experiments = dict((self.data or {}).get("experiments") or {})
        self.study = study
        self.manifest_path = None
        self.source_kind = "memory"
        if data_role:
            metadata["data_role"] = str(data_role)
        if non_imaging:
            metadata["non_imaging"] = True
        self.import_metadata = self.import_metadata or {"subject": self.study_name, "experiments": {}}
        self.import_metadata.setdefault("experiments", {})[exp_name] = metadata
        self.data = _study_to_session_data(
            study,
            self.import_metadata,
            source_root=Path.cwd(),
            source_kind="memory",
        )
        for old_name, old_data in previous_experiments.items():
            if old_name not in self.data.get("experiments", {}):
                continue
            target = self.data["experiments"][old_name]
            for key in ("fitted_maps", "fitted_map_units", "plots"):
                if key in old_data:
                    target[key] = old_data[key]
            if old_data.get("original_mr_image") is not None:
                target["original_mr_image"] = old_data["original_mr_image"]
        self._ensure_original_image_snapshots()
        if select or self.selected_experiment is None:
            self.selected_experiment = exp_name
            self._restore_or_reset_current_view_state()
        self._slice_cache.clear()
        return exp_name

    def rename_experiment(self, old_name: str, new_name: str) -> str:
        old = str(old_name or "").strip()
        new = str(new_name or "").strip()
        if not old:
            raise ValueError("No experiment is selected")
        if old not in self.experiment_names:
            raise KeyError(f"Unknown experiment '{old}'")
        if not new:
            raise ValueError("Experiment name cannot be empty")
        if old == new:
            return old
        if new in self.experiment_names:
            raise ValueError(f"Experiment '{new}' already exists")
        self._save_current_view_state()
        self.data["experiments"] = {
            (new if name == old else name): value
            for name, value in (self.data.get("experiments") or {}).items()
        }
        manifest = self.data.get("manifest") or {}
        summary = manifest.get("summary") or {}
        rows = summary.get("experiments") or []
        for row in rows:
            if isinstance(row, MutableMapping) and str(row.get("name")) == old:
                row["name"] = new
        if self.import_metadata and isinstance(self.import_metadata.get("experiments"), MutableMapping):
            experiments = self.import_metadata["experiments"]
            if old in experiments:
                experiments[new] = experiments.pop(old)
        if self.selected_experiment == old:
            self.selected_experiment = new
        if old in self.view_states:
            self.view_states[new] = self.view_states.pop(old)
        self._rename_experiment_references(old, new)
        self._rename_study_entry(old, new)
        self.manifest_path = None
        self.source_kind = "memory"
        self._slice_cache.clear()
        self._restore_or_reset_current_view_state()
        return new

    def update_experiment_metadata(
        self,
        name: str | None = None,
        *,
        exp_type: str | None = None,
        model_name: str | None = None,
        observables: Mapping[str, Quantity] | None = None,
        extra_metadata: Mapping[str, Any] | None = None,
        replace_observables: bool = True,
    ) -> None:
        """Update editable experiment metadata and rebuild backing Study entries."""

        if not self.data:
            raise RuntimeError("No study is loaded")
        exp_name = str(name or self.selected_experiment or "")
        if exp_name not in self.experiment_names:
            raise KeyError(f"Unknown experiment '{exp_name}'")
        exp_data = self.data["experiments"][exp_name]
        if exp_type is not None:
            exp_data["exp_type"] = str(exp_type or "")
        if model_name is not None:
            exp_data["model"] = str(model_name or "identity")
        if observables is not None:
            obs = require_quantity_dict(dict(observables), "Observable")
            exp_data["observables"] = obs if replace_observables else {**dict(exp_data.get("observables") or {}), **obs}
            exp_data["observable_units"] = {
                key: value.si_unit for key, value in dict(exp_data.get("observables") or {}).items()
            }
        if extra_metadata is not None:
            existing = dict(exp_data.get("extra_metadata") or {})
            existing.update(dict(extra_metadata or {}))
            exp_data["extra_metadata"] = existing
            exp_data["data_role"] = existing.get("data_role") or exp_data.get("exp_type")
            exp_data["non_imaging"] = bool(existing.get("non_imaging", exp_data.get("non_imaging", False)))
            self.import_metadata = self.import_metadata or {"subject": self.study_name, "experiments": {}}
            self.import_metadata.setdefault("experiments", {}).setdefault(exp_name, {}).update(existing)
        self._sync_summary_row(exp_name)
        self._rebuild_study_from_session()
        self.manifest_path = None
        self.source_kind = "memory"
        self._slice_cache.clear()

    def _rename_experiment_references(self, old: str, new: str) -> None:
        def rename_ref(ref: str | None) -> str | None:
            if not ref:
                return ref
            text = str(ref)
            prefix = f"{old}{_MAP_REF_SEP}"
            return f"{new}{_MAP_REF_SEP}{text[len(prefix):]}" if text.startswith(prefix) else text

        def update_settings(settings: FitSettings) -> None:
            if settings.sat_trans_reference_scan == old:
                settings.sat_trans_reference_scan = new
            settings.ab_mt_t1_map_ref = rename_ref(settings.ab_mt_t1_map_ref)
            settings.ab_mt_b1_map_ref = rename_ref(settings.ab_mt_b1_map_ref)
            settings.ab_mt_b0_map_ref = rename_ref(settings.ab_mt_b0_map_ref)

        update_settings(self.fit_settings)
        for state in self.view_states.values():
            update_settings(state.fit_settings)

    def _rename_study_entry(self, old: str, new: str) -> None:
        if self.study is None or old not in (self.study.names or ()):
            self._rebuild_study_from_session()
            return
        names = [new if name == old else name for name in self.study.names]
        self.study = Study(
            list(self.study.experiments),
            list(self.study.datasets),
            list(self.study.models),
            names=names,
            study_globals=dict(self.study.study_globals or {}),
        )

    def _sync_summary_row(self, name: str) -> None:
        if not self.data or name not in self.data.get("experiments", {}):
            return
        exp_data = self.data["experiments"][name]
        dims = self.summary_by_name.get(name, {}).get("dims") or {}
        if not dims:
            dims = _infer_dims_from_session_entry(exp_data)
        summary = self.data.setdefault("manifest", {}).setdefault("summary", {})
        rows = summary.setdefault("experiments", [])
        row = None
        for candidate in rows:
            if isinstance(candidate, MutableMapping) and str(candidate.get("name")) == name:
                row = candidate
                break
        if row is None:
            row = {"name": name}
            rows.append(row)
        row.update(
            {
                "name": name,
                "status": "loaded",
                "model": exp_data.get("model"),
                "exp_type": exp_data.get("exp_type"),
                "dataset": name,
                "dims": {str(k): int(v) for k, v in dict(dims).items()},
                "observables": sorted(dict(exp_data.get("observables") or {}).keys()),
                "non_imaging": bool(exp_data.get("non_imaging", False)),
                "data_role": exp_data.get("data_role") or exp_data.get("exp_type") or "image",
            }
        )

    def _rebuild_study_from_session(self) -> None:
        study = self.fit_study()
        if study is not None:
            self.study = study

    @property
    def is_loaded(self) -> bool:
        return bool(self.data and self.selected_experiment)

    @property
    def experiment_names(self) -> list[str]:
        if not self.data:
            return []
        return list((self.data.get("experiments") or {}).keys())

    @property
    def summary_by_name(self) -> dict[str, dict[str, Any]]:
        if not self.data:
            return {}
        summary = self.data.get("manifest", {}).get("summary", {})
        entries = summary.get("experiments", []) if isinstance(summary, dict) else []
        return {str(entry.get("name")): dict(entry) for entry in entries if entry.get("name")}

    def select_experiment(self, name: str) -> None:
        if name not in self.experiment_names:
            raise KeyError(f"Unknown experiment '{name}'")
        if name == self.selected_experiment:
            return
        self._save_current_view_state()
        self.selected_experiment = name
        self._restore_or_reset_current_view_state()
        self._slice_cache.clear()

    def _save_current_view_state(self) -> None:
        if not self.selected_experiment:
            return
        self.view_states[self.selected_experiment] = ExperimentViewState(
            active_frame=int(self.active_frame),
            active_z=int(self.active_z),
            active_indices=dict(self.active_indices),
            active_image_plane=self.active_image_plane,
            active_map=self.active_map,
            active_kspace_mode=self.active_kspace_mode,
            active_kspace_centering=self.active_kspace_centering,
            active_reconstruction_component=self.active_reconstruction_component,
            active_reconstruction_coil=self.active_reconstruction_coil,
            active_reconstruction_unwrap_phase=self.active_reconstruction_unwrap_phase,
            rois=[_copy_roi(roi) for roi in self.rois],
            fit_settings=replace(self.fit_settings),
            frame_mask=None if self.frame_mask is None else FrameMask(self.frame_mask.as_array().copy()),
            visual_state=replace(self.visual_state),
        )

    def _restore_or_reset_current_view_state(self) -> None:
        if self.selected_experiment and self.selected_experiment in self.view_states:
            self._apply_view_state(self.view_states[self.selected_experiment])
        else:
            self._reset_current_view_state()

    def _apply_view_state(self, state: ExperimentViewState) -> None:
        self.active_frame = int(state.active_frame)
        self.active_z = int(state.active_z)
        self.active_indices = dict(state.active_indices)
        self.active_image_plane = _normalize_image_plane(state.active_image_plane)
        self.active_map = state.active_map if state.active_map in self.map_names else (self.map_names[0] if self.map_names else None)
        self.active_kspace_mode = state.active_kspace_mode
        self.active_kspace_centering = state.active_kspace_centering
        component = str(state.active_reconstruction_component or "magnitude").lower()
        self.active_reconstruction_component = (
            component if component in {"magnitude", "phase"} else "magnitude"
        )
        coil = state.active_reconstruction_coil
        self.active_reconstruction_coil = (
            int(coil)
            if coil is not None and 0 <= int(coil) < self.reconstruction_coil_count
            else None
        )
        self.active_reconstruction_unwrap_phase = bool(
            state.active_reconstruction_unwrap_phase
        )
        self.rois = [_copy_roi(roi) for roi in state.rois]
        self.fit_settings = replace(state.fit_settings)
        if self.fit_settings.model_name is None:
            self.fit_settings.model_name = normalize_model_key(self.current_model_name)
        self.frame_mask = None if state.frame_mask is None else FrameMask(state.frame_mask.as_array().copy())
        if self.frame_mask is None:
            self.frame_mask = FrameMask.all(max(1, self.frame_count))
        else:
            self.frame_mask.resize(max(1, self.frame_count))
        self.visual_state = replace(state.visual_state)

    def _reset_current_view_state(self) -> None:
        self.active_frame = 0
        self.active_z = 0
        self.active_indices.clear()
        self.active_image_plane = self.native_image_plane
        self.active_map = self.map_names[0] if self.map_names else None
        self.active_kspace_mode = "log_magnitude"
        self.active_kspace_centering = "auto"
        self.active_reconstruction_component = "magnitude"
        self.active_reconstruction_coil = None
        self.active_reconstruction_unwrap_phase = True
        # Shared ROIs are available in the ROI list, but a newly visited experiment
        # starts with them disabled. "Apply All" creates explicit per-experiment
        # states when the user wants every experiment activated.
        self.rois = []
        self.fit_settings = FitSettings(model_name=normalize_model_key(self.current_model_name))
        self.frame_mask = FrameMask.all(max(1, self.frame_count)) if self.is_loaded else None
        self.visual_state = VisualViewState()
        self._slice_cache.clear()

    @property
    def current_experiment(self) -> dict[str, Any]:
        if not self.data or not self.selected_experiment:
            raise RuntimeError("No MRax manifest is loaded")
        return self.data["experiments"][self.selected_experiment]

    def current_study_entry(self) -> tuple[Experiment, Any, DataSet] | None:
        if self.study is None or self.selected_experiment is None:
            return None
        entries = self.study.entries_by_name()
        return entries.get(self.selected_experiment)

    def fit_study(self) -> Study | None:
        """Return a Study view of loaded GUI data for cross-scan fit preparation."""

        if self.study is not None:
            return self.study
        if not self.data:
            return None

        experiments: list[Experiment] = []
        datasets: list[DataSet] = []
        models: list[Any] = []
        names: list[str] = []
        for name in self.experiment_names:
            exp_data = self.data["experiments"].get(name) or {}
            summary = self.summary_by_name.get(name, {})
            dims = {str(k): int(v) for k, v in dict(summary.get("dims") or {}).items()}
            if not dims:
                image = _as_numpy(exp_data.get("mr_image"))
                if image.ndim <= 2:
                    dims = {"x": int(image.shape[-2]), "y": int(image.shape[-1])}
                else:
                    dims = {"frame": int(image.shape[0]), "x": int(image.shape[-2]), "y": int(image.shape[-1])}
            experiment = Experiment(
                dims=dims,
                observables=dict(exp_data.get("observables") or {}),
                exp_type=exp_data.get("exp_type"),
            )
            dataset = DataSet(
                data=jnp.asarray(_as_numpy(exp_data.get("mr_image"))),
                name=name,
                kspace=None if exp_data.get("raw_kspace") is None else jnp.asarray(exp_data.get("raw_kspace")),
                spatial_geometry=_spatial_geometry_from_experiment_data(exp_data, dims=dims),
                acquisition=exp_data.get("acquisition"),
                trajectory=exp_data.get("trajectory"), density=exp_data.get("density"),
                sampling_mask=exp_data.get("sampling_mask"),
            )
            experiments.append(experiment)
            datasets.append(dataset)
            models.append(make_model(exp_data.get("model") or "identity",
                param_overrides=exp_data.get("model_parameter_specs") or None))
            names.append(name)
        if not experiments:
            return None
        return Study(
            experiments,
            datasets,
            models,
            names=names,
            study_globals=dict((self.data or {}).get("study_globals") or {}),
        )

    def sat_trans_reference_choices(self) -> list[str]:
        """Eligible single-frame saturation-transfer references for the selected scan."""

        study = self.fit_study()
        if study is None or self.selected_experiment is None or self.selected_experiment not in study.names:
            return []
        target_idx = list(study.names).index(self.selected_experiment)
        target_exp = study.experiments[target_idx]
        if target_exp.exp_type != "sat_trans":
            return []
        target_shape = spatial_shape_from_dims(target_exp.dims)
        choices: list[str] = []
        for idx, (name, exp) in enumerate(zip(study.names, study.experiments)):
            if idx == target_idx:
                continue
            if exp.exp_type != "sat_trans" or int(exp.frame_count) != 1:
                continue
            if spatial_shape_from_dims(exp.dims) == target_shape:
                choices.append(str(name))
        return choices

    def ab_mt_map_choices(self, kind: str) -> list[dict[str, str]]:
        """Fitted-map candidates usable as AB-MT prior maps."""

        if not self.data or not self.selected_experiment:
            return []
        dims = self.experiment_dims
        if not dims:
            return []
        target_shape = tuple(int(v) for v in spatial_shape_from_dims(dims))
        choices: list[dict[str, str]] = []
        for exp_name in self.experiment_names:
            exp_data = self.data["experiments"].get(exp_name) or {}
            for map_name, value in dict(exp_data.get("fitted_maps") or {}).items():
                arr = _as_numpy(value)
                if tuple(arr.shape) != target_shape:
                    continue
                unit = value.si_unit if isinstance(value, Quantity) else str((exp_data.get("fitted_map_units") or {}).get(map_name) or "")
                score = _ab_mt_map_score(kind, str(map_name), unit)
                if score is None:
                    continue
                ref = _encode_map_ref(exp_name, str(map_name))
                label = f"{exp_name} / {map_name}" + (f" [{unit}]" if unit else "")
                choices.append(
                    {
                        "ref": ref,
                        "label": label,
                        "experiment": str(exp_name),
                        "map": str(map_name),
                        "unit": unit,
                        "score": str(score),
                    }
                )
        choices.sort(key=lambda item: (int(item["score"]), item["label"].lower()))
        return choices

    @property
    def current_summary(self) -> dict[str, Any]:
        if not self.selected_experiment:
            return {}
        return self.summary_by_name.get(self.selected_experiment, {})

    @property
    def current_model_name(self) -> str | None:
        try:
            return self.current_experiment.get("model")
        except RuntimeError:
            return None

    @property
    def experiment_dims(self) -> dict[str, int]:
        dims = self.current_summary.get("dims") or {}
        return {str(key): int(value) for key, value in dims.items()}

    @property
    def spatial_dim_names(self) -> list[str]:
        dims = self.experiment_dims
        return spatial_names(dims) if dims else ["x", "y"]

    @property
    def spatial_shape(self) -> tuple[int, ...]:
        dims = self.experiment_dims
        if dims:
            return tuple(int(dims[name]) for name in self.spatial_dim_names)
        image = np.asarray(self.frame_image())
        return tuple(int(v) for v in image.shape[-2:])

    @property
    def non_spatial_dim_names(self) -> list[str]:
        dims = self.experiment_dims
        return non_spatial_names(dims) if dims else (["frame"] if self.image_array.ndim >= 3 else [])

    @property
    def frame_count(self) -> int:
        dims = self.experiment_dims
        if dims:
            return max(1, frame_count_from_dims(dims))
        image = self.image_array
        return int(image.shape[0]) if image.ndim >= 3 else 1

    @property
    def is_non_imaging(self) -> bool:
        return _session_experiment_is_non_imaging(self.current_experiment, self.experiment_dims)

    @property
    def data_role(self) -> str:
        return _session_experiment_data_role(self.current_experiment, self.experiment_dims)

    @property
    def image_array(self) -> np.ndarray:
        return _as_numpy(self.current_experiment.get("mr_image"))

    @property
    def original_image_array(self) -> np.ndarray:
        exp = self.current_experiment
        original = exp.get("original_mr_image")
        if original is None:
            original = exp.get("mr_image")
        return _as_numpy(original)

    def image_data_is_original(self) -> bool:
        try:
            current = np.asarray(self.image_array)
            original = np.asarray(self.original_image_array)
        except Exception:
            return True
        return tuple(current.shape) == tuple(original.shape) and np.array_equal(current, original)

    @property
    def processed_arrays(self) -> Mapping[str, Any]:
        return self.current_experiment.get("processed") or {}

    @property
    def reconstruction_array(self) -> np.ndarray | None:
        try:
            return reconstruction_variant(self.processed_arrays, component="magnitude")
        except (RuntimeError, ValueError):
            return None

    @property
    def reconstruction_coil_count(self) -> int:
        return reconstruction_coil_count(self.processed_arrays)

    def reconstruction_display_array(
        self,
        *,
        component: str | None = None,
        coil: int | None = None,
        frame: int | None = None,
        combined: bool = False,
        unwrap_phase: bool | None = None,
    ) -> np.ndarray:
        """Return the selected real reconstruction component before 2D slicing."""

        component_name = component or self.active_reconstruction_component
        coil_index = None if combined else (self.active_reconstruction_coil if coil is None else int(coil))
        should_unwrap = (
            self.active_reconstruction_unwrap_phase
            if unwrap_phase is None
            else bool(unwrap_phase)
        )
        return reconstruction_variant(
            self.processed_arrays,
            component=component_name,
            coil=coil_index,
            frame=frame,
            unwrap_phase=bool(should_unwrap and component_name == "phase"),
            spatial_ndim=len(self.spatial_dim_names),
        )

    @property
    def kspace_array(self) -> np.ndarray | None:
        value = self.current_experiment.get("raw_kspace")
        return None if value is None else _as_numpy(value)

    def raw_data_loaded(self, name: str | None = None) -> bool:
        exp_data = self.current_experiment if name is None else (self.data or {}).get("experiments", {}).get(name, {})
        if not exp_data:
            return False
        if exp_data.get("raw_kspace") is not None:
            return True
        processed = exp_data.get("processed") or {}
        return (
            processed.get("raw_reconstruction") is not None
            or processed.get("raw_reconstruction_coils") is not None
        )

    def can_lazy_load_raw_data(self, name: str | None = None) -> bool:
        if self.source_kind != "bruker" or not self.data:
            return False
        exp_name = name or self.selected_experiment
        if not exp_name:
            return False
        exp_data = self.data.get("experiments", {}).get(exp_name) or {}
        if self.raw_data_loaded(exp_name):
            return False
        meta = dict(exp_data.get("kspace_metadata") or {})
        if meta.get("status") in {"loaded", "unavailable", "error"}:
            return False
        extra = dict(exp_data.get("extra_metadata") or {})
        path = extra.get("path")
        return bool(path and Path(path).exists())

    def load_raw_data_for_experiment(
        self,
        name: str | None = None,
        *,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> bool:
        if not self.data:
            raise RuntimeError("No study is loaded")
        exp_name = name or self.selected_experiment
        if not exp_name:
            raise RuntimeError("No experiment is selected")
        if self.raw_data_loaded(exp_name):
            return True
        exp_data = self.data["experiments"].get(exp_name) or {}
        extra = dict(exp_data.get("extra_metadata") or {})
        exp_path = Path(str(extra.get("path") or ""))
        if not exp_path.exists():
            raise RuntimeError(f"No source path is available for lazy raw import of experiment {exp_name}")
        if progress_callback is not None:
            progress_callback(0, 2, f"Loading raw k-space for scan {exp_name}")
        from mrax.io.bruker_import import import_bruker_experiment

        experiment, dataset, _model, meta = import_bruker_experiment(
            exp_path,
            override_type=exp_data.get("exp_type"),
            study_globals=dict((self.data or {}).get("study_globals") or {}),
            pdata_index=str(extra.get("pdata_index") or "1"),
            load_raw_kspace=True,
        )
        if progress_callback is not None:
            progress_callback(1, 2, f"Attaching raw data for scan {exp_name}")
        exp_data["raw_kspace"] = dataset.kspace
        exp_data["acquisition"] = dataset.acquisition
        exp_data["trajectory"] = dataset.trajectory
        exp_data["density"] = dataset.density
        exp_data["sampling_mask"] = dataset.sampling_mask
        exp_data["processed"] = dict(dataset.processed or {})
        loaded = dataset.kspace is not None or (dataset.processed or {}).get("raw_reconstruction") is not None
        status = "loaded" if loaded else "unavailable"
        exp_data["kspace_metadata"] = {"status": status, **dict(dataset.kspace_metadata or {})}
        exp_data["extra_metadata"] = {**extra, **dict(meta or {})}
        if self.import_metadata and isinstance(self.import_metadata.get("experiments"), dict):
            imp_exp = self.import_metadata["experiments"].setdefault(exp_name, {})
            imp_exp["raw_import"] = dict(dataset.kspace_metadata or {})
        self._update_study_raw_data(exp_name, dataset)
        self._slice_cache.clear()
        if progress_callback is not None:
            progress_callback(2, 2, f"Loaded raw data for scan {exp_name}")
        return loaded

    def load_all_lazy_raw_data(
        self,
        *,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> list[str]:
        names = [name for name in self.experiment_names if self.can_lazy_load_raw_data(name)]
        loaded: list[str] = []
        total = max(1, len(names))
        for idx, name in enumerate(names, start=1):
            if progress_callback is not None:
                progress_callback(idx - 1, total, f"Loading raw data for scan {name}")
            try:
                if self.load_raw_data_for_experiment(name):
                    loaded.append(name)
            except Exception as exc:
                exp_data = (self.data or {}).get("experiments", {}).get(name) or {}
                exp_data["kspace_metadata"] = {
                    **dict(exp_data.get("kspace_metadata") or {}),
                    "status": "error",
                    "message": str(exc),
                    "error_type": type(exc).__name__,
                }
                if self.import_metadata is not None:
                    issues = self.import_metadata.setdefault("import_issues", [])
                    issues.append(
                        {
                            "scan_no": str(name),
                            "status": "skipped",
                            "reason": "raw_import_failed",
                            "message": str(exc),
                            "error_type": type(exc).__name__,
                        }
                    )
        if progress_callback is not None:
            progress_callback(total, total, "Raw data loading complete")
        return loaded

    def can_reconstruct_raw_data(self, name: str | None = None) -> bool:
        entry = self.current_experiment if name is None else (self.data or {}).get("experiments", {}).get(name, {})
        acquisition = entry.get("acquisition")
        return acquisition is not None and acquisition.domain == "kspace"

    def reconstruct_raw_data_for_experiment(self, name: str | None = None, *, method="fft",
                                           device: DeviceLike = "cpu", **options):
        """Run the generic reconstruction stage after raw data have been imported."""
        from mrax.reco import reconstruct_acquisition
        from mrax.reco.components import RAW_RECONSTRUCTION_FREQUENCY_KEY

        exp_name = name or self.selected_experiment
        if not exp_name or not self.can_reconstruct_raw_data(exp_name):
            raise RuntimeError("This experiment has no prepared MRI acquisition")
        entry = self.data["experiments"][exp_name]
        acquisition = entry["acquisition"]
        result = reconstruct_acquisition(acquisition, method=method, device=device, **options)
        processed = dict(entry.get("processed") or {})
        processed["raw_reconstruction"] = np.asarray(result.image)
        processed["raw_reconstruction_coils"] = np.asarray(result.coil_images)
        if acquisition.recipe.output_product == "frequency_hz":
            processed[RAW_RECONSTRUCTION_FREQUENCY_KEY] = np.asarray(result.image)
        else:
            processed.pop(RAW_RECONSTRUCTION_FREQUENCY_KEY, None)
        entry["processed"] = processed
        entry["kspace_metadata"] = {**dict(entry.get("kspace_metadata") or {}),
            "reconstruction_status": "completed", "output_product": acquisition.recipe.output_product}
        if self.study is not None and exp_name in self.study.names:
            target = self.study.datasets[list(self.study.names).index(exp_name)]
            target.processed = processed
            target.kspace_metadata = dict(entry["kspace_metadata"])
        self._slice_cache.clear()
        return result

    def prepare_raw_view(self, view: str, *, progress_callback=None, device: DeviceLike = "cpu") -> bool:
        """Import missing samples, then reconstruct only when the chosen view needs it."""
        if not self.raw_data_loaded():
            self.load_raw_data_for_experiment(progress_callback=progress_callback)
        if view in {"mrax_reconstruction", "vendor_vs_mrax", "vendor_mrax_diff"} and self.reconstruction_array is None:
            if progress_callback is not None:
                progress_callback(0, 1, "Reconstructing selected experiment")
            self.reconstruct_raw_data_for_experiment(device=device)
            if progress_callback is not None:
                progress_callback(1, 1, "Reconstruction ready")
        return self.raw_data_loaded()

    def _update_study_raw_data(self, name: str, dataset: DataSet) -> None:
        if self.study is None or name not in (self.study.names or ()):
            return
        idx = list(self.study.names).index(name)
        target = self.study.datasets[idx]
        target.kspace = None if dataset.kspace is None else jnp.asarray(dataset.kspace)
        target.processed = dict(dataset.processed or {})
        target.kspace_metadata = dict(dataset.kspace_metadata or {})
        target.acquisition = dataset.acquisition
        target.trajectory = dataset.trajectory
        target.density = dataset.density
        target.sampling_mask = dataset.sampling_mask
        if dataset.spatial_geometry is not None:
            target.spatial_geometry = dataset.spatial_geometry

    @property
    def fitted_maps(self) -> Mapping[str, Any]:
        return self.current_experiment.get("fitted_maps") or {}

    @property
    def map_names(self) -> list[str]:
        try:
            return list(self.fitted_maps.keys())
        except RuntimeError:
            return []

    def map_array(self, name: str | None = None) -> np.ndarray:
        map_name = name or self.active_map
        if not map_name:
            raise KeyError("No fitted map is selected")
        return _as_numpy(self.fitted_maps[map_name])

    def map_unit(self, name: str | None = None) -> str | None:
        map_name = name or self.active_map
        if not map_name:
            return None
        value = self.fitted_maps.get(map_name)
        if isinstance(value, Quantity):
            return value.si_unit
        return (self.current_experiment.get("fitted_map_units") or {}).get(map_name)

    def replace_fitted_maps(self, fields: Mapping[str, Any] | None) -> None:
        """Replace fitted maps for the current experiment and invalidate display slices."""

        maps = dict(fields or {})
        self.current_experiment["fitted_maps"] = maps
        self.active_map = next(iter(maps), None)
        self._slice_cache.clear()

    def replace_image_data(
        self,
        data: Any,
        *,
        processed_key: str | None = None,
        processed_value: Any | None = None,
    ) -> None:
        """Replace current experiment image data in the session and backing study."""

        self._ensure_original_image_snapshot(self.current_experiment)
        arr = jnp.asarray(data)
        self.current_experiment["mr_image"] = arr
        if processed_key:
            processed = dict(self.current_experiment.get("processed") or {})
            processed[processed_key] = processed_value
            self.current_experiment["processed"] = processed
        if self.study is not None and self.selected_experiment in (self.study.names or ()):
            idx = list(self.study.names).index(self.selected_experiment)
            target = self.study.datasets[idx]
            target.data = arr
            if processed_key:
                target_processed = dict(target.processed or {})
                target_processed[processed_key] = processed_value
                target.processed = target_processed
        self._slice_cache.clear()

    def restore_original_image_data(self) -> bool:
        """Restore the current experiment image data from its protected source snapshot."""

        exp = self.current_experiment
        original = exp.get("original_mr_image")
        if original is None:
            return False
        original_arr = jnp.asarray(np.asarray(original).copy())
        try:
            changed = not np.array_equal(np.asarray(exp.get("mr_image")), np.asarray(original_arr))
        except Exception:
            changed = True
        exp["mr_image"] = original_arr
        if self.study is not None and self.selected_experiment in (self.study.names or ()):
            idx = list(self.study.names).index(self.selected_experiment)
            self.study.datasets[idx].data = original_arr
        self._slice_cache.clear()
        return changed

    def dimension_controls(self, kind: str = "image") -> list[dict[str, int | str]]:
        """Return named controls for axes outside the displayed 2D plane."""

        if not self.is_loaded:
            return []
        kind = kind.lower().replace("-", "_").replace(" ", "_")
        if self.is_non_imaging and kind == "image":
            return _axis_controls_for_non_image(self.image_array, self.experiment_dims)
        if kind == "k_space":
            arr = self.kspace_array
            return [] if arr is None else _axis_controls_for_kspace(arr, self.experiment_dims)
        if kind in {"mrax_reconstruction", "vendor_vs_mrax", "vendor_mrax_diff"}:
            try:
                arr = (
                    self.reconstruction_display_array(unwrap_phase=False)
                    if kind == "mrax_reconstruction"
                    else self.reconstruction_display_array(component="magnitude", combined=True)
                )
            except (RuntimeError, ValueError, IndexError):
                arr = None
            return (
                []
                if arr is None
                else _axis_controls_for_image(
                    arr,
                    self.experiment_dims,
                    plane_dims=self.image_plane_dims,
                )
            )
        if kind == "map":
            try:
                return _axis_controls_for_map(
                    self.map_array(),
                    self.experiment_dims,
                    plane_dims=self.image_plane_dims,
                )
            except Exception:
                return []
        return _axis_controls_for_image(
            self.image_array,
            self.experiment_dims,
            plane_dims=self.image_plane_dims,
        )

    @property
    def image_plane_dims(self) -> tuple[str, str]:
        plane = _normalize_image_plane(self.active_image_plane)
        return self._image_plane_dims_for_experiment()[plane]

    def _image_plane_dims_for_experiment(
        self,
        name: str | None = None,
    ) -> dict[str, tuple[str, str]]:
        if not self.data:
            return dict(IMAGE_PLANE_DIMS)
        exp_name = name or self.selected_experiment
        exp_data = self.data.get("experiments", {}).get(exp_name or "")
        if not isinstance(exp_data, Mapping):
            return dict(IMAGE_PLANE_DIMS)
        summary = self.summary_by_name.get(exp_name or "", {})
        dims = {
            str(key): int(value)
            for key, value in dict(summary.get("dims") or {}).items()
        }
        spatial_dims = tuple(spatial_names(dims))
        if spatial_dims != ("x", "y", "z"):
            return dict(IMAGE_PLANE_DIMS)
        geometry = _spatial_geometry_from_experiment_data(exp_data, dims=dims)
        labels = geometry.axis_anatomical_labels
        if labels is None:
            extra = exp_data.get("extra_metadata")
            metadata_labels = (
                extra.get("image_axis_anatomical_labels")
                if isinstance(extra, Mapping)
                else None
            )
            try:
                labels = tuple(str(value).lower() for value in metadata_labels)  # type: ignore[arg-type]
            except TypeError:
                labels = None
        if labels is None or len(labels) != 3 or set(labels) != {"x", "y", "z"}:
            return dict(IMAGE_PLANE_DIMS)
        dimension_for_axis = dict(zip(labels, spatial_dims))
        return {
            plane: (
                dimension_for_axis[anatomical_axes[0]],
                dimension_for_axis[anatomical_axes[1]],
            )
            for plane, anatomical_axes in IMAGE_PLANE_DIMS.items()
        }

    @property
    def native_image_plane(self) -> str:
        return self.native_image_plane_for_experiment()

    def native_image_plane_for_experiment(self, name: str | None = None) -> str:
        plane_dims = self._image_plane_dims_for_experiment(name)
        for plane, dims in plane_dims.items():
            if set(dims) == {"x", "y"}:
                return plane
        if self.data:
            exp_name = name or self.selected_experiment
            exp_data = self.data.get("experiments", {}).get(exp_name or "")
            extra = exp_data.get("extra_metadata") if isinstance(exp_data, Mapping) else None
            metadata_plane = extra.get("native_image_plane") if isinstance(extra, Mapping) else None
            if metadata_plane:
                try:
                    return _normalize_image_plane(str(metadata_plane))
                except ValueError:
                    pass
        return "axial"

    @property
    def image_slice_dim(self) -> str:
        plane = set(self.image_plane_dims)
        return next((name for name in ("x", "y", "z") if name not in plane), "z")

    @property
    def image_plane_voxel_sizes(self) -> tuple[float, float]:
        """Physical pixel sizes for the active anatomical plane, or unit spacing."""

        if not self.is_loaded:
            return (1.0, 1.0)
        geometry = _spatial_geometry_from_experiment_data(
            self.current_experiment,
            dims=self.experiment_dims,
        )
        spatial_dims = tuple(spatial_names(self.experiment_dims))
        if geometry.voxel_sizes is None or len(geometry.voxel_sizes) < len(spatial_dims):
            return (1.0, 1.0)
        spacing_by_dim = dict(zip(spatial_dims, geometry.voxel_sizes))
        try:
            first, second = (float(spacing_by_dim[name]) for name in self.image_plane_dims)
        except (KeyError, TypeError, ValueError):
            return (1.0, 1.0)
        if first <= 0 or second <= 0 or not np.isfinite((first, second)).all():
            return (1.0, 1.0)
        return first, second

    def supports_image_planes(self, kind: str = "image") -> bool:
        kind = kind.lower().replace("-", "_").replace(" ", "_")
        if self.is_non_imaging or kind == "k_space":
            return False
        if not {"x", "y", "z"}.issubset(self.experiment_dims):
            return False
        if kind in {"mrax_reconstruction", "vendor_vs_mrax", "vendor_mrax_diff"}:
            try:
                array = (
                    self.reconstruction_display_array(unwrap_phase=False)
                    if kind == "mrax_reconstruction"
                    else self.reconstruction_display_array(component="magnitude", combined=True)
                )
            except (RuntimeError, ValueError, IndexError):
                array = None
        elif kind == "map":
            try:
                array = self.map_array()
            except Exception:
                return False
        else:
            array = self.image_array
        return array is not None and np.asarray(array).ndim >= 3

    def set_image_plane(self, plane: str) -> None:
        normalized = _normalize_image_plane(plane)
        if normalized == self.active_image_plane:
            return
        self.active_image_plane = normalized
        self._slice_cache.clear()

    def set_reconstruction_component(self, component: str) -> None:
        component_name = str(component or "magnitude").strip().lower()
        if component_name not in {"magnitude", "phase"}:
            raise ValueError("Reconstruction component must be 'magnitude' or 'phase'.")
        self.active_reconstruction_component = component_name
        self._slice_cache.clear()

    def set_reconstruction_coil(self, coil: int | None) -> None:
        if coil is None:
            self.active_reconstruction_coil = None
        else:
            coil_index = int(coil)
            count = self.reconstruction_coil_count
            if coil_index < 0 or coil_index >= count:
                raise IndexError(
                    f"Reconstruction coil {coil_index + 1} is outside the available range 1..{count}."
                )
            self.active_reconstruction_coil = coil_index
        self._slice_cache.clear()

    def set_reconstruction_phase_unwrap(self, enabled: bool) -> None:
        self.active_reconstruction_unwrap_phase = bool(enabled)
        self._slice_cache.clear()

    def set_active_index(self, key: str, value: int) -> None:
        size = None
        for kind in ("image", "k_space", "map"):
            for control in self.dimension_controls(kind):
                if control["key"] == key:
                    size = int(control["size"])
                    break
            if size is not None:
                break
        value = max(0, int(value))
        if size is not None:
            value = min(value, max(0, size - 1))
        self.active_indices[key] = value
        if key == "frame":
            self.active_frame = value
        elif key == "z":
            self.active_z = value
        self._slice_cache.clear()

    def frame_image(
        self,
        frame: int | None = None,
        z: int | None = None,
        *,
        original: bool = False,
    ) -> np.ndarray:
        if self.is_non_imaging:
            vector = self.non_imaging_vector(frame=frame, original=original)
            return vector.reshape((1, int(vector.size)))
        frame_idx = self._bounded_frame(frame)
        z_idx = self.active_z if z is None else max(0, int(z))
        indices = dict(self.active_indices)
        indices["frame"] = frame_idx
        indices["z"] = z_idx
        source = "original" if original else "current"
        key = (
            "image",
            source,
            self.selected_experiment,
            self.active_image_plane,
            tuple(sorted(indices.items())),
        )
        if key not in self._slice_cache:
            self._slice_cache[key] = _display_slice(
                self.original_image_array if original else self.image_array,
                dims=self.experiment_dims,
                indices=indices,
                kind="image",
                plane_dims=self.image_plane_dims,
            )
        return self._slice_cache[key]

    def non_imaging_vector(self, frame: int | None = None, *, original: bool = False) -> np.ndarray:
        arr = np.asarray(self.original_image_array if original else self.image_array)
        if arr.ndim <= 1:
            return np.asarray(arr).reshape((-1,))
        indices = dict(self.active_indices)
        if frame is not None:
            indices["frame"] = self._bounded_frame(frame)
        controls = _axis_controls_for_non_image(arr, self.experiment_dims)
        slicer: list[object] = [slice(None)] * arr.ndim
        for control in controls:
            axis = int(control["axis"])
            key = str(control["key"])
            value = int(indices.get(key, 0))
            slicer[axis] = max(0, min(value, arr.shape[axis] - 1))
        selected = np.asarray(arr[tuple(slicer)])
        while selected.ndim > 1:
            selected = np.take(selected, 0, axis=0)
        return np.asarray(selected).reshape((-1,))

    def reconstruction_image(
        self,
        frame: int | None = None,
        z: int | None = None,
        *,
        component: str | None = None,
        coil: int | None = None,
        combined: bool = False,
        unwrap_phase: bool | None = None,
    ) -> np.ndarray:
        frame_idx = self._bounded_frame(frame)
        reconstruction = self.reconstruction_display_array(
            component=component,
            coil=coil,
            frame=frame_idx,
            combined=combined,
            unwrap_phase=unwrap_phase,
        )
        z_idx = self.active_z if z is None else max(0, int(z))
        indices = dict(self.active_indices)
        indices["frame"] = frame_idx
        indices["z"] = z_idx
        display_indices = dict(indices)
        display_indices["frame"] = 0
        key = (
            "reconstruction",
            self.selected_experiment,
            component or self.active_reconstruction_component,
            None if combined else (self.active_reconstruction_coil if coil is None else int(coil)),
            self.active_reconstruction_unwrap_phase if unwrap_phase is None else bool(unwrap_phase),
            self.active_image_plane,
            tuple(sorted(indices.items())),
        )
        if key not in self._slice_cache:
            self._slice_cache[key] = _display_slice(
                reconstruction,
                dims=self.experiment_dims,
                indices=display_indices,
                kind="image",
                plane_dims=self.image_plane_dims,
            )
        return self._slice_cache[key]

    def kspace_image(
        self,
        frame: int | None = None,
        z: int | None = None,
        mode: str | None = None,
    ) -> np.ndarray:
        kspace = self.kspace_array
        if kspace is None:
            raise RuntimeError("Current experiment does not contain raw k-space")
        frame_idx = self._bounded_frame(frame)
        z_idx = self.active_z if z is None else max(0, int(z))
        mode_name = mode or self.active_kspace_mode
        indices = dict(self.active_indices)
        indices["frame"] = frame_idx
        indices["z"] = z_idx
        key = (
            "kspace",
            self.selected_experiment,
            tuple(sorted(indices.items())),
            mode_name,
            self.active_kspace_centering,
        )
        if key not in self._slice_cache:
            slc = _display_slice(kspace, dims=self.experiment_dims, indices=indices, kind="kspace")
            self._slice_cache[key] = transform_kspace(slc, mode_name, center=self.active_kspace_centering)
        return self._slice_cache[key]

    def map_image(self, name: str | None = None, z: int | None = None) -> np.ndarray:
        map_name = name or self.active_map
        if not map_name:
            raise KeyError("No fitted map is selected")
        z_idx = self.active_z if z is None else max(0, int(z))
        indices = dict(self.active_indices)
        indices["z"] = z_idx
        key = (
            "map",
            self.selected_experiment,
            map_name,
            self.active_image_plane,
            tuple(sorted(indices.items())),
        )
        if key not in self._slice_cache:
            self._slice_cache[key] = _display_slice(
                self.map_array(map_name),
                dims=self.experiment_dims,
                indices=indices,
                kind="map",
                plane_dims=self.image_plane_dims,
            )
        return self._slice_cache[key]

    def add_roi(self, roi: ROI) -> None:
        roi = self._roi_with_inferred_dims(roi)
        self.rois = [existing for existing in self.rois if existing.name != roi.name]
        self.rois.append(roi)
        self._sync_shared_roi(roi)

    def remove_roi(self, name: str) -> None:
        self.rois = [roi for roi in self.rois if roi.name != name]

    def clear_rois(self) -> None:
        self.rois.clear()

    def enable_shared_roi(self, name: str) -> bool:
        for roi in self.shared_rois:
            if roi.name == name:
                self.rois = [existing for existing in self.rois if existing.name != name]
                self.rois.append(_copy_roi(roi))
                return True
        return False

    def remove_shared_roi(self, name: str) -> None:
        self.shared_rois = [roi for roi in self.shared_rois if roi.name != name]
        self.rois = [roi for roi in self.rois if roi.name != name]
        if self.visual_state.mask_roi == name:
            self.visual_state.mask_roi = None
        for state in self.view_states.values():
            state.rois = [roi for roi in state.rois if roi.name != name]
            if state.visual_state.mask_roi == name:
                state.visual_state.mask_roi = None

    def rename_roi(self, old_name: str, new_name: str) -> bool:
        old = str(old_name or "").strip()
        new = str(new_name or "").strip()
        if not old or not new:
            raise ValueError("ROI names cannot be empty")
        if old == new:
            return False
        all_names = {roi.name for roi in self.shared_rois}
        all_names.update(roi.name for roi in self.rois)
        for state in self.view_states.values():
            all_names.update(roi.name for roi in state.rois)
        if new in all_names:
            raise ValueError(f"ROI '{new}' already exists")

        def renamed(roi: ROI) -> ROI:
            if roi.name != old:
                return roi
            return ROI(new, np.asarray(roi.mask).copy(), dims=roi.dims, metadata=dict(roi.metadata))

        changed = any(roi.name == old for roi in self.shared_rois)
        changed = changed or any(roi.name == old for roi in self.rois)
        self.shared_rois = [renamed(roi) for roi in self.shared_rois]
        self.rois = [renamed(roi) for roi in self.rois]
        if self.fit_settings.selected_roi == old:
            self.fit_settings.selected_roi = new
            changed = True
        if self.visual_state.mask_roi == old:
            self.visual_state.mask_roi = new
            changed = True
        for state in self.view_states.values():
            if any(roi.name == old for roi in state.rois):
                changed = True
            state.rois = [renamed(roi) for roi in state.rois]
            if state.fit_settings.selected_roi == old:
                state.fit_settings.selected_roi = new
                changed = True
            if state.visual_state.mask_roi == old:
                state.visual_state.mask_roi = new
                changed = True
        return changed

    def _sync_shared_roi(self, roi: ROI) -> None:
        synced = _copy_roi(roi)
        self.shared_rois = [existing for existing in self.shared_rois if existing.name != roi.name]
        self.shared_rois.append(synced)
        for state in self.view_states.values():
            if any(existing.name == roi.name for existing in state.rois):
                state.rois = [
                    _copy_roi(roi) if existing.name == roi.name else existing
                    for existing in state.rois
                ]

    def apply_rois_to_all_experiments(self) -> None:
        current_rois = [_copy_roi(roi) for roi in self.rois]
        selected_roi = self.fit_settings.selected_roi
        current_names = {roi.name for roi in current_rois}
        if selected_roi not in current_names:
            selected_roi = current_rois[0].name if current_rois else None
        self.fit_settings.selected_roi = selected_roi
        self.shared_rois = [_copy_roi(roi) for roi in current_rois]
        current = self.selected_experiment
        if current:
            self._save_current_view_state()
        for name in self.experiment_names:
            state = self.view_states.get(name)
            if state is None:
                self.view_states[name] = ExperimentViewState(
                    active_map=None,
                    active_kspace_mode=self.active_kspace_mode,
                    active_kspace_centering=self.active_kspace_centering,
                    active_reconstruction_component=self.active_reconstruction_component,
                    active_reconstruction_coil=self.active_reconstruction_coil,
                    active_reconstruction_unwrap_phase=self.active_reconstruction_unwrap_phase,
                    rois=[_copy_roi(roi) for roi in current_rois],
                    fit_settings=FitSettings(
                        model_name=normalize_model_key((self.data["experiments"][name] or {}).get("model")),
                        selected_roi=selected_roi,
                    ),
                    frame_mask=FrameMask.all(max(1, self.frame_count_for_experiment(name))),
                )
            else:
                state.rois = [_copy_roi(roi) for roi in current_rois]
                state.fit_settings.selected_roi = selected_roi
        if current:
            self.selected_experiment = current
            self._restore_or_reset_current_view_state()
            self._slice_cache.clear()

    def frame_count_for_experiment(self, name: str) -> int:
        summary = self.summary_by_name.get(name, {})
        dims = {str(k): int(v) for k, v in dict(summary.get("dims") or {}).items()}
        if dims:
            return max(1, frame_count_from_dims(dims))
        exp = (self.data or {}).get("experiments", {}).get(name, {})
        image = _as_numpy(exp.get("mr_image"))
        return int(image.shape[0]) if image.ndim >= 3 else 1

    def roi(self, name: str | None = None) -> ROI | None:
        if not self.rois:
            return None
        if name is None:
            selected = self.fit_settings.selected_roi
            if selected:
                name = selected
            else:
                return self.rois[0]
        for roi in self.rois:
            if roi.name == name:
                return roi
        return None

    def roi_stats(
        self,
        *,
        map_names: Sequence[str] | None = None,
        roi_names: Sequence[str] | None = None,
        stats: Iterable[str] = ("mean", "variance"),
    ) -> list[dict[str, Any]]:
        names = list(map_names or self.map_names)
        maps = {name: self.fitted_maps[name] for name in names if name in self.fitted_maps}
        selected_rois = [
            roi for roi in self.rois if roi_names is None or roi.name in set(roi_names)
        ]
        spatial_dims = tuple(self.spatial_dim_names)
        target_dims = (
            spatial_dims
            if spatial_dims and all(_as_numpy(value).ndim == len(spatial_dims) for value in maps.values())
            else None
        )
        return collect_roi_stats(maps, selected_rois, stats=stats, target_dims=target_dims)

    def extract_timecourse(
        self,
        roi_name: str | None = None,
        *,
        source: str = "image",
        map_name: str | None = None,
        stat: str = "mean",
    ) -> dict[str, Any]:
        roi = self.roi(roi_name)
        if roi is None:
            raise RuntimeError("At least one ROI is required for timecourse extraction")

        if source == "map":
            if map_name is None:
                map_name = self.active_map
            if map_name is None:
                raise RuntimeError("No fitted map is selected")
            timecourse = extract_timecourse_array(self.fitted_maps[map_name], roi, stat=stat)
            return {"timecourse": timecourse, "dim_names": (), "observables": {}, "source": map_name}

        dims = self.experiment_dims
        if dims and self.image_array.ndim == len(dims):
            experiment = self.build_experiment()
            result = extract_timecourse_from_experiment(self.image_array, experiment, roi, stat=stat)
            result["source"] = "image"
            return result
        timecourse = extract_timecourse_array(self.image_array, roi, stat=stat)
        return {"timecourse": timecourse, "dim_names": (), "observables": {}, "source": "image"}

    def build_experiment(self) -> Experiment:
        entry = self.current_study_entry()
        if entry is not None:
            return entry[0]
        dims = self.experiment_dims
        if not dims:
            image = self.image_array
            if image.ndim <= 2:
                dims = {"x": int(image.shape[-2]), "y": int(image.shape[-1])}
            else:
                dims = {"frame": int(image.shape[0]), "x": int(image.shape[-2]), "y": int(image.shape[-1])}
        return Experiment(
            dims=dims,
            observables=dict(self.current_experiment.get("observables") or {}),
            exp_type=self.current_experiment.get("exp_type"),
        )

    def build_dataset(self) -> DataSet:
        entry = self.current_study_entry()
        if entry is not None:
            return entry[2]
        return DataSet(
            data=jnp.asarray(self.image_array),
            name=self.selected_experiment or "gui_dataset",
            processed=dict(self.processed_arrays or {}),
            kspace=None if self.kspace_array is None else jnp.asarray(self.kspace_array),
            spatial_geometry=_spatial_geometry_from_experiment_data(
                self.current_experiment,
                dims=self.experiment_dims,
            ),
            acquisition=self.current_experiment.get("acquisition"),
            trajectory=self.current_experiment.get("trajectory"),
            density=self.current_experiment.get("density"),
            sampling_mask=self.current_experiment.get("sampling_mask"),
        )

    def build_fit_inputs(self, model_name: str | None = None) -> tuple[Experiment, Any, DataSet, Study]:
        entry = self.current_study_entry()
        selected_key = normalize_model_key(model_name or self.fit_settings.model_name or self.current_model_name)
        source_model = entry[1] if entry is not None else None
        source_key = normalize_model_key(getattr(source_model, "name", None) if source_model is not None else self.current_model_name)
        saved_specs = {}
        if selected_key == source_key:
            if source_model is None:
                saved_specs = self.current_experiment.get("model_parameter_specs", {}) or {}
            else:
                saved_specs = (getattr(source_model, "parameter_specs", None) or
                    getattr(getattr(source_model, "parametrization_factory", None), "parameter_specs", {})) or {}
        parameter_overrides = {str(key): dict(value) for key, value in saved_specs.items()}
        for key, value in self.fit_settings.param_overrides.items():
            parameter_overrides[key] = {**parameter_overrides.get(key, {}), **value}
        if entry is not None:
            experiment, existing_model, dataset = entry
            selected = model_name or self.fit_settings.model_name
            if (
                self.fit_settings.param_overrides
                or self.fit_settings.selected_pools
                or (
                    selected
                    and normalize_model_key(selected)
                    != normalize_model_key(getattr(existing_model, "name", None))
                )
            ):
                model = make_model(
                    selected or getattr(existing_model, "name", None),
                    pools=self.fit_settings.selected_pools or None,
                    param_overrides=parameter_overrides or None,
                    pool_overrides=self.fit_settings.pool_overrides or None,
                    parameter_constant_dims=self.fit_settings.parameter_constant_dims or None,
                )
            else:
                model = existing_model
        else:
            experiment = self.build_experiment()
            model = make_model(
                model_name or self.fit_settings.model_name or self.current_model_name,
                pools=self.fit_settings.selected_pools or None,
                param_overrides=parameter_overrides or None,
                pool_overrides=self.fit_settings.pool_overrides or None,
                parameter_constant_dims=self.fit_settings.parameter_constant_dims or None,
            )
            dataset = self.build_dataset()
        if experiment.exp_type == "sat_trans":
            source_study = self.fit_study()
            if source_study is None or self.selected_experiment is None:
                raise RuntimeError("A loaded study is required to normalize saturation-transfer scans")
            experiment, dataset, _result = prepare_ab_mt_z_spectra(
                source_study,
                self.selected_experiment,
                reference=self.fit_settings.sat_trans_reference_scan,
            )
        if normalize_model_key(getattr(model, "name", None)) in {"ab_mt_n", "zaiss_full"}:
            experiment = self._insert_ab_mt_prior_maps(experiment)
        study = Study(
            [experiment],
            [dataset],
            [model],
            names=[self.selected_experiment or dataset.name],
            study_globals=dict((self.data or {}).get("study_globals") or {}),
        )
        return experiment, model, dataset, study

    def _insert_ab_mt_prior_maps(self, experiment: Experiment) -> Experiment:
        updates: dict[str, Quantity] = {}
        required = (
            ("t1", "t1_map", self.fit_settings.ab_mt_t1_map_ref),
            ("b1", "b1_map", self.fit_settings.ab_mt_b1_map_ref),
            ("b0", "b0_map", self.fit_settings.ab_mt_b0_map_ref),
        )
        missing: list[str] = []
        for kind, observable_name, selected_ref in required:
            choice = self._resolve_ab_mt_map_choice(kind, selected_ref)
            if choice is None:
                missing.append(kind.upper())
                continue
            value = self._map_from_ref(choice["ref"])
            if value is None:
                missing.append(kind.upper())
                continue
            updates[observable_name] = self._ab_mt_observable_from_choice(kind, choice, value)
        if missing:
            raise RuntimeError(
                "AB-MT/Zaiss fitting requires fitted prior maps from this study: "
                + ", ".join(missing)
                + ". Run or load those fits, then select them in the Fit tab."
            )
        return experiment.insert_observables(updates)

    def _resolve_ab_mt_map_choice(self, kind: str, selected_ref: str | None) -> dict[str, str] | None:
        choices = self.ab_mt_map_choices(kind)
        if selected_ref:
            for choice in choices:
                if choice["ref"] == selected_ref:
                    return choice
            return None
        return choices[0] if choices else None

    def _map_from_ref(self, ref: str) -> Any | None:
        decoded = _decode_map_ref(ref)
        if decoded is None or not self.data:
            return None
        exp_name, map_name = decoded
        exp_data = self.data["experiments"].get(exp_name) or {}
        return dict(exp_data.get("fitted_maps") or {}).get(map_name)

    def _ab_mt_observable_from_choice(self, kind: str, choice: Mapping[str, str], value: Any) -> Quantity:
        unit = choice.get("unit") or ""
        arr = _as_numpy(value)
        if kind == "b1" and str(choice.get("map") or "").lower() == "b1" and unit.lower() == "ut":
            decoded = _decode_map_ref(choice["ref"])
            nominal = None
            if decoded is not None and self.data:
                exp_data = self.data["experiments"].get(decoded[0]) or {}
                sat_b1 = dict(exp_data.get("observables") or {}).get("sat_b1")
                if sat_b1 is not None:
                    sat_arr = _as_numpy(sat_b1).reshape((-1,))
                    if sat_arr.size:
                        nominal = float(sat_arr[0])
            if nominal is None or abs(nominal) < 1e-12:
                raise RuntimeError(
                    f"Cannot convert {choice.get('label', 'selected B1 map')} from uT to percent: "
                    "source scan is missing a non-zero sat_b1 observable."
                )
            return Quantity(jnp.asarray((arr - nominal) / nominal * 100.0), "%")
        if isinstance(value, Quantity):
            return value
        return Quantity(jnp.asarray(arr), unit or _default_ab_mt_map_unit(kind))

    def run_current_fit(
        self,
        *,
        model_name: str | None = None,
        roi_name: str | None = None,
        use_roi: bool = True,
        input_experiment: Experiment | None = None,
        input_dataset: DataSet | None = None,
        input_roi: ROI | None = None,
        num_steps: int | None = None,
        step_size: float | None = None,
        optimizer_name: str | None = None,
        compute_derived_maps: bool | None = None,
        progress_callback: Callable[[int, int, str], None] | None = None,
        device: DeviceLike = "cpu",
    ) -> dict[str, Any]:
        steps = self.fit_settings.num_steps if num_steps is None else int(num_steps)
        if progress_callback is not None:
            progress_callback(0, max(1, steps), "Preparing full-map fit")
        experiment, model, dataset, _study = self.build_fit_inputs(model_name)
        if input_experiment is not None or input_dataset is not None:
            if input_experiment is None or input_dataset is None:
                raise RuntimeError("View-state fitting requires both an experiment and dataset override")
            experiment = input_experiment
            dataset = input_dataset
        roi = input_roi if input_roi is not None else (self.roi(roi_name) if use_roi else None)
        mask = roi.mask if roi is not None else None
        mask_dims = self._roi_mask_dims(roi, tuple(experiment.spatial_dim_names)) if roi is not None else None
        frame_mask = self.frame_mask.as_array() if self.frame_mask is not None else None
        initial_params, fit_notes = self._dti_initial_params(
            experiment,
            model,
            dataset,
            frame_mask=frame_mask,
        )
        fields, losses, fit_info = run_fit(
            experiment,
            model,
            dataset,
            optimizer_name=optimizer_name or self.fit_settings.optimizer_name,
            num_steps=steps,
            step_size=self.fit_settings.step_size if step_size is None else float(step_size),
            mask=mask,
            mask_dims=mask_dims,
            frame_mask=frame_mask,
            mask_fill_value=self.fit_settings.mask_fill_value,
            study_globals=dict((self.data or {}).get("study_globals") or {}),
            auto_mask_nan_observables=self.fit_settings.auto_mask_nan_observables,
            regularizer_configs=self.fit_settings.regularizer_configs or None,
            initial_params=initial_params,
            compute_derived_maps=(
                self.fit_settings.compute_derived_maps
                if compute_derived_maps is None
                else compute_derived_maps
            ),
            progress_callback=progress_callback,
            return_fit_info=True,
            device=device,
        )
        if progress_callback is not None:
            progress_callback(max(1, steps), max(1, steps), "Finalizing fitted maps")
        fields, replacement_notes = self._maybe_replace_degenerate_dti_maps(
            fields,
            experiment,
            model,
            dataset,
            mask=mask,
            mask_dims=mask_dims,
            frame_mask=frame_mask,
        )
        fit_notes.extend(replacement_notes)
        if fit_info.regularization_mode == "smoothed_tv":
            fit_notes.append(f"Warning: {fit_info.optimizer_name} uses smoothed TV gradients; choose auto, FISTA, or split_bregman for true TV proximal fitting.")
        return {
            "name": self.selected_experiment or "gui_fit",
            "model": model.name,
            "fields": fields,
            "loss_history": losses,
            "cached": False,
            "fit_notes": fit_notes,
            "optimizer_name": fit_info.optimizer_name,
            "regularization_mode": fit_info.regularization_mode,
            "spatial_geometry": (
                dataset.spatial_geometry.to_metadata()
                if dataset.spatial_geometry is not None
                else None
            ),
        }

    def _dti_initial_params(
        self,
        experiment: Experiment,
        model: Any,
        dataset: DataSet,
        *,
        frame_mask: np.ndarray | None,
    ) -> tuple[dict[str, Any] | None, list[str]]:
        model_name = str(getattr(model, "name", "") or "").lower()
        if model_name not in {"dti", "dti_tensor", "diffusion"}:
            return None, []
        obs = dict(experiment.observables)
        bvals = obs.get("bvals") or obs.get("bval")
        bvecs = obs.get("bvecs") or obs.get("bvec")
        if bvals is None or bvecs is None:
            return None, []
        try:
            data = reshape_dims_to_frames(_as_numpy(dataset.data), experiment.dims)
            bvals_fit = bvals
            bvecs_fit = bvecs
            if frame_mask is not None:
                fm = np.asarray(frame_mask, dtype=bool).reshape((-1,))
                if int(fm.size) == int(data.shape[0]):
                    data = data[fm]
                    bvals_fit = _slice_frame_values(bvals, fm)
                    bvecs_fit = _slice_frame_values(bvecs, fm)
            rank = dti_design_rank(bvals_fit, bvecs_fit)
            if rank < 7:
                return None, [
                    f"DTI tensor initialization skipped because the active frame set has design rank {rank}/7; include b0 frames and at least six independent diffusion directions for tensor fitting."
                ]
            direct = fit_dti_log_linear(data, bvals_fit, bvecs_fit, mask=None)
            initial_params = _fields_to_initial_params(model, direct)
        except Exception:
            return None, []
        if initial_params is None:
            return None, []
        return initial_params, [
            f"DTI tensor fit initialized from a full-rank log-linear estimate (design rank {rank}/7)."
        ]

    def _maybe_replace_degenerate_dti_maps(
        self,
        fields: Mapping[str, Any],
        experiment: Experiment,
        model: Any,
        dataset: DataSet,
        *,
        mask: Any | None,
        mask_dims: tuple[str, ...] | None,
        frame_mask: np.ndarray | None,
    ) -> tuple[dict[str, Any], list[str]]:
        model_name = str(getattr(model, "name", "") or "").lower()
        if model_name not in {"dti", "dti_tensor", "diffusion", "dki"}:
            return dict(fields), []
        obs = dict(experiment.observables)
        bvals = obs.get("bvals") or obs.get("bval")
        bvecs = obs.get("bvecs") or obs.get("bvec")
        if bvals is None:
            return dict(fields), []
        spatial_mask = None
        try:
            data = reshape_dims_to_frames(_as_numpy(dataset.data), experiment.dims)
            bvals_fit = bvals
            bvecs_fit = bvecs
            if frame_mask is not None:
                fm = np.asarray(frame_mask, dtype=bool).reshape((-1,))
                if int(fm.size) == int(data.shape[0]):
                    data = data[fm]
                    bvals_fit = _slice_frame_values(bvals, fm)
                    bvecs_fit = _slice_frame_values(bvecs, fm)
            if mask is not None:
                spatial_mask = broadcast_mask(
                    mask,
                    experiment.spatial_shape,
                    mask_dims=mask_dims,
                    target_dims=tuple(experiment.spatial_dim_names),
                )
            active_rank = dti_design_rank(bvals_fit, bvecs_fit)
            if active_rank >= 7:
                return dict(fields), []
            direct = fit_dti_log_linear(data, bvals_fit, bvecs_fit, mask=spatial_mask)
        except Exception:
            return dict(fields), []

        current_adc = _field_array(fields.get("adc"))
        direct_adc = np.asarray(direct.get("adc"), dtype=float)
        if spatial_mask is not None:
            try:
                mask_arr = np.asarray(spatial_mask, dtype=bool)
                if current_adc.shape == mask_arr.shape:
                    current_adc = np.where(mask_arr, current_adc, np.nan)
                if direct_adc.shape == mask_arr.shape:
                    direct_adc = np.where(mask_arr, direct_adc, np.nan)
            except Exception:
                _LOG.debug("Ignored optional operation failure", exc_info=True)
        current_std = _finite_std(current_adc)
        direct_std = _finite_std(direct_adc)
        current_finite = int(np.isfinite(current_adc).sum()) if current_adc.size else 0
        direct_finite = int(np.isfinite(direct_adc).sum()) if direct_adc.size else 0
        replace = (
            direct_finite > 0
            and (
                current_finite == 0
                or not np.isfinite(current_std)
                or current_std <= max(float(direct_std) * 0.05, 1e-12)
            )
        )
        if not replace:
            return dict(fields), []

        merged = dict(fields)
        parameter_units = getattr(model, "parameter_units", {}) or {}
        derived_units = getattr(model, "derived_map_units", {}) or {}
        for name, value in direct.items():
            unit = parameter_units.get(name) or derived_units.get(name)
            merged[name] = Quantity(jnp.asarray(value), unit) if unit else jnp.asarray(value)
        return merged, [
            f"DTI tensor maps were replaced with an isotropic log-linear estimate because the active frame set is rank-deficient ({active_rank}/7). Include the b0 frames to run the tensor model."
        ]

    def run_fit_sanity_check(
        self,
        *,
        model_name: str | None = None,
        input_experiment: Experiment | None = None,
        input_dataset: DataSet | None = None,
    ) -> dict[str, Any]:
        """Run the core MRax pre-fit sanity check for current GUI fit inputs."""

        try:
            experiment, model, dataset, _study = self.build_fit_inputs(model_name)
            if input_experiment is not None or input_dataset is not None:
                if input_experiment is None or input_dataset is None:
                    raise RuntimeError("View-state fitting requires both an experiment and dataset override")
                experiment = input_experiment
                dataset = input_dataset
            report = experiment.sanity_check(
                model,
                dataset,
                study_globals=dict((self.data or {}).get("study_globals") or {}),
            )
            text = report.explain()
            ok = bool(report.ok)
        except Exception as exc:
            text = "Fit sanity check\nerrors:\n  - " + str(exc) + "\nstatus: fix required items before running run_fit"
            ok = False
        return {
            "name": self.selected_experiment or "gui_fit",
            "model": model_name or self.fit_settings.model_name or self.current_model_name,
            "ok": ok,
            "report": text,
        }

    def run_preview_fit(
        self,
        *,
        model_name: str | None = None,
        roi_name: str | None = None,
        input_experiment: Experiment | None = None,
        input_dataset: DataSet | None = None,
        input_roi: ROI | None = None,
        num_steps: int | None = None,
        step_size: float | None = None,
        progress_callback: Callable[[int, int, str], None] | None = None,
        device: DeviceLike = "cpu",
    ) -> dict[str, Any]:
        steps = self.fit_settings.num_steps if num_steps is None else int(num_steps)
        if progress_callback is not None:
            progress_callback(0, max(1, steps), "Preparing ROI preview fit")
        roi = input_roi if input_roi is not None else self.roi(roi_name)
        if roi is None:
            raise RuntimeError("At least one ROI is required for preview fitting")
        experiment, model, dataset, _study = self.build_fit_inputs(model_name)
        if input_experiment is not None or input_dataset is not None:
            if input_experiment is None or input_dataset is None:
                raise RuntimeError("View-state fitting requires both an experiment and dataset override")
            experiment = input_experiment
            dataset = input_dataset
        extracted = extract_timecourse_from_experiment(np.asarray(dataset.data), experiment, roi, stat="mean")
        observed = _as_numpy(extracted["timecourse"])
        non_spatial = tuple(int(experiment.dims[name]) for name in experiment.non_spatial_dim_names)
        preview_dims = {name: int(experiment.dims[name]) for name in experiment.non_spatial_dim_names}
        preview_dims.update({"x": 1, "y": 1})
        preview_data = observed.reshape(non_spatial + (1, 1))
        preview_exp = Experiment(
            dims=preview_dims,
            observables=dict(experiment.observables),
            exp_type=experiment.exp_type,
        )
        preview_ds = DataSet(data=jnp.asarray(preview_data), name="roi_preview")
        frame_mask = self.frame_mask.as_array() if self.frame_mask is not None else None
        fields, losses, model_fn = run_fit(
            preview_exp,
            model,
            preview_ds,
            num_steps=steps,
            step_size=self.fit_settings.step_size if step_size is None else float(step_size),
            frame_mask=frame_mask,
            study_globals=dict((self.data or {}).get("study_globals") or {}),
            auto_mask_nan_observables=self.fit_settings.auto_mask_nan_observables,
            regularizer_configs=self.fit_settings.regularizer_configs or None,
            return_model_callable=True,
            progress_callback=progress_callback,
            device=device,
        )
        if progress_callback is not None:
            progress_callback(max(1, steps), max(1, steps), "Rendering preview fit")
        predicted = model_fn(fields)
        x_name, x_values = frame_axis_values(experiment)
        return {
            "observed": reshape_dims_to_frames(preview_data, preview_dims).reshape((-1,)),
            "predicted": reshape_dims_to_frames(_as_numpy(predicted), preview_dims).reshape((-1,)),
            "included": frame_mask if frame_mask is not None else np.ones((self.frame_count,), dtype=bool),
            "x": x_values,
            "x_name": x_name,
            "fields": fields,
            "loss_history": losses,
            "model": model.name,
        }

    def save_fit_result(self, fit_result: Mapping[str, Any], output_dir: str | Path, *, bundle=True, progress_callback=None) -> Path:
        _experiment, model, _dataset, study = self.build_fit_inputs(fit_result.get("model"))
        return save_study_results(
            study,
            [dict(fit_result)],
            output_dir=output_dir,
            study_name=Path(output_dir).stem,
            bundle=bundle,
            progress_callback=progress_callback,
        )

    def save_imported_study(
        self,
        output_dir: str | Path,
        *,
        study_name: str | None = None,
        bundle: bool = True,
        progress_callback=None,
    ) -> Path:
        """Save the current study, with a portable bundle by default."""

        study = self.fit_study()
        if study is None:
            raise RuntimeError("No imported Study is available to save")
        out = Path(output_dir)
        results = [
            {
                "name": name,
                "model": model.name,
                "fields": dict(((self.data or {}).get("experiments", {}).get(name, {})).get("fitted_maps", {})),
                "loss_history": [],
                "cached": False,
            }
            for name, _experiment, model, _dataset in study.iter_entries()
        ]
        metadata = (self.import_metadata or {}).get("experiments")
        if not isinstance(metadata, dict):
            metadata = {name: dict(entry.get("extra_metadata") or {})
                        for name, entry in ((self.data or {}).get("experiments") or {}).items()}
        study_to_save = _study_with_save_compatible_image_dims(
            study,
            metadata if isinstance(metadata, dict) else None,
        )
        return save_study_results(
            study_to_save,
            results,
            output_dir=out,
            study_name=study_name or out.stem,
            bundle=bundle,
            progress_callback=progress_callback,
            experiment_metadata=metadata if isinstance(metadata, dict) else None,
        )

    def diagnostics(self) -> list[Diagnostic]:
        if not self.is_loaded:
            return [Diagnostic("info", "No MRax manifest is loaded.")]
        exp = self.current_experiment
        dims = self.experiment_dims
        diagnostics: list[Diagnostic] = []
        image = self.image_array
        diagnostics.append(Diagnostic("info", f"Image shape: {tuple(image.shape)}"))
        if self.is_non_imaging:
            diagnostics.append(Diagnostic("info", f"Non-imaging sequence: {self.data_role}. Data is plotted as a 1D signal."))
        if dims:
            diagnostics.append(Diagnostic("info", f"Experiment dims: {dims}"))
            expected = tuple(int(v) for v in dims.values())
            if tuple(image.shape[: len(expected)]) != expected:
                diagnostics.append(
                    Diagnostic("warning", f"Image shape {tuple(image.shape)} does not match dims {expected}.")
                )
        else:
            diagnostics.append(Diagnostic("warning", "Manifest summary does not include experiment dims."))

        if np.isfinite(image).all():
            diagnostics.append(Diagnostic("ok", "Image contains no NaN/inf values."))
        else:
            invalid = int(np.size(image) - np.isfinite(image).sum())
            diagnostics.append(Diagnostic("warning", f"Image contains {invalid} NaN/inf values."))

        if exp.get("raw_kspace") is None:
            diagnostics.append(Diagnostic("info", "No raw k-space stored for this experiment."))
        else:
            diagnostics.append(Diagnostic("ok", f"K-space shape: {tuple(self.kspace_array.shape)}"))
            try:
                slc = _display_slice(
                    self.kspace_array,
                    dims=self.experiment_dims,
                    indices=dict(self.active_indices),
                    kind="kspace",
                )
                diagnostics.append(
                    Diagnostic(
                        "info",
                        f"K-space display centering: {kspace_centering_decision(slc, self.active_kspace_centering)}.",
                    )
                )
            except Exception:
                _LOG.debug("Ignored optional operation failure", exc_info=True)

        if self.map_names:
            diagnostics.append(Diagnostic("ok", f"Fitted maps: {', '.join(self.map_names)}"))
        else:
            diagnostics.append(Diagnostic("info", "No fitted maps stored for this experiment."))

        for roi in self.rois:
            if dims:
                spatial_shape = tuple(int(dims[name]) for name in spatial_names(dims))
                spatial_dim_names = tuple(spatial_names(dims))
                mask_dims = self._roi_mask_dims(roi, spatial_dim_names)
                try:
                    broadcast_mask(
                        roi.mask,
                        spatial_shape,
                        mask_dims=mask_dims,
                        target_dims=spatial_dim_names,
                    )
                except Exception:
                    diagnostics.append(
                        Diagnostic(
                            "warning",
                            f"ROI '{roi.name}' shape {roi.mask.shape} does not match spatial shape {spatial_shape}.",
                        )
                    )
        try:
            report = self.build_experiment().explain_sanity(
                make_model(
                    self.fit_settings.model_name or self.current_model_name,
                    pools=self.fit_settings.selected_pools or None,
                    param_overrides=self.fit_settings.param_overrides or None,
                    pool_overrides=self.fit_settings.pool_overrides or None,
                    parameter_constant_dims=self.fit_settings.parameter_constant_dims or None,
                ),
                self.build_dataset(),
                study_globals=dict((self.data or {}).get("study_globals") or {}),
            )
            first = report.splitlines()[0] if report else "Fit sanity check completed."
            diagnostics.append(Diagnostic("info", first))
        except Exception as exc:
            diagnostics.append(Diagnostic("warning", f"Fit sanity check failed: {exc}"))
        return diagnostics

    def _bounded_frame(self, frame: int | None) -> int:
        frame_count = max(1, self.frame_count)
        if frame is None:
            frame = self.active_frame
        return max(0, min(int(frame), frame_count - 1))

    def _roi_with_inferred_dims(self, roi: ROI) -> ROI:
        if roi.dims is not None or not self.is_loaded:
            return roi
        spatial_dims = tuple(self.spatial_dim_names)
        inferred = self._infer_mask_dims(np.asarray(roi.mask).shape, spatial_dims)
        if inferred is None:
            return roi
        return ROI(roi.name, roi.mask, dims=inferred, metadata=dict(roi.metadata))

    def _roi_mask_dims(self, roi: ROI | None, spatial_dims: Sequence[str] | None = None) -> tuple[str, ...] | None:
        if roi is None:
            return None
        if roi.dims is not None:
            return tuple(roi.dims)
        return self._infer_mask_dims(np.asarray(roi.mask).shape, tuple(spatial_dims or self.spatial_dim_names))

    def _infer_mask_dims(
        self,
        mask_shape: Sequence[int],
        spatial_dims: Sequence[str],
    ) -> tuple[str, ...] | None:
        spatial_dim_names = tuple(str(dim) for dim in spatial_dims)
        if not spatial_dim_names:
            return None
        try:
            spatial_shape = tuple(int(self.experiment_dims[dim]) for dim in spatial_dim_names)
        except Exception:
            return None
        shape = tuple(int(v) for v in mask_shape)
        if shape == spatial_shape:
            return spatial_dim_names
        if len(shape) <= len(spatial_shape) and shape == spatial_shape[: len(shape)]:
            return spatial_dim_names[: len(shape)]
        if len(shape) <= len(spatial_shape) and shape == spatial_shape[-len(shape) :]:
            return spatial_dim_names[-len(shape) :]
        return None

    def _ensure_original_image_snapshots(self) -> None:
        for exp_data in ((self.data or {}).get("experiments") or {}).values():
            self._ensure_original_image_snapshot(exp_data)

    @staticmethod
    def _ensure_original_image_snapshot(exp_data: MutableMapping[str, Any]) -> None:
        if exp_data.get("original_mr_image") is not None or exp_data.get("mr_image") is None:
            return
        exp_data["original_mr_image"] = np.asarray(exp_data.get("mr_image")).copy()


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, Quantity):
        value = value.values
    return np.asarray(value)


def _empty_session_data(study_name: str, *, study_globals: Mapping[str, Quantity] | None = None) -> dict[str, Any]:
    return {
        "manifest": {
            "study_name": str(study_name or "Untitled Study"),
            "summary": {"experiments": []},
            "targets": [],
            "source": {"kind": "memory", "name": str(study_name or "Untitled Study")},
        },
        "experiments": {},
        "study_globals": dict(study_globals or {}),
    }


def _validated_new_experiment_name(name: str, existing: Sequence[str]) -> str:
    value = str(name or "").strip()
    if not value:
        raise ValueError("Experiment name cannot be empty")
    if value in set(existing):
        raise ValueError(f"Experiment '{value}' already exists")
    return value


def _validated_dims(dims: Mapping[str, int]) -> dict[str, int]:
    result: dict[str, int] = {}
    for key, value in dict(dims or {}).items():
        name = str(key or "").strip()
        if not name:
            raise ValueError("Dimension names cannot be empty")
        size = int(value)
        if size <= 0:
            raise ValueError(f"Dimension '{name}' must have a positive size")
        if name in result:
            raise ValueError(f"Duplicate dimension name '{name}'")
        result[name] = size
    if not result:
        raise ValueError("At least one dimension is required")
    return result


def _infer_dims_from_session_entry(exp_data: Mapping[str, Any]) -> dict[str, int]:
    image = _as_numpy(exp_data.get("mr_image"))
    if image.ndim == 0:
        return {"sample": 1}
    if image.ndim == 1:
        return {"sample": int(image.shape[0])}
    if image.ndim == 2:
        return {"x": int(image.shape[0]), "y": int(image.shape[1])}
    if image.ndim == 3:
        return {"frame": int(image.shape[0]), "x": int(image.shape[1]), "y": int(image.shape[2])}
    dims = {"frame": int(image.shape[0])}
    for idx, size in enumerate(image.shape[1:]):
        dims[f"dim_{idx}"] = int(size)
    return dims


def _field_array(value: Any) -> np.ndarray:
    if value is None:
        return np.asarray([])
    return np.asarray(value.values if isinstance(value, Quantity) else value, dtype=float)


def _fields_to_initial_params(model: Any, fields: Mapping[str, Any]) -> dict[str, Any] | None:
    specs = getattr(model, "_mrax_param_specs", None)
    if not isinstance(specs, Mapping):
        return None
    spatial_shape = None
    for value in fields.values():
        arr = _field_array(value)
        if arr.ndim:
            spatial_shape = tuple(int(v) for v in arr.shape)
            break
    if spatial_shape is None:
        return None
    try:
        default_params, unpack = model.parametrization_factory(spatial_shape)
        default_fields = unpack(default_params)
    except Exception:
        return None

    initial: dict[str, Any] = {}
    for name, cfg_obj in specs.items():
        if not isinstance(cfg_obj, Mapping) or not cfg_obj.get("trainable", True):
            continue
        key = f"{name}_raw"
        if key not in default_params or name not in fields:
            continue
        cfg = dict(cfg_obj)
        value = _field_array(fields[name])
        if value.shape != spatial_shape:
            try:
                value = np.broadcast_to(value, spatial_shape)
            except Exception:
                continue
        default_value = _field_array(default_fields.get(name))
        if default_value.shape != spatial_shape:
            default_value = np.broadcast_to(default_value, spatial_shape)
        value = np.where(np.isfinite(value), value, default_value)
        try:
            initial[key] = jnp.asarray(_inverse_model_transform(value, cfg))
        except Exception:
            initial[key] = default_params[key]
    if set(initial) != set(default_params):
        return None
    return initial


def _inverse_model_transform(value: np.ndarray, cfg: Mapping[str, Any]) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    eps = 1e-6
    transform = str(cfg.get("transform", "softplus"))
    if transform == "softplus":
        scale = float(cfg.get("scale", 1.0))
        offset = float(cfg.get("offset", 0.0))
        scaled = np.maximum((arr - offset) / max(scale, eps), eps)
        return np.where(scaled > 20.0, scaled, np.log(np.expm1(scaled)))
    if transform == "bounded":
        lo, hi = [float(v) for v in cfg["bounds"]]
        denom = hi - lo
        if abs(denom) < eps:
            return np.zeros_like(arr)
        frac = np.clip((arr - lo) / denom, eps, 1.0 - eps)
        return np.log(frac) - np.log1p(-frac)
    if transform == "tanh":
        scale = float(cfg.get("scale", 1.0))
        scaled = np.clip(arr / max(abs(scale), eps), -1.0 + eps, 1.0 - eps)
        return np.arctanh(scaled)
    if transform == "linear":
        scale = float(cfg.get("scale", 1.0))
        offset = float(cfg.get("offset", 0.0))
        return (arr - offset) / max(scale, eps)
    if transform == "identity":
        return arr
    raise ValueError(f"Unknown transform '{transform}'")


def _slice_frame_values(value: Any, frame_mask: np.ndarray) -> Any:
    if value is None:
        return None
    arr = np.asarray(value.values if isinstance(value, Quantity) else value)
    mask = np.asarray(frame_mask, dtype=bool).reshape((-1,))
    if arr.ndim >= 1 and int(arr.shape[0]) == int(mask.size):
        sliced = arr[mask]
    elif int(arr.size) == int(mask.size):
        sliced = arr.reshape((-1,))[mask]
    else:
        return value
    if isinstance(value, Quantity):
        return Quantity(jnp.asarray(sliced), value.si_unit)
    return sliced


def _finite_std(value: np.ndarray) -> float:
    arr = np.asarray(value, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(np.std(finite))


def _study_with_save_compatible_image_dims(
    study: Study,
    experiment_metadata: Mapping[str, Any] | None = None,
) -> Study:
    experiments: list[Experiment] = []
    datasets: list[DataSet] = []
    changed = False
    for name, experiment, _model, dataset in study.iter_entries():
        metadata = dict((experiment_metadata or {}).get(name) or {})
        updated_experiment, updated_dataset = _entry_with_save_compatible_image_dims(experiment, dataset, metadata)
        experiments.append(updated_experiment)
        datasets.append(updated_dataset)
        changed = changed or updated_experiment is not experiment or updated_dataset is not dataset
    if not changed:
        return study
    return Study(
        experiments,
        datasets,
        list(study.models),
        names=list(study.names) if study.names is not None else None,
        study_globals=dict(study.study_globals or {}),
    )


def _entry_with_save_compatible_image_dims(
    experiment: Experiment,
    dataset: DataSet,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[Experiment, DataSet]:
    if _study_entry_is_non_imaging(experiment, dataset, metadata):
        return experiment, dataset
    data = np.asarray(dataset.data)
    names = [str(name) for name in experiment.dims.keys()]
    sizes = {str(name): int(size) for name, size in experiment.dims.items()}
    if data.ndim < len(names):
        return experiment, dataset
    if "x" in sizes and "y" in sizes and data.ndim == len(names):
        return experiment, dataset

    arr = data
    inserted_axes: list[int] = []

    def add_dim(dim: str) -> None:
        nonlocal arr, names, sizes, inserted_axes
        if dim in sizes:
            return
        if arr.ndim > len(names):
            axis = len(names)
            names.insert(axis, dim)
            sizes[dim] = int(arr.shape[axis])
            return
        if dim == "x" and "y" in names:
            axis = names.index("y")
        elif dim == "y" and "x" in names:
            axis = names.index("x") + 1
        else:
            axis = len(names)
        arr = np.expand_dims(arr, axis=axis)
        names.insert(axis, dim)
        sizes[dim] = 1
        inserted_axes.append(axis)

    add_dim("x")
    add_dim("y")
    if arr.ndim > len(names) and "z" not in sizes:
        axis = len(names)
        names.insert(axis, "z")
        sizes["z"] = int(arr.shape[axis])

    if arr.ndim != len(names):
        return experiment, dataset

    updated_dims = {name: int(sizes[name]) for name in names}
    updated_experiment = _copy_experiment_with_dims(experiment, updated_dims)
    updated_kspace = _expand_kspace_for_inserted_image_axes(
        dataset.kspace,
        inserted_axes=inserted_axes,
        original_data_ndim=data.ndim,
        original_non_spatial_count=len(non_spatial_names(experiment.dims)),
    )
    updated_dataset = replace(
        dataset,
        data=jnp.asarray(arr),
        kspace=updated_kspace,
    )
    return updated_experiment, updated_dataset


def _study_entry_is_non_imaging(
    experiment: Experiment,
    dataset: DataSet | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> bool:
    if bool(getattr(experiment, "_mrax_non_imaging", False)):
        return True
    meta = dict(metadata or {})
    if bool(meta.get("non_imaging")):
        return True
    exp_type = str(experiment.exp_type or "").strip().lower()
    if exp_type in {"spectroscopy", "mrs", "fid", "non_imaging", "non-imaging"}:
        return True
    return False


def _study_entry_data_role(
    experiment: Experiment,
    dataset: DataSet | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> str:
    meta = dict(metadata or {})
    role = str(
        meta.get("data_role")
        or getattr(experiment, "_mrax_data_role", "")
        or experiment.exp_type
        or ""
    ).strip()
    if role:
        return role
    return "non_imaging" if _study_entry_is_non_imaging(experiment, dataset, metadata) else "image"


def _session_experiment_is_non_imaging(exp_data: Mapping[str, Any], dims: Mapping[str, int] | None = None) -> bool:
    if bool(exp_data.get("non_imaging")):
        return True
    extra = dict(exp_data.get("extra_metadata") or {})
    if bool(extra.get("non_imaging")):
        return True
    exp_type = str(exp_data.get("exp_type") or "").strip().lower()
    return exp_type in {"spectroscopy", "mrs", "fid", "non_imaging", "non-imaging"}


def _session_experiment_data_role(exp_data: Mapping[str, Any], dims: Mapping[str, int] | None = None) -> str:
    extra = dict(exp_data.get("extra_metadata") or {})
    role = str(extra.get("data_role") or exp_data.get("data_role") or exp_data.get("exp_type") or "").strip()
    if role:
        return role
    return "non_imaging" if _session_experiment_is_non_imaging(exp_data, dims) else "image"


def _spatial_geometry_from_metadata(
    metadata: Mapping[str, Any] | None,
    *,
    spatial_shape: Sequence[int] | None = None,
) -> SpatialGeometry:
    values = dict(metadata or {})
    geometry_value = values.get("spatial_geometry")
    if isinstance(geometry_value, Mapping):
        geometry = SpatialGeometry.from_metadata(geometry_value)
        if geometry.spatial_shape is not None and spatial_shape is not None:
            if tuple(geometry.spatial_shape) != tuple(int(value) for value in spatial_shape):
                raise ValueError(
                    f"Spatial geometry shape {geometry.spatial_shape} does not match "
                    f"image spatial shape {tuple(spatial_shape)}."
                )
        if geometry.spatial_shape is not None or spatial_shape is None:
            return geometry
        payload = geometry.to_metadata()
        payload["spatial_shape"] = list(spatial_shape)
        return SpatialGeometry.from_metadata(payload)
    affine_values = values.get("affines")
    if affine_values is None:
        affine_values = values.get("affine")
    source = str(values.get("import_format") or values.get("format") or "unknown")
    return SpatialGeometry.from_affines(
        affine_values,
        spatial_shape=spatial_shape,
        source=source,
    )


def _spatial_geometry_from_experiment_data(
    exp_data: Mapping[str, Any],
    *,
    dims: Mapping[str, int] | None = None,
) -> SpatialGeometry:
    dims_map = dict(dims or exp_data.get("dims") or {})
    spatial_shape = tuple(int(dims_map[name]) for name in spatial_names(dims_map)) or None
    direct = exp_data.get("spatial_geometry")
    if isinstance(direct, Mapping):
        return _spatial_geometry_from_metadata(
            {"spatial_geometry": direct},
            spatial_shape=spatial_shape,
        )
    return _spatial_geometry_from_metadata(
        exp_data.get("extra_metadata") if isinstance(exp_data.get("extra_metadata"), Mapping) else exp_data,
        spatial_shape=spatial_shape,
    )


def _expand_kspace_for_inserted_image_axes(
    kspace: Any,
    *,
    inserted_axes: Sequence[int],
    original_data_ndim: int,
    original_non_spatial_count: int,
) -> Any:
    if kspace is None or not inserted_axes:
        return kspace
    arr = np.asarray(kspace)
    if arr.ndim == original_data_ndim:
        for axis in inserted_axes:
            arr = np.expand_dims(arr, axis=int(axis))
        return jnp.asarray(arr)
    if arr.ndim == original_data_ndim + 1:
        for axis in inserted_axes:
            kspace_axis = int(axis) + 1 if int(axis) >= int(original_non_spatial_count) else int(axis)
            arr = np.expand_dims(arr, axis=kspace_axis)
        return jnp.asarray(arr)
    return kspace


def _copy_experiment_with_dims(experiment: Experiment, dims: Mapping[str, int]) -> Experiment:
    updated = Experiment(
        dims={str(name): int(size) for name, size in dims.items()},
        observables=dict(experiment.observables),
        exp_type=experiment.exp_type,
    )
    for attr in (
        "study_globals",
        "_mrax_dim_observables",
        "_mrax_default_identity_fit_warning",
        "_mrax_manual_type_override_warning",
    ):
        if hasattr(experiment, attr):
            setattr(updated, attr, getattr(experiment, attr))
    return updated


def _copy_roi(roi: ROI) -> ROI:
    return ROI(
        roi.name,
        np.asarray(roi.mask).copy(),
        dims=None if roi.dims is None else tuple(roi.dims),
        metadata=dict(roi.metadata),
    )


_MAP_REF_SEP = "::"


def _encode_map_ref(experiment_name: str, map_name: str) -> str:
    return f"{experiment_name}{_MAP_REF_SEP}{map_name}"


def _decode_map_ref(ref: str) -> tuple[str, str] | None:
    if _MAP_REF_SEP not in str(ref):
        return None
    exp_name, map_name = str(ref).split(_MAP_REF_SEP, 1)
    if not exp_name or not map_name:
        return None
    return exp_name, map_name


def _ab_mt_map_score(kind: str, map_name: str, unit: str | None = None) -> int | None:
    name = map_name.lower()
    unit_key = (unit or "").lower()
    if kind == "t1":
        if name in {"t1_map", "t1"}:
            return 0
        if "t1" in name and unit_key == "ms":
            return 1
    if kind == "b1":
        if name in {"b1_map", "b1map"}:
            return 0
        if name == "b1" and unit_key in {"%", "percent"}:
            return 1
        if name == "b1" and unit_key in {"ut", "µt", "μt"}:
            return 1
        if "b1" in name and unit_key in {"%", "percent"}:
            return 2
    if kind == "b0":
        if name in {"b0_map", "b0map"}:
            return 0
        if name in {"delta", "b0"} and unit_key == "ppm":
            return 1
        if "b0" in name and unit_key == "ppm":
            return 2
    return None


def _default_ab_mt_map_unit(kind: str) -> str:
    return {"t1": "ms", "b1": "%", "b0": "ppm"}[kind]


def _study_to_session_data(
    study: Study,
    metadata: dict[str, Any] | None,
    *,
    source_root: Path,
    source_kind: str = "bruker",
) -> dict[str, Any]:
    metadata = metadata or {}
    experiments: dict[str, Any] = {}
    summary_entries: list[dict[str, Any]] = []
    extra_by_name = metadata.get("experiments") if isinstance(metadata.get("experiments"), dict) else {}
    for name, experiment, model, dataset in study.iter_entries():
        fitted_units = getattr(model, "parameter_units", {}) or {}
        extra = dict(extra_by_name.get(name, {}) or {})
        non_imaging = _study_entry_is_non_imaging(experiment, dataset, extra)
        data_role = _study_entry_data_role(experiment, dataset, extra)
        image = dataset.data
        experiments[name] = {
            "mr_image": image,
            "original_mr_image": np.asarray(image).copy(),
            "raw_kspace": dataset.kspace,
            "acquisition": dataset.acquisition,
            "trajectory": dataset.trajectory, "density": dataset.density,
            "sampling_mask": dataset.sampling_mask,
            "fitted_maps": {},
            "model": model.name,
            "exp_type": experiment.exp_type,
            "non_imaging": non_imaging,
            "data_role": data_role,
            "observable_units": {key: value.si_unit for key, value in experiment.observables.items()},
            "observables": dict(experiment.observables),
            "fitted_map_units": fitted_units,
            "processed": dict(dataset.processed or {}),
            "kspace_metadata": dict(dataset.kspace_metadata or {}),
            "extra_metadata": extra,
            "spatial_geometry": (
                dataset.spatial_geometry.to_metadata()
                if dataset.spatial_geometry is not None
                else _spatial_geometry_from_metadata(
                    extra,
                    spatial_shape=experiment.spatial_shape,
                ).to_metadata()
            ),
            "plots": {},
        }
        summary_entries.append(
            {
                "name": name,
                "status": "loaded",
                "model": model.name,
                "exp_type": experiment.exp_type,
                "dataset": dataset.name,
                "dims": dict(experiment.dims),
                "observables": sorted(experiment.observables.keys()),
                "non_imaging": non_imaging,
                "data_role": data_role,
            }
        )
    skipped = metadata.get("skipped_experiments") if isinstance(metadata.get("skipped_experiments"), dict) else {}
    for name, issue in sorted(skipped.items(), key=lambda item: item[0]):
        issue = dict(issue or {})
        summary_entries.append(
            {
                "name": str(name),
                "status": issue.get("status", "skipped"),
                "exp_type": "",
                "model": "",
                "dataset": "",
                "dims": "",
                "observables": "",
                "issue": issue.get("message") or issue.get("reason") or "",
            }
        )

    return {
        "manifest": {
            "study_name": metadata.get("subject") or source_root.name,
            "summary": {"experiments": summary_entries},
            "targets": [],
            "source": {"kind": source_kind, "path": str(source_root)},
        },
        "experiments": experiments,
        "study_globals": dict(study.study_globals or {}),
    }


def _normalize_image_plane(plane: str | None) -> str:
    normalized = str(plane or "axial").strip().lower()
    aliases = {
        "transverse": "axial",
        "xy": "axial",
        "xz": "coronal",
        "yz": "sagittal",
        "saggital": "sagittal",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in IMAGE_PLANE_DIMS:
        raise ValueError("Image plane must be axial, coronal, or sagittal")
    return normalized


def _display_slice(
    array: np.ndarray,
    *,
    dims: Mapping[str, int] | None = None,
    indices: Mapping[str, int] | None = None,
    kind: str = "image",
    plane_dims: Sequence[str] = ("x", "y"),
) -> np.ndarray:
    arr = np.asarray(array)
    if arr.ndim == 0:
        return arr.reshape((1, 1))
    if arr.ndim == 1:
        return arr.reshape((1, arr.shape[0]))
    if arr.ndim == 2:
        return arr

    controls = _axis_controls_for_kind(arr, dims or {}, kind, plane_dims=plane_dims)
    index_map = dict(indices or {})
    slicer: list[object] = [slice(None)] * arr.ndim
    for control in controls:
        axis = int(control["axis"])
        key = str(control["key"])
        if 0 <= axis < arr.ndim:
            value = int(index_map.get(key, 0))
            slicer[axis] = max(0, min(value, arr.shape[axis] - 1))
    selected = np.asarray(arr[tuple(slicer)])
    if selected.ndim == 2:
        dim_names = list((dims or {}).keys()) if len(dims or {}) == arr.ndim else []
        if dim_names:
            sliced_axes = {int(control["axis"]) for control in controls}
            selected_dims = [name for axis, name in enumerate(dim_names) if axis not in sliced_axes]
            requested = [str(name) for name in plane_dims]
            if len(selected_dims) == 2 and set(selected_dims) == set(requested):
                order = [selected_dims.index(name) for name in requested]
                selected = np.transpose(selected, order)
        return selected
    selected = np.squeeze(selected)
    if selected.ndim == 0:
        return selected.reshape((1, 1))
    if selected.ndim == 1:
        return selected.reshape((1, selected.shape[0]))
    if selected.ndim == 2:
        return selected
    # If some unknown dimension remains, keep the first plane instead of
    # flattening unrelated axes into a confusing preview.
    while selected.ndim > 2:
        selected = np.take(selected, 0, axis=0)
    return selected


def _axis_controls_for_kind(
    arr: np.ndarray,
    dims: Mapping[str, int],
    kind: str,
    *,
    plane_dims: Sequence[str] = ("x", "y"),
) -> list[dict[str, int | str]]:
    if kind == "kspace":
        return _axis_controls_for_kspace(arr, dims)
    if kind == "map":
        return _axis_controls_for_map(arr, dims, plane_dims=plane_dims)
    return _axis_controls_for_image(arr, dims, plane_dims=plane_dims)


def _axis_controls_for_image(
    arr: np.ndarray,
    dims: Mapping[str, int],
    *,
    plane_dims: Sequence[str] = ("x", "y"),
) -> list[dict[str, int | str]]:
    arr = np.asarray(arr)
    if arr.ndim <= 2:
        return []
    dim_names = list(dims.keys()) if len(dims) == arr.ndim else []
    controls: list[dict[str, int | str]] = []
    if dim_names:
        displayed = {str(name) for name in plane_dims}
        for axis, (name, size) in enumerate(zip(dim_names, arr.shape)):
            if name in displayed:
                continue
            key = "frame" if name in ("frame", "t") and len(controls) == 0 else name
            if name == "z":
                key = "z"
            controls.append({"key": key, "label": name, "axis": axis, "size": int(size)})
        return controls
    for axis, size in enumerate(arr.shape[:-2]):
        key = "frame" if axis == 0 else f"axis_{axis}"
        controls.append({"key": key, "label": key, "axis": axis, "size": int(size)})
    return controls


def _axis_controls_for_non_image(arr: np.ndarray, dims: Mapping[str, int]) -> list[dict[str, int | str]]:
    arr = np.asarray(arr)
    if arr.ndim <= 1:
        return []
    dim_names = list(dims.keys()) if len(dims) == arr.ndim else []
    controls: list[dict[str, int | str]] = []
    axes = range(0, arr.ndim - 1)
    for axis in axes:
        name = dim_names[axis] if dim_names else ("frame" if axis == 0 else f"axis_{axis}")
        key = "frame" if name in {"frame", "t"} and axis == 0 else str(name)
        controls.append({"key": key, "label": str(name), "axis": int(axis), "size": int(arr.shape[axis])})
    return controls


def _axis_controls_for_map(
    arr: np.ndarray,
    dims: Mapping[str, int],
    *,
    plane_dims: Sequence[str] = ("x", "y"),
) -> list[dict[str, int | str]]:
    arr = np.asarray(arr)
    if arr.ndim <= 2:
        return []
    controls: list[dict[str, int | str]] = []
    displayed = {str(name) for name in plane_dims}
    if dims:
        spatial = spatial_names(dims)
        spatial_shape = tuple(int(dims[name]) for name in spatial)
        if spatial and tuple(int(v) for v in arr.shape) == spatial_shape:
            for axis, name in enumerate(spatial):
                if name in displayed:
                    continue
                controls.append({"key": name, "label": name, "axis": axis, "size": int(arr.shape[axis])})
            return controls

        dim_names = list(dims.keys()) if len(dims) == arr.ndim else []
        if dim_names:
            for axis, (name, size) in enumerate(zip(dim_names, arr.shape)):
                if name in displayed:
                    continue
                key = "frame" if name in ("frame", "t") and axis == 0 else name
                controls.append({"key": key, "label": name, "axis": axis, "size": int(size)})
            return controls

    for axis, size in enumerate(arr.shape[:-2]):
        key = "z" if axis == arr.ndim - 3 else f"map_axis_{axis}"
        controls.append({"key": key, "label": key, "axis": axis, "size": int(size)})
    return controls


def _axis_controls_for_kspace(arr: np.ndarray, dims: Mapping[str, int]) -> list[dict[str, int | str]]:
    arr = np.asarray(arr)
    if arr.ndim <= 2:
        return []
    non_spatial = non_spatial_names(dims)
    labels: list[str] = []
    for axis, size in enumerate(arr.shape[:-2]):
        if axis < len(non_spatial) and int(dims.get(non_spatial[axis], -1)) == int(size):
            labels.append(non_spatial[axis])
        elif "z" in dims and int(dims.get("z", -1)) == int(size) and "z" not in labels:
            labels.append("z")
        elif size <= 64 and "coil" not in labels:
            labels.append("coil")
        elif "z" in dims and "partition" not in labels:
            labels.append("partition")
        else:
            labels.append(f"k_axis_{axis}")
    controls = []
    for axis, (label, size) in enumerate(zip(labels, arr.shape[:-2])):
        key = "frame" if label in ("frame", "t") and axis == 0 else label
        controls.append({"key": key, "label": label, "axis": axis, "size": int(size)})
    return controls


def transform_kspace(array: np.ndarray, mode: str = "log_magnitude", *, center: str = "auto") -> np.ndarray:
    """Convert k-space to a displayable real-valued 2D image."""

    arr = np.asarray(array)
    mode = mode.lower().replace("-", "_")
    if mode == "phase":
        display = np.angle(arr)
    elif mode == "real":
        display = np.real(arr)
    elif mode == "imag":
        display = np.imag(arr)
    else:
        magnitude = np.abs(arr)
        if mode in ("log", "log_magnitude", "logmag"):
            display = np.log1p(magnitude)
        else:
            display = magnitude
    display = np.asarray(display)
    if display.ndim >= 2 and kspace_centering_decision(arr, center) == "fftshift":
        return np.fft.fftshift(display, axes=(-2, -1))
    return display


def kspace_centering_decision(array: np.ndarray, center: str = "auto") -> str:
    """Return the display-centering operation for a k-space plane."""

    mode = str(center or "auto").strip().lower().replace("-", "_")
    if mode in {"none", "raw", "off", "no_shift", "unshifted"}:
        return "none"
    if mode in {"fftshift", "shift", "center", "centered", "on"}:
        return "fftshift"
    arr = np.asarray(array)
    if arr.ndim < 2:
        return "none"
    arr = np.abs(arr)
    if not np.isfinite(arr).any():
        return "none"
    raw_ratio = _kspace_center_energy_ratio(arr)
    shifted_ratio = _kspace_center_energy_ratio(np.fft.fftshift(arr, axes=(-2, -1)))
    return "fftshift" if shifted_ratio > raw_ratio * 1.05 else "none"


def _kspace_center_energy_ratio(array: np.ndarray) -> float:
    arr = np.asarray(array, dtype=float)
    h, w = arr.shape[-2], arr.shape[-1]
    ph = max(1, h // 8)
    pw = max(1, w // 8)
    cy = h // 2
    cx = w // 2
    y0 = max(0, cy - ph)
    y1 = min(h, cy + ph + 1)
    x0 = max(0, cx - pw)
    x1 = min(w, cx + pw + 1)
    center_energy = float(np.nansum(arr[..., y0:y1, x0:x1]))
    corner_energy = float(
        np.nansum(arr[..., :ph, :pw])
        + np.nansum(arr[..., :ph, -pw:])
        + np.nansum(arr[..., -ph:, :pw])
        + np.nansum(arr[..., -ph:, -pw:])
    )
    return center_energy / max(corner_energy, 1e-12)


def frame_axis_values(experiment: Experiment) -> tuple[str, np.ndarray]:
    """Return a sensible x-axis for frame plots."""

    frame_count = frame_count_from_dims(experiment.dims)
    for name, qty in experiment.observables.items():
        arr = _as_numpy(qty)
        if arr.ndim == 1 and int(arr.shape[0]) == frame_count:
            return name, arr
    for name in non_spatial_names(experiment.dims):
        size = int(experiment.dims[name])
        if size == frame_count:
            return name, np.arange(frame_count)
    return "frame", np.arange(frame_count)


__all__ = [
    "Diagnostic",
    "ExperimentViewState",
    "FitSettings",
    "FrameMask",
    "GuiSession",
    "kspace_centering_decision",
    "frame_axis_values",
    "transform_kspace",
]
