"""Modular experiment/model/data abstractions."""

from .dataset import DataSet
from .dimensions import (
    FRAME_DIM,
    SPATIAL_DIMS,
    TEMPORAL_DIM,
    DimensionSpec,
    build_experiment_dims,
    expand_observables_to_frames,
    pick_observable_for_frames,
    reshape_dims_to_frames,
    reshape_frames_to_dims,
    reshape_observables_to_dims,
)
from .experiment import Experiment
from .geometry import (
    NATIVE_AXIS_CONVENTION,
    NIFTI_AFFINE_CONVENTION,
    SPATIAL_GEOMETRY_VERSION,
    ResolvedSpatialGeometry,
    SpatialGeometry,
    geometries_are_compatible,
    resolve_spatial_geometry,
)
from .joint import JointExperiment, JointSource, join_experiments, select_experiment_frames
from .model import Model
from .quantity import Quantity
from .storage import load_precomputed_fields, load_study_results, save_study_results
from .study import Study

__all__ = [
    "DataSet",
    "DimensionSpec",
    "Experiment",
    "FRAME_DIM",
    "JointExperiment",
    "JointSource",
    "Model",
    "NATIVE_AXIS_CONVENTION",
    "NIFTI_AFFINE_CONVENTION",
    "Quantity",
    "ResolvedSpatialGeometry",
    "SPATIAL_GEOMETRY_VERSION",
    "SPATIAL_DIMS",
    "Study",
    "SpatialGeometry",
    "TEMPORAL_DIM",
    "build_experiment_dims",
    "expand_observables_to_frames",
    "pick_observable_for_frames",
    "join_experiments",
    "geometries_are_compatible",
    "load_precomputed_fields",
    "load_study_results",
    "save_study_results",
    "select_experiment_frames",
    "reshape_frames_to_dims",
    "reshape_dims_to_frames",
    "reshape_observables_to_dims",
    "resolve_spatial_geometry",
]
