"""Portable acquisition persistence using normalized arrays and metadata."""
from __future__ import annotations

import json
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np

from mrax.data_struct.acquisition import AcquisitionData, ReadoutModel, ReconstructionRecipe
from mrax.data_struct.geometry import SpatialGeometry


def save_acquisition(path: str | Path, acquisition: AcquisitionData) -> None:
    """Save the acquisition contract losslessly as arrays and JSON."""
    arrays = {"samples": acquisition.samples}
    for name in ("acquired", "trajectory", "density", "statistical_weights", "readout_polarity", "readout_coordinates"):
        value = getattr(acquisition, name)
        if value is not None:
            arrays[name] = value
    auxiliary = list(acquisition.auxiliary_samples)
    for index, name in enumerate(auxiliary):
        arrays[f"auxiliary_{index}"] = np.asarray(acquisition.auxiliary_samples[name])
    auxiliary_coordinates = list(acquisition.auxiliary_coordinates)
    for index, name in enumerate(auxiliary_coordinates):
        arrays[f"auxiliary_coordinate_{index}"] = np.asarray(acquisition.auxiliary_coordinates[name])
    coordinates = list(acquisition.frame_coordinates)
    for index, name in enumerate(coordinates):
        array = np.asarray(acquisition.frame_coordinates[name])
        if array.dtype.hasobject:
            raise ValueError("Portable frame coordinates cannot contain Python objects")
        arrays[f"coordinate_{index}"] = array
    metadata = {
        "schema_version": acquisition.schema_version,
        "kind": acquisition.kind, "domain": acquisition.domain,
        "encoding_shape": acquisition.encoding_shape, "encoding_axes": acquisition.encoding_axes,
        "geometry": None if acquisition.geometry is None else acquisition.geometry.to_metadata(),
        "recipe": asdict(acquisition.recipe), "frame_coordinates": coordinates,
        "auxiliary_samples": auxiliary,
        "auxiliary_coordinates": auxiliary_coordinates,
        "auxiliary_frame_axes": dict(acquisition.auxiliary_frame_axes),
    }
    if acquisition.readout_model is not None:
        readout = acquisition.readout_model
        arrays["profile_coordinates"] = readout.coordinates
        arrays["profile_target_trajectory"] = readout.target_trajectory
        metadata["readout_model"] = {"image_points": readout.image_points,
            "target_points": readout.target_points, "relative_cutoff": readout.relative_cutoff}
    arrays["metadata"] = np.asarray(json.dumps(metadata))
    # Opening the stream explicitly prevents NumPy from changing the filename.
    with Path(path).open("wb") as stream:
        np.savez_compressed(stream, **arrays)


def load_acquisition(path: str | Path) -> AcquisitionData:
    """Load and validate an acquisition; archives containing pickle are rejected."""
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        if metadata.get("schema_version") != 1:
            raise ValueError("Unsupported acquisition archive schema")
        options = {name: np.array(archive[name]) for name in
            ("acquired", "trajectory", "density", "statistical_weights", "readout_polarity", "readout_coordinates") if name in archive}
        options["auxiliary_samples"] = {name: np.array(archive[f"auxiliary_{index}"])
            for index, name in enumerate(metadata.get("auxiliary_samples", []))}
        options["auxiliary_coordinates"] = {name: np.array(archive[f"auxiliary_coordinate_{index}"])
            for index, name in enumerate(metadata.get("auxiliary_coordinates", []))}
        options["auxiliary_frame_axes"] = metadata.get("auxiliary_frame_axes", {})
        if "readout_model" in metadata:
            options["readout_model"] = ReadoutModel(np.array(archive["profile_coordinates"]),
                target_trajectory=np.array(archive["profile_target_trajectory"]), **metadata["readout_model"])
        recipe = dict(metadata["recipe"])
        for field in fields(ReconstructionRecipe):
            if isinstance(field.default, tuple) and field.name in recipe:
                recipe[field.name] = tuple(tuple(value) if isinstance(value, list) else value
                                           for value in recipe[field.name])
        names = metadata["frame_coordinates"]
        coordinates = {name: np.array(archive[f"coordinate_{index}"]) for index, name in enumerate(names)}
        return AcquisitionData(np.array(archive["samples"]), tuple(metadata["encoding_shape"]),
            tuple(metadata["encoding_axes"]), kind=metadata["kind"], domain=metadata["domain"],
            geometry=None if metadata["geometry"] is None else SpatialGeometry.from_metadata(metadata["geometry"]),
            frame_coordinates=coordinates, recipe=ReconstructionRecipe(**recipe), **options)
