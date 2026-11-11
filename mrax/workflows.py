"""Versioned, data-free workflow documents shared by Python and the browser GUI.

The filename suffix describes the workflow kind; all three use UTF-8 JSON.
See docs/workflows.rst for the compatibility and execution contract.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from mrax._version import MRAX_VERSION

WorkflowKind = Literal["processing", "reconstruction", "fitting"]
EXTENSIONS = {"processing": ".mpip", "reconstruction": ".mrec", "fitting": ".mfit"}
MAX_DOCUMENT_BYTES = 1024 * 1024
FORMAT_VERSION = 1
COMPOSITE_FORMAT_VERSION = 2


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class WorkflowBinding(_StrictModel):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    kind: Literal[
        "roi",
        "experiment",
        "map",
        "sat_trans_reference_scan",
        "ab_mt_t1_map_ref",
        "ab_mt_b1_map_ref",
        "ab_mt_b0_map_ref",
    ]
    label: str = Field(min_length=1, max_length=160)
    experiment: str | None = None
    default: Literal["target"] | None = None


class WorkflowStep(_StrictModel):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    operation: Literal["process", "register", "drift", "reconstruct", "fit"]
    version: int = Field(ge=1, json_schema_extra={"const": 1})
    enabled: bool = True
    settings: dict[str, Any]

    @field_validator("version")
    @classmethod
    def supported_operation(cls, value):
        if value != 1:
            raise ValueError(
                "Unsupported operation version; update MRax before opening this workflow"
            )
        return value


class WorkflowDocument(_StrictModel):
    format: Literal["mrax.workflow"]
    format_version: int = Field(
        json_schema_extra={"enum": [FORMAT_VERSION, COMPOSITE_FORMAT_VERSION]}
    )
    kind: WorkflowKind
    mrax_version: str = Field(pattern=r"^\d+\.\d+\.\d+[a-zA-Z0-9.+_-]*$", max_length=80)
    name: str = Field(min_length=1, max_length=160)
    description: str = Field(default="", max_length=8000)
    bindings: list[WorkflowBinding] = Field(default_factory=list, max_length=128)
    steps: list[WorkflowStep] = Field(min_length=1, max_length=64)
    extensions: dict[str, Any] = Field(default_factory=dict)

    @field_validator("format_version")
    @classmethod
    def supported_format(cls, value):
        if value not in {FORMAT_VERSION, COMPOSITE_FORMAT_VERSION}:
            raise ValueError(
                "Unsupported workflow format version; no automatic conversion is attempted"
            )
        return value

    @field_validator("name")
    @classmethod
    def meaningful_name(cls, value):
        if not value.strip():
            raise ValueError("Give the workflow a name")
        return value.strip()

    @model_validator(mode="after")
    def consistent_document(self):
        allowed = {
            "processing": {"process", "register", "drift", "reconstruct", "fit"}
            if self.format_version == COMPOSITE_FORMAT_VERSION
            else {"process", "register", "drift"},
            "reconstruction": {"reconstruct"},
            "fitting": {"fit"},
        }[self.kind]
        composite = self.kind == "processing" and any(
            s.operation in {"reconstruct", "fit"} for s in self.steps
        )
        if self.format_version == COMPOSITE_FORMAT_VERSION and not composite:
            raise ValueError(
                "Workflow format version 2 is reserved for processing pipelines with workflow blocks"
            )
        reconstruction = [
            index for index, step in enumerate(self.steps) if step.operation == "reconstruct"
        ]
        if reconstruction and self.kind == "processing" and reconstruction != [0]:
            raise ValueError("A reconstruction block can occur only once, first")
        if any(step.operation not in allowed for step in self.steps):
            raise ValueError("The workflow contains an operation from another workflow kind")
        if self.kind != "processing" and len(self.steps) != 1:
            raise ValueError("A reconstruction or fitting workflow has exactly one configuration")
        if not any(step.enabled for step in self.steps):
            raise ValueError("Enable at least one workflow step")
        for values in (self.steps, self.bindings):
            if len({value.id for value in values}) != len(values):
                raise ValueError("Workflow step and binding IDs must be unique within their lists")
        if any("." not in key for key in self.extensions):
            raise ValueError("Extension names must be namespaced, for example org.example.notes")
        used = set()

        def inspect(value, depth=0):
            if depth > 24:
                raise ValueError("Workflow settings are nested too deeply")
            if isinstance(value, dict):
                if "$binding" in value:
                    if set(value) != {"$binding"} or not isinstance(value["$binding"], str):
                        raise ValueError("A binding reference contains only the $binding string")
                    used.add(value["$binding"])
                for key, item in value.items():
                    if not isinstance(key, str) or len(key) > 256:
                        raise ValueError("Invalid workflow setting key")
                    inspect(item, depth + 1)
            elif isinstance(value, list):
                if len(value) > 4096:
                    raise ValueError("Workflow setting arrays may contain at most 4096 entries")
                for item in value:
                    inspect(item, depth + 1)
            elif isinstance(value, float):
                if not math.isfinite(value):
                    raise ValueError("Workflow numbers must be finite")
            elif isinstance(value, str):
                if len(value) > 8000:
                    raise ValueError("Workflow setting strings are too long")
            elif value is not None and type(value) not in {int, bool}:
                raise ValueError("Workflow settings must contain only JSON values")

        for step in self.steps:
            # Embedded documents own a separate binding namespace and validate it below.
            inspect({key: value for key, value in step.settings.items() if key != "workflow"})
        declared = {binding.id for binding in self.bindings}
        for binding in self.bindings:
            if binding.kind == "experiment" and binding.experiment is not None:
                raise ValueError("Experiment bindings cannot depend on other experiment bindings")
            if binding.experiment is not None:
                used.add(binding.experiment)
            if binding.experiment is not None and not any(
                value.id == binding.experiment and value.kind == "experiment"
                for value in self.bindings
            ):
                raise ValueError("A resource's experiment must reference an experiment binding")
            if binding.default is not None and binding.kind != "experiment":
                raise ValueError("Only experiment bindings support the target default")
        if used != declared:
            raise ValueError("Every binding must be declared and used by a step")
        # Extensions are metadata only, never executable parameters or bindings.
        inspect(self.extensions)
        validate_operations(self)
        return self


def validate_operations(document):
    """Reject unknown executable fields instead of silently changing their meaning."""
    from mrax.gui.session import FitSettings

    portable_fit = set(FitSettings.__dataclass_fields__) - {
        "output_dir",
        "selected_roi",
        "use_view_state_data",
    }
    allowed = {
        "process": {
            "method",
            "input",
            "component",
            "axes",
            "axis",
            "parameters",
            "roi_names",
            "noise_roi_names",
            "roi_name",
            "reference_experiment",
            "reference_frame",
            "scope",
            "indices",
            "plane_dims",
        },
        "reconstruct": {"component", "pipeline"},
        "register": {
            "reference_experiment",
            "moving_source",
            "reference_source",
            "moving_frame",
            "reference_frame",
            "moving_map",
            "reference_map",
            "settings",
            "plane",
            "reference_roi",
            "moving_roi",
            "scope",
            "manual",
            "manual_settings",
            "indices",
        },
        "drift": {"settings", "roi_name"},
        "fit": {"settings", "excluded_frames", "use_roi", "roi_names"},
    }
    for step in document.steps:
        settings = step.settings
        if "workflow" in settings:
            if (
                document.kind != "processing"
                or document.format_version != COMPOSITE_FORMAT_VERSION
                or step.operation not in {"reconstruct", "fit"}
            ):
                raise ValueError(
                    "Only processing pipelines can contain reconstruction or fitting workflow blocks"
                )
            if set(settings) - {"workflow", "bindings"}:
                raise ValueError("A workflow block contains its document and input bindings only")
            expected = "reconstruction" if step.operation == "reconstruct" else "fitting"
            if (
                not isinstance(settings["workflow"], dict)
                or settings["workflow"].get("kind") != expected
            ):
                raise ValueError("The embedded workflow kind does not match its block")
            child = WorkflowDocument.model_validate(settings["workflow"])
            bindings = settings.get("bindings", {})
            if not isinstance(bindings, dict) or set(bindings) != {
                binding.id for binding in child.bindings
            }:
                raise ValueError("Assign every embedded workflow input")
            for binding in child.bindings:
                _require_binding(document, bindings[binding.id], binding.kind, binding.label)
            continue
        if document.kind == "processing" and step.operation in {"reconstruct", "fit"}:
            raise ValueError(
                "Reconstruction and fitting blocks must embed their .mrec or .mfit workflow"
            )
        if set(settings) - allowed[step.operation]:
            raise ValueError("Unknown " + step.operation + " workflow setting")
        for key in ("settings", "parameters", "manual_settings"):
            if key in settings and not isinstance(settings[key], dict):
                raise ValueError(key + " must be an object")
        if step.operation == "process":
            if not isinstance(settings.get("method"), str):
                raise ValueError("A processing step needs a method")
            if settings.get("input", "image") == "view":
                raise ValueError(
                    "Use an explicit image source; display adjustments are not portable"
                )
            source = settings.get("input", "image")
            if isinstance(source, dict):
                _require_binding(document, source, "map", "Processing map")
            elif not isinstance(source, str) or source not in {
                "image",
                "original",
                "reconstruction",
            }:
                raise ValueError("Use image, original, reconstruction or a map binding as input")
            if "axes" in settings and (
                not isinstance(settings["axes"], list)
                or not all(isinstance(value, str) for value in settings["axes"])
            ):
                raise ValueError("Filter axes must be a list of dimension names")
            if settings.get("axis") is not None and not isinstance(settings["axis"], str):
                raise ValueError("The acquisition axis must be a dimension name")
            if "scope" in settings and (
                not isinstance(settings["scope"], str)
                or settings["scope"] not in {"image", "package", "dataset"}
            ):
                raise ValueError("Choose image, package or dataset denoising scope")
            if {"scope", "indices", "plane_dims"} & settings.keys():
                from mrax.gui.services.processing import FILTERS

                if settings["method"] not in FILTERS:
                    raise ValueError("Denoising scopes are not time-series settings")
                if "scope" not in settings:
                    raise ValueError("Denoising target coordinates require a scope")
            if "indices" in settings and (
                not isinstance(settings["indices"], dict)
                or any(
                    not isinstance(name, str) or type(index) is not int or index < 0
                    for name, index in settings["indices"].items()
                )
            ):
                raise ValueError("Denoising indices must name non-negative integer coordinates")
            if "plane_dims" in settings and (
                not isinstance(settings["plane_dims"], list)
                or len(settings["plane_dims"]) != 2
                or any(
                    not isinstance(name, str) or name not in {"x", "y", "z"}
                    for name in settings["plane_dims"]
                )
                or len(set(settings["plane_dims"])) != 2
            ):
                raise ValueError("Viewed-image denoising requires two spatial dimensions")
        if step.operation == "reconstruct" and (
            not isinstance(settings.get("component"), str)
            or settings.get("component")
            not in {
                "magnitude",
                "phase",
            }
        ):
            raise ValueError("Reconstruction requires magnitude or phase")
        if step.operation == "reconstruct" and settings.get("pipeline") is not None:
            from mrax.reco.plans import normalize_plan

            normalize_plan(settings["pipeline"])
        if step.operation == "fit":
            fit = settings.get("settings")
            if not isinstance(fit, dict) or set(fit) - portable_fit or not fit.get("model_name"):
                raise ValueError("A fitting workflow needs a model and portable fit settings")
            if not isinstance(fit["model_name"], str):
                raise ValueError("The fitting model must be a registered model name")
            for key in (
                "param_overrides",
                "pool_overrides",
                "regularizer_configs",
                "parameter_constant_dims",
            ):
                if key in fit and not isinstance(fit[key], dict):
                    raise ValueError(key + " must be an object")
            if "selected_pools" in fit and (
                not isinstance(fit["selected_pools"], list)
                or not all(isinstance(value, str) for value in fit["selected_pools"])
            ):
                raise ValueError("Selected pools must be a list of pool names")
            for key in ("auto_mask_nan_observables", "compute_derived_maps"):
                if fit.get(key) is not None and type(fit[key]) is not bool:
                    raise ValueError(key + " must be a boolean or null")
            if type(settings.get("use_roi", True)) is not bool:
                raise ValueError("use_roi must be a boolean")
            if "num_steps" in fit and type(fit["num_steps"]) is not int:
                raise ValueError("Fit step count must be an integer")
            if "step_size" in fit and type(fit["step_size"]) not in {int, float}:
                raise ValueError("Fit step size must be a number")
            frames = settings.get("excluded_frames", [])
            if not isinstance(frames, list) or any(type(n) is not int or n < 0 for n in frames):
                raise ValueError("Excluded frames are zero-based non-negative integers")
        for key in ("roi_names", "noise_roi_names"):
            if (
                key in settings
                and settings[key] is not None
                and not isinstance(settings[key], list)
            ):
                raise ValueError("ROI selections must be lists or null for all enabled ROIs")
        # Resource names are supplied by the target session, never frozen in a recipe.
        for key in (
            "roi_name",
            "reference_experiment",
            "moving_map",
            "reference_map",
            "reference_roi",
            "moving_roi",
        ):
            if settings.get(key) is not None:
                _require_binding(
                    document,
                    settings[key],
                    "experiment"
                    if key == "reference_experiment"
                    else "map"
                    if key.endswith("_map")
                    else "roi",
                    key,
                )
        for key in ("roi_names", "noise_roi_names"):
            for value in settings.get(key) or []:
                _require_binding(document, value, "roi", key)
        for key in (
            "sat_trans_reference_scan",
            "ab_mt_t1_map_ref",
            "ab_mt_b1_map_ref",
            "ab_mt_b0_map_ref",
        ):
            value = settings.get("settings", {}).get(key)
            if value is not None:
                _require_binding(document, value, key, key)


def _require_binding(document, value, kind, label):
    if not _is_binding(value) or not any(
        binding.id == value["$binding"] and binding.kind == kind for binding in document.bindings
    ):
        raise ValueError(label + " must use a " + kind + " binding")


def _is_binding(value):
    return isinstance(value, dict) and set(value) == {"$binding"}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key: " + key)
        result[key] = value
    return result


def loads_workflow(text: str | bytes, *, suffix: str | None = None) -> WorkflowDocument:
    """Read a bounded, strict JSON document without evaluating code or loading arrays."""
    if len(text.encode("utf-8") if isinstance(text, str) else text) > MAX_DOCUMENT_BYTES:
        raise ValueError("Workflow files must be smaller than 1 MiB")
    try:
        if isinstance(text, bytes):
            text = text.decode("utf-8-sig")
        values = json.loads(text.lstrip("\ufeff"), object_pairs_hook=_unique_object)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("Choose a valid UTF-8 JSON workflow file") from exc
    document = WorkflowDocument.model_validate(values)
    if suffix is not None and suffix.lower() != EXTENSIONS[document.kind]:
        raise ValueError("The file extension does not match its workflow kind")
    return document


def dumps_workflow(document: WorkflowDocument) -> str:
    document = WorkflowDocument.model_validate(document.model_dump())
    text = json.dumps(document.model_dump(), indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if len(text.encode("utf-8")) > MAX_DOCUMENT_BYTES:
        raise ValueError("Workflow files must be smaller than 1 MiB")
    return text


def load_workflow(path: str | Path) -> WorkflowDocument:
    path = Path(path)
    if path.stat().st_size > MAX_DOCUMENT_BYTES:
        raise ValueError("Workflow files must be smaller than 1 MiB")
    return loads_workflow(path.read_bytes(), suffix=path.suffix)


def save_workflow(document: WorkflowDocument, path: str | Path, *, overwrite=False):
    from mrax.gui.services.file_output import write_file_atomic

    path = Path(path)
    if path.suffix.lower() != EXTENSIONS[document.kind]:
        raise ValueError("Save this workflow with a " + EXTENSIONS[document.kind] + " extension")
    text = dumps_workflow(document)
    if path.exists() and overwrite:
        # A native extension alone must never turn an unrelated input into an output.
        load_workflow(path)
    write_file_atomic(
        path,
        lambda destination: destination.write_text(text, encoding="utf-8"),
        overwrite=overwrite,
    )


def new_workflow(kind, steps, *, name=None, description="", bindings=()):
    return WorkflowDocument.model_validate(
        {
            "format": "mrax.workflow",
            "format_version": COMPOSITE_FORMAT_VERSION
            if kind == "processing"
            and any(step["operation"] in {"reconstruct", "fit"} for step in steps)
            else FORMAT_VERSION,
            "kind": kind,
            "mrax_version": MRAX_VERSION,
            "name": name or kind.capitalize() + " workflow",
            "description": description,
            "bindings": list(bindings),
            "steps": steps,
        }
    )
