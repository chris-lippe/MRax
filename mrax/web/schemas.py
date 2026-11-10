"""Validated browser commands. Scientific defaults stay in the existing model library."""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, field_validator, model_validator


class Command(BaseModel):
    model_config = ConfigDict(extra="forbid")


class OverlayWindow(Command):
    center: FiniteFloat
    width: Annotated[FiniteFloat, Field(gt=0)]


class OverlaySource(Command):
    experiment: str
    kind: Literal["image", "map", "reconstruction", "registration"]
    map_name: str | None = None
    registration_key: str | None = None


class OverlayLayer(OverlaySource):
    id: str = Field(min_length=1, max_length=80, pattern=r"^[\w-]+$")
    frame: int | None = Field(default=None, ge=0)
    component: Literal["auto", "magnitude", "phase", "real", "imag"] = "auto"
    colormap: Literal["gray", "viridis", "diverging", "hot"] = "viridis"
    opacity: Annotated[FiniteFloat, Field(ge=0, le=1)] = 0.5
    visible: bool = True
    window: OverlayWindow | None = None


class ComparisonSource(Command):
    experiment: str
    kind: Literal["original", "processed", "map", "reconstruction", "registration"]
    map_name: str | None = None
    registration_key: str | None = None
    frame: int | None = Field(default=None, ge=0)
    component: Literal["auto", "magnitude", "phase", "real", "imag"] = "auto"
    colormap: Literal["gray", "viridis", "diverging", "hot"] = "gray"
    window: OverlayWindow | None = None


class ComparisonSettings(Command):
    enabled: bool = False
    mode: Literal["overlay", "split"] = "split"
    opacity: Annotated[FiniteFloat, Field(ge=0, le=1)] = 0.5
    frame: Annotated[int, Field(ge=0)] | None = None
    plane: Literal["axial", "coronal", "sagittal"] | None = None
    indices: dict[str, Annotated[int, Field(ge=0)]] | None = None
    layout: Literal["swipe", "side-by-side"] = "swipe"
    position: Annotated[FiniteFloat, Field(ge=0.05, le=0.95)] = 0.5
    linked_window: bool = True
    left: ComparisonSource
    right: ComparisonSource


class ComparisonChoice(BaseModel):
    experiment: str
    kind: Literal["original", "processed", "map", "reconstruction", "registration"]
    map_name: str | None = None
    registration_key: str | None = None
    label: str
    frames: int
    dimensions: dict[str, int]


class RoiAppearance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    color: Annotated[str, Field(pattern=r"^#[0-9a-fA-F]{6}$")] | None = None
    opacity: Annotated[FiniteFloat, Field(ge=0, le=1)] | None = None
    label_colors: dict[str, Annotated[str, Field(pattern=r"^#[0-9a-fA-F]{6}$")]] = {}


class SceneUpdate(Command):
    layers: list[OverlayLayer] | None = Field(default=None, max_length=8)
    hidden_rois: list[str] | None = None
    roi_opacity: Annotated[FiniteFloat, Field(ge=0, le=1)] | None = None
    roi_styles: dict[str, RoiAppearance] | None = None
    fit_region: Literal["rois", "all"] | None = None
    fit_rois: list[str] | None = None
    comparison: ComparisonSettings | None = None

    @model_validator(mode="after")
    def explicit_values(self):
        if any(getattr(self, key) is None for key in self.model_fields_set - {"fit_rois"}):
            raise ValueError("Scene settings require explicit values")
        return self


class OverlayChoice(OverlaySource):
    label: str
    frames: int


class OverlayInfo(OverlayLayer):
    label: str
    unavailable: str | None = None


class SceneInfo(BaseModel):
    layers: list[OverlayInfo]
    sources: list[OverlayChoice]
    hidden_rois: list[str]
    roi_opacity: float
    roi_styles: dict[str, RoiAppearance] = {}
    fit_region: Literal["rois", "all"]
    fit_rois: list[str] | None
    comparison: ComparisonSettings
    comparison_sources: list[ComparisonChoice]


class ExperimentInfo(BaseModel):
    reconstruction_label: str = "Reconstruction"
    name: str
    frames: int
    maps: list[str]
    has_reconstruction: bool
    rois: list[str]


class NavigationCommand(Command):
    revision: int = Field(ge=0)
    view_revision: int = Field(ge=0)
    indices: dict[str, int] = Field(min_length=1, max_length=16)
    commit: bool = True
    include_images: bool = True


class ViewCommand(Command):
    revision: int = Field(ge=0)
    frame: int | None = Field(default=None, ge=0)
    z: int | None = Field(default=None, ge=0)
    experiment: str | None = None
    plane: Literal["axial", "coronal", "sagittal"] | None = None
    view: str | None = None
    map_name: str | None = None
    indices: dict[str, int] | None = None
    component: Literal["auto", "magnitude", "phase", "real", "imag"] | None = None
    reconstruction_component: Literal["magnitude", "phase"] | None = None
    coil: int | None = None
    unwrap_phase: bool | None = None
    kspace_mode: str | None = None
    kspace_centering: Literal["auto", "none", "fftshift"] | None = None
    plot_axis: str | None = None
    colormap: Literal["gray", "viridis", "diverging", "hot"] | None = None
    scene: SceneUpdate | None = None
    selected_roi: str | None = None


class RectangleCommand(Command):
    revision: int = Field(ge=0)
    view_revision: int = Field(ge=0)
    name: str = Field(min_length=1, max_length=80, pattern=r"^[\w .-]+$")
    bounds: tuple[
        FiniteFloat,
        FiniteFloat,
        Annotated[FiniteFloat, Field(gt=0)],
        Annotated[FiniteFloat, Field(gt=0)],
    ]

    @field_validator("name")
    @classmethod
    def nonempty_name(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("ROI name must not be empty")
        return value


class CloseCommand(Command):
    revision: int = Field(ge=0)
    discard: bool = False


class PreviewCommand(Command):
    revision: int = Field(ge=0)
    roi_name: str = Field(min_length=1, max_length=80)
    steps: int = Field(default=600, ge=1, le=5000)
    device: str = "cpu"


class TimecourseCommand(Command):
    revision: int = Field(ge=0)
    view_revision: int = Field(ge=0)
    input: Literal["image", "original", "view", "reconstruction"] = "image"
    component: Literal["auto", "magnitude", "phase", "real", "imag"] = "auto"
    statistic: Literal["mean", "median", "std", "variance", "sum"] = "mean"
    axis: str = ""
    roi_names: list[str] = Field(min_length=1, max_length=64)
    respect_exclusions: bool = False


class TimecourseSeries(BaseModel):
    roi: str
    values: list[FiniteFloat | None]
    counts: list[Annotated[int, Field(ge=0)]]


class TimecourseResult(BaseModel):
    workspace_id: str
    experiment: str
    revision: int
    view_revision: int
    input: str
    component: str
    statistic: str
    axis: str
    x: list[FiniteFloat | None]
    x_name: str
    unit: str
    included: list[bool]
    respect_exclusions: bool
    series: list[TimecourseSeries]


class CurveInfo(BaseModel):
    x: list[FiniteFloat | None]
    observed: list[FiniteFloat | None]
    x_name: str
    x_unit: str | None = None
    included: list[bool] | None = None
    dimension: str | None = None

    @model_validator(mode="after")
    def matching_curve_lengths(self):
        if not self.x or len(self.x) != len(self.observed):
            raise ValueError("Curve axes and samples must have matching nonzero lengths")
        return self


class PreviewResult(CurveInfo):
    predicted: list[FiniteFloat | None]
    included: list[bool]
    model: str
    parameters: dict[str, FiniteFloat] | None = None
    relative_rmse: FiniteFloat | None = None
    roi_name: str | None = None

    @model_validator(mode="after")
    def matching_prediction_lengths(self):
        if len(self.x) != len(self.predicted) or len(self.x) != len(self.included):
            raise ValueError("Fit predictions and frame masks must match the curve axis")
        if not any(
            o is not None and p is not None and included
            for o, p, included in zip(self.observed, self.predicted, self.included)
        ):
            raise ValueError("A preview requires finite included observations and predictions")
        return self


class MaskPreview(BaseModel):
    name: str
    dims: list[str]
    shape: list[int]
    plane_dims: list[str]
    plane_shape: list[int]
    slice_dim: str | None
    slice_index: int
    slice_count: int
    dtype: str
    labels: list[int]
    pixels: int
    runs: list[list[int]]


class RoiInfo(BaseModel):
    name: str
    pixels: int
    mask: list[list[int]]
    runs: list[list[int]] = []
    labels: list[int] = []
    enabled: bool = True
    visible: bool = True
    metadata: dict[str, Any] = {}


class PipelineStep(BaseModel):
    id: str
    version: str
    kind: str
    method: str
    family: str
    label: str
    enabled: bool
    options: dict[str, Any]
    result_key: str | None = None
    input_grid: dict[str, Any] | None = None
    output_grid: dict[str, Any] | None = None
    plane_dims: list[str] = []
    noise_estimate: dict[str, Any] | None = None
    result_maps: list[str] = []


class PipelineSource(BaseModel):
    kind: Literal["image", "kspace"]
    format: str
    location: str
    path: str | None = None


class PipelineInfo(BaseModel):
    sources: dict[str, PipelineSource] = {}
    enabled: bool = True
    generation: int = 0
    signature: str = "original"
    steps: list[PipelineStep] = []
    fits_stale: bool = False


class WorkspaceInfo(BaseModel):
    file_location: dict[str, str] | None = None
    id: str
    label: str
    revision: int
    view_revision: int
    frame: int
    z: int
    frame_count: int
    slice_count: int
    shape: list[int]
    spacing: list[float]
    experiment: str
    rois: list[RoiInfo]
    timecourse: CurveInfo | None
    preview: PreviewResult | None
    dirty: bool
    loaded: bool = True
    experiments: list[str] = []
    experiment_catalog: list[ExperimentInfo] = []
    scene: SceneInfo
    slice_dim: str = "z"
    spacing_unit: str = "mm"
    plane: str = "axial"
    plane_dims: list[str] = ["x", "y"]
    plane_axis_labels: list[str] = []
    indices: dict[str, int] = {}
    dimension_controls: list[dict[str, Any]] = []
    dimensions: dict[str, int] = {}
    view: str = "Image"
    editable: bool = True
    display: dict[str, Any] = {}
    channels: int = 1
    image_dtype: str = "float32-le"
    unit: str | None = None
    maps: list[dict[str, Any]] = []
    active_map: str | None = None
    reconstruction_component: str = "magnitude"
    reconstruction_label: str = "Reconstruction"
    reconstruction_coil: int | None = None
    reconstruction_unavailable: str | None = None
    reconstruction_encoding: str | None = None
    coil_count: int = 0
    unwrap_phase: bool = True
    kspace_mode: str = "log_magnitude"
    kspace_centering: str = "auto"
    has_reconstruction: bool = False
    has_kspace: bool = False
    can_reconstruct: bool = False
    can_load_raw: bool = False
    selected_roi: str | None = None
    roi_undo: bool = False
    model: str | None = None
    fit_settings: dict[str, Any] = {}
    frame_exclusions: list[int] = []
    visual_state: dict[str, Any] = {}
    registrations: list[dict[str, Any]] = []
    drifts: list[dict[str, Any]] = []
    fit_info: dict[str, Any] | None = None
    pipeline: PipelineInfo
    diagnostics_count: int = 0
    models: list[dict[str, str]] = []


class PipelineStepProgress(BaseModel):
    id: str
    client_id: str
    label: str
    kind: str
    family: str
    method: str
    status: Literal[
        "pending", "running", "completed", "skipped", "unchanged", "failed", "cancelled"
    ]
    progress: float


class PipelineProgress(BaseModel):
    experiment: str
    status: Literal["running", "completed", "failed", "cancelled", "stale"]
    steps: list[PipelineStepProgress]


class JobInfo(BaseModel):
    id: str
    workspace_id: str
    status: str
    progress: float | None = None
    detail: str
    result: PreviewResult | None = None
    operation: str = "preview"
    summary: dict[str, Any] | None = None
    pipeline: PipelineProgress | None = None


class ActionCommand(Command):
    revision: int = Field(ge=0)
    view_revision: int | None = None
    action: str = Field(min_length=1, max_length=80)
    options: dict[str, Any] = {}


class TaskCommand(Command):
    revision: int = Field(ge=0)
    operation: Literal[
        "preview",
        "full_fit",
        "register",
        "drift",
        "reconstruct",
        "process",
        "pipeline",
        "save",
        "export",
    ]
    options: dict[str, Any] = {}


class NoiseEstimateCommand(Command):
    revision: int = Field(ge=0)
    options: dict[str, Any] = {}


class ProcessingField(BaseModel):
    key: str
    label: str
    kind: Literal["number", "integer", "boolean", "choice"]
    default: Any
    choices: list[str]
    minimum: float | None
    maximum: float | None
    nullable: bool


class ProcessingMethod(BaseModel):
    id: str
    label: str
    group: str
    fields: list[ProcessingField]
    axes: Literal["filter", "series", "none"]
    reference: Literal["frame", "series"] | None
    unavailable: str | None


class ProcessingCatalog(BaseModel):
    methods: list[ProcessingMethod]
    sources: list[dict[str, str]]
    axes: list[str]
    spatial_axes: list[str]
    series_axes: list[str]
    history: list[dict[str, Any]]


class ImportCommand(Command):
    workspace_id: str | None = None
    revision: int | None = None
    root: str | None = None
    path: str = ""
    upload: str | None = None
    options: dict[str, Any] = {}
