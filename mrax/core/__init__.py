from mrax.devices import get_available_devices

from .batch import discover_study_dirs, run_batch_studies
from .fit_execution import FitError, FitInfo, FitState
from .fit_sanity import FitDebugReport, FitSanityReport, check_fit_sanity, debug_fit_prototype
from .loss_builder import build_loss, make_model_callable, run_fit
from .study_runner import collect_skipped_experiments, execute_study_plan, plan_study_fits
from .synthetic import make_example_suite, make_synthetic_study
from .variance import VarianceEstimate, estimate_data_variance

__all__ = [
    "discover_study_dirs",
    "get_available_devices",
    "build_loss",
    "check_fit_sanity",
    "debug_fit_prototype",
    "make_model_callable",
    "run_fit",
    "FitError",
    "FitInfo",
    "FitState",
    "FitDebugReport",
    "FitSanityReport",
    "make_example_suite",
    "make_synthetic_study",
    "VarianceEstimate",
    "estimate_data_variance",
    "run_batch_studies",
    "plan_study_fits",
    "execute_study_plan",
    "collect_skipped_experiments",
]
