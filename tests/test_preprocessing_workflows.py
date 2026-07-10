import jax.numpy as jnp
import numpy as np

from mrax.data_struct import DataSet, Experiment
from mrax.preprocessing.drift import (
    linear_drift_correct_dataset,
    linear_drift_correct_framestack,
    negative_mean_squared_error,
)


class AdditiveTransform:
    def identity_parameters(self, image):
        return jnp.zeros(1, dtype=jnp.float32)

    def apply(self, image, parameters):
        return jnp.asarray(image) + jnp.asarray(parameters)[0]


def test_linear_drift_corrects_synthetic_additive_framestack():
    times = np.linspace(0.0, 1.0, 6)
    frames = np.stack(
        [np.full((4, 4), -2.5 * time, dtype=np.float32) for time in times],
        axis=0,
    )

    result = linear_drift_correct_framestack(
        frames,
        transform=AdditiveTransform(),
        metric=negative_mean_squared_error,
        times=times,
        max_parameter_change=4.0,
        optimizer_name="sgd",
        step_size=0.6,
        num_steps=40,
    )

    assert result.diagnostics.success
    assert result.parameters.shape == (6, 1)
    assert np.allclose(result.parameters[:, 0], 2.5 * times, atol=1e-4)
    assert np.allclose(result.corrected, 0.0, atol=1e-4)
    assert result.similarities_after[-1] > result.similarities_before[-1]


def test_linear_drift_dataset_workflow_preserves_experiment_shape():
    experiment = Experiment(dims={"t": 5, "x": 2, "y": 2}, exp_type="dce")
    times = np.linspace(0.0, 1.0, 5)
    data = np.stack(
        [np.full((2, 2), -1.5 * time, dtype=np.float32) for time in times],
        axis=0,
    )

    result = linear_drift_correct_dataset(
        experiment,
        DataSet(jnp.asarray(data), name="drift_demo"),
        dimensions=("t",),
        transform=AdditiveTransform(),
        metric=negative_mean_squared_error,
        times=times,
        max_parameter_change=3.0,
        optimizer_name="sgd",
        step_size=0.6,
        num_steps=40,
    )

    assert result.dataset.shape == experiment.shape
    assert result.dimensions == ("t",)
    assert result.group_dimensions == ()
    assert np.allclose(np.asarray(result.dataset.data), 0.0, atol=1e-4)
