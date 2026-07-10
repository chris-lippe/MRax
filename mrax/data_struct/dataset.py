from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass
class DataSet:
    """Container for quantitative MRI data and optional raw k-space.

    Non-Cartesian acquisitions store samples in ``kspace`` with the sample axis last
    and store the corresponding trajectory in ``trajectory`` as ``(num_samples, 2)``.
    Cartesian and EPI acquisitions can also provide ``sampling_mask`` for
    undersampled grids.
    """

    data: jnp.ndarray
    name: str = "default"
    noise_level: float | None = None
    processed: dict | None = None
    kspace: jnp.ndarray | None = None
    trajectory: jnp.ndarray | None = None
    density: jnp.ndarray | None = None
    sampling_mask: jnp.ndarray | None = None
    kspace_metadata: dict | None = None

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.data.shape)

    @property
    def spatial_shape(self) -> tuple[int, ...]:
        if self.data.ndim >= 2:
            return tuple(self.data.shape[-2:])
        return tuple(self.data.shape)
