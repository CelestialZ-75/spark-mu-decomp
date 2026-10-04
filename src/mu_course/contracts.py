"""The expanded matrix is [features, samples]; all results are host arrays."""

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.floating]


class BackendUnavailable(RuntimeError):
    """A requested runtime cannot execute; no alternative is selected."""


class MatrixBackend(Protocol):
    @property
    def shape(self) -> tuple[int, int]: ...

    def prepare(self, matrix: Array) -> None: ...

    def correlation(self) -> Array:
        """Return X @ X.T / global_sample_count."""
        ...

    def project(self, vector: Array) -> Array:
        """Return vector @ X in original sample order."""
        ...

    def weighted_sum(self, weights: Array) -> Array:
        """Return X @ weights, without per-block or global normalization."""
        ...

    def synchronize(self) -> None: ...

    def close(self) -> None: ...
