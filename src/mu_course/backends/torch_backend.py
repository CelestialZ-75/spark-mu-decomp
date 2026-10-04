from numbers import Integral

import torch

from mu_course.contracts import Array, BackendUnavailable
from mu_course.validation import finite_array, finite_result, floating_dtype, matrix_array


class TorchBackend:
    def __init__(self, device: str = "cpu", dtype: str = "float64", threads: int = 1):
        if device not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda")
        if isinstance(threads, bool) or not isinstance(threads, Integral) or threads < 1:
            raise ValueError("threads must be a positive integer")
        self.dtype = floating_dtype(dtype)
        if device == "cuda" and not torch.cuda.is_available():
            raise BackendUnavailable("CUDA unavailable; CPU fallback is disabled")
        self.device = device
        torch.set_num_threads(threads)
        self._matrix = None

    @property
    def shape(self) -> tuple[int, int]:
        if self._matrix is None:
            raise RuntimeError("Call prepare before calculating")
        return tuple(self._matrix.shape)

    def prepare(self, matrix: Array) -> None:
        # Own a copy so a caller cannot mutate an already prepared experiment.
        array = matrix_array(matrix, self.dtype)
        self._matrix = torch.from_numpy(array).to(self.device)

    def correlation(self) -> Array:
        samples = self.shape[1]
        result = (self._matrix @ self._matrix.T) / samples
        return finite_result(result.cpu().numpy())

    def project(self, vector: Array) -> Array:
        array = finite_array(vector, self.dtype, (self.shape[0],))
        result = torch.from_numpy(array).to(self.device) @ self._matrix
        return finite_result(result.cpu().numpy())

    def weighted_sum(self, weights: Array) -> Array:
        array = finite_array(weights, self.dtype, (self.shape[1],))
        result = self._matrix @ torch.from_numpy(array).to(self.device)
        return finite_result(result.cpu().numpy())

    def synchronize(self) -> None:
        if self.device == "cuda":
            torch.cuda.synchronize()

    def close(self) -> None:
        self.synchronize()
        self._matrix = None
