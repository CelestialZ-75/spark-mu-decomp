import numpy as np
import pytest
import torch

from mu_course.backends.spark_backend import SparkBackend
from mu_course.backends.torch_backend import TorchBackend
from mu_course.contracts import BackendUnavailable


def check_operations(backend, dtype):
    rng = np.random.default_rng(13)
    matrix = rng.normal(size=(5, 17)).astype(dtype)
    matrix[4] = matrix[0]  # Rank deficiency must not invalidate these operations.
    vector = rng.normal(size=5).astype(dtype)
    weights = rng.normal(size=17).astype(dtype)
    backend.prepare(matrix)
    tolerance = (
        {"rtol": 1e-8, "atol": 1e-10} if dtype == "float64" else {"rtol": 1e-4, "atol": 1e-5}
    )
    np.testing.assert_allclose(backend.correlation(), matrix @ matrix.T / 17, **tolerance)
    np.testing.assert_allclose(backend.project(vector), vector @ matrix, **tolerance)
    np.testing.assert_allclose(backend.weighted_sum(weights), matrix @ weights, **tolerance)
    np.testing.assert_array_equal(backend.weighted_sum(np.zeros(17, dtype=dtype)), np.zeros(5))
    assert backend.project(vector).dtype == matrix.dtype


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_cpu_operations(dtype):
    backend = TorchBackend(dtype=dtype)
    try:
        check_operations(backend, dtype)
    finally:
        backend.close()


@pytest.mark.parametrize(
    "partitions,dtype", [(1, "float64"), (2, "float64"), (4, "float64"), (3, "float32")]
)
def test_spark_operations(spark, partitions, dtype):
    backend = SparkBackend(spark, partitions=partitions, dtype=dtype)
    try:
        check_operations(backend, dtype)
        assert len(backend.workers) == partitions
        assert sum(worker["samples"] for worker in backend.workers) == 17
        cached = backend._blocks
    finally:
        backend.close()
    assert not cached.is_cached
    assert spark.sparkContext.parallelize([1], 1).sum() == 1


def test_gpu_cannot_silently_fall_back(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(BackendUnavailable):
        TorchBackend(device="cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime is not installed/available")
def test_real_cuda_operations():
    backend = TorchBackend(device="cuda")
    try:
        check_operations(backend, "float64")
    finally:
        backend.close()


@pytest.mark.parametrize(
    "bad",
    [
        np.ones((0, 3)),
        np.ones(3),
        np.array([[np.nan]]),
        np.array([[np.inf]]),
        np.ones((2, 3), dtype=int),
    ],
)
def test_invalid_matrices(bad):
    with pytest.raises(ValueError):
        TorchBackend().prepare(bad)


def test_owned_input_shape_checks_and_close():
    backend = TorchBackend()
    with pytest.raises(RuntimeError):
        backend.correlation()
    matrix = np.ones((2, 3))
    backend.prepare(matrix)
    matrix[:] = 99
    np.testing.assert_array_equal(backend.project(np.ones(2)), np.full(3, 2.0))
    with pytest.raises(ValueError):
        backend.project(np.ones(3))
    with pytest.raises(ValueError):
        backend.weighted_sum(np.full(3, np.nan))
    backend.close()
    backend.close()
    with pytest.raises(RuntimeError):
        backend.project(np.ones(2))


def test_zero_matrix_is_well_defined():
    backend = TorchBackend()
    backend.prepare(np.zeros((2, 3)))
    np.testing.assert_array_equal(backend.correlation(), np.zeros((2, 2)))
    np.testing.assert_array_equal(backend.project(np.ones(2)), np.zeros(3))
    np.testing.assert_array_equal(backend.weighted_sum(np.ones(3)), np.zeros(2))
    backend.close()


def test_dtype_conversion_cannot_introduce_infinity():
    with pytest.raises(ValueError, match="finite"):
        TorchBackend(dtype="float32").prepare(np.full((2, 3), 1e60))


def test_calculation_overflow_is_reported():
    backend = TorchBackend(dtype="float32")
    backend.prepare(np.full((2, 3), 1e30, dtype=np.float32))
    with pytest.raises(FloatingPointError):
        backend.correlation()
    backend.close()
