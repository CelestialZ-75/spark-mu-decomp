"""Generic distributed linear algebra; no CKC candidate or acceptance policy."""

import operator
import os
import platform
import socket

import numpy as np
from pyspark import StorageLevel, TaskContext
from pyspark.sql import SparkSession

from mu_course import __version__
from mu_course.contracts import Array
from mu_course.partitioning import assemble_projection, time_ranges
from mu_course.validation import finite_array, finite_result, floating_dtype, matrix_array


def _tensor(array):
    import torch

    torch.set_num_threads(1)
    return torch.from_numpy(array)


def _correlation(block):
    matrix = _tensor(block[1])
    return (matrix @ matrix.T).numpy()


def _project(block, vector):
    return block[0], (_tensor(vector) @ _tensor(block[1])).numpy()


def _weighted_sum(block, weights):
    start, matrix = block
    return (_tensor(matrix) @ _tensor(weights[start : start + matrix.shape[1]])).numpy()


def _describe(block):
    import torch

    return {
        "start": block[0],
        "samples": block[1].shape[1],
        "partition": TaskContext.get().partitionId(),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "course": __version__,
    }


class SparkBackend:
    """Own the cached blocks; the caller owns the supplied SparkSession."""

    def __init__(self, session: SparkSession, partitions: int = 2, dtype: str = "float64"):
        self.session = session
        self.partitions = partitions
        self.dtype = floating_dtype(dtype)
        self._blocks = None
        self._shape = None
        self.workers = []

    @property
    def shape(self) -> tuple[int, int]:
        if self._shape is None:
            raise RuntimeError("Call prepare before calculating")
        return self._shape

    def prepare(self, matrix: Array) -> None:
        import torch

        array = matrix_array(matrix, self.dtype)
        ranges = time_ranges(array.shape[1], self.partitions)
        self.close()
        blocks = [(start, array[:, start:stop].copy()) for start, stop in ranges]
        rdd = self.session.sparkContext.parallelize(blocks, self.partitions).persist(
            StorageLevel.MEMORY_AND_DISK
        )
        try:
            # This action both materializes the cache and probes the actual worker runtimes.
            workers = rdd.map(_describe).collect()
            expected = {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "course": __version__,
            }
            if any(any(row[key] != value for key, value in expected.items()) for row in workers):
                raise RuntimeError("Driver/executor runtime versions differ")
        except Exception:
            rdd.unpersist(blocking=True)
            raise
        self._blocks = rdd
        self._shape = array.shape
        self.workers = workers

    def correlation(self) -> Array:
        samples = self.shape[1]
        summed = self._blocks.map(_correlation).treeReduce(operator.add)
        return finite_result(summed / samples)

    def project(self, vector: Array) -> Array:
        array = finite_array(vector, self.dtype, (self.shape[0],))
        broadcast = self.session.sparkContext.broadcast(array)
        try:
            parts = self._blocks.map(lambda block: _project(block, broadcast.value)).collect()
            return finite_result(assemble_projection(parts, self.shape[1]))
        finally:
            broadcast.destroy(blocking=True)

    def weighted_sum(self, weights: Array) -> Array:
        array = finite_array(weights, self.dtype, (self.shape[1],))
        broadcast = self.session.sparkContext.broadcast(array)
        try:
            result = self._blocks.map(
                lambda block: _weighted_sum(block, broadcast.value)
            ).treeReduce(operator.add)
            return finite_result(result)
        finally:
            broadcast.destroy(blocking=True)

    def synchronize(self) -> None:
        # Every public calculation ends in a completed Spark action.
        return None

    def close(self) -> None:
        if self._blocks is not None:
            self._blocks.unpersist(blocking=True)
        self._blocks = None
        self._shape = None
        self.workers = []
