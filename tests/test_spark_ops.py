import numpy as np
import pytest
import torch

from mu_course.spark_ops import SparkOps


@pytest.mark.parametrize("partitions", [1, 2, 4])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_time_block_math(spark, partitions, dtype):
    generator = torch.Generator().manual_seed(13)
    matrix = torch.randn(5, 17, generator=generator, dtype=dtype)
    vector = torch.randn(5, generator=generator, dtype=dtype)
    weights = torch.randn(17, generator=generator, dtype=dtype)
    weights[:2] = weights[-2:] = 0  # The CKC gradient excludes edge samples.
    ops = SparkOps(spark, matrix, partitions)
    try:
        tolerance = 1e-5 if dtype == torch.float32 else 1e-10
        torch.testing.assert_close(
            ops.correlation(), matrix @ matrix.T / 17, rtol=tolerance, atol=tolerance
        )
        torch.testing.assert_close(
            ops.project(vector), vector @ matrix, rtol=tolerance, atol=tolerance
        )
        torch.testing.assert_close(
            ops.weighted_sum(weights), matrix @ weights, rtol=tolerance, atol=tolerance
        )
        np.testing.assert_array_equal(ops.weighted_sum(torch.zeros_like(weights)), np.zeros(5))
    finally:
        ops.close()
    assert not ops.blocks.is_cached
