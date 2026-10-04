import numpy as np
import pytest

from mu_course.partitioning import assemble_projection, time_ranges


def test_uneven_ranges_cover_samples_once():
    ranges = time_ranges(17, 4)
    assert ranges == [(0, 5), (5, 9), (9, 13), (13, 17)]
    assert [sample for start, stop in ranges for sample in range(start, stop)] == list(range(17))


@pytest.mark.parametrize("samples,partitions", [(0, 1), (3, 0), (3, 4), (3, True), (3, 1.5)])
def test_invalid_partition_counts(samples, partitions):
    with pytest.raises(ValueError):
        time_ranges(samples, partitions)


def test_projection_reassembly_orders_by_global_offset():
    result = assemble_projection([(2, np.array([3.0])), (0, np.array([1.0, 2.0]))], 3)
    np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])


@pytest.mark.parametrize("start", [1, 3])
def test_reassembly_rejects_overlap_and_gap(start):
    with pytest.raises(ValueError):
        assemble_projection([(0, np.ones(2)), (start, np.ones(2))], 4)
