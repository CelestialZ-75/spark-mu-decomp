"""Contiguous time partitions keep all feature rows and global sample offsets."""

from numbers import Integral

import numpy as np

from mu_course.contracts import Array


def time_ranges(samples: int, partitions: int) -> list[tuple[int, int]]:
    if any(isinstance(v, bool) or not isinstance(v, Integral) for v in (samples, partitions)):
        raise ValueError("samples and partitions must be integers")
    if samples < 1 or not 1 <= partitions <= samples:
        raise ValueError("Require 1 <= partitions <= samples")
    width, remainder = divmod(samples, partitions)
    start = 0
    ranges = []
    for index in range(partitions):
        stop = start + width + (index < remainder)
        ranges.append((start, stop))
        start = stop
    return ranges


def assemble_projection(parts: list[tuple[int, Array]], samples: int) -> Array:
    ordered = sorted(parts, key=lambda part: part[0])
    offset = 0
    arrays = []
    for start, values in ordered:
        if start != offset or values.ndim != 1 or values.size == 0:
            raise ValueError("Projection blocks overlap, have gaps, or have invalid shapes")
        arrays.append(values)
        offset += values.size
    if offset != samples:
        raise ValueError("Projection blocks do not cover the original sample axis")
    return np.concatenate(arrays)
