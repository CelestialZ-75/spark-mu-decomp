import numpy as np

from mu_course.contracts import Array


def floating_dtype(value: str) -> np.dtype:
    dtype = np.dtype(value)
    if dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("dtype must be float32 or float64")
    return dtype


def finite_array(value: Array, dtype: np.dtype, shape: tuple | None = None) -> Array:
    original = np.asarray(value)
    if original.dtype.kind != "f":
        raise ValueError("Input must contain real floating-point values")
    with np.errstate(over="ignore", invalid="ignore"):
        array = np.array(original, dtype=dtype, order="C", copy=True)
    if shape is not None and array.shape != shape:
        raise ValueError(f"Expected shape {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("Input must remain finite in the selected dtype")
    return array


def matrix_array(value: Array, dtype: np.dtype) -> Array:
    array = finite_array(value, dtype)
    if array.ndim != 2 or min(array.shape) == 0:
        raise ValueError("Expected a nonempty [features, samples] matrix")
    return array


def finite_result(value: Array) -> Array:
    if not np.isfinite(value).all():
        raise FloatingPointError("Backend calculation produced non-finite values")
    return value
