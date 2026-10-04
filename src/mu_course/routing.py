from mu_course.contracts import BackendUnavailable, MatrixBackend


def make_backend(
    name: str,
    *,
    dtype: str = "float64",
    threads: int = 1,
    session=None,
    partitions: int = 2,
) -> MatrixBackend:
    if name not in ("cpu", "gpu", "spark"):
        raise ValueError(f"Unknown backend: {name}")
    try:
        if name == "spark":
            from mu_course.backends.spark_backend import SparkBackend

            if session is None:
                raise ValueError("Spark backend requires a caller-owned SparkSession")
            return SparkBackend(session, partitions=partitions, dtype=dtype)
        from mu_course.backends.torch_backend import TorchBackend

        return TorchBackend("cuda" if name == "gpu" else "cpu", dtype=dtype, threads=threads)
    except ModuleNotFoundError as error:
        raise BackendUnavailable(
            f"Missing dependency: {error.name}; install cpu or gpu extra"
        ) from error
