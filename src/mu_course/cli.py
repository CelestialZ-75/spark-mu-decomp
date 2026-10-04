"""Infrastructure checks only; there is no public replacement for the private CKC engine."""

import argparse
import json
import os
import platform
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mu-decomp", description=__doc__)
    parser.add_argument("command", choices=["doctor", "kernel-check"])
    parser.add_argument("--backend", choices=["cpu", "gpu", "spark"], required=True)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--master", default="local[2]")
    parser.add_argument("--partitions", type=int, default=2)
    parser.add_argument("--require-two-hosts", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.threads < 1 or args.partitions < 1:
        parser.error("threads and partitions must be positive")
    if args.backend == "spark" and args.threads != 1:
        parser.error("Spark block operations require --threads 1")
    if args.require_two_hosts and args.backend != "spark":
        parser.error("--require-two-hosts requires --backend spark")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = str(1 if args.backend == "spark" else args.threads)

    from mu_course.contracts import BackendUnavailable

    backend = session = None
    try:
        import numpy as np
        import torch

        from mu_course import __version__
        from mu_course.routing import make_backend

        if args.backend == "spark":
            from pyspark.sql import SparkSession

            os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
            builder = SparkSession.builder.master(args.master).appName("mu-course-kernel-check")
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
                builder = builder.config(f"spark.executorEnv.{name}", "1")
            session = builder.getOrCreate()
        backend = make_backend(
            args.backend,
            dtype=args.dtype,
            threads=args.threads,
            session=session,
            partitions=args.partitions,
        )
        rng = np.random.default_rng(42)
        matrix = rng.normal(size=(7, 17)).astype(args.dtype)
        vector = rng.normal(size=7).astype(args.dtype)
        weights = rng.normal(size=17).astype(args.dtype)
        backend.prepare(matrix)
        workers = getattr(backend, "workers", [])
        if args.require_two_hosts and len({worker["host"] for worker in workers}) < 2:
            raise RuntimeError("No evidence of execution on two distinct worker hosts")
        report = {
            "scope": "linear_algebra_only",
            "command": args.command,
            "backend": args.backend,
            "device": getattr(backend, "device", "distributed_cpu"),
            "dtype": args.dtype,
            "shape": list(backend.shape),
            "versions": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "course": __version__,
            },
            "workers": workers,
            "ckc_engine": "not_integrated",
        }
        if args.command == "kernel-check":
            comparisons = {
                "correlation": (backend.correlation(), matrix @ matrix.T / matrix.shape[1]),
                "projection": (backend.project(vector), vector @ matrix),
                "weighted_sum": (backend.weighted_sum(weights), matrix @ weights),
            }
            rtol, atol = (1e-8, 1e-10) if args.dtype == "float64" else (1e-4, 1e-5)
            report["max_absolute_error"] = {}
            for name, (actual, expected) in comparisons.items():
                np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
                report["max_absolute_error"][name] = float(np.max(np.abs(actual - expected)))
        else:
            backend.project(vector)
        backend.synchronize()
        text = json.dumps(report, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(text + "\n")
        print(text)
        return 0
    except (BackendUnavailable, ModuleNotFoundError) as error:
        print(str(error), file=sys.stderr)
        return 3
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2
    except AssertionError as error:
        print(str(error), file=sys.stderr)
        return 5
    except (RuntimeError, FloatingPointError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 4
    finally:
        try:
            if backend is not None:
                backend.close()
        finally:
            if session is not None:
                session.stop()
