"""Check runtime imports and real tensor/Spark work; this does not run CKC."""

import argparse
import importlib.metadata
import json
import os
import platform
import socket

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def probe_partition(values):
    import torch

    torch.set_num_threads(1)
    items = list(values)
    result = torch.tensor(items, dtype=torch.float64).square().sum().item()
    yield {
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "items": items,
        "sum_squares": result,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["cpu", "gpu", "spark"], required=True)
    parser.add_argument("--master", default="local[2]")
    parser.add_argument("--require-two-hosts", action="store_true")
    args = parser.parse_args()
    if args.require_two_hosts and args.backend != "spark":
        parser.error("--require-two-hosts requires --backend spark")

    import h5py  # noqa: F401
    import numpy  # noqa: F401
    import scipy  # noqa: F401
    import torch
    import yaml  # noqa: F401

    torch.set_num_threads(1)
    report = {
        "scope": "environment_only",
        "core_check": "not_implemented",
        "backend": args.backend,
        "python": platform.python_version(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ["torch", "numpy", "scipy", "h5py", "pyyaml", "pyspark"]
        },
    }
    if args.backend == "spark":
        from pyspark.sql import SparkSession

        spark = (
            SparkSession.builder.master(args.master).appName("mu-environment-smoke").getOrCreate()
        )
        try:
            rows = (
                spark.sparkContext.parallelize(range(16), 2)
                .mapPartitions(probe_partition)
                .collect()
            )
            if sum(row["sum_squares"] for row in rows) != 1240:
                raise RuntimeError("Distributed tensor calculation failed")
            if sorted(item for row in rows for item in row["items"]) != list(range(16)):
                raise RuntimeError("Partition coverage failed")
            for row in rows:
                if row["torch"] != torch.__version__ or row["python"] != platform.python_version():
                    raise RuntimeError("Driver/executor runtime mismatch")
            if args.require_two_hosts and len({row["host"] for row in rows}) < 2:
                raise RuntimeError(
                    "No evidence of work on two distinct worker containers; retry and inspect UI"
                )
            report["partitions"] = rows
        finally:
            spark.stop()
    else:
        device = "cuda" if args.backend == "gpu" else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; CPU fallback is disabled")
        tensor = torch.arange(16, dtype=torch.float64, device=device)
        actual = tensor.square().sum()
        if device == "cuda":
            torch.cuda.synchronize()
            report["gpu"] = torch.cuda.get_device_name()
        if actual.item() != 1240:
            raise RuntimeError("Tensor calculation failed")
        report["device"] = str(tensor.device)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
