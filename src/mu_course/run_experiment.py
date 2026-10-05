"""Run the existing private MU algorithm with a few editable experiment parameters."""

import argparse
import json
import os
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch


def experiment_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--backend", choices=["cpu", "gpu", "spark"], default="cpu")
    parser.add_argument("--fsamp", type=int, default=2048)
    parser.add_argument("--start", type=float, default=3.0)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--adapter", type=int, choices=range(4), default=0)
    parser.add_argument("--ext-factor", type=int, default=10)
    parser.add_argument("--n-mu-max", type=int, default=5)
    parser.add_argument("--n-iter", type=int, default=30)
    parser.add_argument("--cost-fcn", type=int, choices=[1, 3, 5], default=1)
    parser.add_argument("--sil-threshold", type=float, default=0.8)
    parser.add_argument("--partitions", type=int, default=2)
    parser.add_argument("--master", default="local[2]")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("outputs/experiment"))
    return parser


def load_emg(path: Path, fsamp: int, start: float, duration: float, adapter: int):
    if fsamp <= 0 or start < 0 or duration <= 0:
        raise ValueError("Use a positive sampling rate/duration and a nonnegative start")
    good = None
    if path.suffix == ".mat":
        # Reuse only the source loader, not its raw-signal filtering pipeline.
        from ckc_preprocess import _load_mat

        data = _load_mat(str(path))
        matrix = data["emg_clean"]
        good = np.asarray(data.get("good_channels", [])).ravel()
    elif path.suffix == ".npz":
        with np.load(path) as data:
            matrix = data["emg"]
    else:
        matrix = np.load(path)
    if matrix.ndim != 2:
        raise ValueError("EMG must have shape [channels, samples]")
    first, stop = round(start * fsamp), round((start + duration) * fsamp)
    if stop > matrix.shape[1] or stop <= first:
        raise ValueError("Requested time interval is outside the recording")
    channels = np.arange(adapter * 64, min((adapter + 1) * 64, matrix.shape[0]))
    if good is not None and good.size:
        if good.size == matrix.shape[0] and np.isin(good, [0, 1]).all():
            channels = channels[good[channels].astype(bool)]
        else:
            channels = np.intersect1d(channels, good.astype(int) - 1)  # MATLAB indices
    emg = np.asarray(matrix[channels, first:stop], dtype=np.float32)
    if not emg.size or not np.isfinite(emg).all():
        raise ValueError("Selected EMG is empty or contains nonfinite samples")
    return emg


def run_experiment(args):
    if args.threads < 1 or args.ext_factor < 1 or args.n_mu_max < 1 or args.n_iter < 1:
        raise ValueError("threads, ext-factor, n-mu-max and n-iter must be positive")
    if not 0 <= args.sil_threshold <= 1:
        raise ValueError("sil-threshold must be between 0 and 1")
    if args.backend == "gpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; install the gpu extra before running GPU")
    sys.path.insert(0, str(args.core_dir.resolve()))
    from ckc_decomposer import CKCDecomposer
    from ckc_extender import CKCExtender

    torch.set_num_threads(args.threads if args.backend != "spark" else 1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    emg = load_emg(args.data, args.fsamp, args.start, args.duration, args.adapter)
    device = "cuda" if args.backend == "gpu" else "cpu"
    if device == "cuda":
        torch.cuda.synchronize()
    total_start = perf_counter()
    extended = CKCExtender(args.ext_factor, device=device).extend(emg)
    if extended.shape[1] <= args.ext_factor * 20:
        raise ValueError("The interval is too short for the algorithm's edge exclusion")
    model = CKCDecomposer(
        fsamp=args.fsamp,
        ext_factor=args.ext_factor,
        n_mu_max=args.n_mu_max,
        n_iter=args.n_iter,
        cost_fcn=args.cost_fcn,
        sil_threshold=args.sil_threshold,
        device=device,
    )
    session = ops = None
    startup_seconds = distribution_seconds = 0.0
    try:
        if args.backend == "spark":
            from pyspark.sql import SparkSession

            from mu_course.spark_ops import SparkOps

            os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
            start = perf_counter()
            session = (
                SparkSession.builder.master(args.master)
                .appName("mu-experiment")
                .config("spark.executorEnv.OMP_NUM_THREADS", "1")
                .config("spark.executorEnv.MKL_NUM_THREADS", "1")
                .config("spark.executorEnv.OPENBLAS_NUM_THREADS", "1")
                .getOrCreate()
            )
            session.sparkContext.setLogLevel("ERROR")
            startup_seconds = perf_counter() - start
            start = perf_counter()
            ops = SparkOps(session, extended, args.partitions)
            distribution_seconds = perf_counter() - start
        if device == "cuda":
            torch.cuda.synchronize()
        start = perf_counter()
        if ops is None:
            model.fit(extended)
        else:
            model.fit(extended, ops=ops)
        if device == "cuda":
            torch.cuda.synchronize()
        fit_seconds = perf_counter() - start
        report = {
            "backend": args.backend,
            "shape": list(emg.shape),
            "mu_count": model.mu_count,
            "fit_seconds": fit_seconds,
            "startup_seconds": startup_seconds,
            "distribution_seconds": distribution_seconds,
            "total_seconds": perf_counter() - total_start,
            "worker_hosts": sorted(set(ops.hosts)) if ops else [],
            "fsamp": args.fsamp,
            "start": args.start,
            "duration": args.duration,
            "adapter": args.adapter,
            "ext_factor": args.ext_factor,
            "n_mu_max": args.n_mu_max,
            "n_iter": args.n_iter,
            "cost_fcn": args.cost_fcn,
            "sil_threshold": args.sil_threshold,
            "partitions": args.partitions,
            "threads": torch.get_num_threads(),
            "seed": args.seed,
            "dtype": "float32",
            "data": str(args.data.resolve()),
            "core_dir": str(args.core_dir.resolve()),
        }
        args.output.mkdir(parents=True, exist_ok=True)
        lengths = [len(pulses) for pulses in model.mu_pulses]
        np.savez_compressed(
            args.output / "mu_results.npz",
            pulses=np.concatenate(model.mu_pulses) if lengths else np.array([], dtype=int),
            offsets=np.r_[0, np.cumsum(lengths)],
            pnrs=model.pnrs,
            sils=model.sils,
            ipts=np.asarray(model.ipts),
            separation_vectors=model.separation_vectors,
        )
        (args.output / "parameters.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        if not model.mu_count:
            print("No MU extracted: check the input/parameters before drawing conclusions.")
        return report
    finally:
        if ops is not None:
            ops.close()
        if session is not None:
            session.stop()


def main(argv=None):
    args = experiment_parser().parse_args(argv)
    run_experiment(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
