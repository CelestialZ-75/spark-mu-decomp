"""Train CPU MLlib Ridge/GBT on one trial and evaluate a separate trial."""

import argparse
import csv
import json
import os
import socket
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

from mu_course.runtime_info import runtime_info


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--model", choices=["ridge", "gbt"], required=True)
    parser.add_argument("--master", default="local[2]")
    parser.add_argument("--partitions", type=int, default=2)
    parser.add_argument("--reg-param", type=float, default=0.1)
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--require-two-hosts", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if min(args.partitions, args.max_iter, args.max_depth) < 1 or args.reg_param < 0:
        parser.error("Counts must be positive; reg-param must be nonnegative")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Output directory is not empty; select a new output directory")
    metadata = json.loads((args.features / "metadata.json").read_text())
    with np.load(args.features / "features.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    train_x, test_x = arrays["train_x"], arrays["test_x"]
    if train_x.ndim != 2 or test_x.ndim != 2 or train_x.shape[1] != test_x.shape[1]:
        raise ValueError("Training/test feature columns differ")
    for split in ["train", "test"]:
        x, y = arrays[f"{split}_x"], arrays[f"{split}_y"]
        if len(y) < 2 or len(x) != len(y) or not x.shape[1]:
            raise ValueError("Expected nonempty aligned features/labels")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Nonfinite features/labels")
    if metadata["trial_sha256"]["train"] == metadata["trial_sha256"]["test"]:
        raise ValueError("Training and test must be separate trials")

    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.regression import GBTRegressor, LinearRegression
    from pyspark.sql import SparkSession

    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    total_start = perf_counter()
    session = SparkSession.builder.master(args.master).appName("force-prediction").getOrCreate()
    session.sparkContext.setLogLevel("WARN")
    startup_seconds = perf_counter() - total_start
    frames = []
    try:
        began = perf_counter()
        columns = [f"feature_{i}" for i in range(train_x.shape[1])]
        assembler = VectorAssembler(inputCols=columns, outputCol="features")
        for split in ["train", "test"]:
            rows = [
                (i, float(y), *map(float, x))
                for i, (x, y) in enumerate(zip(arrays[f"{split}_x"], arrays[f"{split}_y"]))
            ]
            frame = (
                assembler.transform(
                    session.createDataFrame(rows, ["row_id", "label", *columns]).repartition(
                        args.partitions
                    )
                )
                .select("row_id", "label", "features")
                .cache()
            )
            frame.count()  # Materialize separately from model fitting.
            frames.append(frame)
        hosts = sorted(
            set(
                frames[0]
                .rdd.mapPartitions(
                    lambda rows: [socket.gethostname()] if next(rows, None) is not None else []
                )
                .collect()
            )
        )
        if args.require_two_hosts and len(hosts) < 2:
            raise RuntimeError(f"Expected two input-data worker hosts, got {hosts}")
        distribution_seconds = perf_counter() - began
        estimator = (
            LinearRegression(
                regParam=args.reg_param,
                elasticNetParam=0.0,
                standardization=True,
                maxIter=args.max_iter,
                solver="normal",
            )
            if args.model == "ridge"
            else GBTRegressor(
                maxIter=args.max_iter,
                maxDepth=args.max_depth,
                seed=args.seed,
                stepSize=0.1,
                maxBins=32,
            )
        )
        began = perf_counter()
        model = estimator.fit(frames[0])
        fit_seconds = perf_counter() - began
        began = perf_counter()
        rows = model.transform(frames[1]).select("row_id", "prediction").orderBy("row_id").collect()
        predictions = np.array([row.prediction for row in rows])
        predict_seconds = perf_counter() - began
        if len(predictions) != len(arrays["test_y"]) or not np.isfinite(predictions).all():
            raise ValueError("Invalid predictions")
        error = predictions - arrays["test_y"]
        constant_error = arrays["test_y"] - arrays["train_y"].mean()
        report = {
            "runtime": runtime_info(),
            "model": args.model,
            "feature_kind": metadata["feature_kind"],
            "label_unit": metadata["label_unit"],
            "offline": metadata["offline"],
            "rmse": float(np.sqrt(np.mean(error**2))),
            "mae": float(np.mean(np.abs(error))),
            "train_mean_baseline_rmse": float(np.sqrt(np.mean(constant_error**2))),
            "train_windows": len(train_x),
            "test_windows": len(test_x),
            "feature_count": train_x.shape[1],
            "input_worker_hosts": hosts,
            "master": args.master,
            "partitions": args.partitions,
            "seed": args.seed,
            "spark_version": session.version,
            "startup_seconds": startup_seconds,
            "distribution_seconds": distribution_seconds,
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
            "total_seconds": perf_counter() - total_start,
            "timing_scope": "Spark startup through collected predictions; excludes I/O and saving",
            "model_parameters": {p.name: v for p, v in estimator.extractParamMap().items()},
        }
        args.output.mkdir(parents=True, exist_ok=True)
        fsamp = metadata["fsamp"]
        with (args.output / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                ["trial_id", "window_start_s", "window_end_s", "label_g", "prediction_g"]
            )
            for i, value in enumerate(predictions):
                writer.writerow(
                    [
                        metadata["test_trial"],
                        arrays["test_starts"][i] / fsamp,
                        arrays["test_stops"][i] / fsamp,
                        arrays["test_y"][i],
                        value,
                    ]
                )
        model.write().save(str((args.output / "model").resolve()))
        (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
        (args.output / "feature_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        times = (arrays["test_starts"] + arrays["test_stops"]) / (2 * fsamp)
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(times, arrays["test_y"], label="Measured force")
        ax.plot(times, predictions, label=f"{metadata['feature_kind'].upper()} + {args.model}")
        ax.set(xlabel="Time from test trial origin (s)", ylabel="Middle flexion force (g)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.output / "prediction.png", dpi=150)
        plt.close(fig)
        print(json.dumps(report, indent=2))
        return 0
    finally:
        for frame in frames:
            frame.unpersist()
        session.stop()


if __name__ == "__main__":
    raise SystemExit(main())
