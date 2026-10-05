"""Repeat experiments sequentially and write a small timing table."""

import csv
from copy import copy

from mu_course.run_experiment import experiment_parser, run_experiment


def main(argv=None):
    parser = experiment_parser()
    parser.add_argument(
        "--backends", nargs="+", choices=["cpu", "gpu", "spark"], default=["cpu", "gpu", "spark"]
    )
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--partition-counts", nargs="+", type=int, default=[1, 2, 4])
    args = parser.parse_args(argv)
    if args.repeat < 1:
        parser.error("repeat must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    # Write each finished run so a later unavailable backend does not lose earlier results.
    with (args.output / "timings.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "backend",
                "partitions",
                "run",
                "mu_count",
                "fit_seconds",
                "startup_seconds",
                "distribution_seconds",
                "total_seconds",
            ],
        )
        writer.writeheader()
        for backend in args.backends:
            partitions = args.partition_counts if backend == "spark" else [1]
            for count in partitions:
                for run in range(args.repeat + 1):  # First run is warm-up, excluded from CSV.
                    trial = copy(args)
                    trial.backend, trial.partitions = backend, count
                    trial.output = args.output / f"{backend}-p{count}" / f"run-{run}"
                    report = run_experiment(trial)
                    if run:
                        row = {key: report[key] for key in writer.fieldnames if key != "run"}
                        row["run"] = run
                        writer.writerow(row)
                        stream.flush()
                        rows.append(row)
    return rows


if __name__ == "__main__":
    main()
