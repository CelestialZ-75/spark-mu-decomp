"""Summarize an existing formal batch, excluding warm-up run 0."""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory", type=Path, help="Formal batch directory, e.g. outputs/formal-..."
    )
    root = parser.parse_args().directory
    records, summary = [], []
    groups = sorted(p for p in root.iterdir() if p.name.startswith(("mu-", "ml-")) and p.is_dir())
    assert groups, "No experiment groups"
    for group in groups:
        assert (group / "SUCCESS").exists(), f"Incomplete group: {group}"
        kind = group.name.split("-")[0]
        filename = "parameters.json" if kind == "mu" else "metrics.json"
        assert (group / "run-0" / filename).exists(), f"Missing warm-up: {group}"
        group_rows = []
        for run in [1, 2, 3]:
            folder = group / f"run-{run}"
            data = json.loads((folder / filename).read_text())
            row = {"kind": kind, "group": group.name, "run": run}
            for field in [
                "total_seconds",
                "fit_seconds",
                "startup_seconds",
                "distribution_seconds",
            ]:
                value = float(data[field])
                assert math.isfinite(value) and value >= 0, (folder, field)
                row[field] = value
            row["wall_seconds"] = float((group / f"run-{run}.wall-seconds.txt").read_text())
            assert math.isfinite(row["wall_seconds"]) and row["wall_seconds"] >= 0
            row.update(
                mu_count=data.get("mu_count", ""),
                rmse="",
                mae="",
                predict_seconds="",
                train_mean_baseline_rmse="",
            )
            if kind == "ml":
                with (folder / "predictions.csv").open() as f:
                    pred = list(csv.DictReader(f))
                assert len(pred) == data["test_windows"] > 0
                residual = [float(r["prediction_g"]) - float(r["label_g"]) for r in pred]
                rmse = math.sqrt(statistics.mean(x * x for x in residual))
                mae = statistics.mean(abs(x) for x in residual)
                assert math.isfinite(rmse) and math.isfinite(mae), folder
                assert math.isclose(rmse, data["rmse"], rel_tol=1e-8, abs_tol=1e-8)
                assert math.isclose(mae, data["mae"], rel_tol=1e-8, abs_tol=1e-8)
                assert math.isfinite(data["predict_seconds"]) and data["predict_seconds"] >= 0
                assert (
                    math.isfinite(data["train_mean_baseline_rmse"])
                    and data["train_mean_baseline_rmse"] >= 0
                )
                if group.name.startswith("ml-cluster2-"):
                    assert set(data["input_worker_hosts"]) == {"worker-1", "worker-2"}, folder
                else:
                    assert data["master"] == "local[1]", folder
                row.update(
                    rmse=rmse,
                    mae=mae,
                    predict_seconds=data["predict_seconds"],
                    train_mean_baseline_rmse=data["train_mean_baseline_rmse"],
                )
            records.append(row)
            group_rows.append(row)
        aggregate = {"kind": kind, "group": group.name, "n": 3}
        for field, label, unit in [
            ("total_seconds", "total", "s"),
            ("startup_seconds", "startup", "s"),
            ("distribution_seconds", "distribution", "s"),
            ("fit_seconds", "fit", "s"),
            ("predict_seconds", "predict", "s"),
            ("wall_seconds", "wall", "s"),
            ("rmse", "rmse", "g"),
            ("mae", "mae", "g"),
            ("train_mean_baseline_rmse", "baseline_rmse", "g"),
        ]:
            values = [r[field] for r in group_rows if r[field] != ""]
            aggregate[f"{label}_mean_{unit}"] = statistics.mean(values) if values else ""
            aggregate[f"{label}_sd_{unit}"] = statistics.stdev(values) if values else ""
        summary.append(aggregate)
    for name, rows in [("formal_runs.csv", records), ("formal_summary.csv", summary)]:
        with (root / name).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = [
        "# Formal experiment summary",
        "",
        "| Group | n | Total mean ± sample SD (s) | RMSE mean (g) | MAE mean (g) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary:
        error = "" if row["rmse_mean_g"] == "" else f"{row['rmse_mean_g']:.4f}"
        mae = "" if row["mae_mean_g"] == "" else f"{row['mae_mean_g']:.4f}"
        lines.append(
            f"| {row['group']} | 3 | {row['total_mean_s']:.4f} ± {row['total_sd_s']:.4f} | {error} | {mae} |"
        )
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for kind in ["mu", "ml"]:
        selected = [r for r in summary if r["kind"] == kind]
        if not selected:
            continue
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(
            [r["group"] for r in selected],
            [r["total_mean_s"] for r in selected],
            yerr=[r["total_sd_s"] for r in selected],
            capsize=4,
        )
        ax.set_ylabel("Total computation time (s), mean ± sample SD, n=3")
        ax.tick_params(axis="x", labelrotation=30)
        fig.tight_layout()
        fig.savefig(root / f"{kind}-timings.png", dpi=160)
        plt.close(fig)
    print("Completed groups:", len(groups), "formal records:", len(records))


if __name__ == "__main__":
    main()
