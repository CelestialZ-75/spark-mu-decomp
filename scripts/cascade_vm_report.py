"""Check VMware Cascade outputs and redraw the course presentation figures."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

PAIRS = (
    ("rms-ridge", "RMS + Ridge"),
    ("mu-ridge", "Cascade MU features + Ridge"),
    ("mu-gbt", "Cascade MU features + GBT"),
)
FINE_GROUPS = ("one-half", "one-full", "two-p1", "two-p2", "two-p4")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def check_pools(root):
    assert (root / "status.txt").read_text().strip() == "complete", "Cascade is incomplete"
    cpu = read_json(root / "cpu/parameters.json")
    spark = read_json(root / "spark/parameters.json")
    scheduling = {"mode", "slots", "partitions"}
    assert cpu.keys() == spark.keys()
    assert all(cpu[key] == spark[key] for key in cpu.keys() - scheduling)
    levels = len(spark["levels_s"])
    assert levels == 3
    for level in range(levels):
        with (
            np.load(root / "cpu" / f"pool_{level}.npz") as a,
            np.load(root / "spark" / f"pool_{level}.npz") as b,
        ):
            np.testing.assert_array_equal(a["lineages"], b["lineages"])
            for key in ("vectors", "pnrs"):
                assert a[key].shape == b[key].shape
                np.testing.assert_allclose(a[key], b[key], rtol=1e-5, atol=1e-6, equal_nan=False)
            count = len(b["lineages"])
            print(f"Level {level + 1}: {count} matching templates")
    assert count > 0, "Final pool is empty"
    for mode in ("cpu", "spark"):
        assert read_json(root / mode / "summary.json")["final_mu_count"] == count
        events = read_json(root / mode / "events.json")
        previous = None
        for level in range(levels):
            tracks = {
                track: [e for e in events if e["level"] == level and e["track"] == track]
                for track in ("A", "B", "merge")
            }
            assert sorted(e["trial"] for e in tracks["A"]) == [0, 1, 2]
            assert sorted(e["trial"] for e in tracks["B"]) == ([0, 1, 2] if level else [])
            (merge,) = tracks["merge"]
            assert all(e["finished"] <= merge["started"] for e in tracks["A"] + tracks["B"])
            if previous is not None:
                start, end = previous["started"], previous["finished"]
                assert all(e["started"] >= end for e in tracks["B"])
                overlap = any(
                    max(start, e["started"]) < min(end, e["finished"]) for e in tracks["A"]
                )
                print(f"{mode}: merge {level} overlaps A at level {level + 1}: {overlap}")
            previous = merge


def check_features(root):
    rms, mu = root / "features-rms", root / "features-mu"
    a, b = (read_json(folder / "metadata.json") for folder in (rms, mu))
    assert a["feature_kind"] == "rms" and b["feature_kind"] == "cascade_must"
    assert a["trial_sha256"] == b["trial_sha256"]
    assert b["feature_count"] == 6 * b["mu_count"]
    with np.load(rms / "features.npz") as x, np.load(mu / "features.npz") as y:
        for split in ("train", "test"):
            for key in ("starts", "stops", "y"):
                name = f"{split}_{key}"
                np.testing.assert_array_equal(x[name], y[name])
    print(f"Aligned train/test windows: RMS {a['feature_count']} / Cascade {b['feature_count']}")


def curve_data(path):
    rows = read_csv(path)
    keys = [(r["trial_id"], r["window_start_s"], r["window_end_s"]) for r in rows]
    time = np.array([(float(start) + float(end)) / 2 for _, start, end in keys])
    force = np.array([float(r["label_g"]) for r in rows])
    prediction = np.array([float(r["prediction_g"]) for r in rows])
    assert len(rows) and np.all(np.diff(time) > 0)
    return keys, time, force, prediction


def prediction_data(root):
    summary = {r["group"]: r for r in read_csv(root / "formal_summary.csv")}
    curves, rows = [], []
    reference = None
    for pair, label in PAIRS:
        result = []
        for profile in ("local1", "cluster2"):
            group = f"ml-{profile}-{pair}"
            run = root / group / "run-1"
            meta = read_json(run / "metrics.json")
            expected_kind = "rms" if pair.startswith("rms") else "cascade_must"
            assert meta["feature_kind"] == expected_kind, (group, "wrong feature version")
            assert summary[group]["n"] == "3"
            keys, time, force, prediction = curve_data(run / "predictions.csv")
            if reference is None:
                reference = (keys, force)
            assert keys == reference[0], (group, "different prediction windows")
            np.testing.assert_array_equal(force, reference[1])
            result.append((summary[group], meta))
            if profile == "cluster2":
                curves.append((label, time, force, prediction, meta["rmse"]))
        rows.append((label, *result))
    return rows, curves


def plot_fine_grained(summary_csv, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = {row["group"]: row for row in read_csv(summary_csv) if row["kind"] == "mu"}
    groups = [summary[f"mu-{name}"] for name in FINE_GROUPS]
    assert all(r["n"] == "3" for r in groups)
    total, startup, distribution, fit = (
        np.array([float(row[f"{name}_mean_s"]) for row in groups])
        for name in ("total", "startup", "distribution", "fit")
    )
    fig, ax = plt.subplots(figsize=(10, 4.5), layout="constrained")
    offset = np.zeros(len(groups))
    for values, name, color in zip(
        (startup, distribution, fit, np.maximum(0, total - startup - distribution - fit)),
        ("Session startup", "Initial input", "Fit (incl. iteration)", "Other"),
        ("#8caec4", "#d4b38d", "#285c81", "#929fa9"),
    ):
        ax.barh(FINE_GROUPS, values, left=offset, label=name, color=color)
        offset += values
    ax.invert_yaxis()
    ax.set(xlabel="Mean time (s), n=3", title="Fine-grained Spark MU (not Cascade)")
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False)
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "fine-grained-mu-components.png", dpi=180)
    plt.close(fig)
    index = FINE_GROUPS.index("two-p2")
    print(f"Fine-grained two-p2 fit / total: {fit[index] / total[index]:.2%}")


def plot_report(cascade, formal):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    meta = read_json(formal / "features-mu/metadata.json")
    for path, field in (
        ("spark/pool_2.npz", "pool_sha256"),
        ("spark/parameters.json", "pool_parameters_sha256"),
    ):
        assert hashlib.sha256((cascade / path).read_bytes()).hexdigest() == meta[field], path
    rows, curves = prediction_data(formal)
    out = formal / "ppt-figures"
    out.mkdir(exist_ok=True)
    elapsed = [
        read_json(cascade / mode / "summary.json")["elapsed_seconds"] for mode in ("cpu", "spark")
    ]
    slots = read_json(cascade / "spark/parameters.json")["slots"]
    fig, ax = plt.subplots(figsize=(9, 3.5), layout="constrained")
    bars = ax.barh(
        ["CPU sequential", f"Spark local[{slots}] + merger"], elapsed, color=["#438d73", "#285c81"]
    )
    ax.bar_label(bars, fmt="%.2f s", padding=5)
    ax.invert_yaxis()
    ax.set(
        xlim=(0, max(elapsed) * 1.22),
        xlabel="Elapsed time (s)",
        title="Three-trial Cascade in VMware (one run per mode)",
    )
    fig.savefig(out / "cascade-vm-time.png", dpi=180)
    plt.close(fig)

    table = [
        (
            "| Method | Features | local RMSE / MAE (g) | cluster RMSE / MAE (g) | "
            "local total (s) | cluster total (s) |"
        ),
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, (local, lm), (cluster, cm) in rows:
        assert lm["feature_count"] == cm["feature_count"]
        errors = [
            f"{float(r['rmse_mean_g']):.2f} / {float(r['mae_mean_g']):.2f}"
            for r in (local, cluster)
        ]
        times = [
            f"{float(r['total_mean_s']):.2f} ± {float(r['total_sd_s']):.2f}"
            for r in (local, cluster)
        ]
        table.append(f"| {label} | {cm['feature_count']} | " + " | ".join(errors + times) + " |")
    table += [
        "",
        "Three measured runs per group; times are mean ± sample SD.",
        "Offline exploration: Cascade pool construction used unlabeled test-trial EMG.",
    ]
    (out / "prediction-table.md").write_text("\n".join(table) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, sharey=True, layout="constrained")
    fig.suptitle("Offline force prediction, two-worker Docker cluster (run 1)")
    for ax, (label, time, force, prediction, rmse) in zip(axes, curves):
        ax.plot(time, force, label="Measured", color="#285c81", lw=1.7)
        ax.plot(time, prediction, label="Predicted", color="#d47745", lw=1.2)
        ax.set(title=f"{label}  |  RMSE {rmse:.2f} g", ylabel="Force (g)")
        ax.grid(alpha=0.2)
    axes[0].legend(ncol=2)
    axes[-1].set_xlabel("Test trial time (s), window midpoint")
    fig.supxlabel("Cascade pool used unlabeled test-trial EMG.", fontsize=9)
    fig.savefig(out / "prediction-curves.png", dpi=180)
    plt.close(fig)

    print(f"Saved presentation figures and table to {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check-pools").add_argument("cascade", type=Path)
    commands.add_parser("check-features").add_argument("formal", type=Path)
    draw = commands.add_parser("plot")
    draw.add_argument("formal", type=Path)
    fine = commands.add_parser("plot-fine-grained")
    fine.add_argument("summary_csv", type=Path)
    fine.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "check-pools":
        check_pools(args.cascade)
    elif args.command == "check-features":
        check_features(args.formal)
    elif args.command == "plot":
        cascade = Path((args.formal / "cascade-source.txt").read_text(encoding="utf-8").strip())
        plot_report(cascade, args.formal)
    else:
        plot_fine_grained(args.summary_csv, args.output)


if __name__ == "__main__":
    main()
