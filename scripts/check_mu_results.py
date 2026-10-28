"""Check existing Docker CPU/Spark MU pulses and quality; do not rerun decomposition."""

import json
from itertools import pairwise
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


def main():
    root = Path("outputs")
    reports, arrays, pulses = [], [], []
    for name in ["docker-cpu", "docker-spark"]:
        folder = root / name
        for filename in ["parameters.json", "mu_results.npz", "mu_pulses.png"]:
            assert (folder / filename).is_file(), folder / filename
        r = json.loads((folder / "parameters.json").read_text())
        with np.load(folder / "mu_results.npz", allow_pickle=False) as saved:
            d = {k: saved[k].copy() for k in saved.files}
        assert all(np.isfinite(v).all() for v in d.values()), folder
        offsets = d["offsets"]
        assert offsets[0] == 0 and offsets[-1] == len(d["pulses"]) and np.all(np.diff(offsets) > 0)
        p = [d["pulses"][a:b] for a, b in pairwise(offsets)]
        assert len(p) == r["mu_count"] == len(d["pnrs"]) == len(d["sils"]) > 0
        assert np.all(d["sils"] > r["sil_threshold"])
        assert all(
            len(v) >= 3 and np.all(np.diff(v) > 0) and v.min() >= 0 and v.max() < d["ipts"].shape[1]
            for v in p
        )
        reports.append(r)
        arrays.append(d)
        pulses.append(p)
    for field in [
        "data",
        "shape",
        "fsamp",
        "start",
        "duration",
        "adapter",
        "ext_factor",
        "n_mu_max",
        "n_iter",
        "cost_fcn",
        "sil_threshold",
        "seed",
        "dtype",
        "mu_count",
    ]:
        assert reports[0][field] == reports[1][field], field
    assert reports[0]["backend"] == "cpu" and reports[1]["backend"] == "spark"
    assert set(reports[1]["worker_hosts"]) == {"worker-1", "worker-2"}
    cost = np.array(
        [
            [np.max(np.abs(a - b)) if len(a) == len(b) else 1e12 for b in pulses[1]]
            for a in pulses[0]
        ]
    )
    left, right = linear_sum_assignment(cost)
    assert cost[left, right].max() <= 1, "Pulse mismatch; preserve both results and investigate"
    rows = []
    for a, b in zip(left, right):
        row = {
            "cpu_mu": int(a),
            "spark_mu": int(b),
            "pulse_count": len(pulses[0][a]),
            "max_shift_samples": float(cost[a, b]),
        }
        for key in ["pnrs", "sils"]:
            row[key] = {
                "cpu": float(arrays[0][key][a]),
                "spark": float(arrays[1][key][b]),
                "abs_difference": float(abs(arrays[0][key][a] - arrays[1][key][b])),
            }
        rows.append(row)
    target = root / "stages-6-8-check"
    target.mkdir(exist_ok=True)
    (target / "mu-check.json").write_text(json.dumps(rows, indent=2) + "\n")
    print("MU checks passed:", len(rows), "matched MUs; see", target / "mu-check.json")


if __name__ == "__main__":
    main()
