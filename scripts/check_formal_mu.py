"""Check pulse and quality agreement for an existing formal MU batch."""

import argparse
import json
from itertools import pairwise
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


def pulses(path):
    with np.load(path, allow_pickle=False) as d:
        offsets = d["offsets"]
        assert offsets[0] == 0 and offsets[-1] == len(d["pulses"])
        assert np.all(np.diff(offsets) > 0)
        for field in ["ipts", "pnrs", "sils", "separation_vectors"]:
            assert np.isfinite(d[field]).all(), (path, field)
        assert np.all(d["sils"] > 0.8), path
        result = [d["pulses"][a:b].copy() for a, b in pairwise(offsets)]
        assert len(result) == len(d["sils"]) == len(d["pnrs"]) > 0
        for p in result:
            assert len(p) >= 3 and np.all(np.diff(p) > 0)
            assert p.min() >= 0 and p.max() < d["ipts"].shape[1]
        return result, d["pnrs"].copy(), d["sils"].copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory", type=Path, help="Formal batch directory, e.g. outputs/formal-..."
    )
    root = parser.parse_args().directory
    reference, reference_pnr, reference_sil = pulses(root / "mu-cpu/run-1/mu_results.npz")
    rows = []
    for group in sorted(p for p in root.glob("mu-*") if p.is_dir()):
        assert (group / "SUCCESS").exists(), f"Incomplete group: {group}"
        for run in range(4):
            folder = group / f"run-{run}"
            report = json.loads((folder / "parameters.json").read_text())
            actual, pnr, sil = pulses(folder / "mu_results.npz")
            assert report["mu_count"] == len(actual) == len(reference), folder
            costs = np.full((len(reference), len(actual)), 1e12)
            for i, a in enumerate(reference):
                for j, b in enumerate(actual):
                    if len(a) == len(b):
                        costs[i, j] = np.max(np.abs(a - b))
            a, b = linear_sum_assignment(costs)
            shift = float(costs[a, b].max())
            assert shift <= 1, (folder, "pulse difference", shift)
            hosts = set(report["worker_hosts"])
            if group.name in ["mu-two-p2", "mu-two-p4"]:
                assert hosts == {"worker-1", "worker-2"}, (folder, hosts)
            if group.name in ["mu-one-half", "mu-one-full"]:
                assert hosts == {"worker-1"}, (folder, hosts)
            rows.append(
                {
                    "group": group.name,
                    "run": run,
                    "max_shift_samples": shift,
                    "mu_count": len(actual),
                    "worker_hosts": sorted(hosts),
                    "cpu_mu_indices": a.tolist(),
                    "matched_mu_indices": b.tolist(),
                    "pnr_db": pnr[b].tolist(),
                    "sil": sil[b].tolist(),
                    "max_pnr_difference_db": float(np.max(np.abs(reference_pnr[a] - pnr[b]))),
                    "max_sil_difference": float(np.max(np.abs(reference_sil[a] - sil[b]))),
                }
            )
    (root / "mu-correctness.json").write_text(json.dumps(rows, indent=2))
    print("MU correctness checks passed:", len(rows), "runs including warm-up")


if __name__ == "__main__":
    main()
