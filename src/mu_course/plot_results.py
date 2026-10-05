"""Plot the saved MU pulses or the benchmark timing table."""

import argparse
import csv
import json
from collections import defaultdict
from itertools import pairwise
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    directory = args.directory
    if (directory / "timings.csv").exists():
        groups = defaultdict(list)
        with (directory / "timings.csv").open(encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                name = row["backend"]
                if name == "spark":
                    name += " p" + row["partitions"]
                groups[name].append(float(row["total_seconds"]))
        if not groups:
            raise ValueError("No completed timing runs to plot")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(list(groups), [np.mean(values) for values in groups.values()])
        ax.set_ylabel("Mean total compute time (s)")
        filename = "timings.png"
    else:
        report = json.loads((directory / "parameters.json").read_text())
        with np.load(directory / "mu_results.npz") as data:
            offsets, pulses = data["offsets"], data["pulses"]
            # The original extender trims ext_factor samples from the input start.
            trains = [
                (pulses[a:b] + report["ext_factor"]) / report["fsamp"] for a, b in pairwise(offsets)
            ]
        fig, ax = plt.subplots(figsize=(8, 4))
        if trains:
            ax.eventplot(trains)
        else:
            ax.text(0.5, 0.5, "No MU extracted", transform=ax.transAxes, ha="center")
        ax.set(
            xlabel="Time within selected input (s)",
            ylabel="MU index",
            title=f"{report['backend']}: {report['mu_count']} MUs",
        )
        filename = "mu_pulses.png"
    fig.tight_layout()
    fig.savefig(directory / filename, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
