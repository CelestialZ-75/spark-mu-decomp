"""Prepare three bundled trials with the same channel order for Cascade CKC."""

import argparse
from functools import reduce
from pathlib import Path

import numpy as np

from mu_course.dataset import DATA_DIR, load_trial


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trials",
        nargs=3,
        type=Path,
        default=[DATA_DIR / f"ldx_0deg_{i}.npz" for i in (1, 2, 3)],
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    trials = [load_trial(path) for path in args.trials]
    if len({int(trial["fsamp"]) for trial in trials}) != 1:
        raise ValueError("Trials have different sampling rates")
    channels = reduce(np.intersect1d, (trial["channel_ids"] for trial in trials))
    if not len(channels):
        raise ValueError("Trials have no common channels")
    args.output.mkdir(parents=True, exist_ok=False)
    for i, trial in enumerate(trials, 1):
        lookup = {int(channel): row for row, channel in enumerate(trial["channel_ids"])}
        rows = [lookup[int(channel)] for channel in channels]
        np.savez_compressed(
            args.output / f"trial{i}.npz",
            emg=trial["emg"][rows],
            fsamp=trial["fsamp"],
            channel_ids=channels,
        )
    print(f"Prepared 3 trials with {len(channels)} shared channels in {args.output}")


if __name__ == "__main__":
    main()
