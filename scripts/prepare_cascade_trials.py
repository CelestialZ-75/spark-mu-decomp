"""Prepare three aligned trials in an ignored local experiment directory."""

import argparse
from pathlib import Path

import numpy as np
from prepare_example_data import good_channels

from mu_course.ckc.ckc_preprocess import _load_mat
from mu_course.dataset import DATA_DIR, load_trial


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--third-mat", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    first = [load_trial(DATA_DIR / f"ldx_0deg_{i}.npz") for i in (1, 2)]
    third = _load_mat(str(args.third_mat))
    channels = np.intersect1d(first[0]["channel_ids"], good_channels(third))
    if not len(channels) or not np.array_equal(first[0]["channel_ids"], first[1]["channel_ids"]):
        raise ValueError("No common channels or bundled channel IDs differ")
    args.output.mkdir(parents=True, exist_ok=False)
    for i, trial in enumerate(first, 1):
        rows = np.flatnonzero(np.isin(trial["channel_ids"], channels))
        np.savez_compressed(
            args.output / f"trial{i}.npz",
            emg=trial["emg"][rows],
            fsamp=trial["fsamp"],
            channel_ids=channels,
        )
    np.savez_compressed(
        args.output / "trial3.npz",
        emg=np.asarray(third["emg_clean"][channels], dtype=np.float32),
        fsamp=np.int64(2048),
        channel_ids=channels,
    )
    print(f"Prepared 3 trials with {len(channels)} shared channels in {args.output}")


if __name__ == "__main__":
    main()
