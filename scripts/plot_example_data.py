"""Plot a few EMG channels and the synchronized force from a bundled trial."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mu_course.dataset import DATA_DIR, load_trial


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_DIR / "ldx_0deg_1.npz")
    parser.add_argument("--output", type=Path, default=Path("outputs/example_data.png"))
    args = parser.parse_args(argv)
    trial = load_trial(args.data)
    times = np.arange(trial["emg"].shape[1]) / trial["fsamp"].item()
    figure, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    for row in range(min(3, trial["emg"].shape[0])):
        axes[0].plot(times, trial["emg"][row], linewidth=0.3, alpha=0.6)
    axes[0].set_ylabel("EMG (source units)")
    for row, label in enumerate(("flexion", "extension")):
        axes[1].plot(times, trial["force"][row], label=label)
    axes[1].set(xlabel="Time (s)", ylabel="Force (g)")
    axes[1].legend()
    figure.suptitle(trial["trial_id"].item())
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=150)
    plt.close(figure)
    print(args.output)


if __name__ == "__main__":
    main()
