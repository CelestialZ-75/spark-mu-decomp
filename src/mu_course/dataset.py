"""Load the bundled, aligned EMG/force trials without the research repository."""

from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parent / "data"


def load_trial(path: Path = DATA_DIR / "ldx_0deg_1.npz") -> dict:
    """Return EMG [C,N], force [2,N] in grams, fsamp and zero-based channel IDs.

    Time starts at the aligned trial origin. Channel 0 of force is flexion;
    channel 1 is extension. No normalization, windowing or filtering is applied.
    """
    with np.load(path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    emg, force = data["emg"], data["force"]
    if emg.ndim != 2 or force.shape != (2, emg.shape[1]):
        raise ValueError("Expected aligned EMG [C,N] and force [2,N]")
    if not np.isfinite(emg).all() or not np.isfinite(force).all():
        raise ValueError("Trial contains nonfinite samples")
    if data["channel_ids"].shape != (emg.shape[0],):
        raise ValueError("Channel IDs do not match EMG channels")
    if data["fsamp"].item() <= 0 or data["force_unit"].item() != "g":
        raise ValueError("Expected a positive sampling rate and force in grams")
    return data
