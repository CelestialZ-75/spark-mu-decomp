"""Re-export the two bundled trials; normal users do not need the source repository."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from mu_course.ckc.ckc_preprocess import _load_mat


def good_channels(data: dict) -> np.ndarray:
    """Return zero-based good channel IDs in the first 64-channel array."""
    count = data["emg_clean"].shape[0]
    good = np.asarray(data.get("good_channels", np.arange(1, count + 1))).ravel()
    if good.size == count and np.isin(good, [0, 1]).all():
        good = np.flatnonzero(good)
    else:
        good = good.astype(int) - 1
    return np.intersect1d(np.arange(min(64, count)), good)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("src/mu_course/data"))
    args = parser.parse_args(argv)
    paths = [
        args.source_dir / session / "aligned_Middle_Trapezoid.mat"
        for session in ("session0_1", "session0_2")
    ]
    trials = [_load_mat(str(path)) for path in paths]
    channels = np.intersect1d(*(good_channels(data) for data in trials))
    if not channels.size:
        raise ValueError("Trials have no common good channels in adapter 0")
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for index, (path, data) in enumerate(zip(paths, trials), start=1):
        emg = np.asarray(data["emg_clean"][channels], dtype=np.float32)
        force_field = "force_corrected" if "force_corrected" in data else "force_aligned"
        force = np.asarray(data[force_field], dtype=np.float32)
        if force.ndim == 1:
            force = force[None, :]
        elif force.shape[0] > force.shape[1]:
            force = force.T
        if force.shape != (2, emg.shape[1]):
            raise ValueError(f"Expected two aligned force channels, got {force.shape}: {path}")
        if not np.isfinite(emg).all() or not np.isfinite(force).all():
            raise ValueError(f"Nonfinite data in {path}")
        output = args.output / f"ldx_0deg_{index}.npz"
        np.savez_compressed(
            output,
            emg=emg,
            force=force,
            fsamp=np.int64(2048),
            channel_ids=channels,
            trial_id=f"session0_{index}",
            force_unit="g",
            flexion_mvc_g=np.float64(2220.267),
        )
        records.append(
            {
                "file": output.name,
                "source": f"processed_0831LDX/{path.parent.name}/{path.name}",
                "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "emg_shape": list(emg.shape),
                "force_shape": list(force.shape),
                "force_field": force_field,
                "duration_seconds": emg.shape[1] / 2048,
                "bytes": output.stat().st_size,
            }
        )
    metadata = {
        "fsamp": 2048,
        "channel_ids": channels.tolist(),
        "channel_index_base": 0,
        "emg_field": "emg_clean",
        "force_channels": ["Middle flexion", "Middle extension"],
        "force_unit": "g",
        "flexion_mvc_g": 2220.267,
        "time_origin": "Original aligned trial start; no time cropping or resampling",
        "trials": records,
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
