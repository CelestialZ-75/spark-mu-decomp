"""Build per-MU raster features from a completed cascade template pool."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from mu_course.dataset import DATA_DIR, load_trial
from mu_course.prepare_prediction import (
    check_pair,
    file_hash,
    save_table,
    summarize_windows,
    window_bounds,
)

FEATURES = ("CFR", "FR", "dCFR", "CVISI", "TSLS", "ACTIVE")


def raster_features(pulses, n_samples, fsamp, starts, stops, hanning_ms=400):
    """Return MU-major columns at window ends; CVISI is NaN when unavailable."""
    from mu_course.ckc.mu_feature import MURateEncoder

    cfr = MURateEncoder(fsamp=fsamp, hanning_ms=hanning_ms).encode(pulses, n_samples)
    timestamps = stops - 1
    output = np.empty((len(starts), len(pulses) * len(FEATURES)), dtype=np.float64)
    for mu, raster in enumerate(pulses):
        raster = np.asarray(raster, dtype=np.int64)
        if np.any((raster < 0) | (raster >= n_samples)) or np.any(np.diff(raster) <= 0):
            raise ValueError("Pulse indices must be sorted, unique, and inside the trial")
        past_end = np.searchsorted(raster, timestamps, side="right")
        history = np.searchsorted(raster, timestamps - int(0.2 * fsamp), side="left")
        count = past_end - history
        cvisi = np.full(len(starts), np.nan)
        tsls = np.full(len(starts), 0.5)
        valid = past_end > 0
        tsls[valid] = np.minimum((timestamps[valid] - raster[past_end[valid] - 1]) / fsamp, 0.5)
        for i, end in enumerate(past_end):
            first = np.searchsorted(raster, timestamps[i] - fsamp, side="left")
            isi = np.diff(raster[first:end]).astype(float)
            if len(isi) >= 3:
                cvisi[i] = isi.std() / isi.mean()
        earlier = np.maximum(0, timestamps - int(0.1 * fsamp))
        output[:, mu * len(FEATURES) : (mu + 1) * len(FEATURES)] = np.column_stack(
            [
                summarize_windows(cfr[:, mu : mu + 1], starts, stops).ravel(),
                count / 0.2,
                (cfr[timestamps, mu] - cfr[earlier, mu]) / 0.1,
                cvisi,
                tsls,
                count > 0,
            ]
        )
    return output


def impute_training_medians(train, test):
    """Fill undefined CVISI with training-column median, or zero if absent."""
    if np.isinf(train).any() or np.isinf(test).any():
        raise ValueError("Features contain infinity")
    fill = np.zeros(train.shape[1])
    for col in range(train.shape[1]):
        finite = train[np.isfinite(train[:, col]), col]
        if finite.size:
            fill[col] = np.median(finite)
    return np.where(np.isnan(train), fill, train), np.where(np.isnan(test), fill, test), fill


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--pool-parameters", type=Path, required=True)
    parser.add_argument("--train", type=Path, default=DATA_DIR / "ldx_0deg_1.npz")
    parser.add_argument("--test", type=Path, default=DATA_DIR / "ldx_0deg_2.npz")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Output directory is not empty")
    train, test = load_trial(args.train), load_trial(args.test)
    check_pair(train, test)
    hashes = {"train": file_hash(args.train), "test": file_hash(args.test)}
    if hashes["train"] == hashes["test"]:
        raise ValueError("Use distinct training and test trials")
    parameters = json.loads(args.pool_parameters.read_text(encoding="utf-8"))
    if (
        parameters["fsamp"] != int(train["fsamp"])
        or parameters["channel_ids"] != train["channel_ids"].tolist()
    ):
        raise ValueError("Pool sampling rate or channel order does not match trials")
    ext = parameters["ext_factor"]
    with np.load(args.pool, allow_pickle=False) as pool:
        vectors = pool["vectors"]
        lineages = pool["lineages"]
    if not len(vectors) or vectors.shape != (len(lineages), len(train["channel_ids"]) * ext):
        raise ValueError("Pool vectors and lineage IDs do not match trial channels")
    import torch

    from mu_course.ckc.cascade_decomposer import refine_filters

    torch.set_num_threads(1)
    started = perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    arrays, counts, missing = {}, {}, {}
    fsamp = int(train["fsamp"])
    margin = 2 * ext + int(400 * fsamp / 2000) + 1
    for split, trial in (("train", train), ("test", test)):
        result = refine_filters(
            vectors,
            trial["emg"],
            fsamp=fsamp,
            ext_factor=ext,
            n_iter=0,
            cost_fcn=1,
            sil_threshold=0.8,
            device="cpu",
            verbose=False,
        )
        pulses = [np.asarray(p, dtype=np.int64) + ext for p in result["mu_pulses"]]
        starts, stops = window_bounds(trial["emg"].shape[1], fsamp, 200, 50, margin)
        x = raster_features(pulses, trial["emg"].shape[1], fsamp, starts, stops)
        y = summarize_windows(trial["force"][0, :, None], starts, stops).ravel()
        arrays.update(
            {f"{split}_x": x, f"{split}_y": y, f"{split}_starts": starts, f"{split}_stops": stops}
        )
        counts[split] = [len(p) for p in pulses]
        missing[split] = int(np.isnan(x).sum())
        np.savez_compressed(
            args.output / f"{split}_pulses.npz",
            pulses=np.concatenate(pulses),
            offsets=np.r_[0, np.cumsum(counts[split])],
        )
    arrays["train_x"], arrays["test_x"], fill = impute_training_medians(
        arrays["train_x"], arrays["test_x"]
    )
    names = [f"MU{i:03d}_{name}" for i in range(len(vectors)) for name in FEATURES]
    for split, path in (("train", args.train), ("test", args.test)):
        save_table(
            args.output / f"{split}.csv",
            path.stem,
            split,
            arrays[f"{split}_x"],
            arrays[f"{split}_y"],
            arrays[f"{split}_starts"],
            arrays[f"{split}_stops"],
            fsamp,
        )
    metadata = {
        "feature_kind": "cascade_must",
        "feature_types": FEATURES,
        "feature_columns": names,
        "feature_count": len(names),
        "mu_count": len(vectors),
        "lineages": lineages.tolist(),
        "label_unit": "g",
        "offline": True,
        "fsamp": fsamp,
        "pulse_sample_offset": ext,
        "window_samples": round(0.2 * fsamp),
        "step_samples": round(0.05 * fsamp),
        "margin_samples": margin,
        "window_convention": "[start, stop), mean force in window",
        "pulse_counts": counts,
        "missing_before_imputation": missing,
        "imputation": "training-column median; zero if entirely missing",
        "imputation_values": fill.tolist(),
        "trial_sha256": hashes,
        "train_trial": args.train.stem,
        "test_trial": args.test.stem,
        "pool_sha256": file_hash(args.pool),
        "pool_source_trials": parameters["trials"],
        "pool_parameters_sha256": file_hash(args.pool_parameters),
        "projection": "research refine_filters n_iter=0; per-trial covariance and spike thresholds",
        "evaluation_note": "Check pool source trial overlap before treating test as independent",
        "preparation_seconds": perf_counter() - started,
    }
    np.savez_compressed(args.output / "features.npz", **arrays)
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"mu_count": len(vectors), "feature_count": len(names), "missing": missing}, indent=2
        )
    )


if __name__ == "__main__":
    main()
