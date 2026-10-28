"""Prepare aligned offline RMS or fixed-template MU features from separate trials."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from mu_course.dataset import DATA_DIR, load_trial
from mu_course.runtime_info import runtime_info


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def window_bounds(n_samples, fsamp, window_ms=200, step_ms=50, margin=0):
    """Return [start, stop) sample indices; discard incomplete/edge windows."""
    width, step = round(window_ms * fsamp / 1000), round(step_ms * fsamp / 1000)
    if width < 1 or step < 1 or margin < 0:
        raise ValueError("Window/step must be positive and margin nonnegative")
    starts = np.arange(margin, n_samples - margin - width + 1, step, dtype=np.int64)
    if len(starts) < 2:
        raise ValueError("Recording is too short for two complete windows")
    return starts, starts + width


def summarize_windows(values, starts, stops, rms=False):
    """Reduce [samples, features] to window means or root mean squares."""
    values = np.asarray(values, dtype=np.float64)
    if rms:
        values = values**2
    result = np.stack([values[a:b].mean(axis=0) for a, b in zip(starts, stops)])
    return np.sqrt(result) if rms else result


def check_pair(train, test):
    if train["fsamp"].item() != test["fsamp"].item():
        raise ValueError("Training and test sampling rates differ")
    if not np.array_equal(train["channel_ids"], test["channel_ids"]):
        raise ValueError("Training and test channel order differs")


def fit_templates(train, args):
    """Fit only the training EMG; persist projection, thresholds and polarity."""
    import torch

    from mu_course.ckc.ckc_decomposer import CKCDecomposer
    from mu_course.ckc.ckc_extender import CKCExtender

    fsamp = int(train["fsamp"])
    first = round(args.template_start * fsamp)
    stop = first + round(args.template_duration * fsamp)
    if first < 0 or stop > train["emg"].shape[1] or stop - first <= args.ext_factor * 22:
        raise ValueError("Template interval is outside training data or too short")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    extended = CKCExtender(args.ext_factor, device="cpu").extend(train["emg"][:, first:stop])
    model = CKCDecomposer(
        fsamp=fsamp,
        ext_factor=args.ext_factor,
        n_mu_max=args.n_mu_max,
        n_iter=args.n_iter,
        sil_threshold=0.8,
        device="cpu",
    ).fit(extended)
    if model.mu_count == 0:
        raise ValueError("No training MU templates found; inspect training interval")
    projection = model.separation_vectors.T @ model._inv_corr_sig.cpu().numpy()
    projected = projection @ extended.numpy()
    # Freeze sign on training data, rather than deciding polarity from each test trial.
    polarities = np.where(-projected.min(axis=1) > projected.max(axis=1), -1, 1)
    template = {
        "projection_matrix": projection,
        "centroids": model.centroids,
        "polarities": polarities,
        "channel_ids": train["channel_ids"],
        "fsamp": fsamp,
        "ext_factor": args.ext_factor,
        "training_start_sample": first,
        "training_stop_sample": stop,
        "training_sha256": file_hash(args.train),
        "seed": args.seed,
    }
    if not np.isfinite(projection).all() or not np.all(model.centroids.mean(axis=1) > 0):
        raise ValueError("Invalid projection or nonpositive training thresholds")
    return template


def template_rates(trial, template, hanning_ms):
    """Return full-trial Hz features; restore extender's leading sample offset."""
    from mu_course.ckc.ckc_decomposer import apply_templates
    from mu_course.ckc.mu_feature import MURateEncoder

    if not np.array_equal(trial["channel_ids"], template["channel_ids"]):
        raise ValueError("Template channel order differs")
    fsamp, ext = int(template["fsamp"]), int(template["ext_factor"])
    if int(trial["fsamp"]) != fsamp:
        raise ValueError("Template sampling rate differs")
    result = apply_templates(
        template["projection_matrix"],
        template["centroids"],
        trial["emg"],
        fsamp=fsamp,
        ext_factor=ext,
        device="cpu",
        polarities=template["polarities"],
    )
    pulses = [np.asarray(p, dtype=np.int64) + ext for p in result["mu_pulses"]]
    rates = MURateEncoder(fsamp=fsamp, hanning_ms=hanning_ms).encode(pulses, trial["emg"].shape[1])
    return rates, pulses


def save_table(path, trial_id, split, x, y, starts, stops, fsamp):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "trial_id",
                "split",
                "window_start_s",
                "window_end_s",
                "label",
                *[f"feature_{i}" for i in range(x.shape[1])],
            ]
        )
        for i in range(len(y)):
            writer.writerow([trial_id, split, starts[i] / fsamp, stops[i] / fsamp, y[i], *x[i]])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", choices=["rms", "mu"], required=True)
    parser.add_argument("--train", type=Path, default=DATA_DIR / "ldx_0deg_1.npz")
    parser.add_argument("--test", type=Path, default=DATA_DIR / "ldx_0deg_2.npz")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-ms", type=float, default=200)
    parser.add_argument("--step-ms", type=float, default=50)
    parser.add_argument("--hanning-ms", type=float, default=400)
    parser.add_argument("--ext-factor", type=int, default=10)
    parser.add_argument("--template-start", type=float, default=3)
    parser.add_argument("--template-duration", type=float, default=2)
    parser.add_argument("--n-mu-max", type=int, default=5)
    parser.add_argument("--n-iter", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Output directory is not empty; select a new output directory")
    if args.ext_factor < 1 or args.hanning_ms <= 0 or min(args.n_mu_max, args.n_iter) < 1:
        parser.error("Extension, Hanning duration and iteration counts must be positive")
    hashes = {
        split: file_hash(path) for split, path in [("train", args.train), ("test", args.test)]
    }
    if hashes["train"] == hashes["test"]:
        raise ValueError("Use distinct training and test trials")
    started = perf_counter()
    train, test = load_trial(args.train), load_trial(args.test)
    check_pair(train, test)
    fsamp = int(train["fsamp"])
    # Both feature families use identical windows, excluding MU edge/smoothing support.
    margin = 2 * args.ext_factor + int(args.hanning_ms * fsamp / 2000) + 1
    args.output.mkdir(parents=True, exist_ok=True)
    template = fit_templates(train, args) if args.features == "mu" else None
    if template is not None:
        np.savez_compressed(args.output / "templates.npz", **template)
        # Exercise the persisted artifact, not only the in-memory model.
        with np.load(args.output / "templates.npz", allow_pickle=False) as saved:
            template = {name: saved[name] for name in saved.files}
    arrays, counts, pulse_counts = {}, {}, {}
    for split, trial, path in [("train", train, args.train), ("test", test, args.test)]:
        starts, stops = window_bounds(
            trial["emg"].shape[1], fsamp, args.window_ms, args.step_ms, margin
        )
        if template is None:
            values = trial["emg"].T
        else:
            values, pulses = template_rates(trial, template, args.hanning_ms)
            pulse_counts[split] = [len(p) for p in pulses]
            np.savez_compressed(
                args.output / f"{split}_pulses.npz",
                pulses=np.concatenate(pulses),
                offsets=np.r_[0, np.cumsum(pulse_counts[split])],
            )
        x = summarize_windows(values, starts, stops, rms=template is None)
        y = summarize_windows(trial["force"][0, :, None], starts, stops).ravel()
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Features or labels contain nonfinite values")
        if split == "train" and not np.any(np.var(x, axis=0) > 0):
            raise ValueError("Training features are all constant; inspect templates/data")
        arrays.update(
            {f"{split}_x": x, f"{split}_y": y, f"{split}_starts": starts, f"{split}_stops": stops}
        )
        counts[split] = len(y)
        save_table(args.output / f"{split}.csv", path.stem, split, x, y, starts, stops, fsamp)
    metadata = {
        "runtime": runtime_info(),
        "feature_kind": args.features,
        "label_unit": "g",
        "force_channel": "Middle flexion",
        "fsamp": fsamp,
        "channel_ids": train["channel_ids"].tolist(),
        "window_samples": round(args.window_ms * fsamp / 1000),
        "step_samples": round(args.step_ms * fsamp / 1000),
        "margin_samples": margin,
        "window_convention": "[start, stop), mean force in the same window",
        "offline": True,
        "hanning_ms": args.hanning_ms,
        "ext_factor": args.ext_factor,
        "pulse_sample_offset": args.ext_factor,
        "pulse_counts": pulse_counts,
        "trial_sha256": hashes,
        "train_trial": args.train.stem,
        "test_trial": args.test.stem,
        "feature_count": arrays["train_x"].shape[1],
        "window_counts": counts,
        "feature_columns": [f"feature_{i}" for i in range(arrays["train_x"].shape[1])],
        "template_interval_s": [args.template_start, args.template_duration] if template else None,
        "seed": args.seed,
        "template_parameters": {
            "n_mu_max": args.n_mu_max,
            "n_iter": args.n_iter,
            "cost_fcn": 1,
            "sil_threshold": 0.8,
            "threads": 1,
        }
        if template
        else None,
        "preparation_seconds": perf_counter() - started,
    }
    np.savez_compressed(args.output / "features.npz", **arrays)
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
