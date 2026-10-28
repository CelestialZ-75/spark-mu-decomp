"""Check existing prediction files, windows, templates and recomputed metrics."""

import csv
import json
from pathlib import Path

import numpy as np


def main():
    root = Path("outputs")
    features, metadata, checks = {}, {}, []
    for kind in ["rms", "mu"]:
        folder = root / f"prediction-{kind}"
        for name in ["features.npz", "metadata.json", "train.csv", "test.csv"]:
            assert (folder / name).is_file() and (folder / name).stat().st_size > 0, folder / name
        metadata[kind] = json.loads((folder / "metadata.json").read_text())
        m = metadata[kind]
        assert m["feature_kind"] == kind and m["label_unit"] == "g" and m["offline"] is True
        assert m["trial_sha256"]["train"] != m["trial_sha256"]["test"]
        with np.load(folder / "features.npz", allow_pickle=False) as d:
            features[kind] = {k: d[k].copy() for k in d.files}
        for split in ["train", "test"]:
            d = features[kind]
            x, y = d[f"{split}_x"], d[f"{split}_y"]
            assert x.shape == (len(y), m["feature_count"]) and len(y) == m["window_counts"][split]
            assert np.isfinite(x).all() and np.isfinite(y).all()
            assert np.all(d[f"{split}_stops"] - d[f"{split}_starts"] == m["window_samples"])
        assert m["feature_count"] > 0
    for field in [
        "trial_sha256",
        "channel_ids",
        "fsamp",
        "window_samples",
        "step_samples",
        "label_unit",
    ]:
        assert metadata["rms"][field] == metadata["mu"][field], field
    for split in ["train", "test"]:
        for field in ["starts", "stops", "y"]:
            np.testing.assert_array_equal(
                features["rms"][f"{split}_{field}"], features["mu"][f"{split}_{field}"]
            )
    with np.load(root / "prediction-mu/templates.npz", allow_pickle=False) as t:
        m = metadata["mu"]
        assert t["training_sha256"].item() == m["trial_sha256"]["train"]
        np.testing.assert_array_equal(t["channel_ids"], m["channel_ids"])
        assert t["fsamp"].item() == m["fsamp"] and t["ext_factor"].item() == m["ext_factor"]
        assert t["projection_matrix"].shape == (
            m["feature_count"],
            len(m["channel_ids"]) * m["ext_factor"],
        )
        assert len(t["centroids"]) == len(t["polarities"]) == m["feature_count"]
        for field in ["projection_matrix", "centroids", "polarities"]:
            assert np.isfinite(t[field]).all(), field
    for name in ["rms-ridge", "mu-gbt", "mu-ridge"]:
        folder = root / name
        if name == "mu-ridge" and not folder.exists():
            print(name, "可选，未运行")
            continue
        for filename in [
            "metrics.json",
            "predictions.csv",
            "prediction.png",
            "feature_metadata.json",
        ]:
            path = folder / filename
            assert path.is_file() and path.stat().st_size > 0, path
        assert (folder / "model/metadata").is_dir() and (folder / "model/data").is_dir(), folder
        kind, model = name.split("-")
        report = json.loads((folder / "metrics.json").read_text())
        m, d = metadata[kind], features[kind]
        assert json.loads((folder / "feature_metadata.json").read_text()) == m, folder
        assert report["feature_kind"] == kind and report["model"] == model
        assert report["label_unit"] == "g" and report["offline"] is True
        assert report["master"] == "spark://master:7077" and report["partitions"] == 2
        assert set(report["input_worker_hosts"]) == {"worker-1", "worker-2"}, folder
        assert report["train_windows"] == len(d["train_y"]) and report["test_windows"] == len(
            d["test_y"]
        )
        assert report["feature_count"] == m["feature_count"]
        for field in [
            "rmse",
            "mae",
            "train_mean_baseline_rmse",
            "startup_seconds",
            "distribution_seconds",
            "fit_seconds",
            "predict_seconds",
            "total_seconds",
        ]:
            assert np.isfinite(report[field]) and report[field] >= 0, (folder, field)
        with (folder / "predictions.csv").open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == len(d["test_y"]) and all(r["trial_id"] == m["test_trial"] for r in rows)
        for field, expected in [
            ("window_start_s", d["test_starts"] / m["fsamp"]),
            ("window_end_s", d["test_stops"] / m["fsamp"]),
            ("label_g", d["test_y"]),
        ]:
            np.testing.assert_allclose(
                [float(r[field]) for r in rows], expected, rtol=1e-10, atol=1e-10
            )
        prediction = np.array([float(r["prediction_g"]) for r in rows])
        assert np.isfinite(prediction).all()
        error = prediction - d["test_y"]
        for field, value in [
            ("rmse", np.sqrt(np.mean(error**2))),
            ("mae", np.mean(abs(error))),
            (
                "train_mean_baseline_rmse",
                np.sqrt(np.mean((d["test_y"] - d["train_y"].mean()) ** 2)),
            ),
        ]:
            np.testing.assert_allclose(report[field], value, rtol=1e-8, atol=1e-8)
        checks.append(
            {
                "name": name,
                "rmse_g": report["rmse"],
                "mae_g": report["mae"],
                "input_worker_hosts": report["input_worker_hosts"],
            }
        )
        print(name, "文件、时间、标签和指标核对通过", checks[-1])
    target = root / "stages-6-8-check"
    target.mkdir(exist_ok=True)
    (target / "prediction-check.json").write_text(json.dumps(checks, indent=2) + "\n")
    print("Prediction artifact checks passed; inspect prediction curves separately")


if __name__ == "__main__":
    main()
