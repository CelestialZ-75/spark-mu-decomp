"""Protect time alignment, fixed-template transfer and held-out trial evaluation."""

import json

import numpy as np
import pytest

from mu_course.dataset import DATA_DIR, load_trial
from mu_course.prepare_prediction import (
    check_pair,
    summarize_windows,
    template_rates,
    window_bounds,
)
from mu_course.prepare_prediction import (
    main as prepare,
)


def test_windows_use_exact_sample_bounds_and_force_mean():
    starts, stops = window_bounds(20, 10, window_ms=400, step_ms=200, margin=2)
    np.testing.assert_array_equal(starts, [2, 4, 6, 8, 10, 12, 14])
    np.testing.assert_array_equal(stops, starts + 4)
    values = np.arange(20)[:, None]
    np.testing.assert_allclose(summarize_windows(values, starts, stops).ravel(), starts + 1.5)
    np.testing.assert_allclose(summarize_windows(np.ones((20, 2)) * -3, starts, stops, rms=True), 3)
    with pytest.raises(ValueError, match="too short"):
        window_bounds(2, 2048)


def test_pair_rejects_channel_permutation_and_same_trial(tmp_path):
    train = load_trial()
    test = dict(train, channel_ids=train["channel_ids"][::-1])
    with pytest.raises(ValueError, match="channel order"):
        check_pair(train, test)
    with pytest.raises(ValueError, match="distinct"):
        prepare(
            [
                "--features",
                "rms",
                "--test",
                str(DATA_DIR / "ldx_0deg_1.npz"),
                "--output",
                str(tmp_path),
            ]
        )


def test_nonempty_template_pulses_restore_original_time_and_keep_silent_column():
    signal = np.zeros((1, 1000), dtype=np.float32)
    signal[0, [100, 300, 600]] = 3
    signal[0, 800] = -20  # Test amplitude must not flip training polarity.
    template = {
        "projection_matrix": np.array([[1, 0], [0, 0]], dtype=np.float32),
        "centroids": np.array([[8, 0], [8, 0]]),
        "polarities": np.array([1, 1]),
        "fsamp": 1000,
        "ext_factor": 2,
        "channel_ids": np.array([0]),
    }
    trial = {"emg": signal, "fsamp": np.array(1000), "channel_ids": np.array([0])}
    rates, pulses = template_rates(trial, template, hanning_ms=20)
    np.testing.assert_array_equal(pulses[0], [100, 300, 600])
    assert pulses[1].size == 0
    assert rates.shape == (1000, 2)
    assert np.all(rates[:, 1] == 0)
    assert rates[100, 0] > 0


def test_real_trial_features_share_windows_and_reload_templates(tmp_path):
    for kind in ["rms", "mu"]:
        prepare(["--features", kind, "--output", str(tmp_path / kind)])
    with np.load(tmp_path / "rms/features.npz") as rms, np.load(tmp_path / "mu/features.npz") as mu:
        for split in ["train", "test"]:
            for field in ["starts", "stops", "y"]:
                np.testing.assert_array_equal(rms[f"{split}_{field}"], mu[f"{split}_{field}"])
            assert np.isfinite(mu[f"{split}_x"]).all()
            assert rms[f"{split}_x"].shape[1] == 60
        assert mu["train_x"].shape[1] == mu["test_x"].shape[1] > 0
    with np.load(tmp_path / "mu/templates.npz", allow_pickle=False) as template:
        assert template["projection_matrix"].shape[1] == 600
        assert template["projection_matrix"].shape[0] == template["centroids"].shape[0]
    meta = json.loads((tmp_path / "mu/metadata.json").read_text())
    assert sum(meta["pulse_counts"]["train"]) > 0
    assert sum(meta["pulse_counts"]["test"]) > 0
    assert meta["window_counts"] == {"train": 611, "test": 612}


@pytest.mark.parametrize("model", ["ridge", "gbt"])
def test_mllib_held_out_predictions_and_saved_model(tmp_path, model):
    pytest.importorskip("pyspark")
    from pyspark.ml.regression import GBTRegressionModel, LinearRegressionModel

    from mu_course.predict_force import main as predict

    directory = tmp_path / "features"
    directory.mkdir()
    x = np.linspace(-1, 1, 60)[:, None]
    z = np.linspace(-0.9, 0.9, 20)[:, None]
    np.savez(
        directory / "features.npz",
        train_x=x,
        test_x=z,
        train_y=3 * x[:, 0] + 2,
        test_y=3 * z[:, 0] + 2,
        test_starts=np.arange(20) * 10,
        test_stops=np.arange(20) * 10 + 20,
    )
    (directory / "metadata.json").write_text(
        json.dumps(
            {
                "feature_kind": "rms",
                "label_unit": "g",
                "offline": True,
                "fsamp": 100,
                "test_trial": "held-out",
                "trial_sha256": {"train": "one", "test": "two"},
            }
        )
    )
    output = tmp_path / model
    predict(
        [
            "--features",
            str(directory),
            "--model",
            model,
            "--master",
            "local[2]",
            "--output",
            str(output),
        ]
    )
    report = json.loads((output / "metrics.json").read_text())
    assert report["rmse"] < report["train_mean_baseline_rmse"]
    assert report["test_windows"] == 20
    assert (output / "prediction.png").is_file()
    # A persisted Spark model must load while a SparkContext is active.
    from pyspark.sql import SparkSession

    session = SparkSession.builder.master("local[2]").getOrCreate()
    try:
        cls = LinearRegressionModel if model == "ridge" else GBTRegressionModel
        restored = cls.load(str(output / "model"))
        assert restored.numFeatures == 1
    finally:
        session.stop()
