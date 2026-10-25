"""Check the clone-and-run path without an external algorithm or data directory."""

import importlib
from pathlib import Path

import numpy as np

from mu_course.dataset import load_trial
from mu_course.run_experiment import experiment_parser, run_experiment


def test_bundled_trials_have_aligned_force_and_fixed_channels():
    directory = experiment_parser().parse_args([]).data.parent
    loaded = load_trial(directory / "ldx_0deg_1.npz")
    assert loaded["force"].shape[1] == loaded["emg"].shape[1]
    with (
        np.load(directory / "ldx_0deg_1.npz") as first,
        np.load(directory / "ldx_0deg_2.npz") as second,
    ):
        np.testing.assert_array_equal(first["channel_ids"], second["channel_ids"])
        for trial in (first, second):
            assert trial["emg"].dtype == np.float32
            assert trial["force"].shape == (2, trial["emg"].shape[1])
            assert trial["fsamp"].item() == 2048
            assert trial["force_unit"].item() == "g"
            assert np.isfinite(trial["emg"]).all()
            assert np.isfinite(trial["force"]).all()
            assert np.ptp(trial["force"][0]) > 0


def test_bundled_core_imports_and_template_path():
    for name in ("ckc_decomposer", "ckc_extender", "ckc_preprocess", "mu_feature", "mu_quality"):
        module = importlib.import_module(f"mu_course.ckc.{name}")
        assert Path(module.__file__).parent.name == "ckc"
    from mu_course.ckc.ckc_decomposer import apply_templates

    result = apply_templates(np.empty((0, 10)), None, np.zeros((1, 100)), device="cpu")
    assert result["n_mus"] == 0


def test_default_real_cpu_experiment(tmp_path):
    args = experiment_parser().parse_args(["--output", str(tmp_path)])
    assert args.core_dir is None
    report = run_experiment(args)
    assert report["device"] == "cpu"
    assert report["mu_count"] > 0
    assert (tmp_path / "mu_results.npz").is_file()
    assert "FingerForce_Decode" not in report["core_dir"]
