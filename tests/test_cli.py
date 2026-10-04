import json

import pytest

from mu_course.cli import main


def test_cpu_cli_result_and_no_overwrite(tmp_path, capsys):
    path = tmp_path / "check.json"
    args = ["kernel-check", "--backend", "cpu", "--output", str(path)]
    assert main(args) == 0
    result = json.loads(path.read_text())
    assert result["scope"] == "linear_algebra_only"
    assert result["ckc_engine"] == "not_integrated"
    assert max(result["max_absolute_error"].values()) < 1e-10
    first = path.read_bytes()
    assert main(args) == 4
    assert path.read_bytes() == first
    capsys.readouterr()


def test_spark_threads_cannot_be_silently_ignored():
    with pytest.raises(SystemExit) as error:
        main(["doctor", "--backend", "spark", "--threads", "2"])
    assert error.value.code == 2
