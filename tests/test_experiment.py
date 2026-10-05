import numpy as np
import pytest

from mu_course.run_experiment import load_emg


def test_input_selection_and_time_bounds(tmp_path):
    path = tmp_path / "emg.npy"
    matrix = np.arange(128 * 100).reshape(128, 100)
    np.save(path, matrix)
    selected = load_emg(path, fsamp=10, start=2, duration=3, adapter=1)
    np.testing.assert_array_equal(selected, matrix[64:128, 20:50])
    assert selected.dtype == np.float32
    with pytest.raises(ValueError, match="outside"):
        load_emg(path, fsamp=10, start=9, duration=2, adapter=0)
