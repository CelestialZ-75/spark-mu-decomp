import numpy as np
import pytest
import torch

from mu_course.run_experiment import experiment_parser, load_emg, run_experiment


def test_input_selection_and_time_bounds(tmp_path):
    path = tmp_path / "emg.npy"
    matrix = np.arange(128 * 100).reshape(128, 100)
    np.save(path, matrix)
    selected = load_emg(path, fsamp=10, start=2, duration=3, adapter=1)
    np.testing.assert_array_equal(selected, matrix[64:128, 20:50])
    assert selected.dtype == np.float32
    with pytest.raises(ValueError, match="outside"):
        load_emg(path, fsamp=10, start=9, duration=2, adapter=0)


def test_gpu_unavailable_does_not_fall_back(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    args = experiment_parser().parse_args(
        ["--core-dir", str(tmp_path), "--data", str(tmp_path / "emg.npy"), "--backend", "gpu"]
    )
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        run_experiment(args)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="A real CUDA device is required")
def test_cuda_matrix_operations():
    torch.set_float32_matmul_precision("highest")
    generator = torch.Generator().manual_seed(13)
    matrix = torch.randn(5, 17, generator=generator)
    vector = torch.randn(5, generator=generator)
    weights = torch.randn(17, generator=generator)
    gpu_matrix, gpu_vector, gpu_weights = [array.cuda() for array in (matrix, vector, weights)]
    results = [gpu_matrix @ gpu_matrix.T / 17, gpu_vector @ gpu_matrix, gpu_matrix @ gpu_weights]
    torch.cuda.synchronize()
    for actual, expected in zip(
        results, [matrix @ matrix.T / 17, vector @ matrix, matrix @ weights]
    ):
        assert actual.device.type == "cuda"
        torch.testing.assert_close(actual.cpu(), expected, rtol=1e-5, atol=1e-5)
