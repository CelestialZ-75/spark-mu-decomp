import json
from pathlib import Path
import numpy as np
import pytest
import torch
from scipy.optimize import linear_sum_assignment
from mu_course.benchmark import main as benchmark
from mu_course.plot_results import main as plot

print('CUDA_RUNTIME', torch.__version__, torch.version.cuda, torch.cuda.is_available(), flush=True)
assert torch.cuda.is_available()
print('GPU', torch.cuda.get_device_name(), flush=True)
status = pytest.main(['-q', 'tests'])
if status:
    raise SystemExit(status)
common = ['--core-dir', '/mnt/d/meta-robotics/Code/FingerForce_Decode/example/gpu_MU_extract', '--data', '/mnt/d/meta-robotics/Code/FingerForce_Decode/data/processed_0831LDX/session0_1/aligned_Middle_Trapezoid.mat', '--start', '3', '--duration', '2', '--n-mu-max', '5', '--n-iter', '30']
benchmark(common + ['--backends', 'cpu', 'gpu', '--repeat', '3', '--output', 'outputs/cuda-stage/comparison'])
root = Path('outputs/cuda-stage/comparison')
plot([str(root)])
for backend in ['cpu', 'gpu']:
    plot([str(root / f'{backend}-p1/run-1')])

def pulse_trains(path):
    with np.load(path) as data:
        return [data['pulses'][a:b] for a,b in zip(data['offsets'][:-1], data['offsets'][1:])]
comparisons = []
for run in range(1, 4):
    cpu = pulse_trains(root / f'cpu-p1/run-{run}/mu_results.npz')
    gpu = pulse_trains(root / f'gpu-p1/run-{run}/mu_results.npz')
    scores = np.array([[2 * sum(np.any(np.abs(y - t) <= 1) for t in x) / (len(x) + len(y)) for y in gpu] for x in cpu])
    indices = linear_sum_assignment(-scores) if scores.size else ([], [])
    row = {'run': run, 'cpu_mu_count': len(cpu), 'gpu_mu_count': len(gpu), 'matched_pulse_agreement_1_sample': scores[indices].tolist() if scores.size else []}
    comparisons.append(row)
    print('CPU_GPU_COMPARISON', json.dumps(row), flush=True)
(root / 'result_comparison.json').write_text(json.dumps(comparisons, indent=2))
print('CUDA_STAGE_COMPLETE', flush=True)
