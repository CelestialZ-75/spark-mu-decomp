import json
import numpy as np
from mu_course.run_experiment import main
from mu_course.plot_results import main as plot
from mu_course.benchmark import main as benchmark
common = ['--core-dir', '/mnt/d/meta-robotics/Code/FingerForce_Decode/example/gpu_MU_extract', '--data', '/mnt/d/meta-robotics/Code/FingerForce_Decode/data/processed_0831LDX/session0_1/aligned_Middle_Trapezoid.mat', '--start', '3', '--duration', '2', '--n-mu-max', '5', '--n-iter', '30']
main(common + ['--backend', 'cpu', '--output', 'outputs/minimal/cpu'])
main(common + ['--backend', 'spark', '--partitions', '2', '--output', 'outputs/minimal/spark'])
plot(['outputs/minimal/cpu'])
plot(['outputs/minimal/spark'])
with np.load('outputs/minimal/cpu/mu_results.npz') as a, np.load('outputs/minimal/spark/mu_results.npz') as b:
    def trains(data):
        return [data['pulses'][a:b] for a,b in zip(data['offsets'][:-1], data['offsets'][1:])]
    cpu, spark = trains(a), trains(b)
    agreement = [[2 * sum(np.any(np.abs(y - t) <= 1) for t in x) / (len(x) + len(y)) for y in spark] for x in cpu]
    result = {'cpu_mu_count': len(cpu), 'spark_mu_count': len(spark), 'pairwise_pulse_agreement_1_sample': agreement}
    print('RESULT_COMPARISON', json.dumps(result))
    with open('outputs/minimal/result_comparison.json', 'w') as f:
        json.dump(result, f, indent=2)
benchmark(common + ['--backends', 'cpu', '--repeat', '1', '--output', 'outputs/minimal/benchmark-check'])
plot(['outputs/minimal/benchmark-check'])
print('INTEGRATION_AND_PLOTS_COMPLETE')
