#!/usr/bin/env bash
set -euo pipefail

batch=${1:?usage: run_cascade_comparison.sh BATCH_DIRECTORY}
python=${PYTHON:-.venv-wsl/bin/python}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
trials=("$batch/inputs/trial1.npz" "$batch/inputs/trial2.npz" "$batch/inputs/trial3.npz")

printf 'cpu\n' > "$batch/status.txt"
if ! "$python" -m mu_course.cascade_experiment --mode cpu --trials "${trials[@]}" --output "$batch/cpu" > "$batch/cpu.log" 2>&1; then
    printf 'cpu_failed\n' > "$batch/status.txt"
    exit 1
fi
printf 'spark\n' > "$batch/status.txt"
if ! "$python" -m mu_course.cascade_experiment --mode spark --trials "${trials[@]}" --output "$batch/spark" > "$batch/spark.log" 2>&1; then
    printf 'spark_failed\n' > "$batch/status.txt"
    exit 1
fi
printf 'complete\n' > "$batch/status.txt"
