# Architecture

## Current components

| Path | Responsibility |
|---|---|
| `pyproject.toml`, `uv.lock` | Linux x86_64 Python dependency specification and resolution |
| `scripts/bootstrap_wsl.sh` | Project-local uv/Python installation and locked dependency sync |
| `scripts/environment_smoke.py` | Real CPU/CUDA tensor work and Spark partition/runtime checks |
| `environment/Dockerfile` | Shared non-root CPU runtime image |
| `environment/compose.yaml` | Isolated master, two workers and client-mode driver |
| `SETUP_PLAN.md` | Short sequence for running and comparing course experiments |
| `SPARK_MU_COURSE_PROJECT_PLAN.md` | Full current course plan, tracked in the private repository |
| `.private/` | Local instructions, validation scripts, source reference and archived original plan |
| `src/mu_course/spark_ops.py` | Three cached time-block operations with NumPy on workers |
| `src/mu_course/run_experiment.py` | Parameters, local algorithm import, input selection and results |
| `src/mu_course/benchmark.py` | Sequential warm-up/repeats and timing CSV |
| `src/mu_course/plot_results.py` | MU pulse raster and mean timing plot |
| `tests/` | Numerical block calculation and input interval checks |

The `mu-decomp` CLI runs the experiment. Private CKC code is provided through `--core-dir`,
not included in this repository. The previous doctor/kernel-check interface is removed.

## Computation

Experiment parameters → existing CKC main loop → Torch CPU/GPU or a few Spark operators
→ saved MU results and timings.

Keep the existing algorithm and use a local directory for its imports. A separate private
package is optional. CPU/GPU retain native Torch device tensors; Spark partitions the time
dimension and uses built-in caching, broadcast and reduction. Driver keeps global updates
and may retain the complete small input.

The four experiment files are implemented. The early protocol, factory, validation and
separate backend classes have been removed. Private `fit(eSig, ops=None)` has optional calls
at the three operation types; native CPU/GPU math remains the default.

MAT input requires Stage 3 `emg_clean` and selects good channels within one adapter. NPY/NPZ
inputs are already-preprocessed arrays. Results are parameter JSON and a numeric NPZ;
the same output directory is overwritten on rerun. Timing excludes reading and saving files.

## Runtime boundary

WSL native environment for early checks. Compose uses a shared image and internal network.
Worker resources: one core each; application total two cores. Driver connects as `driver`.
Only loopback web UIs are published. Data is mounted read-only with separate output mounts.
Run complete MU experiments in WSL/local first, then demonstrate two Docker workers.
GPU containers and identical container conditions across all comparisons are optional.
Local private algorithm integration and native WSL CUDA execution are verified. GPU math
uses the original Torch loop with highest float32 matmul precision and synchronized timing.
Device, runtime versions and peak allocated GPU memory are saved with each result.
Docker runtime remains pending; the smoke script continues to check the environment only.
