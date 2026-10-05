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
| `src/mu_course/contracts.py`, `validation.py` | Shared matrix protocol and finite/shape checks |
| `src/mu_course/backends/` | Torch CPU/CUDA and cached Spark block operations |
| `src/mu_course/partitioning.py` | Contiguous time ranges and validated projection reassembly |
| `src/mu_course/routing.py`, `cli.py` | Explicit backend selection and doctor/kernel-check commands |
| `tests/` | Numerical reference, partition boundaries, lifecycle and CLI checks |

The installable `mu-decomp` CLI exposes doctor and kernel-check. No private CKC engine,
real-data run command or MU decomposition implementation is included.

## Planned simplification

Experiment parameters → existing CKC main loop → Torch CPU/GPU or a few Spark operators
→ saved MU results and timings.

Keep the existing algorithm and use a small local adapter for its imports. A separate private
package is optional. CPU/GPU retain native Torch device tensors; Spark partitions the time
dimension and uses built-in caching, broadcast and reduction. Driver keeps global updates
and may retain the complete small input.

Proposed files are `spark_ops.py`, `run_experiment.py`, `benchmark.py` and `plot_results.py`.
They are not implemented yet. Existing protocol, factory and validation layers are early
matrix-check code, not requirements for the final experiment.

## Runtime boundary

WSL native environment for early checks. Compose uses a shared image and internal network.
Worker resources: one core each; application total two cores. Driver connects as `driver`.
Only loopback web UIs are published. Data is mounted read-only with separate output mounts.
Run complete MU experiments in WSL/local first, then demonstrate two Docker workers.
GPU containers and identical container conditions across all comparisons are optional.
Private algorithm integration is pending; the smoke script does not simulate it.
