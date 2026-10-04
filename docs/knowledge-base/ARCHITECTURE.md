# Architecture

## Current components

| Path | Responsibility |
|---|---|
| `pyproject.toml`, `uv.lock` | Linux x86_64 Python dependency specification and resolution |
| `scripts/bootstrap_wsl.sh` | Project-local uv/Python installation and locked dependency sync |
| `scripts/environment_smoke.py` | Real CPU/CUDA tensor work and Spark partition/runtime checks |
| `environment/Dockerfile` | Shared non-root CPU runtime image |
| `environment/compose.yaml` | Isolated master, two workers and client-mode driver |
| `SETUP_PLAN.md` | Staged implementation, acceptance and public/private boundary |
| `src/mu_course/contracts.py`, `validation.py` | Shared matrix protocol and finite/shape checks |
| `src/mu_course/backends/` | Torch CPU/CUDA and cached Spark block operations |
| `src/mu_course/partitioning.py` | Contiguous time ranges and validated projection reassembly |
| `src/mu_course/routing.py`, `cli.py` | Explicit backend selection and doctor/kernel-check commands |
| `tests/` | Numerical reference, partition boundaries, lifecycle and CLI checks |

The installable `mu-decomp` CLI exposes doctor and kernel-check. No private CKC engine,
real-data run command or MU decomposition implementation is included.

## Intended computation

CLI → one private CKC engine → CPU / CUDA / Spark operations → consistent artifacts.
Spark partitions the time dimension. Workers calculate block contributions; driver performs
global aggregation, normalization, convergence and quality control. First version retains
the input and small global sequences on driver and makes no out-of-core scalability claim.

## Runtime boundary

WSL native environment for early checks. Compose uses a shared image and internal network.
Worker resources: one core each; application total two cores. Driver connects as `driver`.
Only loopback web UIs are published. Data is mounted read-only with separate output mounts.
Private core packaging and container injection are pending, not simulated by the smoke script.
