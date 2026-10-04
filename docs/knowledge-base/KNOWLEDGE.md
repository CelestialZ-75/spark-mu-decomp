# Project knowledge

## Conventions

- Repository instructions live in `AGENTS.md`; this directory stores durable facts.
- Python 3.11, uv lock, explicit CPU/CUDA selection; no automatic GPU fallback.
- Preserve the original course plan; use `SETUP_PLAN.md` for implementation stages.
- Public repository holds orchestration, not private core sources or real data.

## Verified environment

- Inspection: 2026-10-04, Ubuntu 24.04.3 LTS on WSL2, x86_64.
- WSL reports 28 logical CPUs, approximately 15 GiB RAM and 4 GiB swap.
- NVIDIA RTX 4070 Laptop GPU, 8188 MiB VRAM, driver 591.86 visible in WSL.
- Initial Ubuntu Python is 3.12.3. Java, uv and Docker initially absent.
- Docker Desktop not found at its default Windows installation path.
- WSL warns about a localhost proxy under NAT; direct PyPI/GitHub HTTPS probes succeeded.
- Initial `sudo` required an interactive password; OpenJDK 17 was subsequently installed
  through the WSL administrative entry point for the requested environment setup.

## Current status

- Added public dependency specification, bootstrap, Docker configuration and smoke script.
- Environment smoke tests synthetic arithmetic, not CKC or scientific output.
- User-approved knowledge directory: `docs/knowledge-base/`.
- User identified an existing research MU example as the source; it is not an independently
  packaged private core. Auxiliary modules still contain research-project imports.
- Local Git initialized on `main`; `origin` points to the user-provided GitHub repository.
  SSH read access succeeded and the remote was empty before initial publication.
- Private source file hashes and source commit recorded in ignored `.private/source_reference.json`.
- Dependency resolution succeeded: 48 packages including optional variants; 30 CPU/development
  packages installed in `.venv-wsl` using Python 3.11.15. Java reports 17.0.20.1.
- User authorized publishing environment files; original full plan remains ignored locally.
- Initial environment commit `a2e3896` was pushed; `main` tracks `origin/main`.
- CPU tensor calculation and Spark `local[2]` smoke passed; two distinct Python worker processes
  executed the two partitions with matching Python/Torch versions. This is not two-host acceptance.
- Ruff, Bash/Python syntax and TOML/YAML parsing passed. Docker runtime and CUDA tests remain pending.
- Local check artifacts are in ignored `outputs/environment/`.

## Architecture implementation

- Installable course package 0.1.0 exposes `mu-decomp doctor` and `kernel-check`.
- CPU/CUDA share TorchBackend. SparkBackend caches time blocks, broadcasts operands,
  reduces contributions globally and validates ordered projections and worker versions.
- Generic matrix math is public; private CKC engine, data preparation and MU output are pending.
- Main suite: 23 passed, 1 actual-CUDA test skipped (CPU Torch installed). Three additional
  zero-input/overflow tests passed. CPU/Spark numerical operations cover uneven partitions,
  float32/64, rank-deficient input, shape/finite checks and resource lifecycle.
- Public wheel build passed. Dockerfile installs the same course package without editable mode.
- Installed CPU and Spark four-partition CLI checks passed; maximum absolute error against
  NumPy was about 1.8e-15 on the deterministic float64 check. This is not a performance result.
- Final CLI tests also passed (including invalid Spark thread settings and no-overwrite output).
  Across the main and targeted runs, 27 distinct tests passed and 1 real-CUDA test was skipped.
  Ruff formatting/lint and Git whitespace checks passed. CUDA and Docker runtime remain unverified.

## Next steps

- Install Docker Desktop through the documented path; validate the Compose runtime.
- Validate CUDA and two-worker execution; freeze actual runtime/image versions.
- Package private core separately with provenance and reference regression, then connect its
  controller to the implemented matrix protocol; do not copy private policies into public backends.
- Freeze one approved plateau input and algorithm policy before backend comparisons.
