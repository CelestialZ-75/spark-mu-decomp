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
- CPU tensor calculation and Spark `local[2]` smoke passed; two distinct Python worker processes
  executed the two partitions with matching Python/Torch versions. This is not two-host acceptance.
- Ruff, Bash/Python syntax and TOML/YAML parsing passed. Docker runtime and CUDA tests remain pending.
- Local check artifacts are in ignored `outputs/environment/`.

## Next steps

- Install Docker Desktop through the documented path; validate the Compose runtime.
- Validate CUDA and two-worker execution; freeze actual runtime/image versions.
- Package private core separately with provenance and regression before implementing CKC routing.
- Freeze one approved plateau input and algorithm policy before backend comparisons.
