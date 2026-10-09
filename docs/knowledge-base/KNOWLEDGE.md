# Project knowledge

## Conventions

- Repository instructions live in `AGENTS.md`; this directory stores durable facts.
- Python 3.11, uv lock, explicit CPU/CUDA selection; no automatic GPU fallback.
- Follow the simplified local course plan and public `SETUP_PLAN.md`.
- Prefer a small experiment wrapper, a few Spark operators and easy parameter changes.
- Repository is private. Track course code, full plans, local experiment notes and
  validation scripts. Keep external algorithm sources, real data, outputs and credentials local.

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
- Initial Ruff, Bash/Python syntax and TOML/YAML parsing passed. CUDA was subsequently
  verified in the stage below; Docker runtime remains pending.
- Local check artifacts are in ignored `outputs/environment/`.

## Earlier matrix implementation (historical)

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

## Current experiment implementation

- Four modules implement Spark operators, experiment parameters/input/results, sequential
  timing repetitions and plots. Protocol, routing, validation and backend classes were removed.
- Native CPU/GPU call the existing algorithm directly. The local private fit method has
  optional Spark operation calls (15 added / 6 removed lines); the private source is not copied
  or committed here. Workers need only public NumPy operations.
- MAT input uses Stage 3 `emg_clean` and good channels in one adapter, without filtering again.
  The inspected older recording lacked this field; a newer locally available Trapezoid trial
  was selected instead. Exact source/data paths and commands are in `.private/LOCAL_EXPERIMENT.md`.
- Confirmed parameters from source knowledge/config: 2048 Hz, extension 10, cost function 1,
  SIL 0.8, sync-time plateaus [3,9), [13,19), [23,29) seconds.
- A two-second, 63-channel CPU run produced four MUs after deduplication with five candidates
  and 30 iterations. This is an integration check, not a formal speed or physiological claim.
- Spark local[2] with the same data and parameters also produced four MUs. The four corresponding
  pulse trains agreed fully within one sample. Saved JSON/NPZ and both raster plots were checked.
- CPU benchmark warm-up plus one recorded repeat completed; CSV and the mean timing plot worked.
  The formal three-repeat CPU/GPU/Spark comparison has not been run.
- Nine current tests passed: float32/64 block math with 1/2/4 partitions, input slicing,
  unavailable-GPU rejection and actual CUDA matrix operations. No CUDA test was skipped.
  Ruff passed; offline locked sync rebuilt the editable package and the new CLI help worked.
- Parameters JSON and numeric NPZ are overwritten if the same output directory is reused.
  Total compute timing includes extension and Spark setup, excludes file I/O and cleanup.

## Next steps

- Demonstrate two Docker workers, mounting the private algorithm only on driver.
- Run the three-backend comparison with warm-up and three repetitions; explain MU and timing differences.

## 2026-10-10 — Private repository content

- User confirmed that the remote repository is private and authorized committing/pushing all
  course code and documents. Anonymous GitHub API lookup returned 404; SSH access succeeded.
- Track the current root plan and `.private/` notes, scripts, source-reference snapshot and
  historical plan backup. The backup is not the active implementation checklist.
- Keep credentials, installed runtimes/environments, caches, real data and generated outputs
  excluded. External FingerForce_Decode changes are outside this repository's commit scope.
- Static syntax checks passed for all 15 project/local Python files; current document links,
  code fences and Git whitespace checks passed. No computation logic changed in this task.

## 2026-10-05 — Documentation simplification

- Rewrote the local root plan and aligned public docs with a small course experiment.
- Removed private packaging, strict version/image freezing, complex artifacts and generic
  backend architecture as prerequisites. The earlier long plan is backed up in ignored `.private/`.
- This session changed documentation only. Existing source and previous check results remain
  as recorded above; full MU integration, real CUDA and Docker execution are still pending.

## 2026-10-05 — Minimal experiment implementation

- Documentation baseline committed as `44d81b1`; the root full plan remains ignored locally.
- Removed generic layers; implemented four experiment modules and small private operator hooks.
- Seven tests passed, real CPU/Spark decomposition matched within one sample, and output/plot/
  repeat checks passed. CUDA and Docker remain pending. Local artifacts are under `outputs/minimal/`.

## 2026-10-05 — Native CUDA stage

- Project environment switched to Torch 2.8.0+cu128 with CUDA 12.8 using the existing locked
  gpu extra. WSL confirmed RTX 4070 Laptop GPU. No system CUDA driver/toolkit was installed.
- Reused the original GPU algorithm; added actual-device checks, highest float32 matmul
  precision and report fields for device/runtime/peak allocated GPU memory. Existing CUDA
  synchronization remains at timing boundaries. CPU-based peak/quality helpers remain as before.
- All nine tests passed. CPU/GPU each warmed up once, then ran the same two-second, 63-channel
  input three times with extension 10, five candidates, 30 iterations, cost 1 and SIL 0.8.
- All runs yielded four MUs. One-to-one pulse matching agreed fully within one sample in
  each paired run. Mean total compute time: CPU 0.887 s, GPU 0.595 s (about 1.49x).
  GPU peak allocated memory was about 40.7 MiB. These numbers apply only to this local small sample.
- Saved timing CSV, result comparison and inspected timing/raster plots in ignored
  `outputs/cuda-stage/comparison/`. Full three-backend repeated comparison and Docker remain pending.
