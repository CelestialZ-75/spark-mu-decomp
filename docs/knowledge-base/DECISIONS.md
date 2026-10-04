# Decisions

## 2026-10-04 — Staged WSL and Compose environment

Status: Accepted as preparation route; runtime acceptance remains pending.

Use existing WSL2 Ubuntu 24.04. Establish public dependencies and native local checks,
then use Compose as the maintained Standalone packaging with master, two workers and a
client-mode driver. Avoid maintaining another permanent process-launch cluster variant.
Native and container benchmark conditions must be explicit; add same-container CPU baseline.

## 2026-10-04 — Separate environment from algorithm acceptance

Status: Accepted.

Use synthetic tensor work for infrastructure checks. A passing smoke does not imply MU
decomposition, private package availability, kernel equivalence or physiological validity.
PyTorch 2.8.0 CPU/CUDA 12.8 is an initial reproducible dependency choice, subject to source
regression. Formal experiments require one frozen core and algorithm policy across backends.

## 2026-10-04 — Protect the existing source boundary

Status: Accepted.

The supplied research example is a source snapshot candidate, not an installable core package.
Inspect it read-only. Package the private core separately in the next stage; do not copy it
into the public course project or introduce runtime imports from the research repository.
No new remote, commit or push is inferred from generating environment files.
