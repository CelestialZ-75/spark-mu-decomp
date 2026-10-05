# Decisions

## 2026-10-05 — Keep this a small course experiment

Status: User approved; four experiment modules implemented.

Preserve the original MU main loop, use Torch device selection for CPU/GPU, and implement
only the needed Spark block operators with PySpark. Keep parameters together in one entry
point or a simple configuration file. Save results and timings without an artifact framework.

A private wheel, generic backend protocol, strict image freezing and performance tracing
are not prerequisites. Run the original algorithm on CPU first, then GPU, Spark local and
the two-worker Docker demonstration. Numerical correctness and fair timing remain necessary.

This decision replaces the earlier architecture requirements below. The original long plan
is retained only as an ignored local backup. A later implementation removed the generic layers
and added optional Spark operation calls to the local private fit method. CPU/GPU keep native
Torch operations; Spark workers use NumPy and return CPU tensors to the driver.

## 2026-10-05 — Public generic matrix backend contract

Status: Historical implementation; no longer the required integration interface.

The early implementation put correlation, projection and weighted-sum operations behind
a protocol and returned host NumPy arrays. It does not implement CKC control logic.
Spark caches time blocks and combines their contributions. These remain current code facts,
but the interface may be removed during simplification. In particular, NumPy returns should
not force GPU-to-CPU transfers at every step of the original Torch algorithm.

## 2026-10-04 — Staged WSL and Compose environment

Status: Environment retained; strict container comparison requirements superseded above.

Use existing WSL2 Ubuntu 24.04. Establish public dependencies and native local checks,
then use Compose with master, two workers and a client-mode driver for demonstration.
Native and container conditions should be stated in the report; an additional container
CPU baseline is optional if it helps explain measured differences.

## 2026-10-04 — Separate environment from algorithm acceptance

Status: Accepted.

Use synthetic tensor work for infrastructure checks. A passing smoke does not imply MU
decomposition, private package availability, kernel equivalence or physiological validity.
PyTorch 2.8.0 CPU/CUDA 12.8 is the initial dependency choice. Backend comparisons should
use the same algorithm implementation, input and parameters.

## 2026-10-04 — Protect the existing source boundary

Status: Private/public separation retained; mandatory packaging superseded above.

The supplied research example is not an installable core package. Keep its source private;
use local import adaptation or installation as needed for the experiment. A separate package
can be added later if it actually simplifies running the code.
No new remote, commit or push is inferred from generating environment files.
