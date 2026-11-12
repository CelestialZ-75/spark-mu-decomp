# Decisions

## 2026-11-12 — Trial-level cascade comparison

Compare the same three-trial CKC cascade as sequential CPU and Spark trial-level tasks. A separate local process merges each level while the next level's Track A can proceed; Track B waits for the previous pool. The initial WSL local run simulates slots, not independent Spark worker nodes, and has no container limits. The third trial remains in ignored local outputs rather than the repository.

## 2026-10-25 — Bundle a standalone course example

Status: User explicitly authorized the two named real trials for the private course repository.

Bundle the five required CKC modules, with package-relative imports, and compact EMG/force
exports from processed_0831LDX/session0_1 and session0_2/aligned_Middle_Trapezoid.mat.
Keep full time trajectories, shared good channels in adapter 0 and aligned force; record
source hashes and units. Include these assets in the Python wheel and Docker build context.
Default commands use bundled assets; --core-dir and --data remain optional overrides.

This supersedes the earlier requirement to keep all research code/data outside this repo.
It is a narrow exception: do not copy the research repository wholesale or add other trials.
It is not authorization to publish publicly, commit, push or distribute a public image.
Prediction training and resource-quota experiments remain separate unfinished work.

## 2026-10-20 — Resource-constrained simulation and CPU prediction

Status: User premise recorded; documentation reorganized. Runtime additions pending.

Use one physical machine to simulate resource-constrained Spark nodes. Small real EMG trials
serve this experiment; dataset size alone is not a reason to abandon Spark. Measure resource
allocation, task partitioning and communication. Do not claim validated multi-server or
out-of-core capability: the MU driver still retains the complete input.

ML is CPU-only, with LDX 0° one training and one test trial. GPU remains an existing MU
optimization reference. The working plan is RMS + MLlib Ridge and MU + MLlib GBT;
MU + Ridge is a suggested control. Transformer and prediction intervals are optional.
Exact trial pairing, resource quotas and optional extensions remain unconfirmed.

Docker changes from a final demonstration to the planned resource-experiment environment.
Add actual CPU/memory limits and a comparable CPU baseline; current Compose has neither.
Compare added per-node resources separately from fixed worker-budget redistribution, and
account for driver/master resources. This supersedes the optional-container-baseline
treatment in the older entries below.

Keep simple functions, centralized parameters and explicit input/output semantics.
Conventions and team work packages live in ../DEVELOPMENT.md. No new framework, prediction
source, quota configuration or runtime verification was added in this documentation task.

## 2026-10-10 — Track all course code and documents in the private repository

Status: User authorized committing and pushing.

The user changed the GitHub repository to private and requested that all code and documents
in this checkout be versioned. Include the root plan, local instructions, source-reference
metadata, inspection/validation scripts and the historical long-plan backup. This replaces
the earlier local-only treatment of these files. The backup remains a historical reference.

Credentials, installed environments, caches, real data and generated results remain excluded.
The external research repository is not copied or committed here. Partial open source is a
later decision, with a separate review of the selected files and history.

## 2026-10-05 — Validate native CUDA with the existing loop

Status: Implemented and verified on the actual GPU.

Use the original Torch GPU operations through `--backend gpu`, without another backend class
or custom CUDA kernels. This stage uses highest float32 matmul precision for the CPU comparison.
Check actual tensor devices, synchronize timing and save device/runtime/peak allocated memory.
The existing CPU peak-detection and quality helpers are retained.

Nine tests passed and the same small real input produced matching CPU/GPU MU pulses across
three recorded repetitions. GPU containers and Spark GPU scheduling remain outside this stage.

## 2026-10-05 — Keep this a small course experiment

Status: User approved; four experiment modules implemented.

Preserve the original MU main loop, use Torch device selection for CPU/GPU, and implement
only the needed Spark block operators with PySpark. Keep parameters together in one entry
point or a simple configuration file. Save results and timings without an artifact framework.

A private wheel, generic backend protocol, strict image freezing and performance tracing
are not prerequisites. Run the original algorithm on CPU first, then GPU, Spark local and
the two-worker Docker demonstration. Numerical correctness and fair timing remain necessary.

This decision replaces the earlier architecture requirements below. The original long plan
was initially an ignored local backup; the 2026-10-10 decision brought it into version control.
A later implementation removed the generic layers
and added optional Spark operation calls to the local private fit method. CPU/GPU keep native
Torch operations; Spark workers use NumPy and return CPU tensors to the driver.

## 2026-10-05 — Public generic matrix backend contract

Status: Historical implementation; no longer the required integration interface.

The early implementation put correlation, projection and weighted-sum operations behind
a protocol and returned host NumPy arrays. It does not implement CKC control logic.
Spark cached time blocks and combined their contributions. The protocol and NumPy-returning
GPU interface were subsequently removed. The original Torch CPU/GPU loop now keeps its
iteration tensors on the selected device.

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
