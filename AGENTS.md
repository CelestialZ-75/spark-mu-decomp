# Repository workflow

- Project knowledge lives in `docs/knowledge-base/`. Read `KNOWLEDGE.md`, then
  `ARCHITECTURE.md`, then relevant sections of `DECISIONS.md`.
- Follow the user's shared operating rules and applicable skills.
- Python uses PEP 8, snake_case functions, four spaces, and explicit device selection.
- Use Python 3.11 and uv; CPU and CUDA extras are mutually exclusive.
- The repository is currently private. Track project code, plans, local experiment
  notes and validation scripts. Keep credentials, environments, caches, other real data,
  wheels and experiment outputs out of Git. Do not copy the research repository
  wholesale. Review tracked content separately before any future public release.
- The user authorized two compact real EMG/force trials in `src/mu_course/data/`
  and the required CKC modules in `src/mu_course/ckc/` for standalone team use.
  These are the narrow exception to excluding research data/source; do not add
  other trials or copy the research repository wholesale.
- Use the revised course plan as the implementation guide; keep the old plan only
  as a local reference. Prefer a few experiment functions and simple parameters.
  Record actual progress separately; environment smoke checks do not run MU decomposition.
- Do not commit or push without user authorization. Never silently replace a remote.
- Prefer simple, readable experiment code that runs from a short command. Keep
  parameters together; document shapes, units, time bases and output fields.
  Reuse the original algorithm and Spark/MLlib rather than building a framework.
- Resource-constrained single-host cluster simulation is the experiment premise.
  ML is CPU-only; GPU is a MU reference only. Distinguish implemented, historically
  verified and planned work. Development conventions live in `docs/DEVELOPMENT.md`.
