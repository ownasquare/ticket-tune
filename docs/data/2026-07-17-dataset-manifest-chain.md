# Dataset Manifest Chain-of-Custody Audit — 2026-07-17

## What changed

TicketTune now fails closed before optional model libraries load unless prepared data remains tied to
its exact source and `manifest.json`. The lightweight verifier checks the strict manifest schema,
source filename and SHA-256, configured seed and split fractions, canonical split filenames and
path safety, manifest counts and globally unique ordered IDs, and exact hashes/counts/IDs for each
requested split.

Real training verifies train and validation. Live generation, including model evaluation, verifies
and binds to the canonical test split. Training run provenance now includes source, dataset
manifest, train, and validation digests.

Adapter generation also validates `adapter_config.json` base identity and any declared revision
before optional imports, requires regular Safetensors weights, and records adapter config/weight
digests. Generated rows carry dataset-manifest, model-revision, and adapter identities; evaluation
rejects conflicting row provenance and promotes consistent provenance into JSON and Markdown
reports.

## Why

Split files, a manifest, a base model, or an adapter can otherwise be edited or substituted after
preparation, producing a training or evaluation artifact whose lineage cannot be reproduced. These
checks make that mismatch an early error instead of an unqualified model-quality claim.

## Affected files

- `src/tickettune/data.py`
- `src/tickettune/training.py`
- `src/tickettune/generation.py`
- `src/tickettune/evaluation.py`
- `tests/test_data.py`
- `tests/test_training.py`
- `tests/test_evaluation.py`
- `docs/training.md`
- `docs/evaluation.md`
- `docs/superpowers/plans/2026-07-17-dataset-manifest-chain.md`

## Branch and commit state

Work was completed in the new standalone TicketTune repository on branch `main`. The repository is
still an uncommitted initial worktree; this bounded lane intentionally did not commit because the
root agent owns final integration and commit scope.

## Validation

- `uv run pytest tests/test_data.py tests/test_training.py tests/test_evaluation.py -q` — 66 passed.
- `uv run pytest -q` — 217 passed.
- Focused `uv run ruff check` for the four source and three test files — passed.
- Focused `uv run ruff format --check` for the scoped files — passed.
- `uv run ruff check .` — passed.
- `uv run mypy src` — passed for 13 source files.
- `git diff --check` — passed.

The repository-wide format check separately reports that `src/tickettune/hardware.py` and
`src/tickettune/run_manifest.py` would be reformatted. Those files were outside this lane and were
not changed here; all scoped files pass the format check.

## Proof boundaries and follow-up

The tests use local prepared fixtures and injected fake model libraries. They prove verification
ordering, exact-byte/ID rejection, provenance propagation, report consistency, and lazy optional
imports without network or model execution. They do not prove a real optimizer step, GPU runtime,
adapter quality improvement, vLLM serving, Ollama import, or hosted deployment. The root agent
retains ownership of final runtime proof, repository-wide formatting decisions, commit, and handoff.
