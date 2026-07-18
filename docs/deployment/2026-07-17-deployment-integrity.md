# Deployment integrity hardening

## What changed

TicketTune's deployment planners now keep immutable model identity, network
policy, subprocess execution, and proof claims aligned:

- vLLM argv and Compose both pass the configured 40-character model revision;
- native vLLM execution is plan-only by default and requires explicit
  `--execute`, with offline Hugging Face/Transformers flags unless
  `--allow-download` is deliberately supplied;
- adapter base identity, Safetensors hashes, and an adapter-declared revision
  (when present) are validated before a vLLM plan is accepted;
- safe-merge planning and execution both enforce that same adapter revision,
  closing the review-to-execution change window;
- Ollama and native vLLM commands emit a pre-execution plan first, then emit a
  final success boundary only after every requested subprocess returns cleanly;
- JSON execution sends the pre-execution plan and child-process output to
  stderr, reserving stdout for one final parseable result.
- generated Ollama Modelfiles and the HTTP chat example share the canonical
  training/evaluation system prompt, and the chat request sends it explicitly.
- the src-layout chat example is documented through `uv run python`, and its
  interpreter/import help path is contract-tested.
- GGUF planning requires an intact safe-merge provenance inventory, verifies
  every source hash, and rejects untracked source files;
- successful conversion atomically records a durable manifest linking verified
  merged inputs and the pinned converter to F16, final GGUF, and Modelfile
  hashes;
- health/chat redirect targets are revalidated, and chat output is schema-
  checked before it can be printed.

## Why

The earlier planners could omit the configured revision from vLLM, leave model
downloads implicit, and describe execution as successful before the underlying
process had returned. Those gaps made a dry plan look stronger than its actual
proof boundary and could corrupt `--json` stdout with child-process logs.

## Affected surfaces

- `src/tickettune/export.py`
- `src/tickettune/cli.py`
- `deploy/vllm/compose.yaml`
- `deploy/vllm/.env.example`
- `deploy/examples/chat.py`
- `deploy/examples/health.py`
- `docs/architecture.md`
- `docs/deployment.md`
- `tests/test_export.py`
- `tests/test_cli.py`
- `tests/fixtures/adapter/adapter_config.json`
- `tests/fixtures/merged/tickettune-merge-provenance.json`

## Validation

Local, offline validation on the working tree:

- `pytest -q tests/test_export.py tests/test_cli.py`: 93 passed;
- focused Ruff check: passed;
- focused mypy check for the changed source modules and chat example: passed;
- Bandit scan across `src`: passed.

These checks prove plan construction, revision/offline gates, CLI result timing,
JSON stream isolation, provenance verification, redirect enforcement, output-
schema checks, and rejection behavior. They do not prove a running Linux/CUDA
vLLM container, real GGUF conversion, Ollama import, health, task quality, load
behavior, or production readiness. The Compose image now uses the reviewed
`v0.24.0` tag and the Docker Hub `linux/amd64` manifest digest read back on
2026-07-17. A production release should still mirror that digest into a
controlled registry.

## Branch and follow-up state

The project is on local branch `main` in a newly initialized repository, with
the broader project still uncommitted while parallel build work is being
integrated. No commit SHA or remote deployment is claimed here. The final
project closeout should run the full repository suite, record the resulting
commit, and keep CUDA vLLM and Ollama runtime proof as explicit environment-
specific follow-up evidence.
