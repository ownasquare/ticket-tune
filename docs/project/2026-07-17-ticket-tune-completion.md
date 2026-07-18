# TicketTune project completion record

Date: 2026-07-17 (America/Los_Angeles; execution continued into 2026-07-18 UTC)

## Outcome

TicketTune is a complete, installable Python project for adapting an instruction model to
structured customer-support triage. It includes a privacy-safe task dataset, deterministic data
provenance, PEFT LoRA and CUDA-gated QLoRA training through TRL, immutable runs, held-out
candidate-versus-baseline evaluation, safe model merge, vLLM adapter-serving plans, pinned GGUF
conversion, Ollama deployment, tests, CI, security controls, and portfolio documentation.

The accessible local path was exercised end to end with the pinned Qwen2.5 0.5B model: one real
CPU LoRA optimizer step, reload and held-out evaluation, byte-safe merge, llama.cpp conversion,
Ollama registration, and loopback inference. This is engineering-execution proof, not a claim that
the one-step model meets task-quality or production-readiness thresholds.

## Delivered system

The checked-in project provides:

- five hardware-aware profiles for explicit CPU LoRA, auto-device smoke LoRA, Apple Silicon LoRA,
  Qwen2.5 7B CUDA QLoRA, and gated Llama 3.1 8B CUDA QLoRA;
- a 56-example synthetic CC0 support-ticket corpus with strict finite labels and placeholder-only
  identity data;
- deterministic, stratified train/validation/test preparation with a source-derived manifest,
  canonical-byte regeneration, bounded allocation search, and TOCTOU checks;
- Transformers, PEFT, and TRL training that records exact model revision, dataset hashes,
  resolved config, dependency inventory, hardware, metrics, adapter inventory, and Git state;
- immutable training and evaluation directories with explicit mutable `latest-*` pointers;
- checkpoint resume restricted to a compatible, hash-verified run-specific checkpoint;
- strict bare-JSON evaluation with schema, label, macro-F1, response-policy, exact-match,
  completeness, latency, absolute-threshold, and candidate-versus-baseline regression checks;
- a vLLM 0.24 static-LoRA plan and Compose deployment pinned to an exact Linux/amd64 image digest;
- a safe, non-quantized PEFT merge followed by a pinned llama.cpp GGUF conversion and Ollama
  Modelfile path;
- an offline-by-default CLI, test suite, CI, dependency audit, package build, security guidance,
  and tracked sanitized proof summaries.

## Architecture and trust boundaries

```text
source JSONL
  -> strict parse, privacy checks, deduplication, deterministic split
  -> canonical split bytes plus manifest hashes
  -> hardware and configuration preflight
  -> immutable TRL plus PEFT training run
  -> adapter inventory and run manifest
  -> exact held-out baseline and candidate generation
  -> immutable evaluation and fail-closed quality decision
  -> vLLM static adapter plan
     or safe merge -> pinned llama.cpp -> GGUF -> Ollama
```

Heavy training and serving libraries are loaded only inside their execution paths. Data,
configuration, provenance, evaluation, and deployment-plan validation stay deterministic and
testable without a GPU or model download. Remote model IDs require full 40-character revisions;
downloads and non-loopback serving require explicit operator choices.

The release audit also closed ambiguous JSON, artifact, and path boundaries: duplicate object keys
and non-finite constants are rejected at input trust boundaries; dataset verification regenerates
canonical splits from the immutable source contract; candidate and baseline reports must have
identical ordered IDs, expected objects, and compatible provenance; adapter config and weights must
be regular non-symlink files; and merge/GGUF source and output trees cannot overlap.

## Dataset evidence

| Artifact | Evidence |
| --- | --- |
| Source | 56 examples; SHA-256 `c8dc4e5bc19e1230bf370a55897c1ab7d8c6dd36a2e14cb14f944a203fcdcb5f` |
| Manifest | SHA-256 `bba1c170b057a116a856006dde2012e4aaee404be1bb226564f0bb1ad2ba007c` |
| Train | 42 examples; SHA-256 `6d7958c457779304be9c48293213445ae88ca646ae5f5fcc9c95fe395516f3a7` |
| Validation | 7 examples; SHA-256 `906c40dabee8175e7744e41d127ff011140308bf810baf4f6a466de87ea021a4` |
| Test | 7 examples; SHA-256 `5439239bf9a2c3c69db4492696a8d05914043e0f5ac4bd1cbf5e4357ece8ebd5` |

The committed raw corpus is demonstration data, not representative customer traffic. Real data
requires a separate legal basis, privacy review, redaction policy, access control, retention policy,
and human-reviewed evaluation set.

## Real CPU training and evaluation evidence

The local machine exposed MPS, but two native MPS attempts ended in backend crashes. The dedicated
CPU profile then completed successfully without entering MPS or CUDA paths.

- Base: `Qwen/Qwen2.5-0.5B-Instruct`
- Revision: `7ae557604adf67be50417f59c2c2f167def9a775`
- Run: `20260718T031845.891541Z-35d6408e9cb4`
- Run manifest SHA-256: `97f34a246373baaeab5c8393c6d47164093a0ece26d4da1dfa73c2ad9d75e6df`
- Method: LoRA, CPU, one optimizer step, seed 42
- Training loss: `2.464804172515869`
- Runtime: `2.7738` seconds
- Adapter config SHA-256: `e55c97a69a91f101620cae926ef0ecb504003c2aaa048136c89bc00a9a8b6eee`
- Adapter weight SHA-256: `7d0f3473628c0958205f2dae3e7438a77466910f5e795acab83ed84d52d4c695`

The immutable evaluation is `20260718T032044.486658Z-1d098b688c10`; its manifest SHA-256 is
`95dea57e51bcd8d4f0b98fd374101f4fa5a5947a73725bd67e501dcfbb9e7ba8`. Candidate and baseline
used the same seven held-out IDs. The candidate passed every zero-regression rule and improved
category accuracy and macro F1 by `0.285714`, but both models failed the configured absolute task
thresholds. Candidate schema-valid rate, priority accuracy, and sentiment accuracy were `0.0`.
The adapter is therefore not quality-qualified.

The run occurred before this repository's first commit. Its manifest truthfully records
`git_revision: null` and `git_dirty: true`; the run is byte-linked to model, data, config, adapter,
and evaluation hashes but not to a Git commit.

## Local deployment evidence

- Safe-merge provenance SHA-256:
  `ad259a4cb2736d597841a554e722f89aed021ea06b01ccc0986e2c055cc6d248`
- Merged model SHA-256:
  `10febff16d9c73c80543d5ea98a11da205d0a6a8795555ff9fa45ff60dba86e9`
- llama.cpp revision: `aedb2a5e9ca3d4064148bbb919e0ddc0c1b70ab3`
- Ollama export provenance SHA-256:
  `563426fd6ec160c0a92c90f58fb548614530f9cb38623a943d060ad8cc2c6924`
- F16 GGUF: 994,156,192 bytes; SHA-256
  `a57b44cbfbf56aed0428d3819b81f868fda805ed117c5018d9807f993d743fe3`
- Q4_K_M GGUF: 397,807,264 bytes; SHA-256
  `502905577a06697170e427032e0e2feba863c742e07c1818a3efa7e0fb040e8e`
- Runtime: Ollama `0.32.1`, loopback `127.0.0.1:11434`
- Registered model: `tickettune-cpu-smoke:latest`, ID `0efa04c454b7`
- Constrained response SHA-256:
  `47fa8f98df421b88f13d0ff1335e7f16c788b5f1658f283145a285d2c01ddb86`

The unconstrained Ollama response was valid JSON but failed the TicketTune domain schema because it
used a non-enum category and a malformed action token; its full response was not retained. A second
request with a JSON schema returned all five required fields and passed `TriageOutput`; the
PII-free result is tracked for independent replay. Functional output parity between adapter-backed
generation and the merged model was not run, so this is byte-safe merge and runtime proof only.

## Final validation

- `uv lock --check`: passed with 119 resolved packages.
- Frozen training and conversion environment: passed with Transformers `5.13.0`, TRL `1.7.1`,
  PEFT `0.19.1`, Datasets `5.0.0`, Accelerate `1.14.0`, bitsandbytes `0.49.2`, and CMake `4.4.0`.
- Ruff lint and formatting: passed across 26 Python files.
- Strict mypy: passed across 14 source files.
- Bandit: passed with no reported source findings.
- `pip-audit`: no known vulnerabilities in auditable dependencies; the local unpublished
  `tickettune` package is correctly reported as absent from PyPI.
- Pytest: **276 passed**.
- Branch/statement coverage: **90.26%**, above the enforced 85% floor.
- Wheel and source distribution: built successfully.
- Source-distribution inventory: required source, tests, profiles, dataset, deployment assets,
  docs, results, lockfile, and project metadata present; ignored runtime/model artifacts absent.
- `make smoke`: source validation, deterministic preparation, hardware report, training dry run,
  threshold-enforced fixture evaluation, vLLM plan, and Ollama export plan passed without model
  download or server startup.
- Docker Compose rendering: passed with the exact vLLM Linux/amd64 digest, pinned model revision,
  required immutable adapter path, loopback publication, offline defaults, read-only mount, dropped
  capabilities, and no-new-privileges.
- Installed console script and `python -m tickettune`: complete command help passed.

The locked environment also builds the wheel and source distribution, includes the operational
profiles, dataset, deployment assets, docs, results summaries, lockfile, and tests in the sdist,
and excludes generated adapters, GGUF files, model weights, caches, and secrets. Docker Compose
renders the pinned vLLM definition, but no vLLM runtime was started on this non-CUDA host.

## Reproduction entry points

```bash
uv sync --frozen --extra train --extra convert
uv run tickettune data validate --config configs/cpu-smoke.yaml
uv run tickettune data prepare --config configs/cpu-smoke.yaml
uv run tickettune doctor --config configs/cpu-smoke.yaml
uv run tickettune train --config configs/cpu-smoke.yaml --dry-run
uv run tickettune evaluate \
  --config configs/smoke.yaml \
  --predictions tests/fixtures/predictions_pass.jsonl \
  --enforce-thresholds
make check
make smoke
```

Model downloads are opt-in. A real run must add `--allow-download` only after checking the exact
upstream model license and immutable revision. Use the run-specific adapter path printed by the CLI;
never treat a mutable `latest-*` pointer as provenance.

## Source-control and release record

- Implementation commit:
  `2186ae9177f1c56da345db0c5da6d1a4184006a7` (`feat: build TicketTune fine-tuning and deployment pipeline`).
- The implementation commit contains only source, configuration, the raw synthetic corpus, tests,
  documentation, deployment assets, lockfile, tiny placeholder fixtures, and sanitized proof
  summaries; generated model weights, GGUF files, prepared data, caches, and distributions are
  ignored.
- This completion record is intentionally added in a following documentation commit so it can cite
  the immutable implementation commit without creating a self-referential hash.

The repository has no configured remote, so no push, pull request, hosted release, or deployment is
claimed. Generated model artifacts remain ignored and local; the repository tracks only source,
tests, documentation, configuration, and sanitized proof summaries.

## Proof boundaries and next qualification layers

Completed locally:

- deterministic control-plane validation and packaging;
- one-step CPU LoRA training on the pinned 0.5B model;
- immutable candidate-versus-baseline held-out evaluation;
- safe merge, pinned GGUF conversion, Ollama import, and loopback inference;
- schema-constrained response validation.

Not completed or not implied:

- a representative, human-reviewed production training corpus;
- a quality-qualified multi-step Qwen2.5 7B or Llama 3.1 8B CUDA QLoRA run;
- functional adapter-versus-merged inference parity;
- a post-commit real training rerun with exact Git provenance;
- a live vLLM server on Linux/CUDA;
- remote or production authentication, TLS, network policy, monitoring, load, rollback, and
  production readback.

## References

- [TRL SFTTrainer](https://huggingface.co/docs/trl/en/sft_trainer)
- [PEFT quantization guide](https://huggingface.co/docs/peft/developer_guides/quantization)
- [vLLM 0.24 OpenAI-compatible server](https://docs.vllm.ai/en/v0.24.0/serving/online_serving/openai_compatible_server/)
- [Ollama model import](https://docs.ollama.com/import)
