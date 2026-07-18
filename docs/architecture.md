# TicketTune architecture

## Purpose

TicketTune is a reproducible supervised fine-tuning system for one bounded
task: turn a support ticket into exactly one strict, bare JSON object with `category`,
`priority`, `sentiment`, `response`, and `next_action` fields. It demonstrates
the full path from governed data through PEFT adaptation, held-out evaluation,
and local serving without pretending that a dry run is a completed GPU run.

The allowed categories are `account_access`, `billing`, `bug`, `cancellation`,
`feature_request`, `shipping`, and `security`. Priorities are `low`, `medium`,
`high`, or `urgent`. Those values are a product contract in
`src/tickettune/schemas.py`, not free-form prompt suggestions.

## System flow

```text
Synthetic source JSONL
  -> strict validation and PII-placeholder checks
  -> content-hash deduplication
  -> human-review qualification for the quality profile
  -> deterministic category-aware train/validation/test splits
  -> conversational prompt/completion projection
  -> TRL SFT with PEFT LoRA or CUDA QLoRA
  -> adapter + run manifest
  -> held-out structured-output and baseline evaluation
  -> safe non-quantized merge -> adapter-versus-merged parity
  -> adapter-first vLLM serving behind a hardened production gateway
     or pinned llama.cpp GGUF -> Ollama
```

## Component boundaries

| Component | Responsibility | Heavy imports |
| --- | --- | --- |
| `config.py` | Reject unknown configuration, validate ranges, resolve repository-relative paths | No |
| `schemas.py` | Define the source conversation and exact output schema | No |
| `data.py` | Validate, deduplicate, split, project, hash, and atomically write JSONL | No |
| `qualification.py` | Bind human-review policy decisions to exact quality-candidate source bytes | No |
| `hardware.py` | Report accelerator truth and reject unsupported method/device combinations | PyTorch only inside probes |
| `training.py` | Build LoRA/QLoRA and TRL configuration, train, checkpoint, and save an adapter | Lazy |
| `generation.py` | Apply one chat-template contract to baseline, adapter, and merged inference | Lazy |
| `evaluation.py` | Parse JSON, score task metrics, enforce regression thresholds, write reports | Model generation is lazy |
| `parity.py` | Bind adapter and merged-model lineage, compare deterministic held-out behavior | Model generation is lazy |
| `export.py` | Validate adapter provenance, merge safely, and render shell-free deployment plans | Lazy for merge only |
| `deployment_proof.py` | Validate/start approved releases, capture endpoint-claim/load evidence, and verify rollback plans | No |
| `cli.py` | Turn service results into human-readable or machine-readable commands | No eager model load |

This separation makes configuration, data, metric, and deployment-plan tests
fully offline. Model downloads, training, merging, and serving remain explicit
operator actions.

## Data contract and leakage controls

Every source line is one `TicketExample` containing exactly three ordered chat
messages: system, user, assistant. The assistant message must parse to the same
strict `TriageOutput` stored in `expected`. Each example declares synthetic
CC0 provenance and states that it contains no real customer data.

PII-like values use visible placeholders such as `[EMAIL]` and `[ORDER_ID]`.
The loader rejects undeclared or unused placeholders. Duplicate identifiers,
normalized-content duplicates, cross-split leakage, malformed JSON, unexpected
roles, and assistant/schema mismatches are hard errors. The generated manifest
records source and split hashes, seed, counts, and label distributions.

## Training design

TicketTune projects the system and user messages to the prompt and the
assistant message to the completion. TRL therefore computes completion-only
loss without assuming that a model chat template can produce a trustworthy
assistant-token mask.

LoRA and QLoRA share PEFT configuration while preserving different hardware
requirements:

- LoRA loads ordinary model weights and can be used on supported CUDA, MPS, or
  CPU paths when memory permits.
- QLoRA uses 4-bit NF4 with double quantization and is restricted to a tested
  CUDA/bitsandbytes path. The training model is not loaded with
  `device_map="auto"`.
- The smoke profile proves integration on a small checkpoint. The 7B and 8B
  profiles are portfolio-scale configurations, not evidence that those runs
  happened on the current machine.
- The separate Qwen 7B quality profile requires an approved review manifest
  bound to at least 1,000 synthetic records before a real run can allocate its
  output directory or import the training stack.

Every real run writes one immutable
`<training.output_dir>/runs/<run-id>/` directory containing `adapter/`, `trainer/`, and the sibling
`manifest.json`. The manifest records resolved inputs, versions, method, seed, dataset hashes, and
the hashes of every run artifact. The output root's `latest-run.json` is an explicitly mutable
convenience pointer. It is never used as immutable provenance.

The framework-free `build_training_plan()` API resolves and validates the plan without importing
Torch or probing hardware. The CLI's `train --dry-run` wrapper additionally runs the hardware
probe, which may lazily import Torch, but it does not enforce compatibility, load model weights, or
construct a trainer. Neither form counts as training evidence.

Real training parses exact read-only run snapshots into in-memory rows before constructing a
Datasets object. Live adapter and merged inference loads through private, verified hard-link trees,
so framework path reopens cannot consume bytes from a transient path replacement.

## Evaluation design

Evaluation uses the held-out test split and deterministic decoding by default.
The scorer first checks that the entire response is one bare JSON object, then validates it against the
same output schema, and then records schema validity, category correctness,
priority correctness, field completeness, and latency. Malformed output remains
an ordinary scored failure instead of crashing or disappearing from the
denominator.

Aggregate reports include threshold decisions. Baseline and fine-tuned results
must use the same examples, tokenizer/chat template, and decoding settings for a
meaningful comparison.

Safe-merge parity is a second gate. It requires both runtimes to use the same ordered held-out IDs,
prepared-data manifest, generation configuration, training manifest, adapter inventory, and merge
provenance. Strict schema validity and every routing field must match; response-text exactness is
diagnostic. The complete adapter and merged-model artifact inventories are rechecked after their
respective inference passes. The final report is published only after both sides complete; an
interrupted run may retain an immutable `.adapter-predictions.jsonl` sidecar, which a retry reuses
only when its semantic prediction bytes still match.

Every live evaluation receives an immutable
`<evaluation.output_dir>/runs/<evaluation-id>/` directory and an
`evaluation-manifest.json` that hashes its predictions and reports. The root
`latest-evaluation.json` is only a mutable discovery pointer.

## Export and serving design

### vLLM

vLLM serves the base plus PEFT adapter directly. For a remote model ID,
`build_vllm_plan()` records the configured pinned base revision, adapter and training-lineage
hashes, lineage boundary, network policy, offline environment,
and a vLLM 0.24 static LoRA descriptor as one JSON argv element with `name`,
`path`, and `base_model_name`. The builder verifies the adapter's declared base,
optional declared revision, and rank, disables request logging, uses vLLM
generation defaults, and binds to loopback unless a caller explicitly opts into
a remote bind. Native execution is plan-only by default and offline unless a
caller explicitly enables the pinned download. The Compose file passes the same
revision, defaults Hugging Face and Transformers to offline mode, binds vLLM to
all interfaces inside the container, and publishes the port only on host
loopback.

Dynamic LoRA management endpoints are not enabled. A request selects the
adapter by using the descriptor's `name` as the OpenAI-compatible `model`.

An existing local base path is not inventoried by the planner. By default, the vLLM planner
requires an exact configured revision, a completed sibling training manifest, and qualification
review/report hashes. When PEFT metadata omits `revision`, the configured revision is still matched
against that manifest. Its plan binds the base/configured revision, adapter config and weights, training config
and dataset hashes, qualification hashes, configured maximum rank, download policy, and network
policy; the builder validates the actual adapter rank without storing it in the returned plan. The
explicit `allow_unqualified_local_smoke` option is the only standalone-adapter path and serializes
`unqualified_local_smoke_override_not_release_evidence`. Evaluation and parity remain outside the
planner; the production release validator is the stronger end-to-end enforcement layer.

The production reference keeps vLLM on an internal model network and exposes only allowlisted API
routes through NGINX TLS. Prometheus reaches a separate metrics-only gateway listener, Alertmanager
is the only notification-egress service, and all images are pinned by Linux/amd64 digest. A schema
`2.0` release manifest binds the adapter to completed clean training, passing qualification,
passing evaluation, and passing parity evidence. The fail-closed start launcher additionally
requires the exact versioned production Compose/support-file profile, scrubs ambient interpolation
overrides, rechecks the entire evidence graph, and only then invokes fixed Compose start arguments.

### Safe merge

An Ollama export begins with a pristine, non-quantized reload of the exact base
model. The adapter metadata must identify that same base. The merge calls PEFT
`merge_and_unload(safe_merge=True)`, requires the completed sibling manifest to
match the configured revision even when the adapter omits one, requires any
adapter-declared revision to match too, and repeats those checks at plan and execution time. It saves
Safetensors plus tokenizer files to a temporary directory, writes checksums and
provenance, and only then renames the directory into place. Existing
destinations are never overwritten. Adapter bytes are hashed at planning,
rechecked before loading, and rechecked after the merge; adapter and output
directory trees may not overlap. Adapter config and Safetensors inputs must be
regular non-symlink files, and an existing local base-model tree cannot contain
or be contained by the merge destination.

Live merge parity repeats the same held-out prompts through base-plus-adapter and merged inference.
Both sides must produce schema-valid JSON, and category, priority, sentiment, `next_action`, plus
the aggregate routing gate must match exactly at `1.0`. Creating merged bytes is not itself parity
proof.

A safe merge from a standalone compatible adapter is available only through the explicitly named
local-smoke override. Its plan and provenance state that the output is not release evidence. The
default merge instead requires exact config/data/qualification lineage and preserves the
`qualified_release_lineage` boundary for downstream parity and conversion.

### Ollama

TicketTune does not generate a direct Qwen `ADAPTER` Modelfile. Ollama's direct
Safetensors-adapter documentation currently lists Llama, Mistral, and Gemma,
not Qwen, and recommends non-QLoRA adapters for those supported paths. The Qwen
route always merges first, then uses llama.cpp tag `b9637` at full commit
`aedb2a5e9ca3d4064148bbb919e0ddc0c1b70ab3` to convert merged Hugging Face
weights to F16 GGUF and optionally quantize them. The generated Modelfile uses
only `FROM ./<merged-model>.gguf`. Planning enforces the safe, non-quantized merge flags,
adapter/base identity, complete source inventory, and qualified training-lineage boundary by
default; it does not itself establish live parity. The local-smoke override accepts a legacy or
unqualified safe merge but carries the non-release boundary through Ollama export provenance.
Conversion refuses source directories whose safe-merge provenance
inventory or hashes no longer match. A successful
conversion atomically writes a second immutable manifest linking that source
inventory and llama.cpp revision to the F16 GGUF, final GGUF, and Modelfile
hashes. The source inventory recursively covers normalized relative paths and
rejects symlinks or untracked nested files. It is revalidated after conversion,
and source/output directory trees may not overlap.

The exporter controls and records the pinned conversion subprocesses, output bytes, and optional
`ollama create` invocation. It does not launch, configure, secure, or own the long-running Ollama
daemon. The daemon's bind policy, authentication boundary, logs, lifecycle, and any non-TicketTune
models remain outside the exporter trust boundary and require separate runtime evidence.

## Trust and proof boundaries

| Claim | Required evidence |
| --- | --- |
| Offline implementation works | Offline unit/integration tests pass with network disabled |
| Dataset is reproducible | Repeated preparation yields matching manifests and split hashes |
| Dataset is review-qualified | Exact source/review hashes and every qualification decision pass; production representativeness remains unproven |
| A model was trained | Adapter bytes and run manifest from a real, compatible execution |
| Fine-tuning improved the task | Baseline and adapted held-out reports using identical settings |
| Merge is valid | Merge provenance, checksums, and adapter-vs-merged parity report |
| vLLM serves | Linux/CUDA container health plus an adapter-selected chat response |
| Ollama serves | GGUF checksum, created local model, health/process evidence, and response |
| Production-ready | Lineage-bound release/start receipt, container/image/mount/digest readback, authenticated TLS endpoint proof, access control, alert delivery, capacity/load proof, rollback rehearsal, immutable release storage, and approved host policy |

Offline tests and dry-run plans deliberately stop before the last four runtime
claims.

The clean post-commit CPU run proved one local LoRA step, immutable held-out generation, and a
byte-safe merge. The one-step adapter failed the configured absolute quality gates. Enforced
adapter-versus-merged parity then failed on all seven held-out prompts, so the merged artifact is
diagnostic evidence only and is not release-eligible. CUDA QLoRA, vLLM runtime, hosted serving,
and production controls remain unverified.

## External contracts

- [vLLM LoRA adapters](https://docs.vllm.ai/en/v0.24.0/features/lora/)
- [vLLM Docker deployment](https://docs.vllm.ai/en/v0.24.0/deployment/docker/)
- [Ollama model import](https://docs.ollama.com/import)
- [Ollama Modelfile reference](https://docs.ollama.com/modelfile)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
