# Deployment runbook

## Supported paths

TicketTune has two model-serving forms plus a hardened vLLM production reference:

| Path | Artifact served | Best fit | Required runtime |
| --- | --- | --- | --- |
| vLLM 0.24 local | Original base plus PEFT adapter | Loopback development, OpenAI-compatible serving, adapter kept separate | CUDA-capable Linux host, NVIDIA Container Toolkit, Docker Compose |
| vLLM 0.24 production reference | Qualified base plus lineage-bound PEFT adapter | TLS gateway, private model network, monitoring, alerts, release/readback/load/rollback proof | Approved Linux/amd64 NVIDIA host and target-host controls |
| Ollama | Merged model converted to GGUF | Developer laptop or small local server, portable quantized artifact | Enough RAM/disk for merge and conversion; Ollama for final import |

The native and simple Compose vLLM paths remain local-only. The production bundle is a reviewed
reference, not evidence that a target host is configured or live. Every deployment claim remains
separate from offline tests, dry runs, static configuration checks, and localhost proof.

## Artifact prerequisites

Before release promotion or an Ollama conversion, operator policy requires:

1. Prepare and hash the dataset.
2. Complete a real LoRA or QLoRA run and retain its run manifest.
3. Evaluate the adapter on the held-out split.
4. Verify `adapter_config.json` declares the exact configured base model.
5. Keep the immutable run directory intact. Its `adapter/` contains Safetensors and tokenizer
   metadata; its sibling `manifest.json` is the authoritative run record.
6. Use TicketTune's safe merge output unchanged: GGUF planning requires its
   `tickettune-merge-provenance.json` inventory and verifies every declared file
   hash before conversion.

The default merge, `serve vllm`, and Ollama planners validate completed sibling training lineage,
bind the manifest to the configured base revision even when PEFT metadata omits `revision`, and
require qualification review/report hashes. They do not require evaluation or parity evidence;
schema `2.0` production release validation remains the stronger end-to-end layer. The explicit
`--allow-unqualified-local-smoke` escape hatch accepts a missing or unqualified lineage only for
local smoke tests and fixtures, and its serialized boundary is never release evidence. A present
malformed, tampered, or revision-mismatched manifest is rejected even with that flag.

The export code rejects missing Safetensors, missing PEFT metadata, base-model
or revision mismatches, invalid ranks, unsafe bind addresses, untracked or
hash-mismatched merged artifacts, and unsupported conversion options before
model execution.

## vLLM 0.24: adapter-first deployment

### Hardware and software

Use a Linux host with a supported NVIDIA GPU, working driver, NVIDIA Container
Toolkit, Docker Engine, and Docker Compose v2. Apple Silicon and CPU-only local
machines can render and test the plan but do not prove this CUDA runtime.

The Compose file pins the exact `linux/amd64` image
`vllm/vllm-openai:v0.24.0@sha256:f9de5cd9fa907fbf6dbba691eb7db095d48ad58ea283e3eba7142f9a91e186e8`.
The platform manifest digest was read back from Docker Hub on 2026-07-17; re-verify it before
changing the image or platform. A controlled production deployment should mirror that digest into
its own registry. The
adapter rank must not exceed `MAX_LORA_RANK`, and
the base model plus KV cache must fit available VRAM at the selected context
and utilization.

### Static plan contract

For a remote model ID, `build_vllm_plan()` returns the configured pinned revision, adapter hashes,
explicit download policy, offline environment overrides, and one shell-free argv list. An existing
local base is not inventoried by the planner, but a release-capable plan still requires an exact
configured revision. Only the explicit non-release local-smoke override may omit it.
`build_vllm_argv()` is the lower-level compatibility helper for callers that
need only argv. Both use the current static descriptor shape documented by
vLLM 0.24:

`VllmServePlan` carries base/configured-revision, adapter config/weight hashes, verified training
manifest/config/dataset/qualification hashes, a machine-readable `lineage_boundary`, download
policy, and network policy. The builder validates the actual adapter rank against the configured
maximum, but the returned plan does not carry the actual rank. Evaluation and parity are still
outside this planner; use schema `2.0` release validation for that stronger boundary.

```json
[
  "vllm",
  "serve",
  "Qwen/Qwen2.5-7B-Instruct",
  "--revision",
  "a09a35458c702b33eeacc393d103063234e8bc28",
  "--host",
  "127.0.0.1",
  "--port",
  "8000",
  "--enable-lora",
  "--max-lora-rank",
  "64",
  "--lora-modules",
  "{\"base_model_name\":\"Qwen/Qwen2.5-7B-Instruct\",\"name\":\"tickettune\",\"path\":\"/absolute/path/to/adapter\"}",
  "--generation-config",
  "vllm",
  "--disable-log-requests"
]
```

Remote model identifiers require a full 40-character commit revision. The
actual builder also emits dtype, model length, tensor parallelism, and GPU
utilization. The JSON descriptor is one argv element, not a shell fragment.
`base_model_name` preserves model lineage in `/v1/models`. `name` is the model
value clients use to select the adapter. Static registration avoids exposing
runtime adapter-management endpoints.

Render and validate the configured command without starting a server. Plan-only
is the CLI default; `--dry-run` remains as an explicit compatibility spelling:

```bash
uv run --no-sync tickettune serve vllm \
  --config configs/qwen-7b-qlora-quality.yaml \
  --adapter artifacts/qwen-7b-quality/runs/<run-id>/adapter \
  --dry-run --json
```

The supported execution path is Docker Compose on a reviewed Linux/CUDA host.
For debugging only, an explicit `--execute` runs the native argv in the
foreground. It is offline by default: the child receives `HF_HUB_OFFLINE=1`
and `TRANSFORMERS_OFFLINE=1`. Add `--allow-download` only for a deliberate
first fetch of the already-pinned revision. A configured non-loopback host also
requires explicit `--allow-remote`; that flag is a safety acknowledgement, not
authentication or TLS.

Programmatic use is shell-free:

```python
from pathlib import Path

from tickettune.export import build_vllm_argv, run_argv

argv = build_vllm_argv(
    "Qwen/Qwen2.5-7B-Instruct",
    Path("artifacts/qwen-7b-quality/runs/<run-id>/adapter"),
    model_revision="a09a35458c702b33eeacc393d103063234e8bc28",
    served_model_name="tickettune",
)
print(argv)
```

Native execution is intentionally not a health proof. If the foreground
process exits cleanly, the final CLI result says only that; health and adapter-
selected inference still need the checks below. In `--json` execution mode,
the reviewed preflight plan and all child-process logs go to stderr, while
stdout contains one final parseable JSON object after a clean exit.

### Compose setup

Run these commands from the repository root:

```bash
cp deploy/vllm/.env.example deploy/vllm/.env
mkdir -p deploy/vllm/.cache/huggingface
docker compose --env-file deploy/vllm/.env -f deploy/vllm/compose.yaml config --quiet
docker compose --env-file deploy/vllm/.env -f deploy/vllm/compose.yaml up -d
```

Edit only the local, ignored `deploy/vllm/.env`, never the example. Make
`BASE_MODEL` exactly match the adapter metadata and set `MODEL_REVISION` to the
same reviewed 40-character commit used for training. `ADAPTER_PATH` is required—there is no
fallback—and must be a host path to one immutable `<output>/runs/<run-id>/adapter` directory,
mounted read-only at `/models/adapter`. `HF_CACHE_PATH` must be writable by UID
2000, group 0. Compose sets both Hugging Face and Transformers offline flags to
`1` by default, so a missing cached revision fails closed. For a deliberate
first download only, set both flags to `0`, run with the pinned revision, then
restore them to `1`. If a gated model requires a token, use the host's secret
manager or a short-lived process-level secret; do not add it to these files.

The planner always verifies adapter base identity and hashes its PEFT config
and Safetensors. If `adapter_config.json` declares a `revision`, that revision
must match. When PEFT omits it, the configured revision is still matched exactly
against the completed sibling `manifest.json`; the plan reports both that
manifest lineage and `revision_not_embedded_in_adapter`.

Inside the container vLLM binds `0.0.0.0:8000`, which Docker needs for port
forwarding. The host publishes only `127.0.0.1:${VLLM_PORT}:8000`. Do not change
that host address to `0.0.0.0` as a convenience.

### Health and inference proof

The container health check calls `/health` without a shell. From the host, use:

```bash
python3 deploy/examples/health.py --base-url http://127.0.0.1:8000
uv run --no-sync python deploy/examples/chat.py \
  --base-url http://127.0.0.1:8000 \
  --model tickettune
```

Then inspect the model lineage with an HTTP client against `GET /v1/models` and
confirm the `tickettune` entry has the configured base as its parent. A passing
health endpoint proves process readiness, not output quality. Save a redacted
chat response, server version, GPU model, image digest, adapter/config hashes,
and held-out report together as deployment evidence.

Both example clients re-check every redirect against the loopback policy. The
chat client also sends the canonical training/evaluation system prompt and
validates returned content against `TriageOutput` before printing it; malformed
or free-form model text is reported as an error rather than emitted downstream.

To stop the local service:

```bash
docker compose --env-file deploy/vllm/.env -f deploy/vllm/compose.yaml down
```

## Hardened vLLM production reference

`deploy/vllm/production/` is the release-oriented path. It pins vLLM, NGINX, Prometheus, and
Alertmanager by Linux/amd64 digest. Only the NGINX TLS listener publishes a host port. vLLM stays
on the model network; Prometheus reaches only NGINX's metrics-only listener; Alertmanager is the
only service with notification egress.

The bundle requires four external group-readable `0440` secret files: TLS certificate, TLS private
key, a high-entropy API key of 32–4096 printable ASCII bytes with no whitespace, and the approved
Alertmanager webhook URL. The API key is validated without following symlinks and is injected only
into the vLLM child environment. Adapter and pinned Hugging Face cache mounts are read-only; a
separate pre-created runtime cache holds all writable framework caches.

Copy `.env.example` to an ignored release-specific env file and replace every sentinel. Compute
`EXPECTED_ADAPTER_SHA256` with TicketTune's canonical adapter-inventory algorithm. Then create a
schema `2.0` release manifest that binds the exact Compose/env/adapter bytes plus:

- the verified prepared dataset manifest and exact prepared test split bytes;
- a clean, completed training manifest for that adapter;
- a passing quality-candidate qualification report;
- a completed, passing adapter-versus-baseline evaluation manifest, candidate report, baseline
  report, prediction artifacts, and scored rows; and
- a passing adapter-versus-merged parity report plus both immutable prediction sidecars.

Release validation rejects a smoke adapter, dirty or failed training, placeholder evidence,
digest mismatch, failed policy decision, or evidence whose adapter/config/dataset lineage differs.
It parses the exact prepared test bytes, binds their ordered IDs and expected labels to the reviewed
cohort, and recomputes evaluation and parity evidence rather than trusting reported pass flags.

Render the bundle without starting a GPU service:

```bash
docker compose \
  --env-file deploy/vllm/production/release.env \
  -f deploy/vllm/production/compose.yaml \
  config --quiet
```

Also validate NGINX with test certificate material, Prometheus with `promtool`, Alertmanager with
`amtool`, and the fail-closed Python entrypoint. These are static checks only. On the approved host,
start the immutable project, then capture authenticated TLS readback and bounded load evidence:

```bash
uv run tickettune deploy readback \
  --base-url https://MODEL_HOSTNAME \
  --api-key-file /secure/operator-copy/tickettune-api-key \
  --ca-cert /secure/issuer-ca.pem \
  --model tickettune-qwen-7b-quality \
  --expected-base-model Qwen/Qwen2.5-7B-Instruct \
  --allow-remote \
  --output artifacts/validation/production-readback.json \
  --enforce \
  --json

uv run tickettune deploy load-test \
  --base-url https://MODEL_HOSTNAME \
  --api-key-file /secure/operator-copy/tickettune-api-key \
  --ca-cert /secure/issuer-ca.pem \
  --model tickettune-qwen-7b-quality \
  --requests 100 \
  --concurrency 4 \
  --min-success-rate 0.99 \
  --min-schema-valid-rate 0.99 \
  --min-request-id-rate 0.99 \
  --max-p95-ms 5000 \
  --allow-remote \
  --output artifacts/validation/production-load.json \
  --enforce \
  --json
```

Validate the complete release evidence graph before target-host work:

```text
uv run tickettune deploy validate-release \
  --manifest /immutable/releases/release-manifest.json \
  --output artifacts/validation/release-validation.json \
  --json
```

The report proves that one stable read of every schema `2.0` release input passed byte, semantic,
and lineage checks. Its stated use limit requires immediate revalidation before deployment; it is
not a durable claim that the external files remain unchanged.

The only supported production start path validates again, requires the exact versioned TicketTune
Compose and support-file profile plus the literal `tickettune-production` project slot, rechecks
every bound file and the adapter inventory, and then invokes fixed shell-free Compose arguments.
It starts with an empty environment, a non-user home, and an explicit
`unix:///var/run/docker.sock` target, so a persisted Docker context or user Compose plugin cannot
redirect the operation:

```text
uv run tickettune deploy start-release \
  --manifest /immutable/releases/release-manifest.json \
  --execute \
  --json
```

Do not use a manual `docker compose up` as release evidence. A zero Compose exit from
`start-release` proves only the validated start invocation; authenticated TLS readback, target-host
controls, monitoring, and runtime quality still require separate proof. The recheck-to-Docker handoff
remains path based: deploy from operator-enforced immutable, content-addressed storage and inspect
the running images, labels, mounts, and in-container adapter digest before acceptance.

The proof clients reject redirects, bound response bodies, omit prompt/completion text from reports,
and require remote HTTPS plus explicit acknowledgement. API readback proves only authenticated
endpoint-reported model/parent claims and schema behavior; it does not independently prove release
or adapter bytes. Combine it with the validated launcher receipt and target-host container/mount
readback. Confirm target health, fire and receive a controlled alert, prove host firewall and
notification-egress policy, and rehearse a rollback plan that revalidates compatible current and
previous manifests immediately before execution. `rollback-plan` returns shell-free Compose argv
and never changes the deployment itself.

See the bundle's [operator README](../deploy/vllm/production/README.md) and the
[CUDA qualification runbook](operations/cuda-qualification-runbook.md) for exact target-host steps.

## Ollama: merged Hugging Face to GGUF

### Why Qwen is merged first

Do not point an Ollama `ADAPTER` instruction at the TicketTune Qwen PEFT
directory. Ollama's documented direct Safetensors-adapter architectures are
Llama, Mistral, and Gemma; Qwen is not listed. Ollama also recommends
non-quantized adapters for listed architectures because quantization methods
vary between frameworks.

TicketTune therefore has one Qwen route:

```text
exact base + PEFT adapter
  -> pristine non-quantized safe merge
  -> merged Hugging Face Safetensors
  -> llama.cpp b9637 / aedb2a5e9ca3d4064148bbb919e0ddc0c1b70ab3
  -> F16 GGUF
  -> optional Q4_K_M, Q5_K_M, or Q8_0 quantization
  -> GGUF-only Modelfile
  -> ollama create
```

### Safe merge

Build and inspect the merge plan before the heavy operation:

```bash
uv run --no-sync --extra train tickettune merge \
  --config configs/qwen-7b-qlora-quality.yaml \
  --adapter artifacts/qwen-7b-quality/runs/<run-id>/adapter \
  --output artifacts/qwen-7b-quality/deployments/<run-id>/merged \
  --dry-run
```

By default, the actual merge sets `local_files_only=true` for both base model
and tokenizer. Pre-populate the reviewed revision in the Hugging Face cache.
If a deliberate first download is required, add `--allow-download`, use a
restricted network path, and record the resolved revision and resulting hashes.
Then rerun without `--dry-run`.

```python
from pathlib import Path

from tickettune.export import build_merge_plan, merge_adapter

plan = build_merge_plan(
    "Qwen/Qwen2.5-7B-Instruct",
    Path("artifacts/qwen-7b-quality/runs/<run-id>/adapter"),
    Path("artifacts/qwen-7b-quality/deployments/<run-id>/merged"),
    dtype="bfloat16",
    model_revision="the-reviewed-model-revision",
    allow_download=False,
)
print(plan.model_dump(mode="json"))
result = merge_adapter(plan)
print(result.model_dump(mode="json"))
```

The merge reloads the base without 4-bit or 8-bit quantization, disables remote
model code, checks adapter/base identity and any adapter-declared revision at
plan time, execution start, and execution completion, uses PEFT safe merge, saves
Safetensors and tokenizer files, and writes
`tickettune-merge-provenance.json`. The destination must not already exist.
The adapter and merge destination must be isolated directory trees; a nested or
ancestor destination is rejected.
Plan enough temporary disk for the base, adapter, merged model, F16 GGUF, and
quantized GGUF at the same time.

By default, merge validation requires the sibling training manifest and carries its training
config, dataset, qualification hashes, and `qualified_release_lineage` boundary into merge
provenance. A compatible standalone or unqualified adapter is accepted only with
`allow_unqualified_local_smoke=True` in Python or `--allow-unqualified-local-smoke` in the CLI; the
resulting `unqualified_local_smoke_override_not_release_evidence` boundary cannot be presented as
release proof.

Before conversion, run live parity with identical held-out prompts and decoding settings. Require
schema-valid JSON on both sides plus exact `1.0` matches for category, priority, sentiment,
`next_action`, and aggregate routing. Response-text exactness is diagnostic; there is no numeric,
logit, or tolerance override. Do not infer parity from a successful save.

### Pinned conversion plan

This planning step enforces the merge's safe/non-quantized flags, adapter/base and revision
identity, adapter-config and adapter-weight digest shapes, merge-provenance file, complete
merged-file inventory, and a qualified training lineage boundary by default. It does not execute
functional parity. The same explicitly named local-smoke override can consume a legacy or
unqualified safe merge, but the downstream plan and export provenance remain marked non-release.

`build_ollama_export_plan()` validates the merged Safetensors directory and returns ordered argv
arrays for:

1. cloning llama.cpp without a checkout;
2. detached checkout of full commit
   `aedb2a5e9ca3d4064148bbb919e0ddc0c1b70ab3` (tag `b9637`);
3. configuring and building only `llama-quantize` when quantization is needed;
4. running `convert_hf_to_gguf.py` with `--outtype f16`;
5. optional GGUF quantization;
6. SHA-256 recording for the F16 GGUF, final GGUF, and Modelfile;
7. Ollama model creation.

Planning first verifies the source merge manifest and carries its SHA-256 plus
the complete verified source inventory into the export plan. After conversion,
TicketTune re-verifies that source inventory before it atomically writes immutable
`tickettune-ollama-export-provenance.json`, linking those source hashes to the
pinned llama.cpp revision and the F16/final/Modelfile hashes.
The merged source and Ollama output must be disjoint directory trees, and
materialization refuses unexpected entries in an existing output directory.

Render the plan through the CLI:

```bash
uv run --no-sync tickettune export ollama \
  --config configs/qwen-7b-qlora-quality.yaml \
  --merged-model artifacts/qwen-7b-quality/deployments/<run-id>/merged \
  --output artifacts/qwen-7b-quality/deployments/<run-id>/ollama \
  --json
```

Add `--materialize` only to create the output directory and Modelfile. It does
not clone, convert, quantize, or invoke Ollama; the default remains plan-only.

After reviewing the complete plan, dependencies, disk/RAM headroom, source
model license, and destination paths, `--execute` runs the pinned llama.cpp
clone, detached checkout, build, conversion, optional quantization, and
checksum arrays in order:

```bash
uv sync --frozen --extra train --extra convert
uv run --no-sync --extra train --extra convert tickettune export ollama \
  --config configs/qwen-7b-qlora-quality.yaml \
  --merged-model artifacts/qwen-7b-quality/deployments/<run-id>/merged \
  --output artifacts/qwen-7b-quality/deployments/<run-id>/ollama \
  --execute --json
```

Add `--create-model` to that explicit execution only when the resulting GGUF
should also be registered with the local Ollama daemon. Existing clone or
artifact destinations are not treated as safe to overwrite; preserve the prior
hashes and choose a new output directory for a new build.

The CLI prints the pre-execution plan before doing any work and emits a success
result only after every requested subprocess returns successfully. In `--json`
mode, that plan plus clone/build/conversion output is routed to stderr; stdout
contains one final JSON result. A failed command produces no
`local_conversion_completed` or `ollama_model_created` success boundary.
The final result includes the durable export-provenance path, manifest hash,
and artifact hashes; provenance write failure also prevents a success boundary.

```python
from pathlib import Path

from tickettune.export import build_ollama_export_plan, materialize_ollama_plan

plan = build_ollama_export_plan(
    Path("artifacts/qwen-7b-quality/deployments/<run-id>/merged"),
    Path("artifacts/qwen-7b-quality/deployments/<run-id>/ollama"),
    model_name="tickettune",
    quantization="Q4_K_M",
)
materialize_ollama_plan(plan)
for argv in plan.commands:
    print(list(argv))
```

TicketTune runs the converter with the active TicketTune Python interpreter;
install the pinned `train` extra first so its Transformers, tokenizer, and
model-format dependencies are present. Review llama.cpp's pinned conversion
inputs as part of the supply-chain boundary. The generated Modelfile is
equivalent to:

```text
FROM ./tickettune-q4_k_m.gguf
PARAMETER temperature 0
PARAMETER num_ctx 2048
SYSTEM """You are TicketTune, a support triage assistant. Return exactly one JSON object with keys category, priority, sentiment, response, and next_action. Do not include markdown or expose private data."""
```

It intentionally has no `ADAPTER` instruction.

### Ollama process boundary

TicketTune constrains the bytes it creates: exact merge inventory, pinned llama.cpp commit,
shell-free subprocess argv, supported quantization, GGUF/Modelfile hashes, and immutable export
provenance. With `--create-model`, it also records that the requested `ollama create` command
completed.

TicketTune does not launch or own the persistent Ollama daemon. It cannot guarantee that daemon's
bind address, access control, log policy, lifecycle, storage permissions, or other installed models.
Treat those as an operator-managed runtime boundary. Keep the daemon loopback-only for this local
proof and add an authenticated TLS gateway before any remote exposure.

### Ollama proof

After the GGUF and Modelfile checksums and immutable export provenance are recorded, run the plan's
`ollama_create_argv`, then:

```bash
ollama show tickettune --modelfile
ollama run tickettune \
  "Classify this support ticket: I was charged twice for [INVOICE_ID]."
```

Validate the response against the TicketTune output schema and repeat the held-
out evaluation through the actual Ollama endpoint. A successful import alone
does not prove merge parity or task quality. Keep the generated export-
provenance manifest with the Ollama version, hardware, and runtime evaluation;
it already links the source merged-model inventory, merge-manifest hash,
llama.cpp revision, F16 and final GGUF hashes, and Modelfile hash.

## Remote server hardening

The supplied artifacts are loopback examples. Before any remote deployment:

- place the model server on a private network behind an authenticated TLS
  gateway;
- add authorization, request/body/context limits, rate limits, timeouts, queue
  limits, and GPU concurrency controls;
- keep prompt and response bodies out of default logs;
- disable dynamic LoRA loading unless a separate authenticated control plane
  validates allowlisted artifact identities and hashes;
- scan and mirror images and dependencies, pin digests, and define patch SLAs;
- monitor health, queue depth, GPU memory/temperature/utilization, latency,
  schema-valid rate, and error rate;
- validate model JSON before downstream use and require humans for consequential
  support actions;
- document rollback to the prior adapter/image/GGUF checksums.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| vLLM adapter fails to load | Exact `base_model_name_or_path`, adapter rank vs `MAX_LORA_RANK`, mounted path and permissions |
| Offline vLLM cannot find weights | Confirm `MODEL_REVISION`, pre-populate that exact revision in `HF_CACHE_PATH`, and keep both offline flags at `1` for the proof run |
| Container stays unhealthy | GPU runtime, driver/image compatibility, model download/cache permissions, VRAM, startup logs without prompt bodies |
| Host cannot connect | Keep container bind at `0.0.0.0`; verify host mapping is `127.0.0.1:<port>:8000` |
| CUDA out of memory | Lower context, concurrency, or GPU utilization; use more GPUs with correct tensor parallelism; do not hide the failure |
| Merge runs out of RAM/disk | Use a smaller model or larger machine and preserve the non-quantized merge requirement |
| llama.cpp conversion changes bytes | Confirm the full pinned commit and conversion dependencies, then compare source and output hashes |
| Ollama output is malformed | Confirm the GGUF came from the validated merged directory, inspect template/tokenizer compatibility, rerun parity and held-out evaluation |

## Validation and release status

This runbook and its command planners are designed for offline validation. The
repository's unit tests exercise plan rendering, safety rejection, version pins,
asset contracts, immutable revision propagation, offline gates, truthful post-
subprocess result emission, merge/export provenance verification, redirect
policy, and output-schema validation without downloading a model or starting a server.

The earlier 2026-07-17/18 local CPU closeout proved Qwen2.5 0.5B merge, pinned llama.cpp conversion,
Ollama import, and local inference. A later clean post-commit run repeated real one-step CPU LoRA
training, held-out evaluation, and safe merge, then enforced adapter-versus-merged parity. Parity
historically rejected all seven schema-invalid rows, but that legacy report cannot distinguish
invalid equality from real cross-side drift. The adapter also failed its absolute quality gates.
The merged artifact is therefore diagnostic local evidence only and is not release-eligible. See
`results/post-commit-cpu/` for the sanitized acceptance result. Linux/CUDA vLLM execution, 7B/8B
QLoRA, hosted deployment, load testing, authenticated TLS controls, and production readback remain
unverified.

References: [vLLM LoRA adapters](https://docs.vllm.ai/en/v0.24.0/features/lora/),
[vLLM Docker](https://docs.vllm.ai/en/v0.24.0/deployment/docker/),
[Ollama imports](https://docs.ollama.com/import), and
[Ollama Modelfile](https://docs.ollama.com/modelfile).
