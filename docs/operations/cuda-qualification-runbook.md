# CUDA qualification and vLLM acceptance runbook

This runbook is the executable handoff for the TicketTune work that cannot run on the current
Apple Silicon host. It keeps dataset review, CUDA training, model quality, merge parity, serving,
and production acceptance as separate gates. Stop at the first failed gate; never weaken a
threshold to promote an artifact.

## 0. Rehearse the contract without claiming CUDA

On a CPU or Apple Silicon development host, validate the portable parts before requesting GPU
capacity:

```bash
uv run tickettune data generate-candidate \
  --output data/qualified/support_tickets.jsonl \
  --seed 42
uv run tickettune data prepare --config configs/qwen-7b-qlora-quality.yaml --json
uv run tickettune rehearse cuda \
  --config configs/qwen-7b-qlora-quality.yaml \
  --output artifacts/validation/cuda-quality-contract.json \
  --enforce \
  --json
```

This prepares the exact 896/112/112 quality-profile split and rehearses the final 7B contract. The
rehearsal may pass its static gates while truthfully reporting `blocked_no_cuda` and pending human
qualification. The tracked fallback summary in
`results/qualified-candidate/cuda-vllm-fallback-summary.json` is an example of that boundary.

<abbr title="A rehearsal verifies declarations and observed incompatibility; it does not execute training.">ⓘ</abbr>

| Evidence | Rehearsal may prove | Real CUDA run must prove |
| --- | --- | --- |
| Config and data | Pinned revision, NF4/double quantization, completion-only loss, verified development split hashes | Exact final profile and prepared-data hashes recorded by training |
| Runtime | Observed local accelerator and a truthful `blocked_no_cuda` result | Linux, CUDA, finite compute capability, bfloat16 support, and compatible preflight |
| Execution | `model_weights_loaded=false`, `optimizer_steps=0`, no adapter or run manifest | Loaded weights, positive optimizer steps, positive peak CUDA memory, adapter hashes, clean run manifest |
| Release | `release_eligible=false` | Qualified data, passing quality, passing parity, serving and readback evidence |

`--enforce` covers the static contract only. A non-CUDA host is an expected external runtime block,
not a reason to manufacture surrogate execution evidence.

## 1. Approve and prepare the target

Use a reviewed Linux/amd64 host with:

- an NVIDIA GPU supporting bfloat16 and at least 24 GB VRAM;
- at least 32 GB host memory and 80 GB free storage;
- a compatible NVIDIA driver, Docker Engine, Compose, and NVIDIA Container Toolkit;
- outbound access only for the deliberate pinned model/image population step; and
- explicit provider, quota, and cost authorization.

Do not provision a paid host from this repository without that authorization. Mirror the pinned
container images into a controlled registry before a production rollout.

Check out the exact reviewed TicketTune implementation commit recorded in the latest handoff. From
the repository root, install the locked training environment:

```bash
uv sync --frozen --extra train
```

Record `git rev-parse HEAD`, `git status --short`, GPU model, driver, CUDA version, host memory,
free disk, Docker version, Compose version, and NVIDIA runtime availability. The working tree must
be clean before training.

## 2. Install reviewed data without committing it

Generate or transfer the exact frozen synthetic source at
`data/qualified/support_tickets.jsonl`. The deterministic generator is fixed to seed 42 and 1,120
rows; do not replace it with real customer messages.

Validate and deterministically prepare the source, then scaffold packet-backed v1.2 evidence:

```bash
uv run tickettune data validate --config configs/qwen-7b-qlora-quality.yaml --json
uv run tickettune data prepare --config configs/qwen-7b-qlora-quality.yaml --json
uv run tickettune qualify scaffold-review \
  --config configs/qwen-7b-qlora-quality.yaml \
  --output-dir data/qualified/review-evidence
```

Give `reviewer-a.json` and `reviewer-b.json` to two distinct real people. Each person independently
reviews all 1,120 rows in source order and approves or rejects the label, response, PII, and license
decision. Placeholder reviewer IDs, missing dates, draft packets, incomplete decisions, repeated
reviewer identity, changed source bytes, or a changed holdout all fail closed. Automated audits do
not count as either person.

After both people finish, set the packet and aggregate dates/statuses honestly. Bind the edited
packet hashes into a new aggregate; binding never changes a decision or grants approval:

```bash
uv run tickettune qualify bind-review \
  --review-manifest data/qualified/review-evidence/review-manifest.json \
  --output data/qualified/review-evidence/review-manifest.bound.json
```

Run the gate and retain its JSON report:

```bash
uv run tickettune qualify dataset \
  --config configs/qwen-7b-qlora-quality.yaml \
  --review-manifest data/qualified/review-evidence/review-manifest.bound.json \
  --output artifacts/validation/dataset-qualification.json \
  --enforce \
  --json
```

A green report is review-attestation evidence only. It does not establish representativeness of
live traffic or model quality. The quality profile re-runs this exact packet/hash gate and requires
the frozen holdout to equal the verified prepared test split before a real training command can
create its run directory or import the ML stack. See `docs/qualification.md` for the field-level
review rules.

## 3. Prepare and preflight

```bash
uv run tickettune doctor --config configs/qwen-7b-qlora-quality.yaml --strict --json
uv run tickettune train --config configs/qwen-7b-qlora-quality.yaml --dry-run --json
```

Require `compatible: true`, accelerator `cuda`, bfloat16 support, the exact Qwen revision
`a09a35458c702b33eeacc393d103063234e8bc28`, verified prepared-data hashes, and sufficient free
VRAM/disk. Stop before downloading weights if any check fails.

## 4. Train one immutable QLoRA adapter

For the first deliberate fetch of the pinned revision:

```bash
uv run tickettune train \
  --config configs/qwen-7b-qlora-quality.yaml \
  --allow-download \
  --json
```

For a populated offline cache, omit `--allow-download`. Retain the returned adapter and sibling
run manifest. Require status `completed`, method `qlora`, the exact source/split/config hashes,
`git_dirty: false`, the reviewed Git revision, CUDA hardware facts, scalar training metrics, peak
memory, a positive integer `optimizer_steps`, and Safetensors hashes. Compute capability must be a
finite observed value of at least 6.0; `nan`, infinities, missing versions, CPU/MPS execution, and
declared-only rehearsal facts fail release validation.

## 5. Enforce absolute and non-regression quality

Use the immutable adapter path returned by training:

```bash
uv run tickettune evaluate \
  --config configs/qwen-7b-qlora-quality.yaml \
  --adapter artifacts/qwen-7b-quality/runs/TICKETTUNE_RUN_ID/adapter \
  --compare-baseline \
  --enforce-thresholds \
  --json
```

Replace `TICKETTUNE_RUN_ID` with the exact immutable run ID. Require every absolute threshold and
every candidate-minus-baseline non-regression decision to pass. Archive the evaluation manifest,
candidate/baseline report hashes, ordered held-out IDs, and aggregate metrics. Keep raw generated
text in restricted runtime storage, not Git.

## 6. Safe merge and functional parity

Merge into a new destination using the same pinned base revision:

```bash
uv run tickettune merge \
  --config configs/qwen-7b-qlora-quality.yaml \
  --adapter artifacts/qwen-7b-quality/runs/TICKETTUNE_RUN_ID/adapter \
  --output artifacts/qwen-7b-quality/deployments/TICKETTUNE_RUN_ID/merged \
  --json
```

Then run live parity with the same deterministic held-out prompts:

```bash
uv run tickettune parity verify \
  --config configs/qwen-7b-qlora-quality.yaml \
  --adapter artifacts/qwen-7b-quality/runs/TICKETTUNE_RUN_ID/adapter \
  --merged-model artifacts/qwen-7b-quality/deployments/TICKETTUNE_RUN_ID/merged \
  --output artifacts/qwen-7b-quality/deployments/TICKETTUNE_RUN_ID/parity.json \
  --enforce \
  --json
```

Live verification has no release-lineage bypass. Require the completed sibling training manifest
to match the active quality config, source/prepared/qualification hashes, adapter bytes, and merge
provenance. Retain the immutable adapter/merged prediction sidecars beside the final report in
restricted artifact storage; do not commit their model text.

Require strict schema validity on both sides and exact category, priority, sentiment, and next
action matches for every held-out ID. Response-text exactness is diagnostic; routing parity is a
gate.

## 7. Start the private production reference

Create the API-key, TLS-certificate, TLS-key, and Alertmanager webhook URL files outside Git. Put
them in the numeric `SECRET_GROUP_ID` and set exact mode `0440`. Copy
`deploy/vllm/production/.env.example` to an ignored release-specific env file, set the exact
adapter/cache/release paths, and validate Compose before launch:

```bash
docker compose \
  --env-file deploy/vllm/production/release.env \
  -f deploy/vllm/production/compose.yaml \
  config --quiet
```

Before launch, create a schema `2.0` release manifest that binds the exact Compose, env, adapter,
prepared dataset manifest, exact prepared test split, completed clean training manifest, passing
qualification report, completed passing evaluation manifest/report, and passing parity report.
Keep the referenced candidate/baseline evaluation artifacts and both parity prediction sidecars in
the same operator-controlled evidence package. Store the complete release in operator-enforced
immutable, content-addressed storage. Validation binds the reviewed cohort to the real test bytes
and recomputes evaluation/parity evidence; then use the only supported production start path:

```bash
uv run tickettune deploy validate-release \
  --manifest /secure/releases/current/release-manifest.json \
  --output artifacts/validation/release-validation.json \
  --json

uv run tickettune deploy start-release \
  --manifest /secure/releases/current/release-manifest.json \
  --execute \
  --json
```

The launcher requires the literal `tickettune-production` slot and exact approved profile, starts
with a clean environment against the local Docker socket, and rejects every placeholder or
evidence mismatch. Its final handoff is still path based, so read back actual running image
digests, labels, mounts, release ID, and in-container adapter digest. Keep vLLM, Prometheus, and
Alertmanager off host ports; only the NGINX TLS gateway may publish. Firewall the gateway to the
approved ingress source. vLLM's non-API endpoints must remain internal. Prove that the gateway
cannot use its non-masqueraded edge network for outbound traffic and that only Alertmanager can
reach the approved notification destination.

## 8. Prove readback, load, alerts, and rollback

Run authenticated TLS readback and a bounded acceptance load through the gateway:

```bash
uv run tickettune deploy readback \
  --base-url https://MODEL_HOSTNAME \
  --api-key-file /secure/path/tickettune-api-key \
  --ca-cert /secure/path/issuer-ca.pem \
  --model tickettune-qwen-7b-quality \
  --expected-base-model Qwen/Qwen2.5-7B-Instruct \
  --allow-remote \
  --output artifacts/validation/production-readback.json \
  --enforce \
  --json

uv run tickettune deploy load-test \
  --base-url https://MODEL_HOSTNAME \
  --api-key-file /secure/path/tickettune-api-key \
  --ca-cert /secure/path/issuer-ca.pem \
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

Replace `MODEL_HOSTNAME` with the approved TLS name. Confirm Prometheus target health, queue and
availability alerts, retention, and the external alert receiver. The API report proves authenticated
endpoint-reported name/parent claims and schema behavior, not adapter bytes by itself; combine it
with the launcher and container readback above. Trigger a controlled alert and record its receipt.
Produce the rollback plan from exact compatible current and previous release manifests, review the
argv, then rehearse it during a maintenance window:

```bash
uv run tickettune deploy rollback-plan \
  --current /secure/releases/current.json \
  --previous /secure/releases/previous.json \
  --output artifacts/validation/rollback-plan.json \
  --json
```

The command does not execute rollback. A human operator must confirm traffic-drain, port/ingress
switching, current-state backup, and the permitted downtime before executing the rendered argv.

## 9. Return evidence

Return only sanitized summaries to Git:

- exact source/review/prepared-data hashes and qualification decisions;
- clean Git revision and immutable training run manifest;
- candidate/baseline aggregate reports and decisions;
- safe-merge provenance and parity report;
- container platform digests, GPU/runtime facts, and Compose validation;
- redacted readback/load/alert/rollback results; and
- explicit hosted-dev versus production endpoint and time window.

Never commit dataset rows, model weights, raw generations, prompts, API keys, certificates, env
files, cache contents, cloud identifiers, or provider credentials.
