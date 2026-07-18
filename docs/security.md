# Security and privacy design

## Security posture

TicketTune is a local training and serving reference, not an internet-facing
control plane. Its safe default is offline validation plus loopback-only
serving. Operators who widen network access own authentication, TLS,
authorization, abuse controls, logging policy, and patch management.

## Threat model

Protected assets include source data, model and adapter weights, model-provider
credentials, generated support responses, host GPU capacity, and the integrity
of evaluation reports. Relevant threats include:

- poisoned or malformed training rows;
- customer PII entering examples, prompts, reports, or logs;
- applying an adapter to the wrong base model;
- arbitrary code execution from a model repository;
- pickle deserialization or tampered model artifacts;
- shell or Modelfile directive injection through paths and names;
- unauthenticated remote inference or dynamic adapter loading;
- denial of service through long contexts, large batches, or GPU exhaustion;
- mutable dependencies producing unreproducible GGUF bytes;
- treating dry-run, fixture-backed, or localhost output as production proof.

## Implemented controls

### Dataset controls

- The bundled corpus is synthetic, CC0-licensed, and declares that it contains
  no real customer data.
- Explicit placeholders such as `[EMAIL]` and `[ORDER_ID]` replace PII-like
  values. Placeholder inventories are validated.
- Strict Pydantic schemas reject extra fields, malformed roles, invalid labels,
  non-finite numeric values, duplicate JSON object keys, and assistant output
  that differs from the gold object.
- One shared strict JSON decoder covers source rows, embedded assistant JSON,
  prepared rows, manifests, prediction envelopes, resume manifests, adapter
  metadata, and export provenance so nested duplicate keys cannot fall back to
  parser-specific last-key-wins behavior.
- Deterministic deduplication and split manifests make leakage auditable.
- Dataset preparation resolves source and output paths and rejects containment
  in either direction before creating or replacing any generated file.
- Prepared-data verification regenerates the deterministic projection from the
  hashed source and compares canonical split bytes, IDs, labels, and hashes;
  recomputing a manifest hash cannot bless edited training rows.

These checks reduce risk; they are not a substitute for a privacy review of a
new dataset. Never add real tickets without documented legal basis, retention,
redaction, access control, and a fresh privacy/security review.

### Model and adapter controls

- `trust_remote_code` defaults to false and export plans do not turn it on.
- Real merges set `local_files_only=true` for the base model and tokenizer by
  default. A network download requires an explicit operator opt-in and should
  be tied to a reviewed immutable model revision.
- Adapters and merged outputs use Safetensors. Pickle-based weight files are not
  accepted by the deployment planner.
- Merge and vLLM paths read `adapter_config.json`, require an exact base-model
  match, reject symlinked adapter configs or Safetensors weights, and verify
  that vLLM's rank limit covers the adapter rank.
- Ollama's Qwen path rejects direct adapter import and requires a safe merge
  into a pristine, non-quantized base before GGUF conversion.
- Merge output is built in a sibling temporary directory, provenance and hashes
  are written, and an existing destination is never overwritten.
- Merge and GGUF export require source/output directory isolation and re-hash
  their adapter or merged-model inputs before and after long-running work. A
  concurrent input change aborts before success provenance is published.
- Merge inventories use normalized relative paths recursively, reject symlinks
  at every depth, and refuse nested bytes absent from the signed provenance.

Model licenses are separate from this repository's MIT license. Review the
base model, dataset, and adapter licenses and any gated-model terms before
training, redistribution, or commercial use.

### Command and artifact controls

- External processes are represented as ordered argv arrays and launched with
  `shell=False`; no command string is evaluated.
- Model names, ports, ranks, dtypes, quantization values, and filenames are
  allowlisted or range-checked.
- Modelfile paths reject NULs, line breaks, whitespace, and non-GGUF suffixes.
  System text rejects delimiter injection.
- llama.cpp is pinned to full commit
  `aedb2a5e9ca3d4064148bbb919e0ddc0c1b70ab3`, and the plan records output
  checksums. Review and deliberately update this pin when security fixes land.
- vLLM Compose uses the exact `vllm/vllm-openai:v0.24.0` tag plus the verified
  `linux/amd64` manifest digest. Re-verify before changing the platform and
  mirror the digest into a controlled registry for production.
- vLLM Compose requires `ADAPTER_PATH`; it has no mutable or legacy default and
  must point at an immutable `runs/<run-id>/adapter` directory.
- GitHub Actions and the conversion toolchain use immutable action revisions
  and pinned dependency versions; CI audits both training and conversion
  extras.

### Serving controls

- Native vLLM argv binds to `127.0.0.1` by default. A non-loopback address is
  rejected unless the caller explicitly opts in.
- Docker Compose publishes `127.0.0.1:<port>` only. The container's
  `0.0.0.0` bind is inside the container namespace and is not a host-wide
  publication.
- The container runs as UID 2000, drops Linux capabilities, sets
  `no-new-privileges`, mounts the adapter read-only, and disables request logs
  and usage telemetry.
- Static LoRA registration is used. Runtime adapter-management endpoints are
  not enabled.
- Client examples reject remote URLs unless `--allow-remote` is explicit.

Loopback is not an authentication boundary against other local users or
processes. Shared hosts still require OS isolation and access control.

## Secrets

The default Qwen models do not require a Hugging Face token. Do not put tokens
in Compose files, committed `.env` files, CLI arguments, model manifests,
notebooks, logs, or support tickets. For a gated model, provide a short-lived,
least-privilege read token through the platform's secret mechanism and mount or
inject it only into the serving/training process that needs it.

Before publishing artifacts, inspect manifests and model cards for local paths,
user names, repository URLs, data samples, and credential material. Checksums
prove byte identity, not that bytes are safe to disclose.

## Internet-facing deployment requirements

Never expose vLLM directly. Its API key protects API prefixes, not every server
endpoint. The hardened reference under `deploy/vllm/production/` therefore
keeps the model network private and publishes only an NGINX TLS gateway with a
route allowlist, bounded bodies, rate and concurrency limits, timeouts, request
IDs, and logs that exclude request bodies. A separate metrics-only listener lets
Prometheus scrape without giving the monitoring container access to model API
routes; Alertmanager receives alerts through a file-backed receiver URL.

That reference still requires reviewed certificate issuance, host firewall and
ingress restrictions, per-tenant authorization where applicable, a supported
NVIDIA runtime, capacity planning, image scanning, tested alert delivery, and
an incident-response owner. Static Compose, NGINX, Prometheus, and Alertmanager
validation proves configuration shape only, not a healthy or production-ready
runtime.

Treat model output as untrusted data. Validate it against `TriageOutput` before
automation, escape it before rendering, and require human approval for refunds,
account changes, security decisions, or other consequential actions.

Production promotion must bind the exact adapter to a clean completed training
manifest, passing dataset-qualification evidence, passing absolute/baseline
evaluation evidence, and passing adapter-to-merged parity evidence. Re-read the
bound bytes immediately before launch; filenames and a plausible release label
are not provenance.

## Verification cadence

Run offline tests and static checks on every change. Re-run held-out evaluation
after any dataset, template, model, adapter, decoding, or dependency change.
Re-run merge parity after each merge. Re-scan dependencies and container images
before a release and on a regular maintenance schedule. Keep local, hosted,
and production proof explicitly separate.
