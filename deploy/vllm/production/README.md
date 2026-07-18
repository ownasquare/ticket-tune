# Hardened vLLM production reference

This bundle is a fail-closed deployment reference for the quality-qualified TicketTune
`Qwen/Qwen2.5-7B-Instruct` LoRA adapter. NGINX is the only host-published service, vLLM has no
egress or host port, Prometheus observes vLLM through an internal gateway listener, and a pinned
Alertmanager sends notifications through the only egress-enabled service network.

Validating these files is **configuration proof only**. It does not prove CUDA or driver
compatibility, model quality, live authentication, certificate trust, firewall enforcement,
capacity, alert delivery, rollback, or a production deployment. Those require readback from the
target host and remain separate evidence.

## Immutable release inputs

- A reviewed Linux/amd64 NVIDIA host with Docker Compose and NVIDIA Container Toolkit.
- A quality-qualified immutable `artifacts/qwen-7b-quality/runs/<run-id>/adapter` directory.
- The exact base-model revision used for qualification. The example pins
  `a09a35458c702b33eeacc393d103063234e8bc28`.
- A pre-populated Hugging Face cache for that revision. The cache is mounted read-only and vLLM is
  forced offline, so runtime model downloads fail closed.
- A pre-created runtime-cache directory writable by container UID `2000`.
- A certificate chain and matching private key for the approved ingress name.
- A 32–4096 byte high-entropy vLLM API key made only from printable ASCII with no whitespace.
- An Alertmanager webhook URL stored as a file secret.
- A numeric host secret group ID shared with the deployment operator.

Copy `.env.example` to an ignored, release-specific environment file. Its release ID, all-zero Git
SHA, and all-zero adapter digest are deliberate sentinels. The Python entrypoint rejects them.
Replace every sentinel, path, and resource setting before startup, and never reuse the environment
file for another adapter.

The release validator treats `.env.example` as an approved operational ceiling: model context is
fixed at 2048, concurrency/rank/tensor-parallel/GPU-utilization overrides cannot exceed the tested
profile, service CPU/memory/shared-memory/retention-size caps cannot be raised, and Prometheus must
retain at least 24 hours. A different capacity profile requires a reviewed code change.

All bind mounts set `create_host_path: false`. Compose will not silently create an empty adapter,
cache, or configuration path; prepare each source path first. The Hugging Face cache must be
readable by UID `2000`, and the runtime cache must be a real directory writable by UID `2000`.
The entrypoint creates and write-tests `HOME`, `XDG_CACHE_HOME`, `VLLM_CACHE_ROOT`,
`TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR`, and `CUDA_CACHE_PATH` beneath that runtime root.

## Secret-file permissions

Keep all four secret files outside the checkout. Put them in the numeric group selected by
`SECRET_GROUP_ID` and make their mode exactly `0440`; the services receive that ID through
`group_add`. For example, on the target host:

```text
sudo chgrp <secret-group-id> /run/tickettune-secrets/vllm-api-key \
  /run/tickettune-secrets/tls-certificate.pem \
  /run/tickettune-secrets/tls-private-key.pem \
  /run/tickettune-secrets/alertmanager-webhook-url
sudo chmod 0440 /run/tickettune-secrets/vllm-api-key \
  /run/tickettune-secrets/tls-certificate.pem \
  /run/tickettune-secrets/tls-private-key.pem \
  /run/tickettune-secrets/alertmanager-webhook-url
```

The startup entrypoint rejects an API-key symlink, non-regular file, wrong group, wrong mode,
non-ASCII byte, whitespace, NUL, or out-of-range length. It passes the accepted key to vLLM only
in the child environment—never in Compose command arguments—and removes the secret-file path from
that child environment. Do not commit a key, private key, webhook URL, populated environment
file, cache, adapter, or generated runtime evidence.

## Adapter identity

`EXPECTED_ADAPTER_SHA256` is not the digest of one weights file. It is the SHA-256 of a canonical
inventory covering every adapter file. For each regular file, record its POSIX-relative `path`,
content `sha256`, and `size_bytes`; sort the list by path; serialize it with
`json.dumps(..., sort_keys=True, separators=(',', ':'), ensure_ascii=False)`; UTF-8 encode that
JSON; then hash the resulting bytes with SHA-256. The entrypoint rejects symlinks, FIFOs, devices,
sockets, empty inventories, and any digest mismatch before vLLM starts. Record the same digest in
the release manifest and the `com.tickettune.adapter-sha256` container label.

Copy the schema `2.0` `release-manifest.example.json` beside the immutable release environment
file. Replace every example digest with the SHA-256 of the exact Compose file, release environment
file, canonical adapter inventory, prepared dataset manifest, exact prepared test split, and
completed training manifest. It also binds a passing dataset qualification report, passing
evaluation manifest, passing evaluation report, and passing parity report. Replace every
`replace-with-...` evidence path, the release ID, project name, and served model as well. Preserve the candidate/baseline evaluation
prediction and scored-row artifacts and both parity prediction sidecars referenced by those
reports; release validation reopens them and recomputes their evidence.

Release validation must read every direct and referenced evidence file as a stable non-symlink
regular-file snapshot, match its recorded digest, bind the reviewed IDs and expected labels to the
actual prepared test bytes, recompute evaluation and parity results, and enforce the
artifact-specific `completed`, `qualified`, or `passed` result. It retains every indirect snapshot
for the immediate pre-start identity recheck. A path, digest, or mutually consistent claim alone is
not acceptance evidence. The manifest contains identity and hashes only; never add secret material,
prompts, or model output.

Keep the prepared manifest and test split together under their canonical names `manifest.json` and
`test.jsonl`. The manifest must also bind the train and validation hashes recorded by the training
run, although those two split bodies are not needed by the deployment launcher.

## Static validation

First validate the schema `2.0` release manifest, every bound evidence file, and their semantic
lineage. The optional output is created immutably and contains hashes and identities, not model
text or secrets:

```text
uv run tickettune deploy validate-release \
  --manifest /immutable/releases/release-manifest.json \
  --output artifacts/validation/release-validation.json \
  --json
```

This report captures a validated snapshot only. Re-run the same validation immediately before
deployment because external release inputs can change after the report is written.

Use the fail-closed launcher as the only supported production start path:

```text
uv run tickettune deploy start-release \
  --manifest /immutable/releases/release-manifest.json \
  --execute \
  --json
```

It accepts only the versioned `tickettune-vllm-production-v1` profile: the exact approved Compose,
NGINX, Prometheus, alert rules, Alertmanager, and vLLM-entrypoint bytes. It also requires every
mandatory Compose interpolation value in the hashed environment file, clears ambient Compose and
release overrides, fixes the project slot to `tickettune-production`, starts from an empty process
environment with a non-user home, explicitly targets `unix:///var/run/docker.sock`, rechecks all
evidence and adapter bytes, and then runs fixed `up -d --wait --remove-orphans` arguments without a
shell. Direct `docker compose up` is outside the supported release boundary. A successful launcher
receipt does not replace TLS readback or runtime proof.

The final handoff to Docker remains path based: Docker and vLLM reopen the validated paths. Store
the complete release in operator-enforced immutable, content-addressed storage, then read back the
actual running image digests, Compose labels, bind mounts, release ID, and in-container adapter
digest. The launcher receipt explicitly records this non-atomic boundary.

Render the Compose model without starting a GPU service:

```text
docker compose --env-file deploy/vllm/production/.env.example \
  -f deploy/vllm/production/compose.yaml config --quiet
```

Run `promtool check config` with the exact pinned Prometheus image:

```text
docker run --rm --platform linux/amd64 --entrypoint promtool \
  -v "$PWD/deploy/vllm/production:/etc/prometheus:ro" \
  prom/prometheus:v3.13.0@sha256:0e698e35e50d1ddc2d11a4a55b089fe62eb71358a5c204dfafd21bdf8ffe04b8 \
  check config /etc/prometheus/prometheus.yml
```

Run `amtool check-config` with the exact pinned Alertmanager image and a readable target-host URL
secret:

```text
docker run --rm --platform linux/amd64 --entrypoint amtool \
  -v "$PWD/deploy/vllm/production/alertmanager.yml:/etc/alertmanager/alertmanager.yml:ro" \
  -v "$ALERTMANAGER_WEBHOOK_URL_FILE:/run/secrets/alertmanager_webhook_url:ro" \
  prom/alertmanager:v0.33.1@sha256:a89f8d4520954079275441eecdb71444328bd90633dd4eddfc33b9ed657f349b \
  check-config /etc/alertmanager/alertmanager.yml
```

Validate NGINX syntax separately with its pinned image and test certificate material. Static
configuration checks stay separate from container healthchecks: healthchecks probe the live NGINX
metrics listener, vLLM health route, Prometheus readiness route, and Alertmanager readiness route.

The checked-in tests additionally verify pinned images, network membership, file-backed secrets,
capability drops, read-only roots, effective process/resource limits, route allowlisting, request
and context limits, redacted logging, TLS policy, canonical adapter verification, and cache
write-preflight behavior.

## Network and observability boundary

The `model_internal` and `observability_internal` networks are internal. Docker does not publish a
container port from an internal bridge, so `edge` is non-internal but disables IP masquerading.
That preserves host-published ingress while blocking ordinary routed container egress. Docker
publishes only NGINX TLS port `8443`; the target-host firewall must independently restrict that
port to the approved source ranges and block unexpected forwarding from the edge subnet. Capture
target-host firewall readback, a failed gateway-egress probe, and an authenticated TLS request as
deployment proof—Compose rendering alone does not prove either direction.

The public NGINX listener permits only `GET /v1/models` and `POST /v1/chat/completions`, forwarding
the Authorization header to vLLM. It returns `404` for every other path. A separate NGINX listener
on `9091` exposes exactly `/metrics` to `observability_internal`, clears Authorization, and returns
`404` for all other paths. Prometheus therefore never joins the model network, and vLLM never
joins the observability network.

Only Alertmanager joins the masqueraded `notification_egress` network, which exists solely to
reach the approved webhook. vLLM and Prometheus have no non-internal network; NGINX's edge bridge
has masquerading disabled. On the target host, prove the webhook destination allowlist and blocked
egress from the other three containers; a successful alert test proves delivery but does not by
itself prove the egress restriction.

Prometheus has no host port, scrapes its own metrics, scrapes Alertmanager, forwards alerts to
Alertmanager, and applies both time- and size-based retention. The storage-capacity alert assumes
the default 5 GB size cap; update and revalidate that threshold if the approved cap changes.
Alertmanager has no host port and reads its generic webhook URL through `url_file` from a Docker
secret.

## Request and runtime controls

NGINX caps bodies at 64 KiB, attaches request IDs, rate-limits by client, and logs only request ID,
method, normalized route, status, and timings. Authorization, query strings, request bodies,
prompts, and model responses are absent from the configured access-log format. Container logs use
bounded local rotation.

vLLM caps context length, concurrent sequences, batched tokens, GPU utilization, and LoRA rank.
Runtime LoRA mutation is disabled. Every container runs as a non-root UID with a read-only root
filesystem, all Linux capabilities dropped, `no-new-privileges`, PID/CPU/memory limits, and only
the minimum writable tmpfs or data volume it needs.

## Target-host release and rollback proof

1. Populate and inspect the pinned model cache through a separately approved restricted-egress
   workflow, then remove its write permission for runtime.
2. Pre-create the adapter and runtime-cache paths, configure the secret group, and verify every
   bind source because `create_host_path: false` intentionally fails on missing paths.
3. Record the release ID, clean Git revision, all image digests, model revision, canonical adapter
   digest, configuration hashes, certificate fingerprint, limits, and previous release ID.
4. Run the static checks above, verify the target-host firewall and egress rules, then start the
   pinned Compose release.
5. Perform authenticated TLS readback of `/v1/models` and `/v1/chat/completions`; verify the served
   name, parent claim, request IDs, and strict TicketTune response schema. This endpoint proof does
   not identify adapter bytes by itself; pair it with container labels, mounts, and digest readback.
6. Confirm all three Prometheus scrape targets, fire a bounded test alert, and prove receipt at the
   approved webhook without recording its secret URL.
7. Run approved load and soak tests, check GPU and queue behavior, then rehearse rollback to a
   compatible preceding release in the same project slot, served-model identity, base/revision,
   profile, immutable environment, adapter digest, and image set.

Production approval must explicitly name the DNS and CA policy, firewall allowlist, notification
egress allowlist, secret backend, limits, retention, SLOs, alert receiver, GPU budget, and rollback
authority. Repository configuration cannot make or prove those target-host decisions.
