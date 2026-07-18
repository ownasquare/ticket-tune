# Security policy

## Supported versions

TicketTune is currently pre-1.0. Security fixes are applied to the latest
revision on the default branch. Historical snapshots and locally exported
models are not patched automatically.

## Reporting a vulnerability

Please use GitHub's private vulnerability reporting for the repository when it
is available. If it is not available, open a minimal issue asking the maintainer
for a private reporting channel; do not include exploit details, credentials,
private data, or vulnerable model artifacts in a public issue.

Include the affected revision, component, impact, prerequisites, a minimal
reproduction that contains no secrets or personal data, and any known
mitigation. Allow the maintainer time to investigate and coordinate a fix
before public disclosure.

## Scope

In scope are vulnerabilities in TicketTune's Python code, data validation,
artifact handling, generated deployment plans, included Compose configuration,
and example clients. Vulnerabilities in Transformers, PEFT, TRL, PyTorch,
vLLM, Ollama, llama.cpp, Docker, CUDA, model repositories, or base-model
behavior should also be reported to their upstream maintainers; a TicketTune
report is welcome when the project integrates them unsafely.

Model quality errors, prompt injection inherent to an untrusted model, and
operator exposure that bypasses the documented gateway are not by themselves
code vulnerabilities. Reports that demonstrate a bypass of a stated control,
release-evidence check, route allowlist, or secret boundary are in scope.

## Operator responsibility

The simple vLLM Compose asset remains loopback-only. The separate production
reference is a hardened starting point, not a managed service or proof of a live
deployment. Internet-facing use still requires an approved Linux/NVIDIA host,
trusted certificate issuance, host firewall and ingress policy, secret-manager
integration, image scanning, capacity planning, monitoring ownership, tested
alert delivery, incident response, backup/rollback rehearsal, and license review.

Generate an unpredictable API key of at least 32 ASCII bytes. Keep API keys,
TLS private keys, alert receiver URLs, provider credentials, real customer
tickets, private checkpoints, adapters, and raw model output outside Git.
Treat every model response as untrusted data: validate `TriageOutput`, preserve
human approval for consequential actions, bound request and response sizes, and
avoid recording prompt or completion bodies in gateway and proof logs.

Release manifests and qualification/evaluation/parity reports are integrity
evidence, not authorization. Verify their exact hashes immediately before use,
run from a clean reviewed revision, and do not infer hosted or production status
from static Compose validation.

## Repository secret checks

Run `make security` before publishing a change. The repository scanner checks
both currently tracked files and every reachable Git blob, so deleting a secret
from the latest revision does not remove it from history.

If a finding is a real credential, revoke or rotate it first, remove it from the
project, and coordinate any history rewrite with maintainers. If it is a public,
non-secret value such as an immutable artifact digest, add only that exact
detector-and-hash finding to `.secrets.baseline` with `is_secret: false`. Explain
the value and review in the pull request, then rerun `make security`.

Do not baseline credentials, private data, or uncertain values. Do not add broad
path, line, or value exclusions; disable a detector; or approve findings in bulk.
