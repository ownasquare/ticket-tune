# TicketTune qualified pipeline completion

Date: 2026-07-18

## Outcome

TicketTune is a complete, reproducible portfolio project for adapting a Qwen instruction model to structured customer-support triage. It includes deterministic synthetic-data preparation, PEFT LoRA and QLoRA training, hardware preflight, run-scoped immutable inputs, offline evaluation, adapter-versus-merged parity, Ollama export, a production-oriented vLLM stack, monitoring, rollback planning, and fail-closed release qualification.

The default portfolio smoke path uses `Qwen/Qwen2.5-0.5B-Instruct` so the repository can be exercised on Apple Silicon or CPU-class hardware. The quality profile targets `Qwen/Qwen2.5-7B-Instruct` with 4-bit QLoRA and requires an approved CUDA host plus an independently reviewed dataset before it can produce release evidence.

## Completed implementation

- Deterministic ticket validation, stratified train/validation/test preparation, manifest hashing, and split identity checks.
- A qualification gate requiring at least 1,000 reviewed records, two distinct reviewers, complete review coverage, at least 100 explicit held-out IDs, approval, and an assertion that no real customer data is present.
- LoRA and QLoRA training plans with pinned model revisions, hardware compatibility checks, strict dataset lineage, resumable run directories, and machine-readable run manifests.
- Stable, no-follow file reads and run-scoped read-only snapshots so post-verification path replacement cannot substitute training bytes.
- Private complete hard-link snapshots for adapter and merged-model inference inputs, with full inventories, inode checks, hashes, and cleanup.
- Schema, classification, response-policy, latency, baseline comparison, and exact adapter-versus-merged parity evaluation.
- Fail-closed merge, vLLM launch, Ollama export, release-manifest, readback, load-test, rollback, and lineage checks.
- A pinned production Compose stack with TLS gateway, internal-only model networking, health checks, resource ceilings, secrets, Prometheus rules, Alertmanager, release validation, and rollback tooling.
- Pinned Python dependencies and CI actions, dependency and source security audits, typed source, package builds, and user/operator documentation.

## Validation evidence

| Check | Result | Proof boundary |
| --- | --- | --- |
| Full Python suite with coverage | 483 passed; 85.18% total coverage | Local committed source and test fixtures |
| Ruff lint | Passed | Local committed source |
| Ruff formatting | 34 files already formatted | Local committed source |
| Mypy | Passed for 17 source files | Local committed source |
| Bandit | Passed for `src` and `deploy` | Static local scan |
| Dependency audit | No known vulnerabilities; the local `tickettune` package is not a PyPI dependency | Locked local environment |
| Lock verification | 119 packages resolved; lock check passed | Local lockfile |
| Project smoke workflow | Passed data validation/preparation, hardware doctor, training plan, fixture evaluation thresholds, vLLM launch plan, and Ollama export plan | Local smoke and plan-only deployment proof |
| Compose render | Passed | Static production configuration |
| Pinned NGINX validation | Passed | Containerized configuration syntax and startup validation |
| Pinned Prometheus validation | Passed; one config and four alert rules | Containerized monitoring configuration |
| Pinned Alertmanager validation | Passed; one receiver | Containerized alert configuration |
| Python source and wheel build | Passed | Local package artifacts |
| Source package contents | Includes user docs, production deployment files, and sanitized CPU-smoke summaries; excludes internal handoffs, plans, and readiness captures | Built source distribution |

## Proof boundaries and remaining acceptance gates

The repository and its local smoke workflow are complete. They do not claim that a 7B QLoRA model was trained or that the vLLM production service was started on this Mac.

The quality-profile release remains deliberately blocked until both of these independently verifiable inputs exist:

1. An approved CUDA environment with bfloat16 support and sufficient GPU memory for the configured Qwen 7B QLoRA and vLLM profiles.
2. An approved synthetic corpus of at least 1,000 records with the complete two-reviewer manifest and at least 100 explicit held-out IDs.

No cloud GPU or paid provider was provisioned because the task supplied neither provider credentials nor spending authority. The runbook in `docs/operations/cuda-qualification-runbook.md` defines the exact commands and evidence required on an approved host.

## Reproduction entry points

```bash
make setup
make smoke
make check
```

For a small real local training run, follow `docs/training.md` with `configs/cpu-smoke.yaml`. For a qualified quality run, follow `docs/operations/cuda-qualification-runbook.md`; do not use the local-smoke override as release evidence.

## Post-commit CPU evidence

The final post-commit adapter training, evaluation, merge, and parity evidence is recorded under `results/post-commit-cpu/`. Training ran from a clean detached worktree at commit `67e254cc29d79827a53f64c1b9e16abad1b1af98`; the immutable training manifest records `git_dirty: false`.

The one-step adapter improved held-out category accuracy from `0.0` to `0.285714` without regressing the configured comparison metrics, but it failed every absolute quality threshold. Safe merge completed, then the enforced adapter-versus-merged exact-parity gate failed on all seven prompts and exited non-zero. The merged model is therefore diagnostic local evidence only and is explicitly not release-eligible. These honest negative results demonstrate the fail-closed gates; they do not satisfy the 7B quality-profile release gates above.
