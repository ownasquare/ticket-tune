# Portfolio demo

This walkthrough shows the core product in a few minutes without presenting a dry run as training
or a local smoke run as production quality.

## 1. Run the offline workflow

```bash
uv sync --frozen
uv run --no-sync tickettune quickstart --details
```

Point out four things in the result:

1. The synthetic dataset was validated and split deterministically.
2. Hardware compatibility was checked.
3. A complete training plan was built without downloading model weights.
4. Fixture predictions passed the same structured-output metrics used for a real model.

The output contract is exactly one bare JSON object with `category`, `priority`, `sentiment`,
`response`, and `next_action`. Extra prose or Markdown fences are failures.

## 2. Show the reproducibility chain

```bash
uv run --no-sync tickettune config show --config configs/smoke.yaml
uv run --no-sync tickettune data prepare --config configs/smoke.yaml
uv run --no-sync tickettune train --config configs/smoke.yaml --dry-run
```

The useful review points are the pinned model revision, fixed seed, prepared split hashes, exact
labels, LoRA targets, hardware decision, and immutable output paths. `--dry-run` validates the plan;
it does not train.

## 3. Optional: run a real tiny LoRA job

```bash
uv sync --frozen --extra train
uv run --no-sync tickettune data prepare --config configs/cpu-smoke.yaml
uv run --no-sync tickettune train --config configs/cpu-smoke.yaml --allow-download
```

The CLI prints the immutable adapter and sibling manifest. Compare that adapter with its baseline
on the same seven held-out tickets:

```bash
uv run --no-sync tickettune evaluate \
  --config configs/cpu-smoke.yaml \
  --adapter artifacts/cpu-smoke/runs/<run-id>/adapter \
  --compare-baseline
```

A one-step training job is an integration check, not a useful benchmark. Do not add
`--enforce-thresholds` merely to make this demo appear green; inspect and explain the report.

## 4. Explain deployment choices

- **vLLM:** serves the base plus named LoRA adapter through an OpenAI-compatible API on a
  Linux/NVIDIA host.
- **Ollama:** consumes a safely merged model converted to GGUF with a pinned llama.cpp revision.
- **Production reference:** adds TLS, authentication, private networking, monitoring, resource
  limits, readback, load proof, and rollback planning.

Use [the deployment guide](deployment.md) for commands. A rendered plan is not a live service, and
a healthy service is not proof of model quality.

## Current evidence to present honestly

| Layer | Result |
| --- | --- |
| Offline control plane | Passed |
| Clean one-step CPU LoRA run | Completed |
| Held-out absolute quality gate | Failed; category accuracy improved from 0% to 28.6% |
| Safe merge | Completed |
| Enforced adapter-versus-merged parity | Failed on all 7 held-out prompts |
| Release eligibility | Rejected |
| CUDA QLoRA, live vLLM, hosted production | Not run |

The failed parity gate is a useful result: the system retained the diagnostic evidence and refused
to label the merged artifact deployable. Exact sanitized results are in
[`results/post-commit-cpu/`](../results/post-commit-cpu/README.md).

## Useful discussion points

- Why LoRA makes task adaptation cheaper to train, store, and version.
- Why QLoRA is explicitly CUDA-gated instead of silently falling back.
- Why task metrics and a held-out baseline matter more than training loss alone.
- Why model revision, data hashes, manifests, and artifact inventories belong in one lineage.
- Why adapter/merged parity is a release gate rather than an assumed property of safe merge.
- How to extend the reference with a new dataset, task schema, or deployment target.

For a new-user walkthrough use [Getting started](getting-started.md); for extension paths use
[Customize TicketTune](customize.md).
