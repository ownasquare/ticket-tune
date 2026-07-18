# Contributing to TicketTune

Thanks for helping make task-specific fine-tuning easier to understand and reproduce. Prefer
small changes with a focused test and an honest proof boundary.

## Set up the project

Use Python 3.12 or 3.13 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --frozen --group dev
uv run --no-sync tickettune quickstart
make check
```

Install model-execution dependencies only when your change needs them:

```bash
uv sync --frozen --group dev --extra train
```

GGUF conversion work also needs the pinned conversion dependencies:

```bash
uv sync --frozen --group dev --extra train --extra convert
```

Offline tests must not download a model, call an API, require a GPU, or read ambient credentials.
Keep real-model, GPU, and serving checks opt-in with the existing test markers.

## Check for secrets

Run `make security` before opening a pull request. It checks the source, dependencies, tracked
files, and every reachable Git revision. If the secret scan reports a finding, follow the
[security policy](SECURITY.md#repository-secret-checks): rotate real credentials and baseline only
an exact, reviewed false positive. Never disable a detector or exclude a whole file to make the
check pass.

## Where to make a change

| Change | Start here | Guidance |
| --- | --- | --- |
| Dataset or examples | `data/`, `src/tickettune/data.py` | [Customization guide](docs/customize.md) |
| CLI or terminal output | `src/tickettune/cli.py` | Keep concise human output and stable `--json` output |
| Training or profiles | `src/tickettune/training.py`, `configs/` | [Training guide](docs/training.md) |
| Metrics or parity | `src/tickettune/evaluation.py`, `src/tickettune/parity.py` | [Evaluation guide](docs/evaluation.md) |
| Merge or serving | `src/tickettune/export.py`, `deploy/` | [Deployment guide](docs/deployment.md) |
| Public task schema | schema, prompts, data, evaluation, and tests together | [Source-level task changes](docs/customize.md#3-change-the-task-contract) |

## Change checklist

1. Add or update a test that describes the user-visible contract.
2. Implement the smallest complete change.
3. Run focused tests, then `make check` when the touched surface is ready.
4. Update the nearest user, architecture, dataset, training, evaluation, deployment, or security guide.
5. State what you actually ran: offline, fixture-backed, real-model, GPU, hosted, or production.

Keep heavy ML imports inside execution functions so quickstart, configuration, and planning stay
lightweight. Preserve deterministic seeds, atomic writes, Safetensors, strict types, and immutable
manifests. A dry run is planning proof; it is not completed training, merging, serving, or release
proof.

## Data and model safety

The bundled dataset is synthetic and CC0. New examples must contain no real customer data,
credentials, or identifying values. Use declared placeholders, preserve compatible licensing and
provenance, and run validation plus leakage checks. Do not add scraped or third-party data without
a documented license and provenance review.

Do not enable remote model code by default, add pickle checkpoints, or pass untrusted text through
a shell. Adapter use must verify the exact base model. Safe merge must reload an ordinary
non-quantized base and use PEFT safe merge.

Keep vLLM loopback-only by default with static adapter registration. For Qwen/Ollama, retain the
merged-Hugging-Face → pinned llama.cpp → GGUF path unless upstream support changes and focused
compatibility proof is added.

## Pull requests

Keep pull requests focused. Include the problem, approach, affected files, tests and audits,
artifact or data provenance changes, and hardware used for any real model run. List proof layers
you did not execute. Never attach credentials, private data, proprietary datasets, or unreviewed
model weights.
