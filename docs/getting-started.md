# Getting started

This guide takes you from a fresh checkout to an offline first success, then to an optional real
one-step LoRA run. The offline path does not need an account, GPU, or model download.

## 1. Prerequisites

- Git
- Python 3.12 or 3.13
- [uv](https://docs.astral.sh/uv/)
- macOS or Linux; Windows users should use WSL 2

Run all commands from the TicketTune repository root.

## 2. Get a first success

```bash
uv tool install .
tickettune quickstart
```

The command should finish with `TicketTune is ready` and short checks for the dataset, hardware,
training plan, and fixture-backed evaluation. It does not load or download model weights, and it
cleans up its temporary demo files when it finishes.

Use `--details` to show paths and the proof boundary, or `--json` for automation:

```bash
tickettune quickstart --details
tickettune quickstart --json
```

To keep the generated splits and reports for inspection, provide a new or empty workspace:

```bash
tickettune quickstart --workspace artifacts/quickstart-demo --details
```

## 3. Understand the generated files

| Path | Purpose | Commit it? |
| --- | --- | --- |
| `data/processed/` | Deterministic train, validation, and test splits | No; regenerate it |
| `artifacts/validation/` | Local validation and qualification reports | Usually no |
| `artifacts/<profile>/runs/<run-id>/` | Immutable training run, adapter, and manifest | No; store deliberately |
| `results/` | Small, sanitized proof summaries | Yes, after review |

The `latest-run.json` and `latest-evaluation.json` files are convenience pointers. The immutable
run directories and their manifests are the provenance records.

## 4. Run a tiny real fine-tune

This optional profile downloads `Qwen/Qwen2.5-0.5B-Instruct` and performs one CPU optimizer step.
It proves execution, not useful model quality.

```bash
uv sync --frozen --extra train
uv run --no-sync tickettune data prepare --config configs/cpu-smoke.yaml
uv run --no-sync tickettune doctor --config configs/cpu-smoke.yaml
uv run --no-sync tickettune train --config configs/cpu-smoke.yaml --allow-download
```

Training prints the exact adapter and manifest paths. Keep that immutable adapter path for
evaluation:

```bash
uv run --no-sync tickettune evaluate \
  --config configs/cpu-smoke.yaml \
  --adapter artifacts/cpu-smoke/runs/<run-id>/adapter \
  --compare-baseline
```

A one-step run is expected to miss the absolute quality thresholds. That is an honest diagnostic,
not a broken installation.

## 5. Create a separate starter project

Install the lightweight CLI once from the checkout, then copy its bundled support-triage starter
into an empty directory:

```bash
uv tool install .
tickettune init ../my-support-model
cd ../my-support-model
tickettune data prepare --config configs/tickettune.yaml
```

The command refuses a file, symlink, or non-empty destination. The generated project contains its
own config, synthetic data, fixture predictions, and next commands. The installed command and
generated files therefore keep working without checkout-owned sample paths.

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `uv` is not found | Install uv, reopen the terminal, and confirm `uv --version` works |
| Python version is rejected | Install Python 3.12 or 3.13, then rerun `uv sync --frozen` |
| A command tries to resolve packages again | Run the frozen sync once, then use `uv run --no-sync ...` |
| Model download is blocked | The quickstart is still available; real training needs explicit `--allow-download` or a populated cache |
| The model fills disk or memory | Use `configs/cpu-smoke.yaml`, clear only your own ignored artifacts/cache, or move to a larger machine |
| QLoRA says CUDA is unavailable | QLoRA requires supported Linux/NVIDIA CUDA; use LoRA locally or move the run to a GPU host |
| Llama access is denied | Accept the upstream license and authenticate with Hugging Face before an explicit download |
| Evaluation exits unsuccessfully | Read the report first; a real smoke run can execute correctly while failing quality thresholds |
| Merge or parity is rejected | Preserve the adapter/run manifest, use a new output path, and do not bypass failed lineage or parity gates |

Next: [customize the model or data](customize.md), read the [training guide](training.md), or
choose a [deployment path](deployment.md).
