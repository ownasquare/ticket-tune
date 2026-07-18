# Training TicketTune

TicketTune fine-tunes an instruction model to turn a support conversation into one strict, bare
JSON triage object. The training path uses TRL supervised fine-tuning and a PEFT adapter while
keeping model downloads and model construction out of plan inspection. The framework-free plan API
does not probe hardware; the CLI dry-run adds a lazy hardware probe but never loads model weights.

## Supported methods

| Method | Base weights | Adapter | Intended hardware | Notes |
| --- | --- | --- | --- | --- |
| LoRA | Native model dtype | LoRA | CPU, Apple Silicon, or CUDA, subject to model size | Portable default for the small smoke profile |
| QLoRA | bitsandbytes 4-bit NF4 | LoRA | Compatible CUDA environment | Uses nested quantization and a bf16, fp16, or fp32 compute dtype |

Every default profile uses PEFT's `target_modules="all-linear"` contract. A profile may supply an
explicit target-module list; TicketTune preserves that list rather than silently replacing it.

The optional training dependencies are pinned as a tested compatibility set:

- Transformers 5.13.0
- TRL 1.7.1
- PEFT 0.19.1
- Datasets 5.0.0
- Accelerate 1.14.0
- bitsandbytes 0.49.2

Install them with:

```bash
uv sync --frozen --extra train
```

## Prepare data first

Training consumes the generated conversational prompt/completion splits, not the raw corpus.

```bash
uv run tickettune data prepare --config configs/smoke.yaml
```

Each row has a `prompt` list containing system and user messages and a `completion` list containing
the assistant JSON. `completion_only_loss=true` is mandatory, so system and user prompt tokens are
context rather than training targets.

### Dataset chain of custody

Preparation writes `manifest.json` beside the three generated splits. Before a real training run
imports Datasets, Torch, Transformers, PEFT, or TRL, TicketTune verifies the following chain:

- the manifest passes its strict, versioned schema;
- the configured source filename and exact SHA-256 match the manifest;
- the configured seed and train/validation/test fractions match;
- the manifest contains exactly the canonical `train.jsonl`, `validation.jsonl`, and `test.jsonl`
  names, with no traversal or split-file symlinks;
- declared split counts and ordered IDs are internally consistent, globally unique, and sum to the
  declared total; and
- all three split files have the exact declared byte hashes, row counts, and ordered IDs.

A missing, stale, edited, or mismatched manifest fails before an output directory or optional ML
library is touched. Re-run `tickettune data prepare` from the reviewed source and configuration;
do not edit a manifest to bless changed bytes.

### Review-gated quality profile

`configs/qwen-7b-qlora-quality.yaml` adds a required dataset-qualification declaration. Its
download-free plan reports whether the packet-backed v1.2 evidence is verified, missing, or
invalid, along with the source, review-manifest, prepared-data, holdout, and reviewer-packet hashes.
A real training run re-applies the policy before optional ML imports or run-directory allocation.

The quality run requires at least 1,000 valid synthetic records, an exact frozen test cohort of at
least 100 IDs, and two distinct real humans independently approving the labels, response, PII, and
license decision for every source row. Both dated packets and the aggregate must be approved and
bound to the exact source, prepared manifest, and holdout bytes. Automated audits are supplemental:
they never count as human review or create release eligibility. Legacy v1.1 count-only attestations
are always unqualified.

Follow [Dataset qualification](qualification.md) to generate the fixed 1,120-row candidate,
prepare it, scaffold private packets, bind the completed packet hashes, and enforce the gate. Smoke
and local-candidate profiles remain integration or measurement evidence only. Qualification does
not replace held-out evaluation, baseline comparison, merge parity, CUDA execution, or production
acceptance.

## Inspect a download-free plan

```bash
uv run tickettune train --config configs/smoke.yaml --dry-run
```

The plan resolves the model revision, source/manifest/train/validation/test hashes, data paths,
example counts, precision, sequence length, PEFT targets, QLoRA settings, output directory, and
resume checkpoint. `build_training_plan()` does not import Torch, Transformers, Datasets, PEFT, or
TRL; load a tokenizer or model; probe an accelerator; or claim that training occurred.

The `tickettune train --dry-run` CLI wraps that framework-free plan with `run_preflight()`. That
probe may lazily import Torch to report observed hardware. Dry-run mode does not call
`require_compatible()`, load model/tokenizer weights, construct `SFTTrainer`, or execute an
optimizer step, so an incompatible target profile can still be inspected safely.

The plan also reports `dataset_manifest_status` as `verified`, `missing`, or `invalid`, plus a
manifest digest or diagnostic as appropriate. This lightweight inspection is useful before an
expensive run, but it is not execution proof. A real run verifies the source, prepared manifest,
and all three prepared splits again immediately before the optional training stack is loaded.

The real run then copies the verified train and validation bytes through no-follow file descriptors
into read-only snapshots inside its unique run directory. TicketTune strictly parses those exact
snapshot bytes in memory and passes the resulting rows to `Dataset.from_list`, so Datasets never
reopens a mutable path. Each snapshot is rechecked after framework ingestion and around final
artifact inventory. The held-out test hash is recorded in plan and manifest lineage, but the test
split is never passed to the trainer.

At the Python API boundary, the equivalent call is:

```python
from pathlib import Path

from tickettune.config import load_config
from tickettune.training import build_training_plan

config = load_config(Path("configs/smoke.yaml"))
plan = build_training_plan(config, allow_download=False)
print(plan.model_dump(mode="json"))
```

## Run training

Run the hardware doctor before an expensive profile, then opt in to model access explicitly:

```bash
uv run tickettune data prepare --config configs/cpu-smoke.yaml
uv run tickettune doctor --config configs/cpu-smoke.yaml
uv run tickettune train --config configs/cpu-smoke.yaml --allow-download
```

The CPU smoke profile sets Transformers' public `use_cpu=true` argument and uses one optimizer
step. It is the most portable real-execution check. The regular smoke profile leaves placement to
Transformers and Accelerate, while the Apple profile intentionally targets MPS.

### Bounded Qwen 0.5B local profile

Use `configs/qwen-0.5b-lora-local.yaml` when you want a meaningful local LoRA exercise instead of
the one-step integration smoke check:

```bash
uv run tickettune data prepare --config configs/qwen-0.5b-lora-local.yaml
uv run tickettune train --config configs/qwen-0.5b-lora-local.yaml --dry-run --details
uv run tickettune train --config configs/qwen-0.5b-lora-local.yaml --details
```

This profile keeps the pinned Qwen 2.5 0.5B base model in FP32 on CPU, uses all 42 prepared
training examples for eight epochs with gradient accumulation, and retains the same absolute
held-out evaluation thresholds as the other profiles. Its processed data, training run,
evaluation, and merge directories are separate from `cpu-smoke`, so the one-step artifact remains
independently reproducible.

The bundled corpus keeps `sentiment` tied to expressed customer tone rather than incident
severity. Every tone appears across every priority band, and the shared prompt defines the
category, priority, sentiment, and `next_action` contracts. When validation data is present,
training evaluates and saves on the same interval, restores the lowest-validation-loss checkpoint,
and records a positive `optimizer_steps` count in the immutable run manifest.

The profile is a local demonstration over 56 synthetic rows (42 train, 7 validation, and 7 held
out for test). It does not establish generalization or release readiness. The reviewed 1,000-row
qualification policy, held-out quality evaluation, baseline comparison, merge parity, and real
target-hardware evidence remain separate release requirements; do not weaken them to accept a
small local run.

### Candidate-quality Qwen 0.5B profile

After generating the balanced 1,120-record review candidate, use
`configs/qwen-0.5b-candidate-local.yaml` for one larger local quality measurement:

```bash
uv run tickettune data generate-candidate \
  --output data/qualified/support_tickets.jsonl \
  --seed 42
uv run tickettune data prepare --config configs/qwen-0.5b-candidate-local.yaml
uv run tickettune train --config configs/qwen-0.5b-candidate-local.yaml --dry-run --details
uv run tickettune train --config configs/qwen-0.5b-candidate-local.yaml --details
```

This profile uses the same pinned 0.5B base and CPU FP32 LoRA, but trains for three epochs over the
896-row training split and applies the stricter 7B quality thresholds. It intentionally does not
accept or emit release qualification: the generated source is still a review candidate until two
independent humans approve every decision in separate, hash-bound per-record packets. Automated
audits do not satisfy that gate. Run the frozen 112-row test evaluation once; use validation loss
rather than test predictions for checkpoint selection or tuning.

`allow_download=false` remains useful when the pinned model and tokenizer revision are already in
the local Hugging Face cache. In that mode, the loaders receive `local_files_only=true` and fail
instead of reaching the network when a required artifact is missing. TicketTune also disables
third-party Hub telemetry before constructing the trainer, so checkpoint creation does not emit a
background network request.

Resume from a compatible Trainer checkpoint with:

```bash
uv run tickettune train \
  --config configs/qwen-7b-qlora.yaml \
  --resume-from-checkpoint artifacts/qwen-7b/runs/<run-id>/trainer/checkpoint-100 \
  --allow-download
```

Resume accepts only a checkpoint inside the same profile's immutable
`<output>/runs/<run-id>/trainer/` tree. TicketTune verifies the originating run manifest, config,
model/revision, method, seed, dataset hashes, exact checkpoint inventory, and trainer state before
loading it. Do not resume a checkpoint with different or edited provenance.

## Pinned TRL and PEFT integration contract

TicketTune follows the actual TRL 1.7.1 constructor rather than examples from older releases:

- `AutoModelForCausalLM` first loads the exact immutable revision with the explicit offline gate,
  `dtype`, and `device_map=None`; this avoids TRL's helper performing an unpinned config lookup.
- `SFTTrainer` receives the instantiated model, `processing_class`, and `peft_config`.
- `SFTConfig` uses `max_length` and `eval_strategy`.
- Warmup uses the current `warmup_steps` field; no deprecated `warmup_ratio` is passed.
- Profiles may set public `use_cpu=true` to keep execution out of visible GPU or MPS backends.
- QLoRA's direct `BitsAndBytesConfig` object is passed to that explicit model load, after which
  PEFT's `prepare_model_for_kbit_training` runs before TRL adds the adapter.
- The model-load arguments use `dtype`, not the removed `torch_dtype` alias.
- Every training path passes `device_map=None` explicitly because TRL's model factory otherwise
  defaults it to `"auto"`; distributed or accelerator placement remains under Trainer and
  Accelerate control.
- Qwen profiles use `<|im_end|>` as the EOS token, and the tokenizer's pad token falls back to that
  EOS token when necessary.
- PEFT receives `task_type="CAUSAL_LM"`. QLoRA uses 4-bit NF4, double quantization, and
  `target_modules="all-linear"` unless an explicit profile list is supplied.

These choices correspond to the pinned [TRL SFTTrainer source](https://github.com/huggingface/trl/blob/v1.7.1/trl/trainer/sft_trainer.py),
[TRL SFTConfig source](https://github.com/huggingface/trl/blob/v1.7.1/trl/trainer/sft_config.py),
and [PEFT 0.19.1 LoRA source](https://github.com/huggingface/peft/blob/v0.19.1/src/peft/tuners/lora/config.py).

## Outputs and immutable provenance

A successful run writes:

```text
<training.output_dir>/
├── latest-run.json                         # mutable convenience pointer
└── runs/
    └── <run-id>/                           # immutable execution directory
        ├── adapter/
        │   ├── adapter_config.json
        │   ├── adapter_model.safetensors
        │   └── tokenizer files
        ├── dataset/
        │   ├── train.jsonl                  # exact run-scoped trainer input
        │   └── validation.jsonl             # exact run-scoped evaluator input
        ├── trainer/
        │   ├── checkpoint-*                # only when save cadence creates one
        │   ├── train_results.json
        │   └── trainer_state.json
        └── manifest.json                   # sibling of adapter/, not inside it
```

The run manifest is a frozen, schema-versioned record containing:

- exact resolved configuration and its SHA-256 hash;
- base model identity and pinned revision;
- LoRA or QLoRA method and seed;
- source, prepared-dataset manifest, train, validation, and held-out test hashes;
- installed versions for the training stack without importing those packages;
- Python and platform metadata;
- the hardware preflight, observed accelerator, and effective execution accelerator;
- Git revision, branch, and dirty state when the run occurs inside a checkout;
- training duration and peak CUDA allocation when the backend exposes it;
- scalar training metrics;
- adapter, trainer, and run-scoped dataset artifact paths, byte sizes, and SHA-256 hashes;
- resume checkpoint or a sanitized failure description.

Manifest creation is atomic. Every invocation allocates a new run ID, so a repeat execution cannot
overwrite a prior adapter, trainer checkpoint, or manifest. A failed execution writes the failure
manifest inside its allocated run directory before re-raising the original error. The CLI returns
the exact adapter and manifest paths and refreshes `latest-run.json` only after success.

## Reproducibility checklist

1. Keep the model revision pinned to an immutable commit SHA.
2. Commit the reviewed source dataset, not generated processed splits or their manifest.
3. Retain the exact processed-manifest and split hashes in the immutable run manifest and tracked,
   sanitized `results/` summaries.
4. Keep the same seed, batch size, gradient accumulation, max length, and precision.
5. Compare package versions and accelerator class before interpreting metric differences.
6. Evaluate the held-out test split with the same chat template and deterministic generation
   settings described in [evaluation.md](evaluation.md).

Exact bit-for-bit GPU training is not guaranteed across hardware, kernel, driver, and distributed
runtime changes. The manifest makes those differences visible instead of implying equivalence.

## Validation and proof boundaries

The offline training tests use injected fakes and do not download or execute a model. They verify
the TRL/PEFT argument contract, QLoRA configuration, tokenizer setup, explicit target modules,
checkpoint propagation, artifact hashing, run-manifest immutability, dataset-manifest
chain-of-custody failures before optional imports, and the dry-run import boundary.

A green dry-run or offline test suite proves orchestration only. It does not prove:

- a real LoRA or QLoRA optimizer step completed;
- the selected model fits local RAM or VRAM;
- the adapter improves held-out quality;
- vLLM or Ollama can serve the artifact;
- any hosted or production deployment occurred.

Record each of those as a separate proof layer.

The project closeout separately executed the CPU smoke profile for one optimizer step and wrote an
immutable adapter/run manifest. That proves the local LoRA path. The corresponding held-out report
failed its configured absolute quality gates, so it is not evidence of production task quality or
of the unexecuted CUDA QLoRA profiles.
