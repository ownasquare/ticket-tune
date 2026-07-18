# Evaluating TicketTune

TicketTune evaluates the behavior the adapter is intended to change: producing a complete, valid,
safe JSON triage object with correct routing labels. It does not substitute generic language-model
loss or a subjective sample for held-out task metrics.

## Score an existing prediction artifact

The fastest offline path uses the committed fixture and never loads a model:

```bash
uv run tickettune evaluate \
  --config configs/smoke.yaml \
  --predictions tests/fixtures/predictions.jsonl
```

Each JSONL row must contain:

```json
{
  "id": "EVAL-001",
  "expected": {
    "category": "billing",
    "priority": "high",
    "sentiment": "frustrated",
    "response": "I will review the duplicate charge and help correct it.",
    "next_action": "review_duplicate_charge"
  },
  "prediction": "{\"category\":\"billing\",...}",
  "latency_ms": 12.5
}
```

`latency_ms` is optional. File and schema errors include the source line number.

## Strict JSON contract and diagnostic recovery

Contract success requires the entire model response, after surrounding whitespace is removed, to
be exactly one bare JSON object. The per-example `strict_json_only` field records this result, and
`strict_json_rate` is an independently configured quality gate.

For diagnosis, the extractor can still recover one JSON object from these common invalid shapes:

- a fenced `json` code block;
- brief prose surrounding an object; or
- output containing multiple JSON objects.

It uses Python's JSON decoder rather than a brace-only regular expression, so escaped strings and
nested objects are handled correctly. A recovered non-bare object can contribute diagnostic field
completeness and label accuracy, but it never counts as strict JSON, schema-valid, or an exact
match. A bare object is validated with the same strict `TriageOutput` schema used by the dataset.
Unknown fields, missing fields, unknown labels, invalid next-action names, and out-of-range response
lengths make the schema metric fail. Output without a decodable object receives zero completeness
and label credit.

## Metrics

| Metric | Definition |
| --- | --- |
| Strict bare-JSON rate | Fraction whose entire output is exactly one JSON object |
| Schema-valid rate | Fraction that are both strict bare JSON and validate as one strict `TriageOutput` |
| Field completeness | Mean fraction of category, priority, sentiment, response, and next_action fields present and non-empty |
| Category accuracy | Exact category matches divided by all examples |
| Category macro F1 | Unweighted one-vs-rest F1 across category labels, including invalid predictions as errors |
| Priority accuracy | Exact priority matches divided by all examples |
| Priority macro F1 | Unweighted one-vs-rest F1 across priority labels |
| Sentiment accuracy | Exact sentiment matches divided by all examples |
| Sentiment macro F1 | Unweighted one-vs-rest F1 across sentiment labels |
| Response-policy rate | Fraction whose response is 12–1200 characters and contains no raw email, phone-like, SSN, or payment-card-like value |
| Exact-match rate | Fraction whose fully validated object equals the expected object |
| Mean, P50, P95 latency | Linear-interpolated millisecond summaries when latency is supplied |

Label accuracy is still calculated for a parsed but otherwise invalid object. This preserves useful
diagnostic signal—for example, a correct category with a missing response—without inflating the
schema-valid or exact-match rates.

Macro F1 is implemented locally, so offline evaluation does not add a scikit-learn dependency. Each
metric always averages over its complete canonical label set—seven categories, four priorities, or
five sentiments—even when a supplied evaluation artifact happens not to observe a label. Common
classes and incomplete fixtures therefore cannot hide failure on an absent or rare class.

## Quality gates

The configuration defines six minimum thresholds:

```yaml
evaluation:
  thresholds:
    strict_json_rate: 0.95
    schema_valid_rate: 0.95
    category_accuracy: 0.85
    priority_accuracy: 0.85
    sentiment_accuracy: 0.80
    response_policy_rate: 0.95
```

Every threshold is emitted with the observed value, minimum, and pass/fail decision. The report's
overall result passes only when all configured gates pass. The Python API returns failed reports by
default so artifacts remain available for diagnosis; callers that require an exception can use:

```python
from pathlib import Path

from tickettune.config import load_config
from tickettune.evaluation import evaluate_predictions

config = load_config(Path("configs/smoke.yaml"))
evaluate_predictions(
    config,
    Path("artifacts/predictions.jsonl"),
    raise_on_failure=True,
)
```

## Generated model evaluation

`run_model_evaluation` renders each held-out conversational prompt with the tokenizer's chat
template and `add_generation_prompt=true`. Generation is deterministic by default
(`do_sample=false`); temperature and top-p are supplied only when sampling is explicitly enabled.
It explicitly places the model on CUDA, MPS, or CPU and never uses `device_map="auto"`.

For `configs/qwen-0.5b-candidate-local.yaml`, checkpoint selection uses validation loss only. The
generated 112-row test cohort is a one-time final candidate measurement: once its predictions have
been inspected, do not tune against that cohort or describe a later rerun as unseen evidence.

Live generation is bound to the exact `test.jsonl` named by the prepared dataset manifest. Before
loading a tokenizer, base model, or PEFT library, TicketTune verifies the strict manifest schema,
configured source hash, seed and split fractions, canonical filenames, manifest counts and IDs,
and the test split's exact SHA-256 and ordered IDs. Passing another file through the Python
`dataset_path` argument is rejected even when its bytes happen to match. This prevents an
unrecorded held-out set from being substituted into an evaluation claim.

When an adapter is supplied, generation also reads `adapter_config.json` before optional imports.
Its `base_model_name_or_path` must match the configured base exactly, and any adapter-declared
revision must match the configured revision. Generation requires regular root-level Safetensors
weights and records exact digests for them and the adapter config. Export and serving retain their
stricter rank, safe-merge, format, and deployment validation.

Before Transformers or PEFT opens a verified local adapter or merged model, TicketTune creates a
private random hard-link snapshot tree on the same filesystem. Every linked inode and complete
file inventory must match the verified digests, and model libraries receive only that private
path. Directory or file-entry swap-and-restore races therefore cannot substitute different model
bytes during loading without duplicating multi-gigabyte weights.

Every generated prediction row carries the prepared dataset-manifest and test-split SHA-256,
generation-contract hash, configured model revision, and—when present—the adapter config and
Safetensors digests. Verified live adapters also carry training-manifest/config/dataset and
qualification hashes; merged generation carries merge/source lineage. `evaluate_predictions`
requires those values to be identical across all provenance-bearing rows and promotes the relevant
lineage into canonical reports. Mixed or conflicting provenance is rejected rather than silently
summarized.

Evaluate an adapter and retain a base-model comparison through the CLI, using the exact immutable
adapter path printed by training:

```bash
uv run tickettune evaluate \
  --config configs/cpu-smoke.yaml \
  --adapter artifacts/cpu-smoke/runs/<run-id>/adapter \
  --compare-baseline
```

Live adapter evaluation requires the adapter's sibling completed training manifest and carries its
config, dataset, qualification, split, generation, adapter-config, and adapter-weight hashes into
the report. `--allow-unverified-adapter` exists only for local fixture experiments; it cannot be
combined with `--enforce-thresholds` and is never valid for release evidence.

Do not add `--enforce-thresholds` to a one-step smoke run whose purpose is to preserve diagnostic
artifacts. Use it in a release gate only after the profile is expected to meet its configured
absolute thresholds and zero-regression comparison.

The equivalent Python API is:

```python
from pathlib import Path

from tickettune.config import load_config
from tickettune.evaluation import run_model_evaluation

config = load_config(Path("configs/cpu-smoke.yaml"))
result = run_model_evaluation(
    config,
    adapter_path=Path("artifacts/cpu-smoke/runs/<run-id>/adapter"),
    compare_baseline=True,
    allow_download=False,
)
print(result.model_dump(mode="json"))
```

With `allow_download=false`, both base and adapter loaders use local-only resolution. A missing
cached revision fails rather than opening a network path. Comparison output reports candidate minus
baseline deltas for schema validity, category/priority/sentiment accuracy and macro F1,
response-policy compliance, and exact match.

Scoring an already-produced file with `--predictions` does not load a model and intentionally does
not require the prepared-data manifest. That fixture path proves only evaluator behavior. The
manifest chain is mandatory for live base or adapter generation through `run_model_evaluation`.

## Adapter-to-merged parity gate

A successful merge proves that model files were written safely; it does not prove that the merged
model behaves like the base-plus-adapter runtime. TicketTune therefore compares both sides on the
same ordered held-out IDs with the same pinned base revision, prepared-data manifest, prompt
template, and deterministic generation settings.

For live verification:

```bash
uv run tickettune parity verify \
  --config configs/qwen-7b-qlora-quality.yaml \
  --adapter artifacts/qwen-7b-quality/runs/<run-id>/adapter \
  --merged-model artifacts/qwen-7b-quality/deployments/<run-id>/merged \
  --output artifacts/qwen-7b-quality/deployments/<run-id>/parity.json \
  --enforce \
  --json
```

The CLI has no live lineage bypass. It requires the adapter's completed sibling training manifest
to match the full active config, source/prepared hashes, and required qualification hashes; merge
provenance must bind that exact adapter config and weights. It writes immutable
`<stem>.adapter-predictions.jsonl` and `<stem>.merged-predictions.jsonl` sidecars beside the report
and rejects final or parent symlinks.

The gate requires strict `TriageOutput` JSON from both sides and exact matches for category,
priority, sentiment, and `next_action` on every ID. Exact response text is retained as a diagnostic
metric because equivalent safe wording may differ. The report binds both prediction byte streams,
adapter and merged-model inventories, merge provenance, active configuration, and prepared dataset.
Inputs are rechecked after inference. The final report waits for both sides; an interrupted run may
leave the immutable adapter-predictions sidecar, and a retry reuses it only when all semantic
prediction bytes match (latency is treated as telemetry rather than identity).

`tickettune parity compare` can inspect two already-produced prediction JSONL files. It is an
offline byte-and-contract check, not live inference proof: it validates each exact prediction file
and their embedded cross-side dataset, generation, adapter, merge, and optional training-lineage
contracts, but it does not reopen the external adapter, merge, dataset, or training-manifest paths.
`--enforce` gates the parity metrics; use live `parity verify` for release evidence.

```bash
uv run tickettune parity compare \
  --adapter-predictions artifacts/validation/adapter-predictions.jsonl \
  --merged-predictions artifacts/validation/merged-predictions.jsonl \
  --output artifacts/validation/offline-parity.json \
  --enforce \
  --json
```

Lineage, schema, symlink, and byte-integrity failures are unconditional errors. For live verify,
`--enforce` changes only whether a metric-gate failure exits non-zero; the CLI offers no
training-lineage bypass.

## Artifacts

Fixture scoring writes its report set atomically to the configured output directory (or an
explicit Python API override):

```text
<evaluation.output_dir>/
├── evaluation-report.json
├── evaluation-report.md
└── scored-predictions.jsonl
```

Generated adapter-versus-baseline evaluation additionally keeps the original candidate and baseline
prediction JSONL files and writes every execution into a new immutable directory:

```text
<evaluation.output_dir>/
├── latest-evaluation.json                  # mutable convenience pointer
└── runs/
    └── <evaluation-id>/
        ├── candidate-predictions.jsonl
        ├── candidate/
        │   ├── evaluation-report.json
        │   ├── evaluation-report.md
        │   └── scored-predictions.jsonl
        ├── baseline-predictions.jsonl      # when --compare-baseline is used
        ├── baseline/                       # when --compare-baseline is used
        └── evaluation-manifest.json        # hashes every generated artifact
```

The JSON report is the canonical machine-readable artifact; Markdown is a concise review surface;
scored JSONL preserves per-example parse errors and partial metrics for debugging. The mutable
pointer is for discovery only. Tracked, sanitized closeout summaries belong under `results/`; raw
generated reports remain ignored under `artifacts/`.

## Validation and proof boundaries

The committed offline tests cover raw, fenced, prose-wrapped, multiple-object, partial,
schema-invalid, and malformed outputs; exact label and object scoring; sentiment; email and phone
response PII policy; canonical-label macro F1; latency; strict and task threshold failure; line-aware
file errors; report persistence; manifest tamper and held-out substitution rejection before optional
imports; adapter base/revision mismatch rejection; and fake adapter generation through the shared
chat template.

Scoring a fixture proves the evaluator, not model quality. Generating locally proves only the exact
local model revision, adapter, prompt template, and runtime used. Neither result proves a vLLM or
Ollama deployment, hosted availability, production latency, or production data behavior.

The closeout CPU run generated candidate and baseline predictions for the same seven held-out IDs.
The candidate's strict-JSON and response-policy rates were `6/7`, category accuracy and canonical
macro F1 were `2/7`, and schema, priority, and sentiment gates failed. No comparison metric
regressed against the baseline, but the overall comparison still failed because the candidate did
not meet the absolute thresholds. This is honest one-step pipeline evidence, not a quality claim.
