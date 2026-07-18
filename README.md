# TicketTune

TicketTune is an open-source **support-triage fine-tuning reference**. It shows how to turn a
small instruction model into a specialist that returns one predictable JSON object for every
support ticket—then measure it, merge it, and prepare it for local serving.

It includes:

- a synthetic, privacy-safe ticket dataset;
- reproducible LoRA and CUDA QLoRA profiles;
- held-out baseline-versus-adapter evaluation;
- safe merge, parity, Ollama, and vLLM paths; and
- manifests and release gates that keep proof claims honest.

Requirements: Git, Python 3.12 or 3.13, and [uv](https://docs.astral.sh/uv/).

```bash
uv tool install .
tickettune quickstart
```

The quickstart is offline: it validates and prepares the sample data, checks your hardware,
builds a training plan, and scores fixture predictions. It does not download model weights, and
its temporary demo files are cleaned up automatically.

## What the model learns

Input:

> I was charged twice for [INVOICE_ID]. Can you help?

Output:

```json
{
  "category": "billing",
  "priority": "high",
  "sentiment": "frustrated",
  "response": "I’m sorry about the duplicate charge. I’ll help review both transactions.",
  "next_action": "review_duplicate_charge"
}
```

Exactly one bare JSON object is the contract—no Markdown fence or surrounding prose.

## The four things you do

| Step | Command | Result |
| --- | --- | --- |
| 1. Quickstart | `make quickstart` | Proves the offline workflow without model weights |
| 2. Train | `make train` | Runs the small CPU LoRA profile and writes an immutable adapter |
| 3. Evaluate | `make evaluate-live ADAPTER=artifacts/.../adapter` | Compares that adapter with its baseline on the same held-out tickets |
| 4. Deploy | Follow [Deploy a model](docs/deployment.md) | Serves an adapter with vLLM or a safely merged GGUF with Ollama |

Start with the [getting-started guide](docs/getting-started.md) for expected output, artifact
locations, and quick fixes. To use your own model or data, see [Customize TicketTune](docs/customize.md).

## Pick the right profile

| Goal | Profile | Hardware |
| --- | --- | --- |
| Learn the workflow | `configs/smoke.yaml` | Any supported development machine; no weights in quickstart |
| Run a tiny real fine-tune | `configs/cpu-smoke.yaml` | CPU; model download and several GB of free disk |
| Run the meaningful local example | `configs/qwen-0.5b-lora-local.yaml` | CPU; 3 GB lower-bound working memory, plus headroom and model storage |
| Develop on a Mac | `configs/apple-silicon.yaml` | Apple Silicon with adequate unified memory |
| Fine-tune Qwen 7B | `configs/qwen-7b-qlora.yaml` | Linux with a supported NVIDIA CUDA GPU |
| Build a release candidate | `configs/qwen-7b-qlora-quality.yaml` | Approved CUDA host plus packet-qualified data |

QLoRA fails with an explanation when CUDA is unavailable; it is never silently downgraded to CPU.
Follow the [bounded Qwen 0.5B recipe](docs/training.md#bounded-qwen-05b-local-profile) for the
meaningful local profile.

## What is proven today

<abbr title="A passed check proves only the row it belongs to, not the rows below it.">ⓘ</abbr>
Proof layers stay separate so a successful dry run is never presented as a trained or deployed model.

| Proof layer | Status |
| --- | --- |
| Offline workflow, tests, packaging, and static deployment checks | Passed locally |
| Deterministic 1,120-row candidate and two automated audits | 1,120/1,120 structural checks and 10,080/10,080 semantic checks passed; human review pending |
| Clean 336-step Qwen 0.5B CPU LoRA and one-shot 112-row test | Executed; every absolute threshold passed and all 112 outputs exactly matched expected JSON |
| Safe FP32 merge and enforced adapter/merged parity | Passed every required and diagnostic rate; raw outputs matched on 112/112 examples |
| Prepared Qwen 7B CUDA QLoRA contract rehearsal | Static contract and exact prepared hashes passed; runtime truthfully blocked without CUDA and human approval |
| Real Qwen 7B CUDA QLoRA and live vLLM | Not yet run |
| Hosted or production serving | Not yet run |

The local run is strong bounded evidence for this deterministic synthetic task, but it is not a
release-qualified model. Two real people must still approve every dataset record, and real Linux
CUDA/vLLM acceptance has not occurred. See the
[qualified-candidate evidence](results/qualified-candidate/README.md), the earlier
[small-corpus experiment](results/qwen-0.5b-local/README.md), and the
[CUDA readiness evidence](results/readiness/README.md).

<details>
<summary><strong>Quality-profile data gate</strong></summary>

The release path starts with a fixed 1,120-row synthetic candidate:

```bash
uv run tickettune data generate-candidate \
  --output data/qualified/support_tickets.jsonl --seed 42
uv run tickettune data prepare --config configs/qwen-7b-qlora-quality.yaml
uv run tickettune qualify scaffold-review \
  --config configs/qwen-7b-qlora-quality.yaml \
  --output-dir data/qualified/review-evidence
```

Two different people then independently review every row in `reviewer-a.json` and
`reviewer-b.json`. After both packets and the aggregate are honestly approved, bind their hashes
and enforce qualification. Automated audits do not count as either reviewer, and no release is
eligible before both humans approve all 1,120 rows. The checked-in 56-row teaching dataset cannot
pass this gate. See [dataset qualification](docs/qualification.md) for the two final commands and
the exact approval rules.

</details>

<details>
<summary><strong>Production vLLM controls</strong></summary>

The production reference adds a private model network, TLS gateway, API key, monitoring, alerts,
resource ceilings, immutable release inputs, readback, load proof, and rollback planning. These
controls are available for an approved Linux/NVIDIA host; their presence is not runtime proof.
See the [production operator guide](deploy/vllm/production/README.md).

</details>

## Repository layout

```text
configs/          Ready-to-copy model and training profiles
data/             Synthetic source data; generated splits are ignored
src/tickettune/   Python package and CLI
deploy/           vLLM, Ollama, monitoring, and example clients
tests/            Offline unit and contract tests
results/          Small, sanitized proof summaries
artifacts/        Ignored local models, reports, and runtime evidence
```

## Documentation

Use the [curated documentation index](docs/README.md). New users usually need only:

- [Getting started](docs/getting-started.md)
- [Customizing models, data, or the task](docs/customize.md)
- [Deployment](docs/deployment.md)
- [Architecture](docs/architecture.md)
- [Dataset card](docs/dataset-card.md)

## Safety and license

The bundled dataset is synthetic and uses visible placeholders instead of real personal data.
Do not use real customer tickets just to satisfy a demo or release gate. Base-model licenses and
access terms remain upstream obligations.

Code is MIT licensed. The bundled synthetic dataset is CC0 1.0. See
[DATA_LICENSE.md](DATA_LICENSE.md), [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md),
[SECURITY.md](SECURITY.md), and [CONTRIBUTING.md](CONTRIBUTING.md).
