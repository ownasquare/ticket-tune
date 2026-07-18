# Customize TicketTune

TicketTune supports three extension levels. Start with configuration or data; change Python source
only when you are intentionally creating a different task contract.

| Goal | Extension level | Main files |
| --- | --- | --- |
| Change model, LoRA settings, or training length | Configuration | `configs/*.yaml` |
| Add support-ticket examples | Data | your JSONL source plus a copied config |
| Change labels, output fields, or task behavior | Source-level task change | schema, prompt, data, evaluation, and tests |

## 1. Change a model or hyperparameters

Copy the nearest profile instead of editing it in place:

```bash
cp configs/cpu-smoke.yaml configs/my-support-model.yaml
uv run --no-sync tickettune config show --config configs/my-support-model.yaml
uv run --no-sync tickettune doctor --config configs/my-support-model.yaml
uv run --no-sync tickettune train --config configs/my-support-model.yaml --dry-run
```

Common settings include the pinned model revision, LoRA rank and target modules, learning rate,
batching, epochs or maximum steps, precision, context length, seed, and output paths. Keep each
experiment in a new output directory. A dry run validates the plan but does not train a model.

Use LoRA for portable development. Choose QLoRA only for a supported Linux/NVIDIA CUDA host; the
4-bit profile intentionally does not fall back to CPU or Apple MPS.

## 2. Add synthetic support-ticket data

Point a copied profile at a new JSONL file. Each line needs a unique ID, three ordered chat
messages (`system`, `user`, `assistant`), the parsed `expected` object, synthetic provenance,
license information, and an explicit statement that no real customer data is present.

```json
{
  "id": "MY-0001",
  "messages": [
    {"role": "system", "content": "Return one support-triage JSON object."},
    {"role": "user", "content": "I was charged twice for [INVOICE_ID]."},
    {"role": "assistant", "content": "{\"category\":\"billing\",\"priority\":\"high\",\"sentiment\":\"frustrated\",\"response\":\"I’ll help review both transactions.\",\"next_action\":\"review_duplicate_charge\"}"}
  ],
  "expected": {
    "category": "billing",
    "priority": "high",
    "sentiment": "frustrated",
    "response": "I’ll help review both transactions.",
    "next_action": "review_duplicate_charge"
  },
  "provenance": {
    "source": "synthetic",
    "created_by": "Your project",
    "license": "CC0-1.0",
    "contains_real_customer_data": false
  },
  "pii_placeholders": ["[INVOICE_ID]"]
}
```

Validate before training:

```bash
uv run --no-sync tickettune data validate --config configs/my-support-model.yaml
uv run --no-sync tickettune data prepare --config configs/my-support-model.yaml
```

Validation rejects schema errors, undeclared placeholders, common unredacted identifiers,
duplicate IDs, and normalized-content duplicates. Read the [dataset card](dataset-card.md) before
changing label balance or split expectations. Never add real customer tickets simply to grow a
demo corpus.

## 3. Change the task contract

Changing categories, output fields, message roles, or success metrics is a source-level extension,
not a YAML-only customization. Update these surfaces together:

1. `src/tickettune/schemas.py` — input and output contracts.
2. `src/tickettune/prompts.py` — canonical system and generation prompts.
3. `src/tickettune/data.py` — validation, projection, balancing, and split rules.
4. `src/tickettune/evaluation.py` and `src/tickettune/parity.py` — metrics and release gates.
5. `tests/` — valid, invalid, leakage, metric, and CLI cases.
6. The dataset card, training guide, and examples that describe the public contract.

Do not keep old support-triage metrics if the new task cannot satisfy their meaning. Define the new
success criteria first, then make data, prompts, evaluation, and serving agree with them.

## Add a deployment target

A new exporter or serving backend should remain explicit and shell-safe:

- build and validate its artifact plan in `src/tickettune/export.py`;
- expose the smallest command in `src/tickettune/cli.py`;
- keep runtime assets under `deploy/<target>/`;
- verify base-model identity, revisions, hashes, and non-overwrite behavior; and
- add focused offline tests plus opt-in live tests for the real runtime.

Treat an emitted plan, created artifact, healthy process, valid response, and production-ready
service as separate proof layers. See [Deployment](deployment.md) and [Architecture](architecture.md)
for the existing vLLM and Ollama boundaries.

## Keep an extension easy to adopt

- Give the common path one clear command and put advanced flags behind `--help`.
- Keep defaults safe, local, and reproducible.
- Print paths to the adapter, manifest, report, and next command after a successful action.
- Preserve normal human output and `--json` for automation.
- Update the nearest guide and add a small example that works offline.

For project conventions and the change map, see [CONTRIBUTING.md](../CONTRIBUTING.md).
