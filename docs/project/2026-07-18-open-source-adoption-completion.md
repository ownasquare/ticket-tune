# TicketTune open-source adoption completion

Date: 2026-07-18

## Honest release assessment

Before this adoption pass, TicketTune was a technically complete portfolio project but not a
comfortable open-source product. The repository assumed a checkout, led with maintainer checks,
printed large payloads on the common path, shipped no starter-project command, and mixed current
proof with stale status text. An experienced ML engineer could operate it; a first-time user was
likely to hesitate or copy a command that did not work in their context.

TicketTune is now ready to promote as a **support-triage fine-tuning reference**. It is not
presented as a generic framework for every task. That narrower promise matches the implemented
schema, prompts, data rules, evaluation metrics, and serving contract.

## Adoption outcomes

| Newcomer need | Completed behavior |
| --- | --- |
| Understand the product | The 135-line README leads with one support-triage example and four core steps |
| Get a first success | `tickettune quickstart` runs offline, downloads no model, and cleans up its temporary files |
| Install once | `uv tool install .` provides a CLI that works outside the repository checkout |
| Start a real project | `tickettune init DIRECTORY` safely creates a config, 56 synthetic tickets, passing fixture predictions, README, and ignore rules |
| Avoid accidental overwrite | Project creation rejects files, symlinks, and non-empty directories and rolls back partial writes |
| See only the core | Main help keeps qualification, parity, merge, and release controls behind `tickettune advanced` |
| Get readable results | Quickstart and training use short human summaries; `--details` and `--json` preserve full and automated output |
| Learn without clutter | GitHub `<details>` and an accessible information tooltip keep secondary proof near the claim without dominating the page |
| Extend deliberately | `docs/customize.md` separates configuration, compatible support-ticket data, source-level task changes, and deployment backends |
| Trust the package | The wheel contains its starter assets, and CI imports and runs the workflow from the built wheel rather than checkout files |

## Public workflow

```bash
uv tool install .
tickettune quickstart
tickettune init my-support-model
cd my-support-model
tickettune data prepare --config configs/tickettune.yaml
```

Repository contributors can use the intentionally small Make surface:

```bash
make setup
make quickstart
make train
make evaluate-live ADAPTER=artifacts/<profile>/runs/<run-id>/adapter
```

Advanced qualification and deployment machinery remains available, but it no longer competes
with the first-run path.

## Extension boundary

The supported low-friction extensions are:

1. Copy a YAML profile to change the pinned model, LoRA settings, or training controls.
2. Point that profile at additional schema-compatible synthetic support-ticket JSONL.
3. Add an explicit exporter or serving backend with its own tests and runtime directory.

Changing labels, output fields, prompt semantics, or quality metrics is honestly documented as a
source-level task fork. A plugin system was not added merely to make the project appear generic;
the current reference remains easier to understand because the public promise matches the code.

## Validation evidence

The final implementation was validated from clean detached commit `431ce57`; its tree is
byte-equivalent to main commit `64fd4a9` before this documentation-only completion record.

| Check | Result | Proof boundary |
| --- | --- | --- |
| Full test suite and branch coverage | 512 passed; 85.07% | Clean committed source and local fixtures |
| Focused newcomer, CLI, and package tests | 79 passed | Starter safety, concise output, docs, and package contracts |
| Ruff lint and formatting | Passed; 38 files formatted | Static committed source |
| Mypy | Passed for 19 source files | Static typing |
| Bandit | Passed for `src` and `deploy` | Static source scan |
| Dependency audit | No known vulnerabilities; local package skipped because it is not a PyPI dependency | Locked local environment |
| Lock verification | Passed; 119 packages resolved | `uv.lock` |
| Source and wheel build | Passed | Local package artifacts |
| Installed-wheel first run | Passed with 5 bundled resource groups and 56 starter rows; no checkout fallback | Built wheel imported directly |
| Offline quickstart | Passed data validation, 42/7/7 preparation, hardware inspection, training-plan validation, and 3-row fixture evaluation | No model download or training |

CI now repeats the installed-wheel proof after every build on Python 3.12 and 3.13.

## Remaining external acceptance gates

This adoption work does not change the model-quality or hosting truth:

- The one-step 0.5B CPU adapter executed but failed the absolute quality thresholds.
- Its safe merge completed, but the historical composite parity gate rejected all seven
  schema-invalid rows. Later semantics showed that report cannot prove seven real cross-side
  drifts; the merged artifact remains rejected for quality and invalid-output evidence.
- Qwen 7B QLoRA still requires an approved Linux/NVIDIA CUDA host with bfloat16 support.
- A release-quality run still requires at least 1,000 approved synthetic records, two independent
  reviewers, complete coverage, and at least 100 explicit held-out IDs.
- No hosted or production serving claim has been made.

Those gates are intentionally visible in the README and remain separate from installation,
offline workflow, package, training-execution, and deployment-configuration proof.
