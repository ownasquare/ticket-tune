# Qwen 0.5B quality, parity, and CUDA rehearsal completion

Date: 2026-07-18

## Outcome

TicketTune now has a meaningful, reproducible Qwen2.5 0.5B CPU LoRA lane, parity reports that
separate invalid outputs from actual adapter-to-merge drift, and a CUDA QLoRA contract rehearsal
that can run without NVIDIA hardware while remaining ineligible as CUDA or release evidence.

The engineering objective is complete. The trained local model is intentionally **not** promoted:
it passed schema, sentiment, response-policy, non-regression, and exact merge-fidelity checks, but
its category and priority accuracies remained below the unchanged `0.85` absolute thresholds.

## Delivered changes

- Added `configs/qwen-0.5b-lora-local.yaml` for eight-epoch FP32 CPU LoRA over the bundled teaching
  corpus, with distinct data, training, evaluation, merge, and parity paths.
- Expanded the shared prompt with the exact categories, priorities, sentiments, JSON keys,
  response policy, and lowercase-snake-case `next_action` contract.
- Reworked the synthetic corpus so sentiment is expressed in the customer text and every sentiment
  appears in every priority band; deterministic source, fixture, and split checks remain enforced.
- Rejects any rendered supervised example that exceeds the configured token budget instead of
  silently training on truncated labels.
- Saves and evaluates on the same interval, restores the lowest-validation-loss checkpoint, and
  records a positive integer optimizer-step count in completed training manifests.
- Separates raw-string equality, parsed-object equality, schema invalidity, actual routing drift,
  and release-blocking IDs in parity reports while keeping every required parity rate fail-closed.
- Carries verified merge dtype into parity and rejects a configured/merged precision mismatch.
- Adds `tickettune rehearse cuda`, a CPU/MPS-safe static QLoRA contract check with immutable truth
  markers and no model loading, optimizer execution, adapter, merge, or run manifest.
- Requires real Linux/CUDA facts, finite compute capability, bfloat16 compatibility, positive peak
  accelerator memory, and positive integer optimizer steps before QLoRA evidence can enter a
  release.
- Disables Hugging Face Hub telemetry before trainer construction. This removes the checkpoint-save
  network request at its source; no warning or error output was suppressed.

## Real 0.5B execution evidence

The corrected run was launched from clean source commit
`1edaf0a122ee0136025be272dfd9643dc027c625`.

| Evidence | Result |
| --- | --- |
| Run | `20260718T194628.319207Z-32c9008dd440`; 88 optimizer steps; 8 epochs; CPU FP32 LoRA |
| Run manifest | SHA-256 `607cd91336821ec789ba77f7f6e3589b0c65b313bb424215995d115e9486701c` |
| Dataset | 56 rows: 42 train, 7 validation, 7 fixed regression-test rows |
| Token budget | 49 train/validation examples checked; maximum 358 of 512 tokens; minimum 46 completion tokens |
| Best checkpoint | Step 50, validation loss `1.3699300289`; weights SHA-256 `a444ad03cc5938683296917dbc349ab1071800fd731149297f0b43be1a9d527e` |
| Final checkpoint | Step 88, validation loss `1.4917595387`; weights SHA-256 `6d7f62b82e994a65367c540ee20369b1e215a1bc60b0fbae3ec023c31e35ef36` |
| Exported adapter | Byte-identical to step 50; SHA-256 `a444ad03cc5938683296917dbc349ab1071800fd731149297f0b43be1a9d527e` |

Attempt 1 exposed two defects: its source labels coupled sentiment to priority, and its final epoch
was exported after validation loss had already worsened. Both defects were repaired before the
corrected run. Because those seven test rows informed the correction, the v2 comparison is labeled
regression evidence rather than a pristine held-out generalization claim.

## Quality decision

Evaluation `20260718T195220.440686Z-18a1c2f4f7f9` used the immutable seven-row v2 regression split.

| Metric | Candidate | Baseline | Gate result |
| --- | ---: | ---: | --- |
| Strict JSON rate | `1.000000` | `0.857143` | Passed |
| Schema-valid rate | `1.000000` | `0.142857` | Passed |
| Category accuracy | `0.714286` | `0.571429` | **Failed absolute `0.85` threshold** |
| Priority accuracy | `0.428571` | `0.142857` | **Failed absolute `0.85` threshold** |
| Sentiment accuracy | `1.000000` | `0.285714` | Passed |
| Response-policy rate | `1.000000` | `0.857143` | Passed |

Every configured non-regression comparison passed. The overall decision remained failed because
absolute category and priority quality did not pass. No threshold was weakened.

Tracked aggregate evidence is under `results/qwen-0.5b-local/`. Generated predictions and weights
remain local-only under ignored `artifacts/` paths.

## Merge and parity decision

- Safe FP32 merge completed with provenance SHA-256
  `a821fe4b8294fe47db3433aa14c38c499a5a1b964b5a160e99ef7e3310109690`.
- Merged model weights SHA-256:
  `f2fcd67d6d27f3d8da9de7d0a103889e4a7cbf2d72b7542b9fcd6c0e9a100fdc`.
- The lineage boundary is `unqualified_local_smoke_override_not_release_evidence`.
- Enforced live parity ID `7288d3e20f938110` passed on all seven examples.
- Both schema-valid rates and category, priority, sentiment, `next_action`, and routing match rates
  were `1.0`; raw prediction, parsed object, and response exactness were also `1.0`.
- There were no contract-invalid, release-blocked, or cross-side mismatch IDs.

This proves the merge preserved the adapter's behavior. It does not override the failed task-quality
decision or create release qualification.

## CUDA fallback evidence

The development `configs/qwen-7b-qlora.yaml` profile was prepared and rehearsed without model
loading. The final rehearsal report has SHA-256
`46d14705a9d52b21a18811addd356c9c3ae6d4a942bb7369c705bbaa6c02bd44`.

| Layer | Result |
| --- | --- |
| Static QLoRA contract | Passed |
| Observed runtime | `blocked_no_cuda` on Darwin/MPS |
| CUDA executed | `false` |
| Model weights loaded | `false` |
| Optimizer steps | `0` |
| Adapter or run manifest written | `false` |
| Release status | `ineligible_rehearsal`; `release_eligible=false` |

The rehearsal covers only the development profile and teaching data. It must be repeated for the
final quality profile after qualified data exists, then followed by real CUDA training. It is not a
CUDA-run equivalent in evidentiary terms; it is the safe contract-level fallback possible without
the hardware.

## Validation

Validation used the detached source with the repository's established Python 3.12 environment.

| Gate | Result |
| --- | --- |
| Ruff lint | Passed |
| Ruff format | Passed after one mechanical `export.py` formatting correction |
| Strict mypy | Passed across 20 source files |
| Bandit | Passed for `src` and `deploy` |
| Dependency audit | No known vulnerabilities; the local unpublished package is correctly listed as unauditable on PyPI |
| Full tests | **562 passed** |
| Coverage | **85.36%**, above the required 85% |
| Source distribution and wheel | Built `tickettune-0.1.0.tar.gz` and `tickettune-0.1.0-py3-none-any.whl` |
| Canonical wheel verifier | Passed; bundled starter assets and offline quickstart were self-contained |
| Literal isolated wheel install | Passed outside the checkout; `--help`, `advanced`, offline `quickstart`, and `rehearse cuda --help` all exited zero |

The validation source/evidence tip before this closeout document was
`3d08c5a4629131ad3864bf31d75c9dcfdb36c54e`.

## Acceptance and remaining gates

Implementation, local execution, parity, packaging, and hardware-free rehearsal are complete.
Release acceptance remains blocked by:

1. An independently reviewed release dataset (at least 1,000 approved records, two independent
   reviewers, complete required-field coverage, and at least 100 frozen holdout IDs).
2. A new independent evaluation cohort and a model/data iteration that passes category and priority
   absolute thresholds without regressions.
3. An approved Linux/NVIDIA host proving bitsandbytes kernels, finite compute capability, bfloat16
   behavior, allocator/peak memory, real 7B optimizer steps, and immutable adapter bytes.
4. Safe merge/live parity from that qualified run, followed by vLLM startup, identity readback,
   schema-valid inference, load/soak, rollback, and hosted acceptance evidence.

## Closeout accounting

- Planned Objective Count: 7 task groups.
- Completed Planned Objective Count: 7 task groups, with failed quality acceptance preserved as an
  explicit result rather than reclassified as success.
- Validated Additional Objective Count: 4 (corpus-label repair, best-checkpoint restoration,
  finite-CUDA-fact hardening, and telemetry source removal).
- Warning/Issue Triage: the only local formatting miss was corrected; the checkpoint-save Hub
  request was traced to TRL telemetry and removed at source.
- Warning Suppression Status: none.
- Terminal Next Step Classification: implementation complete; release acceptance requires new data
  and external CUDA/serving evidence.
- Extra Mile Candidate Cleanup: legacy parity language was corrected across public adoption,
  dataset, deployment, and sanitized-result documentation.
- Extra Mile Score: 4/5.
- Extra Mile Rationale: the work included a second real training run, best-checkpoint byte proof,
  installed-wheel verification, and a cross-document evidence audit beyond the minimal feature work.
