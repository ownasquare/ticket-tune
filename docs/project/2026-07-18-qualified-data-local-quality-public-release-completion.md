# Qualified data, local quality, and public-release completion

Date: 2026-07-18

## Outcome

TicketTune now has a complete, reproducible small-model fine-tuning path that a new contributor can
install, understand, run, inspect, and extend. The project generated a deterministic 1,120-record
synthetic candidate, prepared an exact 896/112/112 split, completed a real Qwen2.5 0.5B CPU LoRA
run, passed a single frozen-test comparison against the base model, merged safely, and passed full
adapter-versus-merged parity.

The codebase is ready for public source adoption. The trained artifact is deliberately **not**
called release-qualified because two real human reviews, real Linux CUDA QLoRA execution, and live
vLLM acceptance remain external gates.

## Completed implementation

- Added a fixed 7-category × 4-priority × 5-sentiment × 8-variant generator: 1,120 unique records.
- Bound deterministic source, prepared splits, holdout identity, and training/evaluation lineage by
  SHA-256.
- Added bounded large-dataset split selection without changing existing small-corpus hashes.
- Added packet-backed qualification schema v1.2 with two distinct human reviewer packets and four
  decisions per row: labels, response, PII, and license.
- Added private review scaffolding and a hash-binding command that never grants approval.
- Made legacy count-only v1.1 evidence permanently unqualified.
- Bound reviewer packets, prepared manifest, and holdout freeze through training and release proof.
- Added two independent automated corpus audits while explicitly preventing either from counting
  as human review.
- Added a deterministic committed-source public exporter that excludes internal handoffs, plans,
  ignored weights, local paths, and dirty-tree ambiguity.
- Added a pinned full-history secret scanner, reviewed false-positive baseline, CI job, contributor
  guidance, and source-distribution coverage.
- Updated README, qualification, training, security, and CUDA runbook paths to one packet-backed
  workflow.

## Dataset evidence

| Evidence | Result |
| --- | --- |
| Source records | 1,120 deterministic CC0 synthetic records |
| Source SHA-256 | `611ae32a0ee3304cca87d3ab610496cc08133a5cf457239fc79574247a46f5b6` |
| Prepared manifest SHA-256 | `479bdf2dbfd93502c7416464e0ddd50fcb26f41b65d6f1eb42b208ca1fe07daa` |
| Train | 896 records, `49ec9cd7506f30ebaaf0bda9814daa331631780eb851af9e747191c87edb2366` |
| Validation | 112 records, `121d7c5eebbdcbdc5339da3c04f847e874a2540ad88449206ea292492760c562` |
| Frozen test | 112 records, `e0cf0900fd3f5def7835fd920ef58440f6817acd7352147e6a629e512945aef3` |
| Automated audit A | 1,120/1,120 structural, provenance, privacy, and safety checks passed |
| Automated audit B | 10,080/10,080 semantic policy checks passed |
| Human review | Pending; two distinct people must independently approve every row |

Automated audits are engineering controls only. They have no release-qualification effect.

## Real Qwen 0.5B training

- Profile: `configs/qwen-0.5b-candidate-local.yaml`
- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Pinned revision: `7ae557604adf67be50417f59c2c2f167def9a775`
- Clean source revision: `196be11ad42771e32e913c39619fb891c22a3623`
- Run ID: `20260718T222130.824188Z-a3827144ce4d`
- Method: CPU FP32 LoRA, completion-only loss, no model download
- Execution: 3 epochs, 336 optimizer steps, 2,692.6495 seconds
- Best checkpoint: step 336, validation loss `0.0007495026220567524`
- Adapter weights SHA-256: `30be9543276563ad2f5ca5da7a99e74486d09a85e42a7ffa4c9b1310bc38ac0f`
- Training manifest SHA-256: `e184c4221a62fb3849e03127ec0f2689f5966112fe3a1905b641865fabe2acc6`

The run manifest records `status=completed`, `git_dirty=false`, the exact model revision, unchanged
dataset hashes, positive optimizer steps, and immutable artifact hashes.

## One-shot held-out evaluation

The frozen 112-row test set was evaluated once after checkpoint selection. No threshold or model
setting was changed after the results were observed.

| Metric | Base model | Adapter | Required | Adapter result |
| --- | ---: | ---: | ---: | --- |
| Strict JSON | 0.982143 | 1.000000 | 0.97 | Pass |
| Schema valid | 0.392857 | 1.000000 | 0.97 | Pass |
| Category accuracy | 0.339286 | 1.000000 | 0.90 | Pass |
| Priority accuracy | 0.553571 | 1.000000 | 0.90 | Pass |
| Sentiment accuracy | 0.357143 | 1.000000 | 0.85 | Pass |
| Response policy | 0.973214 | 1.000000 | 0.97 | Pass |
| Exact match | 0.000000 | 1.000000 | Diagnostic | 112/112 |

- Evaluation ID: `20260718T230712.403488Z-daa7cf36d6b3`
- Evaluation manifest SHA-256: `a56408a7a2b5b97c171f5977d48a7350a1bc2e5368d94f2d1f3c2322fa36102a`
- Candidate predictions SHA-256: `1b15262430311092b93f7272b90cada438214e1af08bdf80891cff236b9810c0`
- Baseline predictions SHA-256: `3c7d1afe26a47ecaa3ff0c409ba93f79a570f9994cd0321a8cac46bae24656c5`

This proves the adapter learned the bounded deterministic synthetic task. It does not prove broad
out-of-template generalization or production fitness.

## Safe merge and parity

- Merge: real PEFT safe merge into FP32, safe serialization enabled.
- Merged weights SHA-256: `3e8d50020fa85d94e5a8a6a889ad539bd516f73ad4e8886f7158561cf5f8eb7e`
- Merge provenance SHA-256: `b41363c031d3f024d6e9d9f5fbd791930beaa6edc3a59f865a7a1f8a474c65d1`
- Lineage label: `unqualified_local_smoke_override_not_release_evidence`
- Parity report SHA-256: `596cdc5df91078adc20c67f5c5a57a28c4bec7533b404133f591323ddce86810`
- Parity examples: 112
- Adapter and merged schema validity: 1.0
- Category, priority, sentiment, next action, and routing match: 1.0
- Raw text, parsed object, and response match: 1.0
- Mismatched or contract-invalid IDs: none

Parity proves merge fidelity, not independent task quality. The evaluation above supplies the
separate task-quality proof.

## CUDA and vLLM fallback

The final 7B quality profile was prepared against the same exact split and passed every static
QLoRA contract gate:

- pinned Qwen 7B revision;
- four-bit NF4 with double quantization;
- bfloat16 compute declaration;
- completion-only loss;
- adapter target modules; and
- verified prepared-data bytes.

The rehearsal correctly records `executed_cuda=false`, `model_weights_loaded=false`,
`optimizer_steps=0`, `runtime_status=blocked_no_cuda`, and `release_eligible=false`. Its SHA-256 is
`c245fef8e80c8b026dc21203f4fcbf42ce58715f8ddf4d5cf899cb595a1da397`.

The vLLM fallback rendered an offline, loopback-only, pinned-revision LoRA launch plan for the real
0.5B adapter. It records `executed=false`, `execution_state=planned`, and
`proof_boundary=launch_plan_only`. No live vLLM health, inference, load, hosted, or production claim
is made.

## Public adoption and security

- README leads with an offline quickstart and a short proof table.
- Advanced qualification and deployment material is progressively disclosed.
- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, issue templates, and PR template are present.
- CI includes tests, packaging checks, static deployment checks, and a full-history secret scan.
- `make security` runs Bandit, dependency audit, and the pinned secret scanner.
- The reviewed secret baseline suppresses exact public digests and intentional negative-test values
  only; new findings fail.
- The public exporter reads committed Git blobs, requires a clean tree, excludes weights/internal
  handoffs/plans, rejects local absolute paths, and writes a deterministic manifest.

## GitHub publication

The sanitized public source is published at
[`ownasquare/ticket-tune`](https://github.com/ownasquare/ticket-tune) as a public repository. Its
history starts with one clean export commit authored through the account's GitHub noreply address;
the private development history, internal handoffs, ignored runtime artifacts, reviewer packets,
and trained weights are not published. The public export manifest binds the published files back to
the reviewed local source revision.

The first hosted secret-scan run exposed a manifest-specific circularity: the generated export
manifest is itself an inventory of high-entropy SHA-256 values, including the reviewed baseline's
hash. The scanner now ignores only strict `sha256` and `source_revision` fields in that one generated
manifest while continuing to scan every other field and detector type. Regression tests prove that
digest inventory passes and an embedded credential still fails.

The first hosted CUDA contract rehearsal also exposed a presentation-only test difference: GitHub's
Linux runner emitted ANSI-styled CLI help while the local runner did not. The help assertion now
forces styled output and removes styling before checking the safety language, preserving the same
release-evidence warning while making the contract deterministic across environments. The clean
public-install check also confirmed that Typer 0.27 no longer declares Click transitively, so Click
is now an explicit core dependency rather than an accidental training-environment dependency.

## Validation environment and boundaries

- Local environment: Apple Silicon, 36 GB unified memory, Torch 2.13.0.
- Real local execution: CPU FP32 LoRA training, base/adapter evaluation, safe merge, and full parity.
- Static-only proof: 7B CUDA QLoRA contract, vLLM launch plan, production configuration tests.
- Mock/fixture scope: offline unit and deployment-policy tests use explicit fixtures where no live
  provider or server exists.
- Not proven: two-human approval, NVIDIA CUDA kernels, bitsandbytes runtime behavior, live vLLM,
  authenticated TLS serving, load/soak, hosted deployment, or production readback.

## Remaining gates

1. Two distinct real people independently review all 1,120 rows, approve every decision, date and
   approve both packets and aggregate, bind hashes, and pass enforced v1.2 qualification.
2. Run the unchanged 7B profile on an authorized Linux NVIDIA host and capture real CUDA, memory,
   optimizer, adapter, evaluation, merge, and parity evidence.
3. Run vLLM health, model identity, schema-valid inference, load, monitoring, security, and rollback
   acceptance on that compatible host.

These are external acceptance or owner actions. They do not invalidate the completed local project,
but they prevent claims that the model or service is release-qualified or production-deployed.
