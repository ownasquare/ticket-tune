# Qualified-candidate evidence

This directory records reviewable evidence for TicketTune's deterministic 1,120-record synthetic
support-ticket candidate. Generate the corpus locally with `tickettune data generate-candidate`;
the large JSONL file and private reviewer packets are intentionally not committed.

## Current status

- Source SHA-256: `611ae32a0ee3304cca87d3ab610496cc08133a5cf457239fc79574247a46f5b6`
- Prepared split: 896 train, 112 validation, 112 frozen test records
- Automated audit A: pass, 1,120/1,120 structural, privacy, provenance, and safety checks
- Automated audit B: pass, 10,080/10,080 semantic policy checks
- Local Qwen 0.5B training: pass, 336 optimizer steps from a clean commit
- One-shot held-out evaluation: pass, 112/112 exact matches and every quality threshold passed
- FP32 safe merge parity: pass, 112/112 exact adapter-versus-merged outputs
- Human review: pending; two distinct people must independently approve every record
- Release qualification: **not granted**

The automated reports are useful engineering evidence, but they are not human review and have no
release-qualification effect. A model trained from this candidate remains local evaluation evidence
until the packet-backed v1.2 qualification gate passes.

## Files

- `automated-audit-a.json` binds deterministic generation, schema, uniqueness, privacy, provenance,
  and response-safety checks to the source hash.
- `automated-audit-b.json` independently checks category, priority, sentiment, next-action, and
  response-policy consistency across all records.
- `training-summary.json` records the clean source revision, dataset identities, selected checkpoint,
  and adapter hashes for the real CPU run.
- `evaluation-summary.json` records the single frozen-test comparison against the pinned base model.
- `merge-parity-summary.json` records the safe FP32 merge and enforced full-test fidelity result.
- `cuda-vllm-fallback-summary.json` records the passing static CUDA contract and launch-plan-only
  vLLM fallback, while preserving the missing-hardware boundary.

See `docs/qualification.md` for the short human-review workflow.
