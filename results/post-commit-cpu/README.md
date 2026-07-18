# Post-commit CPU proof

This directory is the sanitized proof index for a real one-step CPU LoRA run executed from a clean detached worktree at commit `67e254cc29d79827a53f64c1b9e16abad1b1af98`. The large ignored model artifacts remain under `artifacts/post-commit-cpu/`; this tracked index contains only hashes, metrics, relative paths, and explicit acceptance results.

## Outcome

- Training completed one optimizer step with pinned `Qwen/Qwen2.5-0.5B-Instruct` revision `7ae557604adf67be50417f59c2c2f167def9a775`.
- The training manifest records `git_dirty: false` and the exact source commit above.
- Held-out comparison improved category accuracy from `0.0` to `0.285714`, while both baseline and candidate correctly failed the configured absolute quality thresholds.
- Safe PEFT merge completed and emitted a complete hash inventory and merge-provenance record.
- The historical parity report failed its composite gate on all seven held-out prompts because both
  sides were schema-invalid. That legacy report did not retain enough semantics to prove seven real
  adapter-versus-merged routing drifts; it proves only that the artifact was correctly rejected.

## Files

- `training-summary.json` records clean source-control provenance, the dataset chain, training metrics, and adapter hashes.
- `evaluation-summary.json` records the immutable candidate/baseline comparison and its failed absolute quality gate.
- `merge-parity-summary.json` records the successful safe merge and the failed exact-parity gate.

## Proof boundary

This proves that the committed pipeline can execute real local training, evaluate immutable model
and dataset lineage, materialize a safe merge, and reject invalid composite parity evidence. It does
not establish actual cross-side drift for that legacy run, acceptable task quality, a qualified
1,000-record dataset, CUDA QLoRA, vLLM runtime health, or production readiness.
