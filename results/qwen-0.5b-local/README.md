# Qwen 0.5B local quality and parity proof

This directory indexes a real CPU LoRA run from clean commit
`1edaf0a122ee0136025be272dfd9643dc027c625`. Large model artifacts and generated text remain in
local-only, ignored `artifacts/`; these tracked files contain only hashes, aggregate metrics, and
decisions.

## Outcome

- Training completed 88 optimizer steps. The exported adapter is byte-identical to checkpoint 50,
  which had the best validation loss (`1.369930`) rather than the final epoch (`1.491760`).
- The v2 corpus makes customer sentiment explicit and covers every sentiment across every priority
  band. The final adapter reached `1.0` strict JSON, schema validity, sentiment accuracy, and
  response-policy compliance.
- It improved every reported baseline metric, but category accuracy (`0.714286`) and priority
  accuracy (`0.428571`) remained below their `0.85` thresholds. The model is not release-ready.
- The FP32 safe merge completed. Enforced adapter-versus-merged parity passed every required rate;
  raw predictions matched on all seven examples.

Attempt 1 is retained in ignored artifacts as a failed experiment. It exposed a sentiment/priority
label coupling and exported the final epoch even after validation loss worsened; both problems were
fixed before this run. The seven attempt-1 test rows are regression evidence, not an unbiased v2
quality claim.

## Files

- `training-summary.json` records source control, data, checkpoint selection, token budget, and
  adapter hashes.
- `evaluation-summary.json` records candidate/baseline metrics and the failed absolute quality
  decision.
- `merge-parity-summary.json` records the safe merge and passing fidelity gate.

## Proof boundary

This proves meaningful local 0.5B training and exact FP32 merge fidelity on seven synthetic test
rows. It does not prove broad generalization, the reviewed 1,000-record release dataset, CUDA
QLoRA, vLLM runtime health, hosted serving, or production readiness.
