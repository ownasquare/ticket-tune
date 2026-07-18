# CPU smoke proof

This directory is the small, tracked proof index for TicketTune's real CPU smoke run. It contains summaries, SHA-256 references, and one explicitly identified PII-free constrained response—no model weights, prompts, raw prediction corpus, sensitive responses, usernames, or device identifiers.

## Outcome

- Training completed one real offline LoRA optimizer step with pinned `Qwen/Qwen2.5-0.5B-Instruct` revision `7ae557604adf67be50417f59c2c2f167def9a775`.
- The immutable run is `20260718T031845.891541Z-35d6408e9cb4`; its adapter weights hash is `7d0f3473628c0958205f2dae3e7438a77466910f5e795acab83ed84d52d4c695`.
- The immutable evaluation is `20260718T032044.486658Z-1d098b688c10`. The candidate passed all zero-regression checks against the base model, including a `+0.285714` category-accuracy delta, but failed every configured absolute quality threshold. This smoke adapter is not a quality-qualified model.
- Safe merge, pinned llama.cpp conversion, Q4_K_M registration, and local loopback inference were exercised with Ollama `0.32.1`.
- Unconstrained Ollama inference produced JSON that failed the TicketTune domain schema; that full response was not retained. A second request using JSON-schema constraints passed, and its PII-free response is retained for independent validation.

## Files

- `training-summary.json` — model, dataset chain, run metrics, and adapter hashes.
- `evaluation-summary.json` — held-out candidate/baseline metrics, thresholds, and explicit failed quality gate.
- `deployment-summary.json` — safe-merge, GGUF, Ollama registration, and sanitized inference proof.
- `ollama-constrained-response.json` — the PII-free constrained response that was
  validated against `TriageOutput`, retained for independent replay.

All paths in the summaries are relative to the repository root. The large immutable artifacts remain under ignored `artifacts/`; these tracked summaries let reviewers verify the exact byte chain without committing model files.

## Proof boundary

This proves the pipeline can train, evaluate, byte-safely merge, convert, register, and serve a tiny local smoke model. The training manifest was created before the repository's first commit (`git_revision` is null and `git_dirty` is true), so exact code-commit provenance is not part of this run. Functional adapter-versus-merged output parity was not executed. Acceptable task quality, CUDA QLoRA training, vLLM runtime behavior, and production deployment also remain separate acceptance layers.
