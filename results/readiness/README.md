# TicketTune readiness evidence

This directory retains small, sanitized readiness summaries. It does not contain model weights,
generated customer-facing text, API keys, certificates, cloud account details, or model-provider
credentials.

## Current CUDA target audit

`cuda-target-audit.json` records the 2026-07-18 local target inspection. The Apple M3 Max host can
exercise MPS LoRA and CPU smoke paths, but it cannot run bitsandbytes QLoRA or an NVIDIA vLLM
container. Docker is available only through a Linux/arm64 VM with 12.5 GB assigned memory and no
NVIDIA runtime. The Qwen2.5 7B quality profile therefore fails its strict doctor check before a
model download or training allocation.

The same download-free plan also reports `data.qualification.required: true` with status `invalid`
because neither the ignored 1,000-plus-record synthetic quality corpus nor its approved review
manifest exists. Hardware access alone will not bypass that independent data-governance gate.

Azure CLI presence is recorded only as a binary-presence fact. No account, subscription, quota,
credential, or billable-resource state was queried. An approved provider/host and spending
authorization remain external inputs.

The recommended first remote target has Linux/amd64, an NVIDIA GPU with bfloat16 support and at
least 24 GB VRAM, at least 32 GB host memory, 80 GB free storage, a compatible driver, and NVIDIA
Container Toolkit. These are conservative single-GPU starting requirements for this bounded 7B
profile, not a guarantee that an arbitrary driver/runtime combination will pass.

## Proof boundary

This evidence proves only why the current target is incompatible. It does not prove remote GPU
capacity, a completed QLoRA run, adapter quality, vLLM runtime health, hosted deployment, or
production acceptance. Follow `docs/operations/cuda-qualification-runbook.md` on an approved
target and return each immutable evidence layer separately.

## CUDA contract rehearsal

`cuda-contract-rehearsal-summary.json` indexes the CPU/MPS-safe rehearsal of the development
`configs/qwen-7b-qlora.yaml` profile from commit `1edaf0a`. The static QLoRA contract and teaching
dataset verified, while observed runtime status remained `blocked_no_cuda`. The report explicitly
records no model load, no CUDA execution, zero optimizer steps, no adapter, and no run manifest.
It is a structural portability check, never a CUDA-run substitute or evidence for the separate
release-quality profile.
