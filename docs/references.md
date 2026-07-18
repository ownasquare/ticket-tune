# Technical references and compatibility decisions

TicketTune pins its training stack as one tested unit. Fine-tuning libraries evolve together, and copying examples across unrelated releases is a common source of silent configuration errors.

## Pinned stack

| Component | Pinned version | Primary reference |
|---|---:|---|
| Python | 3.12 primary; 3.13 supported | [Python documentation](https://docs.python.org/3.12/) |
| Transformers | 5.13.0 | [Transformers documentation](https://huggingface.co/docs/transformers/) |
| TRL | 1.7.1 | [SFTTrainer](https://huggingface.co/docs/trl/en/sft_trainer) |
| PEFT | 0.19.1 | [PEFT documentation](https://huggingface.co/docs/peft/) |
| Datasets | 5.0.0 | [Datasets documentation](https://huggingface.co/docs/datasets/) |
| Accelerate | 1.14.0 | [Accelerate documentation](https://huggingface.co/docs/accelerate/) |
| bitsandbytes | 0.49.2 | [hardware compatibility](https://huggingface.co/docs/bitsandbytes/installation) |
| vLLM | 0.24.0 | [vLLM documentation](https://docs.vllm.ai/en/v0.24.0/) |
| NGINX | 1.30.3 Alpine | [NGINX downloads](https://nginx.org/en/download.html) |
| Prometheus | 3.13.0 | [Prometheus downloads](https://prometheus.io/download/) |
| Alertmanager | 0.33.1 | [Prometheus downloads](https://prometheus.io/download/) |
| CMake conversion tool | 4.4.0 | [CMake documentation](https://cmake.org/documentation/) |
| llama.cpp converter | `aedb2a5e9ca3d4064148bbb919e0ddc0c1b70ab3` (`b9637`) | [llama.cpp](https://github.com/ggml-org/llama.cpp/tree/aedb2a5e9ca3d4064148bbb919e0ddc0c1b70ab3) |

`uv.lock` is the executable resolution record. This table explains the intentional top-level pins.

## Training API decisions

- Source data uses ordinary `system`, `user`, and `assistant` messages. The processed training view separates conversational `prompt` and `completion`, allowing TRL completion-only loss without depending on a tokenizer-specific assistant mask.
- The tokenizer chat template is applied once by TRL. Source records are not pre-rendered into strings.
- Qwen2.5 profiles use `<|im_end|>` as the EOS token, matching the model family’s chat template.
- LoRA targets `all-linear`, which is the PEFT-recommended QLoRA-style coverage for unknown transformer module names.
- The checked-in 7B/8B QLoRA profiles use NF4, double quantization, and bfloat16
  compute after the CUDA/bitsandbytes preflight passes; the engine derives its
  compute dtype from each validated profile.
- Distributed training never uses `device_map="auto"`; placement belongs to Accelerate or the distributed launcher.

The installed TRL 1.7.1 runtime is treated as executable truth. Its `SFTTrainer` accepts
`processing_class` and `peft_config` but not a standalone `quantization_config` keyword.
TicketTune explicitly loads the model with Transformers before constructing the trainer because
TRL's string-model helper otherwise performs a separate config lookup and defaults a missing
device map to `"auto"`; the explicit load keeps revision, offline, dtype, and device policy intact.

Primary guidance:

- [TRL supervised fine-tuning](https://huggingface.co/docs/trl/en/sft_trainer)
- [TRL and PEFT integration](https://huggingface.co/docs/trl/main/peft_integration)
- [PEFT quantization guide](https://huggingface.co/docs/peft/developer_guides/quantization)
- [Transformers chat templates](https://huggingface.co/docs/transformers/en/chat_templating)

## Export decisions

The adapter and merged model serve different purposes:

- The PEFT adapter is small, auditable, and ideal for vLLM’s named LoRA serving.
- The merged model is reloaded from a pristine non-quantized base and merged with `safe_merge=True`. It is the portable source for GGUF conversion.

TicketTune does not merge into the 4-bit training object. It also does not send a Qwen Safetensors adapter directly to Ollama because Ollama’s documented direct adapter families do not include Qwen and its import guide warns that base/adapter mismatches produce erratic behavior.

Adapters live under immutable `<training.output_dir>/runs/<run-id>/adapter` paths; the sibling
`manifest.json` supplies run provenance. `latest-run.json` is only a mutable discovery pointer.
The vLLM Compose asset additionally pins the exact `linux/amd64` image
`vllm/vllm-openai:v0.24.0@sha256:f9de5cd9fa907fbf6dbba691eb7db095d48ad58ea283e3eba7142f9a91e186e8`.

Primary guidance:

- [PEFT checkpoint format and merging](https://huggingface.co/docs/peft/main/developer_guides/checkpoint)
- [vLLM LoRA serving](https://docs.vllm.ai/en/stable/features/lora/)
- [vLLM OpenAI-compatible server](https://docs.vllm.ai/en/v0.24.0/serving/online_serving/openai_compatible_server/)
- [Ollama model import](https://docs.ollama.com/import)
- [llama.cpp Hugging Face to GGUF converter](https://github.com/ggml-org/llama.cpp/blob/master/convert_hf_to_gguf.py)

## Production serving decisions

- [vLLM security guidance](https://docs.vllm.ai/en/stable/usage/security/) states that the API key
  protects API routes rather than every server endpoint. TicketTune therefore keeps vLLM private
  and exposes an allowlisted TLS gateway instead of publishing the vLLM port.
- [Docker Compose secrets](https://docs.docker.com/compose/how-tos/use-secrets/) are mounted as
  files. TicketTune validates a bounded file value inside the container and never places the API
  key in Compose environment, image layers, or command arguments.
- [Docker port-publishing guidance](https://docs.docker.com/engine/network/port-publishing/)
  informs the explicit gateway-only host port and the requirement for host firewall policy.
- vLLM's official Prometheus endpoint is scraped through a metrics-only gateway listener. The
  Prometheus container has no route to vLLM's model API network.
- Prometheus forwards alerts to an internal Alertmanager. The generic receiver URL is loaded with
  Alertmanager's
  [`url_file`](https://prometheus.io/docs/alerting/latest/configuration/#webhook_config) option so
  it is not committed in configuration or exposed in process arguments.

The production image references include exact Linux/amd64 manifest digests. Version labels make
the bundle understandable; digests are the executable identity and must be rescanned before use.

## Model references

- [Qwen2.5 0.5B Instruct model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- [Qwen2.5 7B Instruct model card](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- [Llama 3.1 8B Instruct model card](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)

Each operator remains responsible for the chosen base model’s license and acceptable-use terms.
