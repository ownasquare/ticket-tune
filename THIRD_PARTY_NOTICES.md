# Third-party notices

TicketTune does not redistribute base-model weights. Operators are responsible for reviewing and
accepting the license attached to every selected model before download, fine-tuning, or
distribution.

## Meta Llama 3.1

The optional `meta-llama/Llama-3.1-8B-Instruct` profile is governed by Meta's
[Llama 3.1 Community License](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/blob/main/LICENSE).
If a Llama-derived model is distributed, the operator must satisfy the upstream naming,
attribution, NOTICE, acceptable-use, and license-copy requirements. TicketTune's profile enforces
a `llama-` prefix for its derived serving names, but that guard does not replace the full license
review or the required distribution materials.

## Qwen2.5

Each Qwen profile pins a specific Hugging Face revision. Review the license and model card present
at that exact revision before redistributing merged weights or adapters.

## Serving and observability containers

The production reference composes upstream containers without redistributing their source code.
Each image remains governed by its upstream license and notices:

- [vLLM](https://github.com/vllm-project/vllm/blob/main/LICENSE) — Apache License 2.0;
- [NGINX](https://github.com/nginx/nginx/blob/master/LICENSE) — two-clause BSD license; and
- [Prometheus](https://github.com/prometheus/prometheus/blob/main/LICENSE) and
  [Alertmanager](https://github.com/prometheus/alertmanager/blob/main/LICENSE) — Apache License 2.0.

TicketTune pins platform-specific image digests in the production Compose file. A digest fixes
container bytes; it does not transfer the upstream trademark, license, vulnerability-remediation,
or source-offer responsibilities. Operators should mirror approved images, retain upstream notices,
and rescan the exact digests before every release.
