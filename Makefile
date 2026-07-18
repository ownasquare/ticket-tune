.DEFAULT_GOAL := help

.PHONY: help help-advanced setup sync lock test coverage lint format typecheck security build check quickstart docs data-validate data-prepare doctor smoke train evaluate evaluate-live require-adapter merge serve-vllm

ADAPTER ?=
MERGED_OUTPUT ?=

help:
	@echo "TicketTune — fine-tune a support-ticket triage model"
	@echo ""
	@echo "make setup       Install the locked development environment"
	@echo "make quickstart  Prove the workflow locally without downloads or training"
	@echo "make train       Fine-tune the small-model profile"
	@echo "make evaluate-live ADAPTER=...  Compare an adapter with its base model"
	@echo "make docs        Show the short documentation map"
	@echo ""
	@echo "More commands: make help-advanced"

help-advanced:
	@echo "make check       Run formatting, lint, typing, security, coverage, and build"
	@echo "make security    Check code, dependencies, and Git history for security issues"
	@echo "make smoke       Exercise every offline pipeline stage"
	@echo "make lock        Refresh uv.lock from declared dependencies"
	@echo "make merge ADAPTER=... MERGED_OUTPUT=...  Plan a safe adapter merge"
	@echo "make serve-vllm ADAPTER=...  Print a vLLM launch plan"

setup: sync

sync:
	uv sync --frozen

quickstart:
	uv run --no-sync tickettune quickstart

docs:
	@echo "Start with docs/README.md, then docs/getting-started.md or docs/customize.md"

lock:
	uv lock

test:
	uv run pytest -q

coverage:
	uv run pytest -q --cov=tickettune --cov-report=term-missing

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

typecheck:
	uv run mypy src

security:
	uv run bandit -q -r src deploy
	uv run pip-audit
	uv run python scripts/scan_secrets.py

build:
	uv build

check: lint typecheck security coverage build

data-validate:
	uv run tickettune data validate --config configs/smoke.yaml

data-prepare:
	uv run tickettune data prepare --config configs/smoke.yaml

doctor:
	uv run tickettune doctor --config configs/smoke.yaml

smoke: data-validate data-prepare doctor
	uv run tickettune train --config configs/smoke.yaml --dry-run
	uv run tickettune evaluate --config configs/smoke.yaml --predictions tests/fixtures/predictions_pass.jsonl --enforce-thresholds
	uv run tickettune serve vllm --config configs/smoke.yaml --adapter tests/fixtures/adapter --allow-unqualified-local-smoke --dry-run
	uv run tickettune export ollama --config configs/smoke.yaml --merged-model tests/fixtures/merged --allow-unqualified-local-smoke --dry-run

train:
	uv sync --frozen --extra train
	uv run tickettune data prepare --config configs/cpu-smoke.yaml
	uv run tickettune doctor --config configs/cpu-smoke.yaml
	uv run tickettune train --config configs/cpu-smoke.yaml --allow-download

evaluate:
	uv run tickettune evaluate --config configs/smoke.yaml --predictions tests/fixtures/predictions_pass.jsonl --enforce-thresholds

require-adapter:
	@test -n "$(ADAPTER)" || (echo "ADAPTER is required; use an immutable <output>/runs/<run-id>/adapter path" >&2; exit 2)

evaluate-live: require-adapter
	uv run tickettune evaluate --config configs/cpu-smoke.yaml --adapter "$(ADAPTER)" --compare-baseline

merge: require-adapter
	@test -n "$(MERGED_OUTPUT)" || (echo "MERGED_OUTPUT is required; use a new run-specific deployment path" >&2; exit 2)
	uv run tickettune merge --config configs/cpu-smoke.yaml --adapter "$(ADAPTER)" --output "$(MERGED_OUTPUT)" --allow-unqualified-local-smoke --dry-run

serve-vllm: require-adapter
	uv run tickettune serve vllm --config configs/qwen-7b-qlora.yaml --adapter "$(ADAPTER)" --dry-run
