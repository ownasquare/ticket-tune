import re
import tomllib
from pathlib import Path

import yaml
from typer.testing import CliRunner

from tickettune import __version__
from tickettune.cli import app

ROOT = Path(__file__).resolve().parents[1]


def test_package_exposes_version_and_help() -> None:
    assert __version__ == "0.1.0"
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Fine-tune" in result.stdout


def test_version_is_discoverable() -> None:
    result = CliRunner().invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == "tickettune 0.1.0"


def test_makefile_exposes_a_small_public_workflow() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")

    for target in ("setup:", "quickstart:", "train:", "evaluate-live:", "docs:"):
        assert target in makefile
    assert "tickettune quickstart" in makefile
    assert "help-advanced:" in makefile


def test_public_readme_is_concise_and_progressively_disclosed() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert len(readme.splitlines()) <= 180
    assert "docs/getting-started.md" in readme
    assert "docs/customize.md" in readme
    assert "<details>" in readme
    assert "docs/superpowers" not in readme
    assert "uv tool install ." in readme


def test_deployment_commands_are_pasteable_shell_commands() -> None:
    deployment = (ROOT / "docs/deployment.md").read_text(encoding="utf-8")

    assert '```json\n["' not in deployment
    assert "uv run --no-sync tickettune serve vllm" in deployment
    assert "docker compose --env-file" in deployment


def test_wheel_maps_checkout_examples_into_package_resources() -> None:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)
    force_include = project["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]

    assert force_include["configs/smoke.yaml"] == ("tickettune/starter/configs/tickettune.yaml")
    assert force_include["data/raw/support_tickets.jsonl"] == (
        "tickettune/starter/data/raw/support_tickets.jsonl"
    )
    assert force_include["tests/fixtures/predictions_pass.jsonl"] == (
        "tickettune/starter/predictions/pass.jsonl"
    )


def test_ci_pins_actions_and_audits_conversion_stack() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    uses = re.findall(r"uses:\s+([^\s#]+)", workflow)

    assert uses
    assert all(re.fullmatch(r"[^@]+@[0-9a-f]{40}", item) for item in uses)
    assert "--extra train --extra convert" in workflow
    assert "cmake --version" in workflow
    assert "bandit -q -r src deploy" in workflow
    assert "scripts/verify_wheel_install.py" in workflow


def test_ci_runs_pinned_production_configuration_validators() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "production-static:" in workflow
    assert "docker compose" in workflow and "config --quiet" in workflow
    assert "nginx:1.30.3-alpine@sha256:" in workflow and "-t" in workflow
    assert "prom/prometheus:v3.13.0@sha256:" in workflow
    assert "check config /etc/prometheus/prometheus.yml" in workflow
    assert "prom/alertmanager:v0.33.1@sha256:" in workflow
    assert "check-config /etc/alertmanager/alertmanager.yml" in workflow


def test_dependabot_tracks_both_vllm_compose_directories() -> None:
    payload = yaml.safe_load((ROOT / ".github/dependabot.yml").read_text(encoding="utf-8"))
    docker_directories = {
        update["directory"]
        for update in payload["updates"]
        if update["package-ecosystem"] == "docker"
    }

    assert docker_directories == {"/deploy/vllm", "/deploy/vllm/production"}


def test_sdist_excludes_internal_handoffs_plans_and_host_readiness() -> None:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)
    sdist = project["tool"]["hatch"]["build"]["targets"]["sdist"]

    assert {"/docs", "/results"} <= set(sdist["include"])
    assert {
        "/docs/handoffs",
        "/docs/superpowers",
        "/results/readiness",
    } <= set(sdist["exclude"])
