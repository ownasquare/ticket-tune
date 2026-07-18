from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from tickettune.public_export import (
    EXCLUDED_PREFIXES,
    MANIFEST_NAME,
    ExportError,
    build_public_export,
)

ROOT = Path(__file__).resolve().parents[1]


def _git(repository: Path, *arguments: str) -> str:
    git = shutil.which("git")
    if git is None:
        pytest.skip("Git is required for public-export tests")
    completed = subprocess.run(  # noqa: S603 - absolute executable, argument list, no shell
        [git, "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


@pytest.fixture
def committed_repository(tmp_path: Path) -> Path:
    repository = tmp_path / "source"
    repository.mkdir()
    _git(repository, "init", "-b", "main")

    (repository / ".gitignore").write_text("artifacts/\n", encoding="utf-8")
    (repository / "README.md").write_text("# Public project\n", encoding="utf-8")
    (repository / "payload.bin").write_bytes(b"\x00tracked-git-tree-bytes\xff")
    (repository / "compose.yaml").write_text(
        "cache: /home/vllm/.cache/huggingface\n",
        encoding="utf-8",
    )
    handoff = repository / "docs" / "handoffs" / "private.mdc"
    handoff.parent.mkdir(parents=True)
    private_mac_path = "/" + "Users" + "/example/workspace"
    handoff.write_text(f"private path: {private_mac_path}\n", encoding="utf-8")
    plan = repository / "docs" / "superpowers" / "plans" / "private.md"
    plan.parent.mkdir(parents=True)
    plan.write_text(f"internal plan: {private_mac_path}\n", encoding="utf-8")

    _git(repository, "add", ".")
    _git(
        repository,
        "-c",
        "user.name=TicketTune Tests",
        "-c",
        "user.email=tests@example.invalid",
        "commit",
        "-m",
        "fixture",
    )
    return repository


def test_export_uses_committed_bytes_and_omits_internal_and_ignored_files(
    committed_repository: Path,
    tmp_path: Path,
) -> None:
    ignored_artifact = committed_repository / "artifacts" / "model.safetensors"
    ignored_artifact.parent.mkdir()
    ignored_artifact.write_bytes(b"ignored model weights")
    assert _git(committed_repository, "status", "--porcelain=v1") == ""

    destination = tmp_path / "public"
    result = build_public_export(committed_repository, destination)

    assert result.destination == destination.resolve()
    assert (destination / "README.md").read_bytes() == b"# Public project\n"
    assert (destination / "payload.bin").read_bytes() == b"\x00tracked-git-tree-bytes\xff"
    assert (destination / "compose.yaml").read_text(encoding="utf-8") == (
        "cache: /home/vllm/.cache/huggingface\n"
    )
    assert not (destination / "docs" / "handoffs").exists()
    assert not (destination / "docs" / "superpowers").exists()
    assert not (destination / "artifacts").exists()


def test_export_refuses_a_dirty_source_without_creating_destination(
    committed_repository: Path,
    tmp_path: Path,
) -> None:
    (committed_repository / "README.md").write_text("changed\n", encoding="utf-8")
    destination = tmp_path / "public"

    with pytest.raises(ExportError, match="source repository is dirty"):
        build_public_export(committed_repository, destination)

    assert not destination.exists()


def test_export_refuses_an_existing_destination_without_changing_it(
    committed_repository: Path,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "public"
    destination.mkdir()
    marker = destination / "keep.txt"
    marker.write_text("keep me\n", encoding="utf-8")

    with pytest.raises(ExportError, match="destination must not exist"):
        build_public_export(committed_repository, destination)

    assert marker.read_text(encoding="utf-8") == "keep me\n"


@pytest.mark.parametrize(
    "private_path",
    [
        "/" + "Users" + "/example/private/model",
        "/" + "home" + "/example/private/model",
        "C:" + chr(92) + "Users" + chr(92) + "example" + chr(92) + "private" + chr(92) + "model",
    ],
)
def test_export_refuses_absolute_user_paths_in_public_files(
    committed_repository: Path,
    tmp_path: Path,
    private_path: str,
) -> None:
    (committed_repository / "README.md").write_text(private_path + "\n", encoding="utf-8")
    _git(committed_repository, "add", "README.md")
    _git(
        committed_repository,
        "-c",
        "user.name=TicketTune Tests",
        "-c",
        "user.email=tests@example.invalid",
        "commit",
        "-m",
        "add private path",
    )
    destination = tmp_path / "public"

    with pytest.raises(ExportError, match="absolute local user path"):
        build_public_export(committed_repository, destination)

    assert not destination.exists()


def test_export_manifest_is_deterministic_and_sorted(
    committed_repository: Path,
    tmp_path: Path,
) -> None:
    first = tmp_path / "public-one"
    second = tmp_path / "public-two"

    build_public_export(committed_repository, first)
    build_public_export(committed_repository, second)

    first_manifest = (first / MANIFEST_NAME).read_bytes()
    second_manifest = (second / MANIFEST_NAME).read_bytes()
    assert first_manifest == second_manifest

    manifest = json.loads(first_manifest)
    paths = [entry["path"] for entry in manifest["files"]]
    assert paths == sorted(paths)
    assert manifest["source_revision"] == _git(committed_repository, "rev-parse", "HEAD")
    assert manifest["excluded_prefixes"] == list(EXCLUDED_PREFIXES)
    assert manifest["file_count"] == len(paths)
    assert MANIFEST_NAME not in paths
    assert all(
        set(entry) == {"mode", "path", "sha256", "size_bytes"} for entry in manifest["files"]
    )


def test_public_community_templates_are_present_and_structured() -> None:
    conduct = (ROOT / "CODE_OF_CONDUCT.md").read_text(encoding="utf-8")
    assert "## Our standards" in conduct
    assert "## Enforcement" in conduct

    for name in ("bug_report.yml", "feature_request.yml"):
        template = yaml.safe_load((ROOT / ".github" / "ISSUE_TEMPLATE" / name).read_text())
        assert template["name"]
        assert template["description"]
        assert template["body"]

    pull_request = (ROOT / ".github" / "pull_request_template.md").read_text(encoding="utf-8")
    assert "## Summary" in pull_request
    assert "## Validation" in pull_request
    assert "proof layer" in pull_request.lower()
