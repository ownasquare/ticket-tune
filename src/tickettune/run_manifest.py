"""Immutable provenance records for TicketTune training runs.

This module deliberately depends only on the Python standard library and
Pydantic. Importing it never imports a model framework or probes hardware.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil

# Used only for fixed, read-only local Git metadata commands.
import subprocess  # nosec B404
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

RUN_MANIFEST_SCHEMA_VERSION: Literal["1.1"] = "1.1"
_TRACKED_PACKAGES = (
    "accelerate",
    "bitsandbytes",
    "datasets",
    "peft",
    "torch",
    "transformers",
    "trl",
)
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(authorization|token|api[_-]?key|password|secret)(\s*[:=]\s*)(?:bearer\s+)?[^\s,;]+"
)
_HF_TOKEN = re.compile(r"\bhf_[A-Za-z0-9]{20,}\b")


class ArtifactDigest(BaseModel):
    """Content-addressed description of one immutable artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)


class RunManifest(BaseModel):
    """Frozen metadata written after a completed or failed training attempt."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    schema_version: Literal["1.1"] = RUN_MANIFEST_SCHEMA_VERSION
    run_id: str
    created_at: datetime
    status: Literal["completed", "failed"]
    project_name: str
    model_name_or_path: str
    model_revision: str | None = None
    method: Literal["lora", "qlora"]
    seed: int
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    config: dict[str, Any]
    dataset_sha256: dict[str, str]
    packages: dict[str, str | None]
    runtime: dict[str, str]
    hardware_preflight: dict[str, Any] | None = None
    git_revision: str | None = None
    git_branch: str | None = None
    git_dirty: bool | None = None
    training_duration_seconds: float | None = Field(default=None, ge=0)
    peak_accelerator_memory_mb: float | None = Field(default=None, ge=0)
    metrics: dict[str, float | int | str | bool | None] = Field(default_factory=dict)
    artifacts: tuple[ArtifactDigest, ...] = ()
    resume_from_checkpoint: str | None = None
    error: str | None = None


def json_safe(value: object) -> Any:
    """Convert common configuration values into deterministic JSON data."""

    if isinstance(value, BaseModel):
        return json_safe(value.model_dump(mode="json"))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat()
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "model_dump"):
        dump = value.model_dump(mode="json")
        return json_safe(dump)
    if hasattr(value, "__dict__"):
        return json_safe(vars(value))
    return str(value)


def canonical_json_bytes(value: object, *, pretty: bool = False) -> bytes:
    """Serialize JSON with stable key ordering for hashing and persistence."""

    separators = None if pretty else (",", ":")
    text = json.dumps(
        json_safe(value),
        ensure_ascii=False,
        allow_nan=False,
        indent=2 if pretty else None,
        separators=separators,
        sort_keys=True,
    )
    if pretty:
        text += "\n"
    return text.encode("utf-8")


def sha256_bytes(value: object) -> str:
    """Hash the canonical JSON representation of ``value``."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def make_run_id(*, config: object, created_at: datetime) -> str:
    """Derive the immutable run-directory name from time and resolved config."""

    timestamp = created_at.astimezone(UTC)
    timestamp_slug = timestamp.strftime("%Y%m%dT%H%M%S.%fZ")
    return f"{timestamp_slug}-{sha256_bytes(config)[:12]}"


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for a regular file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_digest(path: Path, *, relative_to: Path | None = None) -> ArtifactDigest:
    """Describe a file without mutating it."""

    resolved = path.resolve()
    display_path = resolved.relative_to(relative_to.resolve()) if relative_to else resolved
    return ArtifactDigest(
        path=str(display_path),
        sha256=sha256_file(resolved),
        size_bytes=resolved.stat().st_size,
    )


def package_versions() -> dict[str, str | None]:
    """Read installed versions without importing any heavyweight package."""

    versions: dict[str, str | None] = {}
    for package in _TRACKED_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def runtime_metadata() -> dict[str, str]:
    """Collect non-sensitive runtime facts useful for reproduction."""

    return {
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
    }


def source_control_metadata(start: Path | None = None) -> dict[str, str | bool | None]:
    """Read non-sensitive Git provenance without making any network request."""

    working_directory = (start or Path.cwd()).resolve()
    git_executable = shutil.which("git")
    if git_executable is None:
        return {"git_revision": None, "git_branch": None, "git_dirty": None}

    def git(*arguments: str) -> tuple[int, str]:
        try:
            # The executable is resolved to an absolute path and argv is fixed.
            result = subprocess.run(  # noqa: S603  # nosec B603
                [git_executable, *arguments],
                cwd=working_directory,
                check=False,
                capture_output=True,
                text=True,
                timeout=3,
            )
        except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
            return 127, ""
        return result.returncode, result.stdout.strip()

    revision_code, revision = git("rev-parse", "HEAD")
    branch_code, branch = git("branch", "--show-current")
    status_code, status = git("status", "--porcelain")
    return {
        "git_revision": revision if revision_code == 0 else None,
        "git_branch": branch if branch_code == 0 and branch else None,
        "git_dirty": bool(status) if status_code == 0 else None,
    }


def sanitize_error(error: BaseException | str, *, limit: int = 4000) -> str:
    """Redact common credential shapes before persisting a bounded error."""

    text = str(error)
    text = _SECRET_ASSIGNMENT.sub(r"\1\2[REDACTED]", text)
    text = _HF_TOKEN.sub("[REDACTED]", text)
    return text[:limit]


def build_run_manifest(
    *,
    config: object,
    status: Literal["completed", "failed"],
    project_name: str,
    model_name_or_path: str,
    model_revision: str | None,
    method: Literal["lora", "qlora"],
    seed: int,
    dataset_sha256: Mapping[str, str],
    metrics: Mapping[str, float | int | str | bool | None] | None = None,
    artifacts: Sequence[ArtifactDigest] = (),
    resume_from_checkpoint: Path | str | None = None,
    error: str | None = None,
    hardware_preflight: object | None = None,
    training_duration_seconds: float | None = None,
    peak_accelerator_memory_mb: float | None = None,
    source_control: Mapping[str, str | bool | None] | None = None,
    created_at: datetime | None = None,
    versions: Mapping[str, str | None] | None = None,
) -> RunManifest:
    """Build a frozen manifest from an already-resolved configuration."""

    config_payload = json_safe(config)
    if not isinstance(config_payload, dict):
        raise TypeError("config must serialize to a JSON object")
    timestamp = (created_at or datetime.now(UTC)).astimezone(UTC)
    config_hash = sha256_bytes(config_payload)
    run_id = make_run_id(config=config_payload, created_at=timestamp)
    git_metadata = dict(source_control or source_control_metadata())
    return RunManifest(
        run_id=run_id,
        created_at=timestamp,
        status=status,
        project_name=project_name,
        model_name_or_path=model_name_or_path,
        model_revision=model_revision,
        method=method,
        seed=seed,
        config_sha256=config_hash,
        config=config_payload,
        dataset_sha256=dict(sorted(dataset_sha256.items())),
        packages=dict(versions) if versions is not None else package_versions(),
        runtime=runtime_metadata(),
        hardware_preflight=(
            json_safe(hardware_preflight) if hardware_preflight is not None else None
        ),
        git_revision=(
            str(git_metadata["git_revision"])
            if git_metadata.get("git_revision") is not None
            else None
        ),
        git_branch=(
            str(git_metadata["git_branch"]) if git_metadata.get("git_branch") is not None else None
        ),
        git_dirty=(
            bool(git_metadata["git_dirty"]) if git_metadata.get("git_dirty") is not None else None
        ),
        training_duration_seconds=training_duration_seconds,
        peak_accelerator_memory_mb=peak_accelerator_memory_mb,
        metrics=dict(metrics or {}),
        artifacts=tuple(artifacts),
        resume_from_checkpoint=str(resume_from_checkpoint) if resume_from_checkpoint else None,
        error=error,
    )


def write_manifest(path: Path, manifest: RunManifest) -> Path:
    """Atomically create an immutable manifest.

    Repeating an identical write is idempotent. Reusing a path for different
    bytes is rejected so provenance cannot silently change underneath a run.
    """

    payload = canonical_json_bytes(manifest.model_dump(mode="json"), pretty=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == payload:
            return path
        raise FileExistsError(f"refusing to overwrite immutable run manifest: {path}")

    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return path
