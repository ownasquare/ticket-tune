#!/usr/bin/env python3
"""Fail-closed production entrypoint for the TicketTune vLLM service."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from pathlib import Path

_API_KEY_MIN_BYTES = 32
_API_KEY_MAX_BYTES = 4096
_DEFAULT_API_KEY_FILE = "/run/secrets/vllm_api_key"
_RELEASE_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]{2,127}\Z")
_GIT_SHA_PATTERN = re.compile(r"[0-9a-f]{40}\Z")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_PLACEHOLDER_MARKERS = (
    "changeme",
    "example",
    "placeholder",
    "replace",
    "sentinel",
    "todo",
)
_CACHE_VARIABLES = (
    "HOME",
    "XDG_CACHE_HOME",
    "VLLM_CACHE_ROOT",
    "TORCHINDUCTOR_CACHE_DIR",
    "TRITON_CACHE_DIR",
    "CUDA_CACHE_PATH",
)

ExecFunction = Callable[[str, list[str], dict[str, str]], None]


def _required(environment: Mapping[str, str], name: str) -> str:
    value = environment.get(name, "")
    if not value:
        raise ValueError(f"required environment variable {name} is empty")
    return value


def _active_group_ids() -> set[int]:
    return {os.getgid(), os.getegid(), *os.getgroups()}


def load_api_key(path: str | Path, *, expected_group_id: int) -> str:
    """Read an API key only when its file metadata and bytes are safe."""

    if expected_group_id < 0:
        raise ValueError("API key expected group ID must be non-negative")
    if expected_group_id not in _active_group_ids():
        raise ValueError("API key group is not active for this process")

    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:  # pragma: no cover - production targets are POSIX Linux.
        raise RuntimeError("API key reads require O_NOFOLLOW support")
    flags = os.O_RDONLY | no_follow | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(Path(path), flags)
    except OSError as error:
        raise ValueError("API key file must be a regular non-symbolic-link file") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("API key file must be a regular file")
        if stat.S_IMODE(metadata.st_mode) != 0o440:
            raise ValueError("API key file mode must be exactly 0440")
        if metadata.st_gid != expected_group_id:
            raise ValueError("API key file group does not match SECRET_GROUP_ID")
        if metadata.st_size > _API_KEY_MAX_BYTES:
            raise ValueError(f"API key must not exceed {_API_KEY_MAX_BYTES} bytes")

        chunks: list[bytes] = []
        remaining = _API_KEY_MAX_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
    except OSError as error:
        raise ValueError(f"API key file cannot be read: {error}") from error
    finally:
        os.close(descriptor)

    if not _API_KEY_MIN_BYTES <= len(payload) <= _API_KEY_MAX_BYTES:
        raise ValueError(
            f"API key must be between {_API_KEY_MIN_BYTES} and {_API_KEY_MAX_BYTES} bytes"
        )
    try:
        value = payload.decode("ascii", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError("API key must contain ASCII bytes only") from error
    if any(character.isspace() or character == "\x00" for character in value):
        raise ValueError("API key must contain ASCII without whitespace or NUL")
    return value


def validate_release_identity(
    release_id: str,
    git_revision: str,
    adapter_sha256: str,
) -> None:
    """Reject mutable, placeholder, or malformed release identity values."""

    if not _RELEASE_ID_PATTERN.fullmatch(release_id):
        raise ValueError("RELEASE_ID is not a valid immutable identifier")
    if any(marker in release_id for marker in _PLACEHOLDER_MARKERS):
        raise ValueError("RELEASE_ID still contains a placeholder marker")
    if not _GIT_SHA_PATTERN.fullmatch(git_revision) or git_revision == "0" * 40:
        raise ValueError("RELEASE_GIT_REVISION must be a non-zero lowercase Git SHA")
    if not _SHA256_PATTERN.fullmatch(adapter_sha256) or adapter_sha256 == "0" * 64:
        raise ValueError("EXPECTED_ADAPTER_SHA256 must be a non-zero lowercase SHA-256")


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size_bytes = 0
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:  # pragma: no cover - production targets are POSIX Linux.
        raise RuntimeError("adapter reads require O_NOFOLLOW support")
    flags = os.O_RDONLY | no_follow | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as handle:
            metadata_before = os.fstat(handle.fileno())
            if not stat.S_ISREG(metadata_before.st_mode):
                raise ValueError(f"adapter entry is not a regular file: {path}")
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
                size_bytes += len(chunk)
            metadata_after = os.fstat(handle.fileno())
    except OSError as error:
        raise ValueError(f"adapter file cannot be read: {path}: {error}") from error
    stable_identity_before = (
        metadata_before.st_dev,
        metadata_before.st_ino,
        metadata_before.st_size,
        metadata_before.st_mtime_ns,
    )
    stable_identity_after = (
        metadata_after.st_dev,
        metadata_after.st_ino,
        metadata_after.st_size,
        metadata_after.st_mtime_ns,
    )
    if stable_identity_before != stable_identity_after or size_bytes != metadata_after.st_size:
        raise ValueError(f"adapter file changed while being hashed: {path}")
    return digest.hexdigest(), size_bytes


def adapter_inventory_sha256(adapter_path: str | Path) -> str:
    """Hash the canonical inventory of all regular adapter files."""

    root = Path(adapter_path)
    try:
        root_metadata = root.lstat()
    except OSError as error:
        raise ValueError(f"adapter directory cannot be inspected: {error}") from error
    if stat.S_ISLNK(root_metadata.st_mode):
        raise ValueError("adapter directory must not be a symbolic link")
    if not stat.S_ISDIR(root_metadata.st_mode):
        raise ValueError("adapter path must be a directory")

    inventory: list[dict[str, str | int]] = []

    def visit(directory: Path) -> None:
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as error:
            raise ValueError(f"adapter directory cannot be read: {directory}: {error}") from error

        for entry in entries:
            entry_path = Path(entry.path)
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError as error:
                raise ValueError(
                    f"adapter entry cannot be inspected: {entry_path}: {error}"
                ) from error
            if stat.S_ISLNK(metadata.st_mode):
                raise ValueError(f"adapter entry is a symbolic link: {entry_path}")
            if stat.S_ISDIR(metadata.st_mode):
                visit(entry_path)
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError(f"adapter entry is not a regular file: {entry_path}")

            file_sha256, size_bytes = _sha256_file(entry_path)
            inventory.append(
                {
                    "path": entry_path.relative_to(root).as_posix(),
                    "sha256": file_sha256,
                    "size_bytes": size_bytes,
                }
            )

    visit(root)
    if not inventory:
        raise ValueError("adapter directory contains no regular files")

    inventory.sort(key=lambda item: str(item["path"]))
    canonical = json.dumps(
        inventory,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _preflight_cache_directory(path: Path, runtime_root: Path, variable: str) -> None:
    try:
        resolved = path.resolve(strict=False)
        resolved.relative_to(runtime_root)
    except (OSError, ValueError) as error:
        raise ValueError(f"{variable} must resolve within TICKETTUNE_RUNTIME_CACHE_ROOT") from error

    try:
        path.mkdir(parents=True, exist_ok=True)
        metadata = path.lstat()
    except OSError as error:
        raise ValueError(f"{variable} cannot be created: {error}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"{variable} must be a real directory, not a link")

    probe_fd: int | None = None
    probe_name: str | None = None
    try:
        probe_fd, probe_name = tempfile.mkstemp(prefix=".tickettune-write-probe-", dir=path)
        os.write(probe_fd, b"ok")
        os.fsync(probe_fd)
    except OSError as error:
        raise ValueError(f"{variable} is not writable: {error}") from error
    finally:
        if probe_fd is not None:
            os.close(probe_fd)
        if probe_name is not None:
            with suppress(FileNotFoundError):
                Path(probe_name).unlink()


def preflight_runtime_caches(environment: Mapping[str, str]) -> None:
    root = Path(_required(environment, "TICKETTUNE_RUNTIME_CACHE_ROOT"))
    try:
        root_metadata = root.lstat()
    except OSError as error:
        raise ValueError(f"runtime cache root cannot be inspected: {error}") from error
    if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
        raise ValueError("runtime cache root must be an existing real directory")
    resolved_root = root.resolve(strict=True)

    for variable in _CACHE_VARIABLES:
        _preflight_cache_directory(
            Path(_required(environment, variable)),
            resolved_root,
            variable,
        )


def _secret_group_id(environment: Mapping[str, str]) -> int:
    raw_group_id = _required(environment, "SECRET_GROUP_ID")
    if not raw_group_id.isascii() or not raw_group_id.isdigit():
        raise ValueError("SECRET_GROUP_ID must be a numeric group ID")
    return int(raw_group_id)


def execute(
    argv: Sequence[str],
    environment: Mapping[str, str],
    *,
    exec_fn: ExecFunction = os.execvpe,
) -> None:
    """Validate the release and replace this process with vLLM."""

    if not argv:
        raise ValueError("a vLLM command is required")

    expected_group_id = _secret_group_id(environment)
    api_key = load_api_key(
        environment.get("VLLM_API_KEY_FILE", _DEFAULT_API_KEY_FILE),
        expected_group_id=expected_group_id,
    )
    release_id = _required(environment, "RELEASE_ID")
    git_revision = _required(environment, "RELEASE_GIT_REVISION")
    expected_adapter_sha256 = _required(environment, "EXPECTED_ADAPTER_SHA256")
    validate_release_identity(release_id, git_revision, expected_adapter_sha256)

    actual_adapter_sha256 = adapter_inventory_sha256(
        _required(environment, "TICKETTUNE_ADAPTER_PATH")
    )
    if actual_adapter_sha256 != expected_adapter_sha256:
        raise ValueError("adapter inventory SHA-256 does not match EXPECTED_ADAPTER_SHA256")

    preflight_runtime_caches(environment)

    child_environment = dict(environment)
    child_environment["VLLM_API_KEY"] = api_key
    child_environment.pop("VLLM_API_KEY_FILE", None)
    command = list(argv)
    exec_fn(command[0], command, child_environment)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a TicketTune release, then start the vLLM server.",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="vLLM command and arguments to execute",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    command = arguments.command
    if command[:1] == ["--"]:
        command = command[1:]
    try:
        execute(command, os.environ)
    except ValueError as error:
        parser.exit(2, f"startup validation failed: {error}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
