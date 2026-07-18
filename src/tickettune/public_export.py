"""Build a deterministic, sanitized public tree from committed Git bytes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess  # nosec B404
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

# Subprocess use is limited to an absolute Git executable with shell execution disabled.

MANIFEST_NAME = "public-export-manifest.json"
EXCLUDED_PREFIXES = (
    "docs/handoffs/",
    "docs/superpowers/",
)

_ALLOWED_FILE_MODES = {"100644", "100755"}
_LOCAL_USER_PATH_MARKERS = (
    b"/" + b"Users" + b"/",
    b"/" + b"home" + b"/",
)
_PUBLIC_RUNTIME_HOME_PATHS = (b"/" + b"home" + b"/vllm/.cache/huggingface",)


class ExportError(RuntimeError):
    """Raised when a public export cannot be created safely."""


@dataclass(frozen=True, slots=True)
class ExportResult:
    """Summary of a completed public export."""

    destination: Path
    manifest_path: Path
    source_revision: str
    file_count: int


@dataclass(frozen=True, slots=True)
class _ExportFile:
    path: str
    mode: str
    payload: bytes


def _run_git(source: Path, *arguments: str) -> bytes:
    git = shutil.which("git")
    if git is None:
        raise ExportError("Git executable was not found")
    environment = os.environ.copy()
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    completed = subprocess.run(  # noqa: S603  # nosec B603
        [git, "-C", str(source), *arguments],
        check=False,
        capture_output=True,
        env=environment,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        if detail:
            raise ExportError(f"Git command failed: {detail}")
        raise ExportError(f"Git command failed with exit code {completed.returncode}")
    return completed.stdout


def _resolve_repository(source: Path) -> Path:
    try:
        resolved = source.expanduser().resolve(strict=True)
    except OSError as exc:
        raise ExportError(f"source repository does not exist: {source}") from exc
    if not resolved.is_dir():
        raise ExportError(f"source repository must be a directory: {resolved}")

    top_level = Path(
        _run_git(resolved, "rev-parse", "--show-toplevel").decode("utf-8").strip()
    ).resolve()
    if top_level != resolved:
        raise ExportError(f"source must be the Git repository root: {top_level}")
    return resolved


def _resolve_new_destination(source: Path, destination: Path) -> Path:
    expanded = destination.expanduser()
    if expanded.exists() or expanded.is_symlink():
        raise ExportError(f"destination must not exist: {expanded}")
    resolved = expanded.resolve(strict=False)
    if resolved.is_relative_to(source):
        raise ExportError("destination must be outside the source repository")
    return resolved


def _require_clean_source(source: Path) -> None:
    status = _run_git(
        source,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        "--ignored=no",
    )
    if status:
        raise ExportError(
            "source repository is dirty; commit, stash, or remove tracked and untracked changes"
        )


def _is_excluded(path: str) -> bool:
    return any(
        path == prefix.removesuffix("/") or path.startswith(prefix) for prefix in EXCLUDED_PREFIXES
    )


def _validate_public_path(path: str) -> None:
    pure_path = PurePosixPath(path)
    if (
        pure_path.is_absolute()
        or not pure_path.parts
        or any(part in {"", ".", ".."} for part in pure_path.parts)
    ):
        raise ExportError(f"tracked tree contains an unsafe path: {path!r}")
    if path == MANIFEST_NAME:
        raise ExportError(f"tracked tree reserves the export manifest path: {MANIFEST_NAME}")


def _reject_local_user_paths(path: str, payload: bytes) -> None:
    normalized = payload.replace(b"\\", b"/")
    for runtime_path in _PUBLIC_RUNTIME_HOME_PATHS:
        normalized = normalized.replace(runtime_path, b"<public-runtime-home>")
    if any(marker in normalized for marker in _LOCAL_USER_PATH_MARKERS):
        raise ExportError(f"public file contains an absolute local user path: {path}")


def _load_export_files(source: Path) -> list[_ExportFile]:
    tree = _run_git(source, "ls-tree", "-r", "-z", "--full-tree", "HEAD")
    files: list[_ExportFile] = []
    for raw_entry in tree.split(b"\0"):
        if not raw_entry:
            continue
        try:
            metadata, raw_path = raw_entry.split(b"\t", 1)
            mode, object_type, object_id = metadata.decode("ascii").split(" ", 2)
            path = raw_path.decode("utf-8")
        except (UnicodeDecodeError, ValueError) as exc:
            raise ExportError("tracked tree contains an unsupported entry") from exc

        if _is_excluded(path):
            continue
        _validate_public_path(path)
        if object_type != "blob" or mode not in _ALLOWED_FILE_MODES:
            raise ExportError(f"tracked tree contains an unsupported entry: {path} ({mode})")

        payload = _run_git(source, "cat-file", "blob", object_id)
        _reject_local_user_paths(path, payload)
        files.append(_ExportFile(path=path, mode=mode, payload=payload))

    return sorted(files, key=lambda item: item.path)


def _manifest_payload(source_revision: str, files: Sequence[_ExportFile]) -> bytes:
    manifest = {
        "excluded_prefixes": list(EXCLUDED_PREFIXES),
        "file_count": len(files),
        "files": [
            {
                "mode": item.mode,
                "path": item.path,
                "sha256": hashlib.sha256(item.payload).hexdigest(),
                "size_bytes": len(item.payload),
            }
            for item in files
        ],
        "schema_version": 1,
        "source_revision": source_revision,
    }
    return (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_export(destination: Path, files: Sequence[_ExportFile], manifest: bytes) -> None:
    created = False
    try:
        destination.mkdir(parents=True, exist_ok=False)
        created = True
        for item in files:
            output = destination.joinpath(*PurePosixPath(item.path).parts)
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("xb") as handle:
                handle.write(item.payload)
            output.chmod(0o755 if item.mode == "100755" else 0o644)

        manifest_path = destination / MANIFEST_NAME
        with manifest_path.open("xb") as handle:
            handle.write(manifest)
        manifest_path.chmod(0o644)
    except OSError as exc:
        if created:
            shutil.rmtree(destination)
        raise ExportError(f"could not write public export: {exc}") from exc


def build_public_export(source: Path, destination: Path) -> ExportResult:
    """Export committed, public-safe files into a destination that does not exist."""

    repository = _resolve_repository(source)
    output = _resolve_new_destination(repository, destination)
    _require_clean_source(repository)
    revision = _run_git(repository, "rev-parse", "--verify", "HEAD^{commit}").decode().strip()
    files = _load_export_files(repository)
    manifest = _manifest_payload(revision, files)
    _write_export(output, files, manifest)
    return ExportResult(
        destination=output,
        manifest_path=output / MANIFEST_NAME,
        source_revision=revision,
        file_count=len(files),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a sanitized public directory from the clean TicketTune Git tree."
    )
    parser.add_argument("destination", type=Path, help="New directory to create outside the repo")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Clean Git repository root (defaults to this TicketTune checkout)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        result = build_public_export(arguments.source, arguments.destination)
    except ExportError as exc:
        print(f"Public export refused: {exc}", file=sys.stderr)
        return 1
    print(
        f"Created {result.file_count}-file public export for {result.source_revision} "
        f"at {result.destination}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
