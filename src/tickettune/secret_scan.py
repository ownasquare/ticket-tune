"""Scan TicketTune's tracked working tree and reachable Git history for secrets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess  # nosec B404
import sys
import tempfile
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path, PurePosixPath
from typing import Literal

from detect_secrets.core import baseline as detect_secrets_baseline
from detect_secrets.core.potential_secret import PotentialSecret
from detect_secrets.core.scan import scan_file
from detect_secrets.exceptions import UnableToReadBaselineError
from detect_secrets.settings import configure_settings_from_baseline, default_settings, get_filters

# Subprocess use is limited to an absolute Git executable with shell execution disabled.

DETECT_SECRETS_VERSION = "1.5.0"
BASELINE_NAME = ".secrets.baseline"

FindingScope = Literal["working-tree", "history"]
_HASH_PATTERN = re.compile(r"[0-9a-f]{40}")
_OBJECT_ID_PATTERN = re.compile(rb"[0-9a-f]{40,64}")


class SecretScanError(RuntimeError):
    """Raised when the repository or baseline cannot be scanned safely."""


@dataclass(frozen=True, slots=True)
class Finding:
    """A redacted detect-secrets finding."""

    scope: FindingScope
    path: str
    line_number: int
    secret_type: str
    secret_hash: str = field(repr=False)
    blob_oid: str | None = None

    @property
    def fingerprint(self) -> str:
        material = f"{self.secret_type}\0{self.secret_hash}".encode()
        return hashlib.sha256(material).hexdigest()[:12]

    @property
    def sort_key(self) -> tuple[int, str, str, int, str, str]:
        return (
            0 if self.scope == "working-tree" else 1,
            self.path,
            self.blob_oid or "",
            self.line_number,
            self.secret_type,
            self.secret_hash,
        )

    def public_dict(self) -> dict[str, object]:
        output: dict[str, object] = {
            "fingerprint": self.fingerprint,
            "line_number": self.line_number,
            "path": self.path,
            "scope": self.scope,
            "type": self.secret_type,
        }
        if self.blob_oid is not None:
            output["blob_oid"] = self.blob_oid
        return output


@dataclass(frozen=True, slots=True)
class ScanReport:
    """Deterministic, secret-free scan results."""

    source_revision: str
    tracked_file_count: int
    history_blob_count: int
    reviewed_suppression_count: int
    findings: tuple[Finding, ...]

    @property
    def passed(self) -> bool:
        return not self.findings

    def public_dict(self) -> dict[str, object]:
        return {
            "detector": {
                "name": "detect-secrets",
                "version": DETECT_SECRETS_VERSION,
            },
            "findings": [finding.public_dict() for finding in self.findings],
            "passed": self.passed,
            "reviewed_suppression_count": self.reviewed_suppression_count,
            "scanned": {
                "tracked_working_tree_files": self.tracked_file_count,
                "unique_reachable_history_blobs": self.history_blob_count,
            },
            "source_revision": self.source_revision,
        }


def _run_git(source: Path, *arguments: str, input_bytes: bytes | None = None) -> bytes:
    git = shutil.which("git")
    if git is None:
        raise SecretScanError("Git executable was not found")
    environment = os.environ.copy()
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    completed = subprocess.run(  # noqa: S603  # nosec B603
        [git, "-C", str(source), *arguments],
        check=False,
        capture_output=True,
        input=input_bytes,
        env=environment,
    )
    if completed.returncode != 0:
        raise SecretScanError(
            f"Git command failed without exposing command output (exit {completed.returncode})"
        )
    return completed.stdout


def _require_detector_version() -> None:
    try:
        installed = version("detect-secrets")
    except PackageNotFoundError as exc:
        raise SecretScanError("detect-secrets is not installed") from exc
    if installed != DETECT_SECRETS_VERSION:
        raise SecretScanError(
            f"detect-secrets {DETECT_SECRETS_VERSION} is required; found {installed}"
        )


def _resolve_repository(source: Path) -> Path:
    try:
        resolved = source.expanduser().resolve(strict=True)
    except OSError as exc:
        raise SecretScanError(f"source repository does not exist: {source}") from exc
    if not resolved.is_dir():
        raise SecretScanError("source repository must be a directory")

    raw_root = _run_git(resolved, "rev-parse", "--show-toplevel")
    try:
        git_root = Path(raw_root.decode("utf-8").strip()).resolve(strict=True)
    except (OSError, UnicodeDecodeError) as exc:
        raise SecretScanError("Git repository root is not a supported path") from exc
    if git_root != resolved:
        raise SecretScanError("source must be the Git repository root")
    return resolved


def _validate_repo_path(path: str) -> tuple[str, ...]:
    pure_path = PurePosixPath(path)
    if (
        pure_path.is_absolute()
        or not pure_path.parts
        or any(part in {"", ".", ".."} for part in pure_path.parts)
    ):
        raise SecretScanError("Git contains an unsupported path")
    return pure_path.parts


def _baseline_to_use(source: Path, baseline_path: Path | None) -> Path | None:
    if baseline_path is None:
        candidate = source / BASELINE_NAME
        return candidate if candidate.is_file() else None

    try:
        resolved = baseline_path.expanduser().resolve(strict=True)
    except OSError as exc:
        raise SecretScanError("reviewed baseline does not exist") from exc
    if not resolved.is_file() or resolved.name != BASELINE_NAME:
        raise SecretScanError(f"reviewed baseline must be a file named {BASELINE_NAME}")
    return resolved


def _reviewed_false_positives(baseline_path: Path | None) -> frozenset[tuple[str, str]]:
    if baseline_path is None:
        return frozenset()
    try:
        payload = detect_secrets_baseline.load_from_file(str(baseline_path))
    except UnableToReadBaselineError as exc:
        raise SecretScanError("reviewed baseline could not be read") from exc

    if payload.get("version") != DETECT_SECRETS_VERSION:
        raise SecretScanError(f"reviewed baseline must use detect-secrets {DETECT_SECRETS_VERSION}")
    results = payload.get("results")
    if not isinstance(results, dict):
        raise SecretScanError("reviewed baseline has no results mapping")

    reviewed: set[tuple[str, str]] = set()
    for entries in results.values():
        if not isinstance(entries, list):
            raise SecretScanError("reviewed baseline has an invalid results entry")
        for entry in entries:
            if not isinstance(entry, dict):
                raise SecretScanError("reviewed baseline has an invalid finding")
            if entry.get("is_secret") is not False:
                continue
            secret_type = entry.get("type")
            secret_hash = entry.get("hashed_secret")
            if not isinstance(secret_type, str) or not isinstance(secret_hash, str):
                raise SecretScanError("reviewed baseline finding is missing type or hash")
            if not _HASH_PATTERN.fullmatch(secret_hash):
                raise SecretScanError("reviewed baseline contains an invalid secret hash")
            reviewed.add((secret_type, secret_hash))
    return frozenset(reviewed)


def _tracked_paths(source: Path) -> tuple[str, ...]:
    raw_paths = _run_git(source, "ls-files", "--cached", "-z")
    paths: list[str] = []
    for raw_path in raw_paths.split(b"\0"):
        if not raw_path:
            continue
        try:
            path = raw_path.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise SecretScanError("tracked file path is not UTF-8") from exc
        _validate_repo_path(path)
        paths.append(path)
    return tuple(sorted(set(paths)))


def _reachable_history_blobs(source: Path) -> tuple[tuple[str, str], ...]:
    raw_objects = _run_git(source, "rev-list", "--objects", "--all")
    object_paths: dict[bytes, str] = {}
    for raw_line in raw_objects.splitlines():
        raw_oid, separator, raw_path = raw_line.partition(b" ")
        if not _OBJECT_ID_PATTERN.fullmatch(raw_oid):
            raise SecretScanError("Git returned an invalid object identifier")
        path = ""
        if separator:
            try:
                path = raw_path.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise SecretScanError("history file path is not UTF-8") from exc
        previous = object_paths.get(raw_oid)
        if previous is None or (path and path < previous):
            object_paths[raw_oid] = path

    if not object_paths:
        return ()
    batch_input = b"".join(oid + b"\n" for oid in sorted(object_paths))
    raw_types = _run_git(
        source,
        "cat-file",
        "--batch-check=%(objectname) %(objecttype)",
        input_bytes=batch_input,
    )
    blobs: list[tuple[str, str]] = []
    for raw_line in raw_types.splitlines():
        raw_oid, separator, raw_type = raw_line.partition(b" ")
        if not separator:
            raise SecretScanError("Git returned an invalid object type record")
        if raw_type != b"blob":
            continue
        path = object_paths[raw_oid]
        if not path:
            path = f"history-blob-{raw_oid.decode('ascii')}.txt"
        _validate_repo_path(path)
        blobs.append((raw_oid.decode("ascii"), path))
    return tuple(sorted(set(blobs)))


def _redact_potential_secret(
    potential: PotentialSecret,
    *,
    scope: FindingScope,
    path: str,
    blob_oid: str | None = None,
) -> Finding:
    finding = Finding(
        scope=scope,
        path=path,
        line_number=int(potential.line_number),
        secret_type=potential.type,
        secret_hash=potential.secret_hash,
        blob_oid=blob_oid,
    )
    potential.secret_value = None
    return finding


def _scan_path(
    actual_path: Path,
    *,
    scope: FindingScope,
    logical_path: str,
    blob_oid: str | None = None,
) -> Iterator[Finding]:
    for potential in scan_file(str(actual_path)):
        yield _redact_potential_secret(
            potential,
            scope=scope,
            path=logical_path,
            blob_oid=blob_oid,
        )


def _scan_working_tree(source: Path, temporary: Path) -> tuple[int, set[Finding]]:
    findings: set[Finding] = set()
    scanned = 0
    for logical_path in _tracked_paths(source):
        parts = _validate_repo_path(logical_path)
        actual_path = source.joinpath(*parts)
        if not actual_path.exists() and not actual_path.is_symlink():
            continue
        resolved_parent = actual_path.parent.resolve(strict=True)
        if not resolved_parent.is_relative_to(source):
            raise SecretScanError("tracked file parent escapes the repository")

        mode = actual_path.lstat().st_mode
        scan_target = actual_path
        if stat.S_ISLNK(mode):
            scan_target = temporary / "working-tree" / logical_path
            scan_target.parent.mkdir(parents=True, exist_ok=True)
            scan_target.write_text(os.readlink(actual_path), encoding="utf-8")
        elif not stat.S_ISREG(mode):
            continue

        scanned += 1
        findings.update(
            _scan_path(
                scan_target,
                scope="working-tree",
                logical_path=logical_path,
            )
        )
    return scanned, findings


def _scan_history(
    source: Path,
    blobs: Sequence[tuple[str, str]],
    temporary: Path,
) -> set[Finding]:
    findings: set[Finding] = set()
    for blob_oid, logical_path in blobs:
        parts = _validate_repo_path(logical_path)
        scan_target = temporary.joinpath("history", blob_oid, *parts)
        scan_target.parent.mkdir(parents=True, exist_ok=True)
        scan_target.write_bytes(_run_git(source, "cat-file", "blob", blob_oid))
        try:
            findings.update(
                _scan_path(
                    scan_target,
                    scope="history",
                    logical_path=logical_path,
                    blob_oid=blob_oid,
                )
            )
        finally:
            scan_target.unlink(missing_ok=True)
    return findings


def scan_repository(source: Path, baseline_path: Path | None = None) -> ScanReport:
    """Scan tracked working bytes and each unique reachable Git blob."""

    _require_detector_version()
    repository = _resolve_repository(source)
    reviewed_baseline = _baseline_to_use(repository, baseline_path)
    reviewed = _reviewed_false_positives(reviewed_baseline)
    revision = _run_git(repository, "rev-parse", "--verify", "HEAD^{commit}").decode().strip()
    history_blobs = _reachable_history_blobs(repository)

    with default_settings():
        configure_settings_from_baseline({}, filename=BASELINE_NAME)
        get_filters.cache_clear()
        with tempfile.TemporaryDirectory(prefix="tickettune-secret-scan-") as temporary_name:
            temporary = Path(temporary_name)
            tracked_count, working_findings = _scan_working_tree(repository, temporary)
            history_findings = _scan_history(repository, history_blobs, temporary)

    all_findings = working_findings | history_findings
    findings: list[Finding] = []
    suppressions = 0
    for finding in all_findings:
        if (finding.secret_type, finding.secret_hash) in reviewed:
            suppressions += 1
        else:
            findings.append(finding)

    return ScanReport(
        source_revision=revision,
        tracked_file_count=tracked_count,
        history_blob_count=len(history_blobs),
        reviewed_suppression_count=suppressions,
        findings=tuple(sorted(findings, key=lambda finding: finding.sort_key)),
    )


def _format_text(report: ScanReport) -> str:
    status = "PASS" if report.passed else "FAIL"
    lines = [
        f"Secret scan: {status}",
        f"Tracked working-tree files scanned: {report.tracked_file_count}",
        f"Unique reachable history blobs scanned: {report.history_blob_count}",
        f"Reviewed false-positive suppressions: {report.reviewed_suppression_count}",
    ]
    for finding in report.findings:
        if finding.scope == "working-tree":
            lines.append(
                f"[working-tree] finding path={finding.path} line={finding.line_number} "
                f"type={finding.secret_type} id={finding.fingerprint}"
            )
        else:
            lines.append(
                f"[history] finding blob={finding.blob_oid} path={finding.path} "
                f"line={finding.line_number} type={finding.secret_type} "
                f"id={finding.fingerprint}"
            )
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scan tracked TicketTune files and every reachable Git blob for secrets."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Git repository root (defaults to this TicketTune checkout)",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        help=f"Reviewed {BASELINE_NAME}; defaults to the repository copy when present",
    )
    parser.add_argument("--json", action="store_true", help="Print deterministic JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        report = scan_repository(arguments.source, arguments.baseline)
    except SecretScanError as exc:
        print(f"Secret scan could not run: {exc}", file=sys.stderr)
        return 2

    if arguments.json:
        print(json.dumps(report.public_dict(), indent=2, sort_keys=True))
    else:
        print(_format_text(report))
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
