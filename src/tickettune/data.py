"""Validated and reproducible dataset lifecycle for TicketTune."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import tempfile
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from tickettune.config import SplitConfig
from tickettune.schemas import (
    CATEGORY_LABELS,
    PRIORITY_LABELS,
    SENTIMENT_LABELS,
    TicketExample,
)
from tickettune.strict_json import (
    DuplicateJSONKeyError as _DuplicateJSONKeyError,
)
from tickettune.strict_json import (
    StrictJSONError,
    loads_strict,
)

DuplicateJSONKeyError = _DuplicateJSONKeyError

SplitName = Literal["train", "validation", "test"]
SPLIT_NAMES: tuple[SplitName, ...] = ("train", "validation", "test")

# Small strata retain the exact historical optimizer. Larger strata use a
# deterministic, label-aware candidate search whose work is capped independently
# of C(n, k). The global dynamic-programming frontier is capped as well.
_EXACT_CHOICE_LIMIT = 128
_BOUNDED_CHOICE_LIMIT = 128
_CHOICE_VARIANTS_PER_MODE = 32
_ALLOCATION_STATE_LIMIT = 4096

_PLACEHOLDER_PATTERN = re.compile(r"\[[A-Z][A-Z0-9_]*\]")
_PII_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "email address",
        re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
    ),
    ("US social-security number", re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")),
    (
        "payment-card-like number",
        re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)"),
    ),
    (
        "phone-like number",
        re.compile(r"(?<![\w.])\+?(?:\d[\s().-]*){7,14}\d(?![\w.])"),
    ),
)


class DatasetValidationError(ValueError):
    """A source-data error carrying its file and, when known, line number."""

    def __init__(self, path: Path, line_number: int | None, message: str) -> None:
        self.path = path
        self.line_number = line_number
        self.detail = message
        location = f"{path}:{line_number}" if line_number is not None else str(path)
        super().__init__(f"{location}: {message}")


class DatasetIntegrityError(ValueError):
    """A prepared dataset no longer matches its provenance manifest."""

    def __init__(self, manifest_path: Path, message: str) -> None:
        self.manifest_path = manifest_path
        self.detail = message
        super().__init__(f"{manifest_path}: {message}")


class LabelDistribution(BaseModel):
    """Counts used to audit the balance of a source or generated split."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    category: dict[str, int]
    priority: dict[str, int]
    sentiment: dict[str, int]


class SplitArtifact(BaseModel):
    """Immutable manifest entry for one generated JSONL split."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    file: str
    count: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    labels: LabelDistribution


class DatasetManifest(BaseModel):
    """Portable content manifest; it contains no machine-specific absolute paths."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    schema_version: Literal["1.0"] = "1.0"
    source_file: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    seed: int = Field(ge=0)
    split_fractions: dict[SplitName, float]
    total_examples: int = Field(ge=1)
    labels: LabelDistribution
    splits: dict[SplitName, SplitArtifact]
    split_ids: dict[SplitName, list[str]]


class PreparedDataset(BaseModel):
    """Result of preparation, including local paths and portable manifest fields."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    output_dir: Path
    manifest_path: Path
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    total_examples: int = Field(ge=1)
    split_ids: dict[SplitName, list[str]]
    splits: dict[SplitName, SplitArtifact]


class DatasetVerification(BaseModel):
    """Exact verified identities for a requested subset of prepared splits."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    status: Literal["verified"] = "verified"
    manifest_path: Path
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_path: Path
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    seed: int = Field(ge=0)
    split_fractions: dict[SplitName, float]
    verified_splits: tuple[SplitName, ...]
    split_paths: dict[SplitName, Path]
    split_counts: dict[SplitName, int]
    split_sha256: dict[SplitName, str]
    split_ids: dict[SplitName, tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class PreparedSplitSnapshot:
    """Stable run-scoped copy used as one exact framework input."""

    split_name: SplitName
    manifest_path: Path
    source_path: Path
    path: Path
    sha256: str
    size_bytes: int
    stat_identity: tuple[int, int, int, int, int]


def sha256_bytes(value: bytes) -> str:
    """Return a lowercase SHA-256 digest."""

    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash exact on-disk bytes without normalizing newlines or JSON."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stat_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _read_stable_regular_bytes(
    path: Path,
    *,
    manifest_path: Path,
    label: str,
) -> tuple[bytes, tuple[int, int, int, int, int]]:
    """Read one final path without following a symlink and bind its path identity."""

    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:  # pragma: no cover - TicketTune targets POSIX hosts.
        raise RuntimeError("stable dataset snapshots require O_NOFOLLOW support")
    flags = os.O_RDONLY | no_follow | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise DatasetIntegrityError(
            manifest_path,
            f"{label} must be a regular non-symlink file: {path}",
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise DatasetIntegrityError(
                manifest_path,
                f"{label} must be a regular non-symlink file: {path}",
            )
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
        current = os.lstat(path)
    except OSError as exc:
        raise DatasetIntegrityError(
            manifest_path,
            f"{label} changed while it was being read: {path}",
        ) from exc
    finally:
        os.close(descriptor)
    payload = b"".join(chunks)
    before_identity = _stat_identity(before)
    if (
        before_identity != _stat_identity(after)
        or (current.st_dev, current.st_ino) != (before.st_dev, before.st_ino)
        or len(payload) != before.st_size
    ):
        raise DatasetIntegrityError(
            manifest_path,
            f"{label} changed while it was being read: {path}",
        )
    return payload, before_identity


def _write_new_snapshot(path: Path, payload: bytes, *, manifest_path: Path) -> None:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:  # pragma: no cover - TicketTune targets POSIX hosts.
        raise RuntimeError("stable dataset snapshots require O_NOFOLLOW support")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | no_follow | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags, 0o400)
    except OSError as exc:
        raise DatasetIntegrityError(
            manifest_path,
            f"run-scoped snapshot must be newly created: {path}",
        ) from exc
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:  # pragma: no cover - defensive kernel boundary.
                raise OSError("snapshot write made no progress")
            offset += written
        os.fsync(descriptor)
    except OSError as exc:
        raise DatasetIntegrityError(
            manifest_path,
            f"run-scoped snapshot could not be written: {path}",
        ) from exc
    finally:
        os.close(descriptor)


def materialize_prepared_split_snapshots(
    verification: DatasetVerification,
    output_dir: Path,
    *,
    split_names: Sequence[SplitName],
) -> dict[SplitName, PreparedSplitSnapshot]:
    """Copy verified split bytes into a new run directory through stable descriptors."""

    requested = tuple(split_names)
    if not requested or len(set(requested)) != len(requested):
        raise DatasetIntegrityError(
            verification.manifest_path,
            "snapshot split names must be a non-empty unique sequence",
        )
    missing = [name for name in requested if name not in verification.verified_splits]
    if missing:
        raise DatasetIntegrityError(
            verification.manifest_path,
            f"snapshot splits were not verified: {missing}",
        )

    destination_dir = output_dir.expanduser()
    try:
        destination_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
    except OSError as exc:
        raise DatasetIntegrityError(
            verification.manifest_path,
            f"run-scoped snapshot directory must be new: {destination_dir}",
        ) from exc

    snapshots: dict[SplitName, PreparedSplitSnapshot] = {}
    for split_name in requested:
        source_path = verification.split_paths[split_name]
        source_payload, _source_identity = _read_stable_regular_bytes(
            source_path,
            manifest_path=verification.manifest_path,
            label=f"{split_name} split",
        )
        source_sha256 = sha256_bytes(source_payload)
        if source_sha256 != verification.split_sha256[split_name]:
            raise DatasetIntegrityError(
                verification.manifest_path,
                f"{split_name} split changed after dataset verification",
            )

        destination = destination_dir / f"{split_name}.jsonl"
        _write_new_snapshot(
            destination,
            source_payload,
            manifest_path=verification.manifest_path,
        )
        snapshot_payload, snapshot_identity = _read_stable_regular_bytes(
            destination,
            manifest_path=verification.manifest_path,
            label=f"run-scoped {split_name} snapshot",
        )
        snapshot = PreparedSplitSnapshot(
            split_name=split_name,
            manifest_path=verification.manifest_path,
            source_path=source_path,
            path=destination,
            sha256=source_sha256,
            size_bytes=len(snapshot_payload),
            stat_identity=snapshot_identity,
        )
        verify_prepared_split_snapshot(snapshot)
        snapshots[split_name] = snapshot

    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    directory_descriptor = os.open(destination_dir, directory_flags)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
    return snapshots


def read_prepared_split_snapshot(snapshot: PreparedSplitSnapshot) -> bytes:
    """Return exact run-scoped bytes after identity and digest verification."""

    payload, observed_identity = _read_stable_regular_bytes(
        snapshot.path,
        manifest_path=snapshot.manifest_path,
        label=f"run-scoped {snapshot.split_name} snapshot",
    )
    if observed_identity != snapshot.stat_identity:
        raise DatasetIntegrityError(
            snapshot.manifest_path,
            f"run-scoped {snapshot.split_name} snapshot identity changed",
        )
    if len(payload) != snapshot.size_bytes or sha256_bytes(payload) != snapshot.sha256:
        raise DatasetIntegrityError(
            snapshot.manifest_path,
            f"run-scoped {snapshot.split_name} snapshot bytes changed",
        )
    return payload


def verify_prepared_split_snapshot(snapshot: PreparedSplitSnapshot) -> None:
    """Reject replacement or byte mutation of a run-scoped framework input."""

    read_prepared_split_snapshot(snapshot)


def load_json_strict(payload: str | bytes | bytearray) -> object:
    """Compatibility wrapper for TicketTune's shared strict JSON decoder."""

    return loads_strict(payload)


def _read_manifest_strict(
    manifest_path: Path,
) -> tuple[DatasetManifest, str, tuple[int, int, int, int, int]]:
    if not manifest_path.is_file():
        raise DatasetIntegrityError(manifest_path, "dataset manifest not found")
    if manifest_path.is_symlink():
        raise DatasetIntegrityError(manifest_path, "dataset manifest must not be a symlink")
    try:
        payload, identity = _read_stable_regular_bytes(
            manifest_path,
            manifest_path=manifest_path,
            label="dataset manifest",
        )
        manifest = DatasetManifest.model_validate(load_json_strict(payload), strict=True)
        return manifest, sha256_bytes(payload), identity
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        StrictJSONError,
        ValidationError,
    ) as exc:
        raise DatasetIntegrityError(
            manifest_path,
            f"manifest schema validation failed: {exc}",
        ) from exc


def _manifest_split_ids(
    payload: bytes,
    *,
    split_name: str,
    manifest_path: Path,
) -> list[str]:
    ids: list[str] = []
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DatasetIntegrityError(
            manifest_path,
            f"{split_name}: invalid UTF-8: {exc}",
        ) from exc
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            raise DatasetIntegrityError(
                manifest_path,
                f"{split_name}:{line_number}: blank lines are not allowed",
            )
        try:
            value = load_json_strict(line)
        except (json.JSONDecodeError, StrictJSONError) as exc:
            raise DatasetIntegrityError(
                manifest_path,
                f"{split_name}:{line_number}: invalid JSON: {exc}",
            ) from exc
        if not isinstance(value, dict):
            raise DatasetIntegrityError(
                manifest_path,
                f"{split_name}:{line_number}: expected a JSON object",
            )
        record_id = value.get("id")
        if not isinstance(record_id, str) or not record_id.strip():
            raise DatasetIntegrityError(
                manifest_path,
                f"{split_name}:{line_number}: expected a non-empty string id",
            )
        ids.append(record_id)
    return ids


def verify_prepared_dataset(
    source_path: Path,
    processed_dir: Path,
    *,
    seed: int,
    splits: SplitConfig,
    required_splits: Sequence[SplitName],
) -> DatasetVerification:
    """Verify configured source and requested prepared splits against ``manifest.json``.

    This path depends only on the lightweight core package. It is safe to call
    before importing Transformers, PEFT, TRL, Torch, or Datasets.
    """

    requested = tuple(required_splits)
    if not requested:
        manifest_path = processed_dir.expanduser().resolve() / "manifest.json"
        raise DatasetIntegrityError(manifest_path, "at least one required split is needed")
    if len(set(requested)) != len(requested) or any(
        split_name not in SPLIT_NAMES for split_name in requested
    ):
        manifest_path = processed_dir.expanduser().resolve() / "manifest.json"
        raise DatasetIntegrityError(
            manifest_path,
            f"required splits must be unique canonical names; received {requested!r}",
        )

    source = source_path.expanduser().resolve()
    output = processed_dir.expanduser().resolve()
    manifest_path = output / "manifest.json"
    manifest, manifest_digest, manifest_identity = _read_manifest_strict(manifest_path)

    if not source.is_file():
        raise DatasetIntegrityError(manifest_path, f"configured source file not found: {source}")
    if manifest.source_file != source.name or Path(manifest.source_file).name != source.name:
        raise DatasetIntegrityError(
            manifest_path,
            "source filename mismatch: "
            f"configured {source.name!r}, manifest declares {manifest.source_file!r}",
        )
    source_payload, source_identity = _read_stable_regular_bytes(
        source,
        manifest_path=manifest_path,
        label="configured source",
    )
    source_digest_before = sha256_bytes(source_payload)
    if source_digest_before != manifest.source_sha256:
        raise DatasetIntegrityError(
            manifest_path,
            "source SHA-256 mismatch: "
            f"expected {manifest.source_sha256}, found {source_digest_before}",
        )
    if manifest.seed != seed:
        raise DatasetIntegrityError(
            manifest_path,
            f"seed mismatch: configured {seed}, manifest declares {manifest.seed}",
        )

    expected_fractions: dict[SplitName, float] = {
        "train": splits.train,
        "validation": splits.validation,
        "test": splits.test,
    }
    if set(manifest.split_fractions) != set(SPLIT_NAMES) or any(
        not math.isclose(
            manifest.split_fractions[split_name],
            expected_fractions[split_name],
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for split_name in SPLIT_NAMES
    ):
        raise DatasetIntegrityError(
            manifest_path,
            "split fractions mismatch: "
            f"configured {expected_fractions}, manifest declares {manifest.split_fractions}",
        )
    if set(manifest.splits) != set(SPLIT_NAMES):
        raise DatasetIntegrityError(
            manifest_path,
            f"manifest splits must be exactly {SPLIT_NAMES!r}",
        )
    if set(manifest.split_ids) != set(SPLIT_NAMES):
        raise DatasetIntegrityError(
            manifest_path,
            f"manifest split_ids must be exactly {SPLIT_NAMES!r}",
        )

    all_ids: list[str] = []
    total_count = 0
    canonical_paths: dict[SplitName, Path] = {}
    for split_name in SPLIT_NAMES:
        artifact = manifest.splits[split_name]
        canonical_filename = f"{split_name}.jsonl"
        if artifact.file != canonical_filename:
            raise DatasetIntegrityError(
                manifest_path,
                f"{split_name} must use canonical filename {canonical_filename!r}; "
                f"manifest declares {artifact.file!r}",
            )
        split_path = output / canonical_filename
        if split_path.is_symlink() or split_path.parent != output:
            raise DatasetIntegrityError(
                manifest_path,
                f"{split_name} split must be a non-symlink file inside {output}",
            )
        canonical_paths[split_name] = split_path
        declared_ids = manifest.split_ids[split_name]
        if len(declared_ids) != artifact.count:
            raise DatasetIntegrityError(
                manifest_path,
                f"{split_name} manifest count mismatch: artifact declares {artifact.count}, "
                f"split_ids contains {len(declared_ids)}",
            )
        if any(not record_id.strip() for record_id in declared_ids):
            raise DatasetIntegrityError(
                manifest_path,
                f"{split_name} split_ids must contain non-empty strings",
            )
        if len(set(declared_ids)) != len(declared_ids):
            raise DatasetIntegrityError(
                manifest_path,
                f"{split_name} split_ids contains duplicate IDs",
            )
        all_ids.extend(declared_ids)
        total_count += artifact.count

    if total_count != manifest.total_examples:
        raise DatasetIntegrityError(
            manifest_path,
            "total example count mismatch: "
            f"splits declare {total_count}, manifest declares {manifest.total_examples}",
        )
    if len(set(all_ids)) != len(all_ids):
        raise DatasetIntegrityError(manifest_path, "manifest IDs overlap across splits")

    try:
        source_examples = _load_examples_from_bytes(source_payload, source)
    except DatasetValidationError as exc:
        raise DatasetIntegrityError(
            manifest_path,
            f"configured source validation failed: {exc}",
        ) from exc

    try:
        expected_splits = split_examples(source_examples, seed=seed, splits=splits)
    except ValueError as exc:
        raise DatasetIntegrityError(
            manifest_path,
            f"deterministic split regeneration failed: {exc}",
        ) from exc
    if manifest.total_examples != len(source_examples):
        raise DatasetIntegrityError(
            manifest_path,
            "total example count mismatch against the validated source: "
            f"manifest declares {manifest.total_examples}, source has {len(source_examples)}",
        )
    expected_source_labels = _distribution(source_examples)
    if manifest.labels != expected_source_labels:
        raise DatasetIntegrityError(
            manifest_path,
            "source label distribution mismatch against the validated source",
        )

    expected_bytes: dict[SplitName, bytes] = {}
    expected_ids_by_split: dict[SplitName, list[str]] = {}
    for split_name in SPLIT_NAMES:
        expected_examples = expected_splits[split_name]
        split_payload = _canonical_split_bytes(expected_examples)
        split_digest = sha256_bytes(split_payload)
        expected_ids = [item.id for item in expected_examples]
        expected_labels = _distribution(expected_examples)
        artifact = manifest.splits[split_name]
        if artifact.count != len(expected_examples):
            raise DatasetIntegrityError(
                manifest_path,
                f"{split_name} count mismatch against deterministic source projection: "
                f"manifest declares {artifact.count}, expected {len(expected_examples)}",
            )
        if manifest.split_ids[split_name] != expected_ids:
            raise DatasetIntegrityError(
                manifest_path,
                f"{split_name} ordered IDs mismatch against deterministic source projection",
            )
        if artifact.labels != expected_labels:
            raise DatasetIntegrityError(
                manifest_path,
                f"{split_name} label distribution mismatch against deterministic source projection",
            )
        if artifact.sha256 != split_digest:
            raise DatasetIntegrityError(
                manifest_path,
                f"{split_name} SHA-256 mismatch against deterministic source projection: "
                f"manifest declares {artifact.sha256}, expected {split_digest}",
            )
        expected_bytes[split_name] = split_payload
        expected_ids_by_split[split_name] = expected_ids

    verified_paths: dict[SplitName, Path] = {}
    verified_counts: dict[SplitName, int] = {}
    verified_hashes: dict[SplitName, str] = {}
    for split_name in requested:
        split_path = canonical_paths[split_name]
        if not split_path.is_file():
            raise DatasetIntegrityError(
                manifest_path,
                f"required split file not found: {split_path}",
            )
        actual_payload, _split_identity = _read_stable_regular_bytes(
            split_path,
            manifest_path=manifest_path,
            label=f"required {split_name} split",
        )
        parse_error: DatasetIntegrityError | None = None
        try:
            actual_ids = _manifest_split_ids(
                actual_payload,
                split_name=split_path.name,
                manifest_path=manifest_path,
            )
        except DatasetIntegrityError as exc:
            if isinstance(exc.__cause__, StrictJSONError):
                raise
            parse_error = exc
            actual_ids = []
        actual_digest = sha256_bytes(actual_payload)
        expected_digest = manifest.splits[split_name].sha256
        if actual_digest != expected_digest:
            raise DatasetIntegrityError(
                manifest_path,
                f"{split_name} SHA-256 mismatch: expected {expected_digest}, found {actual_digest}",
            )
        if actual_payload != expected_bytes[split_name]:
            raise DatasetIntegrityError(
                manifest_path,
                f"{split_name} bytes do not match the deterministic source projection",
            )
        if parse_error is not None:
            raise parse_error
        expected_ids = manifest.split_ids[split_name]
        if len(actual_ids) != manifest.splits[split_name].count:
            raise DatasetIntegrityError(
                manifest_path,
                f"{split_name} count mismatch: manifest declares "
                f"{manifest.splits[split_name].count}, file contains {len(actual_ids)}",
            )
        if actual_ids != expected_ids:
            raise DatasetIntegrityError(
                manifest_path,
                f"{split_name} ordered IDs mismatch between manifest and file",
            )
        verified_paths[split_name] = split_path
        verified_counts[split_name] = len(actual_ids)
        verified_hashes[split_name] = actual_digest

    source_payload_final, source_identity_final = _read_stable_regular_bytes(
        source,
        manifest_path=manifest_path,
        label="configured source",
    )
    if source_identity_final != source_identity or source_payload_final != source_payload:
        raise DatasetIntegrityError(
            manifest_path,
            "configured source changed during prepared-dataset verification",
        )
    manifest_payload_final, manifest_identity_final = _read_stable_regular_bytes(
        manifest_path,
        manifest_path=manifest_path,
        label="dataset manifest",
    )
    if (
        manifest_identity_final != manifest_identity
        or sha256_bytes(manifest_payload_final) != manifest_digest
    ):
        raise DatasetIntegrityError(
            manifest_path,
            "dataset manifest changed during verification",
        )

    return DatasetVerification(
        manifest_path=manifest_path,
        manifest_sha256=manifest_digest,
        source_path=source,
        source_sha256=source_digest_before,
        seed=seed,
        split_fractions=expected_fractions,
        verified_splits=requested,
        split_paths=verified_paths,
        split_counts=verified_counts,
        split_sha256=verified_hashes,
        split_ids={name: tuple(expected_ids_by_split[name]) for name in requested},
    )


def normalize_content(value: str) -> str:
    """Normalize text for content-level duplicate and leakage checks."""

    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = _PLACEHOLDER_PATTERN.sub("[pii]", normalized)
    normalized = re.sub(r"[^\w]+", " ", normalized, flags=re.UNICODE)
    return " ".join(normalized.split())


def content_fingerprint(example: TicketExample) -> str:
    """Hash normalized user intent rather than mutable record metadata."""

    return sha256_bytes(normalize_content(example.messages[1].content).encode("utf-8"))


def _find_unredacted_pii(example: TicketExample) -> list[str]:
    text = "\n".join(message.content for message in example.messages[1:])
    findings: list[str] = []
    for label, pattern in _PII_PATTERNS:
        if pattern.search(text):
            findings.append(label)
    return findings


def _load_examples_from_bytes(payload: bytes, source_path: Path) -> list[TicketExample]:
    """Parse one already-snapshotted JSONL payload with source-aware errors."""

    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DatasetValidationError(source_path, None, f"invalid UTF-8: {exc}") from exc
    examples: list[TicketExample] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            raise DatasetValidationError(source_path, line_number, "blank lines are not allowed")
        try:
            examples.append(TicketExample.model_validate(load_json_strict(line), strict=True))
        except (json.JSONDecodeError, StrictJSONError, ValidationError) as exc:
            raise DatasetValidationError(source_path, line_number, str(exc)) from exc
    if not examples:
        raise DatasetValidationError(source_path, None, "dataset must contain at least one example")
    validate_examples(examples, source_path=source_path)
    return examples


def load_examples(path: Path) -> list[TicketExample]:
    """Load exact stable JSONL bytes with source-line-aware validation failures."""

    source_path = path.expanduser().resolve()
    try:
        payload, _identity = _read_stable_regular_bytes(
            source_path,
            manifest_path=source_path,
            label="dataset source",
        )
    except DatasetIntegrityError as exc:
        raise DatasetValidationError(source_path, None, exc.detail) from exc
    return _load_examples_from_bytes(payload, source_path)


def validate_examples(
    examples: Sequence[TicketExample], *, source_path: Path = Path("<memory>")
) -> None:
    """Reject duplicate IDs, duplicate normalized intents, and unredacted PII."""

    if not examples:
        raise DatasetValidationError(source_path, None, "dataset must contain at least one example")

    id_lines: dict[str, int] = {}
    fingerprint_lines: dict[str, tuple[int, str]] = {}
    for line_number, example in enumerate(examples, 1):
        if example.id in id_lines:
            first_line = id_lines[example.id]
            raise DatasetValidationError(
                source_path,
                line_number,
                f"duplicate id {example.id!r}; first seen on line {first_line}",
            )
        id_lines[example.id] = line_number

        fingerprint = content_fingerprint(example)
        if fingerprint in fingerprint_lines:
            first_line, first_id = fingerprint_lines[fingerprint]
            raise DatasetValidationError(
                source_path,
                line_number,
                "duplicate normalized user content for "
                f"{example.id!r}; matches {first_id!r} on line {first_line}",
            )
        fingerprint_lines[fingerprint] = (line_number, example.id)

        pii_findings = _find_unredacted_pii(example)
        if pii_findings:
            raise DatasetValidationError(
                source_path,
                line_number,
                "possible unredacted PII detected: " + ", ".join(pii_findings),
            )


def _order_key(seed: int, scope: str, example_id: str) -> bytes:
    return hashlib.sha256(f"{seed}:{scope}:{example_id}".encode()).digest()


def _allocate_counts(size: int, splits: SplitConfig) -> dict[SplitName, int]:
    fractions: dict[SplitName, float] = {
        "train": splits.train,
        "validation": splits.validation,
        "test": splits.test,
    }
    raw = {name: size * fraction for name, fraction in fractions.items()}
    counts = {name: int(raw[name]) for name in SPLIT_NAMES}
    remaining = size - sum(counts.values())
    order = sorted(
        SPLIT_NAMES,
        key=lambda name: (-(raw[name] - counts[name]), SPLIT_NAMES.index(name)),
    )
    for name in order[:remaining]:
        counts[name] += 1

    # If a category has enough records for every split, preserve category
    # coverage even when the largest-remainder result rounds a small split to 0.
    if size >= len(SPLIT_NAMES):
        for empty_name in (name for name in SPLIT_NAMES if counts[name] == 0):
            donor = max(SPLIT_NAMES, key=lambda name: counts[name])
            if counts[donor] > 1:
                counts[donor] -= 1
                counts[empty_name] += 1
    return counts


def _bounded_category_choices(
    available: Sequence[TicketExample],
    required: int,
    *,
    seed: int,
    split_name: SplitName,
    category: str,
) -> tuple[tuple[TicketExample, ...], ...]:
    """Return a deterministic bounded set of label-balanced subset candidates.

    Exact enumeration is retained only when the complete candidate space is at
    most ``_EXACT_CHOICE_LIMIT``. For a larger C(n, k), fixed-count variants
    round-robin through joint, priority, and sentiment buckets with seeded,
    stable bucket order and rotations. The returned candidate count never
    depends on C(n, k).
    """

    if required < 0 or required > len(available):
        raise ValueError("required category choice count is outside available bounds")
    combination_count = math.comb(len(available), required)
    if combination_count <= _EXACT_CHOICE_LIMIT:
        return tuple(combinations(available, required))

    position = {item.id: index for index, item in enumerate(available)}
    candidates: list[tuple[TicketExample, ...]] = []
    seen: set[tuple[str, ...]] = set()

    def remember(selected: Sequence[TicketExample]) -> None:
        choice = tuple(sorted(selected, key=lambda item: position[item.id]))
        identity = tuple(item.id for item in choice)
        if len(choice) == required and identity not in seen:
            seen.add(identity)
            candidates.append(choice)

    remember(available[:required])
    bucket_modes = (
        "joint",
        "priority",
        "sentiment",
    )
    for mode in bucket_modes:
        buckets: defaultdict[tuple[int, ...], list[TicketExample]] = defaultdict(list)
        for item in available:
            priority_index = PRIORITY_LABELS.index(item.expected.priority)
            sentiment_index = SENTIMENT_LABELS.index(item.expected.sentiment)
            bucket_key: tuple[int, ...]
            if mode == "joint":
                bucket_key = (priority_index, sentiment_index)
            elif mode == "priority":
                bucket_key = (priority_index,)
            else:
                bucket_key = (sentiment_index,)
            buckets[bucket_key].append(item)

        for variant in range(_CHOICE_VARIANTS_PER_MODE):
            bucket_order = sorted(
                buckets,
                key=lambda current_bucket: _order_key(
                    seed,
                    f"{split_name}:{category}:bounded:{mode}:{variant}",
                    ":".join(str(value) for value in current_bucket),
                ),
            )
            cursors = {
                current_bucket: variant % len(buckets[current_bucket])
                for current_bucket in bucket_order
            }
            consumed = {current_bucket: 0 for current_bucket in bucket_order}
            selected: list[TicketExample] = []
            while len(selected) < required:
                progressed = False
                for current_bucket in bucket_order:
                    items = buckets[current_bucket]
                    if consumed[current_bucket] >= len(items):
                        continue
                    index = (cursors[current_bucket] + consumed[current_bucket]) % len(items)
                    selected.append(items[index])
                    consumed[current_bucket] += 1
                    progressed = True
                    if len(selected) == required:
                        break
                if not progressed:
                    break
            remember(selected)
            if len(candidates) >= _BOUNDED_CHOICE_LIMIT:
                return tuple(candidates)
    return tuple(candidates)


def _select_bounded_holdout_greedily(
    grouped: Mapping[str, Sequence[TicketExample]],
    requested: Mapping[str, int],
    *,
    excluded_ids: set[str],
    seed: int,
    split_name: SplitName,
) -> list[TicketExample]:
    """Select large strata incrementally with the deterministic allocation score."""

    priority_counts = (0,) * len(PRIORITY_LABELS)
    sentiment_counts = (0,) * len(SENTIMENT_LABELS)
    selected: tuple[TicketExample, ...] = ()
    tie_key: tuple[bytes, ...] = ()

    for category in sorted(grouped):
        required = requested[category]
        available = sorted(
            (item for item in grouped[category] if item.id not in excluded_ids),
            key=lambda item: _order_key(seed, f"{split_name}:{category}", item.id),
        )
        if required > len(available):
            raise ValueError(
                f"cannot allocate {required} {split_name} examples for {category!r}; "
                f"only {len(available)} remain"
            )
        choices = _bounded_category_choices(
            available,
            required,
            seed=seed,
            split_name=split_name,
            category=category,
        )
        candidates: list[
            tuple[
                tuple[int, int, int, int, int, tuple[bytes, ...]],
                tuple[TicketExample, ...],
                tuple[int, ...],
                tuple[int, ...],
                tuple[bytes, ...],
            ]
        ] = []
        for choice in choices:
            next_priorities = list(priority_counts)
            next_sentiments = list(sentiment_counts)
            for item in choice:
                next_priorities[PRIORITY_LABELS.index(item.expected.priority)] += 1
                next_sentiments[SENTIMENT_LABELS.index(item.expected.sentiment)] += 1
            choice_tie_key = tuple(
                _order_key(
                    seed,
                    f"{split_name}:{category}:choice:{slot}",
                    item.id,
                )
                for slot, item in enumerate(choice)
            )
            next_tie_key = tie_key + choice_tie_key
            candidates.append(
                (
                    (
                        sum(count == 0 for count in next_sentiments),
                        sum(count == 0 for count in next_priorities),
                        max(next_priorities) - min(next_priorities),
                        sum(count * count for count in next_priorities),
                        max(next_sentiments) - min(next_sentiments),
                        next_tie_key,
                    ),
                    choice,
                    tuple(next_priorities),
                    tuple(next_sentiments),
                    choice_tie_key,
                )
            )
        if not candidates:
            raise ValueError(f"no valid deterministic allocation exists for {split_name}")
        (
            _score,
            choice,
            priority_counts,
            sentiment_counts,
            choice_tie_key,
        ) = min(candidates, key=lambda candidate: candidate[0])
        selected += choice
        tie_key += choice_tie_key
    return list(selected)


def _bounded_greedy_candidate_preserves_coverage(
    grouped: Mapping[str, Sequence[TicketExample]],
    candidate: Sequence[TicketExample],
) -> bool:
    source = [item for category in sorted(grouped) for item in grouped[category]]
    mirrored: dict[SplitName, Sequence[TicketExample]] = {
        "validation": candidate,
        "test": candidate,
    }
    try:
        assert_holdout_label_coverage(source, mirrored)
    except ValueError:
        return False
    return True


def _select_holdout(
    grouped: Mapping[str, Sequence[TicketExample]],
    requested: Mapping[str, int],
    *,
    excluded_ids: set[str],
    seed: int,
    split_name: SplitName,
) -> list[TicketExample]:
    """Select a category-stratified holdout with bounded label-aware search."""

    def category_uses_bounded_choices(category: str) -> bool:
        available_count = sum(item.id not in excluded_ids for item in grouped[category])
        required = requested[category]
        return (
            0 <= required <= available_count
            and math.comb(available_count, required) > _EXACT_CHOICE_LIMIT
        )

    uses_large_multicategory_search = len(grouped) > 1 and any(
        category_uses_bounded_choices(category) for category in grouped
    )
    if uses_large_multicategory_search:
        # A full beam multiplies thousands of aggregate states by every bounded
        # choice for every category. Select incrementally with the same score,
        # retain the coverage audit as a fail-closed guard, and fall back to the
        # beam below only when the bounded candidate misses a feasible label.
        candidate = _select_bounded_holdout_greedily(
            grouped,
            requested,
            excluded_ids=excluded_ids,
            seed=seed,
            split_name=split_name,
        )
        if _bounded_greedy_candidate_preserves_coverage(grouped, candidate):
            return candidate

    zero_priorities = (0,) * len(PRIORITY_LABELS)
    zero_sentiments = (0,) * len(SENTIMENT_LABELS)
    state_key = (zero_priorities, zero_sentiments)
    states: dict[
        tuple[tuple[int, ...], tuple[int, ...]],
        tuple[tuple[TicketExample, ...], tuple[bytes, ...]],
    ] = {state_key: ((), ())}

    def allocation_score(
        state: tuple[
            tuple[tuple[int, ...], tuple[int, ...]],
            tuple[tuple[TicketExample, ...], tuple[bytes, ...]],
        ],
    ) -> tuple[int, int, int, int, int, tuple[bytes, ...]]:
        (priority_counts, sentiment_counts), (_selected, tie_key) = state
        priority_range = max(priority_counts) - min(priority_counts)
        sentiment_range = max(sentiment_counts) - min(sentiment_counts)
        return (
            sum(count == 0 for count in sentiment_counts),
            sum(count == 0 for count in priority_counts),
            priority_range,
            sum(count * count for count in priority_counts),
            sentiment_range,
            tie_key,
        )

    for category in sorted(grouped):
        required = requested[category]
        available = sorted(
            (item for item in grouped[category] if item.id not in excluded_ids),
            key=lambda item: _order_key(seed, f"{split_name}:{category}", item.id),
        )
        if required > len(available):
            raise ValueError(
                f"cannot allocate {required} {split_name} examples for {category!r}; "
                f"only {len(available)} remain"
            )
        choices = _bounded_category_choices(
            available,
            required,
            seed=seed,
            split_name=split_name,
            category=category,
        )
        choice_metadata: list[
            tuple[
                tuple[TicketExample, ...],
                tuple[int, ...],
                tuple[int, ...],
                tuple[bytes, ...],
            ]
        ] = []
        for choice in choices:
            choice_priority_delta = [0] * len(PRIORITY_LABELS)
            choice_sentiment_delta = [0] * len(SENTIMENT_LABELS)
            for item in choice:
                choice_priority_delta[PRIORITY_LABELS.index(item.expected.priority)] += 1
                choice_sentiment_delta[SENTIMENT_LABELS.index(item.expected.sentiment)] += 1
            choice_metadata.append(
                (
                    choice,
                    tuple(choice_priority_delta),
                    tuple(choice_sentiment_delta),
                    tuple(
                        _order_key(
                            seed,
                            f"{split_name}:{category}:choice:{slot}",
                            item.id,
                        )
                        for slot, item in enumerate(choice)
                    ),
                )
            )
        next_states: dict[
            tuple[tuple[int, ...], tuple[int, ...]],
            tuple[tuple[TicketExample, ...], tuple[bytes, ...]],
        ] = {}
        for (priority_counts, sentiment_counts), (selected, tie_key) in states.items():
            for (
                choice,
                cached_priority_delta,
                cached_sentiment_delta,
                choice_tie_key,
            ) in choice_metadata:
                next_key = (
                    tuple(
                        observed + delta
                        for observed, delta in zip(
                            priority_counts,
                            cached_priority_delta,
                            strict=True,
                        )
                    ),
                    tuple(
                        observed + delta
                        for observed, delta in zip(
                            sentiment_counts,
                            cached_sentiment_delta,
                            strict=True,
                        )
                    ),
                )
                next_selected = selected + choice
                next_tie_key = tie_key + choice_tie_key
                current = next_states.get(next_key)
                if current is None or next_tie_key < current[1]:
                    next_states[next_key] = (next_selected, next_tie_key)
        if len(next_states) > _ALLOCATION_STATE_LIMIT:
            states = dict(
                sorted(next_states.items(), key=allocation_score)[:_ALLOCATION_STATE_LIMIT]
            )
        else:
            states = next_states

    if not states:
        raise ValueError(f"no valid deterministic allocation exists for {split_name}")

    return list(min(states.items(), key=allocation_score)[1][0])


def assert_holdout_label_coverage(
    examples: Sequence[TicketExample],
    splits: Mapping[SplitName, Sequence[TicketExample]],
) -> None:
    """Fail closed when feasible holdouts omit canonical labels or skew priorities."""

    holdout_names: tuple[SplitName, SplitName] = ("validation", "test")
    dimensions = (
        ("category", CATEGORY_LABELS),
        ("priority", PRIORITY_LABELS),
        ("sentiment", SENTIMENT_LABELS),
    )
    for field, canonical_labels in dimensions:
        source_counts = Counter(str(getattr(item.expected, field)) for item in examples)
        canonical = {str(label) for label in canonical_labels}
        feasible = (
            set(source_counts) == canonical
            and all(source_counts[label] >= len(holdout_names) for label in canonical)
            and all(len(splits.get(name, ())) >= len(canonical) for name in holdout_names)
        )
        if not feasible:
            continue
        for name in holdout_names:
            counts = Counter(str(getattr(item.expected, field)) for item in splits.get(name, ()))
            missing = sorted(canonical - set(counts))
            if missing:
                raise ValueError(
                    f"holdout label coverage error: {name} is missing canonical "
                    f"{field} labels {missing}"
                )
            if field == "priority":
                observed = [counts[label] for label in canonical_labels]
                if max(observed) - min(observed) > 1:
                    raise ValueError(
                        f"holdout label coverage error: {name} priority labels "
                        f"are imbalanced: {dict(sorted(counts.items()))}"
                    )


def split_examples(
    examples: Sequence[TicketExample], *, seed: int, splits: SplitConfig
) -> dict[SplitName, list[TicketExample]]:
    """Deterministically stratify categories and balance holdout task labels."""

    grouped: defaultdict[str, list[TicketExample]] = defaultdict(list)
    for example in examples:
        grouped[example.expected.category].append(example)

    counts_by_category = {
        category: _allocate_counts(len(items), splits) for category, items in grouped.items()
    }
    result: dict[SplitName, list[TicketExample]] = {name: [] for name in SPLIT_NAMES}
    result["validation"] = _select_holdout(
        grouped,
        {category: counts["validation"] for category, counts in counts_by_category.items()},
        excluded_ids=set(),
        seed=seed,
        split_name="validation",
    )
    validation_ids = {item.id for item in result["validation"]}
    result["test"] = _select_holdout(
        grouped,
        {category: counts["test"] for category, counts in counts_by_category.items()},
        excluded_ids=validation_ids,
        seed=seed,
        split_name="test",
    )
    holdout_ids = validation_ids | {item.id for item in result["test"]}
    for category in sorted(grouped):
        remaining = [item for item in grouped[category] if item.id not in holdout_ids]
        expected_train = counts_by_category[category]["train"]
        if len(remaining) != expected_train:
            raise ValueError(
                f"internal split allocation error for category {category!r}: "
                f"expected {expected_train} train examples, found {len(remaining)}"
            )
        result["train"].extend(remaining)

    # A second deterministic shuffle prevents category blocks from affecting
    # batching while retaining exact membership and reproducibility.
    for name in SPLIT_NAMES:
        result[name].sort(key=lambda item: _order_key(seed, f"split:{name}", item.id))
    assert_no_split_leakage(result)
    assert_holdout_label_coverage(examples, result)
    return result


def assert_no_split_leakage(splits: Mapping[SplitName, Sequence[TicketExample]]) -> None:
    """Fail if an ID or normalized user intent appears in multiple splits."""

    seen_ids: dict[str, SplitName] = {}
    seen_content: dict[str, tuple[SplitName, str]] = {}
    for split_name in SPLIT_NAMES:
        for example in splits.get(split_name, []):
            prior_id_split = seen_ids.get(example.id)
            if prior_id_split is not None and prior_id_split != split_name:
                raise ValueError(
                    f"split leakage: id {example.id!r} appears in {prior_id_split} and {split_name}"
                )
            seen_ids[example.id] = split_name

            fingerprint = content_fingerprint(example)
            prior_content = seen_content.get(fingerprint)
            if prior_content is not None and prior_content[0] != split_name:
                raise ValueError(
                    "split leakage: normalized content for "
                    f"{example.id!r} matches {prior_content[1]!r} across "
                    f"{prior_content[0]} and {split_name}"
                )
            seen_content[fingerprint] = (split_name, example.id)


def project_for_trl(example: TicketExample) -> dict[str, object]:
    """Project a source example to TRL conversational prompt/completion form."""

    return {
        "id": example.id,
        "prompt": [message.model_dump(mode="json") for message in example.prompt_messages],
        "completion": [message.model_dump(mode="json") for message in example.completion_messages],
        "expected": example.expected.model_dump(mode="json"),
        "provenance": example.provenance.model_dump(mode="json"),
        "pii_placeholders": list(example.pii_placeholders),
    }


def _canonical_json_line(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _canonical_split_bytes(examples: Sequence[TicketExample]) -> bytes:
    records = (project_for_trl(example) for example in examples)
    return "".join(f"{_canonical_json_line(record)}\n" for record in records).encode("utf-8")


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            temporary_name = handle.name
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None:
            temporary_path = Path(temporary_name)
            if temporary_path.exists():
                temporary_path.unlink()


def _distribution(examples: Sequence[TicketExample]) -> LabelDistribution:
    return LabelDistribution(
        category=dict(sorted(Counter(item.expected.category for item in examples).items())),
        priority=dict(sorted(Counter(item.expected.priority for item in examples).items())),
        sentiment=dict(sorted(Counter(item.expected.sentiment for item in examples).items())),
    )


def prepare_dataset(
    source_path: Path,
    output_dir: Path,
    *,
    seed: int = 42,
    splits: SplitConfig | None = None,
) -> PreparedDataset:
    """Validate, split, atomically write, and hash a TicketTune dataset."""

    resolved_source = source_path.expanduser().resolve()
    resolved_output = output_dir.expanduser().resolve()
    if (
        resolved_source == resolved_output
        or resolved_source.is_relative_to(resolved_output)
        or resolved_output.is_relative_to(resolved_source)
    ):
        raise DatasetValidationError(
            resolved_source,
            None,
            "source path and output directory must not overlap after symlink resolution",
        )
    split_config = splits or SplitConfig()
    pending_manifest = resolved_output / "manifest.json"
    try:
        source_payload, source_identity = _read_stable_regular_bytes(
            resolved_source,
            manifest_path=pending_manifest,
            label="dataset source",
        )
    except DatasetIntegrityError as exc:
        raise DatasetValidationError(resolved_source, None, exc.detail) from exc
    source_digest = sha256_bytes(source_payload)
    examples = _load_examples_from_bytes(source_payload, resolved_source)
    generated_splits = split_examples(examples, seed=seed, splits=split_config)

    split_artifacts: dict[SplitName, SplitArtifact] = {}
    split_ids: dict[SplitName, list[str]] = {}
    for split_name in SPLIT_NAMES:
        split_path = resolved_output / f"{split_name}.jsonl"
        records = generated_splits[split_name]
        content = _canonical_split_bytes(records)
        _atomic_write_text(split_path, content.decode("utf-8"))
        split_artifacts[split_name] = SplitArtifact(
            file=split_path.name,
            count=len(records),
            sha256=sha256_file(split_path),
            labels=_distribution(generated_splits[split_name]),
        )
        split_ids[split_name] = [example.id for example in generated_splits[split_name]]

    try:
        final_source_payload, final_source_identity = _read_stable_regular_bytes(
            resolved_source,
            manifest_path=pending_manifest,
            label="dataset source",
        )
    except DatasetIntegrityError as exc:
        raise DatasetValidationError(resolved_source, None, exc.detail) from exc
    if final_source_identity != source_identity or final_source_payload != source_payload:
        raise DatasetValidationError(
            resolved_source,
            None,
            "source changed during dataset preparation",
        )

    manifest = DatasetManifest(
        source_file=resolved_source.name,
        source_sha256=source_digest,
        seed=seed,
        split_fractions={
            "train": split_config.train,
            "validation": split_config.validation,
            "test": split_config.test,
        },
        total_examples=len(examples),
        labels=_distribution(examples),
        splits=split_artifacts,
        split_ids=split_ids,
    )
    manifest_path = resolved_output / "manifest.json"
    manifest_content = (
        json.dumps(
            manifest.model_dump(mode="json"),
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    _atomic_write_text(manifest_path, manifest_content)
    return PreparedDataset(
        output_dir=resolved_output,
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        source_sha256=source_digest,
        total_examples=len(examples),
        split_ids=split_ids,
        splits=split_artifacts,
    )
