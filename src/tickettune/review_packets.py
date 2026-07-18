"""Strict reviewer-packet models and deterministic draft scaffolding.

The objects in this module are evidence declarations, not approvals.  Draft
builders deliberately use replacement reviewer identifiers and pending record
decisions so generated scaffolding can never qualify a dataset by itself.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
from dataclasses import dataclass
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Sha256Digest = str
ReviewOutcome = Literal["pending", "approved", "rejected"]

DRAFT_REVIEWER_IDS = (
    "REPLACE_WITH_REVIEWER_A",
    "REPLACE_WITH_REVIEWER_B",
)


@dataclass(frozen=True, slots=True)
class ReviewScaffoldArtifact:
    """Paths and hashes for one deliberately non-approving review workspace."""

    directory: Path
    review_manifest_path: Path
    prepared_manifest_path: Path
    holdout_freeze_path: Path
    reviewer_packet_paths: tuple[Path, Path]
    source_sha256: Sha256Digest
    prepared_manifest_sha256: Sha256Digest
    holdout_freeze_sha256: Sha256Digest
    reviewer_packet_sha256: tuple[Sha256Digest, Sha256Digest]
    record_count: int
    held_out_count: int
    status: Literal["pending_two_independent_human_reviews"] = (
        "pending_two_independent_human_reviews"
    )
    release_eligible: Literal[False] = False
    proof_boundary: str = (
        "draft_review_scaffold_only; generated_packets_are_not_human_review_evidence"
    )


class _FrozenModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        validate_default=True,
        allow_inf_nan=False,
    )


class EvidenceFileReference(_FrozenModel):
    """Manifest-relative identity for one immutable evidence file."""

    path: str = Field(min_length=1, max_length=500)
    sha256: Sha256Digest = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        candidate = PurePosixPath(value)
        if (
            candidate.is_absolute()
            or value != candidate.as_posix()
            or candidate == PurePosixPath(".")
            or any(part in {"", ".", ".."} for part in candidate.parts)
        ):
            raise ValueError("evidence paths must be normalized relative paths without traversal")
        return value


class RecordReviewDecision(_FrozenModel):
    """One human reviewer's four independent decisions for one record."""

    record_id: str = Field(min_length=1, max_length=200)
    labels: ReviewOutcome
    response: ReviewOutcome
    pii: ReviewOutcome
    license: ReviewOutcome


class ReviewerPacket(_FrozenModel):
    """Review decisions made by one identified human reviewer."""

    schema_version: Literal["1.0"] = "1.0"
    reviewer_id: str = Field(min_length=3, max_length=200)
    reviewer_kind: Literal["human"] = "human"
    status: Literal["draft", "approved", "rejected"]
    source_sha256: Sha256Digest = Field(pattern=r"^[0-9a-f]{64}$")
    prepared_manifest_sha256: Sha256Digest = Field(pattern=r"^[0-9a-f]{64}$")
    holdout_freeze_sha256: Sha256Digest = Field(pattern=r"^[0-9a-f]{64}$")
    review_date: date | None
    decisions: tuple[RecordReviewDecision, ...] = Field(min_length=1)

    @field_validator("decisions")
    @classmethod
    def validate_unique_decision_ids(
        cls,
        value: tuple[RecordReviewDecision, ...],
    ) -> tuple[RecordReviewDecision, ...]:
        record_ids = tuple(item.record_id for item in value)
        if len(record_ids) != len(set(record_ids)):
            raise ValueError("review packet decisions must contain unique record IDs")
        return value


class HoldoutFreeze(_FrozenModel):
    """Exact prepared test cohort frozen before model tuning and evaluation."""

    schema_version: Literal["1.0"] = "1.0"
    source_sha256: Sha256Digest = Field(pattern=r"^[0-9a-f]{64}$")
    prepared_manifest_sha256: Sha256Digest = Field(pattern=r"^[0-9a-f]{64}$")
    held_out_ids: tuple[str, ...] = Field(min_length=1)

    @field_validator("held_out_ids")
    @classmethod
    def validate_unique_held_out_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not record_id.strip() for record_id in value):
            raise ValueError("held_out_ids must contain non-empty strings")
        if len(value) != len(set(value)):
            raise ValueError("held_out_ids must not contain duplicates")
        return value


class DatasetReviewManifestV12(_FrozenModel):
    """Aggregate review attestation with transitive file identities."""

    schema_version: Literal["1.2"] = "1.2"
    source_sha256: Sha256Digest = Field(pattern=r"^[0-9a-f]{64}$")
    record_count: int = Field(ge=1)
    review_date: date | None
    intended_domain: str = Field(min_length=12, max_length=500)
    consent_or_license_statement: str = Field(min_length=12, max_length=2_000)
    pii_decision: Literal["no_real_customer_data"]
    isolated_test_set_statement: str = Field(min_length=12, max_length=2_000)
    prepared_manifest: EvidenceFileReference
    holdout_freeze: EvidenceFileReference
    reviewer_packets: tuple[EvidenceFileReference, EvidenceFileReference]
    approval_status: Literal["draft", "approved", "rejected"]

    @model_validator(mode="after")
    def validate_distinct_references(self) -> Self:
        packet_paths = tuple(item.path for item in self.reviewer_packets)
        if len(set(packet_paths)) != len(packet_paths):
            raise ValueError("reviewer packet paths must be distinct")
        return self


def canonical_evidence_bytes(model: BaseModel) -> bytes:
    """Serialize an evidence model deterministically for hashing and writing."""

    payload = json.dumps(
        model.model_dump(mode="json"),
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    )
    return (payload + "\n").encode("utf-8")


def evidence_sha256(model: BaseModel) -> Sha256Digest:
    """Hash the canonical bytes a scaffold writer should persist."""

    return hashlib.sha256(canonical_evidence_bytes(model)).hexdigest()


def build_holdout_freeze(
    *,
    source_sha256: Sha256Digest,
    prepared_manifest_sha256: Sha256Digest,
    held_out_ids: tuple[str, ...],
) -> HoldoutFreeze:
    """Build a deterministic freeze declaration for an already prepared test split."""

    return HoldoutFreeze(
        source_sha256=source_sha256,
        prepared_manifest_sha256=prepared_manifest_sha256,
        held_out_ids=held_out_ids,
    )


def build_draft_reviewer_packet(
    *,
    reviewer_id: str,
    source_sha256: Sha256Digest,
    prepared_manifest_sha256: Sha256Digest,
    holdout_freeze_sha256: Sha256Digest,
    ordered_record_ids: tuple[str, ...],
) -> ReviewerPacket:
    """Build one deliberately non-approving human-review template."""

    return ReviewerPacket(
        reviewer_id=reviewer_id,
        reviewer_kind="human",
        status="draft",
        source_sha256=source_sha256,
        prepared_manifest_sha256=prepared_manifest_sha256,
        holdout_freeze_sha256=holdout_freeze_sha256,
        review_date=None,
        decisions=tuple(
            RecordReviewDecision(
                record_id=record_id,
                labels="pending",
                response="pending",
                pii="pending",
                license="pending",
            )
            for record_id in ordered_record_ids
        ),
    )


def build_draft_reviewer_packets(
    *,
    source_sha256: Sha256Digest,
    prepared_manifest_sha256: Sha256Digest,
    holdout_freeze_sha256: Sha256Digest,
    ordered_record_ids: tuple[str, ...],
) -> tuple[ReviewerPacket, ReviewerPacket]:
    """Build the two deterministic replacement-ID packets used by scaffolding."""

    reviewer_a, reviewer_b = DRAFT_REVIEWER_IDS
    return (
        build_draft_reviewer_packet(
            reviewer_id=reviewer_a,
            source_sha256=source_sha256,
            prepared_manifest_sha256=prepared_manifest_sha256,
            holdout_freeze_sha256=holdout_freeze_sha256,
            ordered_record_ids=ordered_record_ids,
        ),
        build_draft_reviewer_packet(
            reviewer_id=reviewer_b,
            source_sha256=source_sha256,
            prepared_manifest_sha256=prepared_manifest_sha256,
            holdout_freeze_sha256=holdout_freeze_sha256,
            ordered_record_ids=ordered_record_ids,
        ),
    )


def build_draft_review_manifest(
    *,
    source_sha256: Sha256Digest,
    record_count: int,
    intended_domain: str,
    consent_or_license_statement: str,
    isolated_test_set_statement: str,
    prepared_manifest: EvidenceFileReference,
    holdout_freeze: EvidenceFileReference,
    reviewer_packets: tuple[EvidenceFileReference, EvidenceFileReference],
) -> DatasetReviewManifestV12:
    """Build a draft aggregate that cannot represent completed review."""

    return DatasetReviewManifestV12(
        source_sha256=source_sha256,
        record_count=record_count,
        review_date=None,
        intended_domain=intended_domain,
        consent_or_license_statement=consent_or_license_statement,
        pii_decision="no_real_customer_data",
        isolated_test_set_statement=isolated_test_set_statement,
        prepared_manifest=prepared_manifest,
        holdout_freeze=holdout_freeze,
        reviewer_packets=reviewer_packets,
        approval_status="draft",
    )


def resolve_evidence_reference(*, aggregate_path: Path, relative_path: str) -> Path:
    """Resolve a validated reference without allowing parent/symlink traversal."""

    base = aggregate_path.parent.resolve()
    candidate = base.joinpath(*PurePosixPath(relative_path).parts).resolve(strict=False)
    if candidate == base or not candidate.is_relative_to(base):
        raise ValueError("evidence reference escapes the aggregate manifest directory")
    return candidate


def bind_review_manifest_references(
    manifest: DatasetReviewManifestV12,
    *,
    aggregate_path: Path,
) -> DatasetReviewManifestV12:
    """Refresh referenced byte hashes without changing any review decision or status."""

    def rebound(reference: EvidenceFileReference) -> EvidenceFileReference:
        path = resolve_evidence_reference(
            aggregate_path=aggregate_path,
            relative_path=reference.path,
        )
        payload = _read_stable_regular_bytes(path)
        return EvidenceFileReference(
            path=reference.path,
            sha256=hashlib.sha256(payload).hexdigest(),
        )

    return manifest.model_copy(
        update={
            "prepared_manifest": rebound(manifest.prepared_manifest),
            "holdout_freeze": rebound(manifest.holdout_freeze),
            "reviewer_packets": tuple(rebound(item) for item in manifest.reviewer_packets),
        }
    )


def _reject_symlink_components(path: Path) -> Path:
    absolute = Path(os.path.abspath(path.expanduser()))
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"review scaffold path must not contain symlinks: {current}")
    return absolute


def _read_stable_regular_bytes(path: Path) -> bytes:
    candidate = _reject_symlink_components(path)
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:  # pragma: no cover - TicketTune targets POSIX hosts.
        raise RuntimeError("review scaffolding requires O_NOFOLLOW support")
    descriptor = os.open(candidate, os.O_RDONLY | no_follow | getattr(os, "O_CLOEXEC", 0))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"prepared manifest must be a regular file: {candidate}")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
        current = os.lstat(candidate)
    finally:
        os.close(descriptor)
    payload = b"".join(chunks)
    if (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or (current.st_dev, current.st_ino) != (before.st_dev, before.st_ino)
        or len(payload) != before.st_size
    ):
        raise ValueError(f"prepared manifest changed while it was read: {candidate}")
    return payload


def _write_new_private_file(path: Path, payload: bytes) -> None:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:  # pragma: no cover - TicketTune targets POSIX hosts.
        raise RuntimeError("review scaffolding requires O_NOFOLLOW support")
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | no_follow | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:  # pragma: no cover - defensive kernel boundary.
                raise OSError("review scaffold write made no progress")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_draft_review_scaffold(
    output_dir: Path,
    *,
    source_sha256: Sha256Digest,
    prepared_manifest_path: Path,
    prepared_manifest_sha256: Sha256Digest,
    ordered_record_ids: tuple[str, ...],
    held_out_ids: tuple[str, ...],
    intended_domain: str = "synthetic customer-support triage benchmark",
    consent_or_license_statement: str = (
        "CC0-1.0 synthetic records; no real customer data or consent dependency."
    ),
    isolated_test_set_statement: str = (
        "The held-out examples were frozen before model training, tuning, or evaluation."
    ),
) -> ReviewScaffoldArtifact:
    """Create two pending reviewer packets and their transitive v1.2 manifest."""

    if not ordered_record_ids or len(set(ordered_record_ids)) != len(ordered_record_ids):
        raise ValueError("ordered_record_ids must be non-empty and unique")
    prepared_payload = _read_stable_regular_bytes(prepared_manifest_path)
    observed_prepared_sha256 = hashlib.sha256(prepared_payload).hexdigest()
    if observed_prepared_sha256 != prepared_manifest_sha256:
        raise ValueError("prepared manifest bytes do not match the verified SHA-256")

    freeze = build_holdout_freeze(
        source_sha256=source_sha256,
        prepared_manifest_sha256=prepared_manifest_sha256,
        held_out_ids=held_out_ids,
    )
    freeze_payload = canonical_evidence_bytes(freeze)
    freeze_sha256 = hashlib.sha256(freeze_payload).hexdigest()
    packets = build_draft_reviewer_packets(
        source_sha256=source_sha256,
        prepared_manifest_sha256=prepared_manifest_sha256,
        holdout_freeze_sha256=freeze_sha256,
        ordered_record_ids=ordered_record_ids,
    )
    packet_payloads = tuple(canonical_evidence_bytes(packet) for packet in packets)
    packet_a_sha256, packet_b_sha256 = (
        hashlib.sha256(payload).hexdigest() for payload in packet_payloads
    )
    packet_sha256 = (packet_a_sha256, packet_b_sha256)
    aggregate = build_draft_review_manifest(
        source_sha256=source_sha256,
        record_count=len(ordered_record_ids),
        intended_domain=intended_domain,
        consent_or_license_statement=consent_or_license_statement,
        isolated_test_set_statement=isolated_test_set_statement,
        prepared_manifest=EvidenceFileReference(
            path="prepared-manifest.json",
            sha256=prepared_manifest_sha256,
        ),
        holdout_freeze=EvidenceFileReference(
            path="holdout-freeze.json",
            sha256=freeze_sha256,
        ),
        reviewer_packets=(
            EvidenceFileReference(path="reviewer-a.json", sha256=packet_sha256[0]),
            EvidenceFileReference(path="reviewer-b.json", sha256=packet_sha256[1]),
        ),
    )

    destination = _reject_symlink_components(output_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination = _reject_symlink_components(destination)
    try:
        destination.mkdir(mode=0o700, exist_ok=False)
    except FileExistsError:
        raise FileExistsError(f"review scaffold directory must be new: {destination}") from None

    prepared_output = destination / "prepared-manifest.json"
    freeze_output = destination / "holdout-freeze.json"
    packet_outputs = (destination / "reviewer-a.json", destination / "reviewer-b.json")
    aggregate_output = destination / "review-manifest.json"
    try:
        _write_new_private_file(prepared_output, prepared_payload)
        _write_new_private_file(freeze_output, freeze_payload)
        for packet_output, packet_payload in zip(packet_outputs, packet_payloads, strict=True):
            _write_new_private_file(packet_output, packet_payload)
        _write_new_private_file(aggregate_output, canonical_evidence_bytes(aggregate))
        directory_descriptor = os.open(
            destination,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        shutil.rmtree(destination)
        raise

    return ReviewScaffoldArtifact(
        directory=destination,
        review_manifest_path=aggregate_output,
        prepared_manifest_path=prepared_output,
        holdout_freeze_path=freeze_output,
        reviewer_packet_paths=packet_outputs,
        source_sha256=source_sha256,
        prepared_manifest_sha256=prepared_manifest_sha256,
        holdout_freeze_sha256=freeze_sha256,
        reviewer_packet_sha256=packet_sha256,
        record_count=len(ordered_record_ids),
        held_out_count=len(held_out_ids),
    )
