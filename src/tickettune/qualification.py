"""Fail-closed review evidence for a quality-candidate dataset.

This module validates evidence about a dataset; it does not claim that a model
trained on the dataset is accurate, representative of production traffic, or
safe for autonomous support actions.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from .data import SPLIT_NAMES, DatasetManifest, DatasetValidationError, load_examples
from .review_packets import (
    DRAFT_REVIEWER_IDS,
    DatasetReviewManifestV12,
    EvidenceFileReference,
    HoldoutFreeze,
    ReviewerPacket,
    resolve_evidence_reference,
)
from .schemas import TicketExample
from .strict_json import StrictJSONError, loads_strict

MIN_QUALIFIED_RECORDS = 1_000
MIN_INDEPENDENT_REVIEWERS = 2
MIN_HELD_OUT_RECORDS = 100

Sha256Digest = str


class _FrozenModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        validate_default=True,
        allow_inf_nan=False,
    )


class DatasetReviewManifest(_FrozenModel):
    """Human-review attestation bound to exact validated source bytes."""

    schema_version: Literal["1.1"] = "1.1"
    source_sha256: Sha256Digest = Field(pattern=r"^[0-9a-f]{64}$")
    record_count: int = Field(ge=1)
    independent_reviewer_count: int = Field(ge=0)
    reviewed_count: int = Field(ge=0)
    held_out_count: int = Field(ge=0)
    held_out_ids: tuple[str, ...] = Field(min_length=1)
    review_date: date
    intended_domain: str = Field(min_length=12, max_length=500)
    consent_or_license_statement: str = Field(min_length=12, max_length=2_000)
    pii_decision: Literal["no_real_customer_data"]
    isolated_test_set_statement: str = Field(min_length=12, max_length=2_000)
    approval_status: Literal["draft", "approved", "rejected"]

    @field_validator("held_out_ids")
    @classmethod
    def validate_held_out_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not record_id.strip() for record_id in value):
            raise ValueError("held_out_ids must contain non-empty strings")
        if len(set(value)) != len(value):
            raise ValueError("held_out_ids must not contain duplicates")
        return value

    @model_validator(mode="after")
    def validate_internal_counts(self) -> DatasetReviewManifest:
        if self.reviewed_count > self.record_count:
            raise ValueError("reviewed_count cannot exceed record_count")
        if self.held_out_count > self.record_count:
            raise ValueError("held_out_count cannot exceed record_count")
        return self


class QualificationDecision(_FrozenModel):
    """One explicit policy result retained in the qualification report."""

    policy: str
    passed: bool
    observed: str
    required: str
    detail: str


class DatasetQualificationReport(_FrozenModel):
    """Deterministic evidence report for one source/review-manifest pair."""

    schema_version: Literal["1.1", "1.2"] = "1.1"
    qualified: bool
    dataset_tier: Literal["portfolio_smoke", "qualification_candidate"]
    source_path: Path
    source_sha256: Sha256Digest = Field(pattern=r"^[0-9a-f]{64}$")
    review_manifest_path: Path
    review_manifest_sha256: Sha256Digest = Field(pattern=r"^[0-9a-f]{64}$")
    record_count: int = Field(ge=1)
    reviewed_count: int = Field(ge=0)
    independent_reviewer_count: int = Field(ge=0)
    held_out_count: int = Field(ge=0)
    held_out_ids: tuple[str, ...] = Field(min_length=1)
    review_date: date | None
    intended_domain: str
    decisions: tuple[QualificationDecision, ...]
    prepared_manifest_path: Path | None = None
    prepared_manifest_sha256: Sha256Digest | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    holdout_freeze_path: Path | None = None
    holdout_freeze_sha256: Sha256Digest | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    reviewer_packet_paths: tuple[Path, ...] = ()
    reviewer_packet_sha256: tuple[Sha256Digest, ...] = ()
    reviewer_ids: tuple[str, ...] = ()
    proof_boundary: str = (
        "review_attestation_and_source_integrity_only; "
        "does_not_prove_model_quality_or_production_representativeness"
    )


class DatasetQualificationError(ValueError):
    """Malformed evidence or an explicitly enforced policy failure."""

    def __init__(
        self,
        message: str,
        *,
        report: DatasetQualificationReport | None = None,
    ) -> None:
        self.report = report
        super().__init__(message)


FileIdentity = tuple[int, int, int, int, int]


def _stat_identity(metadata: os.stat_result) -> FileIdentity:
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
    label: str,
) -> tuple[bytes, Path, str, FileIdentity]:
    """Read and hash one exact non-symlink inode, then bind it to its path."""

    candidate = Path(os.path.abspath(path.expanduser()))
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:  # pragma: no cover - TicketTune targets POSIX hosts.
        raise DatasetQualificationError(f"{label} reads require O_NOFOLLOW support")
    flags = os.O_RDONLY | no_follow | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(candidate, flags)
    except OSError as exc:
        raise DatasetQualificationError(
            f"{label} must be a regular non-symlink file: {candidate}"
        ) from exc

    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise DatasetQualificationError(
                f"{label} must be a regular non-symlink file: {candidate}"
            )
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
        current = os.lstat(candidate)
    except DatasetQualificationError:
        raise
    except OSError as exc:
        raise DatasetQualificationError(
            f"{label} changed while it was being read: {candidate}"
        ) from exc
    finally:
        os.close(descriptor)

    payload = b"".join(chunks)
    before_identity = _stat_identity(before)
    if (
        before_identity != _stat_identity(after)
        or not stat.S_ISREG(current.st_mode)
        or (current.st_dev, current.st_ino) != (before.st_dev, before.st_ino)
        or len(payload) != before.st_size
    ):
        raise DatasetQualificationError(f"{label} changed while it was being read: {candidate}")
    return payload, candidate, hashlib.sha256(payload).hexdigest(), before_identity


def _write_read_only_snapshot(path: Path, payload: bytes) -> None:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:  # pragma: no cover - TicketTune targets POSIX hosts.
        raise DatasetQualificationError("dataset snapshot writes require O_NOFOLLOW support")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | no_follow | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags, 0o400)
    except OSError as exc:
        raise DatasetQualificationError("could not create private dataset snapshot") from exc
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:  # pragma: no cover - defensive kernel boundary.
                raise OSError("snapshot write made no progress")
            offset += written
        os.fsync(descriptor)
    except OSError as exc:
        raise DatasetQualificationError("could not write private dataset snapshot") from exc
    finally:
        os.close(descriptor)


def _load_examples_from_stable_bytes(
    payload: bytes,
    *,
    source_path: Path,
) -> list[TicketExample]:
    """Validate exact source bytes through a private, read-only temporary snapshot."""

    with tempfile.TemporaryDirectory(prefix="tickettune-qualification-") as directory:
        snapshot = Path(directory) / "source.jsonl"
        _write_read_only_snapshot(snapshot, payload)
        snapshot_before, _path, digest_before, identity_before = _read_stable_regular_bytes(
            snapshot,
            label="dataset source snapshot",
        )
        if snapshot_before != payload:
            raise DatasetQualificationError("private dataset snapshot bytes did not match source")
        try:
            examples = load_examples(snapshot)
        except DatasetValidationError as exc:
            location = (
                f"{source_path}:{exc.line_number}"
                if exc.line_number is not None
                else str(source_path)
            )
            raise DatasetQualificationError(
                f"dataset source validation failed: {location}: {exc.detail}"
            ) from exc
        except OSError as exc:
            raise DatasetQualificationError(f"dataset source validation failed: {exc}") from exc
        finally:
            snapshot_after, _path, digest_after, identity_after = _read_stable_regular_bytes(
                snapshot,
                label="dataset source snapshot",
            )
        if (
            snapshot_after != payload
            or digest_after != digest_before
            or identity_after != identity_before
        ):
            raise DatasetQualificationError(
                "private dataset snapshot changed while it was being validated"
            )
        return examples


ReviewManifest = DatasetReviewManifest | DatasetReviewManifestV12


def _read_review_manifest(path: Path) -> tuple[ReviewManifest, Path, str]:
    payload, manifest_path, manifest_digest, _identity = _read_stable_regular_bytes(
        path,
        label="review manifest",
    )
    try:
        decoded = loads_strict(payload)
        if not isinstance(decoded, dict):
            raise ValueError("review manifest must be a JSON object")
        schema_version = decoded.get("schema_version")
        if schema_version == "1.1":
            manifest: ReviewManifest = DatasetReviewManifest.model_validate_json(
                payload,
                strict=True,
            )
        elif schema_version == "1.2":
            manifest = DatasetReviewManifestV12.model_validate_json(payload, strict=True)
        else:
            raise ValueError("review manifest schema_version must be '1.1' or '1.2'")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise DatasetQualificationError(f"invalid review manifest {manifest_path}: {exc}") from exc
    return manifest, manifest_path, manifest_digest


def load_review_manifest(path: Path) -> ReviewManifest:
    """Load a strict, duplicate-key-free, non-symlink review manifest."""

    manifest, _path, _digest = _read_review_manifest(path)
    return manifest


def _decision(
    policy: str,
    *,
    passed: bool,
    observed: object,
    required: object,
    detail: str,
) -> QualificationDecision:
    return QualificationDecision(
        policy=policy,
        passed=passed,
        observed=str(observed),
        required=str(required),
        detail=detail,
    )


def _read_referenced_evidence(
    *,
    aggregate_path: Path,
    reference: EvidenceFileReference,
    label: str,
) -> tuple[bytes, Path, str]:
    try:
        evidence_path = resolve_evidence_reference(
            aggregate_path=aggregate_path,
            relative_path=reference.path,
        )
    except ValueError as exc:
        raise DatasetQualificationError(f"invalid {label} reference: {exc}") from exc
    payload, stable_path, digest, _identity = _read_stable_regular_bytes(
        evidence_path,
        label=label,
    )
    return payload, stable_path, digest


def _parse_prepared_manifest(payload: bytes, *, path: Path) -> DatasetManifest:
    try:
        loads_strict(payload)
        return DatasetManifest.model_validate_json(payload, strict=True)
    except (UnicodeDecodeError, json.JSONDecodeError, StrictJSONError, ValidationError) as exc:
        raise DatasetQualificationError(f"invalid prepared manifest {path}: {exc}") from exc


def _parse_holdout_freeze(payload: bytes, *, path: Path) -> HoldoutFreeze:
    try:
        loads_strict(payload)
        return HoldoutFreeze.model_validate_json(payload, strict=True)
    except (UnicodeDecodeError, json.JSONDecodeError, StrictJSONError, ValidationError) as exc:
        raise DatasetQualificationError(f"invalid holdout freeze {path}: {exc}") from exc


def _parse_reviewer_packet(payload: bytes, *, path: Path) -> ReviewerPacket:
    try:
        loads_strict(payload)
        return ReviewerPacket.model_validate_json(payload, strict=True)
    except (UnicodeDecodeError, json.JSONDecodeError, StrictJSONError, ValidationError) as exc:
        raise DatasetQualificationError(f"invalid reviewer packet {path}: {exc}") from exc


def _reviewer_id_is_non_placeholder(reviewer_id: str) -> bool:
    normalized = reviewer_id.strip().lower()
    placeholder_markers = ("replace", "placeholder", "example", "todo", "tbd")
    return (
        reviewer_id not in DRAFT_REVIEWER_IDS
        and not any(marker in normalized for marker in placeholder_markers)
        and normalized not in {"reviewer-a", "reviewer_a", "reviewer-b", "reviewer_b"}
        and not (normalized.startswith("<") and normalized.endswith(">"))
    )


def _record_decision_is_approved(packet_decision: object) -> bool:
    return all(
        getattr(packet_decision, field) == "approved"
        for field in ("labels", "response", "pii", "license")
    )


def _legacy_packet_decisions() -> tuple[QualificationDecision, ...]:
    detail = "Legacy v1.1 review counts are self-asserted and are not release evidence."
    return tuple(
        _decision(
            policy,
            passed=False,
            observed="not_present_in_v1.1",
            required="transitively_hashed_v1.2_evidence",
            detail=detail,
        )
        for policy in (
            "review_evidence_schema_v1_2",
            "prepared_manifest_sha256_matches",
            "prepared_manifest_source_matches",
            "prepared_manifest_record_ids_match",
            "holdout_freeze_sha256_matches",
            "holdout_freeze_matches_prepared_test",
            "minimum_independent_review_packets",
            "review_packet_files_distinct",
            "review_packet_hashes_match",
            "reviewer_ids_distinct",
            "reviewer_ids_non_placeholder",
            "review_packets_bind_source",
            "review_packets_bind_prepared_manifest",
            "review_packets_bind_holdout_freeze",
            "review_packets_cover_ordered_source",
            "review_packets_approved",
            "review_packet_decisions_approved",
        )
    )


def _qualify_legacy_manifest(
    *,
    source: Path,
    source_digest: str,
    examples: list[TicketExample],
    manifest: DatasetReviewManifest,
    manifest_path: Path,
    manifest_digest: str,
    enforce: bool,
) -> DatasetQualificationReport:
    record_count = len(examples)
    source_ids = {example.id for example in examples}
    held_out_ids = manifest.held_out_ids
    decisions = (
        _decision(
            "source_sha256_matches",
            passed=manifest.source_sha256 == source_digest,
            observed=manifest.source_sha256,
            required=source_digest,
            detail="The review must attest to the exact validated source bytes.",
        ),
        _decision(
            "record_count_matches",
            passed=manifest.record_count == record_count,
            observed=manifest.record_count,
            required=record_count,
            detail="The reviewed count declaration must match parsed source records.",
        ),
        _decision(
            "minimum_record_count",
            passed=record_count >= MIN_QUALIFIED_RECORDS,
            observed=record_count,
            required=f">={MIN_QUALIFIED_RECORDS}",
            detail="Quality-candidate evidence requires more than smoke data.",
        ),
        _decision(
            "minimum_independent_reviewers",
            passed=False,
            observed=f"self_asserted_count={manifest.independent_reviewer_count}",
            required=f">={MIN_INDEPENDENT_REVIEWERS} verified reviewer packets",
            detail="A legacy integer cannot prove independent human review.",
        ),
        _decision(
            "full_record_review",
            passed=manifest.reviewed_count == record_count,
            observed=manifest.reviewed_count,
            required=record_count,
            detail="This legacy declaration is retained for diagnostics only.",
        ),
        _decision(
            "minimum_held_out_examples",
            passed=len(held_out_ids) >= MIN_HELD_OUT_RECORDS,
            observed=len(held_out_ids),
            required=f">={MIN_HELD_OUT_RECORDS}",
            detail="The isolated held-out cohort must support more than smoke-level scoring.",
        ),
        _decision(
            "held_out_within_dataset",
            passed=len(held_out_ids) <= record_count,
            observed=len(held_out_ids),
            required=f"<={record_count}",
            detail="The held-out declaration cannot exceed the source population.",
        ),
        _decision(
            "held_out_count_matches_ids",
            passed=manifest.held_out_count == len(held_out_ids),
            observed=manifest.held_out_count,
            required=len(held_out_ids),
            detail="The declared held-out count must equal the explicit ID cohort.",
        ),
        _decision(
            "held_out_ids_within_source",
            passed=set(held_out_ids).issubset(source_ids),
            observed=sum(record_id in source_ids for record_id in held_out_ids),
            required=len(held_out_ids),
            detail="Every held-out ID must exist in the exact source bytes.",
        ),
        _decision(
            "approved_status",
            passed=manifest.approval_status == "approved",
            observed=manifest.approval_status,
            required="approved",
            detail="Draft or rejected review evidence cannot qualify a dataset.",
        ),
        *_legacy_packet_decisions(),
    )
    report = DatasetQualificationReport(
        schema_version="1.1",
        qualified=False,
        dataset_tier=(
            "qualification_candidate"
            if record_count >= MIN_QUALIFIED_RECORDS
            else "portfolio_smoke"
        ),
        source_path=source,
        source_sha256=source_digest,
        review_manifest_path=manifest_path,
        review_manifest_sha256=manifest_digest,
        record_count=record_count,
        reviewed_count=manifest.reviewed_count,
        independent_reviewer_count=manifest.independent_reviewer_count,
        held_out_count=len(held_out_ids),
        held_out_ids=held_out_ids,
        review_date=manifest.review_date,
        intended_domain=manifest.intended_domain,
        decisions=decisions,
    )
    if enforce:
        failed = ", ".join(item.policy for item in decisions if not item.passed)
        raise DatasetQualificationError(
            f"dataset qualification failed: {failed}",
            report=report,
        )
    return report


def _qualify_v12_manifest(
    *,
    source: Path,
    source_digest: str,
    examples: list[TicketExample],
    manifest: DatasetReviewManifestV12,
    manifest_path: Path,
    manifest_digest: str,
    enforce: bool,
) -> DatasetQualificationReport:
    prepared_payload, prepared_path, prepared_digest = _read_referenced_evidence(
        aggregate_path=manifest_path,
        reference=manifest.prepared_manifest,
        label="prepared manifest",
    )
    prepared = _parse_prepared_manifest(prepared_payload, path=prepared_path)
    freeze_payload, freeze_path, freeze_digest = _read_referenced_evidence(
        aggregate_path=manifest_path,
        reference=manifest.holdout_freeze,
        label="holdout freeze",
    )
    freeze = _parse_holdout_freeze(freeze_payload, path=freeze_path)

    packet_paths: list[Path] = []
    packet_digests: list[str] = []
    packets: list[ReviewerPacket] = []
    for reference in manifest.reviewer_packets:
        payload, packet_path, packet_digest = _read_referenced_evidence(
            aggregate_path=manifest_path,
            reference=reference,
            label="reviewer packet",
        )
        packet_paths.append(packet_path)
        packet_digests.append(packet_digest)
        packets.append(_parse_reviewer_packet(payload, path=packet_path))

    record_count = len(examples)
    ordered_source_ids = tuple(example.id for example in examples)
    source_id_set = set(ordered_source_ids)
    prepared_ids = tuple(
        record_id for split_name in SPLIT_NAMES for record_id in prepared.split_ids[split_name]
    )
    prepared_ids_match = (
        len(prepared_ids) == record_count
        and len(set(prepared_ids)) == record_count
        and set(prepared_ids) == source_id_set
        and prepared.total_examples == record_count
    )
    held_out_ids = tuple(freeze.held_out_ids)
    expected_held_out_ids = tuple(prepared.split_ids["test"])
    packet_ids = tuple(
        tuple(decision.record_id for decision in packet.decisions) for packet in packets
    )
    reviewer_ids = tuple(packet.reviewer_id for packet in packets)
    approved_sets = [
        {
            decision.record_id
            for decision in packet.decisions
            if _record_decision_is_approved(decision)
        }
        for packet in packets
    ]
    approved_by_all = set.intersection(*approved_sets) if approved_sets else set()
    reviewer_ids_valid = tuple(_reviewer_id_is_non_placeholder(value) for value in reviewer_ids)
    packet_files_distinct = len(set(packet_paths)) == len(packet_paths) and len(
        set(packet_digests)
    ) == len(packet_digests)
    decisions = (
        _decision(
            "review_evidence_schema_v1_2",
            passed=True,
            observed=manifest.schema_version,
            required="1.2",
            detail="The aggregate uses transitive reviewer-packet evidence.",
        ),
        _decision(
            "source_sha256_matches",
            passed=manifest.source_sha256 == source_digest,
            observed=manifest.source_sha256,
            required=source_digest,
            detail="The review must attest to the exact validated source bytes.",
        ),
        _decision(
            "record_count_matches",
            passed=manifest.record_count == record_count,
            observed=manifest.record_count,
            required=record_count,
            detail="The aggregate count must match parsed source records.",
        ),
        _decision(
            "minimum_record_count",
            passed=record_count >= MIN_QUALIFIED_RECORDS,
            observed=record_count,
            required=f">={MIN_QUALIFIED_RECORDS}",
            detail="Quality-candidate evidence requires more than smoke data.",
        ),
        _decision(
            "prepared_manifest_sha256_matches",
            passed=manifest.prepared_manifest.sha256 == prepared_digest,
            observed=manifest.prepared_manifest.sha256,
            required=prepared_digest,
            detail="The aggregate must hash the exact prepared manifest bytes.",
        ),
        _decision(
            "prepared_manifest_source_matches",
            passed=prepared.source_sha256 == source_digest,
            observed=prepared.source_sha256,
            required=source_digest,
            detail="Prepared data must derive from the exact reviewed source.",
        ),
        _decision(
            "prepared_manifest_record_ids_match",
            passed=prepared_ids_match,
            observed=len(set(prepared_ids)),
            required=record_count,
            detail="Prepared split IDs must partition every source ID exactly once.",
        ),
        _decision(
            "holdout_freeze_sha256_matches",
            passed=manifest.holdout_freeze.sha256 == freeze_digest,
            observed=manifest.holdout_freeze.sha256,
            required=freeze_digest,
            detail="The aggregate must hash the exact holdout-freeze bytes.",
        ),
        _decision(
            "holdout_freeze_matches_prepared_test",
            passed=(
                freeze.source_sha256 == source_digest
                and freeze.prepared_manifest_sha256 == prepared_digest
                and held_out_ids == expected_held_out_ids
            ),
            observed=len(held_out_ids),
            required=f"exact ordered prepared test IDs ({len(expected_held_out_ids)})",
            detail="The frozen cohort must equal the prepared test split in order.",
        ),
        _decision(
            "minimum_independent_review_packets",
            passed=len(packets) >= MIN_INDEPENDENT_REVIEWERS,
            observed=len(packets),
            required=f">={MIN_INDEPENDENT_REVIEWERS}",
            detail="Qualification requires two independently hashed reviewer packets.",
        ),
        _decision(
            "review_packet_files_distinct",
            passed=packet_files_distinct,
            observed=f"paths={len(set(packet_paths))},digests={len(set(packet_digests))}",
            required="two distinct paths and two distinct file digests",
            detail="One packet file cannot stand in for independent review.",
        ),
        _decision(
            "review_packet_hashes_match",
            passed=all(
                reference.sha256 == digest
                for reference, digest in zip(
                    manifest.reviewer_packets,
                    packet_digests,
                    strict=True,
                )
            ),
            observed=sum(
                reference.sha256 == digest
                for reference, digest in zip(
                    manifest.reviewer_packets,
                    packet_digests,
                    strict=True,
                )
            ),
            required=len(packet_digests),
            detail="Every packet reference must match the exact packet bytes.",
        ),
        _decision(
            "reviewer_ids_distinct",
            passed=len(set(reviewer_ids)) == len(reviewer_ids),
            observed=len(set(reviewer_ids)),
            required=len(reviewer_ids),
            detail="Each packet must identify a different reviewer.",
        ),
        _decision(
            "reviewer_ids_non_placeholder",
            passed=all(reviewer_ids_valid),
            observed=sum(reviewer_ids_valid),
            required=len(reviewer_ids),
            detail="Replacement IDs and scaffold placeholders cannot approve review.",
        ),
        _decision(
            "review_packets_bind_source",
            passed=all(packet.source_sha256 == source_digest for packet in packets),
            observed=sum(packet.source_sha256 == source_digest for packet in packets),
            required=len(packets),
            detail="Every packet must bind the exact source bytes.",
        ),
        _decision(
            "review_packets_bind_prepared_manifest",
            passed=all(packet.prepared_manifest_sha256 == prepared_digest for packet in packets),
            observed=sum(packet.prepared_manifest_sha256 == prepared_digest for packet in packets),
            required=len(packets),
            detail="Every packet must bind the exact prepared manifest bytes.",
        ),
        _decision(
            "review_packets_bind_holdout_freeze",
            passed=all(packet.holdout_freeze_sha256 == freeze_digest for packet in packets),
            observed=sum(packet.holdout_freeze_sha256 == freeze_digest for packet in packets),
            required=len(packets),
            detail="Every packet must bind the exact frozen holdout bytes.",
        ),
        _decision(
            "review_packets_cover_ordered_source",
            passed=all(ids == ordered_source_ids for ids in packet_ids),
            observed=sum(ids == ordered_source_ids for ids in packet_ids),
            required=len(packet_ids),
            detail="Each reviewer must decide every source record in source order.",
        ),
        _decision(
            "review_packets_approved",
            passed=all(
                packet.status == "approved" and packet.review_date is not None for packet in packets
            ),
            observed=sum(
                packet.status == "approved" and packet.review_date is not None for packet in packets
            ),
            required=len(packets),
            detail="Draft, rejected, or undated packets cannot qualify.",
        ),
        _decision(
            "review_packet_decisions_approved",
            passed=all(
                _record_decision_is_approved(decision)
                for packet in packets
                for decision in packet.decisions
            ),
            observed=len(approved_by_all),
            required=record_count,
            detail="Every reviewer must approve labels, response, PII, and license per record.",
        ),
        _decision(
            "full_record_review",
            passed=len(approved_by_all) == record_count,
            observed=len(approved_by_all),
            required=record_count,
            detail="All records must be approved by both reviewers.",
        ),
        _decision(
            "minimum_held_out_examples",
            passed=len(held_out_ids) >= MIN_HELD_OUT_RECORDS,
            observed=len(held_out_ids),
            required=f">={MIN_HELD_OUT_RECORDS}",
            detail="The frozen cohort must support more than smoke-level scoring.",
        ),
        _decision(
            "held_out_within_dataset",
            passed=len(held_out_ids) <= record_count,
            observed=len(held_out_ids),
            required=f"<={record_count}",
            detail="The frozen cohort cannot exceed the source population.",
        ),
        _decision(
            "held_out_count_matches_ids",
            passed=len(held_out_ids) == len(expected_held_out_ids),
            observed=len(held_out_ids),
            required=len(expected_held_out_ids),
            detail="The freeze count must equal the prepared test count.",
        ),
        _decision(
            "held_out_ids_within_source",
            passed=set(held_out_ids).issubset(source_id_set),
            observed=sum(record_id in source_id_set for record_id in held_out_ids),
            required=len(held_out_ids),
            detail="Every frozen ID must exist in the exact source bytes.",
        ),
        _decision(
            "approved_status",
            passed=manifest.approval_status == "approved" and manifest.review_date is not None,
            observed=(
                manifest.approval_status
                if manifest.review_date is not None
                else f"{manifest.approval_status}; review_date=null"
            ),
            required="approved with review_date",
            detail="A draft, rejected, or undated aggregate cannot qualify.",
        ),
    )
    qualified = all(item.passed for item in decisions)
    report = DatasetQualificationReport(
        schema_version="1.2",
        qualified=qualified,
        dataset_tier=(
            "qualification_candidate"
            if record_count >= MIN_QUALIFIED_RECORDS
            else "portfolio_smoke"
        ),
        source_path=source,
        source_sha256=source_digest,
        review_manifest_path=manifest_path,
        review_manifest_sha256=manifest_digest,
        record_count=record_count,
        reviewed_count=len(approved_by_all),
        independent_reviewer_count=len(set(reviewer_ids)),
        held_out_count=len(held_out_ids),
        held_out_ids=held_out_ids,
        review_date=manifest.review_date,
        intended_domain=manifest.intended_domain,
        decisions=decisions,
        prepared_manifest_path=prepared_path,
        prepared_manifest_sha256=prepared_digest,
        holdout_freeze_path=freeze_path,
        holdout_freeze_sha256=freeze_digest,
        reviewer_packet_paths=tuple(packet_paths),
        reviewer_packet_sha256=tuple(packet_digests),
        reviewer_ids=reviewer_ids,
    )
    if enforce and not qualified:
        failed = ", ".join(item.policy for item in decisions if not item.passed)
        raise DatasetQualificationError(
            f"dataset qualification failed: {failed}",
            report=report,
        )
    return report


def qualify_dataset(
    source_path: Path,
    review_manifest_path: Path,
    *,
    enforce: bool = False,
) -> DatasetQualificationReport:
    """Validate source bytes, review attestation, and minimum quality policy.

    Policy failures are returned as explicit decisions unless ``enforce`` is
    true. Malformed, changing, or symlinked inputs always raise.
    """

    source_payload, source, source_digest, _source_identity = _read_stable_regular_bytes(
        source_path,
        label="dataset source",
    )
    examples = _load_examples_from_stable_bytes(source_payload, source_path=source)

    manifest, manifest_path, manifest_digest = _read_review_manifest(review_manifest_path)
    if isinstance(manifest, DatasetReviewManifest):
        return _qualify_legacy_manifest(
            source=source,
            source_digest=source_digest,
            examples=examples,
            manifest=manifest,
            manifest_path=manifest_path,
            manifest_digest=manifest_digest,
            enforce=enforce,
        )
    return _qualify_v12_manifest(
        source=source,
        source_digest=source_digest,
        examples=examples,
        manifest=manifest,
        manifest_path=manifest_path,
        manifest_digest=manifest_digest,
        enforce=enforce,
    )
