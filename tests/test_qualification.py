from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import BaseModel

import tickettune.qualification as qualification_module
from tickettune.config import SplitConfig, load_config
from tickettune.data import PreparedDataset, prepare_dataset
from tickettune.prompts import SYSTEM_PROMPT
from tickettune.qualification import (
    DatasetQualificationError,
    load_review_manifest,
    qualify_dataset,
)
from tickettune.review_packets import (
    DatasetReviewManifestV12,
    EvidenceFileReference,
    RecordReviewDecision,
    ReviewerPacket,
    build_draft_review_manifest,
    build_draft_reviewer_packets,
    build_holdout_freeze,
    canonical_evidence_bytes,
    evidence_sha256,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CURRENT_SYNTHETIC_SOURCE = PROJECT_ROOT / "data/raw/support_tickets.jsonl"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_source(path: Path, count: int) -> Path:
    categories = (
        "account_access",
        "billing",
        "bug",
        "cancellation",
        "feature_request",
        "shipping",
        "security",
    )
    priorities = ("low", "medium", "high", "urgent")
    sentiments = ("positive", "neutral", "frustrated", "angry", "worried")
    with path.open("w", encoding="utf-8") as handle:
        for index in range(count):
            expected = {
                "category": categories[index % len(categories)],
                "priority": priorities[index % len(priorities)],
                "sentiment": sentiments[index % len(sentiments)],
                "response": f"I will review synthetic case {index:04d} without claiming a fix.",
                "next_action": "review_synthetic_case",
            }
            record = {
                "id": f"QUAL-{index:05d}",
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Synthetic support scenario {index:05d} for [ACCOUNT_ID]; "
                            "please classify it."
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps(expected, sort_keys=True, separators=(",", ":")),
                    },
                ],
                "expected": expected,
                "provenance": {
                    "source": "synthetic",
                    "created_by": "TicketTune qualification test generator",
                    "license": "CC0-1.0",
                    "contains_real_customer_data": False,
                },
                "pii_placeholders": ["[ACCOUNT_ID]"],
            }
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    return path


@pytest.fixture(scope="module")
def qualified_source(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    source = _write_source(tmp_path_factory.mktemp("qualified-source") / "tickets.jsonl", 1000)
    yield source


@pytest.fixture(scope="module")
def qualified_prepared(
    tmp_path_factory: pytest.TempPathFactory,
    qualified_source: Path,
) -> PreparedDataset:
    return prepare_dataset(
        qualified_source,
        tmp_path_factory.mktemp("qualified-prepared"),
        seed=42,
        splits=SplitConfig(),
    )


def _manifest_payload(
    source: Path,
    *,
    record_count: int = 1000,
    reviewers: int = 2,
    reviewed_count: int = 1000,
    held_out_count: int = 100,
    held_out_ids: list[str] | None = None,
    approval_status: str = "approved",
) -> dict[str, object]:
    if held_out_ids is None:
        source_ids = [
            json.loads(line)["id"]
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        held_out_ids = source_ids[:held_out_count]
    return {
        "schema_version": "1.1",
        "source_sha256": _sha256(source),
        "record_count": record_count,
        "independent_reviewer_count": reviewers,
        "reviewed_count": reviewed_count,
        "held_out_count": held_out_count,
        "held_out_ids": held_out_ids,
        "review_date": "2026-07-18",
        "intended_domain": "synthetic customer-support triage benchmark",
        "consent_or_license_statement": (
            "CC0-1.0 synthetic records; no real customer data or consent dependency."
        ),
        "pii_decision": "no_real_customer_data",
        "isolated_test_set_statement": (
            "The held-out examples were isolated before model training and tuning."
        ),
        "approval_status": approval_status,
    }


def _write_manifest(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_evidence(path: Path, model: BaseModel) -> Path:
    path.write_bytes(canonical_evidence_bytes(model))
    return path


def _source_ids(source: Path) -> tuple[str, ...]:
    return tuple(
        json.loads(line)["id"]
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _write_v12_evidence(
    root: Path,
    *,
    source: Path,
    prepared: PreparedDataset,
    draft: bool = False,
) -> tuple[Path, tuple[Path, Path], Path]:
    source_digest = _sha256(source)
    prepared_digest = _sha256(prepared.manifest_path)
    freeze = build_holdout_freeze(
        source_sha256=source_digest,
        prepared_manifest_sha256=prepared_digest,
        held_out_ids=tuple(prepared.split_ids["test"]),
    )
    freeze_path = _write_evidence(root / "holdout-freeze.json", freeze)
    freeze_digest = _sha256(freeze_path)
    ordered_ids = _source_ids(source)

    if draft:
        packet_models = build_draft_reviewer_packets(
            source_sha256=source_digest,
            prepared_manifest_sha256=prepared_digest,
            holdout_freeze_sha256=freeze_digest,
            ordered_record_ids=ordered_ids,
        )
    else:
        decisions = tuple(
            RecordReviewDecision(
                record_id=record_id,
                labels="approved",
                response="approved",
                pii="approved",
                license="approved",
            )
            for record_id in ordered_ids
        )
        packet_models = (
            ReviewerPacket(
                reviewer_id="human-reviewer-01",
                reviewer_kind="human",
                status="approved",
                source_sha256=source_digest,
                prepared_manifest_sha256=prepared_digest,
                holdout_freeze_sha256=freeze_digest,
                review_date="2026-07-18",
                decisions=decisions,
            ),
            ReviewerPacket(
                reviewer_id="human-reviewer-02",
                reviewer_kind="human",
                status="approved",
                source_sha256=source_digest,
                prepared_manifest_sha256=prepared_digest,
                holdout_freeze_sha256=freeze_digest,
                review_date="2026-07-18",
                decisions=decisions,
            ),
        )

    packet_paths = (
        _write_evidence(root / "reviewer-a.json", packet_models[0]),
        _write_evidence(root / "reviewer-b.json", packet_models[1]),
    )
    prepared_copy = root / "prepared-manifest.json"
    prepared_copy.write_bytes(prepared.manifest_path.read_bytes())
    references = (
        EvidenceFileReference(path=packet_paths[0].name, sha256=_sha256(packet_paths[0])),
        EvidenceFileReference(path=packet_paths[1].name, sha256=_sha256(packet_paths[1])),
    )
    intended_domain = "synthetic customer-support triage benchmark"
    license_statement = "CC0-1.0 synthetic records; no real customer data or consent dependency."
    isolation_statement = "The frozen test IDs were isolated before model training and tuning."
    prepared_reference = EvidenceFileReference(
        path=prepared_copy.name,
        sha256=_sha256(prepared_copy),
    )
    freeze_reference = EvidenceFileReference(
        path=freeze_path.name,
        sha256=freeze_digest,
    )
    if draft:
        aggregate = build_draft_review_manifest(
            source_sha256=source_digest,
            record_count=len(ordered_ids),
            intended_domain=intended_domain,
            consent_or_license_statement=license_statement,
            isolated_test_set_statement=isolation_statement,
            prepared_manifest=prepared_reference,
            holdout_freeze=freeze_reference,
            reviewer_packets=references,
        )
    else:
        aggregate = DatasetReviewManifestV12(
            source_sha256=source_digest,
            record_count=len(ordered_ids),
            review_date="2026-07-18",
            intended_domain=intended_domain,
            consent_or_license_statement=license_statement,
            pii_decision="no_real_customer_data",
            isolated_test_set_statement=isolation_statement,
            prepared_manifest=prepared_reference,
            holdout_freeze=freeze_reference,
            reviewer_packets=references,
            approval_status="approved",
        )
    aggregate_path = _write_evidence(root / "review-manifest.json", aggregate)
    return aggregate_path, packet_paths, freeze_path


def _decision_map(report: object) -> dict[str, bool]:
    return {item.policy: item.passed for item in report.decisions}  # type: ignore[attr-defined]


def _swap_path_during_first_descriptor_read(
    monkeypatch: pytest.MonkeyPatch,
    *,
    target: Path,
    replacement: Path,
    backup: Path,
    restore_before_return: bool,
) -> dict[str, bool]:
    original_read = qualification_module.os.read
    state = {"triggered": False}

    def swapping_read(descriptor: int, size: int) -> bytes:
        if state["triggered"]:
            return original_read(descriptor, size)
        state["triggered"] = True
        target.rename(backup)
        replacement.rename(target)
        try:
            return original_read(descriptor, size)
        finally:
            if restore_before_return:
                target.rename(replacement)
                backup.rename(target)

    monkeypatch.setattr(qualification_module.os, "read", swapping_read)
    return state


def _restore_replaced_path(*, target: Path, replacement: Path, backup: Path) -> None:
    if backup.exists():
        if target.exists():
            target.rename(replacement)
        backup.rename(target)


def test_review_manifest_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    manifest = tmp_path / "review.json"
    manifest.write_text(
        '{"schema_version":"1.1","source_sha256":"'
        + ("0" * 64)
        + '","source_sha256":"'
        + ("1" * 64)
        + '"}',
        encoding="utf-8",
    )

    with pytest.raises(DatasetQualificationError, match="duplicate"):
        load_review_manifest(manifest)


def test_review_manifest_rejects_unknown_keys(tmp_path: Path, qualified_source: Path) -> None:
    payload = _manifest_payload(qualified_source)
    payload["unexpected"] = True
    manifest = _write_manifest(tmp_path / "review.json", payload)

    with pytest.raises(DatasetQualificationError, match="invalid review manifest"):
        load_review_manifest(manifest)


def test_review_manifest_and_source_reject_symlinks(tmp_path: Path, qualified_source: Path) -> None:
    manifest = _write_manifest(tmp_path / "review.json", _manifest_payload(qualified_source))
    manifest_link = tmp_path / "review-link.json"
    manifest_link.symlink_to(manifest)
    source_link = tmp_path / "source-link.jsonl"
    source_link.symlink_to(qualified_source)

    with pytest.raises(DatasetQualificationError, match="non-symlink"):
        load_review_manifest(manifest_link)
    with pytest.raises(DatasetQualificationError, match="non-symlink"):
        qualify_dataset(source_link, manifest)


def test_review_manifest_swap_and_restore_during_read_fails_closed(
    tmp_path: Path,
    qualified_source: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _write_manifest(tmp_path / "review.json", _manifest_payload(qualified_source))
    replacement_payload = _manifest_payload(qualified_source, approval_status="rejected")
    replacement = _write_manifest(tmp_path / "replacement.json", replacement_payload)
    backup = tmp_path / "review.original.json"
    state = _swap_path_during_first_descriptor_read(
        monkeypatch,
        target=manifest,
        replacement=replacement,
        backup=backup,
        restore_before_return=True,
    )

    with pytest.raises(DatasetQualificationError, match="changed while it was being read"):
        load_review_manifest(manifest)

    assert state["triggered"] is True


def test_review_manifest_path_replacement_during_read_fails_closed(
    tmp_path: Path,
    qualified_source: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _write_manifest(tmp_path / "review.json", _manifest_payload(qualified_source))
    replacement = _write_manifest(
        tmp_path / "replacement.json",
        _manifest_payload(qualified_source, approval_status="rejected"),
    )
    backup = tmp_path / "review.original.json"
    state = _swap_path_during_first_descriptor_read(
        monkeypatch,
        target=manifest,
        replacement=replacement,
        backup=backup,
        restore_before_return=False,
    )

    try:
        with pytest.raises(DatasetQualificationError, match="changed while it was being read"):
            load_review_manifest(manifest)
    finally:
        _restore_replaced_path(target=manifest, replacement=replacement, backup=backup)

    assert state["triggered"] is True


def test_dataset_source_swap_and_restore_during_read_fails_closed(
    tmp_path: Path,
    qualified_source: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "tickets.jsonl"
    source.write_bytes(qualified_source.read_bytes())
    manifest = _write_manifest(tmp_path / "review.json", _manifest_payload(source))
    replacement = tmp_path / "replacement.jsonl"
    replacement.write_text('{"attacker":true}\n', encoding="utf-8")
    backup = tmp_path / "tickets.original.jsonl"
    state = _swap_path_during_first_descriptor_read(
        monkeypatch,
        target=source,
        replacement=replacement,
        backup=backup,
        restore_before_return=True,
    )

    with pytest.raises(DatasetQualificationError, match="changed while it was being read"):
        qualify_dataset(source, manifest, enforce=True)

    assert state["triggered"] is True


def test_dataset_source_path_replacement_during_read_fails_closed(
    tmp_path: Path,
    qualified_source: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "tickets.jsonl"
    source.write_bytes(qualified_source.read_bytes())
    manifest = _write_manifest(tmp_path / "review.json", _manifest_payload(source))
    replacement = tmp_path / "replacement.jsonl"
    replacement.write_text('{"attacker":true}\n', encoding="utf-8")
    backup = tmp_path / "tickets.original.jsonl"
    state = _swap_path_during_first_descriptor_read(
        monkeypatch,
        target=source,
        replacement=replacement,
        backup=backup,
        restore_before_return=False,
    )

    try:
        with pytest.raises(DatasetQualificationError, match="changed while it was being read"):
            qualify_dataset(source, manifest)
    finally:
        _restore_replaced_path(target=source, replacement=replacement, backup=backup)

    assert state["triggered"] is True


@pytest.mark.parametrize(
    ("mutation", "failed_policy"),
    [
        ({"source_sha256": "f" * 64}, "source_sha256_matches"),
        ({"record_count": 1001}, "record_count_matches"),
        ({"reviewed_count": 999}, "full_record_review"),
        ({"independent_reviewer_count": 1}, "minimum_independent_reviewers"),
        ({"held_out_count": 99}, "minimum_held_out_examples"),
        ({"approval_status": "draft"}, "approved_status"),
    ],
)
def test_policy_failures_are_explicit_and_enforceable(
    tmp_path: Path,
    qualified_source: Path,
    mutation: dict[str, object],
    failed_policy: str,
) -> None:
    payload = _manifest_payload(qualified_source)
    payload.update(mutation)
    if "held_out_count" in mutation:
        held_out_count = int(mutation["held_out_count"])
        payload["held_out_ids"] = list(payload["held_out_ids"])[:held_out_count]  # type: ignore[arg-type]
    manifest = _write_manifest(tmp_path / "review.json", payload)

    report = qualify_dataset(qualified_source, manifest)

    assert report.qualified is False
    assert _decision_map(report)[failed_policy] is False
    with pytest.raises(DatasetQualificationError, match="dataset qualification failed"):
        qualify_dataset(qualified_source, manifest, enforce=True)


def test_declared_holdout_count_must_match_explicit_ids(
    tmp_path: Path, qualified_source: Path
) -> None:
    payload = _manifest_payload(qualified_source)
    payload["held_out_count"] = 101
    manifest = _write_manifest(tmp_path / "review.json", payload)

    report = qualify_dataset(qualified_source, manifest)

    assert report.qualified is False
    assert _decision_map(report)["held_out_count_matches_ids"] is False


def test_every_explicit_holdout_id_must_exist_in_source(
    tmp_path: Path, qualified_source: Path
) -> None:
    payload = _manifest_payload(qualified_source)
    held_out_ids = list(payload["held_out_ids"])  # type: ignore[arg-type]
    held_out_ids[-1] = "QUAL-NOT-IN-SOURCE"
    payload["held_out_ids"] = held_out_ids
    manifest = _write_manifest(tmp_path / "review.json", payload)

    report = qualify_dataset(qualified_source, manifest)

    assert report.qualified is False
    assert _decision_map(report)["held_out_ids_within_source"] is False


def test_current_synthetic_corpus_is_portfolio_smoke_only(tmp_path: Path) -> None:
    payload = _manifest_payload(
        CURRENT_SYNTHETIC_SOURCE,
        record_count=56,
        reviewed_count=56,
        held_out_count=7,
    )
    manifest = _write_manifest(tmp_path / "review.json", payload)

    report = qualify_dataset(CURRENT_SYNTHETIC_SOURCE, manifest)

    assert report.qualified is False
    assert report.dataset_tier == "portfolio_smoke"
    decisions = _decision_map(report)
    assert decisions["minimum_record_count"] is False
    assert decisions["minimum_held_out_examples"] is False
    assert "does_not_prove_model_quality" in report.proof_boundary


def test_legacy_v11_never_qualifies_even_with_self_asserted_approval(
    tmp_path: Path, qualified_source: Path
) -> None:
    manifest = _write_manifest(tmp_path / "review.json", _manifest_payload(qualified_source))

    report = qualify_dataset(qualified_source, manifest)

    assert report.qualified is False
    assert report.schema_version == "1.1"
    assert report.dataset_tier == "qualification_candidate"
    assert report.source_sha256 == _sha256(qualified_source)
    assert report.review_manifest_sha256 == _sha256(manifest)
    assert report.record_count == 1000
    assert report.held_out_count == 100
    assert len(report.held_out_ids) == 100
    decisions = _decision_map(report)
    assert decisions["review_evidence_schema_v1_2"] is False
    assert decisions["minimum_independent_reviewers"] is False
    with pytest.raises(DatasetQualificationError, match="dataset qualification failed"):
        qualify_dataset(qualified_source, manifest, enforce=True)


def test_v12_two_packet_chain_qualifies_with_transitive_hashes(
    tmp_path: Path,
    qualified_source: Path,
    qualified_prepared: PreparedDataset,
) -> None:
    aggregate, packets, freeze = _write_v12_evidence(
        tmp_path,
        source=qualified_source,
        prepared=qualified_prepared,
    )

    report = qualify_dataset(qualified_source, aggregate, enforce=True)

    assert report.qualified is True
    assert report.schema_version == "1.2"
    assert report.review_manifest_sha256 == _sha256(aggregate)
    assert report.prepared_manifest_sha256 == qualified_prepared.manifest_sha256
    assert report.holdout_freeze_sha256 == _sha256(freeze)
    assert report.reviewer_packet_sha256 == tuple(_sha256(path) for path in packets)
    assert report.reviewer_ids == ("human-reviewer-01", "human-reviewer-02")
    assert report.reviewed_count == 1000
    assert report.independent_reviewer_count == 2
    assert all(item.passed for item in report.decisions)


def test_deterministic_draft_packets_are_pending_and_never_qualify(
    tmp_path: Path,
    qualified_source: Path,
    qualified_prepared: PreparedDataset,
) -> None:
    aggregate, packets, _freeze = _write_v12_evidence(
        tmp_path,
        source=qualified_source,
        prepared=qualified_prepared,
        draft=True,
    )

    packet_a = packets[0].read_bytes()
    parsed = json.loads(packet_a)
    assert parsed["reviewer_id"] == "REPLACE_WITH_REVIEWER_A"
    assert parsed["review_date"] is None
    assert parsed["status"] == "draft"
    assert all(set(decision.values()) >= {"pending"} for decision in parsed["decisions"])
    assert evidence_sha256(load_review_manifest(aggregate)) == _sha256(aggregate)

    report = qualify_dataset(qualified_source, aggregate)

    assert report.qualified is False
    decisions = _decision_map(report)
    assert decisions["reviewer_ids_non_placeholder"] is False
    assert decisions["review_packets_approved"] is False
    assert decisions["review_packet_decisions_approved"] is False
    with pytest.raises(DatasetQualificationError, match="dataset qualification failed"):
        qualify_dataset(qualified_source, aggregate, enforce=True)


def test_v12_rejects_packet_reference_traversal(
    tmp_path: Path,
    qualified_source: Path,
    qualified_prepared: PreparedDataset,
) -> None:
    aggregate, _packets, _freeze = _write_v12_evidence(
        tmp_path,
        source=qualified_source,
        prepared=qualified_prepared,
    )
    payload = json.loads(aggregate.read_text(encoding="utf-8"))
    payload["reviewer_packets"][0]["path"] = "../reviewer-a.json"
    _write_manifest(aggregate, payload)

    with pytest.raises(DatasetQualificationError, match="invalid review manifest"):
        qualify_dataset(qualified_source, aggregate)


def test_v12_packet_order_and_hash_mismatch_fail_explicitly(
    tmp_path: Path,
    qualified_source: Path,
    qualified_prepared: PreparedDataset,
) -> None:
    aggregate, packets, _freeze = _write_v12_evidence(
        tmp_path,
        source=qualified_source,
        prepared=qualified_prepared,
    )
    packet_payload = json.loads(packets[0].read_text(encoding="utf-8"))
    packet_payload["decisions"][0], packet_payload["decisions"][1] = (
        packet_payload["decisions"][1],
        packet_payload["decisions"][0],
    )
    _write_manifest(packets[0], packet_payload)

    report = qualify_dataset(qualified_source, aggregate)

    decisions = _decision_map(report)
    assert report.qualified is False
    assert decisions["review_packet_hashes_match"] is False
    assert decisions["review_packets_cover_ordered_source"] is False


def test_quality_profile_targets_separate_qualified_data_and_artifacts() -> None:
    config = load_config(PROJECT_ROOT / "configs/qwen-7b-qlora-quality.yaml")

    assert config.model.name_or_path == "Qwen/Qwen2.5-7B-Instruct"
    assert config.lora.method == "qlora"
    assert config.data.source_path == PROJECT_ROOT / "data/qualified/support_tickets.jsonl"
    assert config.data.processed_dir == PROJECT_ROOT / "data/processed/qwen-7b-quality"
    assert config.training.output_dir == PROJECT_ROOT / "artifacts/qwen-7b-quality"
    assert config.evaluation.thresholds.strict_json_rate == 0.97


def test_example_review_manifest_is_deliberately_unapproved() -> None:
    manifest = load_review_manifest(PROJECT_ROOT / "data/qualified/review-manifest.example.json")

    assert isinstance(manifest, DatasetReviewManifestV12)
    assert manifest.record_count == 1120
    assert manifest.review_date is None
    assert len(manifest.reviewer_packets) == 2
    assert len({packet.path for packet in manifest.reviewer_packets}) == 2
    assert manifest.pii_decision == "no_real_customer_data"
    assert manifest.approval_status == "draft"
