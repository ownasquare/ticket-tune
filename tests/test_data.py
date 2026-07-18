from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

import tickettune.data as data_module
from tickettune.config import SplitConfig
from tickettune.data import (
    DatasetIntegrityError,
    DatasetValidationError,
    assert_holdout_label_coverage,
    assert_no_split_leakage,
    load_examples,
    materialize_prepared_split_snapshots,
    prepare_dataset,
    project_for_trl,
    sha256_file,
    split_examples,
    validate_examples,
    verify_prepared_dataset,
    verify_prepared_split_snapshot,
)
from tickettune.prompts import OUTPUT_FIELDS, SYSTEM_PROMPT
from tickettune.schemas import (
    CATEGORY_LABELS,
    PRIORITY_LABELS,
    SENTIMENT_LABELS,
    TicketExample,
)
from tickettune.synthetic import canonical_qualified_synthetic_bytes

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).parent / "fixtures" / "tickets.jsonl"
SOURCE = ROOT / "data" / "raw" / "support_tickets.jsonl"


def _fixture_records() -> list[dict[str, object]]:
    return [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines()]


def _write_records(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text("".join(f"{json.dumps(record)}\n" for record in records), encoding="utf-8")


def _read_manifest(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write_manifest(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _large_single_category_examples(size: int = 600) -> list[TicketExample]:
    template = load_examples(FIXTURE)[0]
    priorities = ("low", "medium", "high", "urgent")
    sentiments = ("positive", "neutral", "frustrated", "angry", "worried")
    examples: list[TicketExample] = []
    for index in range(size):
        expected = template.expected.model_copy(
            update={
                "priority": priorities[index % len(priorities)],
                "sentiment": sentiments[index % len(sentiments)],
            }
        )
        messages = [
            template.messages[0],
            template.messages[1].model_copy(
                update={
                    "content": (
                        f"Synthetic large-stratum ticket {index} for [CUSTOMER_NAME] on [DEVICE]."
                    )
                }
            ),
            template.messages[2].model_copy(
                update={
                    "content": json.dumps(
                        expected.model_dump(mode="json"),
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                }
            ),
        ]
        examples.append(
            TicketExample.model_validate(
                template.model_copy(
                    update={
                        "id": f"TT-LARGE-{index:05d}",
                        "messages": messages,
                        "expected": expected,
                    }
                ).model_dump(mode="json")
            )
        )
    return examples


def test_source_corpus_is_balanced_synthetic_and_unique() -> None:
    examples = load_examples(SOURCE)

    assert len(examples) == 56
    assert len({example.id for example in examples}) == 56
    assert Counter(example.expected.category for example in examples) == {
        "account_access": 8,
        "billing": 8,
        "bug": 8,
        "cancellation": 8,
        "feature_request": 8,
        "shipping": 8,
        "security": 8,
    }
    assert Counter(example.expected.priority for example in examples) == {
        "low": 14,
        "medium": 14,
        "high": 14,
        "urgent": 14,
    }
    assert all(example.provenance.source == "synthetic" for example in examples)
    assert all(example.provenance.contains_real_customer_data is False for example in examples)
    assert all(example.pii_placeholders for example in examples)


def test_system_prompt_is_a_self_contained_output_contract() -> None:
    assert "Return exactly one JSON object and no markdown" in SYSTEM_PROMPT
    assert "Use exactly these five keys and no others: " + ", ".join(OUTPUT_FIELDS) in SYSTEM_PROMPT
    for label in CATEGORY_LABELS:
        assert label in SYSTEM_PROMPT
    for label in PRIORITY_LABELS:
        assert label in SYSTEM_PROMPT
    for label in SENTIMENT_LABELS:
        assert label in SYSTEM_PROMPT
    assert "response must be 12 to 1200 characters" in SYSTEM_PROMPT
    assert "must not reveal raw private data" in SYSTEM_PROMPT
    assert "next_action must match ^[a-z][a-z0-9_]{2,63}$" in SYSTEM_PROMPT
    assert "never use spaces or uppercase" in SYSTEM_PROMPT
    assert "feature_request is requested new or changed behavior" in SYSTEM_PROMPT
    assert "bug is a malfunction against current behavior" in SYSTEM_PROMPT
    assert (
        "high is a material blocker, billing risk, or time-sensitive cancellation" in SYSTEM_PROMPT
    )
    assert (
        "urgent is an active security threat, outage, data-loss risk, or launch blocker"
        in SYSTEM_PROMPT
    )
    assert "angry is explicit anger or accusation" in SYSTEM_PROMPT
    assert "worried is fear or anxiety" in SYSTEM_PROMPT

    examples = load_examples(SOURCE)
    assert all(example.messages[0].content == SYSTEM_PROMPT for example in examples)


def test_source_sentiments_are_explicit_and_not_priority_aliases() -> None:
    cues = {
        "positive": "I appreciate the help so far. ",
        "neutral": "For context, ",
        "frustrated": "I'm frustrated by this. ",
        "angry": "I'm angry that this happened. ",
        "worried": "I'm worried about the impact. ",
    }
    examples = load_examples(SOURCE)

    priorities_by_sentiment: dict[str, set[str]] = {
        sentiment: set() for sentiment in SENTIMENT_LABELS
    }
    for example in examples:
        user_message = next(
            message.content for message in example.messages if message.role == "user"
        )
        assert user_message.startswith(cues[example.expected.sentiment])
        priorities_by_sentiment[example.expected.sentiment].add(example.expected.priority)

    assert priorities_by_sentiment == {
        sentiment: set(PRIORITY_LABELS) for sentiment in SENTIMENT_LABELS
    }


def test_prepare_dataset_is_reproducible(tmp_path: Path) -> None:
    split_config = SplitConfig(train=0.75, validation=0.125, test=0.125)

    first = prepare_dataset(FIXTURE, tmp_path / "one", seed=42, splits=split_config)
    second = prepare_dataset(FIXTURE, tmp_path / "two", seed=42, splits=split_config)

    assert first.source_sha256 == second.source_sha256
    assert first.split_ids == second.split_ids
    assert {name: artifact.sha256 for name, artifact in first.splits.items()} == {
        name: artifact.sha256 for name, artifact in second.splits.items()
    }
    assert first.manifest_sha256 == second.manifest_sha256


def test_prepare_dataset_writes_auditable_hashes_and_strata(tmp_path: Path) -> None:
    prepared = prepare_dataset(FIXTURE, tmp_path / "prepared", seed=17)

    assert prepared.total_examples == 16
    assert prepared.manifest_sha256 == sha256_file(prepared.manifest_path)
    manifest = json.loads(prepared.manifest_path.read_text(encoding="utf-8"))
    assert manifest["source_sha256"] == sha256_file(FIXTURE)
    assert sum(item["count"] for item in manifest["splits"].values()) == 16
    for split_name in ("train", "validation", "test"):
        artifact = prepared.splits[split_name]
        assert artifact.sha256 == sha256_file(prepared.output_dir / artifact.file)
        assert set(artifact.labels.category) == {"account_access", "billing"}


@pytest.mark.parametrize("source_name", ["train.jsonl", "manifest.json"])
def test_prepare_dataset_rejects_source_at_reserved_output_path_without_mutation(
    tmp_path: Path,
    source_name: str,
) -> None:
    output_dir = tmp_path / "prepared"
    output_dir.mkdir()
    source_path = output_dir / source_name
    original = FIXTURE.read_bytes()
    source_path.write_bytes(original)

    with pytest.raises(DatasetValidationError, match="must not overlap"):
        prepare_dataset(source_path, output_dir, seed=17)

    assert source_path.read_bytes() == original
    assert sorted(path.name for path in output_dir.iterdir()) == [source_name]


def test_prepare_dataset_rejects_symlink_resolved_source_output_overlap(
    tmp_path: Path,
) -> None:
    real_output = tmp_path / "real-output"
    real_output.mkdir()
    source_target = real_output / "source.jsonl"
    original = FIXTURE.read_bytes()
    source_target.write_bytes(original)
    source_alias = tmp_path / "source-alias.jsonl"
    source_alias.symlink_to(source_target)
    output_alias = tmp_path / "output-alias"
    output_alias.symlink_to(real_output, target_is_directory=True)

    with pytest.raises(DatasetValidationError, match="after symlink resolution"):
        prepare_dataset(source_alias, output_alias, seed=17)

    assert source_target.read_bytes() == original
    assert sorted(path.name for path in real_output.iterdir()) == [source_target.name]


def test_verify_prepared_dataset_accepts_exact_manifest_chain(tmp_path: Path) -> None:
    prepared = prepare_dataset(FIXTURE, tmp_path / "prepared", seed=17)

    verified = verify_prepared_dataset(
        FIXTURE,
        prepared.output_dir,
        seed=17,
        splits=SplitConfig(),
        required_splits=("train", "validation"),
    )

    assert verified.status == "verified"
    assert verified.manifest_path == prepared.manifest_path
    assert verified.manifest_sha256 == prepared.manifest_sha256
    assert verified.source_sha256 == prepared.source_sha256
    assert verified.verified_splits == ("train", "validation")
    assert verified.split_counts == {
        "train": prepared.splits["train"].count,
        "validation": prepared.splits["validation"].count,
    }
    assert verified.split_sha256 == {
        "train": prepared.splits["train"].sha256,
        "validation": prepared.splits["validation"].sha256,
    }
    assert verified.split_paths["train"] == prepared.output_dir / "train.jsonl"


def test_materialize_prepared_split_snapshots_copies_exact_verified_bytes(
    tmp_path: Path,
) -> None:
    prepared = prepare_dataset(FIXTURE, tmp_path / "prepared", seed=17)
    verified = verify_prepared_dataset(
        FIXTURE,
        prepared.output_dir,
        seed=17,
        splits=SplitConfig(),
        required_splits=("train", "validation", "test"),
    )

    snapshots = materialize_prepared_split_snapshots(
        verified,
        tmp_path / "run" / "dataset",
        split_names=("train", "validation"),
    )

    assert tuple(snapshots) == ("train", "validation")
    for split_name, snapshot in snapshots.items():
        assert snapshot.split_name == split_name
        assert snapshot.source_path == prepared.output_dir / f"{split_name}.jsonl"
        assert snapshot.path == tmp_path / "run" / "dataset" / f"{split_name}.jsonl"
        assert snapshot.path.read_bytes() == snapshot.source_path.read_bytes()
        assert snapshot.sha256 == verified.split_sha256[split_name]
        assert snapshot.size_bytes == snapshot.path.stat().st_size
        verify_prepared_split_snapshot(snapshot)


def test_materialize_prepared_split_snapshots_rejects_post_verification_symlink(
    tmp_path: Path,
) -> None:
    prepared = prepare_dataset(FIXTURE, tmp_path / "prepared", seed=17)
    verified = verify_prepared_dataset(
        FIXTURE,
        prepared.output_dir,
        seed=17,
        splits=SplitConfig(),
        required_splits=("train", "validation", "test"),
    )
    train_path = prepared.output_dir / "train.jsonl"
    same_bytes = tmp_path / "same-train.jsonl"
    same_bytes.write_bytes(train_path.read_bytes())
    train_path.unlink()
    train_path.symlink_to(same_bytes)

    with pytest.raises(DatasetIntegrityError, match="train split must be a regular non-symlink"):
        materialize_prepared_split_snapshots(
            verified,
            tmp_path / "run" / "dataset",
            split_names=("train", "validation"),
        )


def test_verify_prepared_split_snapshot_rejects_same_byte_path_replacement(
    tmp_path: Path,
) -> None:
    prepared = prepare_dataset(FIXTURE, tmp_path / "prepared", seed=17)
    verified = verify_prepared_dataset(
        FIXTURE,
        prepared.output_dir,
        seed=17,
        splits=SplitConfig(),
        required_splits=("train", "validation", "test"),
    )
    snapshot = materialize_prepared_split_snapshots(
        verified,
        tmp_path / "run" / "dataset",
        split_names=("train",),
    )["train"]
    payload = snapshot.path.read_bytes()
    snapshot.path.unlink()
    snapshot.path.write_bytes(payload)

    with pytest.raises(DatasetIntegrityError, match="snapshot identity changed"):
        verify_prepared_split_snapshot(snapshot)


def test_verify_prepared_dataset_rejects_missing_manifest(tmp_path: Path) -> None:
    prepared = prepare_dataset(FIXTURE, tmp_path / "prepared", seed=17)
    prepared.manifest_path.unlink()

    with pytest.raises(DatasetIntegrityError, match="manifest not found"):
        verify_prepared_dataset(
            FIXTURE,
            prepared.output_dir,
            seed=17,
            splits=SplitConfig(),
            required_splits=("test",),
        )


def test_verify_prepared_dataset_rejects_duplicate_manifest_key(tmp_path: Path) -> None:
    prepared = prepare_dataset(FIXTURE, tmp_path / "prepared", seed=17)
    manifest_text = prepared.manifest_path.read_text(encoding="utf-8")
    prepared.manifest_path.write_text(
        manifest_text.replace(
            '  "seed": 17,\n',
            '  "seed": 99,\n  "seed": 17,\n',
            1,
        ),
        encoding="utf-8",
    )

    with pytest.raises(DatasetIntegrityError, match=r"duplicate JSON object key 'seed'"):
        verify_prepared_dataset(
            FIXTURE,
            prepared.output_dir,
            seed=17,
            splits=SplitConfig(),
            required_splits=("train",),
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"schema_version": "2.0"}, "manifest schema"),
        ({"seed": 99}, "seed mismatch"),
        (
            {"split_fractions": {"train": 0.5, "validation": 0.25, "test": 0.25}},
            "split fractions mismatch",
        ),
    ],
)
def test_verify_prepared_dataset_rejects_manifest_configuration_mismatch(
    tmp_path: Path,
    mutation: dict[str, object],
    message: str,
) -> None:
    prepared = prepare_dataset(FIXTURE, tmp_path / "prepared", seed=17)
    manifest = _read_manifest(prepared.manifest_path)
    manifest.update(mutation)
    _write_manifest(prepared.manifest_path, manifest)

    with pytest.raises(DatasetIntegrityError, match=message):
        verify_prepared_dataset(
            FIXTURE,
            prepared.output_dir,
            seed=17,
            splits=SplitConfig(),
            required_splits=("train",),
        )


def test_verify_prepared_dataset_rejects_source_hash_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_bytes(FIXTURE.read_bytes())
    prepared = prepare_dataset(source, tmp_path / "prepared", seed=17)
    source.write_bytes(source.read_bytes() + b"\n")

    with pytest.raises(DatasetIntegrityError, match="source SHA-256 mismatch"):
        verify_prepared_dataset(
            source,
            prepared.output_dir,
            seed=17,
            splits=SplitConfig(),
            required_splits=("test",),
        )


def test_verify_prepared_dataset_rejects_traversal_filename(tmp_path: Path) -> None:
    prepared = prepare_dataset(FIXTURE, tmp_path / "prepared", seed=17)
    manifest = _read_manifest(prepared.manifest_path)
    manifest_splits = manifest["splits"]
    assert isinstance(manifest_splits, dict)
    train = manifest_splits["train"]
    assert isinstance(train, dict)
    train["file"] = "../train.jsonl"
    _write_manifest(prepared.manifest_path, manifest)

    with pytest.raises(DatasetIntegrityError, match="canonical filename"):
        verify_prepared_dataset(
            FIXTURE,
            prepared.output_dir,
            seed=17,
            splits=SplitConfig(),
            required_splits=("train",),
        )


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("count", "count mismatch"),
        ("ids", "ordered IDs mismatch"),
        ("sha256", "SHA-256 mismatch"),
    ],
)
def test_verify_prepared_dataset_rejects_tampered_split_metadata(
    tmp_path: Path,
    field: str,
    message: str,
) -> None:
    prepared = prepare_dataset(FIXTURE, tmp_path / "prepared", seed=17)
    manifest = _read_manifest(prepared.manifest_path)
    manifest_splits = manifest["splits"]
    split_ids = manifest["split_ids"]
    assert isinstance(manifest_splits, dict)
    assert isinstance(split_ids, dict)
    train = manifest_splits["train"]
    assert isinstance(train, dict)
    if field == "count":
        train["count"] = int(train["count"]) + 1
    elif field == "ids":
        train_ids = split_ids["train"]
        assert isinstance(train_ids, list)
        train_ids[0] = "TT-TAMPERED"
    else:
        train["sha256"] = "0" * 64
    _write_manifest(prepared.manifest_path, manifest)

    with pytest.raises(DatasetIntegrityError, match=message):
        verify_prepared_dataset(
            FIXTURE,
            prepared.output_dir,
            seed=17,
            splits=SplitConfig(),
            required_splits=("train",),
        )


def test_verify_prepared_dataset_rejects_changed_split_bytes(tmp_path: Path) -> None:
    prepared = prepare_dataset(FIXTURE, tmp_path / "prepared", seed=17)
    test_path = prepared.output_dir / "test.jsonl"
    test_path.write_bytes(test_path.read_bytes() + b"\n")

    with pytest.raises(DatasetIntegrityError, match="SHA-256 mismatch"):
        verify_prepared_dataset(
            FIXTURE,
            prepared.output_dir,
            seed=17,
            splits=SplitConfig(),
            required_splits=("test",),
        )


def test_verify_rejects_semantically_poisoned_split_with_recomputed_hash(
    tmp_path: Path,
) -> None:
    prepared = prepare_dataset(FIXTURE, tmp_path / "prepared", seed=17)
    train_path = prepared.output_dir / "train.jsonl"
    records = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines()]
    first = records[0]
    first["prompt"][1]["content"] += " Poisoned deterministic projection."
    first["expected"]["next_action"] = "poisoned_action"
    completion = json.loads(first["completion"][0]["content"])
    completion["next_action"] = "poisoned_action"
    first["completion"][0]["content"] = json.dumps(
        completion,
        sort_keys=True,
        separators=(",", ":"),
    )
    _write_records(train_path, records)

    manifest = _read_manifest(prepared.manifest_path)
    manifest_splits = manifest["splits"]
    assert isinstance(manifest_splits, dict)
    train_manifest = manifest_splits["train"]
    assert isinstance(train_manifest, dict)
    train_manifest["sha256"] = sha256_file(train_path)
    _write_manifest(prepared.manifest_path, manifest)

    with pytest.raises(DatasetIntegrityError, match="deterministic source projection"):
        verify_prepared_dataset(
            FIXTURE,
            prepared.output_dir,
            seed=17,
            splits=SplitConfig(),
            required_splits=("train",),
        )


def test_verify_rejects_duplicate_json_key_in_processed_row(tmp_path: Path) -> None:
    prepared = prepare_dataset(FIXTURE, tmp_path / "prepared", seed=17)
    train_path = prepared.output_dir / "train.jsonl"
    rows = train_path.read_text(encoding="utf-8").splitlines()
    rows[0] = rows[0].replace('"expected":{', '"expected":{"category":"billing",', 1)
    train_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    with pytest.raises(
        DatasetIntegrityError,
        match=r"train\.jsonl:1: invalid JSON: duplicate JSON object key 'category'",
    ):
        verify_prepared_dataset(
            FIXTURE,
            prepared.output_dir,
            seed=17,
            splits=SplitConfig(),
            required_splits=("train",),
        )


def test_full_corpus_splits_preserve_category_and_priority_balance(tmp_path: Path) -> None:
    prepared = prepare_dataset(SOURCE, tmp_path / "full", seed=42)

    assert {name: artifact.count for name, artifact in prepared.splits.items()} == {
        "train": 42,
        "validation": 7,
        "test": 7,
    }
    for split_name, expected_category_count in {
        "train": 6,
        "validation": 1,
        "test": 1,
    }.items():
        artifact = prepared.splits[split_name]
        assert set(artifact.labels.category.values()) == {expected_category_count}
        priority_counts = artifact.labels.priority.values()
        assert max(priority_counts) - min(priority_counts) <= 1, artifact.labels.priority
        if split_name in {"validation", "test"}:
            assert set(artifact.labels.sentiment) == {
                "positive",
                "neutral",
                "frustrated",
                "angry",
                "worried",
            }


def test_full_corpus_split_hashes_remain_stable(tmp_path: Path) -> None:
    prepared = prepare_dataset(SOURCE, tmp_path / "full", seed=42)

    assert prepared.manifest_sha256 == (
        "c800b714dd8b4f3098f90dcce384f5146c5e8d6dc13e6f9725c66e3b1fe3e1ed"
    )
    assert {name: artifact.sha256 for name, artifact in prepared.splits.items()} == {
        "train": "f026b75de35a603802c66042ebc292921f99df1f97ed5c006b6d829635795798",
        "validation": "7e80845a9e72b7dcf62afbac8102d8de3c48826ef7454bd044405ff2490b47b2",
        "test": "08d8433b2e2fdeaf99946b1266a6958bac507a8e736e4f98aae2114bd02a6bb0",
    }


def test_large_category_uses_bounded_deterministic_holdout_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    examples = _large_single_category_examples()

    def reject_exhaustive_combinations(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("large strata must not enumerate itertools.combinations")

    monkeypatch.setattr(data_module, "combinations", reject_exhaustive_combinations)
    first = split_examples(examples, seed=42, splits=SplitConfig())
    second = split_examples(examples, seed=42, splits=SplitConfig())

    assert {name: len(items) for name, items in first.items()} == {
        "train": 450,
        "validation": 75,
        "test": 75,
    }
    assert {name: [item.id for item in items] for name, items in first.items()} == {
        name: [item.id for item in items] for name, items in second.items()
    }
    for split_name in ("validation", "test"):
        priority_counts = Counter(item.expected.priority for item in first[split_name])
        assert max(priority_counts.values()) - min(priority_counts.values()) <= 1
        assert {item.expected.sentiment for item in first[split_name]} == {
            "positive",
            "neutral",
            "frustrated",
            "angry",
            "worried",
        }


def test_qualified_candidate_split_reuses_bounded_choice_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "qualified-synthetic.jsonl"
    source.write_bytes(canonical_qualified_synthetic_bytes())
    split_config = SplitConfig(train=0.8, validation=0.1, test=0.1)
    original_order_key = data_module._order_key
    call_limit = 100_000
    call_count = 0

    def bounded_order_key(seed: int, scope: str, example_id: str) -> bytes:
        nonlocal call_count
        call_count += 1
        if call_count > call_limit:
            raise AssertionError("large-corpus allocation repeated invariant choice hashing")
        return original_order_key(seed, scope, example_id)

    monkeypatch.setattr(data_module, "_order_key", bounded_order_key)
    first = prepare_dataset(source, tmp_path / "first", seed=42, splits=split_config)
    first_call_count = call_count
    call_count = 0
    second = prepare_dataset(source, tmp_path / "second", seed=42, splits=split_config)

    assert 0 < first_call_count <= call_limit
    assert 0 < call_count <= call_limit
    assert {name: artifact.count for name, artifact in first.splits.items()} == {
        "train": 896,
        "validation": 112,
        "test": 112,
    }
    assert first.split_ids == second.split_ids
    assert {name: artifact.sha256 for name, artifact in first.splits.items()} == {
        name: artifact.sha256 for name, artifact in second.splits.items()
    }


def test_prepare_rejects_source_changed_while_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.jsonl"
    source.write_bytes(FIXTURE.read_bytes())
    original_parse = data_module._load_examples_from_bytes

    def parse_then_mutate(payload: bytes, path: Path) -> list[TicketExample]:
        examples = original_parse(payload, path)
        path.write_bytes(payload + b"\n")
        return examples

    monkeypatch.setattr(data_module, "_load_examples_from_bytes", parse_then_mutate)

    with pytest.raises(DatasetValidationError, match="source changed during dataset preparation"):
        prepare_dataset(source, tmp_path / "prepared", seed=17)


def test_verify_rejects_source_changed_while_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.jsonl"
    source.write_bytes(FIXTURE.read_bytes())
    prepared = prepare_dataset(source, tmp_path / "prepared", seed=17)
    original_parse = data_module._load_examples_from_bytes

    def parse_then_mutate(payload: bytes, path: Path) -> list[TicketExample]:
        examples = original_parse(payload, path)
        path.write_bytes(payload + b"\n")
        return examples

    monkeypatch.setattr(data_module, "_load_examples_from_bytes", parse_then_mutate)

    with pytest.raises(
        DatasetIntegrityError,
        match="source changed during prepared-dataset verification",
    ):
        verify_prepared_dataset(
            source,
            prepared.output_dir,
            seed=17,
            splits=SplitConfig(),
            required_splits=("train",),
        )


@pytest.mark.parametrize("operation", ["prepare", "verify"])
def test_dataset_consumers_reject_swap_and_restore_after_using_verified_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    source = tmp_path / "source.jsonl"
    original_payload = FIXTURE.read_bytes()
    source.write_bytes(original_payload)
    prepared = prepare_dataset(source, tmp_path / "prepared", seed=17)
    original_parse = data_module._load_examples_from_bytes
    parsed_payloads: list[bytes] = []

    def parse_during_swap(payload: bytes, path: Path) -> list[TicketExample]:
        backup = path.with_suffix(".verified-original")
        path.rename(backup)
        path.write_bytes(b'{"id":"attacker-controlled"}\n')
        try:
            parsed_payloads.append(payload)
            return original_parse(payload, path)
        finally:
            path.unlink()
            backup.rename(path)

    monkeypatch.setattr(data_module, "_load_examples_from_bytes", parse_during_swap)

    if operation == "prepare":
        with pytest.raises(DatasetValidationError, match="source changed"):
            prepare_dataset(source, tmp_path / "prepared-again", seed=17)
    else:
        with pytest.raises(DatasetIntegrityError, match="source changed"):
            verify_prepared_dataset(
                source,
                prepared.output_dir,
                seed=17,
                splits=SplitConfig(),
                required_splits=("train",),
            )

    assert parsed_payloads == [original_payload]


def test_holdout_coverage_audit_fails_when_feasible_sentiment_is_missing() -> None:
    examples = load_examples(SOURCE)
    splits = split_examples(examples, seed=42, splits=SplitConfig())
    altered = {name: list(items) for name, items in splits.items()}
    removed = next(item for item in altered["validation"] if item.expected.sentiment == "angry")
    replacement = next(
        item
        for item in altered["train"]
        if item.expected.category == removed.expected.category
        and item.expected.priority == removed.expected.priority
        and item.expected.sentiment != "angry"
    )
    altered["validation"].remove(removed)
    altered["validation"].append(replacement)
    altered["train"].remove(replacement)
    altered["train"].append(removed)

    with pytest.raises(ValueError, match="missing canonical sentiment labels"):
        assert_holdout_label_coverage(examples, altered)


def test_duplicate_ids_report_the_second_source_line(tmp_path: Path) -> None:
    records = _fixture_records()[:2]
    records[1]["id"] = records[0]["id"]
    path = tmp_path / "duplicate-id.jsonl"
    _write_records(path, records)

    with pytest.raises(DatasetValidationError, match=r":2: duplicate id"):
        load_examples(path)


def test_duplicate_normalized_content_is_rejected(tmp_path: Path) -> None:
    record = _fixture_records()[0]
    duplicate = json.loads(json.dumps(record))
    duplicate["id"] = "TT-DUPLICATE"
    path = tmp_path / "duplicate-content.jsonl"
    _write_records(path, [record, duplicate])

    with pytest.raises(DatasetValidationError, match="duplicate normalized user content"):
        load_examples(path)


def test_duplicate_normalization_ignores_case_spacing_and_punctuation(tmp_path: Path) -> None:
    record = _fixture_records()[0]
    duplicate = json.loads(json.dumps(record))
    duplicate["id"] = "TT-NORMALIZED-DUPLICATE"
    messages = duplicate["messages"]
    assert isinstance(messages, list)
    user_message = messages[1]
    assert isinstance(user_message, dict)
    user_message["content"] = f"  {str(user_message['content']).upper()} !!!  "
    path = tmp_path / "normalized-duplicate.jsonl"
    _write_records(path, [record, duplicate])

    with pytest.raises(DatasetValidationError, match="duplicate normalized user content"):
        load_examples(path)


def test_possible_unredacted_pii_is_rejected(tmp_path: Path) -> None:
    record = _fixture_records()[0]
    messages = record["messages"]
    assert isinstance(messages, list)
    user_message = messages[1]
    assert isinstance(user_message, dict)
    user_message["content"] = str(user_message["content"]) + " Contact jane@example.test."
    path = tmp_path / "pii.jsonl"
    _write_records(path, [record])

    with pytest.raises(DatasetValidationError, match="email address"):
        load_examples(path)


def test_invalid_json_reports_line_number(tmp_path: Path) -> None:
    path = tmp_path / "invalid.jsonl"
    path.write_text(FIXTURE.read_text(encoding="utf-8").splitlines()[0] + "\n{bad json}\n")

    with pytest.raises(DatasetValidationError, match=r":2:"):
        load_examples(path)


def test_duplicate_json_key_in_source_reports_line_and_key(tmp_path: Path) -> None:
    first_line = FIXTURE.read_text(encoding="utf-8").splitlines()[0]
    path = tmp_path / "duplicate-key.jsonl"
    path.write_text(first_line.replace("{", '{"id":"TT-SHADOW",', 1) + "\n", encoding="utf-8")

    with pytest.raises(
        DatasetValidationError,
        match=r":1: duplicate JSON object key 'id'",
    ):
        load_examples(path)


@pytest.mark.parametrize("content", ["", "\n"])
def test_empty_and_blank_datasets_fail_closed(tmp_path: Path, content: str) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(DatasetValidationError):
        load_examples(path)

    with pytest.raises(DatasetValidationError, match="at least one example"):
        validate_examples([], source_path=path)


def test_small_categories_still_cover_every_split() -> None:
    examples = load_examples(FIXTURE)[:3]

    splits = split_examples(
        examples,
        seed=42,
        splits=SplitConfig(train=0.8, validation=0.1, test=0.1),
    )

    assert {name: len(items) for name, items in splits.items()} == {
        "train": 1,
        "validation": 1,
        "test": 1,
    }


def test_cross_split_content_leak_is_rejected() -> None:
    example = load_examples(FIXTURE)[0]

    with pytest.raises(ValueError, match="split leakage"):
        assert_no_split_leakage({"train": [example], "validation": [], "test": [example]})


def test_cross_split_normalized_content_leak_is_rejected_with_distinct_ids() -> None:
    record = _fixture_records()[0]
    duplicate = json.loads(json.dumps(record))
    duplicate["id"] = "TT-LEAK-DUPLICATE"
    first = TicketExample.model_validate(record)
    second = TicketExample.model_validate(duplicate)

    with pytest.raises(ValueError, match="normalized content"):
        assert_no_split_leakage({"train": [first], "validation": [], "test": [second]})


def test_trl_projection_uses_completion_only_boundary() -> None:
    example = TicketExample.model_validate(_fixture_records()[0])

    projected = project_for_trl(example)

    assert [message["role"] for message in projected["prompt"]] == ["system", "user"]
    assert [message["role"] for message in projected["completion"]] == ["assistant"]
    assert json.loads(projected["completion"][0]["content"]) == projected["expected"]
