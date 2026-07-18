from __future__ import annotations

import hashlib
import json
import subprocess  # nosec B404 - isolated import contract only
import sys
from collections import Counter
from pathlib import Path

import pytest

from tickettune.data import load_examples, validate_examples
from tickettune.schemas import CATEGORY_LABELS, PRIORITY_LABELS, SENTIMENT_LABELS
from tickettune.synthetic import (
    QUALIFIED_SYNTHETIC_RECORD_COUNT,
    build_qualified_synthetic_examples,
    canonical_qualified_synthetic_bytes,
    write_qualified_synthetic_corpus,
)

ROOT = Path(__file__).resolve().parents[1]


def test_qualified_synthetic_grid_is_exact_balanced_and_deterministic() -> None:
    first = build_qualified_synthetic_examples()
    second = build_qualified_synthetic_examples()

    assert first == second
    assert len(first) == QUALIFIED_SYNTHETIC_RECORD_COUNT == 1_120
    assert len({example.id for example in first}) == 1_120
    assert Counter(example.expected.category for example in first) == {
        label: 160 for label in CATEGORY_LABELS
    }
    assert Counter(example.expected.priority for example in first) == {
        label: 280 for label in PRIORITY_LABELS
    }
    assert Counter(example.expected.sentiment for example in first) == {
        label: 224 for label in SENTIMENT_LABELS
    }
    assert Counter(
        (
            example.expected.category,
            example.expected.priority,
            example.expected.sentiment,
        )
        for example in first
    ) == {
        (category, priority, sentiment): 8
        for category in CATEGORY_LABELS
        for priority in PRIORITY_LABELS
        for sentiment in SENTIMENT_LABELS
    }
    assert all(example.provenance.source == "synthetic" for example in first)
    assert all(example.provenance.license == "CC0-1.0" for example in first)
    assert all(example.provenance.contains_real_customer_data is False for example in first)
    assert all(example.pii_placeholders for example in first)

    validate_examples(first)


def test_qualified_synthetic_bytes_are_canonical_and_round_trip(tmp_path: Path) -> None:
    first = canonical_qualified_synthetic_bytes()
    second = canonical_qualified_synthetic_bytes()

    assert first == second
    assert first.endswith(b"\n")
    assert len(first.splitlines()) == QUALIFIED_SYNTHETIC_RECORD_COUNT
    for encoded_line in first.splitlines():
        decoded = json.loads(encoded_line)
        assert encoded_line.decode("utf-8") == json.dumps(
            decoded,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    destination = tmp_path / "round-trip.jsonl"
    destination.write_bytes(first)
    loaded = load_examples(destination)
    assert tuple(loaded) == build_qualified_synthetic_examples()


def test_writer_creates_exact_bytes_once_and_never_overwrites(tmp_path: Path) -> None:
    destination = tmp_path / "qualified.jsonl"
    artifact = write_qualified_synthetic_corpus(destination)
    expected = canonical_qualified_synthetic_bytes()

    assert artifact.path == destination.resolve()
    assert artifact.record_count == QUALIFIED_SYNTHETIC_RECORD_COUNT
    assert artifact.size_bytes == len(expected)
    assert artifact.sha256 == hashlib.sha256(expected).hexdigest()
    assert destination.read_bytes() == expected

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_qualified_synthetic_corpus(destination)
    assert destination.read_bytes() == expected


def test_writer_rejects_destination_or_parent_symlinks(tmp_path: Path) -> None:
    target = tmp_path / "target.jsonl"
    target.write_text("sentinel\n", encoding="utf-8")
    destination_link = tmp_path / "destination-link.jsonl"
    destination_link.symlink_to(target)

    with pytest.raises(ValueError, match="symlink"):
        write_qualified_synthetic_corpus(destination_link)
    assert target.read_text(encoding="utf-8") == "sentinel\n"

    real_directory = tmp_path / "real"
    real_directory.mkdir()
    directory_link = tmp_path / "linked-directory"
    directory_link.symlink_to(real_directory, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        write_qualified_synthetic_corpus(directory_link / "qualified.jsonl")
    assert list(real_directory.iterdir()) == []


def test_import_does_not_load_training_or_network_frameworks() -> None:
    source_root = ROOT / "src"
    script = (
        "import sys; "
        f"sys.path.insert(0, {str(source_root)!r}); "
        "import tickettune.synthetic; "
        "forbidden=('torch','transformers','datasets','peft','trl','httpx','requests'); "
        "loaded=[name for name in forbidden if name in sys.modules]; "
        "raise SystemExit('unexpected imports: '+repr(loaded) if loaded else 0)"
    )
    result = subprocess.run(  # noqa: S603  # nosec B603
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr or result.stdout
