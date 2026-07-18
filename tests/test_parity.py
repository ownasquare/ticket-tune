from __future__ import annotations

import hashlib
import json
import os
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

import tickettune.export as export_module
import tickettune.generation as generation_module
import tickettune.parity as parity_module
from tickettune.config import (
    DataConfig,
    DeploymentConfig,
    EvaluationConfig,
    FineTuneConfig,
    LoraConfig,
    ModelConfig,
    TrainingConfig,
)
from tickettune.data import prepare_dataset
from tickettune.export import ExportValidationError, build_merge_plan, verify_merged_model
from tickettune.generation import (
    GeneratedPrediction,
    GenerationLibraries,
    GenerationOutputError,
    generate_predictions,
    validate_adapter_compatibility,
)
from tickettune.parity import (
    ParityThresholdError,
    ParityValidationError,
    compare_prediction_files,
    verify_live_parity,
)
from tickettune.run_manifest import ArtifactDigest, RunManifest, canonical_json_bytes

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
DATASET_SHA256 = "a" * 64
ADAPTER_CONFIG_SHA256 = "b" * 64
ADAPTER_WEIGHT_SHA256 = "c" * 64
MERGE_PROVENANCE_SHA256 = "d" * 64
MERGED_WEIGHT_SHA256 = "e" * 64
DATASET_SPLIT_SHA256 = "1" * 64
GENERATION_CONFIG_SHA256 = "2" * 64
TEST_SOURCE = Path(__file__).parent / "fixtures" / "tickets.jsonl"


def _expected(
    *, response: str = "I will secure the account and investigate this activity."
) -> dict[str, str]:
    return {
        "category": "security",
        "priority": "urgent",
        "sentiment": "worried",
        "response": response,
        "next_action": "lock_and_review_account",
    }


def _prediction_row(
    identifier: str,
    *,
    side: str,
    response: str = "I will secure the account and investigate this activity.",
    prediction: str | None = None,
    dataset_sha256: str = DATASET_SHA256,
    latency_ms: float = 1.0,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": identifier,
        "expected": _expected(),
        "prediction": prediction or json.dumps(_expected(response=response), sort_keys=True),
        "latency_ms": latency_ms,
        "prompt_tokens": 10,
        "generated_tokens": 20,
        "model_name_or_path": BASE_MODEL,
        "model_revision": MODEL_REVISION,
        "dataset_manifest_sha256": dataset_sha256,
        "dataset_split_sha256": DATASET_SPLIT_SHA256,
        "generation_config_sha256": GENERATION_CONFIG_SHA256,
        "adapter_path": None,
        "adapter_config_sha256": None,
        "adapter_weight_sha256": {},
        "merged_model_path": None,
        "merge_provenance_sha256": None,
        "merged_artifact_sha256": {},
        "merged_adapter_config_sha256": None,
        "merged_adapter_weight_files": [],
        "merged_adapter_weight_sha256": [],
    }
    if side == "adapter":
        payload.update(
            adapter_path="/immutable/adapter",
            adapter_config_sha256=ADAPTER_CONFIG_SHA256,
            adapter_weight_sha256={"adapter_model.safetensors": ADAPTER_WEIGHT_SHA256},
        )
    elif side == "merged":
        payload.update(
            merged_model_path="/immutable/merged",
            merge_provenance_sha256=MERGE_PROVENANCE_SHA256,
            merged_artifact_sha256={"model.safetensors": MERGED_WEIGHT_SHA256},
            merged_adapter_config_sha256=ADAPTER_CONFIG_SHA256,
            merged_adapter_weight_files=["adapter_model.safetensors"],
            merged_adapter_weight_sha256=[ADAPTER_WEIGHT_SHA256],
        )
    else:  # pragma: no cover - test-helper misuse
        raise AssertionError(f"unknown side: {side}")
    return payload


def _write_rows(path: Path, rows: list[dict[str, object]], *, allow_nan: bool = False) -> Path:
    path.write_text(
        "".join(
            json.dumps(row, allow_nan=allow_nan, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    return path


def _merged_model(tmp_path: Path) -> Path:
    merged = tmp_path / "merged"
    merged.mkdir()
    config = merged / "config.json"
    weights = merged / "model.safetensors"
    config.write_text(json.dumps({"model_type": "qwen2"}), encoding="utf-8")
    weights.write_bytes(b"verified-merged-weights")
    artifact_hashes = {
        item.name: hashlib.sha256(item.read_bytes()).hexdigest() for item in (config, weights)
    }
    adapter_config = hashlib.sha256(b"adapter-config").hexdigest()
    adapter_weight = hashlib.sha256(b"adapter-weight").hexdigest()
    provenance = {
        "schema_version": 1,
        "operation": "peft_safe_merge",
        "base_model": BASE_MODEL,
        "model_revision": MODEL_REVISION,
        "adapter_base_model": BASE_MODEL,
        "adapter_revision": MODEL_REVISION,
        "adapter_config_sha256": adapter_config,
        "adapter_weight_files": ["adapter_model.safetensors"],
        "adapter_weight_sha256": [adapter_weight],
        "dtype": "float32",
        "load_in_4bit": False,
        "load_in_8bit": False,
        "safe_merge": True,
        "safe_serialization": True,
        "trust_remote_code": False,
        "artifact_sha256": artifact_hashes,
    }
    (merged / "tickettune-merge-provenance.json").write_text(
        json.dumps(provenance, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return merged


def _adapter(tmp_path: Path, merged: Path) -> Path:
    provenance = json.loads(
        (merged / "tickettune-merge-provenance.json").read_text(encoding="utf-8")
    )
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    config = {
        "base_model_name_or_path": BASE_MODEL,
        "revision": MODEL_REVISION,
        "r": 8,
    }
    config_path = adapter / "adapter_config.json"
    weight_path = adapter / "adapter_model.safetensors"
    config_path.write_bytes(b"adapter-config")
    # The compatibility validator needs JSON, so bind the provenance to its real bytes afterward.
    config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")
    weight_path.write_bytes(b"adapter-weight")
    provenance["adapter_config_sha256"] = hashlib.sha256(config_path.read_bytes()).hexdigest()
    provenance["adapter_weight_sha256"] = [hashlib.sha256(weight_path.read_bytes()).hexdigest()]
    (merged / "tickettune-merge-provenance.json").write_text(
        json.dumps(provenance, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return adapter


def _attach_training_manifest(adapter: Path, config: FineTuneConfig) -> Path:
    config_payload = config.model_dump(mode="json")
    adapter_files = (adapter / "adapter_config.json", adapter / "adapter_model.safetensors")
    artifacts = tuple(
        ArtifactDigest(
            path=f"{adapter.name}/{path.name}",
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            size_bytes=path.stat().st_size,
        )
        for path in adapter_files
    )
    manifest = RunManifest(
        run_id=adapter.parent.name,
        created_at=datetime(2026, 7, 18, tzinfo=UTC),
        status="completed",
        project_name=config.project_name,
        model_name_or_path=config.model.name_or_path,
        model_revision=config.model.revision,
        method=config.lora.method,
        seed=config.seed,
        config_sha256=hashlib.sha256(canonical_json_bytes(config_payload)).hexdigest(),
        config=config_payload,
        dataset_sha256={
            "source": "1" * 64,
            "manifest": "2" * 64,
            "train": "3" * 64,
            "validation": "4" * 64,
            "test": "7" * 64,
            "qualification_review_manifest": "5" * 64,
            "qualification_report": "6" * 64,
        },
        packages={},
        runtime={},
        artifacts=artifacts,
    )
    manifest_path = adapter.parent / "manifest.json"
    manifest_path.write_bytes(canonical_json_bytes(manifest.model_dump(mode="json"), pretty=True))
    return manifest_path


def _config(tmp_path: Path) -> FineTuneConfig:
    return FineTuneConfig(
        model=ModelConfig(
            name_or_path=BASE_MODEL,
            revision=MODEL_REVISION,
            parameters_b=0.5,
            torch_dtype="float32",
            max_sequence_length=512,
        ),
        data=DataConfig(source_path=TEST_SOURCE, processed_dir=tmp_path / "processed"),
        lora=LoraConfig(),
        training=TrainingConfig(output_dir=tmp_path / "training", use_cpu=True),
        evaluation=EvaluationConfig(output_dir=tmp_path / "evaluation"),
        deployment=DeploymentConfig(merged_model_dir=tmp_path / "merged-output"),
    )


def test_compare_prediction_files_requires_exact_routing_and_reports_response_diagnostic(
    tmp_path: Path,
) -> None:
    adapter_path = _write_rows(
        tmp_path / "adapter.jsonl",
        [_prediction_row("A", side="adapter"), _prediction_row("B", side="adapter")],
    )
    merged_path = _write_rows(
        tmp_path / "merged.jsonl",
        [
            _prediction_row("A", side="merged"),
            _prediction_row("B", side="merged", response="A different but valid response text."),
        ],
    )
    output_path = tmp_path / "parity.json"

    first = compare_prediction_files(
        adapter_path,
        merged_path,
        output_path=output_path,
        enforce=True,
    )
    second = compare_prediction_files(
        adapter_path,
        merged_path,
        output_path=output_path,
        enforce=True,
    )

    assert first.report.passed is True
    assert first.report.metrics.routing_match_rate == 1.0
    assert first.report.metrics.response_exact_rate == 0.5
    assert first.report.metrics.raw_prediction_exact_rate == 0.5
    assert first.report.metrics.parsed_object_exact_rate == 0.5
    assert first.report.mismatched_ids == ()
    assert first.report.contract_invalid_ids == ()
    assert first.report.release_blocked_ids == ()
    assert first.report.raw_prediction_mismatched_ids == ("B",)
    assert first.report.parsed_object_mismatched_ids == ("B",)
    assert first.report.response_mismatched_ids == ("B",)
    assert first.report.adapter_predictions.sha256 != first.report.merged_predictions.sha256
    assert first.report.merged_predictions.merge_provenance_sha256 == MERGE_PROVENANCE_SHA256
    assert second.report_sha256 == first.report_sha256
    assert output_path.is_file()
    assert first.report_sha256 == hashlib.sha256(output_path.read_bytes()).hexdigest()

    output_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ParityValidationError, match="different parity report"):
        compare_prediction_files(adapter_path, merged_path, output_path=output_path)


def test_release_parity_requires_and_binds_exact_training_test_hash(
    tmp_path: Path,
) -> None:
    lineage = {
        "training_manifest_sha256": "3" * 64,
        "training_config_sha256": "4" * 64,
        "training_dataset_sha256": {
            "source": "5" * 64,
            "manifest": DATASET_SHA256,
            "train": "6" * 64,
            "validation": "7" * 64,
            "test": DATASET_SPLIT_SHA256,
        },
    }
    adapter_row = _prediction_row("A", side="adapter") | lineage
    merged_row = _prediction_row("A", side="merged") | lineage
    adapter_path = _write_rows(tmp_path / "release-adapter.jsonl", [adapter_row])
    merged_path = _write_rows(tmp_path / "release-merged.jsonl", [merged_row])

    artifacts = compare_prediction_files(
        adapter_path,
        merged_path,
        _require_release_lineage=True,
    )

    assert artifacts.report.training_dataset_sha256["test"] == DATASET_SPLIT_SHA256

    missing_test = dict(lineage["training_dataset_sha256"])
    missing_test.pop("test")
    _write_rows(
        adapter_path,
        [
            _prediction_row("A", side="adapter")
            | lineage
            | {"training_dataset_sha256": missing_test}
        ],
    )
    with pytest.raises(ParityValidationError, match="training lineage is invalid"):
        compare_prediction_files(
            adapter_path,
            merged_path,
            _require_release_lineage=True,
        )

    mismatched_test = dict(lineage["training_dataset_sha256"])
    mismatched_test["test"] = "8" * 64
    _write_rows(
        adapter_path,
        [
            _prediction_row("A", side="adapter")
            | lineage
            | {"training_dataset_sha256": mismatched_test}
        ],
    )
    with pytest.raises(ParityValidationError, match="training lineage is invalid"):
        compare_prediction_files(
            adapter_path,
            merged_path,
            _require_release_lineage=True,
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("merged_adapter_config_sha256", "f" * 64),
        ("merged_adapter_weight_sha256", ["f" * 64]),
        ("dataset_split_sha256", "f" * 64),
        ("generation_config_sha256", "f" * 64),
    ],
)
def test_compare_rejects_adapter_merge_or_generation_contract_mismatch(
    tmp_path: Path,
    field: str,
    replacement: object,
) -> None:
    adapter_path = _write_rows(
        tmp_path / "adapter.jsonl",
        [_prediction_row("A", side="adapter")],
    )
    merged_row = _prediction_row("A", side="merged")
    merged_row[field] = replacement
    merged_path = _write_rows(tmp_path / "merged.jsonl", [merged_row])

    with pytest.raises(ParityValidationError):
        compare_prediction_files(adapter_path, merged_path)


def test_prediction_digest_is_for_the_exact_bytes_that_were_parsed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter_path = _write_rows(
        tmp_path / "adapter.jsonl",
        [_prediction_row("A", side="adapter")],
    )
    original = adapter_path.read_bytes()
    replacement = (
        json.dumps(
            _prediction_row("A", side="adapter", latency_ms=9.0),
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()
    merged_path = _write_rows(
        tmp_path / "merged.jsonl",
        [_prediction_row("A", side="merged")],
    )
    real_open = Path.open
    adapter_opens = 0

    def racing_open(path: Path, *args: object, **kwargs: object) -> Any:
        nonlocal adapter_opens
        if path == adapter_path.resolve():
            adapter_opens += 1
            if adapter_opens == 2:
                descriptor = os.open(path, os.O_WRONLY | os.O_TRUNC)
                try:
                    os.write(descriptor, replacement)
                finally:
                    os.close(descriptor)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", racing_open)
    result = compare_prediction_files(adapter_path, merged_path)

    assert result.report.adapter_predictions.sha256 == hashlib.sha256(original).hexdigest()


def test_merged_verifier_does_not_mix_two_provenance_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    merged = _merged_model(tmp_path)
    provenance_path = merged / "tickettune-merge-provenance.json"
    expected = json.loads(provenance_path.read_text(encoding="utf-8"))
    real_read = export_module._read_json_object
    provenance_reads = 0

    def racing_read(path: Path, *, label: str) -> dict[str, Any]:
        nonlocal provenance_reads
        value = real_read(path, label=label)
        if path == provenance_path:
            provenance_reads += 1
            if provenance_reads > 1:
                value = dict(value)
                value["adapter_config_sha256"] = "f" * 64
        return value

    monkeypatch.setattr(export_module, "_read_json_object", racing_read)
    verified = verify_merged_model(merged)

    assert verified.adapter_config_sha256 == expected["adapter_config_sha256"]


@pytest.mark.parametrize("defect", ["duplicate", "reordered", "provenance", "expected"])
def test_compare_prediction_files_rejects_incomparable_sequences(
    tmp_path: Path,
    defect: str,
) -> None:
    adapter_rows = [_prediction_row("A", side="adapter"), _prediction_row("B", side="adapter")]
    merged_rows = [_prediction_row("A", side="merged"), _prediction_row("B", side="merged")]
    if defect == "duplicate":
        adapter_rows[1]["id"] = "A"
    elif defect == "reordered":
        merged_rows.reverse()
    elif defect == "provenance":
        merged_rows[1]["dataset_manifest_sha256"] = "f" * 64
    else:
        merged_rows[1]["expected"] = _expected(response="A different expected response value.")

    adapter_path = _write_rows(tmp_path / "adapter.jsonl", adapter_rows)
    merged_path = _write_rows(tmp_path / "merged.jsonl", merged_rows)

    with pytest.raises(ParityValidationError):
        compare_prediction_files(adapter_path, merged_path)


def test_compare_prediction_files_rejects_nonfinite_latency(tmp_path: Path) -> None:
    adapter_path = _write_rows(
        tmp_path / "adapter.jsonl",
        [_prediction_row("A", side="adapter", latency_ms=float("nan"))],
        allow_nan=True,
    )
    merged_path = _write_rows(
        tmp_path / "merged.jsonl",
        [_prediction_row("A", side="merged")],
    )

    with pytest.raises(ParityValidationError, match="invalid JSON"):
        compare_prediction_files(adapter_path, merged_path)


def test_malformed_schema_is_a_failed_report_and_enforcement_is_fail_closed(
    tmp_path: Path,
) -> None:
    adapter_path = _write_rows(
        tmp_path / "adapter.jsonl",
        [_prediction_row("A", side="adapter", prediction="not-json")],
    )
    merged_path = _write_rows(
        tmp_path / "merged.jsonl",
        [_prediction_row("A", side="merged")],
    )

    artifacts = compare_prediction_files(adapter_path, merged_path)

    assert artifacts.report.passed is False
    assert artifacts.report.adapter_schema_invalid_ids == ("A",)
    assert artifacts.report.mismatched_ids == ()
    assert artifacts.report.contract_invalid_ids == ("A",)
    assert artifacts.report.release_blocked_ids == ("A",)
    with pytest.raises(ParityThresholdError, match="parity thresholds failed"):
        compare_prediction_files(adapter_path, merged_path, enforce=True)


def test_identical_invalid_objects_are_equal_diagnostics_but_release_blocked(
    tmp_path: Path,
) -> None:
    invalid_object = json.dumps({"category": "security"}, sort_keys=True)
    adapter_path = _write_rows(
        tmp_path / "adapter.jsonl",
        [_prediction_row("A", side="adapter", prediction=invalid_object)],
    )
    merged_path = _write_rows(
        tmp_path / "merged.jsonl",
        [_prediction_row("A", side="merged", prediction=invalid_object)],
    )

    artifacts = compare_prediction_files(adapter_path, merged_path)

    assert artifacts.report.passed is False
    assert artifacts.report.metrics.raw_prediction_exact_rate == 1.0
    assert artifacts.report.metrics.parsed_object_exact_rate == 1.0
    assert artifacts.report.raw_prediction_mismatched_ids == ()
    assert artifacts.report.parsed_object_mismatched_ids == ()
    assert artifacts.report.routing_mismatches == {
        "category": (),
        "priority": (),
        "sentiment": (),
        "next_action": (),
    }
    assert artifacts.report.mismatched_ids == ()
    assert artifacts.report.contract_invalid_ids == ("A",)
    assert artifacts.report.release_blocked_ids == ("A",)


def test_valid_routing_drift_is_an_actual_mismatch_and_fails_release_gate(
    tmp_path: Path,
) -> None:
    adapter_path = _write_rows(
        tmp_path / "adapter.jsonl",
        [_prediction_row("A", side="adapter")],
    )
    merged_prediction = _expected() | {"priority": "high"}
    merged_path = _write_rows(
        tmp_path / "merged.jsonl",
        [
            _prediction_row(
                "A",
                side="merged",
                prediction=json.dumps(merged_prediction, sort_keys=True),
            )
        ],
    )

    artifacts = compare_prediction_files(adapter_path, merged_path)

    assert artifacts.report.passed is False
    assert artifacts.report.routing_mismatches["priority"] == ("A",)
    assert artifacts.report.mismatched_ids == ("A",)
    assert artifacts.report.contract_invalid_ids == ()
    assert artifacts.report.release_blocked_ids == ("A",)
    assert artifacts.report.metrics.raw_prediction_exact_rate == 0.0
    assert artifacts.report.metrics.parsed_object_exact_rate == 0.0


def test_live_parity_rejects_merge_precision_mismatch_before_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    merged = _merged_model(tmp_path)
    adapter = _adapter(tmp_path, merged)
    provenance_path = merged / "tickettune-merge-provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["dtype"] = "float16"
    provenance_path.write_text(
        json.dumps(provenance, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    generation_calls = 0

    def unexpected_generation(*args: object, **kwargs: object) -> None:
        del args, kwargs
        nonlocal generation_calls
        generation_calls += 1

    monkeypatch.setattr(parity_module, "generate_predictions", unexpected_generation)

    with pytest.raises(ParityValidationError, match="precision mismatch"):
        verify_live_parity(
            config,
            adapter,
            merged,
            output_path=tmp_path / "parity" / "report.json",
            allow_unverified_training_manifest=True,
        )

    assert generation_calls == 0


def test_public_merged_model_verifier_requires_safe_merge_lineage_and_exact_bytes(
    tmp_path: Path,
) -> None:
    merged = _merged_model(tmp_path)

    verified = verify_merged_model(
        merged,
        expected_base_model=BASE_MODEL,
        expected_model_revision=MODEL_REVISION,
    )

    assert verified.merged_model == str(merged.resolve())
    assert verified.base_model == BASE_MODEL
    assert verified.model_revision == MODEL_REVISION
    assert verified.merge_dtype == "float32"
    assert verified.safe_merge is True
    assert len(verified.provenance_sha256) == 64
    assert (
        dict(verified.artifact_sha256)["model.safetensors"]
        == hashlib.sha256(b"verified-merged-weights").hexdigest()
    )

    arbitrary = tmp_path / "arbitrary"
    arbitrary.mkdir()
    (arbitrary / "config.json").write_text('{"model_type":"qwen2"}', encoding="utf-8")
    (arbitrary / "model.safetensors").write_bytes(b"arbitrary")
    with pytest.raises(ExportValidationError, match="safe merge"):
        verify_merged_model(arbitrary)

    provenance_path = merged / "tickettune-merge-provenance.json"
    unsupported_dtype = json.loads(provenance_path.read_text(encoding="utf-8"))
    unsupported_dtype["dtype"] = "float64"
    provenance_path.write_text(
        json.dumps(unsupported_dtype, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ExportValidationError, match="dtype"):
        verify_merged_model(merged)
    unsupported_dtype["dtype"] = "float32"
    provenance_path.write_text(
        json.dumps(unsupported_dtype, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    historical_v1 = json.loads(provenance_path.read_text(encoding="utf-8"))
    historical_v1.pop("adapter_weight_files")
    provenance_path.write_text(
        json.dumps(historical_v1, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    historical_verified = verify_merged_model(merged)
    assert historical_verified.adapter_weight_files == ()
    assert historical_verified.adapter_weight_sha256 == tuple(
        historical_v1["adapter_weight_sha256"]
    )

    (merged / "model.safetensors").write_bytes(b"tampered")
    with pytest.raises(ExportValidationError, match="hash mismatch"):
        verify_merged_model(merged)


def test_adapter_training_manifest_lineage_is_verified_and_carried_into_merge_plan(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    merged = _merged_model(tmp_path)
    adapter = _adapter(tmp_path, merged)
    manifest_path = _attach_training_manifest(adapter, config)

    provenance = validate_adapter_compatibility(
        adapter,
        model_name_or_path=config.model.name_or_path,
        model_revision=config.model.revision,
        require_training_manifest=True,
    )
    plan = build_merge_plan(
        config.model.name_or_path,
        adapter,
        tmp_path / "new-merged",
        model_revision=config.model.revision,
    )

    assert provenance.training_manifest_path == manifest_path.resolve()
    assert (
        provenance.training_manifest_sha256
        == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    )
    assert provenance.training_config_sha256 == plan.training_config_sha256
    assert dict(plan.training_dataset_sha256)["source"] == "1" * 64
    assert dict(plan.qualification_sha256) == {
        "qualification_report": "6" * 64,
        "qualification_review_manifest": "5" * 64,
    }


def test_generation_loads_verified_merged_directory_without_reapplying_peft(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    prepare_dataset(
        config.data.source_path,
        config.data.processed_dir,
        seed=config.seed,
        splits=config.data.splits,
    )
    merged = _merged_model(tmp_path)
    state: dict[str, Any] = {}

    class FakeTensor:
        shape = (1, 3)

        def to(self, device: str) -> FakeTensor:
            return self

    class FakeTokenizer:
        eos_token_id = 2
        pad_token_id = 2
        eos_token = "</s>"
        pad_token = "</s>"

        def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
            return "prompt"

        def __call__(self, text: str, **kwargs: Any) -> dict[str, FakeTensor]:
            return {"input_ids": FakeTensor()}

        def decode(self, tokens: list[int], **kwargs: Any) -> str:
            return json.dumps(_expected())

    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, model: str, **kwargs: Any) -> FakeTokenizer:
            state["tokenizer_load"] = (model, kwargs)
            return FakeTokenizer()

    class FakeModel:
        def to(self, device: str) -> None:
            return None

        def eval(self) -> None:
            return None

        def generate(self, **kwargs: Any) -> list[list[int]]:
            return [[1, 2, 3, 9, 10]]

    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, model: str, **kwargs: Any) -> FakeModel:
            state["model_load"] = (model, kwargs)
            backup = merged.with_name("verified-original-merged")
            merged.rename(backup)
            merged.mkdir()
            (merged / "config.json").write_text('{"model_type":"qwen2"}', encoding="utf-8")
            (merged / "model.safetensors").write_bytes(b"attacker")
            try:
                state["merged_snapshot_weight"] = (Path(model) / "model.safetensors").read_bytes()
            finally:
                for artifact in merged.iterdir():
                    artifact.unlink()
                merged.rmdir()
                backup.rename(merged)
            return FakeModel()

    class FakePeftModel:
        @classmethod
        def from_pretrained(cls, *args: Any, **kwargs: Any) -> None:
            raise AssertionError("PEFT must not be applied to a verified merged model")

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class FakeTorch:
        cuda = FakeCuda()
        backends = type("Backends", (), {"mps": FakeCuda()})()
        bfloat16 = "torch.bfloat16"
        float16 = "torch.float16"
        float32 = "torch.float32"

        @staticmethod
        def manual_seed(seed: int) -> None:
            return None

        @staticmethod
        def inference_mode() -> nullcontext[None]:
            return nullcontext()

    predictions = generate_predictions(
        config,
        merged_model_path=merged,
        _libraries=GenerationLibraries(
            torch=FakeTorch,
            AutoModelForCausalLM=FakeAutoModel,
            AutoTokenizer=FakeAutoTokenizer,
            PeftModel=FakePeftModel,
        ),
    )

    loaded_model_path = Path(state["model_load"][0])
    assert loaded_model_path != merged.resolve()
    assert state["merged_snapshot_weight"] == b"verified-merged-weights"
    assert not loaded_model_path.exists()
    assert state["model_load"][1]["local_files_only"] is True
    assert "revision" not in state["model_load"][1]
    assert predictions[0].model_name_or_path == BASE_MODEL
    assert predictions[0].model_revision == MODEL_REVISION
    assert predictions[0].merged_model_path == str(merged.resolve())
    assert predictions[0].merge_provenance_sha256 == verify_merged_model(merged).provenance_sha256
    assert predictions[0].dataset_split_sha256 is not None
    assert predictions[0].generation_config_sha256 is not None
    assert (
        predictions[0].merged_adapter_config_sha256
        == verify_merged_model(merged).adapter_config_sha256
    )
    assert predictions[0].merged_adapter_weight_files == ("adapter_model.safetensors",)
    assert predictions[0].adapter_path is None


def test_live_parity_orchestrates_adapter_then_verified_merged_and_writes_immutable_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    merged = _merged_model(tmp_path)
    adapter = _adapter(tmp_path, merged)
    calls: list[str] = []

    def fake_generate(
        config_arg: FineTuneConfig,
        *,
        dataset_path: Path | None = None,
        adapter_path: Path | None = None,
        merged_model_path: Path | None = None,
        output_path: Path | None = None,
        allow_download: bool = False,
        require_training_manifest: bool = False,
        _libraries: GenerationLibraries | None = None,
    ) -> tuple[GeneratedPrediction, ...]:
        del dataset_path, allow_download, require_training_manifest, _libraries
        assert config_arg is config
        assert output_path is not None
        side = "adapter" if adapter_path is not None else "merged"
        calls.append(side)
        verified = verify_merged_model(merged) if merged_model_path is not None else None
        row = GeneratedPrediction.model_validate(
            _prediction_row("A", side=side)
            | (
                {
                    "adapter_path": str(adapter.resolve()),
                    "adapter_config_sha256": hashlib.sha256(
                        (adapter / "adapter_config.json").read_bytes()
                    ).hexdigest(),
                    "adapter_weight_sha256": {
                        "adapter_model.safetensors": hashlib.sha256(
                            (adapter / "adapter_model.safetensors").read_bytes()
                        ).hexdigest()
                    },
                }
                if side == "adapter"
                else {
                    "merged_model_path": verified.merged_model,
                    "merge_provenance_sha256": verified.provenance_sha256,
                    "merged_artifact_sha256": dict(verified.artifact_sha256),
                    "merged_adapter_config_sha256": verified.adapter_config_sha256,
                    "merged_adapter_weight_files": list(verified.adapter_weight_files),
                    "merged_adapter_weight_sha256": list(verified.adapter_weight_sha256),
                }
            )
        )
        output_path.write_bytes(canonical_json_bytes(row.model_dump(mode="json")) + b"\n")
        return (row,)

    monkeypatch.setattr(parity_module, "generate_predictions", fake_generate)
    output_path = tmp_path / "parity-run" / "parity-report.json"

    artifacts = verify_live_parity(
        config,
        adapter,
        merged,
        output_path=output_path,
        enforce=True,
        allow_unverified_training_manifest=True,
    )

    assert calls == ["adapter", "merged"]
    assert artifacts.report.passed is True
    assert artifacts.report.merge_dtype == "float32"
    assert Path(artifacts.report.adapter_predictions.path).is_file()
    assert Path(artifacts.report.merged_predictions.path).is_file()
    assert output_path.is_file()


def test_live_parity_release_path_requires_sibling_training_manifest(tmp_path: Path) -> None:
    config = _config(tmp_path)
    merged = _merged_model(tmp_path)
    adapter = _adapter(tmp_path, merged)

    with pytest.raises(ParityValidationError, match="training manifest"):
        verify_live_parity(
            config,
            adapter,
            merged,
            output_path=tmp_path / "parity" / "report.json",
        )


def test_prediction_writer_reuses_semantically_identical_telemetry_only_retry(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "predictions.jsonl"
    first = GeneratedPrediction.model_validate(_prediction_row("A", side="adapter", latency_ms=1.0))
    retried = GeneratedPrediction.model_validate(
        _prediction_row("A", side="adapter", latency_ms=99.0)
    )

    generation_module._write_predictions(destination, (first,))
    original = destination.read_bytes()
    generation_module._write_predictions(destination, (retried,))

    assert destination.read_bytes() == original

    changed = GeneratedPrediction.model_validate(
        _prediction_row(
            "A",
            side="adapter",
            prediction=json.dumps(_expected() | {"priority": "high"}),
        )
    )
    with pytest.raises(GenerationOutputError, match="different predictions"):
        generation_module._write_predictions(destination, (changed,))


@pytest.mark.parametrize("mutation_side", ["adapter", "merged"])
def test_live_parity_rechecks_all_model_artifacts_after_each_inference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation_side: str,
) -> None:
    config = _config(tmp_path)
    merged = _merged_model(tmp_path)
    adapter = _adapter(tmp_path, merged)

    def fake_generate(
        config_arg: FineTuneConfig,
        *,
        dataset_path: Path | None = None,
        adapter_path: Path | None = None,
        merged_model_path: Path | None = None,
        output_path: Path | None = None,
        allow_download: bool = False,
        require_training_manifest: bool = False,
        _libraries: GenerationLibraries | None = None,
    ) -> tuple[GeneratedPrediction, ...]:
        del config_arg, dataset_path, allow_download, require_training_manifest, _libraries
        assert output_path is not None
        side = "adapter" if adapter_path is not None else "merged"
        verified = verify_merged_model(merged)
        row = GeneratedPrediction.model_validate(
            _prediction_row("A", side=side)
            | (
                {
                    "adapter_path": str(adapter.resolve()),
                    "adapter_config_sha256": hashlib.sha256(
                        (adapter / "adapter_config.json").read_bytes()
                    ).hexdigest(),
                    "adapter_weight_sha256": {
                        "adapter_model.safetensors": hashlib.sha256(
                            (adapter / "adapter_model.safetensors").read_bytes()
                        ).hexdigest()
                    },
                }
                if side == "adapter"
                else {
                    "merged_model_path": verified.merged_model,
                    "merge_provenance_sha256": verified.provenance_sha256,
                    "merged_artifact_sha256": dict(verified.artifact_sha256),
                    "merged_adapter_config_sha256": verified.adapter_config_sha256,
                    "merged_adapter_weight_files": list(verified.adapter_weight_files),
                    "merged_adapter_weight_sha256": list(verified.adapter_weight_sha256),
                }
            )
        )
        output_path.write_bytes(canonical_json_bytes(row.model_dump(mode="json")) + b"\n")
        if side == mutation_side:
            target = (
                adapter / "adapter_model.safetensors"
                if side == "adapter"
                else merged / "model.safetensors"
            )
            target.write_bytes(b"mutated-during-inference")
        return (row,)

    monkeypatch.setattr(parity_module, "generate_predictions", fake_generate)

    with pytest.raises(ParityValidationError, match="changed"):
        verify_live_parity(
            config,
            adapter,
            merged,
            output_path=tmp_path / "parity" / "report.json",
            allow_unverified_training_manifest=True,
        )


@pytest.mark.parametrize("symlink_kind", ["final", "parent"])
def test_parity_report_rejects_unresolved_symlink_outputs(
    tmp_path: Path,
    symlink_kind: str,
) -> None:
    adapter_path = _write_rows(
        tmp_path / "adapter.jsonl",
        [_prediction_row("A", side="adapter")],
    )
    merged_path = _write_rows(
        tmp_path / "merged.jsonl",
        [_prediction_row("A", side="merged")],
    )
    if symlink_kind == "final":
        output_path = tmp_path / "report-link.json"
        output_path.symlink_to(tmp_path / "escaped-report.json")
    else:
        actual_parent = tmp_path / "actual-parent"
        actual_parent.mkdir()
        linked_parent = tmp_path / "linked-parent"
        linked_parent.symlink_to(actual_parent, target_is_directory=True)
        output_path = linked_parent / "report.json"

    with pytest.raises(ParityValidationError, match="symbolic link"):
        compare_prediction_files(adapter_path, merged_path, output_path=output_path)
