from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, cast

import pytest
from pydantic import ValidationError

import tickettune.training as training_module
from tickettune.config import (
    DataConfig,
    DataQualificationConfig,
    DeploymentConfig,
    EvaluationConfig,
    FineTuneConfig,
    GenerationConfig,
    LoraConfig,
    ModelConfig,
    TrainingConfig,
)
from tickettune.data import DatasetIntegrityError, prepare_dataset, sha256_file
from tickettune.qualification import DatasetQualificationError, qualify_dataset
from tickettune.review_packets import (
    DatasetReviewManifestV12,
    EvidenceFileReference,
    RecordReviewDecision,
    ReviewerPacket,
    build_holdout_freeze,
    canonical_evidence_bytes,
)
from tickettune.run_manifest import (
    artifact_digest,
    build_run_manifest,
    canonical_json_bytes,
    json_safe,
    sanitize_error,
    source_control_metadata,
    write_manifest,
)
from tickettune.training import (
    QWEN_EOS_TOKEN,
    ResumeCompatibilityError,
    TrainingLibraries,
    build_training_plan,
    resolve_target_modules,
    run_training,
)

TEST_SOURCE = Path(__file__).parent / "fixtures" / "tickets.jsonl"
TEST_MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"


def _config(tmp_path: Path, *, method: str = "qlora") -> FineTuneConfig:
    processed = tmp_path / "processed"
    prepare_dataset(TEST_SOURCE, processed, seed=7)
    return FineTuneConfig(
        project_name="ticket-test",
        seed=7,
        model=ModelConfig(
            name_or_path="Qwen/Qwen2.5-0.5B-Instruct",
            revision=TEST_MODEL_REVISION,
            parameters_b=0.5,
            torch_dtype="bfloat16",
            max_sequence_length=512,
        ),
        data=DataConfig(source_path=TEST_SOURCE, processed_dir=processed),
        lora=LoraConfig(method=method),
        training=TrainingConfig(
            output_dir=tmp_path / "training",
            max_steps=1,
            bf16=True,
            report_to=[],
        ),
        generation=GenerationConfig(),
        evaluation=EvaluationConfig(output_dir=tmp_path / "evaluation"),
        deployment=DeploymentConfig(merged_model_dir=tmp_path / "merged"),
    )


def _qualified_config(
    tmp_path: Path,
    *,
    approval_status: str = "approved",
    declared_source_sha256: str | None = None,
) -> tuple[FineTuneConfig, Path]:
    source = tmp_path / "qualified.jsonl"
    template = json.loads(TEST_SOURCE.read_text(encoding="utf-8").splitlines()[0])
    with source.open("w", encoding="utf-8") as handle:
        for index in range(1_000):
            record = json.loads(json.dumps(template))
            record["id"] = f"QUAL-{index:05d}"
            record["messages"][1]["content"] = (
                f"Synthetic case {index:05d}: " + record["messages"][1]["content"]
            )
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")

    processed = tmp_path / "qualified-processed"
    prepared = prepare_dataset(source, processed, seed=7)
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    prepared_sha256 = prepared.manifest_sha256
    evidence_dir = tmp_path / "review-evidence"
    evidence_dir.mkdir()
    freeze = build_holdout_freeze(
        source_sha256=source_sha256,
        prepared_manifest_sha256=prepared_sha256,
        held_out_ids=tuple(prepared.split_ids["test"]),
    )
    freeze_path = evidence_dir / "holdout-freeze.json"
    freeze_path.write_bytes(canonical_evidence_bytes(freeze))
    freeze_sha256 = hashlib.sha256(freeze_path.read_bytes()).hexdigest()
    source_ids = tuple(
        json.loads(line)["id"]
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    decisions = tuple(
        RecordReviewDecision(
            record_id=record_id,
            labels="approved",
            response="approved",
            pii="approved",
            license="approved",
        )
        for record_id in source_ids
    )
    packet_models = (
        ReviewerPacket(
            reviewer_id="human-reviewer-01",
            reviewer_kind="human",
            status="approved",
            source_sha256=source_sha256,
            prepared_manifest_sha256=prepared_sha256,
            holdout_freeze_sha256=freeze_sha256,
            review_date="2026-07-18",
            decisions=decisions,
        ),
        ReviewerPacket(
            reviewer_id="human-reviewer-02",
            reviewer_kind="human",
            status="approved",
            source_sha256=source_sha256,
            prepared_manifest_sha256=prepared_sha256,
            holdout_freeze_sha256=freeze_sha256,
            review_date="2026-07-18",
            decisions=decisions,
        ),
    )
    packet_paths = (evidence_dir / "reviewer-a.json", evidence_dir / "reviewer-b.json")
    for path, packet in zip(packet_paths, packet_models, strict=True):
        path.write_bytes(canonical_evidence_bytes(packet))
    review_manifest = tmp_path / "review-manifest.json"
    aggregate = DatasetReviewManifestV12(
        source_sha256=declared_source_sha256 or source_sha256,
        record_count=1_000,
        review_date="2026-07-18",
        intended_domain="synthetic customer-support triage benchmark",
        consent_or_license_statement=("CC0-1.0 synthetic records with no real customer data."),
        pii_decision="no_real_customer_data",
        isolated_test_set_statement=(
            "The held-out examples were isolated before model training and tuning."
        ),
        prepared_manifest=EvidenceFileReference(
            path=prepared.manifest_path.relative_to(tmp_path).as_posix(),
            sha256=prepared_sha256,
        ),
        holdout_freeze=EvidenceFileReference(
            path=freeze_path.relative_to(tmp_path).as_posix(),
            sha256=freeze_sha256,
        ),
        reviewer_packets=(
            EvidenceFileReference(
                path=packet_paths[0].relative_to(tmp_path).as_posix(),
                sha256=hashlib.sha256(packet_paths[0].read_bytes()).hexdigest(),
            ),
            EvidenceFileReference(
                path=packet_paths[1].relative_to(tmp_path).as_posix(),
                sha256=hashlib.sha256(packet_paths[1].read_bytes()).hexdigest(),
            ),
        ),
        approval_status=cast(
            Literal["draft", "approved", "rejected"],
            approval_status,
        ),
    )
    review_manifest.write_bytes(canonical_evidence_bytes(aggregate))
    config = _config(tmp_path).model_copy(
        update={
            "project_name": "ticket-quality-test",
            "data": DataConfig(
                source_path=source,
                processed_dir=processed,
                qualification=DataQualificationConfig(
                    required=True,
                    review_manifest=review_manifest,
                ),
            ),
        }
    )
    return config, review_manifest


def _fake_libraries() -> tuple[TrainingLibraries, dict[str, Any]]:
    state: dict[str, Any] = {
        "tokenizer_loads": [],
        "model_loads": [],
        "dataset_loads": [],
        "quantization": [],
        "lora": [],
        "sft_config": [],
        "trainer": [],
        "resume": [],
    }

    class FakeTorch:
        bfloat16 = "torch.bfloat16"
        float16 = "torch.float16"
        float32 = "torch.float32"

    class FakeDataset:
        @classmethod
        def from_list(cls, rows: list[dict[str, Any]]) -> dict[str, object]:
            state["dataset_loads"].append(rows)
            return {"rows": rows}

    class FakeTokenizer:
        eos_token: str | None = "</s>"
        pad_token: str | None = None
        padding_side = "left"

        def apply_chat_template(
            self,
            messages: list[dict[str, str]],
            *,
            tokenize: bool,
            add_generation_prompt: bool,
        ) -> list[int]:
            assert tokenize is True
            token_count = sum(len(message["content"].split()) + 3 for message in messages)
            if add_generation_prompt:
                token_count += 3
            return list(range(token_count))

        def save_pretrained(self, path: str) -> None:
            destination = Path(path)
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")

    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, model: str, **kwargs: Any) -> FakeTokenizer:
            state["tokenizer_loads"].append((model, kwargs))
            tokenizer = FakeTokenizer()
            state["tokenizer"] = tokenizer
            return tokenizer

    class FakeAutoModelForCausalLM:
        @classmethod
        def from_pretrained(cls, model: str, **kwargs: Any) -> object:
            loaded = SimpleNamespace(model_id=model, kwargs=kwargs)
            state["model_loads"].append(loaded)
            state["model"] = loaded
            return loaded

    class FakeBitsAndBytesConfig:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            state["quantization"].append(self)

    class FakeLoraConfig:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            state["lora"].append(self)

    class FakeSFTConfig:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            state["sft_config"].append(self)

    def fake_prepare_model_for_kbit_training(model: object, **kwargs: Any) -> object:
        state["kbit_preparations"] = [(model, kwargs)]
        return model

    class FakeTrainer:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            state["trainer"].append(self)

        def train(self, *, resume_from_checkpoint: str | None) -> SimpleNamespace:
            state["resume"].append(resume_from_checkpoint)
            return SimpleNamespace(
                global_step=1,
                metrics={"train_loss": 0.125, "train_steps": 1},
            )

        def save_model(self, path: str) -> None:
            destination = Path(path)
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "adapter_model.safetensors").write_bytes(b"fake-adapter")
            (destination / "adapter_config.json").write_text("{}\n", encoding="utf-8")

        def save_metrics(self, split: str, metrics: dict[str, object]) -> None:
            state["saved_metrics"] = (split, metrics)

        def save_state(self) -> None:
            state["saved_state"] = True

    return (
        TrainingLibraries(
            torch=FakeTorch,
            Dataset=FakeDataset,
            AutoModelForCausalLM=FakeAutoModelForCausalLM,
            AutoTokenizer=FakeAutoTokenizer,
            BitsAndBytesConfig=FakeBitsAndBytesConfig,
            PeftLoraConfig=FakeLoraConfig,
            prepare_model_for_kbit_training=fake_prepare_model_for_kbit_training,
            SFTConfig=FakeSFTConfig,
            SFTTrainer=FakeTrainer,
        ),
        state,
    )


def test_training_plan_is_offline_and_uses_current_argument_names(tmp_path: Path) -> None:
    config = _config(tmp_path)

    plan = build_training_plan(config)

    assert plan.model_name_or_path == "Qwen/Qwen2.5-0.5B-Instruct"
    assert plan.allow_download is False
    assert plan.dataset_manifest_status == "verified"
    assert plan.dataset_manifest_sha256 is not None
    assert plan.dataset_manifest_error is None
    assert plan.qualification.status == "not_required"
    assert plan.qualification.schema_version is None
    assert plan.qualification.dataset_tier == "not_required"
    assert plan.qualification.source_sha256 is None
    assert plan.qualification.review_manifest_sha256 is None
    assert plan.qualification.report_sha256 is None
    assert plan.qualification.prepared_manifest_sha256 is None
    assert plan.qualification.holdout_freeze_sha256 is None
    assert plan.qualification.reviewer_packet_sha256 == ()
    assert plan.qualification.reviewer_ids == ()
    assert plan.qualification.error is None
    assert plan.dataset_sha256["source"] == sha256_file(TEST_SOURCE)
    assert plan.dataset_sha256["manifest"] == plan.dataset_manifest_sha256
    assert set(plan.dataset_sha256) == {
        "source",
        "manifest",
        "train",
        "validation",
        "test",
    }
    assert plan.target_modules == "all-linear"
    assert plan.completion_only_loss is True
    assert plan.eos_token == QWEN_EOS_TOKEN
    assert plan.sft_arguments["max_length"] == 512
    assert plan.sft_arguments["eval_strategy"] == "steps"
    assert plan.sft_arguments["load_best_model_at_end"] is True
    assert plan.sft_arguments["metric_for_best_model"] == "eval_loss"
    assert plan.sft_arguments["greater_is_better"] is False
    assert plan.sft_arguments["warmup_steps"] == 0
    assert plan.sft_arguments["use_cpu"] is False
    assert "warmup_ratio" not in plan.sft_arguments
    assert "evaluation_strategy" not in plan.sft_arguments
    assert plan.quantization == {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": "bf16",
    }


def test_quality_training_plan_exposes_and_binds_qualification_evidence(
    tmp_path: Path,
) -> None:
    config, review_manifest = _qualified_config(tmp_path)

    plan = build_training_plan(config)

    assert plan.qualification.status == "qualified"
    assert plan.qualification.schema_version == "1.2"
    assert plan.qualification.dataset_tier == "qualification_candidate"
    assert plan.qualification.source_sha256 == sha256_file(config.data.source_path)
    assert plan.qualification.review_manifest_sha256 == sha256_file(review_manifest)
    assert plan.qualification.report_sha256 is not None
    assert plan.qualification.prepared_manifest_sha256 == plan.dataset_manifest_sha256
    assert plan.qualification.holdout_freeze_sha256 is not None
    assert len(plan.qualification.reviewer_packet_sha256) == 2
    assert plan.qualification.reviewer_ids == ("human-reviewer-01", "human-reviewer-02")
    assert plan.qualification.error is None
    assert plan.dataset_sha256["source"] == plan.qualification.source_sha256
    assert (
        plan.dataset_sha256["qualification_review_manifest"]
        == plan.qualification.review_manifest_sha256
    )
    assert plan.dataset_sha256["qualification_report"] == plan.qualification.report_sha256
    assert set(plan.dataset_sha256) == {
        "source",
        "manifest",
        "train",
        "validation",
        "test",
        "qualification_review_manifest",
        "qualification_report",
    }


def test_quality_training_binds_qualification_evidence_into_run_manifest(
    tmp_path: Path,
) -> None:
    config, _review_manifest = _qualified_config(tmp_path)
    libraries, _state = _fake_libraries()

    result = run_training(config, _libraries=libraries)

    manifest = json.loads(Path(result.manifest_path or "").read_text(encoding="utf-8"))
    assert manifest["dataset_sha256"]["source"] == result.plan.qualification.source_sha256
    assert manifest["dataset_sha256"]["qualification_review_manifest"] == (
        result.plan.qualification.review_manifest_sha256
    )
    assert manifest["dataset_sha256"]["qualification_report"] == (
        result.plan.qualification.report_sha256
    )
    assert set(manifest["dataset_sha256"]) == {
        "source",
        "manifest",
        "train",
        "validation",
        "test",
        "qualification_review_manifest",
        "qualification_report",
    }


def test_quality_training_rejects_reviewed_cohort_that_is_not_exact_test_split(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, review_manifest = _qualified_config(tmp_path)
    review_payload = json.loads(review_manifest.read_text(encoding="utf-8"))
    freeze_path = review_manifest.parent / review_payload["holdout_freeze"]["path"]
    freeze_payload = json.loads(freeze_path.read_text(encoding="utf-8"))
    dataset_manifest = json.loads((config.data.processed_dir / "manifest.json").read_text())
    freeze_payload["held_out_ids"][-1] = dataset_manifest["split_ids"]["train"][0]
    freeze_path.write_text(
        json.dumps(freeze_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    loaded = False

    def fail_if_loaded() -> None:
        nonlocal loaded
        loaded = True
        raise AssertionError("optional libraries must not load")

    monkeypatch.setattr("tickettune.training._load_training_libraries", fail_if_loaded)

    plan = build_training_plan(config)

    assert plan.qualification.status == "rejected"
    assert plan.qualification.error is not None
    assert "holdout_freeze_sha256_matches" in plan.qualification.error
    with pytest.raises(DatasetQualificationError, match="holdout_freeze_sha256_matches"):
        run_training(config)
    assert loaded is False
    assert not config.training.output_dir.exists()


def test_quality_plan_rejects_qualification_source_different_from_prepared_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _review_manifest = _qualified_config(tmp_path)
    verification = training_module._verify_training_dataset(config)
    mismatched = verification.model_copy(update={"source_sha256": "f" * 64})
    monkeypatch.setattr(training_module, "_verify_training_dataset", lambda _config: mismatched)

    plan = build_training_plan(config)

    assert plan.qualification.status == "invalid"
    assert plan.qualification.error is not None
    assert "source SHA-256" in plan.qualification.error


def test_quality_plan_rejects_qualification_prepared_manifest_hash_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _review_manifest = _qualified_config(tmp_path)
    verification = training_module._verify_training_dataset(config)
    mismatched = verification.model_copy(update={"manifest_sha256": "f" * 64})
    monkeypatch.setattr(training_module, "_verify_training_dataset", lambda _config: mismatched)

    plan = build_training_plan(config)

    assert plan.qualification.status == "invalid"
    assert plan.qualification.error is not None
    assert "prepared-manifest SHA-256" in plan.qualification.error


def test_quality_plan_rejects_qualification_count_different_from_prepared_test(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _review_manifest = _qualified_config(tmp_path)
    verification = training_module._verify_training_dataset(config)
    split_counts = dict(verification.split_counts)
    split_counts["test"] += 1
    mismatched = verification.model_copy(update={"split_counts": split_counts})
    monkeypatch.setattr(training_module, "_verify_training_dataset", lambda _config: mismatched)

    plan = build_training_plan(config)

    assert plan.qualification.status == "invalid"
    assert plan.qualification.error is not None
    assert "test split count" in plan.qualification.error


def test_quality_training_rejects_missing_review_evidence_before_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, review_manifest = _qualified_config(tmp_path)
    review_manifest.unlink()
    loaded = False

    def fail_if_loaded() -> None:
        nonlocal loaded
        loaded = True
        raise AssertionError("optional libraries must not load")

    monkeypatch.setattr("tickettune.training._load_training_libraries", fail_if_loaded)

    plan = build_training_plan(config)
    assert plan.qualification.status == "invalid"
    assert plan.qualification.error is not None
    assert "review manifest must be a regular" in plan.qualification.error
    with pytest.raises(DatasetQualificationError, match="review manifest must be a regular"):
        run_training(config)
    assert loaded is False
    assert not config.training.output_dir.exists()


@pytest.mark.parametrize(
    ("approval_status", "declared_source_sha256", "failure"),
    [
        ("draft", None, "approved_status"),
        ("approved", "f" * 64, "source_sha256_matches"),
    ],
)
def test_quality_training_rejects_policy_failures_before_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    approval_status: str,
    declared_source_sha256: str | None,
    failure: str,
) -> None:
    config, _review_manifest = _qualified_config(
        tmp_path,
        approval_status=approval_status,
        declared_source_sha256=declared_source_sha256,
    )
    loaded = False

    def fail_if_loaded() -> None:
        nonlocal loaded
        loaded = True
        raise AssertionError("optional libraries must not load")

    monkeypatch.setattr("tickettune.training._load_training_libraries", fail_if_loaded)

    plan = build_training_plan(config)
    assert plan.qualification.status == "rejected"
    assert plan.qualification.error is not None
    assert failure in plan.qualification.error
    with pytest.raises(DatasetQualificationError, match=failure):
        run_training(config)
    assert loaded is False
    assert not config.training.output_dir.exists()


def test_quality_training_rejects_review_evidence_changed_after_planning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, review_manifest = _qualified_config(tmp_path)
    original_qualify = qualify_dataset
    calls = 0

    def qualify_then_change(*args: Any, **kwargs: Any) -> object:
        nonlocal calls
        report = original_qualify(*args, **kwargs)
        calls += 1
        if calls == 1:
            payload = json.loads(review_manifest.read_text(encoding="utf-8"))
            payload["intended_domain"] += " with a changed attestation"
            review_manifest.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        return report

    loaded = False

    def fail_if_loaded() -> None:
        nonlocal loaded
        loaded = True
        raise AssertionError("optional libraries must not load")

    monkeypatch.setattr(training_module, "qualify_dataset", qualify_then_change)
    monkeypatch.setattr(training_module, "_load_training_libraries", fail_if_loaded)

    with pytest.raises(DatasetQualificationError, match="evidence changed after plan creation"):
        run_training(config)
    assert calls == 2
    assert loaded is False
    assert not config.training.output_dir.exists()


def test_run_training_loads_exact_model_before_constructing_sft_trainer(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    libraries, state = _fake_libraries()

    result = run_training(config, _libraries=libraries)

    assert result.executed is True
    assert result.run_id is not None
    assert Path(result.run_dir or "").name == result.run_id
    assert Path(result.adapter_path or "").is_dir()
    assert Path(result.adapter_path or "").parent == Path(result.run_dir or "")
    assert Path(result.manifest_path or "") == Path(result.run_dir or "") / "manifest.json"
    trainer_kwargs = state["trainer"][0].kwargs
    assert trainer_kwargs["model"] is state["model"]
    assert trainer_kwargs["processing_class"] is state["tokenizer"]
    assert trainer_kwargs["peft_config"] is state["lora"][0]
    assert "quantization_config" not in trainer_kwargs

    model_init_kwargs = state["model"].kwargs
    assert model_init_kwargs["quantization_config"] is state["quantization"][0]
    assert model_init_kwargs["dtype"] == "torch.bfloat16"
    assert model_init_kwargs["device_map"] is None
    assert model_init_kwargs["local_files_only"] is True
    assert model_init_kwargs["revision"] == TEST_MODEL_REVISION
    assert state["kbit_preparations"] == [(state["model"], {"use_gradient_checkpointing": True})]
    sft_kwargs = state["sft_config"][0].kwargs
    assert "model_init_kwargs" not in sft_kwargs
    assert sft_kwargs["max_length"] == 512
    assert sft_kwargs["output_dir"] == str(Path(result.run_dir or "") / "trainer")
    assert sft_kwargs["eval_strategy"] == "steps"
    assert sft_kwargs["load_best_model_at_end"] is True
    assert sft_kwargs["metric_for_best_model"] == "eval_loss"
    assert sft_kwargs["greater_is_better"] is False
    assert sft_kwargs["completion_only_loss"] is True
    assert state["lora"][0].kwargs["target_modules"] == "all-linear"
    assert state["lora"][0].kwargs["base_model_name_or_path"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert state["lora"][0].kwargs["revision"] == TEST_MODEL_REVISION
    assert state["tokenizer"].eos_token == QWEN_EOS_TOKEN
    assert state["tokenizer"].pad_token == QWEN_EOS_TOKEN
    assert result.token_budget is not None
    assert result.token_budget.examples == 14
    assert result.token_budget.max_full_tokens <= 512
    assert result.token_budget.min_completion_tokens > 0
    assert result.metrics["optimizer_steps"] == 1
    assert os.environ["HF_HUB_DISABLE_TELEMETRY"] == "1"

    run_dir = Path(result.run_dir or "")
    snapshot_paths = [run_dir / "dataset" / "train.jsonl", run_dir / "dataset" / "validation.jsonl"]
    expected_rows = [
        [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        for path in snapshot_paths
    ]
    assert state["dataset_loads"] == expected_rows
    for split_name, snapshot_path in zip(("train", "validation"), snapshot_paths, strict=True):
        assert (
            snapshot_path.read_bytes()
            == (config.data.processed_dir / f"{split_name}.jsonl").read_bytes()
        )

    manifest_path = Path(result.manifest_path or "")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "1.1"
    assert manifest["status"] == "completed"
    assert manifest["method"] == "qlora"
    assert manifest["dataset_sha256"].keys() == {
        "source",
        "manifest",
        "train",
        "validation",
        "test",
    }
    assert {item["path"] for item in manifest["artifacts"]} >= {
        "adapter/adapter_config.json",
        "adapter/adapter_model.safetensors",
        "adapter/tokenizer_config.json",
        "dataset/train.jsonl",
        "dataset/validation.jsonl",
    }
    artifact_sha256 = {item["path"]: item["sha256"] for item in manifest["artifacts"]}
    assert artifact_sha256["dataset/train.jsonl"] == manifest["dataset_sha256"]["train"]
    assert artifact_sha256["dataset/validation.jsonl"] == manifest["dataset_sha256"]["validation"]
    assert manifest["training_duration_seconds"] is None
    assert manifest["peak_accelerator_memory_mb"] is None
    assert manifest["metrics"]["token_budget_examples"] == 14
    assert manifest["metrics"]["token_budget_max_full_tokens"] <= 512
    assert manifest["metrics"]["token_budget_min_completion_tokens"] > 0
    assert set(manifest) >= {"git_revision", "git_branch", "git_dirty"}
    pointer_path = Path(result.latest_pointer_path or "")
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    assert pointer["run_id"] == result.run_id
    assert pointer["adapter_path"] == f"runs/{result.run_id}/adapter"
    assert pointer["manifest_path"] == f"runs/{result.run_id}/manifest.json"


def test_run_training_rejects_truncation_before_loading_model(tmp_path: Path) -> None:
    config = _config(tmp_path)
    libraries, state = _fake_libraries()
    tokenizer = libraries.AutoTokenizer.from_pretrained("test")

    def oversized_chat_template(
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> list[int]:
        del tokenize
        count = 513 if len(messages) == 3 and not add_generation_prompt else 20
        return list(range(count))

    tokenizer.apply_chat_template = oversized_chat_template

    class ReusedTokenizer:
        @classmethod
        def from_pretrained(cls, *args: object, **kwargs: object) -> object:
            del args, kwargs
            return tokenizer

    libraries = replace(libraries, AutoTokenizer=ReusedTokenizer)

    with pytest.raises(ValueError, match="would truncate"):
        run_training(config, _libraries=libraries)

    assert state["model_loads"] == []
    assert state["trainer"] == []


def test_run_training_rejects_snapshot_mutation_during_framework_load(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    libraries, state = _fake_libraries()
    loaded_paths: list[Path] = []

    class TamperingDataset:
        @classmethod
        def from_list(cls, rows: list[dict[str, Any]]) -> dict[str, object]:
            snapshot_path = next(config.training.output_dir.glob("runs/*/dataset/train.jsonl"))
            loaded_paths.append(snapshot_path)
            snapshot_path.chmod(0o600)
            snapshot_path.write_bytes(snapshot_path.read_bytes() + b"\n")
            return {"rows": rows}

    libraries = replace(libraries, Dataset=TamperingDataset)

    with pytest.raises(DatasetIntegrityError, match="run-scoped train snapshot"):
        run_training(config, _libraries=libraries)

    assert len(loaded_paths) == 1
    assert loaded_paths[0].name == "train.jsonl"
    assert loaded_paths[0].parent.name == "dataset"
    assert state["model_loads"] == []
    manifests = tuple(config.training.output_dir.glob("runs/*/manifest.json"))
    assert len(manifests) == 1
    failed = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert failed["status"] == "failed"
    assert "run-scoped train snapshot" in failed["error"]


def test_run_training_rejects_snapshot_swap_after_consuming_verified_rows(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    libraries, state = _fake_libraries()
    observed_ids: list[list[str]] = []

    class SwappingDataset:
        @classmethod
        def from_list(cls, rows: list[dict[str, Any]]) -> dict[str, object]:
            split_name = "train" if not observed_ids else "validation"
            snapshot_path = next(
                config.training.output_dir.glob(f"runs/*/dataset/{split_name}.jsonl")
            )
            original_path = snapshot_path.with_suffix(".original")
            snapshot_path.rename(original_path)
            snapshot_path.write_text('{"id":"attacker-controlled"}\n', encoding="utf-8")
            try:
                observed_ids.append([str(row["id"]) for row in rows])
            finally:
                snapshot_path.unlink()
                original_path.rename(snapshot_path)
            return {"rows": rows}

    libraries = replace(libraries, Dataset=SwappingDataset)

    with pytest.raises(DatasetIntegrityError, match="snapshot identity changed"):
        run_training(config, _libraries=libraries)

    expected_ids = []
    expected_ids.append(
        [
            json.loads(line)["id"]
            for line in (config.data.processed_dir / "train.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
    )
    assert observed_ids == expected_ids
    assert state["model_loads"] == []


def test_run_training_rejects_snapshot_mutation_during_artifact_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    libraries, state = _fake_libraries()
    original_run_artifacts = training_module._run_artifacts
    tampered = False

    def inventory_then_tamper(run_dir: Path) -> tuple[object, ...]:
        nonlocal tampered
        artifacts = original_run_artifacts(run_dir)
        if not tampered:
            snapshot_path = run_dir / "dataset" / "validation.jsonl"
            snapshot_path.chmod(0o600)
            snapshot_path.write_bytes(snapshot_path.read_bytes() + b"\n")
            tampered = True
        return artifacts

    monkeypatch.setattr(training_module, "_run_artifacts", inventory_then_tamper)

    with pytest.raises(DatasetIntegrityError, match="run-scoped validation snapshot"):
        run_training(config, _libraries=libraries)

    assert tampered is True
    assert state["model_loads"]
    manifests = tuple(config.training.output_dir.glob("runs/*/manifest.json"))
    assert len(manifests) == 1
    failed = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert failed["status"] == "failed"
    assert "run-scoped validation snapshot" in failed["error"]


def test_cpu_policy_reaches_public_sft_config(tmp_path: Path) -> None:
    config = _config(tmp_path, method="lora")
    config = config.model_copy(
        update={"training": config.training.model_copy(update={"use_cpu": True, "warmup_steps": 2})}
    )
    libraries, state = _fake_libraries()

    run_training(config, _libraries=libraries)

    sft_kwargs = state["sft_config"][0].kwargs
    assert sft_kwargs["use_cpu"] is True
    assert sft_kwargs["warmup_steps"] == 2
    assert "warmup_ratio" not in sft_kwargs


def test_lora_omits_quantization_but_keeps_all_linear_targets(tmp_path: Path) -> None:
    config = _config(tmp_path, method="lora")
    libraries, state = _fake_libraries()

    run_training(config, _libraries=libraries)

    model_init_kwargs = state["model"].kwargs
    assert "quantization_config" not in model_init_kwargs
    assert state["quantization"] == []
    assert "kbit_preparations" not in state
    assert state["lora"][0].kwargs["target_modules"] == resolve_target_modules("llama")


def test_explicit_target_modules_are_preserved(tmp_path: Path) -> None:
    config = _config(tmp_path, method="lora")
    config = config.model_copy(
        update={"lora": config.lora.model_copy(update={"target_modules": ["q_proj", "v_proj"]})}
    )
    libraries, state = _fake_libraries()

    plan = build_training_plan(config)
    run_training(config, _libraries=libraries)

    assert plan.target_modules == ("q_proj", "v_proj")
    assert state["lora"][0].kwargs["target_modules"] == ["q_proj", "v_proj"]


def test_dry_run_never_loads_optional_libraries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)

    def fail_if_loaded() -> None:
        raise AssertionError("heavy libraries must remain lazy")

    monkeypatch.setattr("tickettune.training._load_training_libraries", fail_if_loaded)

    result = run_training(config, dry_run=True)

    assert result.executed is False
    assert result.manifest_path is None
    assert result.proof_boundary.startswith("plan_only")


def test_dry_run_reports_invalid_manifest_without_loading_optional_libraries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    (config.data.processed_dir / "train.jsonl").write_bytes(b"tampered\n")

    def fail_if_loaded() -> None:
        raise AssertionError("heavy libraries must remain lazy")

    monkeypatch.setattr("tickettune.training._load_training_libraries", fail_if_loaded)

    result = run_training(config, dry_run=True)

    assert result.executed is False
    assert result.plan.dataset_manifest_status == "invalid"
    assert result.plan.dataset_manifest_sha256 is None
    assert result.plan.dataset_manifest_error is not None
    assert "SHA-256 mismatch" in result.plan.dataset_manifest_error


def test_real_training_rejects_missing_manifest_before_optional_imports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    (config.data.processed_dir / "manifest.json").unlink()
    loaded = False

    def fail_if_loaded() -> None:
        nonlocal loaded
        loaded = True
        raise AssertionError("optional libraries must not load")

    monkeypatch.setattr("tickettune.training._load_training_libraries", fail_if_loaded)

    with pytest.raises(DatasetIntegrityError, match="manifest not found"):
        run_training(config)
    assert loaded is False
    assert not config.training.output_dir.exists()


def test_real_training_rejects_manifest_mismatch_before_optional_imports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    manifest_path = config.data.processed_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["seed"] = config.seed + 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    loaded = False

    def fail_if_loaded() -> None:
        nonlocal loaded
        loaded = True
        raise AssertionError("optional libraries must not load")

    monkeypatch.setattr("tickettune.training._load_training_libraries", fail_if_loaded)

    with pytest.raises(DatasetIntegrityError, match="seed mismatch"):
        run_training(config)
    assert loaded is False
    assert not config.training.output_dir.exists()


def test_dependency_failure_still_writes_immutable_failed_run_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)

    def fail_to_load() -> None:
        raise RuntimeError("optional stack unavailable")

    monkeypatch.setattr("tickettune.training._load_training_libraries", fail_to_load)

    with pytest.raises(RuntimeError, match="optional stack unavailable"):
        run_training(config)

    manifests = list((config.training.output_dir / "runs").glob("*/manifest.json"))
    assert len(manifests) == 1
    payload = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["error"] == "RuntimeError: optional stack unavailable"


def test_repeated_training_uses_distinct_immutable_run_directories(tmp_path: Path) -> None:
    config = _config(tmp_path, method="lora")
    first_libraries, _ = _fake_libraries()
    second_libraries, _ = _fake_libraries()

    first = run_training(config, _libraries=first_libraries)
    first_adapter = Path(first.adapter_path or "") / "adapter_model.safetensors"
    first_manifest = Path(first.manifest_path or "")
    first_adapter_bytes = first_adapter.read_bytes()
    first_manifest_bytes = first_manifest.read_bytes()

    second = run_training(config, _libraries=second_libraries)

    assert first.run_id != second.run_id
    assert first.run_dir != second.run_dir
    assert first.adapter_path != second.adapter_path
    assert first_adapter.read_bytes() == first_adapter_bytes
    assert first_manifest.read_bytes() == first_manifest_bytes
    pointer = json.loads(Path(second.latest_pointer_path or "").read_text(encoding="utf-8"))
    assert pointer["run_id"] == second.run_id
    assert len(list((config.training.output_dir / "runs").glob("*/manifest.json"))) == 2


def _write_resume_run(config: FineTuneConfig, *, tamper: bool = False) -> Path:
    plan = build_training_plan(config)
    created_at = datetime(2026, 7, 17, 20, 0, tzinfo=UTC)
    provisional = build_run_manifest(
        config=json_safe(config),
        status="completed",
        project_name=config.project_name,
        model_name_or_path=config.model.name_or_path,
        model_revision=config.model.revision,
        method=config.lora.method,
        seed=config.seed,
        dataset_sha256=plan.dataset_sha256,
        created_at=created_at,
        versions={},
    )
    run_dir = config.training.output_dir / "runs" / provisional.run_id
    checkpoint = run_dir / "trainer" / "checkpoint-1"
    checkpoint.mkdir(parents=True)
    trainer_state = checkpoint / "trainer_state.json"
    trainer_state.write_text('{"global_step":1}\n', encoding="utf-8")
    (checkpoint / "adapter_model.safetensors").write_bytes(b"checkpoint-adapter")
    artifacts = tuple(
        artifact_digest(path, relative_to=run_dir) for path in sorted(checkpoint.iterdir())
    )
    manifest = build_run_manifest(
        config=json_safe(config),
        status="completed",
        project_name=config.project_name,
        model_name_or_path=config.model.name_or_path,
        model_revision=config.model.revision,
        method=config.lora.method,
        seed=config.seed,
        dataset_sha256=plan.dataset_sha256,
        artifacts=artifacts,
        created_at=created_at,
        versions={},
    )
    write_manifest(run_dir / "manifest.json", manifest)
    if tamper:
        trainer_state.write_text('{"global_step":2}\n', encoding="utf-8")
    return checkpoint


def test_resume_requires_matching_manifest_and_checkpoint_hashes(tmp_path: Path) -> None:
    config = _config(tmp_path, method="lora")
    checkpoint = _write_resume_run(config)
    libraries, state = _fake_libraries()

    result = run_training(
        config,
        resume_from_checkpoint=checkpoint,
        _libraries=libraries,
    )

    assert state["resume"] == [str(checkpoint.resolve())]
    assert result.run_dir != str(checkpoint.parent.parent)
    manifest = json.loads(Path(result.manifest_path or "").read_text(encoding="utf-8"))
    assert manifest["resume_from_checkpoint"] == str(checkpoint.resolve())


def test_resume_rejects_tampered_checkpoint_before_optional_imports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, method="lora")
    checkpoint = _write_resume_run(config, tamper=True)

    def fail_if_loaded() -> None:
        raise AssertionError("optional libraries must not load")

    monkeypatch.setattr("tickettune.training._load_training_libraries", fail_if_loaded)

    with pytest.raises(ResumeCompatibilityError, match="artifact hash mismatch"):
        run_training(config, resume_from_checkpoint=checkpoint)


def test_resume_rejects_duplicate_manifest_field_before_optional_imports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, method="lora")
    checkpoint = _write_resume_run(config)
    manifest_path = checkpoint.parent.parent / "manifest.json"
    manifest_path.write_text(
        manifest_path.read_text(encoding="utf-8").replace(
            "{",
            '{"run_id":"shadowed",',
            1,
        ),
        encoding="utf-8",
    )

    def fail_if_loaded() -> None:
        raise AssertionError("optional libraries must not load")

    monkeypatch.setattr("tickettune.training._load_training_libraries", fail_if_loaded)

    with pytest.raises(ResumeCompatibilityError, match="duplicate JSON object key 'run_id'"):
        run_training(config, resume_from_checkpoint=checkpoint)


def test_resume_rejects_untracked_checkpoint_before_optional_imports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, method="lora")
    checkpoint = config.training.output_dir / "runs" / "unknown" / "trainer" / "checkpoint-1"
    checkpoint.mkdir(parents=True)
    (checkpoint / "trainer_state.json").write_text("{}\n", encoding="utf-8")

    def fail_if_loaded() -> None:
        raise AssertionError("optional libraries must not load")

    monkeypatch.setattr("tickettune.training._load_training_libraries", fail_if_loaded)

    with pytest.raises(ResumeCompatibilityError, match="no regular sibling run manifest"):
        run_training(config, resume_from_checkpoint=checkpoint)


def test_run_manifest_is_frozen_and_refuses_changed_overwrite(tmp_path: Path) -> None:
    manifest = build_run_manifest(
        config={"seed": 42},
        status="completed",
        project_name="tickettune",
        model_name_or_path="Qwen/Qwen2.5-0.5B-Instruct",
        model_revision="main",
        method="lora",
        seed=42,
        dataset_sha256={"train": "a" * 64},
        created_at=datetime(2026, 7, 17, tzinfo=UTC),
        versions={},
    )
    path = tmp_path / "run.json"

    write_manifest(path, manifest)
    write_manifest(path, manifest)

    with pytest.raises(ValidationError):
        manifest.seed = 3  # type: ignore[misc]
    changed = manifest.model_copy(update={"metrics": {"loss": 1.0}})
    with pytest.raises(FileExistsError, match="immutable run manifest"):
        write_manifest(path, changed)


def test_canonical_json_rejects_nested_non_finite_metrics() -> None:
    with pytest.raises(ValueError, match="Out of range float values"):
        canonical_json_bytes(
            {"metrics": {"validation": {"losses": [0.25, float("inf")]}}},
            pretty=True,
        )


def test_manifest_error_sanitizer_redacts_tokens() -> None:
    message = "request failed token=hf_abcdefghijklmnopqrstuvwxyz123456 and password=hunter2"

    sanitized = sanitize_error(message)

    assert "hf_" not in sanitized
    assert "hunter2" not in sanitized
    assert sanitized.count("[REDACTED]") == 2


def test_source_control_metadata_is_bounded_and_non_sensitive(tmp_path: Path) -> None:
    metadata = source_control_metadata(tmp_path)

    assert metadata == {
        "git_revision": None,
        "git_branch": None,
        "git_dirty": None,
    }


def test_plan_reports_fp16_and_missing_validation_without_loading_libraries(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    (config.data.processed_dir / "validation.jsonl").unlink()
    config = config.model_copy(
        update={"training": config.training.model_copy(update={"bf16": False, "fp16": True})}
    )

    plan = build_training_plan(config)

    assert plan.precision == "fp16"
    assert plan.quantization is not None
    assert plan.quantization["bnb_4bit_compute_dtype"] == "fp16"
    assert plan.sft_arguments["eval_strategy"] == "no"
    assert "load_best_model_at_end" not in plan.sft_arguments
    assert "metric_for_best_model" not in plan.sft_arguments
    assert "greater_is_better" not in plan.sft_arguments
    assert plan.sft_arguments["max_steps"] == 1
    assert plan.missing_datasets == (str(config.data.processed_dir / "validation.jsonl"),)
    with pytest.raises(FileNotFoundError, match="prepare the dataset"):
        run_training(config)


def test_metric_normalization_and_empty_artifact_inventory(tmp_path: Path) -> None:
    class Scalar:
        def item(self) -> float:
            return 2.5

    metrics = training_module._numeric_metrics(
        {
            "loss": Scalar(),
            "steps": 3,
            "ignored": object(),
        }
    )

    assert metrics == {"loss": 2.5, "steps": 3}
    assert training_module._adapter_artifacts(tmp_path / "missing", tmp_path) == ()
