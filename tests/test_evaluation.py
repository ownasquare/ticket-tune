from __future__ import annotations

import json
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

import tickettune.evaluation as evaluation_module
import tickettune.generation as generation_module
from tickettune.config import (
    DataConfig,
    DeploymentConfig,
    EvaluationConfig,
    EvaluationThresholds,
    FineTuneConfig,
    LoraConfig,
    ModelConfig,
    TrainingConfig,
)
from tickettune.data import (
    DatasetIntegrityError,
    prepare_dataset,
    sha256_file,
    verify_prepared_dataset,
)
from tickettune.evaluation import (
    EvaluationThresholdError,
    evaluate_predictions,
    extract_json_object,
    macro_f1,
    run_model_evaluation,
    score_prediction,
)
from tickettune.generation import (
    AdapterCompatibilityError,
    GeneratedPrediction,
    GenerationLibraries,
    generate_predictions,
)
from tickettune.run_manifest import (
    artifact_digest,
    build_run_manifest,
    json_safe,
    write_manifest,
)
from tickettune.schemas import CATEGORY_LABELS, TriageOutput

FIXTURE = Path(__file__).parent / "fixtures" / "predictions.jsonl"
TEST_SOURCE = Path(__file__).parent / "fixtures" / "tickets.jsonl"


def _config(
    tmp_path: Path,
    *,
    strict_json_rate: float = 0.30,
    schema_valid_rate: float = 0.30,
    category_accuracy: float = 0.60,
    priority_accuracy: float = 0.60,
    sentiment_accuracy: float = 0.60,
    response_policy_rate: float = 0.60,
) -> FineTuneConfig:
    return FineTuneConfig(
        model=ModelConfig(
            name_or_path="Qwen/Qwen2.5-0.5B-Instruct",
            revision="7ae557604adf67be50417f59c2c2f167def9a775",
            parameters_b=0.5,
            max_sequence_length=512,
        ),
        data=DataConfig(
            source_path=TEST_SOURCE,
            processed_dir=tmp_path / "processed",
        ),
        lora=LoraConfig(),
        training=TrainingConfig(output_dir=tmp_path / "training"),
        evaluation=EvaluationConfig(
            output_dir=tmp_path / "evaluation",
            thresholds=EvaluationThresholds(
                strict_json_rate=strict_json_rate,
                schema_valid_rate=schema_valid_rate,
                category_accuracy=category_accuracy,
                priority_accuracy=priority_accuracy,
                sentiment_accuracy=sentiment_accuracy,
                response_policy_rate=response_policy_rate,
            ),
        ),
        deployment=DeploymentConfig(merged_model_dir=tmp_path / "merged"),
    )


def _expected() -> TriageOutput:
    return TriageOutput(
        category="security",
        priority="urgent",
        sentiment="worried",
        response="I will secure the account and investigate this activity.",
        next_action="lock_and_review_account",
    )


def _adapter(
    config: FineTuneConfig,
    root: Path,
    *,
    with_training_manifest: bool,
    dataset_overrides: dict[str, str] | None = None,
) -> Path:
    prepare_dataset(
        config.data.source_path,
        config.data.processed_dir,
        seed=config.seed,
        splits=config.data.splits,
    )
    verification = verify_prepared_dataset(
        config.data.source_path,
        config.data.processed_dir,
        seed=config.seed,
        splits=config.data.splits,
        required_splits=("train", "validation", "test"),
    )
    dataset_sha256 = {
        "source": verification.source_sha256,
        "manifest": verification.manifest_sha256,
        **verification.split_sha256,
        **(dataset_overrides or {}),
    }
    created_at = datetime(2026, 7, 18, tzinfo=UTC)
    provisional = build_run_manifest(
        config=json_safe(config),
        status="completed",
        project_name=config.project_name,
        model_name_or_path=config.model.name_or_path,
        model_revision=config.model.revision,
        method=config.lora.method,
        seed=config.seed,
        dataset_sha256=dataset_sha256,
        created_at=created_at,
        versions={},
    )
    run_dir = root / "runs" / provisional.run_id
    adapter = run_dir / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": config.model.name_or_path,
                "revision": config.model.revision,
            }
        ),
        encoding="utf-8",
    )
    (adapter / "adapter_model.safetensors").write_bytes(b"offline-adapter")
    if with_training_manifest:
        artifacts = tuple(
            artifact_digest(path, relative_to=run_dir) for path in sorted(adapter.iterdir())
        )
        manifest = build_run_manifest(
            config=json_safe(config),
            status="completed",
            project_name=config.project_name,
            model_name_or_path=config.model.name_or_path,
            model_revision=config.model.revision,
            method=config.lora.method,
            seed=config.seed,
            dataset_sha256=dataset_sha256,
            artifacts=artifacts,
            created_at=created_at,
            versions={},
        )
        write_manifest(run_dir / "manifest.json", manifest)
    return adapter


def _prepared_generation_cohort(
    config: FineTuneConfig,
) -> generation_module.GenerationCohort:
    prepare_dataset(
        config.data.source_path,
        config.data.processed_dir,
        seed=config.seed,
        splits=config.data.splits,
    )
    return generation_module._load_generation_cohort(config)


def _fake_cohort_predictions(
    config: FineTuneConfig,
    cohort: generation_module.GenerationCohort,
    *,
    adapter_path: Path | None,
    with_training_lineage: bool,
    forced_adapter_path: Path | None = None,
) -> tuple[GeneratedPrediction, ...]:
    verification = cohort.verification
    training_dataset_sha256 = (
        {
            "source": verification.source_sha256,
            "manifest": verification.manifest_sha256,
            **verification.split_sha256,
            "qualification_review_manifest": "5" * 64,
            "qualification_report": "6" * 64,
        }
        if with_training_lineage
        else {}
    )
    declared_adapter = forced_adapter_path or adapter_path
    return tuple(
        GeneratedPrediction(
            id=identifier,
            expected=json.loads(expected_payload),
            prediction=expected_payload.decode("utf-8"),
            latency_ms=5.0,
            prompt_tokens=10,
            generated_tokens=20,
            model_name_or_path=config.model.name_or_path,
            model_revision=config.model.revision,
            dataset_manifest_sha256=verification.manifest_sha256,
            dataset_split_sha256=verification.split_sha256["test"],
            generation_config_sha256="e" * 64,
            training_manifest_sha256="f" * 64 if with_training_lineage else None,
            training_config_sha256="1" * 64 if with_training_lineage else None,
            training_dataset_sha256=training_dataset_sha256,
            qualification_sha256={
                "qualification_review_manifest": "5" * 64,
                "qualification_report": "6" * 64,
            }
            if with_training_lineage
            else {},
            adapter_path=str(declared_adapter) if declared_adapter else None,
            adapter_config_sha256="b" * 64 if declared_adapter else None,
            adapter_weight_sha256={"adapter_model.safetensors": "c" * 64}
            if declared_adapter
            else {},
        )
        for identifier, expected_payload in zip(
            cohort.ordered_ids,
            cohort.expected_json,
            strict=True,
        )
    )


def test_extract_json_object_recovers_fence_and_prose() -> None:
    text = (
        "Result follows:\n```json\n"
        '{"category":"security","response":"Keep the {account} locked."}'
        "\n```\nDone."
    )

    assert extract_json_object(text) == {
        "category": "security",
        "response": "Keep the {account} locked.",
    }


def test_extract_json_object_returns_none_for_malformed_output() -> None:
    assert extract_json_object("not JSON {still broken") is None


def test_score_prediction_rewards_valid_exact_labels() -> None:
    expected = _expected()

    score = score_prediction(expected, json.dumps(expected.model_dump(mode="json")))

    assert score.strict_json_only is True
    assert score.schema_valid is True
    assert score.field_completeness == 1.0
    assert score.category_correct is True
    assert score.priority_correct is True
    assert score.sentiment_correct is True
    assert score.response_policy_compliant is True
    assert score.exact_match is True
    assert score.parse_error is None


def test_score_prediction_keeps_partial_metrics_for_schema_error() -> None:
    expected = _expected()
    partial = {
        "category": "security",
        "priority": "urgent",
        "sentiment": "neutral",
    }

    score = score_prediction(expected, partial)

    assert score.strict_json_only is True
    assert score.schema_valid is False
    assert score.field_completeness == pytest.approx(3 / 5)
    assert score.category_correct is True
    assert score.priority_correct is True
    assert score.sentiment_correct is False
    assert score.response_policy_compliant is False
    assert score.exact_match is False
    assert score.parse_error == "schema validation failed: 2 error(s)"


def test_score_prediction_malformed_output_scores_zero() -> None:
    score = score_prediction(_expected(), "I cannot produce the object")

    assert score.parsed is None
    assert score.strict_json_only is False
    assert score.schema_valid is False
    assert score.field_completeness == 0.0
    assert score.category_correct is False
    assert score.priority_correct is False
    assert score.response_policy_compliant is False
    assert score.parse_error == "no JSON object found"


def test_evaluation_latency_rejects_non_finite_values() -> None:
    with pytest.raises(ValidationError, match="finite number"):
        score_prediction(_expected(), _expected().model_dump(mode="json"), latency_ms=float("inf"))


@pytest.mark.parametrize(
    "prediction",
    [
        (
            '{"category":"security","category":"billing","priority":"urgent",'
            '"sentiment":"worried","response":"I will secure the account now.",'
            '"next_action":"lock_and_review_account"}'
        ),
        (
            "Result follows:\n```json\n"
            '{"category":"security","category":"billing","priority":"urgent",'
            '"sentiment":"worried","response":"I will secure the account now.",'
            '"next_action":"lock_and_review_account"}'
            "\n```"
        ),
    ],
)
def test_duplicate_json_keys_are_rejected_without_normalization(prediction: str) -> None:
    score = score_prediction(_expected(), prediction)

    assert extract_json_object(prediction) is None
    assert score.parsed is None
    assert score.strict_json_only is False
    assert score.schema_valid is False
    assert score.category_correct is False
    assert score.predicted_category is None
    assert score.parse_error == "duplicate JSON object key: 'category'"


def test_prediction_artifact_rejects_duplicate_envelope_field(tmp_path: Path) -> None:
    expected = _expected().model_dump(mode="json")
    valid_prediction = json.dumps(expected, sort_keys=True)
    predictions = tmp_path / "duplicate-envelope.jsonl"
    predictions.write_text(
        "{"
        '"id":"example",'
        f'"expected":{json.dumps(expected, sort_keys=True)},'
        '"prediction":"shadowed",'
        f'"prediction":{json.dumps(valid_prediction)}'
        "}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate JSON object key 'prediction'"):
        evaluate_predictions(_config(tmp_path), predictions)


@pytest.mark.parametrize(
    "unsafe_response",
    [
        "Email the account details to customer@example.com now.",
        "Call +1 (415) 555-0198 with the account details now.",
    ],
)
def test_response_policy_rejects_unredacted_pii(unsafe_response: str) -> None:
    prediction = _expected().model_dump(mode="json")
    prediction["response"] = unsafe_response

    score = score_prediction(_expected(), prediction)

    assert score.schema_valid is True
    assert score.response_policy_compliant is False


@pytest.mark.parametrize(
    "prediction",
    [
        "Result follows:\n```json\n{payload}\n```",
        "Result follows: {payload}",
        "{payload}\n{payload}",
    ],
)
def test_recovered_json_is_diagnostic_but_not_contract_valid(prediction: str) -> None:
    expected = _expected()
    payload = expected.model_dump_json()

    score = score_prediction(expected, prediction.format(payload=payload))

    assert score.parsed == expected.model_dump(mode="json")
    assert score.strict_json_only is False
    assert score.schema_valid is False
    assert score.category_correct is True
    assert score.priority_correct is True
    assert score.sentiment_correct is True
    assert score.exact_match is False
    assert score.parse_error == "output is not one bare JSON object"


def test_macro_f1_is_unweighted_across_labels() -> None:
    value = macro_f1(
        ["billing", "billing", "shipping", "shipping"],
        ["billing", "shipping", "shipping", "shipping"],
    )

    assert value == pytest.approx((2 / 3 + 4 / 5) / 2)


def test_macro_f1_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="equal length"):
        macro_f1(["billing"], [])


def test_macro_f1_includes_unobserved_canonical_labels() -> None:
    assert macro_f1(
        ["billing"],
        ["billing"],
        labels=CATEGORY_LABELS,
    ) == pytest.approx(1 / len(CATEGORY_LABELS))


def test_evaluate_predictions_writes_all_reports_and_latency_metrics(tmp_path: Path) -> None:
    config = _config(tmp_path)

    artifacts = evaluate_predictions(config, FIXTURE)

    assert artifacts.passed is True
    assert artifacts.report.summary.examples == 3
    assert artifacts.report.summary.strict_json_rate == pytest.approx(1 / 3)
    assert artifacts.report.summary.schema_valid_rate == pytest.approx(1 / 3)
    assert artifacts.report.summary.category_accuracy == pytest.approx(2 / 3)
    assert artifacts.report.summary.category_macro_f1 == pytest.approx(2 / 7)
    assert artifacts.report.summary.priority_macro_f1 == pytest.approx(0.5)
    assert artifacts.report.summary.sentiment_accuracy == pytest.approx(2 / 3)
    assert artifacts.report.summary.sentiment_macro_f1 == pytest.approx(0.4)
    assert artifacts.report.summary.response_policy_rate == pytest.approx(2 / 3)
    assert artifacts.report.summary.latency_mean_ms == pytest.approx(20.0)
    assert artifacts.report.summary.latency_p50_ms == pytest.approx(20.0)
    assert artifacts.report.summary.latency_p95_ms == pytest.approx(29.0)

    json_path = Path(artifacts.json_report_path)
    markdown_path = Path(artifacts.markdown_report_path)
    scored_path = Path(artifacts.scored_predictions_path)
    assert json_path.is_file()
    assert markdown_path.is_file()
    assert scored_path.is_file()
    assert len(scored_path.read_text(encoding="utf-8").splitlines()) == 3
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "Strict bare-JSON rate" in markdown
    assert "Category macro F1" in markdown
    assert "Response-policy compliance" in markdown
    assert "Overall threshold result: **PASS**" in markdown
    report_json = json.loads(json_path.read_text(encoding="utf-8"))
    assert report_json["summary"]["priority_accuracy"] == pytest.approx(2 / 3)


def test_threshold_failure_is_reported_and_optionally_raised(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        strict_json_rate=0.90,
        schema_valid_rate=0.90,
        category_accuracy=0.90,
        priority_accuracy=0.90,
    )

    artifacts = evaluate_predictions(config, FIXTURE)

    assert artifacts.passed is False
    assert {item.metric for item in artifacts.report.thresholds if not item.passed} == {
        "strict_json_rate",
        "schema_valid_rate",
        "category_accuracy",
        "priority_accuracy",
    }
    with pytest.raises(EvaluationThresholdError, match="thresholds failed"):
        evaluate_predictions(
            config,
            FIXTURE,
            output_dir=tmp_path / "raised",
            raise_on_failure=True,
        )
    assert (tmp_path / "raised" / "evaluation-report.json").is_file()


def test_predictions_file_errors_include_line_number(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"id":"ok"}\nnot-json\n', encoding="utf-8")

    with pytest.raises(ValueError, match=r"bad\.jsonl:1: missing expected"):
        evaluate_predictions(_config(tmp_path), path)


def test_generation_uses_chat_template_adapter_and_no_automatic_device_map(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    prepare_dataset(
        config.data.source_path,
        config.data.processed_dir,
        seed=config.seed,
        splits=config.data.splits,
    )
    expected = _expected().model_dump(mode="json")
    state: dict[str, Any] = {}

    class FakeTensor:
        shape = (1, 3)

        def to(self, device: str) -> FakeTensor:
            state["tensor_device"] = device
            return self

    class FakeTokenizer:
        eos_token_id = 2
        pad_token_id = 2

        def __init__(self) -> None:
            self.eos_token: str | None = "</s>"
            self.pad_token: str | None = state.get("preset_pad")

        def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
            state["chat_template"] = (messages, kwargs)
            return "rendered prompt"

        def __call__(self, text: str, **kwargs: Any) -> dict[str, FakeTensor]:
            state["tokenizer_call"] = (text, kwargs)
            return {"input_ids": FakeTensor()}

        def decode(self, tokens: list[int], **kwargs: Any) -> str:
            state["decode"] = (tokens, kwargs)
            return json.dumps(expected)

    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, model: str, **kwargs: Any) -> FakeTokenizer:
            state["tokenizer_load"] = (model, kwargs)
            tokenizer = FakeTokenizer()
            state["tokenizer"] = tokenizer
            return tokenizer

    class FakeModel:
        def to(self, device: str) -> None:
            state["model_device"] = device

        def eval(self) -> None:
            state["eval"] = True

        def generate(self, **kwargs: Any) -> list[list[int]]:
            state["generate"] = kwargs
            return [[1, 2, 3, 9, 10]]

    model = FakeModel()

    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: Any) -> FakeModel:
            state["model_load"] = (model_id, kwargs)
            return model

    class FakePeftModel:
        @classmethod
        def from_pretrained(cls, base_model: FakeModel, adapter: str, **kwargs: Any) -> FakeModel:
            state["adapter_load"] = (base_model, adapter, kwargs)
            backup = Path(adapter).parent / "verified-original-adapter"
            original_adapter = tmp_path / "adapter"
            original_adapter.rename(backup)
            original_adapter.mkdir()
            (original_adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
            (original_adapter / "adapter_model.safetensors").write_bytes(b"attacker")
            try:
                state["adapter_snapshot_weight"] = (
                    Path(adapter) / "adapter_model.safetensors"
                ).read_bytes()
            finally:
                for artifact in original_adapter.iterdir():
                    artifact.unlink()
                original_adapter.rmdir()
                backup.rename(original_adapter)
            return base_model

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class FakeMps:
        @staticmethod
        def is_available() -> bool:
            return False

    class FakeTorch:
        cuda = FakeCuda()
        backends = type("Backends", (), {"mps": FakeMps()})()
        bfloat16 = "torch.bfloat16"
        float16 = "torch.float16"
        float32 = "torch.float32"

        @staticmethod
        def manual_seed(seed: int) -> None:
            state["seed"] = seed

        @staticmethod
        def inference_mode() -> nullcontext[None]:
            return nullcontext()

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": config.model.name_or_path,
                "revision": config.model.revision,
            }
        ),
        encoding="utf-8",
    )
    (adapter / "adapter_model.safetensors").write_bytes(b"offline-adapter")
    candidate_path = tmp_path / "generated" / "candidate-predictions.jsonl"
    predictions = generate_predictions(
        config,
        adapter_path=adapter,
        output_path=candidate_path,
        _libraries=GenerationLibraries(
            torch=FakeTorch,
            AutoModelForCausalLM=FakeAutoModel,
            AutoTokenizer=FakeAutoTokenizer,
            PeftModel=FakePeftModel,
        ),
    )

    assert len(predictions) == 2
    assert state["seed"] == config.seed
    assert predictions[0].generated_tokens == 2
    assert predictions[0].prediction == json.dumps(expected)
    assert predictions[0].model_revision == config.model.revision
    assert predictions[0].dataset_manifest_sha256 == sha256_file(
        config.data.processed_dir / "manifest.json"
    )
    assert predictions[0].dataset_split_sha256 == sha256_file(
        config.data.processed_dir / "test.jsonl"
    )
    assert predictions[0].generation_config_sha256 is not None
    assert predictions[0].training_manifest_sha256 is None
    assert predictions[0].training_config_sha256 is None
    assert predictions[0].training_dataset_sha256 == {}
    assert predictions[0].qualification_sha256 == {}
    assert predictions[0].adapter_config_sha256 == sha256_file(adapter / "adapter_config.json")
    assert predictions[0].adapter_weight_sha256 == {
        "adapter_model.safetensors": sha256_file(adapter / "adapter_model.safetensors")
    }
    provenance_artifacts = evaluate_predictions(
        config,
        candidate_path,
        output_dir=tmp_path / "provenance-report",
    )
    provenance = provenance_artifacts.report.provenance
    assert provenance is not None
    assert provenance.dataset_manifest_sha256 == predictions[0].dataset_manifest_sha256
    assert provenance.dataset_split_sha256 == predictions[0].dataset_split_sha256
    assert provenance.generation_config_sha256 == predictions[0].generation_config_sha256
    assert provenance.training_manifest_sha256 is None
    assert provenance.training_config_sha256 is None
    assert provenance.training_dataset_sha256 == {}
    assert provenance.qualification_sha256 == {}
    assert provenance.model_revision == config.model.revision
    assert provenance.adapter_config_sha256 == predictions[0].adapter_config_sha256
    assert provenance.adapter_weight_sha256 == predictions[0].adapter_weight_sha256
    assert "## Provenance" in Path(provenance_artifacts.markdown_report_path).read_text(
        encoding="utf-8"
    )
    model_kwargs = state["model_load"][1]
    assert model_kwargs["dtype"] == "auto"
    assert model_kwargs["local_files_only"] is True
    assert model_kwargs["device_map"] is None
    assert state["chat_template"][1] == {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    generation_kwargs = state["generate"]
    assert generation_kwargs["do_sample"] is False
    assert "temperature" not in generation_kwargs
    assert state["adapter_load"][2]["is_trainable"] is False
    loaded_adapter_path = Path(state["adapter_load"][1])
    assert loaded_adapter_path != adapter.resolve()
    assert state["adapter_snapshot_weight"] == b"offline-adapter"
    assert not loaded_adapter_path.exists()

    llama_config = config.model_copy(
        update={
            "model": config.model.model_copy(
                update={
                    "name_or_path": "meta-llama/Llama-3.1-8B-Instruct",
                    "torch_dtype": "float16",
                    "eos_token": "<|eot_id|>",
                }
            ),
            "generation": config.generation.model_copy(
                update={"do_sample": True, "temperature": 0.7, "top_p": 0.8}
            ),
        }
    )
    state["preset_pad"] = "<pad>"
    output_path = tmp_path / "generated" / "predictions.jsonl"

    second = generate_predictions(
        llama_config,
        output_path=output_path,
        _libraries=GenerationLibraries(
            torch=FakeTorch,
            AutoModelForCausalLM=FakeAutoModel,
            AutoTokenizer=FakeAutoTokenizer,
            PeftModel=FakePeftModel,
        ),
    )

    assert second[0].adapter_path is None
    assert second[0].adapter_config_sha256 is None
    assert second[0].adapter_weight_sha256 == {}
    assert second[0].dataset_manifest_sha256 == predictions[0].dataset_manifest_sha256
    assert output_path.is_file()
    assert state["model_load"][1]["dtype"] == "torch.float16"
    assert state["generate"]["temperature"] == 0.7
    assert state["generate"]["top_p"] == 0.8
    assert state["tokenizer"].eos_token == "<|eot_id|>"
    assert state["tokenizer"].pad_token == "<pad>"


@pytest.mark.parametrize("tamper", ["missing_manifest", "changed_test_bytes"])
def test_generation_rejects_broken_dataset_chain_before_optional_imports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    config = _config(tmp_path)
    prepare_dataset(
        config.data.source_path,
        config.data.processed_dir,
        seed=config.seed,
        splits=config.data.splits,
    )
    if tamper == "missing_manifest":
        (config.data.processed_dir / "manifest.json").unlink()
        message = "manifest not found"
    else:
        test_path = config.data.processed_dir / "test.jsonl"
        test_path.write_bytes(test_path.read_bytes() + b"\n")
        message = "SHA-256 mismatch"
    loaded = False

    def fail_if_loaded() -> None:
        nonlocal loaded
        loaded = True
        raise AssertionError("generation libraries must not load")

    monkeypatch.setattr("tickettune.generation._load_generation_libraries", fail_if_loaded)

    with pytest.raises(DatasetIntegrityError, match=message):
        generate_predictions(config)
    assert loaded is False


def test_generation_requires_exact_manifest_test_path_before_optional_imports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    prepare_dataset(
        config.data.source_path,
        config.data.processed_dir,
        seed=config.seed,
        splits=config.data.splits,
    )
    alternate = tmp_path / "alternate-test.jsonl"
    alternate.write_bytes((config.data.processed_dir / "test.jsonl").read_bytes())
    loaded = False

    def fail_if_loaded() -> None:
        nonlocal loaded
        loaded = True
        raise AssertionError("generation libraries must not load")

    monkeypatch.setattr("tickettune.generation._load_generation_libraries", fail_if_loaded)

    with pytest.raises(DatasetIntegrityError, match="canonical manifest test split"):
        generate_predictions(config, dataset_path=alternate)
    assert loaded is False


@pytest.mark.parametrize(
    ("adapter_config", "message"),
    [
        (
            {
                "base_model_name_or_path": "Qwen/Qwen2.5-1.5B-Instruct",
                "revision": "main",
            },
            "base model mismatch",
        ),
        (
            {
                "base_model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct",
                "revision": "different-revision",
            },
            "revision mismatch",
        ),
    ],
)
def test_generation_rejects_adapter_provenance_mismatch_before_optional_imports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    adapter_config: dict[str, str],
    message: str,
) -> None:
    config = _config(tmp_path)
    prepare_dataset(
        config.data.source_path,
        config.data.processed_dir,
        seed=config.seed,
        splits=config.data.splits,
    )
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps(adapter_config),
        encoding="utf-8",
    )
    loaded = False

    def fail_if_loaded() -> None:
        nonlocal loaded
        loaded = True
        raise AssertionError("generation libraries must not load")

    monkeypatch.setattr("tickettune.generation._load_generation_libraries", fail_if_loaded)

    with pytest.raises(AdapterCompatibilityError, match=message):
        generate_predictions(config, adapter_path=adapter)
    assert loaded is False


def test_generation_rejects_duplicate_adapter_config_key(tmp_path: Path) -> None:
    config = _config(tmp_path)
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    adapter.joinpath("adapter_config.json").write_text(
        "{"
        '"base_model_name_or_path":"shadowed/model",'
        f'"base_model_name_or_path":{json.dumps(config.model.name_or_path)},'
        f'"revision":{json.dumps(config.model.revision)}'
        "}",
        encoding="utf-8",
    )
    adapter.joinpath("adapter_model.safetensors").write_bytes(b"offline-adapter")

    with pytest.raises(AdapterCompatibilityError, match="duplicate JSON object key"):
        generation_module.validate_adapter_compatibility(
            adapter,
            model_name_or_path=config.model.name_or_path,
            model_revision=config.model.revision,
        )


def test_generation_helpers_cover_devices_dtypes_and_input_errors(tmp_path: Path) -> None:
    cuda = SimpleNamespace(is_available=lambda: True)
    mps = SimpleNamespace(is_available=lambda: True)
    assert generation_module._device(SimpleNamespace(cuda=cuda)) == "cuda"
    assert generation_module._device(SimpleNamespace(cuda=cuda), force_cpu=True) == "cpu"
    assert (
        generation_module._device(
            SimpleNamespace(
                cuda=SimpleNamespace(is_available=lambda: False),
                backends=SimpleNamespace(mps=mps),
            )
        )
        == "mps"
    )

    seeded: list[int] = []
    cuda_seeds: list[int] = []
    generation_module._seed(
        SimpleNamespace(
            manual_seed=seeded.append,
            cuda=SimpleNamespace(
                is_available=lambda: True,
                manual_seed_all=cuda_seeds.append,
            ),
        ),
        42,
    )
    assert seeded == [42]
    assert cuda_seeds == [42]

    config = _config(tmp_path)
    libraries = GenerationLibraries(
        torch=SimpleNamespace(
            bfloat16="bf16",
            float16="fp16",
            float32="fp32",
        ),
        AutoModelForCausalLM=None,
        AutoTokenizer=None,
        PeftModel=None,
    )
    bf16_config = config.model_copy(
        update={"model": config.model.model_copy(update={"torch_dtype": "bfloat16"})}
    )
    float_config = config.model_copy(
        update={"model": config.model.model_copy(update={"torch_dtype": "float32"})}
    )
    assert generation_module._dtype(bf16_config, libraries) == "bf16"
    assert generation_module._dtype(float_config, libraries) == "fp32"

    assert generation_module._completion_to_expected(json.dumps({"ok": True})) == {"ok": True}
    assert generation_module._completion_to_expected({"ok": True}) == {"ok": True}
    with pytest.raises(ValueError, match="valid expected JSON"):
        generation_module._completion_to_expected("not-json")
    with pytest.raises(ValueError, match="must contain"):
        generation_module._completion_to_expected([])

    malformed = tmp_path / "malformed.jsonl"
    malformed.write_text("not-json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        generation_module._read_generation_records(malformed)
    non_object = tmp_path / "array.jsonl"
    non_object.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected a JSON object"):
        generation_module._read_generation_records(non_object)
    assert generation_module._token_count(object()) == 0


def test_run_model_evaluation_compares_fake_candidate_and_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(
        tmp_path,
        strict_json_rate=0.0,
        schema_valid_rate=0.0,
        category_accuracy=0.0,
        priority_accuracy=0.0,
        sentiment_accuracy=0.0,
        response_policy_rate=0.0,
    )
    cohort = _prepared_generation_cohort(config)
    calls: list[tuple[Path | None, bool]] = []

    def fake_generate(
        _config_value: FineTuneConfig,
        *,
        adapter_path: Path | None = None,
        allow_download: bool = False,
        require_training_manifest: bool = False,
    ) -> tuple[GeneratedPrediction, ...]:
        assert allow_download is False
        calls.append((adapter_path, require_training_manifest))
        return _fake_cohort_predictions(
            config,
            cohort,
            adapter_path=adapter_path,
            with_training_lineage=require_training_manifest,
        )

    monkeypatch.setattr("tickettune.evaluation.generate_predictions", fake_generate)
    adapter = tmp_path / "adapter"
    adapter.mkdir()

    result = run_model_evaluation(config, adapter_path=adapter, compare_baseline=True)

    assert calls == [(adapter, True), (None, False)]
    assert result.candidate.passed is True
    assert result.baseline is not None
    assert result.comparison is not None
    assert result.comparison.metric_deltas["strict_json_rate"] == 0.0
    assert result.comparison.metric_deltas["sentiment_accuracy"] == 0.0
    assert result.comparison.metric_deltas["response_policy_rate"] == 0.0
    assert result.comparison.passed is True
    assert all(item.passed for item in result.comparison.non_regression)
    assert result.candidate.report.provenance is not None
    assert result.candidate.report.provenance.adapter_path == str(adapter)
    assert (
        result.candidate.report.provenance.dataset_split_sha256
        == cohort.verification.split_sha256["test"]
    )
    assert result.candidate.report.provenance.generation_config_sha256 == "e" * 64
    assert result.candidate.report.provenance.training_manifest_sha256 == "f" * 64
    assert result.candidate.report.provenance.training_config_sha256 == "1" * 64
    assert (
        result.candidate.report.provenance.training_dataset_sha256["manifest"]
        == cohort.verification.manifest_sha256
    )
    assert (
        result.candidate.report.provenance.training_dataset_sha256["test"]
        == cohort.verification.split_sha256["test"]
    )
    assert result.candidate.report.provenance.qualification_sha256 == {
        "qualification_review_manifest": "5" * 64,
        "qualification_report": "6" * 64,
    }
    assert result.baseline.report.provenance is not None
    assert result.baseline.report.provenance.adapter_path is None
    assert Path(result.output_dir).parent.name == "runs"
    assert Path(result.output_dir).name == result.evaluation_id
    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["evaluation_id"] == result.evaluation_id
    assert (
        manifest["artifact_sha256"]["candidate-predictions.jsonl"]
        == result.candidate.report.predictions_sha256
    )
    assert set(manifest["artifact_sha256"]) == {
        "baseline-predictions.jsonl",
        "baseline/evaluation-report.json",
        "baseline/evaluation-report.md",
        "baseline/scored-predictions.jsonl",
        "candidate-predictions.jsonl",
        "candidate/evaluation-report.json",
        "candidate/evaluation-report.md",
        "candidate/scored-predictions.jsonl",
    }
    pointer = json.loads(Path(result.latest_pointer_path).read_text(encoding="utf-8"))
    assert pointer["evaluation_id"] == result.evaluation_id
    with pytest.raises(ValueError, match="requires an adapter_path"):
        run_model_evaluation(config, compare_baseline=True)


@pytest.mark.parametrize("mutation", ["id", "expected"])
def test_live_evaluation_binds_in_memory_rows_to_exact_verified_test_objects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    config = _config(tmp_path)
    cohort = _prepared_generation_cohort(config)

    def fake_generate(
        _config_value: FineTuneConfig,
        *,
        adapter_path: Path | None = None,
        allow_download: bool = False,
        require_training_manifest: bool = False,
    ) -> tuple[GeneratedPrediction, ...]:
        del allow_download, require_training_manifest
        predictions = list(
            _fake_cohort_predictions(
                config,
                cohort,
                adapter_path=adapter_path,
                with_training_lineage=False,
            )
        )
        first = predictions[0]
        if mutation == "id":
            predictions[0] = first.model_copy(update={"id": "REPLACED-ID"})
        else:
            predictions[0] = first.model_copy(
                update={"expected": first.expected | {"category": "shipping"}}
            )
        return tuple(predictions)

    monkeypatch.setattr(evaluation_module, "generate_predictions", fake_generate)

    expected_error = "ordered IDs" if mutation == "id" else "expected objects"
    with pytest.raises(ValueError, match=expected_error):
        run_model_evaluation(config)

    manifests = tuple(
        (Path(config.evaluation.output_dir) / "runs").glob("*/evaluation-manifest.json")
    )
    assert len(manifests) == 1
    assert json.loads(manifests[0].read_text(encoding="utf-8"))["status"] == "failed"
    assert not tuple(
        (Path(config.evaluation.output_dir) / "runs").glob("*/candidate-predictions.jsonl")
    )


def test_live_evaluation_detects_prediction_replacement_before_success_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(
        tmp_path,
        strict_json_rate=0.0,
        schema_valid_rate=0.0,
        category_accuracy=0.0,
        priority_accuracy=0.0,
        sentiment_accuracy=0.0,
        response_policy_rate=0.0,
    )
    cohort = _prepared_generation_cohort(config)

    def fake_generate(
        _config_value: FineTuneConfig,
        *,
        adapter_path: Path | None = None,
        allow_download: bool = False,
        require_training_manifest: bool = False,
    ) -> tuple[GeneratedPrediction, ...]:
        del allow_download, require_training_manifest
        return _fake_cohort_predictions(
            config,
            cohort,
            adapter_path=adapter_path,
            with_training_lineage=False,
        )

    original_evaluate = evaluation_module._evaluate_generated_predictions

    def replace_then_score(
        config_value: FineTuneConfig,
        predictions: tuple[GeneratedPrediction, ...],
        *,
        predictions_path: Path,
        predictions_payload: bytes,
        output_dir: Path,
    ) -> evaluation_module.EvaluationArtifacts:
        predictions_path.unlink()
        predictions_path.write_bytes(b'{"replacement":true}\n')
        return original_evaluate(
            config_value,
            predictions,
            predictions_path=predictions_path,
            predictions_payload=predictions_payload,
            output_dir=output_dir,
        )

    monkeypatch.setattr(evaluation_module, "generate_predictions", fake_generate)
    monkeypatch.setattr(
        evaluation_module,
        "_evaluate_generated_predictions",
        replace_then_score,
    )

    with pytest.raises(ValueError, match="changed before success"):
        run_model_evaluation(config)

    run_dir = next((Path(config.evaluation.output_dir) / "runs").iterdir())
    assert (run_dir / "candidate/evaluation-report.json").is_file()
    assert (
        json.loads((run_dir / "evaluation-manifest.json").read_text(encoding="utf-8"))["status"]
        == "failed"
    )
    assert not (Path(config.evaluation.output_dir) / "latest-evaluation.json").exists()


def test_live_evaluation_refuses_to_reuse_a_precreated_prediction_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    cohort = _prepared_generation_cohort(config)

    def fake_generate(
        _config_value: FineTuneConfig,
        *,
        adapter_path: Path | None = None,
        allow_download: bool = False,
        require_training_manifest: bool = False,
    ) -> tuple[GeneratedPrediction, ...]:
        del allow_download, require_training_manifest
        predictions = _fake_cohort_predictions(
            config,
            cohort,
            adapter_path=adapter_path,
            with_training_lineage=False,
        )
        run_dir = next((Path(config.evaluation.output_dir) / "runs").iterdir())
        (run_dir / "candidate-predictions.jsonl").write_bytes(
            evaluation_module._generated_payload(predictions)
        )
        return predictions

    monkeypatch.setattr(evaluation_module, "generate_predictions", fake_generate)

    with pytest.raises(ValueError, match="refusing to reuse existing"):
        run_model_evaluation(config)


def test_live_adapter_evaluation_requires_a_sibling_training_manifest_before_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    adapter = _adapter(config, tmp_path / "naked", with_training_manifest=False)
    loaded = False

    def fail_if_loaded() -> None:
        nonlocal loaded
        loaded = True
        raise AssertionError("generation libraries must not load")

    monkeypatch.setattr(generation_module, "_load_generation_libraries", fail_if_loaded)

    with pytest.raises(AdapterCompatibilityError, match="requires its sibling training manifest"):
        run_model_evaluation(config, adapter_path=adapter)
    assert loaded is False


def test_live_adapter_evaluation_rejects_training_dataset_lineage_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    adapter = _adapter(
        config,
        tmp_path / "mismatched",
        with_training_manifest=True,
        dataset_overrides={"source": "f" * 64},
    )
    loaded = False

    def fail_if_loaded() -> None:
        nonlocal loaded
        loaded = True
        raise AssertionError("generation libraries must not load")

    monkeypatch.setattr(generation_module, "_load_generation_libraries", fail_if_loaded)

    with pytest.raises(
        AdapterCompatibilityError,
        match="source, prepared-manifest, or split hashes",
    ):
        run_model_evaluation(config, adapter_path=adapter)
    assert loaded is False


def test_live_adapter_evaluation_requires_exact_training_test_split_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    adapter = _adapter(
        config,
        tmp_path / "mismatched-test",
        with_training_manifest=True,
        dataset_overrides={"test": "f" * 64},
    )
    loaded = False

    def fail_if_loaded() -> None:
        nonlocal loaded
        loaded = True
        raise AssertionError("generation libraries must not load")

    monkeypatch.setattr(generation_module, "_load_generation_libraries", fail_if_loaded)

    with pytest.raises(
        AdapterCompatibilityError,
        match="source, prepared-manifest, or split hashes",
    ):
        run_model_evaluation(config, adapter_path=adapter)
    assert loaded is False


def test_unverified_adapter_override_is_local_only_and_cannot_enforce_thresholds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(
        tmp_path,
        strict_json_rate=0.0,
        schema_valid_rate=0.0,
        category_accuracy=0.0,
        priority_accuracy=0.0,
        sentiment_accuracy=0.0,
        response_policy_rate=0.0,
    )
    adapter = tmp_path / "naked-adapter"
    adapter.mkdir()
    cohort = _prepared_generation_cohort(config)
    requirements: list[bool] = []

    def fake_generate(
        _config_value: FineTuneConfig,
        *,
        adapter_path: Path | None = None,
        allow_download: bool = False,
        require_training_manifest: bool = False,
    ) -> tuple[GeneratedPrediction, ...]:
        del allow_download
        requirements.append(require_training_manifest)
        return _fake_cohort_predictions(
            config,
            cohort,
            adapter_path=adapter_path,
            with_training_lineage=False,
        )

    monkeypatch.setattr(evaluation_module, "generate_predictions", fake_generate)

    result = run_model_evaluation(
        config,
        adapter_path=adapter,
        allow_unverified_adapter=True,
    )
    assert result.candidate.report.provenance is not None
    assert result.candidate.report.provenance.training_manifest_sha256 is None
    assert requirements == [False]

    with pytest.raises(ValueError, match="cannot be combined with enforced release thresholds"):
        run_model_evaluation(
            config,
            adapter_path=adapter,
            allow_unverified_adapter=True,
            enforce_thresholds=True,
        )


def test_comparison_accepts_matching_fixture_reports_without_provenance(tmp_path: Path) -> None:
    config = _config(tmp_path)
    candidate = evaluate_predictions(config, FIXTURE, output_dir=tmp_path / "candidate")
    baseline = evaluate_predictions(config, FIXTURE, output_dir=tmp_path / "baseline")

    comparison = evaluation_module._compare(candidate, baseline)

    assert candidate.report.provenance is None
    assert baseline.report.provenance is None
    assert all(delta == 0.0 for delta in comparison.metric_deltas.values())


def test_comparison_rejects_reordered_ids(tmp_path: Path) -> None:
    config = _config(tmp_path)
    baseline = evaluate_predictions(config, FIXTURE, output_dir=tmp_path / "baseline")
    candidate = baseline.model_copy(
        update={
            "report": baseline.report.model_copy(
                update={"results": tuple(reversed(baseline.report.results))}
            )
        }
    )

    with pytest.raises(ValueError, match="ordered IDs differ"):
        evaluation_module._compare(candidate, baseline)


def test_comparison_rejects_missing_ids(tmp_path: Path) -> None:
    config = _config(tmp_path)
    baseline = evaluate_predictions(config, FIXTURE, output_dir=tmp_path / "baseline")
    candidate = baseline.model_copy(
        update={
            "report": baseline.report.model_copy(update={"results": baseline.report.results[:-1]})
        }
    )

    with pytest.raises(ValueError, match="ordered IDs must match exactly"):
        evaluation_module._compare(candidate, baseline)


@pytest.mark.parametrize("duplicate_side", ["candidate", "baseline"])
def test_comparison_rejects_duplicate_ids(tmp_path: Path, duplicate_side: str) -> None:
    config = _config(tmp_path)
    original = evaluate_predictions(config, FIXTURE, output_dir=tmp_path / "original")
    duplicated_results = (
        original.report.results[0],
        original.report.results[0],
        *original.report.results[2:],
    )
    duplicated = original.model_copy(
        update={"report": original.report.model_copy(update={"results": duplicated_results})}
    )
    candidate = duplicated if duplicate_side == "candidate" else original
    baseline = duplicated if duplicate_side == "baseline" else original

    with pytest.raises(ValueError, match=rf"{duplicate_side} evaluation contains duplicate IDs"):
        evaluation_module._compare(candidate, baseline)


def test_comparison_rejects_changed_expected_output(tmp_path: Path) -> None:
    config = _config(tmp_path)
    baseline = evaluate_predictions(config, FIXTURE, output_dir=tmp_path / "baseline")
    first = baseline.report.results[0]
    changed = first.model_copy(
        update={
            "expected": first.expected.model_copy(update={"category": "shipping"}),
        }
    )
    candidate = baseline.model_copy(
        update={
            "report": baseline.report.model_copy(
                update={"results": (changed, *baseline.report.results[1:])}
            )
        }
    )

    with pytest.raises(ValueError, match=r"expected output differs for ID 'EVAL-001'"):
        evaluation_module._compare(candidate, baseline)


def _provenance(**updates: Any) -> evaluation_module.EvaluationProvenance:
    values: dict[str, Any] = {
        "dataset_manifest_sha256": "a" * 64,
        "dataset_split_sha256": "b" * 64,
        "generation_config_sha256": "c" * 64,
        "training_manifest_sha256": None,
        "training_config_sha256": None,
        "training_dataset_sha256": {},
        "qualification_sha256": {},
        "model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct",
        "model_revision": "7ae557604adf67be50417f59c2c2f167def9a775",
        "adapter_path": None,
        "adapter_config_sha256": None,
        "adapter_weight_sha256": {},
    }
    values.update(updates)
    return evaluation_module.EvaluationProvenance.model_validate(values)


def _with_provenance(
    artifacts: evaluation_module.EvaluationArtifacts,
    provenance: evaluation_module.EvaluationProvenance | None,
) -> evaluation_module.EvaluationArtifacts:
    return artifacts.model_copy(
        update={"report": artifacts.report.model_copy(update={"provenance": provenance})}
    )


def test_comparison_rejects_one_sided_provenance(tmp_path: Path) -> None:
    config = _config(tmp_path)
    original = evaluate_predictions(config, FIXTURE, output_dir=tmp_path / "original")
    candidate = _with_provenance(original, _provenance())

    with pytest.raises(ValueError, match="must both be present or both be absent"):
        evaluation_module._compare(candidate, original)


def test_evaluation_provenance_rejects_incomplete_or_inconsistent_training_lineage() -> None:
    with pytest.raises(ValidationError, match="training lineage must be complete"):
        _provenance(training_manifest_sha256="d" * 64)

    with pytest.raises(ValidationError, match="qualification hashes must match"):
        _provenance(
            training_manifest_sha256="d" * 64,
            training_config_sha256="e" * 64,
            training_dataset_sha256={
                "source": "f" * 64,
                "manifest": "a" * 64,
                "train": "1" * 64,
                "validation": "2" * 64,
                "test": "b" * 64,
                "qualification_report": "3" * 64,
            },
            qualification_sha256={},
        )

    with pytest.raises(ValidationError, match="test hash differs"):
        _provenance(
            training_manifest_sha256="d" * 64,
            training_config_sha256="e" * 64,
            training_dataset_sha256={
                "source": "f" * 64,
                "manifest": "a" * 64,
                "train": "1" * 64,
                "validation": "2" * 64,
                "test": "9" * 64,
            },
        )


def test_comparison_rejects_training_lineage_on_base_model_baseline(tmp_path: Path) -> None:
    config = _config(tmp_path)
    original = evaluate_predictions(config, FIXTURE, output_dir=tmp_path / "original")
    lineage = {
        "training_manifest_sha256": "d" * 64,
        "training_config_sha256": "e" * 64,
        "training_dataset_sha256": {
            "source": "f" * 64,
            "manifest": "a" * 64,
            "train": "1" * 64,
            "validation": "2" * 64,
            "test": "b" * 64,
        },
    }
    candidate = _with_provenance(
        original,
        _provenance(
            **lineage,
            adapter_path="/tmp/adapter",
            adapter_config_sha256="3" * 64,
            adapter_weight_sha256={"adapter_model.safetensors": "4" * 64},
        ),
    )
    baseline = _with_provenance(original, _provenance(**lineage))

    with pytest.raises(ValueError, match="baseline provenance must be base-model-only"):
        evaluation_module._compare(candidate, baseline)


@pytest.mark.parametrize(
    ("field", "changed_value"),
    [
        ("dataset_manifest_sha256", "d" * 64),
        ("dataset_split_sha256", "d" * 64),
        ("generation_config_sha256", "d" * 64),
        ("model_name_or_path", "Qwen/Qwen2.5-1.5B-Instruct"),
        ("model_revision", "e" * 40),
    ],
)
def test_comparison_rejects_provenance_identity_mismatch(
    tmp_path: Path,
    field: str,
    changed_value: str,
) -> None:
    config = _config(tmp_path)
    original = evaluate_predictions(config, FIXTURE, output_dir=tmp_path / "original")
    baseline = _with_provenance(original, _provenance())
    candidate = _with_provenance(original, _provenance(**{field: changed_value}))

    with pytest.raises(ValueError, match=rf"provenance differs for {field}"):
        evaluation_module._compare(candidate, baseline)


def test_run_model_evaluation_rejects_adapter_backed_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    cohort = _prepared_generation_cohort(config)

    def fake_generate(
        _config_value: FineTuneConfig,
        *,
        adapter_path: Path | None = None,
        allow_download: bool = False,
        require_training_manifest: bool = False,
    ) -> tuple[GeneratedPrediction, ...]:
        assert allow_download is False
        return _fake_cohort_predictions(
            config,
            cohort,
            adapter_path=adapter_path,
            with_training_lineage=require_training_manifest,
            forced_adapter_path=(
                None if adapter_path is not None else tmp_path / "unexpected-baseline-adapter"
            ),
        )

    monkeypatch.setattr("tickettune.evaluation.generate_predictions", fake_generate)
    adapter = tmp_path / "adapter"
    adapter.mkdir()

    with pytest.raises(ValueError, match="baseline provenance must be base-model-only"):
        run_model_evaluation(config, adapter_path=adapter, compare_baseline=True)


def test_comparison_fails_when_candidate_regresses_a_quality_metric(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        strict_json_rate=0.0,
        schema_valid_rate=0.0,
        category_accuracy=0.0,
        priority_accuracy=0.0,
        sentiment_accuracy=0.0,
        response_policy_rate=0.0,
    )
    baseline = evaluate_predictions(config, FIXTURE, output_dir=tmp_path / "baseline")
    regressed_summary = baseline.report.summary.model_copy(
        update={"category_accuracy": max(0.0, baseline.report.summary.category_accuracy - 0.1)}
    )
    candidate = baseline.model_copy(
        update={
            "report": baseline.report.model_copy(update={"summary": regressed_summary}),
        }
    )

    comparison = evaluation_module._compare(candidate, baseline)

    category_gate = next(
        item for item in comparison.non_regression if item.metric == "category_accuracy"
    )
    assert category_gate.value < 0
    assert category_gate.passed is False
    assert comparison.passed is False
