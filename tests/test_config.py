from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError
from yaml.constructor import ConstructorError

from tickettune.config import FineTuneConfig, LoraConfig, SplitConfig, load_config

ROOT = Path(__file__).resolve().parents[1]

PROFILE_REVISIONS = {
    "smoke.yaml": "7ae557604adf67be50417f59c2c2f167def9a775",
    "cpu-smoke.yaml": "7ae557604adf67be50417f59c2c2f167def9a775",
    "qwen-0.5b-lora-local.yaml": "7ae557604adf67be50417f59c2c2f167def9a775",
    "qwen-0.5b-candidate-local.yaml": "7ae557604adf67be50417f59c2c2f167def9a775",
    "apple-silicon.yaml": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
    "qwen-7b-qlora.yaml": "a09a35458c702b33eeacc393d103063234e8bc28",
    "llama-8b-qlora.yaml": "0e9e39f249a16976918f6564b8830bc894c89659",
}
CPU_PROFILES = {
    "cpu-smoke.yaml",
    "qwen-0.5b-lora-local.yaml",
    "qwen-0.5b-candidate-local.yaml",
}


def _raw_smoke() -> dict[str, object]:
    value = yaml.safe_load((ROOT / "configs" / "smoke.yaml").read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_split_fractions_must_total_one() -> None:
    with pytest.raises(ValueError, match=r"sum to 1\.0"):
        SplitConfig(train=0.8, validation=0.1, test=0.2)


@pytest.mark.parametrize(("profile", "revision"), PROFILE_REVISIONS.items())
def test_profiles_are_strict_resolved_and_revision_pinned(profile: str, revision: str) -> None:
    config = load_config(ROOT / "configs" / profile)

    assert config.model.revision == revision
    expected_source = (
        ROOT / "data/qualified/support_tickets.jsonl"
        if profile == "qwen-0.5b-candidate-local.yaml"
        else ROOT / "data/raw/support_tickets.jsonl"
    )
    assert config.data.source_path == expected_source.resolve()
    assert config.data.processed_dir.is_absolute()
    assert config.training.output_dir.is_absolute()
    assert config.evaluation.output_dir.is_absolute()
    assert config.deployment.merged_model_dir.is_absolute()
    assert config.training.completion_only_loss is True


def test_remote_model_revision_must_be_a_full_commit_sha() -> None:
    raw = _raw_smoke()
    model = raw["model"]
    assert isinstance(model, dict)
    model["revision"] = "main"

    with pytest.raises(ValidationError, match="full 40-character commit SHA"):
        FineTuneConfig.model_validate(raw)

    model["name_or_path"] = "./models/local-base"
    assert FineTuneConfig.model_validate(raw).model.revision == "main"


def test_cpu_smoke_profile_is_explicit_and_current() -> None:
    config = load_config(ROOT / "configs" / "cpu-smoke.yaml")

    assert config.training.use_cpu is True
    assert config.training.warmup_steps == 0
    assert config.model.max_sequence_length == 256
    assert config.generation.max_new_tokens == 128
    assert config.deployment.max_model_len == 512


def test_qwen_half_b_local_profile_is_a_bounded_multi_epoch_cpu_run() -> None:
    config = load_config(ROOT / "configs" / "qwen-0.5b-lora-local.yaml")
    smoke = load_config(ROOT / "configs" / "cpu-smoke.yaml")

    assert config.project_name == "tickettune-qwen-0.5b-lora-local"
    assert config.model.name_or_path == "Qwen/Qwen2.5-0.5B-Instruct"
    assert config.model.revision == "7ae557604adf67be50417f59c2c2f167def9a775"
    assert config.model.torch_dtype == "float32"
    assert config.model.max_sequence_length == 512
    assert config.lora.method == "lora"
    assert config.lora.r == 8
    assert config.lora.alpha == 16
    assert config.training.use_cpu is True
    assert config.training.bf16 is False
    assert config.training.fp16 is False
    assert config.training.max_steps is None
    assert config.training.num_train_epochs == 8.0
    assert config.training.gradient_accumulation_steps == 4
    assert config.training.learning_rate == 0.0001
    assert config.training.warmup_steps == 5
    assert config.training.weight_decay == 0.01
    assert config.training.save_steps == config.training.eval_steps == 25
    assert config.generation.max_new_tokens == 192
    assert config.evaluation.thresholds == smoke.evaluation.thresholds
    assert config.data.processed_dir != smoke.data.processed_dir
    assert config.training.output_dir != smoke.training.output_dir
    assert config.evaluation.output_dir != smoke.evaluation.output_dir
    assert config.deployment.merged_model_dir != smoke.deployment.merged_model_dir


def test_qwen_half_b_candidate_profile_uses_quality_gates_without_release_claim() -> None:
    config = load_config(ROOT / "configs" / "qwen-0.5b-candidate-local.yaml")
    quality = load_config(ROOT / "configs" / "qwen-7b-qlora-quality.yaml")

    assert config.project_name == ("tickettune-qwen-0.5b-candidate-local-not-release-qualified")
    assert config.model.name_or_path == "Qwen/Qwen2.5-0.5B-Instruct"
    assert config.model.revision == "7ae557604adf67be50417f59c2c2f167def9a775"
    assert config.model.torch_dtype == "float32"
    assert config.model.max_sequence_length == 512
    assert config.lora.method == "lora"
    assert config.training.use_cpu is True
    assert config.training.num_train_epochs == 3.0
    assert config.training.gradient_accumulation_steps == 8
    assert config.training.learning_rate == 0.0001
    assert config.training.save_steps == config.training.eval_steps == 50
    assert config.data.qualification is None
    assert config.data.source_path == (ROOT / "data/qualified/support_tickets.jsonl").resolve()
    assert config.evaluation.thresholds == quality.evaluation.thresholds
    assert "candidate-local" in str(config.data.processed_dir)
    assert "candidate-local" in str(config.training.output_dir)


def test_quality_profile_requires_resolved_review_evidence() -> None:
    config = load_config(ROOT / "configs" / "qwen-7b-qlora-quality.yaml")

    assert config.data.qualification is not None
    assert config.data.qualification.required is True
    assert (
        config.data.qualification.review_manifest
        == (ROOT / "data/qualified/review-evidence/review-manifest.bound.json").resolve()
    )


def test_local_profiles_do_not_require_review_evidence() -> None:
    for profile in (
        "smoke.yaml",
        "cpu-smoke.yaml",
        "qwen-0.5b-lora-local.yaml",
        "qwen-0.5b-candidate-local.yaml",
    ):
        assert load_config(ROOT / "configs" / profile).data.qualification is None


def test_required_qualification_needs_a_review_manifest() -> None:
    raw = _raw_smoke()
    data = raw["data"]
    assert isinstance(data, dict)
    data["qualification"] = {"required": True}

    with pytest.raises(ValidationError, match="review_manifest is required"):
        FineTuneConfig.model_validate(raw)


def test_accelerator_profiles_do_not_force_cpu() -> None:
    for profile in PROFILE_REVISIONS:
        if profile not in CPU_PROFILES:
            assert load_config(ROOT / "configs" / profile).training.use_cpu is False


def test_load_config_resolves_paths_from_explicit_project_root(tmp_path: Path) -> None:
    config_path = tmp_path / "elsewhere" / "profile.yaml"
    config_path.parent.mkdir()
    config_path.write_text(
        (ROOT / "configs" / "smoke.yaml").read_text(encoding="utf-8"), encoding="utf-8"
    )

    config = load_config(config_path, project_root=tmp_path)

    assert config.data.source_path == (tmp_path / "data/raw/support_tickets.jsonl").resolve()
    assert config.training.output_dir == (tmp_path / "artifacts/smoke").resolve()


def test_unknown_nested_keys_are_rejected() -> None:
    raw = _raw_smoke()
    training = raw["training"]
    assert isinstance(training, dict)
    training["gradient_accumlation_steps"] = 4

    with pytest.raises(ValidationError, match="gradient_accumlation_steps"):
        FineTuneConfig.model_validate(raw)


@pytest.mark.parametrize("literal", [".inf", ".nan"])
def test_yaml_config_rejects_non_finite_numbers(tmp_path: Path, literal: str) -> None:
    source = (ROOT / "configs" / "smoke.yaml").read_text(encoding="utf-8")
    path = tmp_path / "non-finite.yaml"
    path.write_text(
        source.replace("parameters_b: 0.5", f"parameters_b: {literal}"),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="finite number"):
        load_config(path, project_root=ROOT)


@pytest.mark.parametrize(
    ("original", "duplicated"),
    [
        ("seed: 42", "seed: 42\nseed: 42"),
        (
            "  trust_remote_code: false",
            "  trust_remote_code: true\n  trust_remote_code: false",
        ),
    ],
)
def test_yaml_config_rejects_duplicate_mapping_keys(
    tmp_path: Path,
    original: str,
    duplicated: str,
) -> None:
    source = (ROOT / "configs" / "smoke.yaml").read_text(encoding="utf-8")
    assert source.count(original) == 1
    path = tmp_path / "duplicate-key.yaml"
    path.write_text(source.replace(original, duplicated, 1), encoding="utf-8")

    with pytest.raises(ConstructorError, match="duplicate mapping key"):
        load_config(path, project_root=ROOT)


def test_bf16_and_fp16_are_mutually_exclusive() -> None:
    raw = _raw_smoke()
    training = raw["training"]
    assert isinstance(training, dict)
    training["bf16"] = True
    training["fp16"] = True

    with pytest.raises(ValidationError, match="cannot both be enabled"):
        FineTuneConfig.model_validate(raw)


def test_completion_only_loss_cannot_be_disabled() -> None:
    raw = _raw_smoke()
    training = raw["training"]
    assert isinstance(training, dict)
    training["completion_only_loss"] = False

    with pytest.raises(ValidationError, match="completion_only_loss=true"):
        FineTuneConfig.model_validate(raw)


def test_sampling_requires_nonzero_temperature() -> None:
    raw = _raw_smoke()
    generation = raw["generation"]
    assert isinstance(generation, dict)
    generation["do_sample"] = True

    with pytest.raises(ValidationError, match="temperature must be greater than zero"):
        FineTuneConfig.model_validate(raw)


def test_training_precision_must_match_model_dtype() -> None:
    raw = _raw_smoke()
    model = raw["model"]
    training = raw["training"]
    assert isinstance(model, dict)
    assert isinstance(training, dict)
    model["torch_dtype"] = "float16"
    training["bf16"] = True

    with pytest.raises(ValidationError, match="bf16 training requires"):
        FineTuneConfig.model_validate(raw)


def test_fp16_training_must_match_model_dtype() -> None:
    raw = _raw_smoke()
    model = raw["model"]
    training = raw["training"]
    assert isinstance(model, dict)
    assert isinstance(training, dict)
    model["torch_dtype"] = "bfloat16"
    training["fp16"] = True

    with pytest.raises(ValidationError, match="fp16 training requires"):
        FineTuneConfig.model_validate(raw)


def test_lora_target_lists_are_nonempty_and_unique() -> None:
    assert LoraConfig(target_modules=["q_proj", "v_proj"]).target_modules == [
        "q_proj",
        "v_proj",
    ]
    with pytest.raises(ValidationError, match="at least one non-empty"):
        LoraConfig(target_modules=[])
    with pytest.raises(ValidationError, match="must not contain duplicates"):
        LoraConfig(target_modules=["q_proj", "q_proj"])


def test_duplicate_reporting_integrations_are_rejected() -> None:
    raw = _raw_smoke()
    training = raw["training"]
    assert isinstance(training, dict)
    training["report_to"] = ["tensorboard", "tensorboard"]

    with pytest.raises(ValidationError, match="report_to integrations"):
        FineTuneConfig.model_validate(raw)


def test_positive_temperature_allows_sampling() -> None:
    raw = _raw_smoke()
    generation = raw["generation"]
    assert isinstance(generation, dict)
    generation["do_sample"] = True
    generation["temperature"] = 0.7

    assert FineTuneConfig.model_validate(raw).generation.do_sample is True


def test_deployment_context_reserves_prompt_budget() -> None:
    raw = _raw_smoke()
    generation = raw["generation"]
    deployment = raw["deployment"]
    assert isinstance(generation, dict)
    assert isinstance(deployment, dict)
    generation["max_new_tokens"] = 256
    deployment["max_model_len"] = 383

    with pytest.raises(ValidationError, match="reserve at least 128 prompt tokens"):
        FineTuneConfig.model_validate(raw)

    deployment["max_model_len"] = 384
    assert FineTuneConfig.model_validate(raw).deployment.max_model_len == 384


def test_meta_llama_deployment_names_are_license_clear() -> None:
    path = ROOT / "configs" / "llama-8b-qlora.yaml"
    config = load_config(path)

    assert config.deployment.ollama_model_name == "llama-tickettune-8b"
    assert config.deployment.vllm_served_model_name == "llama-tickettune-8b"

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    deployment = raw["deployment"]
    assert isinstance(deployment, dict)
    deployment["ollama_model_name"] = "tickettune-llama-8b"
    deployment["vllm_served_model_name"] = "TicketTune-Llama-8B"

    with pytest.raises(ValidationError, match="must start with 'llama'"):
        FineTuneConfig.model_validate(raw)


def test_absolute_paths_remain_absolute(tmp_path: Path) -> None:
    raw = _raw_smoke()
    data = raw["data"]
    assert isinstance(data, dict)
    absolute_source = tmp_path / "source.jsonl"
    data["source_path"] = str(absolute_source)

    config = FineTuneConfig.model_validate(raw).resolve_paths(ROOT)

    assert config.data.source_path == absolute_source


def test_config_root_fallbacks_do_not_depend_on_cwd(tmp_path: Path) -> None:
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    nested_profile = configs_dir / "profile.yaml"
    nested_profile.write_text(
        (ROOT / "configs" / "smoke.yaml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    plain_profile = tmp_path / "plain-profile.yaml"
    plain_profile.write_text(nested_profile.read_text(encoding="utf-8"), encoding="utf-8")

    assert (
        load_config(nested_profile).data.source_path
        == (tmp_path / "data/raw/support_tickets.jsonl").resolve()
    )
    assert (
        load_config(plain_profile).data.source_path
        == (tmp_path / "data/raw/support_tickets.jsonl").resolve()
    )


@pytest.mark.parametrize("content", ["", "- not-a-mapping\n"])
def test_config_root_must_be_a_nonempty_mapping(tmp_path: Path, content: str) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match=r"configuration (file is empty|root must be a mapping)"):
        load_config(path)


def test_evaluation_thresholds_cover_all_structured_quality_signals() -> None:
    config = load_config(ROOT / "configs" / "smoke.yaml")

    assert config.evaluation.thresholds.strict_json_rate == 0.95
    assert config.evaluation.thresholds.schema_valid_rate == 0.95
    assert config.evaluation.thresholds.category_accuracy == 0.85
    assert config.evaluation.thresholds.priority_accuracy == 0.85
    assert config.evaluation.thresholds.sentiment_accuracy == 0.8
    assert config.evaluation.thresholds.response_policy_rate == 0.95
