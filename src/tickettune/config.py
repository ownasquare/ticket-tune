"""Strict, repository-relative configuration for TicketTune.

The module deliberately depends only on Pydantic and PyYAML.  Training and
serving libraries are loaded later by their owning execution paths, keeping
configuration inspection safe on machines without a GPU toolchain.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate keys at every mapping depth."""

    def construct_mapping(
        self,
        node: MappingNode,
        deep: bool = False,
    ) -> dict[object, object]:
        self.flatten_mapping(node)
        mapping: dict[object, object] = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                duplicate = key in mapping
            except TypeError as exc:
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    "found an unhashable mapping key",
                    key_node.start_mark,
                ) from exc
            if duplicate:
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"duplicate mapping key {key!r}",
                    key_node.start_mark,
                )
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


class StrictModel(BaseModel):
    """Base model that rejects misspelled or obsolete configuration keys."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
        validate_default=True,
        allow_inf_nan=False,
    )


class SplitConfig(StrictModel):
    """Fractions used for deterministic train/validation/test splitting."""

    train: float = Field(default=0.75, gt=0.0, lt=1.0)
    validation: float = Field(default=0.125, gt=0.0, lt=1.0)
    test: float = Field(default=0.125, gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def validate_total(self) -> Self:
        total = self.train + self.validation + self.test
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"split fractions must sum to 1.0; received {total:.12g}")
        return self


class ModelConfig(StrictModel):
    """Base-model identity and tokenizer/model loading controls."""

    name_or_path: str = Field(min_length=1)
    revision: str = Field(default="main", min_length=1)
    parameters_b: float = Field(gt=0.0)
    torch_dtype: Literal["auto", "float16", "bfloat16", "float32"] = "auto"
    trust_remote_code: bool = False
    max_sequence_length: int = Field(default=2048, ge=128, le=131_072)
    eos_token: str | None = None

    @model_validator(mode="after")
    def validate_remote_revision_pin(self) -> Self:
        model_id = self.name_or_path.strip()
        is_explicit_local = model_id.startswith(("./", "../", "/", "~")) or re.match(
            r"^[A-Za-z]:[\\/]", model_id
        )
        looks_like_hub_id = re.fullmatch(r"[^/\s]+/[^/\s]+", model_id) is not None
        if (
            not is_explicit_local
            and looks_like_hub_id
            and re.fullmatch(r"[0-9a-fA-F]{40}", self.revision) is None
        ):
            raise ValueError(
                "remote Hugging Face models require revision to be a full 40-character commit SHA"
            )
        return self


class DataConfig(StrictModel):
    """Source and generated-dataset locations."""

    source_path: Path
    processed_dir: Path
    splits: SplitConfig = Field(default_factory=SplitConfig)
    qualification: DataQualificationConfig | None = None


class DataQualificationConfig(StrictModel):
    """Optional fail-closed human-review evidence required before training."""

    required: bool = False
    review_manifest: Path | None = None

    @model_validator(mode="after")
    def validate_required_manifest(self) -> Self:
        if self.required and self.review_manifest is None:
            raise ValueError("review_manifest is required when data qualification is required")
        if not self.required and self.review_manifest is not None:
            raise ValueError("review_manifest requires data qualification required=true")
        return self


class LoraConfig(StrictModel):
    """PEFT adapter and quantization method controls."""

    method: Literal["lora", "qlora"] = "lora"
    r: int = Field(default=16, ge=1, le=1024)
    alpha: int = Field(default=32, ge=1, le=4096)
    dropout: float = Field(default=0.05, ge=0.0, lt=1.0)
    bias: Literal["none", "all", "lora_only"] = "none"
    target_modules: Literal["all-linear"] | list[str] = "all-linear"
    use_rslora: bool = False

    @model_validator(mode="after")
    def validate_targets(self) -> Self:
        if isinstance(self.target_modules, list):
            cleaned = [target.strip() for target in self.target_modules]
            if not cleaned or any(not target for target in cleaned):
                raise ValueError("target_modules must contain at least one non-empty name")
            if len(set(cleaned)) != len(cleaned):
                raise ValueError("target_modules must not contain duplicates")
        return self


class TrainingConfig(StrictModel):
    """TRL SFT controls shared by smoke and portfolio-scale profiles."""

    output_dir: Path
    num_train_epochs: float = Field(default=1.0, gt=0.0)
    max_steps: int | None = Field(default=None, ge=1)
    per_device_train_batch_size: int = Field(default=1, ge=1)
    per_device_eval_batch_size: int = Field(default=1, ge=1)
    gradient_accumulation_steps: int = Field(default=8, ge=1)
    learning_rate: float = Field(default=2e-4, gt=0.0)
    warmup_steps: int = Field(default=0, ge=0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    logging_steps: int = Field(default=1, ge=1)
    save_steps: int = Field(default=25, ge=1)
    eval_steps: int = Field(default=25, ge=1)
    gradient_checkpointing: bool = True
    bf16: bool = False
    fp16: bool = False
    packing: bool = False
    completion_only_loss: bool = True
    optim: str = Field(default="adamw_torch", min_length=1)
    report_to: list[str] = Field(default_factory=list)
    use_cpu: bool = False

    @model_validator(mode="after")
    def validate_precision_and_loss(self) -> Self:
        if self.bf16 and self.fp16:
            raise ValueError("bf16 and fp16 cannot both be enabled")
        if not self.completion_only_loss:
            raise ValueError(
                "TicketTune requires completion_only_loss=true so user prompts are not "
                "training targets"
            )
        if len(set(self.report_to)) != len(self.report_to):
            raise ValueError("report_to integrations must not contain duplicates")
        return self


class GenerationConfig(StrictModel):
    """Deterministic-by-default generation settings."""

    max_new_tokens: int = Field(default=256, ge=16, le=8192)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    do_sample: bool = False
    repetition_penalty: float = Field(default=1.0, ge=0.5, le=2.0)

    @model_validator(mode="after")
    def validate_sampling(self) -> Self:
        if self.do_sample and self.temperature == 0:
            raise ValueError("temperature must be greater than zero when do_sample=true")
        return self


class EvaluationThresholds(StrictModel):
    """Minimum held-out metrics required for a successful evaluation."""

    strict_json_rate: float = Field(default=0.95, ge=0.0, le=1.0)
    schema_valid_rate: float = Field(default=0.95, ge=0.0, le=1.0)
    category_accuracy: float = Field(default=0.85, ge=0.0, le=1.0)
    priority_accuracy: float = Field(default=0.85, ge=0.0, le=1.0)
    sentiment_accuracy: float = Field(default=0.80, ge=0.0, le=1.0)
    response_policy_rate: float = Field(default=0.95, ge=0.0, le=1.0)


class EvaluationConfig(StrictModel):
    """Evaluation output location and regression gates."""

    output_dir: Path
    thresholds: EvaluationThresholds = Field(default_factory=EvaluationThresholds)


class DeploymentConfig(StrictModel):
    """Ollama export and vLLM serving-plan settings."""

    merged_model_dir: Path
    ollama_model_name: str = Field(default="tickettune", pattern=r"^[a-z0-9][a-z0-9._-]*$")
    ollama_quantization: Literal["Q4_K_M", "Q5_K_M", "Q8_0", "F16"] = "Q4_K_M"
    vllm_served_model_name: str = Field(default="tickettune", min_length=1)
    vllm_host: str = "127.0.0.1"
    vllm_port: int = Field(default=8000, ge=1, le=65_535)
    vllm_dtype: Literal["auto", "half", "bfloat16", "float", "float16", "float32"] = "auto"
    max_model_len: int = Field(default=2048, ge=128, le=131_072)
    tensor_parallel_size: int = Field(default=1, ge=1)
    gpu_memory_utilization: float = Field(default=0.9, gt=0.0, le=1.0)


class FineTuneConfig(StrictModel):
    """Complete TicketTune pipeline configuration."""

    project_name: str = Field(default="tickettune", min_length=1)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    model: ModelConfig
    data: DataConfig
    lora: LoraConfig
    training: TrainingConfig
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    evaluation: EvaluationConfig
    deployment: DeploymentConfig

    @model_validator(mode="after")
    def validate_precision_contract(self) -> Self:
        if self.training.bf16 and self.model.torch_dtype not in {"auto", "bfloat16"}:
            raise ValueError("bf16 training requires model.torch_dtype to be auto or bfloat16")
        if self.training.fp16 and self.model.torch_dtype not in {"auto", "float16"}:
            raise ValueError("fp16 training requires model.torch_dtype to be auto or float16")
        reserved_prompt_tokens = self.deployment.max_model_len - self.generation.max_new_tokens
        if reserved_prompt_tokens < 128:
            raise ValueError(
                "deployment.max_model_len must reserve at least 128 prompt tokens "
                "after generation.max_new_tokens"
            )
        if self.model.name_or_path.casefold().startswith("meta-llama/"):
            deployment_names = {
                "ollama_model_name": self.deployment.ollama_model_name,
                "vllm_served_model_name": self.deployment.vllm_served_model_name,
            }
            invalid_names = [
                field
                for field, value in deployment_names.items()
                if not value.casefold().startswith("llama")
            ]
            if invalid_names:
                raise ValueError(
                    "Meta Llama deployment names must start with 'llama': "
                    + ", ".join(invalid_names)
                )
        return self

    def resolve_paths(self, project_root: Path) -> FineTuneConfig:
        """Return a copy with every configured path anchored to ``project_root``."""

        root = project_root.expanduser().resolve()

        def resolved(path: Path) -> Path:
            expanded = path.expanduser()
            return expanded.resolve() if expanded.is_absolute() else (root / expanded).resolve()

        return self.model_copy(
            update={
                "data": self.data.model_copy(
                    update={
                        "source_path": resolved(self.data.source_path),
                        "processed_dir": resolved(self.data.processed_dir),
                        "qualification": (
                            self.data.qualification.model_copy(
                                update={
                                    "review_manifest": resolved(
                                        self.data.qualification.review_manifest
                                    )
                                }
                            )
                            if self.data.qualification is not None
                            and self.data.qualification.review_manifest is not None
                            else self.data.qualification
                        ),
                    }
                ),
                "training": self.training.model_copy(
                    update={"output_dir": resolved(self.training.output_dir)}
                ),
                "evaluation": self.evaluation.model_copy(
                    update={"output_dir": resolved(self.evaluation.output_dir)}
                ),
                "deployment": self.deployment.model_copy(
                    update={"merged_model_dir": resolved(self.deployment.merged_model_dir)}
                ),
            }
        )


def _discover_project_root(config_path: Path) -> Path:
    """Find the nearest package root without depending on the caller's CWD."""

    for candidate in (config_path.parent, *config_path.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    if config_path.parent.name == "configs":
        return config_path.parent.parent
    return config_path.parent


def load_config(path: Path, *, project_root: Path | None = None) -> FineTuneConfig:
    """Load and validate YAML, rejecting unknown keys at every nesting level."""

    config_path = path.expanduser().resolve()
    loader = _UniqueKeySafeLoader(config_path.read_text(encoding="utf-8"))
    try:
        raw = loader.get_single_data()
    finally:
        loader.dispose()  # type: ignore[no-untyped-call]  # PyYAML lacks this annotation.
    if raw is None:
        raise ValueError(f"configuration file is empty: {config_path}")
    if not isinstance(raw, dict):
        raise ValueError(f"configuration root must be a mapping: {config_path}")
    config = FineTuneConfig.model_validate(raw)
    root = project_root or _discover_project_root(config_path)
    return config.resolve_paths(root)
