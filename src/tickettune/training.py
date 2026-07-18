"""LoRA and QLoRA training orchestration with lazy heavyweight imports."""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .data import (
    DatasetIntegrityError,
    DatasetVerification,
    PreparedSplitSnapshot,
    SplitName,
    materialize_prepared_split_snapshots,
    read_prepared_split_snapshot,
    verify_prepared_dataset,
    verify_prepared_split_snapshot,
)
from .qualification import (
    DatasetQualificationError,
    DatasetQualificationReport,
    qualify_dataset,
)
from .run_manifest import (
    ArtifactDigest,
    RunManifest,
    artifact_digest,
    build_run_manifest,
    canonical_json_bytes,
    json_safe,
    make_run_id,
    sanitize_error,
    sha256_bytes,
    sha256_file,
    write_manifest,
)
from .strict_json import StrictJSONError, loads_strict

if TYPE_CHECKING:
    from .config import FineTuneConfig
    from .hardware import HardwarePreflight

# Qwen chat-template terminator; this public tokenizer sentinel is not a credential.
QWEN_EOS_TOKEN = "<|im_end|>"  # noqa: S105  # nosec B105
Sha256Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class ResumeCompatibilityError(ValueError):
    """A requested checkpoint is not an immutable compatible TicketTune run."""


class TrainingQualification(BaseModel):
    """Framework-light qualification evidence resolved before training."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    required: bool
    schema_version: Literal["1.1", "1.2"] | None = None
    review_manifest_path: str | None = None
    status: Literal["not_required", "qualified", "rejected", "invalid"]
    dataset_tier: Literal[
        "not_required",
        "unknown",
        "portfolio_smoke",
        "qualification_candidate",
    ]
    source_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    review_manifest_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    report_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    prepared_manifest_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    holdout_freeze_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    reviewer_packet_sha256: tuple[Sha256Digest, ...] = ()
    reviewer_ids: tuple[str, ...] = ()
    held_out_count: int = Field(default=0, ge=0)
    held_out_ids: tuple[str, ...] = ()
    error: str | None = None


class TrainingPlan(BaseModel):
    """Serializable, download-free description of a requested training run."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    project_name: str
    model_name_or_path: str
    model_revision: str | None
    method: Literal["lora", "qlora"]
    output_dir: str
    train_dataset: str
    validation_dataset: str
    dataset_counts: dict[str, int]
    dataset_sha256: dict[str, str]
    missing_datasets: tuple[str, ...]
    dataset_manifest_path: str
    dataset_manifest_status: Literal["verified", "missing", "invalid"]
    dataset_manifest_sha256: str | None
    dataset_manifest_error: str | None
    qualification: TrainingQualification
    precision: Literal["bf16", "fp16", "fp32"]
    max_length: int
    target_modules: Literal["all-linear"] | tuple[str, ...] = "all-linear"
    completion_only_loss: Literal[True] = True
    eos_token: str | None
    quantization: dict[str, object] | None
    sft_arguments: dict[str, object]
    allow_download: bool
    resume_from_checkpoint: str | None
    proof_boundary: str = (
        "plan_only: no model, tokenizer, dataset framework, or accelerator was loaded"
    )


class TrainingTokenBudget(BaseModel):
    """Observed chat-template token bounds checked before model loading."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    examples: int = Field(ge=1)
    max_length: int = Field(ge=1)
    max_full_tokens: int = Field(ge=1)
    min_completion_tokens: int = Field(ge=1)


class TrainingResult(BaseModel):
    """Machine-readable result shared by the CLI and tests."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    plan: TrainingPlan
    executed: bool
    run_id: str | None = None
    run_dir: str | None = None
    adapter_path: str | None = None
    manifest_path: str | None = None
    latest_pointer_path: str | None = None
    metrics: dict[str, float | int | str | bool | None] = Field(default_factory=dict)
    token_budget: TrainingTokenBudget | None = None
    proof_boundary: str


@dataclass(frozen=True)
class TrainingLibraries:
    """Injected heavy-library adapter used to keep offline tests lightweight."""

    torch: Any
    Dataset: Any
    AutoModelForCausalLM: Any
    AutoTokenizer: Any
    BitsAndBytesConfig: Any
    PeftLoraConfig: Any
    prepare_model_for_kbit_training: Any
    SFTConfig: Any
    SFTTrainer: Any


def _load_training_libraries() -> TrainingLibraries:
    """Import the optional training stack only for a real execution."""

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig as PeftLoraConfig
        from peft import prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl.trainer.sft_config import SFTConfig
        from trl.trainer.sft_trainer import SFTTrainer
    except ImportError as exc:  # pragma: no cover - environment-specific guidance
        raise RuntimeError(
            "Training dependencies are not installed. Run `uv sync --extra train` first."
        ) from exc
    return TrainingLibraries(
        torch=torch,
        Dataset=Dataset,
        AutoModelForCausalLM=AutoModelForCausalLM,
        AutoTokenizer=AutoTokenizer,
        BitsAndBytesConfig=BitsAndBytesConfig,
        PeftLoraConfig=PeftLoraConfig,
        prepare_model_for_kbit_training=prepare_model_for_kbit_training,
        SFTConfig=SFTConfig,
        SFTTrainer=SFTTrainer,
    )


def resolve_target_modules(_model_name_or_family: str | None = None) -> Literal["all-linear"]:
    """Return the architecture-independent default for legacy callers."""

    return "all-linear"


def _split_path(config: FineTuneConfig, split: str) -> Path:
    return Path(config.data.processed_dir) / f"{split}.jsonl"


def _verify_training_dataset(config: FineTuneConfig) -> DatasetVerification:
    return verify_prepared_dataset(
        config.data.source_path,
        config.data.processed_dir,
        seed=config.seed,
        splits=config.data.splits,
        required_splits=("train", "validation", "test"),
    )


def _qualification_from_report(
    report: DatasetQualificationReport,
    *,
    status: Literal["qualified", "rejected"],
    error: str | None = None,
) -> TrainingQualification:
    return TrainingQualification(
        required=True,
        schema_version=report.schema_version,
        review_manifest_path=str(report.review_manifest_path),
        status=status,
        dataset_tier=report.dataset_tier,
        source_sha256=report.source_sha256,
        review_manifest_sha256=report.review_manifest_sha256,
        report_sha256=sha256_bytes(json_safe(report)),
        prepared_manifest_sha256=report.prepared_manifest_sha256,
        holdout_freeze_sha256=report.holdout_freeze_sha256,
        reviewer_packet_sha256=report.reviewer_packet_sha256,
        reviewer_ids=report.reviewer_ids,
        held_out_count=report.held_out_count,
        held_out_ids=report.held_out_ids,
        error=error,
    )


def _bind_qualification_to_test_split(
    qualification: TrainingQualification,
    verification: DatasetVerification | None,
) -> TrainingQualification:
    """Require the reviewed holdout cohort to be the exact prepared test split."""

    if qualification.status != "qualified":
        return qualification
    if verification is None:
        return qualification.model_copy(
            update={
                "status": "invalid",
                "error": (
                    "qualified held-out IDs cannot be bound without a verified prepared test split"
                ),
            }
        )
    if qualification.source_sha256 != verification.source_sha256:
        return qualification.model_copy(
            update={
                "status": "invalid",
                "error": (
                    "qualified source SHA-256 must equal the verified prepared-dataset "
                    "source SHA-256"
                ),
            }
        )
    if qualification.prepared_manifest_sha256 != verification.manifest_sha256:
        return qualification.model_copy(
            update={
                "status": "invalid",
                "error": (
                    "qualified prepared-manifest SHA-256 must equal the verified "
                    "prepared-dataset manifest SHA-256"
                ),
            }
        )
    test_ids = verification.split_ids.get("test")
    test_count = verification.split_counts.get("test")
    if test_count is None or qualification.held_out_count != test_count:
        return qualification.model_copy(
            update={
                "status": "invalid",
                "error": (
                    "qualified held-out count must equal the verified prepared test split count"
                ),
            }
        )
    if test_ids is None or qualification.held_out_ids != test_ids:
        return qualification.model_copy(
            update={
                "status": "invalid",
                "error": (
                    "review manifest held_out_ids must exactly equal the prepared test "
                    "split ordered IDs"
                ),
            }
        )
    return qualification


def _qualification_plan(
    config: FineTuneConfig,
    *,
    fallback_source_sha256: str | None,
) -> TrainingQualification:
    configured = config.data.qualification
    if configured is None or not configured.required:
        return TrainingQualification(
            required=False,
            status="not_required",
            dataset_tier="not_required",
        )
    if configured.review_manifest is None:
        return TrainingQualification(
            required=True,
            status="invalid",
            dataset_tier="unknown",
            source_sha256=fallback_source_sha256,
            error="required data qualification has no review manifest",
        )
    try:
        report = qualify_dataset(
            config.data.source_path,
            configured.review_manifest,
            enforce=True,
        )
    except DatasetQualificationError as exc:
        if exc.report is not None:
            return _qualification_from_report(
                exc.report,
                status="rejected",
                error=str(exc),
            )
        return TrainingQualification(
            required=True,
            review_manifest_path=str(configured.review_manifest),
            status="invalid",
            dataset_tier="unknown",
            source_sha256=fallback_source_sha256,
            error=str(exc),
        )
    return _qualification_from_report(report, status="qualified")


def _require_stable_qualification(
    config: FineTuneConfig,
    planned: TrainingQualification,
    verification: DatasetVerification,
) -> None:
    configured = config.data.qualification
    if configured is None or not configured.required:
        return
    if configured.review_manifest is None:
        raise DatasetQualificationError("required data qualification has no review manifest")
    if planned.status != "qualified":
        raise DatasetQualificationError(
            planned.error or "required dataset qualification is not release-qualified"
        )
    report = qualify_dataset(
        config.data.source_path,
        configured.review_manifest,
        enforce=True,
    )
    current = _bind_qualification_to_test_split(
        _qualification_from_report(report, status="qualified"),
        verification,
    )
    if current.status != "qualified":
        raise DatasetQualificationError(
            current.error or "reviewed holdout cohort does not match the prepared test split"
        )
    planned_identity = (
        planned.status,
        planned.schema_version,
        planned.source_sha256,
        planned.review_manifest_sha256,
        planned.report_sha256,
        planned.prepared_manifest_sha256,
        planned.holdout_freeze_sha256,
        planned.reviewer_packet_sha256,
        planned.reviewer_ids,
        planned.held_out_count,
        planned.held_out_ids,
    )
    current_identity = (
        current.status,
        current.schema_version,
        current.source_sha256,
        current.review_manifest_sha256,
        current.report_sha256,
        current.prepared_manifest_sha256,
        current.holdout_freeze_sha256,
        current.reviewer_packet_sha256,
        current.reviewer_ids,
        current.held_out_count,
        current.held_out_ids,
    )
    if planned_identity != current_identity:
        raise DatasetQualificationError(
            "dataset qualification evidence changed after plan creation"
        )


def _precision(config: FineTuneConfig) -> Literal["bf16", "fp16", "fp32"]:
    if config.training.bf16:
        return "bf16"
    if config.training.fp16:
        return "fp16"
    return "fp32"


def _eos_token(config: FineTuneConfig) -> str | None:
    if "qwen" in config.model.name_or_path.casefold():
        return QWEN_EOS_TOKEN
    return config.model.eos_token


def _quantization_plan(config: FineTuneConfig) -> dict[str, object] | None:
    if config.lora.method != "qlora":
        return None
    return {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": _precision(config),
    }


def _public_sft_arguments(config: FineTuneConfig, *, has_evaluation: bool) -> dict[str, object]:
    """Return JSON-safe TRL 1.7 argument names for dry-run inspection."""

    arguments: dict[str, object] = {
        "output_dir": str(config.training.output_dir),
        "num_train_epochs": config.training.num_train_epochs,
        "max_steps": config.training.max_steps if config.training.max_steps is not None else -1,
        "per_device_train_batch_size": config.training.per_device_train_batch_size,
        "per_device_eval_batch_size": config.training.per_device_eval_batch_size,
        "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
        "learning_rate": config.training.learning_rate,
        "warmup_steps": config.training.warmup_steps,
        "weight_decay": config.training.weight_decay,
        "logging_steps": config.training.logging_steps,
        "save_steps": config.training.save_steps,
        "gradient_checkpointing": config.training.gradient_checkpointing,
        "bf16": config.training.bf16,
        "fp16": config.training.fp16,
        "packing": config.training.packing,
        "completion_only_loss": True,
        "optim": config.training.optim,
        "report_to": list(config.training.report_to),
        "max_length": config.model.max_sequence_length,
        "eval_strategy": "steps" if has_evaluation else "no",
        "seed": config.seed,
        "data_seed": config.seed,
        "use_cpu": config.training.use_cpu,
    }
    if has_evaluation:
        arguments["eval_steps"] = config.training.eval_steps
        arguments["load_best_model_at_end"] = True
        arguments["metric_for_best_model"] = "eval_loss"
        arguments["greater_is_better"] = False
    eos_token = _eos_token(config)
    if eos_token:
        arguments["eos_token"] = eos_token
    return arguments


def build_training_plan(
    config: FineTuneConfig,
    *,
    allow_download: bool = False,
    resume_from_checkpoint: Path | str | None = None,
) -> TrainingPlan:
    """Resolve a complete execution plan without importing the training stack."""

    train_path = _split_path(config, "train")
    validation_path = _split_path(config, "validation")
    paths = {"train": train_path, "validation": validation_path}
    counts = {name: 0 for name in paths}
    hashes: dict[str, str] = {}
    missing = tuple(str(path) for path in paths.values() if not path.is_file())
    manifest_path = Path(config.data.processed_dir) / "manifest.json"
    manifest_status: Literal["verified", "missing", "invalid"]
    manifest_sha256: str | None = None
    manifest_error: str | None = None
    verification: DatasetVerification | None = None
    try:
        verification = _verify_training_dataset(config)
    except DatasetIntegrityError as exc:
        manifest_status = "missing" if not manifest_path.is_file() else "invalid"
        manifest_error = str(exc)
    else:
        manifest_status = "verified"
        manifest_sha256 = verification.manifest_sha256
        counts = {str(name): count for name, count in verification.split_counts.items()}
        hashes = {
            "source": verification.source_sha256,
            "manifest": verification.manifest_sha256,
            **{str(name): digest for name, digest in verification.split_sha256.items()},
        }
    qualification = _qualification_plan(
        config,
        fallback_source_sha256=hashes.get("source"),
    )
    qualification = _bind_qualification_to_test_split(qualification, verification)
    if qualification.status == "qualified":
        if qualification.review_manifest_sha256 is None or qualification.report_sha256 is None:
            raise RuntimeError("qualified training plan is missing immutable evidence hashes")
        hashes["qualification_review_manifest"] = qualification.review_manifest_sha256
        hashes["qualification_report"] = qualification.report_sha256
    has_evaluation = validation_path.is_file() and counts["validation"] > 0
    return TrainingPlan(
        project_name=config.project_name,
        model_name_or_path=config.model.name_or_path,
        model_revision=config.model.revision,
        method=config.lora.method,
        output_dir=str(config.training.output_dir),
        train_dataset=str(train_path),
        validation_dataset=str(validation_path),
        dataset_counts=counts,
        dataset_sha256=hashes,
        missing_datasets=missing,
        dataset_manifest_path=str(manifest_path),
        dataset_manifest_status=manifest_status,
        dataset_manifest_sha256=manifest_sha256,
        dataset_manifest_error=manifest_error,
        qualification=qualification,
        precision=_precision(config),
        max_length=config.model.max_sequence_length,
        target_modules=(
            tuple(config.lora.target_modules)
            if isinstance(config.lora.target_modules, list)
            else config.lora.target_modules
        ),
        eos_token=_eos_token(config),
        quantization=_quantization_plan(config),
        sft_arguments=_public_sft_arguments(config, has_evaluation=has_evaluation),
        allow_download=allow_download,
        resume_from_checkpoint=str(resume_from_checkpoint) if resume_from_checkpoint else None,
    )


def _torch_dtype(config: FineTuneConfig, libraries: TrainingLibraries) -> Any:
    precision = _precision(config)
    if precision == "bf16":
        return libraries.torch.bfloat16
    if precision == "fp16":
        return libraries.torch.float16
    return libraries.torch.float32


def build_quantization_config(config: FineTuneConfig, libraries: TrainingLibraries) -> Any | None:
    """Create the direct Transformers quantization object used by TRL."""

    if config.lora.method != "qlora":
        return None
    return libraries.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=_torch_dtype(config, libraries),
    )


def _model_init_kwargs(
    config: FineTuneConfig,
    libraries: TrainingLibraries,
    allow_download: bool,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "dtype": _torch_dtype(config, libraries),
        # TRL's model factory otherwise inserts device_map="auto" before
        # from_pretrained(). Keep placement under Trainer/Accelerate control;
        # this is also required for SFTConfig(use_cpu=True) to remain CPU-only.
        "device_map": None,
        "local_files_only": not allow_download,
        "trust_remote_code": config.model.trust_remote_code,
    }
    if config.model.revision:
        kwargs["revision"] = config.model.revision
    quantization_config = build_quantization_config(config, libraries)
    if quantization_config is not None:
        # TRL 1.7.1 accepts this direct object through SFTConfig.model_init_kwargs
        # when SFTTrainer receives a model identifier string.
        kwargs["quantization_config"] = quantization_config
    return kwargs


def _build_sft_config(
    config: FineTuneConfig,
    libraries: TrainingLibraries,
    *,
    has_evaluation: bool,
    output_dir: Path | None = None,
) -> Any:
    kwargs: dict[str, Any] = dict(_public_sft_arguments(config, has_evaluation=has_evaluation))
    kwargs.update(
        {
            "save_strategy": "steps",
            "logging_strategy": "steps",
            "do_eval": has_evaluation,
        }
    )
    if output_dir is not None:
        kwargs["output_dir"] = str(output_dir)
    return libraries.SFTConfig(**kwargs)


def _load_model(
    config: FineTuneConfig,
    libraries: TrainingLibraries,
    *,
    allow_download: bool,
) -> Any:
    """Load the exact base without TRL's implicit device-map or config lookup."""

    model = libraries.AutoModelForCausalLM.from_pretrained(
        config.model.name_or_path,
        **_model_init_kwargs(config, libraries, allow_download),
    )
    if config.lora.method == "qlora":
        model = libraries.prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=config.training.gradient_checkpointing,
        )
    return model


def _build_peft_config(config: FineTuneConfig, libraries: TrainingLibraries) -> Any:
    target_modules: str | list[str] = (
        list(config.lora.target_modules)
        if isinstance(config.lora.target_modules, list)
        else config.lora.target_modules
    )
    return libraries.PeftLoraConfig(
        task_type="CAUSAL_LM",
        base_model_name_or_path=config.model.name_or_path,
        revision=config.model.revision,
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        bias=config.lora.bias,
        target_modules=target_modules,
        use_rslora=config.lora.use_rslora,
    )


def _load_tokenizer(
    config: FineTuneConfig, libraries: TrainingLibraries, *, allow_download: bool
) -> Any:
    kwargs: dict[str, Any] = {
        "local_files_only": not allow_download,
        "trust_remote_code": config.model.trust_remote_code,
    }
    if config.model.revision:
        kwargs["revision"] = config.model.revision
    tokenizer = libraries.AutoTokenizer.from_pretrained(config.model.name_or_path, **kwargs)
    eos_token = _eos_token(config)
    if eos_token:
        tokenizer.eos_token = eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def _numeric_metrics(
    values: Mapping[str, object] | None,
) -> dict[str, float | int | str | bool | None]:
    metrics: dict[str, float | int | str | bool | None] = {}
    for key, value in (values or {}).items():
        if value is None or isinstance(value, (float, int, str, bool)):
            metrics[str(key)] = value
        elif hasattr(value, "item"):
            scalar = value.item()
            if scalar is None or isinstance(scalar, (float, int, str, bool)):
                metrics[str(key)] = scalar
    return metrics


def _run_artifacts(run_dir: Path) -> tuple[ArtifactDigest, ...]:
    """Inventory adapter and Trainer products relative to one immutable run."""

    files: list[Path] = []
    for product_dir in (run_dir / "adapter", run_dir / "dataset", run_dir / "trainer"):
        if product_dir.is_dir():
            files.extend(path for path in product_dir.rglob("*") if path.is_file())
    return tuple(
        artifact_digest(path, relative_to=run_dir)
        for path in sorted(files, key=lambda item: item.as_posix())
    )


def _snapshot_rows(snapshot: PreparedSplitSnapshot) -> list[dict[str, Any]]:
    """Parse one verified run-scoped snapshot without framework imports."""

    payload = read_prepared_split_snapshot(snapshot)
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DatasetIntegrityError(
            snapshot.manifest_path,
            f"run-scoped {snapshot.split_name} snapshot is not UTF-8",
        ) from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            raise DatasetIntegrityError(
                snapshot.manifest_path,
                f"run-scoped {snapshot.split_name} snapshot has a blank line at {line_number}",
            )
        try:
            value = loads_strict(line)
        except (json.JSONDecodeError, StrictJSONError) as exc:
            raise DatasetIntegrityError(
                snapshot.manifest_path,
                f"run-scoped {snapshot.split_name} snapshot has invalid JSON at {line_number}",
            ) from exc
        if not isinstance(value, dict):
            raise DatasetIntegrityError(
                snapshot.manifest_path,
                f"run-scoped {snapshot.split_name} snapshot row {line_number} is not an object",
            )
        rows.append(value)
    if not rows:
        raise DatasetIntegrityError(
            snapshot.manifest_path,
            f"run-scoped {snapshot.split_name} snapshot is empty",
        )
    return rows


def _load_snapshot_dataset(
    libraries: TrainingLibraries,
    snapshot: PreparedSplitSnapshot,
) -> Any:
    """Build a framework dataset from the exact verified in-memory snapshot rows."""

    rows = _snapshot_rows(snapshot)
    dataset = libraries.Dataset.from_list(rows)
    verify_prepared_split_snapshot(snapshot)
    return dataset


def _chat_token_count(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    add_generation_prompt: bool,
) -> int:
    """Count one chat-template rendering without truncation or tensor allocation."""

    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
    )
    if isinstance(encoded, Mapping):
        encoded = encoded.get("input_ids")
    shape = getattr(encoded, "shape", None)
    if shape is not None:
        if not shape:
            raise TypeError("tokenizer chat template returned a scalar token container")
        return int(shape[-1])
    if isinstance(encoded, (list, tuple)):
        if encoded and isinstance(encoded[0], (list, tuple)):
            if len(encoded) != 1:
                raise TypeError("tokenizer chat template returned an unexpected batch")
            return len(encoded[0])
        return len(encoded)
    raise TypeError("tokenizer chat template did not return countable token IDs")


def _validate_token_budget(
    tokenizer: Any,
    snapshots: Mapping[SplitName, PreparedSplitSnapshot],
    *,
    max_length: int,
) -> TrainingTokenBudget:
    """Fail before model loading when TRL would truncate any supervised completion."""

    full_counts: list[int] = []
    completion_counts: list[int] = []
    for split_name in ("train", "validation"):
        snapshot = snapshots[split_name]
        for row_number, row in enumerate(_snapshot_rows(snapshot), 1):
            prompt = row.get("prompt")
            completion = row.get("completion")
            if not isinstance(prompt, list) or not isinstance(completion, list):
                raise DatasetIntegrityError(
                    snapshot.manifest_path,
                    f"run-scoped {split_name} snapshot row {row_number} lacks chat messages",
                )
            messages = [*prompt, *completion]
            if not all(isinstance(message, dict) for message in messages):
                raise DatasetIntegrityError(
                    snapshot.manifest_path,
                    f"run-scoped {split_name} snapshot row {row_number} has invalid messages",
                )
            full_count = _chat_token_count(
                tokenizer,
                messages,
                add_generation_prompt=False,
            )
            prompt_count = _chat_token_count(
                tokenizer,
                prompt,
                add_generation_prompt=True,
            )
            identifier = str(row.get("id", f"row-{row_number}"))
            if full_count > max_length:
                raise ValueError(
                    f"training example {identifier} would truncate at {max_length} tokens "
                    f"(rendered length: {full_count})"
                )
            completion_count = full_count - prompt_count
            if completion_count <= 0:
                raise ValueError(
                    f"training example {identifier} has no label-bearing completion tokens"
                )
            full_counts.append(full_count)
            completion_counts.append(completion_count)
    return TrainingTokenBudget(
        examples=len(full_counts),
        max_length=max_length,
        max_full_tokens=max(full_counts),
        min_completion_tokens=min(completion_counts),
    )


def _verified_run_artifacts(
    run_dir: Path,
    snapshots: Mapping[SplitName, PreparedSplitSnapshot],
) -> tuple[ArtifactDigest, ...]:
    """Inventory products while proving snapshot artifacts retain planned bytes."""

    for snapshot in snapshots.values():
        verify_prepared_split_snapshot(snapshot)
    artifacts = _run_artifacts(run_dir)
    for snapshot in snapshots.values():
        verify_prepared_split_snapshot(snapshot)
    artifact_by_path = {artifact.path: artifact for artifact in artifacts}
    for snapshot in snapshots.values():
        relative = snapshot.path.relative_to(run_dir).as_posix()
        artifact = artifact_by_path.get(relative)
        if (
            artifact is None
            or artifact.sha256 != snapshot.sha256
            or artifact.size_bytes != snapshot.size_bytes
        ):
            raise DatasetIntegrityError(
                snapshot.manifest_path,
                f"run-scoped {snapshot.split_name} snapshot inventory changed",
            )
    return artifacts


def _adapter_artifacts(adapter_path: Path, output_dir: Path) -> tuple[ArtifactDigest, ...]:
    """Compatibility helper retained for callers that inventory one adapter."""

    if not adapter_path.is_dir():
        return ()
    files = sorted(path for path in adapter_path.rglob("*") if path.is_file())
    return tuple(artifact_digest(path, relative_to=output_dir) for path in files)


def _validate_resume_checkpoint(
    config: FineTuneConfig,
    plan: TrainingPlan,
    checkpoint: Path | str,
) -> Path:
    """Verify checkpoint lineage, config/data identity, and recorded bytes."""

    candidate = Path(checkpoint).expanduser()
    if candidate.is_symlink() or not candidate.is_dir():
        raise ResumeCompatibilityError(
            f"resume checkpoint must be a regular directory, not a symlink: {candidate}"
        )
    resolved = candidate.resolve()
    if re.fullmatch(r"checkpoint-[1-9][0-9]*", resolved.name) is None:
        raise ResumeCompatibilityError("resume checkpoint must be named checkpoint-<positive-step>")
    if resolved.parent.name != "trainer":
        raise ResumeCompatibilityError(
            "resume checkpoint must be inside <output>/runs/<run-id>/trainer"
        )

    run_dir = resolved.parent.parent
    expected_runs_root = (Path(config.training.output_dir) / "runs").resolve()
    if run_dir.parent != expected_runs_root:
        raise ResumeCompatibilityError(
            "resume checkpoint must belong to this profile's immutable runs directory"
        )
    manifest_path = run_dir / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ResumeCompatibilityError(
            f"resume checkpoint has no regular sibling run manifest: {manifest_path}"
        )
    try:
        manifest_payload = manifest_path.read_bytes()
        loads_strict(manifest_payload)
        manifest = RunManifest.model_validate_json(manifest_payload, strict=True)
    except (OSError, ValueError) as exc:
        raise ResumeCompatibilityError(f"resume run manifest is invalid: {exc}") from exc
    if run_dir.name != manifest.run_id:
        raise ResumeCompatibilityError("resume run directory does not match the manifest run_id")

    config_hash = sha256_bytes(json_safe(config))
    if manifest.config_sha256 != config_hash:
        raise ResumeCompatibilityError("resume config hash does not match the current profile")
    if manifest.dataset_sha256 != plan.dataset_sha256:
        raise ResumeCompatibilityError("resume dataset hashes do not match the verified dataset")
    if (
        manifest.model_name_or_path != config.model.name_or_path
        or manifest.model_revision != config.model.revision
        or manifest.method != config.lora.method
        or manifest.seed != config.seed
    ):
        raise ResumeCompatibilityError("resume model, method, revision, or seed is incompatible")

    checkpoint_prefix = f"trainer/{resolved.name}/"
    expected = {
        item.path: item for item in manifest.artifacts if item.path.startswith(checkpoint_prefix)
    }
    actual_files = sorted(path for path in resolved.rglob("*") if path.is_file())
    actual_paths: dict[str, Path] = {}
    for path in actual_files:
        if path.is_symlink():
            raise ResumeCompatibilityError(f"resume checkpoint contains a symlink: {path}")
        relative = path.relative_to(run_dir).as_posix()
        actual_paths[relative] = path
    if not expected:
        raise ResumeCompatibilityError(
            "resume checkpoint has no artifact inventory in its manifest"
        )
    if set(actual_paths) != set(expected):
        raise ResumeCompatibilityError("resume checkpoint files differ from the manifest inventory")
    for relative, path in actual_paths.items():
        recorded = expected[relative]
        if path.stat().st_size != recorded.size_bytes or sha256_file(path) != recorded.sha256:
            raise ResumeCompatibilityError(f"resume checkpoint artifact hash mismatch: {relative}")
    if "trainer_state.json" not in {path.name for path in actual_files}:
        raise ResumeCompatibilityError("resume checkpoint is missing trainer_state.json")
    return resolved


def _write_latest_pointer(output_dir: Path, manifest: RunManifest, run_dir: Path) -> Path:
    """Atomically update the explicit mutable convenience pointer after success."""

    destination = output_dir / "latest-run.json"
    if destination.exists() and (destination.is_symlink() or not destination.is_file()):
        raise FileExistsError(f"latest-run pointer must be a regular file: {destination}")
    payload = canonical_json_bytes(
        {
            "schema_version": "1.0",
            "run_id": manifest.run_id,
            "status": manifest.status,
            "run_dir": run_dir.relative_to(output_dir).as_posix(),
            "adapter_path": (run_dir / "adapter").relative_to(output_dir).as_posix(),
            "manifest_path": (run_dir / "manifest.json").relative_to(output_dir).as_posix(),
            "updated_at": datetime.now(UTC),
        },
        pretty=True,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=output_dir)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    return destination


def _peak_accelerator_memory_mb(libraries: TrainingLibraries) -> float | None:
    cuda = getattr(libraries.torch, "cuda", None)
    if cuda is None or not callable(getattr(cuda, "is_available", None)):
        return None
    if not cuda.is_available():
        return None
    max_memory_allocated = getattr(cuda, "max_memory_allocated", None)
    if not callable(max_memory_allocated):
        return None
    return round(float(max_memory_allocated()) / 1024**2, 3)


def run_training(
    config: FineTuneConfig,
    *,
    dry_run: bool = False,
    allow_download: bool = False,
    resume_from_checkpoint: Path | str | None = None,
    hardware_preflight: HardwarePreflight | None = None,
    _libraries: TrainingLibraries | None = None,
) -> TrainingResult:
    """Run SFT with a PEFT adapter, or return the exact offline plan."""

    plan = build_training_plan(
        config,
        allow_download=allow_download,
        resume_from_checkpoint=resume_from_checkpoint,
    )
    if dry_run:
        return TrainingResult(
            plan=plan,
            executed=False,
            proof_boundary=plan.proof_boundary,
        )
    if plan.missing_datasets:
        joined = ", ".join(plan.missing_datasets)
        raise FileNotFoundError(f"prepare the dataset before training; missing: {joined}")
    verification = _verify_training_dataset(config)
    current_dataset_sha256 = {
        "source": verification.source_sha256,
        "manifest": verification.manifest_sha256,
        **{str(name): digest for name, digest in verification.split_sha256.items()},
    }
    if any(
        plan.dataset_sha256.get(name) != digest for name, digest in current_dataset_sha256.items()
    ):
        raise DatasetIntegrityError(
            verification.manifest_path,
            "verified dataset changed after training plan creation",
        )

    resolved_checkpoint = (
        _validate_resume_checkpoint(config, plan, resume_from_checkpoint)
        if resume_from_checkpoint is not None
        else None
    )
    _require_stable_qualification(config, plan.qualification, verification)
    output_dir = Path(config.training.output_dir)
    config_payload = json_safe(config)
    started_at = datetime.now(UTC)
    run_id = make_run_id(config=config_payload, created_at=started_at)
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_dir = runs_dir / run_id
    run_dir.mkdir(exist_ok=False)
    trainer_output_dir = run_dir / "trainer"
    adapter_path = run_dir / "adapter"
    manifest_path = run_dir / "manifest.json"
    try:
        snapshots = materialize_prepared_split_snapshots(
            verification,
            run_dir / "dataset",
            split_names=("train", "validation"),
        )
        # TRL emits a background Hub telemetry request when its trainer is constructed.
        # Disable that request for privacy and to keep offline runs network-silent.
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        libraries = _libraries or _load_training_libraries()
        tokenizer = _load_tokenizer(config, libraries, allow_download=allow_download)
        token_budget = _validate_token_budget(
            tokenizer,
            snapshots,
            max_length=config.model.max_sequence_length,
        )
        train_dataset = _load_snapshot_dataset(libraries, snapshots["train"])
        evaluation_dataset = (
            _load_snapshot_dataset(libraries, snapshots["validation"])
            if plan.dataset_counts["validation"] > 0
            else None
        )
        model = _load_model(config, libraries, allow_download=allow_download)
        sft_config = _build_sft_config(
            config,
            libraries,
            has_evaluation=evaluation_dataset is not None,
            output_dir=trainer_output_dir,
        )
        peft_config = _build_peft_config(config, libraries)
        trainer = libraries.SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=train_dataset,
            eval_dataset=evaluation_dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
        )
        train_output = trainer.train(
            resume_from_checkpoint=(
                str(resolved_checkpoint) if resolved_checkpoint is not None else None
            )
        )
        optimizer_steps = getattr(train_output, "global_step", None)
        if (
            isinstance(optimizer_steps, bool)
            or not isinstance(optimizer_steps, int)
            or optimizer_steps <= 0
        ):
            raise RuntimeError("training completed without positive optimizer step evidence")
        metrics = _numeric_metrics(getattr(train_output, "metrics", None))
        metrics.update(
            {
                "optimizer_steps": optimizer_steps,
                "token_budget_examples": token_budget.examples,
                "token_budget_max_full_tokens": token_budget.max_full_tokens,
                "token_budget_min_completion_tokens": token_budget.min_completion_tokens,
            }
        )
        runtime_metric = metrics.get("train_runtime")
        training_duration_seconds = (
            float(runtime_metric)
            if isinstance(runtime_metric, (float, int)) and not isinstance(runtime_metric, bool)
            else None
        )
        peak_accelerator_memory_mb = _peak_accelerator_memory_mb(libraries)
        trainer.save_model(str(adapter_path))
        tokenizer.save_pretrained(str(adapter_path))
        if hasattr(trainer, "save_metrics"):
            trainer.save_metrics("train", metrics)
        if hasattr(trainer, "save_state"):
            trainer.save_state()
        run_artifacts = _verified_run_artifacts(run_dir, snapshots)
        manifest = build_run_manifest(
            config=config_payload,
            status="completed",
            project_name=config.project_name,
            model_name_or_path=config.model.name_or_path,
            model_revision=config.model.revision,
            method=config.lora.method,
            seed=config.seed,
            dataset_sha256=plan.dataset_sha256,
            metrics=metrics,
            artifacts=run_artifacts,
            resume_from_checkpoint=resolved_checkpoint,
            hardware_preflight=hardware_preflight,
            training_duration_seconds=training_duration_seconds,
            peak_accelerator_memory_mb=peak_accelerator_memory_mb,
            created_at=started_at,
        )
        if manifest.run_id != run_id:
            raise RuntimeError("run manifest identity diverged from the allocated run directory")
        write_manifest(manifest_path, manifest)
    except Exception as exc:
        failed_manifest = build_run_manifest(
            config=config_payload,
            status="failed",
            project_name=config.project_name,
            model_name_or_path=config.model.name_or_path,
            model_revision=config.model.revision,
            method=config.lora.method,
            seed=config.seed,
            dataset_sha256=plan.dataset_sha256,
            artifacts=_run_artifacts(run_dir),
            resume_from_checkpoint=resolved_checkpoint,
            error=f"{type(exc).__name__}: {sanitize_error(exc)}",
            hardware_preflight=hardware_preflight,
            created_at=started_at,
        )
        if not manifest_path.exists():
            write_manifest(manifest_path, failed_manifest)
        raise

    latest_pointer = _write_latest_pointer(output_dir, manifest, run_dir)

    return TrainingResult(
        plan=plan,
        executed=True,
        run_id=run_id,
        run_dir=str(run_dir),
        adapter_path=str(adapter_path),
        manifest_path=str(manifest_path),
        latest_pointer_path=str(latest_pointer),
        metrics=metrics,
        token_budget=token_budget,
        proof_boundary=(
            "local_training_executed: run-specific adapter, trainer products, and immutable "
            "manifest were written; latest-run.json is an explicit mutable convenience pointer"
        ),
    )
