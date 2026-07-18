"""Fail-closed functional parity for PEFT adapters and verified safe merges."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .export import ExportValidationError, VerifiedMergedModel, verify_merged_model
from .generation import (
    REQUIRED_TRAINING_DATASET_HASHES,
    AdapterCompatibilityError,
    AdapterProvenance,
    GeneratedPrediction,
    GenerationLibraries,
    GenerationOutputError,
    Sha256Digest,
    _read_regular_snapshot,
    _validate_immutable_output_path,
    _write_immutable_payload,
    generate_predictions,
    validate_adapter_compatibility,
)
from .run_manifest import canonical_json_bytes
from .schemas import TriageOutput
from .strict_json import StrictJSONError, loads_strict

if TYPE_CHECKING:
    from .config import FineTuneConfig

_ROUTING_FIELDS = ("category", "priority", "sentiment", "next_action")
MergeDtype = Literal["bfloat16", "float16", "float32"]


class ParityValidationError(ValueError):
    """Prediction artifacts cannot support a trustworthy parity comparison."""


class ParityDecision(BaseModel):
    """One exact functional-parity gate."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    metric: str
    value: float = Field(ge=0, le=1)
    required: float = Field(default=1.0, ge=0, le=1)
    passed: bool


class ParityMetrics(BaseModel):
    """Aggregate schema, routing, and response diagnostics."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    examples: int = Field(ge=1)
    adapter_schema_valid_rate: float = Field(ge=0, le=1)
    merged_schema_valid_rate: float = Field(ge=0, le=1)
    category_match_rate: float = Field(ge=0, le=1)
    priority_match_rate: float = Field(ge=0, le=1)
    sentiment_match_rate: float = Field(ge=0, le=1)
    next_action_match_rate: float = Field(ge=0, le=1)
    routing_match_rate: float = Field(ge=0, le=1)
    response_exact_rate: float = Field(ge=0, le=1)
    raw_prediction_exact_rate: float = Field(default=0.0, ge=0, le=1)
    parsed_object_exact_rate: float = Field(default=0.0, ge=0, le=1)


class PredictionArtifact(BaseModel):
    """Immutable prediction artifact identity used by one parity side."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    role: Literal["adapter", "merged"]
    path: str
    sha256: Sha256Digest
    model_name_or_path: str
    model_revision: str | None = None
    dataset_manifest_sha256: Sha256Digest
    dataset_split_sha256: Sha256Digest
    generation_config_sha256: Sha256Digest
    training_manifest_sha256: Sha256Digest | None = None
    training_config_sha256: Sha256Digest | None = None
    training_dataset_sha256: dict[str, Sha256Digest] = Field(default_factory=dict)
    qualification_sha256: dict[str, Sha256Digest] = Field(default_factory=dict)
    adapter_path: str | None = None
    adapter_config_sha256: Sha256Digest | None = None
    adapter_weight_sha256: dict[str, Sha256Digest] = Field(default_factory=dict)
    merged_model_path: str | None = None
    merge_provenance_sha256: Sha256Digest | None = None
    merged_artifact_sha256: dict[str, Sha256Digest] = Field(default_factory=dict)
    merged_adapter_config_sha256: Sha256Digest | None = None
    merged_adapter_weight_files: tuple[str, ...] = ()
    merged_adapter_weight_sha256: tuple[Sha256Digest, ...] = ()


class PredictionContract(BaseModel):
    """Cross-side generation and training lineage that must remain identical."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    dataset_manifest_sha256: Sha256Digest
    dataset_split_sha256: Sha256Digest
    generation_config_sha256: Sha256Digest
    training_manifest_sha256: Sha256Digest | None = None
    training_config_sha256: Sha256Digest | None = None
    training_dataset_sha256: dict[str, Sha256Digest] = Field(default_factory=dict)
    qualification_sha256: dict[str, Sha256Digest] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_training_lineage(self) -> PredictionContract:
        presence = (
            self.training_manifest_sha256 is not None,
            self.training_config_sha256 is not None,
            bool(self.training_dataset_sha256),
        )
        if any(presence) and not all(presence):
            raise ValueError("parity training lineage must be complete")
        if self.training_dataset_sha256:
            if not REQUIRED_TRAINING_DATASET_HASHES.issubset(self.training_dataset_sha256):
                raise ValueError("parity training dataset hashes are incomplete")
            if self.training_dataset_sha256["manifest"] != self.dataset_manifest_sha256:
                raise ValueError("parity training manifest hash differs from the dataset")
            if self.training_dataset_sha256["test"] != self.dataset_split_sha256:
                raise ValueError("parity training test hash differs from the evaluated split")
        expected_qualification = {
            name: digest
            for name, digest in self.training_dataset_sha256.items()
            if name.startswith("qualification_")
        }
        if self.qualification_sha256 != expected_qualification:
            raise ValueError("parity qualification hashes differ from training lineage")
        return self


class ParityReport(BaseModel):
    """Deterministic functional-parity report with immutable input identities."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    schema_version: Literal["1.0"] = "1.0"
    parity_id: str = Field(pattern=r"^[0-9a-f]{16}$")
    dataset_manifest_sha256: Sha256Digest
    dataset_split_sha256: Sha256Digest
    generation_config_sha256: Sha256Digest
    training_manifest_sha256: Sha256Digest | None = None
    training_config_sha256: Sha256Digest | None = None
    training_dataset_sha256: dict[str, Sha256Digest] = Field(default_factory=dict)
    qualification_sha256: dict[str, Sha256Digest] = Field(default_factory=dict)
    merge_dtype: MergeDtype | None = None
    ordered_ids: tuple[str, ...]
    adapter_predictions: PredictionArtifact
    merged_predictions: PredictionArtifact
    metrics: ParityMetrics
    adapter_schema_invalid_ids: tuple[str, ...]
    merged_schema_invalid_ids: tuple[str, ...]
    routing_mismatches: dict[str, tuple[str, ...]]
    mismatched_ids: tuple[str, ...]
    response_mismatched_ids: tuple[str, ...]
    raw_prediction_mismatched_ids: tuple[str, ...] = ()
    parsed_object_mismatched_ids: tuple[str, ...] = ()
    contract_invalid_ids: tuple[str, ...] = ()
    release_blocked_ids: tuple[str, ...] = ()
    thresholds: tuple[ParityDecision, ...]
    passed: bool
    proof_boundary: str


class ParityArtifacts(BaseModel):
    """In-memory report plus optional immutable report file."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    report: ParityReport
    report_path: str | None = None
    report_sha256: Sha256Digest
    passed: bool


class ParityThresholdError(RuntimeError):
    """Exact schema or routing parity failed under enforcement."""

    def __init__(self, report: ParityReport) -> None:
        self.report = report
        self.failures = tuple(item for item in report.thresholds if not item.passed)
        detail = ", ".join(
            f"{item.metric}={item.value:.4f} < {item.required:.4f}" for item in self.failures
        )
        super().__init__(f"parity thresholds failed: {detail}")


def _read_prediction_artifact(
    path: Path,
    *,
    role: Literal["adapter", "merged"],
) -> tuple[Path, tuple[GeneratedPrediction, ...], PredictionArtifact, PredictionContract]:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise ParityValidationError(f"{role} predictions cannot be a symlink: {path}")
    resolved = expanded.resolve()
    if not resolved.is_file():
        raise ParityValidationError(f"{role} predictions must be a regular file: {resolved}")

    try:
        payload = _read_regular_snapshot(resolved, label=f"{role} predictions")
        text = payload.decode("utf-8")
    except (AdapterCompatibilityError, UnicodeDecodeError) as exc:
        raise ParityValidationError(f"cannot read stable {role} predictions: {resolved}") from exc
    rows: list[GeneratedPrediction] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            decoded = loads_strict(line)
        except (json.JSONDecodeError, StrictJSONError) as exc:
            detail = exc.msg if isinstance(exc, json.JSONDecodeError) else str(exc)
            raise ParityValidationError(
                f"{resolved}:{line_number}: invalid JSON: {detail}"
            ) from exc
        if not isinstance(decoded, dict):
            raise ParityValidationError(f"{resolved}:{line_number}: expected a JSON object")
        try:
            row = GeneratedPrediction.model_validate(decoded)
            TriageOutput.model_validate(row.expected)
        except ValidationError as exc:
            raise ParityValidationError(
                f"{resolved}:{line_number}: invalid prediction row: {exc.error_count()} error(s)"
            ) from exc
        rows.append(row)
    if not rows:
        raise ParityValidationError(f"{role} predictions file is empty: {resolved}")

    identifiers = [row.id for row in rows]
    if any(not identifier for identifier in identifiers):
        raise ParityValidationError(f"{role} predictions contain an empty ID")
    duplicate_ids = sorted(
        {identifier for identifier in identifiers if identifiers.count(identifier) > 1}
    )
    if duplicate_ids:
        raise ParityValidationError(f"{role} predictions contain duplicate IDs: {duplicate_ids}")

    def one_value(attribute: str) -> object:
        values = [getattr(row, attribute) for row in rows]
        if any(value != values[0] for value in values[1:]):
            raise ParityValidationError(
                f"{role} prediction provenance differs across rows: {attribute}"
            )
        return values[0]

    dataset_sha256 = one_value("dataset_manifest_sha256")
    if dataset_sha256 is None:
        raise ParityValidationError(
            f"{role} predictions require dataset_manifest_sha256 on every row"
        )
    dataset_split_sha256 = one_value("dataset_split_sha256")
    generation_config_sha256 = one_value("generation_config_sha256")
    if dataset_split_sha256 is None or generation_config_sha256 is None:
        raise ParityValidationError(
            f"{role} predictions require dataset split and generation config hashes"
        )
    training_manifest_sha256 = one_value("training_manifest_sha256")
    training_config_sha256 = one_value("training_config_sha256")
    training_dataset_sha256 = one_value("training_dataset_sha256")
    qualification_sha256 = one_value("qualification_sha256")
    if (training_manifest_sha256 is None) != (training_config_sha256 is None):
        raise ParityValidationError(f"{role} predictions contain incomplete training lineage")
    if training_manifest_sha256 is None and (training_dataset_sha256 or qualification_sha256):
        raise ParityValidationError(f"{role} predictions contain orphaned training hashes")
    model_name = one_value("model_name_or_path")
    model_revision = one_value("model_revision")
    adapter_path = one_value("adapter_path")
    adapter_config_sha256 = one_value("adapter_config_sha256")
    adapter_weight_sha256 = one_value("adapter_weight_sha256")
    merged_model_path = one_value("merged_model_path")
    merge_provenance_sha256 = one_value("merge_provenance_sha256")
    merged_artifact_sha256 = one_value("merged_artifact_sha256")
    merged_adapter_config_sha256 = one_value("merged_adapter_config_sha256")
    merged_adapter_weight_files = one_value("merged_adapter_weight_files")
    merged_adapter_weight_sha256 = one_value("merged_adapter_weight_sha256")

    if role == "adapter":
        if (
            not isinstance(adapter_path, str)
            or not adapter_path
            or adapter_config_sha256 is None
            or not adapter_weight_sha256
        ):
            raise ParityValidationError(
                "adapter predictions require adapter path, config hash, and weight hashes"
            )
        if (
            merged_model_path is not None
            or merge_provenance_sha256 is not None
            or merged_artifact_sha256
            or merged_adapter_config_sha256 is not None
            or merged_adapter_weight_files
            or merged_adapter_weight_sha256
        ):
            raise ParityValidationError(
                "adapter predictions cannot also declare merged-model provenance"
            )
    else:
        if (
            not isinstance(merged_model_path, str)
            or not merged_model_path
            or merge_provenance_sha256 is None
            or not merged_artifact_sha256
            or merged_adapter_config_sha256 is None
            or not merged_adapter_weight_files
            or not merged_adapter_weight_sha256
        ):
            raise ParityValidationError(
                "merged predictions require merged and source-adapter lineage"
            )
        if adapter_path is not None or adapter_config_sha256 is not None or adapter_weight_sha256:
            raise ParityValidationError("merged predictions cannot apply or declare a PEFT adapter")
        if (
            not isinstance(merged_adapter_weight_files, tuple)
            or not isinstance(merged_adapter_weight_sha256, tuple)
            or len(merged_adapter_weight_files) != len(merged_adapter_weight_sha256)
            or len(set(merged_adapter_weight_files)) != len(merged_adapter_weight_files)
        ):
            raise ParityValidationError("merged predictions contain invalid adapter weight lineage")

    artifact = PredictionArtifact(
        role=role,
        path=str(resolved),
        sha256=hashlib.sha256(payload).hexdigest(),
        model_name_or_path=str(model_name),
        model_revision=model_revision if isinstance(model_revision, str) else None,
        dataset_manifest_sha256=str(dataset_sha256),
        dataset_split_sha256=str(dataset_split_sha256),
        generation_config_sha256=str(generation_config_sha256),
        training_manifest_sha256=(
            training_manifest_sha256 if isinstance(training_manifest_sha256, str) else None
        ),
        training_config_sha256=(
            training_config_sha256 if isinstance(training_config_sha256, str) else None
        ),
        training_dataset_sha256=(
            dict(training_dataset_sha256) if isinstance(training_dataset_sha256, dict) else {}
        ),
        qualification_sha256=(
            dict(qualification_sha256) if isinstance(qualification_sha256, dict) else {}
        ),
        adapter_path=adapter_path if isinstance(adapter_path, str) else None,
        adapter_config_sha256=(
            adapter_config_sha256 if isinstance(adapter_config_sha256, str) else None
        ),
        adapter_weight_sha256=(
            dict(adapter_weight_sha256) if isinstance(adapter_weight_sha256, dict) else {}
        ),
        merged_model_path=(merged_model_path if isinstance(merged_model_path, str) else None),
        merge_provenance_sha256=(
            merge_provenance_sha256 if isinstance(merge_provenance_sha256, str) else None
        ),
        merged_artifact_sha256=(
            dict(merged_artifact_sha256) if isinstance(merged_artifact_sha256, dict) else {}
        ),
        merged_adapter_config_sha256=(
            merged_adapter_config_sha256 if isinstance(merged_adapter_config_sha256, str) else None
        ),
        merged_adapter_weight_files=(
            merged_adapter_weight_files if isinstance(merged_adapter_weight_files, tuple) else ()
        ),
        merged_adapter_weight_sha256=(
            merged_adapter_weight_sha256 if isinstance(merged_adapter_weight_sha256, tuple) else ()
        ),
    )
    try:
        contract = PredictionContract(
            dataset_manifest_sha256=str(dataset_sha256),
            dataset_split_sha256=str(dataset_split_sha256),
            generation_config_sha256=str(generation_config_sha256),
            training_manifest_sha256=artifact.training_manifest_sha256,
            training_config_sha256=artifact.training_config_sha256,
            training_dataset_sha256=artifact.training_dataset_sha256,
            qualification_sha256=artifact.qualification_sha256,
        )
    except ValidationError as exc:
        raise ParityValidationError(
            f"{role} prediction training lineage is invalid: {exc.error_count()} error(s)"
        ) from exc
    return resolved, tuple(rows), artifact, contract


def _parse_output(value: str) -> tuple[TriageOutput | None, dict[str, object] | None]:
    try:
        decoded = loads_strict(value.strip())
    except (json.JSONDecodeError, StrictJSONError):
        return None, None
    if not isinstance(decoded, dict):
        return None, None
    try:
        return TriageOutput.model_validate(decoded), decoded
    except ValidationError:
        return None, decoded


def _rate(matches: Sequence[bool]) -> float:
    return sum(matches) / len(matches)


def _build_report(
    adapter_rows: tuple[GeneratedPrediction, ...],
    merged_rows: tuple[GeneratedPrediction, ...],
    *,
    adapter_artifact: PredictionArtifact,
    merged_artifact: PredictionArtifact,
    contract: PredictionContract,
    merge_dtype: MergeDtype | None = None,
) -> ParityReport:
    adapter_ids = tuple(row.id for row in adapter_rows)
    merged_ids = tuple(row.id for row in merged_rows)
    if adapter_ids != merged_ids:
        raise ParityValidationError(
            "adapter and merged predictions must contain identical ordered IDs"
        )
    if adapter_artifact.model_name_or_path != merged_artifact.model_name_or_path:
        raise ParityValidationError("adapter and merged predictions use different base models")
    if adapter_artifact.model_revision != merged_artifact.model_revision:
        raise ParityValidationError("adapter and merged predictions use different model revisions")
    if adapter_artifact.adapter_config_sha256 != merged_artifact.merged_adapter_config_sha256:
        raise ParityValidationError(
            "merged model was not produced from the compared adapter config"
        )
    merged_adapter_weights = dict(
        zip(
            merged_artifact.merged_adapter_weight_files,
            merged_artifact.merged_adapter_weight_sha256,
            strict=True,
        )
    )
    if adapter_artifact.adapter_weight_sha256 != merged_adapter_weights:
        raise ParityValidationError(
            "merged model was not produced from the compared adapter weights"
        )
    for identifier, adapter_row, merged_row in zip(
        adapter_ids, adapter_rows, merged_rows, strict=True
    ):
        if adapter_row.expected != merged_row.expected:
            raise ParityValidationError(
                f"adapter and merged expected objects differ for ID {identifier!r}"
            )
        try:
            TriageOutput.model_validate(adapter_row.expected)
            TriageOutput.model_validate(merged_row.expected)
        except ValidationError as exc:  # defensive; rows are validated during loading
            raise ParityValidationError(
                f"expected object is invalid for ID {identifier!r}"
            ) from exc

    adapter_outputs = [_parse_output(row.prediction) for row in adapter_rows]
    merged_outputs = [_parse_output(row.prediction) for row in merged_rows]
    adapter_schema_valid = [validated is not None for validated, _raw in adapter_outputs]
    merged_schema_valid = [validated is not None for validated, _raw in merged_outputs]
    adapter_invalid_ids = tuple(
        identifier
        for identifier, valid in zip(adapter_ids, adapter_schema_valid, strict=True)
        if not valid
    )
    merged_invalid_ids = tuple(
        identifier
        for identifier, valid in zip(adapter_ids, merged_schema_valid, strict=True)
        if not valid
    )

    field_matches: dict[str, list[bool]] = {field: [] for field in _ROUTING_FIELDS}
    routing_matches: list[bool] = []
    response_matches: list[bool] = []
    raw_prediction_matches: list[bool] = []
    parsed_object_matches: list[bool] = []
    routing_mismatch_ids: dict[str, list[str]] = {field: [] for field in _ROUTING_FIELDS}
    response_mismatch_ids: list[str] = []
    raw_prediction_mismatch_ids: list[str] = []
    parsed_object_mismatch_ids: list[str] = []
    for identifier, adapter_row, merged_row, adapter_output, merged_output in zip(
        adapter_ids,
        adapter_rows,
        merged_rows,
        adapter_outputs,
        merged_outputs,
        strict=True,
    ):
        adapter_validated, adapter_raw = adapter_output
        merged_validated, merged_raw = merged_output
        raw_prediction_match = adapter_row.prediction == merged_row.prediction
        raw_prediction_matches.append(raw_prediction_match)
        if not raw_prediction_match:
            raw_prediction_mismatch_ids.append(identifier)

        objects_comparable = adapter_raw is not None and merged_raw is not None
        parsed_object_match = bool(objects_comparable and adapter_raw == merged_raw)
        parsed_object_matches.append(parsed_object_match)
        if objects_comparable and not parsed_object_match:
            parsed_object_mismatch_ids.append(identifier)

        contracts_valid = adapter_validated is not None and merged_validated is not None
        row_field_matches: list[bool] = []
        for field in _ROUTING_FIELDS:
            matches = bool(
                contracts_valid
                and getattr(adapter_validated, field) == getattr(merged_validated, field)
            )
            field_matches[field].append(matches)
            row_field_matches.append(matches)
            if contracts_valid and not matches:
                routing_mismatch_ids[field].append(identifier)
        routing_matches.append(all(row_field_matches))
        response_matches_for_row = bool(
            contracts_valid
            and adapter_raw is not None
            and merged_raw is not None
            and adapter_raw.get("response") == merged_raw.get("response")
        )
        response_matches.append(response_matches_for_row)
        if contracts_valid and not response_matches_for_row:
            response_mismatch_ids.append(identifier)

    metrics = ParityMetrics(
        examples=len(adapter_rows),
        adapter_schema_valid_rate=_rate(adapter_schema_valid),
        merged_schema_valid_rate=_rate(merged_schema_valid),
        category_match_rate=_rate(field_matches["category"]),
        priority_match_rate=_rate(field_matches["priority"]),
        sentiment_match_rate=_rate(field_matches["sentiment"]),
        next_action_match_rate=_rate(field_matches["next_action"]),
        routing_match_rate=_rate(routing_matches),
        response_exact_rate=_rate(response_matches),
        raw_prediction_exact_rate=_rate(raw_prediction_matches),
        parsed_object_exact_rate=_rate(parsed_object_matches),
    )
    threshold_values = (
        ("adapter_schema_valid_rate", metrics.adapter_schema_valid_rate),
        ("merged_schema_valid_rate", metrics.merged_schema_valid_rate),
        ("category_match_rate", metrics.category_match_rate),
        ("priority_match_rate", metrics.priority_match_rate),
        ("sentiment_match_rate", metrics.sentiment_match_rate),
        ("next_action_match_rate", metrics.next_action_match_rate),
        ("routing_match_rate", metrics.routing_match_rate),
    )
    thresholds = tuple(
        ParityDecision(metric=metric, value=value, passed=value == 1.0)
        for metric, value in threshold_values
    )
    mismatched: set[str] = set()
    for identifiers in routing_mismatch_ids.values():
        mismatched.update(identifiers)
    contract_invalid = set(adapter_invalid_ids) | set(merged_invalid_ids)
    release_blocked = contract_invalid | mismatched

    identity_payload = {
        "dataset_manifest_sha256": contract.dataset_manifest_sha256,
        "dataset_split_sha256": contract.dataset_split_sha256,
        "generation_config_sha256": contract.generation_config_sha256,
        "training_manifest_sha256": contract.training_manifest_sha256,
        "merge_dtype": merge_dtype,
        "ordered_ids": adapter_ids,
        "adapter_predictions_sha256": adapter_artifact.sha256,
        "merged_predictions_sha256": merged_artifact.sha256,
        "merge_provenance_sha256": merged_artifact.merge_provenance_sha256,
    }
    return ParityReport(
        parity_id=hashlib.sha256(canonical_json_bytes(identity_payload)).hexdigest()[:16],
        dataset_manifest_sha256=contract.dataset_manifest_sha256,
        dataset_split_sha256=contract.dataset_split_sha256,
        generation_config_sha256=contract.generation_config_sha256,
        training_manifest_sha256=contract.training_manifest_sha256,
        training_config_sha256=contract.training_config_sha256,
        training_dataset_sha256=contract.training_dataset_sha256,
        qualification_sha256=contract.qualification_sha256,
        merge_dtype=merge_dtype,
        ordered_ids=adapter_ids,
        adapter_predictions=adapter_artifact,
        merged_predictions=merged_artifact,
        metrics=metrics,
        adapter_schema_invalid_ids=adapter_invalid_ids,
        merged_schema_invalid_ids=merged_invalid_ids,
        routing_mismatches={
            field: tuple(identifiers) for field, identifiers in routing_mismatch_ids.items()
        },
        mismatched_ids=tuple(identifier for identifier in adapter_ids if identifier in mismatched),
        response_mismatched_ids=tuple(response_mismatch_ids),
        raw_prediction_mismatched_ids=tuple(raw_prediction_mismatch_ids),
        parsed_object_mismatched_ids=tuple(parsed_object_mismatch_ids),
        contract_invalid_ids=tuple(
            identifier for identifier in adapter_ids if identifier in contract_invalid
        ),
        release_blocked_ids=tuple(
            identifier for identifier in adapter_ids if identifier in release_blocked
        ),
        thresholds=thresholds,
        passed=all(item.passed for item in thresholds),
        proof_boundary=(
            "Release parity requires exact schema validity and routing equality on one "
            "immutable ordered prediction set; raw, parsed-object, and response equality "
            "remain diagnostics and this does not prove task quality, serving health, or "
            "production readiness."
        ),
    )


def _write_report(path: Path, report: ParityReport) -> tuple[Path, str]:
    payload = canonical_json_bytes(report.model_dump(mode="json")) + b"\n"
    try:
        destination = _write_immutable_payload(
            path,
            payload,
            label="parity report",
        )
    except GenerationOutputError as exc:
        raise ParityValidationError(str(exc)) from exc
    return destination, hashlib.sha256(payload).hexdigest()


def compare_prediction_files(
    adapter_predictions_path: Path,
    merged_predictions_path: Path,
    *,
    output_path: Path | None = None,
    enforce: bool = False,
    _require_release_lineage: bool = False,
    _merge_dtype: MergeDtype | None = None,
) -> ParityArtifacts:
    """Compare existing immutable adapter and merged prediction JSONL files."""

    _adapter_path, adapter_rows, adapter_artifact, adapter_contract = _read_prediction_artifact(
        adapter_predictions_path, role="adapter"
    )
    _merged_path, merged_rows, merged_artifact, merged_contract = _read_prediction_artifact(
        merged_predictions_path, role="merged"
    )
    if adapter_contract != merged_contract:
        raise ParityValidationError("adapter and merged predictions use different run contracts")
    if _require_release_lineage and (
        adapter_contract.training_manifest_sha256 is None
        or adapter_contract.training_config_sha256 is None
        or not REQUIRED_TRAINING_DATASET_HASHES.issubset(adapter_contract.training_dataset_sha256)
    ):
        raise ParityValidationError("release parity requires completed training-manifest lineage")
    report = _build_report(
        adapter_rows,
        merged_rows,
        adapter_artifact=adapter_artifact,
        merged_artifact=merged_artifact,
        contract=adapter_contract,
        merge_dtype=_merge_dtype,
    )
    report_payload = canonical_json_bytes(report.model_dump(mode="json")) + b"\n"
    report_path: str | None = None
    report_sha256 = hashlib.sha256(report_payload).hexdigest()
    if output_path is not None:
        written, report_sha256 = _write_report(output_path, report)
        report_path = str(written)
    artifacts = ParityArtifacts(
        report=report,
        report_path=report_path,
        report_sha256=report_sha256,
        passed=report.passed,
    )
    if enforce and not report.passed:
        raise ParityThresholdError(report)
    return artifacts


def _require_adapter_matches_merge(
    adapter_path: Path,
    verified: VerifiedMergedModel,
    *,
    model_name_or_path: str,
    model_revision: str | None,
    require_training_manifest: bool,
) -> AdapterProvenance:
    try:
        adapter = validate_adapter_compatibility(
            adapter_path,
            model_name_or_path=model_name_or_path,
            model_revision=model_revision,
            require_training_manifest=require_training_manifest,
        )
    except AdapterCompatibilityError as exc:
        raise ParityValidationError(str(exc)) from exc
    if adapter.config_sha256 != verified.adapter_config_sha256:
        raise ParityValidationError(
            "live adapter config hash does not match the verified safe-merge provenance"
        )
    if tuple(adapter.weight_sha256.values()) != verified.adapter_weight_sha256:
        raise ParityValidationError(
            "live adapter weight hashes do not match the verified safe-merge provenance"
        )
    if not verified.adapter_weight_files:
        raise ParityValidationError("verified safe-merge provenance lacks adapter weight filenames")
    if tuple(adapter.weight_sha256) != verified.adapter_weight_files:
        raise ParityValidationError(
            "live adapter weight files do not match the verified safe-merge provenance"
        )
    if (
        adapter.training_manifest_sha256 != verified.training_manifest_sha256
        or adapter.training_config_sha256 != verified.training_config_sha256
        or tuple(sorted(adapter.training_dataset_sha256.items()))
        != verified.training_dataset_sha256
        or tuple(sorted(adapter.qualification_sha256.items())) != verified.qualification_sha256
    ):
        raise ParityValidationError(
            "live adapter training lineage does not match the verified safe merge"
        )
    if require_training_manifest and adapter.training_manifest_sha256 is None:
        raise ParityValidationError("release parity requires sibling training-manifest lineage")
    return adapter


def _require_live_merge_precision(
    config: FineTuneConfig,
    verified: VerifiedMergedModel,
) -> MergeDtype:
    """Bind live adapter/merge comparison to one proved model precision."""

    raw_merge_dtype = verified.merge_dtype
    if raw_merge_dtype is None:
        raise ParityValidationError(
            "live parity requires merge provenance with an explicit precision dtype"
        )
    if raw_merge_dtype not in {"bfloat16", "float16", "float32"}:  # pragma: no cover
        raise ParityValidationError("verified safe merge contains an unsupported precision dtype")
    merge_dtype = cast(MergeDtype, raw_merge_dtype)
    configured_dtype = config.model.torch_dtype
    if configured_dtype == "auto":
        raise ParityValidationError(
            "live parity requires an explicit configured precision instead of auto"
        )
    if merge_dtype != configured_dtype:
        raise ParityValidationError(
            "live parity precision mismatch: "
            f"config requests {configured_dtype}, merge provenance declares {merge_dtype}"
        )
    return merge_dtype


def _require_live_inputs_unchanged(
    adapter_path: Path,
    merged_model_path: Path,
    *,
    expected_adapter: AdapterProvenance,
    expected_merged: VerifiedMergedModel,
    model_name_or_path: str,
    model_revision: str | None,
    require_training_manifest: bool,
) -> None:
    try:
        current_adapter = validate_adapter_compatibility(
            adapter_path,
            model_name_or_path=model_name_or_path,
            model_revision=model_revision,
            require_training_manifest=require_training_manifest,
        )
    except AdapterCompatibilityError as exc:
        raise ParityValidationError(
            "adapter bytes, inventory, or training lineage changed"
        ) from exc
    if current_adapter != expected_adapter:
        raise ParityValidationError("adapter bytes, inventory, or training lineage changed")
    try:
        current_merged = verify_merged_model(
            merged_model_path,
            expected_base_model=model_name_or_path,
            expected_model_revision=model_revision,
        )
    except ExportValidationError as exc:
        raise ParityValidationError("merged-model bytes, inventory, or provenance changed") from exc
    if current_merged != expected_merged:
        raise ParityValidationError("merged-model bytes, inventory, or provenance changed")


def verify_live_parity(
    config: FineTuneConfig,
    adapter_path: Path,
    merged_model_path: Path,
    *,
    output_path: Path,
    dataset_path: Path | None = None,
    allow_download: bool = False,
    enforce: bool = False,
    allow_unverified_training_manifest: bool = False,
    _libraries: GenerationLibraries | None = None,
) -> ParityArtifacts:
    """Generate identical held-out prompts through an adapter and its safe merge."""

    require_training_manifest = not allow_unverified_training_manifest
    try:
        report_destination = _validate_immutable_output_path(
            output_path,
            label="parity report",
        )
    except GenerationOutputError as exc:
        raise ParityValidationError(str(exc)) from exc
    stem = report_destination.stem
    adapter_predictions = report_destination.with_name(f"{stem}.adapter-predictions.jsonl")
    merged_predictions = report_destination.with_name(f"{stem}.merged-predictions.jsonl")
    try:
        _validate_immutable_output_path(
            adapter_predictions,
            label="adapter predictions output",
        )
        _validate_immutable_output_path(
            merged_predictions,
            label="merged predictions output",
        )
    except GenerationOutputError as exc:
        raise ParityValidationError(str(exc)) from exc

    verified = verify_merged_model(
        merged_model_path,
        expected_base_model=config.model.name_or_path,
        expected_model_revision=config.model.revision,
    )
    merge_dtype = _require_live_merge_precision(config, verified)
    adapter = _require_adapter_matches_merge(
        adapter_path,
        verified,
        model_name_or_path=config.model.name_or_path,
        model_revision=config.model.revision,
        require_training_manifest=require_training_manifest,
    )

    generate_predictions(
        config,
        dataset_path=dataset_path,
        adapter_path=adapter_path,
        output_path=adapter_predictions,
        allow_download=allow_download,
        require_training_manifest=require_training_manifest,
        _libraries=_libraries,
    )
    _require_live_inputs_unchanged(
        adapter_path,
        merged_model_path,
        expected_adapter=adapter,
        expected_merged=verified,
        model_name_or_path=config.model.name_or_path,
        model_revision=config.model.revision,
        require_training_manifest=require_training_manifest,
    )
    generate_predictions(
        config,
        dataset_path=dataset_path,
        merged_model_path=merged_model_path,
        output_path=merged_predictions,
        allow_download=False,
        require_training_manifest=require_training_manifest,
        _libraries=_libraries,
    )
    _require_live_inputs_unchanged(
        adapter_path,
        merged_model_path,
        expected_adapter=adapter,
        expected_merged=verified,
        model_name_or_path=config.model.name_or_path,
        model_revision=config.model.revision,
        require_training_manifest=require_training_manifest,
    )
    return compare_prediction_files(
        adapter_predictions,
        merged_predictions,
        output_path=report_destination,
        enforce=enforce,
        _require_release_lineage=require_training_manifest,
        _merge_dtype=merge_dtype,
    )


__all__ = [
    "ParityArtifacts",
    "ParityDecision",
    "ParityMetrics",
    "ParityReport",
    "ParityThresholdError",
    "ParityValidationError",
    "PredictionArtifact",
    "compare_prediction_files",
    "verify_live_parity",
]
