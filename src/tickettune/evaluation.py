"""Task-specific JSON scoring, macro F1, reports, and quality thresholds."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .generation import (
    REQUIRED_TRAINING_DATASET_HASHES,
    AdapterCompatibilityError,
    GeneratedPrediction,
    GenerationOutputError,
    Sha256Digest,
    _load_generation_cohort,
    _read_regular_snapshot,
    _validate_predictions_against_cohort,
    _write_immutable_payload,
    generate_predictions,
)
from .run_manifest import (
    canonical_json_bytes,
    json_safe,
    make_run_id,
    sanitize_error,
    sha256_bytes,
)
from .schemas import CATEGORY_LABELS, PRIORITY_LABELS, SENTIMENT_LABELS, TriageOutput
from .strict_json import (
    DuplicateJSONKeyError,
    StrictJSONError,
    loads_strict,
    strict_json_decoder,
)

if TYPE_CHECKING:
    from .config import FineTuneConfig

_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", flags=re.IGNORECASE | re.DOTALL)
_INVALID_LABEL = "__invalid__"
_UNREDACTED_RESPONSE_PATTERNS = (
    re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
    re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)"),
    re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)"),
    re.compile(r"(?<![\w.])\+?(?:\d[\s().-]*){7,14}\d(?![\w.])"),
)


class PredictionScore(BaseModel):
    """Deterministic score for one held-out prediction."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    id: str
    expected: TriageOutput
    prediction: str
    parsed: dict[str, Any] | None
    parse_error: str | None
    strict_json_only: bool
    schema_valid: bool
    field_completeness: float = Field(ge=0, le=1)
    category_correct: bool
    priority_correct: bool
    sentiment_correct: bool
    response_policy_compliant: bool
    exact_match: bool
    predicted_category: str | None
    predicted_priority: str | None
    predicted_sentiment: str | None
    latency_ms: float | None = Field(default=None, ge=0)


class EvaluationSummary(BaseModel):
    """Aggregate structured-output and latency metrics."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    examples: int = Field(ge=0)
    strict_json_rate: float = Field(ge=0, le=1)
    schema_valid_rate: float = Field(ge=0, le=1)
    field_completeness_rate: float = Field(ge=0, le=1)
    category_accuracy: float = Field(ge=0, le=1)
    category_macro_f1: float = Field(ge=0, le=1)
    priority_accuracy: float = Field(ge=0, le=1)
    priority_macro_f1: float = Field(ge=0, le=1)
    sentiment_accuracy: float = Field(ge=0, le=1)
    sentiment_macro_f1: float = Field(ge=0, le=1)
    response_policy_rate: float = Field(ge=0, le=1)
    exact_match_rate: float = Field(ge=0, le=1)
    latency_mean_ms: float | None = Field(default=None, ge=0)
    latency_p50_ms: float | None = Field(default=None, ge=0)
    latency_p95_ms: float | None = Field(default=None, ge=0)


class ThresholdDecision(BaseModel):
    """One explicit absolute quality gate."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    metric: str
    value: float
    minimum: float
    passed: bool


class EvaluationProvenance(BaseModel):
    """Input identities shared by every row in a generated evaluation artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    dataset_manifest_sha256: Sha256Digest
    dataset_split_sha256: Sha256Digest
    generation_config_sha256: Sha256Digest
    model_name_or_path: str
    model_revision: str | None = None
    training_manifest_sha256: Sha256Digest | None = None
    training_config_sha256: Sha256Digest | None = None
    training_dataset_sha256: dict[str, Sha256Digest] = Field(default_factory=dict)
    qualification_sha256: dict[str, Sha256Digest] = Field(default_factory=dict)
    adapter_path: str | None = None
    adapter_config_sha256: Sha256Digest | None = None
    adapter_weight_sha256: dict[str, Sha256Digest] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_training_lineage(self) -> EvaluationProvenance:
        lineage_presence = (
            self.training_manifest_sha256 is not None,
            self.training_config_sha256 is not None,
            bool(self.training_dataset_sha256),
        )
        if any(lineage_presence) and not all(lineage_presence):
            raise ValueError(
                "training lineage must be complete: manifest, config, and dataset hashes"
            )
        if self.training_dataset_sha256:
            if not REQUIRED_TRAINING_DATASET_HASHES.issubset(self.training_dataset_sha256):
                raise ValueError("training lineage dataset hashes are incomplete")
            if self.training_dataset_sha256["manifest"] != self.dataset_manifest_sha256:
                raise ValueError("training lineage prepared-manifest hash differs from evaluation")
            if self.training_dataset_sha256["test"] != self.dataset_split_sha256:
                raise ValueError("training lineage test hash differs from evaluation")
        expected_qualification = {
            name: digest
            for name, digest in self.training_dataset_sha256.items()
            if name.startswith("qualification_")
        }
        if self.qualification_sha256 != expected_qualification:
            raise ValueError(
                "qualification hashes must match the qualification entries in training lineage"
            )
        return self


class EvaluationReport(BaseModel):
    """Complete immutable in-memory report before artifact rendering."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    schema_version: str = "1.3"
    generated_at: datetime
    predictions_path: str
    predictions_sha256: Sha256Digest
    provenance: EvaluationProvenance | None = None
    summary: EvaluationSummary
    thresholds: tuple[ThresholdDecision, ...]
    passed: bool
    results: tuple[PredictionScore, ...]


class EvaluationArtifacts(BaseModel):
    """Paths and report returned to the CLI."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    report: EvaluationReport
    json_report_path: str
    markdown_report_path: str
    scored_predictions_path: str
    passed: bool


class EvaluationComparison(BaseModel):
    """Candidate minus baseline metric deltas."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    metric_deltas: dict[str, float]
    non_regression: tuple[ThresholdDecision, ...]
    passed: bool
    candidate_passed: bool
    baseline_passed: bool


class ModelEvaluationResult(BaseModel):
    """Generated candidate report plus an optional baseline comparison."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    evaluation_id: str
    output_dir: str
    manifest_path: str
    latest_pointer_path: str
    candidate: EvaluationArtifacts
    baseline: EvaluationArtifacts | None = None
    comparison: EvaluationComparison | None = None


class EvaluationThresholdError(RuntimeError):
    """Raised when a caller elects to enforce failed thresholds."""

    def __init__(self, failures: Sequence[ThresholdDecision]) -> None:
        self.failures = tuple(failures)
        detail = ", ".join(
            f"{item.metric}={item.value:.4f} < {item.minimum:.4f}" for item in failures
        )
        super().__init__(f"evaluation thresholds failed: {detail}")


def _decode_json_candidate(text: str) -> dict[str, Any] | None:
    candidate = text.strip()
    if not candidate:
        return None
    try:
        value = loads_strict(candidate)
    except json.JSONDecodeError:
        value = None
    if isinstance(value, dict):
        return value

    decoder = strict_json_decoder()
    for offset, character in enumerate(candidate):
        if character != "{":
            continue
        try:
            value, _end = decoder.raw_decode(candidate[offset:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _extract_json_object_fail_closed(
    value: str | Mapping[str, Any],
) -> dict[str, Any] | None:
    if isinstance(value, Mapping):
        return dict(value)
    direct = _decode_json_candidate(value)
    if direct is not None:
        return direct
    for fenced in _JSON_FENCE.findall(value):
        decoded = _decode_json_candidate(fenced)
        if decoded is not None:
            return decoded
    return None


def extract_json_object(value: str | Mapping[str, Any]) -> dict[str, Any] | None:
    """Recover an unambiguous JSON object from raw, fenced, or wrapped output."""

    try:
        return _extract_json_object_fail_closed(value)
    except StrictJSONError:
        return None


def _strict_json_object(value: str | Mapping[str, Any]) -> dict[str, Any] | None:
    """Return an object only when the entire prediction is exactly one JSON object."""

    if isinstance(value, Mapping):
        return dict(value)
    candidate = value.strip()
    if not candidate:
        return None
    try:
        decoded = loads_strict(candidate)
    except json.JSONDecodeError:
        return None
    return decoded if isinstance(decoded, dict) else None


def _as_expected(value: TriageOutput | Mapping[str, Any]) -> TriageOutput:
    if isinstance(value, TriageOutput):
        return value
    return TriageOutput.model_validate(value)


def _raw_label(parsed: Mapping[str, Any] | None, field: str) -> str | None:
    if parsed is None:
        return None
    value = parsed.get(field)
    return value if isinstance(value, str) else None


def response_policy_compliant(parsed: Mapping[str, Any] | None) -> bool:
    """Check the response field's length and synthetic-data PII safety contract."""

    if parsed is None:
        return False
    response = parsed.get("response")
    if not isinstance(response, str):
        return False
    normalized = response.strip()
    if not 12 <= len(normalized) <= 1200:
        return False
    return not any(pattern.search(normalized) for pattern in _UNREDACTED_RESPONSE_PATTERNS)


def score_prediction(
    expected: TriageOutput | Mapping[str, Any],
    prediction: str | Mapping[str, Any],
    *,
    id: str = "example",
    latency_ms: float | None = None,
) -> PredictionScore:
    """Score parsing, schema, field, label, and full-object correctness."""

    expected_output = _as_expected(expected)
    strict_decode_error: StrictJSONError | None = None
    try:
        strict_parsed = _strict_json_object(prediction)
        parsed = _extract_json_object_fail_closed(prediction)
    except StrictJSONError as exc:
        strict_parsed = None
        parsed = None
        strict_decode_error = exc
    strict_json_only = strict_parsed is not None
    required_fields = tuple(TriageOutput.model_fields)
    complete = 0
    if parsed is not None:
        complete = sum(
            field in parsed and parsed[field] is not None and parsed[field] != ""
            for field in required_fields
        )
    field_completeness = complete / len(required_fields)
    validated: TriageOutput | None = None
    parse_error: str | None = None
    if isinstance(strict_decode_error, DuplicateJSONKeyError):
        parse_error = f"duplicate JSON object key: {strict_decode_error.key!r}"
    elif strict_decode_error is not None:
        parse_error = str(strict_decode_error)
    elif parsed is None:
        parse_error = "no JSON object found"
    else:
        try:
            validated = TriageOutput.model_validate(parsed)
        except ValidationError as exc:
            parse_error = f"schema validation failed: {exc.error_count()} error(s)"
        if not strict_json_only:
            parse_error = "output is not one bare JSON object"

    predicted_category = _raw_label(parsed, "category")
    predicted_priority = _raw_label(parsed, "priority")
    predicted_sentiment = _raw_label(parsed, "sentiment")
    prediction_text = (
        json.dumps(dict(prediction), ensure_ascii=False, allow_nan=False, sort_keys=True)
        if isinstance(prediction, Mapping)
        else prediction
    )
    return PredictionScore(
        id=id,
        expected=expected_output,
        prediction=prediction_text,
        parsed=parsed,
        parse_error=parse_error,
        strict_json_only=strict_json_only,
        schema_valid=strict_json_only and validated is not None,
        field_completeness=field_completeness,
        category_correct=predicted_category == expected_output.category,
        priority_correct=predicted_priority == expected_output.priority,
        sentiment_correct=predicted_sentiment == expected_output.sentiment,
        response_policy_compliant=response_policy_compliant(parsed),
        exact_match=(
            validated == expected_output if strict_json_only and validated is not None else False
        ),
        predicted_category=predicted_category,
        predicted_priority=predicted_priority,
        predicted_sentiment=predicted_sentiment,
        latency_ms=latency_ms,
    )


def macro_f1(
    expected: Sequence[str],
    predicted: Sequence[str | None],
    *,
    labels: Sequence[str] | None = None,
) -> float:
    """Compute unweighted multiclass F1 without a scikit-learn dependency."""

    if len(expected) != len(predicted):
        raise ValueError("expected and predicted labels must have equal length")
    if not expected:
        return 0.0
    normalized_predictions = [item if item is not None else _INVALID_LABEL for item in predicted]
    label_set = set(labels or ())
    if not label_set:
        label_set.update(expected)
        label_set.update(normalized_predictions)
    if not label_set:
        return 0.0
    scores: list[float] = []
    for label in sorted(label_set):
        true_positive = sum(
            actual == label and guessed == label
            for actual, guessed in zip(expected, normalized_predictions, strict=True)
        )
        false_positive = sum(
            actual != label and guessed == label
            for actual, guessed in zip(expected, normalized_predictions, strict=True)
        )
        false_negative = sum(
            actual == label and guessed != label
            for actual, guessed in zip(expected, normalized_predictions, strict=True)
        )
        denominator = (2 * true_positive) + false_positive + false_negative
        scores.append((2 * true_positive) / denominator if denominator else 0.0)
    return sum(scores) / len(scores)


def _mean(values: Sequence[float | bool]) -> float:
    return sum(float(value) for value in values) / len(values) if values else 0.0


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] + ((ordered[upper] - ordered[lower]) * weight)


def summarize_scores(scores: Sequence[PredictionScore]) -> EvaluationSummary:
    """Aggregate exact metrics from per-example scores."""

    categories = [str(item.expected.category) for item in scores]
    priorities = [str(item.expected.priority) for item in scores]
    sentiments = [str(item.expected.sentiment) for item in scores]
    latencies = [item.latency_ms for item in scores if item.latency_ms is not None]
    return EvaluationSummary(
        examples=len(scores),
        strict_json_rate=_mean([item.strict_json_only for item in scores]),
        schema_valid_rate=_mean([item.schema_valid for item in scores]),
        field_completeness_rate=_mean([item.field_completeness for item in scores]),
        category_accuracy=_mean([item.category_correct for item in scores]),
        category_macro_f1=macro_f1(
            categories,
            [item.predicted_category for item in scores],
            labels=CATEGORY_LABELS,
        ),
        priority_accuracy=_mean([item.priority_correct for item in scores]),
        priority_macro_f1=macro_f1(
            priorities,
            [item.predicted_priority for item in scores],
            labels=PRIORITY_LABELS,
        ),
        sentiment_accuracy=_mean([item.sentiment_correct for item in scores]),
        sentiment_macro_f1=macro_f1(
            sentiments,
            [item.predicted_sentiment for item in scores],
            labels=SENTIMENT_LABELS,
        ),
        response_policy_rate=_mean([item.response_policy_compliant for item in scores]),
        exact_match_rate=_mean([item.exact_match for item in scores]),
        latency_mean_ms=_mean(latencies) if latencies else None,
        latency_p50_ms=_percentile(latencies, 0.50),
        latency_p95_ms=_percentile(latencies, 0.95),
    )


def _threshold_decisions(
    config: FineTuneConfig, summary: EvaluationSummary
) -> tuple[ThresholdDecision, ...]:
    configured = config.evaluation.thresholds
    values = {
        "strict_json_rate": (summary.strict_json_rate, configured.strict_json_rate),
        "schema_valid_rate": (summary.schema_valid_rate, configured.schema_valid_rate),
        "category_accuracy": (summary.category_accuracy, configured.category_accuracy),
        "priority_accuracy": (summary.priority_accuracy, configured.priority_accuracy),
        "sentiment_accuracy": (summary.sentiment_accuracy, configured.sentiment_accuracy),
        "response_policy_rate": (
            summary.response_policy_rate,
            configured.response_policy_rate,
        ),
    }
    return tuple(
        ThresholdDecision(metric=name, value=value, minimum=minimum, passed=value >= minimum)
        for name, (value, minimum) in values.items()
    )


def _parse_prediction_payload(path: Path, payload: bytes) -> list[dict[str, Any]]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"predictions file is not UTF-8: {path}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = loads_strict(line)
        except (json.JSONDecodeError, StrictJSONError) as exc:
            detail = exc.msg if isinstance(exc, json.JSONDecodeError) else str(exc)
            raise ValueError(f"{path}:{line_number}: invalid JSON: {detail}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        if "expected" not in row:
            raise ValueError(f"{path}:{line_number}: missing expected output")
        if "prediction" not in row:
            raise ValueError(f"{path}:{line_number}: missing prediction text")
        rows.append(row)
    if not rows:
        raise ValueError(f"predictions file is empty: {path}")
    return rows


def _read_prediction_rows(path: Path) -> tuple[list[dict[str, Any]], bytes]:
    try:
        payload = _read_regular_snapshot(path, label="prediction artifact")
    except AdapterCompatibilityError as exc:
        raise ValueError(str(exc)) from exc
    return _parse_prediction_payload(path, payload), payload


def _load_prediction_rows(path: Path) -> list[dict[str, Any]]:
    """Compatibility helper for callers that only need one stable row snapshot."""

    rows, _payload = _read_prediction_rows(path)
    return rows


def _evaluation_provenance(rows: Sequence[Mapping[str, Any]]) -> EvaluationProvenance | None:
    manifest_values = [row.get("dataset_manifest_sha256") for row in rows]
    if all(value is None for value in manifest_values):
        return None
    if any(value is None for value in manifest_values):
        raise ValueError("prediction artifact mixes generated rows with and without provenance")
    keys = (
        "dataset_manifest_sha256",
        "dataset_split_sha256",
        "generation_config_sha256",
        "training_manifest_sha256",
        "training_config_sha256",
        "training_dataset_sha256",
        "qualification_sha256",
        "model_name_or_path",
        "model_revision",
        "adapter_path",
        "adapter_config_sha256",
        "adapter_weight_sha256",
    )
    first_payload = {key: rows[0].get(key) for key in keys}
    provenance = EvaluationProvenance.model_validate(first_payload, strict=True)
    canonical = provenance.model_dump(mode="json")
    for index, row in enumerate(rows[1:], 2):
        candidate = EvaluationProvenance.model_validate(
            {key: row.get(key) for key in keys},
            strict=True,
        ).model_dump(mode="json")
        if candidate != canonical:
            raise ValueError(f"prediction artifact provenance mismatch on row {index}")
    return provenance


def _atomic_write(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return path


def render_markdown_report(report: EvaluationReport) -> str:
    """Render a compact human-readable report from the canonical JSON model."""

    summary = report.summary
    lines = [
        "# TicketTune Evaluation Report",
        "",
        f"- Generated: `{report.generated_at.isoformat()}`",
        f"- Predictions: `{report.predictions_path}`",
        f"- Predictions SHA-256: `{report.predictions_sha256}`",
        f"- Overall threshold result: **{'PASS' if report.passed else 'FAIL'}**",
        "",
    ]
    if report.provenance is not None:
        provenance = report.provenance
        lines.extend(
            [
                "## Provenance",
                "",
                f"- Dataset manifest SHA-256: `{provenance.dataset_manifest_sha256}`",
                f"- Dataset split SHA-256: `{provenance.dataset_split_sha256}`",
                f"- Generation config SHA-256: `{provenance.generation_config_sha256}`",
                f"- Model: `{provenance.model_name_or_path}`",
                f"- Model revision: `{provenance.model_revision or 'not declared'}`",
                f"- Adapter: `{provenance.adapter_path or 'base model only'}`",
                f"- Adapter config SHA-256: "
                f"`{provenance.adapter_config_sha256 or 'not applicable'}`",
                f"- Training manifest SHA-256: "
                f"`{provenance.training_manifest_sha256 or 'not applicable'}`",
                f"- Training config SHA-256: "
                f"`{provenance.training_config_sha256 or 'not applicable'}`",
            ]
        )
        for name, digest in sorted(provenance.training_dataset_sha256.items()):
            lines.append(f"- Training dataset SHA-256 (`{name}`): `{digest}`")
        for name, digest in sorted(provenance.qualification_sha256.items()):
            lines.append(f"- Qualification SHA-256 (`{name}`): `{digest}`")
        for name, digest in sorted(provenance.adapter_weight_sha256.items()):
            lines.append(f"- Adapter weight SHA-256 (`{name}`): `{digest}`")
        lines.append("")
    lines.extend(
        [
            "## Quality metrics",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| Examples | {summary.examples} |",
            f"| Strict bare-JSON rate | {summary.strict_json_rate:.4f} |",
            f"| Schema-valid rate | {summary.schema_valid_rate:.4f} |",
            f"| Field completeness | {summary.field_completeness_rate:.4f} |",
            f"| Category accuracy | {summary.category_accuracy:.4f} |",
            f"| Category macro F1 | {summary.category_macro_f1:.4f} |",
            f"| Priority accuracy | {summary.priority_accuracy:.4f} |",
            f"| Priority macro F1 | {summary.priority_macro_f1:.4f} |",
            f"| Sentiment accuracy | {summary.sentiment_accuracy:.4f} |",
            f"| Sentiment macro F1 | {summary.sentiment_macro_f1:.4f} |",
            f"| Response-policy compliance | {summary.response_policy_rate:.4f} |",
            f"| Exact match | {summary.exact_match_rate:.4f} |",
            "",
            "## Thresholds",
            "",
            "| Metric | Value | Minimum | Result |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    lines.extend(
        f"| {item.metric} | {item.value:.4f} | {item.minimum:.4f} | "
        f"{'PASS' if item.passed else 'FAIL'} |"
        for item in report.thresholds
    )
    lines.extend(["", "## Latency", ""])
    if summary.latency_mean_ms is None:
        lines.append("Latency was not supplied in the predictions file.")
    else:
        lines.extend(
            [
                "| Statistic | Milliseconds |",
                "| --- | ---: |",
                f"| Mean | {summary.latency_mean_ms:.3f} |",
                f"| P50 | {summary.latency_p50_ms:.3f} |",
                f"| P95 | {summary.latency_p95_ms:.3f} |",
            ]
        )
    lines.extend(
        [
            "",
            "## Proof boundary",
            "",
            "This report scores the supplied prediction artifact only. It does not "
            "prove that a model was fine-tuned, served, or deployed.",
            "",
        ]
    )
    return "\n".join(lines)


def _persist_evaluation_payload(path: Path, payload: bytes, *, immutable: bool) -> Path:
    if not immutable:
        return _atomic_write(path, payload)
    try:
        destination = _write_immutable_payload(
            path,
            payload,
            label="live evaluation artifact",
            require_absent=True,
        )
        observed = _read_regular_snapshot(destination, label="live evaluation artifact")
    except (AdapterCompatibilityError, GenerationOutputError) as exc:
        raise ValueError(str(exc)) from exc
    if observed != payload:
        raise ValueError(f"live evaluation artifact changed after publication: {destination}")
    return destination


def _evaluate_rows(
    config: FineTuneConfig,
    rows: Sequence[Mapping[str, Any]],
    *,
    predictions_path: Path,
    predictions_payload: bytes,
    output_dir: Path | None,
    raise_on_failure: bool,
    immutable_outputs: bool,
) -> EvaluationArtifacts:
    """Score one already-snapshotted row sequence and persist its exact report bytes."""

    provenance = _evaluation_provenance(rows)
    scores = tuple(
        score_prediction(
            row["expected"],
            row["prediction"],
            id=str(row.get("id", f"example-{index + 1:05d}")),
            latency_ms=float(row["latency_ms"]) if row.get("latency_ms") is not None else None,
        )
        for index, row in enumerate(rows)
    )
    summary = summarize_scores(scores)
    decisions = _threshold_decisions(config, summary)
    passed = all(item.passed for item in decisions)
    report = EvaluationReport(
        generated_at=datetime.now(UTC),
        predictions_path=str(predictions_path),
        predictions_sha256=hashlib.sha256(predictions_payload).hexdigest(),
        provenance=provenance,
        summary=summary,
        thresholds=decisions,
        passed=passed,
        results=scores,
    )
    destination = output_dir or Path(config.evaluation.output_dir)
    json_path = destination / "evaluation-report.json"
    markdown_path = destination / "evaluation-report.md"
    scored_path = destination / "scored-predictions.jsonl"
    _persist_evaluation_payload(
        json_path,
        canonical_json_bytes(report.model_dump(mode="json"), pretty=True),
        immutable=immutable_outputs,
    )
    _persist_evaluation_payload(
        markdown_path,
        render_markdown_report(report).encode("utf-8"),
        immutable=immutable_outputs,
    )
    scored_payload = b"".join(
        canonical_json_bytes(item.model_dump(mode="json")) + b"\n" for item in scores
    )
    _persist_evaluation_payload(scored_path, scored_payload, immutable=immutable_outputs)
    artifacts = EvaluationArtifacts(
        report=report,
        json_report_path=str(json_path),
        markdown_report_path=str(markdown_path),
        scored_predictions_path=str(scored_path),
        passed=passed,
    )
    if raise_on_failure and not passed:
        raise EvaluationThresholdError([item for item in decisions if not item.passed])
    return artifacts


def evaluate_predictions(
    config: FineTuneConfig,
    predictions_path: Path,
    *,
    output_dir: Path | None = None,
    raise_on_failure: bool = False,
) -> EvaluationArtifacts:
    """Score one stable JSONL snapshot and write JSON, JSONL, and Markdown reports."""

    rows, payload = _read_prediction_rows(predictions_path)
    return _evaluate_rows(
        config,
        rows,
        predictions_path=predictions_path,
        predictions_payload=payload,
        output_dir=output_dir,
        raise_on_failure=raise_on_failure,
        immutable_outputs=False,
    )


def _generated_payload(predictions: Sequence[GeneratedPrediction]) -> bytes:
    return b"".join(
        canonical_json_bytes(item.model_dump(mode="json")) + b"\n" for item in predictions
    )


def _write_generated(
    path: Path,
    predictions: Sequence[GeneratedPrediction],
) -> tuple[Path, bytes]:
    payload = _generated_payload(predictions)
    try:
        destination = _write_immutable_payload(
            path,
            payload,
            label="live generated predictions",
            require_absent=True,
        )
        observed = _read_regular_snapshot(destination, label="live generated predictions")
    except (AdapterCompatibilityError, GenerationOutputError) as exc:
        raise ValueError(str(exc)) from exc
    if observed != payload:
        raise ValueError(f"live generated predictions changed after publication: {destination}")
    return destination, payload


def _evaluate_generated_predictions(
    config: FineTuneConfig,
    predictions: tuple[GeneratedPrediction, ...],
    *,
    predictions_path: Path,
    predictions_payload: bytes,
    output_dir: Path,
) -> EvaluationArtifacts:
    """Score exact generated objects without reopening their mutable path."""

    rows = tuple(item.model_dump(mode="json") for item in predictions)
    return _evaluate_rows(
        config,
        rows,
        predictions_path=predictions_path,
        predictions_payload=predictions_payload,
        output_dir=output_dir,
        raise_on_failure=False,
        immutable_outputs=True,
    )


def _duplicate_ids(ids: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for item in ids:
        if item in seen and item not in duplicates:
            duplicates.append(item)
        seen.add(item)
    return tuple(duplicates)


def _validate_comparable_reports(
    candidate: EvaluationReport,
    baseline: EvaluationReport,
) -> None:
    """Fail closed unless reports describe the same ordered evaluation cohort."""

    candidate_ids = tuple(item.id for item in candidate.results)
    baseline_ids = tuple(item.id for item in baseline.results)
    for label, ids in (("candidate", candidate_ids), ("baseline", baseline_ids)):
        duplicates = _duplicate_ids(ids)
        if duplicates:
            rendered = ", ".join(repr(item) for item in duplicates)
            raise ValueError(f"{label} evaluation contains duplicate IDs: {rendered}")

    if candidate_ids != baseline_ids:
        if len(candidate_ids) == len(baseline_ids) and set(candidate_ids) == set(baseline_ids):
            raise ValueError("candidate and baseline ordered IDs differ")
        candidate_set = set(candidate_ids)
        baseline_set = set(baseline_ids)
        missing_from_candidate = [item for item in baseline_ids if item not in candidate_set]
        missing_from_baseline = [item for item in candidate_ids if item not in baseline_set]
        details: list[str] = []
        if missing_from_candidate:
            details.append(f"missing from candidate: {missing_from_candidate!r}")
        if missing_from_baseline:
            details.append(f"missing from baseline: {missing_from_baseline!r}")
        detail = f" ({'; '.join(details)})" if details else ""
        raise ValueError(f"candidate and baseline ordered IDs must match exactly{detail}")

    for candidate_result, baseline_result in zip(
        candidate.results,
        baseline.results,
        strict=True,
    ):
        if candidate_result.expected != baseline_result.expected:
            raise ValueError(
                f"candidate and baseline expected output differs for ID {candidate_result.id!r}"
            )

    candidate_provenance = candidate.provenance
    baseline_provenance = baseline.provenance
    if (candidate_provenance is None) != (baseline_provenance is None):
        raise ValueError("candidate and baseline provenance must both be present or both be absent")
    if candidate_provenance is None or baseline_provenance is None:
        return

    for field in (
        "dataset_manifest_sha256",
        "dataset_split_sha256",
        "generation_config_sha256",
        "model_name_or_path",
        "model_revision",
    ):
        if getattr(candidate_provenance, field) != getattr(baseline_provenance, field):
            raise ValueError(f"candidate and baseline provenance differs for {field}")

    if (
        baseline_provenance.adapter_path is not None
        or baseline_provenance.adapter_config_sha256 is not None
        or baseline_provenance.adapter_weight_sha256
        or baseline_provenance.training_manifest_sha256 is not None
        or baseline_provenance.training_config_sha256 is not None
        or baseline_provenance.training_dataset_sha256
        or baseline_provenance.qualification_sha256
    ):
        raise ValueError(
            "baseline provenance must be base-model-only without adapter identity or weights"
        )


def _validate_live_adapter_predictions(
    predictions: Sequence[GeneratedPrediction],
    *,
    adapter_path: Path,
    require_training_lineage: bool,
) -> None:
    expected_adapter = str(adapter_path.expanduser().resolve())
    for index, prediction in enumerate(predictions, 1):
        if prediction.adapter_path != expected_adapter:
            raise ValueError(
                f"live adapter prediction {index} is not bound to the requested adapter"
            )
        if not require_training_lineage:
            continue
        if (
            prediction.training_manifest_sha256 is None
            or prediction.training_config_sha256 is None
            or not prediction.training_dataset_sha256
        ):
            raise ValueError(f"live adapter prediction {index} lacks verified training lineage")
        if (
            prediction.dataset_manifest_sha256 is None
            or prediction.training_dataset_sha256.get("manifest")
            != prediction.dataset_manifest_sha256
            or prediction.dataset_split_sha256 is None
            or prediction.training_dataset_sha256.get("test") != prediction.dataset_split_sha256
        ):
            raise ValueError(
                f"live adapter prediction {index} training dataset differs from evaluation"
            )


def _compare(candidate: EvaluationArtifacts, baseline: EvaluationArtifacts) -> EvaluationComparison:
    _validate_comparable_reports(candidate.report, baseline.report)
    candidate_summary = candidate.report.summary
    baseline_summary = baseline.report.summary
    names = (
        "strict_json_rate",
        "schema_valid_rate",
        "category_accuracy",
        "category_macro_f1",
        "priority_accuracy",
        "priority_macro_f1",
        "sentiment_accuracy",
        "sentiment_macro_f1",
        "response_policy_rate",
        "exact_match_rate",
    )
    deltas = {
        name: float(getattr(candidate_summary, name)) - float(getattr(baseline_summary, name))
        for name in names
    }
    non_regression = tuple(
        ThresholdDecision(metric=name, value=delta, minimum=0.0, passed=delta >= 0.0)
        for name, delta in deltas.items()
    )
    return EvaluationComparison(
        metric_deltas=deltas,
        non_regression=non_regression,
        passed=candidate.passed and all(item.passed for item in non_regression),
        candidate_passed=candidate.passed,
        baseline_passed=baseline.passed,
    )


def _evaluation_artifact_payloads(
    artifacts: EvaluationArtifacts,
    *,
    predictions_path: Path,
    predictions_payload: bytes,
) -> dict[Path, bytes]:
    report = artifacts.report
    return {
        predictions_path: predictions_payload,
        Path(artifacts.json_report_path): canonical_json_bytes(
            report.model_dump(mode="json"), pretty=True
        ),
        Path(artifacts.markdown_report_path): render_markdown_report(report).encode("utf-8"),
        Path(artifacts.scored_predictions_path): b"".join(
            canonical_json_bytes(item.model_dump(mode="json")) + b"\n" for item in report.results
        ),
    }


def _recheck_evaluation_artifacts(
    output_dir: Path,
    expected_payloads: Mapping[Path, bytes],
) -> dict[str, str]:
    """Recheck exact stable bytes immediately before publishing completed evidence."""

    hashes: dict[str, str] = {}
    for path, expected in sorted(expected_payloads.items(), key=lambda item: item[0].as_posix()):
        try:
            observed = _read_regular_snapshot(path, label="live evaluation artifact")
        except AdapterCompatibilityError as exc:
            raise ValueError(str(exc)) from exc
        if observed != expected:
            raise ValueError(f"live evaluation artifact changed before success: {path}")
        try:
            relative = path.relative_to(output_dir).as_posix()
        except ValueError as exc:
            raise ValueError(f"evaluation artifact escapes its run directory: {path}") from exc
        hashes[relative] = hashlib.sha256(observed).hexdigest()
    return hashes


def _write_evaluation_success(
    config: FineTuneConfig,
    result: ModelEvaluationResult,
    *,
    created_at: datetime,
    expected_artifact_payloads: Mapping[Path, bytes],
) -> None:
    output_dir = Path(result.output_dir)
    artifact_sha256 = _recheck_evaluation_artifacts(
        output_dir,
        expected_artifact_payloads,
    )
    payload = {
        "schema_version": "1.0",
        "evaluation_id": result.evaluation_id,
        "created_at": created_at,
        "status": "completed",
        "config_sha256": sha256_bytes(json_safe(config)),
        "candidate_passed": result.candidate.passed,
        "comparison_passed": result.comparison.passed if result.comparison else None,
        "artifact_sha256": artifact_sha256,
        "result": result,
    }
    _persist_evaluation_payload(
        Path(result.manifest_path),
        canonical_json_bytes(payload, pretty=True),
        immutable=True,
    )
    base_output = Path(config.evaluation.output_dir)
    pointer = {
        "schema_version": "1.0",
        "evaluation_id": result.evaluation_id,
        "status": "completed",
        "output_dir": output_dir.relative_to(base_output).as_posix(),
        "manifest_path": Path(result.manifest_path).relative_to(base_output).as_posix(),
        "updated_at": datetime.now(UTC),
    }
    _atomic_write(
        Path(result.latest_pointer_path),
        canonical_json_bytes(pointer, pretty=True),
    )


def run_model_evaluation(
    config: FineTuneConfig,
    *,
    adapter_path: Path | None = None,
    compare_baseline: bool = False,
    allow_download: bool = False,
    allow_unverified_adapter: bool = False,
    enforce_thresholds: bool = False,
) -> ModelEvaluationResult:
    """Generate and score adapter predictions, optionally alongside the base model."""

    if compare_baseline and adapter_path is None:
        raise ValueError("compare_baseline requires an adapter_path")
    if allow_unverified_adapter and adapter_path is None:
        raise ValueError("allow_unverified_adapter requires an adapter_path")
    if allow_unverified_adapter and enforce_thresholds:
        raise ValueError(
            "allow_unverified_adapter cannot be combined with enforced release thresholds"
        )
    require_training_lineage = adapter_path is not None and not allow_unverified_adapter
    created_at = datetime.now(UTC)
    identity = {
        "config": json_safe(config),
        "adapter_path": str(adapter_path.resolve()) if adapter_path else None,
        "compare_baseline": compare_baseline,
        "allow_unverified_adapter": allow_unverified_adapter,
    }
    evaluation_id = make_run_id(config=identity, created_at=created_at)
    base_output = Path(config.evaluation.output_dir)
    output_root = base_output / "runs" / evaluation_id
    output_root.mkdir(parents=True, exist_ok=False)
    manifest_path = output_root / "evaluation-manifest.json"
    latest_pointer_path = base_output / "latest-evaluation.json"
    try:
        cohort = _load_generation_cohort(config)
        candidate_predictions = generate_predictions(
            config,
            adapter_path=adapter_path,
            allow_download=allow_download,
            require_training_manifest=require_training_lineage,
        )
        _validate_predictions_against_cohort(
            candidate_predictions,
            cohort,
            label="candidate",
        )
        if adapter_path is not None:
            _validate_live_adapter_predictions(
                candidate_predictions,
                adapter_path=adapter_path,
                require_training_lineage=require_training_lineage,
            )
        candidate_path, candidate_payload = _write_generated(
            output_root / "candidate-predictions.jsonl", candidate_predictions
        )
        candidate = _evaluate_generated_predictions(
            config,
            candidate_predictions,
            predictions_path=candidate_path,
            predictions_payload=candidate_payload,
            output_dir=output_root / "candidate",
        )
        expected_artifact_payloads = _evaluation_artifact_payloads(
            candidate,
            predictions_path=candidate_path,
            predictions_payload=candidate_payload,
        )
        baseline: EvaluationArtifacts | None = None
        comparison: EvaluationComparison | None = None
        baseline_predictions: tuple[GeneratedPrediction, ...] | None = None
        if compare_baseline:
            baseline_predictions = generate_predictions(
                config,
                allow_download=allow_download,
                require_training_manifest=False,
            )
            _validate_predictions_against_cohort(
                baseline_predictions,
                cohort,
                label="baseline",
            )
            baseline_path, baseline_payload = _write_generated(
                output_root / "baseline-predictions.jsonl", baseline_predictions
            )
            baseline = _evaluate_generated_predictions(
                config,
                baseline_predictions,
                predictions_path=baseline_path,
                predictions_payload=baseline_payload,
                output_dir=output_root / "baseline",
            )
            expected_artifact_payloads.update(
                _evaluation_artifact_payloads(
                    baseline,
                    predictions_path=baseline_path,
                    predictions_payload=baseline_payload,
                )
            )
            comparison = _compare(candidate, baseline)
        result = ModelEvaluationResult(
            evaluation_id=evaluation_id,
            output_dir=str(output_root),
            manifest_path=str(manifest_path),
            latest_pointer_path=str(latest_pointer_path),
            candidate=candidate,
            baseline=baseline,
            comparison=comparison,
        )
        final_cohort = _load_generation_cohort(config)
        _validate_predictions_against_cohort(
            candidate_predictions,
            final_cohort,
            label="candidate",
        )
        if baseline_predictions is not None:
            _validate_predictions_against_cohort(
                baseline_predictions,
                final_cohort,
                label="baseline",
            )
        if enforce_thresholds and not result.candidate.passed:
            raise RuntimeError("candidate evaluation did not meet configured thresholds")
        if enforce_thresholds and result.comparison is not None and not result.comparison.passed:
            raise RuntimeError("candidate evaluation regressed against the baseline")
        _write_evaluation_success(
            config,
            result,
            created_at=created_at,
            expected_artifact_payloads=expected_artifact_payloads,
        )
    except Exception as exc:
        if not manifest_path.exists():
            _atomic_write(
                manifest_path,
                canonical_json_bytes(
                    {
                        "schema_version": "1.0",
                        "evaluation_id": evaluation_id,
                        "created_at": created_at,
                        "status": "failed",
                        "config_sha256": sha256_bytes(json_safe(config)),
                        "error": f"{type(exc).__name__}: {sanitize_error(exc)}",
                    },
                    pretty=True,
                ),
            )
        raise
    return result
