"""Authenticated deployment readback, load evidence, and rollback planning."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import ssl
import stat
import tempfile
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .data import DatasetManifest as _DatasetManifestEvidence
from .evaluation import (
    EvaluationArtifacts as _EvaluationCandidateEvidence,
)
from .evaluation import (
    EvaluationReport as _EvaluationReportEvidence,
)
from .evaluation import (
    ModelEvaluationResult as _EvaluationResultEvidence,
)
from .evaluation import (
    PredictionScore as _EvaluationScoreEvidence,
)
from .evaluation import (
    ThresholdDecision as _EvaluationThresholdEvidence,
)
from .evaluation import (
    score_prediction as _score_prediction,
)
from .evaluation import (
    summarize_scores as _summarize_scores,
)
from .generation import GeneratedPrediction as _GeneratedPredictionEvidence
from .hardware import HardwarePreflight
from .parity import (
    ParityReport as _ParityEvidence,
)
from .parity import (
    PredictionArtifact as _ParityPredictionEvidence,
)
from .parity import (
    PredictionContract as _ParityPredictionContract,
)
from .parity import (
    _build_report as _build_parity_report,
)
from .prompts import SYSTEM_PROMPT
from .qualification import DatasetQualificationReport as _QualificationEvidence
from .run_manifest import RunManifest as _TrainingEvidence
from .run_manifest import canonical_json_bytes, sha256_bytes
from .schemas import TicketExample, TriageOutput
from .strict_json import StrictJSONError, loads_strict

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})
_DEFAULT_TICKET = "I was charged twice for invoice [INVOICE_ID]. Please help."
MAX_RESPONSE_BYTES = 1024 * 1024
MIN_API_KEY_BYTES = 32
MAX_LOAD_REQUESTS = 10_000
MAX_LOAD_CONCURRENCY = 128
MAX_CPU_LIMIT = 256.0
MAX_MEMORY_BYTES = 1024**4
MIN_PROMETHEUS_RETENTION_SECONDS = 60 * 60
MAX_PROMETHEUS_RETENTION_SECONDS = 31 * 24 * 60 * 60
APPROVED_PRODUCTION_PROFILE = "tickettune-vllm-production-v1"
APPROVED_PRODUCTION_PROJECT = "tickettune-production"
APPROVED_DOCKER_EXECUTABLE = "/usr/bin/docker"
APPROVED_DOCKER_HOST = "unix:///var/run/docker.sock"
APPROVED_PRODUCTION_COMPOSE_SHA256 = (
    "5cdd6cbb34ab784ad864be4b8ffd4fac8d43c6fa6c2b38af929a9c9cde964f22"
)
_APPROVED_PRODUCTION_SUPPORT_SHA256 = (
    (
        "nginx.conf",
        "f3d5adcb1474b1345875bfdf1acc944f0bf0a8c00fdfa2ca2a985468b8beeab9",
    ),
    (
        "prometheus.yml",
        "46ab0172970e47559debeba92234c21178784c4ad8077d26dec8b378c4618a24",
    ),
    (
        "alerts.yml",
        "b43759d0227f41102b42484943798d000f703b3caabefcb12945a94eadf986a7",
    ),
    (
        "alertmanager.yml",
        "3fddd5088186a769d96908e1815b2a7aa061790159367a7a571514951bdc2eee",
    ),
    (
        "vllm-entrypoint.py",
        "d49dc9332c91d1224ca16272fafb179b2fe249283b9a0249f9dda8618bcff747",
    ),
)
_APPROVED_EXEC_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
_APPROVED_EXEC_HOME = "/var/empty"
_QUALIFICATION_POLICIES = frozenset(
    {
        "review_evidence_schema_v1_2",
        "source_sha256_matches",
        "record_count_matches",
        "minimum_record_count",
        "prepared_manifest_sha256_matches",
        "prepared_manifest_source_matches",
        "prepared_manifest_record_ids_match",
        "holdout_freeze_sha256_matches",
        "holdout_freeze_matches_prepared_test",
        "minimum_independent_review_packets",
        "review_packet_files_distinct",
        "review_packet_hashes_match",
        "reviewer_ids_distinct",
        "reviewer_ids_non_placeholder",
        "review_packets_bind_source",
        "review_packets_bind_prepared_manifest",
        "review_packets_bind_holdout_freeze",
        "review_packets_cover_ordered_source",
        "review_packets_approved",
        "review_packet_decisions_approved",
        "full_record_review",
        "minimum_held_out_examples",
        "held_out_within_dataset",
        "held_out_count_matches_ids",
        "held_out_ids_within_source",
        "approved_status",
    }
)
_EVALUATION_ABSOLUTE_METRICS = (
    "strict_json_rate",
    "schema_valid_rate",
    "category_accuracy",
    "priority_accuracy",
    "sentiment_accuracy",
    "response_policy_rate",
)
_EVALUATION_NON_REGRESSION_METRICS = (
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
_PARITY_GATE_METRICS = (
    "adapter_schema_valid_rate",
    "merged_schema_valid_rate",
    "category_match_rate",
    "priority_match_rate",
    "sentiment_match_rate",
    "next_action_match_rate",
    "routing_match_rate",
)
_TRAINING_DATASET_HASH_KEYS = frozenset(
    {
        "source",
        "manifest",
        "train",
        "validation",
        "test",
        "qualification_review_manifest",
        "qualification_report",
    }
)
_SPLIT_NAMES: tuple[Literal["train", "validation", "test"], ...] = (
    "train",
    "validation",
    "test",
)


class ResponseTooLargeError(ValueError):
    """A deployment endpoint exceeded the bounded proof-response budget."""


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


@dataclass(frozen=True)
class HttpResponse:
    """Transport-neutral HTTP response used by proof clients and tests."""

    status: int
    headers: Mapping[str, str]
    body: bytes


Transport = Callable[[str, str, dict[str, str], bytes | None, float, Path | None], HttpResponse]
ComposeRunner = Callable[[tuple[str, ...]], int]


class ReadbackReport(_FrozenModel):
    """Redacted endpoint claims and schema readback from one serving origin."""

    schema_version: Literal["1.1"] = "1.1"
    created_at: datetime
    base_url: str
    model: str
    expected_base_model: str
    models_status: int = Field(ge=0, le=599)
    chat_status: int = Field(ge=0, le=599)
    model_found: bool
    parent_matches: bool
    chat_schema_valid: bool
    request_ids_received: int = Field(ge=0, le=2)
    request_id_rate: float = Field(ge=0, le=1)
    passed: bool
    failure_codes: tuple[str, ...] = ()
    proof_boundary: Literal[
        "authenticated_tls_serving_claim_readback",
        "authenticated_loopback_http_serving_claim_readback",
    ]
    identity_limit: Literal[
        "endpoint_reported_model_and_parent_only; release_and_adapter_bytes_not_proven"
    ] = "endpoint_reported_model_and_parent_only; release_and_adapter_bytes_not_proven"


class LoadTestReport(_FrozenModel):
    """Redacted bounded-load summary with explicit acceptance thresholds."""

    schema_version: Literal["1.0"] = "1.0"
    created_at: datetime
    base_url: str
    model: str
    requests: int = Field(ge=1)
    concurrency: int = Field(ge=1)
    successes: int = Field(ge=0)
    schema_valid_responses: int = Field(ge=0)
    request_ids_received: int = Field(ge=0)
    success_rate: float = Field(ge=0, le=1)
    schema_valid_rate: float = Field(ge=0, le=1)
    request_id_rate: float = Field(ge=0, le=1)
    latency_p50_ms: float | None = Field(default=None, ge=0)
    latency_p95_ms: float | None = Field(default=None, ge=0)
    status_counts: dict[str, int]
    error_counts: dict[str, int]
    min_success_rate: float = Field(ge=0, le=1)
    min_schema_valid_rate: float = Field(ge=0, le=1)
    min_request_id_rate: float = Field(ge=0, le=1)
    max_p95_ms: float = Field(gt=0)
    passed: bool
    proof_boundary: Literal["bounded_authenticated_load"] = "bounded_authenticated_load"


class DeploymentReleaseManifest(_FrozenModel):
    """Non-secret identity of one immutable Compose release."""

    schema_version: Literal["2.0"] = "2.0"
    release_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{2,127}$")
    project_name: Literal["tickettune-production"]
    compose_file: str
    compose_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    env_file: str
    env_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model: str = Field(min_length=1)
    adapter_path: str = Field(min_length=1)
    adapter_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_manifest_file: str = Field(min_length=1)
    dataset_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    test_split_file: str = Field(min_length=1)
    test_split_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    training_manifest_file: str = Field(min_length=1)
    training_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    qualification_report_file: str = Field(min_length=1)
    qualification_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluation_manifest_file: str = Field(min_length=1)
    evaluation_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluation_report_file: str = Field(min_length=1)
    evaluation_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parity_report_file: str = Field(min_length=1)
    parity_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class _EvidenceView(BaseModel):
    """Framework-light strict view over release evidence produced elsewhere."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


_EvaluationBaselineEvidence = _EvaluationCandidateEvidence
_EvaluationBaselineReportEvidence = _EvaluationReportEvidence


class _EvaluationManifestEvidence(_EvidenceView):
    schema_version: Literal["1.0"]
    evaluation_id: str
    created_at: datetime
    status: Literal["completed", "failed"]
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_passed: bool
    comparison_passed: bool | None
    artifact_sha256: dict[str, str]
    result: _EvaluationResultEvidence


class ReleaseValidationReport(_FrozenModel):
    """Non-secret result of complete schema-2 release-evidence validation."""

    schema_version: Literal["1.0"] = "1.0"
    release_id: str
    project_name: str
    model: str
    release_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    compose_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    env_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    adapter_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    test_split_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    training_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    qualification_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluation_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluation_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parity_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    git_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    base_model: str
    model_revision: str | None
    production_profile: Literal["tickettune-vllm-production-v1"] = "tickettune-vllm-production-v1"
    production_profile_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    passed: Literal[True] = True
    proof_boundary: Literal["semantic_release_validation"] = "semantic_release_validation"
    check_use_limit: Literal[
        "artifacts_revalidated_before_report_only; revalidate_immediately_before_deployment"
    ] = "artifacts_revalidated_before_report_only; revalidate_immediately_before_deployment"


class ReleaseStartReport(_FrozenModel):
    """Receipt for a rechecked snapshot followed by a path-based Compose start."""

    schema_version: Literal["1.0"] = "1.0"
    release_id: str
    project_name: str
    model: str
    release_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    compose_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    adapter_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    production_profile: Literal["tickettune-vllm-production-v1"] = "tickettune-vllm-production-v1"
    production_profile_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    returncode: Literal[0] = 0
    executed: Literal[True] = True
    passed: Literal[True] = True
    proof_boundary: Literal["validated_compose_start_invocation"] = (
        "validated_compose_start_invocation"
    )
    acceptance_limit: Literal[
        "compose_exit_zero_only; authenticated_tls_readback_and_runtime_proof_remain_required"
    ] = "compose_exit_zero_only; authenticated_tls_readback_and_runtime_proof_remain_required"
    path_binding_limit: Literal[
        "snapshot_recheck_precedes_path_based_docker_and_in_container_opens; "
        "atomic_byte_binding_requires_immutable_release_storage"
    ] = (
        "snapshot_recheck_precedes_path_based_docker_and_in_container_opens; "
        "atomic_byte_binding_requires_immutable_release_storage"
    )


class RollbackPlan(_FrozenModel):
    """Shell-free, non-executing plan to replace current with previous release."""

    schema_version: Literal["1.0"] = "1.0"
    current_release_id: str
    previous_release_id: str
    current_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    previous_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    current_adapter_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    previous_adapter_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    stop_current_argv: tuple[str, ...]
    start_previous_argv: tuple[str, ...]
    executed: Literal[False] = False
    proof_boundary: Literal["rollback_plan_only"] = "rollback_plan_only"
    check_use_limit: Literal[
        "artifacts_verified_at_plan_creation_only; revalidate_immediately_before_execution"
    ] = "artifacts_verified_at_plan_creation_only; revalidate_immediately_before_execution"
    path_binding_limit: Literal[
        "rendered_argv_reopens_paths; atomic_byte_binding_requires_immutable_release_storage"
    ] = "rendered_argv_reopens_paths; atomic_byte_binding_requires_immutable_release_storage"


@dataclass(frozen=True)
class _RegularSnapshot:
    path: Path
    label: str
    payload: bytes
    sha256: str


@dataclass(frozen=True)
class _HeldOutExample:
    id: str
    expected: TriageOutput


@dataclass(frozen=True)
class _AdapterIdentity:
    path: Path
    inventory: tuple[dict[str, object], ...]
    inventory_sha256: str
    config_sha256: str
    weight_sha256: dict[str, str]


@dataclass(frozen=True)
class _ValidatedRelease:
    manifest_snapshot: _RegularSnapshot
    release: DeploymentReleaseManifest
    compose_snapshot: _RegularSnapshot
    env_snapshot: _RegularSnapshot
    adapter: _AdapterIdentity
    evidence_snapshots: tuple[_RegularSnapshot, ...]
    profile_snapshots: tuple[_RegularSnapshot, ...]
    production_profile_sha256: str
    training: _TrainingEvidence


@dataclass(frozen=True)
class _Outcome:
    status: int
    latency_ms: float
    schema_valid: bool
    request_id: bool
    error_code: str | None = None


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Never forward authorization headers to a redirect target."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        return None


def validate_base_url(base_url: str, *, allow_remote: bool) -> str:
    """Validate a serving origin without permitting paths, credentials, or silent remote use."""

    candidate = base_url.strip().rstrip("/")
    parsed = urlparse(candidate)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("base URL must be an absolute HTTP(S) origin")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("base URL must not contain credentials")
    if parsed.path not in {"", "/"} or parsed.params or parsed.query or parsed.fragment:
        raise ValueError("base URL must not contain a path, parameters, query, or fragment")
    remote = parsed.hostname.casefold() not in _LOOPBACK_HOSTS
    if remote and not allow_remote:
        raise ValueError("remote deployment proof requests require --allow-remote")
    if remote and parsed.scheme != "https":
        raise ValueError("remote deployment proof requests require HTTPS")
    return candidate


def _regular_file(path: Path, *, label: str) -> Path:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise ValueError(f"{label} must be a regular non-symlink file")
    resolved = expanded.resolve()
    if not resolved.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    return resolved


def read_api_key(path: Path) -> str:
    """Read a bounded API key from a file without accepting ambiguous whitespace."""

    expanded = path.expanduser()
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:  # pragma: no cover - TicketTune deployment targets are POSIX.
        raise RuntimeError("API key reads require O_NOFOLLOW support")
    flags = os.O_RDONLY | no_follow | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(expanded, flags)
    except OSError as exc:
        raise ValueError("API key file must be a regular non-symlink file") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("API key file must be a regular non-symlink file")
        if metadata.st_uid != os.getuid():
            raise ValueError("API key file must be owned by the current user")
        if stat.S_IMODE(metadata.st_mode) & 0o077:
            raise ValueError("API key file permissions must not grant group or world access")
        if metadata.st_size > 4096:
            raise ValueError("API key file exceeds 4096 bytes")
        chunks: list[bytes] = []
        remaining = 4097
        while remaining:
            chunk = os.read(descriptor, remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        if len(payload) > 4096:
            raise ValueError("API key file exceeds 4096 bytes")
    finally:
        os.close(descriptor)
    try:
        value = payload.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError("API key file must contain ASCII only") from exc
    if any(byte < 0x21 or byte > 0x7E for byte in payload):
        raise ValueError("API key must contain printable ASCII without whitespace")
    if len(payload) < MIN_API_KEY_BYTES:
        raise ValueError(f"API key must contain at least {MIN_API_KEY_BYTES} bytes")
    return value


def _validate_ca_cert(path: Path | None) -> Path | None:
    return _regular_file(path, label="CA certificate") if path is not None else None


def _default_transport(
    method: str,
    url: str,
    headers: dict[str, str],
    body: bytes | None,
    timeout: float,
    ca_cert: Path | None,
) -> HttpResponse:
    context = ssl.create_default_context(cafile=str(ca_cert) if ca_cert else None)
    # ``url`` is constructed only from the origin validated by ``validate_base_url``.
    request = urllib.request.Request(  # noqa: S310
        url, data=body, headers=headers, method=method
    )
    opener = urllib.request.build_opener(
        urllib.request.HTTPSHandler(context=context),
        _NoRedirectHandler(),
    )
    try:
        with opener.open(request, timeout=timeout) as response:  # nosec B310
            return HttpResponse(
                status=int(response.status),
                headers=dict(response.headers.items()),
                body=_read_bounded_response_body(response),
            )
    except urllib.error.HTTPError as exc:
        return HttpResponse(
            status=int(exc.code),
            headers=dict(exc.headers.items()) if exc.headers else {},
            body=_read_bounded_response_body(exc),
        )


def _read_bounded_response_body(response: Any) -> bytes:
    """Read at most the redacted proof budget from an HTTP response."""

    headers = getattr(response, "headers", None)
    content_length = headers.get("Content-Length") if headers is not None else None
    if content_length is not None:
        try:
            declared_length = int(content_length)
        except (TypeError, ValueError):
            declared_length = None
        if declared_length is not None and declared_length > MAX_RESPONSE_BYTES:
            raise ResponseTooLargeError(
                f"response body exceeds {MAX_RESPONSE_BYTES} byte proof limit"
            )
    payload = response.read(MAX_RESPONSE_BYTES + 1)
    if len(payload) > MAX_RESPONSE_BYTES:
        raise ResponseTooLargeError(f"response body exceeds {MAX_RESPONSE_BYTES} byte proof limit")
    return bytes(payload)


def _header_present(headers: Mapping[str, str], name: str) -> bool:
    expected = name.casefold()
    return any(key.casefold() == expected and bool(value.strip()) for key, value in headers.items())


def _json_object(body: bytes) -> dict[str, Any] | None:
    try:
        value = loads_strict(body)
    except (json.JSONDecodeError, UnicodeDecodeError, StrictJSONError):
        return None
    return value if isinstance(value, dict) else None


def _chat_schema_valid(body: bytes) -> bool:
    payload = _json_object(body)
    if payload is None:
        return False
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return False
    if not isinstance(content, str):
        return False
    try:
        decoded = loads_strict(content)
        TriageOutput.model_validate(decoded)
    except (json.JSONDecodeError, StrictJSONError, ValueError):
        return False
    return True


def _request_body(model: str) -> bytes:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _DEFAULT_TICKET},
        ],
        "temperature": 0,
        "max_tokens": 256,
    }
    return canonical_json_bytes(payload)


def _authorization_headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def _models_serving_claims(
    body: bytes,
    *,
    model: str,
    expected_base_model: str,
) -> tuple[bool, bool]:
    payload = _json_object(body)
    records = payload.get("data") if payload is not None else None
    if not isinstance(records, list):
        return False, False
    for record in records:
        if isinstance(record, dict) and record.get("id") == model:
            return True, record.get("parent") == expected_base_model
    return False, False


def run_readback(
    *,
    base_url: str,
    api_key_file: Path,
    model: str,
    expected_base_model: str,
    ca_cert: Path | None = None,
    timeout: float = 30,
    allow_remote: bool = False,
    output_path: Path | None = None,
    _transport: Transport | None = None,
) -> ReadbackReport:
    """Check endpoint-reported model claims and one response without retaining model text."""

    origin = validate_base_url(base_url, allow_remote=allow_remote)
    uses_tls = urlparse(origin).scheme == "https"
    key = read_api_key(api_key_file)
    certificate = _validate_ca_cert(ca_cert)
    if timeout <= 0 or not math.isfinite(timeout):
        raise ValueError("timeout must be a finite positive number")
    transport = _transport or _default_transport
    headers = _authorization_headers(key)
    failures: list[str] = []

    try:
        models_response = transport(
            "GET", f"{origin}/v1/models", headers, None, timeout, certificate
        )
        model_found, parent_matches = _models_serving_claims(
            models_response.body,
            model=model,
            expected_base_model=expected_base_model,
        )
        models_request_id = _header_present(models_response.headers, "x-request-id")
    except Exception as exc:  # proof must retain a redacted failure result
        models_response = HttpResponse(status=0, headers={}, body=b"")
        model_found = False
        parent_matches = False
        models_request_id = False
        failures.append(f"models_{type(exc).__name__}")

    try:
        chat_response = transport(
            "POST",
            f"{origin}/v1/chat/completions",
            headers,
            _request_body(model),
            timeout,
            certificate,
        )
        chat_valid = chat_response.status == 200 and _chat_schema_valid(chat_response.body)
        chat_request_id = _header_present(chat_response.headers, "x-request-id")
    except Exception as exc:  # proof must retain a redacted failure result
        chat_response = HttpResponse(status=0, headers={}, body=b"")
        chat_valid = False
        chat_request_id = False
        failures.append(f"chat_{type(exc).__name__}")

    if models_response.status != 200:
        failures.append("models_http_status")
    if not model_found:
        failures.append("model_not_found")
    if not parent_matches:
        failures.append("base_model_parent_mismatch")
    if chat_response.status != 200:
        failures.append("chat_http_status")
    if not chat_valid:
        failures.append("chat_schema_invalid")
    request_ids = int(models_request_id) + int(chat_request_id)
    if request_ids != 2:
        failures.append("request_id_missing")
    report = ReadbackReport(
        created_at=datetime.now(UTC),
        base_url=origin,
        model=model,
        expected_base_model=expected_base_model,
        models_status=models_response.status,
        chat_status=chat_response.status,
        model_found=model_found,
        parent_matches=parent_matches,
        chat_schema_valid=chat_valid,
        request_ids_received=request_ids,
        request_id_rate=request_ids / 2,
        passed=(
            models_response.status == 200
            and chat_response.status == 200
            and model_found
            and parent_matches
            and chat_valid
            and request_ids == 2
        ),
        failure_codes=tuple(dict.fromkeys(failures)),
        proof_boundary=(
            "authenticated_tls_serving_claim_readback"
            if uses_tls
            else "authenticated_loopback_http_serving_claim_readback"
        ),
    )
    if output_path is not None:
        write_proof_report(output_path, report)
    return report


def _one_load_request(
    *,
    origin: str,
    headers: dict[str, str],
    body: bytes,
    timeout: float,
    certificate: Path | None,
    transport: Transport,
) -> _Outcome:
    started = time.perf_counter()
    try:
        response = transport(
            "POST",
            f"{origin}/v1/chat/completions",
            headers,
            body,
            timeout,
            certificate,
        )
        elapsed = (time.perf_counter() - started) * 1000
        return _Outcome(
            status=response.status,
            latency_ms=elapsed,
            schema_valid=response.status == 200 and _chat_schema_valid(response.body),
            request_id=_header_present(response.headers, "x-request-id"),
        )
    except Exception as exc:
        elapsed = (time.perf_counter() - started) * 1000
        return _Outcome(
            status=0,
            latency_ms=elapsed,
            schema_valid=False,
            request_id=False,
            error_code=type(exc).__name__,
        )


def _nearest_rank(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(quantile * len(ordered)))
    return ordered[rank - 1]


def _rate(value: float, *, label: str) -> float:
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{label} must be between 0 and 1")
    return value


def run_load_test(
    *,
    base_url: str,
    api_key_file: Path,
    model: str,
    requests: int = 20,
    concurrency: int = 2,
    min_success_rate: float = 1.0,
    min_schema_valid_rate: float = 1.0,
    min_request_id_rate: float = 1.0,
    max_p95_ms: float = 5000,
    ca_cert: Path | None = None,
    timeout: float = 60,
    allow_remote: bool = False,
    output_path: Path | None = None,
    _transport: Transport | None = None,
) -> LoadTestReport:
    """Run a bounded authenticated load check and retain only aggregate evidence."""

    if requests < 1:
        raise ValueError("requests must be at least 1")
    if requests > MAX_LOAD_REQUESTS:
        raise ValueError(f"requests cannot exceed {MAX_LOAD_REQUESTS}")
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    if concurrency > MAX_LOAD_CONCURRENCY:
        raise ValueError(f"concurrency cannot exceed {MAX_LOAD_CONCURRENCY}")
    if concurrency > requests:
        raise ValueError("concurrency cannot exceed requests")
    minimum_success = _rate(min_success_rate, label="min_success_rate")
    minimum_schema = _rate(min_schema_valid_rate, label="min_schema_valid_rate")
    minimum_request_id = _rate(min_request_id_rate, label="min_request_id_rate")
    if not math.isfinite(max_p95_ms) or max_p95_ms <= 0:
        raise ValueError("max_p95_ms must be a finite positive number")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be a finite positive number")

    origin = validate_base_url(base_url, allow_remote=allow_remote)
    key = read_api_key(api_key_file)
    certificate = _validate_ca_cert(ca_cert)
    transport = _transport or _default_transport
    headers = _authorization_headers(key)
    body = _request_body(model)
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(
                _one_load_request,
                origin=origin,
                headers=headers,
                body=body,
                timeout=timeout,
                certificate=certificate,
                transport=transport,
            )
            for _index in range(requests)
        ]
        outcomes = [future.result() for future in futures]

    successes = sum(item.status == 200 for item in outcomes)
    schema_valid = sum(item.schema_valid for item in outcomes)
    request_ids = sum(item.request_id for item in outcomes)
    success_rate = successes / requests
    schema_rate = schema_valid / requests
    request_id_rate = request_ids / requests
    p50 = _nearest_rank([item.latency_ms for item in outcomes], 0.50)
    p95 = _nearest_rank([item.latency_ms for item in outcomes], 0.95)
    report = LoadTestReport(
        created_at=datetime.now(UTC),
        base_url=origin,
        model=model,
        requests=requests,
        concurrency=concurrency,
        successes=successes,
        schema_valid_responses=schema_valid,
        request_ids_received=request_ids,
        success_rate=success_rate,
        schema_valid_rate=schema_rate,
        request_id_rate=request_id_rate,
        latency_p50_ms=p50,
        latency_p95_ms=p95,
        status_counts=dict(
            sorted(Counter(str(item.status) for item in outcomes if item.status).items())
        ),
        error_counts=dict(
            sorted(Counter(item.error_code for item in outcomes if item.error_code).items())
        ),
        min_success_rate=minimum_success,
        min_schema_valid_rate=minimum_schema,
        min_request_id_rate=minimum_request_id,
        max_p95_ms=max_p95_ms,
        passed=(
            success_rate >= minimum_success
            and schema_rate >= minimum_schema
            and request_id_rate >= minimum_request_id
            and p95 is not None
            and p95 <= max_p95_ms
        ),
    )
    if output_path is not None:
        write_proof_report(output_path, report)
    return report


def write_proof_report(path: Path, value: object) -> Path:
    """Atomically create an immutable redacted proof report."""

    payload = canonical_json_bytes(value, pretty=True)
    destination = _reject_symlink_components(path, label="proof report path")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination = _reject_symlink_components(destination, label="proof report path")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination, follow_symlinks=False)
        except FileExistsError:
            if _existing_regular_file_matches(destination, payload):
                return destination
            raise FileExistsError(
                f"refusing to overwrite immutable proof report: {destination}"
            ) from None
        _fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _reject_symlink_components(path: Path, *, label: str) -> Path:
    """Return an absolute lexical path after rejecting every existing symlink component."""

    expanded = path.expanduser()
    absolute = Path(os.path.abspath(expanded))
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"{label} must not contain symlink components: {current}")
    return absolute


def _stat_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _read_regular_snapshot(path: Path, *, label: str) -> _RegularSnapshot:
    """Read one stable regular-file snapshot without following any symlink component."""

    absolute = _reject_symlink_components(path, label=label)
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:  # pragma: no cover - deployment targets are POSIX Linux.
        raise RuntimeError("release evidence reads require O_NOFOLLOW support")
    flags = os.O_RDONLY | no_follow | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(absolute, flags)
    except OSError as exc:
        raise ValueError(f"{label} must be a readable regular non-symlink file") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} must be a regular non-symlink file")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
        current = os.lstat(absolute)
    except OSError as exc:
        raise ValueError(f"{label} changed while it was being read") from exc
    finally:
        os.close(descriptor)
    if (
        _stat_identity(before) != _stat_identity(after)
        or (current.st_dev, current.st_ino) != (before.st_dev, before.st_ino)
        or len(payload) != before.st_size
    ):
        raise ValueError(f"{label} changed while it was being read")
    return _RegularSnapshot(
        path=absolute,
        label=label,
        payload=payload,
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def _strict_json_object(snapshot: _RegularSnapshot) -> dict[str, Any]:
    try:
        payload = loads_strict(snapshot.payload.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, StrictJSONError) as exc:
        raise ValueError(f"invalid {snapshot.label}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"invalid {snapshot.label}: expected a JSON object")
    return payload


def _evidence_model[EvidenceModel: BaseModel](
    snapshot: _RegularSnapshot,
    model: type[EvidenceModel],
) -> tuple[dict[str, Any], EvidenceModel]:
    payload = _strict_json_object(snapshot)
    try:
        evidence = model.model_validate_json(snapshot.payload, strict=True)
    except ValidationError as exc:
        raise ValueError(f"invalid {snapshot.label}: {exc}") from exc
    return payload, evidence


def _existing_regular_file_matches(path: Path, expected: bytes) -> bool:
    """Read one existing final component without following a symlink."""

    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:  # pragma: no cover - TicketTune deployment targets are POSIX.
        return False
    try:
        descriptor = os.open(path, os.O_RDONLY | no_follow | getattr(os, "O_CLOEXEC", 0))
    except OSError:
        return False
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != len(expected):
            return False
        chunks: list[bytes] = []
        remaining = len(expected) + 1
        while remaining:
            chunk = os.read(descriptor, remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks) == expected
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


_RELEASE_DIGEST_FIELDS = (
    "compose_sha256",
    "env_sha256",
    "adapter_sha256",
    "dataset_manifest_sha256",
    "test_split_sha256",
    "training_manifest_sha256",
    "qualification_report_sha256",
    "evaluation_manifest_sha256",
    "evaluation_report_sha256",
    "parity_report_sha256",
)
_RELEASE_PATH_FIELDS = (
    "compose_file",
    "env_file",
    "adapter_path",
    "dataset_manifest_file",
    "test_split_file",
    "training_manifest_file",
    "qualification_report_file",
    "evaluation_manifest_file",
    "evaluation_report_file",
    "parity_report_file",
)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_GIT_SHA_PATTERN = re.compile(r"[0-9a-f]{40}\Z")
_COMPOSE_REQUIRED_VARIABLE_PATTERN = re.compile(rb"\$\{([A-Za-z_][A-Za-z0-9_]*):\?")


def _load_release_manifest(path: Path) -> tuple[_RegularSnapshot, DeploymentReleaseManifest]:
    snapshot = _read_regular_snapshot(path, label="release manifest")
    payload = _strict_json_object(snapshot)
    try:
        release = DeploymentReleaseManifest.model_validate(payload, strict=True)
    except ValidationError as exc:
        raise ValueError(f"invalid release manifest: {exc}") from exc
    return snapshot, release


def _validate_release_sentinels(release: DeploymentReleaseManifest) -> None:
    for field in _RELEASE_DIGEST_FIELDS:
        value = getattr(release, field)
        if len(set(value)) == 1:
            raise ValueError(f"release {field} is a repeated-character digest sentinel")
    for field in _RELEASE_PATH_FIELDS:
        value = getattr(release, field)
        if "replace-with" in value.casefold():
            raise ValueError(f"release {field} still contains a placeholder path")
    if "replace-with" in release.release_id.casefold():
        raise ValueError("release ID still contains a placeholder")


def _located_release_path(manifest_path: Path, raw_path: str) -> Path:
    candidate = Path(raw_path).expanduser()
    return candidate if candidate.is_absolute() else manifest_path.parent / candidate


def _bound_release_file(
    manifest_path: Path,
    raw_path: str,
    expected_sha256: str,
    *,
    label: str,
) -> _RegularSnapshot:
    snapshot = _read_regular_snapshot(
        _located_release_path(manifest_path, raw_path),
        label=label,
    )
    if snapshot.sha256 != expected_sha256:
        raise ValueError(f"{label} SHA-256 mismatch")
    return snapshot


def _validate_production_profile(
    compose_snapshot: _RegularSnapshot,
) -> tuple[tuple[_RegularSnapshot, ...], str]:
    if compose_snapshot.sha256 != APPROVED_PRODUCTION_COMPOSE_SHA256:
        raise ValueError("compose file does not match the approved TicketTune production profile")
    snapshots: list[_RegularSnapshot] = []
    for filename, expected_sha256 in _APPROVED_PRODUCTION_SUPPORT_SHA256:
        snapshot = _read_regular_snapshot(
            compose_snapshot.path.parent / filename,
            label=f"production profile {filename}",
        )
        if snapshot.sha256 != expected_sha256:
            raise ValueError(
                f"{filename} does not match the approved TicketTune production profile"
            )
        snapshots.append(snapshot)
    inventory = {
        "profile": APPROVED_PRODUCTION_PROFILE,
        "files": [
            {"path": "compose.yaml", "sha256": compose_snapshot.sha256},
            *[
                {"path": filename, "sha256": expected_sha256}
                for filename, expected_sha256 in _APPROVED_PRODUCTION_SUPPORT_SHA256
            ],
        ],
    }
    return tuple(snapshots), sha256_bytes(inventory)


def _regular_adapter_directory(path: Path) -> Path:
    absolute = _reject_symlink_components(path, label="adapter directory")
    try:
        metadata = absolute.lstat()
    except OSError as exc:
        raise ValueError("adapter directory must exist") from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("adapter directory must be a regular non-symlink directory")
    return absolute


def _hash_regular_file_snapshot(path: Path, *, label: str) -> tuple[str, int]:
    absolute = _reject_symlink_components(path, label=label)
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:  # pragma: no cover - deployment targets are POSIX Linux.
        raise RuntimeError("adapter reads require O_NOFOLLOW support")
    flags = os.O_RDONLY | no_follow | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(absolute, flags)
    except OSError as exc:
        raise ValueError(f"{label} must be a regular non-symlink file") from exc
    digest = hashlib.sha256()
    size_bytes = 0
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} must be a regular file")
        while block := os.read(descriptor, 1024 * 1024):
            digest.update(block)
            size_bytes += len(block)
        after = os.fstat(descriptor)
        current = os.lstat(absolute)
    except OSError as exc:
        raise ValueError(f"{label} changed while it was being hashed") from exc
    finally:
        os.close(descriptor)
    if (
        _stat_identity(before) != _stat_identity(after)
        or (current.st_dev, current.st_ino) != (before.st_dev, before.st_ino)
        or size_bytes != before.st_size
    ):
        raise ValueError(f"{label} changed while it was being hashed")
    return digest.hexdigest(), size_bytes


def _adapter_inventory(path: Path) -> tuple[Path, tuple[dict[str, object], ...]]:
    adapter = _regular_adapter_directory(path)
    entries: list[dict[str, object]] = []
    for candidate in sorted(adapter.rglob("*"), key=lambda item: item.as_posix()):
        try:
            metadata = candidate.lstat()
        except OSError as exc:
            raise ValueError(f"adapter entry changed during inventory: {candidate}") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"adapter directory contains a symlink: {candidate}")
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"adapter directory contains a non-regular file: {candidate}")
        digest, size_bytes = _hash_regular_file_snapshot(
            candidate,
            label="adapter file",
        )
        entries.append(
            {
                "path": candidate.relative_to(adapter).as_posix(),
                "sha256": digest,
                "size_bytes": size_bytes,
            }
        )
    if not entries:
        raise ValueError("adapter directory must contain at least one regular file")
    return adapter, tuple(entries)


def adapter_inventory_sha256(path: Path) -> str:
    """Hash a deterministic, non-symlink adapter-directory inventory."""

    _adapter, inventory = _adapter_inventory(path)
    return sha256_bytes(inventory)


def _bound_release_adapter(
    manifest_path: Path,
    raw_path: str,
    expected_sha256: str,
) -> _AdapterIdentity:
    adapter, inventory = _adapter_inventory(_located_release_path(manifest_path, raw_path))
    inventory_sha256 = sha256_bytes(inventory)
    if inventory_sha256 != expected_sha256:
        raise ValueError("adapter directory SHA-256 mismatch")
    config_entry = next(
        (entry for entry in inventory if entry["path"] == "adapter_config.json"),
        None,
    )
    if config_entry is None:
        raise ValueError("adapter must contain adapter_config.json")
    weights = {
        str(entry["path"]): str(entry["sha256"])
        for entry in inventory
        if str(entry["path"]).endswith(".safetensors") and "/" not in str(entry["path"])
    }
    if not weights:
        raise ValueError("adapter must contain at least one root Safetensors weight file")
    return _AdapterIdentity(
        path=adapter,
        inventory=inventory,
        inventory_sha256=inventory_sha256,
        config_sha256=str(config_entry["sha256"]),
        weight_sha256=weights,
    )


def _literal_release_env(snapshot: _RegularSnapshot) -> dict[str, str]:
    try:
        text = snapshot.payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("env file must be readable UTF-8") from exc
    if "\x00" in text:
        raise ValueError("env file must not contain NUL bytes")
    values: dict[str, str] = {}
    for line_number, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator or re.fullmatch(r"[A-Z][A-Z0-9_]*", key) is None:
            raise ValueError(f"env file line {line_number} is not a literal KEY=value entry")
        if key in values:
            raise ValueError(f"env file contains duplicate key {key}")
        if not value or value != value.strip():
            raise ValueError(f"env {key} must contain one non-empty literal value")
        if any(token in value for token in ("$", "`", "\r", "\n")):
            raise ValueError(f"env {key} must not use expansion or command syntax")
        values[key] = value
    return values


def _literal_bound_path(raw_path: str, *, anchor: Path, label: str) -> Path:
    candidate = Path(raw_path).expanduser()
    located = candidate if candidate.is_absolute() else anchor / candidate
    return _reject_symlink_components(located, label=label)


def _bounded_integer(
    value: str,
    *,
    label: str,
    minimum: int,
    maximum: int,
) -> int:
    if re.fullmatch(r"[0-9]+", value) is None:
        raise ValueError(f"env {label} must be an integer")
    parsed = int(value)
    if not minimum <= parsed <= maximum:
        raise ValueError(f"env {label} must be between {minimum} and {maximum}")
    return parsed


def _bounded_float(
    value: str,
    *,
    label: str,
    minimum_exclusive: float,
    maximum: float,
) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"env {label} must be numeric") from exc
    if not math.isfinite(parsed) or not minimum_exclusive < parsed <= maximum:
        raise ValueError(
            f"env {label} must be greater than {minimum_exclusive:g} and at most {maximum:g}"
        )
    return parsed


def _bounded_memory_bytes(value: str, *, label: str) -> int:
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([kKmMgGtT]?)(?:[bB])?", value)
    if match is None:
        raise ValueError(f"env {label} must be a positive Docker byte size")
    amount = float(match.group(1))
    if not math.isfinite(amount) or amount <= 0:
        raise ValueError(f"env {label} must be a positive Docker byte size")
    multiplier = {
        "": 1,
        "K": 1024,
        "M": 1024**2,
        "G": 1024**3,
        "T": 1024**4,
    }[match.group(2).upper()]
    parsed = int(amount * multiplier)
    if not 1 <= parsed <= MAX_MEMORY_BYTES:
        raise ValueError(f"env {label} must resolve to between 1 byte and {MAX_MEMORY_BYTES} bytes")
    return parsed


def _bounded_retention_seconds(value: str) -> int:
    match = re.fullmatch(r"([1-9][0-9]*)(ms|s|m|h|d|w|y)", value)
    if match is None:
        raise ValueError("env PROMETHEUS_RETENTION must be one positive duration")
    multiplier = {
        "s": 1,
        "m": 60,
        "h": 60 * 60,
        "d": 24 * 60 * 60,
        "w": 7 * 24 * 60 * 60,
        "y": 365 * 24 * 60 * 60,
    }
    amount = int(match.group(1))
    unit = match.group(2)
    seconds = amount // 1000 if unit == "ms" else amount * multiplier[unit]
    if not MIN_PROMETHEUS_RETENTION_SECONDS <= seconds <= MAX_PROMETHEUS_RETENTION_SECONDS:
        raise ValueError("env PROMETHEUS_RETENTION must be between 1 hour and 31 days")
    return seconds


def _require_env_operational_invariants(
    values: Mapping[str, str],
    adapter: _AdapterIdentity,
) -> None:
    effective = {
        "BIND_ADDRESS": "127.0.0.1",
        "TLS_PORT": "8443",
        "VLLM_DTYPE": "auto",
        "MAX_MODEL_LEN": "2048",
        "MAX_NUM_SEQS": "8",
        "MAX_NUM_BATCHED_TOKENS": "4096",
        "MAX_LORA_RANK": "64",
        "TENSOR_PARALLEL_SIZE": "1",
        "GPU_MEMORY_UTILIZATION": "0.90",
        "VLLM_SHM_SIZE": "8gb",
        "GATEWAY_CPUS": "1.0",
        "GATEWAY_MEMORY": "512M",
        "VLLM_CPUS": "8.0",
        "VLLM_MEMORY": "32G",
        "PROMETHEUS_CPUS": "1.0",
        "PROMETHEUS_MEMORY": "1G",
        "PROMETHEUS_RETENTION": "24h",
        "PROMETHEUS_RETENTION_SIZE": "5GB",
        "ALERTMANAGER_CPUS": "0.5",
        "ALERTMANAGER_MEMORY": "256M",
    }
    effective.update({name: value for name, value in values.items() if name in effective})
    if effective["BIND_ADDRESS"] not in {"127.0.0.1", "localhost", "[::1]"}:
        raise ValueError("env BIND_ADDRESS must bind the production gateway to loopback")
    _bounded_integer(
        effective["TLS_PORT"],
        label="TLS_PORT",
        minimum=1,
        maximum=65_535,
    )
    _bounded_integer(
        effective["MAX_MODEL_LEN"],
        label="MAX_MODEL_LEN",
        minimum=2_048,
        maximum=2_048,
    )
    _bounded_integer(
        effective["MAX_NUM_SEQS"],
        label="MAX_NUM_SEQS",
        minimum=1,
        maximum=8,
    )
    _bounded_integer(
        effective["MAX_NUM_BATCHED_TOKENS"],
        label="MAX_NUM_BATCHED_TOKENS",
        minimum=2_048,
        maximum=4_096,
    )
    max_lora_rank = _bounded_integer(
        effective["MAX_LORA_RANK"],
        label="MAX_LORA_RANK",
        minimum=1,
        maximum=64,
    )
    _bounded_integer(
        effective["TENSOR_PARALLEL_SIZE"],
        label="TENSOR_PARALLEL_SIZE",
        minimum=1,
        maximum=8,
    )
    _bounded_float(
        effective["GPU_MEMORY_UTILIZATION"],
        label="GPU_MEMORY_UTILIZATION",
        minimum_exclusive=0,
        maximum=0.90,
    )
    if effective["VLLM_DTYPE"] not in {"auto", "bfloat16"}:
        raise ValueError("env VLLM_DTYPE must be auto or bfloat16 for the approved profile")
    approved_cpu_limits = {
        "GATEWAY_CPUS": 1.0,
        "VLLM_CPUS": 8.0,
        "PROMETHEUS_CPUS": 1.0,
        "ALERTMANAGER_CPUS": 0.5,
    }
    for name, maximum in approved_cpu_limits.items():
        _bounded_float(
            effective[name],
            label=name,
            minimum_exclusive=0,
            maximum=maximum,
        )
    approved_memory_limits = {
        "VLLM_SHM_SIZE": 8 * 1024**3,
        "GATEWAY_MEMORY": 512 * 1024**2,
        "VLLM_MEMORY": 32 * 1024**3,
        "PROMETHEUS_MEMORY": 1024**3,
        "PROMETHEUS_RETENTION_SIZE": 5 * 1024**3,
        "ALERTMANAGER_MEMORY": 256 * 1024**2,
    }
    for name, maximum in approved_memory_limits.items():
        if _bounded_memory_bytes(effective[name], label=name) > maximum:
            raise ValueError(f"env {name} exceeds the approved production profile maximum")
    if _bounded_retention_seconds(effective["PROMETHEUS_RETENTION"]) < 24 * 60 * 60:
        raise ValueError("env PROMETHEUS_RETENTION is below the approved 24-hour minimum")

    secret_group_id = values.get("SECRET_GROUP_ID")
    if secret_group_id is None:  # pragma: no cover - required by Compose contract.
        raise ValueError("env SECRET_GROUP_ID is required")
    _bounded_integer(
        secret_group_id,
        label="SECRET_GROUP_ID",
        minimum=1,
        maximum=2_147_483_647,
    )
    adapter_config = _adapter_config_payload(adapter)
    rank = adapter_config.get("r")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        raise ValueError("adapter config requires a positive integer rank")
    if max_lora_rank < rank:
        raise ValueError("env MAX_LORA_RANK is below the release adapter rank")


def _require_env_identity(
    snapshot: _RegularSnapshot,
    release: DeploymentReleaseManifest,
    adapter: _AdapterIdentity,
    training: _TrainingEvidence,
) -> dict[str, str]:
    values = _literal_release_env(snapshot)
    required = {
        "RELEASE_ID",
        "RELEASE_GIT_REVISION",
        "SERVED_MODEL_NAME",
        "ADAPTER_PATH",
        "EXPECTED_ADAPTER_SHA256",
        "BASE_MODEL",
        "MODEL_REVISION",
    }
    missing = sorted(required - values.keys())
    if missing:
        raise ValueError(f"env file is missing release identity keys: {', '.join(missing)}")
    if values["RELEASE_ID"] != release.release_id:
        raise ValueError("env RELEASE_ID does not match the release manifest")
    if values["RELEASE_GIT_REVISION"] != training.git_revision:
        raise ValueError("env RELEASE_GIT_REVISION does not match training evidence")
    if values["SERVED_MODEL_NAME"] != release.model:
        raise ValueError("env SERVED_MODEL_NAME does not match the release manifest")
    env_adapter = _literal_bound_path(
        values["ADAPTER_PATH"],
        anchor=snapshot.path.parent,
        label="env ADAPTER_PATH",
    )
    if env_adapter != adapter.path:
        raise ValueError("env ADAPTER_PATH does not resolve to the release adapter directory")
    if values["EXPECTED_ADAPTER_SHA256"] != release.adapter_sha256:
        raise ValueError("env EXPECTED_ADAPTER_SHA256 does not match the release adapter")
    if values["BASE_MODEL"] != training.model_name_or_path:
        raise ValueError("env BASE_MODEL does not match training evidence")
    if values["MODEL_REVISION"] != training.model_revision:
        raise ValueError("env MODEL_REVISION does not match training evidence")
    _require_env_operational_invariants(values, adapter)
    return values


def _require_compose_env_contract(
    compose_snapshot: _RegularSnapshot,
    values: Mapping[str, str],
) -> None:
    required = {
        match.group(1).decode("ascii")
        for match in _COMPOSE_REQUIRED_VARIABLE_PATTERN.finditer(compose_snapshot.payload)
    }
    missing = sorted(required - values.keys())
    if missing:
        raise ValueError(
            "env file is missing required production Compose keys: " + ", ".join(missing)
        )


def _require_digest_mapping(values: Mapping[str, str], *, label: str) -> None:
    if not values:
        raise ValueError(f"{label} must not be empty")
    for name, digest in values.items():
        if not name or _SHA256_PATTERN.fullmatch(digest) is None:
            raise ValueError(f"{label} contains an invalid SHA-256")
        if len(set(digest)) == 1:
            raise ValueError(f"{label} contains a repeated-character digest sentinel")


def _require_unique_exact_names(
    names: list[str],
    expected: tuple[str, ...] | frozenset[str],
    *,
    label: str,
) -> None:
    if len(names) != len(set(names)):
        raise ValueError(f"{label} contains duplicate names")
    expected_names = set(expected)
    observed_names = set(names)
    if observed_names != expected_names:
        missing = sorted(expected_names - observed_names)
        unexpected = sorted(observed_names - expected_names)
        raise ValueError(
            f"{label} metric/policy set is incomplete: missing={missing}, unexpected={unexpected}"
        )


def _same_number(first: float, second: float) -> bool:
    return math.isclose(first, second, rel_tol=0, abs_tol=1e-12)


def _evaluation_thresholds_by_metric(
    thresholds: Sequence[_EvaluationThresholdEvidence],
    *,
    expected: tuple[str, ...],
    label: str,
) -> dict[str, _EvaluationThresholdEvidence]:
    _require_unique_exact_names(
        [item.metric for item in thresholds],
        expected,
        label=label,
    )
    return {item.metric: item for item in thresholds}


def _evaluation_minimums(training: _TrainingEvidence) -> dict[str, float]:
    evaluation = training.config.get("evaluation")
    thresholds = evaluation.get("thresholds") if isinstance(evaluation, dict) else None
    if not isinstance(thresholds, dict):
        raise ValueError("training config has no evaluation threshold contract")
    values: dict[str, float] = {}
    for metric in _EVALUATION_ABSOLUTE_METRICS:
        value = thresholds.get(metric)
        if isinstance(value, bool) or not isinstance(value, (float, int)):
            raise ValueError(f"training evaluation threshold {metric} is invalid")
        numeric = float(value)
        if not math.isfinite(numeric) or not 0 <= numeric <= 1:
            raise ValueError(f"training evaluation threshold {metric} is invalid")
        values[metric] = numeric
    return values


def _validate_absolute_evaluation_report(
    report: _EvaluationReportEvidence | _EvaluationBaselineReportEvidence,
    *,
    minimums: Mapping[str, float],
    label: str,
) -> tuple[str, ...]:
    identifiers = tuple(item.id for item in report.results)
    if not identifiers or len(identifiers) != len(set(identifiers)):
        raise ValueError(f"{label} results require non-empty unique ordered IDs")
    if report.summary.examples != len(identifiers):
        raise ValueError(f"{label} summary example count differs from its results")
    thresholds = _evaluation_thresholds_by_metric(
        report.thresholds,
        expected=_EVALUATION_ABSOLUTE_METRICS,
        label=f"{label} absolute thresholds",
    )
    for metric in _EVALUATION_ABSOLUTE_METRICS:
        threshold = thresholds[metric]
        summary_value = float(getattr(report.summary, metric))
        minimum = minimums[metric]
        if not _same_number(threshold.value, summary_value):
            raise ValueError(f"{label} threshold {metric} differs from its summary value")
        if not _same_number(threshold.minimum, minimum):
            raise ValueError(f"{label} threshold {metric} differs from the training config")
        if threshold.passed != (threshold.value >= threshold.minimum):
            raise ValueError(f"{label} threshold {metric} has an inconsistent pass result")
    expected_passed = all(item.passed for item in thresholds.values())
    if report.passed != expected_passed:
        raise ValueError(f"{label} overall pass result is inconsistent with its thresholds")
    return identifiers


def _generated_prediction_rows(
    snapshot: _RegularSnapshot,
    *,
    label: str,
) -> tuple[_GeneratedPredictionEvidence, ...]:
    try:
        text = snapshot.payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} must be UTF-8 JSONL") from exc
    rows: list[_GeneratedPredictionEvidence] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            raise ValueError(f"{label} line {line_number} must not be blank")
        try:
            decoded = loads_strict(line)
            if not isinstance(decoded, dict):
                raise ValueError("expected a JSON object")
            row = _GeneratedPredictionEvidence.model_validate_json(line, strict=True)
        except (json.JSONDecodeError, StrictJSONError, ValidationError, ValueError) as exc:
            raise ValueError(
                f"{label} line {line_number} is not a producer-valid prediction"
            ) from exc
        rows.append(row)
    if not rows:
        raise ValueError(f"{label} must contain at least one prediction")
    identifiers = [row.id for row in rows]
    if any(not identifier for identifier in identifiers) or len(identifiers) != len(
        set(identifiers)
    ):
        raise ValueError(f"{label} require non-empty unique ordered IDs")
    canonical_payload = b"".join(
        canonical_json_bytes(row.model_dump(mode="json")) + b"\n" for row in rows
    )
    if snapshot.payload != canonical_payload:
        raise ValueError(f"{label} is not canonical producer JSONL")
    return tuple(rows)


def _scored_prediction_rows(
    snapshot: _RegularSnapshot,
    *,
    label: str,
) -> tuple[_EvaluationScoreEvidence, ...]:
    try:
        text = snapshot.payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} must be UTF-8 JSONL") from exc
    rows: list[_EvaluationScoreEvidence] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            raise ValueError(f"{label} line {line_number} must not be blank")
        try:
            decoded = loads_strict(line)
            if not isinstance(decoded, dict):
                raise ValueError("expected a JSON object")
            row = _EvaluationScoreEvidence.model_validate_json(line, strict=True)
        except (json.JSONDecodeError, StrictJSONError, ValidationError, ValueError) as exc:
            raise ValueError(f"{label} line {line_number} is not a producer-valid score") from exc
        rows.append(row)
    canonical_payload = b"".join(
        canonical_json_bytes(row.model_dump(mode="json")) + b"\n" for row in rows
    )
    if snapshot.payload != canonical_payload:
        raise ValueError(f"{label} is not canonical producer JSONL")
    return tuple(rows)


def _validate_report_recomputation(
    report: _EvaluationReportEvidence,
    prediction_rows: tuple[_GeneratedPredictionEvidence, ...],
    scored_rows: tuple[_EvaluationScoreEvidence, ...],
    expected_examples: tuple[_HeldOutExample, ...],
    *,
    label: str,
) -> None:
    expected_ids = tuple(item.id for item in expected_examples)
    prediction_ids = tuple(item.id for item in prediction_rows)
    if prediction_ids != expected_ids:
        raise ValueError(
            f"{label} prediction IDs differ from the prepared test split and qualified "
            "held-out cohort"
        )
    if tuple(item.id for item in report.results) != expected_ids:
        raise ValueError(
            f"{label} report IDs differ from the prepared test split and qualified held-out cohort"
        )
    expected_objects = tuple(item.expected.model_dump(mode="json") for item in expected_examples)
    if tuple(item.expected for item in prediction_rows) != expected_objects:
        raise ValueError(f"{label} prediction expected objects differ from the prepared test split")
    if tuple(item.expected.model_dump(mode="json") for item in report.results) != expected_objects:
        raise ValueError(f"{label} report expected objects differ from the prepared test split")

    recomputed = tuple(
        _score_prediction(
            row.expected,
            row.prediction,
            id=row.id,
            latency_ms=row.latency_ms,
        )
        for row in prediction_rows
    )
    recomputed_payload = tuple(item.model_dump(mode="json") for item in recomputed)
    report_payload = tuple(item.model_dump(mode="json") for item in report.results)
    scored_payload = tuple(item.model_dump(mode="json") for item in scored_rows)
    if report_payload != recomputed_payload:
        raise ValueError(f"{label} report scores differ from recomputed predictions")
    if scored_payload != recomputed_payload:
        raise ValueError(f"{label} scored prediction artifact differs from recomputed predictions")
    recomputed_summary = _summarize_scores(recomputed)
    if report.summary.model_dump(mode="json") != recomputed_summary.model_dump(mode="json"):
        raise ValueError(f"{label} summary differs from recomputed predictions")

    provenance = report.provenance
    if provenance is None:
        raise ValueError(f"{label} requires prediction provenance")
    for field in (
        "dataset_manifest_sha256",
        "dataset_split_sha256",
        "generation_config_sha256",
        "model_name_or_path",
        "model_revision",
        "training_manifest_sha256",
        "training_config_sha256",
        "training_dataset_sha256",
        "qualification_sha256",
        "adapter_path",
        "adapter_config_sha256",
        "adapter_weight_sha256",
    ):
        expected_value = getattr(provenance, field)
        if any(getattr(row, field) != expected_value for row in prediction_rows):
            raise ValueError(f"{label} row provenance differs for {field}")


def _evaluation_artifact_snapshots(
    manifest_snapshot: _RegularSnapshot,
    manifest: _EvaluationManifestEvidence,
    candidate_report_snapshot: _RegularSnapshot,
    baseline_report_snapshot: _RegularSnapshot,
) -> tuple[
    tuple[_RegularSnapshot, ...],
    tuple[_GeneratedPredictionEvidence, ...],
    tuple[_EvaluationScoreEvidence, ...],
    tuple[_GeneratedPredictionEvidence, ...],
    tuple[_EvaluationScoreEvidence, ...],
]:
    result = manifest.result
    if result.evaluation_id != manifest.evaluation_id:
        raise ValueError("evaluation result ID differs from the evaluation manifest")
    output_dir = _literal_bound_path(
        result.output_dir,
        anchor=manifest_snapshot.path.parent,
        label="evaluation output directory lineage",
    )
    if output_dir != manifest_snapshot.path.parent:
        raise ValueError("evaluation output directory differs from the manifest directory")
    result_manifest_path = _literal_bound_path(
        result.manifest_path,
        anchor=manifest_snapshot.path.parent,
        label="evaluation manifest path lineage",
    )
    if result_manifest_path != manifest_snapshot.path:
        raise ValueError("evaluation result manifest path differs from release evidence")
    baseline = result.baseline
    if baseline is None:  # pragma: no cover - checked by caller.
        raise ValueError("evaluation baseline evidence is missing")
    roles = (
        ("candidate", result.candidate, candidate_report_snapshot),
        ("baseline", baseline, baseline_report_snapshot),
    )
    expected_paths: dict[str, tuple[str, Path, _RegularSnapshot | None]] = {}
    for role, artifacts, report_snapshot in roles:
        raw_paths = (
            ("predictions", artifacts.report.predictions_path, None),
            ("report", artifacts.json_report_path, report_snapshot),
            ("markdown report", artifacts.markdown_report_path, None),
            ("scored predictions", artifacts.scored_predictions_path, None),
        )
        for artifact_name, raw_path, existing_snapshot in raw_paths:
            path = _literal_bound_path(
                raw_path,
                anchor=manifest_snapshot.path.parent,
                label=f"evaluation {role} {artifact_name} path lineage",
            )
            try:
                key = path.relative_to(output_dir).as_posix()
            except ValueError as exc:
                raise ValueError(
                    f"evaluation {role} {artifact_name} is outside the evaluation output"
                ) from exc
            if key in expected_paths:
                raise ValueError("evaluation artifacts contain duplicate paths")
            expected_paths[key] = (f"evaluation {role} {artifact_name}", path, existing_snapshot)
    if set(manifest.artifact_sha256) != set(expected_paths):
        missing = sorted(set(expected_paths) - set(manifest.artifact_sha256))
        unexpected = sorted(set(manifest.artifact_sha256) - set(expected_paths))
        raise ValueError(
            "evaluation artifact lineage must contain the exact producer artifact set: "
            f"missing={missing}, unexpected={unexpected}"
        )

    snapshots: dict[str, _RegularSnapshot] = {}
    for key, (label, path, existing_snapshot) in expected_paths.items():
        snapshot = existing_snapshot or _read_regular_snapshot(path, label=label)
        if snapshot.path != path or snapshot.sha256 != manifest.artifact_sha256[key]:
            raise ValueError(f"{label} SHA-256 differs from the evaluation manifest")
        snapshots[key] = snapshot
    candidate_predictions = _generated_prediction_rows(
        snapshots["candidate-predictions.jsonl"],
        label="evaluation candidate predictions",
    )
    if (
        result.candidate.report.predictions_sha256
        != snapshots["candidate-predictions.jsonl"].sha256
    ):
        raise ValueError("evaluation candidate prediction SHA-256 differs from its report")
    candidate_scores = _scored_prediction_rows(
        snapshots["candidate/scored-predictions.jsonl"],
        label="evaluation candidate scored predictions",
    )
    baseline_predictions = _generated_prediction_rows(
        snapshots["baseline-predictions.jsonl"],
        label="evaluation baseline predictions",
    )
    if baseline.report.predictions_sha256 != snapshots["baseline-predictions.jsonl"].sha256:
        raise ValueError("evaluation baseline prediction SHA-256 differs from its report")
    baseline_scores = _scored_prediction_rows(
        snapshots["baseline/scored-predictions.jsonl"],
        label="evaluation baseline scored predictions",
    )
    indirect = tuple(
        snapshot
        for key, snapshot in snapshots.items()
        if key not in {"candidate/evaluation-report.json", "baseline/evaluation-report.json"}
    )
    return (
        indirect,
        candidate_predictions,
        candidate_scores,
        baseline_predictions,
        baseline_scores,
    )


def _adapter_config_payload(adapter: _AdapterIdentity) -> dict[str, Any]:
    snapshot = _read_regular_snapshot(
        adapter.path / "adapter_config.json",
        label="adapter config",
    )
    if snapshot.sha256 != adapter.config_sha256:
        raise ValueError("adapter config changed after inventory")
    return _strict_json_object(snapshot)


def _require_qlora_cuda_evidence(
    evidence: _TrainingEvidence,
    *,
    model_config: dict[str, Any],
    training_config: dict[str, Any],
) -> None:
    """Reject declared or rehearsal-only hardware as QLoRA release evidence."""

    if evidence.hardware_preflight is None:
        raise ValueError("QLoRA release requires hardware preflight evidence")
    raw_preflight = dict(evidence.hardware_preflight)
    compatible_claim = raw_preflight.pop("compatible", None)
    if not isinstance(compatible_claim, bool):
        raise ValueError("QLoRA hardware preflight requires an explicit compatibility claim")
    try:
        preflight = HardwarePreflight.model_validate(raw_preflight)
    except ValidationError as exc:
        raise ValueError("QLoRA hardware preflight evidence is invalid") from exc
    if compatible_claim != preflight.compatible:
        raise ValueError("QLoRA hardware preflight compatibility claim is inconsistent")
    if not preflight.compatible or any(finding.level == "error" for finding in preflight.findings):
        raise ValueError("QLoRA release requires a compatible hardware preflight")
    if preflight.method != "qlora":
        raise ValueError("QLoRA hardware preflight method differs from training")

    parameters_b = model_config.get("parameters_b")
    if (
        isinstance(parameters_b, bool)
        or not isinstance(parameters_b, (float, int))
        or float(parameters_b) <= 0
        or preflight.model_parameters_b != float(parameters_b)
    ):
        raise ValueError("QLoRA hardware preflight parameter count differs from training")
    if training_config.get("use_cpu") is not False:
        raise ValueError("QLoRA release training config forces CPU or omits accelerator intent")

    report = preflight.report
    if report.accelerator != "cuda":
        raise ValueError("QLoRA release requires an observed CUDA accelerator")
    if preflight.execution_accelerator != "cuda":
        raise ValueError("QLoRA release must prove it executed on CUDA")
    if report.platform_system != "Linux":
        raise ValueError("QLoRA release requires an observed Linux CUDA host")
    if not isinstance(report.torch_version, str) or not report.torch_version.strip():
        raise ValueError("QLoRA release requires an observed Torch version")
    if not isinstance(report.cuda_version, str) or not report.cuda_version.strip():
        raise ValueError("QLoRA release requires an observed CUDA version")
    try:
        compute_capability = float(report.compute_capability or "")
    except ValueError as exc:
        raise ValueError("QLoRA release requires a verified CUDA compute capability") from exc
    if not math.isfinite(compute_capability):
        raise ValueError("QLoRA release requires a verified CUDA compute capability")
    if compute_capability < 6.0:
        raise ValueError("QLoRA release CUDA compute capability is below 6.0")
    if training_config.get("bf16") is True and not report.supports_bfloat16:
        raise ValueError("QLoRA release bfloat16 execution is unsupported by the observed GPU")
    optimizer_steps = evidence.metrics.get("optimizer_steps")
    if (
        isinstance(optimizer_steps, bool)
        or not isinstance(optimizer_steps, int)
        or optimizer_steps <= 0
    ):
        raise ValueError("QLoRA release requires positive optimizer step evidence")
    if evidence.peak_accelerator_memory_mb is None or evidence.peak_accelerator_memory_mb <= 0:
        raise ValueError("QLoRA release requires positive peak accelerator memory evidence")


def _validate_training_evidence(
    snapshot: _RegularSnapshot,
    adapter: _AdapterIdentity,
) -> tuple[dict[str, Any], _TrainingEvidence]:
    payload, evidence = _evidence_model(snapshot, _TrainingEvidence)
    if not isinstance(evidence, _TrainingEvidence):  # pragma: no cover - type narrowing.
        raise TypeError("training evidence model mismatch")
    if evidence.status != "completed" or evidence.error is not None:
        raise ValueError("training manifest is not a clean completed run")
    if evidence.git_dirty is not False:
        raise ValueError("training manifest does not prove a clean Git checkout")
    if evidence.git_revision is None or _GIT_SHA_PATTERN.fullmatch(evidence.git_revision) is None:
        raise ValueError("training manifest requires an exact 40-hex Git revision")
    if len(set(evidence.git_revision)) == 1:
        raise ValueError("training manifest Git revision is a placeholder")
    if (
        evidence.model_revision is None
        or _GIT_SHA_PATTERN.fullmatch(evidence.model_revision) is None
        or len(set(evidence.model_revision)) == 1
    ):
        raise ValueError("training manifest requires an exact model commit revision")
    if sha256_bytes(evidence.config) != evidence.config_sha256:
        raise ValueError("training manifest config SHA-256 is invalid")

    if set(evidence.dataset_sha256) != _TRAINING_DATASET_HASH_KEYS:
        missing = sorted(_TRAINING_DATASET_HASH_KEYS - set(evidence.dataset_sha256))
        unexpected = sorted(set(evidence.dataset_sha256) - _TRAINING_DATASET_HASH_KEYS)
        raise ValueError(
            "training manifest dataset lineage must use the exact release key set: "
            f"missing={missing}, unexpected={unexpected}"
        )
    _require_digest_mapping(evidence.dataset_sha256, label="training dataset lineage")

    expected_manifest = adapter.path.parent / "manifest.json"
    if snapshot.path != expected_manifest or evidence.run_id != adapter.path.parent.name:
        raise ValueError("training manifest is not the adapter run's sibling manifest")

    model_config = evidence.config.get("model")
    if not isinstance(model_config, dict):
        raise ValueError("training config has no model identity")
    if model_config.get("name_or_path") != evidence.model_name_or_path:
        raise ValueError("training config base model differs from the training manifest")
    if model_config.get("revision") != evidence.model_revision:
        raise ValueError("training config model revision differs from the training manifest")
    lora_config = evidence.config.get("lora")
    if not isinstance(lora_config, dict) or lora_config.get("method") != evidence.method:
        raise ValueError("training manifest method differs from the training config")
    if evidence.method == "qlora":
        training_config = evidence.config.get("training")
        if not isinstance(training_config, dict):
            raise ValueError("QLoRA training config is missing execution settings")
        _require_qlora_cuda_evidence(
            evidence,
            model_config=model_config,
            training_config=training_config,
        )

    prefix = f"{adapter.path.name}/"
    recorded: dict[str, tuple[str, int]] = {}
    for artifact in evidence.artifacts:
        if not artifact.path.startswith(prefix):
            continue
        relative = artifact.path.removeprefix(prefix)
        if not relative or relative in recorded or ".." in Path(relative).parts:
            raise ValueError("training adapter artifact inventory contains an invalid path")
        recorded[relative] = (artifact.sha256, artifact.size_bytes)
    actual = {
        str(item["path"]): (str(item["sha256"]), int(str(item["size_bytes"])))
        for item in adapter.inventory
    }
    if recorded != actual:
        raise ValueError("training adapter artifact inventory does not exactly match adapter bytes")

    adapter_config = _adapter_config_payload(adapter)
    if adapter_config.get("base_model_name_or_path") != evidence.model_name_or_path:
        raise ValueError("adapter config base model differs from training evidence")
    adapter_revision = adapter_config.get("revision")
    if adapter_revision is not None and adapter_revision != evidence.model_revision:
        raise ValueError("adapter config model revision differs from training evidence")
    return payload, evidence


def _validate_prepared_test_split(
    snapshot: _RegularSnapshot,
) -> tuple[_HeldOutExample, ...]:
    try:
        text = snapshot.payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("prepared test split must be UTF-8 JSONL") from exc
    rows: list[dict[str, Any]] = []
    examples: list[_HeldOutExample] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            raise ValueError(f"prepared test split line {line_number} must not be blank")
        try:
            row = loads_strict(line)
        except (json.JSONDecodeError, StrictJSONError) as exc:
            raise ValueError(f"invalid prepared test split JSON on line {line_number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"prepared test split line {line_number} must be a JSON object")
        expected_fields = {
            "id",
            "prompt",
            "completion",
            "expected",
            "provenance",
            "pii_placeholders",
        }
        if set(row) != expected_fields:
            raise ValueError(
                f"prepared test split line {line_number} is not a complete prepared record"
            )
        prompt = row["prompt"]
        completion = row["completion"]
        if not isinstance(prompt, list) or not isinstance(completion, list):
            raise ValueError(f"prepared test split line {line_number} has invalid conversations")
        try:
            source_example = TicketExample.model_validate(
                {
                    "id": row["id"],
                    "messages": [*prompt, *completion],
                    "expected": row["expected"],
                    "provenance": row["provenance"],
                    "pii_placeholders": row["pii_placeholders"],
                },
                strict=True,
            )
        except ValidationError as exc:
            raise ValueError(
                f"prepared test split line {line_number} is producer-invalid: {exc}"
            ) from exc
        if len(prompt) != 2 or len(completion) != 1:
            raise ValueError(
                f"prepared test split line {line_number} has invalid prompt/completion shape"
            )
        rows.append(row)
        examples.append(_HeldOutExample(id=source_example.id, expected=source_example.expected))
    if not examples:
        raise ValueError("prepared test split must contain at least one record")
    identifiers = [item.id for item in examples]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("prepared test split contains duplicate IDs")
    canonical_payload = b"".join(
        (
            json.dumps(
                row,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )
        for row in rows
    )
    if snapshot.payload != canonical_payload:
        raise ValueError("prepared test split is not canonical producer JSONL")
    return tuple(examples)


def _validate_dataset_evidence(
    manifest_snapshot: _RegularSnapshot,
    test_snapshot: _RegularSnapshot,
    training: _TrainingEvidence,
) -> tuple[_DatasetManifestEvidence, tuple[_HeldOutExample, ...]]:
    _payload, manifest = _evidence_model(
        manifest_snapshot,
        _DatasetManifestEvidence,
    )
    if not isinstance(manifest, _DatasetManifestEvidence):  # pragma: no cover
        raise TypeError("dataset manifest evidence model mismatch")
    dataset = training.dataset_sha256
    if manifest_snapshot.sha256 != dataset["manifest"]:
        raise ValueError("dataset manifest bytes differ from training lineage")
    if test_snapshot.sha256 != dataset["test"]:
        raise ValueError("prepared test split bytes differ from training lineage")
    if manifest.source_sha256 != dataset["source"]:
        raise ValueError("dataset manifest source lineage differs from training")
    if set(manifest.split_fractions) != set(_SPLIT_NAMES):
        raise ValueError("dataset manifest split fractions are incomplete")
    if set(manifest.splits) != set(_SPLIT_NAMES):
        raise ValueError("dataset manifest split artifacts are incomplete")
    if set(manifest.split_ids) != set(_SPLIT_NAMES):
        raise ValueError("dataset manifest split ID inventories are incomplete")

    all_ids: list[str] = []
    total = 0
    for split_name in _SPLIT_NAMES:
        artifact = manifest.splits[split_name]
        split_ids = manifest.split_ids[split_name]
        if artifact.file != f"{split_name}.jsonl":
            raise ValueError(
                f"dataset manifest {split_name} artifact does not use its canonical filename"
            )
        if artifact.sha256 != dataset[split_name]:
            raise ValueError(f"dataset manifest {split_name} lineage differs from training")
        if artifact.count != len(split_ids):
            raise ValueError(f"dataset manifest {split_name} count differs from its ID inventory")
        if any(not identifier for identifier in split_ids) or len(split_ids) != len(set(split_ids)):
            raise ValueError(f"dataset manifest {split_name} IDs must be non-empty and unique")
        all_ids.extend(split_ids)
        total += artifact.count
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("dataset manifest IDs overlap across splits")
    if total != manifest.total_examples:
        raise ValueError("dataset manifest total count differs from its split artifacts")

    expected_test_path = manifest_snapshot.path.parent / manifest.splits["test"].file
    if test_snapshot.path != expected_test_path:
        raise ValueError("prepared test split path differs from the dataset manifest")
    if manifest.splits["test"].sha256 != test_snapshot.sha256:
        raise ValueError("prepared test split SHA-256 differs from the dataset manifest")
    examples = _validate_prepared_test_split(test_snapshot)
    test_ids = tuple(item.id for item in examples)
    if test_ids != tuple(manifest.split_ids["test"]):
        raise ValueError("prepared test split ordered IDs differ from the dataset manifest")
    if len(examples) != manifest.splits["test"].count:
        raise ValueError("prepared test split count differs from the dataset manifest")
    test_labels = manifest.splits["test"].labels
    expected_labels = {
        "category": dict(sorted(Counter(str(item.expected.category) for item in examples).items())),
        "priority": dict(sorted(Counter(str(item.expected.priority) for item in examples).items())),
        "sentiment": dict(
            sorted(Counter(str(item.expected.sentiment) for item in examples).items())
        ),
    }
    if test_labels.model_dump(mode="json") != expected_labels:
        raise ValueError("prepared test split labels differ from the dataset manifest")
    return manifest, examples


def _validate_qualification_evidence(
    snapshot: _RegularSnapshot,
    training: _TrainingEvidence,
) -> tuple[dict[str, Any], _QualificationEvidence]:
    payload, evidence = _evidence_model(snapshot, _QualificationEvidence)
    if not isinstance(evidence, _QualificationEvidence):  # pragma: no cover - type narrowing.
        raise TypeError("qualification evidence model mismatch")
    if not evidence.qualified:
        raise ValueError("dataset qualification report is not qualified")
    if evidence.schema_version != "1.2":
        raise ValueError("dataset qualification report requires v1.2 reviewer-packet evidence")
    if evidence.dataset_tier != "qualification_candidate":
        raise ValueError("dataset evidence is not qualification_candidate tier")
    _require_unique_exact_names(
        [item.policy for item in evidence.decisions],
        _QUALIFICATION_POLICIES,
        label="dataset qualification decisions",
    )
    if not all(item.passed for item in evidence.decisions):
        raise ValueError("dataset qualification decisions are not all passing")
    if (
        not evidence.held_out_ids
        or len(evidence.held_out_ids) != len(set(evidence.held_out_ids))
        or evidence.held_out_count != len(evidence.held_out_ids)
    ):
        raise ValueError(
            "dataset qualification held-out count and unique ID inventory do not match"
        )
    dataset = training.dataset_sha256
    if evidence.source_sha256 != dataset["source"]:
        raise ValueError("qualification source lineage differs from training")
    if evidence.prepared_manifest_sha256 != dataset["manifest"]:
        raise ValueError("qualification prepared-manifest lineage differs from training")
    if evidence.holdout_freeze_sha256 is None:
        raise ValueError("qualification evidence has no frozen holdout identity")
    if (
        len(evidence.reviewer_packet_paths) != 2
        or len(evidence.reviewer_packet_sha256) != 2
        or len(set(evidence.reviewer_packet_paths)) != 2
        or len(set(evidence.reviewer_packet_sha256)) != 2
        or len(evidence.reviewer_ids) != 2
        or len(set(evidence.reviewer_ids)) != 2
    ):
        raise ValueError("qualification evidence requires two distinct reviewer packet identities")
    if evidence.review_manifest_sha256 != dataset["qualification_review_manifest"]:
        raise ValueError("qualification review lineage differs from training")
    if sha256_bytes(payload) != dataset["qualification_report"]:
        raise ValueError("qualification semantic-report lineage differs from training")
    return payload, evidence


def _require_adapter_path_lineage(
    raw_path: str,
    *,
    snapshot: _RegularSnapshot,
    adapter: _AdapterIdentity,
    label: str,
) -> None:
    located = _literal_bound_path(
        raw_path,
        anchor=snapshot.path.parent,
        label=label,
    )
    if located != adapter.path:
        raise ValueError(f"{label} differs from the release adapter")


def _qualification_lineage(training: _TrainingEvidence) -> dict[str, str]:
    return {
        name: digest
        for name, digest in training.dataset_sha256.items()
        if name.startswith("qualification_")
    }


def _validate_evaluation_evidence(
    manifest_snapshot: _RegularSnapshot,
    report_snapshot: _RegularSnapshot,
    training_snapshot: _RegularSnapshot,
    training: _TrainingEvidence,
    adapter: _AdapterIdentity,
    *,
    expected_examples: tuple[_HeldOutExample, ...],
) -> tuple[_EvaluationReportEvidence, tuple[_RegularSnapshot, ...]]:
    report_payload, report_view = _evidence_model(
        report_snapshot,
        _EvaluationReportEvidence,
    )
    if not isinstance(report_view, _EvaluationReportEvidence):  # pragma: no cover
        raise TypeError("evaluation report model mismatch")
    manifest_payload, manifest_view = _evidence_model(
        manifest_snapshot,
        _EvaluationManifestEvidence,
    )
    if not isinstance(manifest_view, _EvaluationManifestEvidence):  # pragma: no cover
        raise TypeError("evaluation manifest model mismatch")
    if manifest_view.status != "completed":
        raise ValueError("evaluation manifest is not completed")
    candidate = manifest_view.result.candidate
    baseline = manifest_view.result.baseline
    comparison = manifest_view.result.comparison
    if baseline is None or comparison is None:
        raise ValueError("evaluation manifest requires completed baseline and comparison evidence")
    if manifest_view.config_sha256 != training.config_sha256:
        raise ValueError("evaluation config lineage differs from training")
    _require_digest_mapping(
        manifest_view.artifact_sha256,
        label="evaluation artifact lineage",
    )
    candidate_report_digests = [
        digest
        for name, digest in manifest_view.artifact_sha256.items()
        if name.replace("\\", "/").endswith("candidate/evaluation-report.json")
    ]
    if candidate_report_digests != [report_snapshot.sha256]:
        raise ValueError("evaluation manifest candidate report SHA-256 mismatch")
    baseline_report_digests = [
        digest
        for name, digest in manifest_view.artifact_sha256.items()
        if name.replace("\\", "/").endswith("baseline/evaluation-report.json")
    ]
    if len(baseline_report_digests) != 1:
        raise ValueError("evaluation manifest requires one baseline report SHA-256")
    candidate_report_path = _literal_bound_path(
        candidate.json_report_path,
        anchor=manifest_snapshot.path.parent,
        label="evaluation candidate report path lineage",
    )
    if candidate_report_path != report_snapshot.path:
        raise ValueError("evaluation candidate report path differs from release evidence")
    baseline_report_path = _literal_bound_path(
        baseline.json_report_path,
        anchor=manifest_snapshot.path.parent,
        label="evaluation baseline report path lineage",
    )
    baseline_snapshot = _read_regular_snapshot(
        baseline_report_path,
        label="evaluation baseline report",
    )
    if baseline_snapshot.sha256 != baseline_report_digests[0]:
        raise ValueError("evaluation manifest baseline report SHA-256 mismatch")
    baseline_payload, baseline_view = _evidence_model(
        baseline_snapshot,
        _EvaluationBaselineReportEvidence,
    )
    if not isinstance(baseline_view, _EvaluationBaselineReportEvidence):  # pragma: no cover
        raise TypeError("evaluation baseline report model mismatch")
    embedded = manifest_payload.get("result")
    if not isinstance(embedded, dict):  # pragma: no cover - caught by the model.
        raise ValueError("evaluation manifest result is invalid")
    embedded_candidate = embedded.get("candidate")
    if not isinstance(embedded_candidate, dict):  # pragma: no cover - caught by the model.
        raise ValueError("evaluation manifest candidate is invalid")
    if embedded_candidate.get("report") != report_payload:
        raise ValueError("evaluation manifest embedded report differs from candidate report")
    embedded_baseline = embedded.get("baseline")
    if not isinstance(embedded_baseline, dict):  # pragma: no cover - caught by the model.
        raise ValueError("evaluation manifest baseline is invalid")
    if embedded_baseline.get("report") != baseline_payload:
        raise ValueError("evaluation manifest embedded report differs from baseline report")

    (
        indirect_artifact_snapshots,
        candidate_prediction_rows,
        candidate_scored_rows,
        baseline_prediction_rows,
        baseline_scored_rows,
    ) = _evaluation_artifact_snapshots(
        manifest_snapshot,
        manifest_view,
        report_snapshot,
        baseline_snapshot,
    )
    _validate_report_recomputation(
        report_view,
        candidate_prediction_rows,
        candidate_scored_rows,
        expected_examples,
        label="candidate evaluation",
    )
    _validate_report_recomputation(
        baseline_view,
        baseline_prediction_rows,
        baseline_scored_rows,
        expected_examples,
        label="baseline evaluation",
    )

    minimums = _evaluation_minimums(training)
    candidate_ids = _validate_absolute_evaluation_report(
        report_view,
        minimums=minimums,
        label="candidate evaluation",
    )
    baseline_ids = _validate_absolute_evaluation_report(
        baseline_view,
        minimums=minimums,
        label="baseline evaluation",
    )
    expected_held_out_ids = tuple(item.id for item in expected_examples)
    if candidate_ids != expected_held_out_ids or baseline_ids != expected_held_out_ids:
        raise ValueError(
            "candidate and baseline evaluation ordered IDs differ from the qualified "
            "held-out cohort"
        )
    if candidate.passed != report_view.passed or baseline.passed != baseline_view.passed:
        raise ValueError("evaluation artifact pass flags differ from their reports")
    if not report_view.passed or not candidate.passed:
        raise ValueError("evaluation candidate has not passed all absolute thresholds")

    non_regression = _evaluation_thresholds_by_metric(
        comparison.non_regression,
        expected=_EVALUATION_NON_REGRESSION_METRICS,
        label="evaluation non-regression thresholds",
    )
    _require_unique_exact_names(
        list(comparison.metric_deltas),
        _EVALUATION_NON_REGRESSION_METRICS,
        label="evaluation metric deltas",
    )
    for metric in _EVALUATION_NON_REGRESSION_METRICS:
        expected_delta = float(getattr(report_view.summary, metric)) - float(
            getattr(baseline_view.summary, metric)
        )
        threshold = non_regression[metric]
        if not _same_number(threshold.minimum, 0):
            raise ValueError(f"evaluation non-regression threshold {metric} must have minimum 0")
        if not _same_number(threshold.value, expected_delta) or not _same_number(
            comparison.metric_deltas[metric],
            expected_delta,
        ):
            raise ValueError(
                f"evaluation non-regression metric {metric} differs from report summaries"
            )
        if threshold.passed != (threshold.value >= threshold.minimum):
            raise ValueError(
                f"evaluation non-regression threshold {metric} has an inconsistent pass result"
            )
    expected_comparison_passed = report_view.passed and all(
        item.passed for item in non_regression.values()
    )
    if (
        comparison.candidate_passed != report_view.passed
        or comparison.baseline_passed != baseline_view.passed
        or comparison.passed != expected_comparison_passed
        or manifest_view.candidate_passed != report_view.passed
        or manifest_view.comparison_passed != comparison.passed
    ):
        raise ValueError("evaluation candidate/baseline/comparison pass flags are inconsistent")
    if not comparison.passed:
        raise ValueError("evaluation comparison has failed non-regression thresholds")

    provenance = report_view.provenance
    baseline_provenance = baseline_view.provenance
    if provenance is None or baseline_provenance is None:
        raise ValueError("candidate and baseline evaluation reports require provenance")
    expected_qualification = _qualification_lineage(training)
    if provenance.training_manifest_sha256 != training_snapshot.sha256:
        raise ValueError("evaluation training manifest lineage differs from release evidence")
    if provenance.training_config_sha256 != training.config_sha256:
        raise ValueError("evaluation training config lineage differs from training")
    if provenance.training_dataset_sha256 != training.dataset_sha256:
        raise ValueError("evaluation training dataset lineage differs from training")
    if provenance.qualification_sha256 != expected_qualification:
        raise ValueError("evaluation qualification lineage differs from training")
    if provenance.dataset_manifest_sha256 != training.dataset_sha256["manifest"]:
        raise ValueError("evaluation dataset lineage differs from training")
    if provenance.dataset_split_sha256 != training.dataset_sha256["test"]:
        raise ValueError("evaluation held-out split lineage differs from training")
    if provenance.model_name_or_path != training.model_name_or_path:
        raise ValueError("evaluation base model lineage differs from training")
    if provenance.model_revision != training.model_revision:
        raise ValueError("evaluation model revision lineage differs from training")
    if provenance.adapter_path is None:
        raise ValueError("candidate evaluation provenance has no adapter path")
    _require_adapter_path_lineage(
        provenance.adapter_path,
        snapshot=report_snapshot,
        adapter=adapter,
        label="evaluation adapter path lineage",
    )
    if provenance.adapter_config_sha256 != adapter.config_sha256:
        raise ValueError("evaluation adapter config lineage differs from adapter bytes")
    if provenance.adapter_weight_sha256 != adapter.weight_sha256:
        raise ValueError("evaluation adapter weight lineage differs from adapter bytes")
    for field in (
        "dataset_manifest_sha256",
        "dataset_split_sha256",
        "generation_config_sha256",
        "model_name_or_path",
        "model_revision",
    ):
        if getattr(baseline_provenance, field) != getattr(provenance, field):
            raise ValueError(f"baseline evaluation provenance differs for {field}")
    if (
        baseline_provenance.training_dataset_sha256
        or baseline_provenance.qualification_sha256
        or baseline_provenance.adapter_weight_sha256
    ):
        raise ValueError("baseline evaluation provenance must be base-model-only")
    return report_view, (baseline_snapshot, *indirect_artifact_snapshots)


def _require_parity_common_lineage(
    evidence: _ParityEvidence | _ParityPredictionEvidence,
    *,
    label: str,
    training_snapshot: _RegularSnapshot,
    training: _TrainingEvidence,
) -> None:
    if evidence.training_manifest_sha256 != training_snapshot.sha256:
        raise ValueError(f"{label} training manifest lineage differs from release evidence")
    if evidence.training_config_sha256 != training.config_sha256:
        raise ValueError(f"{label} training config lineage differs from training")
    if evidence.training_dataset_sha256 != training.dataset_sha256:
        raise ValueError(f"{label} training dataset lineage differs from training")
    if evidence.qualification_sha256 != _qualification_lineage(training):
        raise ValueError(f"{label} qualification lineage differs from training")
    if evidence.dataset_manifest_sha256 != training.dataset_sha256["manifest"]:
        raise ValueError(f"{label} dataset lineage differs from training")
    if evidence.dataset_split_sha256 != training.dataset_sha256["test"]:
        raise ValueError(f"{label} held-out split lineage differs from training")


def _parity_prediction_snapshot(
    prediction: _ParityPredictionEvidence,
    *,
    parity_snapshot: _RegularSnapshot,
    label: str,
) -> tuple[_RegularSnapshot, tuple[_GeneratedPredictionEvidence, ...]]:
    path = _literal_bound_path(
        prediction.path,
        anchor=parity_snapshot.path.parent,
        label=f"{label} path lineage",
    )
    snapshot = _read_regular_snapshot(path, label=label)
    if snapshot.sha256 != prediction.sha256:
        raise ValueError(f"{label} SHA-256 differs from the parity report")
    return snapshot, _generated_prediction_rows(snapshot, label=label)


def _validate_parity_prediction_rows(
    rows: tuple[_GeneratedPredictionEvidence, ...],
    artifact: _ParityPredictionEvidence,
    expected_examples: tuple[_HeldOutExample, ...],
    *,
    label: str,
) -> None:
    expected_ids = tuple(item.id for item in expected_examples)
    if tuple(item.id for item in rows) != expected_ids:
        raise ValueError(f"{label} IDs differ from the prepared test split")
    expected_objects = tuple(item.expected.model_dump(mode="json") for item in expected_examples)
    if tuple(item.expected for item in rows) != expected_objects:
        raise ValueError(f"{label} expected objects differ from the prepared test split")
    for field in (
        "model_name_or_path",
        "model_revision",
        "dataset_manifest_sha256",
        "dataset_split_sha256",
        "generation_config_sha256",
        "training_manifest_sha256",
        "training_config_sha256",
        "training_dataset_sha256",
        "qualification_sha256",
        "adapter_path",
        "adapter_config_sha256",
        "adapter_weight_sha256",
        "merged_model_path",
        "merge_provenance_sha256",
        "merged_artifact_sha256",
        "merged_adapter_config_sha256",
        "merged_adapter_weight_files",
        "merged_adapter_weight_sha256",
    ):
        expected_value = getattr(artifact, field)
        if any(getattr(row, field) != expected_value for row in rows):
            raise ValueError(f"{label} row provenance differs for {field}")


def _validate_parity_evidence(
    snapshot: _RegularSnapshot,
    training_snapshot: _RegularSnapshot,
    training: _TrainingEvidence,
    adapter: _AdapterIdentity,
    *,
    expected_generation_config_sha256: str,
    expected_examples: tuple[_HeldOutExample, ...],
) -> tuple[_RegularSnapshot, _RegularSnapshot]:
    _payload, evidence = _evidence_model(snapshot, _ParityEvidence)
    if not isinstance(evidence, _ParityEvidence):  # pragma: no cover - type narrowing.
        raise TypeError("parity evidence model mismatch")
    model_config = training.config.get("model")
    configured_dtype = model_config.get("torch_dtype") if isinstance(model_config, dict) else None
    if (
        configured_dtype not in {"bfloat16", "float16", "float32"}
        or evidence.merge_dtype != configured_dtype
    ):
        raise ValueError("parity merge precision differs from the explicit training precision")
    expected_held_out_ids = tuple(item.id for item in expected_examples)
    if tuple(evidence.ordered_ids) != expected_held_out_ids:
        raise ValueError("parity ordered IDs differ from the qualified held-out cohort")
    if evidence.metrics.examples != len(evidence.ordered_ids):
        raise ValueError("parity metric example count differs from its ordered IDs")
    _require_unique_exact_names(
        [item.metric for item in evidence.thresholds],
        _PARITY_GATE_METRICS,
        label="parity thresholds",
    )
    parity_thresholds = {item.metric: item for item in evidence.thresholds}
    for metric in _PARITY_GATE_METRICS:
        threshold = parity_thresholds[metric]
        metric_value = float(getattr(evidence.metrics, metric))
        if not _same_number(threshold.value, metric_value):
            raise ValueError(f"parity threshold {metric} differs from its metric value")
        if not _same_number(threshold.required, 1):
            raise ValueError(f"parity threshold {metric} must require exact 1.0 parity")
        if threshold.passed != (threshold.value >= threshold.required):
            raise ValueError(f"parity threshold {metric} has an inconsistent pass result")
        if not _same_number(threshold.value, 1):
            raise ValueError(f"parity threshold {metric} did not achieve exact 1.0 parity")
    expected_parity_passed = all(item.passed for item in parity_thresholds.values())
    if evidence.passed != expected_parity_passed or not evidence.passed:
        raise ValueError("parity report has not passed every exact threshold")
    if evidence.generation_config_sha256 != expected_generation_config_sha256:
        raise ValueError("parity generation config lineage differs from evaluation")
    _require_parity_common_lineage(
        evidence,
        label="parity",
        training_snapshot=training_snapshot,
        training=training,
    )
    for role, prediction in (
        ("parity adapter predictions", evidence.adapter_predictions),
        ("parity merged predictions", evidence.merged_predictions),
    ):
        _require_parity_common_lineage(
            prediction,
            label=role,
            training_snapshot=training_snapshot,
            training=training,
        )
        if prediction.model_name_or_path != training.model_name_or_path:
            raise ValueError(f"{role} base model lineage differs from training")
        if prediction.model_revision != training.model_revision:
            raise ValueError(f"{role} model revision lineage differs from training")
        if prediction.generation_config_sha256 != evidence.generation_config_sha256:
            raise ValueError(f"{role} generation config lineage differs from parity report")
    adapter_prediction = evidence.adapter_predictions
    merged_prediction = evidence.merged_predictions
    if adapter_prediction.role != "adapter" or merged_prediction.role != "merged":
        raise ValueError("parity prediction roles are invalid")
    if adapter_prediction.adapter_path is None:
        raise ValueError("parity adapter prediction has no adapter path")
    _require_adapter_path_lineage(
        adapter_prediction.adapter_path,
        snapshot=snapshot,
        adapter=adapter,
        label="parity adapter path lineage",
    )
    if adapter_prediction.adapter_config_sha256 != adapter.config_sha256:
        raise ValueError("parity adapter config lineage differs from adapter bytes")
    if adapter_prediction.adapter_weight_sha256 != adapter.weight_sha256:
        raise ValueError("parity adapter weight lineage differs from adapter bytes")
    if (
        adapter_prediction.merged_model_path is not None
        or adapter_prediction.merge_provenance_sha256 is not None
        or adapter_prediction.merged_artifact_sha256
        or adapter_prediction.merged_adapter_config_sha256 is not None
        or adapter_prediction.merged_adapter_weight_files
        or adapter_prediction.merged_adapter_weight_sha256
    ):
        raise ValueError("parity adapter predictions contain merged-model provenance")
    if (
        merged_prediction.adapter_path is not None
        or merged_prediction.adapter_config_sha256 is not None
        or merged_prediction.adapter_weight_sha256
    ):
        raise ValueError("parity merged predictions contain applied-adapter provenance")
    if (
        not merged_prediction.merged_model_path
        or merged_prediction.merge_provenance_sha256 is None
        or not merged_prediction.merged_artifact_sha256
        or merged_prediction.merged_adapter_config_sha256 is None
        or not merged_prediction.merged_adapter_weight_files
        or not merged_prediction.merged_adapter_weight_sha256
    ):
        raise ValueError("parity merged predictions lack complete safe-merge provenance")
    _require_digest_mapping(
        merged_prediction.merged_artifact_sha256,
        label="parity merged artifact lineage",
    )
    _require_digest_mapping(
        {"merge_provenance": merged_prediction.merge_provenance_sha256},
        label="parity merge provenance lineage",
    )
    if merged_prediction.merged_adapter_config_sha256 != adapter.config_sha256:
        raise ValueError("parity merged adapter config lineage differs from adapter bytes")
    weight_files = merged_prediction.merged_adapter_weight_files
    weight_hashes = merged_prediction.merged_adapter_weight_sha256
    if (
        len(weight_files) != len(weight_hashes)
        or len(weight_files) != len(set(weight_files))
        or any(
            Path(name).name != name or not name.endswith(".safetensors") for name in weight_files
        )
    ):
        raise ValueError("parity merged adapter weight identity is invalid")
    merged_adapter_weights = dict(zip(weight_files, weight_hashes, strict=True))
    _require_digest_mapping(
        merged_adapter_weights,
        label="parity merged adapter weight lineage",
    )
    if merged_adapter_weights != adapter.weight_sha256:
        raise ValueError("parity merged adapter weight lineage differs from adapter bytes")
    adapter_snapshot, adapter_rows = _parity_prediction_snapshot(
        adapter_prediction,
        parity_snapshot=snapshot,
        label="parity adapter predictions",
    )
    merged_snapshot, merged_rows = _parity_prediction_snapshot(
        merged_prediction,
        parity_snapshot=snapshot,
        label="parity merged predictions",
    )
    adapter_ids = tuple(item.id for item in adapter_rows)
    merged_ids = tuple(item.id for item in merged_rows)
    if adapter_ids != expected_held_out_ids or merged_ids != expected_held_out_ids:
        raise ValueError(
            "adapter and merged parity-side ordered IDs differ from the qualified held-out cohort"
        )
    if adapter_ids != tuple(evidence.ordered_ids) or merged_ids != tuple(evidence.ordered_ids):
        raise ValueError("parity-side ordered IDs differ from the parity report")
    _validate_parity_prediction_rows(
        adapter_rows,
        adapter_prediction,
        expected_examples,
        label="parity adapter predictions",
    )
    _validate_parity_prediction_rows(
        merged_rows,
        merged_prediction,
        expected_examples,
        label="parity merged predictions",
    )
    contract = _ParityPredictionContract(
        dataset_manifest_sha256=evidence.dataset_manifest_sha256,
        dataset_split_sha256=evidence.dataset_split_sha256,
        generation_config_sha256=evidence.generation_config_sha256,
        training_manifest_sha256=evidence.training_manifest_sha256,
        training_config_sha256=evidence.training_config_sha256,
        training_dataset_sha256=evidence.training_dataset_sha256,
        qualification_sha256=evidence.qualification_sha256,
    )
    try:
        recomputed = _build_parity_report(
            adapter_rows,
            merged_rows,
            adapter_artifact=adapter_prediction,
            merged_artifact=merged_prediction,
            contract=contract,
            merge_dtype=evidence.merge_dtype,
        )
    except ValueError as exc:
        raise ValueError(f"parity sidecars cannot reproduce the parity report: {exc}") from exc
    if recomputed.model_dump(mode="json") != evidence.model_dump(mode="json"):
        raise ValueError("parity report differs from recomputed immutable sidecars")
    return adapter_snapshot, merged_snapshot


def _validate_release(path: Path) -> _ValidatedRelease:
    manifest_snapshot, release = _load_release_manifest(path)
    _validate_release_sentinels(release)

    compose_snapshot = _bound_release_file(
        manifest_snapshot.path,
        release.compose_file,
        release.compose_sha256,
        label="compose file",
    )
    profile_snapshots, production_profile_sha256 = _validate_production_profile(compose_snapshot)
    env_snapshot = _bound_release_file(
        manifest_snapshot.path,
        release.env_file,
        release.env_sha256,
        label="env file",
    )
    adapter = _bound_release_adapter(
        manifest_snapshot.path,
        release.adapter_path,
        release.adapter_sha256,
    )
    dataset_manifest_snapshot = _bound_release_file(
        manifest_snapshot.path,
        release.dataset_manifest_file,
        release.dataset_manifest_sha256,
        label="dataset manifest",
    )
    test_split_snapshot = _bound_release_file(
        manifest_snapshot.path,
        release.test_split_file,
        release.test_split_sha256,
        label="prepared test split",
    )
    training_snapshot = _bound_release_file(
        manifest_snapshot.path,
        release.training_manifest_file,
        release.training_manifest_sha256,
        label="training manifest",
    )
    qualification_snapshot = _bound_release_file(
        manifest_snapshot.path,
        release.qualification_report_file,
        release.qualification_report_sha256,
        label="qualification report",
    )
    evaluation_manifest_snapshot = _bound_release_file(
        manifest_snapshot.path,
        release.evaluation_manifest_file,
        release.evaluation_manifest_sha256,
        label="evaluation manifest",
    )
    evaluation_report_snapshot = _bound_release_file(
        manifest_snapshot.path,
        release.evaluation_report_file,
        release.evaluation_report_sha256,
        label="evaluation report",
    )
    parity_snapshot = _bound_release_file(
        manifest_snapshot.path,
        release.parity_report_file,
        release.parity_report_sha256,
        label="parity report",
    )

    _training_payload, training = _validate_training_evidence(
        training_snapshot,
        adapter,
    )
    _dataset_manifest, test_examples = _validate_dataset_evidence(
        dataset_manifest_snapshot,
        test_split_snapshot,
        training,
    )
    _qualification_payload, qualification = _validate_qualification_evidence(
        qualification_snapshot,
        training,
    )
    held_out_ids = tuple(qualification.held_out_ids)
    test_ids = tuple(item.id for item in test_examples)
    if held_out_ids != test_ids:
        raise ValueError("qualification held-out IDs differ from the prepared test split")
    evaluation, evaluation_indirect_snapshots = _validate_evaluation_evidence(
        evaluation_manifest_snapshot,
        evaluation_report_snapshot,
        training_snapshot,
        training,
        adapter,
        expected_examples=test_examples,
    )
    evaluation_provenance = evaluation.provenance
    if evaluation_provenance is None:  # pragma: no cover - semantic validator requires it.
        raise ValueError("candidate evaluation report requires provenance")
    parity_indirect_snapshots = _validate_parity_evidence(
        parity_snapshot,
        training_snapshot,
        training,
        adapter,
        expected_generation_config_sha256=evaluation_provenance.generation_config_sha256,
        expected_examples=test_examples,
    )
    env_values = _require_env_identity(env_snapshot, release, adapter, training)
    _require_compose_env_contract(compose_snapshot, env_values)

    return _ValidatedRelease(
        manifest_snapshot=manifest_snapshot,
        release=release,
        compose_snapshot=compose_snapshot,
        env_snapshot=env_snapshot,
        adapter=adapter,
        evidence_snapshots=(
            training_snapshot,
            dataset_manifest_snapshot,
            test_split_snapshot,
            qualification_snapshot,
            evaluation_manifest_snapshot,
            evaluation_report_snapshot,
            parity_snapshot,
            *evaluation_indirect_snapshots,
            *parity_indirect_snapshots,
        ),
        profile_snapshots=profile_snapshots,
        production_profile_sha256=production_profile_sha256,
        training=training,
    )


def _recheck_snapshot(snapshot: _RegularSnapshot, *, purpose: str) -> None:
    current = _read_regular_snapshot(snapshot.path, label=snapshot.label)
    if current.payload != snapshot.payload:
        raise ValueError(f"{snapshot.label} changed before {purpose} completion")


def _recheck_validated_release(binding: _ValidatedRelease, *, purpose: str) -> None:
    for snapshot in (
        binding.manifest_snapshot,
        binding.compose_snapshot,
        binding.env_snapshot,
        *binding.evidence_snapshots,
        *binding.profile_snapshots,
    ):
        _recheck_snapshot(snapshot, purpose=purpose)
    adapter_path, inventory = _adapter_inventory(binding.adapter.path)
    if adapter_path != binding.adapter.path or inventory != binding.adapter.inventory:
        raise ValueError(f"adapter directory changed before {purpose} completion")


def _compose_command_prefix(_binding: _ValidatedRelease) -> tuple[str, ...]:
    """Select a fixed local daemon and ignore ambient Docker/Compose configuration."""

    return (
        "/usr/bin/env",
        "-i",
        f"PATH={_APPROVED_EXEC_PATH}",
        f"HOME={_APPROVED_EXEC_HOME}",
        APPROVED_DOCKER_EXECUTABLE,
        "--host",
        APPROVED_DOCKER_HOST,
        "compose",
    )


def _require_rollback_compatibility(
    current: _ValidatedRelease,
    previous: _ValidatedRelease,
) -> None:
    """Require both manifests to describe revisions of one approved deployment slot."""

    comparisons = (
        (
            "project slot",
            current.release.project_name,
            previous.release.project_name,
        ),
        ("served model", current.release.model, previous.release.model),
        (
            "production profile",
            current.production_profile_sha256,
            previous.production_profile_sha256,
        ),
        (
            "base model",
            current.training.model_name_or_path,
            previous.training.model_name_or_path,
        ),
        (
            "model revision",
            current.training.model_revision,
            previous.training.model_revision,
        ),
    )
    for label, current_value, previous_value in comparisons:
        if current_value != previous_value:
            raise ValueError(f"rollback {label} differs between current and previous release")


def _evidence_snapshot(binding: _ValidatedRelease, label: str) -> _RegularSnapshot:
    matches = [snapshot for snapshot in binding.evidence_snapshots if snapshot.label == label]
    if len(matches) != 1:  # pragma: no cover - internal binding construction invariant.
        raise RuntimeError(f"release binding has {len(matches)} snapshots labelled {label!r}")
    return matches[0]


def validate_release_manifest(path: Path) -> ReleaseValidationReport:
    """Semantically validate one schema-2 release from stable, non-symlink snapshots."""

    binding = _validate_release(path)
    _recheck_validated_release(binding, purpose="release validation")
    release = binding.release
    git_revision = binding.training.git_revision
    if git_revision is None:  # pragma: no cover - guaranteed by semantic validation.
        raise ValueError("training Git revision is missing")
    return ReleaseValidationReport(
        release_id=release.release_id,
        project_name=release.project_name,
        model=release.model,
        release_manifest_sha256=binding.manifest_snapshot.sha256,
        compose_sha256=binding.compose_snapshot.sha256,
        env_sha256=binding.env_snapshot.sha256,
        adapter_sha256=binding.adapter.inventory_sha256,
        dataset_manifest_sha256=_evidence_snapshot(binding, "dataset manifest").sha256,
        test_split_sha256=_evidence_snapshot(binding, "prepared test split").sha256,
        training_manifest_sha256=_evidence_snapshot(binding, "training manifest").sha256,
        qualification_report_sha256=_evidence_snapshot(binding, "qualification report").sha256,
        evaluation_manifest_sha256=_evidence_snapshot(binding, "evaluation manifest").sha256,
        evaluation_report_sha256=_evidence_snapshot(binding, "evaluation report").sha256,
        parity_report_sha256=_evidence_snapshot(binding, "parity report").sha256,
        git_revision=git_revision,
        base_model=binding.training.model_name_or_path,
        model_revision=binding.training.model_revision,
        production_profile_sha256=binding.production_profile_sha256,
    )


def _default_compose_runner(argv: tuple[str, ...]) -> int:
    # The executable and every operation token are fixed by this module; no shell is involved.
    return os.spawnv(os.P_WAIT, argv[0], argv)  # noqa: S606  # nosec B606


def start_release(
    path: Path,
    *,
    _runner: ComposeRunner | None = None,
) -> ReleaseStartReport:
    """Validate and immediately start one approved production release via Compose."""

    binding = _validate_release(path)
    release = binding.release
    argv = (
        *_compose_command_prefix(binding),
        "--project-name",
        release.project_name,
        "--env-file",
        str(binding.env_snapshot.path),
        "-f",
        str(binding.compose_snapshot.path),
        "up",
        "-d",
        "--wait",
        "--remove-orphans",
    )
    _recheck_validated_release(binding, purpose="release start")
    returncode = (_runner or _default_compose_runner)(argv)
    if isinstance(returncode, bool) or not isinstance(returncode, int):
        raise TypeError("Compose runner must return an integer exit code")
    if returncode != 0:
        raise RuntimeError(f"docker compose start failed with exit code {returncode}")
    return ReleaseStartReport(
        release_id=release.release_id,
        project_name=release.project_name,
        model=release.model,
        release_manifest_sha256=binding.manifest_snapshot.sha256,
        compose_sha256=binding.compose_snapshot.sha256,
        adapter_sha256=binding.adapter.inventory_sha256,
        production_profile_sha256=binding.production_profile_sha256,
    )


def build_rollback_plan(current: Path, previous: Path) -> RollbackPlan:
    """Verify two release manifests and render rollback argv without executing it."""

    current_binding = _validate_release(current)
    previous_binding = _validate_release(previous)
    current_release = current_binding.release
    previous_release = previous_binding.release
    if current_release.release_id == previous_release.release_id:
        raise ValueError("current and previous release IDs must differ")
    _require_rollback_compatibility(current_binding, previous_binding)
    current_compose_prefix = _compose_command_prefix(current_binding)
    previous_compose_prefix = _compose_command_prefix(previous_binding)
    plan = RollbackPlan(
        current_release_id=current_release.release_id,
        previous_release_id=previous_release.release_id,
        current_manifest_sha256=current_binding.manifest_snapshot.sha256,
        previous_manifest_sha256=previous_binding.manifest_snapshot.sha256,
        current_adapter_sha256=current_release.adapter_sha256,
        previous_adapter_sha256=previous_release.adapter_sha256,
        stop_current_argv=(
            *current_compose_prefix,
            "--project-name",
            current_release.project_name,
            "--env-file",
            str(current_binding.env_snapshot.path),
            "-f",
            str(current_binding.compose_snapshot.path),
            "down",
        ),
        start_previous_argv=(
            *previous_compose_prefix,
            "--project-name",
            previous_release.project_name,
            "--env-file",
            str(previous_binding.env_snapshot.path),
            "-f",
            str(previous_binding.compose_snapshot.path),
            "up",
            "-d",
            "--wait",
            "--remove-orphans",
        ),
    )
    _recheck_validated_release(current_binding, purpose="rollback plan")
    _recheck_validated_release(previous_binding, purpose="rollback plan")
    return plan
