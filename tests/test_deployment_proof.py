from __future__ import annotations

import hashlib
import io
import json
import threading
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from email.message import Message
from pathlib import Path
from typing import Any

import pytest

import tickettune.deployment_proof as deployment_proof
from tickettune.deployment_proof import (
    MAX_RESPONSE_BYTES,
    HttpResponse,
    ResponseTooLargeError,
    _default_transport,
    adapter_inventory_sha256,
    build_rollback_plan,
    read_api_key,
    run_load_test,
    run_readback,
    validate_base_url,
    write_proof_report,
)
from tickettune.evaluation import score_prediction, summarize_scores
from tickettune.generation import GeneratedPrediction
from tickettune.parity import PredictionArtifact, PredictionContract, _build_report
from tickettune.prompts import SYSTEM_PROMPT
from tickettune.run_manifest import canonical_json_bytes, sha256_bytes, sha256_file
from tickettune.schemas import CATEGORY_LABELS, PRIORITY_LABELS, SENTIMENT_LABELS


def _key(tmp_path: Path) -> Path:
    path = tmp_path / "api-key"
    path.write_text("test-key-0123456789abcdef01234567", encoding="utf-8")
    path.chmod(0o600)
    return path


def _completion() -> dict[str, object]:
    content = {
        "category": "billing",
        "priority": "high",
        "sentiment": "frustrated",
        "response": "I will review the duplicate invoice charge and report the next step.",
        "next_action": "review_duplicate_charge",
    }
    return {"choices": [{"message": {"content": json.dumps(content)}}]}


def test_remote_endpoint_requires_https_and_explicit_acknowledgement() -> None:
    assert validate_base_url("http://127.0.0.1:8000", allow_remote=False) == (
        "http://127.0.0.1:8000"
    )
    assert validate_base_url("https://models.example.test", allow_remote=True) == (
        "https://models.example.test"
    )

    with pytest.raises(ValueError, match="require --allow-remote"):
        validate_base_url("https://models.example.test", allow_remote=False)
    with pytest.raises(ValueError, match="require HTTPS"):
        validate_base_url("http://models.example.test", allow_remote=True)
    with pytest.raises(ValueError, match="must not contain"):
        validate_base_url("https://models.example.test/path?token=nope", allow_remote=True)
    with pytest.raises(ValueError, match="credentials"):
        validate_base_url("https://user:pass@models.example.test", allow_remote=True)


def test_api_key_is_file_only_and_never_returned_in_reports(tmp_path: Path) -> None:
    path = _key(tmp_path)
    assert read_api_key(path).startswith("test-key-")

    link = tmp_path / "link"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="non-symlink"):
        read_api_key(link)

    whitespace = tmp_path / "whitespace"
    whitespace.write_text("  secret-value-that-is-long  ", encoding="utf-8")
    whitespace.chmod(0o600)
    with pytest.raises(ValueError, match="whitespace"):
        read_api_key(whitespace)

    internal_whitespace = tmp_path / "internal-whitespace"
    internal_whitespace.write_text("secret-value-with internal-space-123456", encoding="utf-8")
    internal_whitespace.chmod(0o600)
    with pytest.raises(ValueError, match="whitespace"):
        read_api_key(internal_whitespace)

    non_ascii = tmp_path / "non-ascii"
    non_ascii.write_text("secret-value-0123456789abcdef-café", encoding="utf-8")
    non_ascii.chmod(0o600)
    with pytest.raises(ValueError, match="ASCII"):
        read_api_key(non_ascii)

    control_character = tmp_path / "control-character"
    control_character.write_bytes(b"printable-prefix-0123456789abcdef\x07suffix")
    control_character.chmod(0o600)
    with pytest.raises(ValueError, match="printable ASCII"):
        read_api_key(control_character)

    too_short = tmp_path / "too-short"
    too_short.write_text("x" * 31, encoding="utf-8")
    too_short.chmod(0o600)
    with pytest.raises(ValueError, match="at least 32 bytes"):
        read_api_key(too_short)

    public = tmp_path / "public-key"
    public.write_text("public-secret-value-0123456789", encoding="utf-8")
    public.chmod(0o644)
    with pytest.raises(ValueError, match="permissions"):
        read_api_key(public)

    oversized = tmp_path / "oversized-key"
    oversized.write_bytes(b"x" * 4097)
    oversized.chmod(0o600)
    with pytest.raises(ValueError, match="4096 bytes"):
        read_api_key(oversized)


class _FakeResponse:
    def __init__(self, body: bytes, *, status: int = 200) -> None:
        self.status = status
        self.headers = {"Content-Length": str(len(body))}
        self._body = io.BytesIO(body)

    def read(self, amount: int = -1) -> bytes:
        return self._body.read(amount)

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None


class _FakeOpener:
    def __init__(self, response: _FakeResponse | BaseException) -> None:
        self.response = response

    def open(self, *_args: object, **_kwargs: object) -> _FakeResponse:
        if isinstance(self.response, BaseException):
            raise self.response
        return self.response


def test_default_transport_bounds_success_and_http_error_bodies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oversized = b"sensitive-response" + (b"x" * MAX_RESPONSE_BYTES)
    opener = _FakeOpener(_FakeResponse(oversized))
    monkeypatch.setattr(urllib.request, "build_opener", lambda *_args: opener)

    with pytest.raises(ResponseTooLargeError, match="response body exceeds"):
        _default_transport("GET", "https://example.test/v1/models", {}, None, 1, None)

    error_headers = Message()
    error_headers["Content-Length"] = str(len(oversized))
    error = urllib.error.HTTPError(
        "https://example.test/v1/models",
        413,
        "too large",
        error_headers,
        io.BytesIO(oversized),
    )
    monkeypatch.setattr(
        urllib.request,
        "build_opener",
        lambda *_args: _FakeOpener(error),
    )
    with pytest.raises(ResponseTooLargeError, match="response body exceeds"):
        _default_transport("GET", "https://example.test/v1/models", {}, None, 1, None)


def test_oversized_readback_failure_is_redacted(tmp_path: Path) -> None:
    def transport(*_args: object, **_kwargs: object) -> HttpResponse:
        raise ResponseTooLargeError("response body exceeds limit: sensitive-response")

    report = run_readback(
        base_url="http://127.0.0.1:8000",
        api_key_file=_key(tmp_path),
        model="tickettune",
        expected_base_model="Qwen/Qwen2.5-7B-Instruct",
        _transport=transport,
    )

    serialized = report.model_dump_json()
    assert report.passed is False
    assert "ResponseTooLargeError" in serialized
    assert "sensitive-response" not in serialized


def test_authenticated_readback_checks_serving_claims_and_schema_without_payload_leak(
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, str, dict[str, str], bytes | None]] = []

    def transport(
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
        timeout: float,
        ca_cert: Path | None,
    ) -> HttpResponse:
        assert timeout == 12
        assert ca_cert is None
        calls.append((method, url, headers, body))
        payload: dict[str, object]
        if url.endswith("/v1/models"):
            payload = {
                "data": [
                    {
                        "id": "tickettune-qwen-7b",
                        "parent": "Qwen/Qwen2.5-7B-Instruct",
                    }
                ]
            }
        else:
            payload = _completion()
        return HttpResponse(
            status=200,
            headers={"x-request-id": "req-123"},
            body=json.dumps(payload).encode(),
        )

    output = tmp_path / "readback.json"
    report = run_readback(
        base_url="http://127.0.0.1:8000",
        api_key_file=_key(tmp_path),
        model="tickettune-qwen-7b",
        expected_base_model="Qwen/Qwen2.5-7B-Instruct",
        timeout=12,
        output_path=output,
        _transport=transport,
    )

    assert report.passed is True
    assert report.model_found is True
    assert report.parent_matches is True
    assert report.chat_schema_valid is True
    assert report.request_id_rate == 1.0
    assert report.proof_boundary == "authenticated_loopback_http_serving_claim_readback"
    assert "release_and_adapter_bytes_not_proven" in report.identity_limit
    assert output.is_file()
    serialized = output.read_text(encoding="utf-8")
    assert "test-key-" not in serialized
    assert "duplicate invoice" not in serialized
    assert calls[0][2]["Authorization"].startswith("Bearer test-key-")
    assert json.loads(calls[1][3] or b"{}")["model"] == "tickettune-qwen-7b"


def test_readback_tls_boundary_is_reported_only_for_https(tmp_path: Path) -> None:
    def transport(
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
        timeout: float,
        ca_cert: Path | None,
    ) -> HttpResponse:
        del method, headers, body, timeout, ca_cert
        payload = (
            {
                "data": [
                    {
                        "id": "tickettune",
                        "parent": "Qwen/Qwen2.5-7B-Instruct",
                    }
                ]
            }
            if url.endswith("/v1/models")
            else _completion()
        )
        return HttpResponse(
            status=200,
            headers={"x-request-id": "request-id"},
            body=json.dumps(payload).encode(),
        )

    report = run_readback(
        base_url="https://models.example.test",
        api_key_file=_key(tmp_path),
        model="tickettune",
        expected_base_model="Qwen/Qwen2.5-7B-Instruct",
        allow_remote=True,
        _transport=transport,
    )

    assert report.passed is True
    assert report.proof_boundary == "authenticated_tls_serving_claim_readback"


def test_readback_fails_closed_on_wrong_parent_or_malformed_chat(tmp_path: Path) -> None:
    responses = iter(
        [
            HttpResponse(
                status=200,
                headers={},
                body=json.dumps({"data": [{"id": "tickettune", "parent": "wrong/base"}]}).encode(),
            ),
            HttpResponse(
                status=200,
                headers={},
                body=json.dumps({"choices": [{"message": {"content": "not-json"}}]}).encode(),
            ),
        ]
    )

    def transport(*_args: object, **_kwargs: object) -> HttpResponse:
        return next(responses)

    report = run_readback(
        base_url="http://localhost:8000",
        api_key_file=_key(tmp_path),
        model="tickettune",
        expected_base_model="Qwen/Qwen2.5-7B-Instruct",
        _transport=transport,
    )

    assert report.passed is False
    assert report.model_found is True
    assert report.parent_matches is False
    assert report.chat_schema_valid is False
    assert report.request_id_rate == 0.0


def test_readback_requires_request_ids_from_both_proof_requests(tmp_path: Path) -> None:
    def transport(
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
        timeout: float,
        ca_cert: Path | None,
    ) -> HttpResponse:
        del method, headers, body, timeout, ca_cert
        payload = (
            {
                "data": [
                    {
                        "id": "tickettune",
                        "parent": "Qwen/Qwen2.5-7B-Instruct",
                    }
                ]
            }
            if url.endswith("/v1/models")
            else _completion()
        )
        request_headers = {"x-request-id": "models-id"} if url.endswith("/v1/models") else {}
        return HttpResponse(
            status=200,
            headers=request_headers,
            body=json.dumps(payload).encode(),
        )

    report = run_readback(
        base_url="http://127.0.0.1:8000",
        api_key_file=_key(tmp_path),
        model="tickettune",
        expected_base_model="Qwen/Qwen2.5-7B-Instruct",
        _transport=transport,
    )

    assert report.model_found is True
    assert report.parent_matches is True
    assert report.chat_schema_valid is True
    assert report.request_ids_received == 1
    assert report.passed is False
    assert "request_id_missing" in report.failure_codes


def test_load_report_enforces_success_schema_latency_and_request_ids(tmp_path: Path) -> None:
    call = 0

    def transport(*_args: object, **_kwargs: object) -> HttpResponse:
        nonlocal call
        call += 1
        if call == 4:
            return HttpResponse(status=503, headers={}, body=b"{}")
        return HttpResponse(
            status=200,
            headers={"X-Request-ID": f"req-{call}"},
            body=json.dumps(_completion()).encode(),
        )

    report = run_load_test(
        base_url="http://127.0.0.1:8000",
        api_key_file=_key(tmp_path),
        model="tickettune",
        requests=4,
        concurrency=2,
        min_success_rate=0.75,
        min_schema_valid_rate=0.75,
        min_request_id_rate=0.75,
        max_p95_ms=10_000,
        _transport=transport,
    )

    assert report.requests == 4
    assert report.successes == 3
    assert report.schema_valid_responses == 3
    assert report.request_ids_received == 3
    assert report.success_rate == 0.75
    assert report.schema_valid_rate == 0.75
    assert report.request_id_rate == 0.75
    assert report.latency_p50_ms is not None
    assert report.latency_p95_ms is not None
    assert report.passed is True
    assert report.status_counts == {"200": 3, "503": 1}


@pytest.mark.parametrize(
    ("requests", "concurrency", "message"),
    [
        (0, 1, "requests"),
        (2, 0, "concurrency"),
        (2, 3, "cannot exceed"),
        (10_001, 1, "cannot exceed 10000"),
        (129, 129, "cannot exceed 128"),
    ],
)
def test_load_test_rejects_invalid_shape(
    tmp_path: Path, requests: int, concurrency: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        run_load_test(
            base_url="http://127.0.0.1:8000",
            api_key_file=_key(tmp_path),
            model="tickettune",
            requests=requests,
            concurrency=concurrency,
        )


class _ReleaseBundle:
    def __init__(
        self,
        *,
        release_manifest: Path,
        compose: Path,
        env: Path,
        adapter: Path,
        dataset_manifest: Path,
        test_split: Path,
        training_manifest: Path,
        qualification_report: Path,
        evaluation_manifest: Path,
        evaluation_report: Path,
        baseline_report: Path,
        candidate_predictions: Path,
        baseline_predictions: Path,
        candidate_scores: Path,
        baseline_scores: Path,
        parity_report: Path,
        parity_adapter_predictions: Path,
        parity_merged_predictions: Path,
    ) -> None:
        self.release_manifest = release_manifest
        self.compose = compose
        self.env = env
        self.adapter = adapter
        self.dataset_manifest = dataset_manifest
        self.test_split = test_split
        self.training_manifest = training_manifest
        self.qualification_report = qualification_report
        self.evaluation_manifest = evaluation_manifest
        self.evaluation_report = evaluation_report
        self.baseline_report = baseline_report
        self.candidate_predictions = candidate_predictions
        self.baseline_predictions = baseline_predictions
        self.candidate_scores = candidate_scores
        self.baseline_scores = baseline_scores
        self.parity_report = parity_report
        self.parity_adapter_predictions = parity_adapter_predictions
        self.parity_merged_predictions = parity_merged_predictions


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in rows))
    return path


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _update_release(bundle: _ReleaseBundle, **updates: object) -> None:
    payload = _read_json(bundle.release_manifest)
    payload.update(updates)
    _write_json(bundle.release_manifest, payload)


def _set_release_env(bundle: _ReleaseBundle, key: str, value: str) -> None:
    lines = bundle.env.read_text(encoding="utf-8").splitlines()
    prefix = f"{key}="
    replaced = False
    updated: list[str] = []
    for line in lines:
        if line.startswith(prefix):
            updated.append(f"{prefix}{value}")
            replaced = True
        else:
            updated.append(line)
    if not replaced:
        updated.append(f"{prefix}{value}")
    bundle.env.write_text("\n".join(updated) + "\n", encoding="utf-8")
    _update_release(bundle, env_sha256=sha256_file(bundle.env))


def _sync_candidate_evaluation(
    bundle: _ReleaseBundle,
    report_payload: dict[str, object],
) -> None:
    _write_json(bundle.evaluation_report, report_payload)
    report_sha256 = sha256_file(bundle.evaluation_report)
    manifest = _read_json(bundle.evaluation_manifest)
    artifacts = manifest["artifact_sha256"]
    result = manifest["result"]
    assert isinstance(artifacts, dict)
    assert isinstance(result, dict)
    candidate = result["candidate"]
    assert isinstance(candidate, dict)
    artifacts["candidate/evaluation-report.json"] = report_sha256
    candidate["report"] = report_payload
    _write_json(bundle.evaluation_manifest, manifest)
    _update_release(
        bundle,
        evaluation_manifest_sha256=sha256_file(bundle.evaluation_manifest),
        evaluation_report_sha256=report_sha256,
    )


def _sync_baseline_evaluation(
    bundle: _ReleaseBundle,
    report_payload: dict[str, object],
) -> None:
    _write_json(bundle.baseline_report, report_payload)
    report_sha256 = sha256_file(bundle.baseline_report)
    manifest = _read_json(bundle.evaluation_manifest)
    artifacts = manifest["artifact_sha256"]
    result = manifest["result"]
    assert isinstance(artifacts, dict)
    assert isinstance(result, dict)
    baseline = result["baseline"]
    assert isinstance(baseline, dict)
    artifacts["baseline/evaluation-report.json"] = report_sha256
    baseline["report"] = report_payload
    _write_json(bundle.evaluation_manifest, manifest)
    _update_release(
        bundle,
        evaluation_manifest_sha256=sha256_file(bundle.evaluation_manifest),
    )


def _sync_parity(bundle: _ReleaseBundle, payload: dict[str, object]) -> None:
    _write_json(bundle.parity_report, payload)
    _update_release(bundle, parity_report_sha256=sha256_file(bundle.parity_report))


def _release_bundle(root: Path, *, slug: str, weight_bytes: bytes) -> _ReleaseBundle:
    root.mkdir(parents=True)
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    model_revision = "a09a35458c702b33eeacc393d103063234e8bc28"
    git_revision = "0123456789abcdef0123456789abcdef01234567"
    served_model = "tickettune-qwen-7b-quality"
    release_date = "20260718" if slug.startswith("current") else "20260717"
    release_id = f"release-{release_date}-{slug}"
    run_id = f"run-{slug}"

    profile_dir = Path(__file__).resolve().parents[1] / "deploy" / "vllm" / "production"
    for filename in (
        "compose.yaml",
        "nginx.conf",
        "prometheus.yml",
        "alerts.yml",
        "alertmanager.yml",
        "vllm-entrypoint.py",
    ):
        (root / filename).write_bytes((profile_dir / filename).read_bytes())
    compose = root / "compose.yaml"

    adapter = root / "artifacts" / "runs" / run_id / "adapter"
    adapter.mkdir(parents=True)
    adapter_config = adapter / "adapter_config.json"
    _write_json(
        adapter_config,
        {
            "base_model_name_or_path": model_name,
            "revision": model_revision,
            "r": 16,
        },
    )
    adapter_weight = adapter / "adapter_model.safetensors"
    adapter_weight.write_bytes(weight_bytes)
    adapter_digest = adapter_inventory_sha256(adapter)
    adapter_config_sha = sha256_file(adapter_config)
    adapter_weight_sha = sha256_file(adapter_weight)

    held_out_ids = [f"TICKET-{index:04d}" for index in range(1, 101)]
    expected_by_id: dict[str, dict[str, object]] = {
        record_id: {
            "category": CATEGORY_LABELS[(index - 1) % len(CATEGORY_LABELS)],
            "priority": PRIORITY_LABELS[(index - 1) % len(PRIORITY_LABELS)],
            "sentiment": SENTIMENT_LABELS[(index - 1) % len(SENTIMENT_LABELS)],
            "response": ("I will review ticket [TICKET_ID] and provide a clear next step."),
            "next_action": "review_ticket",
        }
        for index, record_id in enumerate(held_out_ids, 1)
    }

    def prepared_record(record_id: str, expected: dict[str, object]) -> dict[str, object]:
        completion = canonical_json_bytes(expected).decode("utf-8")
        return {
            "id": record_id,
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": "Please triage support ticket [TICKET_ID].",
                },
            ],
            "completion": [{"role": "assistant", "content": completion}],
            "expected": expected,
            "provenance": {
                "source": "synthetic",
                "created_by": "TicketTune test fixture",
                "license": "CC0-1.0",
                "contains_real_customer_data": False,
            },
            "pii_placeholders": ["[TICKET_ID]"],
        }

    dataset_dir = root / "dataset"
    test_split = _write_jsonl(
        dataset_dir / "test.jsonl",
        [prepared_record(record_id, expected_by_id[record_id]) for record_id in held_out_ids],
    )
    train_ids = ["TRAIN-0001"]
    validation_ids = ["VALIDATION-0001"]
    split_expected = {
        "train": [expected_by_id[held_out_ids[0]]],
        "validation": [expected_by_id[held_out_ids[1]]],
        "test": [expected_by_id[record_id] for record_id in held_out_ids],
    }

    def labels(values: list[dict[str, object]]) -> dict[str, dict[str, int]]:
        return {
            name: dict(sorted(Counter(str(value[name]) for value in values).items()))
            for name in ("category", "priority", "sentiment")
        }

    train_sha = _digest(f"{slug}-train")
    validation_sha = _digest(f"{slug}-validation")
    test_sha = sha256_file(test_split)
    all_expected = [
        *split_expected["train"],
        *split_expected["validation"],
        *split_expected["test"],
    ]
    dataset_manifest_payload: dict[str, object] = {
        "schema_version": "1.0",
        "source_file": "dataset.jsonl",
        "source_sha256": _digest(f"{slug}-source"),
        "seed": 42,
        "split_fractions": {"train": 0.8, "validation": 0.1, "test": 0.1},
        "total_examples": len(train_ids) + len(validation_ids) + len(held_out_ids),
        "labels": labels(all_expected),
        "splits": {
            "train": {
                "file": "train.jsonl",
                "count": len(train_ids),
                "sha256": train_sha,
                "labels": labels(split_expected["train"]),
            },
            "validation": {
                "file": "validation.jsonl",
                "count": len(validation_ids),
                "sha256": validation_sha,
                "labels": labels(split_expected["validation"]),
            },
            "test": {
                "file": "test.jsonl",
                "count": len(held_out_ids),
                "sha256": test_sha,
                "labels": labels(split_expected["test"]),
            },
        },
        "split_ids": {
            "train": train_ids,
            "validation": validation_ids,
            "test": held_out_ids,
        },
    }
    dataset_manifest = _write_json(
        dataset_dir / "manifest.json",
        dataset_manifest_payload,
    )
    qualification_policies = (
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
    )

    qualification_payload: dict[str, object] = {
        "schema_version": "1.2",
        "qualified": True,
        "dataset_tier": "qualification_candidate",
        "source_path": str(root / "dataset.jsonl"),
        "source_sha256": dataset_manifest_payload["source_sha256"],
        "review_manifest_path": str(root / "review-manifest.json"),
        "review_manifest_sha256": _digest(f"{slug}-review"),
        "record_count": 1000,
        "reviewed_count": 1000,
        "independent_reviewer_count": 2,
        "held_out_count": 100,
        "held_out_ids": held_out_ids,
        "review_date": "2026-07-18",
        "intended_domain": "Synthetic support-ticket triage",
        "decisions": [
            {
                "policy": policy,
                "passed": True,
                "observed": "passing",
                "required": "passing",
                "detail": "Complete qualification evidence.",
            }
            for policy in qualification_policies
        ],
        "prepared_manifest_path": str(dataset_manifest),
        "prepared_manifest_sha256": sha256_file(dataset_manifest),
        "holdout_freeze_path": str(root / "holdout-freeze.json"),
        "holdout_freeze_sha256": _digest(f"{slug}-holdout-freeze"),
        "reviewer_packet_paths": [
            str(root / "reviewer-a.json"),
            str(root / "reviewer-b.json"),
        ],
        "reviewer_packet_sha256": [
            _digest(f"{slug}-reviewer-a"),
            _digest(f"{slug}-reviewer-b"),
        ],
        "reviewer_ids": ["human-reviewer-01", "human-reviewer-02"],
        "proof_boundary": "review evidence only",
    }
    qualification_report = _write_json(
        root / "qualification-report.json",
        qualification_payload,
    )

    dataset_sha256 = {
        "source": str(qualification_payload["source_sha256"]),
        "manifest": sha256_file(dataset_manifest),
        "train": train_sha,
        "validation": validation_sha,
        "test": test_sha,
        "qualification_review_manifest": str(qualification_payload["review_manifest_sha256"]),
        "qualification_report": sha256_bytes(qualification_payload),
    }
    training_config: dict[str, object] = {
        "project_name": "ticket-tune",
        "model": {
            "name_or_path": model_name,
            "revision": model_revision,
            "parameters_b": 7.0,
            "torch_dtype": "bfloat16",
        },
        "lora": {"method": "qlora"},
        "training": {
            "bf16": True,
            "fp16": False,
            "use_cpu": False,
        },
        "seed": 42,
        "evaluation": {
            "thresholds": {
                "strict_json_rate": 0.9,
                "schema_valid_rate": 0.9,
                "category_accuracy": 0.9,
                "priority_accuracy": 0.9,
                "sentiment_accuracy": 0.9,
                "response_policy_rate": 0.9,
            }
        },
    }
    training_payload: dict[str, object] = {
        "schema_version": "1.1",
        "run_id": run_id,
        "created_at": "2026-07-18T00:00:00Z",
        "status": "completed",
        "project_name": "ticket-tune",
        "model_name_or_path": model_name,
        "model_revision": model_revision,
        "method": "qlora",
        "seed": 42,
        "config_sha256": sha256_bytes(training_config),
        "config": training_config,
        "dataset_sha256": dataset_sha256,
        "packages": {},
        "runtime": {},
        "hardware_preflight": {
            "report": {
                "accelerator": "cuda",
                "total_memory_gb": 24.0,
                "device_name": "test NVIDIA GPU",
                "accelerator_count": 1,
                "torch_version": "2.8.0",
                "cuda_version": "12.8",
                "compute_capability": "8.0",
                "supports_bfloat16": True,
                "platform_system": "Linux",
                "platform_machine": "x86_64",
            },
            "execution_accelerator": "cuda",
            "method": "qlora",
            "model_parameters_b": 7.0,
            "estimated_training_memory_gb": 5.6,
            "findings": [
                {
                    "level": "info",
                    "code": "cuda_adapter_training",
                    "message": "CUDA adapter training is compatible with the selected method.",
                    "remediation": None,
                }
            ],
            "compatible": True,
        },
        "git_revision": git_revision,
        "git_branch": "main",
        "git_dirty": False,
        "training_duration_seconds": 120.0,
        "peak_accelerator_memory_mb": 8192.0,
        "metrics": {"optimizer_steps": 1, "train_loss": 0.25},
        "artifacts": [
            {
                "path": "adapter/adapter_config.json",
                "sha256": adapter_config_sha,
                "size_bytes": adapter_config.stat().st_size,
            },
            {
                "path": "adapter/adapter_model.safetensors",
                "sha256": adapter_weight_sha,
                "size_bytes": adapter_weight.stat().st_size,
            },
        ],
        "error": None,
    }
    training_manifest = _write_json(adapter.parent / "manifest.json", training_payload)
    training_manifest_sha = sha256_file(training_manifest)

    qualification_lineage = {
        key: value for key, value in dataset_sha256.items() if key.startswith("qualification_")
    }
    provenance: dict[str, object] = {
        "dataset_manifest_sha256": dataset_sha256["manifest"],
        "dataset_split_sha256": dataset_sha256["test"],
        "generation_config_sha256": _digest(f"{slug}-generation-config"),
        "model_name_or_path": model_name,
        "model_revision": model_revision,
        "training_manifest_sha256": training_manifest_sha,
        "training_config_sha256": training_payload["config_sha256"],
        "training_dataset_sha256": dataset_sha256,
        "qualification_sha256": qualification_lineage,
        "adapter_path": str(adapter),
        "adapter_config_sha256": adapter_config_sha,
        "adapter_weight_sha256": {adapter_weight.name: adapter_weight_sha},
    }

    def generated_prediction(
        record_id: str,
        *,
        candidate: bool,
    ) -> GeneratedPrediction:
        lineage: dict[str, object] = {}
        if candidate:
            lineage = {
                "training_manifest_sha256": training_manifest_sha,
                "training_config_sha256": training_payload["config_sha256"],
                "training_dataset_sha256": dataset_sha256,
                "qualification_sha256": qualification_lineage,
                "adapter_path": str(adapter),
                "adapter_config_sha256": adapter_config_sha,
                "adapter_weight_sha256": {adapter_weight.name: adapter_weight_sha},
            }
        expected = expected_by_id[record_id]
        return GeneratedPrediction(
            id=record_id,
            expected=expected,
            prediction=canonical_json_bytes(expected).decode("utf-8"),
            latency_ms=10.0,
            prompt_tokens=64,
            generated_tokens=32,
            model_name_or_path=model_name,
            model_revision=model_revision,
            dataset_manifest_sha256=dataset_sha256["manifest"],
            dataset_split_sha256=dataset_sha256["test"],
            generation_config_sha256=provenance["generation_config_sha256"],
            **lineage,
        )

    candidate_prediction_models = tuple(
        generated_prediction(record_id, candidate=True) for record_id in held_out_ids
    )
    baseline_prediction_models = tuple(
        generated_prediction(record_id, candidate=False) for record_id in held_out_ids
    )
    evaluation_root = root / "evaluation"
    candidate_predictions = _write_jsonl(
        evaluation_root / "candidate-predictions.jsonl",
        [item.model_dump(mode="json") for item in candidate_prediction_models],
    )
    baseline_predictions = _write_jsonl(
        evaluation_root / "baseline-predictions.jsonl",
        [item.model_dump(mode="json") for item in baseline_prediction_models],
    )
    candidate_score_models = tuple(
        score_prediction(
            row.expected,
            row.prediction,
            id=row.id,
            latency_ms=row.latency_ms,
        )
        for row in candidate_prediction_models
    )
    baseline_score_models = tuple(
        score_prediction(
            row.expected,
            row.prediction,
            id=row.id,
            latency_ms=row.latency_ms,
        )
        for row in baseline_prediction_models
    )
    candidate_summary = summarize_scores(candidate_score_models).model_dump(mode="json")
    baseline_summary = summarize_scores(baseline_score_models).model_dump(mode="json")
    absolute_minimums = training_config["evaluation"]
    assert isinstance(absolute_minimums, dict)
    absolute_minimums = absolute_minimums["thresholds"]
    assert isinstance(absolute_minimums, dict)

    def absolute_thresholds(summary: dict[str, object]) -> list[dict[str, object]]:
        return [
            {
                "metric": metric,
                "value": summary[metric],
                "minimum": absolute_minimums[metric],
                "passed": float(summary[metric]) >= float(absolute_minimums[metric]),
            }
            for metric in deployment_proof._EVALUATION_ABSOLUTE_METRICS
        ]

    evaluation_report_payload: dict[str, object] = {
        "schema_version": "1.3",
        "generated_at": "2026-07-18T01:00:00Z",
        "predictions_path": str(candidate_predictions),
        "predictions_sha256": sha256_file(candidate_predictions),
        "provenance": provenance,
        "summary": candidate_summary,
        "thresholds": absolute_thresholds(candidate_summary),
        "passed": True,
        "results": [item.model_dump(mode="json") for item in candidate_score_models],
    }
    evaluation_report = _write_json(
        evaluation_root / "candidate" / "evaluation-report.json",
        evaluation_report_payload,
    )
    candidate_markdown = evaluation_root / "candidate" / "evaluation-report.md"
    candidate_markdown.write_text("# Candidate evaluation\n", encoding="utf-8")
    candidate_scores = _write_jsonl(
        evaluation_root / "candidate" / "scored-predictions.jsonl",
        [item.model_dump(mode="json") for item in candidate_score_models],
    )
    evaluation_report_sha = sha256_file(evaluation_report)
    baseline_provenance = {
        "dataset_manifest_sha256": dataset_sha256["manifest"],
        "dataset_split_sha256": dataset_sha256["test"],
        "generation_config_sha256": provenance["generation_config_sha256"],
        "model_name_or_path": model_name,
        "model_revision": model_revision,
        "training_manifest_sha256": None,
        "training_config_sha256": None,
        "training_dataset_sha256": {},
        "qualification_sha256": {},
        "adapter_path": None,
        "adapter_config_sha256": None,
        "adapter_weight_sha256": {},
    }
    baseline_report_payload: dict[str, object] = {
        "schema_version": "1.3",
        "generated_at": "2026-07-18T01:00:00Z",
        "predictions_path": str(baseline_predictions),
        "predictions_sha256": sha256_file(baseline_predictions),
        "provenance": baseline_provenance,
        "summary": baseline_summary,
        "thresholds": absolute_thresholds(baseline_summary),
        "passed": True,
        "results": [item.model_dump(mode="json") for item in baseline_score_models],
    }
    baseline_report = _write_json(
        evaluation_root / "baseline" / "evaluation-report.json",
        baseline_report_payload,
    )
    baseline_markdown = evaluation_root / "baseline" / "evaluation-report.md"
    baseline_markdown.write_text("# Baseline evaluation\n", encoding="utf-8")
    baseline_scores = _write_jsonl(
        evaluation_root / "baseline" / "scored-predictions.jsonl",
        [item.model_dump(mode="json") for item in baseline_score_models],
    )
    baseline_report_sha = sha256_file(baseline_report)
    metric_deltas = {
        metric: float(candidate_summary[metric]) - float(baseline_summary[metric])
        for metric in deployment_proof._EVALUATION_NON_REGRESSION_METRICS
    }
    non_regression = [
        {
            "metric": metric,
            "value": metric_deltas[metric],
            "minimum": 0.0,
            "passed": metric_deltas[metric] >= 0,
        }
        for metric in deployment_proof._EVALUATION_NON_REGRESSION_METRICS
    ]
    evaluation_id = f"evaluation-{slug}"
    evaluation_manifest = evaluation_root / "evaluation-manifest.json"
    candidate_artifacts: dict[str, object] = {
        "report": evaluation_report_payload,
        "json_report_path": str(evaluation_report),
        "markdown_report_path": str(candidate_markdown),
        "scored_predictions_path": str(candidate_scores),
        "passed": True,
    }
    baseline_artifacts: dict[str, object] = {
        "report": baseline_report_payload,
        "json_report_path": str(baseline_report),
        "markdown_report_path": str(baseline_markdown),
        "scored_predictions_path": str(baseline_scores),
        "passed": True,
    }
    evaluation_manifest_payload: dict[str, object] = {
        "schema_version": "1.0",
        "evaluation_id": evaluation_id,
        "created_at": "2026-07-18T01:00:00Z",
        "status": "completed",
        "config_sha256": training_payload["config_sha256"],
        "candidate_passed": True,
        "comparison_passed": True,
        "artifact_sha256": {
            "candidate-predictions.jsonl": sha256_file(candidate_predictions),
            "candidate/evaluation-report.json": evaluation_report_sha,
            "candidate/evaluation-report.md": sha256_file(candidate_markdown),
            "candidate/scored-predictions.jsonl": sha256_file(candidate_scores),
            "baseline-predictions.jsonl": sha256_file(baseline_predictions),
            "baseline/evaluation-report.json": baseline_report_sha,
            "baseline/evaluation-report.md": sha256_file(baseline_markdown),
            "baseline/scored-predictions.jsonl": sha256_file(baseline_scores),
        },
        "result": {
            "evaluation_id": evaluation_id,
            "output_dir": str(evaluation_root),
            "manifest_path": str(evaluation_manifest),
            "latest_pointer_path": str(root / "latest-evaluation.json"),
            "candidate": candidate_artifacts,
            "baseline": baseline_artifacts,
            "comparison": {
                "passed": True,
                "candidate_passed": True,
                "baseline_passed": True,
                "metric_deltas": metric_deltas,
                "non_regression": non_regression,
            },
        },
    }
    _write_json(evaluation_manifest, evaluation_manifest_payload)

    parity_adapter_models = candidate_prediction_models
    merge_provenance_sha = _digest(f"{slug}-merge-provenance")
    merged_artifact_sha = {
        "config.json": _digest(f"{slug}-merged-config"),
        "model.safetensors": _digest(f"{slug}-merged-model"),
    }
    parity_merged_models = tuple(
        GeneratedPrediction(
            id=row.id,
            expected=row.expected,
            prediction=row.prediction,
            latency_ms=row.latency_ms,
            prompt_tokens=row.prompt_tokens,
            generated_tokens=row.generated_tokens,
            model_name_or_path=row.model_name_or_path,
            model_revision=row.model_revision,
            dataset_manifest_sha256=row.dataset_manifest_sha256,
            dataset_split_sha256=row.dataset_split_sha256,
            generation_config_sha256=row.generation_config_sha256,
            training_manifest_sha256=row.training_manifest_sha256,
            training_config_sha256=row.training_config_sha256,
            training_dataset_sha256=row.training_dataset_sha256,
            qualification_sha256=row.qualification_sha256,
            merged_model_path=str(root / "merged"),
            merge_provenance_sha256=merge_provenance_sha,
            merged_artifact_sha256=merged_artifact_sha,
            merged_adapter_config_sha256=adapter_config_sha,
            merged_adapter_weight_files=(adapter_weight.name,),
            merged_adapter_weight_sha256=(adapter_weight_sha,),
        )
        for row in candidate_prediction_models
    )
    parity_adapter_predictions = _write_jsonl(
        root / "parity-adapter-predictions.jsonl",
        [item.model_dump(mode="json") for item in parity_adapter_models],
    )
    parity_merged_predictions = _write_jsonl(
        root / "parity-merged-predictions.jsonl",
        [item.model_dump(mode="json") for item in parity_merged_models],
    )
    adapter_prediction_artifact = PredictionArtifact(
        role="adapter",
        path=str(parity_adapter_predictions),
        sha256=sha256_file(parity_adapter_predictions),
        model_name_or_path=model_name,
        model_revision=model_revision,
        dataset_manifest_sha256=dataset_sha256["manifest"],
        dataset_split_sha256=dataset_sha256["test"],
        generation_config_sha256=str(provenance["generation_config_sha256"]),
        training_manifest_sha256=training_manifest_sha,
        training_config_sha256=str(training_payload["config_sha256"]),
        training_dataset_sha256=dataset_sha256,
        qualification_sha256=qualification_lineage,
        adapter_path=str(adapter),
        adapter_config_sha256=adapter_config_sha,
        adapter_weight_sha256={adapter_weight.name: adapter_weight_sha},
    )
    merged_prediction_artifact = PredictionArtifact(
        role="merged",
        path=str(parity_merged_predictions),
        sha256=sha256_file(parity_merged_predictions),
        model_name_or_path=model_name,
        model_revision=model_revision,
        dataset_manifest_sha256=dataset_sha256["manifest"],
        dataset_split_sha256=dataset_sha256["test"],
        generation_config_sha256=str(provenance["generation_config_sha256"]),
        training_manifest_sha256=training_manifest_sha,
        training_config_sha256=str(training_payload["config_sha256"]),
        training_dataset_sha256=dataset_sha256,
        qualification_sha256=qualification_lineage,
        merged_model_path=str(root / "merged"),
        merge_provenance_sha256=merge_provenance_sha,
        merged_artifact_sha256=merged_artifact_sha,
        merged_adapter_config_sha256=adapter_config_sha,
        merged_adapter_weight_files=(adapter_weight.name,),
        merged_adapter_weight_sha256=(adapter_weight_sha,),
    )
    parity_contract = PredictionContract(
        dataset_manifest_sha256=dataset_sha256["manifest"],
        dataset_split_sha256=dataset_sha256["test"],
        generation_config_sha256=str(provenance["generation_config_sha256"]),
        training_manifest_sha256=training_manifest_sha,
        training_config_sha256=str(training_payload["config_sha256"]),
        training_dataset_sha256=dataset_sha256,
        qualification_sha256=qualification_lineage,
    )
    parity_model = _build_report(
        parity_adapter_models,
        parity_merged_models,
        adapter_artifact=adapter_prediction_artifact,
        merged_artifact=merged_prediction_artifact,
        contract=parity_contract,
        merge_dtype="bfloat16",
    )
    parity_report = _write_json(
        root / "parity-report.json",
        parity_model.model_dump(mode="json"),
    )

    env = root / "release.env"
    env.write_text(
        "\n".join(
            (
                f"RELEASE_ID={release_id}",
                f"RELEASE_GIT_REVISION={git_revision}",
                f"SERVED_MODEL_NAME={served_model}",
                f"ADAPTER_PATH={adapter}",
                f"EXPECTED_ADAPTER_SHA256={adapter_digest}",
                f"BASE_MODEL={model_name}",
                f"MODEL_REVISION={model_revision}",
                "SECRET_GROUP_ID=1234",
                f"HF_CACHE_PATH={root / 'hf-cache'}",
                f"RUNTIME_CACHE_PATH={root / 'runtime-cache'}",
                f"VLLM_API_KEY_FILE={root / 'secrets' / 'vllm-api-key'}",
                f"TLS_CERTIFICATE_FILE={root / 'secrets' / 'tls-certificate.pem'}",
                f"TLS_PRIVATE_KEY_FILE={root / 'secrets' / 'tls-private-key.pem'}",
                (f"ALERTMANAGER_WEBHOOK_URL_FILE={root / 'secrets' / 'alertmanager-webhook-url'}"),
                "",
            )
        ),
        encoding="utf-8",
    )
    release_manifest = _write_json(
        root / "release-manifest.json",
        {
            "schema_version": "2.0",
            "release_id": release_id,
            "project_name": deployment_proof.APPROVED_PRODUCTION_PROJECT,
            "compose_file": str(compose),
            "compose_sha256": sha256_file(compose),
            "env_file": str(env),
            "env_sha256": sha256_file(env),
            "model": served_model,
            "adapter_path": str(adapter),
            "adapter_sha256": adapter_digest,
            "dataset_manifest_file": str(dataset_manifest),
            "dataset_manifest_sha256": sha256_file(dataset_manifest),
            "test_split_file": str(test_split),
            "test_split_sha256": sha256_file(test_split),
            "training_manifest_file": str(training_manifest),
            "training_manifest_sha256": training_manifest_sha,
            "qualification_report_file": str(qualification_report),
            "qualification_report_sha256": sha256_file(qualification_report),
            "evaluation_manifest_file": str(evaluation_manifest),
            "evaluation_manifest_sha256": sha256_file(evaluation_manifest),
            "evaluation_report_file": str(evaluation_report),
            "evaluation_report_sha256": evaluation_report_sha,
            "parity_report_file": str(parity_report),
            "parity_report_sha256": sha256_file(parity_report),
        },
    )
    return _ReleaseBundle(
        release_manifest=release_manifest,
        compose=compose,
        env=env,
        adapter=adapter,
        dataset_manifest=dataset_manifest,
        test_split=test_split,
        training_manifest=training_manifest,
        qualification_report=qualification_report,
        evaluation_manifest=evaluation_manifest,
        evaluation_report=evaluation_report,
        baseline_report=baseline_report,
        candidate_predictions=candidate_predictions,
        baseline_predictions=baseline_predictions,
        candidate_scores=candidate_scores,
        baseline_scores=baseline_scores,
        parity_report=parity_report,
        parity_adapter_predictions=parity_adapter_predictions,
        parity_merged_predictions=parity_merged_predictions,
    )


def test_rollback_plan_binds_both_immutable_releases_without_execution(tmp_path: Path) -> None:
    current = _release_bundle(
        tmp_path / "current",
        slug="current",
        weight_bytes=b"current-adapter",
    )
    previous = _release_bundle(
        tmp_path / "previous",
        slug="previous",
        weight_bytes=b"previous-adapter",
    )

    validation = deployment_proof.validate_release_manifest(current.release_manifest)
    plan = build_rollback_plan(current.release_manifest, previous.release_manifest)

    assert validation.passed is True
    assert validation.proof_boundary == "semantic_release_validation"
    assert validation.release_id == "release-20260718-current"
    assert validation.base_model == "Qwen/Qwen2.5-7B-Instruct"
    assert validation.production_profile == "tickettune-vllm-production-v1"
    assert str(current.adapter) not in validation.model_dump_json()
    assert plan.executed is False
    assert plan.proof_boundary == "rollback_plan_only"
    assert "revalidate" in plan.check_use_limit
    assert plan.current_release_id == "release-20260718-current"
    assert plan.previous_release_id == "release-20260717-previous"
    assert plan.stop_current_argv[-1] == "down"
    assert plan.start_previous_argv[-2:] == ("--wait", "--remove-orphans")
    expected_prefix = (
        "/usr/bin/env",
        "-i",
        "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "HOME=/var/empty",
        deployment_proof.APPROVED_DOCKER_EXECUTABLE,
        "--host",
        deployment_proof.APPROVED_DOCKER_HOST,
        "compose",
    )
    assert plan.stop_current_argv[: len(expected_prefix)] == expected_prefix
    assert plan.start_previous_argv[: len(expected_prefix)] == expected_prefix
    assert deployment_proof.APPROVED_PRODUCTION_PROJECT in plan.stop_current_argv
    assert deployment_proof.APPROVED_PRODUCTION_PROJECT in plan.start_previous_argv
    assert "atomic_byte_binding_requires_immutable_release_storage" in (plan.path_binding_limit)
    assert plan.current_adapter_sha256 == adapter_inventory_sha256(current.adapter)
    assert plan.previous_adapter_sha256 == adapter_inventory_sha256(previous.adapter)

    current.env.write_text("RELEASE_ID=tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="env file SHA-256 mismatch"):
        build_rollback_plan(current.release_manifest, previous.release_manifest)


def test_release_start_validates_rechecks_and_runs_only_shell_free_profile_argv(
    tmp_path: Path,
) -> None:
    release = _release_bundle(
        tmp_path / "release",
        slug="current",
        weight_bytes=b"approved-adapter",
    )
    calls: list[tuple[str, ...]] = []

    def runner(argv: tuple[str, ...]) -> int:
        calls.append(argv)
        return 0

    report = deployment_proof.start_release(
        release.release_manifest,
        _runner=runner,
    )

    assert report.executed is True
    assert report.passed is True
    assert report.proof_boundary == "validated_compose_start_invocation"
    assert report.production_profile == "tickettune-vllm-production-v1"
    assert "atomic_byte_binding_requires_immutable_release_storage" in (report.path_binding_limit)
    assert len(calls) == 1
    argv = calls[0]
    assert isinstance(argv, tuple)
    assert argv[:8] == (
        "/usr/bin/env",
        "-i",
        "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "HOME=/var/empty",
        deployment_proof.APPROVED_DOCKER_EXECUTABLE,
        "--host",
        deployment_proof.APPROVED_DOCKER_HOST,
        "compose",
    )
    assert argv[-4:] == ("up", "-d", "--wait", "--remove-orphans")
    assert not any("DOCKER_CONTEXT" in token for token in argv)
    assert not any("DOCKER_CONFIG" in token for token in argv)


def test_release_manifest_requires_the_owned_production_project_slot(tmp_path: Path) -> None:
    release = _release_bundle(
        tmp_path / "release",
        slug="current",
        weight_bytes=b"approved-adapter",
    )
    _update_release(release, project_name="unrelated-production")

    with pytest.raises(ValueError, match="tickettune-production"):
        deployment_proof.validate_release_manifest(release.release_manifest)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("project", "project slot"),
        ("served_model", "served model"),
        ("profile", "production profile"),
        ("base_model", "base model"),
        ("model_revision", "model revision"),
    ],
)
def test_rollback_requires_one_compatible_deployment_slot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    message: str,
) -> None:
    current = _release_bundle(
        tmp_path / "current",
        slug="current",
        weight_bytes=b"current-adapter",
    )
    previous = _release_bundle(
        tmp_path / "previous",
        slug="previous",
        weight_bytes=b"previous-adapter",
    )
    current_binding = deployment_proof._validate_release(current.release_manifest)
    previous_binding = deployment_proof._validate_release(previous.release_manifest)

    if mutation == "project":
        previous_binding = replace(
            previous_binding,
            release=previous_binding.release.model_copy(
                update={"project_name": "unrelated-production"}
            ),
        )
    elif mutation == "served_model":
        previous_binding = replace(
            previous_binding,
            release=previous_binding.release.model_copy(update={"model": "other-model"}),
        )
    elif mutation == "profile":
        previous_binding = replace(
            previous_binding,
            production_profile_sha256="f" * 64,
        )
    elif mutation == "base_model":
        previous_binding = replace(
            previous_binding,
            training=previous_binding.training.model_copy(
                update={"model_name_or_path": "Other/Model"}
            ),
        )
    else:
        previous_binding = replace(
            previous_binding,
            training=previous_binding.training.model_copy(update={"model_revision": "f" * 40}),
        )

    bindings = iter((current_binding, previous_binding))
    monkeypatch.setattr(
        deployment_proof,
        "_validate_release",
        lambda _path: next(bindings),
    )

    with pytest.raises(ValueError, match=message):
        build_rollback_plan(current.release_manifest, previous.release_manifest)


def test_release_start_rejects_unapproved_profile_and_support_bytes(
    tmp_path: Path,
) -> None:
    release = _release_bundle(
        tmp_path / "release",
        slug="current",
        weight_bytes=b"approved-adapter",
    )
    calls: list[tuple[str, ...]] = []

    def forbidden_runner(argv: tuple[str, ...]) -> int:
        calls.append(argv)
        return 0

    release.compose.write_text(
        release.compose.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    _update_release(release, compose_sha256=sha256_file(release.compose))
    with pytest.raises(ValueError, match="approved TicketTune production profile"):
        deployment_proof.start_release(
            release.release_manifest,
            _runner=forbidden_runner,
        )
    assert calls == []

    release = _release_bundle(
        tmp_path / "release-support",
        slug="current-support",
        weight_bytes=b"approved-adapter",
    )
    (release.compose.parent / "nginx.conf").write_text("events {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"nginx.conf.*approved"):
        deployment_proof.start_release(
            release.release_manifest,
            _runner=forbidden_runner,
        )
    assert calls == []


@pytest.mark.parametrize("mutation", ["evidence", "profile", "adapter"])
def test_release_start_fails_closed_before_runner_on_recheck_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    release = _release_bundle(
        tmp_path / "release",
        slug="current",
        weight_bytes=b"approved-adapter",
    )
    calls: list[tuple[str, ...]] = []
    original = deployment_proof._recheck_validated_release

    def racing_recheck(binding: Any, *, purpose: str) -> None:
        if mutation == "evidence":
            snapshot = binding.evidence_snapshots[-1]
            snapshot.path.write_bytes(snapshot.payload + b" ")
        elif mutation == "profile":
            snapshot = binding.profile_snapshots[0]
            snapshot.path.write_bytes(snapshot.payload + b" ")
        else:
            (binding.adapter.path / "adapter_model.safetensors").write_bytes(
                b"changed-after-validation"
            )
        original(binding, purpose=purpose)

    monkeypatch.setattr(
        deployment_proof,
        "_recheck_validated_release",
        racing_recheck,
    )

    def forbidden_runner(argv: tuple[str, ...]) -> int:
        calls.append(argv)
        return 0

    with pytest.raises(ValueError, match="changed before release start"):
        deployment_proof.start_release(
            release.release_manifest,
            _runner=forbidden_runner,
        )
    assert calls == []


def test_release_start_surfaces_compose_failure_without_false_receipt(tmp_path: Path) -> None:
    release = _release_bundle(
        tmp_path / "release",
        slug="current",
        weight_bytes=b"approved-adapter",
    )
    with pytest.raises(RuntimeError, match="exit code 17"):
        deployment_proof.start_release(
            release.release_manifest,
            _runner=lambda _argv: 17,
        )


def test_release_start_requires_every_fail_fast_compose_input(tmp_path: Path) -> None:
    release = _release_bundle(
        tmp_path / "release",
        slug="current",
        weight_bytes=b"approved-adapter",
    )
    env_lines = release.env.read_text(encoding="utf-8").splitlines()
    release.env.write_text(
        "\n".join(line for line in env_lines if not line.startswith("TLS_PRIVATE_KEY_FILE="))
        + "\n",
        encoding="utf-8",
    )
    _update_release(release, env_sha256=sha256_file(release.env))

    with pytest.raises(ValueError, match="TLS_PRIVATE_KEY_FILE"):
        deployment_proof.start_release(
            release.release_manifest,
            _runner=lambda _argv: pytest.fail("runner must not be called"),
        )


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("BIND_ADDRESS", "0.0.0.0", "loopback"),  # noqa: S104 - rejection fixture
        ("TLS_PORT", "0", "between 1 and 65535"),
        ("MAX_MODEL_LEN", "131073", "MAX_MODEL_LEN"),
        ("MAX_NUM_SEQS", "0", "MAX_NUM_SEQS"),
        ("MAX_NUM_BATCHED_TOKENS", "0", "MAX_NUM_BATCHED_TOKENS"),
        ("MAX_LORA_RANK", "8", "below the release adapter rank"),
        ("TENSOR_PARALLEL_SIZE", "65", "TENSOR_PARALLEL_SIZE"),
        ("GPU_MEMORY_UTILIZATION", "1.01", "GPU_MEMORY_UTILIZATION"),
        ("GATEWAY_CPUS", "0", "GATEWAY_CPUS"),
        ("VLLM_MEMORY", "2T", "VLLM_MEMORY"),
        ("PROMETHEUS_RETENTION", "10m", "between 1 hour and 31 days"),
        ("PROMETHEUS_RETENTION_SIZE", "0", "PROMETHEUS_RETENTION_SIZE"),
        ("SECRET_GROUP_ID", "0", "SECRET_GROUP_ID"),
    ],
)
def test_release_rejects_unsafe_or_unbounded_operational_env(
    tmp_path: Path,
    key: str,
    value: str,
    message: str,
) -> None:
    release = _release_bundle(
        tmp_path / "release",
        slug="current",
        weight_bytes=b"approved-adapter",
    )
    _set_release_env(release, key, value)

    with pytest.raises(ValueError, match=message):
        deployment_proof.validate_release_manifest(release.release_manifest)


def test_rollback_rejects_adapter_mismatch_env_mismatch_and_symlinks(tmp_path: Path) -> None:
    current = _release_bundle(tmp_path / "current", slug="current", weight_bytes=b"adapter")
    previous = _release_bundle(
        tmp_path / "previous",
        slug="previous",
        weight_bytes=b"previous",
    )
    wrong_adapter = tmp_path / "wrong-adapter"
    wrong_adapter.mkdir()
    current.env.write_text(
        current.env.read_text(encoding="utf-8").replace(
            f"ADAPTER_PATH={current.adapter}",
            f"ADAPTER_PATH={wrong_adapter}",
        ),
        encoding="utf-8",
    )
    _update_release(current, env_sha256=sha256_file(current.env))

    with pytest.raises(ValueError, match="ADAPTER_PATH"):
        build_rollback_plan(current.release_manifest, previous.release_manifest)

    current = _release_bundle(tmp_path / "current-2", slug="current-2", weight_bytes=b"adapter")
    (current.adapter / "adapter_model.safetensors").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="adapter directory SHA-256 mismatch"):
        build_rollback_plan(current.release_manifest, previous.release_manifest)

    link = previous.adapter / "linked-weight.safetensors"
    link.symlink_to(previous.adapter / "adapter_model.safetensors")
    with pytest.raises(ValueError, match="symlink"):
        adapter_inventory_sha256(previous.adapter)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("parity_report_sha256", "a" * 64, "repeated-character"),
        ("parity_report_file", "replace-with-passing-parity-report.json", "placeholder"),
    ],
)
def test_release_rejects_digest_and_path_placeholders(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    current = _release_bundle(tmp_path / "current", slug="current", weight_bytes=b"current")
    previous = _release_bundle(tmp_path / "previous", slug="previous", weight_bytes=b"previous")
    _update_release(current, **{field: value})

    with pytest.raises(ValueError, match=message):
        build_rollback_plan(current.release_manifest, previous.release_manifest)


@pytest.mark.parametrize(
    ("qualified", "dataset_tier", "message"),
    [
        (True, "portfolio_smoke", "qualification_candidate"),
        (False, "qualification_candidate", "not qualified"),
    ],
)
def test_release_rejects_smoke_or_unqualified_dataset_evidence(
    tmp_path: Path,
    qualified: bool,
    dataset_tier: str,
    message: str,
) -> None:
    current = _release_bundle(tmp_path / "current", slug="current", weight_bytes=b"current")
    previous = _release_bundle(tmp_path / "previous", slug="previous", weight_bytes=b"previous")
    payload = _read_json(current.qualification_report)
    payload["qualified"] = qualified
    payload["dataset_tier"] = dataset_tier
    _write_json(current.qualification_report, payload)
    _update_release(
        current,
        qualification_report_sha256=sha256_file(current.qualification_report),
    )

    with pytest.raises(ValueError, match=message):
        build_rollback_plan(current.release_manifest, previous.release_manifest)


def test_release_rejects_legacy_self_asserted_review_evidence(tmp_path: Path) -> None:
    release = _release_bundle(tmp_path / "release", slug="current", weight_bytes=b"current")
    payload = _read_json(release.qualification_report)
    payload["schema_version"] = "1.1"
    _write_json(release.qualification_report, payload)
    _update_release(
        release,
        qualification_report_sha256=sha256_file(release.qualification_report),
    )

    with pytest.raises(ValueError, match=r"requires v1\.2 reviewer-packet evidence"):
        deployment_proof.validate_release_manifest(release.release_manifest)


def test_release_rejects_duplicate_reviewer_packet_identity(tmp_path: Path) -> None:
    release = _release_bundle(tmp_path / "release", slug="current", weight_bytes=b"current")
    payload = _read_json(release.qualification_report)
    packet_hashes = payload["reviewer_packet_sha256"]
    assert isinstance(packet_hashes, list)
    packet_hashes[1] = packet_hashes[0]
    _write_json(release.qualification_report, payload)
    _update_release(
        release,
        qualification_report_sha256=sha256_file(release.qualification_report),
    )

    with pytest.raises(ValueError, match="two distinct reviewer packet identities"):
        deployment_proof.validate_release_manifest(release.release_manifest)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "unexpected"])
def test_release_requires_the_complete_unique_qualification_policy_set(
    tmp_path: Path,
    mutation: str,
) -> None:
    release = _release_bundle(tmp_path / "release", slug="current", weight_bytes=b"current")
    payload = _read_json(release.qualification_report)
    decisions = payload["decisions"]
    assert isinstance(decisions, list)
    if mutation == "missing":
        decisions.pop()
    elif mutation == "duplicate":
        decisions[-1] = dict(decisions[0])
    else:
        decision = decisions[-1]
        assert isinstance(decision, dict)
        decision["policy"] = "invented_quality_gate"
    _write_json(release.qualification_report, payload)
    _update_release(
        release,
        qualification_report_sha256=sha256_file(release.qualification_report),
    )

    with pytest.raises(ValueError, match="qualification decisions"):
        deployment_proof.validate_release_manifest(release.release_manifest)


@pytest.mark.parametrize("surface", ["candidate", "baseline", "parity"])
def test_release_binds_every_quality_surface_to_the_qualified_held_out_ids(
    tmp_path: Path,
    surface: str,
) -> None:
    release = _release_bundle(tmp_path / "release", slug="current", weight_bytes=b"current")
    if surface == "candidate":
        payload = _read_json(release.evaluation_report)
        results = payload["results"]
        assert isinstance(results, list)
        result = results[0]
        assert isinstance(result, dict)
        result["id"] = "DIFFERENT-CANDIDATE-ID"
        _sync_candidate_evaluation(release, payload)
    elif surface == "baseline":
        payload = _read_json(release.baseline_report)
        results = payload["results"]
        assert isinstance(results, list)
        result = results[0]
        assert isinstance(result, dict)
        result["id"] = "DIFFERENT-BASELINE-ID"
        _sync_baseline_evaluation(release, payload)
    else:
        payload = _read_json(release.parity_report)
        ordered_ids = payload["ordered_ids"]
        assert isinstance(ordered_ids, list)
        ordered_ids[0] = "DIFFERENT-PARITY-ID"
        _sync_parity(release, payload)

    with pytest.raises(ValueError, match="qualified held-out cohort"):
        deployment_proof.validate_release_manifest(release.release_manifest)


def test_release_rejects_dirty_training_and_incomplete_adapter_inventory(tmp_path: Path) -> None:
    current = _release_bundle(tmp_path / "current", slug="current", weight_bytes=b"current")
    previous = _release_bundle(tmp_path / "previous", slug="previous", weight_bytes=b"previous")
    training = _read_json(current.training_manifest)
    training["git_dirty"] = True
    _write_json(current.training_manifest, training)
    _update_release(current, training_manifest_sha256=sha256_file(current.training_manifest))

    with pytest.raises(ValueError, match="clean Git"):
        build_rollback_plan(current.release_manifest, previous.release_manifest)

    current = _release_bundle(tmp_path / "current-2", slug="current-2", weight_bytes=b"current")
    (current.adapter / "unrecorded-tokenizer.json").write_text("{}\n", encoding="utf-8")
    adapter_digest = adapter_inventory_sha256(current.adapter)
    release_payload = _read_json(current.release_manifest)
    recorded_adapter_digest = release_payload["adapter_sha256"]
    assert isinstance(recorded_adapter_digest, str)
    current.env.write_text(
        current.env.read_text(encoding="utf-8").replace(
            recorded_adapter_digest,
            adapter_digest,
        ),
        encoding="utf-8",
    )
    _update_release(
        current,
        adapter_sha256=adapter_digest,
        env_sha256=sha256_file(current.env),
    )
    with pytest.raises(ValueError, match="artifact inventory"):
        build_rollback_plan(current.release_manifest, previous.release_manifest)


@pytest.mark.parametrize(
    ("defect", "expected"),
    [
        ("method_mismatch", "method differs"),
        ("missing_preflight", "requires hardware preflight"),
        ("cpu_report", "observed CUDA"),
        ("cpu_execution", "executed on CUDA"),
        ("non_linux", "Linux"),
        ("bf16_unsupported", "bfloat16"),
        ("unknown_compute", "compute capability"),
        ("nan_compute", "compute capability"),
        ("infinite_compute", "compute capability"),
        ("negative_infinite_compute", "compute capability"),
        ("low_compute", "compute capability"),
        ("missing_cuda_version", "CUDA version"),
        ("missing_torch_version", "Torch version"),
        ("preflight_error", "compatible hardware preflight"),
        ("compatibility_lie", "compatibility claim"),
        ("parameter_mismatch", "parameter count"),
        ("forced_cpu_config", "forces CPU"),
        ("missing_optimizer_steps", "positive optimizer step"),
        ("zero_optimizer_steps", "positive optimizer step"),
        ("boolean_optimizer_steps", "positive optimizer step"),
        ("string_optimizer_steps", "positive optimizer step"),
        ("missing_peak", "peak accelerator memory"),
        ("zero_peak", "peak accelerator memory"),
    ],
)
def test_qlora_release_requires_real_cuda_execution_evidence(
    tmp_path: Path,
    defect: str,
    expected: str,
) -> None:
    release = _release_bundle(tmp_path / defect, slug=f"current-{defect}", weight_bytes=b"x")
    training = _read_json(release.training_manifest)
    preflight = training["hardware_preflight"]
    assert isinstance(preflight, dict)
    report = preflight["report"]
    assert isinstance(report, dict)
    config = training["config"]
    assert isinstance(config, dict)

    if defect == "method_mismatch":
        training["method"] = "lora"
    elif defect == "missing_preflight":
        training["hardware_preflight"] = None
    elif defect == "cpu_report":
        report["accelerator"] = "cpu"
    elif defect == "cpu_execution":
        preflight["execution_accelerator"] = "cpu"
    elif defect == "non_linux":
        report["platform_system"] = "Darwin"
    elif defect == "bf16_unsupported":
        report["supports_bfloat16"] = False
    elif defect == "unknown_compute":
        report["compute_capability"] = None
    elif defect == "nan_compute":
        report["compute_capability"] = "nan"
    elif defect == "infinite_compute":
        report["compute_capability"] = "inf"
    elif defect == "negative_infinite_compute":
        report["compute_capability"] = "-inf"
    elif defect == "low_compute":
        report["compute_capability"] = "5.0"
    elif defect == "missing_cuda_version":
        report["cuda_version"] = None
    elif defect == "missing_torch_version":
        report["torch_version"] = None
    elif defect == "preflight_error":
        findings = preflight["findings"]
        assert isinstance(findings, list)
        findings.append(
            {
                "level": "error",
                "code": "test_incompatible",
                "message": "Test-only incompatible hardware.",
                "remediation": None,
            }
        )
        preflight["compatible"] = False
    elif defect == "compatibility_lie":
        preflight["compatible"] = False
    elif defect == "parameter_mismatch":
        preflight["model_parameters_b"] = 8.0
    elif defect == "forced_cpu_config":
        training_config = config["training"]
        assert isinstance(training_config, dict)
        training_config["use_cpu"] = True
        training["config_sha256"] = sha256_bytes(config)
    elif defect == "missing_optimizer_steps":
        metrics = training["metrics"]
        assert isinstance(metrics, dict)
        metrics.pop("optimizer_steps")
    elif defect == "zero_optimizer_steps":
        metrics = training["metrics"]
        assert isinstance(metrics, dict)
        metrics["optimizer_steps"] = 0
    elif defect == "boolean_optimizer_steps":
        metrics = training["metrics"]
        assert isinstance(metrics, dict)
        metrics["optimizer_steps"] = True
    elif defect == "string_optimizer_steps":
        metrics = training["metrics"]
        assert isinstance(metrics, dict)
        metrics["optimizer_steps"] = "1"
    elif defect == "missing_peak":
        training["peak_accelerator_memory_mb"] = None
    else:
        training["peak_accelerator_memory_mb"] = 0.0

    _write_json(release.training_manifest, training)
    _update_release(
        release,
        training_manifest_sha256=sha256_file(release.training_manifest),
    )

    with pytest.raises(ValueError, match=expected):
        deployment_proof.validate_release_manifest(release.release_manifest)


@pytest.mark.parametrize("evidence", ["evaluation", "parity"])
def test_release_rejects_failed_evaluation_or_parity(
    tmp_path: Path,
    evidence: str,
) -> None:
    current = _release_bundle(tmp_path / "current", slug="current", weight_bytes=b"current")
    previous = _release_bundle(tmp_path / "previous", slug="previous", weight_bytes=b"previous")
    if evidence == "evaluation":
        payload = _read_json(current.evaluation_manifest)
        payload["candidate_passed"] = False
        _write_json(current.evaluation_manifest, payload)
        _update_release(
            current,
            evaluation_manifest_sha256=sha256_file(current.evaluation_manifest),
        )
    else:
        payload = _read_json(current.parity_report)
        payload["passed"] = False
        thresholds = payload["thresholds"]
        assert isinstance(thresholds, list)
        threshold = thresholds[0]
        assert isinstance(threshold, dict)
        threshold["passed"] = False
        _write_json(current.parity_report, payload)
        _update_release(current, parity_report_sha256=sha256_file(current.parity_report))

    with pytest.raises(ValueError, match="pass"):
        build_rollback_plan(current.release_manifest, previous.release_manifest)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("absolute_missing", "absolute thresholds"),
        ("absolute_duplicate", "duplicate"),
        ("absolute_minimum", "differs from the training config"),
        ("non_regression_missing", "non-regression thresholds"),
        ("non_regression_delta", "differs from report summaries"),
    ],
)
def test_release_requires_exact_internally_consistent_evaluation_gates(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    release = _release_bundle(tmp_path / "release", slug="current", weight_bytes=b"current")
    if mutation.startswith("absolute"):
        report = _read_json(release.evaluation_report)
        thresholds = report["thresholds"]
        assert isinstance(thresholds, list)
        if mutation == "absolute_missing":
            thresholds.pop()
        elif mutation == "absolute_duplicate":
            thresholds[-1] = dict(thresholds[0])
        else:
            threshold = thresholds[0]
            assert isinstance(threshold, dict)
            threshold["minimum"] = 0.1
        _sync_candidate_evaluation(release, report)
    else:
        manifest = _read_json(release.evaluation_manifest)
        result = manifest["result"]
        assert isinstance(result, dict)
        comparison = result["comparison"]
        assert isinstance(comparison, dict)
        non_regression = comparison["non_regression"]
        assert isinstance(non_regression, list)
        if mutation == "non_regression_missing":
            non_regression.pop()
        else:
            threshold = non_regression[0]
            assert isinstance(threshold, dict)
            threshold["value"] = 0.25
            deltas = comparison["metric_deltas"]
            assert isinstance(deltas, dict)
            deltas[str(threshold["metric"])] = 0.25
        _write_json(release.evaluation_manifest, manifest)
        _update_release(
            release,
            evaluation_manifest_sha256=sha256_file(release.evaluation_manifest),
        )

    with pytest.raises(ValueError, match=message):
        deployment_proof.validate_release_manifest(release.release_manifest)


@pytest.mark.parametrize("mutation", ["missing_embedded_baseline", "missing_baseline_file"])
def test_release_requires_a_bound_completed_baseline_report(
    tmp_path: Path,
    mutation: str,
) -> None:
    release = _release_bundle(tmp_path / "release", slug="current", weight_bytes=b"current")
    if mutation == "missing_embedded_baseline":
        manifest = _read_json(release.evaluation_manifest)
        result = manifest["result"]
        assert isinstance(result, dict)
        result.pop("baseline")
        _write_json(release.evaluation_manifest, manifest)
        _update_release(
            release,
            evaluation_manifest_sha256=sha256_file(release.evaluation_manifest),
        )
    else:
        release.baseline_report.unlink()

    with pytest.raises(ValueError, match="baseline"):
        deployment_proof.validate_release_manifest(release.release_manifest)


def test_release_rejects_evaluation_and_parity_lineage_mismatch(tmp_path: Path) -> None:
    current = _release_bundle(tmp_path / "current", slug="current", weight_bytes=b"current")
    previous = _release_bundle(tmp_path / "previous", slug="previous", weight_bytes=b"previous")
    parity = _read_json(current.parity_report)
    parity["training_config_sha256"] = _digest("wrong-training-config")
    _write_json(current.parity_report, parity)
    _update_release(current, parity_report_sha256=sha256_file(current.parity_report))

    with pytest.raises(ValueError, match="parity training config lineage"):
        build_rollback_plan(current.release_manifest, previous.release_manifest)

    current = _release_bundle(tmp_path / "current-2", slug="current-2", weight_bytes=b"current")
    evaluation = _read_json(current.evaluation_report)
    provenance = evaluation["provenance"]
    assert isinstance(provenance, dict)
    provenance["dataset_manifest_sha256"] = _digest("wrong-dataset")
    _write_json(current.evaluation_report, evaluation)
    evaluation_report_sha = sha256_file(current.evaluation_report)
    evaluation_manifest = _read_json(current.evaluation_manifest)
    artifact_sha256 = evaluation_manifest["artifact_sha256"]
    result = evaluation_manifest["result"]
    assert isinstance(artifact_sha256, dict)
    assert isinstance(result, dict)
    candidate = result["candidate"]
    assert isinstance(candidate, dict)
    artifact_sha256["candidate/evaluation-report.json"] = evaluation_report_sha
    candidate["report"] = evaluation
    _write_json(current.evaluation_manifest, evaluation_manifest)
    _update_release(
        current,
        evaluation_manifest_sha256=sha256_file(current.evaluation_manifest),
        evaluation_report_sha256=evaluation_report_sha,
    )

    with pytest.raises(ValueError, match="prepared-manifest hash differs"):
        build_rollback_plan(current.release_manifest, previous.release_manifest)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("threshold_missing", "parity thresholds"),
        ("threshold_duplicate", "duplicate"),
        ("threshold_required", "require exact 1.0"),
        ("threshold_value", "did not achieve exact 1.0"),
        ("merged_provenance_missing", "complete safe-merge provenance"),
        ("merged_adapter_config", "merged adapter config lineage"),
        ("merged_adapter_weight", "merged adapter weight lineage"),
        ("merged_artifact_sentinel", "repeated-character"),
        ("raw_diagnostic", "recomputed immutable sidecars"),
        ("merge_dtype", "merge precision"),
    ],
)
def test_release_requires_exact_parity_gates_and_complete_merge_binding(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    release = _release_bundle(tmp_path / "release", slug="current", weight_bytes=b"current")
    payload = _read_json(release.parity_report)
    thresholds = payload["thresholds"]
    merged = payload["merged_predictions"]
    metrics = payload["metrics"]
    assert isinstance(thresholds, list)
    assert isinstance(merged, dict)
    assert isinstance(metrics, dict)
    if mutation == "threshold_missing":
        thresholds.pop()
    elif mutation == "threshold_duplicate":
        thresholds[-1] = dict(thresholds[0])
    elif mutation == "threshold_required":
        threshold = thresholds[0]
        assert isinstance(threshold, dict)
        threshold["required"] = 0.9
    elif mutation == "threshold_value":
        threshold = thresholds[0]
        assert isinstance(threshold, dict)
        metric = str(threshold["metric"])
        threshold["value"] = 0.9
        threshold["passed"] = False
        metrics[metric] = 0.9
        payload["passed"] = False
    elif mutation == "merged_provenance_missing":
        merged["merge_provenance_sha256"] = None
    elif mutation == "merged_adapter_config":
        merged["merged_adapter_config_sha256"] = _digest("wrong-adapter-config")
    elif mutation == "merged_adapter_weight":
        merged["merged_adapter_weight_sha256"] = [_digest("wrong-adapter-weight")]
    elif mutation == "raw_diagnostic":
        metrics["raw_prediction_exact_rate"] = 0.5
    elif mutation == "merge_dtype":
        payload["merge_dtype"] = "float32"
    else:
        merged["merged_artifact_sha256"] = {"config.json": "a" * 64}
    _sync_parity(release, payload)

    with pytest.raises(ValueError, match=message):
        deployment_proof.validate_release_manifest(release.release_manifest)


def test_release_rejects_symlinked_and_changing_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = _release_bundle(tmp_path / "current", slug="current", weight_bytes=b"current")
    previous = _release_bundle(tmp_path / "previous", slug="previous", weight_bytes=b"previous")
    linked_parent = tmp_path / "linked-evidence"
    linked_parent.symlink_to(current.parity_report.parent, target_is_directory=True)
    _update_release(
        current,
        parity_report_file=str(linked_parent / current.parity_report.name),
    )
    with pytest.raises(ValueError, match="symlink components"):
        build_rollback_plan(current.release_manifest, previous.release_manifest)

    current = _release_bundle(tmp_path / "current-2", slug="current-2", weight_bytes=b"current")
    original = deployment_proof._read_regular_snapshot
    changed = False

    def racing_snapshot(path: Path, *, label: str) -> Any:
        nonlocal changed
        snapshot = original(path, label=label)
        if Path(path) == current.qualification_report and not changed:
            current.qualification_report.write_bytes(snapshot.payload + b" ")
            changed = True
        return snapshot

    monkeypatch.setattr(deployment_proof, "_read_regular_snapshot", racing_snapshot)
    with pytest.raises(ValueError, match="changed before rollback plan"):
        build_rollback_plan(current.release_manifest, previous.release_manifest)


def test_proof_reports_are_immutable(tmp_path: Path) -> None:
    path = tmp_path / "proof.json"
    first = {"proof": "one"}
    assert write_proof_report(path, first) == path
    assert write_proof_report(path, first) == path
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_proof_report(path, {"proof": "two"})

    target = tmp_path / "target.json"
    target.write_bytes(path.read_bytes())
    link = tmp_path / "proof-link.json"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="symlink components"):
        write_proof_report(link, first)

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink components"):
        write_proof_report(linked_parent / "redirected.json", first)
    assert not (real_parent / "redirected.json").exists()


def test_concurrent_proof_writers_cannot_overwrite_each_other(tmp_path: Path) -> None:
    path = tmp_path / "proof.json"
    writers = 12
    barrier = threading.Barrier(writers)

    def write(writer: int) -> tuple[str, int]:
        barrier.wait()
        try:
            write_proof_report(
                path,
                {"writer": writer, "padding": "x" * 250_000},
            )
        except FileExistsError:
            return "rejected", writer
        return "created", writer

    with ThreadPoolExecutor(max_workers=writers) as executor:
        results = list(executor.map(write, range(writers)))

    created = [writer for status, writer in results if status == "created"]
    assert len(created) == 1
    assert json.loads(path.read_text(encoding="utf-8"))["writer"] == created[0]
