from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

import tickettune.cuda_rehearsal as rehearsal_module
import tickettune.training as training_module
from tickettune.config import load_config
from tickettune.cuda_rehearsal import (
    CudaRehearsalGate,
    CudaRehearsalReport,
    run_cuda_rehearsal,
)
from tickettune.hardware import HardwareReport
from tickettune.run_manifest import RunManifest

ROOT = Path(__file__).resolve().parents[1]
QLORA_CONFIG = ROOT / "configs" / "qwen-7b-qlora.yaml"
LORA_CONFIG = ROOT / "configs" / "smoke.yaml"
CREATED_AT = datetime(2026, 7, 18, 12, 0, tzinfo=UTC)


def _cpu_report() -> HardwareReport:
    return HardwareReport(
        accelerator="cpu",
        total_memory_gb=32.0,
        device_name="test CPU",
        torch_version="2.8.0",
        platform_system="Linux",
        platform_machine="x86_64",
    )


def _cuda_report() -> HardwareReport:
    return HardwareReport(
        accelerator="cuda",
        total_memory_gb=24.0,
        device_name="test NVIDIA GPU",
        torch_version="2.8.0",
        cuda_version="12.8",
        compute_capability="8.0",
        supports_bfloat16=True,
        platform_system="Linux",
        platform_machine="x86_64",
    )


def _gate(report: CudaRehearsalReport, code: str) -> CudaRehearsalGate:
    return next(gate for gate in report.gates if gate.code == code)


def test_cpu_rehearsal_is_static_only_and_release_ineligible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(QLORA_CONFIG)

    def forbidden_training_import() -> None:
        raise AssertionError("the rehearsal must not import or load the training stack")

    monkeypatch.setattr(training_module, "_load_training_libraries", forbidden_training_import)

    report = run_cuda_rehearsal(
        config,
        hardware_report=_cpu_report(),
        created_at=CREATED_AT,
    )

    assert report.static_contract_passed is True
    assert report.runtime_status == "blocked_no_cuda"
    assert report.executed_cuda is False
    assert report.model_weights_loaded is False
    assert report.optimizer_steps == 0
    assert report.adapter_artifacts_written is False
    assert report.run_manifest_written is False
    assert report.release_eligible is False
    assert report.release_status == "ineligible_rehearsal"
    assert report.training_plan.allow_download is False
    assert _gate(report, "observed_cuda_runtime").state == "blocked_external"


def test_compatible_observed_cuda_only_marks_host_eligible_for_a_real_run() -> None:
    report = run_cuda_rehearsal(
        load_config(QLORA_CONFIG),
        hardware_report=_cuda_report(),
        created_at=CREATED_AT,
    )

    assert report.runtime_status == "eligible_for_real_cuda_run"
    assert _gate(report, "observed_cuda_runtime").state == "passed"
    assert report.executed_cuda is False
    assert report.optimizer_steps == 0
    assert report.release_eligible is False


@pytest.mark.parametrize("compute_capability", ["nan", "inf", "-inf"])
def test_non_finite_compute_capability_blocks_cuda_host_eligibility(
    compute_capability: str,
) -> None:
    hardware_report = _cuda_report().model_copy(update={"compute_capability": compute_capability})

    report = run_cuda_rehearsal(
        load_config(QLORA_CONFIG),
        hardware_report=hardware_report,
        created_at=CREATED_AT,
    )

    assert report.runtime_status == "blocked_no_cuda"
    runtime_gate = _gate(report, "observed_cuda_runtime")
    assert runtime_gate.state == "blocked_external"
    assert "compute capability" in runtime_gate.message


def test_missing_release_qualification_is_an_external_blocker_not_a_static_failure() -> None:
    report = run_cuda_rehearsal(
        load_config(QLORA_CONFIG),
        hardware_report=_cpu_report(),
        created_at=CREATED_AT,
    )

    qualification = _gate(report, "release_dataset_qualification")
    assert qualification.state == "blocked_external"
    assert qualification.scope == "external_evidence"
    assert report.static_contract_passed is True


def test_rehearsal_refuses_a_lora_profile() -> None:
    with pytest.raises(ValueError, match="requires a QLoRA profile"):
        run_cuda_rehearsal(
            load_config(LORA_CONFIG),
            hardware_report=_cpu_report(),
            created_at=CREATED_AT,
        )


def test_static_contract_failure_is_separate_from_runtime_blockers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(QLORA_CONFIG)
    original = rehearsal_module.build_training_plan

    def malformed_plan(*args: object, **kwargs: object) -> object:
        plan = original(*args, **kwargs)
        return plan.model_copy(update={"quantization": None})

    monkeypatch.setattr(rehearsal_module, "build_training_plan", malformed_plan)
    report = run_cuda_rehearsal(
        config,
        hardware_report=_cpu_report(),
        created_at=CREATED_AT,
    )

    assert report.static_contract_passed is False
    assert _gate(report, "nf4_double_quantization").state == "failed"
    assert report.runtime_status == "blocked_no_cuda"


def test_rehearsal_report_is_immutable_and_refuses_symlink_outputs(tmp_path: Path) -> None:
    config = load_config(QLORA_CONFIG)
    output = tmp_path / "rehearsal.json"
    report = run_cuda_rehearsal(
        config,
        output_path=output,
        hardware_report=_cpu_report(),
        created_at=CREATED_AT,
    )
    assert output.is_file()
    assert report.release_eligible is False

    with pytest.raises(FileExistsError, match="refusing to overwrite immutable proof report"):
        run_cuda_rehearsal(
            config,
            output_path=output,
            hardware_report=_cpu_report(),
            created_at=CREATED_AT + timedelta(seconds=1),
        )

    target = tmp_path / "target.json"
    target.write_text("{}\n", encoding="utf-8")
    symlink = tmp_path / "symlink.json"
    symlink.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        run_cuda_rehearsal(
            config,
            output_path=symlink,
            hardware_report=_cpu_report(),
            created_at=CREATED_AT,
        )


def test_rehearsal_report_cannot_validate_as_a_training_run_manifest() -> None:
    report = run_cuda_rehearsal(
        load_config(QLORA_CONFIG),
        hardware_report=_cpu_report(),
        created_at=CREATED_AT,
    )

    with pytest.raises(ValidationError):
        RunManifest.model_validate(report.model_dump(mode="json"))
