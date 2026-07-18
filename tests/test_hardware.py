from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import tickettune.hardware as hardware_module
from tickettune.config import load_config
from tickettune.hardware import (
    HardwareCompatibilityError,
    HardwareReport,
    estimate_training_memory_gb,
    probe_hardware,
    require_compatible,
    run_preflight,
    validate_deployment_compatibility,
    validate_training_compatibility,
)

ROOT = Path(__file__).resolve().parents[1]


class _UnavailableCuda:
    @staticmethod
    def is_available() -> bool:
        return False


class _AvailableMps:
    @staticmethod
    def is_available() -> bool:
        return True


class _UnavailableMps:
    @staticmethod
    def is_available() -> bool:
        return False


class _AvailableCuda:
    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def get_device_properties(_index: int) -> SimpleNamespace:
        return SimpleNamespace(total_memory=24 * 1024**3, name="Test CUDA GPU")

    @staticmethod
    def get_device_capability(_index: int) -> tuple[int, int]:
        return (8, 0)

    @staticmethod
    def device_count() -> int:
        return 2

    @staticmethod
    def is_bf16_supported() -> bool:
        return True


def test_probe_hardware_accepts_an_injected_mps_probe() -> None:
    fake_torch = SimpleNamespace(
        __version__="test",
        cuda=_UnavailableCuda(),
        backends=SimpleNamespace(mps=_AvailableMps()),
    )

    report = probe_hardware(fake_torch)

    assert report.accelerator == "mps"
    assert report.torch_version == "test"
    assert report.total_memory_gb > 0


def test_hardware_report_rejects_non_finite_memory() -> None:
    with pytest.raises(ValidationError, match="finite number"):
        HardwareReport(accelerator="cpu", total_memory_gb=float("inf"))


def test_probe_hardware_reports_injected_cuda_details() -> None:
    fake_torch = SimpleNamespace(
        __version__="test",
        version=SimpleNamespace(cuda="12.8"),
        cuda=_AvailableCuda(),
        backends=SimpleNamespace(mps=_UnavailableMps()),
    )

    report = probe_hardware(fake_torch)

    assert report.accelerator == "cuda"
    assert report.device_name == "Test CUDA GPU"
    assert report.accelerator_count == 2
    assert report.cuda_version == "12.8"
    assert report.compute_capability == "8.0"
    assert report.supports_bfloat16 is True


def test_probe_hardware_falls_back_when_torch_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing_torch(_name: str) -> None:
        raise ImportError

    monkeypatch.setattr(hardware_module.importlib, "import_module", missing_torch)

    assert probe_hardware().accelerator == "cpu"


def test_probe_hardware_handles_probe_errors_and_memory_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenCuda:
        @staticmethod
        def is_available() -> bool:
            raise RuntimeError("probe failed")

    def broken_sysconf(_name: str) -> int:
        raise OSError("not available")

    monkeypatch.setattr(hardware_module.os, "sysconf", broken_sysconf)
    fake_torch = SimpleNamespace(
        __version__="test",
        cuda=BrokenCuda(),
        backends=SimpleNamespace(mps=_UnavailableMps()),
    )

    report = probe_hardware(fake_torch)

    assert report.accelerator == "cpu"
    assert report.total_memory_gb == 1.0


def test_qlora_rejects_mps() -> None:
    report = HardwareReport(accelerator="mps", total_memory_gb=36)

    findings = validate_training_compatibility(report, method="qlora", model_parameters_b=7)

    assert any(item.level == "error" and "CUDA" in item.message for item in findings)


def test_qlora_accepts_sufficient_supported_cuda() -> None:
    report = HardwareReport(
        accelerator="cuda",
        total_memory_gb=24,
        compute_capability="8.0",
        supports_bfloat16=True,
        platform_system="Linux",
    )

    findings = validate_training_compatibility(report, method="qlora", model_parameters_b=7)

    assert not any(item.level == "error" for item in findings)
    assert any(item.code == "cuda_adapter_training" for item in findings)


def test_qlora_rejects_old_compute_capability() -> None:
    report = HardwareReport(accelerator="cuda", total_memory_gb=24, compute_capability="5.2")

    findings = validate_training_compatibility(report, method="qlora", model_parameters_b=7)

    assert any(item.code == "compute_capability_too_low" for item in findings)


def test_qlora_fails_closed_when_compute_capability_is_unknown() -> None:
    report = HardwareReport(accelerator="cuda", total_memory_gb=24)

    findings = validate_training_compatibility(report, method="qlora", model_parameters_b=7)

    assert any(
        item.level == "error" and item.code == "compute_capability_unknown" for item in findings
    )
    assert not any(item.code == "cuda_adapter_training" for item in findings)


@pytest.mark.parametrize("compute_capability", ["not-a-number", "nan", "inf", "-inf"])
def test_invalid_compute_capability_is_treated_as_unknown(compute_capability: str) -> None:
    report = HardwareReport(
        accelerator="cuda", total_memory_gb=24, compute_capability=compute_capability
    )

    findings = validate_training_compatibility(report, method="qlora", model_parameters_b=7)

    assert any(item.code == "compute_capability_unknown" for item in findings)


def test_bfloat16_requirement_is_checked_explicitly() -> None:
    report = HardwareReport(
        accelerator="cuda",
        total_memory_gb=24,
        compute_capability="8.0",
        supports_bfloat16=False,
    )

    findings = validate_training_compatibility(
        report,
        method="qlora",
        model_parameters_b=7,
        requires_bfloat16=True,
    )

    assert any(item.code == "bfloat16_unsupported" for item in findings)


def test_memory_estimates_are_method_specific() -> None:
    assert estimate_training_memory_gb(7, "full") == 112.0
    assert estimate_training_memory_gb(7, "lora") == 42.0
    assert estimate_training_memory_gb(7, "qlora") == 5.6

    with pytest.raises(ValueError, match="unsupported training method"):
        estimate_training_memory_gb(7, "invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="greater than zero"):
        estimate_training_memory_gb(0, "lora")


def test_lora_mps_and_low_memory_findings_are_actionable() -> None:
    report = HardwareReport(accelerator="mps", total_memory_gb=4)

    findings = validate_training_compatibility(report, method="lora", model_parameters_b=1)

    assert any(item.code == "mps_lora_supported" for item in findings)
    assert any(item.code == "insufficient_memory" for item in findings)


def test_preflight_uses_configured_model_size_and_method() -> None:
    config = load_config(ROOT / "configs" / "smoke.yaml")
    report = HardwareReport(accelerator="cpu", total_memory_gb=64)

    preflight = run_preflight(config, report=report)

    assert preflight.method == "lora"
    assert preflight.execution_accelerator == "cpu"
    assert preflight.model_parameters_b == 0.5
    assert preflight.compatible is True
    assert any(item.code == "cpu_training_slow" for item in preflight.findings)


def test_cpu_profile_preserves_observed_hardware_and_forces_execution() -> None:
    config = load_config(ROOT / "configs" / "cpu-smoke.yaml")
    report = HardwareReport(accelerator="mps", total_memory_gb=36)

    preflight = run_preflight(config, report=report)

    assert preflight.report.accelerator == "mps"
    assert preflight.execution_accelerator == "cpu"
    assert preflight.compatible is True
    assert any(item.code == "cpu_forced" for item in preflight.findings)
    assert any(item.code == "cpu_training_slow" for item in preflight.findings)


def test_require_compatible_raises_for_errors() -> None:
    config = load_config(ROOT / "configs" / "qwen-7b-qlora.yaml")
    preflight = run_preflight(config, report=HardwareReport(accelerator="mps", total_memory_gb=36))

    with pytest.raises(HardwareCompatibilityError, match="requires a supported NVIDIA CUDA"):
        require_compatible(preflight)


def test_vllm_runtime_is_not_claimed_on_macos_mps() -> None:
    report = HardwareReport(accelerator="mps", total_memory_gb=36, platform_system="Darwin")

    findings = validate_deployment_compatibility(report, target="vllm")

    assert {item.code for item in findings if item.level == "error"} == {
        "vllm_requires_linux",
        "vllm_requires_cuda",
    }


def test_supported_vllm_and_ollama_findings_keep_runtime_boundaries() -> None:
    report = HardwareReport(accelerator="cuda", total_memory_gb=24, platform_system="Linux")

    assert [item.code for item in validate_deployment_compatibility(report, target="vllm")] == [
        "vllm_runtime_supported"
    ]
    assert [item.code for item in validate_deployment_compatibility(report, target="ollama")] == [
        "ollama_requires_runtime_check"
    ]


def test_run_preflight_can_probe_and_compatible_result_does_not_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(ROOT / "configs" / "smoke.yaml")
    report = HardwareReport(accelerator="cpu", total_memory_gb=64)
    monkeypatch.setattr(hardware_module, "probe_hardware", lambda: report)

    preflight = run_preflight(config)

    require_compatible(preflight)
    assert preflight.compatible is True
