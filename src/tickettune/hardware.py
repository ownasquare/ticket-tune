"""Truthful, lazy hardware inspection and training-method preflight."""

from __future__ import annotations

import importlib
import math
import os
import platform
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field

from tickettune.config import FineTuneConfig

Accelerator = Literal["cuda", "mps", "cpu"]
TrainingMethod = Literal["full", "lora", "qlora"]
FindingLevel = Literal["info", "warning", "error"]


class HardwareReport(BaseModel):
    """Portable snapshot of the resources visible to the current process."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    accelerator: Accelerator
    total_memory_gb: float = Field(gt=0.0)
    device_name: str = "unknown"
    accelerator_count: int = Field(default=1, ge=1)
    torch_version: str | None = None
    cuda_version: str | None = None
    compute_capability: str | None = None
    supports_bfloat16: bool = False
    platform_system: str = Field(default_factory=platform.system)
    platform_machine: str = Field(default_factory=platform.machine)


class PreflightFinding(BaseModel):
    """One actionable compatibility observation."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    level: FindingLevel
    code: str = Field(pattern=r"^[a-z][a-z0-9_]+$")
    message: str
    remediation: str | None = None


class HardwarePreflight(BaseModel):
    """Hardware report plus the lower-bound estimate and compatibility result."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    report: HardwareReport
    execution_accelerator: Accelerator
    method: TrainingMethod
    model_parameters_b: float = Field(gt=0.0)
    estimated_training_memory_gb: float = Field(gt=0.0)
    findings: list[PreflightFinding]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def compatible(self) -> bool:
        return not any(finding.level == "error" for finding in self.findings)


class HardwareCompatibilityError(RuntimeError):
    """Raised when an execution path requires a compatible preflight."""


def _system_memory_gb() -> float:
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        page_count = int(os.sysconf("SC_PHYS_PAGES"))
        return round(page_size * page_count / 1024**3, 1)
    except (AttributeError, OSError, TypeError, ValueError):
        # The fallback is deliberately explicit rather than pretending that
        # memory is unbounded on an unsupported inspection platform.
        return 1.0


def _safe_call(callable_value: Any, default: Any) -> Any:
    try:
        return callable_value()
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return default


def probe_hardware(torch_module: Any | None = None) -> HardwareReport:
    """Inspect PyTorch lazily; tests can inject a deterministic fake module."""

    torch_value = torch_module
    if torch_value is None:
        try:
            torch_value = importlib.import_module("torch")
        except ImportError:
            return HardwareReport(
                accelerator="cpu",
                total_memory_gb=_system_memory_gb(),
                device_name=platform.processor() or platform.machine() or "CPU",
            )

    torch_version = str(getattr(torch_value, "__version__", "unknown"))
    cuda = getattr(torch_value, "cuda", None)
    if cuda is not None and _safe_call(cuda.is_available, False):
        properties = cuda.get_device_properties(0)
        capability = _safe_call(lambda: cuda.get_device_capability(0), None)
        capability_text = (
            f"{capability[0]}.{capability[1]}"
            if isinstance(capability, (tuple, list)) and len(capability) == 2
            else None
        )
        return HardwareReport(
            accelerator="cuda",
            total_memory_gb=round(float(properties.total_memory) / 1024**3, 1),
            device_name=str(getattr(properties, "name", "CUDA device")),
            accelerator_count=int(_safe_call(cuda.device_count, 1)),
            torch_version=torch_version,
            cuda_version=str(getattr(getattr(torch_value, "version", None), "cuda", None))
            if getattr(getattr(torch_value, "version", None), "cuda", None) is not None
            else None,
            compute_capability=capability_text,
            supports_bfloat16=bool(_safe_call(cuda.is_bf16_supported, False)),
        )

    backends = getattr(torch_value, "backends", None)
    mps = getattr(backends, "mps", None)
    if mps is not None and _safe_call(mps.is_available, False):
        return HardwareReport(
            accelerator="mps",
            total_memory_gb=_system_memory_gb(),
            device_name=f"Apple {platform.machine()} unified memory",
            torch_version=torch_version,
            supports_bfloat16=False,
        )

    return HardwareReport(
        accelerator="cpu",
        total_memory_gb=_system_memory_gb(),
        device_name=platform.processor() or platform.machine() or "CPU",
        torch_version=torch_version,
    )


def estimate_training_memory_gb(parameters_b: float, method: TrainingMethod) -> float:
    """Estimate a conservative lower bound before activations and allocator spikes."""

    if parameters_b <= 0:
        raise ValueError("parameters_b must be greater than zero")
    if method not in {"full", "lora", "qlora"}:
        raise ValueError(f"unsupported training method: {method}")
    bytes_per_parameter = {"full": 16.0, "lora": 6.0, "qlora": 0.8}[method]
    return round(parameters_b * bytes_per_parameter, 1)


def _compute_capability_value(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        capability = float(value)
    except ValueError:
        return None
    return capability if math.isfinite(capability) else None


def validate_training_compatibility(
    report: HardwareReport,
    *,
    method: TrainingMethod,
    model_parameters_b: float,
    requires_bfloat16: bool = False,
) -> list[PreflightFinding]:
    """Return fail-closed findings for the requested method and model size."""

    estimate = estimate_training_memory_gb(model_parameters_b, method)
    findings: list[PreflightFinding] = [
        PreflightFinding(
            level="info",
            code="memory_estimate",
            message=(
                f"Estimated {estimate:.1f} GB lower-bound memory for {model_parameters_b:g}B "
                f"parameters with {method}; activation and allocator spikes require extra headroom."
            ),
        )
    ]

    if method == "qlora" and report.accelerator != "cuda":
        findings.append(
            PreflightFinding(
                level="error",
                code="qlora_requires_cuda",
                message=(
                    "TicketTune QLoRA requires a supported NVIDIA CUDA device; "
                    f"the detected accelerator is {report.accelerator}."
                ),
                remediation=(
                    "Use the LoRA profile locally, or run QLoRA on Linux with an NVIDIA GPU. "
                    "bitsandbytes Apple Silicon support is experimental and is not accepted "
                    "by this reproducibility preflight."
                ),
            )
        )

    capability = _compute_capability_value(report.compute_capability)
    if method == "qlora" and report.accelerator == "cuda":
        if capability is None:
            findings.append(
                PreflightFinding(
                    level="error",
                    code="compute_capability_unknown",
                    message="CUDA compute capability could not be verified for NF4 quantization.",
                    remediation=(
                        "Confirm an NVIDIA compute capability of at least 6.0 before training."
                    ),
                )
            )
        elif capability < 6.0:
            findings.append(
                PreflightFinding(
                    level="error",
                    code="compute_capability_too_low",
                    message=(
                        f"CUDA compute capability {capability:g} is below the bitsandbytes "
                        "NF4/FP4 requirement of 6.0."
                    ),
                    remediation="Use a Pascal-generation or newer NVIDIA GPU.",
                )
            )

    if requires_bfloat16 and not report.supports_bfloat16:
        findings.append(
            PreflightFinding(
                level="error",
                code="bfloat16_unsupported",
                message=(
                    "The selected profile requires bfloat16, but the device did not report support."
                ),
                remediation="Use an fp16 profile or a CUDA device with bfloat16 support.",
            )
        )

    if method == "lora" and report.accelerator == "mps":
        findings.append(
            PreflightFinding(
                level="info",
                code="mps_lora_supported",
                message="Standard LoRA can use PyTorch MPS and Apple unified memory.",
            )
        )
    elif method in {"lora", "full"} and report.accelerator == "cpu":
        findings.append(
            PreflightFinding(
                level="warning",
                code="cpu_training_slow",
                message=(
                    "CPU training is supported for correctness but will be substantially slower."
                ),
                remediation="Use a small smoke model or move the run to CUDA/MPS hardware.",
            )
        )

    required_with_headroom = round(estimate * 1.2, 1)
    if report.total_memory_gb < required_with_headroom:
        findings.append(
            PreflightFinding(
                level="error",
                code="insufficient_memory",
                message=(
                    f"Detected {report.total_memory_gb:.1f} GB, below the "
                    f"{required_with_headroom:.1f} GB "
                    "minimum including 20% headroom."
                ),
                remediation=(
                    "Choose a smaller model, shorten sequences/batches, or use a larger device."
                ),
            )
        )

    if (
        report.accelerator == "cuda"
        and method in {"lora", "qlora"}
        and not any(finding.level == "error" for finding in findings)
    ):
        findings.append(
            PreflightFinding(
                level="info",
                code="cuda_adapter_training",
                message="CUDA adapter training is compatible with the selected method.",
            )
        )
    return findings


def validate_deployment_compatibility(
    report: HardwareReport, *, target: Literal["vllm", "ollama"]
) -> list[PreflightFinding]:
    """Keep deployment-plan support separate from runtime support."""

    if target == "vllm":
        findings: list[PreflightFinding] = []
        if report.platform_system != "Linux":
            findings.append(
                PreflightFinding(
                    level="error",
                    code="vllm_requires_linux",
                    message="The supported TicketTune vLLM runtime path requires Linux.",
                    remediation=(
                        "Render the plan locally, then run the container on a Linux server."
                    ),
                )
            )
        if report.accelerator != "cuda":
            findings.append(
                PreflightFinding(
                    level="error",
                    code="vllm_requires_cuda",
                    message="The TicketTune vLLM profile requires an NVIDIA CUDA accelerator.",
                    remediation="Use Ollama locally or deploy vLLM to a Linux CUDA host.",
                )
            )
        if not findings:
            findings.append(
                PreflightFinding(
                    level="info",
                    code="vllm_runtime_supported",
                    message="Linux and CUDA requirements for the TicketTune vLLM profile are met.",
                )
            )
        return findings
    return [
        PreflightFinding(
            level="info",
            code="ollama_requires_runtime_check",
            message=(
                "Hardware is not enough to prove Ollama availability; verify the installed "
                "runtime separately."
            ),
        )
    ]


def run_preflight(
    config: FineTuneConfig, *, report: HardwareReport | None = None
) -> HardwarePreflight:
    """Evaluate the configured method without importing any training library."""

    observed = report or probe_hardware()
    execution_accelerator: Accelerator = "cpu" if config.training.use_cpu else observed.accelerator
    effective = observed
    if config.training.use_cpu:
        effective = observed.model_copy(
            update={
                "accelerator": "cpu",
                "device_name": "CPU forced by training.use_cpu",
                "accelerator_count": 1,
                "cuda_version": None,
                "compute_capability": None,
                "supports_bfloat16": False,
            }
        )
    findings = validate_training_compatibility(
        effective,
        method=config.lora.method,
        model_parameters_b=config.model.parameters_b,
        requires_bfloat16=config.training.bf16,
    )
    if config.training.use_cpu:
        findings.insert(
            0,
            PreflightFinding(
                level="info",
                code="cpu_forced",
                message=(
                    "This profile forces Transformers and Accelerate to use CPU even when "
                    "another accelerator is visible."
                ),
            ),
        )
    return HardwarePreflight(
        report=observed,
        execution_accelerator=execution_accelerator,
        method=config.lora.method,
        model_parameters_b=config.model.parameters_b,
        estimated_training_memory_gb=estimate_training_memory_gb(
            config.model.parameters_b, config.lora.method
        ),
        findings=findings,
    )


def require_compatible(preflight: HardwarePreflight) -> None:
    """Raise a concise exception for execution commands that may not continue."""

    errors = [finding.message for finding in preflight.findings if finding.level == "error"]
    if errors:
        raise HardwareCompatibilityError("; ".join(errors))
