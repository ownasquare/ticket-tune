"""CPU-safe rehearsal of the declared CUDA QLoRA contract.

The rehearsal deliberately stops before importing the training stack or loading
model weights.  It proves that a configuration describes the expected QLoRA
shape and records whether the *observed* host is eligible for a later real CUDA
run; it is never training or release evidence.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field

from .config import FineTuneConfig
from .deployment_proof import write_proof_report
from .hardware import HardwarePreflight, HardwareReport, run_preflight
from .run_manifest import sha256_bytes
from .training import TrainingPlan, build_training_plan

GateState = Literal["passed", "failed", "blocked_external", "not_applicable"]
EvidenceSource = Literal[
    "observed",
    "declared_requirement",
    "fixture_contract",
    "executed_cpu_surrogate",
]
GateScope = Literal["static_contract", "observed_runtime", "external_evidence"]
RuntimeStatus = Literal["blocked_no_cuda", "eligible_for_real_cuda_run"]

_PINNED_REVISION = re.compile(r"^[0-9a-fA-F]{40}$")
_EXPECTED_QUANTIZATION: dict[str, object] = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True,
}


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class CudaTargetRequirements(_FrozenModel):
    """Declared requirements a later real QLoRA execution must satisfy."""

    platform_system: Literal["Linux"] = "Linux"
    accelerator: Literal["cuda"] = "cuda"
    training_method: Literal["qlora"] = "qlora"
    quantization: Literal["nf4"] = "nf4"
    double_quantization: Literal[True] = True
    minimum_compute_capability: float = Field(default=6.0, ge=6.0)
    requires_bfloat16: bool
    minimum_memory_with_headroom_gb: float = Field(gt=0.0)


class CudaRehearsalGate(_FrozenModel):
    """One static, observed-runtime, or external-evidence rehearsal check."""

    code: str = Field(pattern=r"^[a-z][a-z0-9_]+$")
    state: GateState
    scope: GateScope
    evidence_source: EvidenceSource
    message: str
    remediation: str | None = None


class CudaRehearsalReport(_FrozenModel):
    """Truth-preserving CUDA contract report that cannot represent a run."""

    schema_version: Literal["1.0"] = "1.0"
    created_at: datetime
    project_name: str
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_name_or_path: str
    model_revision: str
    training_plan: TrainingPlan
    hardware_preflight: HardwarePreflight
    target_requirements: CudaTargetRequirements
    gates: tuple[CudaRehearsalGate, ...]
    executed_cuda: Literal[False] = False
    model_weights_loaded: Literal[False] = False
    optimizer_steps: Literal[0] = 0
    adapter_artifacts_written: Literal[False] = False
    run_manifest_written: Literal[False] = False
    release_eligible: Literal[False] = False
    release_status: Literal["ineligible_rehearsal"] = "ineligible_rehearsal"
    proof_boundary: Literal[
        "static QLoRA contract and observed hardware only; no CUDA execution or model loading"
    ] = "static QLoRA contract and observed hardware only; no CUDA execution or model loading"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def static_contract_passed(self) -> bool:
        """Fail only for malformed declarations, never for absent external resources."""

        return not any(
            gate.scope == "static_contract" and gate.state == "failed" for gate in self.gates
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def runtime_status(self) -> RuntimeStatus:
        """Classify only the hardware currently observed by the preflight."""

        preflight = self.hardware_preflight
        report = preflight.report
        eligible = (
            report.accelerator == "cuda"
            and preflight.execution_accelerator == "cuda"
            and report.platform_system == self.target_requirements.platform_system
            and preflight.compatible
        )
        return "eligible_for_real_cuda_run" if eligible else "blocked_no_cuda"


def _gate(
    code: str,
    *,
    passed: bool,
    scope: GateScope,
    evidence_source: EvidenceSource,
    success: str,
    failure: str,
    blocked_external: bool = False,
    remediation: str | None = None,
) -> CudaRehearsalGate:
    if passed:
        state: GateState = "passed"
        message = success
    else:
        state = "blocked_external" if blocked_external else "failed"
        message = failure
    return CudaRehearsalGate(
        code=code,
        state=state,
        scope=scope,
        evidence_source=evidence_source,
        message=message,
        remediation=remediation if not passed else None,
    )


def _static_gates(plan: TrainingPlan) -> list[CudaRehearsalGate]:
    quantization = plan.quantization or {}
    quantization_matches = (
        all(quantization.get(name) == expected for name, expected in _EXPECTED_QUANTIZATION.items())
        and quantization.get("bnb_4bit_compute_dtype") == plan.precision
    )
    targets_valid = plan.target_modules == "all-linear" or bool(plan.target_modules)
    return [
        _gate(
            "pinned_model_revision",
            passed=_PINNED_REVISION.fullmatch(plan.model_revision or "") is not None,
            scope="static_contract",
            evidence_source="declared_requirement",
            success="The remote model revision is pinned to a full commit SHA.",
            failure="The remote model revision is not pinned to a full 40-character commit SHA.",
            remediation="Pin model.revision to the reviewed Hugging Face commit SHA.",
        ),
        _gate(
            "nf4_double_quantization",
            passed=quantization_matches,
            scope="static_contract",
            evidence_source="declared_requirement",
            success="The plan declares four-bit NF4 with double quantization.",
            failure="The plan does not declare the canonical NF4 double-quantization contract.",
            remediation="Use QLoRA with NF4, double quantization, and matching compute precision.",
        ),
        _gate(
            "adapter_target_modules",
            passed=targets_valid,
            scope="static_contract",
            evidence_source="declared_requirement",
            success="The plan declares at least one adapter target module.",
            failure="The plan has no adapter target modules.",
            remediation="Declare all-linear or an explicit non-empty target-module list.",
        ),
        _gate(
            "completion_only_loss",
            passed=plan.completion_only_loss is True,
            scope="static_contract",
            evidence_source="declared_requirement",
            success="Only assistant completions are declared as training targets.",
            failure="The plan would train on prompt tokens.",
            remediation="Set completion_only_loss=true.",
        ),
        _gate(
            "training_precision",
            passed=plan.precision in {"bf16", "fp16"},
            scope="static_contract",
            evidence_source="declared_requirement",
            success=f"The QLoRA compute precision is {plan.precision}.",
            failure=f"The QLoRA compute precision {plan.precision} is not bf16 or fp16.",
            remediation="Use bf16 on supported GPUs or an explicitly reviewed fp16 profile.",
        ),
    ]


def _external_gates(plan: TrainingPlan) -> list[CudaRehearsalGate]:
    dataset_ready = plan.dataset_manifest_status == "verified" and not plan.missing_datasets
    qualification_ready = plan.qualification.status == "qualified"
    qualification_error = plan.qualification.error or (
        "The profile does not require independently reviewed release qualification."
        if plan.qualification.status == "not_required"
        else f"Qualification status is {plan.qualification.status}."
    )
    return [
        _gate(
            "prepared_dataset",
            passed=dataset_ready,
            scope="external_evidence",
            evidence_source="observed",
            success="The prepared dataset manifest and split bytes verify.",
            failure=plan.dataset_manifest_error
            or "Prepared dataset bytes are missing or do not verify.",
            blocked_external=True,
            remediation="Run data prepare for this exact config before a real CUDA run.",
        ),
        _gate(
            "release_dataset_qualification",
            passed=qualification_ready,
            scope="external_evidence",
            evidence_source="observed",
            success="The exact prepared holdout is bound to independent review evidence.",
            failure=qualification_error,
            blocked_external=True,
            remediation=(
                "Use the quality profile and complete its independent dataset-review manifest."
            ),
        ),
    ]


def _runtime_gate(preflight: HardwarePreflight) -> CudaRehearsalGate:
    report = preflight.report
    eligible = (
        report.accelerator == "cuda"
        and preflight.execution_accelerator == "cuda"
        and report.platform_system == "Linux"
        and preflight.compatible
    )
    errors = [finding.message for finding in preflight.findings if finding.level == "error"]
    failure = "; ".join(errors) or (
        "A real CUDA rehearsal target requires an observed compatible Linux CUDA host."
    )
    return _gate(
        "observed_cuda_runtime",
        passed=eligible,
        scope="observed_runtime",
        evidence_source="observed",
        success="This host is eligible to start a separately authorized real CUDA run.",
        failure=failure,
        blocked_external=True,
        remediation="Move the unchanged config and prepared data to a compatible Linux CUDA host.",
    )


def run_cuda_rehearsal(
    config: FineTuneConfig,
    *,
    output_path: Path | None = None,
    hardware_report: HardwareReport | None = None,
    created_at: datetime | None = None,
) -> CudaRehearsalReport:
    """Build and optionally persist a CUDA contract report without training.

    This function intentionally calls only the download-free training planner and
    hardware preflight.  It never imports the optional training libraries, calls
    ``run_training``, writes adapter bytes, or creates a ``RunManifest``.
    """

    if config.lora.method != "qlora":
        raise ValueError("CUDA contract rehearsal requires a QLoRA profile")

    plan = build_training_plan(config, allow_download=False)
    preflight = run_preflight(config, report=hardware_report)
    requirements = CudaTargetRequirements(
        requires_bfloat16=config.training.bf16,
        minimum_memory_with_headroom_gb=round(
            preflight.estimated_training_memory_gb * 1.2,
            1,
        ),
    )
    gates = (*_static_gates(plan), *_external_gates(plan), _runtime_gate(preflight))
    report = CudaRehearsalReport(
        created_at=(created_at or datetime.now(UTC)).astimezone(UTC),
        project_name=config.project_name,
        config_sha256=sha256_bytes(config),
        model_name_or_path=config.model.name_or_path,
        model_revision=config.model.revision,
        training_plan=plan,
        hardware_preflight=preflight,
        target_requirements=requirements,
        gates=gates,
    )
    if output_path is not None:
        write_proof_report(output_path, report)
    return report
