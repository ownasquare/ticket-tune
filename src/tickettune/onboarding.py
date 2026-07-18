"""Self-contained onboarding workflows for first-time TicketTune users."""

from __future__ import annotations

import tempfile
from contextlib import suppress
from dataclasses import dataclass
from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from .config import load_config
from .data import load_examples, prepare_dataset
from .evaluation import evaluate_predictions
from .hardware import run_preflight
from .training import run_training

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True, slots=True)
class InitResult:
    """Files created by :func:`init_project`."""

    destination: Path
    config_path: Path
    dataset_path: Path
    predictions_path: Path
    readme_path: Path
    gitignore_path: Path
    created_files: tuple[Path, ...]


class QuickstartResult(BaseModel):
    """Compact, machine-readable result for the offline first-success workflow."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    workspace: Path | None
    config_path: Path | None
    model_name: str
    method: str
    source_examples: int = Field(ge=1)
    split_counts: dict[str, int]
    accelerator: str
    hardware_compatible: bool
    training_plan_ready: bool
    evaluation_passed: bool
    evaluation_examples: int = Field(ge=1)
    manifest_path: Path | None
    evaluation_report_path: Path | None
    artifacts_retained: bool
    proof_boundary: str


def starter_project_root() -> Traversable:
    """Return the importlib-resources root bundled into the wheel."""

    return resources.files("tickettune.starter")


def _resource(relative_path: str, checkout_fallback: Path) -> Traversable:
    candidate = starter_project_root().joinpath(*relative_path.split("/"))
    if candidate.is_file():
        return candidate
    if checkout_fallback.is_file():
        return checkout_fallback
    raise FileNotFoundError(f"TicketTune starter resource is missing: {relative_path}")


def starter_config_path() -> Traversable:
    """Return the starter YAML profile without assuming a repository checkout."""

    return _resource("configs/tickettune.yaml", _REPOSITORY_ROOT / "configs" / "smoke.yaml")


def starter_dataset_path() -> Traversable:
    """Return the bundled 56-row synthetic support-ticket dataset."""

    return _resource(
        "data/raw/support_tickets.jsonl",
        _REPOSITORY_ROOT / "data" / "raw" / "support_tickets.jsonl",
    )


def starter_predictions_path() -> Traversable:
    """Return passing fixture predictions used only by the offline quickstart."""

    return _resource(
        "predictions/pass.jsonl",
        _REPOSITORY_ROOT / "tests" / "fixtures" / "predictions_pass.jsonl",
    )


def starter_readme_path() -> Traversable:
    """Return the short README copied into generated projects."""

    return _resource(
        "PROJECT_README.md",
        Path(__file__).with_name("starter") / "PROJECT_README.md",
    )


def starter_gitignore_path() -> Traversable:
    """Return the ignore rules copied into generated projects."""

    return _resource(
        "gitignore.txt",
        Path(__file__).with_name("starter") / "gitignore.txt",
    )


def _starter_payloads() -> tuple[tuple[Path, bytes], ...]:
    return (
        (Path("configs/tickettune.yaml"), starter_config_path().read_bytes()),
        (Path("data/raw/support_tickets.jsonl"), starter_dataset_path().read_bytes()),
        (Path("predictions/pass.jsonl"), starter_predictions_path().read_bytes()),
        (Path("README.md"), starter_readme_path().read_bytes()),
        (Path(".gitignore"), starter_gitignore_path().read_bytes()),
    )


def init_project(destination: Path) -> InitResult:
    """Create a starter project without following or overwriting an existing path."""

    requested = destination.expanduser()
    if requested.is_symlink():
        raise ValueError(f"starter destination must not be a symlink: {requested}")

    existed = requested.exists()
    if existed:
        if not requested.is_dir():
            raise ValueError(f"starter destination must be a directory: {requested}")
        if any(requested.iterdir()):
            raise ValueError(f"starter destination must be empty: {requested}")

    payloads = _starter_payloads()
    created_files: list[Path] = []
    created_directories: set[Path] = set()
    try:
        requested.mkdir(parents=True, exist_ok=existed)
        destination_root = requested.resolve()
        for relative_path, payload in payloads:
            target = destination_root / relative_path
            missing_parents = [
                parent
                for parent in (target.parent, *target.parent.parents)
                if parent.is_relative_to(destination_root) and not parent.exists()
            ]
            target.parent.mkdir(parents=True, exist_ok=True)
            created_directories.update(missing_parents)
            with target.open("xb") as handle:
                handle.write(payload)
            created_files.append(target)
    except Exception:
        for path in reversed(created_files):
            path.unlink(missing_ok=True)
        for path in sorted(created_directories, key=lambda item: len(item.parts), reverse=True):
            with suppress(OSError):
                path.rmdir()
        if not existed:
            with suppress(OSError):
                requested.rmdir()
        raise

    by_name = {path.relative_to(destination_root).as_posix(): path for path in created_files}
    return InitResult(
        destination=destination_root,
        config_path=by_name["configs/tickettune.yaml"],
        dataset_path=by_name["data/raw/support_tickets.jsonl"],
        predictions_path=by_name["predictions/pass.jsonl"],
        readme_path=by_name["README.md"],
        gitignore_path=by_name[".gitignore"],
        created_files=tuple(created_files),
    )


def _run_quickstart_in(workspace: Path) -> QuickstartResult:
    initialized = init_project(workspace.expanduser())
    config = load_config(initialized.config_path)
    examples = load_examples(config.data.source_path)
    prepared = prepare_dataset(
        config.data.source_path,
        config.data.processed_dir,
        seed=config.seed,
        splits=config.data.splits,
    )
    preflight = run_preflight(config)
    training = run_training(
        config,
        dry_run=True,
        allow_download=False,
        resume_from_checkpoint=None,
        hardware_preflight=preflight,
    )
    evaluation = evaluate_predictions(
        config,
        initialized.predictions_path,
        raise_on_failure=True,
    )

    return QuickstartResult(
        workspace=initialized.destination,
        config_path=initialized.config_path,
        model_name=config.model.name_or_path,
        method=config.lora.method,
        source_examples=len(examples),
        split_counts={name: artifact.count for name, artifact in prepared.splits.items()},
        accelerator=preflight.execution_accelerator,
        hardware_compatible=preflight.compatible,
        training_plan_ready=(
            training.plan.dataset_manifest_status == "verified"
            and not training.plan.missing_datasets
            and not training.executed
        ),
        evaluation_passed=evaluation.passed,
        evaluation_examples=evaluation.report.summary.examples,
        manifest_path=prepared.manifest_path,
        evaluation_report_path=Path(evaluation.json_report_path),
        artifacts_retained=True,
        proof_boundary=(
            "offline_demo_only: dataset preparation, hardware inspection, training-plan "
            "validation, and fixture scoring passed; No model weights were downloaded or trained"
        ),
    )


def run_quickstart(workspace: Path | None = None) -> QuickstartResult:
    """Run the complete lightweight workflow without downloading or training a model."""

    if workspace is not None:
        return _run_quickstart_in(workspace)

    with tempfile.TemporaryDirectory(prefix="tickettune-quickstart-") as temporary:
        result = _run_quickstart_in(Path(temporary))
    return result.model_copy(
        update={
            "workspace": None,
            "config_path": None,
            "manifest_path": None,
            "evaluation_report_path": None,
            "artifacts_retained": False,
            "proof_boundary": (
                f"{result.proof_boundary}; temporary demo artifacts were cleaned up"
            ),
        }
    )
