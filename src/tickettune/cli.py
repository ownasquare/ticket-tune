"""Command-line orchestration for the TicketTune pipeline."""

from __future__ import annotations

import dataclasses
import json
import shlex
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Any

import typer
from pydantic import BaseModel

from tickettune import __version__
from tickettune.config import FineTuneConfig, load_config
from tickettune.cuda_rehearsal import CudaRehearsalReport, run_cuda_rehearsal
from tickettune.data import load_examples, prepare_dataset, verify_prepared_dataset
from tickettune.deployment_proof import (
    build_rollback_plan,
    run_load_test,
    run_readback,
    start_release,
    validate_release_manifest,
    write_proof_report,
)
from tickettune.evaluation import evaluate_predictions, run_model_evaluation
from tickettune.export import (
    build_merge_plan,
    build_ollama_export_plan,
    build_vllm_plan,
    materialize_ollama_plan,
    merge_adapter,
    run_argv,
    write_ollama_export_provenance,
)
from tickettune.hardware import HardwarePreflight, require_compatible, run_preflight
from tickettune.onboarding import InitResult, QuickstartResult, init_project, run_quickstart
from tickettune.parity import compare_prediction_files, verify_live_parity
from tickettune.qualification import load_review_manifest, qualify_dataset
from tickettune.review_packets import (
    DatasetReviewManifestV12,
    bind_review_manifest_references,
    write_draft_review_scaffold,
)
from tickettune.synthetic import write_qualified_synthetic_corpus
from tickettune.training import TrainingResult, run_training

app = typer.Typer(
    name="tickettune",
    help="Fine-tune and run a small model for structured support-ticket triage.",
    no_args_is_help=True,
    rich_markup_mode="markdown",
)
config_app = typer.Typer(help="Inspect validated configuration.", no_args_is_help=True)
data_app = typer.Typer(help="Validate and prepare the training dataset.", no_args_is_help=True)
export_app = typer.Typer(help="Prepare portable deployment artifacts.", no_args_is_help=True)
serve_app = typer.Typer(help="Run or inspect a model-serving command.", no_args_is_help=True)
deploy_app = typer.Typer(
    help="Collect redacted deployment and rollback proof.", no_args_is_help=True
)
qualify_app = typer.Typer(help="Apply fail-closed qualification gates.", no_args_is_help=True)
parity_app = typer.Typer(help="Verify adapter and safe-merge parity.", no_args_is_help=True)
rehearse_app = typer.Typer(
    help="Rehearse hardware-specific contracts safely.", no_args_is_help=True
)
app.add_typer(config_app, name="config", hidden=True)
app.add_typer(data_app, name="data")
app.add_typer(export_app, name="export")
app.add_typer(serve_app, name="serve")
app.add_typer(deploy_app, name="deploy", hidden=True)
app.add_typer(qualify_app, name="qualify", hidden=True)
app.add_typer(parity_app, name="parity", hidden=True)
app.add_typer(rehearse_app, name="rehearse", hidden=True)

ConfigOption = Annotated[
    Path,
    typer.Option(
        "--config",
        "-c",
        help=(
            "Validated TicketTune YAML profile. Defaults are checkout-relative; pass "
            "--config when using the installed CLI outside this repository."
        ),
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
]
JsonOption = Annotated[
    bool,
    typer.Option("--json", help="Emit compact machine-readable JSON."),
]
DetailsOption = Annotated[
    bool,
    typer.Option(
        "--details",
        help="Show the complete human-readable result instead of the short summary.",
    ),
]
LocalSmokeOption = Annotated[
    bool,
    typer.Option(
        "--allow-unqualified-local-smoke",
        help=(
            "Permit missing qualification lineage only for local smoke tests and fixtures; "
            "the serialized plan is never release evidence."
        ),
    ),
]


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"tickettune {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show the installed version and exit.",
        ),
    ] = False,
) -> None:
    """Run the reproducible support-triage workflow."""


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(item) for item in value]
    return value


def _emit(value: Any, *, compact: bool, err: bool = False) -> None:
    payload = json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":") if compact else None,
        indent=None if compact else 2,
    )
    typer.echo(payload, err=err)


@contextmanager
def _friendly_errors() -> Iterator[None]:
    try:
        yield
    except typer.Exit:
        raise
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=2) from exc


def _load(path: Path) -> FineTuneConfig:
    return load_config(path)


def _emit_init_summary(result: InitResult) -> None:
    typer.echo("✓ Starter project created")
    typer.echo(f"  Location: {result.destination}")
    typer.echo("  Includes: config, 56 synthetic tickets, and passing fixture predictions")
    typer.echo("Next:")
    typer.echo(f"  cd {shlex.quote(str(result.destination))}")
    typer.echo("  tickettune data prepare --config configs/tickettune.yaml")


def _emit_quickstart_summary(result: QuickstartResult) -> None:
    split_counts = result.split_counts
    typer.echo("TicketTune is ready")
    typer.echo(f"✓ Dataset: {result.source_examples} synthetic tickets validated")
    typer.echo(
        "✓ Splits: "
        f"{split_counts.get('train', 0)} train / "
        f"{split_counts.get('validation', 0)} validation / "
        f"{split_counts.get('test', 0)} test"
    )
    hardware_mark = "✓" if result.hardware_compatible else "△"
    typer.echo(
        f"{hardware_mark} Hardware: {result.accelerator} checked for {result.method.upper()}"
    )
    typer.echo("✓ Training plan: valid; no model download")
    typer.echo(f"✓ Evaluation contract: passed on {result.evaluation_examples} fixture predictions")
    if result.artifacts_retained:
        typer.echo(f"ⓘ Demo files: {result.workspace}")
    else:
        typer.echo("ⓘ Temporary demo files were cleaned up.")
    typer.echo("ⓘ No model weights were downloaded or trained.")
    typer.echo("Next: tickettune init my-support-model")


@app.command("init")
def initialize(
    destination: Annotated[
        Path,
        typer.Argument(help="New or empty directory for the self-contained starter project."),
    ],
    as_json: JsonOption = False,
) -> None:
    """Create a safe, self-contained support-triage starter project."""

    with _friendly_errors():
        result = init_project(destination)
        if as_json:
            _emit(result, compact=True)
        else:
            _emit_init_summary(result)


@app.command("quickstart")
def quickstart(
    workspace: Annotated[
        Path | None,
        typer.Option(
            "--workspace",
            help="Optional new or empty directory in which to keep the offline demo artifacts.",
        ),
    ] = None,
    details: DetailsOption = False,
    as_json: JsonOption = False,
) -> None:
    """Prove the local workflow without downloading or training a model."""

    with _friendly_errors():
        result = run_quickstart(workspace)
        if as_json or details:
            _emit(result, compact=as_json)
        else:
            _emit_quickstart_summary(result)


@app.command("advanced")
def advanced() -> None:
    """Show qualification, parity, release, and inspection commands."""

    typer.echo("Advanced TicketTune commands")
    typer.echo("  tickettune config --help     Inspect validated profiles")
    typer.echo("  tickettune qualify --help    Bind independent dataset review")
    typer.echo("  tickettune parity --help     Compare adapter and merged outputs")
    typer.echo("  tickettune rehearse --help   Verify CUDA contracts without claiming a run")
    typer.echo("  tickettune merge --help      Plan or create a safe merge")
    typer.echo("  tickettune deploy --help     Collect release and rollback proof")


def _emit_training_summary(
    preflight: HardwarePreflight,
    result: TrainingResult,
    *,
    config_path: Path,
) -> None:
    plan = result.plan
    if not result.executed:
        typer.echo("✓ Training plan is valid")
        typer.echo(f"  Model: {plan.model_name_or_path}")
        typer.echo(f"  Method: {plan.method} on {preflight.execution_accelerator}")
        typer.echo(
            "  Data: "
            f"{plan.dataset_counts.get('train', 0)} train / "
            f"{plan.dataset_counts.get('validation', 0)} validation"
        )
        typer.echo("ⓘ No model weights were loaded or changed.")
        typer.echo("Next: rerun without --dry-run and add --allow-download when ready.")
        return

    typer.echo("✓ Training completed")
    typer.echo(f"  Run: {result.run_id}")
    typer.echo(f"  Adapter: {result.adapter_path}")
    typer.echo(f"  Manifest: {result.manifest_path}")
    if "train_loss" in result.metrics:
        typer.echo(f"  Train loss: {result.metrics['train_loss']}")
    typer.echo("Next: compare the adapter with the base model:")
    typer.echo(
        "  uv run --no-sync tickettune evaluate "
        f"--config {config_path} --adapter {result.adapter_path} --compare-baseline"
    )


@config_app.command("show")
def show_config(
    config: ConfigOption = Path("configs/smoke.yaml"),
    as_json: JsonOption = False,
) -> None:
    """Load, validate, and print a fully resolved profile."""

    with _friendly_errors():
        _emit(_load(config), compact=as_json)


@data_app.command("validate")
def validate_data(
    config: ConfigOption = Path("configs/smoke.yaml"),
    as_json: JsonOption = False,
) -> None:
    """Validate source schema, IDs, PII placeholders, and duplicate content."""

    with _friendly_errors():
        resolved = _load(config)
        examples = load_examples(resolved.data.source_path)
        report = {
            "source_path": resolved.data.source_path,
            "examples": len(examples),
            "categories": dict(
                sorted(Counter(item.expected.category for item in examples).items())
            ),
            "priorities": dict(
                sorted(Counter(item.expected.priority for item in examples).items())
            ),
            "sentiments": dict(
                sorted(Counter(item.expected.sentiment for item in examples).items())
            ),
            "validation": "passed",
            "proof_boundary": "source_data_only",
        }
        _emit(report, compact=as_json)


@data_app.command("prepare")
def prepare_data(
    config: ConfigOption = Path("configs/smoke.yaml"),
    as_json: JsonOption = False,
) -> None:
    """Create deterministic TRL prompt/completion splits and a hash manifest."""

    with _friendly_errors():
        resolved = _load(config)
        result = prepare_dataset(
            resolved.data.source_path,
            resolved.data.processed_dir,
            seed=resolved.seed,
            splits=resolved.data.splits,
        )
        _emit(result, compact=as_json)


@data_app.command("generate-candidate")
def generate_candidate_data(
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            help="New JSONL path for the fixed 1,120-record synthetic review candidate.",
            resolve_path=True,
        ),
    ],
    seed: Annotated[
        int,
        typer.Option(
            "--seed",
            min=0,
            help="Reproducibility contract; the published candidate is fixed at seed 42.",
        ),
    ] = 42,
    as_json: JsonOption = False,
) -> None:
    """Create the deterministic, privacy-safe corpus that humans can review."""

    with _friendly_errors():
        if seed != 42:
            raise ValueError(
                "the published qualification candidate is fixed at seed 42; "
                "choose --seed 42 for byte-for-byte reproducibility"
            )
        artifact = write_qualified_synthetic_corpus(output)
        _emit(
            {
                "artifact": artifact,
                "seed": seed,
                "review_status": "pending_two_independent_human_reviews",
                "release_eligible": False,
                "proof_boundary": (
                    "deterministic_synthetic_candidate_only; "
                    "not_human_reviewed_or_release_qualified"
                ),
            },
            compact=as_json,
        )


@qualify_app.command("dataset")
def qualify_data(
    review_manifest: Annotated[
        Path,
        typer.Option(
            "--review-manifest",
            help="Strict human-review manifest bound to the configured source bytes.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    config: ConfigOption = Path("configs/qwen-7b-qlora-quality.yaml"),
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Create an immutable qualification-report JSON file."),
    ] = None,
    enforce: Annotated[
        bool,
        typer.Option("--enforce", help="Exit non-zero when any qualification policy fails."),
    ] = False,
    as_json: JsonOption = False,
) -> None:
    """Bind reviewed dataset evidence to the exact configured source."""

    with _friendly_errors():
        resolved = _load(config)
        report = qualify_dataset(
            resolved.data.source_path,
            review_manifest,
            enforce=False,
        )
        if output is not None:
            write_proof_report(output, report)
        _emit(report, compact=as_json)
        if enforce and not report.qualified:
            raise typer.Exit(code=1)


@qualify_app.command("scaffold-review")
def scaffold_review(
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            help="New private directory for two pending human-review packets.",
            resolve_path=True,
        ),
    ],
    config: ConfigOption = Path("configs/qwen-7b-qlora-quality.yaml"),
    as_json: JsonOption = False,
) -> None:
    """Create a fail-closed v1.2 review workspace from verified prepared data."""

    with _friendly_errors():
        resolved = _load(config)
        verification = verify_prepared_dataset(
            resolved.data.source_path,
            resolved.data.processed_dir,
            seed=resolved.seed,
            splits=resolved.data.splits,
            required_splits=("train", "validation", "test"),
        )
        ordered_record_ids = tuple(
            example.id for example in load_examples(resolved.data.source_path)
        )
        artifact = write_draft_review_scaffold(
            output_dir,
            source_sha256=verification.source_sha256,
            prepared_manifest_path=verification.manifest_path,
            prepared_manifest_sha256=verification.manifest_sha256,
            ordered_record_ids=ordered_record_ids,
            held_out_ids=verification.split_ids["test"],
        )
        _emit(artifact, compact=as_json)


@qualify_app.command("bind-review")
def bind_review(
    review_manifest: Annotated[
        Path,
        typer.Option(
            "--review-manifest",
            help="Edited v1.2 aggregate whose companion file hashes must be refreshed.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            help="New bound aggregate path; the edited input is never overwritten.",
            resolve_path=True,
        ),
    ],
    as_json: JsonOption = False,
) -> None:
    """Rehash edited review packets without approving or changing their decisions."""

    with _friendly_errors():
        manifest = load_review_manifest(review_manifest)
        if not isinstance(manifest, DatasetReviewManifestV12):
            raise ValueError("bind-review requires a v1.2 reviewer-packet manifest")
        rebound = bind_review_manifest_references(
            manifest,
            aggregate_path=review_manifest,
        )
        written = write_proof_report(output, rebound)
        _emit(
            {
                "output": written,
                "approval_status": rebound.approval_status,
                "review_date": rebound.review_date,
                "reviewer_packet_sha256": [
                    reference.sha256 for reference in rebound.reviewer_packets
                ],
                "release_eligible": False,
                "proof_boundary": (
                    "reference_hash_refresh_only; no_review_decision_or_approval_was_created"
                ),
            },
            compact=as_json,
        )


@app.command("doctor")
def doctor(
    config: ConfigOption = Path("configs/smoke.yaml"),
    strict: Annotated[
        bool,
        typer.Option(
            "--strict/--advisory",
            help="Exit non-zero when the selected training method is incompatible.",
        ),
    ] = True,
    as_json: JsonOption = False,
) -> None:
    """Inspect the actual accelerator and fail closed on incompatible methods."""

    with _friendly_errors():
        preflight = run_preflight(_load(config))
        _emit(preflight, compact=as_json)
        if strict and not preflight.compatible:
            raise typer.Exit(code=2)


@rehearse_app.command("cuda")
def rehearse_cuda(
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            help="New immutable CUDA contract rehearsal JSON file.",
            resolve_path=True,
        ),
    ],
    config: ConfigOption = Path("configs/qwen-7b-qlora.yaml"),
    enforce: Annotated[
        bool,
        typer.Option(
            "--enforce",
            help=(
                "Exit non-zero only when the declared static QLoRA contract fails; "
                "missing CUDA remains a truthful external blocker."
            ),
        ),
    ] = False,
    as_json: JsonOption = False,
) -> None:
    """Validate a CUDA QLoRA contract without loading weights or training."""

    with _friendly_errors():
        report: CudaRehearsalReport = run_cuda_rehearsal(
            _load(config),
            output_path=output,
        )
        _emit(report, compact=as_json)
        if enforce and not report.static_contract_passed:
            raise typer.Exit(code=1)


@app.command("train")
def train(
    config: ConfigOption = Path("configs/smoke.yaml"),
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Print the complete plan without loading model weights."),
    ] = False,
    allow_download: Annotated[
        bool,
        typer.Option("--allow-download", help="Permit an explicit Hugging Face model download."),
    ] = False,
    resume_from_checkpoint: Annotated[
        Path | None,
        typer.Option(
            "--resume-from-checkpoint",
            help="Resume from a compatible local Trainer checkpoint.",
            exists=True,
            file_okay=False,
            dir_okay=True,
        ),
    ] = None,
    details: DetailsOption = False,
    as_json: JsonOption = False,
) -> None:
    """Run TRL supervised fine-tuning with a PEFT LoRA or QLoRA adapter."""

    with _friendly_errors():
        resolved = _load(config)
        preflight = run_preflight(resolved)
        if not dry_run:
            require_compatible(preflight)
        result = run_training(
            resolved,
            dry_run=dry_run,
            allow_download=allow_download,
            resume_from_checkpoint=resume_from_checkpoint,
            hardware_preflight=preflight,
        )
        payload = {"hardware_preflight": preflight, "training": result}
        if as_json or details:
            _emit(payload, compact=as_json)
        else:
            _emit_training_summary(preflight, result, config_path=config)


@app.command("evaluate")
def evaluate(
    config: ConfigOption = Path("configs/smoke.yaml"),
    predictions: Annotated[
        Path | None,
        typer.Option(
            "--predictions",
            help="Score existing prediction JSONL instead of loading a model.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ] = None,
    adapter: Annotated[
        Path | None,
        typer.Option(
            "--adapter",
            help="PEFT adapter used for live generation.",
            exists=True,
            file_okay=False,
            dir_okay=True,
        ),
    ] = None,
    compare_baseline: Annotated[
        bool,
        typer.Option(
            "--compare-baseline",
            help="Generate the same held-out IDs with the base model.",
        ),
    ] = False,
    allow_download: Annotated[
        bool,
        typer.Option("--allow-download", help="Permit an explicit Hugging Face model download."),
    ] = False,
    allow_unverified_adapter: Annotated[
        bool,
        typer.Option(
            "--allow-unverified-adapter",
            help=(
                "Permit a naked adapter only for local fixture evaluation; disables release gating."
            ),
        ),
    ] = False,
    enforce_thresholds: Annotated[
        bool,
        typer.Option("--enforce-thresholds", help="Exit non-zero when configured gates fail."),
    ] = False,
    as_json: JsonOption = False,
) -> None:
    """Score fixture predictions or generate held-out baseline/adapter reports."""

    with _friendly_errors():
        resolved = _load(config)
        if predictions is not None:
            if (
                adapter is not None
                or compare_baseline
                or allow_download
                or allow_unverified_adapter
            ):
                raise ValueError(
                    "--predictions is fixture scoring only and cannot be combined with "
                    "--adapter, --compare-baseline, --allow-download, or "
                    "--allow-unverified-adapter"
                )
            artifacts = evaluate_predictions(
                resolved,
                predictions,
                raise_on_failure=enforce_thresholds,
            )
            _emit(artifacts, compact=as_json)
            return
        model_result = run_model_evaluation(
            resolved,
            adapter_path=adapter,
            compare_baseline=compare_baseline,
            allow_download=allow_download,
            allow_unverified_adapter=allow_unverified_adapter,
            enforce_thresholds=enforce_thresholds,
        )
        _emit(model_result, compact=as_json)


def _merge_dtype(config: FineTuneConfig) -> str:
    if config.training.bf16 or config.model.torch_dtype == "bfloat16":
        return "bfloat16"
    if config.training.fp16 or config.model.torch_dtype == "float16":
        return "float16"
    return "float32"


@app.command("merge", hidden=True)
def merge(
    adapter: Annotated[
        Path,
        typer.Option(
            "--adapter",
            help="Validated PEFT adapter directory.",
            exists=True,
            file_okay=False,
            dir_okay=True,
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            help="New merged Hugging Face model directory.",
            resolve_path=True,
        ),
    ],
    config: ConfigOption = Path("configs/smoke.yaml"),
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Validate and print the merge plan without loading weights.",
        ),
    ] = False,
    allow_download: Annotated[
        bool,
        typer.Option("--allow-download", help="Permit an explicit base-model download."),
    ] = False,
    allow_unqualified_local_smoke: LocalSmokeOption = False,
    as_json: JsonOption = False,
) -> None:
    """Safely merge an adapter into a pristine non-quantized base model."""

    with _friendly_errors():
        resolved = _load(config)
        plan = build_merge_plan(
            resolved.model.name_or_path,
            adapter,
            output,
            dtype=_merge_dtype(resolved),
            model_revision=resolved.model.revision,
            allow_download=allow_download,
            allow_unqualified_local_smoke=allow_unqualified_local_smoke,
        )
        result = merge_adapter(plan, dry_run=dry_run)
        _emit(result, compact=as_json)


@parity_app.command("compare")
def parity_compare(
    adapter_predictions: Annotated[
        Path,
        typer.Option(
            "--adapter-predictions",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ],
    merged_predictions: Annotated[
        Path,
        typer.Option(
            "--merged-predictions",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Create an immutable parity-report JSON file."),
    ] = None,
    enforce: Annotated[
        bool,
        typer.Option("--enforce", help="Exit non-zero unless every parity gate passes."),
    ] = False,
    as_json: JsonOption = False,
) -> None:
    """Compare existing adapter and merged prediction artifacts."""

    with _friendly_errors():
        result = compare_prediction_files(
            adapter_predictions,
            merged_predictions,
            output_path=output,
            enforce=False,
        )
        _emit(result, compact=as_json)
        if enforce and not result.passed:
            raise typer.Exit(code=1)


@parity_app.command("verify")
def parity_verify(
    adapter: Annotated[
        Path,
        typer.Option(
            "--adapter",
            help="Validated PEFT adapter used by the safe merge.",
            exists=True,
            file_okay=False,
            dir_okay=True,
        ),
    ],
    merged_model: Annotated[
        Path,
        typer.Option(
            "--merged-model",
            help="Verified TicketTune safe-merged Hugging Face model.",
            exists=True,
            file_okay=False,
            dir_okay=True,
        ),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", help="New immutable parity-report JSON file."),
    ],
    config: ConfigOption = Path("configs/smoke.yaml"),
    allow_download: Annotated[
        bool,
        typer.Option(
            "--allow-download",
            help="Permit only the pinned base revision for adapter-side generation.",
        ),
    ] = False,
    enforce: Annotated[
        bool,
        typer.Option("--enforce", help="Exit non-zero unless every parity gate passes."),
    ] = False,
    as_json: JsonOption = False,
) -> None:
    """Generate identical held-out prompts through an adapter and its safe merge."""

    with _friendly_errors():
        result = verify_live_parity(
            _load(config),
            adapter,
            merged_model,
            output_path=output,
            allow_download=allow_download,
            enforce=False,
        )
        _emit(result, compact=as_json)
        if enforce and not result.passed:
            raise typer.Exit(code=1)


@serve_app.command("vllm")
def serve_vllm(
    adapter: Annotated[
        Path,
        typer.Option(
            "--adapter",
            help="Validated PEFT adapter directory.",
            exists=True,
            file_okay=False,
            dir_okay=True,
        ),
    ],
    config: ConfigOption = Path("configs/smoke.yaml"),
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Explicitly render the plan only (the default)."),
    ] = False,
    execute: Annotated[
        bool,
        typer.Option(
            "--execute",
            help="Run the native vLLM process in the foreground on a compatible Linux CUDA host.",
        ),
    ] = False,
    allow_download: Annotated[
        bool,
        typer.Option(
            "--allow-download",
            help="Permit the pinned base-model revision to be fetched during execution.",
        ),
    ] = False,
    allow_remote: Annotated[
        bool,
        typer.Option(
            "--allow-remote",
            help="Acknowledge that a non-loopback bind needs authenticated TLS in front of it.",
        ),
    ] = False,
    allow_unqualified_local_smoke: LocalSmokeOption = False,
    as_json: JsonOption = False,
) -> None:
    """Render a vLLM plan or deliberately run it as a foreground process."""

    with _friendly_errors():
        if dry_run and execute:
            raise ValueError("--dry-run cannot be combined with --execute")
        resolved = _load(config)
        deployment = resolved.deployment
        plan = build_vllm_plan(
            resolved.model.name_or_path,
            adapter,
            model_revision=resolved.model.revision,
            allow_download=allow_download,
            served_model_name=deployment.vllm_served_model_name,
            host=deployment.vllm_host,
            port=deployment.vllm_port,
            max_lora_rank=resolved.lora.r,
            tensor_parallel_size=deployment.tensor_parallel_size,
            gpu_memory_utilization=deployment.gpu_memory_utilization,
            dtype=deployment.vllm_dtype,
            max_model_len=deployment.max_model_len,
            allow_remote=allow_remote,
            allow_unqualified_local_smoke=allow_unqualified_local_smoke,
        )
        planned = {
            "plan": plan,
            "executed": False,
            "execution_state": "planned",
            "proof_boundary": "launch_plan_only",
        }
        if not execute:
            _emit(planned, compact=as_json)
            return

        # In JSON mode both the reviewed pre-execution plan and native process
        # logs go to stderr, preserving stdout for one final machine result.
        _emit(planned, compact=as_json, err=as_json)
        run_argv(
            plan.argv,
            environment_overrides=dict(plan.environment_overrides),
            route_output_to_stderr=as_json,
        )
        _emit(
            {
                "plan": plan,
                "executed": True,
                "execution_state": "process_exited_cleanly",
                "proof_boundary": (
                    "foreground_process_exited_cleanly; live_health_and_inference_not_proven"
                ),
            },
            compact=as_json,
        )


@export_app.command("ollama")
def export_ollama(
    merged_model: Annotated[
        Path,
        typer.Option(
            "--merged-model",
            help="Merged Hugging Face model directory containing safetensors.",
            exists=True,
            file_okay=False,
            dir_okay=True,
        ),
    ],
    config: ConfigOption = Path("configs/smoke.yaml"),
    output: Annotated[
        Path | None,
        typer.Option("--output", help="GGUF and Modelfile output directory.", resolve_path=True),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Print the complete export plan without writing files."),
    ] = False,
    materialize: Annotated[
        bool,
        typer.Option("--materialize", help="Write the deterministic Modelfile."),
    ] = False,
    execute: Annotated[
        bool,
        typer.Option(
            "--execute",
            help="Clone pinned llama.cpp, build, convert, quantize, and checksum the model.",
        ),
    ] = False,
    create_model: Annotated[
        bool,
        typer.Option("--create-model", help="Run `ollama create` after a successful conversion."),
    ] = False,
    allow_unqualified_local_smoke: LocalSmokeOption = False,
    as_json: JsonOption = False,
) -> None:
    """Plan or execute the verified merged-Qwen-to-GGUF Ollama path."""

    with _friendly_errors():
        if dry_run and (materialize or execute or create_model):
            raise ValueError(
                "--dry-run cannot be combined with --materialize, --execute, or --create-model"
            )
        if create_model and not execute:
            raise ValueError("--create-model requires --execute")
        resolved = _load(config)
        deployment = resolved.deployment
        destination = output or deployment.merged_model_dir.parent / "ollama"
        plan = build_ollama_export_plan(
            merged_model,
            destination,
            model_name=deployment.ollama_model_name,
            quantization=deployment.ollama_quantization,
            context_length=deployment.max_model_len,
            allow_unqualified_local_smoke=allow_unqualified_local_smoke,
        )
        planned = {
            "plan": plan,
            "materialized": False,
            "executed": False,
            "ollama_model_created": False,
            "execution_state": "planned",
            "proof_boundary": "export_plan_only",
        }
        if not materialize and not execute:
            _emit(planned, compact=as_json)
            return

        # Never publish a success boundary before filesystem and subprocess
        # work succeeds. JSON execution routes progress and child output to
        # stderr so stdout remains one parseable final object.
        _emit(planned, compact=as_json, err=as_json)
        if materialize or execute:
            materialize_ollama_plan(plan)
        completed_commands = 0
        export_provenance = None
        if execute:
            Path(plan.clone_argv[-1]).parent.mkdir(parents=True, exist_ok=True)
            commands = (
                plan.clone_argv,
                plan.checkout_argv,
                plan.configure_argv,
                plan.build_argv,
                plan.conversion_argv,
                plan.quantize_argv,
                plan.checksum_argv,
            )
            for argv in commands:
                if argv:
                    run_argv(argv, route_output_to_stderr=as_json)
                    completed_commands += 1
            export_provenance = write_ollama_export_provenance(plan)
            if create_model:
                run_argv(plan.ollama_create_argv, route_output_to_stderr=as_json)
                completed_commands += 1

        proof_boundary = "modelfile_materialized_only"
        execution_state = "materialized"
        if execute and create_model:
            proof_boundary = "ollama_model_created; runtime_inference_not_proven"
            execution_state = "conversion_and_model_creation_completed"
        elif execute:
            proof_boundary = "local_conversion_completed; ollama_model_not_created"
            execution_state = "conversion_completed"
        _emit(
            {
                "plan": plan,
                "materialized": True,
                "executed": execute,
                "completed_commands": completed_commands,
                "export_provenance": export_provenance,
                "ollama_model_created": execute and create_model,
                "execution_state": execution_state,
                "proof_boundary": proof_boundary,
            },
            compact=as_json,
        )


@deploy_app.command("readback")
def deployment_readback(
    api_key_file: Annotated[
        Path,
        typer.Option(
            "--api-key-file",
            help="Regular, non-symlink file containing the serving API key.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    base_url: Annotated[
        str,
        typer.Option("--base-url", help="Serving origin; remote origins require HTTPS."),
    ] = "https://127.0.0.1:8443",
    model: Annotated[
        str,
        typer.Option("--model", help="Expected served adapter model ID."),
    ] = "tickettune-qwen-7b",
    expected_base_model: Annotated[
        str,
        typer.Option("--expected-base-model", help="Expected parent base-model ID."),
    ] = "Qwen/Qwen2.5-7B-Instruct",
    ca_cert: Annotated[
        Path | None,
        typer.Option(
            "--ca-cert",
            help="Optional CA certificate for a private TLS issuer.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Create an immutable redacted JSON report."),
    ] = None,
    timeout: Annotated[
        float,
        typer.Option("--timeout", min=0.1, help="Per-request timeout in seconds."),
    ] = 30,
    allow_remote: Annotated[
        bool,
        typer.Option("--allow-remote", help="Permit an explicit remote HTTPS origin."),
    ] = False,
    enforce: Annotated[
        bool,
        typer.Option("--enforce", help="Exit non-zero when readback does not pass."),
    ] = False,
    as_json: JsonOption = False,
) -> None:
    """Verify endpoint-reported model/base claims and one schema-valid response."""

    with _friendly_errors():
        report = run_readback(
            base_url=base_url,
            api_key_file=api_key_file,
            model=model,
            expected_base_model=expected_base_model,
            ca_cert=ca_cert,
            timeout=timeout,
            allow_remote=allow_remote,
            output_path=output,
        )
        _emit(report, compact=as_json)
        if enforce and not report.passed:
            raise typer.Exit(code=1)


@deploy_app.command("load-test")
def deployment_load_test(
    api_key_file: Annotated[
        Path,
        typer.Option(
            "--api-key-file",
            help="Regular, non-symlink file containing the serving API key.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    base_url: Annotated[str, typer.Option("--base-url")] = "https://127.0.0.1:8443",
    model: Annotated[str, typer.Option("--model")] = "tickettune-qwen-7b",
    requests: Annotated[int, typer.Option("--requests", min=1, max=10_000)] = 20,
    concurrency: Annotated[int, typer.Option("--concurrency", min=1, max=128)] = 2,
    min_success_rate: Annotated[float, typer.Option("--min-success-rate", min=0.0, max=1.0)] = 1.0,
    min_schema_valid_rate: Annotated[
        float, typer.Option("--min-schema-valid-rate", min=0.0, max=1.0)
    ] = 1.0,
    min_request_id_rate: Annotated[
        float, typer.Option("--min-request-id-rate", min=0.0, max=1.0)
    ] = 1.0,
    max_p95_ms: Annotated[float, typer.Option("--max-p95-ms", min=0.1)] = 5000,
    ca_cert: Annotated[
        Path | None,
        typer.Option(
            "--ca-cert",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ] = None,
    output: Annotated[Path | None, typer.Option("--output")] = None,
    timeout: Annotated[float, typer.Option("--timeout", min=0.1)] = 60,
    allow_remote: Annotated[bool, typer.Option("--allow-remote")] = False,
    enforce: Annotated[
        bool,
        typer.Option("--enforce", help="Exit non-zero when any load threshold fails."),
    ] = False,
    as_json: JsonOption = False,
) -> None:
    """Run a bounded authenticated load check without retaining model text."""

    with _friendly_errors():
        report = run_load_test(
            base_url=base_url,
            api_key_file=api_key_file,
            model=model,
            requests=requests,
            concurrency=concurrency,
            min_success_rate=min_success_rate,
            min_schema_valid_rate=min_schema_valid_rate,
            min_request_id_rate=min_request_id_rate,
            max_p95_ms=max_p95_ms,
            ca_cert=ca_cert,
            timeout=timeout,
            allow_remote=allow_remote,
            output_path=output,
        )
        _emit(report, compact=as_json)
        if enforce and not report.passed:
            raise typer.Exit(code=1)


@deploy_app.command("validate-release")
def deployment_validate_release(
    manifest: Annotated[
        Path,
        typer.Option(
            "--manifest",
            help="Schema-2 release manifest whose complete evidence graph must pass.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Create an immutable release-validation report."),
    ] = None,
    as_json: JsonOption = False,
) -> None:
    """Validate release bytes, evidence semantics, and end-to-end lineage."""

    with _friendly_errors():
        report = validate_release_manifest(manifest)
        if output is not None:
            write_proof_report(output, report)
        _emit(report, compact=as_json)


@deploy_app.command("start-release")
def deployment_start_release(
    manifest: Annotated[
        Path,
        typer.Option(
            "--manifest",
            help="Schema-2 release manifest for the approved production profile.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    execute: Annotated[
        bool,
        typer.Option(
            "--execute",
            help="Acknowledge that the validated Compose release will be started.",
        ),
    ] = False,
    as_json: JsonOption = False,
) -> None:
    """Validate and start only the exact approved production Compose profile."""

    with _friendly_errors():
        if not execute:
            raise ValueError("start-release requires --execute; use validate-release for dry proof")
        report = start_release(manifest)
        _emit(report, compact=as_json)


@deploy_app.command("rollback-plan")
def deployment_rollback_plan(
    current: Annotated[
        Path,
        typer.Option(
            "--current",
            help="Current immutable release manifest.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    previous: Annotated[
        Path,
        typer.Option(
            "--previous",
            help="Previous immutable release manifest.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Create an immutable rollback-plan JSON file."),
    ] = None,
    as_json: JsonOption = False,
) -> None:
    """Validate two releases and render shell-free rollback argv without executing."""

    with _friendly_errors():
        plan = build_rollback_plan(current, previous)
        if output is not None:
            write_proof_report(output, plan)
        _emit(plan, compact=as_json)


if __name__ == "__main__":  # pragma: no cover - exercised through the package entry point
    app()
