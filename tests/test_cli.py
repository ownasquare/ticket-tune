import hashlib
import json
from pathlib import Path

import pytest
from click import unstyle
from typer.testing import CliRunner

from tickettune import cli as cli_module
from tickettune.cli import app

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "smoke.yaml"
ADAPTER = ROOT / "tests" / "fixtures" / "adapter"
MERGED = ROOT / "tests" / "fixtures" / "merged"


@pytest.mark.parametrize(
    "command",
    [
        ["init", "--help"],
        ["quickstart", "--help"],
        ["advanced", "--help"],
        ["config", "show", "--help"],
        ["data", "validate", "--help"],
        ["data", "prepare", "--help"],
        ["data", "generate-candidate", "--help"],
        ["doctor", "--help"],
        ["train", "--help"],
        ["evaluate", "--help"],
        ["merge", "--help"],
        ["export", "ollama", "--help"],
        ["serve", "vllm", "--help"],
        ["deploy", "readback", "--help"],
        ["deploy", "load-test", "--help"],
        ["deploy", "validate-release", "--help"],
        ["deploy", "start-release", "--help"],
        ["deploy", "rollback-plan", "--help"],
        ["qualify", "dataset", "--help"],
        ["qualify", "scaffold-review", "--help"],
        ["qualify", "bind-review", "--help"],
        ["parity", "compare", "--help"],
        ["parity", "verify", "--help"],
        ["rehearse", "cuda", "--help"],
    ],
)
def test_commands_are_discoverable(command: list[str]) -> None:
    result = CliRunner().invoke(app, command)
    assert result.exit_code == 0, result.output


@pytest.mark.parametrize(
    "command",
    [
        ["merge", "--help"],
        ["serve", "vllm", "--help"],
        ["export", "ollama", "--help"],
    ],
)
def test_export_and_serve_help_marks_local_smoke_override(command: list[str]) -> None:
    result = CliRunner().invoke(app, command, terminal_width=160, color=True)
    output = unstyle(result.output)

    assert result.exit_code == 0, output
    assert "--allow-unqualified-loca" in output
    assert "release" in output
    assert "evidence" in output


def test_export_and_serve_cli_defaults_reject_unqualified_fixtures(tmp_path: Path) -> None:
    commands = (
        (
            [
                "merge",
                "--config",
                str(CONFIG),
                "--adapter",
                str(ADAPTER),
                "--output",
                str(tmp_path / "merged"),
                "--dry-run",
            ],
            "completed sibling training manifest",
        ),
        (
            [
                "serve",
                "vllm",
                "--config",
                str(CONFIG),
                "--adapter",
                str(ADAPTER),
            ],
            "completed sibling training manifest",
        ),
        (
            [
                "export",
                "ollama",
                "--config",
                str(CONFIG),
                "--merged-model",
                str(MERGED),
            ],
            "qualified merge lineage",
        ),
    )

    for command, expected in commands:
        result = CliRunner().invoke(app, command)
        assert result.exit_code == 2
        assert expected in result.output


def test_config_show_emits_machine_readable_resolved_paths() -> None:
    result = CliRunner().invoke(
        app,
        ["config", "show", "--config", str(CONFIG), "--json"],
    )
    assert result.exit_code == 0, result.output
    assert '"project_name":"tickettune-smoke"' in result.stdout
    assert str(ROOT / "data" / "raw" / "support_tickets.jsonl") in result.stdout


def test_training_dry_run_does_not_require_model_weights() -> None:
    result = CliRunner().invoke(
        app,
        ["train", "--config", str(CONFIG), "--dry-run", "--json"],
    )
    assert result.exit_code == 0, result.output
    assert '"executed":false' in result.stdout
    assert '"allow_download":false' in result.stdout


def test_training_dry_run_default_output_is_concise() -> None:
    result = CliRunner().invoke(
        app,
        ["train", "--config", str(CONFIG), "--dry-run"],
    )

    assert result.exit_code == 0, result.output
    assert "Training plan is valid" in result.stdout
    assert "No model weights were loaded or changed" in result.stdout
    assert not result.stdout.lstrip().startswith("{")


def test_training_details_preserve_the_complete_payload() -> None:
    result = CliRunner().invoke(
        app,
        ["train", "--config", str(CONFIG), "--dry-run", "--details"],
    )

    assert result.exit_code == 0, result.output
    assert '"hardware_preflight"' in result.stdout
    assert '"training"' in result.stdout


def test_cuda_rehearsal_enforces_static_contract_without_claiming_a_run(
    tmp_path: Path,
) -> None:
    output = tmp_path / "cuda-contract-rehearsal.json"
    result = CliRunner().invoke(
        app,
        [
            "rehearse",
            "cuda",
            "--config",
            str(ROOT / "configs" / "qwen-7b-qlora.yaml"),
            "--output",
            str(output),
            "--enforce",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["static_contract_passed"] is True
    assert payload["runtime_status"] in {
        "blocked_no_cuda",
        "eligible_for_real_cuda_run",
    }
    assert payload["executed_cuda"] is False
    assert payload["model_weights_loaded"] is False
    assert payload["optimizer_steps"] == 0
    assert payload["release_eligible"] is False
    assert payload["release_status"] == "ineligible_rehearsal"
    assert output.is_file()


def test_cuda_rehearsal_cli_refuses_lora_profile(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "rehearse",
            "cuda",
            "--config",
            str(CONFIG),
            "--output",
            str(tmp_path / "cuda-contract-rehearsal.json"),
        ],
    )

    assert result.exit_code == 2
    assert "requires a QLoRA profile" in result.output


def test_cuda_rehearsal_enforce_rejects_only_a_static_contract_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = cli_module._load(ROOT / "configs" / "qwen-7b-qlora.yaml")
    report = cli_module.run_cuda_rehearsal(config)
    failed_gate = report.gates[0].model_copy(
        update={
            "state": "failed",
            "scope": "static_contract",
            "message": "test-only malformed static contract",
        }
    )
    failed_report = report.model_copy(update={"gates": (failed_gate, *report.gates[1:])})
    monkeypatch.setattr(
        cli_module,
        "run_cuda_rehearsal",
        lambda *_args, **_kwargs: failed_report,
    )

    result = CliRunner().invoke(
        app,
        [
            "rehearse",
            "cuda",
            "--config",
            str(ROOT / "configs" / "qwen-7b-qlora.yaml"),
            "--output",
            str(tmp_path / "cuda-contract-rehearsal.json"),
            "--enforce",
            "--json",
        ],
    )

    assert result.exit_code == 1
    assert json.loads(result.stdout)["static_contract_passed"] is False


def test_init_command_creates_a_starter_project(tmp_path: Path) -> None:
    destination = tmp_path / "my-support-model"
    result = CliRunner().invoke(app, ["init", str(destination)])

    assert result.exit_code == 0, result.output
    assert "Starter project created" in result.stdout
    assert (destination / "configs/tickettune.yaml").is_file()
    assert (destination / "data/raw/support_tickets.jsonl").is_file()
    assert (destination / "predictions/pass.jsonl").is_file()
    assert f"cd {destination}" in result.stdout
    assert "tickettune data prepare --config configs/tickettune.yaml" in result.stdout


def test_init_command_json_is_machine_readable(tmp_path: Path) -> None:
    destination = tmp_path / "my-support-model"
    result = CliRunner().invoke(app, ["init", str(destination), "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["destination"] == str(destination.resolve())
    assert len(payload["created_files"]) == 5


def test_data_commands_run_the_public_prepare_path() -> None:
    validated = CliRunner().invoke(
        app,
        ["data", "validate", "--config", str(CONFIG), "--json"],
    )
    prepared = CliRunner().invoke(
        app,
        ["data", "prepare", "--config", str(CONFIG), "--json"],
    )

    assert validated.exit_code == 0, validated.output
    assert json.loads(validated.stdout)["examples"] == 56
    assert prepared.exit_code == 0, prepared.output
    assert json.loads(prepared.stdout)["total_examples"] == 56


def test_generate_candidate_creates_fixed_non_release_corpus(tmp_path: Path) -> None:
    output = tmp_path / "qualified-candidate.jsonl"

    result = CliRunner().invoke(
        app,
        [
            "data",
            "generate-candidate",
            "--output",
            str(output),
            "--seed",
            "42",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["artifact"]["record_count"] == 1_120
    assert payload["seed"] == 42
    assert payload["release_eligible"] is False
    assert payload["review_status"] == "pending_two_independent_human_reviews"
    assert output.is_file()


def test_generate_candidate_rejects_noncanonical_seed(tmp_path: Path) -> None:
    output = tmp_path / "qualified-candidate.jsonl"

    result = CliRunner().invoke(
        app,
        [
            "data",
            "generate-candidate",
            "--output",
            str(output),
            "--seed",
            "7",
        ],
    )

    assert result.exit_code == 2
    assert "fixed at seed 42" in result.output
    assert not output.exists()


def test_scaffold_review_creates_two_pending_human_packets(tmp_path: Path) -> None:
    resolved = cli_module._load(CONFIG)
    cli_module.prepare_dataset(
        resolved.data.source_path,
        resolved.data.processed_dir,
        seed=resolved.seed,
        splits=resolved.data.splits,
    )
    output_dir = tmp_path / "review"

    result = CliRunner().invoke(
        app,
        [
            "qualify",
            "scaffold-review",
            "--config",
            str(CONFIG),
            "--output-dir",
            str(output_dir),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["record_count"] == 56
    assert payload["held_out_count"] == 7
    assert payload["status"] == "pending_two_independent_human_reviews"
    assert payload["release_eligible"] is False
    packet_paths = [Path(path) for path in payload["reviewer_packet_paths"]]
    assert len(packet_paths) == 2
    for path in packet_paths:
        packet = json.loads(path.read_text(encoding="utf-8"))
        assert packet["status"] == "draft"
        assert packet["review_date"] is None
        assert len(packet["decisions"]) == 56
        assert all(decision["labels"] == "pending" for decision in packet["decisions"])

    first_packet = json.loads(packet_paths[0].read_text(encoding="utf-8"))
    first_packet["reviewer_id"] = "human-reviewer-01"
    packet_paths[0].write_text(
        json.dumps(first_packet, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    bound_path = output_dir / "review-manifest.bound.json"
    bound = CliRunner().invoke(
        app,
        [
            "qualify",
            "bind-review",
            "--review-manifest",
            str(output_dir / "review-manifest.json"),
            "--output",
            str(bound_path),
            "--json",
        ],
    )
    assert bound.exit_code == 0, bound.output
    bound_payload = json.loads(bound.stdout)
    assert bound_payload["approval_status"] == "draft"
    assert bound_payload["release_eligible"] is False
    rebound_manifest = json.loads(bound_path.read_text(encoding="utf-8"))
    assert (
        rebound_manifest["reviewer_packets"][0]["sha256"]
        == hashlib.sha256(packet_paths[0].read_bytes()).hexdigest()
    )

    qualification = CliRunner().invoke(
        app,
        [
            "qualify",
            "dataset",
            "--config",
            str(CONFIG),
            "--review-manifest",
            str(bound_path),
            "--json",
        ],
    )
    assert qualification.exit_code == 0, qualification.output
    assert json.loads(qualification.stdout)["qualified"] is False


def test_completed_training_summary_prints_artifacts_and_next_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolved = cli_module._load(CONFIG)
    cli_module.prepare_dataset(
        resolved.data.source_path,
        resolved.data.processed_dir,
        seed=resolved.seed,
        splits=resolved.data.splits,
    )
    preflight = cli_module.run_preflight(resolved)
    planned = cli_module.run_training(
        resolved,
        dry_run=True,
        allow_download=False,
        resume_from_checkpoint=None,
        hardware_preflight=preflight,
    )
    completed = planned.model_copy(
        update={
            "executed": True,
            "run_id": "run-123",
            "adapter_path": "artifacts/run-123/adapter",
            "manifest_path": "artifacts/run-123/manifest.json",
            "metrics": {"train_loss": 1.25},
            "proof_boundary": "test fixture",
        }
    )
    monkeypatch.setattr(cli_module, "require_compatible", lambda _preflight: None)
    monkeypatch.setattr(cli_module, "run_training", lambda *_args, **_kwargs: completed)

    result = CliRunner().invoke(app, ["train", "--config", str(CONFIG)])

    assert result.exit_code == 0, result.output
    assert "Training completed" in result.stdout
    assert "run-123" in result.stdout
    assert "Train loss: 1.25" in result.stdout
    assert "compare the adapter" in result.stdout


def test_quickstart_default_output_is_concise(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        ["quickstart", "--workspace", str(tmp_path / "demo")],
    )

    assert result.exit_code == 0, result.output
    assert "TicketTune is ready" in result.stdout
    assert "Dataset: 56 synthetic tickets validated" in result.stdout
    assert "Training plan: valid" in result.stdout
    assert "Evaluation contract: passed" in result.stdout
    assert "No model weights were downloaded or trained" in result.stdout
    assert f"Demo files: {(tmp_path / 'demo').resolve()}" in result.stdout
    assert "Next: tickettune init my-support-model" in result.stdout
    assert not result.stdout.lstrip().startswith("{")


def test_quickstart_details_explain_the_proof_boundary(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        ["quickstart", "--workspace", str(tmp_path / "demo"), "--details"],
    )

    assert result.exit_code == 0, result.output
    assert '"manifest_path"' in result.stdout
    assert '"evaluation_report_path"' in result.stdout
    assert "No model weights were downloaded or trained" in result.stdout


def test_quickstart_json_is_compact_and_machine_readable(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        ["quickstart", "--workspace", str(tmp_path / "demo"), "--json"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["source_examples"] == 56
    assert payload["split_counts"] == {"test": 7, "train": 42, "validation": 7}
    assert payload["training_plan_ready"] is True
    assert payload["evaluation_passed"] is True
    assert payload["artifacts_retained"] is True


def test_root_help_keeps_advanced_controls_out_of_the_core_menu() -> None:
    result = CliRunner().invoke(app, ["--help"], terminal_width=160)

    assert result.exit_code == 0, result.output
    assert "advanced" in result.stdout
    assert "quickstart" in result.stdout
    assert "Collect redacted deployment and rollback proof" not in result.stdout
    assert "Verify adapter and safe-merge parity" not in result.stdout


def test_advanced_command_reveals_hidden_controls() -> None:
    result = CliRunner().invoke(app, ["advanced"])

    assert result.exit_code == 0, result.output
    for command in ("config", "qualify", "parity", "rehearse", "merge", "deploy"):
        assert f"tickettune {command} --help" in result.stdout


@pytest.mark.parametrize(
    "live_options",
    [
        ["--adapter", str(ADAPTER)],
        ["--compare-baseline"],
        ["--allow-download"],
        ["--allow-unverified-adapter"],
    ],
)
def test_evaluate_rejects_fixture_and_live_option_mix(live_options: list[str]) -> None:
    result = CliRunner().invoke(
        app,
        [
            "evaluate",
            "--config",
            str(CONFIG),
            "--predictions",
            str(ROOT / "tests" / "fixtures" / "predictions_pass.jsonl"),
            *live_options,
        ],
    )

    assert result.exit_code == 2
    assert "fixture scoring only" in result.output


def test_evaluate_forwards_explicit_local_unverified_adapter_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_run(_config: object, **kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return {"evaluation_id": "local-fixture-evaluation"}

    monkeypatch.setattr(cli_module, "run_model_evaluation", fake_run)

    result = CliRunner().invoke(
        app,
        [
            "evaluate",
            "--config",
            str(CONFIG),
            "--adapter",
            str(ADAPTER),
            "--allow-unverified-adapter",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == [
        {
            "adapter_path": ADAPTER.resolve(),
            "compare_baseline": False,
            "allow_download": False,
            "allow_unverified_adapter": True,
            "enforce_thresholds": False,
        }
    ]


def test_evaluate_rejects_unverified_adapter_override_with_release_thresholds() -> None:
    result = CliRunner().invoke(
        app,
        [
            "evaluate",
            "--config",
            str(CONFIG),
            "--adapter",
            str(ADAPTER),
            "--allow-unverified-adapter",
            "--enforce-thresholds",
        ],
    )

    assert result.exit_code == 2
    assert "cannot be combined with enforced release thresholds" in result.output


def test_vllm_dry_run_uses_named_static_adapter() -> None:
    result = CliRunner().invoke(
        app,
        [
            "serve",
            "vllm",
            "--config",
            str(CONFIG),
            "--adapter",
            str(ADAPTER),
            "--allow-unqualified-local-smoke",
            "--dry-run",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    assert '"--enable-lora"' in result.stdout
    assert '"--lora-modules"' in result.stdout
    assert '"--revision"' in result.stdout
    assert '"adapter_revision":"7ae557604adf67be50417f59c2c2f167def9a775"' in result.stdout
    assert '"allow_download":false' in result.stdout
    assert '"HF_HUB_OFFLINE","1"' in result.stdout
    assert '"launch_plan_only"' in result.stdout
    assert '"unqualified_local_smoke_override_not_release_evidence"' in result.stdout


def test_vllm_is_plan_only_by_default() -> None:
    result = CliRunner().invoke(
        app,
        [
            "serve",
            "vllm",
            "--config",
            str(CONFIG),
            "--adapter",
            str(ADAPTER),
            "--allow-unqualified-local-smoke",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["executed"] is False
    assert payload["execution_state"] == "planned"
    assert payload["plan"]["model_revision"] == "7ae557604adf67be50417f59c2c2f167def9a775"
    assert (
        payload["plan"]["lineage_boundary"]
        == "unqualified_local_smoke_override_not_release_evidence"
    )


def test_vllm_execute_emits_final_json_only_after_process_returns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[str, ...] | list[str], dict[str, object]]] = []

    def fake_run(argv: tuple[str, ...] | list[str], **kwargs: object) -> None:
        calls.append((argv, kwargs))

    monkeypatch.setattr(cli_module, "run_argv", fake_run)
    result = CliRunner().invoke(
        app,
        [
            "serve",
            "vllm",
            "--config",
            str(CONFIG),
            "--adapter",
            str(ADAPTER),
            "--allow-unqualified-local-smoke",
            "--execute",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["executed"] is True
    assert payload["execution_state"] == "process_exited_cleanly"
    assert "live_health_and_inference_not_proven" in payload["proof_boundary"]
    assert len(calls) == 1
    assert calls[0][1]["route_output_to_stderr"] is True
    assert calls[0][1]["environment_overrides"] == {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }


def test_vllm_dry_run_and_execute_are_mutually_exclusive() -> None:
    result = CliRunner().invoke(
        app,
        [
            "serve",
            "vllm",
            "--config",
            str(CONFIG),
            "--adapter",
            str(ADAPTER),
            "--dry-run",
            "--execute",
        ],
    )

    assert result.exit_code == 2
    assert "cannot be combined" in result.output


def test_ollama_dry_run_uses_merged_gguf_path() -> None:
    result = CliRunner().invoke(
        app,
        [
            "export",
            "ollama",
            "--config",
            str(CONFIG),
            "--merged-model",
            str(MERGED),
            "--allow-unqualified-local-smoke",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    assert '"direct_adapter_supported":false' in result.stdout
    assert '"export_plan_only"' in result.stdout
    assert '"unqualified_local_smoke_override_not_release_evidence"' in result.stdout


def test_ollama_cli_preserves_merged_root_symlink_for_library_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    merged = tmp_path / "merged"
    merged.mkdir()
    linked = tmp_path / "linked-merged"
    linked.symlink_to(merged, target_is_directory=True)
    calls: list[Path] = []

    def fake_build(path: Path, _output: Path, **_kwargs: object) -> dict[str, object]:
        calls.append(path)
        if path.is_symlink():
            raise ValueError("merged root rejected by library validator")
        return {"merged_model": str(path)}

    monkeypatch.setattr(cli_module, "build_ollama_export_plan", fake_build)
    result = CliRunner().invoke(
        app,
        [
            "export",
            "ollama",
            "--config",
            str(CONFIG),
            "--merged-model",
            str(linked),
            "--json",
        ],
    )

    assert result.exit_code == 2
    assert "merged root rejected by library validator" in result.output
    assert calls == [linked]


def test_ollama_execute_emits_success_only_after_all_commands_return(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, ...] | list[str]] = []

    def fake_run(argv: tuple[str, ...] | list[str], **kwargs: object) -> None:
        assert kwargs["route_output_to_stderr"] is True
        calls.append(argv)

    monkeypatch.setattr(cli_module, "run_argv", fake_run)
    monkeypatch.setattr(
        cli_module,
        "write_ollama_export_provenance",
        lambda plan: {
            "provenance_path": plan.export_provenance_path,
            "provenance_sha256": "a" * 64,
            "artifact_sha256": [],
        },
    )
    result = CliRunner().invoke(
        app,
        [
            "export",
            "ollama",
            "--config",
            str(CONFIG),
            "--merged-model",
            str(MERGED),
            "--allow-unqualified-local-smoke",
            "--output",
            str(tmp_path / "ollama"),
            "--execute",
            "--create-model",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["completed_commands"] == len(calls) == 8
    assert payload["executed"] is True
    assert payload["ollama_model_created"] is True
    assert payload["execution_state"] == "conversion_and_model_creation_completed"
    assert payload["export_provenance"]["provenance_sha256"] == "a" * 64


def test_ollama_failed_command_does_not_emit_success_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(argv: tuple[str, ...] | list[str], **kwargs: object) -> None:
        del argv, kwargs
        raise RuntimeError("simulated conversion failure")

    monkeypatch.setattr(cli_module, "run_argv", fail)
    result = CliRunner().invoke(
        app,
        [
            "export",
            "ollama",
            "--config",
            str(CONFIG),
            "--merged-model",
            str(MERGED),
            "--allow-unqualified-local-smoke",
            "--output",
            str(tmp_path / "ollama"),
            "--execute",
            "--json",
        ],
    )

    assert result.exit_code == 2
    combined = result.stdout + getattr(result, "stderr", "")
    assert "simulated conversion failure" in combined
    assert "local_conversion_completed" not in combined
    assert "ollama_model_created;" not in combined


def test_ollama_create_model_requires_execution() -> None:
    result = CliRunner().invoke(
        app,
        [
            "export",
            "ollama",
            "--config",
            str(CONFIG),
            "--merged-model",
            str(MERGED),
            "--create-model",
        ],
    )

    assert result.exit_code == 2
    assert "requires --execute" in result.output


def test_merge_dry_run_is_download_closed(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "merge",
            "--config",
            str(CONFIG),
            "--adapter",
            str(ADAPTER),
            "--allow-unqualified-local-smoke",
            "--output",
            str(tmp_path / "merged"),
            "--dry-run",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    assert '"allow_download":false' in result.stdout
    assert '"safe_merge":true' in result.stdout


def test_deployment_readback_enforcement_uses_redacted_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key = tmp_path / "api-key"
    key.write_text("not-emitted-0123456789", encoding="utf-8")

    class Report(dict[str, object]):
        passed = False

    report = Report(passed=False, failure_codes=["model_not_found"])
    monkeypatch.setattr(cli_module, "run_readback", lambda **_kwargs: report)

    result = CliRunner().invoke(
        app,
        [
            "deploy",
            "readback",
            "--api-key-file",
            str(key),
            "--enforce",
            "--json",
        ],
    )

    assert result.exit_code == 1
    assert '"passed":false' in result.stdout
    assert "not-emitted" not in result.stdout


def test_deployment_load_command_forwards_explicit_thresholds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key = tmp_path / "api-key"
    key.write_text("not-emitted-0123456789", encoding="utf-8")
    calls: list[dict[str, object]] = []

    class Report(dict[str, object]):
        passed = True

    report = Report(passed=True)

    def fake_load(**kwargs: object) -> object:
        calls.append(kwargs)
        return report

    monkeypatch.setattr(cli_module, "run_load_test", fake_load)
    result = CliRunner().invoke(
        app,
        [
            "deploy",
            "load-test",
            "--api-key-file",
            str(key),
            "--requests",
            "8",
            "--concurrency",
            "4",
            "--min-success-rate",
            "0.99",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls[0]["requests"] == 8
    assert calls[0]["concurrency"] == 4
    assert calls[0]["min_success_rate"] == 0.99
    assert "not-emitted" not in result.stdout


def test_deployment_validate_release_writes_immutable_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "release-manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "release-validation.json"
    calls: list[Path] = []
    report = {
        "schema_version": "1.0",
        "release_id": "release-20260718",
        "passed": True,
        "proof_boundary": "semantic_release_validation",
    }

    def fake_validate(path: Path) -> dict[str, object]:
        calls.append(path)
        return report

    monkeypatch.setattr(cli_module, "validate_release_manifest", fake_validate)
    result = CliRunner().invoke(
        app,
        [
            "deploy",
            "validate-release",
            "--manifest",
            str(manifest),
            "--output",
            str(output),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == [manifest.resolve()]
    assert json.loads(result.stdout) == report
    assert json.loads(output.read_text(encoding="utf-8")) == report


def test_deployment_validate_release_preserves_symlink_for_library_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "release-manifest.json"
    target.write_text("{}\n", encoding="utf-8")
    linked = tmp_path / "linked-release-manifest.json"
    linked.symlink_to(target)
    calls: list[Path] = []

    def fake_validate(path: Path) -> dict[str, object]:
        calls.append(path)
        return {"passed": True}

    monkeypatch.setattr(cli_module, "validate_release_manifest", fake_validate)
    result = CliRunner().invoke(
        app,
        ["deploy", "validate-release", "--manifest", str(linked), "--json"],
    )

    assert result.exit_code == 0, result.output
    assert calls == [linked]


def test_deployment_start_release_requires_explicit_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "release-manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    calls: list[Path] = []

    def fake_start(path: Path) -> dict[str, object]:
        calls.append(path)
        return {"passed": True}

    monkeypatch.setattr(cli_module, "start_release", fake_start)
    refused = CliRunner().invoke(
        app,
        ["deploy", "start-release", "--manifest", str(manifest), "--json"],
    )
    assert refused.exit_code == 2
    assert "requires --execute" in refused.output
    assert calls == []

    started = CliRunner().invoke(
        app,
        [
            "deploy",
            "start-release",
            "--manifest",
            str(manifest),
            "--execute",
            "--json",
        ],
    )
    assert started.exit_code == 0, started.output
    assert calls == [manifest.resolve()]
    assert json.loads(started.stdout) == {"passed": True}


def test_dataset_qualification_enforcement_emits_failed_report_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Report(dict[str, object]):
        qualified = False

    calls: list[tuple[Path, Path, bool]] = []

    def fake_qualify(source: Path, manifest: Path, *, enforce: bool) -> Report:
        calls.append((source, manifest, enforce))
        return Report(
            qualified=False,
            dataset_tier="portfolio_smoke",
            decisions=[{"policy": "minimum_record_count", "passed": False}],
        )

    monkeypatch.setattr(cli_module, "qualify_dataset", fake_qualify)
    manifest = ROOT / "data" / "qualified" / "review-manifest.example.json"
    result = CliRunner().invoke(
        app,
        [
            "qualify",
            "dataset",
            "--config",
            str(CONFIG),
            "--review-manifest",
            str(manifest),
            "--enforce",
            "--json",
        ],
    )

    assert result.exit_code == 1
    assert '"qualified":false' in result.stdout
    assert calls == [(ROOT / "data" / "raw" / "support_tickets.jsonl", manifest, False)]


def test_parity_compare_enforcement_emits_failed_report_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Result(dict[str, object]):
        passed = False

    calls: list[dict[str, object]] = []

    def fake_compare(*_args: Path, **kwargs: object) -> Result:
        calls.append(kwargs)
        return Result(passed=False, report={"mismatched_ids": ["TT-0001"]})

    monkeypatch.setattr(cli_module, "compare_prediction_files", fake_compare)
    predictions = ROOT / "tests" / "fixtures" / "predictions.jsonl"
    result = CliRunner().invoke(
        app,
        [
            "parity",
            "compare",
            "--adapter-predictions",
            str(predictions),
            "--merged-predictions",
            str(predictions),
            "--enforce",
            "--json",
        ],
    )

    assert result.exit_code == 1
    assert '"passed":false' in result.stdout
    assert calls == [{"output_path": None, "enforce": False}]


def test_live_parity_command_forwards_verified_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Result(dict[str, object]):
        passed = True

    calls: list[tuple[object, ...]] = []

    def fake_verify(*args: object, **kwargs: object) -> Result:
        calls.append((*args, kwargs))
        return Result(passed=True, report={"proof_boundary": "test"})

    monkeypatch.setattr(cli_module, "verify_live_parity", fake_verify)
    output = tmp_path / "parity.json"
    result = CliRunner().invoke(
        app,
        [
            "parity",
            "verify",
            "--config",
            str(CONFIG),
            "--adapter",
            str(ADAPTER),
            "--merged-model",
            str(MERGED),
            "--output",
            str(output),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls[0][1] == ADAPTER.resolve()
    assert calls[0][2] == MERGED.resolve()
    assert calls[0][3]["output_path"] == output
    assert calls[0][3]["allow_download"] is False
    assert calls[0][3]["enforce"] is False
