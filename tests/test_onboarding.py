from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from tickettune import onboarding as onboarding_module
from tickettune.onboarding import (
    init_project,
    run_quickstart,
    starter_config_path,
    starter_dataset_path,
    starter_gitignore_path,
    starter_predictions_path,
    starter_readme_path,
)

ROOT = Path(__file__).resolve().parents[1]


def test_starter_resources_are_complete() -> None:
    assert starter_config_path().read_bytes() == (ROOT / "configs/smoke.yaml").read_bytes()
    assert (
        starter_dataset_path().read_bytes()
        == (ROOT / "data/raw/support_tickets.jsonl").read_bytes()
    )
    assert len(starter_dataset_path().read_text(encoding="utf-8").splitlines()) == 56
    assert len(starter_predictions_path().read_text(encoding="utf-8").splitlines()) == 3
    assert "My TicketTune project" in starter_readme_path().read_text(encoding="utf-8")
    assert "artifacts/" in starter_gitignore_path().read_text(encoding="utf-8")


@pytest.mark.parametrize("precreate", [False, True])
def test_init_project_creates_a_self_contained_project(
    tmp_path: Path,
    precreate: bool,
) -> None:
    destination = tmp_path / "support-model"
    if precreate:
        destination.mkdir()

    result = init_project(destination)

    assert result.destination == destination.resolve()
    assert result.config_path.is_file()
    assert result.dataset_path.is_file()
    assert result.predictions_path.is_file()
    assert result.readme_path.is_file()
    assert result.gitignore_path.is_file()
    assert set(result.created_files) == {
        result.config_path,
        result.dataset_path,
        result.predictions_path,
        result.readme_path,
        result.gitignore_path,
    }
    with pytest.raises(FrozenInstanceError):
        result.destination = tmp_path  # type: ignore[misc]


def test_init_project_rejects_a_file_without_changing_it(tmp_path: Path) -> None:
    destination = tmp_path / "occupied"
    destination.write_text("keep me", encoding="utf-8")

    with pytest.raises(ValueError, match="must be a directory"):
        init_project(destination)

    assert destination.read_text(encoding="utf-8") == "keep me"


def test_init_project_rejects_a_symlink_without_touching_target(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    destination = tmp_path / "linked"
    destination.symlink_to(target, target_is_directory=True)

    with pytest.raises(ValueError, match="must not be a symlink"):
        init_project(destination)

    assert destination.is_symlink()
    assert list(target.iterdir()) == []


def test_init_project_rejects_a_nonempty_directory(tmp_path: Path) -> None:
    destination = tmp_path / "occupied"
    destination.mkdir()
    marker = destination / "keep.txt"
    marker.write_text("keep me", encoding="utf-8")

    with pytest.raises(ValueError, match="must be empty"):
        init_project(destination)

    assert marker.read_text(encoding="utf-8") == "keep me"


def test_init_project_rolls_back_files_created_before_a_write_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "starter"
    duplicate = Path("nested/duplicate.txt")
    monkeypatch.setattr(
        onboarding_module,
        "_starter_payloads",
        lambda: ((duplicate, b"first"), (duplicate, b"second")),
    )

    with pytest.raises(FileExistsError):
        init_project(destination)

    assert not destination.exists()


def test_default_quickstart_cleans_up_its_temporary_artifacts() -> None:
    result = run_quickstart()

    assert result.workspace is None
    assert result.config_path is None
    assert result.manifest_path is None
    assert result.evaluation_report_path is None
    assert result.artifacts_retained is False
    assert "cleaned up" in result.proof_boundary
