from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from tickettune.secret_scan import DETECT_SECRETS_VERSION, main, scan_repository


def _git(repository: Path, *arguments: str) -> str:
    git = shutil.which("git")
    if git is None:
        pytest.skip("Git is required for secret-scan tests")
    completed = subprocess.run(  # noqa: S603 - absolute executable, argument list, no shell
        [git, "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _commit(repository: Path, message: str) -> None:
    _git(repository, "add", "-A")
    _git(
        repository,
        "-c",
        "user.name=TicketTune Tests",
        "-c",
        "user.email=tests@example.invalid",
        "commit",
        "-m",
        message,
    )


def _fake_access_key() -> str:
    return "".join(("AK", "IA", "IOSF", "ODNN", "7EXA", "MPLE"))


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    source = tmp_path / "repository"
    source.mkdir()
    _git(source, "init", "-b", "main")
    (source / "README.md").write_text("# Clean fixture\n", encoding="utf-8")
    _commit(source, "initial")
    return source


def _write_secret(path: Path) -> None:
    path.write_text(f"aws_access_key_id = {_fake_access_key()}\n", encoding="utf-8")


def _write_baseline(
    path: Path,
    findings: set[tuple[str, str]],
    *,
    review: bool | None,
) -> None:
    entries: list[dict[str, object]] = []
    for secret_type, secret_hash in sorted(findings):
        entry: dict[str, object] = {
            "type": secret_type,
            "hashed_secret": secret_hash,
            "line_number": 1,
            "is_verified": False,
        }
        if review is not None:
            entry["is_secret"] = review
        entries.append(entry)

    path.write_text(
        json.dumps(
            {
                "version": DETECT_SECRETS_VERSION,
                "plugins_used": [],
                "filters_used": [],
                "results": {"credentials.env": entries},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def test_scan_uses_modified_tracked_working_tree_bytes(repository: Path) -> None:
    _write_secret(repository / "README.md")

    report = scan_repository(repository)

    assert report.tracked_file_count == 1
    assert report.history_blob_count == 1
    assert report.findings
    assert {finding.scope for finding in report.findings} == {"working-tree"}
    assert {finding.path for finding in report.findings} == {"README.md"}


def test_scan_ignores_untracked_working_tree_files(repository: Path) -> None:
    _write_secret(repository / "untracked.env")

    report = scan_repository(repository)

    assert report.findings == ()
    assert report.tracked_file_count == 1


def test_scan_finds_secret_removed_from_head_in_unique_history_blob(
    repository: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret_file = repository / "credentials.env"
    _write_secret(secret_file)
    _commit(repository, "add historical credential")
    secret_file.unlink()
    _commit(repository, "remove historical credential")

    report = scan_repository(repository)

    assert report.tracked_file_count == 1
    assert report.history_blob_count == 2
    assert report.findings
    assert {finding.scope for finding in report.findings} == {"history"}
    assert {finding.path for finding in report.findings} == {"credentials.env"}
    assert len({finding.blob_oid for finding in report.findings}) == 1

    assert main(["--source", str(repository)]) == 1
    output = capsys.readouterr()
    assert "[history]" in output.out
    assert "[working-tree] finding" not in output.out
    assert _fake_access_key() not in output.out
    assert _fake_access_key() not in output.err


def test_json_output_distinguishes_scopes_without_secret_values(
    repository: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_secret(repository / "README.md")

    assert main(["--source", str(repository), "--json"]) == 1
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert {finding["scope"] for finding in payload["findings"]} == {"working-tree"}
    assert all("secret" not in key for finding in payload["findings"] for key in finding)
    assert _fake_access_key() not in captured.out
    assert _fake_access_key() not in captured.err


def test_reviewed_false_positive_baseline_suppresses_matching_hashes(
    repository: Path,
) -> None:
    _write_secret(repository / "credentials.env")
    _commit(repository, "add reviewed fixture")
    initial = scan_repository(repository)
    identities = {(finding.secret_type, finding.secret_hash) for finding in initial.findings}
    baseline = repository / ".secrets.baseline"
    _write_baseline(baseline, identities, review=False)
    _commit(repository, "add reviewed baseline")

    report = scan_repository(repository, baseline)

    assert report.findings == ()
    assert report.reviewed_suppression_count == len(initial.findings)


@pytest.mark.parametrize("review", [None, True])
def test_unreviewed_or_true_baseline_entries_do_not_suppress_findings(
    repository: Path,
    review: bool | None,
) -> None:
    _write_secret(repository / "credentials.env")
    _commit(repository, "add credential")
    initial = scan_repository(repository)
    identities = {(finding.secret_type, finding.secret_hash) for finding in initial.findings}
    baseline = repository / ".secrets.baseline"
    _write_baseline(baseline, identities, review=review)
    _commit(repository, "add baseline")

    report = scan_repository(repository, baseline)

    assert report.findings
    assert report.reviewed_suppression_count == 0


def test_report_order_is_deterministic(repository: Path) -> None:
    _write_secret(repository / "credentials.env")
    _commit(repository, "add credential")

    first = scan_repository(repository)
    second = scan_repository(repository)

    assert first == second
    assert list(first.findings) == sorted(
        first.findings,
        key=lambda finding: finding.sort_key,
    )
