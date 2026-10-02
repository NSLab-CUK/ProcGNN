"""Exercise the cleanup policy on disposable fixtures, not the research archive."""

from pathlib import Path
import shutil
import subprocess

import pytest


SHELL = shutil.which("pwsh") or shutil.which("powershell")
pytestmark = pytest.mark.skipif(SHELL is None, reason="PowerShell is unavailable")
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "cleanup_obsolete_artifacts.ps1"


@pytest.fixture
def archive(tmp_path):
    workspace = tmp_path / "project"
    (workspace / "scripts").mkdir(parents=True)
    shutil.copyfile(SCRIPT, workspace / "scripts" / SCRIPT.name)
    protected = {
        "outputs/debug_run/result.csv": b"debug result",
        "outputs/cache/__pycache__/keep.pyc": b"output cache",
        "data/__pycache__/keep.pyc": b"data cache",
        "archive/old_run/result.json": b"archived result",
        "validation.xlsx": b"workbook",
        "logs/old.log": b"research log",
    }
    for relative, content in protected.items():
        path = workspace / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    cache = workspace / "scripts" / "__pycache__" / "old.pyc"
    cache.parent.mkdir()
    cache.write_bytes(b"disposable cache")
    backup = tmp_path / "_ProcGNN_test_backup"
    return workspace, backup, protected


def run_cleanup(workspace, *arguments):
    return subprocess.run(
        [SHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
         str(workspace / "scripts" / SCRIPT.name), *map(str, arguments)],
        capture_output=True, text=True, errors="replace", timeout=30,
    )


def assert_protected(workspace, protected):
    for relative, content in protected.items():
        assert (workspace / relative).read_bytes() == content


def test_preview_is_non_mutating(archive):
    workspace, backup, protected = archive
    result = run_cleanup(workspace)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Preview only" in result.stdout
    assert (workspace / "scripts/__pycache__/old.pyc").exists()
    assert not backup.exists()
    assert_protected(workspace, protected)


def test_execute_is_recoverable_and_preserves_all_outputs(archive):
    workspace, backup, protected = archive
    result = run_cleanup(workspace, "-Execute", "-BackupDirectory", backup)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (workspace / "scripts/__pycache__").exists()
    assert (backup / "files/scripts/__pycache__/old.pyc").read_bytes() == b"disposable cache"
    assert (backup / "restore_manifest.csv").exists()
    assert (backup / "file_checksums.csv").exists()
    assert_protected(workspace, protected)


def test_artifact_inside_cache_blocks_cleanup(archive):
    workspace, backup, protected = archive
    artifact = workspace / "scripts/__pycache__/checkpoint.pt"
    artifact.write_bytes(b"checkpoint")
    result = run_cleanup(workspace, "-Execute", "-BackupDirectory", backup)
    assert result.returncode != 0
    assert artifact.read_bytes() == b"checkpoint"
    assert not backup.exists()
    assert_protected(workspace, protected)


def test_backup_inside_workspace_is_rejected(archive):
    workspace, _, protected = archive
    backup = workspace / "_ProcGNN_unsafe_backup"
    result = run_cleanup(workspace, "-Execute", "-BackupDirectory", backup)
    assert result.returncode != 0
    assert not backup.exists()
    assert (workspace / "scripts/__pycache__/old.pyc").exists()
    assert_protected(workspace, protected)


def test_changed_duplicate_document_is_not_archived(archive):
    workspace, backup, protected = archive
    (workspace / "docs").mkdir()
    duplicate = workspace / "docs/final_model.md"
    duplicate.write_text("not identical", encoding="utf-8")
    (workspace / "docs/MODEL_final.md").write_text("canonical", encoding="utf-8")
    result = run_cleanup(workspace, "-Execute", "-IncludeLegacyHelpers", "-BackupDirectory", backup)
    assert result.returncode == 0, result.stdout + result.stderr
    assert duplicate.read_text(encoding="utf-8") == "not identical"
    assert_protected(workspace, protected)
