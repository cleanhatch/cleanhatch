from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from cleanhatch import cli
from cleanhatch.cli import main

GIT = shutil.which("git")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
APPROVED_USER_NOREPLY = "354919554+uncio-lintel@users.noreply.github.com"
GITHUB_WEBFLOW = "noreply@github.com"
HOOK_CANDIDATES = (
    PROJECT_ROOT / ".githooks" / "pre-commit",
    PROJECT_ROOT / ".git" / "hooks" / "pre-commit",
)
HOOK_SOURCE = next((path for path in HOOK_CANDIDATES if path.is_file()), HOOK_CANDIDATES[0])

pytestmark = pytest.mark.skipif(GIT is None, reason="git is not installed")


def _run(
    repo: Path,
    *arguments: str,
    env: dict[str, str] | None = None,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    complete_env = os.environ.copy()
    for key in ("GIT_AUTHOR_EMAIL", "GIT_COMMITTER_EMAIL"):
        complete_env.pop(key, None)
    if env:
        complete_env.update(env)
    return subprocess.run(
        [GIT, *arguments],
        cwd=repo,
        capture_output=True,
        text=True,
        env=complete_env,
        check=check,
    )


def _init_repo(
    path: Path,
    *,
    email: str = "uncio@cleanhatch.dev",
    name: str = "Uncio Lintel",
) -> None:
    _run(path, "init", "-q", check=True)
    _run(path, "config", "user.name", name, check=True)
    _run(path, "config", "user.email", email, check=True)
    (path / "file.txt").write_text("x", encoding="utf-8")
    _run(path, "add", "file.txt", check=True)


def _copy_hook(path: Path) -> Path:
    if not HOOK_SOURCE.is_file():
        pytest.skip("local pre-commit hook is not installed")
    target = path / ".git" / "hooks" / "pre-commit"
    shutil.copy2(HOOK_SOURCE, target)
    return target


def test_hook_blocks_non_cleanhatch_email(tmp_path: Path) -> None:
    _init_repo(tmp_path, email="attacker@gmail.com")
    _copy_hook(tmp_path)

    result = _run(tmp_path, "hook", "run", "pre-commit")

    assert result.returncode != 0
    assert "OpSec" in result.stderr


def test_hook_blocks_author_email_override(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    _copy_hook(tmp_path)

    result = _run(
        tmp_path,
        "hook",
        "run",
        "pre-commit",
        env={"GIT_AUTHOR_EMAIL": "attacker@gmail.com"},
    )

    assert result.returncode != 0
    assert "GIT_AUTHOR_EMAIL" in result.stderr


def test_hook_blocks_commit_author_override(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    _copy_hook(tmp_path)

    result = _run(
        tmp_path,
        "commit",
        "-m",
        "test",
        "--author=Attacker <attacker@gmail.com>",
    )

    assert result.returncode != 0
    assert "GIT_AUTHOR_EMAIL" in result.stderr
    assert _run(tmp_path, "log", "-1", "--format=%ae").stdout.strip() == ""


def test_hook_allows_compliant_identity(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    _copy_hook(tmp_path)

    result = _run(tmp_path, "commit", "-m", "test")

    assert result.returncode == 0
    assert _run(tmp_path, "log", "-1", "--format=%ae").stdout.strip() == "uncio@cleanhatch.dev"


@pytest.mark.parametrize(
    "email",
    [APPROVED_USER_NOREPLY, GITHUB_WEBFLOW],
    ids=["user-noreply", "github-webflow"],
)
def test_hook_allows_only_exact_approved_public_emails(tmp_path: Path, email: str) -> None:
    _init_repo(tmp_path, email=email)
    _copy_hook(tmp_path)

    result = _run(tmp_path, "commit", "-m", "test")

    assert result.returncode == 0
    assert _run(tmp_path, "log", "-1", "--format=%ae").stdout.strip() == email


def test_hook_blocks_unapproved_github_noreply(tmp_path: Path) -> None:
    _init_repo(tmp_path, email="12345+other@users.noreply.github.com")
    _copy_hook(tmp_path)

    result = _run(tmp_path, "commit", "-m", "test")

    assert result.returncode != 0
    assert "OpSec" in result.stderr


def test_email_policy_allows_only_exact_approved_public_identities() -> None:
    assert cli._email_is_compliant(APPROVED_USER_NOREPLY)
    assert cli._email_is_compliant(GITHUB_WEBFLOW)
    assert cli._email_is_compliant("uncio@cleanhatch.dev")
    assert not cli._email_is_compliant("12345+other@users.noreply.github.com")
    assert not cli._email_is_compliant("someone@users.noreply.github.com")
    assert not cli._email_is_compliant("someone@gmail.com")
    assert not cli._email_is_compliant("uncio@example.com")
    assert not cli._email_is_compliant("")


def test_hook_blocks_staged_project_identifier(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    _copy_hook(tmp_path)
    marker = "PIT_" + "Gate"
    (tmp_path / "leak.txt").write_text(f"internal project: {marker}\n", encoding="utf-8")
    _run(tmp_path, "add", "leak.txt", check=True)

    result = _run(tmp_path, "commit", "-m", "test")

    assert result.returncode != 0
    assert "identifier" in result.stderr


def test_hook_blocks_internal_research_paths_unless_bypassed(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    _copy_hook(tmp_path)
    internal = tmp_path / "docs" / "notes.md"
    internal.parent.mkdir()
    internal.write_text("private research\n", encoding="utf-8")
    _run(tmp_path, "add", "docs/notes.md", check=True)

    blocked = _run(tmp_path, "commit", "-m", "internal")

    assert blocked.returncode != 0
    assert "internal research" in blocked.stderr

    allowed = _run(
        tmp_path,
        "commit",
        "-m",
        "internal",
        env={"CLEANHATCH_ALLOW_INTERNAL": "1"},
    )

    assert allowed.returncode == 0


def test_doctor_accepts_compliant_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _init_repo(tmp_path)
    _copy_hook(tmp_path)
    commit = _run(tmp_path, "commit", "-m", "init")
    assert commit.returncode == 0

    monkeypatch.chdir(tmp_path)
    exit_code = main(["doctor"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["git_opsec"]["email_ok"] is True
    assert payload["git_opsec"]["pre_commit_hook_policy_ok"] is True
    assert payload["git_opsec"]["history_authors_ok"] is True
    assert payload["git_opsec"]["ok"] is True


def test_doctor_accepts_mixed_approved_public_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _init_repo(tmp_path, email=APPROVED_USER_NOREPLY)
    _copy_hook(tmp_path)
    commit = _run(
        tmp_path,
        "commit",
        "-m",
        "web-flow",
        env={
            "GIT_COMMITTER_NAME": "GitHub",
            "GIT_COMMITTER_EMAIL": GITHUB_WEBFLOW,
        },
    )
    assert commit.returncode == 0

    monkeypatch.chdir(tmp_path)
    exit_code = main(["doctor"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["git_opsec"]["history_authors_ok"] is True
    assert payload["git_opsec"]["non_compliant_author_email_count"] == 0


def test_doctor_flags_non_compliant_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _init_repo(tmp_path, email="attacker@gmail.com")
    hook = _copy_hook(tmp_path)
    hook.unlink()
    commit = _run(
        tmp_path,
        "commit",
        "-m",
        "history",
        "--author=Attacker <attacker@gmail.com>",
    )
    assert commit.returncode == 0
    shutil.copy2(HOOK_SOURCE, hook)
    _run(tmp_path, "config", "user.email", "uncio@cleanhatch.dev")

    monkeypatch.chdir(tmp_path)
    exit_code = main(["doctor"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert payload["git_opsec"]["email_ok"] is True
    assert payload["git_opsec"]["pre_commit_hook_policy_ok"] is True
    assert payload["git_opsec"]["history_authors_ok"] is False
    assert payload["git_opsec"]["ok"] is False
    assert "attacker@gmail.com" in payload["git_opsec"]["non_compliant_author_emails"]
