"""Doctor diagnostics: dependency floors and local OpSec remediation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cleanhatch import cli
from cleanhatch.cli import main


def _run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    monkeypatch.chdir(tmp_path)
    exit_code = main(["doctor"])
    return exit_code, json.loads(capsys.readouterr().out)


def test_doctor_has_no_license_server_section(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code, payload = _run(tmp_path, monkeypatch, capsys)

    assert "license_server" not in payload
    assert exit_code == 0


def test_doctor_flags_dependency_below_floor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    real_version = cli._package_version
    monkeypatch.setattr(
        cli,
        "_package_version",
        lambda name: "0.9.0" if name == "polars" else real_version(name),
    )

    exit_code, payload = _run(tmp_path, monkeypatch, capsys)

    polars = payload["packages"]["polars"]
    assert polars["required"] == ">=1.0.0"
    assert polars["found"] == "0.9.0"
    assert polars["ok"] is False
    assert polars["remediation"] == 'python -m pip install --upgrade "polars>=1.0.0"'
    assert payload["ok"] is False
    assert exit_code == 1


def test_doctor_emits_one_line_remediation_for_every_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    real_version = cli._package_version
    monkeypatch.setattr(
        cli,
        "_package_version",
        lambda name: "0.1.0" if name == "duckdb" else real_version(name),
    )
    monkeypatch.setattr(
        cli,
        "_git_opsec",
        lambda: {"repository": True, "ok": False},
    )

    exit_code, payload = _run(tmp_path, monkeypatch, capsys)

    assert exit_code == 1
    remediations = [
        payload["packages"]["duckdb"]["remediation"],
        payload["git_opsec"]["remediation"],
    ]
    assert all(isinstance(item, str) and item for item in remediations)
    assert all("\n" not in item for item in remediations)
