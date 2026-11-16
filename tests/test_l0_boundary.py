"""Negative boundary checks for the Apache-2.0 L0 distribution.

These tests exercise actual imports, filesystem placement, CLI parsing, and a
fresh subprocess.  They deliberately do not grep source text for module names.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

import cleanhatch
import cleanhatch.packs as packs_module
from cleanhatch import PackNotInstalledError, available_packs, get_pack
from cleanhatch.cli import main

_COMMERCIAL_MODULES = (
    "cleanhatch.license",
    "cleanhatch.license_client",
    "cleanhatch.report",
    "cleanhatch.bundle",
    "cleanhatch.packs.market_bars_pit",
    "cleanhatch.packs.sec_fundamentals_pit",
    "cleanhatch.packs.ai_rag_ledger_guard",
)
_COMMERCIAL_API_NAMES = (
    "AuditReportBundle",
    "BundleArtifact",
    "LicenseCertificate",
    "LicenseVerificationResult",
    "ReleaseBundleManifest",
    "ReportVerificationResult",
    "build_l1_recipe_bundle",
    "sign_bundle_manifest",
    "sign_license_cert",
    "verify_bundle_manifest",
    "verify_license_cert",
    "verify_report_html",
)


def test_l0_package_tree_has_no_commercial_modules() -> None:
    package_root = Path(cleanhatch.__file__).resolve().parent

    for relative in (
        "license.py",
        "license_client.py",
        "report.py",
        "bundle.py",
        "packs/market_bars_pit.py",
        "packs/sec_fundamentals_pit.py",
        "packs/ai_rag_ledger_guard.py",
    ):
        assert not (package_root / relative).exists(), relative


def test_l0_import_specs_resolve_only_core_modules() -> None:
    for module_name in _COMMERCIAL_MODULES:
        assert importlib.util.find_spec(module_name) is None, module_name


def test_l0_top_level_exports_have_no_commercial_api() -> None:
    for name in _COMMERCIAL_API_NAMES:
        assert name not in cleanhatch.__all__
        assert not hasattr(cleanhatch, name)


def test_l0_help_exposes_core_commands_only(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0
    help_text = capsys.readouterr().out

    for command in ("doctor", "validate", "diff", "demo", "canary", "init-recipe"):
        assert command in help_text
    assert "verify-report" not in help_text
    assert "license" not in help_text


def test_l0_validate_and_diff_reject_report_flags(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["validate", "source.parquet", "--report-html", "audit.html"])
    assert excinfo.value.code == 2
    assert "--report-html" in capsys.readouterr().err

    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "diff",
                "before.parquet",
                "after.parquet",
                "--license-cert",
                "license.cert",
            ]
        )
    assert excinfo.value.code == 2
    assert "--license-cert" in capsys.readouterr().err


def test_l0_doctor_has_no_license_server_section(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)

    exit_code = main(["doctor"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert "license_server" not in payload
    assert "warnings" not in payload


def test_l0_pack_listing_reads_metadata_without_importing_l1(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class UnloadedEntryPoint:
        name = "l1-metadata-only"

        def load(self) -> object:
            raise AssertionError("available_packs must not load plugin code")

    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda *, group: [UnloadedEntryPoint()] if group == "cleanhatch.packs" else [],
    )
    monkeypatch.setattr(packs_module, "_REGISTERED_PACKS", {})

    assert available_packs() == ("l1-metadata-only", "temporal-core")


def test_l0_pack_lookup_reports_install_guidance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda *, group: [])
    monkeypatch.setattr(packs_module, "_REGISTERED_PACKS", {})

    with pytest.raises(PackNotInstalledError, match="cleanhatch-l1"):
        get_pack("market-bars-pit")


def test_l0_cli_reports_install_guidance_for_an_l1_pack(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda *, group: [])
    monkeypatch.setattr(packs_module, "_REGISTERED_PACKS", {})
    source = tmp_path / "sample.csv"
    source.write_text(
        "entity_id,event_time,available_at,value\n"
        "A,2026-01-01T00:00:00Z,2026-01-01T00:00:00Z,1.0\n",
        encoding="utf-8",
    )

    exit_code = main(["validate", str(source), "--pack", "market-bars-pit"])

    assert exit_code == 1
    payload = json.loads(capsys.readouterr().err)
    assert payload["error_code"] == "PACK_NOT_INSTALLED"
    assert "cleanhatch-l1" in payload["message"]


def test_core_import_does_not_load_l1_in_a_fresh_process(tmp_path: Path) -> None:
    script = "\n".join(
        (
            "import sys",
            "import cleanhatch",
            "import cleanhatch.cli",
            "import cleanhatch.packs as packs",
            "packs.available_packs()",
            "assert 'cleanhatch_l1' not in sys.modules, 'L0 import path loaded L1'",
            "assert 'cleanhatch.report' not in sys.modules",
            "assert 'cleanhatch.license' not in sys.modules",
        )
    )

    result = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
