"""Assert that built L0/L1 artifacts respect the package boundary.

Usage:
    python scripts/check_dist.py l0 dist-l0
    python scripts/check_dist.py l1 dist-l1
"""

from __future__ import annotations

import argparse
import sys
import tarfile
import zipfile
from pathlib import Path

_COMMERCIAL_MODULE_SUFFIXES = (
    "license.py",
    "license_client.py",
    "report.py",
    "bundle.py",
    "market_bars_pit.py",
    "sec_fundamentals_pit.py",
    "ai_rag_ledger_guard.py",
)
_COMMERCIAL_IDENTIFIERS = (
    b"class MarketBarsPack",
    b"class SecFundamentalsPack",
    b"class AiRagLedgerPack",
    b"class LicenseCertificate",
    b"class AuditReportBundle",
    b"def build_l1_recipe_bundle",
)
_L1_REQUIRED = (
    "cleanhatch_l1/__init__.py",
    "cleanhatch_l1/cli.py",
    "cleanhatch_l1/license.py",
    "cleanhatch_l1/license_client.py",
    "cleanhatch_l1/report.py",
    "cleanhatch_l1/bundle.py",
    "cleanhatch_l1/packs/market_bars_pit.py",
    "cleanhatch_l1/packs/sec_fundamentals_pit.py",
    "cleanhatch_l1/packs/ai_rag_ledger_guard.py",
)


def _wheel_members(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as archive:
        return {name: archive.read(name) for name in archive.namelist() if not name.endswith("/")}


def _sdist_members(path: Path) -> dict[str, bytes]:
    with tarfile.open(path) as archive:
        members: dict[str, bytes] = {}
        for entry in archive.getmembers():
            if not entry.isfile():
                continue
            extracted = archive.extractfile(entry)
            if extracted is None:
                continue
            members[entry.name] = extracted.read()
        return members


def _strip_prefix(names: list[str]) -> list[str]:
    prefixes = {name.split("/", 1)[0] for name in names if "/" in name}
    if len(prefixes) == 1 and all("/" in name for name in names):
        prefix = prefixes.pop() + "/"
        return [name[len(prefix) :] for name in names]
    return names


def _check_no_forbidden(names: list[str], blobs: dict[str, bytes]) -> None:
    for name in names:
        assert not name.startswith(("cleanhatch_l1/", "packages/l1/")), name
        assert "/tests/private/" not in f"/{name}", name
        assert not name.startswith(("benchmarks/", "site/", "edge/", "docs/", "reffer/")), name
        assert not name.endswith(_COMMERCIAL_MODULE_SUFFIXES), name
    for name, blob in blobs.items():
        if not name.startswith("cleanhatch/"):
            continue
        for identifier in _COMMERCIAL_IDENTIFIERS:
            assert identifier not in blob, f"{identifier!r} leaked into {name}"


def check_l0(dist_dir: Path) -> None:
    artifacts = sorted(dist_dir.iterdir())
    assert artifacts, f"{dist_dir} is empty"
    wheels = [path for path in artifacts if path.suffix == ".whl"]
    sdists = [path for path in artifacts if path.name.endswith(".tar.gz")]
    assert len(wheels) == 1, wheels
    assert len(sdists) == 1, sdists

    wheel_members = _wheel_members(wheels[0])
    wheel_names = list(wheel_members)
    _check_no_forbidden(wheel_names, wheel_members)
    assert "cleanhatch/cli.py" in wheel_names
    assert "cleanhatch/analysis.py" in wheel_names
    assert "cleanhatch/packs/temporal_core.py" in wheel_names
    assert any(name.endswith("dist-info/licenses/LICENSE") for name in wheel_names)
    assert not any(
        name.endswith("dist-info/licenses/LICENSE-COMMERCIAL.md") for name in wheel_names
    )

    sdist_members = _sdist_members(sdists[0])
    sdist_names = _strip_prefix(list(sdist_members))
    sdist_blobs = {
        name.split("/", 1)[1]: blob for name, blob in sdist_members.items() if "/" in name
    }
    _check_no_forbidden(sdist_names, sdist_blobs)
    assert "cleanhatch/cli.py" in sdist_names
    assert "cleanhatch/analysis.py" in sdist_names
    assert "tests/test_l0_boundary.py" in sdist_names
    assert "tests/test_contribution_flow.py" in sdist_names
    assert "tests/private/test_site_and_geo.py" not in sdist_names
    for required in (
        "CONTRIBUTING.md",
        "README_PUBLIC.md",
        ".githooks/pre-commit",
        ".github/PULL_REQUEST_TEMPLATE.md",
        ".github/ISSUE_TEMPLATE/bug_report.yml",
        ".github/ISSUE_TEMPLATE/config.yml",
        ".github/ISSUE_TEMPLATE/silent_killer_pitfall.yml",
    ):
        assert required in sdist_names, required
    assert not any(name.startswith(".github/workflows/") for name in sdist_names)
    assert "LICENSE" in sdist_names
    assert "MIGRATION.md" in sdist_names
    assert "pyproject.toml" in sdist_names
    print(f"L0 OK: {wheels[0].name}, {sdists[0].name}")


def check_l1(dist_dir: Path) -> None:
    artifacts = sorted(dist_dir.iterdir())
    assert artifacts, f"{dist_dir} is empty"
    wheels = [path for path in artifacts if path.suffix == ".whl"]
    sdists = [path for path in artifacts if path.name.endswith(".tar.gz")]
    assert len(wheels) == 1, wheels
    assert len(sdists) == 1, sdists

    wheel_members = _wheel_members(wheels[0])
    wheel_names = list(wheel_members)
    for required in _L1_REQUIRED:
        assert required in wheel_names, required
    assert not any(name.startswith("cleanhatch/") for name in wheel_names)
    assert any(name.endswith("dist-info/licenses/LICENSE") for name in wheel_names)

    metadata_name = next(name for name in wheel_names if name.endswith(".dist-info/METADATA"))
    metadata = wheel_members[metadata_name].decode("utf-8")
    assert "Private :: Do Not Upload" in metadata
    assert "License :: Other/Proprietary License" in metadata
    assert "License :: OSI Approved :: Apache Software License" not in metadata

    entry_points_name = next(
        name for name in wheel_names if name.endswith(".dist-info/entry_points.txt")
    )
    entry_points = wheel_members[entry_points_name].decode("utf-8")
    for pack in ("market-bars-pit", "sec-fundamentals-pit", "ai-rag-ledger-guard"):
        assert f"{pack} = " in entry_points, pack
    assert "cleanhatch-l1 = " in entry_points

    sdist_members = _sdist_members(sdists[0])
    sdist_names = _strip_prefix(list(sdist_members))
    for required in _L1_REQUIRED:
        assert f"src/{required}" in sdist_names, required
    assert "tests/test_report_and_license.py" in sdist_names
    assert "LICENSE" in sdist_names
    assert "README.md" in sdist_names
    print(f"L1 OK: {wheels[0].name}, {sdists[0].name}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("layer", choices=("l0", "l1"))
    parser.add_argument("dist_dir", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.layer == "l0":
            check_l0(args.dist_dir)
        else:
            check_l1(args.dist_dir)
    except AssertionError as error:
        print(f"distribution boundary check failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
