"""Zero-framework command line entry points for the L0 cleanhatch kernel."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import re
import shutil
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import polars as pl
from pydantic import ValidationError

from .__about__ import __version__
from .analysis import run_diff, run_validate, write_ai_summary
from .demo import run_demo_suite
from .errors import CleanHatchError
from .invar import Stage
from .packs import available_packs, get_pack
from .spec import TemporalSpec


def main(argv: Sequence[str] | None = None) -> int:
    """Run one CLI command and return its process exit code."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except CleanHatchError as error:
        print(json.dumps(error.to_dict(), indent=2, ensure_ascii=False), file=sys.stderr)
        return 1
    except (OSError, ValidationError, duckdb.Error, pl.exceptions.PolarsError) as error:
        print(
            json.dumps(
                {"error_type": type(error).__name__, "message": str(error)},
                indent=2,
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cleanhatch",
        description="Point-in-time, causal, invariant, and lineage gates.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="inspect runtime and local OpSec setup")
    doctor.set_defaults(handler=_doctor)

    validate = subparsers.add_parser("validate", help="run an invariant gate on a source")
    add_validate_arguments(validate)
    validate.set_defaults(handler=_validate)

    diff = subparsers.add_parser("diff", help="compare two temporal snapshots")
    add_diff_arguments(diff)
    diff.set_defaults(handler=_diff)

    demo = subparsers.add_parser(
        "demo",
        help="run the five silent-killer showcase scenarios",
    )
    demo.add_argument("--json", action="store_true", help="emit structured JSON")
    demo.set_defaults(handler=_demo)

    init_recipe = subparsers.add_parser(
        "init-recipe",
        help="write a declarative configuration recipe for one pack",
    )
    init_recipe.add_argument("pack")
    init_recipe.add_argument(
        "--out",
        "-o",
        required=True,
        help="target directory (a recipe file name is derived) or a .py file",
    )
    init_recipe.add_argument(
        "--agents-md",
        action="store_true",
        help="also write an AGENTS.md guardrail file",
    )
    init_recipe.set_defaults(handler=_init_recipe)

    canary = subparsers.add_parser(
        "canary",
        help="run pack canary suites and report the intercept rate",
    )
    canary.add_argument(
        "--pack",
        default=None,
        help="run one pack (default: every installed or registered pack)",
    )
    canary.set_defaults(handler=_canary)

    return parser


def add_validate_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the shared ``validate`` arguments for the L0 and L1 CLIs."""

    parser.add_argument("source", help="Parquet, CSV, JSON, glob, or directory")
    parser.add_argument("--as-of", type=parse_as_of, default=None)
    add_spec_arguments(parser)
    parser.add_argument(
        "--stage",
        choices=[stage.value for stage in Stage],
        default=Stage.CLEAN.value,
    )
    parser.add_argument("--value-col", default="value")
    parser.add_argument("--ai-summary", type=Path, default=None)
    parser.add_argument(
        "--no-drift-baseline",
        action="store_true",
        help="skip the automated rolling PIT drift baseline at the clean stage",
    )
    parser.add_argument(
        "--baseline-windows",
        type=int,
        default=20,
        help="number of past PIT windows used for the clean-stage drift baseline",
    )
    parser.add_argument(
        "--baseline-window-hours",
        type=float,
        default=24.0,
        help="width of each rolling baseline window in hours",
    )
    parser.add_argument(
        "--pack",
        default=None,
        help="run one installed or registered industry gate pack",
    )


def add_diff_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the shared ``diff`` arguments for the L0 and L1 CLIs."""

    parser.add_argument("source_a")
    parser.add_argument("source_b")
    parser.add_argument("--as-of", type=parse_as_of, default=None)
    add_spec_arguments(parser)


def add_spec_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the domain-neutral temporal coordinate arguments."""

    parser.add_argument("--entity-col", default="entity_id")
    parser.add_argument("--event-time-col", default="event_time")
    parser.add_argument("--knowledge-time-col", default="available_at")
    parser.add_argument("--decision-time-col", default="decision_time")


def spec_from_args(args: argparse.Namespace) -> TemporalSpec:
    """Build a :class:`TemporalSpec` from parsed coordinate arguments."""

    return TemporalSpec(
        entity_col=args.entity_col,
        event_time_col=args.event_time_col,
        knowledge_time_col=args.knowledge_time_col,
        decision_time_col=args.decision_time_col,
    )


def _doctor(_args: argparse.Namespace) -> int:
    python_ok = sys.version_info >= (3, 11)
    packages = _package_report()
    packages_ok = all(bool(item["ok"]) for item in packages.values())
    git = _git_opsec()
    if not bool(git["ok"]):
        git["remediation"] = (
            "review `git log --format=%ae`; fix the local identity with "
            f'`git config user.email "{_APPROVED_USER_NOREPLY}"` and install the hook '
            "with `git config core.hooksPath .githooks`"
        )
    report = {
        "python": {
            "version": platform.python_version(),
            "required": ">=3.11",
            "supported": python_ok,
            "remediation": (
                None
                if python_ok
                else "install Python 3.11 or newer from https://www.python.org/downloads/"
            ),
        },
        "packages": packages,
        "git_opsec": git,
        "ok": python_ok and packages_ok and bool(git["ok"]),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["ok"] else 1


_REQUIRED_PACKAGES: dict[str, str] = {
    "polars": "1.0.0",
    "duckdb": "1.0.0",
    "pydantic": "2.7.0",
}


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _version_tuple(value: str) -> tuple[int, int, int]:
    core = value.split("+", 1)[0].split("-", 1)[0]
    parts = [int(part) for part in re.findall(r"\d+", core)[:3]]
    padded = parts + [0, 0, 0]
    return padded[0], padded[1], padded[2]


def _package_report() -> dict[str, dict[str, object]]:
    report: dict[str, dict[str, object]] = {}
    for name, required in _REQUIRED_PACKAGES.items():
        found = _package_version(name)
        ok = found is not None and _version_tuple(found) >= _version_tuple(required)
        report[name] = {
            "required": f">={required}",
            "found": found,
            "ok": ok,
            "remediation": (
                None if ok else f'python -m pip install --upgrade "{name}>={required}"'
            ),
        }
    return report


# Assemble the first fragment from pieces so the public source does not embed
# the raw personal identifier that this OpSec check is meant to reject.
_FORBIDDEN_EMAIL_FRAGMENTS = ("jal" + "an", "qq.com", "163.com")
_OPSE_EMAIL_DOMAIN = "@cleanhatch.dev"

# Exact, user-approved public identities.  Keep this as an exact allowlist:
# a domain-wide users.noreply.github.com rule would admit unrelated accounts.
_APPROVED_USER_NOREPLY = "354919554+uncio-lintel@users.noreply.github.com"
_APPROVED_PUBLIC_EMAILS = frozenset(
    {
        _APPROVED_USER_NOREPLY,
        "noreply@github.com",  # GitHub Web-flow committer, not a personal mailbox
    }
)


def _email_is_compliant(email: str | None) -> bool:
    if not email:
        return False
    lower = email.lower()
    if lower in _APPROVED_PUBLIC_EMAILS:
        return True
    return lower.endswith(_OPSE_EMAIL_DOMAIN) and not any(
        fragment in lower for fragment in _FORBIDDEN_EMAIL_FRAGMENTS
    )


def _hook_policy_is_current(hook_path: Path) -> bool:
    try:
        content = hook_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return _OPSE_EMAIL_DOMAIN in content and "GIT_AUTHOR_EMAIL" in content


def _git_opsec() -> dict[str, object]:
    git = shutil.which("git")
    if git is None:
        return {"repository": False, "ok": True, "reason": "git is not installed"}

    root_result = _run_git(git, ["rev-parse", "--show-toplevel"])
    if root_result.returncode != 0:
        return {"repository": False, "ok": True}
    root = Path(root_result.stdout.strip())

    name = _git_value(git, root, ["config", "user.name"])
    email = _git_value(git, root, ["config", "user.email"])
    hook_result = _run_git(git, ["-C", str(root), "rev-parse", "--git-path", "hooks/pre-commit"])
    hook_candidates: list[Path] = []
    if hook_result.returncode == 0:
        candidate = Path(hook_result.stdout.strip())
        if not candidate.is_absolute():
            candidate = root / candidate
        hook_candidates.append(candidate)
    configured_hooks_path = _git_value(git, root, ["config", "core.hooksPath"])
    if configured_hooks_path is not None:
        hooks_dir = Path(configured_hooks_path)
        if not hooks_dir.is_absolute():
            hooks_dir = root / hooks_dir
        hook_candidates.append(hooks_dir / "pre-commit")
    hook_path = next((path for path in hook_candidates if path.is_file()), None)
    hook_present = hook_path is not None
    hook_policy_ok = hook_present and hook_path is not None and _hook_policy_is_current(hook_path)

    history_result = _run_git(git, ["-C", str(root), "log", "--format=%ae%n%ce"])
    history_emails = (
        [line.strip() for line in history_result.stdout.splitlines() if line.strip()]
        if history_result.returncode == 0
        else []
    )
    non_compliant = sorted(
        {address for address in history_emails if not _email_is_compliant(address)}
    )

    email_ok = _email_is_compliant(email)
    history_ok = not non_compliant
    return {
        "repository": True,
        "root": str(root),
        "user_name": name,
        "user_email": email,
        "email_ok": email_ok,
        "pre_commit_hook": hook_present,
        "pre_commit_hook_policy_ok": hook_policy_ok,
        "history_authors_checked": len(history_emails),
        "history_authors_ok": history_ok,
        "non_compliant_author_emails": non_compliant[:10],
        "non_compliant_author_email_count": len(non_compliant),
        "ok": email_ok and hook_policy_ok and history_ok,
    }


def _git_value(git: str, root: Path, arguments: Sequence[str]) -> str | None:
    result = _run_git(git, ["-C", str(root), *arguments])
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else None


def _run_git(git: str, arguments: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [git, *arguments],
        capture_output=True,
        text=True,
        check=False,
    )


def _validate(args: argparse.Namespace) -> int:
    outcome = run_validate(
        args.source,
        spec=spec_from_args(args),
        as_of=args.as_of,
        stage=Stage(args.stage),
        pack=args.pack,
        value_col=args.value_col,
        no_drift_baseline=args.no_drift_baseline,
        baseline_windows=args.baseline_windows,
        baseline_window_hours=args.baseline_window_hours,
    )
    print(json.dumps(outcome.report.to_dict(), indent=2, ensure_ascii=False, default=str))
    if args.ai_summary is not None:
        write_ai_summary(outcome.manifest, args.ai_summary)
    return 0 if outcome.report.passed else 2


def _init_recipe(args: argparse.Namespace) -> int:
    pack = get_pack(args.pack)
    raw = str(args.out)
    target = Path(raw)
    is_directory = target.is_dir() or raw.endswith(("/", "\\"))
    agents_path: Path | None = None
    if is_directory:
        target.mkdir(parents=True, exist_ok=True)
        recipe_path = target / f"{args.pack.replace('-', '_')}_recipe.py"
        if args.agents_md:
            agents_path = target / "AGENTS.md"
    else:
        if target.suffix.lower() != ".py":
            raise CleanHatchError("--out must be an existing directory or a .py file path")
        recipe_path = target
        recipe_path.parent.mkdir(parents=True, exist_ok=True)
        if args.agents_md:
            agents_path = recipe_path.parent / "AGENTS.md"
    recipe_path.write_text(pack.render_recipe_scaffold(), encoding="utf-8")
    if agents_path is not None:
        agents_path.write_text(pack.render_agents_md(), encoding="utf-8")
    print(
        json.dumps(
            {
                "pack": args.pack,
                "recipe": str(recipe_path),
                "agents_md": str(agents_path) if agents_path is not None else None,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def _canary(args: argparse.Namespace) -> int:
    if args.pack is not None:
        suite = get_pack(args.pack).run_canary_suite()
        payload: dict[str, object] = suite.to_dict()
        all_caught = suite.all_caught
    else:
        suites = [get_pack(name).run_canary_suite() for name in available_packs()]
        all_caught = all(suite.all_caught for suite in suites)
        payload = {
            "all_caught": all_caught,
            "packs": [suite.to_dict() for suite in suites],
        }
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    return 0 if all_caught else 1


def _demo(args: argparse.Namespace) -> int:
    result = run_demo_suite()
    print(result.to_json() if args.json else result.to_markdown())
    return 0 if result.all_caught else 1


def _diff(args: argparse.Namespace) -> int:
    outcome = run_diff(
        args.source_a,
        args.source_b,
        spec=spec_from_args(args),
        as_of=args.as_of,
    )
    print(json.dumps(outcome.report.to_dict(), indent=2, ensure_ascii=False, default=str))
    return 0


def parse_as_of(value: str) -> datetime:
    """Parse a timezone-aware ISO timestamp for ``--as-of``."""

    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid ISO timestamp: {value!r}") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError("--as-of must include a timezone offset")
    return parsed.astimezone(UTC)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "add_diff_arguments",
    "add_spec_arguments",
    "add_validate_arguments",
    "main",
    "parse_as_of",
    "spec_from_args",
]
