"""Compare matched cleanhatch, GX, and Pandera checks on the same bar rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from importlib.metadata import version
from pathlib import Path

import great_expectations as gx
import pandas as pd
import pandera.pandas as pa
import polars as pl
from great_expectations.expectations.core.expect_column_values_to_be_between import (
    ExpectColumnValuesToBeBetween,
)
from great_expectations.expectations.core.expect_column_values_to_not_be_null import (
    ExpectColumnValuesToNotBeNull,
)
from great_expectations.expectations.core.expect_compound_columns_to_be_unique import (
    ExpectCompoundColumnsToBeUnique,
)

from cleanhatch.invar import GateReport, InvarGate, InvarRule, Stage
from cleanhatch.packs import GatePack, PackNotInstalledError, get_pack
from cleanhatch.spec import TemporalSpec
from examples.repro.market_bars.demo import (
    inject_negative_value,
    inject_pre_close_availability,
    load_market_bars,
)

_DEFAULT_ARCHIVE = Path(__file__).resolve().parents[1] / "data" / "BTCUSDT-1h-2024-01.zip"
_DEFAULT_RUNS = 5
_PreparedFrames = tuple[pl.DataFrame, pd.DataFrame]
_Validator = Callable[[_PreparedFrames], bool]


def _common_frame(bars: pl.DataFrame) -> tuple[pl.DataFrame, pd.DataFrame]:
    cleanhatch_frame = bars.select(
        "symbol",
        "decision_time",
        pl.col("close").alias("value"),
    )
    pandas_frame = pd.DataFrame(cleanhatch_frame.to_dict(as_series=False))
    return cleanhatch_frame, pandas_frame


def _cleanhatch_validator() -> _Validator:
    spec = TemporalSpec(
        entity_col="symbol",
        decision_time_col="decision_time",
    )
    gate = InvarGate(spec, max_null_ratio=0.0)
    rule = InvarRule(
        name="positive_value",
        predicate=pl.col("value") > 0,
        message="value must be greater than zero",
    )

    def validate(frames: _PreparedFrames) -> bool:
        report = gate.check_feature(
            frames[0],
            value_col="value",
            rules=(rule,),
            raise_on_failure=False,
        )
        return report.passed

    return validate


def _great_expectations_validator() -> _Validator:
    os.environ["GX_ANALYTICS_ENABLED"] = "False"
    context = gx.get_context(mode="ephemeral")
    context.variables.progress_bars = {  # pyright: ignore[reportAttributeAccessIssue]
        "globally": False,
        "metric_calculations": False,
    }
    data_source = context.data_sources.add_pandas(name="market_bars_pandas")
    asset = data_source.add_dataframe_asset(name="market_bars")
    batch_definition = asset.add_batch_definition_whole_dataframe(name="market_bars_batch")
    suite = gx.ExpectationSuite(name="market_bars_common_rules")
    suite.add_expectation(ExpectColumnValuesToNotBeNull(column="value"))
    suite.add_expectation(
        ExpectColumnValuesToBeBetween(
            column="value",
            min_value=0.0,
            max_value=sys.float_info.max,
            strict_min=True,
        )
    )
    suite.add_expectation(ExpectCompoundColumnsToBeUnique(column_list=["symbol", "decision_time"]))
    context.suites.add(suite)
    validation = gx.ValidationDefinition(
        name="market_bars_common_validation",
        data=batch_definition,
        suite=suite,
    )
    context.validation_definitions.add(validation)

    def validate(frames: _PreparedFrames) -> bool:
        result = validation.run(batch_parameters={"dataframe": frames[1]})
        return bool(result.success)

    return validate


def _pandera_validator() -> _Validator:
    schema = pa.DataFrameSchema(
        {
            "symbol": pa.Column(nullable=True),
            "decision_time": pa.Column(nullable=True),
            "value": pa.Column(
                float,
                checks=[
                    pa.Check.gt(0),
                    pa.Check.le(sys.float_info.max),
                ],
                nullable=False,
            ),
        },
        unique=["symbol", "decision_time"],
    )

    def validate(frames: _PreparedFrames) -> bool:
        try:
            schema.validate(frames[1], lazy=True)
        except (pa.errors.SchemaError, pa.errors.SchemaErrors):
            return False
        return True

    return validate


def _timed_median(
    validate: _Validator,
    frames: _PreparedFrames,
    *,
    expected: bool,
    runs: int,
) -> dict[str, object]:
    warmed = validate(frames)
    if warmed is not expected:
        raise AssertionError(
            f"validation warm-up mismatch: expected passed={expected}, got {warmed}"
        )
    durations: list[float] = []
    for _ in range(runs):
        started = time.perf_counter()
        passed = validate(frames)
        durations.append((time.perf_counter() - started) * 1000.0)
        if passed is not expected:
            raise AssertionError(
                f"validation verdict mismatch: expected passed={expected}, got {passed}"
            )
    return {
        "verdict": "PASSED" if expected else "BLOCKED",
        "median_ms": round(statistics.median(durations), 4),
        "runs": runs,
    }


def _l1_report(pack: GatePack, frame: pl.DataFrame) -> GateReport:
    return pack.validate(frame, stage=Stage.INGEST, raise_on_failure=False)


def _timed_l1_case(
    pack: GatePack,
    frame: pl.DataFrame,
    *,
    expected_failures: frozenset[str],
    exact_failures: bool,
    runs: int,
) -> dict[str, object]:
    def validate() -> GateReport:
        return _l1_report(pack, frame)

    def check(report: GateReport) -> bool:
        failures = frozenset(result.name for result in report.checks if result.status == "FAILED")
        if not expected_failures:
            return report.passed and not failures
        if report.passed or not expected_failures.issubset(failures):
            return False
        return not exact_failures or failures == expected_failures

    warmed = validate()
    if not check(warmed):
        raise AssertionError(
            f"L1 validation warm-up mismatch: expected failures "
            f"{sorted(expected_failures)}, got "
            f"{[result.name for result in warmed.checks if result.status == 'FAILED']}"
        )

    durations: list[float] = []
    report = warmed
    for _ in range(runs):
        started = time.perf_counter()
        report = validate()
        durations.append((time.perf_counter() - started) * 1000.0)
        if not check(report):
            raise AssertionError(
                f"L1 validation verdict mismatch: expected failures "
                f"{sorted(expected_failures)}, got "
                f"{[result.name for result in report.checks if result.status == 'FAILED']}"
            )

    failures = [result.name for result in report.checks if result.status == "FAILED"]
    warnings = [result.name for result in report.checks if result.status == "WARNING"]
    return {
        "verdict": "BLOCKED"
        if failures
        else "CONTINUE"
        if report.status == "WARNING"
        else "PASSED",
        "status": report.status,
        "failed_checks": failures,
        "warnings": warnings,
        "median_ms": round(statistics.median(durations), 4),
        "runs": runs,
    }


def _l1_pack_comparison(bars: pl.DataFrame, *, runs: int) -> dict[str, object]:
    pack = get_pack("market-bars-pit")
    cases = {
        "clean": (bars, frozenset(), True),
        "negative_close": (
            inject_negative_value(bars, "close"),
            frozenset({"POSITIVE_FINITE_PRICES"}),
            False,
        ),
        "pre_close_availability": (
            inject_pre_close_availability(bars),
            frozenset({"BAR_CLOSE_CAUSALITY"}),
            True,
        ),
    }
    results = {
        name: _timed_l1_case(
            pack,
            frame,
            expected_failures=expected_failures,
            exact_failures=exact_failures,
            runs=runs,
        )
        for name, (frame, expected_failures, exact_failures) in cases.items()
    }
    return {
        "pack": pack.name,
        "scope": "full L1 market-bars-pit pack; separate from matched GE/Pandera rules",
        "results": results,
    }


def run_comparison(archive_path: Path, *, runs: int, with_l1: bool = False) -> dict[str, object]:
    bars = load_market_bars(archive_path)
    clean_pl, clean_pd = _common_frame(bars)
    bad_pl = inject_negative_value(clean_pl, "value")
    bad_pd = pd.DataFrame(bad_pl.to_dict(as_series=False))

    clean_frames = (clean_pl, clean_pd)
    bad_frames = (bad_pl, bad_pd)
    validators: Sequence[tuple[str, _Validator]] = (
        ("Cleanhatch L0", _cleanhatch_validator()),
        ("Great Expectations", _great_expectations_validator()),
        ("Pandera", _pandera_validator()),
    )
    results: dict[str, dict[str, dict[str, object]]] = {"clean": {}, "negative_close": {}}
    for name, validate in validators:
        results["clean"][name] = _timed_median(
            validate,
            clean_frames,
            expected=True,
            runs=runs,
        )
        results["negative_close"][name] = _timed_median(
            validate,
            bad_frames,
            expected=False,
            runs=runs,
        )

    versions = {
        "python": platform.python_version(),
        "cleanhatch": version("cleanhatch"),
        "polars": version("polars"),
        "great-expectations": version("great-expectations"),
        "pandas": version("pandas"),
        "pandera": version("pandera"),
    }
    output: dict[str, object] = {
        "dataset": archive_path.name,
        "archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        "rows": bars.height,
        "common_input_columns": ["symbol", "decision_time", "value=close"],
        "matched_rules": [
            "value is non-null and finite",
            "value > 0",
            "(symbol, decision_time) is unique",
        ],
        "timing_scope": (
            "one untimed warm-up per validator and case; measured median covers "
            "validation calls only, while ZIP reading, normalization, "
            "Polars-to-pandas conversion, rule construction, and Python/library "
            "startup are excluded"
        ),
        "execution": "serial; each validator sees the same prepared clean and poisoned rows",
        "great_expectations_analytics": "disabled",
        "versions": versions,
        "results": results,
    }
    if with_l1:
        output["l1_full_pack"] = _l1_pack_comparison(bars, runs=runs)
        versions["cleanhatch-l1"] = version("cleanhatch-l1")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=_DEFAULT_ARCHIVE)
    parser.add_argument("--runs", type=int, default=_DEFAULT_RUNS)
    parser.add_argument(
        "--with-l1",
        action="store_true",
        help="also time the full installed L1 market-bars-pit pack separately",
    )
    args = parser.parse_args(argv)
    if args.runs < 1:
        parser.error("--runs must be positive")

    try:
        result = run_comparison(args.zip, runs=args.runs, with_l1=args.with_l1)
    except PackNotInstalledError as error:
        print(error, file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
