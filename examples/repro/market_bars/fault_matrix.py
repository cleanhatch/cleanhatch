"""Construct and run one deterministic input for every L1 market-bars check."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import cast

import polars as pl

from cleanhatch.invar import Stage
from cleanhatch.packs import GatePack, PackNotInstalledError, get_pack
from examples.repro.market_bars.demo import load_market_bars

_DEFAULT_ARCHIVE = Path(__file__).with_name("data") / "BTCUSDT-1h-2024-01.zip"
_FAULT_ROW = 40
_SPIKE_ROW = 100


@dataclass(frozen=True, slots=True)
class FaultScenario:
    """One constructed input and its expected status for a single pack check."""

    check_name: str
    construction: str
    frame: pl.DataFrame
    expected_check_status: str
    expected_failed_checks: frozenset[str] = frozenset()
    exact_failed_checks: bool = True
    evidence_time: datetime | None = None


@dataclass(frozen=True, slots=True)
class FaultOutcome:
    """Observed check result and its construction evidence."""

    check_name: str
    construction: str
    expected_check_status: str
    actual_check_status: str
    gate_verdict: str
    failed_checks: tuple[str, ...]
    target_details: dict[str, object]
    matched_injected_spike: bool | None
    passed: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _set_row(
    frame: pl.DataFrame,
    row_index: int,
    values: dict[str, pl.Expr | float | int | bool],
) -> pl.DataFrame:
    selected = pl.int_range(0, pl.len()) == row_index
    return frame.with_columns(
        pl.when(selected).then(value).otherwise(pl.col(column)).alias(column)
        for column, value in values.items()
    )


def build_fault_scenarios(bars: pl.DataFrame) -> tuple[FaultScenario, ...]:
    """Create 12 reproducible single-purpose cases from the bundled sample."""

    if bars.height <= _SPIKE_ROW:
        raise ValueError(f"fault examples require at least {_SPIKE_ROW + 1} rows")

    open_ = pl.col("open")
    close = pl.col("close")
    volume = pl.col("volume")
    minimum = pl.min_horizontal(open_, close)
    maximum = pl.max_horizontal(open_, close)

    high_below_close = _set_row(
        bars,
        _FAULT_ROW,
        {
            "high": minimum - 1.0,
            "low": minimum - 2.0,
            "quote_volume": volume * (minimum - 1.5),
        },
    )
    low_above_close = _set_row(
        bars,
        _FAULT_ROW,
        {
            "high": maximum + 2.0,
            "low": minimum + 1.0,
            "quote_volume": volume * (minimum + maximum + 3.0) / 2.0,
        },
    )
    high_below_low = _set_row(
        bars,
        _FAULT_ROW,
        {
            "open": pl.lit(100.0),
            "close": pl.lit(100.0),
            "high": pl.lit(99.0),
            "low": pl.lit(101.0),
            "quote_volume": volume * 100.0,
        },
    )
    non_positive_price = _set_row(bars, _FAULT_ROW, {"low": pl.lit(0.0)})
    negative_volume = _set_row(
        bars,
        _FAULT_ROW,
        {"volume": pl.lit(-1.0), "quote_volume": -close},
    )
    non_flat_halt = _set_row(bars, _FAULT_ROW, {"is_trading": pl.lit(False)})
    out_of_envelope_vwap = _set_row(
        bars,
        _FAULT_ROW,
        {"quote_volume": volume * pl.col("high") * 1.01},
    )
    before_close = _set_row(
        bars,
        _FAULT_ROW,
        {
            "available_at": pl.col("bar_close_time") - pl.duration(milliseconds=1),
            "decision_time": pl.col("bar_close_time") - pl.duration(milliseconds=1),
        },
    )
    duplicate_coordinate = pl.concat([bars, bars.slice(_FAULT_ROW, 1)])
    warmup_only = bars.head(10)

    dead_values = bars.with_columns(
        pl.lit(100.0).alias("open"),
        pl.lit(100.0).alias("high"),
        pl.lit(100.0).alias("low"),
        pl.lit(100.0).alias("close"),
        pl.lit(1_000.0).alias("volume"),
        pl.lit(100_000.0).alias("quote_volume"),
        pl.lit(True).alias("is_trading"),
    )

    spike_close = close * 2.0
    spike_high = pl.max_horizontal(open_, spike_close) + 1.0
    spike_low = pl.min_horizontal(open_, spike_close) - 1.0
    rolling_spike = _set_row(
        bars,
        _SPIKE_ROW,
        {
            "close": spike_close,
            "high": spike_high,
            "low": spike_low,
            "quote_volume": volume * (spike_high + spike_low) / 2.0,
        },
    )
    spike_time = cast(datetime, bars["bar_close_time"][_SPIKE_ROW])

    return (
        FaultScenario(
            "OHLC_HIGH_GE_OPEN_CLOSE",
            f"sorted row {_FAULT_ROW}: high := min(open, close)-1; "
            "low := min(open, close)-2; quote_volume reset inside the new range",
            high_below_close,
            "FAILED",
            frozenset({"OHLC_HIGH_GE_OPEN_CLOSE"}),
        ),
        FaultScenario(
            "OHLC_LOW_LE_OPEN_CLOSE",
            f"sorted row {_FAULT_ROW}: low := min(open, close)+1; "
            "high := max(open, close)+2; quote_volume reset inside the new range",
            low_above_close,
            "FAILED",
            frozenset({"OHLC_LOW_LE_OPEN_CLOSE"}),
        ),
        FaultScenario(
            "OHLC_HIGH_GE_LOW",
            f"sorted row {_FAULT_ROW}: open=close=100, high=99, low=101; "
            "impossible range also violates related high/open and low/open rules",
            high_below_low,
            "FAILED",
            frozenset(
                {
                    "OHLC_HIGH_GE_LOW",
                    "OHLC_HIGH_GE_OPEN_CLOSE",
                    "OHLC_LOW_LE_OPEN_CLOSE",
                    "VWAP_ENVELOPE",
                }
            ),
        ),
        FaultScenario(
            "POSITIVE_FINITE_PRICES",
            f"sorted row {_FAULT_ROW}: low := 0",
            non_positive_price,
            "FAILED",
            frozenset({"POSITIVE_FINITE_PRICES"}),
        ),
        FaultScenario(
            "NON_NEGATIVE_FINITE_VOLUMES",
            f"sorted row {_FAULT_ROW}: volume := -1 and quote_volume := -close",
            negative_volume,
            "FAILED",
            frozenset({"NON_NEGATIVE_FINITE_VOLUMES"}),
        ),
        FaultScenario(
            "HALT_ZERO_AMPLITUDE",
            f"sorted row {_FAULT_ROW}: is_trading := false while its original "
            "OHLC range remains non-flat",
            non_flat_halt,
            "FAILED",
            frozenset({"HALT_ZERO_AMPLITUDE"}),
        ),
        FaultScenario(
            "VWAP_ENVELOPE",
            f"sorted row {_FAULT_ROW}: quote_volume := volume * high * 1.01, "
            "exceeding the 1 bps envelope tolerance",
            out_of_envelope_vwap,
            "FAILED",
            frozenset({"VWAP_ENVELOPE"}),
        ),
        FaultScenario(
            "BAR_CLOSE_CAUSALITY",
            f"sorted row {_FAULT_ROW}: available_at and decision_time := bar_close_time-1ms",
            before_close,
            "FAILED",
            frozenset({"BAR_CLOSE_CAUSALITY"}),
        ),
        FaultScenario(
            "UNIQUE_BAR_COORDINATE",
            f"append a duplicate of sorted row {_FAULT_ROW} with the same timestamp, "
            "symbol, and availability coordinate",
            duplicate_coordinate,
            "FAILED",
            frozenset({"UNIQUE_BAR_COORDINATE"}),
        ),
        FaultScenario(
            "WARMUP_WINDOW_SKIP",
            "use the first 10 rows; the default 10-bar warm-up skips every row",
            warmup_only,
            "SKIPPED",
        ),
        FaultScenario(
            "ACTIVE_DEAD_VALUE_RUN",
            "all rows: OHLC=100, volume=1000, quote_volume=100000, "
            "is_trading=true; constant active prices create a dead-value run",
            dead_values,
            "WARNING",
        ),
        FaultScenario(
            "ROLLING_MAD_SPIKE",
            f"sorted row {_SPIKE_ROW}: double close and expand high/low to keep "
            "OHLC valid; reset quote_volume to the midpoint and verify this "
            "timestamp appears in spike_rows",
            rolling_spike,
            "WARNING",
            evidence_time=spike_time,
        ),
    )


def evaluate_fault_scenario(scenario: FaultScenario, pack: GatePack) -> FaultOutcome:
    report = pack.validate(scenario.frame, stage=Stage.INGEST, raise_on_failure=False)
    target = next(check for check in report.checks if check.name == scenario.check_name)
    failed = tuple(check.name for check in report.checks if check.status == "FAILED")
    detail_evidence: dict[str, object] = dict(target.details)

    matched_spike: bool | None = None
    if scenario.evidence_time is not None:
        spike_rows = cast(list[dict[str, object]], target.details.get("spike_rows", []))
        matching_rows = [
            row for row in spike_rows if row.get("bar_close_time") == scenario.evidence_time
        ]
        matched_spike = bool(matching_rows)
        detail_evidence["injected_spike_rows"] = matching_rows

    expected_failures_seen = scenario.expected_failed_checks.issubset(failed)
    if scenario.exact_failed_checks:
        expected_failures_seen = expected_failures_seen and (
            frozenset(failed) == scenario.expected_failed_checks
        )
    gate_verdict = (
        "BLOCKED" if not report.passed else "CONTINUE" if report.status == "WARNING" else "PASSED"
    )
    expected_gate_pass = scenario.expected_check_status != "FAILED"
    passed = (
        target.status == scenario.expected_check_status
        and expected_failures_seen
        and report.passed is expected_gate_pass
        and matched_spike is not False
    )
    return FaultOutcome(
        check_name=scenario.check_name,
        construction=scenario.construction,
        expected_check_status=scenario.expected_check_status,
        actual_check_status=target.status,
        gate_verdict=gate_verdict,
        failed_checks=failed,
        target_details=detail_evidence,
        matched_injected_spike=matched_spike,
        passed=passed,
    )


def run_fault_matrix(bars: pl.DataFrame, pack: GatePack) -> tuple[FaultOutcome, ...]:
    """Evaluate all twelve checks and fail if any expected evidence is absent."""

    outcomes = tuple(
        evaluate_fault_scenario(scenario, pack) for scenario in build_fault_scenarios(bars)
    )
    for index, outcome in enumerate(outcomes, start=1):
        print(
            f"L1 fault {index:02}/12 {outcome.check_name}: "
            f"{outcome.gate_verdict} target={outcome.actual_check_status} "
            f"expected={outcome.expected_check_status} "
            f"failed={list(outcome.failed_checks)}"
        )
        print(f"  construct: {outcome.construction}")
        if outcome.check_name == "WARMUP_WINDOW_SKIP":
            print(
                "  coverage: "
                f"skipped_rows={outcome.target_details.get('skipped_rows')} "
                f"effective_coverage={outcome.target_details.get('effective_coverage')}"
            )
        if outcome.matched_injected_spike is not None:
            print(
                "  injected spike timestamp matched details.spike_rows: "
                f"{outcome.matched_injected_spike} "
                f"rows={outcome.target_details.get('injected_spike_rows')}"
            )
        if not outcome.passed:
            print(f"  unexpected result: {json.dumps(outcome.to_dict(), default=str)}")
    if not all(outcome.passed for outcome in outcomes):
        raise AssertionError("one or more L1 market-bars fault cases missed their expected result")
    return outcomes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=_DEFAULT_ARCHIVE)
    args = parser.parse_args(argv)
    try:
        bars = load_market_bars(args.zip)
        pack = get_pack("market-bars-pit")
    except PackNotInstalledError as error:
        print(error, file=sys.stderr)
        return 1
    run_fault_matrix(bars, pack)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
