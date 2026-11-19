"""Run the L0 market-bar gates on the bundled or downloaded Binance Vision ZIP."""

from __future__ import annotations

import argparse
import hashlib
import io
import sys
import zipfile
from pathlib import Path

import polars as pl

from cleanhatch.invar import GateReport, InvarGate, InvarRule, Stage
from cleanhatch.packs import PackNotInstalledError, get_pack
from cleanhatch.spec import TemporalSpec

_DEFAULT_ARCHIVE = Path(__file__).with_name("data") / "BTCUSDT-1h-2024-01.zip"
_MAX_UNCOMPRESSED_BYTES = 16 * 1024 * 1024
_CSV_COLUMNS = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_base_volume",
    "taker_quote_volume",
    "ignore",
)


def load_market_bars(archive_path: Path) -> pl.DataFrame:
    with zipfile.ZipFile(archive_path) as archive:
        csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(csv_names) != 1:
            raise ValueError(f"expected one CSV in the Binance ZIP, found {len(csv_names)}")
        with archive.open(csv_names[0]) as source:
            csv_bytes = source.read(_MAX_UNCOMPRESSED_BYTES + 1)
    if len(csv_bytes) > _MAX_UNCOMPRESSED_BYTES:
        raise ValueError(f"uncompressed Binance CSV exceeds {_MAX_UNCOMPRESSED_BYTES} bytes")

    raw = pl.read_csv(
        io.BytesIO(csv_bytes),
        has_header=False,
        new_columns=list(_CSV_COLUMNS),
        infer_schema_length=1000,
    )
    if raw.width != len(_CSV_COLUMNS):
        raise ValueError(f"expected 12 Binance kline columns, got {raw.width}")
    bars = raw.select(
        pl.lit("BTCUSDT").alias("symbol"),
        pl.col("open").cast(pl.Float64),
        pl.col("high").cast(pl.Float64),
        pl.col("low").cast(pl.Float64),
        pl.col("close").cast(pl.Float64),
        pl.col("volume").cast(pl.Float64),
        pl.col("quote_volume").cast(pl.Float64),
        (pl.col("close_time").cast(pl.Int64) * 1000)
        .cast(pl.Datetime("us", "UTC"))
        .alias("bar_close_time"),
    )
    return (
        bars.with_columns(
            (pl.col("bar_close_time") + pl.duration(milliseconds=1)).alias("available_at"),
            (pl.col("volume") > 0).alias("is_trading"),
        )
        .with_columns(pl.col("available_at").alias("decision_time"))
        .sort("bar_close_time")
    )


def _l0_gate() -> InvarGate:
    spec = TemporalSpec(
        entity_col="symbol",
        event_time_col="bar_close_time",
        knowledge_time_col="available_at",
        decision_time_col="decision_time",
    )
    return InvarGate(spec)


def _positive_close_rule() -> InvarRule:
    return InvarRule(
        name="POSITIVE_FINITE_CLOSE",
        predicate=(pl.col("close").is_finite() & (pl.col("close") > 0)),
        message="close must be finite and greater than zero",
    )


def _availability_before_decision_rule() -> InvarRule:
    return InvarRule(
        name="AVAILABILITY_BEFORE_DECISION",
        predicate=pl.col("available_at") <= pl.col("decision_time"),
        message="available_at must not follow decision_time",
    )


def inject_negative_value(frame: pl.DataFrame, column: str) -> pl.DataFrame:
    return frame.with_columns(
        pl.when(pl.int_range(0, pl.len()) == 0)
        .then(pl.lit(-1.0))
        .otherwise(pl.col(column))
        .alias(column)
    )


def inject_pre_close_availability(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(
        pl.when(pl.int_range(0, pl.len()) == 0)
        .then(pl.col("bar_close_time") - pl.duration(milliseconds=1))
        .otherwise(pl.col("available_at"))
        .alias("available_at")
    ).with_columns(pl.col("available_at").alias("decision_time"))


def inject_after_decision_availability(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(
        pl.when(pl.int_range(0, pl.len()) == 0)
        .then(pl.col("available_at") + pl.duration(milliseconds=1))
        .otherwise(pl.col("available_at"))
        .alias("available_at")
    )


def inject_non_finite_value(frame: pl.DataFrame, column: str) -> pl.DataFrame:
    return frame.with_columns(
        pl.when(pl.int_range(0, pl.len()) == 0)
        .then(pl.lit(float("nan")))
        .otherwise(pl.col(column))
        .alias(column)
    )


def run_l0(
    frame: pl.DataFrame,
) -> tuple[GateReport, GateReport, GateReport, GateReport, GateReport]:
    gate = _l0_gate()
    rules = (_positive_close_rule(), _availability_before_decision_rule())
    clean = gate.check_ingest(
        frame, required_columns=("close",), rules=rules, raise_on_failure=False
    )
    bad_close = inject_negative_value(frame, "close")
    negative_close = gate.check_ingest(
        bad_close,
        required_columns=("close",),
        rules=rules,
        raise_on_failure=False,
    )
    early_availability = inject_pre_close_availability(frame)
    pit_violation = gate.check_ingest(
        early_availability,
        required_columns=("close",),
        rules=rules,
        raise_on_failure=False,
    )
    after_decision = gate.check_ingest(
        inject_after_decision_availability(frame),
        required_columns=("close",),
        rules=rules,
        raise_on_failure=False,
    )
    non_finite_close = gate.check_ingest(
        inject_non_finite_value(frame, "close"),
        required_columns=("close",),
        rules=rules,
        raise_on_failure=False,
    )
    return clean, negative_close, pit_violation, after_decision, non_finite_close


def _print_report(label: str, report: GateReport, *, expected_block: bool = False) -> None:
    failed = [check.name for check in report.checks if check.status == "FAILED"]
    warnings = [check.name for check in report.checks if check.status == "WARNING"]
    if expected_block:
        verdict = "BLOCKED" if not report.passed else "MISSED"
    elif report.passed:
        verdict = "CONTINUE" if report.status == "WARNING" else "PASSED"
    else:
        verdict = "FAILED"
    print(f"{label}: {verdict} (status={report.status}, failed={failed}, warnings={warnings})")


def _failed_check_names(report: GateReport) -> set[str]:
    return {check.name for check in report.checks if check.status == "FAILED"}


def run_l1(frame: pl.DataFrame) -> tuple[GateReport, GateReport, GateReport]:
    pack = get_pack("market-bars-pit")
    clean = pack.validate(frame, stage=Stage.INGEST, raise_on_failure=False)
    bad_close = inject_negative_value(frame, "close")
    negative_close = pack.validate(bad_close, stage=Stage.INGEST, raise_on_failure=False)
    early_availability = inject_pre_close_availability(frame)
    pit_violation = pack.validate(
        early_availability,
        stage=Stage.INGEST,
        raise_on_failure=False,
    )
    return clean, negative_close, pit_violation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=_DEFAULT_ARCHIVE, help="Binance Vision ZIP")
    parser.add_argument(
        "--with-l1",
        action="store_true",
        help="also run the installed private L1 market-bars-pit pack",
    )
    parser.add_argument(
        "--fault-matrix",
        action="store_true",
        help="run deterministic examples for all 12 L1 market-bars-pit checks",
    )
    args = parser.parse_args(argv)
    if args.fault_matrix and not args.with_l1:
        parser.error("--fault-matrix requires --with-l1")

    try:
        bars = load_market_bars(args.zip)
    except (OSError, ValueError, zipfile.BadZipFile, pl.exceptions.PolarsError) as error:
        print(f"could not load Binance sample: {error}", file=sys.stderr)
        return 1

    digest = hashlib.sha256(args.zip.read_bytes()).hexdigest()
    print(f"dataset: {args.zip.name} rows={bars.height} sha256={digest}")
    clean, negative_close, pit_violation, after_decision, non_finite_close = run_l0(bars)
    _print_report("L0 real sample", clean)
    _print_report("L0 injected negative close", negative_close, expected_block=True)
    _print_report("L0 injected pre-close availability", pit_violation, expected_block=True)
    _print_report("L0 injected availability after decision", after_decision, expected_block=True)
    _print_report("L0 injected non-finite close", non_finite_close, expected_block=True)
    if (
        not clean.passed
        or _failed_check_names(negative_close) != {"POSITIVE_FINITE_CLOSE"}
        or _failed_check_names(pit_violation) != {"knowledge_time_causality"}
        or "AVAILABILITY_BEFORE_DECISION" not in _failed_check_names(after_decision)
        or _failed_check_names(non_finite_close) != {"POSITIVE_FINITE_CLOSE"}
    ):
        return 1

    if args.with_l1:
        try:
            l1_clean, l1_negative, l1_pit = run_l1(bars)
        except PackNotInstalledError as error:
            print(error, file=sys.stderr)
            return 1
        _print_report("L1 market-bars-pit real sample", l1_clean)
        _print_report("L1 injected negative close", l1_negative, expected_block=True)
        _print_report("L1 injected pre-close availability", l1_pit, expected_block=True)
        if (
            not l1_clean.passed
            or "POSITIVE_FINITE_PRICES" not in _failed_check_names(l1_negative)
            or _failed_check_names(l1_pit) != {"BAR_CLOSE_CAUSALITY"}
        ):
            return 1
        if args.fault_matrix:
            from examples.repro.market_bars.fault_matrix import run_fault_matrix

            run_fault_matrix(bars, get_pack("market-bars-pit"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
