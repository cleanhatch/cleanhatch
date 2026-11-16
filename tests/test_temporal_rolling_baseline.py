"""Rolling PIT baselines: window edges, restatements, and robust drift."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from cleanhatch import InvarGate, TemporalGate, TemporalSpec
from cleanhatch.cli import main

BASE = datetime(2026, 1, 1, tzinfo=UTC)
SPEC = TemporalSpec()


def _frame(rows: list[tuple[datetime, datetime, float]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "entity_id": ["A"] * len(rows),
            "event_time": [row[0] for row in rows],
            "available_at": [row[1] for row in rows],
            "value": [row[2] for row in rows],
        }
    )


def test_rolling_baseline_returns_only_windows_with_visible_rows() -> None:
    as_of = BASE + timedelta(days=4)
    frame = _frame([(BASE + timedelta(days=3, hours=12), BASE + timedelta(days=3, hours=13), 1.0)])

    baseline = TemporalGate(frame, spec=SPEC).rolling_baseline(
        as_of=as_of,
        n_windows=3,
        window=timedelta(days=1),
    )

    assert baseline.height == 1
    assert baseline["__baseline_window_start"].to_list() == [BASE + timedelta(days=3)]
    assert baseline["__baseline_window_end"].to_list() == [BASE + timedelta(days=4)]


def test_rolling_baseline_returns_exactly_n_windows() -> None:
    as_of = BASE + timedelta(days=3)
    rows = [
        (
            BASE + timedelta(days=index, hours=12),
            BASE + timedelta(days=index, hours=13),
            float(index),
        )
        for index in range(3)
    ]

    baseline = TemporalGate(_frame(rows), spec=SPEC).rolling_baseline(
        as_of=as_of,
        n_windows=3,
        window=timedelta(days=1),
    )

    assert baseline.height == 3
    assert baseline["value"].to_list() == [0.0, 1.0, 2.0]


def test_rolling_baseline_does_not_rewrite_a_window_with_a_later_restatement() -> None:
    event = BASE + timedelta(days=1, hours=12)
    original = event + timedelta(hours=1)
    restatement = BASE + timedelta(days=3, hours=12)
    frame = _frame(
        [
            (event, original, 100.0),
            (event, restatement, 999.0),
        ]
    )

    baseline = TemporalGate(frame, spec=SPEC).rolling_baseline(
        as_of=BASE + timedelta(days=4),
        n_windows=3,
        window=timedelta(days=1),
    )

    assert baseline.height == 1
    assert baseline["value"].to_list() == [100.0]


def test_rolling_baseline_aligns_rows_on_window_boundaries() -> None:
    frame = _frame(
        [
            (BASE + timedelta(days=0), BASE + timedelta(hours=1), 10.0),
            (BASE + timedelta(days=1), BASE + timedelta(days=1, hours=1), 20.0),
        ]
    )

    baseline = TemporalGate(frame, spec=SPEC).rolling_baseline(
        as_of=BASE + timedelta(days=2),
        n_windows=2,
        window=timedelta(days=1),
    )

    assert baseline["__baseline_window_start"].to_list() == [
        BASE,
        BASE + timedelta(days=1),
    ]
    assert baseline["value"].to_list() == [10.0, 20.0]


def _daily_frame(values: list[float]) -> pl.DataFrame:
    rows = [
        (
            BASE + timedelta(days=index),
            BASE + timedelta(days=index, hours=1),
            value,
        )
        for index, value in enumerate(values)
    ]
    return _frame(rows)


def test_yacht_scale_values_do_not_trigger_false_drift() -> None:
    values = [1_000_000_000_000.0 + index * 1_000_000.0 for index in range(40)]
    frame = _daily_frame(values)
    gate = TemporalGate(frame, spec=SPEC)
    baseline = gate.rolling_baseline(
        as_of=BASE + timedelta(days=40),
        n_windows=20,
        window=timedelta(days=1),
    )

    report = InvarGate(SPEC).check_clean(
        frame.tail(5),
        baseline=baseline,
        numeric_cols=("value",),
        raise_on_failure=False,
    )

    drift = [check for check in report.checks if check.name == "zscore_drift:value"]
    assert drift
    assert drift[0].status == "PASSED"
    assert report.passed is True


def test_a_single_spike_in_the_baseline_does_not_poison_drift() -> None:
    values = [1_000_000_000_000.0 + index * 1_000.0 for index in range(40)]
    values[20] = 10_000_000_000_000_000.0
    frame = _daily_frame(values)
    baseline = TemporalGate(frame, spec=SPEC).rolling_baseline(
        as_of=BASE + timedelta(days=40),
        n_windows=20,
        window=timedelta(days=1),
    )

    report = InvarGate(SPEC).check_clean(
        frame.tail(5),
        baseline=baseline,
        numeric_cols=("value",),
        raise_on_failure=False,
    )

    drift = [check for check in report.checks if check.name == "zscore_drift:value"]
    assert drift
    assert drift[0].status == "PASSED"


def test_cli_clean_stage_uses_the_rolling_baseline_by_default(
    tmp_path: Path,
    capsys,
) -> None:
    source = tmp_path / "history.parquet"
    values = [100.0 + index for index in range(40)]
    _daily_frame(values).write_parquet(source)

    exit_code = main(["validate", str(source), "--stage", "clean"])
    report = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert any(check["name"] == "zscore_drift:value" for check in report["checks"])


def test_cli_clean_stage_can_disable_the_baseline(
    tmp_path: Path,
    capsys,
) -> None:
    source = tmp_path / "history.parquet"
    values = [100.0 + index for index in range(40)]
    _daily_frame(values).write_parquet(source)

    exit_code = main(["validate", str(source), "--stage", "clean", "--no-drift-baseline"])
    report = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert not any(check["name"].startswith("zscore_drift:") for check in report["checks"])
