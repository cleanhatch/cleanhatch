import math
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from cleanhatch import CausalFailure, CausalGate, TemporalSpec, compare_causal_values
from cleanhatch.causal import (
    _DEFAULT_WARMUP_BUFFER,
    _local_slice_around,
    _perturb_future,
    _warmup_boundary,
)

BASE = datetime(2026, 1, 1, tzinfo=UTC)
SPEC = TemporalSpec(signed_numeric_cols=("value",))


def _frame(values: list[float | int]) -> pl.DataFrame:
    times = [BASE + timedelta(hours=index) for index in range(len(values))]
    return pl.DataFrame(
        {
            "entity_id": ["A"] * len(values),
            "event_time": times,
            "available_at": times,
            "value": values,
        }
    )


def _full_sample_zscore(frame: pl.LazyFrame) -> pl.LazyFrame:
    return frame.with_columns(
        ((pl.col("value") - pl.col("value").mean()) / pl.col("value").std()).alias("factor")
    ).select(
        pl.col("event_time").alias("decision_time"),
        "entity_id",
        pl.col("factor").alias("value"),
    )


def _future_shift(frame: pl.LazyFrame) -> pl.LazyFrame:
    return (
        frame.sort("event_time")
        .with_columns(pl.col("value").shift(-1).alias("value"))
        .select(
            pl.col("event_time").alias("decision_time"),
            "entity_id",
            "value",
        )
    )


def _rolling_mean(frame: pl.LazyFrame) -> pl.LazyFrame:
    return (
        frame.sort("event_time")
        .with_columns(pl.col("value").rolling_mean(window_size=3).alias("factor"))
        .select(
            pl.col("event_time").alias("decision_time"),
            "entity_id",
            pl.col("factor").alias("value"),
        )
    )


def test_full_sample_normalization_canary_fails_future_perturbation() -> None:
    gate = CausalGate(SPEC)
    with pytest.raises(CausalFailure) as caught:
        gate.run(_full_sample_zscore, _frame([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))

    methods = {check.method for check in caught.value.failed_checks}
    assert "future_perturbation" in methods


def test_future_shift_canary_fails_availability_truncation() -> None:
    gate = CausalGate(SPEC)
    with pytest.raises(CausalFailure) as caught:
        gate.run(_future_shift, _frame([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))

    methods = {check.method for check in caught.value.failed_checks}
    assert "availability_truncation" in methods


def test_legal_rolling_mean_passes_with_integer_input() -> None:
    result = CausalGate(SPEC).run(
        _rolling_mean,
        _frame([10, 20, 30, 40, 50, 60]),
    )

    assert result.status == "PASSED"
    assert result.sample_points >= 3


def test_future_perturbation_preserves_integer_schema() -> None:
    frame = _frame([10, 20, 30, 40])
    causal = _perturb_future(
        frame,
        spec=SPEC,
        decision_time=BASE + timedelta(hours=1),
        numeric_cols=("value",),
        heavy_cols=frozenset(),
        mask_cols=(),
    )

    assert causal.schema["value"] == pl.Int64
    assert causal["value"].to_list()[2] != 30


def test_empty_cross_section_never_passes() -> None:
    check, drifts = compare_causal_values(
        method="future_perturbation",
        decision_time=BASE,
        expected={},
        actual={},
        tolerance=1e-8,
        rtol=1e-7,
        atol=1e-12,
    )

    assert check.compared_entities == 0
    assert check.passed is False
    assert drifts == ()


def test_warmup_boundary_crosses_fragile_first_points() -> None:
    times = [BASE + timedelta(hours=index) for index in range(10)]
    baseline = pl.DataFrame(
        {
            "decision_time": times,
            "value": [float(index) for index in range(10)],
        }
    )

    boundary = _warmup_boundary(
        baseline,
        decision_col="decision_time",
        value_col="value",
    )

    assert boundary == times[_DEFAULT_WARMUP_BUFFER]


def test_local_slice_excludes_ancient_future_restatements() -> None:
    times = [BASE + timedelta(hours=index) for index in range(4)]
    frame = pl.DataFrame(
        {
            "entity_id": ["A", "A", "A", "A", "A"],
            "event_time": [times[0], times[1], times[2], times[3], times[0]],
            "available_at": [times[0], times[1], times[2], times[3], BASE + timedelta(hours=5)],
            "value": [1.0, 2.0, 3.0, 4.0, 999.0],
        }
    )

    sliced = _local_slice_around(
        frame,
        times[2],
        spec=SPEC,
        safe_bars=1,
        future_bars=None,
        unique_times=times,
    )

    assert sliced.filter(
        (pl.col("event_time") == times[0]) & (pl.col("available_at") > times[2])
    ).is_empty()
    assert math.isclose(sliced["value"].max(), 4.0)
