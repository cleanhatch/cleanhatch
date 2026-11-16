from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from cleanhatch import (
    CausalResult,
    GateCheck,
    GateReport,
    InvarGate,
    InvarRule,
    Stage,
    TemporalSpec,
)

BASE = datetime(2026, 1, 1, tzinfo=UTC)


def _frame(values: list[float], *, mask: list[bool] | None = None) -> pl.DataFrame:
    size = len(values)
    data: dict[str, object] = {
        "entity_id": ["A"] * size,
        "event_time": [BASE + timedelta(minutes=index) for index in range(size)],
        "available_at": [BASE + timedelta(minutes=index) for index in range(size)],
        "value": values,
    }
    if mask is not None:
        data["is_trading"] = mask
    return pl.DataFrame(data)


def test_gate_report_aggregates_skipped_state() -> None:
    skipped = GateReport(
        Stage.CLEAN,
        (GateCheck("a", Stage.CLEAN, "SKIPPED", "no samples"),),
    )
    assert skipped.status == "SKIPPED"
    assert skipped.passed is True

    warned = GateReport(
        Stage.CLEAN,
        (GateCheck("a", Stage.CLEAN, "PASSED", "ok"),),
        warnings=("drift",),
    )
    assert warned.status == "WARNING"

    failed = GateReport(
        Stage.CLEAN,
        (GateCheck("a", Stage.CLEAN, "FAILED", "bad"),),
    )
    assert failed.status == "FAILED"
    assert failed.passed is False


def test_gate_report_without_checks_is_skipped() -> None:
    assert GateReport(Stage.CLEAN, ()).status == "SKIPPED"


def test_causal_gate_requires_explicit_passed_status() -> None:
    features = pl.DataFrame(
        {
            "decision_time": [BASE],
            "entity_id": ["A"],
            "value": [1.0],
        }
    )
    invalid_payloads: tuple[object, ...] = (None, {"status": "PASSED"})
    for payload in invalid_payloads:
        report = InvarGate().check_feature(
            features,
            causal_check=lambda payload=payload: payload,
            raise_on_failure=False,
        )
        check = next(c for c in report.checks if c.name == "causal_gate")

        assert check.status == "FAILED"
        assert check.message.startswith("causal gate returned invalid or failed status")

    passed = InvarGate().check_feature(
        features,
        causal_check=lambda: CausalResult(status="PASSED", sample_points=0, checks=()),
    )
    passed_check = next(c for c in passed.checks if c.name == "causal_gate")

    assert passed_check.status == "PASSED"


def test_run_accepts_string_stage_values() -> None:
    frame = _frame([1.0, 2.0])

    by_member = InvarGate().run(frame, stage=Stage.CLEAN)
    by_string = InvarGate().run(frame, stage="clean")

    assert by_member.to_dict() == by_string.to_dict()


def test_run_rejects_unknown_stage() -> None:
    with pytest.raises(ValueError, match="not a valid Stage"):
        InvarGate().run(_frame([1.0]), stage="unknown")


def test_protect_accepts_string_stage() -> None:
    @InvarGate.protect(stage="clean")
    def transform(frame: pl.DataFrame) -> pl.DataFrame:
        return frame

    result = transform(_frame([1.0, 2.0]))

    assert result.height == 2
    assert transform.last_report is not None
    assert transform.last_report.stage is Stage.CLEAN


def test_empty_clean_frame_is_skipped_not_passed() -> None:
    empty = pl.DataFrame(
        schema={
            "entity_id": pl.String,
            "event_time": pl.Datetime("us", "UTC"),
            "available_at": pl.Datetime("us", "UTC"),
            "value": pl.Float64,
        }
    )

    report = InvarGate().check_clean(empty, numeric_cols=("value",))

    assert report.status == "SKIPPED"
    assert report.passed is True
    assert all(check.status == "SKIPPED" for check in report.checks)


def test_empty_feature_frame_is_skipped() -> None:
    empty = pl.DataFrame(
        schema={
            "decision_time": pl.Datetime("us", "UTC"),
            "entity_id": pl.String,
            "value": pl.Float64,
        }
    )

    report = InvarGate().check_feature(empty)

    assert report.status == "SKIPPED"


def test_robust_drift_ignores_single_spike() -> None:
    baseline = _frame([float(index) for index in range(1, 101)])
    spiked = _frame([float(index) for index in range(1, 101)] + [100_000.0])
    gate = InvarGate(max_zscore_drift=4.0)

    robust = gate.check_clean(spiked, baseline=baseline, numeric_cols=("value",))
    mean_based = gate.check_clean(
        spiked,
        baseline=baseline,
        numeric_cols=("value",),
        robust_drift=False,
    )

    assert robust.status == "PASSED"
    assert mean_based.status == "WARNING"


def test_robust_drift_detects_whole_distribution_shift() -> None:
    baseline = _frame([float(index) for index in range(1, 101)])
    shifted = _frame([float(1000 + index) for index in range(1, 101)])

    report = InvarGate(max_zscore_drift=4.0).check_clean(
        shifted,
        baseline=baseline,
        numeric_cols=("value",),
    )

    assert report.status == "WARNING"


def test_dead_value_check_ignores_inactive_market_rows() -> None:
    values = [float(index) for index in range(10)] + [0.0] * 90
    mask = [True] * 10 + [False] * 90
    frame = _frame(values, mask=mask)

    unmasked = InvarGate().check_clean(
        frame,
        numeric_cols=("value",),
        raise_on_failure=False,
    )
    masked = InvarGate(spec=TemporalSpec(boolean_mask_cols=("is_trading",))).check_clean(
        frame,
        numeric_cols=("value",),
    )

    assert unmasked.status == "FAILED"
    assert masked.status == "PASSED"


def test_custom_rule_can_be_attached_to_clean_stage() -> None:
    rule = InvarRule(
        name="non_negative_value",
        predicate=pl.col("value") >= 0,
        message="value must be non-negative",
    )

    report = InvarGate().check_clean(
        _frame([-1.0, 2.0]),
        numeric_cols=(),
        rules=(rule,),
        raise_on_failure=False,
    )

    assert report.status == "FAILED"
    assert any(check.name == "non_negative_value" for check in report.checks)


def test_custom_rule_can_be_attached_to_feature_stage() -> None:
    features = pl.DataFrame(
        {
            "decision_time": [BASE, BASE + timedelta(hours=1)],
            "entity_id": ["A", "A"],
            "value": [1.0, -2.0],
        }
    )
    rule = InvarRule(
        name="positive_feature",
        predicate=pl.col("value") > 0,
    )

    report = InvarGate().check_feature(features, rules=(rule,), raise_on_failure=False)

    assert report.status == "FAILED"


def test_ingest_rejects_naive_time_columns() -> None:
    frame = pl.DataFrame(
        {
            "entity_id": ["A", "A"],
            "event_time": [datetime(2026, 1, 1), datetime(2026, 1, 1, 0, 1)],
            "available_at": [datetime(2026, 1, 1), datetime(2026, 1, 1, 0, 1)],
            "value": [1.0, 2.0],
        }
    )

    report = InvarGate().check_ingest(frame, raise_on_failure=False)
    check = next(c for c in report.checks if c.name == "timezone_aware_coordinates")

    assert check.status == "FAILED"
    assert report.status == "FAILED"
    assert set(check.details["non_tz_columns"]) == {"event_time", "available_at"}


def test_ingest_rejects_string_time_columns_without_crashing() -> None:
    frame = _frame([1.0, 2.0]).with_columns(
        pl.col("event_time").cast(pl.String),
        pl.col("available_at").cast(pl.String),
    )

    report = InvarGate().check_ingest(frame, raise_on_failure=False)
    timezone_check = next(c for c in report.checks if c.name == "timezone_aware_coordinates")
    causality_check = next(c for c in report.checks if c.name == "knowledge_time_causality")

    assert report.status == "FAILED"
    assert timezone_check.status == "FAILED"
    assert causality_check.status == "FAILED"
    assert "Datetime" in causality_check.message


def test_empty_ingest_frame_skips_timezone_check_but_validates_schema() -> None:
    empty = pl.DataFrame(
        schema={
            "entity_id": pl.String,
            "event_time": pl.Datetime("us", "UTC"),
            "available_at": pl.Datetime("us", "UTC"),
            "value": pl.Float64,
        }
    )

    report = InvarGate().check_ingest(empty, raise_on_failure=False)
    timezone_check = next(c for c in report.checks if c.name == "timezone_aware_coordinates")

    assert timezone_check.status == "SKIPPED"
    assert "schema" in timezone_check.message


def test_causality_compares_cross_timezone_columns_by_instant() -> None:
    frame = pl.DataFrame(
        {
            "entity_id": ["A", "A"],
            "event_time": [
                datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
                datetime(2026, 1, 1, 13, 0, tzinfo=UTC),
            ],
            "available_at": [
                datetime(2026, 1, 1, 13, 0, tzinfo=UTC),
                datetime(2026, 1, 1, 14, 0, tzinfo=UTC),
            ],
            "value": [1.0, 2.0],
        }
    ).with_columns(pl.col("available_at").dt.convert_time_zone("America/New_York"))

    report = InvarGate().check_ingest(frame, raise_on_failure=False)
    causality_check = next(c for c in report.checks if c.name == "knowledge_time_causality")

    assert causality_check.status == "PASSED"

    violated = frame.with_columns(
        pl.col("available_at").dt.convert_time_zone("UTC").dt.offset_by("-2h")
    )
    violation_report = InvarGate().check_ingest(violated, raise_on_failure=False)
    violation_check = next(
        c for c in violation_report.checks if c.name == "knowledge_time_causality"
    )

    assert violation_check.status == "FAILED"


def test_grid_continuity_can_allow_multiples() -> None:
    frame = pl.DataFrame(
        {
            "entity_id": ["A", "A", "A"],
            "event_time": [BASE, BASE + timedelta(hours=2), BASE + timedelta(hours=4)],
            "available_at": [BASE, BASE + timedelta(hours=2), BASE + timedelta(hours=4)],
        }
    )
    gate = InvarGate()

    strict = gate.check_clean(
        frame,
        grid_freq_ms=3_600_000,
        raise_on_failure=False,
    )
    permissive = gate.check_clean(
        frame,
        grid_freq_ms=3_600_000,
        allow_multiples=True,
    )

    assert strict.status == "FAILED"
    assert permissive.status == "PASSED"
