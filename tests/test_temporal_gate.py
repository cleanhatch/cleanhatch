from datetime import UTC, datetime, timedelta

import duckdb
import polars as pl
import pytest

from cleanhatch import (
    TemporalContractError,
    TemporalGate,
    TemporalSpec,
    VersionMutationError,
)

BASE = datetime(2026, 1, 1, tzinfo=UTC)


def _revisioned_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "entity_id": ["A", "A"],
            "event_time": [BASE, BASE],
            "available_at": [BASE + timedelta(hours=1), BASE + timedelta(hours=2)],
            "value": [100.0, 120.0],
        }
    )


def test_read_as_of_keeps_latest_visible_revision() -> None:
    frame = _revisioned_frame()
    gate = TemporalGate(frame, spec=TemporalSpec())

    latest = gate.read_as_of(as_of=BASE + timedelta(hours=2))
    assert latest.height == 1
    assert latest["value"].to_list() == [120.0]

    earlier = gate.read_as_of(as_of=BASE + timedelta(hours=1))
    assert earlier.height == 1
    assert earlier["value"].to_list() == [100.0]


def test_read_as_of_compares_non_utc_timezones_by_instant() -> None:
    frame = _revisioned_frame().with_columns(
        pl.col("event_time").dt.convert_time_zone("Asia/Shanghai"),
        pl.col("available_at").dt.convert_time_zone("Asia/Shanghai"),
    )
    gate = TemporalGate(frame, spec=TemporalSpec())

    latest = gate.read_as_of(as_of=BASE + timedelta(hours=2))

    assert latest.height == 1
    assert latest["value"].to_list() == [120.0]


def test_read_as_of_can_return_all_revisions() -> None:
    gate = TemporalGate(_revisioned_frame(), spec=TemporalSpec())
    all_rows = gate.read_as_of(
        as_of=BASE + timedelta(hours=2),
        deduplicate=False,
    )
    assert all_rows.height == 2
    assert all_rows["value"].to_list() == [100.0, 120.0]


def test_parquet_as_of_qualifies_latest_revision_without_knowledge_column(
    tmp_path,
) -> None:
    path = tmp_path / "bars.parquet"
    _revisioned_frame().write_parquet(path)
    connection = duckdb.connect()
    try:
        result = TemporalGate.read_parquet_as_of(
            connection,
            [str(path)],
            spec=TemporalSpec(),
            as_of=BASE + timedelta(hours=2),
            columns=["event_time", "entity_id", "value"],
        )
    finally:
        connection.close()

    assert result.height == 1
    assert result["value"].to_list() == [120.0]


def test_parquet_as_of_can_skip_qualify(tmp_path) -> None:
    path = tmp_path / "bars.parquet"
    _revisioned_frame().write_parquet(path)
    connection = duckdb.connect()
    try:
        result = TemporalGate.read_parquet_as_of(
            connection,
            [str(path)],
            spec=TemporalSpec(),
            as_of=BASE + timedelta(hours=2),
            deduplicate=False,
        )
    finally:
        connection.close()

    assert result.height == 2


def test_scd2_join_returns_latest_interval_attribute() -> None:
    start = BASE
    change = BASE + timedelta(hours=1)
    available = BASE + timedelta(hours=2)
    as_of = BASE + timedelta(hours=3)
    dimension = pl.DataFrame(
        {
            "entity_id": ["A", "A"],
            "valid_from": [start, change],
            "valid_to": [change, None],
            "available_at": [start, available],
            "sector": ["Z_Sector", "A_Sector"],
        }
    )
    base = pl.DataFrame(
        {
            "entity_id": ["A"],
            "event_time": [change + timedelta(minutes=30)],
            "available_at": [as_of],
        }
    )

    result = TemporalGate(base, spec=TemporalSpec()).asof_join_scd2(
        base,
        dimension,
        as_of=as_of,
        attributes=["sector"],
    )

    assert result["sector"].to_list() == ["A_Sector"]


def test_scd2_join_uses_knowledge_time_not_alphabetical_order(monkeypatch) -> None:
    # Overlapping bitemporal revisions are normally rejected.  This test
    # isolates the aggregation tie-break: if both revisions are visible, the
    # later knowledge time must win even when its attribute sorts first.
    monkeypatch.setattr(
        "cleanhatch.temporal._reject_overlapping_intervals",
        lambda *_args, **_kwargs: None,
    )
    start = BASE
    as_of = BASE + timedelta(hours=3)
    dimension = pl.DataFrame(
        {
            "entity_id": ["A", "A"],
            "valid_from": [start, start],
            "valid_to": pl.Series([None, None], dtype=pl.Datetime("us", "UTC")),
            "available_at": [BASE + timedelta(hours=1), BASE + timedelta(hours=2)],
            "sector": ["Z_Sector", "A_Sector"],
        }
    )
    base = pl.DataFrame(
        {
            "entity_id": ["A"],
            "event_time": [start + timedelta(minutes=30)],
            "available_at": [as_of],
        }
    )

    result = TemporalGate(base, spec=TemporalSpec()).asof_join_scd2(
        base,
        dimension,
        as_of=as_of,
        attributes=["sector"],
    )

    assert result["sector"].to_list() == ["A_Sector"]


def test_scd2_join_rejects_overlapping_intervals() -> None:
    dimension = pl.DataFrame(
        {
            "entity_id": ["A", "A"],
            "valid_from": [BASE, BASE + timedelta(hours=1)],
            "valid_to": [BASE + timedelta(hours=2), BASE + timedelta(hours=3)],
            "available_at": [BASE, BASE],
            "sector": ["first", "second"],
        }
    )
    base = pl.DataFrame(
        {
            "entity_id": ["A"],
            "event_time": [BASE + timedelta(minutes=30)],
            "available_at": [BASE + timedelta(hours=4)],
        }
    )

    with pytest.raises(TemporalContractError, match="overlap"):
        TemporalGate(base, spec=TemporalSpec()).asof_join_scd2(
            base,
            dimension,
            as_of=BASE + timedelta(hours=4),
            attributes=["sector"],
        )


def test_diff_without_as_of_deduplicates_latest_revision() -> None:
    frame_a = _revisioned_frame()
    frame_b = pl.concat(
        [
            frame_a,
            pl.DataFrame(
                {
                    "entity_id": ["A"],
                    "event_time": [BASE],
                    "available_at": [BASE + timedelta(hours=3)],
                    "value": [130.0],
                }
            ),
        ]
    )
    gate = TemporalGate(frame_a, spec=TemporalSpec())

    report = gate.diff_versions(frame_a, frame_b, as_of=None)

    assert report.as_of is None
    assert report.rows_a == 1
    assert report.rows_b == 1
    assert report.rows_added == 0
    assert report.rows_removed == 0
    assert report.common_rows == 1
    assert report.changed_cell_ratio > 0.0
    assert report.columns[0].changed_rows == 1
    assert report.columns[0].max_abs_change == pytest.approx(10.0)


def test_unversioned_store_rejects_diff_version_tags() -> None:
    gate = TemporalGate(_revisioned_frame(), spec=TemporalSpec())

    with pytest.raises(TemporalContractError, match="unversioned"):
        gate.diff_versions("v1", "v2")


def test_immutable_version_tag_rejects_different_content() -> None:
    first = pl.DataFrame({"value": [1.0]})
    second = pl.DataFrame({"value": [2.0]})
    gate = TemporalGate(first, spec=TemporalSpec())

    gate.assert_version_immutable("v1", first)
    with pytest.raises(VersionMutationError):
        gate.assert_version_immutable("v1", second)
