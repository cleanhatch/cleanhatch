from datetime import UTC, datetime
from time import perf_counter

import polars as pl

from cleanhatch import content_hash


def test_negative_zero_matches_positive_zero() -> None:
    positive = pl.DataFrame({"value": [0.0, 1.5, -2.25]})
    negative = pl.DataFrame({"value": [-0.0, 1.5, -2.25]})

    assert content_hash(positive) == content_hash(negative)


def test_negative_zero_matches_for_float32() -> None:
    positive = pl.DataFrame({"value": pl.Series([0.0], dtype=pl.Float32)})
    negative = pl.DataFrame({"value": pl.Series([-0.0], dtype=pl.Float32)})

    assert content_hash(positive) == content_hash(negative)


def test_nan_payloads_are_canonicalized() -> None:
    first = pl.DataFrame({"value": [float("nan"), 1.0]})
    second = pl.DataFrame({"value": [-float("nan"), 1.0]})

    assert content_hash(first) == content_hash(second)


def test_aware_datetimes_hash_by_instant() -> None:
    instant = datetime(2026, 1, 1, 12, tzinfo=UTC)
    utc = pl.DataFrame({"timestamp": pl.Series([instant], dtype=pl.Datetime("us", "UTC"))})
    new_york = utc.with_columns(pl.col("timestamp").dt.convert_time_zone("America/New_York"))

    assert content_hash(utc) == content_hash(new_york)


def test_nested_list_and_array_floats_are_canonicalized() -> None:
    list_positive = pl.DataFrame(
        {"value": pl.Series([[0.0, float("nan"), 1.5]], dtype=pl.List(pl.Float64))}
    )
    list_negative = pl.DataFrame(
        {"value": pl.Series([[-0.0, float("nan"), 1.5]], dtype=pl.List(pl.Float64))}
    )
    array_positive = pl.DataFrame({"value": pl.Series([[0.0, 1.5]], dtype=pl.Array(pl.Float64, 2))})
    array_negative = pl.DataFrame(
        {"value": pl.Series([[-0.0, 1.5]], dtype=pl.Array(pl.Float64, 2))}
    )

    assert content_hash(list_positive) == content_hash(list_negative)
    assert content_hash(array_positive) == content_hash(array_negative)


def test_nested_struct_floats_and_datetimes_are_canonicalized() -> None:
    instant = datetime(2026, 1, 1, 12, tzinfo=UTC)
    positive = pl.DataFrame({"row": [{"value": 0.0, "timestamp": instant}]})
    negative = pl.DataFrame({"row": [{"value": -0.0, "timestamp": instant}]})
    utc = pl.DataFrame(
        {
            "row": pl.Series(
                [{"value": 0.0, "timestamp": instant}],
                dtype=pl.Struct({"value": pl.Float64, "timestamp": pl.Datetime("us", "UTC")}),
            )
        }
    )
    new_york = pl.DataFrame(
        {
            "row": pl.Series(
                [{"value": 0.0, "timestamp": instant}],
                dtype=pl.Struct(
                    {"value": pl.Float64, "timestamp": pl.Datetime("us", "America/New_York")}
                ),
            )
        }
    )

    assert content_hash(positive) == content_hash(negative)
    assert content_hash(utc) == content_hash(new_york)


def test_nested_list_of_structs_is_canonicalized() -> None:
    dtype = pl.List(pl.Struct({"value": pl.Float64}))
    positive = pl.DataFrame({"rows": pl.Series([[{"value": 0.0}]], dtype=dtype)})
    negative = pl.DataFrame({"rows": pl.Series([[{"value": -0.0}]], dtype=dtype)})

    assert content_hash(positive) == content_hash(negative)


def test_struct_with_nested_list_float_is_canonicalized() -> None:
    positive = pl.DataFrame({"row": [{"inner": [0.0, float("nan")]}]})
    negative = pl.DataFrame({"row": [{"inner": [-0.0, float("nan")]}]})

    assert content_hash(positive) == content_hash(negative)


def test_nested_list_and_array_datetimes_hash_by_instant() -> None:
    instant = datetime(2026, 1, 1, 12, tzinfo=UTC)
    list_utc = pl.DataFrame(
        {"value": pl.Series([[instant]], dtype=pl.List(pl.Datetime("us", "UTC")))}
    )
    list_new_york = pl.DataFrame(
        {"value": pl.Series([[instant]], dtype=pl.List(pl.Datetime("us", "America/New_York")))}
    )
    array_utc = pl.DataFrame(
        {"value": pl.Series([[instant, instant]], dtype=pl.Array(pl.Datetime("us", "UTC"), 2))}
    )
    array_new_york = pl.DataFrame(
        {
            "value": pl.Series(
                [[instant, instant]],
                dtype=pl.Array(pl.Datetime("us", "America/New_York"), 2),
            )
        }
    )

    assert content_hash(list_utc) == content_hash(list_new_york)
    assert content_hash(array_utc) == content_hash(array_new_york)


def test_any_cell_change_changes_hash() -> None:
    base = pl.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})
    changed = base.with_columns(pl.when(pl.col("a") == 2).then(20).otherwise(pl.col("a")))

    assert content_hash(base) != content_hash(changed)


def test_column_order_changes_hash() -> None:
    base = pl.DataFrame({"a": [1, 2], "b": [3, 4]})

    assert content_hash(base) != content_hash(base.select(["b", "a"]))


def test_null_position_changes_hash() -> None:
    first = pl.DataFrame({"a": [1, None, 3]})
    second = pl.DataFrame({"a": [1, 3, None]})

    assert content_hash(first) != content_hash(second)


def test_dtype_change_changes_hash() -> None:
    integer = pl.DataFrame({"a": [1, 2, 3]})
    floating = pl.DataFrame({"a": [1.0, 2.0, 3.0]})

    assert content_hash(integer) != content_hash(floating)


def test_hash_is_deterministic() -> None:
    frame = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})

    assert content_hash(frame) == content_hash(frame)


def test_vectorized_hash_performance() -> None:
    frame = pl.DataFrame(
        {f"column_{index}": [float(row) for row in range(100_000)] for index in range(10)}
    )

    started = perf_counter()
    digest = content_hash(frame)
    elapsed = perf_counter() - started

    assert len(digest) == 64
    assert elapsed < 0.05, f"content_hash took {elapsed * 1000:.1f}ms"
