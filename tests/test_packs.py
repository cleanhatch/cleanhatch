"""Comprehensive tests for the free temporal-core pack and the shared pack protocol."""

from __future__ import annotations

import importlib.metadata
import warnings
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

import cleanhatch.packs as packs_module
from cleanhatch import (
    CleanHatchError,
    GateCheck,
    GatePack,
    GateReport,
    InvarGate,
    InvarRule,
    LineageManifest,
    NonIsolatedAblationWarning,
    PackNotInstalledError,
    TransformSignature,
    VersionMutationError,
    available_packs,
    get_pack,
    register_pack,
)
from cleanhatch.packs.temporal_core import (
    TemporalCoreColMap,
    TemporalCorePack,
    apply_publication_lag,
)

BASE = datetime(2026, 1, 1, tzinfo=UTC)


def _check(report: GateReport, name: str) -> GateCheck:
    return next(check for check in report.checks if check.name == name)


# --- InvarRule three-state and severity extensions -------------------------


def test_warn_severity_does_not_fail_the_gate() -> None:
    frame = pl.DataFrame(
        {
            "entity_id": ["A", "A"],
            "event_time": [BASE, BASE + timedelta(hours=1)],
            "available_at": [BASE, BASE + timedelta(hours=1)],
            "value": [1.0, 3.0],
        }
    )
    rule = InvarRule(
        name="value_below_ceiling",
        predicate=pl.col("value") < 2.0,
        message="value exceeded the review ceiling",
        severity="WARN",
    )

    report = InvarGate().check_ingest(frame, rules=(rule,))
    check = _check(report, "value_below_ceiling")

    assert check.status == "WARNING"
    assert check.details["failure_count"] == 1
    assert report.status == "WARNING"
    assert report.passed is True
    assert report.warnings == ("value_below_ceiling: value exceeded the review ceiling",)


def test_skip_when_excludes_rows_from_failure_and_coverage() -> None:
    frame = pl.DataFrame(
        {
            "entity_id": ["A", "A", "A", "A"],
            "event_time": [BASE + timedelta(minutes=index) for index in range(4)],
            "available_at": [BASE + timedelta(minutes=index) for index in range(4)],
            "row_index": [0, 1, 2, 3],
            "value": [-1.0, -2.0, 1.0, 2.0],
        }
    )
    rule = InvarRule(
        name="positive_after_warmup",
        predicate=pl.col("value") > 0,
        skip_when=pl.col("row_index") < 2,
    )

    report = InvarGate().check_ingest(frame, rules=(rule,))
    check = _check(report, "positive_after_warmup")

    assert check.status == "PASSED"
    assert check.details["skipped_rows"] == 2
    assert check.details["effective_coverage"] == pytest.approx(0.5)


def test_skip_when_all_rows_returns_skipped() -> None:
    frame = pl.DataFrame(
        {
            "entity_id": ["A", "A"],
            "event_time": [BASE, BASE + timedelta(hours=1)],
            "available_at": [BASE, BASE + timedelta(hours=1)],
            "value": [1.0, 2.0],
        }
    )
    rule = InvarRule(
        name="fully_skipped",
        predicate=pl.col("value") > 0,
        skip_when=pl.col("value") > 0,
    )

    report = InvarGate().check_ingest(frame, rules=(rule,))
    check = _check(report, "fully_skipped")

    assert check.status == "SKIPPED"
    assert check.details["skipped_rows"] == 2
    assert check.details["effective_coverage"] == 0.0


# --- Pack registry ---------------------------------------------------------


def test_available_packs_without_plugins_is_temporal_core_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda *, group: [])
    monkeypatch.setattr(packs_module, "_REGISTERED_PACKS", {})

    assert available_packs() == ("temporal-core",)


def test_get_pack_rejects_unknown_names_and_options() -> None:
    with pytest.raises(KeyError, match="unknown gate pack"):
        get_pack("not-a-pack")
    with pytest.raises(TypeError, match="unsupported temporal-core pack options"):
        get_pack("temporal-core", unknown_option=True)


def test_register_pack_adds_and_removes_a_custom_pack() -> None:
    name = "test-custom-pack"

    def factory(**kwargs: object) -> GatePack:
        return get_pack("temporal-core", **kwargs)

    register_pack(name, factory)
    try:
        assert name in available_packs()
        assert get_pack(name).name == "temporal-core"
    finally:
        packs_module._REGISTERED_PACKS.pop(name, None)

    assert name not in available_packs()
    with pytest.raises(KeyError, match="unknown gate pack"):
        get_pack(name)


def test_entry_point_packs_are_discovered(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeEntryPoint:
        name = "entry-point-pack"

        def load(self) -> Callable[..., GatePack]:
            return lambda **kwargs: get_pack("temporal-core", **kwargs)

    def fake_entry_points(*, group: str) -> list[FakeEntryPoint]:
        return [FakeEntryPoint()] if group == "cleanhatch.packs" else []

    monkeypatch.setattr(importlib.metadata, "entry_points", fake_entry_points)

    assert "entry-point-pack" in available_packs()
    assert get_pack("entry-point-pack").name == "temporal-core"


def test_available_packs_lists_entry_point_names_without_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class UnloadedEntryPoint:
        name = "not-loaded-pack"

        def load(self) -> Callable[..., GatePack]:
            raise AssertionError("listing packs must not import plugin implementations")

    def fake_entry_points(*, group: str) -> list[UnloadedEntryPoint]:
        return [UnloadedEntryPoint()] if group == "cleanhatch.packs" else []

    monkeypatch.setattr(importlib.metadata, "entry_points", fake_entry_points)

    assert "not-loaded-pack" in available_packs()


def test_l0_blocks_commercial_l1_packs_with_install_guidance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda *, group: [])
    monkeypatch.setattr(packs_module, "_REGISTERED_PACKS", {})

    assert available_packs() == ("temporal-core",)

    with pytest.raises(PackNotInstalledError) as excinfo:
        get_pack("market-bars-pit")

    error = excinfo.value
    assert isinstance(error, KeyError)
    assert isinstance(error, CleanHatchError)
    assert "L1 Pro Suite" in str(error)
    assert "cleanhatch-l1" in str(error)


@pytest.mark.parametrize("pack_name", ["temporal-core"])
def test_every_pack_conforms_to_the_gate_pack_protocol(pack_name: str) -> None:
    pack = get_pack(pack_name)

    assert isinstance(pack, GatePack)
    assert pack.name == pack_name

    recipe = pack.render_recipe_scaffold()
    agents = pack.render_agents_md()
    assert "cleanhatch" in recipe
    assert "1A-Guard" in recipe
    assert agents.startswith("# AGENTS.md")
    assert "1A-Guard" in agents
    assert "Do not edit a signed license badge or trace code by hand." in agents

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NonIsolatedAblationWarning)
        suite = pack.run_canary_suite()

    assert suite.pack_name == pack_name
    assert suite.checks
    assert suite.all_caught is True


# --- Pack 1: temporal-core -------------------------------------------------


def _temporal_panel() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "entity_id": ["A", "A", "B"],
            "event_time": [BASE, BASE + timedelta(hours=1), BASE],
            "available_at": [BASE + timedelta(hours=1), BASE + timedelta(hours=2), BASE],
            "decision_time": [BASE + timedelta(hours=1)] * 3,
            "valid_from": [BASE, BASE + timedelta(hours=1), BASE],
            "valid_to": [BASE + timedelta(hours=1), BASE + timedelta(hours=2), None],
            "value": [1.0, 2.0, 3.0],
        }
    )


def test_temporal_core_clean_panel_passes() -> None:
    report = TemporalCorePack().validate(_temporal_panel())

    assert report.status == "PASSED"
    assert report.passed is True
    for name in (
        "TZ_AWARE_COORDINATES",
        "CAUSAL_AVAILABILITY",
        "PIT_VERSION_DEDUP",
        "SCD2_NON_OVERLAPPING",
    ):
        assert _check(report, name).status == "PASSED"
    assert _check(report, "IMMUTABLE_VERSION_TAG").status == "SKIPPED"
    assert _check(report, "SINGLE_VARIABLE_ABLATION").status == "SKIPPED"


def test_temporal_core_custom_col_map_avoids_renaming() -> None:
    col_map = TemporalCoreColMap(
        entity_col="ticker",
        event_time_col="event_ts",
        knowledge_time_col="published_at",
        decision_time_col="decided_at",
        valid_from_col="from_ts",
        valid_to_col="to_ts",
    )
    frame = pl.DataFrame(
        {
            "ticker": ["AAPL", "AAPL"],
            "event_ts": [BASE, BASE + timedelta(hours=1)],
            "published_at": [BASE + timedelta(hours=2), BASE + timedelta(hours=3)],
            "decided_at": [BASE + timedelta(hours=2), BASE + timedelta(hours=3)],
            "from_ts": [BASE, BASE + timedelta(hours=1)],
            "to_ts": [BASE + timedelta(hours=1), BASE + timedelta(hours=2)],
            "value": [1.0, 2.0],
        }
    )
    original_columns = frame.columns
    pack = TemporalCorePack(col_map=col_map)

    report = pack.validate(frame)

    assert report.passed is True
    assert pack.spec.entity_col == "ticker"
    assert pack.spec.knowledge_time_col == "published_at"
    assert frame.columns == original_columns

    violated = frame.with_columns(
        (pl.col("published_at") - pl.duration(hours=3)).alias("published_at")
    )
    assert _check(pack.validate(violated), "CAUSAL_AVAILABILITY").status == "FAILED"


def test_temporal_core_flags_naive_coordinates() -> None:
    frame = pl.DataFrame(
        {
            "entity_id": ["A"],
            "event_time": [datetime(2026, 1, 1)],
            "available_at": [datetime(2026, 1, 1, 1)],
            "value": [1.0],
        }
    )

    report = TemporalCorePack().validate(frame)
    check = _check(report, "TZ_AWARE_COORDINATES")

    assert check.status == "FAILED"
    assert set(check.details["non_tz_columns"]) == {"event_time", "available_at"}
    assert report.passed is False


def test_temporal_core_flags_non_causal_availability() -> None:
    frame = _temporal_panel().with_columns(
        (pl.col("available_at") - pl.duration(hours=2)).alias("available_at")
    )

    report = TemporalCorePack().validate(frame)
    check = _check(report, "CAUSAL_AVAILABILITY")

    assert check.status == "FAILED"
    assert check.details["failure_count"] >= 1


def test_temporal_core_dedup_and_read_as_of_convergence() -> None:
    pack = TemporalCorePack()
    conflicting = pl.DataFrame(
        {
            "entity_id": ["A", "A"],
            "event_time": [BASE, BASE],
            "available_at": [BASE + timedelta(hours=1), BASE + timedelta(hours=1)],
            "value": [1.0, 2.0],
        }
    )

    conflict_check = _check(pack.validate(conflicting), "PIT_VERSION_DEDUP")
    assert conflict_check.status == "FAILED"
    assert conflict_check.details["conflict_group_count"] == 1
    assert conflict_check.details["failure_count"] == 2

    revisions = pl.DataFrame(
        {
            "entity_id": ["A", "A"],
            "event_time": [BASE, BASE],
            "available_at": [BASE + timedelta(hours=1), BASE + timedelta(hours=2)],
            "value": [100.0, 120.0],
        }
    )
    assert _check(pack.validate(revisions), "PIT_VERSION_DEDUP").status == "PASSED"

    early = pack.read_as_of(revisions, as_of=BASE + timedelta(hours=1, minutes=30))
    latest = pack.read_as_of(revisions, as_of=BASE + timedelta(hours=3))
    assert early.get_column("value").to_list() == [100.0]
    assert latest.get_column("value").to_list() == [120.0]


def test_temporal_core_rejects_mutated_version_tag() -> None:
    pack = TemporalCorePack()
    clean = _temporal_panel()

    report = pack.validate(clean, expected_content_hash="0" * 64)
    check = _check(report, "IMMUTABLE_VERSION_TAG")

    assert check.status == "FAILED"
    assert "different content" in check.message
    assert "expected_hash" in check.details
    with pytest.raises(VersionMutationError):
        pack.validate(clean, expected_content_hash="0" * 64, raise_on_failure=True)

    immutable_pack = TemporalCorePack()
    digest = immutable_pack.assert_version_immutable(clean)
    assert len(digest) == 64
    mutated = clean.with_columns((pl.col("value") + 1.0).alias("value"))
    with pytest.raises(VersionMutationError, match="different content"):
        immutable_pack.assert_version_immutable(mutated)


def test_temporal_core_flags_overlapping_scd2_intervals() -> None:
    frame = pl.DataFrame(
        {
            "entity_id": ["A", "A"],
            "event_time": [BASE, BASE + timedelta(hours=3)],
            "available_at": [BASE + timedelta(hours=4), BASE + timedelta(hours=4)],
            "valid_from": [BASE, BASE + timedelta(hours=1)],
            "valid_to": [BASE + timedelta(hours=2), BASE + timedelta(hours=3)],
            "value": [1.0, 2.0],
        }
    )

    report = TemporalCorePack().validate(frame)
    check = _check(report, "SCD2_NON_OVERLAPPING")

    assert check.status == "FAILED"
    assert check.details["failure_count"] == 1


def test_temporal_core_ablation_isolation() -> None:
    base = LineageManifest.capture(
        cleaning_config={"drop_nulls": True},
        transform=TransformSignature(code_hash="transform-a"),
    )
    changed_transform = LineageManifest.capture(
        cleaning_config={"drop_nulls": True},
        transform=TransformSignature(code_hash="transform-b"),
    )
    changed_two_axes = LineageManifest.capture(
        cleaning_config={"drop_nulls": False},
        transform=TransformSignature(code_hash="transform-b"),
    )

    isolated = TemporalCorePack().validate(
        _temporal_panel(),
        base_manifest=base,
        candidate_manifest=changed_transform,
    )
    assert _check(isolated, "SINGLE_VARIABLE_ABLATION").status == "PASSED"

    with pytest.warns(NonIsolatedAblationWarning):
        report = TemporalCorePack().validate(
            _temporal_panel(),
            base_manifest=base,
            candidate_manifest=changed_two_axes,
        )
    check = _check(report, "SINGLE_VARIABLE_ABLATION")
    assert check.status == "WARNING"
    assert check.details["changed_axes"] == ["cleaning", "transform"]
    assert report.status == "WARNING"
    assert report.passed is True
    assert report.warnings


def test_apply_publication_lag_fixed_and_calendar() -> None:
    frame = pl.DataFrame(
        {
            "available_at": [
                datetime(2026, 1, 1, 3, 0, tzinfo=UTC),
                datetime(2026, 1, 1, 23, 30, tzinfo=UTC),
            ]
        }
    )

    lagged = apply_publication_lag(frame, fixed_lag=timedelta(hours=6))
    assert lagged.get_column("available_at").to_list() == [
        datetime(2026, 1, 1, 9, 0, tzinfo=UTC),
        datetime(2026, 1, 2, 5, 30, tzinfo=UTC),
    ]

    aligned = apply_publication_lag(frame, calendar_align_hour_utc=22)
    assert aligned.get_column("available_at").to_list() == [
        datetime(2026, 1, 1, 22, 0, tzinfo=UTC),
        datetime(2026, 1, 2, 22, 0, tzinfo=UTC),
    ]

    custom = apply_publication_lag(
        frame.rename({"available_at": "published_at"}),
        col_map=TemporalCoreColMap(knowledge_time_col="published_at"),
        fixed_lag=timedelta(hours=1),
    )
    assert custom.get_column("published_at").to_list() == [
        datetime(2026, 1, 1, 4, 0, tzinfo=UTC),
        datetime(2026, 1, 2, 0, 30, tzinfo=UTC),
    ]

    with pytest.raises(ValueError, match="provide fixed_lag"):
        apply_publication_lag(frame)
    with pytest.raises(ValueError, match="non-negative"):
        apply_publication_lag(frame, fixed_lag=timedelta(hours=-1))
    with pytest.raises(ValueError, match=r"\[0, 23\]"):
        apply_publication_lag(frame, calendar_align_hour_utc=24)
    with pytest.raises(ValueError, match="missing knowledge column"):
        apply_publication_lag(pl.DataFrame({"value": [1.0]}), fixed_lag=timedelta(hours=1))
    with pytest.raises(ValueError, match="timezone-aware"):
        apply_publication_lag(
            pl.DataFrame({"available_at": [datetime(2026, 1, 1, 3, 0)]}),
            calendar_align_hour_utc=22,
        )
