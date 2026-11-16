"""`temporal-core`: dual-timeline, version, SCD2, and ablation gates.

The pack is the physical-law layer for every other cleanhatch pack.  It keeps
the six core rules in one place while exposing declarative column mapping so
callers never need to copy the operators into their own repositories.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import cast

import polars as pl
from pydantic import BaseModel, ConfigDict

from ..errors import InvarViolationError, VersionMutationError
from ..hashing import content_hash
from ..invar import GateCheck, GateReport, InvarRule, Stage
from ..lineage import LineageManifest, TransformSignature
from ..spec import TemporalSpec
from ..temporal import TemporalGate
from ..tolerance import ToleranceProfile, at_least
from . import CanaryCheckResult, CanarySuiteReport

_TZ_RULE = "TZ_AWARE_COORDINATES"
_CAUSAL_RULE = "CAUSAL_AVAILABILITY"
_DEDUP_RULE = "PIT_VERSION_DEDUP"
_IMMUTABLE_RULE = "IMMUTABLE_VERSION_TAG"
_SCD2_RULE = "SCD2_NON_OVERLAPPING"
_ABLATION_RULE = "SINGLE_VARIABLE_ABLATION"


class TemporalCoreColMap(BaseModel):
    """Physical column names for the temporal-core contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    entity_col: str = "entity_id"
    event_time_col: str = "event_time"
    knowledge_time_col: str = "available_at"
    decision_time_col: str = "decision_time"
    valid_from_col: str = "valid_from"
    valid_to_col: str = "valid_to"
    version_tag: str = "v1"

    def to_spec(self) -> TemporalSpec:
        return TemporalSpec(
            entity_col=self.entity_col,
            event_time_col=self.event_time_col,
            knowledge_time_col=self.knowledge_time_col,
            decision_time_col=self.decision_time_col,
            version_tag=self.version_tag,
        )


class TemporalCorePack:
    """Six-rule gate pack for point-in-time temporal tables."""

    name = "temporal-core"

    def __init__(
        self,
        *,
        col_map: TemporalCoreColMap | None = None,
        tolerance: ToleranceProfile | None = None,
    ) -> None:
        self._col_map = col_map if col_map is not None else TemporalCoreColMap()
        self._spec = self._col_map.to_spec()
        # temporal-core has no monetary identity; its coordinate and version
        # comparisons are exact by design.  The shared profile keeps the tier
        # explicit when the pack is composed with the financial packs.
        self._tolerance = (
            tolerance if tolerance is not None else ToleranceProfile(mode="exact_int_cents")
        )
        self._version_hashes: dict[str, str] = {}

    @property
    def spec(self) -> TemporalSpec:
        return self._spec

    @property
    def col_map(self) -> TemporalCoreColMap:
        return self._col_map

    @property
    def tolerance(self) -> ToleranceProfile:
        return self._tolerance

    def rules(self, stage: Stage = Stage.INGEST) -> tuple[InvarRule, ...]:
        """Return the row-vectorized subset of the six core rules."""

        event = self._col_map.event_time_col
        knowledge = self._col_map.knowledge_time_col
        return (
            InvarRule(
                name=_CAUSAL_RULE,
                predicate=at_least(
                    pl.col(knowledge).dt.convert_time_zone("UTC"),
                    pl.col(event).dt.convert_time_zone("UTC"),
                    profile=self._tolerance,
                ),
                message="available_at must be greater than or equal to event_time",
                severity="ERROR",
            ),
        )

    def validate(
        self,
        frame: pl.DataFrame,
        *,
        stage: Stage | str = Stage.INGEST,
        raise_on_failure: bool = False,
        **kwargs: object,
    ) -> GateReport:
        """Run all six temporal-core checks and return structured evidence.

        Recognized options: ``expected_content_hash``, ``temporal_gate``,
        ``base_manifest``, ``candidate_manifest``, ``scd2_dimension``, and
        ``as_of``.
        """

        expected_content_hash = cast(str | None, kwargs.pop("expected_content_hash", None))
        temporal_gate = cast(TemporalGate | None, kwargs.pop("temporal_gate", None))
        base_manifest = cast(LineageManifest | None, kwargs.pop("base_manifest", None))
        candidate_manifest = cast(LineageManifest | None, kwargs.pop("candidate_manifest", None))
        scd2_dimension = cast(pl.DataFrame | None, kwargs.pop("scd2_dimension", None))
        as_of = cast(datetime | None, kwargs.pop("as_of", None))
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"unsupported temporal-core validation options: {unexpected}")
        active_stage = Stage(stage)
        checks: list[GateCheck] = [
            self._check_timezone_aware(frame, stage=active_stage),
            self._check_causal_availability(frame, stage=active_stage),
            self._check_pit_version_dedup(frame, stage=active_stage),
        ]
        immutable_check, mutation_error = self._check_immutable_version(
            frame,
            stage=active_stage,
            expected_content_hash=expected_content_hash,
            temporal_gate=temporal_gate,
        )
        checks.append(immutable_check)
        checks.append(
            self._check_scd2_non_overlapping(
                frame,
                stage=active_stage,
                scd2_dimension=scd2_dimension,
                as_of=as_of,
            )
        )
        warnings: list[str] = []
        if base_manifest is not None and candidate_manifest is not None:
            ablation_check = self.check_ablation(
                base_manifest,
                candidate_manifest,
                stage=active_stage,
            )
            checks.append(ablation_check)
            if ablation_check.status == "WARNING":
                warnings.append(f"{_ABLATION_RULE}: {ablation_check.message}")
        else:
            checks.append(
                GateCheck(
                    name=_ABLATION_RULE,
                    stage=active_stage,
                    status="SKIPPED",
                    message="base_manifest and candidate_manifest were not both supplied",
                )
            )
        if mutation_error is not None and raise_on_failure:
            raise mutation_error
        report = GateReport(active_stage, tuple(checks), tuple(warnings))
        if raise_on_failure and not report.passed:
            raise InvarViolationError(
                f"{active_stage.value} temporal-core gate failed",
                details={"gate_report": report.to_dict()},
            )
        return report

    def read_as_of(self, frame: pl.DataFrame, *, as_of: datetime) -> pl.DataFrame:
        """Converge to the latest revision visible at ``as_of``."""

        return TemporalGate(frame, spec=self._spec).read_as_of(as_of=as_of)

    def assert_version_immutable(
        self,
        frame: pl.DataFrame,
        *,
        expected_content_hash: str | None = None,
        version_tag: str | None = None,
        gate: TemporalGate | None = None,
    ) -> str:
        """Reject content drift under a reused version tag."""

        tag = version_tag or self._col_map.version_tag
        digest = content_hash(frame)
        if expected_content_hash is not None and expected_content_hash != digest:
            raise VersionMutationError(
                f"version tag {tag!r} refers to different content; bump the version tag "
                "instead of overwriting history",
                details={
                    "version_tag": tag,
                    "expected_hash": expected_content_hash,
                    "current_hash": digest,
                },
            )
        previous = self._version_hashes.get(tag)
        if expected_content_hash is None and previous is not None and previous != digest:
            raise VersionMutationError(
                f"version tag {tag!r} refers to different content; bump the version tag "
                "instead of overwriting history",
                details={"version_tag": tag, "previous_hash": previous, "current_hash": digest},
            )
        if gate is not None:
            gate.assert_version_immutable(tag, frame)
        self._version_hashes[tag] = digest
        return digest

    def check_ablation(
        self,
        base: LineageManifest,
        candidate: LineageManifest,
        *,
        stage: Stage = Stage.INGEST,
    ) -> GateCheck:
        """Compare two manifests and expose non-isolated ablations as warnings."""

        report = LineageManifest.ablation_diff(base, candidate)
        changed = ", ".join(report.changed_axes) if report.changed_axes else "none"
        details: dict[str, object] = {
            "changed_axes": list(report.changed_axes),
            "is_clean_ablation": report.is_clean_ablation,
            "is_no_op": report.is_no_op,
        }
        if not report.is_clean_ablation:
            message = (
                report.warnings[0]
                if report.warnings
                else f"ablation changed more than one variable at once: {changed}"
            )
            return GateCheck(
                name=_ABLATION_RULE,
                stage=stage,
                status="WARNING",
                message=message,
                details=details,
            )
        return GateCheck(
            name=_ABLATION_RULE,
            stage=stage,
            status="PASSED",
            message=f"ablation changes at most one variable (changed axes: {changed})",
            details=details,
        )

    def run_canary_suite(self) -> CanarySuiteReport:
        """Prove every core rule catches its poisoned frame."""

        base = datetime(2026, 1, 1, tzinfo=UTC)
        entity = self._col_map.entity_col
        event = self._col_map.event_time_col
        knowledge = self._col_map.knowledge_time_col
        results: list[CanaryCheckResult] = []

        naive = pl.DataFrame(
            {
                entity: ["A"],
                event: [datetime(2026, 1, 1)],
                knowledge: [datetime(2026, 1, 1, 1)],
                "value": [1.0],
            }
        )
        results.append(
            _canary_result(
                "canary_naive_coordinates",
                _TZ_RULE,
                self.validate(naive),
                accepted=("FAILED",),
            )
        )

        causal = pl.DataFrame(
            {
                entity: ["A"],
                event: [base],
                knowledge: [base - timedelta(hours=1)],
                "value": [1.0],
            }
        )
        results.append(
            _canary_result(
                "canary_future_knowledge_time",
                _CAUSAL_RULE,
                self.validate(causal),
                accepted=("FAILED",),
            )
        )

        conflicting = pl.DataFrame(
            {
                entity: ["A", "A"],
                event: [base, base],
                knowledge: [base + timedelta(hours=1), base + timedelta(hours=1)],
                "value": [1.0, 2.0],
            }
        )
        results.append(
            _canary_result(
                "canary_conflicting_revision",
                _DEDUP_RULE,
                self.validate(conflicting),
                accepted=("FAILED",),
            )
        )

        clean = self._clean_frame()
        results.append(
            _canary_result(
                "canary_mutated_version_snapshot",
                _IMMUTABLE_RULE,
                self.validate(clean, expected_content_hash="0" * 64),
                accepted=("FAILED",),
            )
        )

        overlapping = pl.DataFrame(
            {
                entity: ["A", "A"],
                event: [base, base + timedelta(hours=3)],
                knowledge: [base + timedelta(hours=4), base + timedelta(hours=4)],
                self._col_map.valid_from_col: [base, base + timedelta(hours=1)],
                self._col_map.valid_to_col: [base + timedelta(hours=2), base + timedelta(hours=3)],
                "value": [1.0, 2.0],
            }
        )
        results.append(
            _canary_result(
                "canary_overlapping_scd2_intervals",
                _SCD2_RULE,
                self.validate(overlapping),
                accepted=("FAILED",),
            )
        )

        base_manifest = LineageManifest.capture(
            cleaning_config={"drop_nulls": True},
            transform=TransformSignature(code_hash="canary-transform-a"),
        )
        candidate_manifest = LineageManifest.capture(
            cleaning_config={"drop_nulls": False},
            transform=TransformSignature(code_hash="canary-transform-b"),
        )
        results.append(
            _canary_result(
                "canary_multi_axis_ablation",
                _ABLATION_RULE,
                self.validate(
                    clean,
                    base_manifest=base_manifest,
                    candidate_manifest=candidate_manifest,
                ),
                accepted=("WARNING",),
            )
        )
        return CanarySuiteReport(self.name, tuple(results))

    def render_recipe_scaffold(self) -> str:
        return '''"""cleanhatch temporal-core recipe (declarative configuration only).

The gate operators and the report seal stay inside the installed cleanhatch
package (1A-Guard). Edit the column map and policy values, not the operators.
"""

from datetime import timedelta
from pathlib import Path

import polars as pl

from cleanhatch.packs.temporal_core import (
    TemporalCoreColMap,
    TemporalCorePack,
    apply_publication_lag,
)

col_map = TemporalCoreColMap(
    entity_col="entity_id",
    event_time_col="event_time",
    knowledge_time_col="available_at",
    decision_time_col="decision_time",
    valid_from_col="valid_from",
    valid_to_col="valid_to",
    version_tag="v1",
)
pack = TemporalCorePack(col_map=col_map)

frame = pl.read_parquet(Path("data/temporal_core.parquet"))
frame = apply_publication_lag(
    frame,
    col_map=col_map,
    fixed_lag=timedelta(hours=6),
    calendar_align_hour_utc=22,
)

report = pack.validate(frame, raise_on_failure=False)
print(report.to_dict())

canary = pack.run_canary_suite()
assert canary.all_caught, canary.to_dict()
'''

    def render_agents_md(self) -> str:
        return """# AGENTS.md — cleanhatch temporal-core

## Scope

This repository runs the `temporal-core` gate pack for point-in-time tables.
The pack owns the operators; this file only declares column roles and policy.

## Non-negotiable rules

1. `event_time` and `available_at` must be timezone-aware datetimes.
2. `available_at` must never precede `event_time`.
3. Duplicate `(entity_id, event_time, available_at)` rows must not carry
   conflicting payload values. Read history through `pack.read_as_of(...)`.
4. A `version_tag` is immutable. Publish a new tag instead of overwriting it.
5. SCD2 intervals `[valid_from, valid_to)` must not overlap per entity.
6. Experiments must change exactly one lineage axis at a time.

## Workflow

- Run `pack.validate(frame, raise_on_failure=False)` before publishing data.
- Treat `ERROR` checks as hard blockers and `WARNING` checks as review items.
- Run `pack.run_canary_suite()` after changing the column map or upgrade.
- Keep `event_time` and `available_at` in UTC for portable evidence.

## Guardrails

- Do not copy gate operators into application code (`1A-Guard`).
- Do not edit a signed license badge or trace code by hand.
"""

    def _check_timezone_aware(self, frame: pl.DataFrame, *, stage: Stage) -> GateCheck:
        columns = (self._col_map.event_time_col, self._col_map.knowledge_time_col)
        missing = sorted(set(columns) - set(frame.columns))
        if missing:
            return GateCheck(
                name=_TZ_RULE,
                stage=stage,
                status="FAILED",
                message=f"time coordinate columns are missing: {missing}",
                details={"missing": missing},
            )
        non_tz = {
            column: str(frame.schema[column])
            for column in columns
            if not _is_aware_datetime(frame.schema[column])
        }
        if non_tz:
            return GateCheck(
                name=_TZ_RULE,
                stage=stage,
                status="FAILED",
                message="event_time and available_at must be timezone-aware Datetime",
                details={"non_tz_columns": non_tz},
            )
        if frame.is_empty():
            return GateCheck(
                name=_TZ_RULE,
                stage=stage,
                status="SKIPPED",
                message="frame is empty; timezone check passed on schema",
            )
        return GateCheck(
            name=_TZ_RULE,
            stage=stage,
            status="PASSED",
            message="time coordinate columns are timezone-aware",
        )

    def _check_causal_availability(self, frame: pl.DataFrame, *, stage: Stage) -> GateCheck:
        event = self._col_map.event_time_col
        knowledge = self._col_map.knowledge_time_col
        missing = sorted({event, knowledge} - set(frame.columns))
        if missing:
            return GateCheck(
                name=_CAUSAL_RULE,
                stage=stage,
                status="FAILED",
                message=f"time coordinate columns are missing: {missing}",
                details={"missing": missing},
            )
        if not (
            _is_aware_datetime(frame.schema[event]) and _is_aware_datetime(frame.schema[knowledge])
        ):
            return GateCheck(
                name=_CAUSAL_RULE,
                stage=stage,
                status="FAILED",
                message="event_time and available_at must be timezone-aware Datetime",
            )
        if frame.is_empty():
            return GateCheck(
                name=_CAUSAL_RULE,
                stage=stage,
                status="SKIPPED",
                message="frame is empty; causality is not applicable",
            )
        violations = frame.filter(_utc_expr(frame, knowledge) < _utc_expr(frame, event)).select(
            self._col_map.entity_col, event, knowledge
        )
        if not violations.is_empty():
            return GateCheck(
                name=_CAUSAL_RULE,
                stage=stage,
                status="FAILED",
                message="available_at must be greater than or equal to event_time",
                details={
                    "failure_count": violations.height,
                    "violation_count": violations.height,
                    "violations": violations.head(20).to_dicts(),
                },
            )
        return GateCheck(
            name=_CAUSAL_RULE,
            stage=stage,
            status="PASSED",
            message="knowledge time never precedes event time",
        )

    def _check_pit_version_dedup(self, frame: pl.DataFrame, *, stage: Stage) -> GateCheck:
        entity = self._col_map.entity_col
        event = self._col_map.event_time_col
        knowledge = self._col_map.knowledge_time_col
        keys = [entity, event, knowledge]
        missing = sorted(set(keys) - set(frame.columns))
        if missing:
            return GateCheck(
                name=_DEDUP_RULE,
                stage=stage,
                status="FAILED",
                message=f"point-in-time key columns are missing: {missing}",
                details={"missing": missing},
            )
        if frame.is_empty():
            return GateCheck(
                name=_DEDUP_RULE,
                stage=stage,
                status="SKIPPED",
                message="frame is empty; version deduplication is not applicable",
            )
        payload = [column for column in frame.columns if column not in keys]
        if not payload:
            return GateCheck(
                name=_DEDUP_RULE,
                stage=stage,
                status="PASSED",
                message="frame has no payload columns that could conflict",
            )
        conflicts = (
            frame.group_by(keys)
            .agg(
                pl.struct(payload).n_unique().alias("distinct_payloads"),
                pl.len().alias("row_count"),
            )
            .filter(pl.col("distinct_payloads") > 1)
        )
        if not conflicts.is_empty():
            return GateCheck(
                name=_DEDUP_RULE,
                stage=stage,
                status="FAILED",
                message=(
                    "duplicate (entity, event_time, available_at) rows carry conflicting "
                    "payload values"
                ),
                details={
                    "failure_count": int(conflicts["row_count"].sum()),
                    "conflict_group_count": conflicts.height,
                    "conflicting_row_count": int(conflicts["row_count"].sum()),
                    "conflicts": conflicts.head(20).to_dicts(),
                },
            )
        return GateCheck(
            name=_DEDUP_RULE,
            stage=stage,
            status="PASSED",
            message="duplicate point-in-time keys never carry conflicting payloads",
        )

    def _check_immutable_version(
        self,
        frame: pl.DataFrame,
        *,
        stage: Stage,
        expected_content_hash: str | None,
        temporal_gate: TemporalGate | None,
    ) -> tuple[GateCheck, VersionMutationError | None]:
        if expected_content_hash is None and temporal_gate is None:
            return (
                GateCheck(
                    name=_IMMUTABLE_RULE,
                    stage=stage,
                    status="SKIPPED",
                    message="no expected content hash or temporal gate was supplied",
                ),
                None,
            )
        try:
            digest = self.assert_version_immutable(
                frame,
                expected_content_hash=expected_content_hash,
                gate=temporal_gate,
            )
        except VersionMutationError as error:
            return (
                GateCheck(
                    name=_IMMUTABLE_RULE,
                    stage=stage,
                    status="FAILED",
                    message=str(error),
                    details=dict(error.details),
                ),
                error,
            )
        return (
            GateCheck(
                name=_IMMUTABLE_RULE,
                stage=stage,
                status="PASSED",
                message="version content hash matches the registered snapshot",
                details={
                    "content_hash": digest,
                    "version_tag": self._col_map.version_tag,
                },
            ),
            None,
        )

    def _check_scd2_non_overlapping(
        self,
        frame: pl.DataFrame,
        *,
        stage: Stage,
        scd2_dimension: pl.DataFrame | None,
        as_of: datetime | None,
    ) -> GateCheck:
        start_col = self._col_map.valid_from_col
        end_col = self._col_map.valid_to_col
        explicit_dimension = scd2_dimension is not None
        if not explicit_dimension and not {start_col, end_col}.issubset(frame.columns):
            return GateCheck(
                name=_SCD2_RULE,
                stage=stage,
                status="SKIPPED",
                message="SCD2 interval columns are not present",
            )
        dimension = scd2_dimension if scd2_dimension is not None else frame
        entity = self._col_map.entity_col
        missing = sorted({entity, start_col, end_col} - set(dimension.columns))
        if missing:
            return GateCheck(
                name=_SCD2_RULE,
                stage=stage,
                status="FAILED",
                message=f"SCD2 dimension columns are missing: {missing}",
                details={"missing": missing},
            )
        knowledge = self._col_map.knowledge_time_col
        if as_of is not None and knowledge in dimension.columns:
            dimension = dimension.filter(_utc_expr(dimension, knowledge) <= pl.lit(as_of))
        if dimension.is_empty():
            return GateCheck(
                name=_SCD2_RULE,
                stage=stage,
                status="SKIPPED",
                message="no SCD2 rows are visible at the requested cutoff",
            )
        ordered = dimension.sort([entity, start_col])
        overlaps = (
            ordered.with_columns(
                pl.col(start_col).shift(1).over(entity).alias("__previous_start"),
                pl.col(end_col).shift(1).over(entity).alias("__previous_end"),
            )
            .filter(
                pl.col("__previous_start").is_not_null()
                & (
                    pl.col("__previous_end").is_null()
                    | (pl.col("__previous_end") > pl.col(start_col))
                )
            )
            .select(entity, "__previous_start", "__previous_end", start_col, end_col)
        )
        if not overlaps.is_empty():
            return GateCheck(
                name=_SCD2_RULE,
                stage=stage,
                status="FAILED",
                message="SCD2 intervals overlap for the same entity",
                details={
                    "failure_count": overlaps.height,
                    "overlap_count": overlaps.height,
                    "overlaps": overlaps.head(20).to_dicts(),
                },
            )
        return GateCheck(
            name=_SCD2_RULE,
            stage=stage,
            status="PASSED",
            message="SCD2 intervals are non-overlapping and half-open",
        )

    def _clean_frame(self) -> pl.DataFrame:
        base = datetime(2026, 1, 1, tzinfo=UTC)
        return pl.DataFrame(
            {
                self._col_map.entity_col: ["A", "A"],
                self._col_map.event_time_col: [base, base + timedelta(hours=1)],
                self._col_map.knowledge_time_col: [
                    base + timedelta(hours=1),
                    base + timedelta(hours=2),
                ],
                "value": [1.0, 2.0],
            }
        )


def apply_publication_lag(
    frame: pl.DataFrame,
    *,
    col_map: TemporalCoreColMap | None = None,
    fixed_lag: timedelta | None = None,
    calendar_align_hour_utc: int | None = None,
) -> pl.DataFrame:
    """Shift knowledge time by a fixed lag and/or a UTC calendar slot."""

    if fixed_lag is None and calendar_align_hour_utc is None:
        raise ValueError("provide fixed_lag or calendar_align_hour_utc")
    if fixed_lag is not None and fixed_lag < timedelta(0):
        raise ValueError("fixed_lag must be non-negative")
    if calendar_align_hour_utc is not None and not 0 <= calendar_align_hour_utc <= 23:
        raise ValueError("calendar_align_hour_utc must be in [0, 23]")
    mapping = col_map if col_map is not None else TemporalCoreColMap()
    knowledge = mapping.knowledge_time_col
    if knowledge not in frame.columns:
        raise ValueError(f"frame is missing knowledge column {knowledge!r}")
    result = frame
    if fixed_lag is not None:
        result = result.with_columns(pl.col(knowledge) + fixed_lag)
    if calendar_align_hour_utc is not None:
        dtype = result.schema[knowledge]
        if not (isinstance(dtype, pl.Datetime) and dtype.time_zone is not None):
            raise ValueError(f"{knowledge} must be timezone-aware for calendar-aligned publication")
        hour = calendar_align_hour_utc
        knowledge_utc = pl.col(knowledge).dt.convert_time_zone("UTC")
        aligned = knowledge_utc.dt.truncate("1d") + timedelta(hours=hour)
        result = result.with_columns(
            pl.when(aligned < knowledge_utc)
            .then(aligned + timedelta(days=1))
            .otherwise(aligned)
            .alias(knowledge)
        )
    return result


def _canary_result(
    name: str,
    gate: str,
    report: GateReport,
    *,
    accepted: tuple[str, ...],
) -> CanaryCheckResult:
    check = next(item for item in report.checks if item.name == gate)
    return CanaryCheckResult(
        name=name,
        caught=check.status in accepted,
        gate=gate,
        message=check.message,
        details=check.to_dict(),
    )


def _is_aware_datetime(dtype: pl.DataType) -> bool:
    return isinstance(dtype, pl.Datetime) and dtype.time_zone is not None


def _utc_expr(frame: pl.DataFrame, name: str) -> pl.Expr:
    dtype = frame.schema[name]
    if isinstance(dtype, pl.Datetime) and dtype.time_zone is not None:
        return pl.col(name).dt.convert_time_zone("UTC")
    return pl.col(name)


__all__ = [
    "TemporalCoreColMap",
    "TemporalCorePack",
    "apply_publication_lag",
]
