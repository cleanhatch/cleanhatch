"""Lifecycle invariant gates for ingest, cleaning, and feature output.

A gate is a small, composable middleware.  It knows only column roles from
:class:`TemporalSpec`; application adapters add domain rules as
:class:`InvarRule` predicates.  Every check returns structured evidence so
the same report can be attached to a lineage manifest.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Literal, cast

import polars as pl

from .errors import CausalFailure, InvarViolationError
from .spec import TemporalSpec

GateStatus = Literal["PASSED", "WARNING", "FAILED", "SKIPPED"]
RuleSeverity = Literal["ERROR", "WARN"]


class Stage(StrEnum):
    """Standard lifecycle points where invariants are evaluated."""

    INGEST = "ingest"
    CLEAN = "clean"
    FEATURE = "feature"


@dataclass(frozen=True, slots=True)
class InvarRule:
    """One row-level predicate added by a domain adapter.

    ``predicate`` must evaluate to a boolean scalar per row.  Null results are
    treated as failures: an unknown invariant result is not a pass.
    ``severity="WARN"`` isolates a rule violation as a warning that does not
    fail the gate, while ``skip_when`` removes provisional rows from the
    effective coverage denominator.
    """

    name: str
    predicate: pl.Expr
    message: str = ""
    severity: RuleSeverity = "ERROR"
    skip_when: pl.Expr | None = None


@dataclass(frozen=True, slots=True)
class GateCheck:
    """One named check inside a gate report."""

    name: str
    stage: Stage
    status: GateStatus
    message: str
    details: dict[str, object] = field(default_factory=dict[str, object])

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["stage"] = self.stage.value
        payload["details"] = dict(self.details)
        return payload


@dataclass(frozen=True, slots=True)
class GateReport:
    """Immutable evidence package produced by one gate invocation."""

    stage: Stage
    checks: tuple[GateCheck, ...]
    warnings: tuple[str, ...] = ()

    @property
    def status(self) -> GateStatus:
        if any(check.status == "FAILED" for check in self.checks):
            return "FAILED"
        if self.warnings or any(check.status == "WARNING" for check in self.checks):
            return "WARNING"
        if not self.checks or all(check.status == "SKIPPED" for check in self.checks):
            return "SKIPPED"
        return "PASSED"

    @property
    def passed(self) -> bool:
        return self.status != "FAILED"

    def to_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage.value,
            "status": self.status,
            "checks": [check.to_dict() for check in self.checks],
            "warnings": list(self.warnings),
        }


class InvarGate:
    """Run generic point-in-time invariants at one or more lifecycle stages."""

    def __init__(
        self,
        spec: TemporalSpec | None = None,
        *,
        max_dead_value_ratio: float = 0.05,
        max_zscore_drift: float = 4.0,
        max_null_ratio: float = 1.0,
    ) -> None:
        if not 0.0 <= max_dead_value_ratio <= 1.0:
            raise ValueError("max_dead_value_ratio must be in [0, 1]")
        if not 0.0 <= max_null_ratio <= 1.0:
            raise ValueError("max_null_ratio must be in [0, 1]")
        if not math.isfinite(max_zscore_drift) or max_zscore_drift < 0:
            raise ValueError("max_zscore_drift must be finite and non-negative")
        self._spec = spec if spec is not None else TemporalSpec()
        self._max_dead_value_ratio = max_dead_value_ratio
        self._max_zscore_drift = max_zscore_drift
        self._max_null_ratio = max_null_ratio

    @property
    def spec(self) -> TemporalSpec:
        return self._spec

    def run(
        self,
        frame: pl.DataFrame,
        *,
        stage: Stage | str,
        raise_on_failure: bool = True,
        **options: object,
    ) -> GateReport:
        """Dispatch to the stage-specific check and optionally raise."""

        stage = Stage(stage)
        if stage is Stage.INGEST:
            report = self.check_ingest(
                frame,
                raise_on_failure=False,
                **options,  # pyright: ignore[reportArgumentType]
            )
        elif stage is Stage.CLEAN:
            report = self.check_clean(
                frame,
                raise_on_failure=False,
                **options,  # pyright: ignore[reportArgumentType]
            )
        elif stage is Stage.FEATURE:
            report = self.check_feature(
                frame,
                raise_on_failure=False,
                **options,  # pyright: ignore[reportArgumentType]
            )
        else:
            raise ValueError(f"unsupported invariant stage: {stage!r}")
        if raise_on_failure and not report.passed:
            raise InvarViolationError(
                f"{stage.value} invariant gate failed",
                details={"gate_report": report.to_dict()},
            )
        return report

    def check_ingest(
        self,
        frame: pl.DataFrame,
        *,
        required_columns: Sequence[str] = (),
        rules: Sequence[InvarRule] = (),
        require_sorted: bool = True,
        raise_on_failure: bool = True,
    ) -> GateReport:
        """Check coordinates, causality, uniqueness, order, and domain rules."""

        spec = self._spec
        checks: list[GateCheck] = []
        checks.append(
            _check_required_columns(
                frame,
                stage=Stage.INGEST,
                required=(*spec.time_columns, spec.entity_col, *required_columns),
            )
        )
        checks.append(_check_coordinate_nulls(frame, spec=spec, stage=Stage.INGEST))
        checks.append(_check_timezone_aware(frame, spec=spec, stage=Stage.INGEST))
        checks.append(_check_causality(frame, spec=spec, stage=Stage.INGEST))
        checks.append(_check_unique_key(frame, spec=spec, stage=Stage.INGEST))
        if require_sorted:
            checks.append(_check_sorted(frame, spec=spec, stage=Stage.INGEST))
        rule_checks, rule_warnings = _check_rules(frame, rules, stage=Stage.INGEST)
        checks.extend(rule_checks)
        report = GateReport(Stage.INGEST, tuple(checks), tuple(rule_warnings))
        _raise_if_needed(report, raise_on_failure)
        return report

    def check_clean(
        self,
        frame: pl.DataFrame,
        *,
        baseline: pl.DataFrame | None = None,
        numeric_cols: Sequence[str] | None = None,
        grid_freq_ms: int | None = None,
        allow_multiples: bool = False,
        robust_drift: bool = True,
        rules: Sequence[InvarRule] = (),
        raise_on_failure: bool = True,
    ) -> GateReport:
        """Check grid continuity, dead values, and distribution drift."""

        spec = self._spec
        checks: list[GateCheck] = []
        required = (*spec.time_columns, spec.entity_col)
        checks.append(_check_required_columns(frame, stage=Stage.CLEAN, required=required))
        if grid_freq_ms is not None:
            checks.append(
                _check_grid_continuity(
                    frame,
                    spec=spec,
                    freq_ms=grid_freq_ms,
                    allow_multiples=allow_multiples,
                )
            )
        columns = tuple(numeric_cols) if numeric_cols is not None else spec.numeric_columns
        if columns:
            checks.append(
                _check_dead_values(
                    frame,
                    spec=spec,
                    columns=columns,
                    threshold=self._max_dead_value_ratio,
                )
            )
            if baseline is not None:
                drift_checks, warnings = _check_zscore_drift(
                    frame,
                    baseline,
                    columns=columns,
                    threshold=self._max_zscore_drift,
                    robust=robust_drift,
                )
                checks.extend(drift_checks)
            else:
                warnings = []
        else:
            warnings = []
        rule_checks, rule_warnings = _check_rules(frame, rules, stage=Stage.CLEAN)
        checks.extend(rule_checks)
        warnings.extend(rule_warnings)
        report = GateReport(Stage.CLEAN, tuple(checks), tuple(warnings))
        _raise_if_needed(report, raise_on_failure)
        return report

    def check_feature(
        self,
        frame: pl.DataFrame,
        *,
        value_col: str = "value",
        required_columns: Sequence[str] = (),
        decision_time_col: str | None = None,
        entity_col: str | None = None,
        causal_check: Callable[[], object] | None = None,
        rules: Sequence[InvarRule] = (),
        raise_on_failure: bool = True,
    ) -> GateReport:
        """Check output key, finiteness, null ratio, and optional causal gate."""

        spec = self._spec
        decision = decision_time_col or spec.decision_time_col
        entity = entity_col or spec.entity_col
        checks: list[GateCheck] = [
            _check_required_columns(
                frame,
                stage=Stage.FEATURE,
                required=(decision, entity, value_col, *required_columns),
            ),
            _check_feature_values(
                frame,
                value_col=value_col,
                max_null_ratio=self._max_null_ratio,
            ),
            _check_unique_decision_key(
                frame,
                decision_col=decision,
                entity_col=entity,
            ),
        ]
        if causal_check is not None:
            checks.append(_check_causal(causal_check))
        rule_checks, rule_warnings = _check_rules(frame, rules, stage=Stage.FEATURE)
        checks.extend(rule_checks)
        report = GateReport(Stage.FEATURE, tuple(checks), tuple(rule_warnings))
        _raise_if_needed(report, raise_on_failure)
        return report

    @classmethod
    def protect(
        cls,
        *,
        stage: Stage | str,
        gate: InvarGate | None = None,
        **options: object,
    ) -> Callable[[Callable[..., object]], Callable[..., object]]:
        """Decorate a transform and run a gate on its materialized output.

        The wrapped function keeps its original return value (lazy stays lazy),
        while ``wrapper.last_report`` exposes the latest :class:`GateReport`.
        """

        stage = Stage(stage)
        active_gate = gate if gate is not None else cls()

        def decorator(function: Callable[..., object]) -> Callable[..., object]:
            @functools.wraps(function)
            def wrapper(*args: object, **kwargs: object) -> object:
                result = function(*args, **kwargs)
                materialized = result.collect() if isinstance(result, pl.LazyFrame) else result
                if not isinstance(materialized, pl.DataFrame):
                    raise InvarViolationError(
                        f"{stage.value} gate requires a DataFrame or LazyFrame result, "
                        f"got {type(result).__name__}"
                    )
                report = active_gate.run(
                    materialized,
                    stage=stage,
                    **options,  # pyright: ignore[reportArgumentType]
                )
                wrapper.__dict__["last_report"] = report
                return result

            wrapper.__dict__["last_report"] = None
            return wrapper

        return decorator


def _raise_if_needed(report: GateReport, raise_on_failure: bool) -> None:
    if raise_on_failure and not report.passed:
        raise InvarViolationError(
            f"{report.stage.value} invariant gate failed",
            details={"gate_report": report.to_dict()},
        )


def _check_required_columns(
    frame: pl.DataFrame, *, stage: Stage, required: Sequence[str]
) -> GateCheck:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        return GateCheck(
            name="required_columns",
            stage=stage,
            status="FAILED",
            message=f"missing required columns: {missing}",
            details={"missing": missing},
        )
    if frame.is_empty():
        return GateCheck(
            name="required_columns",
            stage=stage,
            status="SKIPPED",
            message="frame is empty; required columns are present but there are no samples",
        )
    return GateCheck(
        name="required_columns",
        stage=stage,
        status="PASSED",
        message="all required columns are present",
    )


def _check_coordinate_nulls(frame: pl.DataFrame, *, spec: TemporalSpec, stage: Stage) -> GateCheck:
    if frame.is_empty():
        return GateCheck(
            name="coordinate_nulls",
            stage=stage,
            status="SKIPPED",
            message="frame is empty; coordinate nulls are not applicable",
        )
    columns = (spec.entity_col, spec.event_time_col, spec.knowledge_time_col)
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        return GateCheck(
            name="coordinate_nulls",
            stage=stage,
            status="FAILED",
            message=f"coordinate columns are missing: {missing}",
            details={"missing": missing},
        )
    nulls = {
        column: int(frame.get_column(column).null_count())
        for column in columns
        if frame.get_column(column).null_count()
    }
    if nulls:
        return GateCheck(
            name="coordinate_nulls",
            stage=stage,
            status="FAILED",
            message="coordinate columns must not contain nulls",
            details={"null_counts": nulls},
        )
    return GateCheck(
        name="coordinate_nulls",
        stage=stage,
        status="PASSED",
        message="coordinate columns contain no nulls",
    )


def _check_timezone_aware(frame: pl.DataFrame, *, spec: TemporalSpec, stage: Stage) -> GateCheck:
    missing = sorted(set(spec.time_columns) - set(frame.columns))
    if missing:
        return GateCheck(
            name="timezone_aware_coordinates",
            stage=stage,
            status="FAILED",
            message=f"time columns are missing: {missing}",
            details={"missing": missing},
        )
    non_tz_columns = {
        column: str(frame.schema[column])
        for column in spec.time_columns
        if not _is_aware_datetime(frame.schema[column])
    }
    if non_tz_columns:
        return GateCheck(
            name="timezone_aware_coordinates",
            stage=stage,
            status="FAILED",
            message="time coordinate columns must be timezone-aware Datetime",
            details={"non_tz_columns": non_tz_columns},
        )
    if frame.is_empty():
        return GateCheck(
            name="timezone_aware_coordinates",
            stage=stage,
            status="SKIPPED",
            message="frame is empty; timezone check passed on schema",
        )
    return GateCheck(
        name="timezone_aware_coordinates",
        stage=stage,
        status="PASSED",
        message="time coordinate columns are timezone-aware",
    )


def _check_causality(frame: pl.DataFrame, *, spec: TemporalSpec, stage: Stage) -> GateCheck:
    if frame.is_empty():
        return GateCheck(
            name="knowledge_time_causality",
            stage=stage,
            status="SKIPPED",
            message="frame is empty; causality is not applicable",
        )
    if not {spec.entity_col, spec.event_time_col, spec.knowledge_time_col}.issubset(frame.columns):
        return GateCheck(
            name="knowledge_time_causality",
            stage=stage,
            status="FAILED",
            message="entity, event, and knowledge columns are required",
        )
    event_dtype = frame.schema[spec.event_time_col]
    knowledge_dtype = frame.schema[spec.knowledge_time_col]
    if not (_is_aware_datetime(event_dtype) and _is_aware_datetime(knowledge_dtype)):
        return GateCheck(
            name="knowledge_time_causality",
            stage=stage,
            status="FAILED",
            message="event_time and knowledge_time must be timezone-aware Datetime",
            details={
                "event_time_dtype": str(event_dtype),
                "knowledge_time_dtype": str(knowledge_dtype),
            },
        )
    violations = frame.filter(
        pl.col(spec.knowledge_time_col).dt.convert_time_zone("UTC")
        < pl.col(spec.event_time_col).dt.convert_time_zone("UTC")
    ).select(spec.entity_col, spec.event_time_col, spec.knowledge_time_col)
    if not violations.is_empty():
        return GateCheck(
            name="knowledge_time_causality",
            stage=stage,
            status="FAILED",
            message="knowledge_time must be >= event_time",
            details={"violations": violations.head(20).to_dicts()},
        )
    return GateCheck(
        name="knowledge_time_causality",
        stage=stage,
        status="PASSED",
        message="knowledge time never precedes event time",
    )


def _check_unique_key(frame: pl.DataFrame, *, spec: TemporalSpec, stage: Stage) -> GateCheck:
    if frame.is_empty():
        return GateCheck(
            name="unique_key",
            stage=stage,
            status="SKIPPED",
            message="frame is empty; key uniqueness is not applicable",
        )
    missing = set(spec.key_columns) - set(frame.columns)
    if missing:
        return GateCheck(
            name="unique_key",
            stage=stage,
            status="FAILED",
            message=f"key columns are missing: {sorted(missing)}",
        )
    duplicates = frame.group_by(list(spec.key_columns)).len().filter(pl.col("len") > 1)
    if not duplicates.is_empty():
        return GateCheck(
            name="unique_key",
            stage=stage,
            status="FAILED",
            message="point-in-time keys must be unique",
            details={"duplicates": duplicates.head(20).to_dicts()},
        )
    return GateCheck(
        name="unique_key",
        stage=stage,
        status="PASSED",
        message="point-in-time keys are unique",
    )


def _check_sorted(frame: pl.DataFrame, *, spec: TemporalSpec, stage: Stage) -> GateCheck:
    missing = set(spec.key_columns) - set(frame.columns)
    if missing:
        return GateCheck(
            name="sorted_key",
            stage=stage,
            status="FAILED",
            message=f"sort columns are missing: {sorted(missing)}",
        )
    if frame.height < 2:
        return GateCheck(
            name="sorted_key",
            stage=stage,
            status="SKIPPED",
            message="frame has fewer than two rows; ordering is not applicable",
        )
    event = pl.col(spec.event_time_col)
    entity = pl.col(spec.entity_col)
    previous_event = event.shift(1)
    previous_entity = entity.shift(1)
    violation = (
        (event < previous_event) | ((event == previous_event) & (entity < previous_entity))
    ).fill_null(False)
    if bool(frame.select(violation.any()).item()):
        return GateCheck(
            name="sorted_key",
            stage=stage,
            status="FAILED",
            message=f"rows must be sorted by {spec.key_columns}",
        )
    return GateCheck(
        name="sorted_key",
        stage=stage,
        status="PASSED",
        message=f"rows are sorted by {spec.key_columns}",
    )


def _check_rules(
    frame: pl.DataFrame, rules: Sequence[InvarRule], *, stage: Stage
) -> tuple[list[GateCheck], list[str]]:
    checks: list[GateCheck] = []
    warnings: list[str] = []
    if frame.is_empty():
        return (
            [
                GateCheck(
                    name=rule.name,
                    stage=stage,
                    status="SKIPPED",
                    message="frame is empty; rule is not applicable",
                )
                for rule in rules
            ],
            [],
        )
    for rule in rules:
        try:
            skipped_rows = 0
            active_frame = frame
            if rule.skip_when is not None:
                skip_mask = rule.skip_when.fill_null(False)
                skipped_rows = int(frame.select(skip_mask.sum()).item() or 0)
                active_frame = frame.filter(~skip_mask)
            if active_frame.is_empty():
                checks.append(
                    GateCheck(
                        name=rule.name,
                        stage=stage,
                        status="SKIPPED",
                        message="all rows skipped by skip_when mask",
                        details={"skipped_rows": skipped_rows, "effective_coverage": 0.0},
                    )
                )
                continue
            result = active_frame.select(rule.predicate.fill_null(False).alias("_ok"))
            failures = int(result.filter(~pl.col("_ok")).height)
        except pl.exceptions.PolarsError as error:
            checks.append(
                GateCheck(
                    name=rule.name,
                    stage=stage,
                    status="FAILED",
                    message=f"rule could not be evaluated: {error}",
                )
            )
            continue
        effective_coverage = round(active_frame.height / frame.height, 6)
        coverage_details: dict[str, object] = (
            {"skipped_rows": skipped_rows, "effective_coverage": effective_coverage}
            if rule.skip_when is not None
            else {}
        )
        if failures:
            details: dict[str, object] = {"failure_count": failures}
            details.update(coverage_details)
            message = rule.message or f"{failures} row(s) violated the invariant"
            if rule.severity == "WARN":
                warnings.append(f"{rule.name}: {message}")
                checks.append(
                    GateCheck(
                        name=rule.name,
                        stage=stage,
                        status="WARNING",
                        message=message,
                        details=details,
                    )
                )
            else:
                checks.append(
                    GateCheck(
                        name=rule.name,
                        stage=stage,
                        status="FAILED",
                        message=message,
                        details=details,
                    )
                )
        else:
            checks.append(
                GateCheck(
                    name=rule.name,
                    stage=stage,
                    status="PASSED",
                    message="invariant holds for every row",
                    details=coverage_details,
                )
            )
    return checks, warnings


def _check_grid_continuity(
    frame: pl.DataFrame,
    *,
    spec: TemporalSpec,
    freq_ms: int,
    allow_multiples: bool = False,
) -> GateCheck:
    if type(freq_ms) is not int or freq_ms < 1:
        raise ValueError("grid_freq_ms must be a positive integer")
    if frame.is_empty():
        return GateCheck(
            name="grid_continuity",
            stage=Stage.CLEAN,
            status="SKIPPED",
            message="frame is empty; grid continuity is not applicable",
        )
    missing = set(spec.key_columns + (spec.entity_col,)) - set(frame.columns)
    if missing:
        return GateCheck(
            name="grid_continuity",
            stage=Stage.CLEAN,
            status="FAILED",
            message=f"grid columns are missing: {sorted(missing)}",
        )
    irregular = (pl.col("__gap_ms") <= 0) | (pl.col("__gap_ms") % freq_ms != 0)
    if not allow_multiples:
        irregular = irregular | (pl.col("__gap_ms") > freq_ms)
    gaps = (
        frame.sort(spec.key_columns)
        .with_columns(
            pl.col(spec.event_time_col)
            .diff()
            .over(spec.entity_col)
            .dt.total_milliseconds()
            .alias("__gap_ms")
        )
        .filter(pl.col("__gap_ms").is_not_null() & irregular)
        .select(spec.entity_col, spec.event_time_col, "__gap_ms")
    )
    if not gaps.is_empty():
        return GateCheck(
            name="grid_continuity",
            stage=Stage.CLEAN,
            status="FAILED",
            message="event-time grid has gaps or irregular steps",
            details={
                "expected_step_ms": freq_ms,
                "violations": gaps.head(20).to_dicts(),
            },
        )
    return GateCheck(
        name="grid_continuity",
        stage=Stage.CLEAN,
        status="PASSED",
        message="event-time grid matches the expected step",
    )


def _check_dead_values(
    frame: pl.DataFrame,
    *,
    spec: TemporalSpec,
    columns: Sequence[str],
    threshold: float,
) -> GateCheck:
    active = frame
    mask_col: str | None = None
    if spec.boolean_mask_cols:
        candidate = spec.boolean_mask_cols[0]
        if candidate in frame.columns:
            mask_col = candidate
            active = frame.filter(pl.col(candidate).fill_null(False))
    if active.is_empty():
        return GateCheck(
            name="dead_value_ratio",
            stage=Stage.CLEAN,
            status="SKIPPED",
            message="no active samples for dead-value analysis",
            details={"threshold": threshold, "ratios": {}, "mask_col": mask_col},
        )
    ratios: dict[str, float] = {}
    analyzed = False
    for column in columns:
        if column not in active.columns:
            continue
        non_null = int(active.get_column(column).is_not_null().sum())
        if non_null == 0:
            ratios[column] = 0.0
            continue
        repeated = (
            active.sort(spec.key_columns)
            .select(
                (
                    (pl.col(column) == pl.col(column).shift(1).over(spec.entity_col))
                    & pl.col(column).is_not_null()
                ).alias("__repeat")
            )
            .select(pl.col("__repeat").sum().alias("repeats"))
            .item()
        )
        repeats = int(repeated) if isinstance(repeated, int) else 0
        ratios[column] = repeats / non_null
        analyzed = True
    if not analyzed:
        return GateCheck(
            name="dead_value_ratio",
            stage=Stage.CLEAN,
            status="SKIPPED",
            message="no non-null active samples for dead-value analysis",
            details={"threshold": threshold, "ratios": ratios, "mask_col": mask_col},
        )
    offenders = {name: ratio for name, ratio in ratios.items() if ratio > threshold}
    if offenders:
        return GateCheck(
            name="dead_value_ratio",
            stage=Stage.CLEAN,
            status="FAILED",
            message="constant runs exceed the dead-value threshold",
            details={
                "threshold": threshold,
                "ratios": ratios,
                "offenders": offenders,
                "mask_col": mask_col,
            },
        )
    return GateCheck(
        name="dead_value_ratio",
        stage=Stage.CLEAN,
        status="PASSED",
        message="constant runs are within the dead-value threshold",
        details={"threshold": threshold, "ratios": ratios, "mask_col": mask_col},
    )


def _check_zscore_drift(
    frame: pl.DataFrame,
    baseline: pl.DataFrame,
    *,
    columns: Sequence[str],
    threshold: float,
    robust: bool = True,
) -> tuple[list[GateCheck], list[str]]:
    checks: list[GateCheck] = []
    warnings: list[str] = []
    for column in columns:
        if column not in frame.columns or column not in baseline.columns:
            continue
        current = frame.get_column(column).cast(pl.Float64)
        previous = baseline.get_column(column).cast(pl.Float64)
        mean_current = _finite_mean(current)
        mean_baseline = _finite_mean(previous)
        std_baseline = _finite_std(previous)
        if mean_current is None or mean_baseline is None:
            checks.append(
                GateCheck(
                    name=f"zscore_drift:{column}",
                    stage=Stage.CLEAN,
                    status="SKIPPED",
                    message="no finite samples for drift comparison",
                    details={"column": column},
                )
            )
            continue
        details: dict[str, object] = {"column": column, "threshold": threshold}
        zscore: float
        median_current = _finite_median(current) if robust else None
        median_baseline = _finite_median(previous) if robust else None
        mad_baseline = _finite_mad(previous) if robust else None
        if (
            median_current is not None
            and median_baseline is not None
            and mad_baseline is not None
            and mad_baseline > 0.0
        ):
            zscore = abs(median_current - median_baseline) / mad_baseline
            details.update(
                {
                    "statistic": "mad",
                    "median_baseline": median_baseline,
                    "median_current": median_current,
                    "mad_baseline": mad_baseline,
                    "zscore": zscore,
                }
            )
        else:
            if std_baseline is None or std_baseline == 0.0:
                zscore = 0.0 if mean_current == mean_baseline else float("inf")
            else:
                zscore = abs(mean_current - mean_baseline) / std_baseline
            details.update(
                {
                    "statistic": "mean_std",
                    "mean_baseline": mean_baseline,
                    "mean_current": mean_current,
                    "std_baseline": std_baseline,
                    "zscore": zscore,
                }
            )
        if zscore > threshold:
            warnings.append(
                f"{column} {details['statistic']} drift z-score {zscore:.3f} exceeds {threshold:g}"
            )
            checks.append(
                GateCheck(
                    name=f"zscore_drift:{column}",
                    stage=Stage.CLEAN,
                    status="WARNING",
                    message="distribution drifted beyond the configured threshold",
                    details=details,
                )
            )
        else:
            checks.append(
                GateCheck(
                    name=f"zscore_drift:{column}",
                    stage=Stage.CLEAN,
                    status="PASSED",
                    message="distribution is within the configured drift threshold",
                    details=details,
                )
            )
    return checks, warnings


def _check_feature_values(
    frame: pl.DataFrame, *, value_col: str, max_null_ratio: float
) -> GateCheck:
    if frame.is_empty():
        return GateCheck(
            name="finite_feature_values",
            stage=Stage.FEATURE,
            status="SKIPPED",
            message="feature frame is empty; finiteness is not applicable",
        )
    if value_col not in frame.columns:
        return GateCheck(
            name="finite_feature_values",
            stage=Stage.FEATURE,
            status="FAILED",
            message=f"feature value column {value_col!r} is missing",
        )
    value_expr = pl.col(value_col).cast(pl.Float64)
    non_null_count = int(frame.select(value_expr.is_not_null().sum()).item() or 0)
    invalid = frame.filter(value_expr.is_not_null() & ~value_expr.is_finite())
    null_ratio = 1.0 - (non_null_count / frame.height)
    if not invalid.is_empty():
        return GateCheck(
            name="finite_feature_values",
            stage=Stage.FEATURE,
            status="FAILED",
            message="feature values must be finite or null",
            details={"invalid_rows": invalid.head(20).to_dicts()},
        )
    if null_ratio > max_null_ratio:
        return GateCheck(
            name="finite_feature_values",
            stage=Stage.FEATURE,
            status="FAILED",
            message="feature null ratio exceeds the configured ceiling",
            details={"null_ratio": null_ratio, "max_null_ratio": max_null_ratio},
        )
    return GateCheck(
        name="finite_feature_values",
        stage=Stage.FEATURE,
        status="PASSED",
        message="feature values are finite or null within the null-ratio ceiling",
        details={"null_ratio": null_ratio},
    )


def _check_unique_decision_key(
    frame: pl.DataFrame, *, decision_col: str, entity_col: str
) -> GateCheck:
    if frame.is_empty():
        return GateCheck(
            name="unique_decision_key",
            stage=Stage.FEATURE,
            status="SKIPPED",
            message="frame is empty; decision-key uniqueness is not applicable",
        )
    missing = {decision_col, entity_col} - set(frame.columns)
    if missing:
        return GateCheck(
            name="unique_decision_key",
            stage=Stage.FEATURE,
            status="FAILED",
            message=f"decision key columns are missing: {sorted(missing)}",
        )
    duplicates = frame.group_by([decision_col, entity_col]).len().filter(pl.col("len") > 1)
    if not duplicates.is_empty():
        return GateCheck(
            name="unique_decision_key",
            stage=Stage.FEATURE,
            status="FAILED",
            message="decision-time keys must be unique",
            details={"duplicates": duplicates.head(20).to_dicts()},
        )
    return GateCheck(
        name="unique_decision_key",
        stage=Stage.FEATURE,
        status="PASSED",
        message="decision-time keys are unique",
    )


def _check_causal(causal_check: Callable[[], object]) -> GateCheck:
    try:
        result = causal_check()
    except CausalFailure as error:
        return GateCheck(
            name="causal_gate",
            stage=Stage.FEATURE,
            status="FAILED",
            message="causal gate detected future-dependent output",
            details=error.details,
        )
    status = getattr(result, "status", None)
    if status == "PASSED":
        gate_status: GateStatus = "PASSED"
        message = f"causal gate status: {status}"
    elif status == "SKIPPED":
        gate_status = "SKIPPED"
        message = f"causal gate status: {status}"
    else:
        gate_status = "FAILED"
        message = f"causal gate returned invalid or failed status: {status!r}"
    return GateCheck(
        name="causal_gate",
        stage=Stage.FEATURE,
        status=gate_status,
        message=message,
    )


def _is_aware_datetime(dtype: pl.DataType) -> bool:
    return isinstance(dtype, pl.Datetime) and dtype.time_zone is not None


def _finite_mean(series: pl.Series) -> float | None:
    value: object = cast(object, series.filter(series.is_finite()).mean())
    return float(value) if isinstance(value, int | float) and math.isfinite(value) else None


def _finite_std(series: pl.Series) -> float | None:
    value: object = cast(object, series.filter(series.is_finite()).std())
    return float(value) if isinstance(value, int | float) and math.isfinite(value) else None


def _finite_median(series: pl.Series) -> float | None:
    finite = series.filter(series.is_finite())
    if finite.is_empty():
        return None
    value: object = cast(object, finite.median())
    return float(value) if isinstance(value, int | float) and math.isfinite(value) else None


def _finite_mad(series: pl.Series) -> float | None:
    finite = series.filter(series.is_finite())
    if finite.is_empty():
        return None
    median = _finite_median(finite)
    if median is None:
        return None
    value: object = cast(object, (finite - median).abs().median())
    if not isinstance(value, int | float) or not math.isfinite(value):
        return None
    return float(value) * 1.4826


__all__ = [
    "GateCheck",
    "GateReport",
    "GateStatus",
    "InvarGate",
    "InvarRule",
    "RuleSeverity",
    "Stage",
]
