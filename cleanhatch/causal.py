"""Domain-neutral causal gate for lookahead leakage.

The gate compares a function's output at sampled decision times against two
counterfactuals:

* future rows are physically removed by knowledge time; and
* future numeric values are poisoned and boolean masks are flipped.

It uses the column roles in :class:`TemporalSpec`, so the same engine works for
market bars, demand forecasts, clinical event tables, or any other temporal
panel.  Application adapters supply the output schema and the perturbation
policy.
"""

from __future__ import annotations

import bisect
import math
import random
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Literal, cast

import polars as pl

from .errors import (
    CausalContractError,
    CausalExecutionError,
    CausalFailure,
)
from .spec import TemporalSpec

CausalTransform = Callable[[pl.LazyFrame], object]
CausalMethod = Literal["availability_truncation", "future_perturbation"]
CheckpointKind = Literal[
    "warmup_boundary",
    "halt_resumption_boundary",
    "median_point",
    "tail_point",
    "random_point",
]
_CHECKPOINT_PRIORITY: tuple[CheckpointKind, ...] = (
    "warmup_boundary",
    "halt_resumption_boundary",
    "median_point",
    "tail_point",
    "random_point",
)
_DEFAULT_WARMUP_BUFFER = 5
MAX_TOLERANCE = 1e-8
"""Hard cap for the mixed tolerance; a larger tolerance could hide real leaks."""


@dataclass(frozen=True, slots=True)
class CausalDrift:
    """A non-zero change that stayed inside tolerance and therefore passed."""

    decision_time: datetime
    method: CausalMethod
    entity_id: str
    before: float | None
    after: float | None
    abs_diff: float
    rel_diff: float | None
    rank_flipped: bool

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["decision_time"] = self.decision_time.isoformat()
        return payload


@dataclass(frozen=True, slots=True)
class CausalCheck:
    method: CausalMethod
    decision_time: datetime
    max_abs_diff: float | None
    mismatched_entities: tuple[str, ...]
    checkpoint_kind: CheckpointKind = "random_point"
    rank_flip_count: int = 0
    compared_entities: int = 0

    @property
    def passed(self) -> bool:
        # An empty cross-section proves nothing; it must not pass silently.
        return self.compared_entities > 0 and not self.mismatched_entities

    def to_dict(self) -> dict[str, object]:
        values = asdict(self)
        values["decision_time"] = self.decision_time.isoformat()
        return values


@dataclass(frozen=True, slots=True)
class CausalResult:
    status: Literal["PASSED", "FAILED"]
    sample_points: int
    checks: tuple[CausalCheck, ...]
    sub_tolerance_drifts: tuple[CausalDrift, ...] = ()
    slicing_mode: Literal["adaptive_window", "prefix"] = "prefix"
    checkpoint_kinds: tuple[CheckpointKind, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "sample_points": self.sample_points,
            "slicing_mode": self.slicing_mode,
            "checkpoint_kinds": list(self.checkpoint_kinds),
            "checks": [check.to_dict() for check in self.checks],
            "sub_tolerance_drifts": [drift.to_dict() for drift in self.sub_tolerance_drifts],
        }


class CausalGate:
    """Run dual-mode causality tests for a temporal transform.

    The gate owns the input/output column roles and perturbation policy; each
    :meth:`run` supplies only the sampling and tolerance budget.  This makes the
    common case a single object with a stable contract while still allowing
    per-run checkpoints.
    """

    def __init__(
        self,
        spec: TemporalSpec | None = None,
        *,
        value_col: str = "value",
        output_decision_time_col: str | None = None,
        output_entity_col: str | None = None,
        adaptive_window: bool = False,
        safe_lookback: int = 0,
        validate_output: Callable[[pl.DataFrame], pl.DataFrame] | None = None,
        perturbation_numeric_cols: Sequence[str] | None = None,
        heavy_perturbation_cols: Sequence[str] = (),
        mask_cols: Sequence[str] | None = None,
        boundary_mask_col: str | None = None,
    ) -> None:
        self._spec = spec if spec is not None else TemporalSpec()
        self._value_col = value_col
        self._output_decision_time_col = output_decision_time_col
        self._output_entity_col = output_entity_col
        self._adaptive_window = adaptive_window
        self._safe_lookback = safe_lookback
        self._validate_output = validate_output
        self._perturbation_numeric_cols = (
            tuple(perturbation_numeric_cols) if perturbation_numeric_cols is not None else None
        )
        self._heavy_perturbation_cols = tuple(heavy_perturbation_cols)
        self._mask_cols = tuple(mask_cols) if mask_cols is not None else None
        self._boundary_mask_col = boundary_mask_col

    @property
    def spec(self) -> TemporalSpec:
        return self._spec

    def run(
        self,
        transform: CausalTransform,
        frame: pl.DataFrame,
        *,
        sample_points: int = 3,
        seed: int = 42,
        tolerance: float = MAX_TOLERANCE,
        rtol: float = 1e-7,
        atol: float = 1e-12,
    ) -> CausalResult:
        """Evaluate one transform against truncated and poisoned futures."""

        return run_causal_analysis(
            transform,
            frame,
            spec=self._spec,
            value_col=self._value_col,
            output_decision_time_col=self._output_decision_time_col,
            output_entity_col=self._output_entity_col,
            sample_points=sample_points,
            seed=seed,
            tolerance=tolerance,
            rtol=rtol,
            atol=atol,
            adaptive_window=self._adaptive_window,
            safe_lookback=self._safe_lookback,
            validate_output=self._validate_output,
            perturbation_numeric_cols=self._perturbation_numeric_cols,
            heavy_perturbation_cols=self._heavy_perturbation_cols,
            mask_cols=self._mask_cols,
            boundary_mask_col=self._boundary_mask_col,
        )

    check = run

    @staticmethod
    def compare(
        *,
        method: CausalMethod,
        decision_time: datetime,
        expected: dict[str, float | None],
        actual: dict[str, float | None],
        tolerance: float,
        rtol: float,
        atol: float,
    ) -> tuple[CausalCheck, tuple[CausalDrift, ...]]:
        """Compare one cross-section with mixed tolerance and rank-flip rejection."""

        return compare_causal_values(
            method=method,
            decision_time=decision_time,
            expected=expected,
            actual=actual,
            tolerance=tolerance,
            rtol=rtol,
            atol=atol,
        )


def run_causal_analysis(
    transform: CausalTransform,
    frame: pl.DataFrame,
    *,
    spec: TemporalSpec,
    value_col: str,
    output_decision_time_col: str | None = None,
    output_entity_col: str | None = None,
    sample_points: int = 3,
    seed: int = 42,
    tolerance: float = MAX_TOLERANCE,
    rtol: float = 1e-7,
    atol: float = 1e-12,
    adaptive_window: bool = False,
    safe_lookback: int = 0,
    validate_output: Callable[[pl.DataFrame], pl.DataFrame] | None = None,
    perturbation_numeric_cols: Sequence[str] | None = None,
    heavy_perturbation_cols: Sequence[str] = (),
    mask_cols: Sequence[str] | None = None,
    boundary_mask_col: str | None = None,
) -> CausalResult:
    """Compare a transform's signal with future rows removed or poisoned.

    ``spec`` describes the input table.  The output table must contain the
    configured decision-time and entity columns plus ``value_col``.  The test
    is deterministic, empirical, and not a proof that arbitrary code is causal.
    """

    _validate_options(sample_points, seed, tolerance, rtol, atol)
    decision_col = output_decision_time_col or spec.decision_time_col
    entity_col = output_entity_col or spec.entity_col
    if frame.is_empty():
        raise CausalContractError("causal gate input frame is empty")
    ordered = frame.sort(spec.key_columns)
    baseline = _evaluate(
        transform,
        ordered,
        decision_col=decision_col,
        entity_col=entity_col,
        value_col=value_col,
        validate_output=validate_output,
    )
    candidate_times = _sampleable_decision_times(
        baseline,
        ordered,
        spec=spec,
        decision_col=decision_col,
        value_col=value_col,
    )
    if len(candidate_times) < sample_points:
        raise CausalContractError(
            f"Causal gate needs at least {sample_points} non-null decision times with future "
            f"observations; found {len(candidate_times)}"
        )
    selected = _stratified_checkpoints(
        baseline,
        ordered,
        candidate_times,
        spec=spec,
        decision_col=decision_col,
        value_col=value_col,
        boundary_mask_col=boundary_mask_col,
        sample_points=sample_points,
        seed=seed,
    )
    selected_times = [decision_time for decision_time, _ in selected]
    kinds: dict[datetime, CheckpointKind] = {
        decision_time: kind for decision_time, kind in selected
    }
    slicing_mode: Literal["adaptive_window", "prefix"] = (
        "adaptive_window" if adaptive_window and safe_lookback > 0 else "prefix"
    )
    safe_window = safe_lookback if slicing_mode == "adaptive_window" else 0
    unique_times = _unique_event_times(ordered, spec)
    numeric_cols = (
        tuple(perturbation_numeric_cols)
        if perturbation_numeric_cols is not None
        else spec.numeric_columns
    )
    masks = tuple(mask_cols) if mask_cols is not None else spec.boolean_mask_cols
    heavy = frozenset(heavy_perturbation_cols)

    checks: list[CausalCheck] = []
    failures: list[CausalCheck] = []
    drifts: list[CausalDrift] = []
    for decision_time in selected_times:
        kind: CheckpointKind = kinds.get(decision_time, "random_point")
        expected = _values_at(
            baseline,
            decision_time,
            decision_col=decision_col,
            entity_col=entity_col,
            value_col=value_col,
        )
        use_local = slicing_mode == "adaptive_window" and safe_window > 0
        if use_local:
            local_baseline = _evaluate(
                transform,
                _local_slice_around(
                    ordered,
                    decision_time,
                    spec=spec,
                    safe_bars=safe_window,
                    future_bars=None,
                    unique_times=unique_times,
                ),
                decision_col=decision_col,
                entity_col=entity_col,
                value_col=value_col,
                validate_output=validate_output,
            )
            use_local = _same_values(
                expected,
                _values_at(
                    local_baseline,
                    decision_time,
                    decision_col=decision_col,
                    entity_col=entity_col,
                    value_col=value_col,
                ),
            )
        for method in ("availability_truncation", "future_perturbation"):
            if method == "availability_truncation":
                candidate_bars = (
                    _local_slice_around(
                        ordered,
                        decision_time,
                        spec=spec,
                        safe_bars=safe_window,
                        future_bars=0,
                        unique_times=unique_times,
                    )
                    if use_local
                    else ordered
                )
                candidate_bars = _truncate(candidate_bars, spec=spec, decision_time=decision_time)
            else:
                candidate_bars = (
                    _local_slice_around(
                        ordered,
                        decision_time,
                        spec=spec,
                        safe_bars=safe_window,
                        future_bars=None,
                        unique_times=unique_times,
                    )
                    if use_local
                    else ordered
                )
                candidate_bars = _perturb_future(
                    candidate_bars,
                    spec=spec,
                    decision_time=decision_time,
                    numeric_cols=numeric_cols,
                    heavy_cols=heavy,
                    mask_cols=masks,
                )
            actual = _evaluate(
                transform,
                candidate_bars,
                decision_col=decision_col,
                entity_col=entity_col,
                value_col=value_col,
                validate_output=validate_output,
            )
            check, check_drifts = compare_causal_values(
                method=_cast_causal_method(method),
                decision_time=decision_time,
                expected=expected,
                actual=_values_at(
                    actual,
                    decision_time,
                    decision_col=decision_col,
                    entity_col=entity_col,
                    value_col=value_col,
                ),
                tolerance=tolerance,
                rtol=rtol,
                atol=atol,
            )
            check = CausalCheck(
                method=check.method,
                decision_time=check.decision_time,
                max_abs_diff=check.max_abs_diff,
                mismatched_entities=check.mismatched_entities,
                checkpoint_kind=kind,
                rank_flip_count=check.rank_flip_count,
                compared_entities=check.compared_entities,
            )
            checks.append(check)
            drifts.extend(check_drifts)
            if not check.passed:
                failures.append(check)

    checkpoint_kinds: tuple[CheckpointKind, ...] = tuple(
        kinds.get(value, "random_point") for value in selected_times
    )
    result = CausalResult(
        "FAILED" if failures else "PASSED",
        len(selected_times),
        tuple(checks),
        tuple(drifts),
        slicing_mode=slicing_mode,
        checkpoint_kinds=checkpoint_kinds,
    )
    if failures:
        raise CausalFailure(
            failed_checks=failures,
            all_checks=checks,
            sub_tolerance_drifts=drifts,
            slicing_mode=slicing_mode,
            checkpoint_kinds=checkpoint_kinds,
            partial_result=result.to_dict(),
        )
    return result


def compare_causal_values(
    *,
    method: CausalMethod,
    decision_time: datetime,
    expected: dict[str, float | None],
    actual: dict[str, float | None],
    tolerance: float,
    rtol: float,
    atol: float,
) -> tuple[CausalCheck, tuple[CausalDrift, ...]]:
    """Compare one cross-section with mixed tolerance and rank-flip rejection."""

    mismatched: list[str] = []
    drifts: list[CausalDrift] = []
    max_abs_diff = 0.0
    expected_ranks = _ranks(expected)
    actual_ranks = _ranks(actual)
    rank_flips = 0

    for entity in sorted(expected.keys() | actual.keys()):
        if entity not in expected or entity not in actual:
            mismatched.append(entity)
            continue
        before = expected[entity]
        after = actual[entity]
        if before is None or after is None:
            if before is not after:
                mismatched.append(entity)
            continue
        difference = abs(before - after)
        if difference > max_abs_diff:
            max_abs_diff = difference
        allowed = min(tolerance, atol + rtol * max(abs(before), abs(after)))
        flipped = expected_ranks.get(entity) != actual_ranks.get(entity)
        if flipped:
            rank_flips += 1
        if (difference > 0.0 and flipped) or difference > allowed:
            mismatched.append(entity)
        elif difference > 0.0:
            largest = max(abs(before), abs(after))
            drifts.append(
                CausalDrift(
                    decision_time=decision_time,
                    method=method,
                    entity_id=entity,
                    before=before,
                    after=after,
                    abs_diff=difference,
                    rel_diff=(difference / largest) if largest > 0 else None,
                    rank_flipped=flipped,
                )
            )
    check = CausalCheck(
        method=method,
        decision_time=decision_time,
        max_abs_diff=max_abs_diff,
        mismatched_entities=tuple(mismatched),
        rank_flip_count=rank_flips,
        compared_entities=len(expected.keys() | actual.keys()),
    )
    return check, tuple(drifts)


def _validate_options(
    sample_points: int, seed: int, tolerance: float, rtol: float, atol: float
) -> None:
    if type(sample_points) is not int or sample_points < 1:
        raise ValueError("sample_points must be a positive integer")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and non-negative")
    if tolerance > MAX_TOLERANCE:
        raise ValueError(f"tolerance must not exceed the hard cap {MAX_TOLERANCE:g}")
    if not math.isfinite(rtol) or rtol < 0:
        raise ValueError("rtol must be finite and non-negative")
    if not math.isfinite(atol) or atol < 0:
        raise ValueError("atol must be finite and non-negative")


def _unique_event_times(frame: pl.DataFrame, spec: TemporalSpec) -> list[datetime]:
    if spec.event_time_col not in frame.columns:
        return []
    return [
        value
        for value in frame.get_column(spec.event_time_col).unique().sort().to_list()
        if isinstance(value, datetime)
    ]


def _history_cutoff(
    unique_times: Sequence[datetime], decision_time: datetime, safe_bars: int
) -> datetime:
    index = bisect.bisect_right(unique_times, decision_time) - 1
    if index < 0:
        return unique_times[0]
    return unique_times[max(0, index - safe_bars)]


def _local_slice_around(
    frame: pl.DataFrame,
    decision_time: datetime,
    *,
    spec: TemporalSpec,
    safe_bars: int,
    future_bars: int | None,
    unique_times: Sequence[datetime] | None = None,
) -> pl.DataFrame:
    if frame.is_empty() or safe_bars <= 0:
        return frame
    times = list(unique_times) if unique_times is not None else _unique_event_times(frame, spec)
    if not times:
        return frame
    cutoff_start = _history_cutoff(times, decision_time, safe_bars)
    keep_history = (pl.col(spec.event_time_col) >= pl.lit(cutoff_start)) & (
        pl.col(spec.knowledge_time_col) <= pl.lit(decision_time)
    )
    if future_bars is None:
        keep_future = (pl.col(spec.knowledge_time_col) > pl.lit(decision_time)) & (
            pl.col(spec.event_time_col) >= pl.lit(cutoff_start)
        )
    elif future_bars <= 0:
        keep_future = pl.lit(False)
    else:
        future_times = [value for value in times if value > decision_time][:future_bars]
        keep_future = (
            (pl.col(spec.knowledge_time_col) > pl.lit(decision_time))
            & (pl.col(spec.event_time_col) >= pl.lit(cutoff_start))
            & (pl.col(spec.event_time_col) <= pl.lit(future_times[-1]))
            if future_times
            else pl.lit(False)
        )
    return frame.filter(keep_history | keep_future)


def _same_values(expected: dict[str, float | None], actual: dict[str, float | None]) -> bool:
    if expected.keys() != actual.keys():
        return False
    return all(expected[key] == actual[key] for key in expected)


def _truncate(frame: pl.DataFrame, *, spec: TemporalSpec, decision_time: datetime) -> pl.DataFrame:
    return frame.filter(pl.col(spec.knowledge_time_col) <= pl.lit(decision_time))


def _stratified_checkpoints(
    baseline: pl.DataFrame,
    frame: pl.DataFrame,
    candidate_times: Sequence[datetime],
    *,
    spec: TemporalSpec,
    decision_col: str,
    value_col: str,
    boundary_mask_col: str | None,
    sample_points: int,
    seed: int,
) -> list[tuple[datetime, CheckpointKind]]:
    ordered = sorted(candidate_times)
    proposals: dict[CheckpointKind, datetime] = {}
    warmup = _warmup_boundary(baseline, decision_col=decision_col, value_col=value_col)
    if warmup is not None and warmup in set(ordered):
        proposals["warmup_boundary"] = warmup
    halt = _halt_resumption_boundary(
        frame,
        spec=spec,
        boundary_mask_col=boundary_mask_col,
    )
    if halt is not None and halt in set(ordered):
        proposals["halt_resumption_boundary"] = halt
    if ordered:
        proposals["median_point"] = ordered[len(ordered) // 2]
        proposals["tail_point"] = ordered[-1]

    selected: list[tuple[datetime, CheckpointKind]] = []
    chosen: set[datetime] = set()
    for kind in _CHECKPOINT_PRIORITY:
        candidate = proposals.get(kind)
        if candidate is None or candidate in chosen:
            continue
        if len(selected) >= sample_points:
            break
        selected.append((candidate, kind))
        chosen.add(candidate)

    remaining = [value for value in ordered if value not in chosen]
    if remaining and len(selected) < sample_points:
        wanted = min(sample_points - len(selected), len(remaining))
        for value in random.Random(seed).sample(remaining, wanted):
            selected.append((value, "random_point"))
            chosen.add(value)

    selected.sort(key=lambda item: item[0])
    return selected


def _warmup_boundary(
    baseline: pl.DataFrame, *, decision_col: str, value_col: str
) -> datetime | None:
    valid_times = (
        baseline.filter(pl.col(value_col).is_not_null())
        .select(decision_col)
        .unique()
        .sort(decision_col)
        .get_column(decision_col)
        .to_list()
    )
    if not valid_times:
        return None
    candidate = (
        valid_times[_DEFAULT_WARMUP_BUFFER]
        if len(valid_times) > _DEFAULT_WARMUP_BUFFER
        else valid_times[0]
    )
    return candidate if isinstance(candidate, datetime) else None


def _halt_resumption_boundary(
    frame: pl.DataFrame,
    *,
    spec: TemporalSpec,
    boundary_mask_col: str | None,
) -> datetime | None:
    column = boundary_mask_col
    if column is None:
        column = spec.boolean_mask_cols[0] if spec.boolean_mask_cols else None
    if column is None or column not in frame.columns or frame.is_empty():
        return None
    mask = pl.col(column).fill_null(False)
    flips = (
        frame.sort(spec.key_columns)
        .with_columns((mask != mask.shift(1).over(spec.entity_col)).alias("_flip"))
        .filter(pl.col("_flip") & pl.col(spec.knowledge_time_col).is_not_null())
        .select(pl.col(spec.knowledge_time_col).min().alias("decision_time"))
        .item()
    )
    return flips if isinstance(flips, datetime) else None


def _evaluate(
    transform: CausalTransform,
    frame: pl.DataFrame,
    *,
    decision_col: str,
    entity_col: str,
    value_col: str,
    validate_output: Callable[[pl.DataFrame], pl.DataFrame] | None,
) -> pl.DataFrame:
    try:
        output: object = transform(frame.lazy())
    except Exception as error:  # noqa: BLE001 - transform crashes are runtime failures
        raise CausalExecutionError(
            f"transform raised {type(error).__name__} during causal evaluation: {error}",
            details={"error_type": type(error).__name__},
        ) from error
    if not isinstance(output, pl.LazyFrame):
        raise CausalContractError("transform must return polars.LazyFrame")
    try:
        collected = output.collect()
    except Exception as error:  # noqa: BLE001 - lazy execution failures are runtime errors
        raise CausalExecutionError(
            f"transform raised {type(error).__name__} while collecting causal output: {error}",
            details={"error_type": type(error).__name__},
        ) from error
    if validate_output is not None:
        # Schema failures intentionally propagate unchanged so host adapters can
        # keep their own exit-code mapping.
        collected = validate_output(collected)
    required = {decision_col, entity_col, value_col}
    missing = required - set(collected.columns)
    if missing:
        raise CausalContractError(f"causal output is missing required columns: {sorted(missing)}")
    return collected


def _sampleable_decision_times(
    factors: pl.DataFrame,
    frame: pl.DataFrame,
    *,
    spec: TemporalSpec,
    decision_col: str,
    value_col: str,
) -> list[datetime]:
    if factors.is_empty() or frame.is_empty():
        return []
    latest_available: object = cast(object, frame.get_column(spec.knowledge_time_col).max())
    if not isinstance(latest_available, datetime):
        return []
    candidates = (
        factors.filter(pl.col(value_col).is_not_null())
        .select(decision_col)
        .unique()
        .sort(decision_col)
        .get_column(decision_col)
        .to_list()
    )
    return [value for value in candidates if value < latest_available]


def _values_at(
    factors: pl.DataFrame,
    decision_time: datetime,
    *,
    decision_col: str,
    entity_col: str,
    value_col: str,
) -> dict[str, float | None]:
    selected = factors.filter(pl.col(decision_col) == pl.lit(decision_time)).select(
        entity_col, value_col
    )
    return {entity: value for entity, value in selected.iter_rows()}


def _ranks(values: dict[str, float | None]) -> dict[str, float]:
    ordered = sorted(
        (entity, value)
        for entity, value in values.items()
        if value is not None and math.isfinite(value)
    )
    ordered.sort(key=lambda item: (item[1], item[0]))
    return {entity: float(position + 1) for position, (entity, _) in enumerate(ordered)}


def _perturb_future(
    frame: pl.DataFrame,
    *,
    spec: TemporalSpec,
    decision_time: datetime,
    numeric_cols: Sequence[str],
    heavy_cols: frozenset[str],
    mask_cols: Sequence[str],
) -> pl.DataFrame:
    is_future = pl.col(spec.knowledge_time_col) > pl.lit(decision_time)
    expressions: list[pl.Expr] = []
    for column in sorted(set(numeric_cols).intersection(frame.columns)):
        dtype = frame.schema[column]
        multiplier = 100 if column in heavy_cols else 10
        if dtype.is_integer():
            perturbed = pl.col(column) * multiplier + 7
        elif dtype.is_float():
            perturbed = pl.col(column) * float(multiplier) + 0.137
        else:
            continue
        expressions.append(
            pl.when(is_future).then(perturbed).otherwise(pl.col(column)).cast(dtype).alias(column)
        )
    for column in sorted(set(mask_cols).intersection(frame.columns)):
        expressions.append(
            pl.when(is_future).then(~pl.col(column)).otherwise(pl.col(column)).alias(column)
        )
    return frame.with_columns(expressions)


def _cast_causal_method(value: str) -> CausalMethod:
    """Narrow a loop literal to the public method type."""

    if value == "availability_truncation":
        return "availability_truncation"
    if value == "future_perturbation":
        return "future_perturbation"
    raise ValueError(f"unknown causal method: {value!r}")


__all__ = [
    "CausalCheck",
    "CausalDrift",
    "CausalGate",
    "CausalMethod",
    "CausalResult",
    "CausalTransform",
    "CheckpointKind",
    "MAX_TOLERANCE",
    "compare_causal_values",
    "run_causal_analysis",
]
