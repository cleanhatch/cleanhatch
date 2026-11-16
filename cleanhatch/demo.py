"""Five self-contained showcase scenarios for ``cleanhatch demo``.

Every scenario builds a realistic temporal panel in memory, runs the guard
that owns the failure mode, and returns the measured verdict.  No network,
file system, or third-party data source is required.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal

import polars as pl

from .__about__ import __version__
from .causal import CausalCheck, CausalGate
from .errors import CausalFailure
from .invar import GateCheck, GateReport, InvarGate, InvarRule, Stage
from .lineage import InputSnapshot, LineageManifest, TransformSignature
from .spec import TemporalSpec
from .temporal import TemporalGate

DemoVerdict = Literal["CAUGHT", "PASSED"]
ScenarioRunner = Callable[[], "_ScenarioOutcome"]


@dataclass(frozen=True, slots=True)
class DemoScenarioResult:
    """One measured showcase scenario."""

    name: str
    domain: str
    silent_killer: str
    gate: str
    verdict: DemoVerdict
    detail: str
    elapsed_ms: float
    ai_summary_bytes: int
    evidence: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "domain": self.domain,
            "silent_killer": self.silent_killer,
            "gate": self.gate,
            "verdict": self.verdict,
            "detail": self.detail,
            "elapsed_ms": self.elapsed_ms,
            "ai_summary_bytes": self.ai_summary_bytes,
            "evidence": dict(self.evidence),
        }


@dataclass(frozen=True, slots=True)
class DemoSuiteResult:
    """Measured output of all five showcase scenarios."""

    scenarios: tuple[DemoScenarioResult, ...]
    total_elapsed_ms: float

    @property
    def all_caught(self) -> bool:
        return all(scenario.verdict == "CAUGHT" for scenario in self.scenarios)

    @property
    def max_ai_summary_bytes(self) -> int:
        return max((scenario.ai_summary_bytes for scenario in self.scenarios), default=0)

    def to_dict(self) -> dict[str, object]:
        return {
            "suite": "cleanhatch demo",
            "version": __version__,
            "all_caught": self.all_caught,
            "total_elapsed_ms": self.total_elapsed_ms,
            "max_ai_summary_bytes": self.max_ai_summary_bytes,
            "scenarios": [scenario.to_dict() for scenario in self.scenarios],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def to_markdown(self) -> str:
        lines = [
            "## cleanhatch demo: 5 silent killers, caught in milliseconds",
            "",
            "| # | Scenario | Silent killer | Gate that caught it | Verdict | Time | AI summary |",
            "|---|----------|---------------|---------------------|---------|------|------------|",
        ]
        for index, scenario in enumerate(self.scenarios, start=1):
            lines.append(
                f"| {index} | `{scenario.name}` | {scenario.silent_killer} | "
                f"{scenario.gate} | **{scenario.verdict}** | "
                f"{scenario.elapsed_ms:.1f} ms | {scenario.ai_summary_bytes} B |"
            )
        lines.extend(
            [
                "",
                f"**{len(self.scenarios)}/{len(self.scenarios)} silent killers caught in "
                f"{self.total_elapsed_ms:.1f} ms.** Largest AI payload: "
                f"{self.max_ai_summary_bytes} B / 2048 B.",
                "",
                "### Evidence",
                "",
            ]
        )
        lines.extend(
            f"- **{scenario.name}** ({scenario.domain}): {scenario.detail}"
            for scenario in self.scenarios
        )
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class _ScenarioOutcome:
    name: str
    domain: str
    silent_killer: str
    gate: str
    verdict: DemoVerdict
    detail: str
    ai_summary_bytes: int
    evidence: dict[str, object]


def run_demo_suite() -> DemoSuiteResult:
    """Run all five showcase scenarios and measure each one."""

    suite_start = time.perf_counter()
    results: list[DemoScenarioResult] = []
    for runner in _SCENARIOS:
        scenario_start = time.perf_counter()
        outcome = runner()
        elapsed_ms = (time.perf_counter() - scenario_start) * 1000.0
        results.append(
            DemoScenarioResult(
                name=outcome.name,
                domain=outcome.domain,
                silent_killer=outcome.silent_killer,
                gate=outcome.gate,
                verdict=outcome.verdict,
                detail=outcome.detail,
                elapsed_ms=round(elapsed_ms, 2),
                ai_summary_bytes=outcome.ai_summary_bytes,
                evidence=outcome.evidence,
            )
        )
    total_ms = (time.perf_counter() - suite_start) * 1000.0
    return DemoSuiteResult(tuple(results), round(total_ms, 2))


def _ai_summary_bytes(
    scenario: str,
    frame: pl.DataFrame,
    gates: Sequence[GateReport],
    *,
    as_of: datetime,
    metrics: dict[str, float],
) -> int:
    manifest = LineageManifest.capture(
        inputs=[
            InputSnapshot.from_frame(
                frame,
                dataset_id=f"demo:{scenario}",
                version_tag="v1",
                as_of=as_of,
            )
        ],
        transform=TransformSignature.from_source(f"demo.{scenario}"),
        gates=list(gates),
        metrics=metrics,
    )
    return len(manifest.to_ai_summary_json().encode("utf-8"))


def _price_panel(*, entities: int, bars: int) -> pl.DataFrame:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    times = [start + timedelta(hours=index) for index in range(bars)]
    frames: list[pl.DataFrame] = []
    for entity in range(entities):
        prices = [
            100.0 + entity * 2.0 + (index % 7) * 0.4 + (index // 7) * 0.05 for index in range(bars)
        ]
        frames.append(
            pl.DataFrame(
                {
                    "entity_id": [f"EQ{entity:02d}"] * bars,
                    "event_time": times,
                    "available_at": [value + timedelta(hours=1) for value in times],
                    "price": prices,
                }
            )
        )
    return pl.concat(frames).sort(["event_time", "entity_id"])


def _full_sample_zscore(frame: pl.LazyFrame) -> pl.LazyFrame:
    price = pl.col("price")
    zscore = (price - price.mean()) / price.std()
    return (
        frame.with_columns(zscore.alias("__zscore"))
        .with_columns(pl.col("__zscore").rolling_mean(3).over("entity_id").alias("__feature"))
        .select(
            pl.col("available_at").alias("decision_time"),
            "entity_id",
            pl.col("__feature").alias("value"),
        )
    )


def _causal_rolling_zscore(frame: pl.LazyFrame) -> pl.LazyFrame:
    price = pl.col("price")
    rolling_mean = price.rolling_mean(3).over("entity_id")
    rolling_std = price.rolling_std(3).over("entity_id")
    zscore = (price - rolling_mean) / rolling_std
    return (
        frame.with_columns(zscore.alias("__zscore"))
        .with_columns(pl.col("__zscore").rolling_mean(3).over("entity_id").alias("__feature"))
        .select(
            pl.col("available_at").alias("decision_time"),
            "entity_id",
            pl.col("__feature").alias("value"),
        )
    )


def _scenario_quant_full_sample_zscore() -> _ScenarioOutcome:
    spec = TemporalSpec(signed_numeric_cols=("price",))
    panel = _price_panel(entities=6, bars=140)
    leaky_gate = CausalGate(spec, value_col="value")
    try:
        leaky_gate.run(_full_sample_zscore, panel, sample_points=4)
    except CausalFailure as error:
        failed = [check for check in error.failed_checks if isinstance(check, CausalCheck)]
        perturbed = [check for check in failed if check.method == "future_perturbation"]
        if not perturbed:
            raise AssertionError("leak must be caught by future perturbation") from error
        max_abs = max((check.max_abs_diff or 0.0 for check in perturbed), default=0.0)
        rank_flips = sum(check.rank_flip_count for check in perturbed)
        methods = sorted({check.method for check in failed})
    else:
        raise AssertionError("full-sample z-score must fail the causal gate")

    safe_gate = CausalGate(spec, value_col="value")
    safe_result = safe_gate.run(_causal_rolling_zscore, panel, sample_points=4)
    if safe_result.status != "PASSED":
        raise AssertionError("rolling z-score must pass the causal gate")

    report = GateReport(
        Stage.FEATURE,
        (
            GateCheck(
                name="full_sample_zscore_leak",
                stage=Stage.FEATURE,
                status="FAILED",
                message="full-sample normalization depends on future observations",
                details={
                    "methods": methods,
                    "max_abs_diff": round(max_abs, 6),
                    "rank_flip_count": rank_flips,
                },
            ),
            GateCheck(
                name="rolling_zscore_causal",
                stage=Stage.FEATURE,
                status="PASSED",
                message="rolling normalization is causal",
                details={"sample_points": safe_result.sample_points},
            ),
        ),
    )
    size = _ai_summary_bytes(
        "quant_full_sample_zscore",
        panel,
        (report,),
        as_of=datetime(2026, 2, 1, tzinfo=UTC),
        metrics={"entities": float(panel["entity_id"].n_unique()), "rows": float(panel.height)},
    )
    return _ScenarioOutcome(
        name="quant_full_sample_zscore",
        domain="Quant / Feature Engineering",
        silent_killer="Full-sample z-score normalizes with observations that did not exist yet",
        gate="CausalGate (future_perturbation)",
        verdict="CAUGHT",
        detail=(
            f"future_perturbation max_abs_diff={max_abs:.4f}, rank_flips={rank_flips}; "
            f"causal rolling twin PASSED"
        ),
        ai_summary_bytes=size,
        evidence={
            "caught_by": methods,
            "max_abs_diff": round(max_abs, 6),
            "rank_flip_count": rank_flips,
            "safe_twin": safe_result.status,
            "entities": panel["entity_id"].n_unique(),
            "rows": panel.height,
        },
    )


def _scenario_sec_restatement_time_travel() -> _ScenarioOutcome:
    event_time = datetime(2026, 3, 31, tzinfo=UTC)
    initial = pl.DataFrame(
        {
            "entity_id": ["ACME"],
            "event_time": [event_time],
            "available_at": [datetime(2026, 5, 1, tzinfo=UTC)],
            "revenue": [100.0],
        }
    )
    restated = pl.concat(
        [
            initial,
            pl.DataFrame(
                {
                    "entity_id": ["ACME"],
                    "event_time": [event_time],
                    "available_at": [datetime(2026, 8, 10, tzinfo=UTC)],
                    "revenue": [62.0],
                }
            ),
        ]
    )
    spec = TemporalSpec(positive_numeric_cols=("revenue",))
    gate = TemporalGate(restated, spec=spec)
    before = float(gate.read_as_of(as_of=datetime(2026, 5, 15, tzinfo=UTC))["revenue"].item())
    after = float(gate.read_as_of(as_of=datetime(2026, 8, 15, tzinfo=UTC))["revenue"].item())
    diff = gate.diff_versions(initial, restated)
    changed = diff.columns[0]
    report = GateReport(
        Stage.CLEAN,
        (
            GateCheck(
                name="restatement_detected",
                stage=Stage.CLEAN,
                status="WARNING",
                message="Q1 revenue was restated after the original filing",
                details={
                    "original_usd_m": before,
                    "restated_usd_m": after,
                    "delta_usd_m": round(after - before, 4),
                    "changed_rows": changed.changed_rows,
                },
            ),
        ),
    )
    size = _ai_summary_bytes(
        "sec_restatement_time_travel",
        restated,
        (report,),
        as_of=datetime(2026, 9, 1, tzinfo=UTC),
        metrics={"changed_rows": float(changed.changed_rows)},
    )
    return _ScenarioOutcome(
        name="sec_restatement_time_travel",
        domain="Fundamental / SEC Filings PIT",
        silent_killer="A later restatement rewrites the revenue visible in a historical backtest",
        gate="TemporalGate.read_as_of + diff_versions",
        verdict="CAUGHT",
        detail=(
            f"as_of 2026-05-15 -> USD {before:.1f}M; as_of 2026-08-15 -> USD {after:.1f}M; "
            f"diff changed_rows={changed.changed_rows}, max_abs_change="
            f"{changed.max_abs_change:.1f}M"
        ),
        ai_summary_bytes=size,
        evidence={
            "as_of_2026_05_15_usd_m": before,
            "as_of_2026_08_15_usd_m": after,
            "diff_changed_rows": changed.changed_rows,
            "diff_max_abs_change_usd_m": changed.max_abs_change,
        },
    )


def _crypto_panel() -> tuple[pl.DataFrame, pl.DataFrame]:
    start = datetime(2026, 2, 1, tzinfo=UTC)
    size = 121
    halt = set(range(40, 60))
    normal_prices = [100.0 + (index % 11) * 0.5 + (index // 11) * 0.03 for index in range(size)]
    for index in halt:
        normal_prices[index] = normal_prices[39]
    spiked_prices = list(normal_prices)
    spiked_prices[80] = 50_000.0
    times = [start + timedelta(hours=index) for index in range(size)]
    mask = [index not in halt for index in range(size)]

    def build(prices: list[float]) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "entity_id": ["BTC-USD"] * size,
                "event_time": times,
                "available_at": [value + timedelta(minutes=1) for value in times],
                "price": prices,
                "is_trading": mask,
            }
        )

    return build(spiked_prices), build(normal_prices)


def _scenario_crypto_halt_and_flash_spike() -> _ScenarioOutcome:
    spiked, normal = _crypto_panel()
    masked_spec = TemporalSpec(
        positive_numeric_cols=("price",),
        boolean_mask_cols=("is_trading",),
    )
    masked_gate = InvarGate(masked_spec, max_zscore_drift=4.0)
    unmasked_gate = InvarGate(
        TemporalSpec(positive_numeric_cols=("price",)),
        max_zscore_drift=4.0,
    )
    masked = masked_gate.check_clean(
        spiked,
        baseline=normal,
        numeric_cols=("price",),
        grid_freq_ms=3_600_000,
        raise_on_failure=False,
    )
    unmasked = unmasked_gate.check_clean(
        spiked,
        numeric_cols=("price",),
        grid_freq_ms=3_600_000,
        raise_on_failure=False,
    )
    regime = masked_gate.check_clean(
        normal.with_columns((pl.col("price") * 3.0).alias("price")),
        baseline=normal,
        numeric_cols=("price",),
        grid_freq_ms=3_600_000,
        raise_on_failure=False,
    )
    dead = next(check for check in masked.checks if check.name == "dead_value_ratio")
    drift = next(check for check in masked.checks if check.name == "zscore_drift:price")
    regime_drift = next(check for check in regime.checks if check.name == "zscore_drift:price")
    if masked.status != "PASSED":
        raise AssertionError("masked halt panel must not raise a false positive")
    if unmasked.status != "FAILED":
        raise AssertionError("unmasked halt panel must expose the dead-value false positive")
    if regime.status != "WARNING":
        raise AssertionError("a threefold regime shift must trigger drift warnings")
    if drift.details.get("statistic") != "mad":
        raise AssertionError("robust drift must use the MAD statistic")

    size = _ai_summary_bytes(
        "crypto_halt_and_flash_spike",
        spiked,
        (masked, regime),
        as_of=datetime(2026, 3, 1, tzinfo=UTC),
        metrics={"halt_bars": 20.0, "rows": float(spiked.height)},
    )
    return _ScenarioOutcome(
        name="crypto_halt_and_flash_spike",
        domain="Market Bars / Distribution Drift",
        silent_killer="Halt bars look dead and one flash spike masks a real regime shift",
        gate="InvarGate (masked dead values + robust MAD)",
        verdict="CAUGHT",
        detail=(
            f"20 halt bars exempted (unmasked gate {unmasked.status}); "
            f"single-bar spike ignored by MAD (z={drift.details.get('zscore')}); "
            f"3x regime shift flagged ({regime.status})"
        ),
        ai_summary_bytes=size,
        evidence={
            "masked_status": masked.status,
            "unmasked_status": unmasked.status,
            "dead_value_ratios": dead.details.get("ratios"),
            "drift_statistic": drift.details.get("statistic"),
            "spike_drift_zscore": drift.details.get("zscore"),
            "regime_status": regime.status,
            "regime_drift_zscore": regime_drift.details.get("zscore"),
        },
    )


def _survivorship_panel() -> tuple[pl.DataFrame, int]:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    full_bars = 120
    frames: list[pl.DataFrame] = []
    for entity in range(8):
        bars = full_bars if entity < 5 else 60
        times = [start + timedelta(hours=index) for index in range(bars)]
        prices = [100.0 + entity + 0.1 * index + (index % 5) * 0.2 for index in range(bars)]
        frames.append(
            pl.DataFrame(
                {
                    "entity_id": [f"FUND{entity:02d}"] * bars,
                    "event_time": times,
                    "available_at": [value + timedelta(hours=1) for value in times],
                    "price": prices,
                }
            )
        )
    return pl.concat(frames).sort(["event_time", "entity_id"]), full_bars


def _scenario_survivorship_bias_filter() -> _ScenarioOutcome:
    panel, full_bars = _survivorship_panel()
    spec = TemporalSpec(signed_numeric_cols=("price",))

    def survivors_only(frame: pl.LazyFrame) -> pl.LazyFrame:
        return (
            frame.with_columns(pl.col("event_time").count().over("entity_id").alias("__bars"))
            .filter(pl.col("__bars") == full_bars)
            .select(
                pl.col("available_at").alias("decision_time"),
                pl.col("entity_id"),
                pl.col("price").alias("value"),
            )
        )

    gate = CausalGate(spec, value_col="value")
    try:
        gate.run(survivors_only, panel, sample_points=3)
    except CausalFailure as error:
        failed = [check for check in error.failed_checks if isinstance(check, CausalCheck)]
        truncated = [check for check in failed if check.method == "availability_truncation"]
        if not truncated:
            raise AssertionError("universe lookahead must fail availability truncation") from error
        methods = sorted({check.method for check in failed})
        mismatched = max(len(check.mismatched_entities) for check in truncated)
    else:
        raise AssertionError("survivorship filter must fail the causal gate")

    report = GateReport(
        Stage.FEATURE,
        (
            GateCheck(
                name="causal_gate:survivorship_filter",
                stage=Stage.FEATURE,
                status="FAILED",
                message="entity set changed when future bars were truncated",
                details={
                    "methods": methods,
                    "mismatched_entities": mismatched,
                },
            ),
        ),
    )
    size = _ai_summary_bytes(
        "survivorship_bias_filter",
        panel,
        (report,),
        as_of=datetime(2026, 2, 1, tzinfo=UTC),
        metrics={"survivors": 5.0, "delisted": 3.0},
    )
    return _ScenarioOutcome(
        name="survivorship_bias_filter",
        domain="Universe Selection / Survivorship Leak",
        silent_killer="Universe keeps only entities that survive to the end of the sample",
        gate="CausalGate (availability_truncation)",
        verdict="CAUGHT",
        detail=(
            f"full-sample filter keeps 5 survivors; truncated history changes the entity "
            f"set ({mismatched} mismatched entities)"
        ),
        ai_summary_bytes=size,
        evidence={
            "full_bars": full_bars,
            "survivors": 5,
            "delisted": 3,
            "caught_by": methods,
            "mismatched_entities": mismatched,
        },
    )


def _scenario_ai_rag_temporal_contamination() -> _ScenarioOutcome:
    decision_time = datetime(2025, 6, 1, tzinfo=UTC)
    chunks = pl.DataFrame(
        {
            "chunk_id": ["doc-contaminated", "doc-clean"],
            "event_time": [
                datetime(2025, 5, 15, tzinfo=UTC),
                datetime(2025, 5, 20, tzinfo=UTC),
            ],
            "available_at": [
                datetime(2026, 1, 20, tzinfo=UTC),
                datetime(2025, 5, 21, tzinfo=UTC),
            ],
            "decision_time": [decision_time, decision_time],
            "relevance": [0.94, 0.82],
        }
    )
    spec = TemporalSpec(
        entity_col="chunk_id",
        positive_numeric_cols=("relevance",),
    )
    visible = TemporalGate(chunks, spec=spec).read_as_of(as_of=decision_time)
    if visible.height != 1 or visible["chunk_id"].to_list() != ["doc-clean"]:
        raise AssertionError("the contaminated chunk must be invisible at decision time")

    rule = InvarRule(
        name="chunk_available_before_decision",
        predicate=pl.col("available_at") <= pl.col("decision_time"),
        message="retrieved chunk was not available at decision time",
    )
    gate_report = InvarGate(spec).check_ingest(
        chunks,
        rules=(rule,),
        raise_on_failure=False,
    )
    if gate_report.status != "FAILED":
        raise AssertionError("the contaminated chunk must fail the ingest gate")

    size = _ai_summary_bytes(
        "ai_rag_temporal_contamination",
        chunks,
        (gate_report,),
        as_of=decision_time,
        metrics={"candidates": 2.0, "blocked": 1.0},
    )
    return _ScenarioOutcome(
        name="ai_rag_temporal_contamination",
        domain="AI Agent / RAG Eval Guard",
        silent_killer="A post-hoc edited document leaks 2026 spoilers into a 2025 decision",
        gate="TemporalGate + InvarGate + ai_summary.json",
        verdict="CAUGHT",
        detail=(
            "1/2 chunks blocked (available_at 2026-01-20 > decision_time 2025-06-01); "
            f"clean chunk retained; AI summary {size} B"
        ),
        ai_summary_bytes=size,
        evidence={
            "decision_time": decision_time.isoformat(),
            "visible_chunks": visible["chunk_id"].to_list(),
            "blocked_chunks": 1,
            "ingest_status": gate_report.status,
            "failed_rule": rule.name,
        },
    )


_SCENARIOS: tuple[ScenarioRunner, ...] = (
    _scenario_quant_full_sample_zscore,
    _scenario_sec_restatement_time_travel,
    _scenario_crypto_halt_and_flash_spike,
    _scenario_survivorship_bias_filter,
    _scenario_ai_rag_temporal_contamination,
)


__all__ = [
    "DemoScenarioResult",
    "DemoSuiteResult",
    "DemoVerdict",
    "run_demo_suite",
]
