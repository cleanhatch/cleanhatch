"""Pure analysis helpers shared by the L0 and L1 command lines.

The functions below contain no licensing, report export, or proprietary pack
logic.  The L0 CLI uses them directly; the private L1 CLI imports them so an
exported report is built from exactly the same typed gate results as the free
CLI.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import duckdb
import polars as pl

from .errors import CleanHatchError
from .hashing import source_ast_hash
from .invar import GateReport, InvarGate, Stage
from .lineage import InputSnapshot, LineageManifest, TransformSignature
from .packs import get_pack
from .spec import TemporalSpec
from .temporal import TemporalGate, VersionDiffReport, _frame_from_relation


@dataclass(frozen=True, slots=True)
class ValidateOutcome:
    """The frame, gate report, and lineage manifest for one validation run."""

    frame: pl.DataFrame
    report: GateReport
    manifest: LineageManifest


@dataclass(frozen=True, slots=True)
class DiffOutcome:
    """The version diff and lineage manifest for one comparison run."""

    report: VersionDiffReport
    manifest: LineageManifest


def load_frame(source: str) -> pl.DataFrame:
    """Load a Parquet, CSV, JSON, glob, or directory source."""

    path = Path(source)
    lower = source.lower()
    if path.is_dir() or any(character in source for character in "*?[]"):
        return _read_parquet(source)
    if lower.endswith(".parquet"):
        return _read_parquet(source)
    if lower.endswith((".csv", ".tsv")):
        separator = "\t" if lower.endswith(".tsv") else ","
        return pl.read_csv(source, separator=separator, try_parse_dates=True)
    if lower.endswith((".jsonl", ".ndjson")):
        return pl.read_ndjson(source)
    if lower.endswith(".json"):
        return pl.read_json(source)
    raise CleanHatchError(f"unsupported source type: {source}")


def build_clean_baseline(
    frame: pl.DataFrame,
    spec: TemporalSpec,
    *,
    as_of: datetime | None,
    no_drift_baseline: bool,
    baseline_windows: int,
    baseline_window_hours: float,
) -> pl.DataFrame | None:
    """Build the rolling PIT drift baseline used by ``--stage clean``."""

    if no_drift_baseline or frame.is_empty():
        return None
    required = {spec.entity_col, spec.event_time_col, spec.knowledge_time_col}
    if required - set(frame.columns):
        return None
    if baseline_windows < 1:
        raise CleanHatchError("--baseline-windows must be at least 1")
    if baseline_window_hours <= 0:
        raise CleanHatchError("--baseline-window-hours must be positive")
    cutoff = as_of
    if cutoff is None:
        latest = frame.get_column(spec.knowledge_time_col).max()
        if latest is None:
            return None
        if not isinstance(latest, datetime):
            raise CleanHatchError(
                f"{spec.knowledge_time_col} must contain datetime values to build a "
                "rolling baseline"
            )
        cutoff = latest + timedelta(microseconds=1)
    return TemporalGate(frame, spec=spec).rolling_baseline(
        as_of=cutoff,
        n_windows=baseline_windows,
        window=timedelta(hours=baseline_window_hours),
    )


def run_validate(
    source: str,
    *,
    spec: TemporalSpec,
    as_of: datetime | None = None,
    stage: Stage = Stage.CLEAN,
    pack: str | None = None,
    value_col: str = "value",
    no_drift_baseline: bool = False,
    baseline_windows: int = 20,
    baseline_window_hours: float = 24.0,
) -> ValidateOutcome:
    """Run one validation and return the result plus its lineage manifest."""

    frame = load_frame(source)
    if as_of is not None:
        frame = TemporalGate(frame, spec=spec).read_as_of(as_of=as_of)
    if pack is not None:
        report = get_pack(pack).validate(
            frame,
            stage=stage,
            raise_on_failure=False,
        )
    else:
        options: dict[str, object] = {}
        if stage is Stage.FEATURE:
            options["value_col"] = value_col
            options["decision_time_col"] = spec.decision_time_col
        elif stage is Stage.CLEAN:
            baseline = build_clean_baseline(
                frame,
                spec,
                as_of=as_of,
                no_drift_baseline=no_drift_baseline,
                baseline_windows=baseline_windows,
                baseline_window_hours=baseline_window_hours,
            )
            if baseline is not None:
                options["baseline"] = baseline
                numeric = spec.numeric_columns or tuple(
                    name for name, dtype in frame.schema.items() if dtype.is_numeric()
                )
                if numeric:
                    options["numeric_cols"] = numeric
        report = InvarGate(spec).run(
            frame,
            stage=stage,
            raise_on_failure=False,
            **options,  # pyright: ignore[reportArgumentType]
        )
    manifest = LineageManifest.capture(
        inputs=[
            InputSnapshot.from_frame(
                frame,
                dataset_id=Path(source).name or source,
                version_tag=pack or "validate",
                as_of=as_of or datetime.now(UTC),
            )
        ],
        transform=TransformSignature(
            code_hash=source_ast_hash("cleanhatch validate"),
            parameters=_validate_parameters(stage=stage, pack=pack),
        ),
        gates=[report],
        metrics={"rows": float(frame.height), "columns": float(frame.width)},
    )
    return ValidateOutcome(frame=frame, report=report, manifest=manifest)


def run_diff(
    source_a: str,
    source_b: str,
    *,
    spec: TemporalSpec,
    as_of: datetime | None = None,
) -> DiffOutcome:
    """Compare two snapshots and return the diff plus its lineage manifest."""

    frame_a = load_frame(source_a)
    frame_b = load_frame(source_b)
    report = TemporalGate(frame_a, spec=spec).diff_versions(
        frame_a,
        frame_b,
        as_of=as_of,
    )
    manifest_as_of = as_of or datetime.now(UTC)
    manifest = LineageManifest.capture(
        inputs=[
            InputSnapshot.from_frame(
                frame_a,
                dataset_id=Path(source_a).name or source_a,
                version_tag="a",
                as_of=manifest_as_of,
            ),
            InputSnapshot.from_frame(
                frame_b,
                dataset_id=Path(source_b).name or source_b,
                version_tag="b",
                as_of=manifest_as_of,
            ),
        ],
        transform=TransformSignature(
            code_hash=source_ast_hash("cleanhatch diff"),
            parameters={"as_of": as_of.isoformat() if as_of is not None else None},
        ),
        gates=[],
        metrics={"rows_a": float(report.rows_a), "rows_b": float(report.rows_b)},
    )
    return DiffOutcome(report=report, manifest=manifest)


def write_ai_summary(manifest: LineageManifest, output: str | Path) -> Path:
    """Write the bounded ``ai_summary.json`` payload."""

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.to_ai_summary_json(), encoding="utf-8")
    return path


def _validate_parameters(*, stage: Stage, pack: str | None) -> dict[str, object]:
    parameters: dict[str, object] = {"stage": stage.value}
    if pack is not None:
        parameters["pack"] = pack
    return parameters


def _read_parquet(source: str) -> pl.DataFrame:
    connection = duckdb.connect(
        ":memory:",
        config={
            "autoinstall_known_extensions": False,
            "autoload_known_extensions": False,
        },
    )
    try:
        connection.execute("SET TimeZone = 'UTC'")
        relation = connection.sql(
            "SELECT * FROM read_parquet(?, hive_partitioning = true, union_by_name = true)",
            params=[source],
        )
        if relation is None:
            raise CleanHatchError("DuckDB did not return a Parquet relation")
        return _frame_from_relation(relation)
    finally:
        connection.close()


__all__ = [
    "DiffOutcome",
    "ValidateOutcome",
    "build_clean_baseline",
    "load_frame",
    "run_diff",
    "run_validate",
    "write_ai_summary",
]
