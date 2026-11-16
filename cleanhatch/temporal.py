"""TemporalGate: domain-neutral as-of reads, SCD2 joins, and version diffs.

The store only knows the coordinates described by :class:`TemporalSpec`.  It
never validates a market-bar schema or assumes what a row means.  Host
applications wrap it with their own schema and source-resolution policies.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import duckdb
import polars as pl

from .errors import TemporalContractError, VersionMutationError
from .hashing import content_hash
from .spec import TemporalSpec

SourceInput = str | Path | Sequence[str | Path] | pl.DataFrame
ConnectionSetup = Callable[[duckdb.DuckDBPyConnection], None]


@dataclass(frozen=True, slots=True)
class ColumnDiff:
    """Distribution and cell-level changes for one numeric column."""

    column: str
    changed_rows: int
    changed_ratio: float
    max_abs_change: float | None
    max_rel_change: float | None
    mean_a: float | None
    mean_b: float | None
    mean_delta: float | None
    std_a: float | None
    std_b: float | None
    std_delta: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "column": self.column,
            "changed_rows": self.changed_rows,
            "changed_ratio": self.changed_ratio,
            "max_abs_change": self.max_abs_change,
            "max_rel_change": self.max_rel_change,
            "mean_a": self.mean_a,
            "mean_b": self.mean_b,
            "mean_delta": self.mean_delta,
            "std_a": self.std_a,
            "std_b": self.std_b,
            "std_delta": self.std_delta,
        }


@dataclass(frozen=True, slots=True)
class VersionDiffReport:
    """Structured comparison of two versions at one availability watermark."""

    version_a: str
    version_b: str
    as_of: datetime | None
    rows_a: int
    rows_b: int
    rows_added: int
    rows_removed: int
    common_rows: int
    entities_added: tuple[str, ...]
    entities_removed: tuple[str, ...]
    columns: tuple[ColumnDiff, ...]
    changed_cell_ratio: float
    content_hash_a: str
    content_hash_b: str

    def to_dict(self) -> dict[str, object]:
        return {
            "version_a": self.version_a,
            "version_b": self.version_b,
            "as_of": self.as_of.isoformat() if self.as_of is not None else None,
            "rows_a": self.rows_a,
            "rows_b": self.rows_b,
            "rows_added": self.rows_added,
            "rows_removed": self.rows_removed,
            "common_rows": self.common_rows,
            "entities_added": list(self.entities_added),
            "entities_removed": list(self.entities_removed),
            "columns": [column.to_dict() for column in self.columns],
            "changed_cell_ratio": self.changed_cell_ratio,
            "content_hash_a": self.content_hash_a,
            "content_hash_b": self.content_hash_b,
        }


class TemporalGate:
    """Read and compare point-in-time tables without domain assumptions.

    ``sources`` may be one source or a mapping from ``version_tag`` to source.
    A source is a Parquet file, glob, directory, ``s3://`` URI, a sequence of
    those, or an already materialized :class:`polars.DataFrame`.
    """

    def __init__(
        self,
        sources: SourceInput | Mapping[str, SourceInput],
        *,
        spec: TemporalSpec | None = None,
        default_version: str | None = None,
        connection_setup: ConnectionSetup | None = None,
    ) -> None:
        self._spec = spec if spec is not None else TemporalSpec()
        self._connection_setup = connection_setup
        self._single_source: SourceInput | None = None
        self._version_hashes: dict[str, str] = {}
        if isinstance(sources, Mapping):
            if not sources:
                raise TemporalContractError("TemporalGate requires at least one versioned source")
            self._versions: dict[str, SourceInput] = dict(sources)
            if default_version is None:
                default_version = next(iter(self._versions))
            if default_version not in self._versions:
                raise TemporalContractError(
                    f"default_version {default_version!r} is not one of {sorted(self._versions)}"
                )
            self._default_version: str | None = default_version
        else:
            self._versions = {}
            self._single_source = sources
            self._default_version = default_version

    @property
    def spec(self) -> TemporalSpec:
        return self._spec

    @property
    def version_tags(self) -> tuple[str, ...]:
        return tuple(self._versions)

    def assert_version_immutable(self, version_tag: str, frame: pl.DataFrame) -> str:
        """Register a full snapshot hash and reject silent overwrites.

        Pass the complete registered snapshot, not a filtered view: two
        different filters of the same version are legitimately different
        frames.  The check is process-local unless a host adapter persists the
        returned hash.
        """

        digest = content_hash(frame)
        previous = self._version_hashes.get(version_tag)
        if previous is not None and previous != digest:
            raise VersionMutationError(
                f"version tag {version_tag!r} refers to different content; bump the "
                "version tag instead of overwriting history",
                details={"previous_hash": previous, "current_hash": digest},
            )
        self._version_hashes[version_tag] = digest
        return digest

    def read_as_of(
        self,
        *,
        as_of: datetime,
        version_tag: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        entities: Sequence[str] | None = None,
        columns: Sequence[str] | None = None,
        deduplicate: bool = True,
    ) -> pl.DataFrame:
        """Read rows whose knowledge time is visible at ``as_of``.

        When ``deduplicate`` is true, each point-in-time key resolves to its
        latest revision visible at the cutoff instead of returning every
        historical revision.
        """

        cutoff = _aware(as_of, "as_of")
        start = _aware(start_time, "start_time") if start_time is not None else None
        end = _aware(end_time, "end_time") if end_time is not None else None
        if start is not None and end is not None and start >= end:
            raise TemporalContractError("start_time must be earlier than end_time")
        source = self._source_for(version_tag)
        if isinstance(source, pl.DataFrame):
            frame = _filter_frame(
                source,
                spec=self._spec,
                as_of=cutoff,
                start_time=start,
                end_time=end,
                entities=entities,
                columns=columns,
                deduplicate=deduplicate,
            )
            return frame.sort(self._spec.key_columns)
        files = _files(source)
        connection = self._connect()
        try:
            frame = self.read_parquet_as_of(
                connection,
                files,
                spec=self._spec,
                as_of=cutoff,
                start_time=start,
                end_time=end,
                entities=entities,
                columns=columns,
                deduplicate=deduplicate,
            )
        finally:
            connection.close()
        return frame.sort(self._spec.key_columns)

    def rolling_baseline(
        self,
        *,
        as_of: datetime,
        n_windows: int,
        window: timedelta,
        version_tag: str | None = None,
        entities: Sequence[str] | None = None,
        columns: Sequence[str] | None = None,
    ) -> pl.DataFrame:
        """Aggregate the previous ``n_windows`` visible snapshots into a baseline.

        Each window is ``[window_end - window, window_end)`` and is read with
        the knowledge cutoff at its own ``window_end``.  A restatement that was
        published after a window ended therefore cannot rewrite that window's
        baseline; it only appears in later windows where it was visible.

        The returned frame has one row per window that contained visible rows,
        numeric medians for the requested columns, and ``__baseline_window_start``
        / ``__baseline_window_end`` alignment columns.  Pass it directly to
        :meth:`cleanhatch.invar.InvarGate.check_clean` as ``baseline``.

        Windows with no visible rows are omitted.  If every window is empty the
        call raises :class:`TemporalContractError` instead of pretending that a
        drift baseline exists.  ``as_of`` should be at or after the latest
        knowledge time that you want visible.
        """

        cutoff = _aware(as_of, "as_of")
        if n_windows < 1:
            raise TemporalContractError("n_windows must be at least 1")
        if window <= timedelta(0):
            raise TemporalContractError("window must be positive")
        requested = (
            None if columns is None else list(dict.fromkeys((*self._spec.key_columns, *columns)))
        )
        frames: list[pl.DataFrame] = []
        for offset in range(n_windows - 1, -1, -1):
            window_end = cutoff - window * offset
            window_start = window_end - window
            snapshot = self.read_as_of(
                as_of=window_end,
                version_tag=version_tag,
                start_time=window_start,
                end_time=window_end,
                entities=entities,
                columns=requested,
                deduplicate=True,
            )
            if snapshot.is_empty():
                continue
            if columns is None:
                numeric = [name for name, dtype in snapshot.schema.items() if dtype.is_numeric()]
            else:
                numeric = [name for name in columns if name in snapshot.columns]
            if not numeric:
                continue
            medians = snapshot.select(
                [pl.col(name).cast(pl.Float64).median().alias(name) for name in numeric]
            ).with_columns(
                pl.lit(window_start).alias("__baseline_window_start"),
                pl.lit(window_end).alias("__baseline_window_end"),
            )
            frames.append(medians)
        if not frames:
            raise TemporalContractError(
                "rolling baseline has no visible rows in the requested windows",
                details={
                    "as_of": cutoff.isoformat(),
                    "n_windows": n_windows,
                    "window_seconds": window.total_seconds(),
                },
            )
        return pl.concat(frames, how="diagonal_relaxed").sort("__baseline_window_end")

    def diff_versions(
        self,
        version_a: str | pl.DataFrame,
        version_b: str | pl.DataFrame,
        *,
        as_of: datetime | None = None,
        value_columns: Sequence[str] | None = None,
    ) -> VersionDiffReport:
        """Compare two versions at the same knowledge watermark.

        Passing the same version tag twice with different content is rejected:
        an immutable version tag must never silently change underneath a
        previously registered experiment.
        """

        cutoff = _aware(as_of, "as_of") if as_of is not None else None
        frame_a = self._version_frame(version_a, cutoff)
        frame_b = self._version_frame(version_b, cutoff)
        label_a = version_a if isinstance(version_a, str) else "<frame-a>"
        label_b = version_b if isinstance(version_b, str) else "<frame-b>"
        hash_a = content_hash(frame_a)
        hash_b = content_hash(frame_b)
        if isinstance(version_a, str) and version_a == version_b and hash_a != hash_b:
            raise VersionMutationError(
                f"version tag {version_a!r} refers to different content; bump the version "
                "tag instead of overwriting history",
                details={"content_hash_a": hash_a, "content_hash_b": hash_b},
            )
        return _diff_frames(
            frame_a,
            frame_b,
            spec=self._spec,
            version_a=label_a,
            version_b=label_b,
            as_of=cutoff,
            value_columns=value_columns,
            hash_a=hash_a,
            hash_b=hash_b,
        )

    def asof_join_scd2(
        self,
        base: pl.DataFrame,
        dimension: pl.DataFrame,
        *,
        as_of: datetime,
        interval_start_col: str = "valid_from",
        interval_end_col: str = "valid_to",
        dimension_knowledge_time_col: str = "available_at",
        attributes: Sequence[str] = (),
        membership_col: str | None = None,
    ) -> pl.DataFrame:
        """Attach slowly-changing-dimension attributes to a base table.

        An interval ``[valid_from, valid_to)`` is visible only when its
        disclosure time is at or before both ``as_of`` and the base row's own
        knowledge time.  Rows outside every interval receive ``False`` for
        ``membership_col`` (or ``temporal_matched`` when no membership column is
        supplied) and null attributes.
        """

        spec = self._spec
        entity = spec.entity_col
        event = spec.event_time_col
        base_knowledge = spec.knowledge_time_col
        cutoff = _aware(as_of, "as_of")
        required_dim = {
            entity,
            interval_start_col,
            dimension_knowledge_time_col,
            *attributes,
        }
        if membership_col is not None:
            required_dim.add(membership_col)
        missing_dim = required_dim - set(dimension.columns)
        if missing_dim:
            raise TemporalContractError(
                f"dimension is missing required columns: {sorted(missing_dim)}"
            )
        missing_base = {entity, event, base_knowledge} - set(base.columns)
        if missing_base:
            raise TemporalContractError(f"base is missing required columns: {sorted(missing_base)}")
        dimension_attributes: set[str] = set(attributes)
        if membership_col is not None:
            dimension_attributes.add(membership_col)
        collisions = dimension_attributes & set(base.columns)
        if collisions:
            raise TemporalContractError(
                f"dimension attributes would overwrite base columns: {sorted(collisions)}"
            )

        visible = dimension.filter(pl.col(dimension_knowledge_time_col) <= pl.lit(cutoff))
        _reject_overlapping_intervals(
            visible,
            entity_col=entity,
            start_col=interval_start_col,
            end_col=interval_end_col,
        )
        base_indexed = base.with_row_index("__temporal_row")
        if visible.is_empty():
            return (
                _fill_unmatched(
                    base_indexed,
                    attributes=attributes,
                    membership_col=membership_col,
                )
                .drop("__temporal_row")
                .sort([event, entity])
            )

        aliased = visible.select(
            pl.col(entity),
            pl.col(interval_start_col).alias("__temporal_start"),
            pl.col(interval_end_col).alias("__temporal_end"),
            pl.col(dimension_knowledge_time_col).alias("__temporal_knowledge"),
            *(
                [pl.col(membership_col).alias("__temporal_membership_source")]
                if membership_col is not None
                else []
            ),
            *[
                pl.col(attribute).alias(f"__temporal_source_{attribute}")
                for attribute in attributes
            ],
        )
        joined = (
            base_indexed.lazy()
            .join(aliased.lazy(), on=entity, how="left")
            .filter(
                (pl.col("__temporal_start") <= pl.col(event))
                & (pl.col("__temporal_end").is_null() | (pl.col(event) < pl.col("__temporal_end")))
                & (pl.col("__temporal_knowledge") <= pl.col(base_knowledge))
            )
            .collect()
        )
        if joined.is_empty():
            return (
                _fill_unmatched(
                    base_indexed,
                    attributes=attributes,
                    membership_col=membership_col,
                )
                .drop("__temporal_row")
                .sort([event, entity])
            )
        aggregations: list[pl.Expr] = []
        if membership_col is not None:
            aggregations.append(
                pl.col("__temporal_membership_source").max().alias("__temporal_membership")
            )
        else:
            aggregations.append(pl.lit(True).alias("__temporal_matched"))
        aggregations.extend(
            pl.col(f"__temporal_source_{attribute}")
            .sort_by(["__temporal_knowledge", "__temporal_start"])
            .last()
            .alias(f"__temporal_{attribute}")
            for attribute in attributes
        )
        flags = (
            joined.group_by("__temporal_row").agg(aggregations)
            if not joined.is_empty()
            else pl.DataFrame(schema={"__temporal_row": pl.UInt32})
        )
        merged = base_indexed.join(flags, on="__temporal_row", how="left")
        if membership_col is not None:
            merged = merged.with_columns(
                pl.col("__temporal_membership").fill_null(False).alias(membership_col)
            ).drop("__temporal_membership")
        else:
            merged = merged.with_columns(
                pl.col("__temporal_matched").fill_null(False).alias("temporal_matched")
            ).drop("__temporal_matched")
        for attribute in attributes:
            merged = merged.with_columns(pl.col(f"__temporal_{attribute}").alias(attribute)).drop(
                f"__temporal_{attribute}"
            )
        return merged.drop("__temporal_row").sort([event, entity])

    @staticmethod
    def read_parquet_as_of(
        connection: duckdb.DuckDBPyConnection,
        files: Sequence[str],
        *,
        spec: TemporalSpec,
        as_of: datetime,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        entities: Sequence[str] | None = None,
        columns: Sequence[str] | None = None,
        deduplicate: bool = True,
    ) -> pl.DataFrame:
        """Execute the generic point-in-time Parquet query.

        This helper is intentionally separate from source resolution and
        credential handling: a host adapter owns those policies.
        """

        cutoff = _aware(as_of, "as_of")
        if columns is not None:
            missing_keys = set(spec.key_columns) - set(columns)
            if missing_keys:
                raise TemporalContractError(
                    f"selected columns must include the point-in-time key: {sorted(missing_keys)}"
                )
        selected = "*" if columns is None else ", ".join(_quote(name) for name in columns)
        conditions = [f"{_quote(spec.knowledge_time_col)} <= ?"]
        parameters: list[object] = [list(files), cutoff]
        if start_time is not None:
            conditions.append(f"{_quote(spec.event_time_col)} >= ?")
            parameters.append(_aware(start_time, "start_time"))
        if end_time is not None:
            conditions.append(f"{_quote(spec.event_time_col)} < ?")
            parameters.append(_aware(end_time, "end_time"))
        if entities is not None:
            conditions.append(f"{_quote(spec.entity_col)} IN (SELECT unnest(?::VARCHAR[]))")
            parameters.append(list(entities))
        query = (
            f"SELECT {selected} "
            "FROM read_parquet(?, hive_partitioning = true, union_by_name = true) "
            f"WHERE {' AND '.join(conditions)}"
        )
        if deduplicate:
            key_columns = ", ".join(_quote(name) for name in spec.key_columns)
            query += (
                f" QUALIFY ROW_NUMBER() OVER (PARTITION BY {key_columns} "
                f"ORDER BY {_quote(spec.knowledge_time_col)} DESC) = 1"
            )
        relation = connection.sql(query, params=parameters)
        if relation is None:
            raise TemporalContractError("DuckDB did not return a result relation")
        return _frame_from_relation(relation)

    def _connect(self) -> duckdb.DuckDBPyConnection:
        try:
            connection = duckdb.connect(
                ":memory:",
                config={
                    "autoinstall_known_extensions": False,
                    "autoload_known_extensions": False,
                },
            )
        except duckdb.Error as error:
            raise TemporalContractError("DuckDB could not open an in-memory connection") from error
        try:
            connection.execute("SET TimeZone = 'UTC'")
            if self._connection_setup is not None:
                self._connection_setup(connection)
        except Exception:
            connection.close()
            raise
        return connection

    def _source_for(self, version_tag: str | None) -> SourceInput:
        if self._versions:
            tag = version_tag or self._default_version
            if tag is None or tag not in self._versions:
                raise TemporalContractError(
                    f"unknown version_tag {tag!r}; available versions: {sorted(self._versions)}"
                )
            return self._versions[tag]
        if version_tag is not None:
            if self._default_version is None:
                raise TemporalContractError(
                    f"this store is unversioned and does not accept version_tag {version_tag!r}"
                )
            if version_tag != self._default_version:
                raise TemporalContractError(
                    f"this store exposes a single version ({self._default_version!r}), "
                    f"not {version_tag!r}"
                )
        source = getattr(self, "_single_source", None)
        if source is None:
            raise TemporalContractError("TemporalGate has no source configured")
        return cast(SourceInput, source)

    def _version_frame(self, version: str | pl.DataFrame, as_of: datetime | None) -> pl.DataFrame:
        if isinstance(version, pl.DataFrame):
            return _filter_frame(
                version,
                spec=self._spec,
                as_of=as_of,
                start_time=None,
                end_time=None,
                entities=None,
                columns=None,
            )
        source = self._source_for(version)
        if isinstance(source, pl.DataFrame):
            return _filter_frame(
                source,
                spec=self._spec,
                as_of=as_of,
                start_time=None,
                end_time=None,
                entities=None,
                columns=None,
            )
        return self.read_as_of(as_of=as_of or datetime.now(UTC), version_tag=version)


def _diff_frames(
    frame_a: pl.DataFrame,
    frame_b: pl.DataFrame,
    *,
    spec: TemporalSpec,
    version_a: str,
    version_b: str,
    as_of: datetime | None,
    value_columns: Sequence[str] | None,
    hash_a: str,
    hash_b: str,
) -> VersionDiffReport:
    entity = spec.entity_col
    key_columns = list(spec.key_columns)
    for frame, label in ((frame_a, "version_a"), (frame_b, "version_b")):
        missing = {*key_columns, entity} - set(frame.columns)
        if missing:
            raise TemporalContractError(f"{label} is missing key columns: {sorted(missing)}")
        duplicates = frame.group_by(key_columns).len().filter(pl.col("len") > 1)
        if not duplicates.is_empty():
            raise TemporalContractError(
                f"{label} contains duplicate point-in-time keys",
                details={"duplicates": duplicates.head(20).to_dicts()},
            )
    ordered_a = frame_a.sort(key_columns)
    ordered_b = frame_b.sort(key_columns)
    added = ordered_b.join(ordered_a.select(key_columns), on=key_columns, how="anti")
    removed = ordered_a.join(ordered_b.select(key_columns), on=key_columns, how="anti")
    common = ordered_a.join(ordered_b, on=key_columns, how="inner", suffix="__b")
    # ``common`` carries a's columns plus ``<name>__b`` for b's non-key columns.
    entities_a = set(ordered_a.get_column(entity).cast(pl.String).to_list())
    entities_b = set(ordered_b.get_column(entity).cast(pl.String).to_list())
    if value_columns is None:
        values = [
            name
            for name, dtype in ordered_a.schema.items()
            if name not in key_columns and dtype.is_numeric() and name in ordered_b.columns
        ]
    else:
        values = list(value_columns)
    columns: list[ColumnDiff] = []
    total_changed = 0
    for name in values:
        if name not in ordered_a.columns or name not in ordered_b.columns:
            raise TemporalContractError(
                f"cannot diff column {name!r}; it is absent from one version"
            )
        a_name = name
        b_name = f"{name}__b"
        a = pl.col(a_name)
        b = pl.col(b_name)
        both_null = a.is_null() & b.is_null()
        both_present = a.is_not_null() & b.is_not_null()
        changed = ~both_null & ((a.is_null() != b.is_null()) | (both_present & (a != b)))
        changed_rows = int(common.select(changed.sum()).item() or 0)
        total_changed += changed_rows
        difference = (a - b).abs()
        max_abs = _float_or_none(common.filter(both_present).select(difference.max()).item())
        denominator = pl.max_horizontal(a.abs(), b.abs())
        max_rel = _float_or_none(
            common.filter(both_present & (denominator > 0))
            .select((difference / denominator).max())
            .item()
        )
        mean_a = _float_or_none(common.select(a.mean()).item())
        mean_b = _float_or_none(common.select(b.mean()).item())
        std_a = _float_or_none(common.select(a.std()).item())
        std_b = _float_or_none(common.select(b.std()).item())
        columns.append(
            ColumnDiff(
                column=name,
                changed_rows=changed_rows,
                changed_ratio=(changed_rows / common.height) if common.height else 0.0,
                max_abs_change=max_abs,
                max_rel_change=max_rel,
                mean_a=mean_a,
                mean_b=mean_b,
                mean_delta=_delta(mean_a, mean_b),
                std_a=std_a,
                std_b=std_b,
                std_delta=_delta(std_a, std_b),
            )
        )
    denominator = common.height * len(columns)
    return VersionDiffReport(
        version_a=version_a,
        version_b=version_b,
        as_of=as_of,
        rows_a=ordered_a.height,
        rows_b=ordered_b.height,
        rows_added=added.height,
        rows_removed=removed.height,
        common_rows=common.height,
        entities_added=tuple(sorted(entities_b - entities_a)),
        entities_removed=tuple(sorted(entities_a - entities_b)),
        columns=tuple(columns),
        changed_cell_ratio=(total_changed / denominator) if denominator else 0.0,
        content_hash_a=hash_a,
        content_hash_b=hash_b,
    )


def _filter_frame(
    frame: pl.DataFrame,
    *,
    spec: TemporalSpec,
    as_of: datetime | None,
    start_time: datetime | None,
    end_time: datetime | None,
    entities: Sequence[str] | None,
    columns: Sequence[str] | None,
    deduplicate: bool = True,
) -> pl.DataFrame:
    required = {spec.entity_col, spec.event_time_col, spec.knowledge_time_col}
    missing = required - set(frame.columns)
    if missing:
        raise TemporalContractError(f"frame is missing coordinate columns: {sorted(missing)}")
    selected = frame
    if as_of is not None:
        knowledge = _utc_expr(selected, spec.knowledge_time_col)
        selected = selected.filter(knowledge <= pl.lit(as_of))
    if deduplicate:
        selected = selected.sort(
            [*spec.key_columns, _utc_expr(selected, spec.knowledge_time_col)],
            maintain_order=True,
        ).unique(
            subset=list(spec.key_columns),
            keep="last",
            maintain_order=True,
        )
    if start_time is not None:
        event = _utc_expr(selected, spec.event_time_col)
        selected = selected.filter(event >= pl.lit(start_time))
    if end_time is not None:
        event = _utc_expr(selected, spec.event_time_col)
        selected = selected.filter(event < pl.lit(end_time))
    if entities is not None:
        selected = selected.filter(pl.col(spec.entity_col).is_in(list(entities)))
    if columns is not None:
        missing_keys = set(spec.key_columns) - set(columns)
        if missing_keys:
            raise TemporalContractError(
                f"requested columns must include the point-in-time key: {sorted(missing_keys)}"
            )
        missing_columns = set(columns) - set(selected.columns)
        if missing_columns:
            raise TemporalContractError(
                f"frame is missing requested columns: {sorted(missing_columns)}"
            )
        selected = selected.select(list(columns))
    return selected


def _reject_overlapping_intervals(
    dimension: pl.DataFrame,
    *,
    entity_col: str,
    start_col: str,
    end_col: str,
) -> None:
    if dimension.is_empty():
        return
    if end_col not in dimension.columns:
        return
    ordered = dimension.sort([entity_col, start_col])
    overlapping = (
        ordered.with_columns(
            pl.col(start_col).shift(1).over(entity_col).alias("__previous_start"),
            pl.col(end_col).shift(1).over(entity_col).alias("__previous_end"),
        )
        .filter(
            pl.col("__previous_start").is_not_null()
            & (pl.col("__previous_end").is_null() | (pl.col("__previous_end") > pl.col(start_col)))
        )
        .select(entity_col, "__previous_start", "__previous_end", start_col, end_col)
    )
    if not overlapping.is_empty():
        raise TemporalContractError(
            "dimension intervals overlap for the same entity",
            details={"conflicts": overlapping.head(20).to_dicts()},
        )


def _fill_unmatched(
    base_indexed: pl.DataFrame,
    *,
    attributes: Sequence[str],
    membership_col: str | None,
) -> pl.DataFrame:
    expressions: list[pl.Expr] = []
    if membership_col is not None:
        expressions.append(pl.lit(False).alias(membership_col))
    else:
        expressions.append(pl.lit(False).alias("temporal_matched"))
    expressions.extend(pl.lit(None).alias(attribute) for attribute in attributes)
    return base_indexed.with_columns(expressions)


def _files(source: SourceInput) -> list[str]:
    if isinstance(source, pl.DataFrame):
        raise TypeError("a materialized frame has no Parquet files")
    if isinstance(source, str | Path):
        return [str(source)]
    return [str(item) for item in source]


def _frame_from_relation(relation: duckdb.DuckDBPyRelation) -> pl.DataFrame:
    """Convert a DuckDB relation through Arrow without requiring pyarrow."""

    value = pl.from_arrow(relation)
    if isinstance(value, pl.DataFrame):
        return value
    return value.struct.unnest()


def _utc_expr(frame: pl.DataFrame, name: str) -> pl.Expr:
    """Return a comparison expression normalized to UTC for aware datetimes."""

    dtype = frame.schema[name]
    if isinstance(dtype, pl.Datetime) and dtype.time_zone is not None:
        return pl.col(name).dt.convert_time_zone("UTC")
    return pl.col(name)


def _quote(identifier: str) -> str:
    if not identifier or '"' in identifier or "\x00" in identifier:
        raise TemporalContractError(f"invalid SQL identifier: {identifier!r}")
    return f'"{identifier}"'


def _aware(value: datetime, name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise TemporalContractError(f"{name} must be timezone-aware")
    return value.astimezone(UTC)


def _float_or_none(value: object) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        candidate = float(value)
        if candidate == candidate and candidate not in (float("inf"), float("-inf")):
            return candidate
    return None


def _delta(before: float | None, after: float | None) -> float | None:
    if before is None or after is None:
        return None
    return after - before


__all__ = [
    "ColumnDiff",
    "TemporalGate",
    "SourceInput",
    "VersionDiffReport",
]
