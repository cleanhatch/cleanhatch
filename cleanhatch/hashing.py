"""Deterministic fingerprints for tables and transformations.

The hash is a canonical, collision-free serialization of the schema and rows.
It is intentionally independent of Polars' internal hash seeds and of the
machine's local timezone, so two processes produce the same value.
"""

from __future__ import annotations

import ast
import hashlib
import io
from typing import Any, cast

import polars as pl


def content_hash(frame: pl.DataFrame) -> str:
    """SHA-256 over the schema and every cell of ``frame``.

    Values are normalized column-by-column, then serialized as an
    uncompressed Arrow IPC stream.  This keeps the work in Polars/Rust instead
    of a Python row loop and gives a byte-level canonical representation.
    """

    digest = hashlib.sha256()
    for name, dtype in frame.schema.items():
        _feed_text(digest, name)
        _feed_text(digest, _schema_dtype_name(dtype))
    buffer = io.BytesIO()
    _canonicalize(frame).write_ipc(buffer, compression="uncompressed")
    digest.update(buffer.getvalue())
    return digest.hexdigest()


def source_ast_hash(source: str) -> str:
    """SHA-256 of a normalized Python AST, ignoring formatting and comments."""

    try:
        tree = ast.parse(source)
    except SyntaxError:
        # A malformed transformation still needs a stable identity for lineage;
        # hash the raw text rather than pretending it has a valid AST.
        return hashlib.sha256(source.encode("utf-8")).hexdigest()
    normalized = ast.dump(tree, annotate_fields=True, include_attributes=False)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _feed_text(digest: Any, text: str) -> None:
    payload = text.encode("utf-8")
    digest.update(len(payload).to_bytes(8, "little"))
    digest.update(payload)


def _schema_dtype_name(dtype: pl.DataType) -> str:
    """Render a dtype without a machine-dependent timezone label.

    Aware datetimes are compared by instant, so their schema identity is
    normalized to UTC.  Naive datetimes keep their literal dtype because a
    wall-clock value without a timezone has no portable instant.  Nested
    containers are rendered recursively so their inner datetimes are
    normalized too.
    """

    if isinstance(dtype, pl.Datetime) and dtype.time_zone is not None:
        return str(pl.Datetime(dtype.time_unit, "UTC"))
    if isinstance(dtype, pl.List):
        return f"List({_schema_dtype_name(cast(pl.DataType, dtype.inner))})"
    if isinstance(dtype, pl.Array):
        return f"Array({_schema_dtype_name(cast(pl.DataType, dtype.inner))}, {dtype.size})"
    if isinstance(dtype, pl.Struct):
        fields = ", ".join(
            f"{field.name}: {_schema_dtype_name(cast(pl.DataType, field.dtype))}"
            for field in dtype.fields
        )
        return f"Struct({{{fields}}})"
    return str(dtype)


def _canonicalize(frame: pl.DataFrame) -> pl.DataFrame:
    """Normalize values that have multiple binary representations."""

    expressions: list[pl.Expr] = []
    for name, dtype in frame.schema.items():
        if _needs_canonicalization(dtype):
            expressions.append(_canonical_expr(pl.col(name), dtype).alias(name))
    normalized = frame.with_columns(expressions) if expressions else frame
    return normalized.rechunk()


def _needs_canonicalization(dtype: pl.DataType) -> bool:
    if dtype.is_float():
        return True
    if isinstance(dtype, pl.Datetime) and dtype.time_zone is not None:
        return True
    if isinstance(dtype, pl.List):
        return _needs_canonicalization(cast(pl.DataType, dtype.inner))
    if isinstance(dtype, pl.Array):
        return _needs_canonicalization(cast(pl.DataType, dtype.inner))
    if isinstance(dtype, pl.Struct):
        return any(
            _needs_canonicalization(cast(pl.DataType, field.dtype)) for field in dtype.fields
        )
    return False


def _canonical_expr(expr: pl.Expr, dtype: pl.DataType) -> pl.Expr:
    if dtype.is_float():
        canonical_nan = pl.lit(float("nan"), dtype=dtype)
        return (
            pl.when(expr.is_nan())
            .then(canonical_nan)
            .when(expr == 0.0)
            .then(pl.lit(0.0, dtype=dtype))
            .otherwise(expr)
        )
    if isinstance(dtype, pl.Datetime) and dtype.time_zone is not None:
        return expr.dt.convert_time_zone("UTC")
    if isinstance(dtype, pl.List):
        inner = cast(pl.DataType, dtype.inner)
        if not _needs_canonicalization(inner):
            return expr
        return expr.list.eval(_canonical_expr(pl.element(), inner))
    if isinstance(dtype, pl.Array):
        inner = cast(pl.DataType, dtype.inner)
        if not _needs_canonicalization(inner):
            return expr
        return expr.arr.eval(_canonical_expr(pl.element(), inner))
    if isinstance(dtype, pl.Struct):
        if not _needs_canonicalization(dtype):
            return expr
        fields = [
            _canonical_expr(expr.struct.field(field.name), cast(pl.DataType, field.dtype)).alias(
                field.name
            )
            for field in dtype.fields
        ]
        return pl.struct(fields)
    return expr


__all__ = ["content_hash", "source_ast_hash"]
