"""Bounded AI-facing summaries for lineage manifests.

An agent context window is a hard budget.  The summary therefore starts with a
compact evidence set and progressively drops lower-value detail until the
UTF-8 payload fits ``MAX_AI_SUMMARY_BYTES``.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .lineage import LineageManifest

MAX_AI_SUMMARY_BYTES = 2048
_FAILED_GATE_LIMIT = 5
_FAILED_MESSAGE_LIMIT = 120
_FAILED_GATE_LIMIT_COMPACT = 1
_FAILED_MESSAGE_LIMIT_COMPACT = 80
_COMPACT_METRIC_LIMIT = 3
_COMPACT_INPUT_LIMIT = 2


def build_ai_summary(
    manifest: LineageManifest,
    *,
    max_bytes: int = MAX_AI_SUMMARY_BYTES,
) -> dict[str, object]:
    """Build a JSON-serializable summary that fits the byte budget."""

    if max_bytes < 256:
        raise ValueError("max_bytes must be at least 256")

    payload = _base_payload(manifest)
    if _serialized_size(payload) <= max_bytes:
        return payload

    payload["truncated"] = True

    payload["metrics"] = dict(list(payload["metrics"].items())[:_COMPACT_METRIC_LIMIT])
    finalized = _finalize(payload, max_bytes)
    if finalized is not None:
        return finalized

    payload["failed_gates"] = payload["failed_gates"][:_FAILED_GATE_LIMIT_COMPACT]
    for failure in payload["failed_gates"]:
        failure["message"] = failure["message"][:_FAILED_MESSAGE_LIMIT_COMPACT]
    finalized = _finalize(payload, max_bytes)
    if finalized is not None:
        return finalized

    inputs = payload["inputs"]
    payload["inputs_omitted"] = max(0, len(inputs) - _COMPACT_INPUT_LIMIT)
    payload["inputs"] = inputs[:_COMPACT_INPUT_LIMIT]
    finalized = _finalize(payload, max_bytes)
    if finalized is not None:
        return finalized

    # Last resort: keep the identity and aggregate status, drop bulky evidence.
    payload["metrics"] = {}
    payload["failed_gates"] = []
    payload["inputs"] = payload["inputs"][:1]
    payload["schema_version"] = payload["schema_version"][:16]
    payload["run_id"] = payload["run_id"][:32]
    for item in payload["inputs"]:
        item["dataset_id"] = item["dataset_id"][:64]
        item["version_tag"] = item["version_tag"][:32]
    payload["emergency_compaction"] = True
    finalized = _finalize(payload, max_bytes)
    if finalized is None:
        raise ValueError(f"AI summary cannot fit within {max_bytes} bytes")
    return finalized


def _base_payload(manifest: LineageManifest) -> dict[str, Any]:
    return {
        "schema_version": manifest.schema_version,
        "run_id": manifest.run_id,
        "status": _aggregate_status(manifest),
        "inputs": [
            {
                "dataset_id": item.dataset_id,
                "version_tag": item.version_tag,
                "content_hash": item.content_hash[:12],
                "row_count": item.row_count,
            }
            for item in manifest.inputs
        ],
        "transform": {"code_hash": manifest.transform.code_hash[:12]},
        "failed_gates": _failed_gates(
            manifest,
            limit=_FAILED_GATE_LIMIT,
            message_limit=_FAILED_MESSAGE_LIMIT,
        ),
        "warnings_count": sum(len(gate.warnings) for gate in manifest.gates),
        "metrics": dict(manifest.metrics),
        "truncated": False,
    }


def _aggregate_status(manifest: LineageManifest) -> str:
    if not manifest.gates:
        return "UNKNOWN"
    statuses = [_normalize_status(gate.status) for gate in manifest.gates]
    if "FAILED" in statuses:
        return "FAILED"
    if "WARNING" in statuses or any(gate.warnings for gate in manifest.gates):
        return "WARNING"
    if all(status == "SKIPPED" for status in statuses):
        return "SKIPPED"
    if all(status in {"PASSED", "SKIPPED"} for status in statuses):
        return "PASSED"
    return "UNKNOWN"


def _normalize_status(status: str) -> str:
    normalized = status.upper()
    if normalized in {"FAILED", "ERROR"}:
        return "FAILED"
    if normalized in {"WARNING", "WARN"}:
        return "WARNING"
    if normalized in {"SKIPPED", "SKIP"}:
        return "SKIPPED"
    if normalized == "PASSED":
        return "PASSED"
    return "UNKNOWN"


def _failed_gates(
    manifest: LineageManifest,
    *,
    limit: int,
    message_limit: int,
) -> list[dict[str, object]]:
    failures: list[dict[str, object]] = []
    for gate in manifest.gates:
        for check in gate.checks:
            status = str(check.get("status", "")).upper()
            if status not in {"FAILED", "ERROR"}:
                continue
            failures.append(
                {
                    "name": str(check.get("name", "unnamed")),
                    "stage": str(check.get("stage", gate.stage)),
                    "message": str(check.get("message", ""))[:message_limit],
                }
            )
            if len(failures) >= limit:
                return failures
    return failures


def _finalize(payload: dict[str, Any], max_bytes: int) -> dict[str, Any] | None:
    candidate = dict(payload)
    for _ in range(4):
        size = _serialized_size(candidate)
        if size > max_bytes:
            return None
        if candidate.get("byte_length") == size:
            return candidate
        candidate["byte_length"] = size
    return candidate if _serialized_size(candidate) <= max_bytes else None


def _serialized_size(payload: dict[str, Any]) -> int:
    return len(_serialize(payload))


def _serialize(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


__all__ = ["MAX_AI_SUMMARY_BYTES", "build_ai_summary"]
