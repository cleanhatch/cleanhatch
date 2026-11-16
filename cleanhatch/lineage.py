"""Immutable lineage manifests and single-variable ablation comparison.

A manifest binds the four axes that determine an experiment's result:

1. the dataset slice and content hash,
2. the cleaning recipe,
3. the transformation AST,
4. downstream parameters.

``ablation_diff`` answers the question an AI agent or researcher actually asks:
"what changed between these two runs, and did I accidentally change more than
one thing?"
"""

from __future__ import annotations

import json
import uuid
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Self, cast

import polars as pl
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator

from .ai_summary import MAX_AI_SUMMARY_BYTES, build_ai_summary
from .errors import NonIsolatedAblationWarning
from .hashing import content_hash, source_ast_hash
from .invar import GateReport


class _LineageModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class InputSnapshot(_LineageModel):
    """One input table and the exact version content it contributed."""

    dataset_id: str = Field(min_length=1)
    version_tag: str = Field(min_length=1)
    content_hash: str = Field(min_length=1)
    as_of: AwareDatetime
    row_count: int = Field(ge=0)
    uri: str | None = None

    @field_validator("as_of")
    @classmethod
    def normalize_as_of(cls, value: datetime) -> datetime:
        return value.astimezone(UTC)

    @classmethod
    def from_frame(
        cls,
        frame: pl.DataFrame,
        *,
        dataset_id: str,
        version_tag: str,
        as_of: datetime,
        uri: str | None = None,
    ) -> Self:
        return cls(
            dataset_id=dataset_id,
            version_tag=version_tag,
            content_hash=content_hash(frame),
            as_of=as_of,
            row_count=frame.height,
            uri=uri,
        )


class TransformSignature(_LineageModel):
    """AST identity and parameters of one transformation."""

    code_hash: str = Field(min_length=1)
    parameters: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_source(cls, source: str, *, parameters: Mapping[str, Any] | None = None) -> Self:
        return cls(
            code_hash=source_ast_hash(source),
            parameters=dict(parameters or {}),
        )


class GateEvidence(_LineageModel):
    """Serialized status and checks from one lifecycle gate."""

    stage: str = Field(min_length=1)
    status: str = Field(min_length=1)
    checks: tuple[dict[str, Any], ...] = ()
    warnings: tuple[str, ...] = ()

    @classmethod
    def from_report(cls, report: GateReport) -> Self:
        payload = report.to_dict()
        checks = cast(list[dict[str, Any]], payload.get("checks", []))
        warnings = cast(list[str], payload.get("warnings", []))
        return cls(
            stage=str(payload["stage"]),
            status=str(payload["status"]),
            checks=tuple(checks),
            warnings=tuple(str(item) for item in warnings),
        )


class LineageManifest(_LineageModel):
    """Immutable provenance bundle for one run."""

    schema_version: str = "1.0.0"
    run_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    created_at: AwareDatetime = Field(default_factory=lambda: datetime.now(UTC))
    inputs: tuple[InputSnapshot, ...] = ()
    cleaning_config: dict[str, Any] = Field(default_factory=dict)
    transform: TransformSignature
    gates: tuple[GateEvidence, ...] = ()
    metrics: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("created_at")
    @classmethod
    def normalize_created_at(cls, value: datetime) -> datetime:
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_inputs(self) -> Self:
        keys = [(item.dataset_id, item.version_tag) for item in self.inputs]
        if len(keys) != len(set(keys)):
            raise ValueError("inputs must not repeat the same (dataset_id, version_tag)")
        return self

    @classmethod
    def capture(
        cls,
        *,
        inputs: Sequence[InputSnapshot] = (),
        cleaning_config: Mapping[str, Any] | None = None,
        transform: TransformSignature,
        gates: Sequence[GateReport | GateEvidence] = (),
        metrics: Mapping[str, float] | None = None,
        metadata: Mapping[str, Any] | None = None,
        run_id: str | None = None,
    ) -> Self:
        evidence: list[GateEvidence] = []
        for gate in gates:
            evidence.append(
                GateEvidence.from_report(gate) if isinstance(gate, GateReport) else gate
            )
        return cls(
            run_id=run_id or uuid.uuid4().hex,
            inputs=tuple(inputs),
            cleaning_config=dict(cleaning_config or {}),
            transform=transform,
            gates=tuple(evidence),
            metrics=dict(metrics or {}),
            metadata=dict(metadata or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(
            self.to_dict(),
            indent=indent,
            ensure_ascii=False,
            allow_nan=False,
            default=str,
        )

    def to_ai_summary(
        self,
        *,
        max_bytes: int = MAX_AI_SUMMARY_BYTES,
    ) -> dict[str, object]:
        """Return a compact, context-window-safe evidence summary."""

        return build_ai_summary(self, max_bytes=max_bytes)

    def to_ai_summary_json(
        self,
        *,
        max_bytes: int = MAX_AI_SUMMARY_BYTES,
    ) -> str:
        """Serialize the AI summary as compact UTF-8 JSON within the budget."""

        return json.dumps(
            self.to_ai_summary(max_bytes=max_bytes),
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )

    @classmethod
    def ablation_diff(cls, base: LineageManifest, candidate: LineageManifest) -> AblationDiffReport:
        """Compare two runs and flag non-isolated ablations."""

        changed: list[str] = []
        if _axis(base.inputs) != _axis(candidate.inputs):
            changed.append("dataset")
        if _axis(base.cleaning_config) != _axis(candidate.cleaning_config):
            changed.append("cleaning")
        if base.transform.code_hash != candidate.transform.code_hash:
            changed.append("transform")
        if _axis(base.transform.parameters) != _axis(candidate.transform.parameters):
            changed.append("parameters")
        isolated = len(changed) <= 1
        warning_messages: list[str] = []
        if not isolated:
            message = (
                "ablation changed more than one variable at once: "
                f"{', '.join(changed)}; the result cannot be attributed to one change"
            )
            warning_messages.append(message)
            warnings.warn(message, NonIsolatedAblationWarning, stacklevel=2)
        metric_deltas: dict[str, float | None] = {}
        for key in sorted(set(base.metrics) | set(candidate.metrics)):
            before = base.metrics.get(key)
            after = candidate.metrics.get(key)
            metric_deltas[key] = (
                after - before if before is not None and after is not None else None
            )
        base_gates = {(gate.stage, gate.status) for gate in base.gates}
        candidate_gates = {(gate.stage, gate.status) for gate in candidate.gates}
        return AblationDiffReport(
            base_run_id=base.run_id,
            candidate_run_id=candidate.run_id,
            changed_axes=tuple(changed),
            is_clean_ablation=isolated,
            is_no_op=not changed,
            metric_deltas=metric_deltas,
            gate_changes=(
                tuple(sorted(base_gates - candidate_gates)),
                tuple(sorted(candidate_gates - base_gates)),
            ),
            warnings=tuple(warning_messages),
        )


@dataclass(frozen=True, slots=True)
class AblationDiffReport:
    """The result of comparing two lineage manifests."""

    base_run_id: str
    candidate_run_id: str
    changed_axes: tuple[str, ...]
    is_clean_ablation: bool
    is_no_op: bool
    metric_deltas: dict[str, float | None]
    gate_changes: tuple[tuple[tuple[str, str], ...], tuple[tuple[str, str], ...]]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        removed, added = self.gate_changes
        return {
            "base_run_id": self.base_run_id,
            "candidate_run_id": self.candidate_run_id,
            "changed_axes": list(self.changed_axes),
            "is_clean_ablation": self.is_clean_ablation,
            "is_no_op": self.is_no_op,
            "metric_deltas": dict(self.metric_deltas),
            "gate_changes": {
                "removed": [{"stage": stage, "status": status} for stage, status in removed],
                "added": [{"stage": stage, "status": status} for stage, status in added],
            },
            "warnings": list(self.warnings),
        }


def _axis(value: object) -> str:
    if isinstance(value, BaseModel):
        return _axis(value.model_dump(mode="json"))
    if isinstance(value, tuple | list):
        items = cast(list[object], value)
        return json.dumps(
            [_axis(item) for item in items],
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        return json.dumps(
            {str(key): _axis(item) for key, item in mapping.items()},
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


__all__ = [
    "AblationDiffReport",
    "GateEvidence",
    "InputSnapshot",
    "LineageManifest",
    "TransformSignature",
]
