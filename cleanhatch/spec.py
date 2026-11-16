"""Domain-neutral table coordinates for the cleanhatch temporal kernel.

The guard kernel never assumes a market-bar schema.  Every table is described
by the four coordinates it actually uses:

``(entity, event_time, knowledge_time, version_tag)``

Application code supplies the concrete column names through
:class:`TemporalSpec`; the kernel itself only ever refers to the configured
roles.  The default names are deliberately generic so this module can be
extracted into a standalone package without renaming a single column.
"""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TemporalSpec(BaseModel):
    """Column roles for one point-in-time table.

    ``positive_numeric_cols`` and ``signed_numeric_cols`` drive generic
    invariant checks and perturbation policies.  A column may appear in one
    numeric role only; boolean masks are kept separate because flipping a mask
    is a different kind of perturbation from scaling a value.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    entity_col: str = Field(default="entity_id", min_length=1, pattern=r"^\S+$")
    event_time_col: str = Field(default="event_time", min_length=1, pattern=r"^\S+$")
    knowledge_time_col: str = Field(default="available_at", min_length=1, pattern=r"^\S+$")
    decision_time_col: str = Field(default="decision_time", min_length=1, pattern=r"^\S+$")
    version_tag: str = Field(default="v1", min_length=1, pattern=r"^\S+$")
    positive_numeric_cols: tuple[str, ...] = ()
    signed_numeric_cols: tuple[str, ...] = ()
    boolean_mask_cols: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_roles(self) -> Self:
        coordinate_names = (
            self.entity_col,
            self.event_time_col,
            self.knowledge_time_col,
            self.decision_time_col,
        )
        if len(set(coordinate_names)) != len(coordinate_names):
            raise ValueError("entity, event, knowledge, and decision columns must be distinct")
        positive = set(self.positive_numeric_cols)
        signed = set(self.signed_numeric_cols)
        masks = set(self.boolean_mask_cols)
        if positive & signed:
            raise ValueError(
                f"a column cannot be both positive and signed numeric: {sorted(positive & signed)}"
            )
        if (positive | signed) & masks:
            raise ValueError(
                "numeric and boolean-mask roles must not overlap: "
                f"{sorted((positive | signed) & masks)}"
            )
        for role in (
            *self.positive_numeric_cols,
            *self.signed_numeric_cols,
            *self.boolean_mask_cols,
        ):
            if not role.strip():
                raise ValueError("column roles must not contain blank names")
        return self

    @property
    def key_columns(self) -> tuple[str, str]:
        """Physical row key used for sorting and uniqueness checks."""

        return (self.event_time_col, self.entity_col)

    @property
    def decision_key_columns(self) -> tuple[str, str]:
        """Factor/weight key used by point-in-time feature tables."""

        return (self.decision_time_col, self.entity_col)

    @property
    def numeric_columns(self) -> tuple[str, ...]:
        """Every numeric column registered by this spec, without duplicates."""

        return tuple(dict.fromkeys((*self.positive_numeric_cols, *self.signed_numeric_cols)))

    @property
    def time_columns(self) -> tuple[str, ...]:
        """Timestamp columns whose values must be timezone-aware milliseconds."""

        return (self.event_time_col, self.knowledge_time_col)
