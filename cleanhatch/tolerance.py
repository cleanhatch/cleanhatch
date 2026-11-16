"""Shared financial tolerance tiers for Cleanhatch pack identities.

Three tiers are deliberately distinct:

``exact_int_cents``
    Integer cents must be exactly equal.  Use this when the upstream contract
    already stores money in integer minor units and any change is a mutation.
``dust``
    A fixed absolute tolerance (``0.01`` by default) absorbs binary floating
    point dust on decimal amounts.
``bps``
    A relative tolerance expressed in basis points (``0.5`` by default, i.e.
    ``5e-5``) for large monetary or ratio-scale values where an absolute cent
    floor is meaningless.  An optional ``absolute_floor`` keeps small values
    from receiving an unreasonably tight relative bound.

The same :func:`within_tolerance` predicate builder is used by every industry
pack so a gate can switch tiers without copying comparison operators.
"""

from __future__ import annotations

from typing import Final, Literal

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

ToleranceMode = Literal["exact_int_cents", "dust", "bps"]

EXACT_INT_CENTS: Final[ToleranceMode] = "exact_int_cents"
DUST_TOLERANCE: Final[float] = 0.01
BPS_TOLERANCE: Final[float] = 0.5

# IEEE-754 subtraction at exactly 0.01 can produce 0.010000000000005116.
# This guard keeps the documented boundary inclusive without widening the
# tolerance for any meaningful amount.
_FLOAT_EPSILON: Final[float] = 1e-12


class ToleranceProfile(BaseModel):
    """Declarative tolerance configuration shared by all packs."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    mode: ToleranceMode = "dust"
    dust_tolerance: float = Field(default=DUST_TOLERANCE, ge=0.0)
    bps_tolerance: float = Field(default=BPS_TOLERANCE, ge=0.0)
    absolute_floor: float = Field(default=0.0, ge=0.0)

    @property
    def relative_tolerance(self) -> float:
        """The ``bps`` tier as a relative fraction."""

        return self.bps_tolerance * 1e-4


def within_tolerance(
    left: pl.Expr,
    right: pl.Expr,
    *,
    profile: ToleranceProfile,
    mode: ToleranceMode | None = None,
) -> pl.Expr:
    """Return a Polars predicate that is true when two expressions match."""

    tier = mode if mode is not None else profile.mode
    if tier == "exact_int_cents":
        return left == right
    if tier == "dust":
        return (left - right).abs() <= pl.lit(profile.dust_tolerance + _FLOAT_EPSILON)
    return (left - right).abs() <= _relative_limit(left, right, profile)


def at_least(
    left: pl.Expr,
    right: pl.Expr,
    *,
    profile: ToleranceProfile,
    mode: ToleranceMode | None = None,
) -> pl.Expr:
    """Return ``left >= right`` with the selected tolerance tier."""

    tier = mode if mode is not None else profile.mode
    if tier == "exact_int_cents":
        return left >= right
    if tier == "dust":
        return (right - left) <= pl.lit(profile.dust_tolerance + _FLOAT_EPSILON)
    return (right - left) <= _relative_limit(left, right, profile)


def at_most(
    left: pl.Expr,
    right: pl.Expr,
    *,
    profile: ToleranceProfile,
    mode: ToleranceMode | None = None,
) -> pl.Expr:
    """Return ``left <= right`` with the selected tolerance tier."""

    tier = mode if mode is not None else profile.mode
    if tier == "exact_int_cents":
        return left <= right
    if tier == "dust":
        return (left - right) <= pl.lit(profile.dust_tolerance + _FLOAT_EPSILON)
    return (left - right) <= _relative_limit(left, right, profile)


def scalar_within_tolerance(
    left: float | None,
    right: float | None,
    *,
    profile: ToleranceProfile,
    mode: ToleranceMode | None = None,
) -> bool:
    """Scalar counterpart for Python-side identity checks."""

    if left is None or right is None:
        return left is None and right is None
    tier = mode if mode is not None else profile.mode
    if tier == "exact_int_cents":
        return left == right
    difference = abs(left - right)
    if tier == "dust":
        return difference <= profile.dust_tolerance + _FLOAT_EPSILON
    limit = max(profile.absolute_floor, profile.relative_tolerance * max(abs(left), abs(right)))
    return difference <= limit + _FLOAT_EPSILON


def _relative_limit(left: pl.Expr, right: pl.Expr, profile: ToleranceProfile) -> pl.Expr:
    scale = pl.max_horizontal(left.abs(), right.abs())
    return pl.max_horizontal(
        pl.lit(profile.absolute_floor + _FLOAT_EPSILON),
        pl.lit(profile.relative_tolerance) * scale,
    )


__all__ = [
    "BPS_TOLERANCE",
    "DUST_TOLERANCE",
    "EXACT_INT_CENTS",
    "ToleranceMode",
    "ToleranceProfile",
    "at_least",
    "at_most",
    "scalar_within_tolerance",
    "within_tolerance",
]
