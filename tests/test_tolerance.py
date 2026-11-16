"""Independent tests for the three shared financial tolerance tiers."""

from __future__ import annotations

import polars as pl
import pytest

from cleanhatch.packs.temporal_core import TemporalCorePack
from cleanhatch.tolerance import (
    BPS_TOLERANCE,
    DUST_TOLERANCE,
    ToleranceProfile,
    at_least,
    at_most,
    scalar_within_tolerance,
    within_tolerance,
)


def _evaluate(left: list[float], right: list[float], profile: ToleranceProfile) -> list[bool]:
    frame = pl.DataFrame({"left": left, "right": right})
    return (
        frame.select(within_tolerance(pl.col("left"), pl.col("right"), profile=profile).alias("ok"))
        .get_column("ok")
        .to_list()
    )


def test_exact_int_cents_requires_exact_integer_cents() -> None:
    profile = ToleranceProfile(mode="exact_int_cents")

    assert _evaluate([100.0, 100.0, 100.0], [100.0, 101.0, 100.4], profile) == [
        True,
        False,
        False,
    ]
    integer_cents = pl.DataFrame({"left": [100, 100], "right": [100, 101]})
    assert integer_cents.select(
        within_tolerance(pl.col("left"), pl.col("right"), profile=profile).alias("ok")
    )["ok"].to_list() == [True, False]


def test_dust_tolerance_accepts_the_cent_boundary() -> None:
    profile = ToleranceProfile(mode="dust", dust_tolerance=DUST_TOLERANCE)

    assert _evaluate([100.0, 100.0], [100.01, 100.0101], profile) == [True, False]


def test_bps_tolerance_accepts_half_a_basis_point() -> None:
    profile = ToleranceProfile(mode="bps", bps_tolerance=BPS_TOLERANCE)

    assert profile.relative_tolerance == pytest.approx(5e-5)
    assert _evaluate(
        [1_000_000.0, 1_000_000.0],
        [1_000_050.0, 1_000_051.0],
        profile,
    ) == [True, False]


def test_bps_absolute_floor_keeps_small_values_usable() -> None:
    profile = ToleranceProfile(mode="bps", bps_tolerance=0.5, absolute_floor=0.01)

    assert _evaluate([1.0, 1.0], [1.01, 1.02], profile) == [True, False]


def test_ordering_helpers_share_the_same_tiers() -> None:
    dust = ToleranceProfile(mode="dust", dust_tolerance=0.01)
    frame = pl.DataFrame({"left": [10.0, 10.0], "right": [10.005, 10.02]})

    at_least_values = frame.select(
        at_least(pl.col("left"), pl.col("right"), profile=dust).alias("ok")
    )["ok"].to_list()
    at_most_values = frame.select(
        at_most(pl.col("right"), pl.col("left"), profile=dust).alias("ok")
    )["ok"].to_list()

    assert at_least_values == [True, False]
    assert at_most_values == [True, False]


def test_scalar_helper_handles_null_pairing() -> None:
    profile = ToleranceProfile(mode="dust", dust_tolerance=0.01)

    assert scalar_within_tolerance(None, None, profile=profile) is True
    assert scalar_within_tolerance(None, 1.0, profile=profile) is False
    assert scalar_within_tolerance(1.0, 1.01, profile=profile) is True
    assert scalar_within_tolerance(1.0, 1.02, profile=profile) is False


def test_temporal_core_pack_exposes_the_shared_tolerance_profile() -> None:
    assert TemporalCorePack().tolerance.mode == "exact_int_cents"
