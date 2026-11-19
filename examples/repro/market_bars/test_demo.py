"""Offline acceptance checks for the public market-bars example."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from cleanhatch.packs import GatePack, PackNotInstalledError, get_pack
from examples.repro.market_bars.demo import load_market_bars, run_l0, run_l1
from examples.repro.market_bars.download import _checksum
from examples.repro.market_bars.fault_matrix import (
    build_fault_scenarios,
    evaluate_fault_scenario,
)

_DATA_DIR = Path(__file__).with_name("data")
_ARCHIVE = _DATA_DIR / "BTCUSDT-1h-2024-01.zip"
_CHECKSUM = _DATA_DIR / "BTCUSDT-1h-2024-01.zip.CHECKSUM"


def test_bundled_binance_sample_matches_official_sha256() -> None:
    expected = _checksum(_CHECKSUM.read_text(encoding="ascii"))
    actual = hashlib.sha256(_ARCHIVE.read_bytes()).hexdigest()

    assert actual == expected


def test_l0_real_sample_passes_and_all_injected_failures_block() -> None:
    bars = load_market_bars(_ARCHIVE)

    clean, negative_close, pit_violation, after_decision, non_finite_close = run_l0(bars)
    negative_close_failures = {
        check.name for check in negative_close.checks if check.status == "FAILED"
    }

    assert bars.height == 744
    assert clean.status == "PASSED"
    assert negative_close.status == "FAILED"
    assert negative_close_failures == {"POSITIVE_FINITE_CLOSE"}
    assert pit_violation.status == "FAILED"
    failed = {check.name for check in pit_violation.checks if check.status == "FAILED"}
    assert failed == {"knowledge_time_causality"}
    after_decision_failures = {
        check.name for check in after_decision.checks if check.status == "FAILED"
    }
    assert "AVAILABILITY_BEFORE_DECISION" in after_decision_failures
    non_finite_failures = {
        check.name for check in non_finite_close.checks if check.status == "FAILED"
    }
    assert non_finite_failures == {"POSITIVE_FINITE_CLOSE"}


def test_l1_fault_matrix_covers_all_twelve_market_bar_checks() -> None:
    bars = load_market_bars(_ARCHIVE)
    scenarios = build_fault_scenarios(bars)

    assert len(scenarios) == 12
    assert {scenario.check_name for scenario in scenarios} == {
        "OHLC_HIGH_GE_OPEN_CLOSE",
        "OHLC_LOW_LE_OPEN_CLOSE",
        "OHLC_HIGH_GE_LOW",
        "POSITIVE_FINITE_PRICES",
        "NON_NEGATIVE_FINITE_VOLUMES",
        "HALT_ZERO_AMPLITUDE",
        "VWAP_ENVELOPE",
        "BAR_CLOSE_CAUSALITY",
        "UNIQUE_BAR_COORDINATE",
        "WARMUP_WINDOW_SKIP",
        "ACTIVE_DEAD_VALUE_RUN",
        "ROLLING_MAD_SPIKE",
    }
    assert all(scenario.construction for scenario in scenarios)


@pytest.mark.parametrize(
    "check_name",
    (
        "OHLC_HIGH_GE_OPEN_CLOSE",
        "OHLC_LOW_LE_OPEN_CLOSE",
        "OHLC_HIGH_GE_LOW",
        "POSITIVE_FINITE_PRICES",
        "NON_NEGATIVE_FINITE_VOLUMES",
        "HALT_ZERO_AMPLITUDE",
        "VWAP_ENVELOPE",
        "BAR_CLOSE_CAUSALITY",
        "UNIQUE_BAR_COORDINATE",
        "WARMUP_WINDOW_SKIP",
        "ACTIVE_DEAD_VALUE_RUN",
        "ROLLING_MAD_SPIKE",
    ),
)
def test_l1_fault_case_observes_expected_check_status(check_name: str) -> None:
    bars = load_market_bars(_ARCHIVE)
    scenario = next(
        scenario for scenario in build_fault_scenarios(bars) if scenario.check_name == check_name
    )
    try:
        pack: GatePack = get_pack("market-bars-pit")
    except PackNotInstalledError as error:
        pytest.skip(str(error))

    outcome = evaluate_fault_scenario(scenario, pack)

    assert outcome.actual_check_status == scenario.expected_check_status
    assert outcome.passed, outcome.to_dict()
    if check_name == "WARMUP_WINDOW_SKIP":
        assert outcome.target_details["skipped_rows"] == 10
        assert outcome.target_details["effective_coverage"] == 0.0
        assert outcome.gate_verdict == "PASSED"
    if check_name == "ACTIVE_DEAD_VALUE_RUN":
        assert outcome.gate_verdict == "CONTINUE"
    if check_name == "ROLLING_MAD_SPIKE":
        assert outcome.matched_injected_spike is True


def test_l1_real_sample_warns_and_both_injected_failures_block() -> None:
    bars = load_market_bars(_ARCHIVE)
    try:
        clean, negative_close, pit_violation = run_l1(bars)
    except PackNotInstalledError as error:
        pytest.skip(str(error))
    clean_warnings = {check.name for check in clean.checks if check.status == "WARNING"}
    negative_close_failures = {
        check.name for check in negative_close.checks if check.status == "FAILED"
    }
    pit_failures = {check.name for check in pit_violation.checks if check.status == "FAILED"}

    assert clean.status == "WARNING"
    assert clean_warnings == {"ROLLING_MAD_SPIKE"}
    assert negative_close.status == "FAILED"
    assert "POSITIVE_FINITE_PRICES" in negative_close_failures
    assert pit_violation.status == "FAILED"
    assert pit_failures == {"BAR_CLOSE_CAUSALITY"}
