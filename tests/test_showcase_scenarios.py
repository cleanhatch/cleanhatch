from __future__ import annotations

import json

from cleanhatch.cli import main
from cleanhatch.demo import run_demo_suite

EXPECTED_NAMES = [
    "quant_full_sample_zscore",
    "sec_restatement_time_travel",
    "crypto_halt_and_flash_spike",
    "survivorship_bias_filter",
    "ai_rag_temporal_contamination",
]


def _scenarios_by_name() -> dict[str, dict[str, object]]:
    result = run_demo_suite()
    return {scenario.name: scenario.to_dict() for scenario in result.scenarios}


def test_demo_suite_catches_all_five_scenarios() -> None:
    result = run_demo_suite()

    assert [scenario.name for scenario in result.scenarios] == EXPECTED_NAMES
    assert result.all_caught is True
    assert len(result.scenarios) == 5
    assert result.total_elapsed_ms >= 0.0
    assert result.max_ai_summary_bytes <= 2048
    for scenario in result.scenarios:
        assert scenario.verdict == "CAUGHT"
        assert scenario.elapsed_ms >= 0.0
        assert 0 < scenario.ai_summary_bytes <= 2048


def test_quant_full_sample_zscore_is_caught_by_future_perturbation() -> None:
    scenario = _scenarios_by_name()["quant_full_sample_zscore"]
    evidence = scenario["evidence"]

    assert isinstance(evidence, dict)
    assert "future_perturbation" in evidence["caught_by"]
    assert evidence["max_abs_diff"] > 0.0
    assert evidence["safe_twin"] == "PASSED"


def test_sec_restatement_recovers_both_point_in_time_views() -> None:
    scenario = _scenarios_by_name()["sec_restatement_time_travel"]
    evidence = scenario["evidence"]

    assert isinstance(evidence, dict)
    assert evidence["as_of_2026_05_15_usd_m"] == 100.0
    assert evidence["as_of_2026_08_15_usd_m"] == 62.0
    assert evidence["diff_changed_rows"] == 1
    assert evidence["diff_max_abs_change_usd_m"] == 38.0


def test_crypto_halt_mask_and_robust_mad_avoid_false_positives() -> None:
    scenario = _scenarios_by_name()["crypto_halt_and_flash_spike"]
    evidence = scenario["evidence"]

    assert isinstance(evidence, dict)
    assert evidence["masked_status"] == "PASSED"
    assert evidence["unmasked_status"] == "FAILED"
    assert evidence["drift_statistic"] == "mad"
    assert evidence["regime_status"] == "WARNING"


def test_survivorship_filter_is_caught_by_availability_truncation() -> None:
    scenario = _scenarios_by_name()["survivorship_bias_filter"]
    evidence = scenario["evidence"]

    assert isinstance(evidence, dict)
    assert evidence["caught_by"] == ["availability_truncation"]
    assert evidence["mismatched_entities"] > 0


def test_rag_contamination_blocks_post_hoc_chunk() -> None:
    scenario = _scenarios_by_name()["ai_rag_temporal_contamination"]
    evidence = scenario["evidence"]

    assert isinstance(evidence, dict)
    assert evidence["visible_chunks"] == ["doc-clean"]
    assert evidence["blocked_chunks"] == 1
    assert evidence["ingest_status"] == "FAILED"
    assert scenario["ai_summary_bytes"] <= 2048


def test_cli_demo_prints_markdown_table(capsys) -> None:
    exit_code = main(["demo"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "cleanhatch demo" in output
    for name in EXPECTED_NAMES:
        assert name in output
    assert output.count("**CAUGHT**") == 5
    assert "AI summary" in output


def test_cli_demo_json_is_structured_and_bounded(capsys) -> None:
    exit_code = main(["demo", "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["suite"] == "cleanhatch demo"
    assert payload["all_caught"] is True
    assert payload["max_ai_summary_bytes"] <= 2048
    assert [scenario["name"] for scenario in payload["scenarios"]] == EXPECTED_NAMES
    assert all(scenario["verdict"] == "CAUGHT" for scenario in payload["scenarios"])
    assert all(scenario["ai_summary_bytes"] <= 2048 for scenario in payload["scenarios"])
