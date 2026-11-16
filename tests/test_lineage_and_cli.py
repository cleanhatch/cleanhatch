import json
import warnings
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from cleanhatch import (
    GateEvidence,
    InputSnapshot,
    LineageManifest,
    NonIsolatedAblationWarning,
    TransformSignature,
)
from cleanhatch.cli import main

BASE = datetime(2026, 1, 1, tzinfo=UTC)


def _snapshot(index: int) -> InputSnapshot:
    return InputSnapshot(
        dataset_id=f"dataset-{index}",
        version_tag=f"v{index}",
        content_hash=f"{index:064x}",
        as_of=BASE,
        row_count=index,
    )


def _manifest(
    *,
    inputs: tuple[InputSnapshot, ...] = (),
    gates: tuple[GateEvidence, ...] = (),
    metrics: dict[str, float] | None = None,
    source: str = "value = 1",
) -> LineageManifest:
    return LineageManifest.capture(
        inputs=inputs,
        transform=TransformSignature.from_source(source),
        gates=gates,
        metrics=metrics,
    )


def test_non_isolated_ablation_warns() -> None:
    base = _manifest(inputs=(_snapshot(1),), source="value = 1")
    candidate = _manifest(inputs=(_snapshot(2),), source="value = 2")

    with pytest.warns(NonIsolatedAblationWarning):
        report = LineageManifest.ablation_diff(base, candidate)

    assert report.changed_axes == ("dataset", "transform")
    assert report.is_clean_ablation is False


def test_ai_summary_is_bounded_under_extreme_pressure() -> None:
    inputs = tuple(_snapshot(index) for index in range(50))
    checks = tuple(
        {
            "name": f"check_{index}",
            "stage": "clean",
            "status": "FAILED",
            "message": "x" * 500,
        }
        for index in range(100)
    )
    gates = (GateEvidence(stage="clean", status="FAILED", checks=checks),)
    manifest = _manifest(
        inputs=inputs,
        gates=gates,
        metrics={f"metric_{index}": float(index) for index in range(50)},
    )

    encoded = manifest.to_ai_summary_json()
    payload = json.loads(encoded)

    assert len(encoded.encode("utf-8")) <= 2048
    assert payload["truncated"] is True
    assert payload["status"] == "FAILED"
    assert payload["inputs_omitted"] == 48
    assert len(payload["inputs"]) == 2
    assert len(payload["failed_gates"]) == 1
    assert len(payload["metrics"]) == 3
    assert payload["byte_length"] == len(encoded.encode("utf-8"))


def test_small_ai_summary_is_not_marked_truncated() -> None:
    manifest = _manifest(inputs=(_snapshot(1),))

    payload = manifest.to_ai_summary()

    assert payload["truncated"] is False
    assert "byte_length" not in payload
    assert payload["status"] == "UNKNOWN"


def _bars() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "entity_id": ["A", "A", "A"],
            "event_time": [BASE, BASE + timedelta(hours=1), BASE + timedelta(hours=2)],
            "available_at": [BASE, BASE + timedelta(hours=1), BASE + timedelta(hours=2)],
            "value": [1.0, 2.0, 3.0],
        }
    )


def test_cli_doctor_outputs_json(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)

    exit_code = main(["doctor"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["python"]["supported"] is True
    assert "polars" in payload["packages"]
    assert "duckdb" in payload["packages"]
    assert payload["git_opsec"]["repository"] is False


def test_cli_validate_passes_and_writes_ai_summary(tmp_path, capsys) -> None:
    source = tmp_path / "bars.parquet"
    summary = tmp_path / "ai_summary.json"
    _bars().write_parquet(source)

    exit_code = main(
        [
            "validate",
            str(source),
            "--stage",
            "clean",
            "--ai-summary",
            str(summary),
        ]
    )
    report = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert report["status"] == "PASSED"
    encoded = summary.read_text(encoding="utf-8")
    assert len(encoded.encode("utf-8")) <= 2048
    summary_payload = json.loads(encoded)
    assert summary_payload["status"] == "PASSED"


def test_cli_validate_returns_two_on_gate_failure(tmp_path, capsys) -> None:
    source = tmp_path / "bad.parquet"
    bad = _bars().with_columns(
        (pl.col("available_at") - pl.duration(hours=2)).alias("available_at")
    )
    bad.write_parquet(source)

    exit_code = main(["validate", str(source), "--stage", "ingest"])
    report = json.loads(capsys.readouterr().out)

    assert exit_code == 2
    assert report["status"] == "FAILED"


def test_cli_validate_as_of_resolves_latest_revision(tmp_path, capsys) -> None:
    source = tmp_path / "revisions.parquet"
    pl.DataFrame(
        {
            "entity_id": ["A", "A"],
            "event_time": [BASE, BASE],
            "available_at": [BASE + timedelta(hours=1), BASE + timedelta(hours=2)],
            "value": [100.0, 120.0],
        }
    ).write_parquet(source)

    exit_code = main(
        [
            "validate",
            str(source),
            "--as-of",
            (BASE + timedelta(hours=3)).isoformat(),
            "--stage",
            "ingest",
        ]
    )
    report = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert report["status"] == "PASSED"


def test_cli_validate_feature_accepts_custom_decision_time(tmp_path, capsys) -> None:
    source = tmp_path / "features.parquet"
    pl.DataFrame(
        {
            "dt": [BASE, BASE + timedelta(hours=1)],
            "entity_id": ["A", "A"],
            "value": [1.0, 2.0],
        }
    ).write_parquet(source)

    exit_code = main(
        [
            "validate",
            str(source),
            "--stage",
            "feature",
            "--decision-time-col",
            "dt",
        ]
    )
    report = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert report["status"] == "PASSED"
    assert report["stage"] == "feature"


def test_cli_rejects_conflicting_coordinate_columns(tmp_path, capsys) -> None:
    source = tmp_path / "bars.parquet"
    _bars().write_parquet(source)

    exit_code = main(
        [
            "validate",
            str(source),
            "--decision-time-col",
            "event_time",
        ]
    )
    payload = json.loads(capsys.readouterr().err)

    assert exit_code == 1
    assert payload["error_type"] == "ValidationError"


def test_cli_diff_reports_changed_cells(tmp_path, capsys) -> None:
    source_a = tmp_path / "a.parquet"
    source_b = tmp_path / "b.parquet"
    _bars().write_parquet(source_a)
    _bars().with_columns((pl.col("value") + 1.0).alias("value")).write_parquet(source_b)

    exit_code = main(["diff", str(source_a), str(source_b)])
    report = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert report["rows_a"] == 3
    assert report["rows_b"] == 3
    assert report["common_rows"] == 3
    assert report["changed_cell_ratio"] > 0.0
    assert report["columns"][0]["changed_rows"] == 3


def test_cli_diff_without_as_of_deduplicates_revisions(tmp_path, capsys) -> None:
    source_a = tmp_path / "revised_a.parquet"
    source_b = tmp_path / "revised_b.parquet"
    revisioned = pl.DataFrame(
        {
            "entity_id": ["A", "A"],
            "event_time": [BASE, BASE],
            "available_at": [BASE + timedelta(hours=1), BASE + timedelta(hours=2)],
            "value": [100.0, 120.0],
        }
    )
    revisioned.write_parquet(source_a)
    pl.concat(
        [
            revisioned,
            pl.DataFrame(
                {
                    "entity_id": ["A"],
                    "event_time": [BASE],
                    "available_at": [BASE + timedelta(hours=3)],
                    "value": [130.0],
                }
            ),
        ]
    ).write_parquet(source_b)

    exit_code = main(["diff", str(source_a), str(source_b)])
    report = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert report["as_of"] is None
    assert report["rows_a"] == 1
    assert report["rows_b"] == 1
    assert report["common_rows"] == 1
    assert report["columns"][0]["changed_rows"] == 1
    assert report["columns"][0]["max_abs_change"] == 10.0


def test_warning_category_is_a_user_warning() -> None:
    assert issubclass(NonIsolatedAblationWarning, Warning)
    with warnings.catch_warnings(record=True) as caught:
        warnings.warn("x", NonIsolatedAblationWarning, stacklevel=2)
    assert caught[0].category is NonIsolatedAblationWarning
