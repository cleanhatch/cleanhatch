# Cleanhatch

**The cleanroom airlock for AI and temporal data pipelines.**

[![PyPI](https://img.shields.io/pypi/v/cleanhatch.svg)](https://pypi.org/project/cleanhatch/)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-3776AB?logo=python&logoColor=white)](https://pypi.org/project/cleanhatch/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Architecture](https://img.shields.io/badge/architecture-zero--server%20%2F%20local--first-10B981.svg)](#design-notes)

```text
  (Ingest / Revisions)
          |
          v
  +----------------------------------------------------------+
  |  CLEANROOM AIRLOCK                                       |
  |  [ TemporalGate | CausalGate | InvarGate ]               |
  +----------------------------------------------------------+
          |
          v
  LineageManifest (<= 2 KB JSON)
  + optional L1 report (HTML / PDF)
```

Cleanhatch is a small, domain-neutral guard kernel for tables that must respect
time. It gives you point-in-time reads, lookahead-leak probes, lifecycle
invariants, and compact lineage evidence without a server, a metadata database,
or a framework-specific schema.

Most temporal data failures are silent: the pipeline finishes green while a
backtest uses a restatement that did not exist yet, a feature is normalized over
the full sample, a market halt is counted as dead data, or a RAG chunk is
retrieved from a document edited six months after the decision date. Cleanhatch
turns those failures into typed, testable evidence at the exact boundary where
they enter the system.

## 30-second quickstart: catch 5 silent killers

```bash
pip install cleanhatch
cleanhatch demo
```

The demo builds five realistic panels in memory and measures the gate that owns
each failure mode. No network, files, or fixtures are required. Output from a
local Windows 11 / Python 3.14 machine:

```text
## cleanhatch demo: 5 silent killers, caught in milliseconds

| # | Scenario | Silent killer | Gate that caught it | Verdict | Time | AI summary |
|---|----------|---------------|---------------------|---------|------|------------|
| 1 | `quant_full_sample_zscore` | Full-sample z-score normalizes with observations that did not exist yet | CausalGate (future_perturbation) | **CAUGHT** | 151.6 ms | 468 B |
| 2 | `sec_restatement_time_travel` | A later restatement rewrites the revenue visible in a historical backtest | TemporalGate.read_as_of + diff_versions | **CAUGHT** | 25.6 ms | 340 B |
| 3 | `crypto_halt_and_flash_spike` | Halt bars look dead and one flash spike masks a real regime shift | InvarGate (masked dead values + robust MAD) | **CAUGHT** | 16.5 ms | 353 B |
| 4 | `survivorship_bias_filter` | Universe keeps only entities that survive to the end of the sample | CausalGate (availability_truncation) | **CAUGHT** | 51.2 ms | 473 B |
| 5 | `ai_rag_temporal_contamination` | A post-hoc edited document leaks 2026 spoilers into a 2025 decision | TemporalGate + InvarGate + ai_summary.json | **CAUGHT** | 6.9 ms | 475 B |

**5/5 silent killers caught in 251.8 ms.** Largest AI payload: 475 B / 2048 B.
```

Use `cleanhatch demo --json` for structured verdicts and evidence.

## Why Cleanhatch? (Architecture & Benchmark Matrix)

| Capability | Ad-hoc Polars/Pandas scripts | Heavy data-quality suites (GX / Soda) | Cleanhatch (L0 kernel + L1 packs) |
| --- | --- | --- | --- |
| Bitemporal as-of & revision diffs | Hand-rolled joins per project; revision ordering bugs surface late | Row/table expectations, not bitemporal semantics | `TemporalGate` resolves the latest visible revision and diffs two as-of states |
| Lookahead-leak probes (`CausalGate`) | None unless you build a custom harness | Monitors data, not transform causality | Re-runs the transform with future rows removed and future values perturbed |
| Inactive/halt-aware rolling MAD (`InvarGate`) | Manual `when(...)` masks drift across every rule | Static thresholds turn halts into false positives or silent `fillna` | `skip_when` masks halt and warm-up rows; rolling MAD resists one flash spike |
| Infrastructure footprint | Notebook + cron + warehouse | Server/cloud runner, metadata store, orchestration | Zero server, in-process Polars; L1 adds a single-file HTML/PDF report |
| Agent repair payload (<= 2,048 B) | Ad-hoc error strings with unbounded context | Dashboard links and long JSON payloads | Bounded `ai_summary.json` with structured, repair-oriented failures |

### Measured 1,000,000-row benchmark

Measured with the project's reproducible benchmark harness. The harness is not
part of the public L0 distribution, so the figures below are reported
measurements rather than a command that ships with this repository.

**Dataset:** 1,000,000 rows x 10 columns, 100 entities, 100,000 revision rows,
900,000 unique point-in-time keys. **Environment:** Windows 11, Python 3.14.6,
Polars 2.1.0, DuckDB 2.0.0, cleanhatch 0.1.0. Median of 3 runs.

| Operation | Workload | Median |
| --- | --- | ---: |
| `TemporalGate.read_as_of` | Polars in-memory, 900,000 unique keys | 144.4 ms |
| `TemporalGate.read_as_of` | DuckDB Parquet + QUALIFY, 900,000 unique keys | 337.3 ms |
| `content_hash` | 1,000,000 rows x 10 columns = 10,000,000 cells | 274.5 ms |
| `InvarGate.check_clean` | hourly grid + masked dead values + robust MAD drift | 429.1 ms |
| `CausalGate.run` | availability truncation + future perturbation, adaptive window | 224.6 ms |
| `LineageManifest.to_ai_summary_json` | bounded agent payload (<=2048 bytes) | 0.1 ms |

Latency includes Python and serialization overhead and is hardware-dependent.
Rerun the script on your machine before quoting these numbers.

## The Four Core Gates (L0 Open Kernel)

All four gates use the same domain-neutral coordinates:

```text
(entity, event_time, knowledge_time, version_tag)
```

Column names come from `TemporalSpec`; no market-specific field is hard-coded.

### `TemporalGate`: point-in-time reads and restatements

```python
from datetime import UTC, datetime

from cleanhatch import TemporalGate, TemporalSpec

spec = TemporalSpec(
    entity_col="entity_id",
    event_time_col="event_time",
    knowledge_time_col="available_at",
)
gate = TemporalGate("filings.parquet", spec=spec)

before = gate.read_as_of(as_of=datetime(2026, 5, 15, tzinfo=UTC))
after = gate.read_as_of(as_of=datetime(2026, 8, 15, tzinfo=UTC))
```

When several revisions of the same `(event_time, entity_id)` are visible at the
cutoff, `read_as_of` keeps the latest knowledge time. Pass
`deduplicate=False` to inspect every historical revision.

`diff_versions` resolves the same latest revision even when no watermark is
given, so `cleanhatch diff a.parquet b.parquet` compares clean point-in-time
states instead of raising a duplicate-key error on revisioned source files:

```python
report = gate.diff_versions(frame_a, frame_b)
print(report.changed_cell_ratio, report.columns[0].max_abs_change)
```

### `CausalGate`: lookahead probes

```python
import polars as pl

from cleanhatch import CausalGate, TemporalSpec

spec = TemporalSpec(
    event_time_col="event_time",
    knowledge_time_col="available_at",
    signed_numeric_cols=("value",),
)


def rolling_feature(frame: pl.LazyFrame) -> pl.LazyFrame:
    return frame.select(
        pl.col("available_at").alias("decision_time"),
        "entity_id",
        pl.col("value").rolling_mean(3).over("entity_id").alias("factor"),
    )


result = CausalGate(spec, value_col="factor").run(rolling_feature, frame)
```

The gate re-runs the transform with future rows removed and with future numeric
values or boolean masks altered. A non-empty cross-section that differs is a
failure; failures carry `rank_flip_count` and `max_abs_diff` evidence. An empty
cross-section never passes.

### `InvarGate`: lifecycle invariants with four-state verdicts

```python
from cleanhatch import InvarGate, InvarRule, Stage

gate = InvarGate(spec)
report = gate.run(frame, stage=Stage.CLEAN)

positive_price = InvarRule(
    name="positive_price",
    predicate=pl.col("close") > 0,
    message="close must be positive",
)
missing_close_review = InvarRule(
    name="missing_close_review",
    predicate=pl.col("close").is_not_null(),
    message="close is missing outside a halt",
    severity="WARN",
    skip_when=~pl.col("is_trading"),
)
feature_report = gate.run(
    features,
    stage=Stage.FEATURE,
    rules=(positive_price, missing_close_review),
)
```

Gate results are four-state:

| Status | Meaning |
| --- | --- |
| `PASSED` | The check ran and found no violation |
| `WARNING` | The check found a suspicious but non-blocking change |
| `FAILED` | The check found a blocking violation |
| `SKIPPED` | There were no applicable samples, for example an empty frame or warm-up period |

A skipped check is reported as `SKIPPED` rather than `PASSED`. It does not block
the pipeline, but it is not evidence that the check succeeded. Set
`severity="WARN"` on a rule to keep a suspicious finding visible without failing
the gate. Use `skip_when` to exempt inactive rows while preserving accurate
coverage accounting.

Market halts and other inactive rows can be exempted with a boolean mask, while
robust MAD drift resists a single flash spike:

```python
spec = TemporalSpec(
    positive_numeric_cols=("close",),
    boolean_mask_cols=("is_trading",),
)
report = InvarGate(spec, max_zscore_drift=4.0).check_clean(
    bars,
    baseline=previous_window,
    grid_freq_ms=3_600_000,
)
```

### `LineageManifest`: bounded agent evidence

```python
from pathlib import Path

from cleanhatch import LineageManifest, TransformSignature

manifest = LineageManifest.capture(
    inputs=[snapshot],
    transform=TransformSignature.from_source(source_code),
    gates=[report],
)
Path("ai_summary.json").write_text(manifest.to_ai_summary_json(), encoding="utf-8")
```

`to_ai_summary_json()` returns compact UTF-8 JSON that is always at most 2,048
bytes. A failed gate exposes its name, stage, and bounded message; the agent can
repair the pipeline, rerun, and compare the new manifest with
`LineageManifest.ablation_diff()` instead of guessing what changed. Large
manifests are progressively compacted rather than flooding the context window.

## Domain Gate Packs & Canary Poisoning Suites (L0 Protocol + L1 Pro Packs)

The L0 open kernel ships the `GatePack` protocol and the free `temporal-core`
pack. The three vertical packs are distributed in the separate private
`cleanhatch-l1` package (or loaded from third-party Python entry points). Every
pack keeps the same interface:

```python
from cleanhatch import available_packs, get_pack

print(available_packs())
report = get_pack("temporal-core").validate(frame)
```

| Pack | Tier | Rule count | Target silent killers | Canary coverage |
| --- | --- | ---: | --- | --- |
| `temporal-core` | L0 Built-in | 6 | Naive timestamps, future knowledge time, conflicting revisions, mutated version tags, overlapping SCD2 intervals, multi-axis ablation | 6/6 poisoned panels caught |
| `market-bars-pit` | L1 Pro | 12 | Broken OHLC identities, non-positive prices, halt bars with fake amplitude, VWAP outside the envelope, duplicate bar coordinates, dead-value runs, flash spikes | 8/8 poisoned panels caught |
| `sec-fundamentals-pit` | L1 Pro | 11 | Same-day filing leakage, balance-sheet and gross-profit identities, 1000x unit jumps, currency switches, restatement time travel, TTM window construction | 11/11 poisoned panels caught |
| `ai-rag-ledger-guard` | L1 Pro | 8 | Order-total and line-item hallucinations, unit-price math, discount and tax bounds, multi-currency drift, mutable invoice versions, RAG post-cutoff contamination | 10/10 poisoned panels caught |

Rule counts are the full gate-check surface, including table-level checks and
declared `SKIPPED` warm-up exemptions. The canary number is the size of the
pack's executable poisoning suite; `cleanhatch canary` verifies that every
poisoned panel is intercepted. Proprietary L1 rule formulas are distributed only
in the private `cleanhatch-l1` package; this README documents their contract and
canary coverage, not their implementation.

```bash
pip install cleanhatch-l1          # private distribution / private index
cleanhatch canary
cleanhatch validate bars.parquet --pack market-bars-pit
cleanhatch init-recipe market-bars-pit --out ./gates/ --agents-md
```

`init-recipe` writes a declarative configuration template plus an `AGENTS.md`
guardrail file. It exposes column mappings, thresholds, and the canary entry
point without ejecting the operator source.

An L0-only installation still discovers packs registered elsewhere, but it does
not import their implementation just to list names:

```python
from cleanhatch import PackNotInstalledError, register_pack

try:
    pack = get_pack("market-bars-pit")
except PackNotInstalledError as error:
    print(error)  # points at the private cleanhatch-l1 distribution

register_pack("in-house-ledger", MyInHousePack)
```

External distributions can publish the same contract without importing private
code:

```toml
[project.entry-points."cleanhatch.packs"]
in-house-ledger = "my_company.packs:MyInHousePack"
```

## L1 Pro Single-File HTML & PDF Audit Reports

Report export is an L1 capability and lives in the private `cleanhatch-l1`
package; the L0 kernel exits before parsing `--report-*` flags.

```bash
cleanhatch-l1 validate bars.parquet --pack market-bars-pit \
  --report-html audit.html --report-pdf audit.pdf
cleanhatch-l1 diff before.parquet after.parquet --report-html diff.html
cleanhatch-l1 verify-report audit.html
```

The HTML report is one self-contained file: inline CSS, inline SVG, an embedded
canonical JSON evidence payload, and an offline WebCrypto verifier. The PDF uses
only PDF 1.4 primitives and standard fonts, so it needs no browser, external
asset, or network. `verify-report` recomputes the report digest and checks the
visible seal against the embedded payload; a modified file exits with code `2`.
This is a **content-consistency check, not a digital signature** and not a
tamper-proof proof.

| Build | Report seal | Intended use |
| --- | --- | --- |
| Community Evaluation Build | `COMMUNITY EVALUATION BUILD — UNLICENSED` badge and watermark | Local evaluation and sharing; every report carries the watermark |
| L1 Pro Compliance Suite | Offline Ed25519 `license.cert` seal with the organization badge and `CH-XXXX-XXXX` trace code | Team review, investment committee packs, and client delivery |

The current L1 certificate is the legacy 14-day offline lease certificate. It is
verified offline with no phone-home and no report-path network call. An expired
or revoked lease falls back to the unlicensed or revoked watermark instead of
silently producing a sealed report. A permanent purchase grant, CMS-signed PDF,
and Marimo Diff/What-if are planned but **not implemented** in this release.

## Contribute a Silent Killer ("Field Pitfalls")

Quant, data, and AI engineers: if you have lost hours to a pipeline that stayed
green while the data was wrong, submit the reproducible scenario through the
**[Silent Killer / Field Pitfall Issue Form](.github/ISSUE_TEMPLATE/silent_killer_pitfall.yml)**.

You choose the attribution. Contributors may be credited under a **GitHub
`@handle` or a custom `Pseudonym`** (for example `QuantAnon-42`) so an employer,
fund, or client is never exposed. Desensitized schema shapes and poison patterns
are enough; do not paste proprietary data.

Two recognition tiers keep the loop honest:

| Tier | What you submit | Recognition |
| --- | --- | --- |
| **Tier 1: Field Pitfall Scout** | A reproducible production pitfall via the Issue Form. We write the rule and add it to the appropriate pack. | Permanent attribution in `/silent-killers/` and release notes, plus a **20% OFF L1 Pro Suite coupon** |
| **Tier 2: Core Recipe Architect** | A merged PR with a vectorized rule and a canary poisoning test (capped at 2 per month). | Permanent attribution plus a complimentary **1-year L1 Contributor License (`Contributor Edition`)** |

Tier 1 turns field experience into coverage without requiring you to write
Polars. Tier 2 rewards engineers who ship the rule and the proof that it catches
the failure. Both tiers are attributed exactly as you request: handle,
pseudonym, or anonymous.

## Support Boundary

Cleanhatch L1 is a **Self-serve Digital Kit — Includes GitHub Issues for bug reports only.**

- **No 1-on-1 code debugging or Slack/Discord support.** A good report is a reproducible defect in the delivered code, gate packs, CLI, report verifier, or documentation; live debugging, private channels, and custom rule writing are out of scope.
- Custom data-source rules, pipeline integration, and private deployment are scoped as an **L3 Custom Airlock Setup** engagement at **$1,500** per fixed-scope delivery.
- Once the License Key is activated or the L1 bundle is downloaded, digital delivery is complete and no-reason refunds are not offered, subject to the terms shown at checkout and any non-waivable statutory rights; see the DRAFT policies below, which require legal and MoR review before going live.
- Read the binding documents before purchase: [Terms of Service](https://cleanhatch.dev/terms/), [Commercial License](https://cleanhatch.dev/terms/), [Privacy Policy](https://cleanhatch.dev/privacy/), and [Refund Policy](https://cleanhatch.dev/refund/). They are marked **DRAFT** pending legal and MoR platform review.

The L0 kernel remains Apache-2.0 and free to evaluate, share, and modify under
that license.

## CLI Reference

```bash
cleanhatch doctor
cleanhatch demo
cleanhatch demo --json
cleanhatch validate bars.parquet --stage clean --ai-summary ai_summary.json
cleanhatch validate bars.parquet --stage clean --no-drift-baseline
cleanhatch validate bars.parquet --stage clean --baseline-windows 30 --baseline-window-hours 24
cleanhatch validate bars.parquet --as-of 2026-01-02T00:00:00Z
cleanhatch validate bars.parquet --pack market-bars-pit
cleanhatch diff snapshot_a.parquet snapshot_b.parquet
cleanhatch canary
cleanhatch canary --pack market-bars-pit
cleanhatch init-recipe market-bars-pit --out ./gates/ --agents-md

cleanhatch-l1 validate bars.parquet --pack market-bars-pit \
  --report-html audit.html --report-pdf audit.pdf
cleanhatch-l1 diff snapshot_a.parquet snapshot_b.parquet --report-html diff.html
cleanhatch-l1 verify-report audit.html
cleanhatch-l1 license health
cleanhatch-l1 license verify-cert license.cert
```

`validate` exits with code `2` when a gate fails, so it can be used directly in
CI. At `--stage clean`, it builds a rolling PIT drift baseline from the previous
20 one-day windows by default; use `--no-drift-baseline` to opt out, or
`--baseline-windows` / `--baseline-window-hours` to change the window set. The
baseline reads each window at its own knowledge cutoff, so a later restatement
cannot rewrite an earlier baseline. `doctor` checks Python and dependency floors
and the local Git identity guard. Every failing check prints a one-line
remediation. `diff` compares latest visible revisions by default; add `--as-of`
to pin both sides to a knowledge watermark. Report flags and license health
belong to `cleanhatch-l1`; the L0 CLI does not parse them.

## Design Notes

- Inputs and outputs are described by `TemporalSpec`; no domain-specific column
  names are hard-coded.
- Unexpected states raise typed errors instead of silently returning defaults.
- Financial identities share three explicit tolerance tiers: integer cents
  (`exact_int_cents`), decimal dust (`dust_tolerance = 0.01`), and a relative
  envelope (`bps_tolerance = 0.5`, i.e. `5e-5`). Version identity checks remain
  exact by design.
- Version tags are immutable within a process: reusing a tag for different
  content raises `VersionMutationError`.
- `content_hash` is a deterministic, vectorized SHA-256 over the schema and
  Arrow data, independent of local timezone and Polars hash seeds.
- Every check returns structured evidence that can be attached to a lineage
  manifest and compressed for an agent context window.
- The L0 kernel is zero-server and local-first: gates, lineage manifests, and
  its optional AI summary all run in-process. L1 reports and offline license
  verification are separate proprietary add-ons.

## Development

```bash
python -m pip install -e ".[dev]"
python -m ruff check .
python -m ruff format --check .
python -m pyright
python -m pytest tests -q
```

CI runs lint, format, type checks, L0 tests, the demo, and an L0 sdist/wheel
build with a package-boundary assertion on Ubuntu and Windows across Python
3.11, 3.12, and 3.13. The private engineering repository additionally runs the
L1 package tests, site/corpus checks, and the benchmark harness; those paths are
not part of the public L0 repository.

## Publishing

Release distributions are built only on GitHub Actions, never on a developer
machine. The `Build and publish` workflow in `.github/workflows/publish.yml`
tests the L0 kernel on Ubuntu and Windows across Python 3.11, 3.12, and 3.13,
builds the wheel and source distribution, checks their metadata, and verifies
installation from both artifacts outside the source checkout.

A normal push to `main` or a pull request runs the test and build jobs only. The
PyPI publish job is restricted to a manual `workflow_dispatch` run on `main` and
uses PyPI OIDC trusted publishing; no API token is stored in the repository.

Before publishing 0.1.0:

1. Confirm the existing PyPI trusted publisher:

   | Field | Value |
   | --- | --- |
   | PyPI project | `cleanhatch` |
   | GitHub owner | `cleanhatch` |
   | GitHub repository | `cleanhatch` |
   | Workflow filename | `publish.yml` |
   | GitHub environment | `pypi` |

2. Confirm the GitHub Actions environment `pypi` matches the publisher.
3. Run `Build and publish` manually from `main`.
4. Verify the new files on PyPI and install them in a fresh environment.

Do not rerun a successful publication of the same version: PyPI does not permit
replacing uploaded files. For the next release, update `cleanhatch/__about__.py`
and the corresponding tests and documentation.

The GitHub repository's visibility is independent of PyPI: publishing makes the
wheel, source distribution, metadata, and README publicly downloadable.

## License

Copyright 2026 Atelier Null.

The L0 kernel is Apache 2.0. See [LICENSE](LICENSE). The separate
`cleanhatch-l1` distribution is proprietary and its commercial terms remain
marked DRAFT; see [MIGRATION.md](MIGRATION.md) for the split and prior-version
rights.
