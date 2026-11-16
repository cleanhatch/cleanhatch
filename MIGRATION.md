# L0/L1 package migration (Cleanhatch 0.1.x split)

This release physically separates the Apache-2.0 **L0 kernel** (`cleanhatch`)
from the proprietary **L1 Pro Suite** (`cleanhatch-l1`).

## What changed

| Removed from the `cleanhatch` distribution | New home |
| --- | --- |
| `cleanhatch.license` | `cleanhatch_l1.license` |
| `cleanhatch.license_client` | `cleanhatch_l1.license_client` |
| `cleanhatch.report` | `cleanhatch_l1.report` |
| `cleanhatch.bundle` | `cleanhatch_l1.bundle` |
| `cleanhatch.packs.market_bars_pit` | `cleanhatch_l1.packs.market_bars_pit` |
| `cleanhatch.packs.sec_fundamentals_pit` | `cleanhatch_l1.packs.sec_fundamentals_pit` |
| `cleanhatch.packs.ai_rag_ledger_guard` | `cleanhatch_l1.packs.ai_rag_ledger_guard` |
| `cleanhatch validate --report-html/--report-pdf` | `cleanhatch-l1 validate ... --report-html/--report-pdf` |
| `cleanhatch diff --report-html/--report-pdf` | `cleanhatch-l1 diff ... --report-html/--report-pdf` |
| `cleanhatch verify-report` | `cleanhatch-l1 verify-report` |
| `cleanhatch doctor` license-server section | `cleanhatch-l1 license health` |

The L0 wheel intentionally ships **no forwarding stubs**. An old import now
fails with the standard interpreter error:

```text
ModuleNotFoundError: No module named 'cleanhatch.report'
```

That is the migration signal, not a data or license failure. Install the private
L1 distribution and update the import:

```powershell
python -m pip install cleanhatch
python -m pip install cleanhatch-l1        # private distribution / private index
```

```python
# Before
from cleanhatch.report import AuditReportBundle

# After
from cleanhatch_l1.report import AuditReportBundle
```

## CLI migration

The free CLI keeps the core commands only:

```text
doctor  validate  diff  demo  canary  init-recipe
```

The L1 CLI adds report and licensing commands:

```powershell
cleanhatch-l1 validate bars.parquet --pack market-bars-pit `
  --report-html audit.html --report-pdf audit.pdf
cleanhatch-l1 diff before.parquet after.parquet --report-html diff.html
cleanhatch-l1 verify-report audit.html
cleanhatch-l1 license health
cleanhatch-l1 license verify-cert license.cert
```

The three L1 packs register through the public `cleanhatch.packs` entry-point
group. With L1 installed, the core CLI can discover and run them:

```powershell
cleanhatch canary --pack market-bars-pit
cleanhatch validate bars.parquet --pack market-bars-pit
cleanhatch init-recipe market-bars-pit --out ./gates --agents-md
```

An L0-only environment lists only `temporal-core`; selecting an L1 pack raises
`PackNotInstalledError` with install guidance. Listing pack names reads
distribution metadata only and does not import L1 implementation modules.

## Evidence boundary (unchanged)

This split is a packaging and entry-point change. It does **not** add:

- a permanent purchase grant;
- a CMS/PAdES-signed PDF;
- Marimo Diff/What-if;
- per-order refund/dispute isolation;
- a new trusted-execution claim.

The current `license.cert` remains the legacy 14-day Ed25519 offline certificate.
The HTML SHA-256 check remains a content-consistency check, not a digital
signature and not a tamper-proof proof. Commercial documents remain marked
**DRAFT** until the external legal/MoR review is complete.

## Prior Apache-2.0 rights

Versions previously published under Apache-2.0 remain usable, modifiable, and
redistributable under that license. Removing L1 files from a **new** L0 release
does not revoke rights already granted for older Apache-2.0 versions. The L0
kernel remains Apache-2.0; its `LICENSE` file is unchanged. The L1 distribution
is proprietary and is not relicensed as Apache-2.0.

## Verification

Public L0 checks (run from the repository root):

```powershell
python -m pytest tests -q
python -m build . --outdir dist-l0
python scripts/check_dist.py l0 dist-l0
python -m twine check dist-l0/*
```

`twine` is a release-environment tool, not a runtime or dev dependency. The
module ownership ADR and private L1 test suite are maintained in the private
engineering repository.

## L0 source distribution

The L0 sdist includes the public contribution materials required by its own
test suite: `CONTRIBUTING.md`, `README_PUBLIC.md`, the public PR/issue
templates, and `.githooks/pre-commit`. These are contribution materials, not
commercial contracts. The sdist still excludes the private L1 suite, benchmarks,
the commercial contracts, and private CI workflows, so `pytest tests` can run
from the unpacked sdist without importing L1.
