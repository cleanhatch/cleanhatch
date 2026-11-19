# Reproduce the market-bars gate results

This source-repository example uses the same Binance Vision BTCUSDT spot
1-hour klines for January 2024 across its offline demo, optional L1 pack run,
and Great Expectations/Pandera comparison. The bundled file contains the
official monthly ZIP and its official SHA-256 sidecar.

The sample data is provided for non-commercial open-source data-science
evaluation under CC BY-NC-SA 4.0. See
[THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md) for attribution and source
terms. The example code follows the repository's Apache-2.0 license. The ZIP
and notice are included in the source repository only. They are excluded from
both L0 and L1 wheels and source distributions.

The companion [reproduction artifacts](../artifacts/README.md) document all
data and licenses. See the recorded [GE/Pandera benchmark results](../artifacts/market-bars-benchmark-results.json)
and [fault replay log](../artifacts/market-bars-fault-results.txt) alongside
the SEC, dbt, CDC, and UCI bundles.

## Run L0 and L1 locally

From the repository root, install L0:

```powershell
python -m pip install -e .
python -m examples.repro.market_bars.demo
```

The demo reads the bundled ZIP without network access, standardizes its 744
hourly rows, runs L0 time-order and price checks, then injects a negative close,
a non-finite close, availability before bar close, and availability after the
decision time. It prints the construction category, clean verdict, and blocked
rules. A zero process exit code means every expected result was observed.

To also execute the private L1 market-bars-pit pack, install L1 from the
configured private distribution source or local wheel, then run:

```powershell
python -m examples.repro.market_bars.demo --with-l1
```

L1 resolves its pack through the regular `cleanhatch.packs` entry-point
registry. The real month currently produces `ROLLING_MAD_SPIKE` warnings for
three rows. The report retains those warnings and allows the clean input to
continue. Both basic injected failures must still block.

To construct an example for every one of the 12 `market-bars-pit` checks, run:

```powershell
python -m examples.repro.market_bars.demo --with-l1 --fault-matrix
```

Each case is built from a copy of the same normalized public month. The output
prints the sorted row or field mutation, expected/actual rule status, overall
allow/block verdict, and relevant evidence. The price-spike case checks that
its injected UTC timestamp appears in `spike_rows`, so the real sample's
pre-existing warnings cannot make that case pass by themselves.

| L1 rule | Deterministic construction | Expected result |
| --- | --- | --- |
| `OHLC_HIGH_GE_OPEN_CLOSE` | Row 40: lower high below open/close and move low lower to keep the range ordered. | Block |
| `OHLC_LOW_LE_OPEN_CLOSE` | Row 40: raise low above open/close and move high higher. | Block |
| `OHLC_HIGH_GE_LOW` | Row 40: set open/close=100, high=99, low=101. This necessarily also violates the related low/high geometry and VWAP envelope. | Block |
| `POSITIVE_FINITE_PRICES` | Row 40: set low=0. | Block |
| `NON_NEGATIVE_FINITE_VOLUMES` | Row 40: set volume=-1 and quote volume=-close. | Block |
| `HALT_ZERO_AMPLITUDE` | Row 40: mark a non-flat bar as not trading. | Block |
| `VWAP_ENVELOPE` | Row 40: set quote volume to volume × high × 1.01, outside the 1 bps tolerance. | Block |
| `BAR_CLOSE_CAUSALITY` | Row 40: set availability and decision to one millisecond before close. | Block |
| `UNIQUE_BAR_COORDINATE` | Append a duplicate of row 40 with the same symbol and timestamps. | Block |
| `WARMUP_WINDOW_SKIP` | Keep only the first 10 rows, all within the default warm-up. | Check is `SKIPPED`; gate continues, with zero effective coverage. |
| `ACTIVE_DEAD_VALUE_RUN` | Set all OHLC=100, volume=1000, quote volume=100000, and active=true. | `WARNING`; gate continues. |
| `ROLLING_MAD_SPIKE` | Double row 100's close and resize high/low and quote volume to keep other bar geometry valid. | `WARNING`; gate continues, and the injected timestamp must be in `spike_rows`. |

The normalized `available_at` field is derived as `bar_close_time + 1ms`
because Binance's kline schema contains no receipt timestamp. It is a
documented demo assumption, not an exchange-reported receive time.

## Download the same public sample

The default run is offline. To fetch the fixed month again, the downloader
retrieves Binance Vision's `.CHECKSUM` and ZIP, verifies SHA-256, then writes
both files into an ignored output directory:

```powershell
python -m examples.repro.market_bars.download
python -m examples.repro.market_bars.demo `
  --zip examples/repro/market_bars/.downloads/BTCUSDT-1h-2024-01.zip
```

The downloader fails if either request, the checksum format, the size limit,
or the digest check fails. It never silently substitutes another dataset.

## Compare common checks with Great Expectations and Pandera

The benchmark pins Python 3.12, Cleanhatch 0.1.2, Great Expectations 1.25.0,
Pandas 2.3.3, and Pandera 0.36.0 in its own `uv` environment. It tests the same
prepared 744-row table serially in each library. The matched rules are
non-null finite close values, close greater than zero, and unique
`(symbol, decision_time)` keys. It runs both the real rows and the same
negative-close mutation, reports each library's verdict and median, and exits
with an error if any validator misses either expected verdict.

Install `uv` if needed, then run from the repository root:

```powershell
python -m pip install uv
python -m uv sync --project examples/repro/market_bars/benchmark --python 3.12
python -m uv run --project examples/repro/market_bars/benchmark `
  python -m examples.repro.market_bars.benchmark.run
```

Each library's rule objects are constructed before timing. ZIP reading,
normalization, Polars-to-Pandas conversion, interpreter startup, and library
imports are excluded. Great Expectations analytics and progress bars are
disabled. The benchmark reports one untimed warm-up, then the
median of five validation calls. All validators run serially to avoid CPU
contention. These are small-sample local timings, not a million-row throughput
claim. The full L1 pack's PIT and market-specific checks can also be timed as a
separate scope; they are not included in the matched-rule comparison.

To include that optional L1 timing, first obtain an authorized L1 wheel from
your configured private distribution source or local wheelhouse. After the
`uv sync` above, install it into the benchmark environment and invoke that
environment's Python directly so a later sync does not remove the private
package:

```powershell
python -m uv pip install `
  --python examples/repro/market_bars/benchmark/.venv/Scripts/python.exe `
  .\dist-l1\cleanhatch_l1-0.1.0-py3-none-any.whl
examples/repro/market_bars/benchmark/.venv/Scripts/python.exe `
  -m examples.repro.market_bars.benchmark.run --with-l1
```

The optional section reports the full `market-bars-pit` pack on the same clean
rows, the same negative-close injection, and the same pre-close-availability
injection used by the demo. It prints L1's installed version, per-case verdict,
warnings, and median validation time. It exits nonzero if an expected block is
missed or the clean sample blocks. The full-pack scope is intentionally reported
beside, and separately from, the GE/Pandera matched-rule results.
