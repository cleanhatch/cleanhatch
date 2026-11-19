# Reproduction data and recorded evidence

These source-repository artifacts accompany the runnable examples in
[`../market_bars/`](../market_bars/README.md). They are not part of either
Cleanhatch wheel or source distribution. The example code remains under the
repository's Apache-2.0 license; each data source has the terms described
below.

## Market-bars benchmark and fault evidence

- [`market-bars-benchmark-results.json`](market-bars-benchmark-results.json)
  records the serial, matched-rule Great Expectations and Pandera timing runs
  on the bundled 744-row Binance Vision BTCUSDT January 2024 sample. This
  historical result was measured with Cleanhatch 0.1.0; the public repository's
  current benchmark lock targets its local Cleanhatch 0.1.2 source.
- [`market-bars-fault-results.txt`](market-bars-fault-results.txt) records the
  real-data and deterministic fault-matrix gate outcomes.
- The raw sample is bundled in
  [`../market_bars/data/`](../market_bars/data/BTCUSDT-1h-2024-01.zip), with a
  checksum sidecar and downloader. Binance Vision identifies the data as CC
  BY-NC-SA 4.0 and permits non-commercial open-source data-science evaluation
  under its terms. See the market-bars
  [third-party notice](../market_bars/THIRD-PARTY-NOTICES.md) for attribution
  and the official terms. The downloader retrieves that same fixed month.

## SEC, dbt, and deterministic CDC replays

- [`sec-edgar-and-dbt-jaffle-samples.zip`](sec-edgar-and-dbt-jaffle-samples.zip)
  contains raw SEC EDGAR XBRL responses for Microsoft (CIK 0000789019), raw
  fictional dbt Jaffle Shop seed CSVs, and normalized CSV/Parquet examples.
  The recorded manifest covers eight SEC fundamentals rows, 64 SEC RAG
  documents, and 113 Jaffle Shop rows. The included dbt seed license is
  Apache-2.0. The SEC responses come from the public EDGAR XBRL API.
- [`sec-dbt-cdc-fault-replays.zip`](sec-dbt-cdc-fault-replays.zip) contains
  four clean Parquet inputs and 15 deterministic faulty replays with
  `CASES.txt`, machine-readable case metadata, and expected rule evidence.
  Five temporal-core cases target public L0. Ten cases use private L1 packs:
  five SEC fundamentals cases use `sec-fundamentals-pit`, while three Jaffle
  ledger and two SEC RAG cases use `ai-rag-ledger-guard`. Rerunning these cases
  requires an authorized L1 installation. The CDC records are synthetic
  deterministic examples, not a vendor event feed. Fault replay cases record
  exit code 2 when the expected gate blocks the faulty input.

## UCI Online Retail original archive

- [`uci-online-retail.zip`](uci-online-retail.zip) is the original UCI archive
  containing `Online Retail.xlsx`; it has not been normalized or transformed.
- Dataset page: [UCI Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail).
  The dataset is licensed CC BY 4.0. Cite it as: Chen, D. (2015). *Online
  Retail* [Dataset]. UCI Machine Learning Repository.
  https://doi.org/10.24432/C5BW33. The associated study is Daqing Chen, Sai
  Laing Sain, and Kun Guo (2012), “Data mining for the online retail industry:
  A case study of RFM model-based customer segmentation using data mining.”
