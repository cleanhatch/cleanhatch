"""Fetch the fixed public Binance Vision sample and verify its official checksum."""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import urllib.request
from pathlib import Path

_FILENAME = "BTCUSDT-1h-2024-01.zip"
_BASE_URL = "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1h"
_MAX_DOWNLOAD_BYTES = 16 * 1024 * 1024


def _get(url: str, *, max_bytes: int) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "cleanhatch-repro/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read(max_bytes + 1)
    if len(payload) > max_bytes:
        raise ValueError(f"download exceeds the {max_bytes}-byte limit: {url}")
    return payload


def _checksum(checksum_text: str) -> str:
    matches = re.findall(r"\b[0-9a-fA-F]{64}\b", checksum_text)
    if len(matches) != 1:
        raise ValueError("Binance checksum response must contain exactly one SHA-256 digest")
    return matches[0].lower()


def download_sample(output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    checksum_path = output_dir / f"{_FILENAME}.CHECKSUM"
    archive_path = output_dir / _FILENAME
    checksum_url = f"{_BASE_URL}/{_FILENAME}.CHECKSUM"
    archive_url = f"{_BASE_URL}/{_FILENAME}"

    checksum_bytes = _get(checksum_url, max_bytes=1024)
    expected = _checksum(checksum_bytes.decode("ascii"))
    archive_bytes = _get(archive_url, max_bytes=_MAX_DOWNLOAD_BYTES)
    actual = hashlib.sha256(archive_bytes).hexdigest()
    if actual != expected:
        raise ValueError(f"Binance SHA-256 mismatch: expected {expected}, got {actual}")

    archive_path.write_bytes(archive_bytes)
    checksum_path.write_bytes(checksum_bytes)
    return archive_path, checksum_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).with_name(".downloads"),
        help="output directory (default: this example's ignored .downloads folder)",
    )
    args = parser.parse_args(argv)

    try:
        archive_path, checksum_path = download_sample(args.out_dir)
    except (OSError, UnicodeDecodeError, ValueError) as error:
        print(f"download failed: {error}", file=sys.stderr)
        return 1

    print(f"verified sample: {archive_path}")
    print(f"official checksum: {checksum_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
