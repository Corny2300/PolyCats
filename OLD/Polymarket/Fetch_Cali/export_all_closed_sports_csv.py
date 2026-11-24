#!/usr/bin/env python3
"""Export one row per closed-sports market with the 24 hourly bins as columns.

Each metric-hour cell contains a JSON list of per-outcome values. This keeps
outcome/token observations separate while making the CSV grain one market/row.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data" / "all_closed_sports"
HOURLY_METRICS = [
    "trade_records",
    "unique_transaction_hashes",
    "unique_wallets_observed",
    "buy_records",
    "sell_records",
    "share_size",
    "trade_side_notional_usdc",
    "vwap_price",
    "median_price",
]


def csv_value(value: Any) -> Any:
    """Keep scalar values native to CSV; encode nested source values as JSON."""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def discover_market_columns(catalogue: Path) -> tuple[list[str], int]:
    columns: set[str] = set()
    count = 0
    with catalogue.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                market = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON in {catalogue}, line {line_number}: {exc}") from exc
            if not isinstance(market, dict):
                raise RuntimeError(f"Expected a JSON object in {catalogue}, line {line_number}")
            columns.update(market.keys())
            count += 1
    if not count:
        raise RuntimeError(f"No market records found in {catalogue}")
    return sorted(columns), count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--force", action="store_true", help="replace an existing output after a successful export")
    parser.add_argument("--progress-every", type=int, default=100_000, help="report progress every N markets")
    args = parser.parse_args()

    data_dir = args.data_dir.expanduser().resolve()
    catalogue = data_dir / "markets.jsonl"
    hourly_dir = data_dir / "hourly_activity_24h"
    output = (args.output.expanduser().resolve() if args.output else
              data_dir / "closed_sports_all_data.csv")
    if not catalogue.is_file():
        parser.error(f"catalogue not found: {catalogue}")
    if not hourly_dir.is_dir():
        parser.error(f"hourly aggregate directory not found: {hourly_dir}")
    if output.exists() and not args.force:
        parser.error(f"output already exists (use --force to replace): {output}")
    if args.progress_every < 1:
        parser.error("--progress-every must be at least 1")

    print(f"scanning catalogue fields: {catalogue}", flush=True)
    market_fields, total = discover_market_columns(catalogue)
    preferred_market_fields = [
        "id", "conditionId", "slug", "question", "startDate", "endDate", "closedTime",
        "outcomes", "outcomePrices", "clobTokenIds", "volume", "volumeNum", "volumeClob",
        "volumeAmm", "liquidity", "liquidityNum", "liquidityClob", "liquidityAmm", "spread",
        "bestBid", "bestAsk", "lastTradePrice",
    ]
    market_order = [field for field in preferred_market_fields if field in market_fields]
    market_order += [field for field in market_fields if field not in market_order]
    fields = market_order + ["hourly_file_exists", "has_hourly_activity", "hourly_read_error"]
    for hour in range(1, 25):
        prefix = f"hour_{hour:02d}"
        fields.extend([f"{prefix}_start_utc", f"{prefix}_end_utc", f"{prefix}_outcomes"])
        fields.extend(f"{prefix}_{metric}" for metric in HOURLY_METRICS)
    print(f"catalogue markets={total:,}; market columns={len(market_fields)}; "
          f"hourly columns={24 * (3 + len(HOURLY_METRICS))}; output={output}", flush=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".partial", dir=output.parent)
    os.close(fd)
    temp_path = Path(temp_name)
    processed = 0
    rows_written = 0
    with_hourly = 0
    files_found = 0
    read_errors = 0
    try:
        with catalogue.open("r", encoding="utf-8") as source, temp_path.open(
            "w", encoding="utf-8", newline="", buffering=1024 * 1024
        ) as target:
            writer = csv.DictWriter(target, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
            writer.writeheader()
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    market = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"Invalid JSON in {catalogue}, line {line_number}: {exc}") from exc
                market_id = str(market.get("id", ""))
                activity_path = hourly_dir / f"{market_id}.json.gz"
                file_exists = activity_path.is_file()
                aggregates: list[dict[str, Any]] = []
                read_error = False
                if file_exists:
                    files_found += 1
                    try:
                        with gzip.open(activity_path, "rt", encoding="utf-8") as handle:
                            payload = json.load(handle)
                        if not isinstance(payload, list):
                            raise ValueError("expected a JSON list of hourly aggregates")
                        aggregates = [row for row in payload if isinstance(row, dict)]
                    except (OSError, EOFError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                        read_errors += 1
                        read_error = True
                        print(f"warning: market={market_id} hourly file unreadable: {type(exc).__name__}: {exc}",
                              file=sys.stderr, flush=True)

                row = {key: csv_value(value) for key, value in market.items()}
                close_time = None
                close_text = market.get("endDate")
                if close_text:
                    try:
                        close_time = datetime.fromisoformat(str(close_text).replace("Z", "+00:00")).astimezone(timezone.utc)
                    except (ValueError, TypeError):
                        close_time = None
                per_hour: dict[int, list[dict[str, Any]]] = {hour: [] for hour in range(1, 25)}
                for aggregate in aggregates:
                    try:
                        hour = int(aggregate["hours_before_close"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    if 1 <= hour <= 24:
                        per_hour[hour].append(aggregate)

                for hour in range(1, 25):
                    prefix = f"hour_{hour:02d}"
                    records = per_hour[hour]
                    starts = {record.get("hour_start_utc") for record in records if record.get("hour_start_utc")}
                    ends = {record.get("hour_end_utc") for record in records if record.get("hour_end_utc")}
                    if close_time is not None:
                        hour_start = (close_time - timedelta(hours=hour)).isoformat()
                        hour_end = (close_time - timedelta(hours=hour - 1)).isoformat()
                    else:
                        hour_start = next(iter(starts), "")
                        hour_end = next(iter(ends), "")
                    row[f"{prefix}_start_utc"] = hour_start
                    row[f"{prefix}_end_utc"] = hour_end
                    identity = [
                        {"asset": record.get("asset", ""), "outcome": record.get("outcome", ""),
                         "outcome_index": record.get("outcome_index")}
                        for record in records
                    ]
                    row[f"{prefix}_outcomes"] = json.dumps(identity, ensure_ascii=False, separators=(",", ":")) if identity else ""
                    for metric in HOURLY_METRICS:
                        # Values align by position with the outcome/token list
                        # above, avoiding repeated long token IDs in every cell.
                        values = [record.get(metric) for record in records]
                        row[f"{prefix}_{metric}"] = json.dumps(values, ensure_ascii=False, separators=(",", ":")) if values else ""

                row["hourly_file_exists"] = "true" if file_exists else "false"
                row["has_hourly_activity"] = "true" if any(per_hour.values()) else "false"
                row["hourly_read_error"] = "true" if read_error else "false"
                if any(per_hour.values()):
                    with_hourly += 1
                writer.writerow(row)
                rows_written += 1

                processed += 1
                if processed % args.progress_every == 0 or processed == total:
                    target.flush()
                    size_mb = temp_path.stat().st_size / (1024 * 1024)
                    print(f"progress markets={processed:,}/{total:,} ({processed / total:.1%}) "
                          f"rows={rows_written:,} with_hourly_activity={with_hourly:,} "
                          f"hourly_files={files_found:,} read_errors={read_errors:,} size={size_mb:,.0f} MiB",
                          flush=True)

        if output.exists() and not args.force:
            raise FileExistsError(f"output appeared during export (not replacing): {output}")
        os.replace(temp_path, output)
    except BaseException:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise

    size_gb = output.stat().st_size / (1024 ** 3)
    print(f"complete: {output}", flush=True)
    print(f"markets={processed:,}; rows={rows_written:,}; markets_with_hourly_activity={with_hourly:,}; "
          f"hourly_files={files_found:,}; unreadable_hourly_files={read_errors:,}; size={size_gb:,.2f} GiB",
          flush=True)
    print("Note: hourly cells contain JSON lists for each outcome/token; individual raw trades were not retained.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
