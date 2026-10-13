#!/usr/bin/env python3
"""Build one forecast row per usable settled binary market from the wide master CSV."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "all_closed_sports"
MASTER = DATA / "closed_sports_all_data.csv"
OUTPUT = DATA / "closed_sports_market_forecasts.csv"
OUTPUT_COLUMNS = [
    "market_id", "question", "slug", "first_outcome", "probability", "actual",
    "hours_before_close", "latest_hour_trade_records",
    "latest_hour_trade_side_notional_usdc", "volumeNum", "liquidityNum", "spread",
]


def as_list(value):
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return []
    return []


def forecasts_for_chunk(chunk: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Use pandas chunk operations to get the nearest-hour binary forecasts."""
    markets = chunk[["id", "question", "slug", "outcomes", "outcomePrices",
                     "volumeNum", "liquidityNum", "spread"]].copy()
    markets["market_id"] = markets["id"].astype("string")
    markets["outcomes"] = markets["outcomes"].map(as_list)
    markets["settled"] = markets["outcomePrices"].map(as_list)
    binary = markets.loc[
        markets["outcomes"].str.len().eq(2) & markets["settled"].str.len().eq(2)
    ].copy()
    binary["first_result"] = pd.to_numeric(binary["settled"].str[0], errors="coerce")
    binary = binary.loc[binary["first_result"].le(0.02) | binary["first_result"].ge(0.98)].copy()
    binary["actual"] = binary["first_result"].ge(0.98).astype("int8")
    binary["first_outcome"] = binary["outcomes"].str[0]
    binary_n = len(binary)
    if binary.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS), binary_n

    metadata = binary.set_index("market_id")[[
        "question", "slug", "first_outcome", "actual", "volumeNum", "liquidityNum", "spread"
    ]]
    activity_parts = []
    for hour in range(1, 25):
        prefix = f"hour_{hour:02d}"
        part = pd.DataFrame(index=binary.index)
        part["market_id"] = binary["market_id"]
        part["token"] = chunk.loc[binary.index, f"{prefix}_outcomes"].map(as_list)
        part["vwap_price"] = chunk.loc[binary.index, f"{prefix}_vwap_price"].map(as_list)
        part["trade_records"] = chunk.loc[binary.index, f"{prefix}_trade_records"].map(as_list)
        part["trade_side_notional_usdc"] = chunk.loc[binary.index, f"{prefix}_trade_side_notional_usdc"].map(as_list)
        part = part.explode(["token", "vwap_price", "trade_records", "trade_side_notional_usdc"])
        part = part.loc[part["token"].map(lambda value: isinstance(value, dict))].copy()
        if part.empty:
            continue
        part["outcome_index"] = pd.to_numeric(part["token"].str.get("outcome_index"), errors="coerce")
        part["vwap_price"] = pd.to_numeric(part["vwap_price"], errors="coerce")
        part["trade_records"] = pd.to_numeric(part["trade_records"], errors="coerce")
        part["trade_side_notional_usdc"] = pd.to_numeric(part["trade_side_notional_usdc"], errors="coerce")
        part["hours_before_close"] = hour
        valid = part["outcome_index"].isin([0, 1]) & part["vwap_price"].between(0, 1)
        activity_parts.append(part.loc[valid, [
            "market_id", "hours_before_close", "outcome_index", "vwap_price",
            "trade_records", "trade_side_notional_usdc",
        ]])

    if not activity_parts:
        return pd.DataFrame(columns=OUTPUT_COLUMNS), binary_n
    activity = pd.concat(activity_parts, ignore_index=True)
    nearest_hour = activity.groupby("market_id")["hours_before_close"].transform("min")
    latest = activity.loc[activity["hours_before_close"].eq(nearest_hour)].copy()
    prices = latest.pivot_table(
        index="market_id", columns="outcome_index", values="vwap_price", aggfunc="first"
    ).reindex(columns=[0, 1])
    prices["probability"] = prices[0].fillna(1 - prices[1])
    aggregates = latest.groupby("market_id").agg(
        hours_before_close=("hours_before_close", "min"),
        latest_hour_trade_records=("trade_records", "sum"),
        latest_hour_trade_side_notional_usdc=("trade_side_notional_usdc", "sum"),
    )
    result = prices[["probability"]].join(aggregates).join(metadata)
    result = result.dropna(subset=["probability"]).reset_index()
    return result.reindex(columns=OUTPUT_COLUMNS), binary_n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Rebuild the CSV even if it already exists.")
    parser.add_argument("--chunk-size", type=int, default=25_000, help="Master CSV rows per pandas chunk.")
    args = parser.parse_args()

    if not MASTER.is_file():
        raise FileNotFoundError(MASTER)
    if OUTPUT.exists() and not args.force and OUTPUT.stat().st_mtime >= MASTER.stat().st_mtime:
        print(f"Already prepared from the current master CSV: {OUTPUT}")
        return

    DATA.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix="closed_sports_forecasts_", suffix=".csv.tmp", dir=DATA)
    os.close(fd)
    temporary = Path(temporary_name)
    catalogue_n = binary_n = forecast_n = 0
    wrote_header = False
    try:
        usecols = ["id", "question", "slug", "outcomes", "outcomePrices",
                   "volumeNum", "liquidityNum", "spread"]
        for hour in range(1, 25):
            prefix = f"hour_{hour:02d}"
            usecols.extend([
                f"{prefix}_outcomes", f"{prefix}_vwap_price", f"{prefix}_trade_records",
                f"{prefix}_trade_side_notional_usdc",
            ])
        for chunk in pd.read_csv(
            MASTER, usecols=usecols, dtype={"id": "string"}, chunksize=args.chunk_size, low_memory=False
        ):
            catalogue_n += len(chunk)
            forecasts, settled_binary_n = forecasts_for_chunk(chunk)
            binary_n += settled_binary_n
            if not forecasts.empty:
                forecasts.to_csv(temporary, mode="a", header=not wrote_header, index=False)
                wrote_header = True
                forecast_n += len(forecasts)

            print(
                f"Scanned {catalogue_n:,} sports markets from master CSV; "
                f"settled binary={binary_n:,}; usable forecasts={forecast_n:,}",
                flush=True,
            )

        if not wrote_header:
            raise ValueError("No usable settled Sports forecasts found; CSV was not created.")
        temporary.replace(OUTPUT)
    finally:
        temporary.unlink(missing_ok=True)

    print(f"Saved {forecast_n:,} one-per-market forecasts to {OUTPUT}")


if __name__ == "__main__":
    main()
