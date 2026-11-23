#!/usr/bin/env python3
"""
F1 Data Store Ingestion Pipeline (Jolpica F1 API -> Local CSV Data Store)

Handles:
- Offset/limit pagination (max limit=100 innermost rows per Jolpica spec)
- Rate-limit compliance (4 req/s burst, 500 req/hr sustained) + exponential backoff on 429/5xx
- Per-season raw JSON caching so historical seasons are never re-fetched
- Normalized relational CSV tables + a merged denormalized master CSV for notebooks

Usage:
    python ingest_f1.py                       # Full backfill (1950-2026)
    python ingest_f1.py --start 2004          # Modern fastest-lap era only
    python ingest_f1.py --refresh-current     # Only re-fetch the latest season & rebuild CSVs
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

BASE_URL = "https://api.jolpi.ca/ergast/f1"
HEADERS = {"User-Agent": "F1DataStore/1.0 (local-analytics)"}

RAW_DIR = Path("data/raw")
STORE_DIR = Path("data/store")

# Polite pacing: ~3 req/s (under Jolpica's 4 req/s burst limit)
REQUEST_DELAY_SEC = 0.35
MAX_RETRIES = 6


def fetch_with_backoff(session: requests.Session, url: str) -> dict[str, Any]:
    """GET a URL with exponential backoff on HTTP 429 or transient 5xx errors."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = session.get(url, headers=HEADERS, timeout=20)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429 or resp.status_code >= 500:
                wait = min(2 ** (attempt + 1), 60)
                print(f"  [HTTP {resp.status_code}] Rate-limited/transient error. Sleeping {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
        except requests.RequestException as exc:
            wait = min(2 ** (attempt + 1), 60)
            print(f"  [Network Error] {exc}. Retrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError(f"Failed to fetch after {MAX_RETRIES} attempts: {url}")


def fetch_paginated(
    session: requests.Session,
    endpoint: str,
    table_key: str,
    list_key: str,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """
    Fetch all pages for a Jolpica endpoint.
    Note: Jolpica counts innermost rows (e.g. Results) against `limit` and `MRData.total`.
    """
    all_items: list[dict[str, Any]] = []
    offset = 0

    while True:
        url = f"{BASE_URL}/{endpoint}.json?limit={limit}&offset={offset}"
        payload = fetch_with_backoff(session, url)
        mrdata = payload["MRData"]

        total = int(mrdata.get("total", 0))
        items = mrdata.get(table_key, {}).get(list_key, [])
        all_items.extend(items)

        offset += int(mrdata.get("limit", limit))
        if offset >= total or not items:
            break
        time.sleep(REQUEST_DELAY_SEC)

    return all_items


def fetch_season_raw(
    session: requests.Session, year: int, force_refresh: bool = False
) -> tuple[list[dict], list[dict]]:
    """Fetch and cache raw race results and WCC standings for a single season."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    res_cache = RAW_DIR / f"results_{year}.json"
    wcc_cache = RAW_DIR / f"wcc_{year}.json"

    if res_cache.exists() and not force_refresh:
        races = json.loads(res_cache.read_text(encoding="utf-8"))
    else:
        races = fetch_paginated(session, f"{year}/results", "RaceTable", "Races")
        res_cache.write_text(json.dumps(races), encoding="utf-8")
        time.sleep(REQUEST_DELAY_SEC)

    # Constructors' Championship started in 1958
    if year < 1958:
        wcc = []
    elif wcc_cache.exists() and not force_refresh:
        wcc = json.loads(wcc_cache.read_text(encoding="utf-8"))
    else:
        wcc = fetch_paginated(
            session, f"{year}/constructorstandings/1", "StandingsTable", "StandingsLists"
        )
        wcc_cache.write_text(json.dumps(wcc), encoding="utf-8")
        time.sleep(REQUEST_DELAY_SEC)

    return races, wcc


def parse_lap_time_ms(time_str: str | None) -> float | None:
    """Convert 'M:SS.mmm' or 'SS.mmm' lap time string to milliseconds."""
    if not time_str:
        return None
    try:
        parts = time_str.split(":")
        if len(parts) == 2:
            return (int(parts[0]) * 60 + float(parts[1])) * 1000.0
        return float(parts[0]) * 1000.0
    except ValueError:
        return None


def build_data_store(start_year: int, end_year: int, refresh_current: bool = False) -> None:
    """Run full ingestion and emit normalized CSV tables + merged master CSV."""
    STORE_DIR.mkdir(parents=True, exist_ok=True)

    races_map: dict[str, dict] = {}
    drivers_map: dict[str, dict] = {}
    constructors_map: dict[str, dict] = {}
    wcc_rows: list[dict] = []
    result_rows: list[dict] = []

    session = requests.Session()

    for year in range(start_year, end_year + 1):
        force = refresh_current and (year == end_year)
        print(f"Processing season {year}{'(refresh)' if force else ''}...")
        races, wcc_lists = fetch_season_raw(session, year, force_refresh=force)

        # Extract season WCC winner
        wcc_constructor_id = None
        if wcc_lists:
            standings = wcc_lists[0].get("ConstructorStandings", [])
            if standings:
                top = standings[0]
                c = top["Constructor"]
                wcc_constructor_id = c["constructorId"]
                wcc_rows.append({
                    "season": year,
                    "wcc_constructor_id": wcc_constructor_id,
                    "wcc_constructor_name": c["name"],
                    "wcc_points": float(top.get("points", 0)),
                    "wcc_wins": int(top.get("wins", 0)),
                })

        # Flatten Race -> Results (safely merges races split across pagination pages)
        for race in races:
            season = int(race["season"])
            rnd = int(race["round"])
            race_id = f"{season}_{rnd:02d}"
            circuit = race.get("Circuit", {})
            loc = circuit.get("Location", {})

            races_map[race_id] = {
                "race_id": race_id,
                "season": season,
                "round": rnd,
                "race_name": race.get("raceName"),
                "date": race.get("date"),
                "circuit_id": circuit.get("circuitId"),
                "circuit_name": circuit.get("circuitName"),
                "locality": loc.get("locality"),
                "country": loc.get("country"),
            }

            for res in race.get("Results", []):
                drv = res["Driver"]
                con = res["Constructor"]
                drv_id = drv["driverId"]
                con_id = con["constructorId"]

                drivers_map[drv_id] = {
                    "driver_id": drv_id,
                    "driver_code": drv.get("code"),
                    "permanent_number": drv.get("permanentNumber"),
                    "full_name": f"{drv.get('givenName', '')} {drv.get('familyName', '')}".strip(),
                    "dob": drv.get("dateOfBirth"),
                    "nationality": drv.get("nationality"),
                }

                constructors_map[con_id] = {
                    "constructor_id": con_id,
                    "constructor_name": con.get("name"),
                    "constructor_nationality": con.get("nationality"),
                }

                fl = res.get("FastestLap", {})
                fl_time_str = fl.get("Time", {}).get("time")

                result_rows.append({
                    "race_id": race_id,
                    "season": season,
                    "round": rnd,
                    "driver_id": drv_id,
                    "constructor_id": con_id,
                    "grid": int(res["grid"]) if str(res.get("grid", "")).isdigit() else None,
                    "position": int(res["position"]) if str(res.get("position", "")).isdigit() else None,
                    "position_text": res.get("positionText"),
                    "points": float(res.get("points", 0)),
                    "laps": int(res["laps"]) if str(res.get("laps", "")).isdigit() else None,
                    "status": res.get("status"),
                    "race_time_ms": int(res["Time"]["millis"]) if "Time" in res and "millis" in res["Time"] else None,
                    "fastest_lap_rank": int(fl["rank"]) if str(fl.get("rank", "")).isdigit() else None,
                    "fastest_lap_num": int(fl["lap"]) if str(fl.get("lap", "")).isdigit() else None,
                    "fastest_lap_time": fl_time_str,
                    "fastest_lap_ms": parse_lap_time_ms(fl_time_str),
                    "fastest_lap_avg_speed": float(fl["AverageSpeed"]["speed"]) if "AverageSpeed" in fl and "speed" in fl["AverageSpeed"] else None,
                    "is_win": res.get("position") == "1",
                    "is_podium": res.get("position") in {"1", "2", "3"},
                    "is_fastest_lap": fl.get("rank") == "1",
                })

    # Convert to DataFrames & deduplicate
    df_races = pd.DataFrame(races_map.values()).sort_values(["season", "round"])
    df_drivers = pd.DataFrame(drivers_map.values()).sort_values("driver_id")
    df_constructors = pd.DataFrame(constructors_map.values()).sort_values("constructor_id")
    df_wcc = pd.DataFrame(wcc_rows).drop_duplicates(subset=["season"]).sort_values("season")
    df_results = (
        pd.DataFrame(result_rows)
        .drop_duplicates(subset=["race_id", "driver_id"])
        .sort_values(["season", "round", "position"])
    )

    # Save normalized tables
    df_races.to_csv(STORE_DIR / "races.csv", index=False)
    df_drivers.to_csv(STORE_DIR / "drivers.csv", index=False)
    df_constructors.to_csv(STORE_DIR / "constructors.csv", index=False)
    df_wcc.to_csv(STORE_DIR / "wcc_champions.csv", index=False)
    df_results.to_csv(STORE_DIR / "results.csv", index=False)

    # Build merged, analysis-ready master CSV
    df_master = (
        df_results
        .merge(df_races[["race_id", "race_name", "date", "circuit_id", "country"]], on="race_id", how="left")
        .merge(df_drivers[["driver_id", "driver_code", "full_name", "nationality"]], on="driver_id", how="left")
        .merge(df_constructors[["constructor_id", "constructor_name"]], on="constructor_id", how="left")
        .merge(df_wcc[["season", "wcc_constructor_id", "wcc_constructor_name"]], on="season", how="left")
    )
    df_master["in_wcc_car"] = df_master["constructor_id"] == df_master["wcc_constructor_id"]
    df_master.to_csv(STORE_DIR / "f1_master.csv", index=False)

    print(f"\nData store built in '{STORE_DIR}/':")
    print(f"  races.csv         : {len(df_races):,} rows")
    print(f"  drivers.csv       : {len(df_drivers):,} rows")
    print(f"  constructors.csv  : {len(df_constructors):,} rows")
    print(f"  wcc_champions.csv : {len(df_wcc):,} rows")
    print(f"  results.csv       : {len(df_results):,} rows")
    print(f"  f1_master.csv     : {len(df_master):,} rows (merged analytics table)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch F1 data from Jolpica API into a CSV data store.")
    parser.add_argument("--start", type=int, default=1950, help="Start season (default: 1950)")
    parser.add_argument("--end", type=int, default=datetime.now().year, help="End season (default: current year)")
    parser.add_argument("--refresh-current", action="store_true", help="Re-fetch only the latest season")
    args = parser.parse_args()

    build_data_store(args.start, args.end, refresh_current=args.refresh_current)