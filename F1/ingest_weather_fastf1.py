#!/usr/bin/env python3
"""
FastF1 Weather Ingestion -> data/store/weather.csv
- Uses laps=False by default (~3 API calls/race instead of ~10) to stay under the 500 calls/hr limit.
- Saves incrementally after every race and exits cleanly on RateLimitExceededError.
- Re-running automatically resumes from where it stopped using data/fastf1_cache/.
"""

import argparse
from datetime import datetime
from pathlib import Path

import fastf1
from fastf1.exceptions import DataNotLoadedError, RateLimitExceededError
import pandas as pd

CACHE_DIR = Path("data/fastf1_cache")
STORE_DIR = Path("data/store")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
STORE_DIR.mkdir(parents=True, exist_ok=True)

fastf1.Cache.enable_cache(str(CACHE_DIR))


def save_progress(existing: pd.DataFrame, new_rows: list[dict], out_path: Path) -> pd.DataFrame:
    if not new_rows:
        return existing
    df_all = (
        pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
        .drop_duplicates(subset=["race_id"])
        .sort_values(["season", "round"])
    )
    df_all.to_csv(out_path, index=False)
    return df_all


def build_weather_table(start_year: int = 2018, end_year: int = 2026, include_laps: bool = False) -> None:
    out_path = STORE_DIR / "weather.csv"
    df_all = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame()
    done_ids = set(df_all["race_id"]) if not df_all.empty else set()

    today = pd.Timestamp.utcnow().tz_localize(None)
    new_rows = []

    try:
        for year in range(start_year, end_year + 1):
            schedule = fastf1.get_event_schedule(year, include_testing=False)
            for _, event in schedule.iterrows():
                rnd = int(event["RoundNumber"])
                if rnd == 0:
                    continue
                race_id = f"{year}_{rnd:02d}"

                if pd.notna(event["EventDate"]) and pd.Timestamp(event["EventDate"]) > today:
                    continue
                if race_id in done_ids:
                    continue

                print(f"Loading {race_id} ({event['EventName']})...")
                try:
                    session = fastf1.get_session(year, rnd, "R")
                    session.load(laps=include_laps, telemetry=False, weather=True, messages=False)
                    w = session.weather_data
                except DataNotLoadedError:
                    print(f"  [Skipped {race_id}] Weather data not published yet.")
                    continue
                rain_pct = float(w["Rainfall"].mean() * 100) if not w.empty else 0.0
                avg_track_temp = float(w["TrackTemp"].mean()) if not w.empty else None
                avg_air_temp = float(w["AirTemp"].mean()) if not w.empty else None

                wet_tire_lap_pct = 0.0
                if include_laps:
                    try:
                        laps = session.laps
                        if not laps.empty and "Compound" in laps.columns:
                            wet_laps = laps["Compound"].isin(["INTERMEDIATE", "WET"]).sum()
                            wet_tire_lap_pct = float(wet_laps / len(laps) * 100)
                    except Exception:
                        pass

                new_rows.append({
                    "race_id": race_id,
                    "season": year,
                    "round": rnd,
                    "rain_pct": round(rain_pct, 1),
                    "wet_tire_lap_pct": round(wet_tire_lap_pct, 1),
                    "is_wet_race": (rain_pct > 0.0) or (wet_tire_lap_pct >= 5.0),
                    "avg_track_temp_c": round(avg_track_temp, 1) if avg_track_temp is not None else None,
                    "avg_air_temp_c": round(avg_air_temp, 1) if avg_air_temp is not None else None,
                })
                done_ids.add(race_id)

                # Save after every race so rate limits never lose work
                df_all = save_progress(df_all, new_rows, out_path)
                new_rows = []

    except RateLimitExceededError:
        df_all = save_progress(df_all, new_rows, out_path)
        print(f"\n[Rate Limit Hit] Saved {len(df_all)} races to {out_path}.")
        print("Wait ~1 hour and re-run the exact same command to resume where you left off.")
        return

    df_all = save_progress(df_all, new_rows, out_path)
    print(f"\nDone! Saved {len(df_all)} races to {out_path} ({int(df_all['is_wet_race'].sum())} wet races).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=2018)
    parser.add_argument("--end", type=int, default=datetime.now().year)
    parser.add_argument("--with-laps", action="store_true", help="Also fetch tire compounds (~3x more API calls)")
    args = parser.parse_args()
    build_weather_table(args.start, args.end, include_laps=args.with_laps)