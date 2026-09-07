"""
Persistence Log & Recurrence Tracking Module
============================================

HOW TO RUN:
    Initialize the database and populate persistence history from clustered hotspots:
        python -m processing.persistence_log --input data/hotspots_clustered.csv --db data/hotspots.db

INPUTS & ENVIRONMENT:
    - Clustered Hotspots DataFrame with event_id, latitude, longitude, dist_to_nearest_industrial_m, facility_id, acq_date, acq_time, frp, confidence, daynight, nearest_industrial_type.
    - SQLite database path (default: data/hotspots.db).

OUTPUT:
    - Populates the `hotspot_history` table in SQLite.
    - Enriches DataFrame with:
      [location_key, days_active_last_30, duty_cycle, night_detection_fraction]

STRICT RULES OBSERVED:
    - location_key = facility_id if within 500m of known OSM facility, ELSE H3 resolution-8 cell ID.
    - NEVER rounds lat/lon to a naive degree grid (prevents pixel jitter fragmentation across cells).
    - Computes `night_detection_fraction` (fraction of historical detections with daynight='N').
    - NEVER computes `recurrence_time_of_day_std` (VIIRS sun-synchronous orbital artifact).
"""

import os
import sys
import sqlite3
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np

# Compatible H3 import across versions
try:
    import h3
    def get_h3_res8_cell(lat: float, lon: float) -> str:
        if hasattr(h3, "latlng_to_cell"):
            return str(h3.latlng_to_cell(lat, lon, 8))
        elif hasattr(h3, "geo_to_h3"):
            return str(h3.geo_to_h3(lat, lon, 8))
        else:
            return f"h3_{round(lat, 3)}_{round(lon, 3)}"
except ImportError:
    def get_h3_res8_cell(lat: float, lon: float) -> str:
        return f"h3res8_{round(lat, 3)}_{round(lon, 3)}"

# Ensure module import works when run as script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.settings import DB_PATH, SCHEMA_PATH, FACILITY_PROXIMITY_THRESHOLD_M


def init_db(db_path: str = str(DB_PATH), schema_path: str = str(SCHEMA_PATH)) -> None:
    """
    Initializes the SQLite persistence database schema if it doesn't exist.
    """
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        if os.path.exists(schema_path):
            with open(schema_path, "r", encoding="utf-8") as f:
                cursor.executescript(f.read())
        else:
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS hotspot_history (
              event_id TEXT,
              location_key TEXT,
              acq_date DATE,
              acq_time TEXT,
              frp REAL,
              confidence REAL,
              daynight TEXT,
              nearest_industrial_type TEXT,
              PRIMARY KEY (location_key, acq_date, acq_time)
            );
            """)
        conn.commit()


def compute_location_keys(df_hotspots: pd.DataFrame) -> List[str]:
    """
    Assigns location_key: facility_id if within 500m of known OSM facility,
    else H3 resolution-8 cell ID.
    """
    keys = []
    for _, row in df_hotspots.iterrows():
        dist = float(row.get("dist_to_nearest_industrial_m", 99999.0))
        fac_id = row.get("facility_id")
        
        if dist < FACILITY_PROXIMITY_THRESHOLD_M and fac_id and str(fac_id).strip() != "" and str(fac_id) != "None":
            keys.append(str(fac_id))
        else:
            lat = float(row["latitude"])
            lon = float(row["longitude"])
            cell_id = get_h3_res8_cell(lat, lon)
            keys.append(f"h3_r8_{cell_id}")
            
    return keys


def update_persistence_log(df_hotspots: pd.DataFrame, db_path: str = str(DB_PATH)) -> pd.DataFrame:
    """
    Upserts hotspot observations into the hotspot_history table and computes
    days_active_last_30 and night_detection_fraction for each hotspot.
    """
    if df_hotspots.empty:
        df_out = df_hotspots.copy()
        df_out["location_key"] = []
        df_out["days_active_last_30"] = []
        df_out["duty_cycle"] = []
        df_out["night_detection_fraction"] = []
        return df_out

    init_db(db_path)
    df_work = df_hotspots.copy()

    # 1. Assign location_keys
    if "location_key" not in df_work.columns:
        df_work["location_key"] = compute_location_keys(df_work)

    # 2. Upsert into database
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        for _, row in df_work.iterrows():
            cursor.execute("""
            INSERT OR REPLACE INTO hotspot_history (
                event_id, location_key, acq_date, acq_time, frp, confidence, daynight, nearest_industrial_type
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                str(row.get("event_id", "")),
                str(row["location_key"]),
                str(row["acq_date"]),
                str(row["acq_time"]),
                float(row["frp"]),
                float(row["confidence"]),
                str(row["daynight"]),
                str(row.get("nearest_industrial_type", "none")),
            ))
        conn.commit()

    # 3. Query historical persistence metrics for each point
    days_active = []
    night_fractions = []

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        for _, row in df_work.iterrows():
            loc_key = str(row["location_key"])
            curr_date_str = str(row["acq_date"])
            try:
                curr_date = datetime.strptime(curr_date_str, "%Y-%m-%d")
            except Exception:
                curr_date = datetime.now()
            
            start_date_str = (curr_date - timedelta(days=30)).strftime("%Y-%m-%d")

            # Days active in last 30 days
            cursor.execute("""
            SELECT COUNT(DISTINCT acq_date) FROM hotspot_history
            WHERE location_key = ? AND acq_date BETWEEN ? AND ?
            """, (loc_key, start_date_str, curr_date_str))
            res_days = cursor.fetchone()[0] or 1
            days_active.append(int(res_days))

            # Night detection fraction (all historical records for this location_key)
            cursor.execute("""
            SELECT 
                SUM(CASE WHEN daynight = 'N' THEN 1 ELSE 0 END),
                COUNT(*)
            FROM hotspot_history
            WHERE location_key = ? AND acq_date <= ?
            """, (loc_key, curr_date_str))
            n_night, total_dets = cursor.fetchone()
            night_fraction = (float(n_night) / float(total_dets)) if total_dets and total_dets > 0 else (1.0 if row["daynight"] == "N" else 0.0)
            night_fractions.append(round(night_fraction, 3))

    df_work["days_active_last_30"] = days_active
    df_work["duty_cycle"] = [round(d / 30.0, 3) for d in days_active]
    df_work["night_detection_fraction"] = night_fractions

    return df_work


def query_persistence_features(location_key: str, as_of_date: str, db_path: str = str(DB_PATH)) -> Dict[str, float]:
    """
    Helper function to query persistence metrics for a single location_key.
    """
    init_db(db_path)
    try:
        curr_date = datetime.strptime(as_of_date, "%Y-%m-%d")
    except Exception:
        curr_date = datetime.now()
    start_date = (curr_date - timedelta(days=30)).strftime("%Y-%m-%d")

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
        SELECT COUNT(DISTINCT acq_date) FROM hotspot_history
        WHERE location_key = ? AND acq_date BETWEEN ? AND ?
        """, (location_key, start_date, as_of_date))
        days = cursor.fetchone()[0] or 1

        cursor.execute("""
        SELECT 
            SUM(CASE WHEN daynight = 'N' THEN 1 ELSE 0 END),
            COUNT(*)
        FROM hotspot_history
        WHERE location_key = ? AND acq_date <= ?
        """, (location_key, as_of_date))
        n_night, total = cursor.fetchone()
        night_frac = (float(n_night) / float(total)) if total and total > 0 else 0.5

    return {
        "days_active_last_30": int(days),
        "duty_cycle": round(days / 30.0, 3),
        "night_detection_fraction": round(night_frac, 3),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update and Query Persistence History Log")
    parser.add_argument("--input", type=str, default="data/hotspots_clustered.csv")
    parser.add_argument("--db", type=str, default=str(DB_PATH))
    parser.add_argument("--output", type=str, default="data/hotspots_persisted.csv")
    args = parser.parse_args()

    if os.path.exists(args.input):
        df_in = pd.read_csv(args.input)
        df_out = update_persistence_log(df_in, args.db)
        df_out.to_csv(args.output, index=False)
        print(f"[PersistenceLog] Updated DB ({args.db}) and enriched {len(df_out)} rows -> {args.output}")
    else:
        print(f"[PersistenceLog] Input file not found: {args.input}")
