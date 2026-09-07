"""
Spatiotemporal Clustering Module (DBSCAN)
=========================================

HOW TO RUN:
    Test space-time clustering on joined hotspots:
        python -m processing.clustering --input data/hotspots_joined.csv --output data/hotspots_clustered.csv

INPUTS & ENVIRONMENT:
    - Hotspots DataFrame containing latitude, longitude, acq_date, acq_time, frp.

OUTPUT:
    - DataFrame enriched with:
      [event_id, cluster_growth_rate]

STRICT RULES OBSERVED:
    - Executed BEFORE feature engineering, not after.
    - Uses DBSCAN with combined space-time distance (eps ≈ 750m spatial, 12h temporal window).
    - Assigns stable `event_id` (e.g., `evt_YYYYMMDD_XXXX`) grouping adjacent detections.
    - Computes `cluster_growth_rate` (rate of change in footprint/FRP across time).
"""

import os
import sys
import argparse
from datetime import datetime
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from shapely.geometry import MultiPoint

# Ensure module import works when run as script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.settings import DBSCAN_EPS_METERS, DBSCAN_TIME_HOURS


def parse_timestamp(acq_date: str, acq_time: str) -> datetime:
    """
    Parses FIRMS acq_date ('YYYY-MM-DD') and acq_time ('0130' or '1330' or 130) into datetime.
    """
    date_str = str(acq_date).strip()
    time_str = str(acq_time).strip().zfill(4)
    try:
        dt_str = f"{date_str} {time_str[:2]}:{time_str[2:4]}"
        return datetime.strptime(dt_str, "%Y-%m-%d %H:%M")
    except Exception:
        try:
            return datetime.strptime(date_str, "%Y-%m-%d")
        except Exception:
            return datetime(2026, 1, 1)


def custom_spacetime_distance_matrix(coords: np.ndarray, timestamps_hours: np.ndarray) -> np.ndarray:
    """
    Computes normalized space-time pairwise distance matrix:
    d = sqrt( (spatial_dist_meters / 750m)^2 + (temporal_dist_hours / 12h)^2 )
    """
    n = len(coords)
    dist_matrix = np.zeros((n, n), dtype=np.float32)
    
    # Earth radius in meters
    R = 6371000.0
    lat_rad = np.radians(coords[:, 0])
    lon_rad = np.radians(coords[:, 1])

    for i in range(n):
        # Haversine spatial distance
        dlat = lat_rad - lat_rad[i]
        dlon = lon_rad - lon_rad[i]
        a = np.sin(dlat / 2.0)**2 + np.cos(lat_rad[i]) * np.cos(lat_rad) * np.sin(dlon / 2.0)**2
        c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        spatial_meters = R * c

        # Temporal distance in hours
        temporal_hours = np.abs(timestamps_hours - timestamps_hours[i])

        # Combined space-time metric
        combined_metric = np.sqrt(
            (spatial_meters / DBSCAN_EPS_METERS) ** 2 +
            (temporal_hours / DBSCAN_TIME_HOURS) ** 2
        )
        dist_matrix[i, :] = combined_metric

    return dist_matrix


def cluster_hotspots_spatiotemporal(df_hotspots: pd.DataFrame) -> pd.DataFrame:
    """
    Groups adjacent detections into stable event_id clusters using DBSCAN
    and computes cluster growth rates.
    """
    if df_hotspots.empty:
        df_out = df_hotspots.copy()
        df_out["event_id"] = []
        df_out["cluster_growth_rate"] = []
        return df_out

    df_work = df_hotspots.copy()

    # Parse timestamps into continuous epoch hours
    dts = [parse_timestamp(d, t) for d, t in zip(df_work["acq_date"], df_work["acq_time"])]
    min_dt = min(dts)
    hours_from_start = np.array([(dt - min_dt).total_seconds() / 3600.0 for dt in dts], dtype=np.float64)
    coords = df_work[["latitude", "longitude"]].values

    # Run DBSCAN with precomputed space-time matrix
    if len(coords) > 1:
        dist_mat = custom_spacetime_distance_matrix(coords, hours_from_start)
        # eps=1.0 on normalized space-time metric
        db = DBSCAN(eps=1.0, min_samples=1, metric="precomputed")
        cluster_labels = db.fit_predict(dist_mat)
    else:
        cluster_labels = np.array([0])

    # Assign stable event_ids: evt_YYYYMMDD_XXXX
    event_ids = []
    for idx, (dt, label) in enumerate(zip(dts, cluster_labels)):
        date_prefix = dt.strftime("%Y%m%d")
        event_ids.append(f"evt_{date_prefix}_{label:04d}")

    df_work["event_id"] = event_ids
    df_work["_timestamp"] = dts

    # Compute cluster growth rates
    # Growth rate: (Total FRP at t2 - Total FRP at t1) / dt_hours (or footprint expansion)
    growth_rates = {}
    for eid, group in df_work.groupby("event_id"):
        if len(group) <= 1:
            growth_rates[eid] = 0.0
            continue
        
        group_sorted = group.sort_values(by="_timestamp")
        # Measure time span
        t_span = (group_sorted["_timestamp"].iloc[-1] - group_sorted["_timestamp"].iloc[0]).total_seconds() / 3600.0
        if t_span < 1.0:
            t_span = 1.0

        # Measure FRP rate of change
        frp_start = float(group_sorted["frp"].iloc[0])
        frp_end = float(group_sorted["frp"].iloc[-1])
        delta_frp = (frp_end - frp_start) / max(frp_start, 1.0)
        
        # Spatial footprint growth (bounding box diagonal)
        pts = group_sorted[["longitude", "latitude"]].values
        if len(pts) >= 3:
            hull = MultiPoint(pts).convex_hull
            area_growth = float(hull.area) * 1e6 # Approx square meters
        else:
            area_growth = float(np.std(pts[:, 0]) + np.std(pts[:, 1])) * 111000.0

        rate = (delta_frp + (area_growth / 1000.0)) / t_span
        growth_rates[eid] = round(float(np.clip(rate, -10.0, 50.0)), 3)

    df_work["cluster_growth_rate"] = df_work["event_id"].map(growth_rates)
    df_work.drop(columns=["_timestamp"], inplace=True)

    return df_work


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform Spatiotemporal Clustering (DBSCAN)")
    parser.add_argument("--input", type=str, default="data/hotspots_joined.csv")
    parser.add_argument("--output", type=str, default="data/hotspots_clustered.csv")
    args = parser.parse_args()

    if os.path.exists(args.input):
        df_in = pd.read_csv(args.input)
        df_out = cluster_hotspots_spatiotemporal(df_in)
        df_out.to_csv(args.output, index=False)
        print(f"[Clustering] Assigned {df_out['event_id'].nunique()} unique event clusters -> {args.output}")
    else:
        print(f"[Clustering] Input file not found: {args.input}")
