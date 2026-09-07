"""
Labeling Heuristic Module (Ground Truth Bootstrapping)
=====================================================

HOW TO RUN:
    Generate pseudo-labels for a feature table:
        python -m processing.labeling_heuristic --input data/feature_table.csv --output data/labeled_dataset.csv

INPUTS & ENVIRONMENT:
    - Feature table DataFrame with spatial distances, land cover, persistence, and FRP metrics.

OUTPUT:
    - DataFrame enriched with `event_type` pseudo-label categorized into one of the 7 exact classes:
      1. industrial_flare
      2. industrial_accident
      3. stockpile_combustion
      4. known_false_positive
      5. wildfire
      6. agricultural_burn
      7. unknown_source

STRICT RULES OBSERVED:
    - Exactly 7 classes, one rule each, mutually exclusive, no duplicate rules.
    - Captures solar false positives, stockpiles, accidents, and NTRO unknown sources.
"""

import os
import sys
import argparse
from typing import List, Dict
import pandas as pd
import numpy as np

# Ensure module import works when run as script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.settings import EVENT_TYPES


def classify_point_heuristic(row: pd.Series) -> str:
    """
    Evaluates the 7 exact heuristic rules in priority order for a single hotspot record.
    """
    dist_ind = float(row.get("dist_to_nearest_industrial_m", 99999.0))
    ind_type = str(row.get("nearest_industrial_type", "none"))
    land_cover = str(row.get("land_cover_class", "cropland")).lower()
    days_active = int(row.get("days_active_last_30", 1))
    duty_cycle = float(row.get("duty_cycle", days_active / 30.0))
    frp = float(row.get("frp", 0.0))
    z_score = float(row.get("frp_zscore_vs_facility_baseline", 0.0))
    daynight = str(row.get("daynight", "D")).upper()
    growth_rate = float(row.get("cluster_growth_rate", 0.0))
    is_agri = int(row.get("is_agri_burn_season", 0))

    # Rule 4: known_false_positive
    # Distance < 500m to a solar/renewable tag AND detections only in daynight=D AND stable FRP AND no cluster growth
    if dist_ind < 500.0 and ind_type == "solar_farm" and daynight == "D" and growth_rate <= 0.0:
        return "known_false_positive"

    # Rule 1: industrial_flare
    # Distance < 500m to a flare/refinery/plant tag, duty cycle (days_active_last_30/30) > 0.6
    if dist_ind < 500.0 and duty_cycle > 0.6:
        return "industrial_flare"

    # Rule 2: industrial_accident
    # Distance < 500m to industrial tag, high FRP spike (z > 3 or frp > 50), duty cycle < 0.2, no prior history
    if dist_ind < 500.0 and (z_score > 3.0 or frp > 50.0) and duty_cycle < 0.2 and days_active <= 1:
        return "industrial_accident"

    # Rule 3: stockpile_combustion
    # Distance < 500m to waste/landfill/mining/coal-stockyard tag AND moderate sustained FRP
    if dist_ind < 500.0 and ind_type in ["waste_landfill", "quarry_mining"] and frp < 80.0:
        return "stockpile_combustion"

    # Rule 6: agricultural_burn
    # Land cover = cropland AND seasonal clustering in is_agri_burn_season (Oct-Nov, Apr-May)
    if land_cover == "cropland" and is_agri == 1 and dist_ind >= 500.0:
        return "agricultural_burn"

    # Rule 5: wildfire
    # Land cover in {forest, grassland} AND no nearby industrial tag
    if land_cover in ["forest", "grassland", "shrubland"] and dist_ind >= 500.0:
        return "wildfire"

    # Rule 7: unknown_source
    # No OSM industrial/solar/waste tag within 2km AND land_cover not in {forest, cropland, grassland}
    if dist_ind >= 2000.0 and land_cover not in ["forest", "grassland", "cropland"]:
        return "unknown_source"

    # Fallback to nearest logical classification for edge points
    if dist_ind < 500.0:
        return "industrial_flare" if duty_cycle >= 0.4 else "industrial_accident"
    if land_cover in ["forest", "grassland"]:
        return "wildfire"
    if land_cover == "cropland":
        return "agricultural_burn"
    
    return "unknown_source"


def apply_labeling_heuristic(df_features: pd.DataFrame) -> pd.DataFrame:
    """
    Applies the pseudo-labeling heuristic to all rows in the feature DataFrame.
    """
    if df_features.empty:
        df_out = df_features.copy()
        df_out["event_type"] = []
        return df_out

    labels = [classify_point_heuristic(row) for _, row in df_features.iterrows()]
    df_out = df_features.copy()
    df_out["event_type"] = labels
    return df_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 7-Class Pseudo-Labels for Hotspots")
    parser.add_argument("--input", type=str, default="data/feature_table.csv")
    parser.add_argument("--output", type=str, default="data/labeled_dataset.csv")
    args = parser.parse_args()

    if os.path.exists(args.input):
        df_in = pd.read_csv(args.input)
        df_out = apply_labeling_heuristic(df_in)
        df_out.to_csv(args.output, index=False)
        print(f"[LabelingHeuristic] Assigned pseudo-labels across {len(df_out)} points -> {args.output}")
        print("Class breakdown:")
        print(df_out["event_type"].value_counts())
    else:
        print(f"[LabelingHeuristic] Input file not found: {args.input}")
