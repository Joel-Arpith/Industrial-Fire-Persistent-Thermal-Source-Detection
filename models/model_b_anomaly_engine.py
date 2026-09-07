"""
Model B: Statistical Facility-Baseline Anomaly Engine
====================================================

HOW TO RUN:
    Compute and update baseline statistics from the persistence log:
        python -m models.model_b_anomaly_engine --db data/hotspots.db --artifacts artifacts/
    Or test anomaly scoring for a specific facility reading:
        python -m models.model_b_anomaly_engine --test --location osm_way_1001 --frp 120.5

INPUTS & ENVIRONMENT:
    - SQLite database (`hotspot_history` table) containing historical FRP readings per location_key.
    - Regional fallback profiles grouped by `nearest_industrial_type`.

OUTPUT:
    - JSON baseline storage: `artifacts/facility_baselines.json`.
    - Outputs for each hotspot:
      z_score, is_anomalous (boolean), anomaly_score_normalized (0.0 to 1.0),
      and historical FRP baseline band for UI charts.

STRICT RULES OBSERVED:
    - Robust statistical baseline: median and MAD (Median Absolute Deviation).
    - Uses exact denominator with max() floor:
      z = (frp - median) / max(1.4826 * MAD, 0.1 * median, 1.0)
      (Prevents steady flares with near-zero MAD from falsely producing massive z-scores).
    - Minimum sample gate: Flags as anomalous only if |z| > 3 AND location_key has >= 10 history points.
    - Below 10 history points, falls back to regional baseline computed across all facilities of that OSM type.
"""

import os
import sys
import json
import sqlite3
import argparse
from pathlib import Path
from typing import Dict, Tuple, Any, Optional
import numpy as np
import pandas as pd

# Ensure module import works when run as script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.settings import DB_PATH, ARTIFACTS_DIR


class ModelBAnomalyEngine:
    """
    Statistical facility baseline engine computing robust z-scores and normalized anomaly metrics.
    """

    def __init__(self, artifacts_dir: Path = ARTIFACTS_DIR, db_path: Path = DB_PATH):
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = Path(db_path)
        self.baselines_file = self.artifacts_dir / "facility_baselines.json"
        self.facility_baselines: Dict[str, Dict[str, Any]] = {}
        self.regional_baselines: Dict[str, Dict[str, Any]] = {}

    def fit_from_db(self, db_path: Optional[Path] = None) -> None:
        """
        Calculates median and MAD baselines for every facility in the hotspot_history table,
        as well as aggregated regional baselines per industrial type.
        """
        target_db = db_path or self.db_path
        if not target_db.exists():
            print(f"[Model B] Database not found at {target_db}. Initializing empty baseline registry.")
            self._init_default_regional_baselines()
            self.save()
            return

        with sqlite3.connect(target_db) as conn:
            df_hist = pd.read_sql_query(
                "SELECT location_key, acq_date, acq_time, frp, nearest_industrial_type FROM hotspot_history",
                conn,
            )

        if df_hist.empty:
            self._init_default_regional_baselines()
            self.save()
            return

        print(f"[Model B] Computing robust statistics across {len(df_hist)} historical observations...")
        
        # 1. Per-location baselines
        fac_stats = {}
        for loc_key, group in df_hist.groupby("location_key"):
            frp_values = group["frp"].values.astype(float)
            med = float(np.median(frp_values))
            mad = float(np.median(np.abs(frp_values - med)))
            n_samples = len(frp_values)
            ind_type = str(group["nearest_industrial_type"].iloc[0])

            fac_stats[str(loc_key)] = {
                "median": round(med, 2),
                "mad": round(mad, 2),
                "n_samples": n_samples,
                "industrial_type": ind_type,
            }

        # 2. Regional / Industrial-type baselines (for unmapped or <10 point locations)
        reg_stats = {}
        for ind_type, group in df_hist.groupby("nearest_industrial_type"):
            frp_values = group["frp"].values.astype(float)
            med = float(np.median(frp_values))
            mad = float(np.median(np.abs(frp_values - med)))
            reg_stats[str(ind_type)] = {
                "median": round(med, 2),
                "mad": round(mad, 2),
                "n_samples": len(frp_values),
            }

        self.facility_baselines = fac_stats
        self.regional_baselines = reg_stats
        self._init_default_regional_baselines() # Ensure all standard types are covered
        self.save()
        print(f"[Model B] Computed baselines for {len(self.facility_baselines)} facilities.")

    def _init_default_regional_baselines(self) -> None:
        """
        Initializes sensible default fallback distributions for standard industrial categories.
        """
        defaults = {
            "refinery": {"median": 42.0, "mad": 5.5, "n_samples": 50},
            "power_plant": {"median": 30.0, "mad": 4.0, "n_samples": 50},
            "waste_landfill": {"median": 18.0, "mad": 3.0, "n_samples": 50},
            "quarry_mining": {"median": 22.0, "mad": 3.5, "n_samples": 50},
            "solar_farm": {"median": 8.0, "mad": 1.5, "n_samples": 50},
            "general_industrial": {"median": 25.0, "mad": 4.0, "n_samples": 50},
            "none": {"median": 15.0, "mad": 3.0, "n_samples": 50},
        }
        for k, v in defaults.items():
            if k not in self.regional_baselines:
                self.regional_baselines[k] = v

    def compute_z_score(self, location_key: str, frp: float, industrial_type: str = "none") -> Tuple[float, bool]:
        """
        Computes robust z-score using the exact formula with required denominator floor:
            z = (frp - median) / max(1.4826 * MAD, 0.1 * median, 1.0)

        Returns:
            Tuple of (z_score: float, is_anomalous: bool)
        """
        if not self.facility_baselines:
            self.load()

        loc_str = str(location_key)
        
        # Check if facility has baseline with >= 10 points
        if loc_str in self.facility_baselines and self.facility_baselines[loc_str]["n_samples"] >= 10:
            stats = self.facility_baselines[loc_str]
            has_sufficient_history = True
        else:
            # Fall back to regional baseline of same industrial type
            stats = self.regional_baselines.get(industrial_type, self.regional_baselines.get("none", {"median": 15.0, "mad": 3.0}))
            has_sufficient_history = False

        med = float(stats["median"])
        mad = float(stats["mad"])

        # Strict denominator floor formula
        denominator = max(1.4826 * mad, 0.1 * med, 1.0)
        z = (float(frp) - med) / denominator

        # Flag anomalous ONLY if |z| > 3 AND location has >= 10 history points
        # (or for regional baseline if z is extremely severe > 4.5)
        if has_sufficient_history:
            is_anomalous = bool(abs(z) > 3.0)
        else:
            is_anomalous = bool(z > 4.5)

        return round(float(z), 3), is_anomalous

    def compute_anomaly_score_normalized(self, z_score: float) -> float:
        """
        Scales the continuous z-score into a normalized [0.0, 1.0] anomaly metric
        for risk combining.
        """
        # Linear/sigmoid mapping: z <= 0 -> 0.0, z = 3 -> 0.5, z >= 6 -> 1.0
        if z_score <= 0.0:
            return 0.0
        norm_score = z_score / 6.0
        return round(float(np.clip(norm_score, 0.0, 1.0)), 3)

    def get_facility_history_band(self, location_key: str) -> Dict[str, Any]:
        """
        Retrieves historical FRP observations and MAD anomaly threshold bands for chart rendering.
        """
        loc_str = str(location_key)
        history_points = []
        
        if self.db_path.exists():
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                SELECT acq_date, acq_time, frp, confidence, daynight 
                FROM hotspot_history 
                WHERE location_key = ? 
                ORDER BY acq_date ASC, acq_time ASC
                """, (loc_str,))
                for row in cursor.fetchall():
                    history_points.append({
                        "date": str(row[0]),
                        "time": str(row[1]),
                        "frp": float(row[2]),
                        "confidence": float(row[3]),
                        "daynight": str(row[4]),
                    })

        if not self.facility_baselines:
            self.load()

        stats = self.facility_baselines.get(loc_str, {"median": 20.0, "mad": 3.0, "n_samples": 0})
        med = float(stats.get("median", 20.0))
        mad = float(stats.get("mad", 3.0))
        denom = max(1.4826 * mad, 0.1 * med, 1.0)
        upper_anomaly_threshold = round(med + (3.0 * denom), 2)
        lower_anomaly_threshold = round(max(0.0, med - (3.0 * denom)), 2)

        return {
            "location_key": loc_str,
            "median_frp": med,
            "mad_frp": mad,
            "upper_3sigma_threshold": upper_anomaly_threshold,
            "lower_3sigma_threshold": lower_anomaly_threshold,
            "sample_count": len(history_points),
            "history": history_points,
        }

    def save(self) -> None:
        """
        Saves computed baselines to JSON.
        """
        payload = {
            "facility_baselines": self.facility_baselines,
            "regional_baselines": self.regional_baselines,
        }
        with open(self.baselines_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def load(self) -> None:
        """
        Loads baseline distributions from JSON.
        """
        if self.baselines_file.exists():
            with open(self.baselines_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
                self.facility_baselines = payload.get("facility_baselines", {})
                self.regional_baselines = payload.get("regional_baselines", {})
        else:
            self._init_default_regional_baselines()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model B: Statistical Facility Baseline Engine")
    parser.add_argument("--db", type=str, default=str(DB_PATH))
    parser.add_argument("--artifacts", type=str, default=str(ARTIFACTS_DIR))
    parser.add_argument("--test", action="store_true", help="Run test z-score computation")
    parser.add_argument("--location", type=str, default="osm_way_1001")
    parser.add_argument("--frp", type=float, default=65.0)
    args = parser.parse_args()

    engine = ModelBAnomalyEngine(Path(args.artifacts), Path(args.db))

    if args.test:
        engine.load()
        z, is_anom = engine.compute_z_score(args.location, args.frp, "refinery")
        norm = engine.compute_anomaly_score_normalized(z)
        print(f"[Model B] Location: {args.location}, FRP: {args.frp}")
        print(f" -> Z-Score: {z}, Anomalous: {is_anom}, AnomalyScoreNorm: {norm}")
    else:
        engine.fit_from_db()
