"""
End-to-End Training & Pipeline Orchestrator
===========================================

HOW TO RUN:
    Run the full end-to-end training and data pipeline:
        python -m models.train_all
    Or with custom parameters:
        python -m models.train_all --days 30 --west 72.5 --south 21.0 --east 73.5 --north 22.0

INPUTS & ENVIRONMENT:
    - FIRMS_MAP_KEY: NASA FIRMS Map Key in .env (falls back to realistic synthetic generator if omitted).
    - DEFAULT_BBOX or CLI arguments for target region coordinates.

OUTPUT:
    - Populates `data/hotspots.db` SQLite database with persistence history.
    - Saves trained model artifacts in `artifacts/`:
      * `artifacts/model_a.joblib` (LightGBM 7-class classifier + SHAP explainer)
      * `artifacts/facility_baselines.json` (Statistical Median + MAD baseline table)
      * `artifacts/model_c.joblib` (LightGBM 24h risk escalation model)

STRICT PIPELINE EXECUTION SEQUENCE OBSERVED:
    Step 1: fetch_firms.py + fetch_osm.py -> spatial_join.py (boundary distance, cached OSM extract)
    Step 2: clustering.py (assign event_id BEFORE feature engineering)
    Step 3: persistence_log.py (build/update hotspot_history table)
    Step 4: feature_engineering.py (full canonical feature table)
    Step 5: labeling_heuristic.py (generate 7-class pseudo-labels)
    Step 6: Train Model A, fit Model B baselines, train Model C (with duty-cycle filter)
    Step 7: Validate Score Combiner on test events
"""

import os
import sys
import time
import argparse

# Ensure module import works when run as script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.settings import DEFAULT_BBOX, DB_PATH, ARTIFACTS_DIR, DATA_DIR, OSM_CACHE_PATH
from data_ingestion.fetch_firms import fetch_firms_hotspots
from data_ingestion.fetch_osm import fetch_osm_industrial_and_exposure
from processing.spatial_join import perform_spatial_join
from processing.clustering import cluster_hotspots_spatiotemporal
from processing.persistence_log import update_persistence_log
from processing.feature_engineering import build_feature_table
from processing.labeling_heuristic import apply_labeling_heuristic
from models.model_a_classifier import ModelAClassifier
from models.model_b_anomaly_engine import ModelBAnomalyEngine
from models.model_c_risk_model import ModelCRiskModel
from models.score_combiner import ScoreCombiner


def run_training_pipeline(
    west: float = DEFAULT_BBOX["west"],
    south: float = DEFAULT_BBOX["south"],
    east: float = DEFAULT_BBOX["east"],
    north: float = DEFAULT_BBOX["north"],
    days: int = 30,
) -> None:
    """
    Executes the 7-step data ingestion, processing, baseline fitting, and model training sequence.
    """
    start_time = time.time()
    print("=" * 70)
    print("  INDUSTRIAL FIRE & PERSISTENT THERMAL SOURCE DETECTION PIPELINE")
    print("=" * 70)
    print(f"Target Bounding Box: West={west}, South={south}, East={east}, North={north}")
    print(f"Historical Window:   {days} days")
    print(f"Artifacts Path:      {ARTIFACTS_DIR}")
    print(f"Database Path:       {DB_PATH}\n")

    # -------------------------------------------------------------------------
    # STEP 1: Ingestion & Spatial Join (Boundary Distances)
    # -------------------------------------------------------------------------
    print("\n>>> [STEP 1/7] Ingesting NASA FIRMS Hotspots & OSM Infrastructure Geometries...")
    df_raw = fetch_firms_hotspots(west=west, south=south, east=east, north=north, total_days=days)
    if df_raw.empty:
        raise RuntimeError("No hotspot data available to train or process pipeline.")

    # Cache OSM infrastructure
    fetch_osm_industrial_and_exposure(west=west, south=south, east=east, north=north)
    
    # Boundary distance spatial join
    df_joined = perform_spatial_join(df_raw, str(OSM_CACHE_PATH))
    print(f" -> Spatial Join complete: {len(df_joined)} points processed with boundary distances.")

    # -------------------------------------------------------------------------
    # STEP 2: Spatiotemporal Clustering (DBSCAN) - BEFORE Feature Engineering
    # -------------------------------------------------------------------------
    print("\n>>> [STEP 2/7] Running Spatiotemporal DBSCAN Clustering (Assign event_ids)...")
    df_clustered = cluster_hotspots_spatiotemporal(df_joined)
    n_events = df_clustered["event_id"].nunique()
    print(f" -> Clustering complete: Formed {n_events} distinct thermal event clusters.")

    # -------------------------------------------------------------------------
    # STEP 3: Build/Update Persistence History Log (hotspot_history Table)
    # -------------------------------------------------------------------------
    print("\n>>> [STEP 3/7] Updating Hotspot Persistence History Database...")
    df_persisted = update_persistence_log(df_clustered, str(DB_PATH))
    print(f" -> Persistence log updated. Computed days_active_last_30 and night_detection_fraction.")

    # -------------------------------------------------------------------------
    # STEP 4: Build Canonical Feature Engineering Table
    # -------------------------------------------------------------------------
    print("\n>>> [STEP 4/7] Constructing Full Feature Engineering Matrix...")
    # Initialize Model B anomaly engine to compute z-scores for feature table
    anomaly_engine = ModelBAnomalyEngine(ARTIFACTS_DIR, DB_PATH)
    anomaly_engine.fit_from_db()
    
    df_features = build_feature_table(df_persisted, anomaly_engine=anomaly_engine)
    feature_csv_path = DATA_DIR / "feature_table.csv"
    df_features.to_csv(feature_csv_path, index=False)
    print(f" -> Feature engineering complete: Saved {len(df_features)} rows to {feature_csv_path}")

    # -------------------------------------------------------------------------
    # STEP 5: Apply Heuristic Pseudo-Labeling (7 Classes)
    # -------------------------------------------------------------------------
    print("\n>>> [STEP 5/7] Bootstrapping Ground Truth via 7-Class Labeling Heuristic...")
    df_labeled = apply_labeling_heuristic(df_features)
    labeled_csv_path = DATA_DIR / "labeled_dataset.csv"
    df_labeled.to_csv(labeled_csv_path, index=False)
    print(f" -> Pseudo-labeling complete. Class distribution:")
    for cls_name, count in df_labeled["event_type"].value_counts().items():
        print(f"    * {cls_name:<25}: {count:>4} samples")

    # -------------------------------------------------------------------------
    # STEP 6: Train Models A, B, and C
    # -------------------------------------------------------------------------
    print("\n>>> [STEP 6/7] Training Models A, B, and C...")

    # Model A: LightGBM Multiclass Classifier + SHAP TreeExplainer
    print("\n--- Training Model A (LightGBM Multi-Class Event Classifier) ---")
    model_a = ModelAClassifier(ARTIFACTS_DIR)
    metrics_a = model_a.train(df_labeled)
    print(f" -> Model A trained. Validation Accuracy: {metrics_a.get('val_accuracy', 0.0):.2%}")

    # Model C: LightGBM 24h Escalation Risk Model (Filtered duty_cycle < 0.2)
    print("\n--- Training Model C (LightGBM 24h Risk Escalation Model) ---")
    model_c = ModelCRiskModel(ARTIFACTS_DIR)
    metrics_c = model_c.train(df_labeled)
    print(f" -> Model C trained. Mean validation escalation probability: {metrics_c.get('mean_risk_val', 0.0):.2%}")

    # -------------------------------------------------------------------------
    # STEP 7: Score Combiner Verification
    # -------------------------------------------------------------------------
    print("\n>>> [STEP 7/7] Verifying Score Combiner on Test Samples...")
    combiner = ScoreCombiner()
    sample_row = df_features.iloc[0]
    pred_a = model_a.predict_point(sample_row)
    z_score, _ = anomaly_engine.compute_z_score(
        sample_row.get("location_key", ""),
        float(sample_row.get("frp", 0.0)),
        str(sample_row.get("nearest_industrial_type", "none")),
    )
    pred_b = {"anomaly_score_normalized": anomaly_engine.compute_anomaly_score_normalized(z_score)}
    pred_c = model_c.predict_point(sample_row)

    combined_score = combiner.combine(
        model_a_output=pred_a,
        model_b_output=pred_b,
        model_c_output=pred_c,
        dist_to_populated_m=float(sample_row.get("dist_to_populated_area_m", 5000.0)),
        dist_to_infra_m=float(sample_row.get("dist_to_critical_infra_m", 5000.0)),
    )

    print(" -> Sample End-to-End Scored Point:")
    import json
    print(json.dumps(combined_score, indent=2))

    elapsed = round(time.time() - start_time, 2)
    print("\n" + "=" * 70)
    print(f"  ALL PIPELINE STEPS & MODELS TRAINED SUCCESSFULLY IN {elapsed}s")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End Industrial Fire Pipeline & Model Trainer")
    parser.add_argument("--west", type=float, default=DEFAULT_BBOX["west"])
    parser.add_argument("--south", type=float, default=DEFAULT_BBOX["south"])
    parser.add_argument("--east", type=float, default=DEFAULT_BBOX["east"])
    parser.add_argument("--north", type=float, default=DEFAULT_BBOX["north"])
    parser.add_argument("--days", type=int, default=30)
    args = parser.parse_args()

    run_training_pipeline(
        west=args.west,
        south=args.south,
        east=args.east,
        north=args.north,
        days=args.days,
    )
