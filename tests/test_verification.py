"""
Automated Verification Suite
============================
Verifies all 7 stages of the pipeline, model artifacts, SHAP attributions,
and FastAPI endpoint response contracts.
"""

import os
import sys
import json
import sqlite3
import pytest
from pathlib import Path
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient

# Ensure root import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.settings import DEFAULT_BBOX, DB_PATH, ARTIFACTS_DIR, OSM_CACHE_PATH, EVENT_TYPES, HAZARD_WEIGHTS
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
from api.main import app


def test_end_to_end_pipeline():
    print("\n--- 1. Testing FIRMS & OSM Ingestion ---")
    df_raw = fetch_firms_hotspots(total_days=7)
    assert not df_raw.empty, "FIRMS hotspot dataframe is empty"
    assert "frp" in df_raw.columns
    assert "daynight" in df_raw.columns

    osm_data = fetch_osm_industrial_and_exposure()
    assert "elements" in osm_data
    assert len(osm_data["elements"]) > 0

    print("\n--- 2. Testing Spatial Join (Boundary Distance) ---")
    df_joined = perform_spatial_join(df_raw, str(OSM_CACHE_PATH))
    assert "dist_to_nearest_industrial_m" in df_joined.columns
    assert "nearest_industrial_type" in df_joined.columns
    assert "dist_to_populated_area_m" in df_joined.columns
    assert "dist_to_critical_infra_m" in df_joined.columns
    # Ensure boundary distance is non-negative
    assert (df_joined["dist_to_nearest_industrial_m"] >= 0).all()

    print("\n--- 3. Testing DBSCAN Spatiotemporal Clustering ---")
    df_clustered = cluster_hotspots_spatiotemporal(df_joined)
    assert "event_id" in df_clustered.columns
    assert "cluster_growth_rate" in df_clustered.columns
    assert df_clustered["event_id"].nunique() > 0

    print("\n--- 4. Testing Persistence Log Database ---")
    df_persisted = update_persistence_log(df_clustered, str(DB_PATH))
    assert "location_key" in df_persisted.columns
    assert "days_active_last_30" in df_persisted.columns
    assert "night_detection_fraction" in df_persisted.columns
    # Verify recurrence_time_of_day_std is NOT present
    assert "recurrence_time_of_day_std" not in df_persisted.columns

    print("\n--- 5. Testing Feature Engineering Matrix ---")
    anomaly_engine = ModelBAnomalyEngine(ARTIFACTS_DIR, DB_PATH)
    anomaly_engine.fit_from_db()
    df_features = build_feature_table(df_persisted, anomaly_engine=anomaly_engine)
    assert "frp_zscore_vs_facility_baseline" in df_features.columns
    assert "month" in df_features.columns
    assert "is_agri_burn_season" in df_features.columns

    print("\n--- 6. Testing 7-Class Pseudo-Labeling Heuristic ---")
    df_labeled = apply_labeling_heuristic(df_features)
    assert "event_type" in df_labeled.columns
    for cls in df_labeled["event_type"].unique():
        assert cls in EVENT_TYPES, f"Unknown class: {cls}"

    print("\n--- 7. Training & Validating Models A, B, C ---")
    model_a = ModelAClassifier(ARTIFACTS_DIR)
    metrics_a = model_a.train(df_labeled)
    assert metrics_a["val_accuracy"] >= 0.0

    sample_row = df_features.iloc[0]
    pred_a = model_a.predict_point(sample_row)
    assert pred_a["event_type"] in EVENT_TYPES
    assert "top_shap_features" in pred_a
    assert len(pred_a["top_shap_features"]) > 0

    # Model B
    z_score, is_anom = anomaly_engine.compute_z_score(
        str(sample_row.get("location_key", "")),
        float(sample_row.get("frp", 20.0)),
        str(sample_row.get("nearest_industrial_type", "none")),
    )
    anomaly_norm = anomaly_engine.compute_anomaly_score_normalized(z_score)
    assert 0.0 <= anomaly_norm <= 1.0

    # Model C
    model_c = ModelCRiskModel(ARTIFACTS_DIR)
    metrics_c = model_c.train(df_labeled)
    pred_c = model_c.predict_point(sample_row)
    assert 0.0 <= pred_c["risk_24h"] <= 1.0
    assert isinstance(pred_c["escalating_24h"], bool)

    print("\n--- 8. Testing Score Combiner ---")
    combiner = ScoreCombiner()
    pred_b = {"anomaly_score_normalized": anomaly_norm}
    score_res = combiner.combine(
        model_a_output=pred_a,
        model_b_output=pred_b,
        model_c_output=pred_c,
        dist_to_populated_m=float(sample_row.get("dist_to_populated_area_m", 1000.0)),
        dist_to_infra_m=float(sample_row.get("dist_to_critical_infra_m", 500.0)),
    )
    assert "final_risk_score" in score_res
    assert 0.0 <= score_res["final_risk_score"] <= 1.0
    assert "needs_manual_review" in score_res
    assert "escalating_24h" in score_res

    print("\n--- 9. Testing FastAPI HTTP Endpoints (including /regions and region selection) ---")
    client = TestClient(app)

    # Test GET /regions
    regions_resp = client.get("/regions")
    assert regions_resp.status_code == 200
    regions_data = regions_resp.json()
    assert isinstance(regions_data, list)
    region_names = [r["name"] for r in regions_data]
    assert "gujarat" in region_names
    assert "maharashtra" in region_names
    assert "odisha" in region_names
    assert "all_india" in region_names
    print(f" -> GET /regions verified: {region_names}")

    # Test GET /hotspots with region query
    resp_reg = client.get("/hotspots?region=gujarat&since_hours=24")
    assert resp_reg.status_code == 200
    assert isinstance(resp_reg.json(), list)

    # Test GET /hotspots with invalid region (should return 400 Bad Request)
    resp_invalid = client.get("/hotspots?region=non_existent_region")
    assert resp_invalid.status_code == 400

    # Test /hotspots default
    resp = client.get("/hotspots?since_hours=24")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    if len(data) > 0:
        first = data[0]
        assert "event_id" in first
        assert "lat" in first
        assert "lon" in first
        # Verify strict observed vs inferred separation
        assert "observed" in first
        assert "inferred" in first
        assert "frp" in first["observed"]
        assert "event_type" in first["inferred"]
        assert "final_risk_score" in first["inferred"]
        assert "top_shap_features" in first["inferred"]

        # Test /hotspot/{event_id}
        eid = first["event_id"]
        detail_resp = client.get(f"/hotspot/{eid}")
        assert detail_resp.status_code == 200
        detail = detail_resp.json()
        assert detail["event_id"] == eid
        assert "full_feature_vector" in detail

    # Test /facility/{location_key}/history
    fac_resp = client.get("/facility/osm_way_1001/history")
    assert fac_resp.status_code == 200
    fac_data = fac_resp.json()
    assert "median_frp" in fac_data
    assert "mad_frp" in fac_data
    assert "history" in fac_data

    print("\n>>> ALL VERIFICATION TESTS PASSED SUCCESSFULLY! <<<")


if __name__ == "__main__":
    test_end_to_end_pipeline()
