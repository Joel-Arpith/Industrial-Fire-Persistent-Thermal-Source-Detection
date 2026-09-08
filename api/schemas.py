"""
Pydantic Response Schemas Module
================================

HOW TO RUN:
    This module defines the response data models used by the FastAPI server in api/main.py.
    To validate schema definitions:
        python -m api.schemas

INPUTS:
    - Raw dictionary payloads from pipeline and inference engines.

OUTPUT:
    - Strictly typed Pydantic models enforcing the exact observed vs inferred separation required by the spec.

STRICT RULES OBSERVED:
    - Response structure separates `observed` (satellite evidence: FRP, confidence, acq_date, industrial type/dist)
      from `inferred` (model output: event_type, anomaly_norm, hazard_weight, exposure, final_risk_score,
      escalating_24h, risk_24h, needs_manual_review, top_shap_features).
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class ObservedData(BaseModel):
    """
    Direct satellite and geospatial observations (pure evidence, zero model inference).
    """
    frp: float = Field(..., description="Fire Radiative Power in MW from FIRMS sensor")
    confidence: float = Field(..., description="Detection confidence percentage (0-100)")
    acq_date: str = Field(..., description="Acquisition date (YYYY-MM-DD)")
    nearest_industrial_type: str = Field(..., description="OSM facility type (refinery, power_plant, etc.)")
    dist_to_industrial_m: float = Field(..., description="Exact distance to nearest facility boundary in meters")


class InferredData(BaseModel):
    """
    Synthesized machine learning and statistical inferences.
    """
    event_type: str = Field(..., description="Predicted class from Model A")
    event_type_confidence: float = Field(..., description="Classification probability (0.0 - 1.0)")
    anomaly_score_normalized: float = Field(..., description="Model B normalized robust z-score deviation (0.0 - 1.0)")
    hazard_weight: float = Field(..., description="Deterministic hazard weight multiplier for this class")
    exposure_factor: float = Field(..., description="Spatial exposure factor derived from populated area & infra distance")
    final_risk_score: float = Field(..., description="Combined risk score: hazard_weight * anomaly_score * exposure_factor")
    escalating_24h: bool = Field(..., description="Model C boolean flag indicating spread/intensification risk")
    risk_24h: float = Field(..., description="Model C escalation probability (0.0 - 1.0)")
    needs_manual_review: bool = Field(..., description="Routing flag for analyst review queue")
    top_shap_features: List[str] = Field(..., description="Top feature drivers from SHAP TreeExplainer")


class HotspotResponse(BaseModel):
    """
    Canonical response schema for each thermal hotspot detection point in /hotspots.
    """
    event_id: str
    lat: float
    lon: float
    observed: ObservedData
    inferred: InferredData


class HotspotDetailResponse(BaseModel):
    """
    Full diagnostic feature vector, SHAP explanation breakdown, and persistence context for /hotspot/{event_id}.
    """
    event_id: str
    location_key: str
    lat: float
    lon: float
    observed: ObservedData
    inferred: InferredData
    full_feature_vector: Dict[str, Any]
    class_probabilities: Dict[str, float]


class FacilityHistoryPoint(BaseModel):
    """
    Single historical time-series observation at a facility.
    """
    date: str
    time: str
    frp: float
    confidence: float
    daynight: str


class FacilityHistoryResponse(BaseModel):
    """
    Time series of FRP readings overlaid with MAD baseline thresholds for /facility/{location_key}/history.
    """
    location_key: str
    median_frp: float
    mad_frp: float
    upper_3sigma_threshold: float
    lower_3sigma_threshold: float
    sample_count: int
    history: List[FacilityHistoryPoint]


class RegionBBox(BaseModel):
    """
    Bounding box coordinates in degrees.
    """
    west: float = Field(..., description="Western longitude boundary")
    south: float = Field(..., description="Southern latitude boundary")
    east: float = Field(..., description="Eastern longitude boundary")
    north: float = Field(..., description="Northern latitude boundary")


class RegionInfo(BaseModel):
    """
    Predefined Indian region metadata for frontend selection dropdowns.
    """
    name: str = Field(..., description="Region key/identifier (e.g. 'gujarat', 'maharashtra')")
    label: str = Field(..., description="Human-readable title and coverage cities")
    bbox: RegionBBox = Field(..., description="Bounding box definition")
    description: Optional[str] = Field("", description="Regional industrial profile description")


if __name__ == "__main__":
    sample = HotspotResponse(
        event_id="evt_20260904_0032",
        lat=21.17,
        lon=72.83,
        observed=ObservedData(
            frp=42.3,
            confidence=87,
            acq_date="2026-09-04",
            nearest_industrial_type="refinery",
            dist_to_industrial_m=210,
        ),
        inferred=InferredData(
            event_type="industrial_flare",
            event_type_confidence=0.91,
            anomaly_score_normalized=0.12,
            hazard_weight=0.1,
            exposure_factor=0.4,
            final_risk_score=0.048,
            escalating_24h=False,
            risk_24h=0.08,
            needs_manual_review=False,
            top_shap_features=["dist_to_nearest_industrial_m", "night_detection_fraction", "daynight"],
        ),
    )
    print("Schema Validation Test Successful:")
    print(sample.model_dump_json(indent=2))
