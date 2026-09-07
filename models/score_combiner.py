"""
Score Combiner & Risk Synthesis Engine
======================================

HOW TO RUN:
    Test score combining logic on test inputs:
        python -m models.score_combiner --event_type industrial_flare --z_score 1.2 --dist_pop 500 --dist_infra 200

INPUTS & ENVIRONMENT:
    - Model A outputs: event_type, event_type_confidence, top_shap_features.
    - Model B outputs: z_score, anomaly_score_normalized.
    - Model C outputs: risk_24h, escalating_24h.
    - Spatial exposure metrics: dist_to_populated_area_m, dist_to_critical_infra_m.

OUTPUT:
    - Synthesized risk dictionary matching the exact inferred schema:
      {
        "event_type": str,
        "event_type_confidence": float,
        "anomaly_score_normalized": float,
        "hazard_weight": float,
        "exposure_factor": float,
        "final_risk_score": float,
        "escalating_24h": bool,
        "risk_24h": float,
        "needs_manual_review": bool,
        "top_shap_features": List[str]
      }

STRICT RULES OBSERVED:
    - Formula: final_risk_score = hazard_weight[event_type] * anomaly_score_normalized * exposure_factor
    - Hazard Weights: wildfire: 1.0, unknown_source: 1.0, industrial_accident: 0.9,
      stockpile_combustion: 0.6, agricultural_burn: 0.3, industrial_flare: 0.1, known_false_positive: 0.0
    - Model A confidence and Model C risk_24h are NEVER multiplied or summed into final_risk_score.
    - Exposed as distinct routing/triage flags: `needs_manual_review` and `escalating_24h`.
"""

import os
import sys
from typing import Dict, Any
import numpy as np

# Ensure module import works when run as script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.settings import HAZARD_WEIGHTS


class ScoreCombiner:
    """
    Combines hazard weight, anomaly score, and exposure into the final risk score.
    """

    def __init__(self):
        self.hazard_weights = HAZARD_WEIGHTS

    def compute_exposure_factor(self, dist_to_populated_m: float, dist_to_infra_m: float) -> float:
        """
        Computes normalized exposure factor (0.0 to 1.0) based on proximity to
        populated residential areas and critical infrastructure pipelines/grids.
        """
        d_pop = max(0.0, float(dist_to_populated_m))
        d_infra = max(0.0, float(dist_to_infra_m))

        # Exponential proximity decay: Closer = higher exposure
        # Populated area decay radius ~2000m, Infrastructure decay radius ~1000m
        pop_component = np.exp(-d_pop / 2000.0)
        infra_component = np.exp(-d_infra / 1000.0)

        # Weighted combination (60% human population, 40% critical infra)
        exposure = 0.60 * pop_component + 0.40 * infra_component
        return round(float(np.clip(exposure, 0.0, 1.0)), 3)

    def combine(
        self,
        model_a_output: Dict[str, Any],
        model_b_output: Dict[str, Any],
        model_c_output: Dict[str, Any],
        dist_to_populated_m: float,
        dist_to_infra_m: float,
    ) -> Dict[str, Any]:
        """
        Combines outputs from Models A, B, and C with spatial exposure.
        """
        event_type = str(model_a_output.get("event_type", "unknown_source"))
        clf_conf = float(model_a_output.get("event_type_confidence", 0.5))
        top_shap = list(model_a_output.get("top_shap_features", []))

        anomaly_norm = float(model_b_output.get("anomaly_score_normalized", 0.0))
        risk_24h = float(model_c_output.get("risk_24h", 0.0))
        escalating_24h = bool(model_c_output.get("escalating_24h", risk_24h >= 0.50))

        # 1. Hazard Weight Lookup
        hazard_w = float(self.hazard_weights.get(event_type, 1.0))

        # 2. Exposure Factor
        exposure_f = self.compute_exposure_factor(dist_to_populated_m, dist_to_infra_m)

        # 3. Final Risk Score (STRICT FORMULA)
        final_score = hazard_w * anomaly_norm * exposure_f

        # 4. Triage and routing flags
        # Flag for manual review if classifier confidence is low or event is unknown/accident
        needs_review = bool(
            clf_conf < 0.60
            or event_type in ["unknown_source", "industrial_accident"]
            or (hazard_w >= 0.9 and final_score >= 0.30)
        )

        return {
            "event_type": event_type,
            "event_type_confidence": round(clf_conf, 2),
            "anomaly_score_normalized": round(anomaly_norm, 2),
            "hazard_weight": round(hazard_w, 2),
            "exposure_factor": round(exposure_f, 2),
            "final_risk_score": round(final_score, 3),
            "escalating_24h": escalating_24h,
            "risk_24h": round(risk_24h, 2),
            "needs_manual_review": needs_review,
            "top_shap_features": top_shap,
        }
