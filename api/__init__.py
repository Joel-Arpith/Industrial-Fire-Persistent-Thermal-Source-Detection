"""
FastAPI Application Package
===========================
Provides REST endpoints for querying active hotspots, full feature/SHAP details, and facility history.
"""

from api.schemas import HotspotResponse, HotspotDetailResponse, FacilityHistoryResponse

__all__ = [
    "HotspotResponse",
    "HotspotDetailResponse",
    "FacilityHistoryResponse",
]
