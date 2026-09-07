"""
Configuration and Settings Module
=================================

HOW TO RUN:
    This module is imported by all pipeline and API components.
    To test configuration loading independently:
        python -m config.settings
    Or from the project root:
        python config/settings.py

ENVIRONMENT VARIABLES / INPUTS:
    - FIRMS_MAP_KEY: NASA FIRMS Map Key (32-character string from NASA EOSDIS).
    - DB_PATH: Path to SQLite database file (default: "data/hotspots.db").
    - DEMO_BBOX_WEST, DEMO_BBOX_SOUTH, DEMO_BBOX_EAST, DEMO_BBOX_NORTH: Default bounding box coordinates.
    - OVERPASS_URL: Overpass API interpreter URL.
    - OSM_CACHE_PATH: Path to cached OSM GeoJSON/JSON extract.

OUTPUT:
    Exposes typed configuration constants, paths, hazard weights, and model hyperparameters.
"""

import os
from pathlib import Path
from typing import Dict, List
from dotenv import load_dotenv

# Base Directory paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DB_DIR = BASE_DIR / "db"

# Ensure runtime directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Load environment variables from .env file
load_dotenv(dotenv_path=BASE_DIR / ".env")

# NASA FIRMS API Settings
FIRMS_MAP_KEY: str = os.getenv("FIRMS_MAP_KEY", "")
FIRMS_BASE_URL: str = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
FIRMS_SOURCE: str = "VIIRS_SNPP_NRT"

# Default Bounding Box for Industrial Monitoring: Gujarat / Hazira / Dahej Industrial Belt
DEFAULT_BBOX: Dict[str, float] = {
    "west": float(os.getenv("DEMO_BBOX_WEST", 72.5)),
    "south": float(os.getenv("DEMO_BBOX_SOUTH", 21.0)),
    "east": float(os.getenv("DEMO_BBOX_EAST", 73.5)),
    "north": float(os.getenv("DEMO_BBOX_NORTH", 22.0)),
}

# OpenStreetMap / Overpass Settings
OVERPASS_URL: str = os.getenv("OVERPASS_URL", "https://overpass-api.de/api/interpreter")
OSM_CACHE_PATH: Path = BASE_DIR / os.getenv("OSM_CACHE_PATH", "data/osm_cache.json")

# Database Path
DB_PATH: Path = BASE_DIR / os.getenv("DB_PATH", "data/hotspots.db")
SCHEMA_PATH: Path = DB_DIR / "schema.sql"

# Open-Meteo Weather Endpoints
OPEN_METEO_ARCHIVE_URL: str = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST_URL: str = "https://api.open-meteo.com/v1/forecast"

# 7 Standardized Event Type Classes (Exact spec requirement)
EVENT_TYPES: List[str] = [
    "industrial_flare",
    "industrial_accident",
    "stockpile_combustion",
    "known_false_positive",
    "wildfire",
    "agricultural_burn",
    "unknown_source",
]

# Hazard Weight Lookup Table (Exact spec requirement)
HAZARD_WEIGHTS: Dict[str, float] = {
    "wildfire": 1.0,
    "unknown_source": 1.0,
    "industrial_accident": 0.9,
    "stockpile_combustion": 0.6,
    "agricultural_burn": 0.3,
    "industrial_flare": 0.1,
    "known_false_positive": 0.0,
}

# India Agricultural Stubble Burning Season Months (Apr-May, Oct-Nov)
AGRI_BURN_MONTHS: List[int] = [4, 5, 10, 11]

# Spatial & Clustering Hyperparameters
DBSCAN_EPS_METERS: float = 750.0
DBSCAN_TIME_HOURS: float = 12.0
FACILITY_PROXIMITY_THRESHOLD_M: float = 500.0
H3_RESOLUTION: int = 8

# Model A (Classifier) Hyperparameters
MODEL_A_PARAMS: Dict = {
    "objective": "multiclass",
    "num_class": len(EVENT_TYPES),
    "num_leaves": 31,
    "learning_rate": 0.05,
    "n_estimators": 300,
    "random_state": 42,
    "verbose": -1,
}

# Model C (24h Risk Escalation) Hyperparameters
MODEL_C_PARAMS: Dict = {
    "objective": "binary",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "n_estimators": 300,
    "random_state": 42,
    "verbose": -1,
}

# API Server Configuration
API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", 8000))

if __name__ == "__main__":
    print("=== Configuration Loaded Successfully ===")
    print(f"Base Directory:     {BASE_DIR}")
    print(f"FIRMS Key Defined:  {bool(FIRMS_MAP_KEY)}")
    print(f"Database Path:      {DB_PATH}")
    print(f"OSM Cache Path:     {OSM_CACHE_PATH}")
    print(f"Default BBox:       {DEFAULT_BBOX}")
    print(f"Event Types:        {EVENT_TYPES}")
