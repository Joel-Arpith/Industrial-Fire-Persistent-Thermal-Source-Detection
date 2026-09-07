"""
Data Ingestion Package
======================
Contains fetchers for NASA FIRMS, OpenStreetMap Overpass, Land Cover point-sampling, and Open-Meteo Weather.
"""

from data_ingestion.fetch_firms import fetch_firms_hotspots
from data_ingestion.fetch_osm import fetch_osm_industrial_and_exposure
from data_ingestion.fetch_landcover import sample_landcover_point
from data_ingestion.fetch_weather import fetch_weather_for_hotspots

__all__ = [
    "fetch_firms_hotspots",
    "fetch_osm_industrial_and_exposure",
    "sample_landcover_point",
    "fetch_weather_for_hotspots",
]
