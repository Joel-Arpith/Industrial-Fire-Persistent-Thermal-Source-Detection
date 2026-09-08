# Frontend & Dashboard Integration Guide: API Output Schema

This document serves as the complete technical contract for frontend and dashboard developers integrating against the **Industrial Fire & Persistent Thermal Source Detection API**.

---

## 1. Core Design Principle: Observed vs. Inferred

All hotspot payload objects strictly separate **Satellite Evidence (Observed)** from **Model AI Predictions (Inferred)**:
- `observed`: Direct, un-manipulated satellite & GIS sensor data (FRP, satellite confidence, acquisition timestamp, nearest OSM facility boundary distance and type).
- `inferred`: Synthesized intelligence produced by the machine learning classifiers and anomaly engine (event classification, anomaly score, hazard weighting, spatial exposure factor, final risk score, 24h escalation probability, and SHAP feature drivers).

> **Frontend Implementation Tip**: In your UI modal/side panel, visually delineate these two sections (e.g., a "Satellite Evidence" card with blue/neutral styling, and an "AI Risk Analysis" card with risk-colored styling).

---

## 2. API Endpoints

Base URL: `http://localhost:8000` (or your deployment host)  
CORS: Enabled for all origins (`*`)  
Format: `application/json`

---

### Endpoint 1: `GET /regions`

Returns the registry of available predefined Indian industrial regions (name, human-readable label, bounding box, and industrial profile description) so that the frontend dashboard can render a dynamic region-picker dropdown.

#### Response Schema (Array of Region Objects)

```typescript
interface RegionInfo {
  name: string;        // Unique identifier key (e.g. "gujarat", "maharashtra", "odisha", "all_india")
  label: string;       // Human-friendly title and coverage cities
  bbox: {
    west: number;      // Western longitude in degrees
    south: number;     // Southern latitude in degrees
    east: number;      // Eastern longitude in degrees
    north: number;     // Northern latitude in degrees
  };
  description: string; // Summary of key industries, refineries, and installations in the region
}
```

#### Sample Response Payload (`GET /regions`)

```json
[
  {
    "name": "gujarat",
    "label": "Gujarat Industrial Belt (Hazira, Dahej, Jamnagar, Surat)",
    "bbox": {
      "west": 69.5,
      "south": 20.5,
      "east": 73.8,
      "north": 23.0
    },
    "description": "Petrochemical refineries, LNG terminals, ports, and heavy chemical clusters in Gujarat."
  },
  {
    "name": "maharashtra",
    "label": "Maharashtra Industrial Belt (Mumbai, MMR, Pune, Raigad, Tarapur)",
    "bbox": {
      "west": 72.6,
      "south": 18.3,
      "east": 74.5,
      "north": 20.0
    },
    "description": "Chemical corridors, manufacturing MIDCs, and energy installations in Maharashtra."
  },
  {
    "name": "odisha",
    "label": "Odisha Industrial Belt (Angul, Jharsuguda, Paradip, Kalinganagar)",
    "bbox": {
      "west": 83.5,
      "south": 19.8,
      "east": 87.0,
      "north": 22.2
    },
    "description": "Steel plants, aluminum smelters, coal mining complexes, and deep-water ports in Odisha."
  },
  {
    "name": "chhattisgarh_jharkhand",
    "label": "East-Central Mining & Steel Belt (Bhilai, Korba, Dhanbad, Jamshedpur)",
    "bbox": {
      "west": 81.0,
      "south": 21.0,
      "east": 86.8,
      "north": 24.2
    },
    "description": "Coal fields, thermal power plants, and integrated steelworks in CG & JH."
  },
  {
    "name": "all_india",
    "label": "All India Coverage",
    "bbox": {
      "west": 68.0,
      "south": 6.5,
      "east": 97.5,
      "north": 37.5
    },
    "description": "Nationwide active fire and thermal anomaly coverage across the Indian subcontinent."
  }
]
```

---

### Endpoint 2: `GET /hotspots`

Retrieves all active thermal hotspots for a specified Indian region or custom spatial bounding box, running the full ML pipeline and returning scored detections.

#### Query Parameters

| Parameter | Type | Required | Default | Description | Example |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `region` | `string` | Optional | `gujarat` | **Primary Interface**: Named Indian region key (`gujarat`, `maharashtra`, `odisha`, `chhattisgarh_jharkhand`, `all_india`). **Takes precedence over `bbox`**. | `maharashtra` |
| `bbox` | `string` | Optional | Derived from region | Advanced manual override: `west,south,east,north` in degrees. Used only if `region` is omitted. | `72.5,21.0,73.5,22.0` |
| `since_hours` | `integer` | Optional | `24` | Historical lookback window in hours. | `48` |

#### Response Schema (Array of Hotspot Objects)

```typescript
interface HotspotResponse {
  event_id: string;                      // Unique cluster ID (e.g. "evt_20260904_0032")
  lat: number;                           // Hotspot latitude in degrees
  lon: number;                           // Hotspot longitude in degrees
  observed: {
    frp: number;                         // Fire Radiative Power in MW (raw sensor signal)
    confidence: number;                  // Detection confidence percentage (0 - 100)
    acq_date: string;                    // Acquisition date in "YYYY-MM-DD"
    nearest_industrial_type: string;     // OSM facility category ("refinery", "power_plant", "waste_landfill", "quarry_mining", "solar_farm", "general_industrial", "none")
    dist_to_industrial_m: number;        // Exact distance in meters to nearest facility boundary (0 = inside facility boundary)
  };
  inferred: {
    event_type: string;                  // Predicted event class (see Event Types table below)
    event_type_confidence: number;       // Classification probability (0.0 to 1.0)
    anomaly_score_normalized: number;    // Robust statistical z-score deviation from facility baseline (0.0 to 1.0)
    hazard_weight: number;               // Multiplier for event type severity (0.0 to 1.0)
    exposure_factor: number;             // Human & infrastructure exposure risk (0.0 to 1.0)
    final_risk_score: number;            // Combined score: hazard_weight * anomaly_score_normalized * exposure_factor (0.0 to 1.0)
    escalating_24h: boolean;             // True if 24h risk is critical (risk_24h >= 0.50)
    risk_24h: number;                    // 24h spread/intensification probability (0.0 to 1.0)
    needs_manual_review: boolean;        // True if low confidence, accident, or candidate illegal unknown source
    top_shap_features: string[];         // Top 3 feature drivers explaining the Model A prediction
  };
}
```

#### Sample Response Payload (`GET /hotspots?bbox=72.5,21.0,73.5,22.0&since_hours=24`)

```json
[
  {
    "event_id": "evt_20260904_0032",
    "lat": 21.1700,
    "lon": 72.8300,
    "observed": {
      "frp": 42.3,
      "confidence": 87.0,
      "acq_date": "2026-09-04",
      "nearest_industrial_type": "refinery",
      "dist_to_industrial_m": 210.0
    },
    "inferred": {
      "event_type": "industrial_flare",
      "event_type_confidence": 0.91,
      "anomaly_score_normalized": 0.12,
      "hazard_weight": 0.1,
      "exposure_factor": 0.40,
      "final_risk_score": 0.048,
      "escalating_24h": false,
      "risk_24h": 0.08,
      "needs_manual_review": false,
      "top_shap_features": [
        "dist_to_nearest_industrial_m",
        "night_detection_fraction",
        "daynight"
      ]
    }
  },
  {
    "event_id": "evt_20260904_0045",
    "lat": 21.6500,
    "lon": 73.1200,
    "observed": {
      "frp": 128.5,
      "confidence": 96.0,
      "acq_date": "2026-09-04",
      "nearest_industrial_type": "none",
      "dist_to_industrial_m": 4200.0
    },
    "inferred": {
      "event_type": "wildfire",
      "event_type_confidence": 0.89,
      "anomaly_score_normalized": 0.75,
      "hazard_weight": 1.0,
      "exposure_factor": 0.65,
      "final_risk_score": 0.488,
      "escalating_24h": true,
      "risk_24h": 0.72,
      "needs_manual_review": true,
      "top_shap_features": [
        "land_cover_class",
        "dist_to_nearest_industrial_m",
        "wind_speed"
      ]
    }
  }
]
```

---

### Endpoint 3: `GET /hotspot/{event_id}`

Fetches the complete diagnostic profile, full 19-feature vector, and 7-class probability breakdown for a specific thermal event.

#### Path Parameters

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `event_id` | `string` | The cluster event ID (e.g. `evt_20260904_0032`). |

#### Response Schema

```typescript
interface HotspotDetailResponse {
  event_id: string;
  location_key: string;                  // Facility OSM ID or H3 resolution-8 cell ID
  lat: number;
  lon: number;
  observed: ObservedData;                // (Same shape as in /hotspots)
  inferred: InferredData;                // (Same shape as in /hotspots)
  full_feature_vector: {
    frp: number;
    brightness_ti4: number;
    brightness_ti5: number;
    confidence: number;
    daynight: "D" | "N";
    dist_to_nearest_industrial_m: number;
    nearest_industrial_type: string;
    land_cover_class: string;
    temperature: number;
    humidity: number;
    wind_speed: number;
    days_active_last_30: number;
    night_detection_fraction: number;
    month: number;
    is_agri_burn_season: 0 | 1;
    frp_zscore_vs_facility_baseline: number;
    dist_to_populated_area_m: number;
    dist_to_critical_infra_m: number;
    cluster_growth_rate: number;
  };
  class_probabilities: {
    industrial_flare: number;
    industrial_accident: number;
    stockpile_combustion: number;
    known_false_positive: number;
    wildfire: number;
    agricultural_burn: number;
    unknown_source: number;
  };
}
```

#### Sample Response Payload (`GET /hotspot/evt_20260904_0032`)

```json
{
  "event_id": "evt_20260904_0032",
  "location_key": "osm_way_1001",
  "lat": 21.1700,
  "lon": 72.8300,
  "observed": {
    "frp": 42.3,
    "confidence": 87.0,
    "acq_date": "2026-09-04",
    "nearest_industrial_type": "refinery",
    "dist_to_industrial_m": 210.0
  },
  "inferred": {
    "event_type": "industrial_flare",
    "event_type_confidence": 0.91,
    "anomaly_score_normalized": 0.12,
    "hazard_weight": 0.1,
    "exposure_factor": 0.40,
    "final_risk_score": 0.048,
    "escalating_24h": false,
    "risk_24h": 0.08,
    "needs_manual_review": false,
    "top_shap_features": [
      "dist_to_nearest_industrial_m",
      "night_detection_fraction",
      "daynight"
    ]
  },
  "full_feature_vector": {
    "frp": 42.3,
    "brightness_ti4": 365.2,
    "brightness_ti5": 305.1,
    "confidence": 87.0,
    "daynight": "N",
    "dist_to_nearest_industrial_m": 210.0,
    "nearest_industrial_type": "refinery",
    "land_cover_class": "built_up",
    "temperature": 28.5,
    "humidity": 68.0,
    "wind_speed": 11.2,
    "days_active_last_30": 26,
    "night_detection_fraction": 0.58,
    "month": 9,
    "is_agri_burn_season": 0,
    "frp_zscore_vs_facility_baseline": 0.72,
    "dist_to_populated_area_m": 1200.0,
    "dist_to_critical_infra_m": 350.0,
    "cluster_growth_rate": 0.02
  },
  "class_probabilities": {
    "industrial_flare": 0.91,
    "industrial_accident": 0.03,
    "stockpile_combustion": 0.01,
    "known_false_positive": 0.00,
    "wildfire": 0.01,
    "agricultural_burn": 0.02,
    "unknown_source": 0.02
  }
}
```

---

### Endpoint 4: `GET /facility/{location_key}/history`

Returns the historical FRP time series along with robust Median + MAD statistical baseline bands for time-series chart rendering.

#### Path Parameters

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `location_key` | `string` | Facility identifier (e.g. `osm_way_1001` or `h3_r8_88609a64d1fffff`). |

#### Response Schema

```typescript
interface FacilityHistoryResponse {
  location_key: string;
  median_frp: number;                    // Facility median FRP baseline in MW
  mad_frp: number;                       // Median Absolute Deviation of FRP
  upper_3sigma_threshold: number;        // Anomaly threshold: median + 3 * max(1.4826*MAD, 0.1*median, 1.0)
  lower_3sigma_threshold: number;        // Lower threshold (clamped to >= 0)
  sample_count: number;                  // Total historical observations logged
  history: Array<{
    date: string;                        // "YYYY-MM-DD"
    time: string;                        // "HHMM" (e.g. "0130" or "1330")
    frp: number;                         // FRP reading in MW
    confidence: number;                  // Detection confidence
    daynight: "D" | "N";                 // Day or Night overpass
  }>;
}
```

#### Sample Response Payload (`GET /facility/osm_way_1001/history`)

```json
{
  "location_key": "osm_way_1001",
  "median_frp": 42.00,
  "mad_frp": 5.50,
  "upper_3sigma_threshold": 66.46,
  "lower_3sigma_threshold": 17.54,
  "sample_count": 28,
  "history": [
    {
      "date": "2026-08-08",
      "time": "0130",
      "frp": 41.2,
      "confidence": 85.0,
      "daynight": "N"
    },
    {
      "date": "2026-08-10",
      "time": "1330",
      "frp": 44.1,
      "confidence": 90.0,
      "daynight": "D"
    },
    {
      "date": "2026-08-15",
      "time": "0130",
      "frp": 39.8,
      "confidence": 88.0,
      "daynight": "N"
    },
    {
      "date": "2026-09-04",
      "time": "0130",
      "frp": 42.3,
      "confidence": 87.0,
      "daynight": "N"
    }
  ]
}
```

---

## 3. UI Implementation & Mapping Guide

### Map Marker Styling (Leaflet / Mapbox GL)

1. **Marker Color** $\rightarrow$ mapped to `inferred.final_risk_score` (0.0 to 1.0):
   - `0.00 - 0.15`: 🟢 Green (Routine flare, known false positive)
   - `0.16 - 0.35`: 🟡 Yellow (Moderate risk / agricultural burn)
   - `0.36 - 0.60`: 🟠 Orange (Stockpile combustion, growing fire)
   - `0.61 - 1.00`: 🔴 Red (Severe wildfire, uncontained industrial accident)

2. **Marker Badges / Flags**:
   - `inferred.needs_manual_review == true`: Display ⚠️ **Review Required** badge.
   - `inferred.escalating_24h == true`: Display 🔥 **Rapid Spread / Surge** badge.

### Time-Series Baseline Chart (Recharts / Chart.js)

When rendering `GET /facility/{location_key}/history`:
- **X-Axis**: `date` + `time` formatted as timestamp.
- **Y-Axis**: FRP in Megawatts (MW).
- **Primary Line/Scatter**: Plot each point from `history[].frp` (colored by `daynight` flag: e.g. blue for `N`, amber for `D`).
- **Median Reference Line**: Draw horizontal dashed line at `median_frp`.
- **Upper Anomaly Threshold Band**: Draw horizontal red reference line or shaded upper band at `upper_3sigma_threshold`. Any FRP point crossing above this threshold is highlighted as an anomaly.

### Reference: 7 Canonical Event Types

| Event Type String | UI Display Name | Default Hazard Weight | Typical Action / Routing |
| :--- | :--- | :--- | :--- |
| `industrial_flare` | Routine Industrial Flare | `0.1` | Low priority; monitor baseline |
| `industrial_accident` | Industrial Fire Accident | `0.9` | High priority emergency alert |
| `stockpile_combustion` | Stockpile / Waste Combustion | `0.6` | Environmental health alert |
| `known_false_positive` | Solar Farm Thermal Reflection | `0.0` | Suppressed / Informational only |
| `wildfire` | Wildfire | `1.0` | Forest service containment routing |
| `agricultural_burn` | Stubble / Agricultural Burn | `0.3` | Seasonal air quality tracking |
| `unknown_source` | Unregistered / Illegal Thermal Source | `1.0` | Route to NTRO / Analyst manual review |
