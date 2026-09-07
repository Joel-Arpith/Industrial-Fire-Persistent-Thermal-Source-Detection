-- =====================================================================
-- SCHEMA: Hotspot Persistence History Log
-- 
-- HOW TO RUN / INITIALIZE:
-- 1. Using SQLite CLI or Python sqlite3:
--    sqlite3 data/hotspots.db < db/schema.sql
--    (Or automatically initialized by processing/persistence_log.py)
--
-- INPUTS:
--    - SQLite database path (default: data/hotspots.db)
--
-- OUTPUT:
--    - Creates hotspot_history table with composite primary key
--      (location_key, acq_date, acq_time).
-- =====================================================================

CREATE TABLE IF NOT EXISTS hotspot_history (
  event_id TEXT,
  location_key TEXT,        -- facility_id if within 500m of a known OSM facility, 
                             -- ELSE an H3 resolution-8 cell id. 
                             -- NEVER round lat/lon to a naive grid — this splits a single 
                             -- facility's history across cells due to ordinary pixel jitter.
  acq_date DATE,
  acq_time TEXT,
  frp REAL,
  confidence REAL,
  daynight TEXT,
  nearest_industrial_type TEXT,
  PRIMARY KEY (location_key, acq_date, acq_time)
);

