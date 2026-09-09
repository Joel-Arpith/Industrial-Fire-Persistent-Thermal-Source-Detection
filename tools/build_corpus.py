"""Build a multi-region, multi-season FIRMS corpus for training.

Recent data alone yields 5 of the 7 label classes: agricultural_burn needs the
Oct-Nov stubble window and known_false_positive needs a solar park in bbox.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))   # data/ paths are repo-root relative

from dotenv import load_dotenv; load_dotenv()
import pandas as pd
from data_ingestion.fetch_firms import fetch_firms_hotspots

PULLS = [
    # (label,                region,                   days, start_date)
    ("industrial-gujarat",   "gujarat",                  30, None),
    ("industrial-odisha",    "odisha",                   30, None),
    ("industrial-cgjh",      "chhattisgarh_jharkhand",   30, None),
    ("solar-bhadla",         "rajasthan_solar",          30, None),
    ("agri-punjab-2025",     "punjab_haryana",           15, "2025-10-28"),
    ("wildfire-odisha",      "odisha",                   20, "2026-03-01"),
    ("wildfire-cgjh",        "chhattisgarh_jharkhand",   20, "2026-03-01"),
]

frames = []
for label, region, days, start in PULLS:
    df = fetch_firms_hotspots(region=region, total_days=days, start_date=start)
    print(f"  {label:22s} {region:24s} {days:3d}d  start={start or 'recent':10s} -> {len(df):6,} rows")
    if len(df):
        df = df.copy(); df["pull"] = label
        frames.append(df)

corpus = pd.concat(frames, ignore_index=True)
corpus.drop_duplicates(subset=["latitude","longitude","acq_date","acq_time"], inplace=True)
corpus.reset_index(drop=True, inplace=True)
corpus.to_csv("data/firms_corpus.csv", index=False)
print(f"\nCORPUS: {len(corpus):,} unique rows -> data/firms_corpus.csv")
print(f"  distance-matrix memory at this size: {len(corpus)**2*4/1e6:.0f} MB")
print(corpus["pull"].value_counts().to_string())
