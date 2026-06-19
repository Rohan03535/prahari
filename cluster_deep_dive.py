"""Get detailed stats for the top blind spot cluster for the narrative."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from src.data_pipeline import build_full_pipeline
from src.spatial_analysis import discover_blind_spots
from math import radians, sin, cos, sqrt, atan2
import pandas as pd

data = build_full_pipeline()
p = data["parking"]
clusters = discover_blind_spots(p)
top = clusters.iloc[0]

print("=" * 60)
print("TOP BLIND SPOT CLUSTER — DEEP DIVE")
print("=" * 60)
print(f"Name: {top['cluster_name']}")
print(f"Violations: {top['count']}")
print(f"Total PCIS: {top['total_pcis']:.0f}")
print(f"Lat: {top['lat']:.6f}, Lng: {top['lng']:.6f}")
print(f"Police Station: {top['police_station']}")
print(f"Top Vehicle: {top['top_vehicle']}")
print(f"Peak Hour (IST): {int(top['peak_hour'])}:00")
print(f"Unique Dates Active: {top['unique_dates']}")
print(f"Heavy Vehicle Ratio: {top['heavy_ratio']:.1%}")
print(f"Main Road Violation Ratio: {top['main_road_ratio']:.1%}")

# Fine estimates (Rs 500 per violation is BTP standard for parking)
est_fines = int(top["count"]) * 500
print(f"\nEstimated uncollected fines: Rs {est_fines:,}")
print(f"Estimated monthly: Rs {est_fines // 6:,}")

# Compare coverage
total_violations = len(p)
no_junc = len(p[p["junction_name"] == "No Junction"])
with_junc = total_violations - no_junc
print(f"\nCoverage analysis:")
print(f"  Total violations: {total_violations:,}")
print(f"  At named junctions: {with_junc:,} ({with_junc/total_violations:.1%})")
print(f"  At NO junction (off-grid): {no_junc:,} ({no_junc/total_violations:.1%})")

# Top 5 clusters summary
print(f"\nTop 5 blind spot clusters:")
for i, row in clusters.head(5).iterrows():
    fines = int(row["count"]) * 500
    print(f"  #{i+1}: {row['cluster_name']} — {row['count']} violations, "
          f"Rs {fines:,} uncollected, peak {int(row['peak_hour'])}:00 IST")

# Named junctions for context
js = data["junction_summary"]
print(f"\nFor comparison — Top 5 NAMED junctions:")
for i, row in js.head(5).iterrows():
    print(f"  {row['junction_name']} — {row['count']} violations, "
          f"PCIS {row['total_pcis']:.0f}")

print(f"\n KEY NARRATIVE STAT:")
top5_blind_pcis = clusters.head(5)["total_pcis"].sum()
top5_junc_pcis = js.head(5)["total_pcis"].sum()
print(f"  Top 5 hidden clusters PCIS: {top5_blind_pcis:,.0f}")
print(f"  Top 5 named junctions PCIS: {top5_junc_pcis:,.0f}")
print(f"  Hidden as % of named: {top5_blind_pcis/top5_junc_pcis*100:.1f}%")
