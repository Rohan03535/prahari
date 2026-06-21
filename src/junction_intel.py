"""Precomputed per-junction intel for route rationale (cloud-safe)."""
import pandas as pd

from config import JUNCTION_INTEL_PARQUET


def build_junction_intel_df(parking_df: pd.DataFrame) -> pd.DataFrame:
    jdf = parking_df[parking_df["junction_name"] != "No Junction"]
    rows = []
    for jname, grp in jdf.groupby("junction_name"):
        top_veh = grp["veh_type_final"].mode().iloc[0] if len(grp) else "vehicle"
        heavy_pct = float((grp["vehicle_weight"] >= 3.0).mean())
        main_pct = float(grp["is_main_road_viol"].mean())
        peak = int(grp.groupby("hour")["pcis"].sum().idxmax()) if len(grp) else 0
        if "violations_list" in grp.columns:
            top_viol = _top_violation_from_lists(grp["violations_list"])
        elif "violation_type" in grp.columns:
            top_viol = str(grp["violation_type"].mode().iloc[0]).lower()
        else:
            top_viol = "parking violations"
        rows.append({
            "junction_name": jname,
            "top_vehicle": top_veh,
            "heavy_pct": heavy_pct,
            "main_road_pct": main_pct,
            "peak_hour": peak,
            "top_violation": top_viol,
        })
    return pd.DataFrame(rows)


def _top_violation_from_lists(series) -> str:
    counts = {}
    for vlist in series:
        if not isinstance(vlist, list):
            continue
        for v in vlist:
            counts[v] = counts.get(v, 0) + 1
    if not counts:
        return "parking violations"
    return max(counts, key=counts.get).lower().replace("_", " ")


def load_junction_intel(parking_df: pd.DataFrame | None = None) -> pd.DataFrame:
    if JUNCTION_INTEL_PARQUET.exists():
        return pd.read_parquet(JUNCTION_INTEL_PARQUET)
    if parking_df is not None:
        return build_junction_intel_df(parking_df)
    return pd.DataFrame(columns=[
        "junction_name", "top_vehicle", "heavy_pct", "main_road_pct",
        "peak_hour", "top_violation",
    ])
