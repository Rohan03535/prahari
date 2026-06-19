"""
PRAHARI REST API — Deployable interface for ASTraM integration.

Endpoints:
  GET  /health              — system health check
  GET  /hotspots             — ranked hotspots with traffic flow impact
  GET  /blind-spots          — discovered hidden enforcement gaps
  POST /patrol-plan          — generate optimized patrol routes
  GET  /predict/{junction}   — next-shift prediction for a junction
  GET  /offenders            — chronic offender intelligence
  GET  /flow-impact          — city-wide traffic flow impact summary
  GET  /stations             — per-station performance comparison
"""
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import sys, os, json
from typing import Optional

sys.path.insert(0, os.path.dirname(__file__))

app = FastAPI(
    title="PRAHARI API",
    description="Parking Intelligence Command System for Bengaluru Traffic Police. "
                "AI-driven hotspot detection, traffic flow impact quantification, "
                "and patrol route optimization.",
    version="2.0.0",
    contact={"name": "Team PRAHARI", "url": "https://github.com/prahari"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


class PatrolRequest(BaseModel):
    n_units: int = 5
    shift_hours: float = 4.0
    top_n_hotspots: int = 25


# ── Lazy-load data once ──
_cache = {}

def get_data():
    if "data" not in _cache:
        from src.data_pipeline import build_full_pipeline
        from src.traffic_flow_impact import compute_flow_impact, aggregate_flow_impact, junction_flow_impact
        from src.spatial_analysis import discover_blind_spots, compute_enforcement_exposure
        from src.analytics import chronic_offender_analysis, system_health_analysis, station_comparison

        data = build_full_pipeline()
        parking = data["parking"]
        parking = compute_flow_impact(parking)

        _cache["data"] = data
        _cache["parking"] = parking
        _cache["flow_summary"] = aggregate_flow_impact(parking)
        _cache["junction_flow"] = junction_flow_impact(parking)
        _cache["blind_spots"] = discover_blind_spots(parking)
        _cache["exposure"] = compute_enforcement_exposure(parking)
        _cache["offenders"] = chronic_offender_analysis(parking)
        _cache["health"] = system_health_analysis(parking)
        _cache["stations"] = station_comparison(parking)
    return _cache


@app.get("/health")
def health():
    return {"status": "ok", "system": "PRAHARI", "version": "2.0.0"}


@app.get("/hotspots")
def get_hotspots(top_n: int = Query(20, ge=1, le=100)):
    """Ranked hotspot junctions with traffic flow impact metrics."""
    d = get_data()
    jf = d["junction_flow"].head(top_n)
    return {
        "count": len(jf),
        "hotspots": json.loads(jf.to_json(orient="records")),
    }


@app.get("/blind-spots")
def get_blind_spots_api(top_n: int = Query(30, ge=1, le=100)):
    """Discovered hidden hotspot clusters invisible to BTP's junction system."""
    d = get_data()
    clusters = d["blind_spots"].head(top_n)
    return {
        "total_clusters_found": len(d["blind_spots"]),
        "showing": len(clusters),
        "total_violations_in_clusters": int(clusters["count"].sum()),
        "est_uncollected_fines_rs": int(clusters["count"].sum() * 500),
        "clusters": json.loads(clusters.to_json(orient="records")),
    }


@app.post("/patrol-plan")
def generate_patrol_plan(req: PatrolRequest):
    """Generate optimized patrol routes using VRP solver."""
    d = get_data()
    from src.optimizer import solve_patrol_routing, compare_with_baselines
    js = d["data"]["junction_summary"]
    result = solve_patrol_routing(js, req.n_units, req.shift_hours, req.top_n_hotspots)
    comparison = compare_with_baselines(js, req.n_units, req.shift_hours, req.top_n_hotspots)
    return {
        "routes": result["routes"],
        "total_intercepted_pcis": result["total_intercepted_pcis"],
        "intercept_coverage_pct": result["intercept_pct"],
        "vs_random_improvement_pct": comparison["improvement_vs_random_pct"],
        "vs_historical_improvement_pct": comparison["improvement_vs_naive_pct"],
    }


@app.get("/predict/{junction_name}")
def predict_junction(junction_name: str):
    """Next-shift violation prediction for a specific junction."""
    d = get_data()
    parking = d["parking"]
    junc_data = parking[parking["junction_name"].str.contains(junction_name, case=False, na=False)]
    if junc_data.empty:
        return {"error": f"Junction '{junction_name}' not found", "available_junctions": 
                parking[parking["junction_name"] != "No Junction"]["junction_name"].unique()[:20].tolist()}
    
    recent = junc_data.sort_values("created_datetime").tail(100)
    hourly_pattern = recent.groupby("hour")["pcis"].sum().to_dict()
    dow_pattern = recent.groupby("dow_name")["pcis"].sum().to_dict()
    
    return {
        "junction": junction_name,
        "total_violations": len(junc_data),
        "total_pcis": float(junc_data["pcis"].sum()),
        "avg_daily_pcis": float(junc_data.groupby("date")["pcis"].sum().mean()),
        "peak_hour_ist": int(junc_data.groupby("hour")["pcis"].sum().idxmax()),
        "hourly_pattern": hourly_pattern,
        "day_of_week_pattern": dow_pattern,
        "risk_level": "HIGH" if junc_data["pcis"].sum() > parking["pcis"].sum() * 0.01 else "MEDIUM",
    }


@app.get("/flow-impact")
def get_flow_impact():
    """City-wide traffic flow impact from parking violations."""
    d = get_data()
    fs = d["flow_summary"]
    return {
        "total_violations_analyzed": fs["total_violations"],
        "carriageway_blocked_m2": round(fs["total_carriageway_blocked_m2"], 1),
        "avg_capacity_reduction_pct": round(fs["avg_capacity_reduction_pct"], 1),
        "total_vehicle_hours_delay": round(fs["total_delay_vehicle_hours"], 1),
        "total_person_hours_delay": round(fs["total_delay_person_hours"], 1),
        "daily_avg_vehicle_hours_delay": round(fs["daily_avg_delay_veh_hours"], 1),
        "heavy_vehicle_road_block_pct": round(fs["heavy_vehicle_blocked_pct"], 1),
        "junction_delay_share_pct": round(fs["junction_delay_pct"], 1),
    }


@app.get("/offenders")
def get_offenders_api(min_violations: int = Query(20, ge=5)):
    """Chronic offender intelligence — vehicles with repeated violations."""
    d = get_data()
    off = d["offenders"][d["offenders"]["violation_count"] >= min_violations]
    return {
        "total_offenders": len(off),
        "worst_offender_tickets": int(off.iloc[0]["violation_count"]) if len(off) > 0 else 0,
        "offenders": json.loads(off.head(50).to_json(orient="records", date_format="iso")),
    }


@app.get("/stations")
def get_stations():
    """Per-station performance comparison."""
    d = get_data()
    stations = d["stations"]
    return {
        "count": len(stations),
        "stations": json.loads(stations.to_json(orient="records")),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
