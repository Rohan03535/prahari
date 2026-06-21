"""One-time script: process raw CSV into deployment-friendly parquet bundles."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from config import (
    PARKING_CSV, PATROL_DEFAULTS, DEFAULT_TOP_HOTSPOTS, DEFAULT_SHIFT_LABEL,
    DEFAULT_PATROL_PLAN_JSON, PARKING_SLIM_COLUMNS, PARKING_SLIM_PARQUET, JUNCTION_INTEL_PARQUET,
)
from src.data_pipeline import build_full_pipeline, load_bundled_pipeline, _optimize_parking_dtypes
from src.junction_intel import build_junction_intel_df
from src.traffic_flow_impact import compute_flow_impact, aggregate_flow_impact, junction_flow_impact
from src.spatial_analysis import discover_blind_spots
from src.patrol_shift import junction_summary_for_shift
from src.optimizer import (
    solve_patrol_routing, compare_with_baselines, simulate_unit_scenarios, compute_plan_metrics,
)

DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
BLIND_SPOTS_TOP_N = 30


def _strip_patrol_for_json(result: dict, comparison: dict) -> tuple[dict, dict]:
    result_clean = {k: v for k, v in result.items() if k != "spots"}
    comparison_clean = {k: v for k, v in comparison.items() if k != "optimized_routes"}
    return result_clean, comparison_clean


def _bundle_default_patrol_plan(parking: pd.DataFrame) -> None:
    n_units = PATROL_DEFAULTS["n_units"]
    shift_hrs = float(PATROL_DEFAULTS["shift_hours"])
    top_hotspots = DEFAULT_TOP_HOTSPOTS
    shift_label = DEFAULT_SHIFT_LABEL

    print("Pre-computing default patrol plan (landing page)...")
    shift_junctions = junction_summary_for_shift(parking, shift_label)
    result = solve_patrol_routing(shift_junctions, n_units, shift_hrs, top_hotspots)
    comparison = compare_with_baselines(shift_junctions, n_units, shift_hrs, top_hotspots)

    off_grid_count = int((parking["junction_name"] == "No Junction").sum())
    plan_metrics = compute_plan_metrics(result, comparison, n_units, shift_hrs, off_grid_count)

    scenario_units = list(range(2, min(n_units + 4, 13)))
    scenario_df = simulate_unit_scenarios(shift_junctions, scenario_units, shift_hrs, top_hotspots)

    shift_differs = (
        junction_summary_for_shift(parking, "Morning (08–12)").head(3)["junction_name"].tolist()
        != junction_summary_for_shift(parking, "Evening (16–20)").head(3)["junction_name"].tolist()
    )

    result_clean, comparison_clean = _strip_patrol_for_json(result, comparison)
    bundle = {
        "params": {
            "n_units": n_units,
            "shift_hours": shift_hrs,
            "top_hotspots": top_hotspots,
            "shift_label": shift_label,
            "scenario_units": scenario_units,
        },
        "result": result_clean,
        "comparison": comparison_clean,
        "plan_metrics": plan_metrics,
        "scenario_df": scenario_df.to_dict(orient="records"),
        "shift_differs": shift_differs,
        "off_grid_count": off_grid_count,
    }
    with open(DEFAULT_PATROL_PLAN_JSON, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2)
    print(f"  default_patrol_plan.json: {DEFAULT_PATROL_PLAN_JSON.stat().st_size / 1024:.1f} KB")


def main():
    csv_path = PARKING_CSV if PARKING_CSV.exists() else None
    if csv_path:
        print(f"Processing {csv_path} ...")
        data = build_full_pipeline(csv_path)
    else:
        print("Raw CSV not found — loading existing parquet bundles ...")
        data = load_bundled_pipeline()
        if data is None:
            raise FileNotFoundError(
                f"Neither raw CSV ({PARKING_CSV}) nor bundled parquet found. "
                "Place raw CSV in data/raw/ or run bundle after initial processing."
            )

    parking = data["parking"].copy()
    if "violations_list" in parking.columns:
        parking["violations_list"] = parking["violations_list"].apply(
            lambda x: list(x) if not isinstance(x, list) else x
        )

    parking.to_parquet(DATA_DIR / "parking.parquet", index=False, compression="snappy")
    data["junction_summary"].to_parquet(DATA_DIR / "junction_summary.parquet", index=False)
    data["cell_agg"].to_parquet(DATA_DIR / "cell_agg.parquet", index=False)
    data["recurrence"].to_parquet(DATA_DIR / "recurrence.parquet", index=False)

    print("Building cloud-safe slim bundle (Streamlit ~1GB RAM)...")
    keep = [c for c in PARKING_SLIM_COLUMNS if c in parking.columns]
    slim = _optimize_parking_dtypes(parking[keep].copy())
    slim.to_parquet(PARKING_SLIM_PARQUET, index=False, compression="snappy")
    build_junction_intel_df(parking).to_parquet(JUNCTION_INTEL_PARQUET, index=False)
    print(f"  parking_slim.parquet: {PARKING_SLIM_PARQUET.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"  junction_intel.parquet: {JUNCTION_INTEL_PARQUET.stat().st_size / 1024:.1f} KB")

    print("Computing flow impact (one-time)...")
    pf = compute_flow_impact(parking)
    fs = aggregate_flow_impact(pf)
    jf = junction_flow_impact(pf)
    with open(DATA_DIR / "flow_summary.json", "w", encoding="utf-8") as f:
        json.dump(fs, f, indent=2)
    jf.to_parquet(DATA_DIR / "junc_flow.parquet", index=False)
    (
        pf.groupby("road_type")
        .agg(
            violations=("pcis", "size"),
            avg_delay=("estimated_delay_veh_hours", "mean"),
            total_delay=("estimated_delay_veh_hours", "sum"),
            avg_cap_reduction=("capacity_reduction_pct", "mean"),
        )
        .reset_index()
        .to_parquet(DATA_DIR / "road_type_flow.parquet", index=False)
    )

    print("Discovering blind spots (one-time HDBSCAN)...")
    blind = discover_blind_spots(parking)
    blind = blind.nlargest(BLIND_SPOTS_TOP_N, "total_pcis").reset_index(drop=True)
    blind.to_parquet(DATA_DIR / "blind_spots.parquet", index=False)
    print(f"  blind_spots.parquet: top {len(blind)} clusters (display limit {BLIND_SPOTS_TOP_N})")

    _bundle_default_patrol_plan(parking)

    sizes = {f.name: f.stat().st_size / 1024 / 1024 for f in DATA_DIR.glob("*") if f.is_file()}
    total_mb = sum(sizes.values())
    print("Bundled to data/:")
    for name, mb in sorted(sizes.items()):
        print(f"  {name}: {mb:.2f} MB")
    print(f"  TOTAL: {total_mb:.2f} MB")
    print("Done.")


if __name__ == "__main__":
    main()
