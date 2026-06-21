"""Extract verified numbers for PPT."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import PATROL_DEFAULTS
from src.data_pipeline import load_bundled_pipeline
from src.enforcement_replay import run_enforcement_replay
from src.prediction import engineer_features, train_hurdle_model

d = load_bundled_pipeline()
p = d["parking"]
fs = json.loads((ROOT / "data" / "flow_summary.json").read_text())
plan = json.loads((ROOT / "data" / "default_patrol_plan.json").read_text())

stats = {
    "total_violations": len(p),
    "off_grid": int((p["junction_name"] == "No Junction").sum()),
    "with_junction": int((p["junction_name"] != "No Junction").sum()),
    "junctions": int(p["junction_name"].nunique()),
    "stations": int(p["police_station"].nunique()),
    "date_start": str(p["date"].min()),
    "date_end": str(p["date"].max()),
    "peak_hour": int(p.groupby("hour")["pcis"].sum().idxmax()),
    "flow": fs,
    "plan": {
        "total_intercepted_pcis": plan["result"]["total_intercepted_pcis"],
        "intercept_pct": plan["result"]["intercept_pct"],
        "improvement_vs_naive": plan["comparison"]["improvement_vs_naive_pct"],
        "load_balance": plan["plan_metrics"]["load_balance_ratio"],
    },
}

for res in ["junction", "hex"]:
    r = run_enforcement_replay(p, PATROL_DEFAULTS["n_units"], PATROL_DEFAULTS["shift_hours"], resolution=res)
    stats[f"replay_{res}"] = {
        "improvement_pct": r["summary"]["improvement_pcis_pct"],
        "actual_pcis": r["summary"]["actual"]["intercepted_pcis"],
        "prahari_pcis": r["summary"]["prahari"]["intercepted_pcis"],
        "holdout": f"{r['split']['holdout_start']} to {r['split']['holdout_end']}",
    }

f = engineer_features(d["cell_agg"], d["recurrence"])
m = train_hurdle_model(f)
stats["model_metrics"] = {k: float(v) if isinstance(v, (int, float)) else v for k, v in m["metrics"].items()}

print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    out = ROOT / "scripts" / "ppt_stats.json"
    out.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
