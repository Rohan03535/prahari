"""Integration test for all PRAHARI modules."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

print("=" * 60)
print("PRAHARI MODULE INTEGRATION TEST")
print("=" * 60)

# 1. Data Pipeline
print("\n[1/5] Testing Data Pipeline...")
from src.data_pipeline import build_full_pipeline
data = build_full_pipeline()
p = data["parking"]
print(f"  Parking records: {len(p):,}")
print(f"  Cell aggregations: {len(data['cell_agg']):,}")
print(f"  Recurrence entries: {len(data['recurrence']):,}")
print(f"  Junction summary: {len(data['junction_summary'])}")
print(f"  PCIS range: {p['pcis'].min():.1f} - {p['pcis'].max():.1f}, mean={p['pcis'].mean():.2f}")
print(f"  Top junction: {data['junction_summary'].iloc[0]['junction_name']}")
print("  PASS")

# 2. Spatial Analysis
print("\n[2/5] Testing Spatial Analysis...")
from src.spatial_analysis import discover_blind_spots, compute_enforcement_exposure
clusters = discover_blind_spots(p)
print(f"  Hidden hotspot clusters: {len(clusters)}")
if not clusters.empty:
    print(f"  Largest cluster: {clusters.iloc[0]['cluster_name']} ({clusters.iloc[0]['count']} violations)")
exposure = compute_enforcement_exposure(p)
blind = exposure[exposure["is_blind_spot"] == 1]
print(f"  Blind spot cells: {len(blind)}")
print(f"  Total cells analyzed: {len(exposure)}")
print("  PASS")

# 3. Analytics
print("\n[3/5] Testing Analytics...")
from src.analytics import enforcement_efficacy, chronic_offender_analysis, system_health_analysis
eff = enforcement_efficacy(p)
s = eff["summary"]
print(f"  Enforcement events: {s.get('total_events', 0)}")
print(f"  Effective: {s.get('effective_events', 0)} ({s.get('effectiveness_rate', 0):.1f}%)")

offenders = chronic_offender_analysis(p)
print(f"  Chronic offenders (5+): {len(offenders)}")
print(f"  Top offender: {offenders.iloc[0]['violation_count']:.0f} tickets, PCIS={offenders.iloc[0]['total_pcis']:.0f}")

health = system_health_analysis(p)
print(f"  City rejection rate: {health['city_avg_rejection_rate']:.1%}")
print("  PASS")

# 4. Optimizer
print("\n[4/5] Testing Patrol Optimizer...")
from src.optimizer import solve_patrol_routing, compare_with_random
js = data["junction_summary"]
result = solve_patrol_routing(js, n_units=5, shift_hours=4, top_n=20)
print(f"  Routes generated: {len(result['routes'])}")
print(f"  Total intercepted PCIS: {result['total_intercepted_pcis']:.0f}")
print(f"  Intercept coverage: {result['intercept_pct']:.1f}%")
for r in result["routes"]:
    stops = [s["junction"][:35] for s in r["stops"]]
    print(f"    Unit {r['unit']}: {r['n_stops']} stops -> {', '.join(stops[:3])}")

comp = compare_with_random(js, 5, 4)
print(f"  Optimized vs Random: +{comp['improvement_pct']:.1f}%")
print("  PASS")

# 5. Prediction
print("\n[5/5] Testing Prediction Model...")
from src.prediction import engineer_features, train_hurdle_model
features = engineer_features(data["cell_agg"], data["recurrence"])
print(f"  Feature matrix: {features.shape}")
model_data = train_hurdle_model(features)
m = model_data["metrics"]
print(f"  Precision@10%: {m.get('precision_at_10pct', 0):.2%}")
print(f"  Hit Rate@10%: {m.get('hit_rate_at_10pct', 0):.2%}")
print(f"  Precision@20%: {m.get('precision_at_20pct', 0):.2%}")
print(f"  Binary F1: {m.get('f1', 0):.2%}")
print(f"  MAE: {m.get('mae', 0):.2f}")
print("  PASS")

print("\n" + "=" * 60)
print("ALL MODULES PASSED")
print("=" * 60)
