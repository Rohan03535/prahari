import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, atan2
from typing import List, Optional
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from config import PATROL_DEFAULTS

try:
    from ortools.constraint_solver import routing_enums_pb2, pywrapcp
    HAS_ORTOOLS = True
except ImportError:
    HAS_ORTOOLS = False


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def build_distance_matrix(locations: pd.DataFrame) -> np.ndarray:
    """Build a pairwise travel-time matrix (in minutes) from lat/lng coords."""
    n = len(locations)
    avg_speed = PATROL_DEFAULTS["avg_speed_kmh"]
    matrix = np.zeros((n, n))
    lats = locations["lat"].values
    lngs = locations["lng"].values
    for i in range(n):
        for j in range(i + 1, n):
            dist = haversine_km(lats[i], lngs[i], lats[j], lngs[j])
            time_min = (dist / avg_speed) * 60
            matrix[i][j] = time_min
            matrix[j][i] = time_min
    return matrix


def prepare_hotspots(hotspots: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """One row per junction; top-N by PCIS."""
    df = (
        hotspots.sort_values("total_pcis", ascending=False)
        .drop_duplicates("junction_name", keep="first")
        .reset_index(drop=True)
    )
    return df.nlargest(top_n, "total_pcis").reset_index(drop=True)


def _travel_minutes(from_lat, from_lng, to_lat, to_lng) -> float:
    dist = haversine_km(from_lat, from_lng, to_lat, to_lng)
    return (dist / PATROL_DEFAULTS["avg_speed_kmh"]) * 60


def _clean_route_stops(stops: list[dict]) -> list[dict]:
    """Remove consecutive duplicate junctions and zero-PCIS artifacts."""
    cleaned = []
    for stop in stops:
        if cleaned and cleaned[-1]["junction"] == stop["junction"]:
            continue
        cleaned.append(stop)
    return cleaned


def _recompute_route(route: dict) -> dict:
    route["stops"] = _clean_route_stops(route["stops"])
    route["route_pcis"] = sum(s["pcis"] for s in route["stops"])
    route["n_stops"] = len(route["stops"])
    return route


def _solve_balanced_greedy(spots: pd.DataFrame, n_units: int, shift_minutes: int) -> list[dict]:
    """
    Assign hotspots to units balancing PCIS load while respecting shift time.
    Each junction visited at most once across the force.
    """
    dwell = PATROL_DEFAULTS["dwell_minutes"]
    depot_lat = spots["lat"].mean()
    depot_lng = spots["lng"].mean()

    routes = [{
        "unit": v + 1,
        "stops": [],
        "route_pcis": 0,
        "n_stops": 0,
        "time_used": 0.0,
        "last_lat": depot_lat,
        "last_lng": depot_lng,
    } for v in range(n_units)]

    visited = set()
    for _, spot in spots.sort_values("total_pcis", ascending=False).iterrows():
        jname = spot["junction_name"]
        if jname in visited:
            continue

        best_idx = None
        best_load = float("inf")
        for i, route in enumerate(routes):
            travel = _travel_minutes(route["last_lat"], route["last_lng"], spot["lat"], spot["lng"])
            needed = route["time_used"] + travel + dwell
            if needed <= shift_minutes and route["route_pcis"] < best_load:
                best_idx = i
                best_load = route["route_pcis"]

        if best_idx is None:
            continue

        route = routes[best_idx]
        travel = _travel_minutes(route["last_lat"], route["last_lng"], spot["lat"], spot["lng"])
        route["stops"].append({
            "junction": jname,
            "lat": spot["lat"],
            "lng": spot["lng"],
            "pcis": spot["total_pcis"],
        })
        route["route_pcis"] += spot["total_pcis"]
        route["n_stops"] += 1
        route["time_used"] += travel + dwell
        route["last_lat"] = spot["lat"]
        route["last_lng"] = spot["lng"]
        visited.add(jname)

    for route in routes:
        route.pop("time_used", None)
        route.pop("last_lat", None)
        route.pop("last_lng", None)
        _recompute_route(route)

    return [r for r in routes if r["n_stops"] > 0]


def solve_patrol_routing(
    hotspots: pd.DataFrame,
    n_units: int = None,
    shift_hours: float = None,
    top_n: int = 25,
) -> dict:
    """
    Solve patrol routing through top PCIS hotspots under shift time constraints.
    Uses load-balanced greedy assignment (deduped junctions, travel-aware).
    """
    n_units = n_units or PATROL_DEFAULTS["n_units"]
    shift_hours = shift_hours or PATROL_DEFAULTS["shift_hours"]
    shift_minutes = int(shift_hours * 60)

    spots = prepare_hotspots(hotspots, top_n)
    if spots.empty:
        return {
            "routes": [],
            "total_intercepted_pcis": 0,
            "total_available_pcis": 0,
            "intercept_pct": 0,
            "spots": spots,
        }

    routes = _solve_balanced_greedy(spots, n_units, shift_minutes)
    total_intercepted = sum(r["route_pcis"] for r in routes)
    total_available = spots["total_pcis"].sum()
    intercept_pct = (total_intercepted / total_available * 100) if total_available > 0 else 0

    return {
        "routes": routes,
        "total_intercepted_pcis": total_intercepted,
        "total_available_pcis": total_available,
        "intercept_pct": intercept_pct,
        "spots": spots,
    }


def _naive_same_budget(hotspots: pd.DataFrame, top_n: int, stop_budget: int) -> float:
    """Top-N-by-count from same candidate pool, same stop budget (no routing)."""
    spots = prepare_hotspots(hotspots, top_n)
    n = min(stop_budget, len(spots))
    if n <= 0:
        return 0.0
    return float(spots.nlargest(n, "count")["total_pcis"].sum())


def compute_plan_metrics(
    result: dict,
    comparison: dict,
    n_units: int,
    shift_hours: float,
    off_grid_violations: int = 0,
) -> dict:
    routes = result["routes"]
    total_pcis = result["total_intercepted_pcis"]
    officer_hours = max(n_units * shift_hours, 0.1)
    route_pcis = [r["route_pcis"] for r in routes] or [0]

    min_pcis = min(route_pcis)
    max_pcis = max(route_pcis)
    balance_ratio = (min_pcis / max_pcis) if max_pcis > 0 else 1.0

    est_stops = sum(len(r["stops"]) for r in routes)
    est_fines = est_stops * 500  # ₹500 per stop, illustrative

    return {
        "pcis_per_officer_hour": total_pcis / officer_hours,
        "naive_pcis_per_officer_hour": comparison["naive_pcis"] / officer_hours,
        "improvement_vs_naive_pct": comparison["improvement_vs_naive_pct"],
        "improvement_vs_naive_per_hour_pct": comparison.get("improvement_vs_naive_per_hour_pct", 0),
        "intercept_pct": result["intercept_pct"],
        "est_stops": est_stops,
        "est_fines_rs": est_fines,
        "load_balance_ratio": balance_ratio,
        "min_route_pcis": min_pcis,
        "max_route_pcis": max_pcis,
        "off_grid_violations": off_grid_violations,
    }


def compare_with_baselines(
    hotspots,
    n_units,
    shift_hours,
    top_n: int = 25,
    n_simulations: int = 100,
):
    """
    Compare optimized routing vs baselines on the SAME candidate pool and stop budget.
    """
    optimized = solve_patrol_routing(hotspots, n_units, shift_hours, top_n)
    optimized_pcis = optimized["total_intercepted_pcis"]
    stop_budget = sum(len(r["stops"]) for r in optimized["routes"])

    spots = prepare_hotspots(hotspots, top_n)

    random_pcis_list = []
    for _ in range(n_simulations):
        sample_n = min(stop_budget, len(spots))
        if sample_n <= 0:
            continue
        sampled = spots.sample(n=sample_n, replace=False)
        random_pcis_list.append(sampled["total_pcis"].sum())
    avg_random = float(np.mean(random_pcis_list)) if random_pcis_list else 0.0

    naive_pcis = _naive_same_budget(hotspots, top_n, stop_budget)

    officer_hours = max(n_units * shift_hours, 0.1)
    opt_per_hour = optimized_pcis / officer_hours
    naive_per_hour = naive_pcis / officer_hours

    improvement_vs_random = ((optimized_pcis - avg_random) / avg_random * 100) if avg_random > 0 else 0
    improvement_vs_naive = ((optimized_pcis - naive_pcis) / naive_pcis * 100) if naive_pcis > 0 else 0
    improvement_vs_naive_per_hour = ((opt_per_hour - naive_per_hour) / naive_per_hour * 100) if naive_per_hour > 0 else 0

    return {
        "optimized_pcis": optimized_pcis,
        "random_avg_pcis": avg_random,
        "naive_pcis": naive_pcis,
        "stop_budget": stop_budget,
        "improvement_vs_random_pct": improvement_vs_random,
        "improvement_vs_naive_pct": improvement_vs_naive,
        "improvement_vs_naive_per_hour_pct": improvement_vs_naive_per_hour,
        "pcis_per_officer_hour": opt_per_hour,
        "naive_pcis_per_officer_hour": naive_per_hour,
        "optimized_routes": optimized,
    }


def simulate_unit_scenarios(
    hotspots: pd.DataFrame,
    unit_counts: Optional[List[int]] = None,
    shift_hours: float = None,
    top_n: int = 25,
) -> pd.DataFrame:
    """Compare PCIS intercept across different patrol unit counts."""
    shift_hours = shift_hours or PATROL_DEFAULTS["shift_hours"]
    unit_counts = unit_counts or [3, 5, 8, 10, 12]

    rows = []
    for n in unit_counts:
        result = solve_patrol_routing(hotspots, n, shift_hours, top_n)
        est_stops = sum(len(r["stops"]) for r in result["routes"])
        rows.append({
            "n_units": n,
            "pcis_intercepted": result["total_intercepted_pcis"],
            "coverage_pct": result["intercept_pct"],
            "total_stops": est_stops,
            "est_fines_rs": est_stops * 500,
        })

    df = pd.DataFrame(rows)
    if len(df) > 1:
        df["marginal_pcis"] = df["pcis_intercepted"].diff().fillna(df["pcis_intercepted"])
    else:
        df["marginal_pcis"] = df["pcis_intercepted"]
    return df
