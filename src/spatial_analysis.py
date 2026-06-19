import numpy as np
import pandas as pd
import h3
from hdbscan import HDBSCAN
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from config import HDBSCAN_MIN_CLUSTER_SIZE, HDBSCAN_MIN_SAMPLES


def discover_blind_spots(parking_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cluster 'No Junction' violations using HDBSCAN on lat/lng.
    Returns a DataFrame of discovered clusters with aggregate stats.
    """
    no_junc = parking_df[parking_df["junction_name"] == "No Junction"].copy()
    if len(no_junc) < HDBSCAN_MIN_CLUSTER_SIZE:
        return pd.DataFrame()

    coords = no_junc[["latitude", "longitude"]].values
    coords_rad = np.radians(coords)

    clusterer = HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric="haversine",
    )
    labels = clusterer.fit_predict(coords_rad)
    no_junc = no_junc.copy()
    no_junc["cluster_id"] = labels

    clustered = no_junc[no_junc["cluster_id"] >= 0]

    cluster_stats = (
        clustered.groupby("cluster_id")
        .agg(
            count=("pcis", "size"),
            total_pcis=("pcis", "sum"),
            mean_pcis=("pcis", "mean"),
            lat=("latitude", "mean"),
            lng=("longitude", "mean"),
            heavy_ratio=("vehicle_weight", lambda x: (x >= 3.0).mean()),
            main_road_ratio=("is_main_road_viol", "mean"),
            top_vehicle=("veh_type_final", lambda x: x.mode().iloc[0] if len(x) > 0 else "UNKNOWN"),
            top_violation=("violation_type", lambda x: x.mode().iloc[0] if len(x) > 0 else "UNKNOWN"),
            police_station=("police_station", lambda x: x.mode().iloc[0] if len(x) > 0 else "UNKNOWN"),
            unique_dates=("date", "nunique"),
            peak_hour=("hour", lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 0),
        )
        .reset_index()
        .sort_values("total_pcis", ascending=False)
    )

    cluster_stats = cluster_stats.reset_index(drop=True)
    cluster_stats["cluster_name"] = [
        f"Hidden Hotspot #{i+1} ({row['police_station']})"
        for i, row in cluster_stats.iterrows()
    ]

    return cluster_stats


def compute_enforcement_exposure(parking_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each H3 cell, compute enforcement exposure = unique officers x unique dates.
    High violations / low exposure = blind spot. Low violations / high exposure = over-patrolled.
    """
    exposure = (
        parking_df.groupby("h3_index")
        .agg(
            total_pcis=("pcis", "sum"),
            violation_count=("pcis", "size"),
            unique_officers=("created_by_id", "nunique"),
            unique_devices=("device_id", "nunique"),
            unique_dates=("date", "nunique"),
            lat=("latitude", "mean"),
            lng=("longitude", "mean"),
            has_junction=("has_junction", "max"),
        )
        .reset_index()
    )

    exposure["enforcement_exposure"] = exposure["unique_officers"] * exposure["unique_dates"]
    exposure["enforcement_exposure"] = exposure["enforcement_exposure"].clip(lower=1)

    exposure["pcis_per_exposure"] = exposure["total_pcis"] / exposure["enforcement_exposure"]

    p75_exposure = exposure["enforcement_exposure"].quantile(0.75)
    p25_exposure = exposure["enforcement_exposure"].quantile(0.25)
    p75_pcis = exposure["total_pcis"].quantile(0.75)

    exposure["is_blind_spot"] = (
        (exposure["enforcement_exposure"] <= p25_exposure) & (exposure["pcis_per_exposure"] > exposure["pcis_per_exposure"].quantile(0.75))
    ).astype(int)

    exposure["is_over_patrolled"] = (
        (exposure["enforcement_exposure"] >= p75_exposure) & (exposure["total_pcis"] < exposure["total_pcis"].quantile(0.25))
    ).astype(int)

    exposure["bias_corrected_score"] = exposure["pcis_per_exposure"] * np.log1p(exposure["violation_count"])

    return exposure.sort_values("bias_corrected_score", ascending=False)


def get_h3_neighbors(h3_cell: str, k: int = 1) -> list[str]:
    """Return the k-ring neighbors of a cell (excluding the cell itself)."""
    disk = h3.grid_disk(h3_cell, k)
    return [c for c in disk if c != h3_cell]


def spatial_autocorrelation(cell_agg: pd.DataFrame) -> pd.DataFrame:
    """Add neighbor-average PCIS as a spatial feature for prediction."""
    cell_pcis = cell_agg.groupby("h3_cell")["total_pcis"].sum().to_dict()

    neighbor_means = {}
    for cell in cell_pcis:
        neighbors = get_h3_neighbors(cell, k=1)
        neighbor_vals = [cell_pcis.get(n, 0) for n in neighbors]
        neighbor_means[cell] = np.mean(neighbor_vals) if neighbor_vals else 0

    result = cell_agg.copy()
    result["neighbor_pcis_mean"] = result["h3_cell"].map(neighbor_means).fillna(0)
    return result


def displacement_analysis(parking_df: pd.DataFrame, junction: str, enforcement_weeks: list[str]) -> pd.DataFrame:
    """
    Check if enforcement at a junction displaced violations to nearby junctions.
    Compare violation counts at neighboring cells during vs. before enforcement weeks.
    """
    junc_records = parking_df[parking_df["junction_name"] == junction]
    if junc_records.empty:
        return pd.DataFrame()

    center_h3 = junc_records["h3_index"].mode().iloc[0]
    neighbors = get_h3_neighbors(center_h3, k=2)

    neighbor_records = parking_df[parking_df["h3_index"].isin(neighbors)]

    neighbor_records = neighbor_records.copy()
    neighbor_records["is_enforcement_period"] = neighbor_records["year_week"].isin(enforcement_weeks).astype(int)

    comparison = (
        neighbor_records.groupby(["h3_index", "is_enforcement_period"])
        .agg(weekly_pcis=("pcis", "sum"), weekly_count=("pcis", "size"))
        .reset_index()
    )

    n_weeks_enforce = len(enforcement_weeks)
    all_weeks = neighbor_records["year_week"].nunique()
    n_weeks_before = max(all_weeks - n_weeks_enforce, 1)

    pivot = comparison.pivot_table(
        index="h3_index", columns="is_enforcement_period",
        values="weekly_pcis", aggfunc="sum", fill_value=0,
    )

    if 0 in pivot.columns and 1 in pivot.columns:
        pivot["before_avg"] = pivot[0] / n_weeks_before
        pivot["during_avg"] = pivot[1] / n_weeks_enforce
        pivot["displacement_pct"] = ((pivot["during_avg"] - pivot["before_avg"]) / pivot["before_avg"].clip(lower=1)) * 100
        return pivot.reset_index()

    return pd.DataFrame()
