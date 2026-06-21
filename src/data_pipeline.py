import pandas as pd
import numpy as np
import json
import h3
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from config import (
    PARKING_CSV, PARKING_PARQUET, JUNCTION_PARQUET, CELL_AGG_PARQUET, RECURRENCE_PARQUET,
    H3_RESOLUTION, H3_RESOLUTION_COARSE,
    VEHICLE_WEIGHTS, DEFAULT_VEHICLE_WEIGHT,
    VIOLATION_SEVERITY, DEFAULT_VIOLATION_SEVERITY,
    PARKING_VIOLATION_KEYWORDS, TIME_WINDOW_HOURS,
)


def _parse_violations(raw: str) -> list[str]:
    if pd.isna(raw):
        return []
    try:
        return json.loads(raw.replace("'", '"'))
    except (json.JSONDecodeError, ValueError):
        return [raw.strip()]


def _max_severity(violations: list[str]) -> float:
    if not violations:
        return DEFAULT_VIOLATION_SEVERITY
    return max(VIOLATION_SEVERITY.get(v, DEFAULT_VIOLATION_SEVERITY) for v in violations)


def _has_parking(violations: list[str]) -> bool:
    return any(v in PARKING_VIOLATION_KEYWORDS for v in violations)


def _is_main_road_violation(violations: list[str]) -> bool:
    return "PARKING IN A MAIN ROAD" in violations or "DOUBLE PARKING" in violations


def load_and_clean(path=None) -> pd.DataFrame:
    """Load raw CSV, parse fields, add temporal and spatial features."""
    path = path or PARKING_CSV
    df = pd.read_csv(path)

    df["violations_list"] = df["violation_type"].apply(_parse_violations)
    df["has_parking"] = df["violations_list"].apply(_has_parking)

    df["veh_type_final"] = df["updated_vehicle_type"].where(
        df["updated_vehicle_type"].notna(), df["vehicle_type"]
    )

    df["created_datetime"] = pd.to_datetime(df["created_datetime"], format="mixed", utc=True)
    df["created_datetime"] = df["created_datetime"].dt.tz_convert("Asia/Kolkata")
    df["hour"] = df["created_datetime"].dt.hour
    df["dow"] = df["created_datetime"].dt.dayofweek
    df["dow_name"] = df["created_datetime"].dt.day_name()
    df["date"] = df["created_datetime"].dt.date
    df["year_week"] = (
        df["created_datetime"].dt.isocalendar().year.astype(str)
        + "-W"
        + df["created_datetime"].dt.isocalendar().week.astype(str).str.zfill(2)
    )
    df["is_weekend"] = df["dow"].isin([5, 6]).astype(int)
    df["time_window"] = df["hour"] // TIME_WINDOW_HOURS
    df["month"] = df["created_datetime"].dt.month

    df["h3_index"] = [
        h3.latlng_to_cell(lat, lng, H3_RESOLUTION)
        for lat, lng in zip(df["latitude"], df["longitude"])
    ]
    df["h3_coarse"] = [
        h3.latlng_to_cell(lat, lng, H3_RESOLUTION_COARSE)
        for lat, lng in zip(df["latitude"], df["longitude"])
    ]

    df["has_junction"] = (df["junction_name"] != "No Junction").astype(int)
    df["is_main_road_viol"] = df["violations_list"].apply(_is_main_road_violation).astype(int)

    return df


def compute_pcis(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-record Parking Congestion Impact Score."""
    df = df.copy()
    df["vehicle_weight"] = df["veh_type_final"].map(VEHICLE_WEIGHTS).fillna(DEFAULT_VEHICLE_WEIGHT)
    df["violation_severity"] = df["violations_list"].apply(_max_severity)
    df["location_weight"] = np.where(
        df["has_junction"] == 1, 2.0, np.where(df["is_main_road_viol"] == 1, 1.5, 1.0)
    )
    df["pcis"] = df["vehicle_weight"] * df["violation_severity"] * df["location_weight"]
    return df


def aggregate_cells(df: pd.DataFrame, resolution: str = "fine") -> pd.DataFrame:
    """Aggregate to (h3_cell x date x time_window) with PCIS totals and composition."""
    h3_col = "h3_index" if resolution == "fine" else "h3_coarse"
    agg = (
        df.groupby([h3_col, "date", "time_window"])
        .agg(
            total_pcis=("pcis", "sum"),
            count=("pcis", "size"),
            mean_pcis=("pcis", "mean"),
            heavy_ratio=("vehicle_weight", lambda x: (x >= 3.0).mean()),
            main_road_ratio=("is_main_road_viol", "mean"),
            unique_devices=("device_id", "nunique"),
            unique_officers=("created_by_id", "nunique"),
            lat=("latitude", "mean"),
            lng=("longitude", "mean"),
        )
        .reset_index()
        .rename(columns={h3_col: "h3_cell"})
    )
    return agg


def compute_recurrence(cell_agg: pd.DataFrame) -> pd.DataFrame:
    """For each (h3_cell, time_window), compute how many unique dates it appears in."""
    total_dates = cell_agg["date"].nunique()
    recurrence = (
        cell_agg.groupby(["h3_cell", "time_window"])
        .agg(
            active_days=("date", "nunique"),
            total_pcis_sum=("total_pcis", "sum"),
            total_count=("count", "sum"),
            avg_daily_pcis=("total_pcis", "mean"),
        )
        .reset_index()
    )
    recurrence["recurrence_ratio"] = recurrence["active_days"] / total_dates
    recurrence["recurrence_factor"] = np.clip(1.0 + 2.0 * recurrence["recurrence_ratio"], 1.0, 3.0)
    recurrence["weighted_pcis"] = recurrence["total_pcis_sum"] * recurrence["recurrence_factor"]
    return recurrence


def get_junction_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate PCIS by BTP junction for patrol-level output (one row per junction)."""
    jdf = df[df["junction_name"] != "No Junction"].copy()
    summary = (
        jdf.groupby("junction_name")
        .agg(
            total_pcis=("pcis", "sum"),
            count=("pcis", "size"),
            mean_pcis=("pcis", "mean"),
            heavy_ratio=("vehicle_weight", lambda x: (x >= 3.0).mean()),
            main_road_ratio=("is_main_road_viol", "mean"),
            lat=("latitude", "mean"),
            lng=("longitude", "mean"),
            unique_devices=("device_id", "nunique"),
            peak_hour=("hour", lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 0),
            police_station=("police_station", lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else ""),
        )
        .reset_index()
        .sort_values("total_pcis", ascending=False)
    )
    return summary


from typing import Optional


def load_bundled_pipeline() -> Optional[dict]:
    """Load pre-processed parquet bundles (for cloud deployment)."""
    if not PARKING_PARQUET.exists():
        return None
    parking_df = pd.read_parquet(PARKING_PARQUET)
    if "created_datetime" in parking_df.columns:
        parking_df["created_datetime"] = pd.to_datetime(parking_df["created_datetime"], utc=True)
    if "date" in parking_df.columns and parking_df["date"].dtype == object:
        parking_df["date"] = pd.to_datetime(parking_df["date"]).dt.date

    junction_summary = pd.read_parquet(JUNCTION_PARQUET) if JUNCTION_PARQUET.exists() else get_junction_summary(parking_df)
    cell_agg = pd.read_parquet(CELL_AGG_PARQUET) if CELL_AGG_PARQUET.exists() else aggregate_cells(parking_df, resolution="fine")
    recurrence = pd.read_parquet(RECURRENCE_PARQUET) if RECURRENCE_PARQUET.exists() else compute_recurrence(cell_agg)

    return {
        "raw": parking_df,
        "parking": parking_df,
        "cell_agg": cell_agg,
        "recurrence": recurrence,
        "junction_summary": junction_summary,
    }


def build_full_pipeline(path=None):
    """Run the complete data pipeline, return all key DataFrames."""
    bundled = load_bundled_pipeline()
    if bundled is not None and path is None:
        return bundled

    raw_df = load_and_clean(path)
    parking_df = raw_df[raw_df["has_parking"]].copy()
    parking_df = compute_pcis(parking_df)
    cell_agg = aggregate_cells(parking_df, resolution="fine")
    cell_agg_coarse = aggregate_cells(parking_df, resolution="coarse")
    recurrence = compute_recurrence(cell_agg)
    junction_summary = get_junction_summary(parking_df)
    return {
        "raw": raw_df,
        "parking": parking_df,
        "cell_agg": cell_agg,
        "cell_agg_coarse": cell_agg_coarse,
        "recurrence": recurrence,
        "junction_summary": junction_summary,
    }
