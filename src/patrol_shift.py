"""Shift-window junction ranking for patrol routing."""
import pandas as pd

SHIFT_HOUR_MAP = {
    "Morning (08–12)": (8, 12),
    "Midday (12–16)": (12, 16),
    "Evening (16–20)": (16, 20),
    "Night (20–24)": (20, 24),
}


def junction_summary_for_shift(parking_df: pd.DataFrame, shift_label: str) -> pd.DataFrame:
    """Re-rank junctions by PCIS within the selected shift window."""
    h_start, h_end = SHIFT_HOUR_MAP[shift_label]
    mask = (parking_df["hour"] >= h_start) & (parking_df["hour"] < h_end)
    jdf = parking_df[(mask) & (parking_df["junction_name"] != "No Junction")].copy()
    if jdf.empty:
        return pd.DataFrame(columns=[
            "junction_name", "total_pcis", "count", "mean_pcis", "heavy_ratio",
            "main_road_ratio", "lat", "lng", "unique_devices", "peak_hour", "police_station",
        ])

    return (
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
