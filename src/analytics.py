import numpy as np
import pandas as pd
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))


def enforcement_efficacy(parking_df: pd.DataFrame) -> dict:
    """
    Measure whether enforcement (high-ticketing weeks) at a junction
    reduced violations in subsequent weeks. Simple before/after comparison.
    """
    junctions = parking_df[parking_df["junction_name"] != "No Junction"]
    weekly = (
        junctions.groupby(["junction_name", "year_week"])
        .agg(weekly_pcis=("pcis", "sum"), weekly_count=("pcis", "size"))
        .reset_index()
        .sort_values(["junction_name", "year_week"])
    )

    results = []
    for junc, gdf in weekly.groupby("junction_name"):
        if len(gdf) < 6:
            continue
        gdf = gdf.sort_values("year_week").reset_index(drop=True)
        median_pcis = gdf["weekly_pcis"].median()
        high_weeks = gdf[gdf["weekly_pcis"] > median_pcis * 1.5]

        for _, hw in high_weeks.iterrows():
            idx = gdf[gdf["year_week"] == hw["year_week"]].index
            if len(idx) == 0:
                continue
            i = idx[0]
            if i < 2 or i >= len(gdf) - 2:
                continue
            before = gdf.iloc[i - 2 : i]["weekly_pcis"].mean()
            after = gdf.iloc[i + 1 : i + 3]["weekly_pcis"].mean()
            if before > 0:
                change_pct = ((after - before) / before) * 100
                results.append({
                    "junction": junc,
                    "enforcement_week": hw["year_week"],
                    "pcis_before": before,
                    "pcis_after": after,
                    "change_pct": change_pct,
                    "effective": change_pct < -10,
                })

    results_df = pd.DataFrame(results)
    if results_df.empty:
        return {"results": results_df, "summary": {}}

    summary = {
        "total_events": len(results_df),
        "effective_events": results_df["effective"].sum(),
        "effectiveness_rate": results_df["effective"].mean() * 100,
        "avg_change_pct": results_df["change_pct"].mean(),
        "best_junction": (
            results_df.loc[results_df["change_pct"].idxmin(), "junction"]
            if not results_df.empty else "N/A"
        ),
        "worst_junction": (
            results_df.loc[results_df["change_pct"].idxmax(), "junction"]
            if not results_df.empty else "N/A"
        ),
    }

    return {"results": results_df, "summary": summary}


def chronic_offender_analysis(parking_df: pd.DataFrame, min_violations: int = 5) -> pd.DataFrame:
    """Rank vehicles by total PCIS impact, not just count."""
    offenders = (
        parking_df.groupby("vehicle_number")
        .agg(
            total_pcis=("pcis", "sum"),
            violation_count=("pcis", "size"),
            vehicle_type=("veh_type_final", lambda x: x.mode().iloc[0] if len(x) > 0 else "UNKNOWN"),
            top_violation=("violation_type", lambda x: x.mode().iloc[0] if len(x) > 0 else "UNKNOWN"),
            top_station=("police_station", lambda x: x.mode().iloc[0] if len(x) > 0 else "UNKNOWN"),
            unique_locations=("h3_index", "nunique"),
            first_seen=("created_datetime", "min"),
            last_seen=("created_datetime", "max"),
            approved_count=("validation_status", lambda x: (x == "approved").sum()),
        )
        .reset_index()
    )
    offenders = offenders[offenders["violation_count"] >= min_violations]
    offenders["days_active"] = (offenders["last_seen"] - offenders["first_seen"]).dt.days + 1
    offenders["violations_per_day"] = offenders["violation_count"] / offenders["days_active"].clip(lower=1)
    offenders["approval_rate"] = offenders["approved_count"] / offenders["violation_count"]
    offenders["is_repeat_location"] = (offenders["unique_locations"] == 1).astype(int)

    return offenders.sort_values("total_pcis", ascending=False).reset_index(drop=True)


def system_health_analysis(parking_df: pd.DataFrame) -> dict:
    """Analyze rejection patterns by station, device, and officer."""
    df = parking_df.copy()
    df["is_rejected"] = (df["validation_status"] == "rejected").astype(int)
    df["is_approved"] = (df["validation_status"] == "approved").astype(int)

    station_quality = (
        df.groupby("police_station")
        .agg(
            total=("pcis", "size"),
            approved=("is_approved", "sum"),
            rejected=("is_rejected", "sum"),
            unique_devices=("device_id", "nunique"),
            unique_officers=("created_by_id", "nunique"),
        )
        .reset_index()
    )
    station_quality["rejection_rate"] = station_quality["rejected"] / station_quality["total"]
    station_quality["approval_rate"] = station_quality["approved"] / station_quality["total"]
    station_quality = station_quality.sort_values("rejection_rate", ascending=False)

    device_quality = (
        df.groupby("device_id")
        .agg(
            total=("pcis", "size"),
            approved=("is_approved", "sum"),
            rejected=("is_rejected", "sum"),
            station=("police_station", lambda x: x.mode().iloc[0] if len(x) > 0 else "UNKNOWN"),
        )
        .reset_index()
    )
    device_quality["rejection_rate"] = device_quality["rejected"] / device_quality["total"]
    device_quality = device_quality[device_quality["total"] >= 50].sort_values("rejection_rate", ascending=False)

    hourly_quality = (
        df.groupby("hour")
        .agg(total=("pcis", "size"), rejected=("is_rejected", "sum"))
        .reset_index()
    )
    hourly_quality["rejection_rate"] = hourly_quality["rejected"] / hourly_quality["total"]

    city_avg_rejection = df["is_rejected"].mean()

    return {
        "station_quality": station_quality,
        "device_quality": device_quality,
        "hourly_quality": hourly_quality,
        "city_avg_rejection_rate": city_avg_rejection,
    }


def temporal_trends(parking_df: pd.DataFrame) -> pd.DataFrame:
    """Weekly PCIS trend for the city and per-station."""
    weekly = (
        parking_df.groupby("year_week")
        .agg(
            total_pcis=("pcis", "sum"),
            count=("pcis", "size"),
            unique_junctions=("junction_name", "nunique"),
            heavy_ratio=("vehicle_weight", lambda x: (x >= 3.0).mean()),
        )
        .reset_index()
        .sort_values("year_week")
    )
    return weekly


def station_comparison(parking_df: pd.DataFrame) -> pd.DataFrame:
    """Compare stations by PCIS per enforcement unit (officer)."""
    station = (
        parking_df.groupby("police_station")
        .agg(
            total_pcis=("pcis", "sum"),
            count=("pcis", "size"),
            unique_officers=("created_by_id", "nunique"),
            unique_junctions=("junction_name", "nunique"),
            heavy_ratio=("vehicle_weight", lambda x: (x >= 3.0).mean()),
            approval_rate=("validation_status", lambda x: (x == "approved").mean()),
        )
        .reset_index()
    )
    station["pcis_per_officer"] = station["total_pcis"] / station["unique_officers"].clip(lower=1)
    return station.sort_values("total_pcis", ascending=False)
