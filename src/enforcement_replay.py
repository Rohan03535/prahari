"""
Enforcement replay — proof-on-held-out-data analysis (junction or hex resolution).

Read-only. Same budget, same intercept radius for both strategies.
"""
from __future__ import annotations

from datetime import datetime
from math import atan2, cos, radians, sin, sqrt
from typing import Literal, Optional

import numpy as np
import pandas as pd

from config import PATROL_DEFAULTS

SHIFT_HOUR_MAP = {
    "Morning (08–12)": (8, 12),
    "Midday (12–16)": (12, 16),
    "Evening (16–20)": (16, 20),
    "Night (20–24)": (20, 24),
}
HOLDOUT_DAYS = 14
INTERCEPT_RADIUS_KM = 0.35  # same for ACTUAL and PRAHARI


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Vectorized haversine — inputs broadcastable."""
    r = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return r * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def stop_budget(n_units: Optional[int] = None, shift_hours: Optional[float] = None) -> int:
    n_units = n_units or PATROL_DEFAULTS["n_units"]
    shift_hours = shift_hours or PATROL_DEFAULTS["shift_hours"]
    dwell = PATROL_DEFAULTS["dwell_minutes"]
    return int((shift_hours * 60) / dwell) * n_units


def _assign_shift(hours: pd.Series) -> pd.Series:
    shift = pd.Series(index=hours.index, dtype=object)
    for label, (h0, h1) in SHIFT_HOUR_MAP.items():
        shift[(hours >= h0) & (hours < h1)] = label
    return shift


def split_holdout(parking_df: pd.DataFrame, holdout_days: int = HOLDOUT_DAYS) -> dict:
    dates = sorted(parking_df["date"].unique())
    if len(dates) <= holdout_days:
        holdout_days = max(1, len(dates) // 5)
    holdout_start = dates[-holdout_days]
    train_end = dates[-holdout_days - 1] if len(dates) > holdout_days else dates[0]
    train = parking_df[parking_df["date"] < holdout_start]
    held_out = parking_df[parking_df["date"] >= holdout_start]
    return {
        "train": train,
        "held_out": held_out,
        "train_start": dates[0],
        "train_end": train_end,
        "holdout_start": holdout_start,
        "holdout_end": dates[-1],
        "holdout_days": holdout_days,
    }


def build_actual_junction_deployment(train_df: pd.DataFrame, budget: int) -> pd.DataFrame:
    """Coded junctions only — ranked by training-period violation count."""
    jdf = train_df[train_df["junction_name"] != "No Junction"]
    return (
        jdf.groupby("junction_name")
        .agg(count=("pcis", "size"), total_pcis=("pcis", "sum"),
             lat=("latitude", "mean"), lng=("longitude", "mean"))
        .reset_index()
        .sort_values("count", ascending=False)
        .head(budget)
        .reset_index(drop=True)
    )


def _rank_junctions_causal(causal: pd.DataFrame, h0: int, h1: int, budget: int) -> pd.DataFrame:
    windowed = causal[
        (causal["hour"] >= h0) & (causal["hour"] < h1)
        & (causal["junction_name"] != "No Junction")
    ]
    if windowed.empty:
        return pd.DataFrame(columns=["junction_name", "total_pcis", "lat", "lng"])
    return (
        windowed.groupby("junction_name")
        .agg(total_pcis=("pcis", "sum"), lat=("latitude", "mean"), lng=("longitude", "mean"))
        .reset_index()
        .sort_values("total_pcis", ascending=False)
        .head(budget)
        .reset_index(drop=True)
    )


def _rank_hexes_causal(causal: pd.DataFrame, h0: int, h1: int, budget: int) -> pd.DataFrame:
    windowed = causal[(causal["hour"] >= h0) & (causal["hour"] < h1)]
    if windowed.empty:
        return pd.DataFrame(columns=["h3_index", "total_pcis", "lat", "lng", "off_grid"])
    ranked = (
        windowed.groupby("h3_index")
        .agg(
            total_pcis=("pcis", "sum"),
            lat=("latitude", "mean"),
            lng=("longitude", "mean"),
            off_grid=("has_junction", lambda x: (x == 0).any()),
        )
        .reset_index()
        .sort_values("total_pcis", ascending=False)
        .head(budget)
        .reset_index(drop=True)
    )
    return ranked


def build_prahari_deployments(
    parking_df: pd.DataFrame,
    holdout_start,
    holdout_end,
    budget: int,
    resolution: Literal["junction", "hex"] = "junction",
) -> dict:
    """Per (date, shift) deployment — causal ranking before each window."""
    deployments = {}
    holdout_dates = sorted(d for d in parking_df["date"].unique() if holdout_start <= d <= holdout_end)
    rank_fn = _rank_hexes_causal if resolution == "hex" else _rank_junctions_causal

    for day in holdout_dates:
        for shift_label, (h0, _) in SHIFT_HOUR_MAP.items():
            as_of = pd.Timestamp(
                datetime.combine(day, datetime.min.time()).replace(hour=h0),
                tz="Asia/Kolkata",
            )
            causal = parking_df[parking_df["created_datetime"] < as_of]
            deployments[(day, shift_label)] = rank_fn(causal, h0, SHIFT_HOUR_MAP[shift_label][1], budget)
    return deployments


def _within_radius(vlat, vlng, pts_lat, pts_lng, radius_km) -> np.ndarray:
    if len(pts_lat) == 0:
        return np.zeros(len(vlat), dtype=bool)
    vlat = np.asarray(vlat)
    vlng = np.asarray(vlng)
    pts_lat = np.asarray(pts_lat)
    pts_lng = np.asarray(pts_lng)
    # min distance to any deployment point
    min_d = np.full(len(vlat), np.inf)
    for plat, plng in zip(pts_lat, pts_lng):
        d = haversine_km(vlat, vlng, plat, plng)
        min_d = np.minimum(min_d, d)
    return min_d <= radius_km


def score_violations_fast(
    held_out: pd.DataFrame,
    actual_deployed: pd.DataFrame,
    prahari_deployments: dict,
    resolution: Literal["junction", "hex"] = "junction",
) -> pd.DataFrame:
    df = held_out.copy()
    df["shift"] = _assign_shift(df["hour"])
    df = df[df["shift"].notna()].copy()

    actual_juncs = set(actual_deployed["junction_name"].values) if not actual_deployed.empty else set()
    actual_lat = actual_deployed["lat"].values if not actual_deployed.empty else np.array([])
    actual_lng = actual_deployed["lng"].values if not actual_deployed.empty else np.array([])

    actual_hit = np.zeros(len(df), dtype=bool)
    if actual_juncs:
        actual_hit |= df["junction_name"].isin(actual_juncs).values
    if len(actual_lat):
        actual_hit |= _within_radius(df["latitude"].values, df["longitude"].values, actual_lat, actual_lng, INTERCEPT_RADIUS_KM)

    prahari_hit = np.zeros(len(df), dtype=bool)
    for (day, shift), grp in df.groupby(["date", "shift"]):
        dep = prahari_deployments.get((day, shift), pd.DataFrame())
        if dep.empty:
            continue
        idx = grp.index
        pos = df.index.get_indexer(idx)

        if resolution == "hex":
            hex_set = set(dep["h3_index"].values)
            prahari_hit[pos] |= grp["h3_index"].isin(hex_set).values
        else:
            jset = set(dep["junction_name"].values) if "junction_name" in dep.columns else set()
            prahari_hit[pos] |= grp["junction_name"].isin(jset).values

        if len(dep):
            radius_hit = _within_radius(
                grp["latitude"].values, grp["longitude"].values,
                dep["lat"].values, dep["lng"].values, INTERCEPT_RADIUS_KM,
            )
            prahari_hit[pos] |= radius_hit

    df["actual_intercepted"] = actual_hit
    df["prahari_intercepted"] = prahari_hit
    return df


def summarize_scores(scored: pd.DataFrame) -> dict:
    def _side(col: str) -> dict:
        hit = scored[scored[col]]
        miss = scored[~scored[col]]
        return {
            "intercepted_count": int(len(hit)),
            "missed_count": int(len(miss)),
            "intercepted_pcis": float(hit["pcis"].sum()),
            "missed_pcis": float(miss["pcis"].sum()),
            "total_count": int(len(scored)),
            "total_pcis": float(scored["pcis"].sum()),
        }

    actual = _side("actual_intercepted")
    prahari = _side("prahari_intercepted")
    pct = (
        (prahari["intercepted_pcis"] - actual["intercepted_pcis"]) / actual["intercepted_pcis"] * 100
        if actual["intercepted_pcis"] > 0 else 0.0
    )
    return {"actual": actual, "prahari": prahari, "improvement_pcis_pct": pct}


def _hex_audit(prahari_deployments: dict, held_out: pd.DataFrame) -> dict:
    """Verify deployed off-grid hexes have real held-out violations."""
    all_hex = []
    off_grid_hex = []
    for dep in prahari_deployments.values():
        if dep.empty or "h3_index" not in dep.columns:
            continue
        all_hex.extend(dep["h3_index"].tolist())
        if "off_grid" in dep.columns:
            off_grid_hex.extend(dep.loc[dep["off_grid"], "h3_index"].tolist())

    held_hex = set(held_out["h3_index"].unique())
    deployed_set = set(all_hex)
    off_grid_set = set(off_grid_hex)
    off_grid_with_violations = off_grid_set & held_hex

    return {
        "deployed_hexes": len(deployed_set),
        "off_grid_hexes_deployed": len(off_grid_set),
        "off_grid_with_heldout_violations": len(off_grid_with_violations),
    }


def pick_representative_day(scored: pd.DataFrame):
    if scored.empty:
        return None
    return scored.groupby("date").size().idxmax()


def day_map_frames(scored_day, actual_deployed, prahari_deployments, day):
    v_act = scored_day.copy()
    v_act["status"] = np.where(v_act["actual_intercepted"], "Intercepted", "Missed")
    v_pra = scored_day.copy()
    v_pra["status"] = np.where(v_pra["prahari_intercepted"], "Intercepted", "Missed")

    prahari_union = []
    for (d, shift), dep in prahari_deployments.items():
        if d == day and not dep.empty:
            tmp = dep.copy()
            tmp["shift"] = shift
            prahari_union.append(tmp)
    prahari_markers = pd.concat(prahari_union, ignore_index=True) if prahari_union else pd.DataFrame()
    actual_markers = actual_deployed.copy()
    return v_act, v_pra, actual_markers, prahari_markers


def run_enforcement_replay(
    parking_df: pd.DataFrame,
    n_units: Optional[int] = None,
    shift_hours: Optional[float] = None,
    holdout_days: int = HOLDOUT_DAYS,
    resolution: Literal["junction", "hex"] = "junction",
) -> dict:
    budget = stop_budget(n_units, shift_hours)
    split = split_holdout(parking_df, holdout_days)

    actual_deployed = build_actual_junction_deployment(split["train"], budget)
    prahari_deployments = build_prahari_deployments(
        parking_df, split["holdout_start"], split["holdout_end"], budget, resolution
    )

    scored = score_violations_fast(
        split["held_out"], actual_deployed, prahari_deployments, resolution
    )
    summary = summarize_scores(scored)
    hex_audit = _hex_audit(prahari_deployments, split["held_out"]) if resolution == "hex" else None
    rep_day = pick_representative_day(scored)

    day_frames = None
    if rep_day is not None:
        day_frames = day_map_frames(
            scored[scored["date"] == rep_day], actual_deployed, prahari_deployments, rep_day
        )

    return {
        "resolution": resolution,
        "split": split,
        "budget": budget,
        "intercept_radius_km": INTERCEPT_RADIUS_KM,
        "n_units": n_units or PATROL_DEFAULTS["n_units"],
        "shift_hours": shift_hours or PATROL_DEFAULTS["shift_hours"],
        "actual_deployed": actual_deployed,
        "prahari_deployments": prahari_deployments,
        "scored": scored,
        "summary": summary,
        "hex_audit": hex_audit,
        "representative_day": rep_day,
        "day_frames": day_frames,
    }
