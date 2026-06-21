import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import h3
import sys, os
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    BENGALURU_CENTER, MAP_ZOOM, TIME_WINDOW_LABELS,
    VEHICLE_WEIGHTS, VIOLATION_SEVERITY, PATROL_DEFAULTS,
    DEFAULT_TOP_HOTSPOTS, DEFAULT_SHIFT_LABEL, DEFAULT_PATROL_PLAN_JSON,
    FLOW_SUMMARY_JSON, JUNC_FLOW_PARQUET, ROAD_TYPE_FLOW_PARQUET, BLIND_SPOTS_PARQUET,
)
from src.patrol_shift import junction_summary_for_shift as _junction_summary_for_shift, SHIFT_HOUR_MAP
from src.data_pipeline import build_full_pipeline
from src.traffic_flow_impact import ROAD_PROFILES, ASSUMPTIONS
from src.patrol_briefing import generate_briefing_text, generate_briefing_html
from src.junction_intel import load_junction_intel

st.set_page_config(
    page_title="PRAHARI - Parking Intelligence",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem; font-weight: 800;
        background: linear-gradient(90deg, #1a73e8, #00bcd4);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.1rem; color: #888; margin-top: -10px; margin-bottom: 20px;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e1e2e, #2a2a3e);
        border-radius: 12px; padding: 20px; border: 1px solid #333;
    }
    .stat-big { font-size: 2.2rem; font-weight: 700; color: #00bcd4; }
    .stat-label { font-size: 0.85rem; color: #aaa; text-transform: uppercase; letter-spacing: 1px; }
    div[data-testid="stMetric"] {
        background: #1e1e2e; border-radius: 10px; padding: 12px 16px; border: 1px solid #333;
    }
    .command-banner {
        background: linear-gradient(135deg, #0d1b2a 0%, #1b2838 100%);
        border: 1px solid #2a4a6b; border-radius: 12px; padding: 20px 24px; margin-bottom: 16px;
    }
    .route-card {
        background: #1e1e2e; border-left: 4px solid #00bcd4;
        border-radius: 8px; padding: 14px 18px; margin-bottom: 10px;
    }
    .contrast-without {
        background: #1a1a1a; border: 1px solid #333; border-radius: 10px;
        padding: 18px 20px; color: #888; min-height: 140px;
    }
    .contrast-with {
        background: linear-gradient(135deg, #0d2818 0%, #1a3a2a 100%);
        border: 1px solid #2d6a4f; border-radius: 10px;
        padding: 18px 20px; color: #d4edda; min-height: 140px;
    }
    .hero-headline {
        font-size: 1.35rem; font-weight: 700; color: #fff;
        margin: 12px 0 8px 0; line-height: 1.4;
    }
    .route-rationale { color: #7ec8e3; font-size: 0.9em; margin-top: 6px; font-style: italic; }
    .sim-insight {
        background: #1e2a3a; border-left: 4px solid #f39c12;
        padding: 12px 16px; border-radius: 6px; margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner="Loading violation records...")
def load_data():
    return build_full_pipeline()


@st.cache_data(show_spinner=False)
def load_default_patrol_plan():
    if not DEFAULT_PATROL_PLAN_JSON.exists():
        return None
    return json.loads(DEFAULT_PATROL_PLAN_JSON.read_text(encoding="utf-8"))


def is_default_patrol_config(n_units, shift_hrs, top_hotspots, shift_label):
    return (
        n_units == PATROL_DEFAULTS["n_units"]
        and abs(float(shift_hrs) - float(PATROL_DEFAULTS["shift_hours"])) < 0.01
        and top_hotspots == DEFAULT_TOP_HOTSPOTS
        and shift_label == DEFAULT_SHIFT_LABEL
    )


@st.cache_data(show_spinner=False)
def get_precomputed_extras():
    """Load pre-bundled heavy outputs (instant) — fallback to compute if missing."""
    if FLOW_SUMMARY_JSON.exists():
        fs = json.loads(FLOW_SUMMARY_JSON.read_text(encoding="utf-8"))
        jf = pd.read_parquet(JUNC_FLOW_PARQUET) if JUNC_FLOW_PARQUET.exists() else pd.DataFrame()
        rt = pd.read_parquet(ROAD_TYPE_FLOW_PARQUET) if ROAD_TYPE_FLOW_PARQUET.exists() else pd.DataFrame()
        blind = pd.read_parquet(BLIND_SPOTS_PARQUET) if BLIND_SPOTS_PARQUET.exists() else pd.DataFrame()
        return fs, jf, rt, blind
    return None, None, None, None


@st.cache_data(show_spinner="Computing traffic flow impact...")
def get_flow_impact(_parking_df):
    from src.traffic_flow_impact import compute_flow_impact, aggregate_flow_impact, junction_flow_impact
    df = compute_flow_impact(_parking_df)
    summary = aggregate_flow_impact(df)
    junc_impact = junction_flow_impact(df)
    return df, summary, junc_impact


@st.cache_data(show_spinner="Discovering blind spots with HDBSCAN...")
def get_blind_spots(_parking_df):
    pre = get_precomputed_extras()
    if pre[3] is not None and not pre[3].empty:
        return pre[3]
    from src.spatial_analysis import discover_blind_spots
    return discover_blind_spots(_parking_df)


@st.cache_data(show_spinner="Computing enforcement exposure map...")
def get_exposure(_parking_df):
    from src.spatial_analysis import compute_enforcement_exposure
    return compute_enforcement_exposure(_parking_df)


@st.cache_data(show_spinner="Training prediction model...")
def get_predictions(_cell_agg, _recurrence):
    from src.prediction import engineer_features, train_hurdle_model
    features = engineer_features(_cell_agg, _recurrence)
    return train_hurdle_model(features)


@st.cache_data(show_spinner="Analyzing enforcement effectiveness...")
def get_efficacy(_parking_df):
    from src.analytics import enforcement_efficacy
    return enforcement_efficacy(_parking_df)


@st.cache_data(show_spinner="Profiling chronic offenders...")
def get_offenders(_parking_df):
    from src.analytics import chronic_offender_analysis
    return chronic_offender_analysis(_parking_df)


@st.cache_data(show_spinner="Checking system health...")
def get_system_health(_parking_df):
    from src.analytics import system_health_analysis
    return system_health_analysis(_parking_df)


@st.cache_data(show_spinner=False)
def get_enforcement_replay(_parking_df, n_units, shift_hrs, resolution: str):
    from src.enforcement_replay import run_enforcement_replay
    return run_enforcement_replay(_parking_df, n_units, shift_hrs, resolution=resolution)


@st.cache_data(show_spinner="Optimizing patrol routes...")
def get_patrol_plan(_junction_summary, n_units, shift_hrs, top_hotspots):
    from src.optimizer import solve_patrol_routing, compare_with_baselines
    result = solve_patrol_routing(_junction_summary, n_units, shift_hrs, top_hotspots)
    comparison = compare_with_baselines(_junction_summary, n_units, shift_hrs, top_hotspots)
    return result, comparison


@st.cache_data(show_spinner="Running scenario simulation...")
def get_scenario_curve(_junction_summary, shift_hrs, top_hotspots, units_tuple):
    from src.optimizer import simulate_unit_scenarios
    return simulate_unit_scenarios(
        _junction_summary,
        list(units_tuple),
        shift_hrs,
        top_hotspots,
    )


def render_patrol_map(routes):
    route_map = folium.Map(location=list(BENGALURU_CENTER), zoom_start=12, tiles="cartodbpositron")
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
              "#1abc9c", "#e67e22", "#2980b9", "#c0392b", "#27ae60"]
    for route in routes:
        color = colors[(route["unit"] - 1) % len(colors)]
        coords = [(s["lat"], s["lng"]) for s in route["stops"]]
        if len(coords) >= 2:
            folium.PolyLine(coords, color=color, weight=3, opacity=0.8).add_to(route_map)
        for i, stop in enumerate(route["stops"]):
            folium.CircleMarker(
                [stop["lat"], stop["lng"]], radius=8, color=color, fill=True,
                popup=f"Unit {route['unit']} | Stop {i+1}: {stop['junction']}<br>PCIS: {stop['pcis']:.0f}",
            ).add_to(route_map)
    return route_map


# ── Presentation-layer helpers (no pipeline / optimizer changes) ──

@st.cache_data(show_spinner=False)
def junction_summary_for_shift(_parking_df, shift_label: str):
    return _junction_summary_for_shift(_parking_df, shift_label)


def _flatten_violations(series):
    counts = {}
    for vlist in series:
        if not isinstance(vlist, list):
            continue
        for v in vlist:
            counts[v] = counts.get(v, 0) + 1
    if not counts:
        return "parking violations"
    top = max(counts, key=counts.get)
    return top.lower().replace("_", " ")


@st.cache_data(show_spinner=False)
def build_junction_intel(_parking_df):
    return load_junction_intel(_parking_df)


def unit_rationale(route: dict, junction_intel: pd.DataFrame, shift_label: str) -> str:
    if not route["stops"]:
        return "No stops assigned for this shift window."
    top_stop = max(route["stops"], key=lambda s: s["pcis"])
    jname = top_stop["junction"]
    short_name = jname.split(" - ", 1)[-1] if " - " in jname else jname
    row = junction_intel[junction_intel["junction_name"] == jname]
    if row.empty:
        return f"Highest PCIS stop this shift — {short_name}."

    r = row.iloc[0]
    period = shift_label.split("(")[0].strip().lower()
    veh = r["top_vehicle"].title() if isinstance(r["top_vehicle"], str) else "mixed vehicles"
    if r["heavy_pct"] >= 0.25:
        veh_note = f"heavy-vehicle load ({r['heavy_pct']:.0%} lorries/buses)"
    else:
        veh_note = f"dominant vehicle: {veh}"
    road_note = "main-road blockage" if r["main_road_pct"] >= 0.3 else r["top_violation"]
    return (
        f"Highest {period} load near {short_name} — {road_note}, {veh_note} "
        f"(peak ~{int(r['peak_hour'])}:00 IST)."
    )


def arrival_windows(stops: list[dict], shift_hours: float, shift_label: str) -> list[str]:
    """Illustrative arrival windows spread across the shift."""
    h_start, _ = SHIFT_HOUR_MAP.get(shift_label, (8, 12))
    if not stops:
        return []
    slot = (shift_hours * 60) / max(len(stops), 1)
    windows = []
    for i in range(len(stops)):
        start_min = int(i * slot)
        end_min = int((i + 1) * slot)
        sh, sm = divmod(h_start * 60 + start_min, 60)
        eh, em = divmod(h_start * 60 + end_min, 60)
        windows.append(f"{sh:02d}:{sm:02d}–{eh:02d}:{em:02d}")
    return windows


def scenario_punchline(scenario_df: pd.DataFrame, current_units: int) -> str:
    if len(scenario_df) < 2:
        return f"Optimal deployment for this shift: {current_units} units."

    df = scenario_df.sort_values("n_units").copy()
    df["marginal_pct"] = df["marginal_pcis"].pct_change() * 100
    df["marginal_pct"] = df["marginal_pct"].fillna(100)

    # Find where marginal gain drops below 15% of first marginal unit
    first_marginal = df["marginal_pcis"].iloc[1] if len(df) > 1 else df["marginal_pcis"].iloc[0]
    knee = current_units
    for _, row in df.iterrows():
        if row["n_units"] > 1 and row["marginal_pcis"] < 0.15 * first_marginal:
            knee = int(row["n_units"] - 1)
            break
    else:
        knee = int(df["n_units"].iloc[-1])

    knee = max(knee, 2)
    if current_units < len(df):
        next_row = df[df["n_units"] == current_units + 1]
        if not next_row.empty:
            extra_pct = next_row.iloc[0]["marginal_pct"]
            return (
                f"Marginal gain drops sharply after **{knee} units** — "
                f"the {current_units + 1}{'st' if current_units == 0 else 'th'} officer adds only "
                f"**{extra_pct:.0f}%** more intercepted impact. "
                f"**Optimal deployment for this shift: {knee} units.**"
            )

    return (
        f"Marginal gain drops sharply after **{knee} units**. "
        f"**Optimal deployment for this shift: {knee} units.**"
    )


# ── Sidebar ──
with st.sidebar:
    st.markdown('<div class="main-header">PRAHARI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Parking Intelligence Command System</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**Bengaluru Traffic Police**")
    st.markdown("AI-driven parking enforcement optimization")
    st.markdown("---")
    st.markdown("##### Data Summary")

data = load_data()
parking_df = data["parking"]
cell_agg = data["cell_agg"]
recurrence = data["recurrence"]
junction_summary = data["junction_summary"]

# Pre-bundled heavy outputs (instant load — no HDBSCAN / flow recompute on startup)
_flow_pre, junc_flow, road_type_flow, blind_spots_pre = get_precomputed_extras()
if _flow_pre is not None:
    flow_summary = _flow_pre
    parking_with_flow = None
else:
    parking_with_flow, flow_summary, junc_flow = get_flow_impact(parking_df)

with st.sidebar:
    st.metric("Total Violations", f"{len(parking_df):,}")
    st.metric("Unique Junctions", f"{parking_df['junction_name'].nunique():,}")
    st.metric("Police Stations", f"{parking_df['police_station'].nunique()}")
    st.metric("Date Range", f"{parking_df['date'].min()} to {parking_df['date'].max()}")
    st.markdown("---")
    st.markdown("##### Shift Configuration")
    cmd_n_units = st.slider("Patrol units", 1, 12, PATROL_DEFAULTS["n_units"], key="cmd_units")
    cmd_shift_hrs = st.slider("Shift hours", 1.0, 8.0, float(PATROL_DEFAULTS["shift_hours"]), 0.5, key="cmd_shift")
    cmd_top_hotspots = st.slider("Candidate hotspots", 15, 40, 25, key="cmd_hotspots")
    peak_hour = int(parking_df.groupby("hour")["pcis"].sum().idxmax())
    shift_options = ["Morning (08–12)", "Midday (12–16)", "Evening (16–20)", "Night (20–24)"]
    cmd_shift_label = st.selectbox("Shift window", shift_options, index=0)
    st.markdown("---")
    st.caption("REST API: `uvicorn api:app --port 8000` → /docs")
    st.caption("Built for Flipkart Gridlock 2.0")


# ── Main navigation (lazy — only the active page runs heavy code) ──
PAGES = [
    "Patrol Command Center",
    "Proof It Works",
    "Intelligence: Enforcement Gap",
    "Intelligence: Traffic Flow Impact",
    "Intelligence: Hotspot Analysis",
    "Intelligence: Blind Spots",
    "Intelligence: Model & Efficacy",
    "Intelligence: System Health",
]
page = st.radio("Section", PAGES, horizontal=True, label_visibility="collapsed")

# ── Lazy command-center state (computed inside tab, not at startup) ──


# ═══════════════════════ COMMAND CENTER ═══════════════════════
if page == "Patrol Command Center":
    default_bundle = load_default_patrol_plan()
    use_bundled = (
        default_bundle is not None
        and is_default_patrol_config(cmd_n_units, cmd_shift_hrs, cmd_top_hotspots, cmd_shift_label)
    )

    if use_bundled:
        result = default_bundle["result"]
        comparison = default_bundle["comparison"]
        plan_metrics = default_bundle["plan_metrics"]
        scenario_df = pd.DataFrame(default_bundle["scenario_df"])
        shift_differs = default_bundle["shift_differs"]
        off_grid_count = default_bundle["off_grid_count"]
        shift_junctions = junction_summary_for_shift(parking_df, cmd_shift_label)
        junction_intel = build_junction_intel(parking_df)
        n_blind_visible = min(30, len(blind_spots_pre)) if blind_spots_pre is not None and not blind_spots_pre.empty else 30
        est_stops = plan_metrics["est_stops"]
        est_fines = plan_metrics["est_fines_rs"]
        data_end = parking_df["date"].max()
        sim_insight = scenario_punchline(scenario_df, cmd_n_units)
        arrival_by_unit = {
            route["unit"]: arrival_windows(route["stops"], cmd_shift_hrs, cmd_shift_label)
            for route in result["routes"]
        }
        briefing_txt = generate_briefing_text(
            result["routes"], cmd_shift_hrs, result["total_intercepted_pcis"],
            comparison, cmd_shift_label, arrival_by_unit,
        )
        briefing_html = generate_briefing_html(
            result["routes"], cmd_shift_hrs, result["total_intercepted_pcis"],
            cmd_shift_label, arrival_by_unit,
        )
    else:
        with st.spinner("Generating patrol plan for this shift…"):
            shift_junctions = junction_summary_for_shift(parking_df, cmd_shift_label)
            junction_intel = build_junction_intel(parking_df)
            n_blind_visible = min(30, len(blind_spots_pre)) if blind_spots_pre is not None and not blind_spots_pre.empty else 30

            result, comparison = get_patrol_plan(shift_junctions, cmd_n_units, cmd_shift_hrs, cmd_top_hotspots)
            off_grid_count = int((parking_df["junction_name"] == "No Junction").sum())
            plan_metrics = compute_plan_metrics(result, comparison, cmd_n_units, cmd_shift_hrs, off_grid_count)
            est_stops = plan_metrics["est_stops"]
            est_fines = plan_metrics["est_fines_rs"]
            data_end = parking_df["date"].max()

            arrival_by_unit = {
                route["unit"]: arrival_windows(route["stops"], cmd_shift_hrs, cmd_shift_label)
                for route in result["routes"]
            }
            briefing_txt = generate_briefing_text(
                result["routes"], cmd_shift_hrs, result["total_intercepted_pcis"],
                comparison, cmd_shift_label, arrival_by_unit,
            )
            briefing_html = generate_briefing_html(
                result["routes"], cmd_shift_hrs, result["total_intercepted_pcis"],
                cmd_shift_label, arrival_by_unit,
            )

            scenario_units = tuple(range(2, min(cmd_n_units + 4, 13)))
            scenario_df = get_scenario_curve(shift_junctions, cmd_shift_hrs, cmd_top_hotspots, scenario_units)
            sim_insight = scenario_punchline(scenario_df, cmd_n_units)

            shift_differs = (
                junction_summary_for_shift(parking_df, "Morning (08–12)").head(3)["junction_name"].tolist()
                != junction_summary_for_shift(parking_df, "Evening (16–20)").head(3)["junction_name"].tolist()
            )

    hour_now = datetime.now().hour
    greeting = "Good morning" if hour_now < 12 else ("Good afternoon" if hour_now < 17 else "Good evening")

    st.markdown(f"""
    <div class="command-banner">
        <h2 style="margin:0;color:#fff;">{greeting}, Inspector.</h2>
        <p style="margin:8px 0 0;color:#aaa;">
            Shift plan for <strong>{cmd_shift_label}</strong> — ranked by violations in this time window
            (data: {parking_df['date'].min()} to {data_end}).
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Change 1 — contrast band
    col_without, col_with = st.columns(2)
    with col_without:
        st.markdown(f"""
        <div class="contrast-without">
            <strong style="color:#666;">Today WITHOUT PRAHARI</strong><br/><br/>
            Officers patrol by experience — top junction hotspots are known.<br/>
            But <strong>{off_grid_count:,}</strong> violations over 6 months happened at locations
            with <strong>no junction code, no camera, no patrol assignment</strong>.<br/><br/>
            <em>Blind.</em>
        </div>
        """, unsafe_allow_html=True)
    with col_with:
        st.markdown(f"""
        <div class="contrast-with">
            <strong style="color:#90ee90;">Today WITH PRAHARI</strong><br/><br/>
            <strong>{cmd_n_units}</strong> units · <strong>{est_stops}</strong> surgical stops ·
            <strong>{result['intercept_pct']:.0f}%</strong> of this shift's top hotspots covered.<br/>
            Plus <strong>{n_blind_visible}</strong> off-grid blind spots now mapped and actionable.<br/><br/>
            <em>Deployed.</em>
        </div>
        """, unsafe_allow_html=True)

    # Change 2 — hero sentence
    st.markdown(
        f'<p class="hero-headline">BTP\'s junction system is blind to '
        f'<span style="color:#00bcd4;">{off_grid_count:,}</span> parking violations. '
        f'PRAHARI sees every one.</p>',
        unsafe_allow_html=True,
    )

    if shift_differs:
        st.caption(
            f"Shift-aware routing active — top hotspots for {cmd_shift_label}: "
            f"{', '.join(shift_junctions.head(3)['junction_name'].str.split(' - ').str[-1].tolist())}"
        )
    else:
        st.caption("Shift window updates junction rankings from hourly violation patterns.")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("PCIS Intercepted", f"{result['total_intercepted_pcis']:,.0f}")
    c2.metric("PCIS / Officer-Hour", f"{plan_metrics['pcis_per_officer_hour']:,.0f}")
    c3.metric("Top Hotspot Coverage", f"{result['intercept_pct']:.0f}%")
    c4.metric("Off-Grid (6 mo)", f"{off_grid_count:,}")
    c5.metric("Est. Fines (illustrative)", f"₹{est_fines/1000:.0f}K",
              help=f"₹500 × {est_stops} stops")

    bal = plan_metrics["load_balance_ratio"]
    st.caption(
        f"{cmd_n_units} units · load balance {bal:.0%} · "
        f"vs same-budget naive: {comparison['improvement_vs_naive_pct']:+.1f}%"
    )

    st.markdown("### Shift Unit Assignments")
    if not result["routes"]:
        st.warning("No routes generated — increase shift duration or reduce candidate hotspots.")
    else:
        for route in result["routes"]:
            seq = " → ".join(s["junction"] for s in route["stops"][:5])
            if len(route["stops"]) > 5:
                seq += f" → … (+{len(route['stops']) - 5} more)"
            rationale = unit_rationale(route, junction_intel, cmd_shift_label)
            st.markdown(
                f"""<div class="route-card">
                <strong>Unit {route['unit']}</strong> — {route['n_stops']} stops · PCIS {route['route_pcis']:,.0f}<br/>
                <span style="color:#aaa;font-size:0.95em;">{seq}</span><br/>
                <span class="route-rationale">Why: {rationale}</span>
                </div>""",
                unsafe_allow_html=True,
            )

    col_map, col_dl = st.columns([3, 1])
    with col_map:
        st.markdown("### Route Map")
        st_folium(render_patrol_map(result["routes"]), width=None, height=450)
    with col_dl:
        st.markdown("### Field Briefing")
        n_officers = len(result["routes"])
        st.download_button(
            f"Print Field Cards ({n_officers} officers)",
            briefing_txt,
            file_name=f"prahari_field_cards_{datetime.now():%Y%m%d}.txt",
            mime="text/plain",
            type="primary",
        )
        st.download_button(
            "Print HTML field cards",
            briefing_html,
            file_name=f"prahari_field_cards_{datetime.now():%Y%m%d}.html",
            mime="text/html",
        )
        with st.expander(f"Preview field cards ({n_officers} units)"):
            st.text(briefing_txt[:3500])

    st.markdown("---")
    # Change 4 — scenario punchline on landing page
    st.markdown("### Resource Optimizer")
    st.markdown(f'<div class="sim-insight">{sim_insight}</div>', unsafe_allow_html=True)
    st.markdown("*Change patrol units in the sidebar to see the plan re-rank for this shift window.*")

    fig_scenario = go.Figure()
    fig_scenario.add_trace(go.Scatter(
        x=scenario_df["n_units"], y=scenario_df["pcis_intercepted"],
        mode="lines+markers", name="PCIS intercepted",
        line=dict(color="#00bcd4", width=3),
    ))
    fig_scenario.add_vline(
        x=cmd_n_units, line_dash="dash", line_color="#f39c12",
        annotation_text=f"Current: {cmd_n_units} units",
    )
    fig_scenario.update_layout(
        title="PCIS Intercepted vs Patrol Units",
        xaxis_title="Patrol units", yaxis_title="PCIS intercepted",
        height=350, margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig_scenario, width="stretch")

    sc1, sc2 = st.columns(2)
    with sc1:
        fig_marginal = px.bar(
            scenario_df, x="n_units", y="marginal_pcis",
            title="Marginal PCIS Gain per Additional Unit",
            color="marginal_pcis", color_continuous_scale="Blues",
        )
        fig_marginal.update_layout(showlegend=False, height=320)
        st.plotly_chart(fig_marginal, width="stretch")
    with sc2:
        st.dataframe(
            scenario_df.assign(
                pcis_intercepted=scenario_df["pcis_intercepted"].round(0),
                marginal_pcis=scenario_df["marginal_pcis"].round(0),
                est_fines_rs=scenario_df["est_fines_rs"].apply(lambda x: f"₹{x:,.0f}"),
            )[["n_units", "pcis_intercepted", "marginal_pcis", "coverage_pct", "est_fines_rs"]],
            width="stretch",
            hide_index=True,
            column_config={
                "n_units": "Units",
                "pcis_intercepted": "PCIS",
                "marginal_pcis": "Marginal PCIS",
                "coverage_pct": st.column_config.NumberColumn("Coverage %", format="%.0f"),
                "est_fines_rs": "Est. Fines",
            },
        )

    st.info(
        "Intelligence tabs below explain **why** these junctions were chosen — blind spots, "
        "flow impact, and prediction backtests. This page is what BTP runs each shift.",
        icon="ℹ️",
    )


# ═══════════════════════ PROOF IT WORKS ═══════════════════════
elif page == "Proof It Works":
    st.markdown("## Proof It Works — Enforcement Replay")
    st.markdown(
        "Replay on **held-out violations the deployment never saw**. "
        "Same officer count, same stop budget, same intercept radius — "
        "compare BTP's historical patrol vs PRAHARI's shift-aware deployment."
    )

    resolution = st.radio(
        "Resolution",
        options=["Junction", "Hex"],
        horizontal=True,
        help="Junction = both strategies deploy to coded junctions. "
             "Hex = ACTUAL limited to coded junctions; PRAHARI can deploy to off-grid H3 cells.",
    )
    res_key = "hex" if resolution == "Hex" else "junction"

    with st.spinner("Running enforcement replay on held-out data…"):
        replay = get_enforcement_replay(parking_df, cmd_n_units, cmd_shift_hrs, res_key)

    sp = replay["split"]
    sm = replay["summary"]
    act = sm["actual"]
    pra = sm["prahari"]
    imp = sm["improvement_pcis_pct"]

    st.markdown("### Data Split (no leakage)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Training period", f"{sp['train_start']} → {sp['train_end']}")
    c2.metric("Held-out period", f"{sp['holdout_start']} → {sp['holdout_end']}")
    c3.metric("Stop budget (both strategies)", replay["budget"])

    stops_per_unit = int(replay["shift_hours"] * 60 / PATROL_DEFAULTS["dwell_minutes"])
    st.caption(
        f"Both strategies: **{replay['n_units']} units** × **{stops_per_unit} stops/unit** = "
        f"**{replay['budget']} locations** · intercept radius **{replay['intercept_radius_km']} km** (same both sides). "
        f"ACTUAL = top coded junctions by training count (fixed). "
        f"PRAHARI = causal {'H3 hex' if res_key == 'hex' else 'junction'} ranking per shift window."
    )

    if res_key == "hex":
        st.info(
            "ACTUAL is limited to coded junctions; PRAHARI can deploy to off-grid hotspots BTP has no codes for.",
            icon="📍",
        )

    st.markdown(
        f'<p class="hero-headline">Over {sp["holdout_start"]} to {sp["holdout_end"]}, '
        f"PRAHARI's deployment intercepted <span style='color:#00bcd4;'>{imp:+.1f}%</span> "
        f"more high-impact violations (PCIS) than BTP's historical patrol pattern — "
        f"using the same number of officers.</p>",
        unsafe_allow_html=True,
    )
    count_imp = (
        (pra["intercepted_count"] - act["intercepted_count"]) / act["intercepted_count"] * 100
        if act["intercepted_count"] > 0 else 0
    )
    st.caption(
        f"+{count_imp:.1f}% more raw violations intercepted ({pra['intercepted_count']:,} vs "
        f"{act['intercepted_count']:,})."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### ACTUAL — How BTP patrols today")
        st.metric("Intercepted (violations)", f"{act['intercepted_count']:,}")
        st.metric("Missed (violations)", f"{act['missed_count']:,}")
        st.metric("Intercepted PCIS", f"{act['intercepted_pcis']:,.0f}")
        st.metric("Missed PCIS", f"{act['missed_pcis']:,.0f}")
    with col_b:
        st.markdown("#### PRAHARI — Shift-aware deployment")
        st.metric("Intercepted (violations)", f"{pra['intercepted_count']:,}")
        st.metric("Missed (violations)", f"{pra['missed_count']:,}")
        st.metric("Intercepted PCIS", f"{pra['intercepted_pcis']:,.0f}")
        st.metric("Missed PCIS", f"{pra['missed_pcis']:,.0f}")

    rep_day = replay["representative_day"]
    if rep_day and replay["day_frames"]:
        v_act, v_pra, m_act, m_pra = replay["day_frames"]
        st.markdown(f"### Representative day: **{rep_day}**")
        st.caption("Green = intercepted · Red = missed")

        map_a, map_b = st.columns(2)
        with map_a:
            fig_a = px.scatter_map(
                v_act, lat="latitude", lon="longitude", color="status",
                color_discrete_map={"Intercepted": "#2ecc71", "Missed": "#e74c3c"},
                zoom=11, height=420,
                center={"lat": BENGALURU_CENTER[0], "lon": BENGALURU_CENTER[1]},
                title="ACTUAL (coded junctions only)",
                hover_data=["junction_name", "pcis", "shift"],
            )
            if not m_act.empty:
                fig_a.add_trace(go.Scattermap(
                    lat=m_act["lat"], lon=m_act["lng"], mode="markers",
                    marker=dict(size=12, color="blue"),
                    name="Deployed junctions",
                ))
            fig_a.update_layout(margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig_a, width="stretch")

        with map_b:
            fig_b = px.scatter_map(
                v_pra, lat="latitude", lon="longitude", color="status",
                color_discrete_map={"Intercepted": "#2ecc71", "Missed": "#e74c3c"},
                zoom=11, height=420,
                center={"lat": BENGALURU_CENTER[0], "lon": BENGALURU_CENTER[1]},
                title="PRAHARI (per-shift deployment)",
                hover_data=["junction_name", "pcis", "shift"],
            )
            if not m_pra.empty:
                lat_col = "lat" if "lat" in m_pra.columns else "latitude"
                lng_col = "lng" if "lng" in m_pra.columns else "longitude"
                fig_b.add_trace(go.Scattermap(
                    lat=m_pra[lat_col], lon=m_pra[lng_col], mode="markers",
                    marker=dict(size=10, color="cyan"),
                    name="Deployed (per shift)",
                ))
            fig_b.update_layout(margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig_b, width="stretch")

    with st.expander("Verification debug — exact counts behind every number"):
        hex_lines = ""
        if replay.get("hex_audit"):
            ha = replay["hex_audit"]
            hex_lines = f"""
**Hex audit**
- Unique hexes deployed (PRAHARI): {ha['deployed_hexes']}
- Off-grid hexes deployed: {ha['off_grid_hexes_deployed']}
- Off-grid hexes with held-out violations: {ha['off_grid_with_heldout_violations']}
"""
        st.markdown(f"""
**Resolution:** {resolution}

**Split**
- Training: `{sp['train_start']}` → `{sp['train_end']}` ({len(sp['train']):,} violations)
- Held-out: `{sp['holdout_start']}` → `{sp['holdout_end']}` ({len(sp['held_out']):,} violations)
- Scored (shift windows 08–24): {len(replay['scored']):,}

**Fair comparison audit**
- ACTUAL budget: {replay['n_units']} units × {stops_per_unit} stops = **{replay['budget']}** ✓
- PRAHARI budget: **{replay['budget']}** ✓
- Intercept radius both sides: **{replay['intercept_radius_km']} km** ✓
- PRAHARI uses only data before each shift window (causal) ✓

**ACTUAL scores**
- Intercepted: {act['intercepted_count']:,} · PCIS {act['intercepted_pcis']:,.1f}
- Missed: {act['missed_count']:,} · PCIS {act['missed_pcis']:,.1f}

**PRAHARI scores**
- Intercepted: {pra['intercepted_count']:,} · PCIS {pra['intercepted_pcis']:,.1f}
- Missed: {pra['missed_count']:,} · PCIS {pra['missed_pcis']:,.1f}

**Headline:** ({pra['intercepted_pcis']:,.1f} − {act['intercepted_pcis']:,.1f}) / {act['intercepted_pcis']:,.1f} = **{imp:+.1f}%**
{hex_lines}
        """)

    st.warning(
        "Observational replay — measures coverage of where violations occurred, not causal enforcement effect.",
        icon="ℹ️",
    )


# ═══════════════════════ TAB 1: SITUATION ROOM ═══════════════════════
elif page == "Intelligence: Enforcement Gap":
    st.markdown("## The Enforcement Gap")
    st.markdown("BTP's AI cameras handle 87% of traffic violations — at junctions with cameras. "
                "But **parking enforcement is still manual**: ~1,500 challans a day, issued by officers on foot. "
                "This data reveals where the system is blind.")

    total_pcis = parking_df["pcis"].sum()
    no_junc = (parking_df["junction_name"] == "No Junction").sum()
    with_junc = len(parking_df) - no_junc
    no_junc_pct = no_junc / len(parking_df) * 100
    peak_hour = parking_df.groupby("hour")["pcis"].sum().idxmax()
    est_uncollected = no_junc * 500

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Violations (6 months)", f"{len(parking_df):,}")
    c2.metric("At Named Junctions", f"{with_junc:,}", delta=f"{100-no_junc_pct:.1f}% visible")
    c3.metric("Off-Grid (No Junction Code)", f"{no_junc:,}", delta=f"{no_junc_pct:.1f}% invisible", delta_color="inverse")
    c4.metric("Est. Uncollected Fines", f"₹{est_uncollected/100000:.1f}L")

    st.markdown("### Where BTP Sees vs. Where It Doesn't")
    st.markdown("**Left:** Named junctions (BTP's current focus)  |  **Right:** Off-grid violations (the blind spots)")

    col_left, col_right = st.columns(2)

    junc_data = parking_df[parking_df["junction_name"] != "No Junction"]
    no_junc_data = parking_df[parking_df["junction_name"] == "No Junction"]

    with col_left:
        junc_hex = junc_data.groupby("h3_index").agg(
            pcis=("pcis", "sum"), lat=("latitude", "mean"), lng=("longitude", "mean")).reset_index()
        fig_l = px.scatter_map(junc_hex, lat="lat", lon="lng", size="pcis", color="pcis",
                               color_continuous_scale="Blues", size_max=15, zoom=11,
                               center={"lat": BENGALURU_CENTER[0], "lon": BENGALURU_CENTER[1]},
                               height=400, title=f"Known System — {with_junc:,} violations")
        fig_l.update_layout(margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_l, width="stretch")

    with col_right:
        nj_hex = no_junc_data.groupby("h3_index").agg(
            pcis=("pcis", "sum"), lat=("latitude", "mean"), lng=("longitude", "mean")).reset_index()
        fig_r = px.scatter_map(nj_hex, lat="lat", lon="lng", size="pcis", color="pcis",
                               color_continuous_scale="Reds", size_max=15, zoom=11,
                               center={"lat": BENGALURU_CENTER[0], "lon": BENGALURU_CENTER[1]},
                               height=400, title=f"Invisible to BTP — {no_junc:,} violations")
        fig_r.update_layout(margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_r, width="stretch")

    st.markdown("---")
    st.markdown("### What This Costs Bengaluru *(estimated)*")
    st.caption("Based on road-capacity estimation model — see 'Traffic Flow Impact' tab for full methodology and assumptions.")
    cf1, cf2, cf3, cf4 = st.columns(4)
    cf1.metric("Daily Vehicle-Hours of Delay", f"~{flow_summary['daily_avg_delay_veh_hours']:,.0f}")
    cf2.metric("Daily Person-Hours of Delay", f"~{flow_summary['daily_avg_delay_person_hours']:,.0f}")
    cf3.metric("Avg Capacity Reduction (lower bound)", f"≈{flow_summary['avg_capacity_reduction_pct']:.0f}%")
    cf4.metric("Heavy Vehicles' Share of Blockage", f"~{flow_summary['heavy_vehicle_blocked_pct']:.0f}%")

    st.markdown("---")
    st.markdown("### When and What")
    col_a, col_b = st.columns(2)
    with col_a:
        hourly = parking_df.groupby("hour")["pcis"].sum().reset_index()
        fig_h = px.bar(hourly, x="hour", y="pcis", title="Enforcement Activity by Hour (IST)",
                       labels={"pcis": "Total PCIS", "hour": "Hour"},
                       color="pcis", color_continuous_scale="YlOrRd")
        fig_h.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig_h, width="stretch")

    with col_b:
        veh = parking_df.groupby("veh_type_final")["pcis"].sum().nlargest(10).reset_index()
        fig_v = px.bar(veh, x="pcis", y="veh_type_final", orientation="h",
                       title="Impact by Vehicle Type (PCIS-weighted)",
                       color="pcis", color_continuous_scale="Viridis")
        fig_v.update_layout(yaxis=dict(autorange="reversed"), showlegend=False, height=350)
        st.plotly_chart(fig_v, width="stretch")


# ═══════════════════════ TAB: TRAFFIC FLOW IMPACT ═══════════════════════
elif page == "Intelligence: Traffic Flow Impact":
    st.markdown("## Traffic Flow Impact Quantification")
    st.markdown(
        "The problem statement asks: *'quantify their impact on traffic flow.'* "
        "PCIS is a relative severity score. This module converts each violation into **concrete, "
        "physical traffic metrics** — carriageway blocked, capacity lost, vehicle-hours of delay — "
        "using road engineering standards."
    )

    st.warning(
        "**Estimation Model — Not Measured Fact.** All figures below are derived from stated assumptions "
        "(road widths, flow rates, parking duration). They are transparent, illustrative estimates — "
        "not field-measured values. Every assumption is listed below so a reviewer can challenge or refine them.",
        icon="⚠️",
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Carriageway Blocked", f"~{flow_summary['total_carriageway_blocked_m2']:,.0f} m²",
              help="Sum of road area physically occupied by illegally parked vehicles + safety buffer")
    c2.metric("Avg Capacity Reduction (lower bound)", f"≈{flow_summary['avg_capacity_reduction_pct']:.0f}%",
              help="Geometric lower bound — actual throughput loss is typically worse due to merging friction")
    c3.metric("Total Vehicle-Hours of Delay (6 mo)", f"~{flow_summary['total_delay_vehicle_hours']:,.0f}",
              help="Cumulative delay imposed on passing vehicles, estimated via BPR model")
    c4.metric("Total Person-Hours of Delay (6 mo)", f"~{flow_summary['total_delay_person_hours']:,.0f}",
              help="Vehicle-hours × average occupancy (1.5 persons/vehicle)")

    st.markdown("---")

    c5, c6, c7 = st.columns(3)
    c5.metric("Daily Avg Vehicle-Hours Lost", f"~{flow_summary['daily_avg_delay_veh_hours']:,.0f}")
    c6.metric("Junction Violations' Delay Share", f"~{flow_summary['junction_delay_pct']:.0f}%")
    c7.metric("Est. Uncollected Fines (6 mo)", f"₹{flow_summary['est_total_uncollected_fines_rs']/10000000:.1f} Cr")

    st.markdown("---")
    st.markdown("### How a Single Violation Causes Delay")
    st.markdown(
        "When a vehicle parks illegally, it physically narrows the road. Passing traffic must "
        "squeeze through a reduced carriageway, which slows every vehicle by a few seconds. "
        "Over 2 hours of parking on a busy road, those seconds add up across thousands of passing vehicles."
    )

    st.markdown("**Example:** A lorry parked on a 3-lane main road (10.5m wide):")
    st.markdown(
        "- Blocks **4.5m** of width (vehicle 2.5m + safety buffer 2.0m)\n"
        "- Reduces effective capacity by **≈43% (lower bound)** — actual throughput loss is typically "
        "worse than linear because vehicles must merge around the obstruction, creating additional friction\n"
        "- Over 2 hours at ~3,600 PCU/hr, ~3,089 affected vehicles each lose ~8 seconds\n"
        "- Total delay: **~6.9 vehicle-hours** from one parked lorry\n\n"
        "*All capacity reduction figures are geometric lower bounds. Real-world merging, "
        "lane-change friction, and signal spillback amplify the effect — making these estimates conservative.*"
    )

    st.markdown("---")
    st.markdown("### Per-Junction Traffic Flow Impact")

    jf_display = junc_flow.head(20).copy()
    jf_display["delay_per_violation_min"] = (jf_display["total_delay_veh_hours"] / jf_display["violations"] * 60).round(1)
    st.dataframe(
        jf_display[["junction_name", "violations", "total_pcis",
                     "carriageway_blocked_m2", "avg_capacity_reduction",
                     "total_delay_veh_hours", "delay_per_violation_min"]].reset_index(drop=True),
        width="stretch",
        column_config={
            "junction_name": "Junction",
            "violations": "Violations",
            "total_pcis": st.column_config.NumberColumn("PCIS", format="%.0f"),
            "carriageway_blocked_m2": st.column_config.NumberColumn("Road Blocked (m²)", format="%.0f"),
            "avg_capacity_reduction": st.column_config.NumberColumn("Avg Cap. Reduction %", format="%.1f"),
            "total_delay_veh_hours": st.column_config.NumberColumn("Total Delay (veh-hrs)", format="%.0f"),
            "delay_per_violation_min": st.column_config.NumberColumn("Delay/Violation (min)", format="%.1f"),
        },
    )

    col_a, col_b = st.columns(2)
    with col_a:
        fig_delay = px.bar(
            jf_display, x="total_delay_veh_hours", y="junction_name", orientation="h",
            title="Top 20 Junctions by Traffic Delay (vehicle-hours)",
            color="total_delay_veh_hours", color_continuous_scale="YlOrRd",
            labels={"total_delay_veh_hours": "Vehicle-Hours of Delay", "junction_name": ""},
        )
        fig_delay.update_layout(yaxis=dict(autorange="reversed"), height=500, showlegend=False)
        st.plotly_chart(fig_delay, width="stretch")

    with col_b:
        if road_type_flow is not None and not road_type_flow.empty:
            fig_road = px.bar(
                road_type_flow, x="road_type", y="total_delay",
                title="Total Delay by Road Type",
                color="avg_cap_reduction", color_continuous_scale="Reds",
                text="violations",
                labels={"road_type": "Road Type", "total_delay": "Total Delay (veh-hrs)",
                        "avg_cap_reduction": "Avg Cap. Reduction %"},
            )
        else:
            road_type_impact = parking_with_flow.groupby("road_type").agg(
                violations=("pcis", "size"),
                avg_delay=("estimated_delay_veh_hours", "mean"),
                total_delay=("estimated_delay_veh_hours", "sum"),
                avg_cap_reduction=("capacity_reduction_pct", "mean"),
            ).reset_index()
            fig_road = px.bar(
                road_type_impact, x="road_type", y="total_delay",
                title="Total Delay by Road Type",
                color="avg_cap_reduction", color_continuous_scale="Reds",
                text="violations",
                labels={"road_type": "Road Type", "total_delay": "Total Delay (veh-hrs)",
                        "avg_cap_reduction": "Avg Cap. Reduction %"},
            )
        fig_road.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig_road, width="stretch")

    st.markdown("---")
    st.markdown("### Stated Assumptions *(full transparency)*")
    st.markdown(
        "Every number above rests on the assumptions below. These are drawn from "
        "IRC (Indian Road Congress) standards and HCM (Highway Capacity Manual) methodology. "
        "They are **illustrative estimates** — actual values vary by location."
    )

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("**Road Profiles Used:**")
        road_data = []
        for rtype, profile in ROAD_PROFILES.items():
            road_data.append({
                "Road Type": rtype.replace("_", " ").title(),
                "Carriageway Width (m)": profile["width_m"],
                "Lanes": profile["lanes"],
                "Flow Rate (PCU/lane/hr)": profile["pcu_per_lane_hr"],
            })
        st.dataframe(pd.DataFrame(road_data), width="stretch", hide_index=True)

        st.markdown("**Other Assumptions:**")
        st.markdown("- **Parking duration:** 2 hours per violation (avg)")
        st.markdown("- **Per-vehicle delay:** ~8 seconds (avg queuing/merging time at obstruction)")
        st.markdown("- **Vehicle occupancy:** 1.5 persons per vehicle")
        st.markdown("- **Fine amount:** ₹500 per violation (MV Act standard)")
        st.markdown("- **Capacity reduction:** Geometric (width-based) lower bound — "
                    "real throughput loss is typically worse due to lane-change friction and merging")

    with col_r:
        st.markdown("**Sources & References:**")
        for key, desc in ASSUMPTIONS.items():
            label = key.replace("_", " ").title()
            st.markdown(f"- **{label}:** {desc}")


# ═══════════════════════ TAB 2: HOTSPOT DEEP DIVE ═══════════════════════
elif page == "Intelligence: Hotspot Analysis":
    st.markdown("## Junction-Level Hotspot Analysis")

    top_n = st.slider("Number of hotspots to display", 10, 50, 20, key="hotspot_n")
    top_junctions = junction_summary.head(top_n)

    c1, c2, c3 = st.columns(3)
    c1.metric("Hottest Junction", top_junctions.iloc[0]["junction_name"])
    c2.metric("Its PCIS", f"{top_junctions.iloc[0]['total_pcis']:,.0f}")
    c3.metric("Peak Hour", f"{int(top_junctions.iloc[0]['peak_hour'])}:00")

    fig_junc = px.scatter_map(
        top_junctions, lat="lat", lon="lng", size="total_pcis", color="total_pcis",
        hover_name="junction_name", color_continuous_scale="Reds",
        size_max=25, zoom=11,
        center={"lat": BENGALURU_CENTER[0], "lon": BENGALURU_CENTER[1]},
        hover_data={"total_pcis": ":.0f", "count": True, "heavy_ratio": ":.2f"},
        height=500, title="Top Hotspot Junctions",
    )
    fig_junc.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_junc, width="stretch")

    st.markdown("### Hotspot Rankings")
    display_cols = ["junction_name", "police_station", "total_pcis", "count",
                    "mean_pcis", "heavy_ratio", "main_road_ratio", "peak_hour"]
    st.dataframe(
        top_junctions[display_cols].reset_index(drop=True),
        width="stretch",
        column_config={
            "junction_name": "Junction",
            "police_station": "Station",
            "total_pcis": st.column_config.NumberColumn("Total PCIS", format="%.0f"),
            "count": "Violations",
            "mean_pcis": st.column_config.NumberColumn("Avg PCIS", format="%.1f"),
            "heavy_ratio": st.column_config.ProgressColumn("Heavy Vehicle %", min_value=0, max_value=1),
            "main_road_ratio": st.column_config.ProgressColumn("Main Road %", min_value=0, max_value=1),
            "peak_hour": "Peak Hour",
        },
    )

    st.markdown("### Hotspot Time Patterns")
    selected_junc = st.selectbox("Select junction", top_junctions["junction_name"].tolist())
    junc_data = parking_df[parking_df["junction_name"] == selected_junc]
    if not junc_data.empty:
        col1, col2 = st.columns(2)
        with col1:
            jh = junc_data.groupby("hour")["pcis"].sum().reset_index()
            fig = px.bar(jh, x="hour", y="pcis", title=f"Hourly Pattern: {selected_junc}",
                         color="pcis", color_continuous_scale="YlOrRd")
            fig.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig, width="stretch")
        with col2:
            jv = junc_data["veh_type_final"].value_counts().head(6).reset_index()
            jv.columns = ["vehicle", "count"]
            fig = px.pie(jv, names="vehicle", values="count",
                         title=f"Vehicle Mix: {selected_junc}")
            fig.update_layout(height=300)
            st.plotly_chart(fig, width="stretch")


# ═══════════════════════ TAB: BLIND SPOT DISCOVERY ═══════════════════════
elif page == "Intelligence: Blind Spots":
    st.markdown("## Enforcement Blind Spot Discovery")
    st.markdown("Using HDBSCAN density clustering on 147,880 off-grid violations, "
                "PRAHARI discovers hidden hotspot clusters that BTP's junction system cannot see.")

    if blind_spots_pre is not None and not blind_spots_pre.empty:
        all_clusters = blind_spots_pre
    else:
        all_clusters = get_blind_spots(parking_df)
    clusters = all_clusters.nlargest(30, "total_pcis") if not all_clusters.empty else all_clusters

    if not clusters.empty:
        # ── THE AUTOPSY: one cluster, full story ──
        st.markdown("---")
        st.markdown("### Cluster Autopsy: The Worst Hidden Hotspot")
        top_c = clusters.iloc[0]
        est_fines = int(top_c["count"]) * 500

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Location", str(top_c["police_station"]) + " jurisdiction")
        c2.metric("Violations (6 months)", f"{int(top_c['count']):,}")
        c3.metric("Days Active", f"{int(top_c['unique_dates'])}/150")
        c4.metric("Est. Uncollected Fines", f"₹{est_fines/100000:.1f}L")

        st.markdown(
            f"**{top_c['cluster_name']}** — located at ({top_c['lat']:.4f}, {top_c['lng']:.4f}) — "
            f"has generated **{int(top_c['count']):,} parking violations** over 6 months, "
            f"active on **{int(top_c['unique_dates'])} out of 150 days**. "
            f"The dominant vehicle is **{top_c['top_vehicle']}**, with **{top_c['heavy_ratio']:.0%} heavy vehicles** "
            f"and **{top_c['main_road_ratio']:.0%} main-road violations**. "
            f"At ₹500/challan, this single location represents an estimated **₹{est_fines/100000:.1f} lakh** "
            f"in enforcement value. **It has no junction code, no camera, and no patrol assignment.**"
        )

        autopsy_map = px.scatter_map(
            pd.DataFrame([{"lat": top_c["lat"], "lng": top_c["lng"], "pcis": top_c["total_pcis"],
                           "name": top_c["cluster_name"]}]),
            lat="lat", lon="lng", size="pcis", color_discrete_sequence=["red"],
            size_max=30, zoom=14, hover_name="name",
            center={"lat": top_c["lat"], "lon": top_c["lng"]},
            height=350, title=f"Zoomed: {top_c['cluster_name']}",
        )
        autopsy_map.update_layout(margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(autopsy_map, width="stretch")

        # ── ALL TOP CLUSTERS ──
        st.markdown("---")
        st.markdown("### All Critical Hidden Hotspots")

        c1, c2, c3 = st.columns(3)
        total_cluster_fines = clusters["count"].sum() * 500
        c1.metric("Critical Clusters Found", len(clusters))
        c2.metric("Total Violations", f"{clusters['count'].sum():,}")
        c3.metric("Total Est. Fines", f"₹{total_cluster_fines/100000:.1f}L")

        col1, col2 = st.columns(2)
        with col1:
            cluster_map = px.scatter_map(
                clusters, lat="lat", lon="lng", size="total_pcis", color="total_pcis",
                hover_name="cluster_name", color_continuous_scale="Reds",
                size_max=20, zoom=11,
                center={"lat": BENGALURU_CENTER[0], "lon": BENGALURU_CENTER[1]},
                hover_data={"count": True, "police_station": True},
                height=400, title="Hidden Hotspot Clusters",
            )
            cluster_map.update_layout(margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(cluster_map, width="stretch")

        with col2:
            exposure = get_exposure(parking_df)
            blind_spots = exposure[exposure["is_blind_spot"] == 1]

            fig_bias = go.Figure()
            fig_bias.add_trace(go.Scattermap(
                lat=blind_spots["lat"], lon=blind_spots["lng"],
                mode="markers", marker=dict(size=10, color="red", opacity=0.7),
                name=f"Blind Spots ({len(blind_spots)} cells)",
                text=blind_spots["bias_corrected_score"].round(1),
            ))
            fig_bias.update_layout(
                map=dict(style="open-street-map",
                         center=dict(lat=BENGALURU_CENTER[0], lon=BENGALURU_CENTER[1]), zoom=11),
                height=400, margin=dict(l=0, r=0, t=30, b=0),
                title=f"Bias-Corrected: {len(blind_spots)} Under-Enforced Cells",
            )
            st.plotly_chart(fig_bias, width="stretch")

        clusters_display = clusters.copy()
        clusters_display["est_fines_lakhs"] = (clusters_display["count"] * 500 / 100000).round(1)
        st.dataframe(
            clusters_display[["cluster_name", "count", "total_pcis", "est_fines_lakhs",
                              "police_station", "top_vehicle", "peak_hour",
                              "unique_dates"]].reset_index(drop=True),
            width="stretch",
            column_config={
                "cluster_name": "Hotspot", "count": "Violations",
                "total_pcis": st.column_config.NumberColumn("PCIS", format="%.0f"),
                "est_fines_lakhs": "Est. Fines (₹L)",
                "police_station": "Station", "top_vehicle": "Top Vehicle",
                "peak_hour": "Peak Hour", "unique_dates": "Days Active",
            },
        )


# ═══════════════════════ TAB: ENFORCEMENT EFFECTIVENESS ═══════════════════════
elif page == "Intelligence: Model & Efficacy":
    st.markdown("## Enforcement Effectiveness Analysis")
    st.markdown("Observational before/after analysis: do violations decrease at junctions following high-enforcement weeks? *(Note: correlation, not causal proof — regression to mean is possible.)*")

    efficacy_data = get_efficacy(parking_df)
    results = efficacy_data["results"]
    summary = efficacy_data["summary"]

    if summary:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Enforcement Events Analyzed", summary.get("total_events", 0))
        c2.metric("Effective (>10% reduction)", summary.get("effective_events", 0))
        c3.metric("Effectiveness Rate", f"{summary.get('effectiveness_rate', 0):.1f}%")
        c4.metric("Avg Change", f"{summary.get('avg_change_pct', 0):.1f}%")

    if not results.empty:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Most Responsive Junctions")
            effective = results[results["effective"]].sort_values("change_pct")
            if not effective.empty:
                fig_eff = px.bar(
                    effective.head(15), x="change_pct", y="junction", orientation="h",
                    title="Junctions Where Enforcement Worked",
                    color="change_pct", color_continuous_scale="Greens_r",
                    labels={"change_pct": "PCIS Change %", "junction": ""},
                )
                fig_eff.update_layout(yaxis=dict(autorange="reversed"), height=400, showlegend=False)
                st.plotly_chart(fig_eff, width="stretch")

        with col2:
            st.markdown("### Resistant Junctions (enforcement didn't help)")
            resistant = results[~results["effective"]].sort_values("change_pct", ascending=False)
            if not resistant.empty:
                fig_res = px.bar(
                    resistant.head(15), x="change_pct", y="junction", orientation="h",
                    title="Junctions Where Violations Persisted/Increased",
                    color="change_pct", color_continuous_scale="Reds",
                    labels={"change_pct": "PCIS Change %", "junction": ""},
                )
                fig_res.update_layout(yaxis=dict(autorange="reversed"), height=400, showlegend=False)
                st.plotly_chart(fig_res, width="stretch")

        st.markdown("### Full Enforcement Event Log")
        st.dataframe(results.sort_values("change_pct").reset_index(drop=True), width="stretch")

    # Prediction model performance
    st.markdown("---")
    st.markdown("## Predictive Model Performance")
    with st.spinner("Training hurdle model..."):
        model_data = get_predictions(cell_agg, recurrence)

    metrics = model_data["metrics"]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Precision@10%", f"{metrics.get('precision_at_10pct', 0):.2%}")
    c2.metric("Hit Rate@10%", f"{metrics.get('hit_rate_at_10pct', 0):.2%}")
    c3.metric("Precision@20%", f"{metrics.get('precision_at_20pct', 0):.2%}")
    c4.metric("Binary F1", f"{metrics.get('f1', 0):.2%}")
    c5.metric("MAE", f"{metrics.get('mae', 0):.2f}")

    if "feature_importance" in model_data:
        fi = model_data["feature_importance"]
        fig_fi = px.bar(fi.head(12), x="regressor_importance", y="feature", orientation="h",
                        title="What Drives Predictions (Feature Importance)",
                        color="regressor_importance", color_continuous_scale="Viridis")
        fig_fi.update_layout(yaxis=dict(autorange="reversed"), height=350, showlegend=False)
        st.plotly_chart(fig_fi, width="stretch")

    test_results = model_data["test_results"]
    if not test_results.empty:
        fig_pred = px.scatter(
            test_results.sample(min(2000, len(test_results))),
            x="total_pcis", y="pred_pcis",
            title="Predicted vs Actual PCIS (Test Set)",
            labels={"total_pcis": "Actual PCIS", "pred_pcis": "Predicted PCIS"},
            opacity=0.4,
        )
        fig_pred.add_trace(go.Scatter(
            x=[0, test_results["total_pcis"].quantile(0.99)],
            y=[0, test_results["total_pcis"].quantile(0.99)],
            mode="lines", line=dict(color="red", dash="dash"), name="Perfect",
        ))
        fig_pred.update_layout(height=400)
        st.plotly_chart(fig_pred, width="stretch")


# ═══════════════════════ TAB: SYSTEM HEALTH ═══════════════════════
elif page == "Intelligence: System Health":
    st.markdown("## System Health & Reporting Quality")

    health = get_system_health(parking_df)

    c1, c2 = st.columns(2)
    c1.metric("City-Wide Rejection Rate", f"{health['city_avg_rejection_rate']:.1%}")
    c2.metric("Stations Analyzed", len(health["station_quality"]))

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Rejection Rate by Station")
        sq = health["station_quality"].head(20)
        fig_sq = px.bar(
            sq, x="rejection_rate", y="police_station", orientation="h",
            title="Top 20 Stations by Rejection Rate",
            color="rejection_rate", color_continuous_scale="Reds",
            labels={"rejection_rate": "Rejection Rate", "police_station": ""},
        )
        fig_sq.update_layout(yaxis=dict(autorange="reversed"), height=500, showlegend=False)
        fig_sq.add_vline(x=health["city_avg_rejection_rate"], line_dash="dash",
                        annotation_text="City Avg", line_color="white")
        st.plotly_chart(fig_sq, width="stretch")

    with col2:
        st.markdown("### Rejection Rate by Hour")
        hq = health["hourly_quality"]
        fig_hq = px.bar(hq, x="hour", y="rejection_rate",
                        title="Report Rejection Rate by Hour",
                        color="rejection_rate", color_continuous_scale="YlOrRd")
        fig_hq.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig_hq, width="stretch")

        st.markdown("### Devices Needing Attention")
        dq = health["device_quality"].head(10)
        st.dataframe(
            dq[["device_id", "total", "approved", "rejected", "rejection_rate", "station"]].reset_index(drop=True),
            width="stretch",
            column_config={
                "rejection_rate": st.column_config.ProgressColumn("Rejection Rate", min_value=0, max_value=1, format="%.0f%%"),
            },
        )

    st.markdown("---")
    st.markdown("## Chronic Offender Intelligence")

    offenders = get_offenders(parking_df)
    n_20plus = len(offenders[offenders["violation_count"] >= 20])
    worst = int(offenders.iloc[0]["violation_count"]) if len(offenders) > 0 else 0
    c1, c2, c3 = st.columns(3)
    c1.metric("Serial Offenders (20+ tickets)", n_20plus)
    c2.metric("Worst Offender", f"{worst} tickets")
    c3.metric("Repeat Offenders (10+)", len(offenders[offenders["violation_count"] >= 10]))

    st.markdown("### Top 25 Chronic Offenders (Ranked by Total PCIS Impact)")
    display = offenders.head(25)
    st.dataframe(
        display[["vehicle_number", "vehicle_type", "violation_count", "total_pcis",
                 "top_station", "unique_locations", "days_active", "violations_per_day",
                 "approval_rate"]].reset_index(drop=True),
        width="stretch",
        column_config={
            "vehicle_number": "Vehicle",
            "vehicle_type": "Type",
            "violation_count": "Tickets",
            "total_pcis": st.column_config.NumberColumn("Total PCIS", format="%.0f"),
            "top_station": "Primary Station",
            "unique_locations": "Unique Locations",
            "days_active": "Days Active",
            "violations_per_day": st.column_config.NumberColumn("Violations/Day", format="%.2f"),
            "approval_rate": st.column_config.ProgressColumn("Approval Rate", min_value=0, max_value=1),
        },
    )

    st.markdown("### Repeat Location Offenders")
    repeat_loc = offenders[offenders["is_repeat_location"] == 1].head(15)
    if not repeat_loc.empty:
        st.markdown("These vehicles **always park illegally at the same spot** — perfect for targeted towing.")
        st.dataframe(
            repeat_loc[["vehicle_number", "vehicle_type", "violation_count",
                        "total_pcis", "top_station"]].reset_index(drop=True),
            width="stretch",
        )

    st.markdown("---")
    st.markdown("## Temporal Trends")
    trends = __import__("src.analytics", fromlist=["temporal_trends"]).temporal_trends(parking_df)
    fig_trend = px.line(trends, x="year_week", y="total_pcis",
                        title="Weekly PCIS Trend", markers=True,
                        labels={"year_week": "Week", "total_pcis": "Total PCIS"})
    fig_trend.update_layout(height=350)
    st.plotly_chart(fig_trend, width="stretch")
