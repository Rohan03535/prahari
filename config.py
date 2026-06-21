from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"

# Bundled processed data (for deployment — no 105MB raw CSV needed)
PARKING_PARQUET = DATA_DIR / "parking.parquet"
JUNCTION_PARQUET = DATA_DIR / "junction_summary.parquet"
CELL_AGG_PARQUET = DATA_DIR / "cell_agg.parquet"
RECURRENCE_PARQUET = DATA_DIR / "recurrence.parquet"
FLOW_SUMMARY_JSON = DATA_DIR / "flow_summary.json"
JUNC_FLOW_PARQUET = DATA_DIR / "junc_flow.parquet"
ROAD_TYPE_FLOW_PARQUET = DATA_DIR / "road_type_flow.parquet"
BLIND_SPOTS_PARQUET = DATA_DIR / "blind_spots.parquet"
PARKING_SLIM_PARQUET = DATA_DIR / "parking_slim.parquet"
JUNCTION_INTEL_PARQUET = DATA_DIR / "junction_intel.parquet"

# Columns kept in cloud bundle (~80MB RAM vs ~500MB for full parking export)
PARKING_SLIM_COLUMNS = [
    "latitude", "longitude", "junction_name", "police_station", "pcis",
    "hour", "date", "h3_index", "veh_type_final", "vehicle_weight",
    "has_junction", "is_main_road_viol", "device_id", "created_by_id",
    "vehicle_number", "validation_status", "violation_type", "year_week",
    "created_datetime",
]

# Optional raw CSV for local re-processing only
PARKING_CSV = RAW_DATA_DIR / "jan_to_may_police_violation_anonymized.csv"

H3_RESOLUTION = 9          # ~174 m edge, ~0.1 km² per hex
H3_RESOLUTION_COARSE = 7   # ~1.2 km edge, overview zoom

TIME_WINDOW_HOURS = 4
TIME_WINDOW_LABELS = {
    0: "00:00–04:00", 1: "04:00–08:00", 2: "08:00–12:00",
    3: "12:00–16:00", 4: "16:00–20:00", 5: "20:00–24:00",
}

VEHICLE_WEIGHTS = {
    "HGV": 5.0, "LORRY/GOODS VEHICLE": 5.0, "TANKER": 5.0,
    "BUS (BMTC/KSRTC)": 4.0, "PRIVATE BUS": 4.0, "FACTORY BUS": 4.0, "TOURIST BUS": 4.0,
    "TEMPO": 3.0, "LGV": 3.0, "MINI LORRY": 3.0, "TRACTOR": 3.5,
    "CAR": 2.0, "MAXI-CAB": 2.0, "JEEP": 2.0, "VAN": 2.0, "SCHOOL VEHICLE": 2.0,
    "PASSENGER AUTO": 1.5, "GOODS AUTO": 1.5, "OTHERS": 1.5,
    "MOTOR CYCLE": 1.0, "SCOOTER": 1.0, "MOPED": 1.0,
}
DEFAULT_VEHICLE_WEIGHT = 1.5

VIOLATION_SEVERITY = {
    "PARKING IN A MAIN ROAD": 3.0,
    "DOUBLE PARKING": 3.0,
    "AGAINST ONE WAY/NO ENTRY": 3.0,
    "PARKING NEAR ROAD CROSSING": 2.5,
    "PARKING NEAR TRAFFIC LIGHT OR ZEBRA CROSS": 2.5,
    "PARKING NEAR BUSTOP/SCHOOL/HOSPITAL ETC": 2.5,
    "STOPING ON WHITE/STOP LINE": 2.5,
    "H T V PROHIBITED": 2.0,
    "PARKING OPPOSITE TO ANOTHER PARKED VEHICLE": 2.0,
    "WRONG PARKING": 2.0,
    "NO PARKING": 2.0,
    "PARKING ON FOOTPATH": 1.5,
    "PARKING OTHER THAN BUS STOP": 1.5,
}
DEFAULT_VIOLATION_SEVERITY = 1.0

PARKING_VIOLATION_KEYWORDS = {
    "WRONG PARKING", "NO PARKING", "PARKING IN A MAIN ROAD",
    "DOUBLE PARKING", "PARKING NEAR ROAD CROSSING",
    "PARKING NEAR TRAFFIC LIGHT OR ZEBRA CROSS",
    "PARKING NEAR BUSTOP/SCHOOL/HOSPITAL ETC",
    "PARKING ON FOOTPATH", "PARKING OPPOSITE TO ANOTHER PARKED VEHICLE",
    "PARKING OTHER THAN BUS STOP", "H T V PROHIBITED",
    "AGAINST ONE WAY/NO ENTRY", "STOPING ON WHITE/STOP LINE",
}

HDBSCAN_MIN_CLUSTER_SIZE = 20
HDBSCAN_MIN_SAMPLES = 10

TRAIN_CUTOFF = "2024-02-15"

BENGALURU_CENTER = (12.9716, 77.5946)
MAP_ZOOM = 12

PATROL_DEFAULTS = {
    "n_units": 5,
    "shift_hours": 4,
    "dwell_minutes": 30,
    "avg_speed_kmh": 15,
}
DEFAULT_TOP_HOTSPOTS = 25
DEFAULT_SHIFT_LABEL = "Morning (08–12)"
DEFAULT_PATROL_PLAN_JSON = DATA_DIR / "default_patrol_plan.json"
