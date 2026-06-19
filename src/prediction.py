import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error
)
import lightgbm as lgb
import sys, warnings
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from config import TRAIN_CUTOFF

warnings.filterwarnings("ignore", category=UserWarning)


def engineer_features(cell_agg: pd.DataFrame, recurrence: pd.DataFrame) -> pd.DataFrame:
    """
    Build the ML-ready feature matrix from cell-level aggregations.
    Expands the grid to include zero-violation time slots for active cells,
    enabling proper hurdle model training.
    """
    df = cell_agg.copy()
    df["date"] = pd.to_datetime(df["date"])

    top_cells = df.groupby("h3_cell")["total_pcis"].sum().nlargest(500).index.tolist()
    df = df[df["h3_cell"].isin(top_cells)]

    all_dates = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    all_tw = range(6)
    full_index = pd.MultiIndex.from_product(
        [top_cells, all_dates, list(all_tw)],
        names=["h3_cell", "date", "time_window"],
    )
    full_df = pd.DataFrame(index=full_index).reset_index()
    df = full_df.merge(df, on=["h3_cell", "date", "time_window"], how="left")
    for col in ["total_pcis", "count", "mean_pcis", "heavy_ratio", "main_road_ratio"]:
        df[col] = df[col].fillna(0)
    df["unique_devices"] = df["unique_devices"].fillna(0)
    df["unique_officers"] = df["unique_officers"].fillna(0)
    df = df.sort_values(["h3_cell", "date", "time_window"])

    recurrence_map = recurrence.set_index(["h3_cell", "time_window"])["recurrence_factor"].to_dict()
    df["recurrence_factor"] = df.apply(
        lambda r: recurrence_map.get((r["h3_cell"], r["time_window"]), 1.0), axis=1
    )

    df["dow"] = df["date"].dt.dayofweek
    df["is_weekend"] = df["dow"].isin([5, 6]).astype(int)
    df["month"] = df["date"].dt.month

    df["hour_proxy"] = df["time_window"] * 4 + 2
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_proxy"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_proxy"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)

    df = df.sort_values(["h3_cell", "date", "time_window"])
    grouped = df.groupby("h3_cell")["total_pcis"]
    df["lag_1"] = grouped.shift(1).fillna(0)
    df["lag_6"] = grouped.shift(6).fillna(0)     # same time_window yesterday
    df["lag_42"] = grouped.shift(42).fillna(0)   # same time_window last week
    df["rolling_6_mean"] = grouped.transform(lambda x: x.rolling(6, min_periods=1).mean())
    df["rolling_42_mean"] = grouped.transform(lambda x: x.rolling(42, min_periods=1).mean())

    df["has_violation"] = (df["total_pcis"] > 0).astype(int)

    return df


def train_hurdle_model(features_df: pd.DataFrame):
    """
    Two-stage hurdle model:
      Stage 1 (binary): will this cell have any violation?
      Stage 2 (regression): if yes, predict PCIS intensity.
    Returns both models and feature columns.
    """
    df = features_df.copy()
    cutoff = pd.Timestamp(TRAIN_CUTOFF)
    train = df[df["date"] < cutoff]
    test = df[df["date"] >= cutoff]

    feature_cols = [
        "time_window", "dow", "is_weekend", "month",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "lag_1", "lag_6", "lag_42",
        "rolling_6_mean", "rolling_42_mean",
        "recurrence_factor",
    ]

    X_train = train[feature_cols].fillna(0)
    y_train_binary = train["has_violation"]
    y_train_intensity = train.loc[train["has_violation"] == 1, "total_pcis"]

    X_test = test[feature_cols].fillna(0)
    y_test = test["total_pcis"]
    y_test_binary = test["has_violation"]

    clf = lgb.LGBMClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        num_leaves=31, min_child_samples=20,
        subsample=0.8, colsample_bytree=0.8,
        verbose=-1, random_state=42,
    )
    clf.fit(X_train, y_train_binary)

    X_train_pos = train.loc[train["has_violation"] == 1, feature_cols].fillna(0)
    reg = lgb.LGBMRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        num_leaves=31, min_child_samples=20,
        subsample=0.8, colsample_bytree=0.8,
        verbose=-1, random_state=42,
    )
    reg.fit(X_train_pos, y_train_intensity)

    pred_binary = clf.predict(X_test)
    pred_proba = clf.predict_proba(X_test)[:, 1]
    pred_intensity = np.zeros(len(X_test))
    pos_mask = pred_binary == 1
    if pos_mask.any():
        pred_intensity[pos_mask] = reg.predict(X_test[pos_mask])
    pred_intensity = np.clip(pred_intensity, 0, None)

    test = test.copy()
    test["pred_pcis"] = pred_intensity
    test["pred_proba"] = pred_proba

    metrics = evaluate_model(test, y_test, pred_intensity, y_test_binary, pred_binary)

    importance = pd.DataFrame({
        "feature": feature_cols,
        "classifier_importance": clf.feature_importances_,
        "regressor_importance": reg.feature_importances_,
    }).sort_values("regressor_importance", ascending=False)

    return {
        "classifier": clf,
        "regressor": reg,
        "feature_cols": feature_cols,
        "test_results": test,
        "metrics": metrics,
        "feature_importance": importance,
    }


def evaluate_model(test_df, y_true, y_pred, y_true_binary, y_pred_binary) -> dict:
    """Compute evaluation metrics including Precision@K."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    if y_true_binary.sum() > 0:
        prec = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        rec = recall_score(y_true_binary, y_pred_binary, zero_division=0)
        f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
    else:
        prec = rec = f1 = 0.0

    metrics = {"mae": mae, "rmse": rmse, "precision": prec, "recall": rec, "f1": f1}

    for k_pct in [5, 10, 20]:
        metrics[f"precision_at_{k_pct}pct"] = precision_at_k(test_df, k_pct)
        metrics[f"hit_rate_at_{k_pct}pct"] = hit_rate_at_k(test_df, k_pct)

    return metrics


def precision_at_k(test_df: pd.DataFrame, k_pct: int) -> float:
    """Of the top-K% predicted hotspots, what fraction were truly hot?"""
    k = max(1, int(len(test_df) * k_pct / 100))
    top_pred = test_df.nlargest(k, "pred_pcis")
    threshold = test_df["total_pcis"].quantile(1 - k_pct / 100)
    truly_hot = (top_pred["total_pcis"] >= threshold).sum()
    return truly_hot / k if k > 0 else 0


def hit_rate_at_k(test_df: pd.DataFrame, k_pct: int) -> float:
    """Of the truly-top-K% hotspots, what fraction did we predict?"""
    k = max(1, int(len(test_df) * k_pct / 100))
    threshold = test_df["total_pcis"].quantile(1 - k_pct / 100)
    actual_hot = test_df[test_df["total_pcis"] >= threshold]
    top_pred_cells = set(test_df.nlargest(k, "pred_pcis")["h3_cell"])
    if len(actual_hot) == 0:
        return 0.0
    hits = actual_hot["h3_cell"].isin(top_pred_cells).sum()
    return hits / len(actual_hot)


def predict_next_shift(model_bundle: dict, current_features: pd.DataFrame) -> pd.DataFrame:
    """Given current state features, predict PCIS for the next time window."""
    clf = model_bundle["classifier"]
    reg = model_bundle["regressor"]
    feature_cols = model_bundle["feature_cols"]

    X = current_features[feature_cols].fillna(0)
    pred_binary = clf.predict(X)
    pred_pcis = np.zeros(len(X))
    pos_mask = pred_binary == 1
    if pos_mask.any():
        pred_pcis[pos_mask] = reg.predict(X[pos_mask])
    pred_pcis = np.clip(pred_pcis, 0, None)

    result = current_features.copy()
    result["pred_pcis"] = pred_pcis
    result["pred_binary"] = pred_binary
    return result
