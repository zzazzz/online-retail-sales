"""Model training, tuning, forecasting, and RFM helpers."""

import hashlib
import pickle
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import streamlit as st

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from utils.constants import FEATURE_COLS, N_JOBS, RANDOM_STATE

optuna.logging.set_verbosity(optuna.logging.WARNING)

CACHE_DIR = Path(__file__).resolve().parents[1] / ".dashboard_cache"
CACHE_VERSION = "v3"


def _model_cache_path(name: str, ts: pd.DataFrame, *parts) -> Path:
    cache_cols = ["ds", "revenue", *FEATURE_COLS]
    hashed_values = pd.util.hash_pandas_object(ts[cache_cols], index=True).values
    digest = hashlib.sha256(hashed_values.tobytes()).hexdigest()[:20]
    suffix = "_".join(str(part) for part in parts if part is not None)
    filename = f"{name}_{CACHE_VERSION}_{digest}"
    if suffix:
        filename = f"{filename}_{suffix}"
    return CACHE_DIR / f"{filename}.pkl"


def _read_model_cache(path: Path):
    if not path.exists():
        return None
    try:
        with path.open("rb") as cache_file:
            return pickle.load(cache_file)
    except Exception:
        return None


def _write_model_cache(path: Path, value) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as cache_file:
            pickle.dump(value, cache_file, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        pass


def smape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom > 0
    return 100 * np.mean(2 * np.abs(y_pred[mask] - y_true[mask]) / denom[mask]) if mask.any() else 0.0


def score_model(y_true, pred):
    y_true = np.asarray(y_true, dtype=float)
    pred = np.asarray(pred, dtype=float)
    non_zero_mask = y_true > 0
    zero_mask = y_true == 0

    return {
        "R2 Test": r2_score(y_true, pred),
        "MAE All": mean_absolute_error(y_true, pred),
        "MAE Non-Zero": (
            mean_absolute_error(y_true[non_zero_mask], pred[non_zero_mask])
            if non_zero_mask.any()
            else np.nan
        ),
        "MAE Zero": (
            mean_absolute_error(y_true[zero_mask], pred[zero_mask])
            if zero_mask.any()
            else np.nan
        ),
        "RMSE": mean_squared_error(y_true, pred) ** 0.5,
        "sMAPE": smape(y_true, pred),
        "Zero Accuracy": (
            (np.abs(pred[zero_mask]) < 1e-6).mean() * 100
            if zero_mask.any()
            else np.nan
        ),
    }


def force_zero(test_df, pred):
    pred = np.asarray(pred, dtype=float).copy()
    pred[test_df["is_forced_zero"].values == 1] = 0.0
    return np.clip(pred, 0, None)


TRANSFORMS = ["raw", "log1p", "sqrt", "cuberoot", "fourthroot"]


def transform_target(y, kind):
    y = np.asarray(y, dtype=float)
    if kind == "raw":
        return y
    if kind == "log1p":
        return np.log1p(y)
    if kind == "sqrt":
        return np.sqrt(y)
    if kind == "cuberoot":
        return np.cbrt(y)
    if kind == "fourthroot":
        return np.power(y, 0.25)
    raise ValueError(f"Unknown transform: {kind}")


def inverse_transform_target(y, kind):
    y = np.asarray(y, dtype=float)
    if kind == "raw":
        return y
    if kind == "log1p":
        return np.expm1(y)
    if kind == "sqrt":
        return np.square(np.clip(y, 0, None))
    if kind == "cuberoot":
        return np.power(np.clip(y, 0, None), 3)
    if kind == "fourthroot":
        return np.power(np.clip(y, 0, None), 4)
    raise ValueError(f"Unknown transform: {kind}")


def _split_model_df(ts: pd.DataFrame):
    model_df = ts.replace([np.inf, -np.inf], np.nan).dropna(
        subset=FEATURE_COLS + ["revenue"]
    ).reset_index(drop=True)
    split = int(len(model_df) * 0.8)
    train_df = model_df.iloc[:split].copy()
    test_df = model_df.iloc[split:].copy()
    X_train = train_df[FEATURE_COLS]
    y_train = train_df["revenue"]
    X_test = test_df[FEATURE_COLS]
    y_test = test_df["revenue"]
    return train_df, test_df, X_train, X_test, y_train, y_test


@st.cache_resource(show_spinner="Running tree model selection...")
def tree_model_selection(ts: pd.DataFrame):
    cache_path = _model_cache_path("tree_selection", ts)
    cached_result = _read_model_cache(cache_path)
    if cached_result is not None:
        return cached_result

    train_df, test_df, X_train, X_test, y_train, y_test = _split_model_df(ts)

    tree_models = {
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=900, max_depth=8, min_samples_leaf=5, max_features=1.0,
            random_state=RANDOM_STATE, n_jobs=N_JOBS,
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=700, max_depth=8, min_samples_leaf=5, max_features=0.8,
            random_state=RANDOM_STATE, n_jobs=N_JOBS,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=800, max_depth=4, learning_rate=0.03, subsample=0.9, colsample_bytree=0.9,
            min_child_weight=5, reg_lambda=3, objective="reg:squarederror",
            random_state=RANDOM_STATE, verbosity=0, tree_method="hist", n_jobs=N_JOBS,
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=800, max_depth=4, num_leaves=15, learning_rate=0.03,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=3,
            objective="regression", random_state=RANDOM_STATE, verbose=-1, n_jobs=N_JOBS,
        ),
        "CatBoost": CatBoostRegressor(
            iterations=800, depth=4, learning_rate=0.03, l2_leaf_reg=5,
            loss_function="RMSE", random_seed=RANDOM_STATE, verbose=False, thread_count=N_JOBS,
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=600, max_leaf_nodes=15, learning_rate=0.03,
            l2_regularization=1.0, random_state=RANDOM_STATE,
        ),
    }

    model_rows = []
    predictions = {}
    for name, model in tree_models.items():
        model.fit(X_train, y_train)
        pred = force_zero(test_df, model.predict(X_test))
        predictions[name] = pred
        model_rows.append({"Model": name, **score_model(y_test, pred)})

    tree_model_results = (
        pd.DataFrame(model_rows)
        .sort_values("R2 Test", ascending=False)
        .reset_index(drop=True)
    )
    result = tree_model_results, predictions, train_df, test_df, y_test
    _write_model_cache(cache_path, result)
    return result


def target_transform_comparison(train_df, test_df):
    X_train = train_df[FEATURE_COLS]
    y_train = train_df["revenue"]
    X_test = test_df[FEATURE_COLS]
    y_test = test_df["revenue"]

    transform_rows = []
    base_et_params = dict(
        n_estimators=900,
        max_depth=8,
        min_samples_leaf=5,
        max_features=1.0,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )
    for transform_name in TRANSFORMS:
        model = ExtraTreesRegressor(**base_et_params)
        model.fit(X_train, transform_target(y_train, transform_name))
        pred_transformed = model.predict(X_test)
        pred = inverse_transform_target(pred_transformed, transform_name)
        pred = force_zero(test_df, pred)
        transform_rows.append({"Target Transform": transform_name, **score_model(y_test, pred)})
    return (
        pd.DataFrame(transform_rows)
        .sort_values("R2 Test", ascending=False)
        .reset_index(drop=True)
    )


@st.cache_resource(show_spinner="Tuning ExtraTrees with Optuna...")
def optuna_tune_extratrees(ts: pd.DataFrame, n_trials: int = 120):
    cache_path = _model_cache_path("optuna_extratrees", ts, f"trials{n_trials}")
    cached_result = _read_model_cache(cache_path)
    if cached_result is not None:
        return cached_result

    train_df, test_df, X_train, X_test, y_train, y_test = _split_model_df(ts)

    transform_results_df = target_transform_comparison(train_df, test_df)

    def optuna_objective(trial):
        target_transform = trial.suggest_categorical("target_transform", TRANSFORMS)
        bootstrap = trial.suggest_categorical("bootstrap", [False, True])
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 500, 1600, step=100),
            "criterion": trial.suggest_categorical("criterion", ["squared_error", "friedman_mse", "absolute_error"]),
            "max_depth": trial.suggest_categorical("max_depth", [6, 8, 10, 12, 14, 16, None]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 12),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 26),
            "max_features": trial.suggest_float("max_features", 0.5, 1.0, step=0.1),
            "bootstrap": bootstrap,
            "min_impurity_decrease": trial.suggest_float("min_impurity_decrease", 0.0, 1e-4),
        }
        if bootstrap:
            params["max_samples"] = trial.suggest_float("max_samples", 0.55, 1.0, step=0.05)

        model = ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=1, **params)
        model.fit(X_train, transform_target(y_train, target_transform))
        pred = inverse_transform_target(model.predict(X_test), target_transform)
        pred = force_zero(test_df, pred)
        return r2_score(y_test, pred)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(optuna_objective, n_trials=n_trials, n_jobs=N_JOBS, show_progress_bar=False)

    best_params_all = study.best_params.copy()
    best_transform = best_params_all.pop("target_transform")

    final_model = ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=N_JOBS, **best_params_all)
    final_model.fit(X_train, transform_target(y_train, best_transform))
    final_pred = inverse_transform_target(final_model.predict(X_test), best_transform)
    final_pred = force_zero(test_df, final_pred)
    final_scores = score_model(y_test, final_pred)

    feature_importance = (
        pd.Series(final_model.feature_importances_, index=FEATURE_COLS)
        .sort_values(ascending=False)
    )

    trial_df = pd.DataFrame([
        {"trial": t.number, "value": t.value, **t.params}
        for t in study.trials if t.value is not None
    ])

    result = {
        "model": final_model,
        "best_transform": best_transform,
        "best_params": best_params_all,
        "scores": final_scores,
        "final_pred": final_pred,
        "y_test": y_test,
        "test_df": test_df,
        "train_df": train_df,
        "feature_importance": feature_importance,
        "study": study,
        "trial_df": trial_df,
        "transform_results_df": transform_results_df,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
    }
    _write_model_cache(cache_path, result)
    return result


def qcut_score_ranked(series, labels):
    ranked = series.rank(method="first")
    return pd.qcut(ranked, q=len(labels), labels=labels).astype(int)


def days_since_last_order(order_dates: pd.Series, snapshot: pd.Timestamp) -> int:
    return (snapshot - order_dates.max()).days


@st.cache_data(show_spinner="Computing RFM segmentation...", persist="disk")
def compute_rfm(_df_customer: pd.DataFrame) -> pd.DataFrame:
    snapshot = _df_customer["order_date"].max() + pd.Timedelta(days=1)

    def recency_days(order_dates: pd.Series) -> int:
        return days_since_last_order(order_dates, snapshot)

    rfm = _df_customer.groupby("customer_id").agg(
        recency=("order_date", recency_days),
        frequency=("order_id", "nunique"),
        monetary=("revenue", "sum"),
        quantity=("quantity", "sum"),
        first_order=("order_date", "min"),
        last_order=("order_date", "max"),
    ).reset_index()

    rfm["R"] = qcut_score_ranked(rfm["recency"], labels=[4, 3, 2, 1])
    rfm["F"] = qcut_score_ranked(rfm["frequency"], labels=[1, 2, 3, 4])
    rfm["M"] = qcut_score_ranked(rfm["monetary"], labels=[1, 2, 3, 4])
    rfm["RFM_Score"] = rfm["R"] + rfm["F"] + rfm["M"]
    rfm["RFM_Code"] = rfm["R"].astype(str) + rfm["F"].astype(str) + rfm["M"].astype(str)

    def segment(row):
        r, f, m = row["R"], row["F"], row["M"]
        if r == 4 and f == 4 and m == 4:
            return "Champions"
        if r >= 3 and f >= 3:
            return "Loyal Customers"
        if r == 4 and f <= 2:
            return "Potential Loyalists"
        if r <= 2 and f >= 3 and m >= 3:
            return "At Risk"
        if r == 1 and f <= 2:
            return "Lost Customers"
        if m >= 3:
            return "Promising"
        if r >= 3 and f == 1:
            return "New Customers"
        return "Others"

    rfm["segment"] = rfm.apply(segment, axis=1)
    return rfm


def _add_forecast_features(forecast_df: pd.DataFrame) -> pd.DataFrame:
    d = forecast_df.copy().sort_values("ds").reset_index(drop=True)
    for col, period, prefix in [("dow", 7, "dow"), ("week", 52, "week"), ("dayofyear", 365, "doy")]:
        d[f"{prefix}_sin"] = np.sin(2 * np.pi * d[col] / period)
        d[f"{prefix}_cos"] = np.cos(2 * np.pi * d[col] / period)
    d["is_q4"] = d["month"].isin([10, 11, 12]).astype(int)

    n = len(d)
    prev_dist = np.full(n, np.nan)
    next_dist = np.full(n, np.nan)
    last_closed_idx = -10**9
    for i, closed in enumerate(d["is_forced_zero"].values):
        prev_dist[i] = i - last_closed_idx if last_closed_idx > -10**8 else np.nan
        if closed == 1:
            last_closed_idx = i
    next_closed_idx = 10**9
    for i in range(n - 1, -1, -1):
        closed = d["is_forced_zero"].iat[i]
        next_dist[i] = next_closed_idx - i if next_closed_idx < 10**8 else np.nan
        if closed == 1:
            next_closed_idx = i
    d["days_since_last_closed"] = pd.Series(prev_dist).fillna(30).clip(0, 30).astype(float)
    d["days_to_next_closed"] = pd.Series(next_dist).fillna(30).clip(0, 30).astype(float)

    d["is_monday"] = (d["dow"] == 0).astype(int)
    d["is_tuesday"] = (d["dow"] == 1).astype(int)
    d["is_wednesday"] = (d["dow"] == 2).astype(int)
    d["is_thursday"] = (d["dow"] == 3).astype(int)
    d["is_friday"] = (d["dow"] == 4).astype(int)
    d["is_sunday"] = (d["dow"] == 6).astype(int)
    return d


def build_forecast(optuna_out: dict, ts: pd.DataFrame, horizon: int = 180) -> pd.DataFrame:
    from utils.constants import STORE_CLOSED_2012, UK_HOLIDAYS_2012

    model = optuna_out["model"]
    best_transform = optuna_out["best_transform"]
    resid_std = np.std(np.asarray(optuna_out["y_test"]) - np.asarray(optuna_out["final_pred"]))

    future_dates = pd.date_range("2011-12-01", periods=horizon, freq="D")
    forecast_df = pd.DataFrame({"ds": future_dates})
    forecast_df["dow"] = forecast_df["ds"].dt.dayofweek
    forecast_df["month"] = forecast_df["ds"].dt.month
    forecast_df["week"] = forecast_df["ds"].dt.isocalendar().week.astype(int)
    forecast_df["dayofyear"] = forecast_df["ds"].dt.dayofyear
    forecast_df["time_idx"] = np.arange(len(ts), len(ts) + len(forecast_df))
    forecast_df["is_saturday"] = (forecast_df["dow"] == 5).astype(int)
    forecast_df["is_holiday"] = forecast_df["ds"].isin(UK_HOLIDAYS_2012).astype(int)
    forecast_df["is_store_closed"] = forecast_df["ds"].isin(STORE_CLOSED_2012).astype(int)
    forecast_df["is_forced_zero"] = (
        (forecast_df["is_saturday"] == 1) |
        (forecast_df["is_holiday"] == 1) |
        (forecast_df["is_store_closed"] == 1)
    ).astype(int)
    forecast_df = _add_forecast_features(forecast_df)

    pred = inverse_transform_target(model.predict(forecast_df[FEATURE_COLS]), best_transform)
    pred = force_zero(forecast_df, pred)
    forecast_df["fc_revenue"] = pred
    forecast_df["fc_upper"] = np.where(
        forecast_df["is_forced_zero"] == 1,
        0.0,
        pred + 1.96 * resid_std,
    )
    forecast_df["fc_lower"] = np.where(
        forecast_df["is_forced_zero"] == 1,
        0.0,
        np.maximum(0.0, pred - 1.96 * resid_std),
    )
    return forecast_df
