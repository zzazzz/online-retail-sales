"""Data loading, cleaning, and aggregation helpers for the dashboard."""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st

from utils.constants import FEATURE_COLS, NON_PRODUCT_CODES, STORE_CLOSED, UK_HOLIDAYS


@st.cache_data(show_spinner=" Loading data", persist="disk")
def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    for col in ["order_id", "product_id", "product_description", "customer_id", "country"]:
        if col in df.columns:
            df[col] = df[col].astype("string")
    return df


@st.cache_data(show_spinner=False, persist="disk")
def line_type_summary(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Revenue split: Product vs Non-product/Admin (pre-cleaning view)."""
    is_cancel = df_raw["order_id"].str.startswith("C", na=False)
    valid = df_raw[(~is_cancel) & (df_raw["quantity"] > 0) & (df_raw["unit_price"] > 0)].copy()
    valid["line_revenue"] = valid["quantity"] * valid["unit_price"]
    valid["product_id_clean"] = valid["product_id"].astype("string").str.strip()
    valid["is_non_product_line"] = valid["product_id_clean"].isin(NON_PRODUCT_CODES)
    valid["line_type"] = np.where(valid["is_non_product_line"], "Non-product/Admin", "Product")
    return (
        valid.groupby("line_type")
        .agg(revenue=("line_revenue", "sum"), rows=("order_id", "count"))
        .reset_index()
        .sort_values("revenue", ascending=False)
    )


@st.cache_data(show_spinner=" Cleaning data (product-based)", persist="disk")
def clean_retail_data_product_revenue(df_raw: pd.DataFrame):
    """Clean product-sales rows without applying percentile outlier filtering."""
    is_cancel = df_raw["order_id"].str.startswith("C", na=False)

    df_cancelled = df_raw[is_cancel].copy()
    df_cancelled["revenue"] = df_cancelled["quantity"] * df_cancelled["unit_price"]
    df_cancelled["month_year"] = df_cancelled["order_date"].dt.to_period("M")

    df_valid = df_raw[(~is_cancel) & (df_raw["quantity"] > 0) & (df_raw["unit_price"] > 0)].copy()
    df_valid["line_revenue"] = df_valid["quantity"] * df_valid["unit_price"]
    df_valid["product_id_clean"] = df_valid["product_id"].astype("string").str.strip()

    df_sales = df_valid[~df_valid["product_id_clean"].isin(NON_PRODUCT_CODES)].copy()
    df_sales = df_sales[df_sales["product_description"].notna()].copy()
    df_sales = df_sales.drop_duplicates()

    df_sales["revenue"] = df_sales["quantity"] * df_sales["unit_price"]
    df_sales["year"] = df_sales["order_date"].dt.year
    df_sales["month"] = df_sales["order_date"].dt.month
    df_sales["month_year"] = df_sales["order_date"].dt.to_period("M")
    df_sales["day_of_week"] = df_sales["order_date"].dt.day_name()

    df_customer = df_sales[df_sales["customer_id"].notna()].copy()

    return df_sales, df_customer, df_cancelled


@st.cache_data(show_spinner=" Computing aggregations", persist="disk")
def compute_aggregations(df_sales, df_cancelled, df_customer):
    monthly = df_sales.groupby("month_year").agg(
        revenue=("revenue", "sum"),
        orders=("order_id", "nunique"),
        customers=("customer_id", "nunique"),
        qty=("quantity", "sum"),
    ).reset_index()
    monthly["month_year_dt"] = monthly["month_year"].dt.to_timestamp()
    monthly["revenue_ma3"] = monthly["revenue"].rolling(3, min_periods=1).mean()
    monthly["aov"] = monthly["revenue"] / monthly["orders"].replace(0, np.nan)

    canc_m = df_cancelled.groupby("month_year").agg(
        canc_revenue=("revenue", "sum"),
        canc_orders=("order_id", "nunique"),
    ).reset_index()
    canc_m["canc_revenue_abs"] = canc_m["canc_revenue"].abs()
    compare = monthly.merge(canc_m, on="month_year", how="left").fillna(0)
    compare["month_year_dt"] = compare["month_year"].dt.to_timestamp()
    compare["refund_rate"] = compare["canc_revenue_abs"] / compare["revenue"].replace(0, np.nan) * 100

    product_after = df_sales.groupby(["product_id", "product_description"]).agg(
        revenue=("revenue", "sum"),
        qty=("quantity", "sum"),
        orders=("order_id", "nunique"),
    ).reset_index().sort_values("revenue", ascending=False)
    product_after["avg_price"] = product_after["revenue"] / product_after["qty"].replace(0, np.nan)
    product_after["product_label"] = (
        product_after["product_id"].astype(str) + "  " +
        product_after["product_description"].astype(str).str[:40]
    )

    country = df_sales.groupby("country").agg(
        revenue=("revenue", "sum"),
        orders=("order_id", "nunique"),
        customers=("customer_id", "nunique"),
    ).reset_index().sort_values("revenue", ascending=False)
    country["revenue_pct"] = (country["revenue"] / country["revenue"].sum() * 100).round(2)

    cust = df_customer.groupby("customer_id").agg(
        revenue=("revenue", "sum"),
        orders=("order_id", "nunique"),
        qty=("quantity", "sum"),
        first_order=("order_date", "min"),
        last_order=("order_date", "max"),
    ).reset_index()
    cust["aov"] = cust["revenue"] / cust["orders"]

    return monthly, compare, product_after, country, cust


@st.cache_data(show_spinner=" Building daily series", persist="disk")
def build_daily_series(df_sales: pd.DataFrame) -> pd.DataFrame:
    """
    Build a complete daily revenue series with the same calendar rules used in the notebook.
    """
    daily = df_sales.groupby(df_sales["order_date"].dt.normalize()).agg(
        revenue=("revenue", "sum"),
    ).reset_index().rename(columns={"order_date": "ds"})
    idx = pd.date_range(daily["ds"].min(), "2011-11-30", freq="D")
    ts = daily.set_index("ds").reindex(idx).rename_axis("ds").reset_index()
    ts["revenue"] = ts["revenue"].fillna(0.0)
    ts["dow"] = ts["ds"].dt.dayofweek
    ts["month"] = ts["ds"].dt.month
    ts["week"] = ts["ds"].dt.isocalendar().week.astype(int)
    ts["dayofyear"] = ts["ds"].dt.dayofyear
    ts["time_idx"] = np.arange(len(ts))
    ts["is_saturday"] = (ts["dow"] == 5).astype(int)
    ts["is_holiday"] = ts["ds"].isin(UK_HOLIDAYS).astype(int)
    ts["is_store_closed"] = ts["ds"].isin(STORE_CLOSED).astype(int)
    ts["is_forced_zero"] = (
        (ts["is_saturday"] == 1) |
        (ts["is_holiday"] == 1) |
        (ts["is_store_closed"] == 1)
    ).astype(int)
    ts.loc[ts["is_forced_zero"] == 1, "revenue"] = 0.0
    return _add_features(ts).replace([np.inf, -np.inf], np.nan).dropna(
        subset=FEATURE_COLS + ["revenue"]
    ).reset_index(drop=True)


def _add_features(ts: pd.DataFrame) -> pd.DataFrame:
    d = ts.copy().sort_values("ds").reset_index(drop=True)
    for col, period, prefix in [("dow", 7, "dow"), ("week", 52, "week"), ("dayofyear", 365, "doy")]:
        d[f"{prefix}_sin"] = np.sin(2 * np.pi * d[col] / period)
        d[f"{prefix}_cos"] = np.cos(2 * np.pi * d[col] / period)
    d["is_q4"] = d["month"].isin([10, 11, 12]).astype(int)

    # Distance-to-closure is calculated from both directions so forecast rows
    # can use the same feature shape as the historical model rows.
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

