"""Shared constants for the retail analytics dashboard."""

import pandas as pd

RANDOM_STATE = 42
N_JOBS = 3

# Stock codes that are fees, postage, adjustments, or other non-product rows.
NON_PRODUCT_CODES = {
    "POST", "DOT", "M", "m", "D", "S", "AMAZONFEE",
    "ADJUST", "B", "BANK CHARGES", "C2", "CRUK",
}

UK_HOLIDAYS = pd.to_datetime([
    "2009-12-25", "2010-01-01", "2010-04-02", "2010-04-05",
    "2010-05-03", "2010-05-31", "2010-08-30", "2010-12-27",
    "2010-12-28", "2011-01-03", "2011-04-22", "2011-04-25",
    "2011-04-29", "2011-05-02", "2011-05-30", "2011-08-29",
])

STORE_CLOSED = pd.to_datetime([
    "2009-12-24", "2009-12-29", "2009-12-30", "2009-12-31",
    "2010-12-24", "2010-12-29", "2010-12-30", "2010-12-31",
    "2009-12-27", "2010-01-03", "2011-01-02", "2011-04-24",
])

UK_HOLIDAYS_2012 = pd.to_datetime([
    "2012-01-02", "2012-04-06", "2012-04-09", "2012-05-07",
    "2012-06-04", "2012-06-05", "2012-08-27",
    "2012-12-25", "2012-12-26",
])
STORE_CLOSED_2012 = pd.to_datetime([
    "2011-12-24", "2011-12-25", "2011-12-26", "2011-12-27",
    "2011-12-28", "2011-12-29", "2011-12-30", "2011-12-31",
    "2012-12-24", "2012-12-27", "2012-12-28",
    "2012-12-29", "2012-12-30", "2012-12-31",
])

PLOTLY_TPL = "plotly_dark"
PAPER_BG = "#0f172a"
PLOT_BG = "#111827"
GRID_COL = "#263348"

SEG_COLORS = {
    "Champions": "#16803c",
    "Loyal Customers": "#1f6feb",
    "Potential Loyalists": "#7c3aed",
    "At Risk": "#d97706",
    "Lost Customers": "#dc2626",
    "Promising": "#0891b2",
    "New Customers": "#db2777",
    "Others": "#94a3b8",
}

FEATURE_COLS = [
    "is_forced_zero", "dow", "dow_sin", "is_q4", "dayofyear", "doy_sin", "doy_cos",
    "week", "week_sin", "week_cos", "month", "time_idx",
    "days_since_last_closed", "days_to_next_closed",
    "is_monday", "is_tuesday", "is_wednesday", "is_thursday", "is_friday", "is_sunday",
]

