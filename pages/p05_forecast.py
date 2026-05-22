"""Revenue forecast page."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.charts import INFO_BOX, dark_fig, kpi


def count_open_days(series):
    return (series == 0).sum()


def render(ts, optuna_out, future_df, forecast_horizon):
    st.markdown("##  Revenue Forecast")

    best_tr = optuna_out["best_transform"]
    scores = optuna_out["scores"]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(kpi("Model", "Optuna ExtraTrees"), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi("Target Transform", best_tr.upper()), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi("R2 Test", f"{scores['R2 Test']:.4f}"), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi("Horizon", f"{forecast_horizon} hari"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    tab1, tab2 = st.tabs([" Daily Forecast", " Monthly Summary"])

    with tab1:
        hist_tail = ts[ts["ds"] >= "2011-06-01"]

        forecast_fig = go.Figure()
        forecast_fig.add_trace(go.Scatter(
            x=hist_tail["ds"], y=hist_tail["revenue"],
            mode="lines", name="Historis",
            line=dict(color="#7a8aa0", width=1.2),
            hovertemplate="<b>%{x|%d %b %Y}</b><br>GBP %{y:,.0f}<extra></extra>",
        ))
        forecast_fig.add_trace(go.Scatter(
            x=pd.concat([future_df["ds"], future_df["ds"][::-1]]),
            y=pd.concat([future_df["fc_upper"], future_df["fc_lower"][::-1]]),
            fill="toself", fillcolor="rgba(59,130,246,0.12)",
            line=dict(width=0), name="95% CI", hoverinfo="skip",
        ))
        forecast_fig.add_trace(go.Scatter(
            x=future_df["ds"], y=future_df["fc_revenue"],
            mode="lines", name="Forecast",
            line=dict(color="#3b82f6", width=2.2),
            hovertemplate="<b>%{x|%d %b %Y}</b><br>Forecast: GBP %{y:,.0f}<extra></extra>",
        ))
        closed = future_df[future_df["is_forced_zero"] == 1]
        forecast_fig.add_trace(go.Scatter(
            x=closed["ds"], y=np.zeros(len(closed)),
            mode="markers", name="Hari Tutup",
            marker=dict(color="#ef4444", size=4, symbol="x", opacity=0.6),
        ))
        forecast_fig.add_shape(
            type="line",
            x0="2011-12-01",
            x1="2011-12-01",
            y0=0,
            y1=1,
            yref="paper",
            line=dict(color="#f59e0b", width=1, dash="dash"),
        )
        forecast_fig.add_annotation(
            x="2011-12-01",
            y=0.98,
            yref="paper",
            text="Forecast Start",
            showarrow=False,
            font=dict(color="#f59e0b", size=11),
            xshift=55,
        )
        dark_fig(forecast_fig, 520, f"Daily Product-Sales Forecast  {forecast_horizon} Hari ke Depan")
        forecast_fig.update_yaxes(tickprefix="GBP ")
        st.plotly_chart(forecast_fig, use_container_width=True)

        open_fc = future_df[future_df["is_forced_zero"] == 0]
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.markdown(kpi("Total Forecast", f"{future_df['fc_revenue'].sum():,.0f}"), unsafe_allow_html=True)
        with s2:
            st.markdown(kpi("Avg Hari Buka", f"{open_fc['fc_revenue'].mean():,.0f}"), unsafe_allow_html=True)
        with s3:
            st.markdown(kpi("Peak Day", f"{future_df['fc_revenue'].max():,.0f}"), unsafe_allow_html=True)
        with s4:
            st.markdown(kpi("Hari Operasional", f"{len(open_fc)} / {forecast_horizon}"), unsafe_allow_html=True)

        st.markdown(INFO_BOX(
            "Forecast menggunakan model Optuna ExtraTrees secara rekursif. "
            "Hari Sabtu, UK bank holidays 2012, dan observed store closures di-force zero. "
            "95% CI = 1.96  residual std dari test set."
        ), unsafe_allow_html=True)

    with tab2:
        forecast_by_day = future_df.copy()
        forecast_by_day["month_year_dt"] = forecast_by_day["ds"].dt.to_period("M").dt.to_timestamp()
        monthly_forecast = forecast_by_day.groupby("month_year_dt").agg(
            forecast_gbp=("fc_revenue", "sum"),
            forecast_upper=("fc_upper", "sum"),
            forecast_lower=("fc_lower", "sum"),
            op_days=("is_forced_zero", count_open_days),
        ).reset_index()

        monthly_fig = go.Figure()
        monthly_fig.add_trace(go.Bar(
            x=monthly_forecast["month_year_dt"].dt.strftime("%b %Y"),
            y=monthly_forecast["forecast_gbp"],
            name="Forecast", marker_color="#3b82f6", opacity=0.85,
            error_y=dict(
                type="data", symmetric=False,
                array=monthly_forecast["forecast_upper"] - monthly_forecast["forecast_gbp"],
                arrayminus=monthly_forecast["forecast_gbp"] - monthly_forecast["forecast_lower"],
                color="#f59e0b",
            ),
            text=(monthly_forecast["forecast_gbp"] / 1e3).map("{:.0f}K".format),
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>GBP %{y:,.0f}<extra></extra>",
        ))
        dark_fig(monthly_fig, 420, f"Monthly Forecast  {forecast_horizon} Hari ke Depan")
        monthly_fig.update_yaxes(tickprefix="GBP ")
        st.plotly_chart(monthly_fig, use_container_width=True)

        monthly_forecast["Bulan"] = monthly_forecast["month_year_dt"].dt.strftime("%B %Y")
        monthly_forecast["Forecast"] = monthly_forecast["forecast_gbp"].map("{:,.0f}".format)
        monthly_forecast["95% CI"] = (
            monthly_forecast["forecast_lower"].map("{:,.0f}".format)
            + "  "
            + monthly_forecast["forecast_upper"].map("{:,.0f}".format)
        )
        monthly_forecast["Hari Buka"] = monthly_forecast["op_days"].map("{} hari".format)
        st.dataframe(monthly_forecast[["Bulan", "Forecast", "95% CI", "Hari Buka"]],
                     use_container_width=True, hide_index=True)

