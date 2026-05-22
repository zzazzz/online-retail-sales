"""Refund and cancellation page."""

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from utils.charts import kpi, WARN_BOX


def render(df_cancelled, compare):
    st.markdown("##  Refund & Cancellation Analysis")

    total_ref = abs(df_cancelled["revenue"].sum())
    avg_rate = compare["refund_rate"].mean()
    n_canc = df_cancelled["order_id"].nunique()
    peak_row = compare.loc[compare["canc_revenue_abs"].idxmax()]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(kpi("Total Refund Value", f"{total_ref/1e3:.1f}K", neg=True), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi("Avg Refund Rate", f"{avg_rate:.1f}%", neg=True), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi("Cancelled Invoices", f"{n_canc:,}"), unsafe_allow_html=True)
    with c4:
        peak_month = peak_row["month_year_dt"].strftime("%b %Y")
        st.markdown(kpi("Peak Refund Month", peak_month, neg=True), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    refund_fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=(
            "Gross Revenue vs Refund Lost",
            "Refund Rate (%) per Month",
            "Cancelled Orders Count",
        ),
    )
    refund_fig.add_trace(go.Bar(
        x=compare["month_year_dt"],
        y=compare["revenue"],
        name="Gross Revenue",
        marker_color="#3b82f6",
        opacity=0.75,
        hovertemplate="<b>%{x|%b %Y}</b><br>Gross: %{y:,.0f}<extra></extra>",
    ), row=1, col=1)
    refund_fig.add_trace(go.Bar(
        x=compare["month_year_dt"],
        y=compare["canc_revenue_abs"],
        name="Refund",
        marker_color="#ef4444",
        opacity=0.85,
        hovertemplate="<b>%{x|%b %Y}</b><br>Refund: %{y:,.0f}<extra></extra>",
    ), row=1, col=1)
    refund_fig.add_trace(go.Scatter(
        x=compare["month_year_dt"],
        y=compare["refund_rate"],
        name="Refund Rate %",
        line=dict(color="#f59e0b", width=2),
        hovertemplate="<b>%{x|%b %Y}</b><br>Rate: %{y:.1f}%<extra></extra>",
    ), row=2, col=1)
    refund_fig.add_hline(
        y=compare["refund_rate"].mean(),
        line_color="#ef4444",
        line_dash="dot",
        row=2,
        col=1,
    )
    refund_fig.add_trace(go.Bar(
        x=compare["month_year_dt"],
        y=compare["canc_orders"],
        name="Cancelled Orders",
        marker_color="#8b5cf6",
        opacity=0.75,
        hovertemplate="<b>%{x|%b %Y}</b><br>%{y:,} cancellations<extra></extra>",
    ), row=3, col=1)
    refund_fig.update_layout(
        paper_bgcolor="#0f172a", plot_bgcolor="#111827",
        font_family="DM Sans", font_color="#dbe7f5",
        height=700, margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        barmode="overlay",
    )
    refund_fig.update_xaxes(showgrid=True, gridcolor="#263348")
    refund_fig.update_yaxes(showgrid=True, gridcolor="#263348")
    st.plotly_chart(refund_fig, use_container_width=True)

    st.markdown(WARN_BOX(
        f"Refund rate rata-rata <b>{avg_rate:.1f}%</b> dari gross revenue. "
        "Desember memiliki refund terbesar  kemungkinan return produk hadiah yang tidak sesuai. "
        "Refund rate yang fluktuatif adalah indikasi masalah kualitas produk atau miskomunikasi deskripsi."
    ), unsafe_allow_html=True)

