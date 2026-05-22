"""RFM customer segmentation page."""

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.charts import INFO_BOX, SEC_TITLE, dark_fig, kpi
from utils.constants import PLOTLY_TPL, SEG_COLORS


def render(rfm):
    st.markdown("##  Customer Segmentation  RFM (9)")

    st.markdown(INFO_BOX(
        "RFM dihitung dari <b>product-sales data</b> (exclude non-product/admin lines). "
        "Skor R, F, M skala 14 (semakin tinggi semakin baik). "
        "R = recency (hari sejak terakhir order), F = order unik, M = total product-sales revenue."
    ), unsafe_allow_html=True)

    segment_summary = rfm.groupby("segment").agg(
        customers=("customer_id", "nunique"),
        revenue=("monetary", "sum"),
        avg_r=("recency", "mean"),
        avg_f=("frequency", "mean"),
        avg_m=("monetary", "mean"),
    ).reset_index().sort_values("revenue", ascending=False)
    segment_summary["customer_pct"] = segment_summary["customers"] / segment_summary["customers"].sum() * 100
    segment_summary["revenue_pct"] = segment_summary["revenue"] / segment_summary["revenue"].sum() * 100

    champs = rfm[rfm["segment"] == "Champions"]
    at_risk = rfm[rfm["segment"] == "At Risk"]
    lost = rfm[rfm["segment"] == "Lost Customers"]
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(kpi("Total Customers", f"{len(rfm):,}"), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi("Champions", f"{len(champs):,}", f"{len(champs)/len(rfm)*100:.1f}%"), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi("At Risk", f"{len(at_risk):,}", f"{len(at_risk)/len(rfm)*100:.1f}%", neg=True), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi("Lost Customers", f"{len(lost):,}", f"{len(lost)/len(rfm)*100:.1f}%", neg=True), unsafe_allow_html=True)
    with c5:
        st.markdown(kpi("Avg LTV", f"{rfm['monetary'].mean():,.0f}"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs([" Segments", " RFM Scatter", " Top Customers"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            segment_revenue = segment_summary.sort_values("revenue")
            segment_revenue["revenue_pct_label"] = segment_revenue["revenue_pct"].map("{:.1f}%".format)
            segment_revenue_fig = px.bar(
                segment_revenue,
                x="revenue",
                y="segment",
                orientation="h",
                color="revenue_pct",
                color_continuous_scale="Blues",
                text="revenue_pct_label",
                template=PLOTLY_TPL,
                title="RFM Segment  Revenue Contribution",
            )
            segment_revenue_fig.update_traces(textposition="outside")
            dark_fig(segment_revenue_fig, 420)
            segment_revenue_fig.update_layout(
                coloraxis_showscale=False,
                yaxis_title="",
                xaxis_tickprefix="GBP ",
            )
            st.plotly_chart(segment_revenue_fig, use_container_width=True)

        with col2:
            segment_count_fig = px.pie(
                segment_summary,
                names="segment",
                values="customers",
                hole=0.45,
                color="segment",
                color_discrete_map=SEG_COLORS,
                template=PLOTLY_TPL,
                title="Customer Count by Segment",
            )
            segment_count_fig.update_traces(
                hovertemplate="<b>%{label}</b><br>%{value} customers (%{percent})<extra></extra>"
            )
            dark_fig(segment_count_fig, 420)
            st.plotly_chart(segment_count_fig, use_container_width=True)

        st.markdown(SEC_TITLE("Segment Summary Table"), unsafe_allow_html=True)
        segment_table = segment_summary.copy()
        segment_table["revenue_fmt"] = segment_table["revenue"].map("{:,.0f}".format)
        segment_table["avg_r_fmt"] = segment_table["avg_r"].map("{:.0f} hari".format)
        segment_table["avg_f_fmt"] = segment_table["avg_f"].map("{:.1f}x".format)
        segment_table["avg_m_fmt"] = segment_table["avg_m"].map("{:,.0f}".format)
        segment_table["cust_pct_fmt"] = segment_table["customer_pct"].map("{:.1f}%".format)
        segment_table["rev_pct_fmt"] = segment_table["revenue_pct"].map("{:.1f}%".format)
        st.dataframe(
            segment_table[
                ["segment", "customers", "cust_pct_fmt", "revenue_fmt", "rev_pct_fmt", "avg_r_fmt", "avg_f_fmt", "avg_m_fmt"]
            ]
               .rename(columns={
                   "segment": "Segment", "customers": "Customers", "cust_pct_fmt": "Cust %",
                   "revenue_fmt": "Revenue", "rev_pct_fmt": "Rev %",
                   "avg_r_fmt": "Avg Recency", "avg_f_fmt": "Avg Frequency", "avg_m_fmt": "Avg LTV",
               }),
            use_container_width=True, hide_index=True,
        )

        st.markdown(INFO_BOX(
            " <b>Champions</b>  VIP reward, early access, referral programme.<br>"
            "<b>Loyal Customers</b>  Loyalty points, upsell ke produk premium.<br>"
            "<b>At Risk</b>  Win-back campaign, personal voucher, kepuasan survey.<br>"
            "<b>Lost Customers</b>  Reaktivasi diskon besar atau accept as churned."
        ), unsafe_allow_html=True)

    with tab2:
        sel_segs = st.multiselect("Filter Segment", sorted(rfm["segment"].unique()),
                                  default=sorted(rfm["segment"].unique()), key="rfm_seg")
        filtered_rfm = rfm[rfm["segment"].isin(sel_segs)] if sel_segs else rfm
        scatter_sample = filtered_rfm.sample(min(3000, len(filtered_rfm)), random_state=42)

        frequency_fig = px.scatter(
            scatter_sample,
            x="frequency",
            y="monetary",
            color="segment",
            color_discrete_map=SEG_COLORS,
            size=np.log1p(scatter_sample["monetary"]).clip(lower=1),
            hover_data={"customer_id": True, "recency": True, "RFM_Code": True},
            template=PLOTLY_TPL,
            labels={"frequency": "Frequency (Orders)", "monetary": "Monetary (GBP)"},
            title="Customer Segmentation  Frequency vs Monetary",
            opacity=0.75,
        )
        frequency_fig.update_traces(hovertemplate=(
            "<b>Customer %{customdata[0]}</b><br>"
            "Recency: %{customdata[1]} days<br>"
            "Freq: %{x}  Revenue: GBP %{y:,.0f}<extra></extra>"
        ))
        dark_fig(frequency_fig, 520)
        st.plotly_chart(frequency_fig, use_container_width=True)

        recency_sample = filtered_rfm.sample(min(3000, len(filtered_rfm)), random_state=99)
        recency_fig = px.scatter(
            recency_sample,
            x="recency",
            y="monetary",
            color="segment",
            color_discrete_map=SEG_COLORS,
            template=PLOTLY_TPL,
            labels={"recency": "Recency (days)", "monetary": "Monetary (GBP)"},
            title="Recency vs Monetary",
            opacity=0.65,
        )
        dark_fig(recency_fig, 400)
        st.plotly_chart(recency_fig, use_container_width=True)

    with tab3:
        n_top = st.slider("Top N customers", 10, 50, 20, key="rfm_top")
        top_customers = rfm.sort_values("monetary", ascending=False).head(n_top).copy()
        top_customers["Revenue"] = top_customers["monetary"].map("{:,.0f}".format)
        top_customers["Recency"] = top_customers["recency"].map("{} days".format)
        top_customers["Frequency"] = top_customers["frequency"].map("{:,}".format)
        top_customers["RFM"] = top_customers["RFM_Code"]
        top_customers["Score"] = top_customers["RFM_Score"]
        st.dataframe(
            top_customers[["customer_id", "segment", "Revenue", "Recency", "Frequency", "RFM", "Score"]]
                 .rename(columns={"customer_id": "Customer ID", "segment": "Segment"}),
            use_container_width=True, hide_index=True, height=420,
        )

