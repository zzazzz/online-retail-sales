"""Data overview and cleaning page."""

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from utils.charts import INFO_BOX, SEC_TITLE, WARN_BOX, dark_fig, kpi
from utils.data import line_type_summary
from utils.constants import NON_PRODUCT_CODES, PLOTLY_TPL


def render(df_raw, df_sales, df_customer, df_cancelled, monthly, product_after, country):
    st.markdown("##  Data Overview & Cleaning")

    st.markdown(SEC_TITLE("2 Raw Dataset Summary"), unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(kpi("Raw Rows", f"{len(df_raw):,}"), unsafe_allow_html=True)
    with c2:
        date_range = (
            f"<span style='font-size:1.02rem;line-height:1.35'>"
            f"{df_raw['order_date'].min().date()}<br>{df_raw['order_date'].max().date()}"
            f"</span>"
        )
        st.markdown(kpi("Date Range", date_range), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi("Unique Products", f"{df_raw['product_id'].nunique():,}"), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi("Unique Customers", f"{df_raw['customer_id'].nunique():,}"), unsafe_allow_html=True)

    st.markdown(INFO_BOX(
        "Tidak memakai outlier filtering percentile. Transaksi produk bernilai besar tetap dipertahankan "
        "selama baris tersebut merupakan <b>product-sales</b> yang valid. Yang dipisahkan hanya baris "
        "non-product/admin seperti POSTAGE, MANUAL, DISCOUNT, FEE, BAD DEBT."
    ), unsafe_allow_html=True)

    st.markdown(SEC_TITLE("4 EDA Before Cleaning: Product vs Non-product Lines"), unsafe_allow_html=True)

    with st.spinner("Computing pre-cleaning line type summary"):
        line_type_totals = line_type_summary(df_raw)

    col1, col2 = st.columns(2, gap="large")
    with col1:
        line_share_fig = px.pie(
            line_type_totals,
            names="line_type",
            values="revenue",
            hole=0.45,
            template=PLOTLY_TPL,
        )
        line_share_fig.update_traces(
            textinfo="label+percent",
            textposition="inside",
            insidetextorientation="radial",
            hovertemplate="%{label}<br>Revenue: GBP %{value:,.0f}<extra></extra>",
        )
        dark_fig(line_share_fig, 430, "Revenue Share")
        line_share_fig.update_layout(showlegend=False)
        st.plotly_chart(line_share_fig, use_container_width=True)

    with col2:
        line_type_sorted = line_type_totals.sort_values("revenue")
        line_type_sorted["revenue_label"] = line_type_sorted["revenue"].map("GBP {:,.0f}".format)
        line_revenue_fig = px.bar(
            line_type_sorted,
            x="revenue",
            y="line_type",
            orientation="h",
            template=PLOTLY_TPL,
            text="revenue_label",
        )
        line_revenue_fig.update_traces(
            marker_color="#3b82f6",
            textposition="auto",
            cliponaxis=False,
        )
        dark_fig(line_revenue_fig, 430, "Revenue by Line Type")
        line_revenue_fig.update_xaxes(range=[0, line_type_sorted["revenue"].max() * 1.12])
        line_revenue_fig.update_layout(yaxis_title="", xaxis_title="Revenue (GBP)")
        st.plotly_chart(line_revenue_fig, use_container_width=True)

    st.markdown(SEC_TITLE("Non-product/Admin Codes Excluded"), unsafe_allow_html=True)
    st.markdown(WARN_BOX(
        f"Codes excluded: <code>{', '.join(sorted(NON_PRODUCT_CODES))}</code><br>"
        "These include POSTAGE (POST), DOTCOM POSTAGE (DOT), Manual (M/m), Discount (D), "
        "Amazon Fee (AMAZONFEE), Bank Charges, CRUK, C2 (carriage)."
    ), unsafe_allow_html=True)

    st.markdown(SEC_TITLE("3 After Product-Based Cleaning"), unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    total_rev = df_sales["revenue"].sum()
    total_ref = abs(df_cancelled["revenue"].sum())
    with c1:
        st.markdown(kpi("Product Sales Rows", f"{len(df_sales):,}"), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi("Unique SKUs", f"{df_sales['product_id'].nunique():,}"), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi("Gross Revenue", f"{total_rev/1e6:.2f}M"), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi("Total Refunds", f"{total_ref/1e3:.0f}K", neg=True), unsafe_allow_html=True)
    with c5:
        st.markdown(kpi("Customers w/ ID", f"{df_customer['customer_id'].nunique():,}"), unsafe_allow_html=True)

    st.markdown(SEC_TITLE("4 Monthly Product-Sales Revenue (After Cleaning)"), unsafe_allow_html=True)

    monthly_fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=("Monthly Product Revenue + 3M MA", "Monthly Orders"),
        vertical_spacing=0.10,
    )
    monthly_fig.add_trace(go.Bar(
        x=monthly["month_year_dt"],
        y=monthly["revenue"],
        name="Revenue",
        marker_color="#3b82f6",
        opacity=0.75,
        hovertemplate="<b>%{x|%b %Y}</b><br>GBP %{y:,.0f}<extra></extra>",
    ), row=1, col=1)
    monthly_fig.add_trace(go.Scatter(
        x=monthly["month_year_dt"],
        y=monthly["revenue_ma3"],
        name="3M MA",
        line=dict(color="#f59e0b", width=2),
        hovertemplate="<b>%{x|%b %Y}</b><br>MA3: GBP %{y:,.0f}<extra></extra>",
    ), row=1, col=1)
    monthly_fig.add_trace(go.Scatter(
        x=monthly["month_year_dt"],
        y=monthly["orders"],
        name="Orders",
        line=dict(color="#8b5cf6", width=2),
        hovertemplate="<b>%{x|%b %Y}</b><br>%{y:,} orders<extra></extra>",
    ), row=2, col=1)
    monthly_fig.update_layout(
        paper_bgcolor="#0f172a",
        plot_bgcolor="#111827",
        font_family="DM Sans",
        font_color="#dbe7f5",
        height=520,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    monthly_fig.update_xaxes(showgrid=True, gridcolor="#263348")
    monthly_fig.update_yaxes(showgrid=True, gridcolor="#263348")
    st.plotly_chart(monthly_fig, use_container_width=True)

    st.markdown(INFO_BOX(
        "Tren revenue meningkat dari 20092011. <b>Seasonality Q4 sangat kuat</b>  "
        "Oktober & November menjadi peak revenue tiap tahun (Christmas gifting demand). "
        "JanuariFeb selalu turun tajam pasca Q4."
    ), unsafe_allow_html=True)

    st.markdown(SEC_TITLE("4 Top 20 Products by Revenue (After Cleaning)"), unsafe_allow_html=True)
    top20 = product_after.head(20).copy()
    top_products_fig = px.bar(
        top20.sort_values("revenue"),
        x="revenue",
        y="product_label",
        orientation="h",
        color="avg_price",
        color_continuous_scale="Viridis",
        template=PLOTLY_TPL,
        labels={"revenue": "Revenue (GBP)", "product_label": "", "avg_price": "Avg Price"},
    )
    top_products_fig.update_traces(hovertemplate="<b>%{y}</b><br>GBP %{x:,.0f}<extra></extra>")
    dark_fig(top_products_fig, 560, "Top 20 Products  Product-Sales Revenue")
    top_products_fig.update_layout(
        coloraxis_colorbar=dict(title="Avg Price ()", thickness=12),
        yaxis_tickfont_size=9,
    )
    st.plotly_chart(top_products_fig, use_container_width=True)

