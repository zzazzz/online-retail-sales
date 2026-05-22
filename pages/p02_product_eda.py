"""Product-level EDA page."""

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.charts import INFO_BOX, SEC_TITLE, dark_fig
from utils.constants import PLOTLY_TPL


def abc_class(cumulative_pct: float) -> str:
    if cumulative_pct <= 70:
        return "A"
    if cumulative_pct <= 90:
        return "B"
    return "C"


def country_for_plot(country_name: str, country_map: dict):
    return country_map.get(country_name, country_name)


def render(df_sales, product_after, country, monthly):
    st.markdown("##  Product Revenue Analysis")

    tab1, tab2, tab3, tab4 = st.tabs([
        " Top Products", " Revenue vs Volume",
        " Geographic", " Seasonality"
    ])

    with tab1:
        n_top = st.slider("Top N products", 10, 50, 20, key="prod_n")

        col1, col2 = st.columns(2)
        top_rev = product_after.nlargest(n_top, "revenue")
        top_qty = product_after.nlargest(n_top, "qty")

        with col1:
            top_revenue_fig = px.bar(
                top_rev.sort_values("revenue"),
                x="revenue",
                y="product_label",
                orientation="h",
                color="avg_price",
                color_continuous_scale="Blues",
                template=PLOTLY_TPL,
                labels={"revenue": "Revenue (GBP)", "product_label": "", "avg_price": "Avg Price"},
            )
            top_revenue_fig.update_traces(hovertemplate="<b>%{y}</b><br>GBP %{x:,.0f}<extra></extra>")
            dark_fig(top_revenue_fig, max(400, n_top * 26), f"Top {n_top} by Revenue")
            top_revenue_fig.update_layout(coloraxis_showscale=False, yaxis_tickfont_size=8)
            st.plotly_chart(top_revenue_fig, use_container_width=True)

        with col2:
            top_quantity_fig = px.bar(
                top_qty.sort_values("qty"),
                x="qty",
                y="product_label",
                orientation="h",
                color="qty",
                color_continuous_scale="Greens",
                template=PLOTLY_TPL,
                labels={"qty": "Qty Sold", "product_label": ""},
            )
            top_quantity_fig.update_traces(hovertemplate="<b>%{y}</b><br>%{x:,} units<extra></extra>")
            dark_fig(top_quantity_fig, max(400, n_top * 26), f"Top {n_top} by Quantity")
            top_quantity_fig.update_layout(coloraxis_showscale=False, yaxis_tickfont_size=8)
            st.plotly_chart(top_quantity_fig, use_container_width=True)

        st.markdown(SEC_TITLE("ABC Classification (Pareto)"), unsafe_allow_html=True)
        product_rank = product_after.sort_values("revenue", ascending=False).copy().reset_index(drop=True)
        product_rank["cum_pct"] = product_rank["revenue"].cumsum() / product_rank["revenue"].sum() * 100
        product_rank["ABC"] = product_rank["cum_pct"].apply(abc_class)
        abc_summary = product_rank.groupby("ABC").agg(
            SKUs=("product_id", "count"),
            Revenue=("revenue", "sum"),
        ).reset_index()
        abc_summary["Rev %"] = (abc_summary["Revenue"] / abc_summary["Revenue"].sum() * 100).round(1)
        abc_summary["SKU %"] = (abc_summary["SKUs"] / abc_summary["SKUs"].sum() * 100).round(1)
        col_a, col_b, col_c = st.columns(3)
        colors = {"A": "#22c55e", "B": "#f59e0b", "C": "#ef4444"}
        for col, (_, row) in zip([col_a, col_b, col_c], abc_summary.iterrows()):
            col.markdown(
                f'<div class="metric-card" style="border-color:{colors[row.ABC]};">'
                f'<div class="metric-label">Class {row.ABC}</div>'
                f'<div class="metric-value">{row.SKUs:,} SKUs</div>'
                f'<div class="metric-delta">{row["Rev %"]:.1f}% revenue  {row["SKU %"]:.1f}% catalogue</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown(SEC_TITLE("Full Product Catalogue"), unsafe_allow_html=True)
        srch = st.text_input(" Search", key="prod_srch")
        sort_c = st.selectbox("Sort by", ["revenue", "qty", "orders"], key="prod_sort")
        catalogue = product_after.copy()
        if srch:
            catalogue = catalogue[catalogue["product_description"].str.contains(srch, case=False, na=False)]
        catalogue = catalogue.sort_values(sort_c, ascending=False)
        catalogue_table = catalogue[
            ["product_id", "product_description", "revenue", "qty", "orders", "avg_price"]
        ].copy()
        catalogue_table.columns = ["SKU", "Product", "Revenue ()", "Qty", "Orders", "Avg Price ()"]
        catalogue_table["Revenue ()"] = catalogue_table["Revenue ()"].map("{:,.0f}".format)
        catalogue_table["Qty"] = catalogue_table["Qty"].map("{:,}".format)
        catalogue_table["Orders"] = catalogue_table["Orders"].map("{:,}".format)
        catalogue_table["Avg Price ()"] = catalogue_table["Avg Price ()"].map("{:.2f}".format)
        st.dataframe(catalogue_table.reset_index(drop=True), use_container_width=True, hide_index=True, height=380)

    with tab2:
        n_bub = st.slider("Top N products", 10, 60, 40, key="bub_n")
        top_n = product_after.nlargest(n_bub, "revenue").copy()

        revenue_volume_fig = px.scatter(
            top_n, x="qty", y="revenue",
            size=np.log1p(top_n["revenue"]) * 2.5,
            color="avg_price",
            hover_name="product_description",
            hover_data={"revenue": ":,.0f", "qty": ":,", "avg_price": ":.2f", "orders": ":,"},
            color_continuous_scale="RdYlGn", template=PLOTLY_TPL,
            labels={"qty": "Qty Sold", "revenue": "Revenue (GBP)", "avg_price": "Avg Price"},
        )
        revenue_volume_fig.update_traces(
            hovertemplate="<b>%{hovertext}</b><br>Qty: %{x:,}<br>Revenue: GBP %{y:,.0f}<extra></extra>"
        )
        dark_fig(revenue_volume_fig, 540, f"Revenue vs Quantity  Top {n_bub} Products (colour = avg price)")
        revenue_volume_fig.update_layout(coloraxis_colorbar=dict(title="Avg Price ()", thickness=12))
        st.plotly_chart(revenue_volume_fig, use_container_width=True)
        st.markdown(INFO_BOX(
            "Upper-left = high-value low-volume (premium). Lower-right = high-volume low-margin. "
            "Hover untuk detail produk."
        ), unsafe_allow_html=True)

    with tab3:
        country_map = {
            "EIRE": "Ireland",
            "USA": "United States",
            "RSA": "South Africa",
            "Korea": "South Korea",
            "Czech Republic": "Czechia",
            "Channel Islands": "United Kingdom",
            "West Indies": "Trinidad and Tobago",
            "European Community": None,
            "Unspecified": None,
        }
        country_for_map = country.copy()
        country_for_map["country_plot"] = country_for_map["country"].apply(country_for_plot, args=(country_map,))
        mapped_countries = country_for_map[country_for_map["country_plot"].notna()].copy()

        map_fig = px.choropleth(
            mapped_countries, locations="country_plot", locationmode="country names",
            color="revenue", hover_name="country_plot",
            hover_data={"revenue": ":,.0f", "orders": ":,", "revenue_pct": ":.2f"},
            color_continuous_scale="Plasma", template=PLOTLY_TPL,
            title="Product-Sales Revenue per Country",
        )
        map_fig.update_layout(
            paper_bgcolor="#0f172a", font_color="#dbe7f5", height=460,
            margin=dict(l=0, r=0, t=40, b=0),
            geo=dict(showframe=False, showcoastlines=True, bgcolor="#0f172a",
                     landcolor="#263348", coastlinecolor="#334155",
                     countrycolor="#334155", showocean=True, oceancolor="#0f172a"),
            coloraxis_colorbar=dict(title="Revenue (GBP)", tickfont_size=10),
        )
        st.plotly_chart(map_fig, use_container_width=True)

        excl_uk = st.checkbox("Exclude United Kingdom", value=False, key="geo_excl")
        ca_disp = country[country["country"] != "United Kingdom"] if excl_uk else country

        country_bar_fig = px.bar(
            ca_disp.head(15).sort_values("revenue"),
            x="revenue",
            y="country",
            orientation="h",
            color="revenue_pct",
            color_continuous_scale="Blues",
            template=PLOTLY_TPL,
        )
        country_bar_fig.update_traces(
            hovertemplate="<b>%{y}</b><br>GBP %{x:,.0f} (%{marker.color:.1f}%)<extra></extra>"
        )
        dark_fig(country_bar_fig, 440, "Top Countries  Product-Sales Revenue")
        country_bar_fig.update_layout(coloraxis_showscale=False, yaxis_title="")
        st.plotly_chart(country_bar_fig, use_container_width=True)

    with tab4:
        monthly2 = monthly.copy()
        monthly2["year"] = monthly2["month_year_dt"].dt.year
        monthly2["month_n"] = monthly2["month_year_dt"].dt.month
        pivot = monthly2.pivot(index="year", columns="month_n", values="revenue")
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

        heatmap_fig = go.Figure(go.Heatmap(
            z=pivot.values, x=[month_names[i-1] for i in pivot.columns],
            y=[str(y) for y in pivot.index],
            colorscale="Blues", hoverongaps=False,
            hovertemplate="<b>%{y} %{x}</b><br>GBP %{z:,.0f}<extra></extra>",
        ))
        dark_fig(heatmap_fig, 260, "Revenue Heatmap  Year  Month")
        st.plotly_chart(heatmap_fig, use_container_width=True)

        dow = df_sales.groupby("day_of_week")["revenue"].mean().reindex(
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        ).fillna(0)
        weekday_fig = px.bar(
            x=dow.index,
            y=dow.values,
            template=PLOTLY_TPL,
            labels={"x": "", "y": "Avg Daily Revenue (GBP)"},
        )
        weekday_fig.update_traces(
            marker_color="#3b82f6",
            hovertemplate="<b>%{x}</b><br>GBP %{y:,.0f}<extra></extra>",
        )
        dark_fig(weekday_fig, 300, "Average Daily Revenue by Day of Week")
        st.plotly_chart(weekday_fig, use_container_width=True)

