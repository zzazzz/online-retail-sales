"""Tree model comparison page."""

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from utils.charts import INFO_BOX, SEC_TITLE, dark_fig, kpi
from utils.constants import PLOTLY_TPL


def render(model_results, preds_dict, train_df, test_df, y_test):
    st.markdown("##  Tree-Based Model Selection (7)")

    st.markdown(INFO_BOX(
        "Train = 80% awal tanggal historis  Test = 20% akhir. "
        "Bagian ini mengikuti notebook: model selection memakai target revenue asli tanpa transformasi. "
        "Model yang dibandingkan: ExtraTrees, RandomForest, XGBoost, LightGBM, CatBoost, dan HistGradientBoosting."
    ), unsafe_allow_html=True)

    st.markdown(SEC_TITLE("Model Comparison - R2 Test"), unsafe_allow_html=True)

    sorted_results = model_results.sort_values("R2 Test")
    sorted_results["r2_label"] = sorted_results["R2 Test"].map("{:.4f}".format)
    score_fig = px.bar(
        sorted_results,
        x="R2 Test",
        y="Model",
        orientation="h",
        template=PLOTLY_TPL,
        text="r2_label",
        color="R2 Test",
        color_continuous_scale="Teal",
        title="Tree-Based Model Selection - R2 Test",
    )
    score_fig.update_traces(textposition="outside")
    xmin = model_results["R2 Test"].min() - 0.03
    xmax = min(1.0, model_results["R2 Test"].max() + 0.03)
    dark_fig(score_fig, 360, "Tree-Based Model Selection - R2 Test")
    score_fig.update_layout(xaxis_range=[xmin, xmax], coloraxis_showscale=False)
    st.plotly_chart(score_fig, use_container_width=True)

    st.markdown(SEC_TITLE("Full Metrics Table"), unsafe_allow_html=True)
    metrics_table = model_results[
        ["Model", "R2 Test", "MAE All", "MAE Non-Zero", "RMSE", "sMAPE", "Zero Accuracy"]
    ].copy()
    metrics_table["MAE All"] = metrics_table["MAE All"].map("{:,.0f}".format)
    metrics_table["MAE Non-Zero"] = metrics_table["MAE Non-Zero"].map("{:,.0f}".format)
    metrics_table["RMSE"] = metrics_table["RMSE"].map("{:,.0f}".format)
    metrics_table["sMAPE"] = metrics_table["sMAPE"].map("{:.2f}%".format)
    metrics_table["Zero Accuracy"] = metrics_table["Zero Accuracy"].map("{:.1f}%".format)
    metrics_table["R2 Test"] = metrics_table["R2 Test"].map("{:.4f}".format)
    st.dataframe(metrics_table, use_container_width=True, hide_index=True)

    best_name = model_results.iloc[0]["Model"]
    best_prediction = preds_dict[best_name]

    st.markdown(SEC_TITLE(f"Best Model: {best_name}  Actual vs Predicted"), unsafe_allow_html=True)

    diagnostics_fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(f"Actual vs Predicted  {best_name}", "Residuals"),
    )
    diagnostics_fig.add_trace(go.Scatter(
        x=test_df["ds"],
        y=y_test,
        mode="lines",
        name="Actual",
        line=dict(color="#16a34a", width=1.5),
    ), row=1, col=1)
    diagnostics_fig.add_trace(go.Scatter(
        x=test_df["ds"],
        y=best_prediction,
        mode="lines",
        name="Predicted",
        line=dict(color="#dc2626", width=1.5, dash="dot"),
    ), row=1, col=1)
    residuals = y_test - best_prediction
    diagnostics_fig.add_trace(go.Scatter(
        x=test_df["ds"],
        y=residuals,
        mode="lines",
        name="Residual",
        line=dict(color="#2563eb", width=1),
    ), row=2, col=1)
    diagnostics_fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

    diagnostics_fig.update_layout(
        paper_bgcolor="#0f172a",
        plot_bgcolor="#111827",
        font_family="DM Sans",
        font_color="#dbe7f5",
        height=640,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    diagnostics_fig.update_xaxes(showgrid=True, gridcolor="#263348")
    diagnostics_fig.update_yaxes(showgrid=True, gridcolor="#263348")
    st.plotly_chart(diagnostics_fig, use_container_width=True)

    st.markdown(SEC_TITLE("KPI Scorecard"), unsafe_allow_html=True)
    model_count = len(model_results)
    cols = st.columns(model_count)
    for col, (_, row) in zip(cols, model_results.iterrows()):
        with col:
            st.markdown(f"**{row['Model']}**")
            st.markdown(kpi("R2 Test", f"{row['R2 Test']:.4f}"), unsafe_allow_html=True)
            st.markdown(kpi("MAE Non-Zero", f"{row['MAE Non-Zero']:,.0f}"), unsafe_allow_html=True)
            st.markdown(kpi("sMAPE", f"{row['sMAPE']:.1f}%"), unsafe_allow_html=True)
            st.markdown(kpi("Zero Acc", f"{row['Zero Accuracy']:.0f}%"), unsafe_allow_html=True)

