"""Optuna tuning and target transformation page."""

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from utils.charts import INFO_BOX, SEC_TITLE, dark_fig, kpi
from utils.constants import PLOTLY_TPL


def render(optuna_out):
    st.markdown("##  Optuna Tuning  ExtraTrees (8)")

    scores = optuna_out["scores"]
    best_tr = optuna_out["best_transform"]
    best_params = optuna_out["best_params"]
    trial_df = optuna_out["trial_df"]
    feature_importance = optuna_out["feature_importance"]
    test_df = optuna_out["test_df"]
    y_test = optuna_out["y_test"]
    final_pred = optuna_out["final_pred"]
    transform_results_df = optuna_out.get("transform_results_df")

    st.markdown(INFO_BOX(
        "Optuna men-tuning ExtraTrees <b>sekaligus memilih target transform</b> "
        "(raw / log1p / sqrt / cuberoot / fourthroot). Semua metrik dihitung di skala revenue asli "
        "setelah inverse transform  perbandingan fair secara bisnis."
    ), unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(kpi("Best Transform", best_tr.upper()), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi("R2 Test", f"{scores['R2 Test']:.4f}"), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi("MAE Non-Zero", f"{scores['MAE Non-Zero']:,.0f}"), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi("sMAPE", f"{scores['sMAPE']:.2f}%"), unsafe_allow_html=True)
    with c5:
        st.markdown(kpi("Zero Accuracy", f"{scores['Zero Accuracy']:.1f}%"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        " Actual vs Predicted", " Feature Importance",
        " Optuna Trials", " Best Hyperparameters", " Target Transform"
    ])

    with tab1:
        diagnostics_fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=("Actual vs Predicted  Optuna Tuned ExtraTrees", "Residuals"),
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
            y=final_pred,
            mode="lines",
            name="Predicted",
            line=dict(color="#dc2626", width=1.5, dash="dot"),
        ), row=1, col=1)
        residuals = y_test - final_pred
        diagnostics_fig.add_trace(go.Scatter(
            x=test_df["ds"],
            y=residuals,
            mode="lines",
            name="Residual",
            line=dict(color="#2563eb", width=1),
        ), row=2, col=1)
        diagnostics_fig.add_hline(y=0, line_dash="dash", line_color="#6b7280", row=2, col=1)
        diagnostics_fig.update_layout(
            paper_bgcolor="#0f172a",
            plot_bgcolor="#111827",
            font_family="DM Sans",
            font_color="#dbe7f5",
            height=680,
            margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            title=dict(
                text="Optuna Tuned ExtraTrees  Test Set",
                font_size=14,
                x=0.01,
                font_color="#f8fafc",
            ),
        )
        diagnostics_fig.update_xaxes(showgrid=True, gridcolor="#263348")
        diagnostics_fig.update_yaxes(showgrid=True, gridcolor="#263348")
        st.plotly_chart(diagnostics_fig, use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            residual_hist = px.histogram(
                x=residuals,
                nbins=50,
                template=PLOTLY_TPL,
                labels={"x": "Residual (GBP)", "y": "Count"},
            )
            residual_hist.update_traces(marker_color="#3b82f6", opacity=0.8)
            dark_fig(residual_hist, 280, "Residual Distribution (Test Set)")
            st.plotly_chart(residual_hist, use_container_width=True)
        with col_b:
            open_day_mask = y_test > 0
            scatter_fig = px.scatter(
                x=y_test[open_day_mask],
                y=final_pred[open_day_mask],
                template=PLOTLY_TPL,
                labels={"x": "Actual (GBP)", "y": "Predicted (GBP)"},
                opacity=0.5,
            )
            scatter_fig.update_traces(marker=dict(color="#22c55e", size=4))
            max_axis = max(y_test[open_day_mask].max(), final_pred[open_day_mask].max())
            scatter_fig.add_trace(go.Scatter(
                x=[0, max_axis],
                y=[0, max_axis],
                mode="lines",
                line=dict(color="#f59e0b", dash="dash", width=1),
                name="Perfect fit",
            ))
            dark_fig(scatter_fig, 280, "Actual vs Predicted Scatter (open days)")
            st.plotly_chart(scatter_fig, use_container_width=True)

    with tab2:
        importance_df = feature_importance.reset_index()
        importance_df.columns = ["Feature", "Importance"]
        importance_fig = px.bar(
            importance_df.head(20),
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="Teal",
            template=PLOTLY_TPL,
        )
        importance_fig.update_traces(hovertemplate="%{y}: %{x:.4f}<extra></extra>")
        dark_fig(importance_fig, 480, "Feature Importance  Optuna ExtraTrees (Top 20)")
        importance_fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(importance_fig, use_container_width=True)
        st.markdown(INFO_BOX(
            "Feature importance mengikuti notebook: forced-zero flag, kalender, cyclical seasonality, "
            "Q4 indicator, time index, dan jarak ke hari tutup. Tidak ada lag/rolling feature pada versi ini."
        ), unsafe_allow_html=True)

    with tab3:
        if trial_df is not None and len(trial_df) > 0:
            trial_df = trial_df.sort_values("trial").copy()
            trial_df["best_so_far"] = trial_df["value"].cummax()
            trial_fig = go.Figure()
            trial_fig.add_trace(go.Scatter(
                x=trial_df["trial"],
                y=trial_df["value"],
                mode="markers",
                name="Trial R2",
                marker=dict(color="#3b82f6", size=5, opacity=0.6),
                hovertemplate="Trial %{x}<br>R2: %{y:.4f}<extra></extra>",
            ))
            trial_fig.add_trace(go.Scatter(
                x=trial_df["trial"],
                y=trial_df["best_so_far"],
                mode="lines",
                name="Best so far",
                line=dict(color="#f59e0b", width=2),
            ))
            dark_fig(trial_fig, 360, "Optuna Convergence - R2 Test")
            trial_fig.update_yaxes(title="R2 Test")
            trial_fig.update_xaxes(title="Trial #")
            st.plotly_chart(trial_fig, use_container_width=True)

            if "target_transform" in trial_df.columns:
                transform_counts = trial_df["target_transform"].value_counts().reset_index()
                transform_counts.columns = ["Transform", "Trials"]
                transform_fig = px.bar(
                    transform_counts,
                    x="Transform",
                    y="Trials",
                    color="Transform",
                    template=PLOTLY_TPL,
                    title="Target Transform Frequency in Optuna Trials",
                )
                dark_fig(transform_fig, 300)
                transform_fig.update_layout(showlegend=False)
                st.plotly_chart(transform_fig, use_container_width=True)
        else:
            st.info("Trial history tidak tersedia.")

    with tab4:
        st.markdown(SEC_TITLE("Best Hyperparameters Found by Optuna"), unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-card" style="text-align:left;margin-bottom:1rem">
            <div class="metric-label">Target Transform</div>
            <div class="metric-value" style="font-size:2rem">{best_tr.upper()}</div>
        </div>
        """, unsafe_allow_html=True)

        for k, v in best_params.items():
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;'
                f'padding:6px 0;border-bottom:1px solid #243149;">'
                f'<code style="color:#a7b3c7">{k}</code>'
                f'<span style="color:#34d399;font-weight:600">{v}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(SEC_TITLE("Final Model Scores"), unsafe_allow_html=True)
        rows = [
            (
                metric,
                f"{value:,.0f}" if "MAE" in metric or "RMSE" in metric else
                f"{value:.4f}" if "R2" in metric else
                f"{value:.2f}%" if "MAPE" in metric or "Accuracy" in metric else
                str(value),
            )
            for metric, value in scores.items()
        ]
        html_rows = "".join(
            f"<tr><td>{metric}</td><td style='text-align:right;font-family:DM Mono'>{value}</td></tr>"
            for metric, value in rows
        )
        st.markdown(f"""
        <table class="custom-table">
            <thead><tr><th>Metric</th><th>Value</th></tr></thead>
            <tbody>{html_rows}</tbody>
        </table>""", unsafe_allow_html=True)

    with tab5:
        if transform_results_df is not None and len(transform_results_df) > 0:
            sorted_transform_results = transform_results_df.sort_values("R2 Test")
            sorted_transform_results["r2_label"] = sorted_transform_results["R2 Test"].map("{:.4f}".format)
            transform_compare_fig = px.bar(
                sorted_transform_results,
                x="R2 Test",
                y="Target Transform",
                orientation="h",
                text="r2_label",
                title="Target Transformation Comparison - ExtraTrees Baseline",
                template=PLOTLY_TPL,
            )
            transform_compare_fig.update_traces(textposition="outside")
            dark_fig(transform_compare_fig, 420)
            st.plotly_chart(transform_compare_fig, use_container_width=True)

            transform_table = transform_results_df.copy()
            for col in ["R2 Test", "sMAPE", "Zero Accuracy"]:
                if col == "R2 Test":
                    transform_table[col] = transform_table[col].map("{:.4f}".format)
                else:
                    transform_table[col] = transform_table[col].map("{:.2f}%".format)
            for col in ["MAE All", "MAE Non-Zero", "MAE Zero", "RMSE"]:
                transform_table[col] = transform_table[col].map("{:,.0f}".format)
            st.dataframe(transform_table, use_container_width=True, hide_index=True)
        else:
            st.info("Target transform comparison tidak tersedia.")

