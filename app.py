"""
Streamlit dashboard for the Online Retail II revenue analysis.
"""

import warnings
warnings.filterwarnings("ignore")

import tempfile
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = APP_DIR / "retail_transaction_data.csv"

st.set_page_config(
    page_title="Retail Analytics  Optuna ExtraTrees",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=DM+Mono:wght@400;500;600&display=swap');

:root {
    --bg: #070b14;
    --panel: #0f172a;
    --panel-2: #111c31;
    --ink: #e8eef8;
    --muted: #a7b3c7;
    --line: #243149;
    --soft-line: #1a2436;
    --blue: #60a5fa;
    --green: #34d399;
    --amber: #fbbf24;
    --red: #fb7185;
    --teal: #2dd4bf;
}

html, body, [class*="css"] {
    font-family: "Inter", "Segoe UI", sans-serif;
    color: var(--ink);
}
.stApp { background: var(--bg); }
[data-testid="stHeader"] { background: #070b14; }
.main .block-container {
    max-width: 1480px;
    padding: 1.1rem 1.8rem 3rem;
}
[data-testid="stSidebar"] {
    background: #0b1220;
    border-right: 1px solid var(--line);
    box-shadow: 8px 0 26px rgba(0, 0, 0, .25);
}
[data-testid="stSidebarNav"] { display: none; }
[data-testid="stSidebar"] * { color: var(--ink) !important; }
[data-testid="stSidebar"] hr { border-color: var(--soft-line); }
[data-testid="stSidebar"] label, [data-testid="stSidebar"] p { font-size: .86rem; }

h1, h2, h3 { color: var(--ink); letter-spacing: 0; }

.hero-panel {
    background: linear-gradient(135deg, #0f172a 0%, #10213a 62%, #0d2b2a 100%);
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 1.25rem 1.35rem;
    margin: .2rem 0 1rem;
    box-shadow: 0 18px 44px rgba(0, 0, 0, .24);
}
.hero-eyebrow {
    font-size: .72rem;
    text-transform: uppercase;
    letter-spacing: .08em;
    color: var(--teal);
    font-weight: 800;
    margin-bottom: .35rem;
}
.hero-title {
    margin: 0;
    color: #f8fafc;
    font-size: clamp(1.45rem, 2.2vw, 2.25rem);
    line-height: 1.15;
    font-weight: 800;
}
.hero-copy {
    max-width: 980px;
    margin-top: .6rem;
    color: var(--muted);
    font-size: .94rem;
    line-height: 1.65;
}

.metric-card {
    background: linear-gradient(180deg, #111c31 0%, #0f172a 100%);
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: .9rem 1rem;
    text-align: left;
    margin-bottom: .55rem;
    min-height: 88px;
    box-shadow: 0 10px 26px rgba(0, 0, 0, .22);
}
.metric-label {
    font-size: .67rem;
    font-weight: 800;
    letter-spacing: .075em;
    text-transform: uppercase;
    color: #93a4bc;
    margin-bottom: .42rem;
}
.metric-value {
    font-family: "DM Mono", Consolas, monospace;
    font-size: clamp(1.08rem, 1.28vw, 1.48rem);
    font-weight: 700;
    color: var(--ink);
    line-height: 1.15;
    overflow-wrap: anywhere;
}
.metric-delta { font-size: .74rem; color: var(--green); margin-top: .35rem; font-weight: 700; }
.metric-delta.neg { color: var(--red); }

.section-title {
    font-size: .84rem;
    font-weight: 850;
    color: #dbeafe;
    margin: 1.35rem 0 .72rem;
    padding-bottom: .48rem;
    border-bottom: 1px solid var(--line);
}
.info-box, .warn-box {
    background: var(--panel);
    border: 1px solid var(--line);
    border-left: 4px solid var(--blue);
    border-radius: 8px;
    padding: .9rem 1rem;
    margin: .65rem 0;
    font-size: .86rem;
    color: #cbd5e1;
    line-height: 1.65;
    box-shadow: 0 8px 22px rgba(0, 0, 0, .18);
}
.warn-box { border-left-color: var(--amber); background: #221a0b; color: #fde68a; }

.stTabs [data-baseweb="tab-list"] {
    gap: .35rem;
    background: transparent;
    border-bottom: 1px solid var(--line);
}
.stTabs [data-baseweb="tab"] {
    background: #111c31;
    border: 1px solid var(--line);
    border-bottom: 0;
    border-radius: 8px 8px 0 0;
    color: #cbd5e1;
    font-weight: 750;
    font-size: .86rem;
    padding: .55rem 1rem;
}
.stTabs [aria-selected="true"] {
    background: #17345b !important;
    color: #dbeafe !important;
    border-color: #60a5fa !important;
}

[data-testid="stDataFrame"], [data-testid="stTable"], div[data-testid="stPlotlyChart"] {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 12px 28px rgba(0, 0, 0, .24);
}
div[data-testid="stPlotlyChart"] { padding: .55rem; }
.custom-table { font-family: "DM Mono", Consolas, monospace; font-size: .76rem; width: 100%; border-collapse: collapse; }
.custom-table th { background: #17223a; color: #9fb0c9; padding: .45rem .7rem; text-align: left; font-weight: 700; text-transform: uppercase; font-size: .66rem; }
.custom-table td { padding: .38rem .7rem; border-bottom: 1px solid var(--soft-line); color: #d7e1ef; }
.custom-table tr:hover td { background: #152035; }

[data-testid="stFileUploaderDropzone"] {
    background: #0f172a !important;
    border: 1px dashed #334155 !important;
}
[data-testid="stFileUploaderDropzone"] * { color: #cbd5e1 !important; }
.stButton button, .stDownloadButton button,
[data-testid="stFileUploaderDropzone"] button {
    background: #17223a !important;
    color: #e8eef8 !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
}
[data-baseweb="slider"] [role="slider"] { background-color: #60a5fa !important; }

#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("""
    <div style="padding:.45rem 0 1rem">
        <div style="font-size:1.32rem;font-weight:850;color:#f8fafc">Retail Analytics</div>
        <div style="font-size:.76rem;color:#a7b3c7;margin-top:.25rem;line-height:1.45">
            Product revenue forecasting, Optuna ExtraTrees, and RFM segmentation
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown("### Dataset")
    if DEFAULT_DATA_PATH.exists():
        st.caption(f"Default aktif: {DEFAULT_DATA_PATH.name}")
    uploaded = st.file_uploader("Upload CSV alternatif", type=["csv"])
    if uploaded:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(uploaded.getvalue())
            st.session_state["data_path"] = tmp.name
        st.success(f"Menggunakan {uploaded.name}")

    DATA_PATH = st.session_state.get("data_path")
    if DATA_PATH is None and DEFAULT_DATA_PATH.exists():
        DATA_PATH = str(DEFAULT_DATA_PATH)

    st.divider()
    st.markdown("### Model Settings")
    n_trials = st.slider(
        "Optuna Trials", 10, 120, 120, step=10,
        help="Notebook memakai 120 trials. Turunkan hanya kalau ingin preview cepat."
    )
    horizon = st.slider("Forecast Horizon (hari)", 30, 365, 180, step=30)

    st.divider()
    st.markdown("### Navigation")
    PAGES = {
        "Overview": "overview",
        "Data & Cleaning": "data",
        "Product EDA": "product",
        "Model Selection": "models",
        "Optuna Tuning": "optuna",
        "Forecast": "forecast",
        "RFM Segmentation": "rfm",
        "Refund Analysis": "refund",
        "Business Insights": "insights",
    }
    page_label = st.radio("Dashboard page", list(PAGES.keys()), label_visibility="collapsed")
    active = PAGES[page_label]

    st.divider()
    st.markdown("""
    <div style="font-size:.72rem;color:#a7b3c7;line-height:1.7">
        <b style="color:#e8eef8">Reference:</b> Retail Analytics HM Sampoerna notebook<br>
        <b style="color:#e8eef8">Scope:</b> product-sales revenue only<br>
        <b style="color:#e8eef8">Model:</b> Optuna ExtraTrees
    </div>
    """, unsafe_allow_html=True)

if DATA_PATH is None:
    st.markdown("""
    <div class="hero-panel" style="max-width:860px;margin:3rem auto">
        <div class="hero-eyebrow">Dataset required</div>
        <h1 class="hero-title">Online Retail II Product Revenue Analytics</h1>
        <div class="hero-copy">
            Upload CSV di sidebar untuk memulai dashboard end-to-end sesuai notebook:
            product-based cleansing, daily revenue forecasting, Optuna ExtraTrees,
            refund review, dan RFM customer segmentation.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

loading_slot = st.empty()
loading_slot.markdown("""
<div class="hero-panel" style="margin-top:.2rem">
    <div class="hero-eyebrow">Preparing analytics</div>
    <h1 class="hero-title">Loading product revenue dashboard...</h1>
    <div class="hero-copy">
        Dataset sedang dibersihkan dan agregasi utama disiapkan. Model dan forecast dihitung
        hanya saat halaman yang membutuhkannya dibuka agar navigasi harian tetap ringan.
    </div>
</div>
""", unsafe_allow_html=True)
from utils.data import load_raw, clean_retail_data_product_revenue, compute_aggregations, build_daily_series
from utils.models import tree_model_selection, optuna_tune_extratrees, compute_rfm, build_forecast
from utils.charts import kpi

df_raw = load_raw(DATA_PATH)
df_sales, df_customer, df_cancelled = clean_retail_data_product_revenue(df_raw)
monthly, compare, product_after, country, cust = compute_aggregations(df_sales, df_cancelled, df_customer)


@st.cache_data(show_spinner=" Building forecast")
def get_forecast(_optuna_out, _ts, _horizon):
    return build_forecast(_optuna_out, _ts, _horizon)


def get_timeseries():
    return build_daily_series(df_sales)


def get_model_selection(_ts):
    return tree_model_selection(_ts)


def get_optuna(_ts):
    return optuna_tune_extratrees(_ts, n_trials=n_trials)


def get_segments():
    return compute_rfm(df_customer)


needs_timeseries = active in {"overview", "models", "optuna", "forecast", "insights"}
needs_model_selection = active == "models"
needs_optuna = active in {"overview", "optuna", "forecast", "insights"}
needs_rfm = active in {"overview", "rfm", "insights"}
needs_forecast = active in {"overview", "forecast"}

ts = get_timeseries() if needs_timeseries else None
model_selection_out = get_model_selection(ts) if needs_model_selection else None
optuna_out = get_optuna(ts) if needs_optuna else None
rfm = get_segments() if needs_rfm else None
future_df = get_forecast(optuna_out, ts, horizon) if needs_forecast else None

loading_slot.empty()

if active == "overview":
    st.markdown("""
    <div class="hero-panel">
        <div class="hero-eyebrow">Notebook-aligned executive dashboard</div>
        <h1 class="hero-title">Online Retail II Product Revenue Analytics</h1>
        <div class="hero-copy">
            Dashboard ini mengikuti alur notebook referensi: product-sales cleansing tanpa
            percentile outlier filter, EDA produk, model selection, Optuna ExtraTrees,
            forecast revenue harian, refund monitoring, dan RFM segmentation.
        </div>
    </div>
    """, unsafe_allow_html=True)

    scores = optuna_out["scores"]
    best_tr = optuna_out["best_transform"]
    total_rev = df_sales["revenue"].sum()
    total_ref = abs(df_cancelled["revenue"].sum())

    import plotly.express as px
    import plotly.graph_objects as go

    from utils.charts import SEC_TITLE, dark_fig
    from utils.constants import PLOTLY_TPL, SEG_COLORS

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.markdown(kpi("Gross Revenue", f"{total_rev/1e6:.2f}M"), unsafe_allow_html=True)
    with c2:
        net_revenue = total_rev - total_ref
        st.markdown(kpi("Net Revenue", f"{net_revenue/1e6:.2f}M"), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi("Unique SKUs", f"{df_sales['product_id'].nunique():,}"), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi("Customers", f"{df_customer['customer_id'].nunique():,}"), unsafe_allow_html=True)
    with c5:
        st.markdown(kpi("Model R2", f"{scores['R2 Test']:.4f}"), unsafe_allow_html=True)
    with c6:
        st.markdown(kpi("Best Transform", best_tr.upper()), unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(SEC_TITLE("Monthly Product-Sales Revenue"), unsafe_allow_html=True)
        revenue_fig = go.Figure()
        revenue_fig.add_trace(go.Bar(
            x=monthly["month_year_dt"],
            y=monthly["revenue"],
            name="Revenue",
            marker_color="#3b82f6",
            opacity=0.75,
            hovertemplate="<b>%{x|%b %Y}</b><br>%{y:,.0f}<extra></extra>",
        ))
        revenue_fig.add_trace(go.Scatter(
            x=monthly["month_year_dt"],
            y=monthly["revenue_ma3"],
            name="3M MA",
            line=dict(color="#f59e0b", width=2),
        ))
        dark_fig(revenue_fig, 340)
        st.plotly_chart(revenue_fig, use_container_width=True)

    with col_b:
        st.markdown(SEC_TITLE("RFM Segment Distribution"), unsafe_allow_html=True)
        segment_counts = rfm.groupby("segment")["customer_id"].count().reset_index()
        segment_counts.columns = ["segment", "count"]
        segment_fig = px.pie(
            segment_counts,
            names="segment",
            values="count",
            hole=0.48,
            color="segment",
            color_discrete_map=SEG_COLORS,
            template=PLOTLY_TPL,
        )
        segment_fig.update_traces(
            hovertemplate="<b>%{label}</b><br>%{value} customers (%{percent})<extra></extra>"
        )
        dark_fig(segment_fig, 340)
        st.plotly_chart(segment_fig, use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown(SEC_TITLE("Forecast vs Historical (Recent Tail)"), unsafe_allow_html=True)
        hist_tail = ts[ts["ds"] >= "2011-06-01"]
        forecast_fig = go.Figure()
        forecast_fig.add_trace(go.Scatter(
            x=hist_tail["ds"],
            y=hist_tail["revenue"],
            mode="lines",
            name="Historis",
            line=dict(color="#7a8aa0", width=1.2),
        ))
        forecast_fig.add_trace(go.Scatter(
            x=future_df["ds"],
            y=future_df["fc_revenue"],
            mode="lines",
            name="Forecast",
            line=dict(color="#3b82f6", width=2, dash="dash"),
        ))
        dark_fig(forecast_fig, 300)
        st.plotly_chart(forecast_fig, use_container_width=True)

    with col_d:
        st.markdown(SEC_TITLE("Top 10 Countries - Revenue"), unsafe_allow_html=True)
        top_countries = country.head(10)
        country_fig = px.bar(
            top_countries.sort_values("revenue"),
            x="revenue",
            y="country",
            orientation="h",
            color="revenue",
            color_continuous_scale="Blues",
            template=PLOTLY_TPL,
        )
        country_fig.update_traces(hovertemplate="<b>%{y}</b><br>%{x:,.0f}<extra></extra>")
        dark_fig(country_fig, 300)
        country_fig.update_layout(coloraxis_showscale=False, yaxis_title="")
        st.plotly_chart(country_fig, use_container_width=True)

elif active == "data":
    from pages.p01_data_overview import render
    render(df_raw, df_sales, df_customer, df_cancelled, monthly, product_after, country)

elif active == "product":
    from pages.p02_product_eda import render
    render(df_sales, product_after, country, monthly)

elif active == "models":
    from pages.p03_model_selection import render
    model_res, preds_dict, train_df, test_df, y_test = model_selection_out
    render(model_res, preds_dict, train_df, test_df, y_test)

elif active == "optuna":
    from pages.p04_optuna import render
    render(optuna_out)

elif active == "forecast":
    from pages.p05_forecast import render
    render(ts, optuna_out, future_df, horizon)

elif active == "rfm":
    from pages.p06_rfm import render
    render(rfm)

elif active == "refund":
    from pages.p07_refund import render
    render(df_cancelled, compare)

elif active == "insights":
    from pages.p08_insights import render
    render(df_sales, compare, optuna_out, rfm, product_after, country)


