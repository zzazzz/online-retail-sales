"""Small chart and HTML helpers used across Streamlit pages."""

import plotly.graph_objects as go
from utils.constants import GRID_COL, PAPER_BG, PLOT_BG


def dark_fig(fig: go.Figure, height: int = 450, title: str = None) -> go.Figure:
    up = dict(
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font_family="Inter, DM Sans, Segoe UI, sans-serif",
        font_color="#dbe7f5",
        height=height,
        margin=dict(l=14, r=14, t=64 if title else 36, b=34),
        legend=dict(
            bgcolor="rgba(15,23,42,0)",
            borderwidth=0,
            font_size=11,
            font=dict(color="#dbe7f5"),
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor=GRID_COL,
            zeroline=False,
            linecolor="#334155",
            ticks="",
            tickfont=dict(color="#b8c5d8"), title_font=dict(color="#dbe7f5"),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=GRID_COL,
            zeroline=False,
            linecolor="#334155",
            ticks="",
            tickfont=dict(color="#b8c5d8"), title_font=dict(color="#dbe7f5"),
        ),
    )
    if title:
        up["title"] = dict(text=title, font_size=15, font_color="#f8fafc", x=0.01, xanchor="left")
    fig.update_layout(**up)
    return fig


def kpi(label: str, value: str, delta: str = None, neg: bool = False) -> str:
    d = ""
    if delta:
        cls = "neg" if neg else ""
        d = f'<div class="metric-delta {cls}">{delta}</div>'
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {d}
    </div>"""


def INFO_BOX(text: str) -> str:
    return f'<div class="info-box">{text}</div>'


def WARN_BOX(text: str) -> str:
    return f'<div class="warn-box">{text}</div>'


def SEC_TITLE(text: str) -> str:
    return f'<div class="section-title">{text}</div>'

