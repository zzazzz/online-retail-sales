"""Business summary and recommendations page."""

import streamlit as st

from utils.charts import SEC_TITLE, kpi


def render(df_sales, compare, optuna_out, rfm, product_after, country):
    st.markdown("##  Final Business Interpretation (10)")

    st.markdown("""
    <div class="info-box">
    Versi ini adalah kompromi yang paling <b>defensible secara bisnis</b>:<br>
     Tidak memakai outlier filtering percentile<br>
     Tidak menghapus produk fisik hanya karena revenue/quantity besar<br>
     Hanya memisahkan line yang bukan produk dagangan (berdasarkan <code>product_id</code>)<br>
     Forecasting target = <b>daily product-sales revenue</b><br>
     RFM memakai product-sales revenue  tidak bias oleh postage/fee/adjustment
    </div>
    """, unsafe_allow_html=True)

    scores = optuna_out["scores"]
    best_tr = optuna_out["best_transform"]

    st.markdown(SEC_TITLE("Executive KPIs"), unsafe_allow_html=True)
    total_rev = df_sales["revenue"].sum()
    total_ref = abs(compare["canc_revenue_abs"].sum())
    top_country = country.iloc[0]
    champs = rfm[rfm["segment"] == "Champions"]
    top_prod = product_after.iloc[0]

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.markdown(kpi("Gross Revenue", f"{total_rev/1e6:.2f}M"), unsafe_allow_html=True)
    with c2:
        net_revenue = total_rev - total_ref
        st.markdown(kpi("Net Revenue", f"{net_revenue/1e6:.2f}M"), unsafe_allow_html=True)
    with c3:
        refund_rate = total_ref / total_rev * 100
        st.markdown(kpi("Refund Rate", f"{refund_rate:.1f}%", neg=True), unsafe_allow_html=True)
    with c4:
        champion_share = len(champs) / len(rfm) * 100
        st.markdown(kpi("Champions", f"{len(champs):,}", f"{champion_share:.1f}% base"), unsafe_allow_html=True)
    with c5:
        st.markdown(kpi("Forecast R2", f"{scores['R2 Test']:.4f}"), unsafe_allow_html=True)
    with c6:
        st.markdown(kpi("Best Transform", best_tr.upper()), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(SEC_TITLE("Modelling Decisions (Notebook 7-8)"), unsafe_allow_html=True)

    cards = [
        ("", "Product-Based Cleansing",
         "Tidak memakai percentile outlier filter. Hanya exclude non-product/admin codes "
         "(POST, DOT, M, D, AMAZONFEE, dll). Produk fisik dengan revenue besar seperti "
         "PAPER CRAFT, LITTLE BIRDIE tetap dipertahankan.",
         "#3b82f6"),
        ("", "Target Transformation",
         f"Optuna memilih <b>{best_tr.upper()}</b> sebagai transformasi terbaik. "
         "Lima pilihan diuji: raw, log1p, sqrt, cuberoot, fourthroot. "
         "Semua metrik dievaluasi di skala revenue asli setelah inverse transform.",
         "#22c55e"),
        ("", "Optuna ExtraTrees",
         f"50 trial TPE search  hyperparameter + target transform simultaneous. "
         f"Hasil: R2 = {scores['R2 Test']:.4f}, MAE Non-Zero = {scores['MAE Non-Zero']:,.0f}, "
         f"sMAPE = {scores['sMAPE']:.1f}%.",
         "#8b5cf6"),
        ("", "Why Lower R is Acceptable",
         "Jika R lebih rendah dibanding versi outlier-filtered, itu wajar karena spike produk fisik besar "
         "tetap dipertahankan. Pendekatan ini lebih mudah dipertanggungjawabkan dalam konteks bisnis "
         "karena tidak membuang data valid.",
         "#f59e0b"),
        ("", "RFM Segmentation",
         "Customer segmentation memakai product-sales revenue  tidak bias oleh postage, "
         "manual adjustment, dan fee. Scoring 14 dengan qcut ranking untuk menghindari "
         "duplicate bin issues pada data imbalanced.",
         "#06b6d4"),
        ("", "Forecast Safety",
         "Semua fitur bersifat forecast-safe: kalender, seasonality, dan pola hari tutup. "
         "Tidak ada same-day leakage dari orders/customers/qty karena variabel tersebut "
         "baru diketahui setelah hari berjalan.",
         "#f472b6"),
    ]
    row1 = st.columns(3)
    row2 = st.columns(3)
    for col, (icon, title, desc, color) in zip(list(row1) + list(row2), cards):
        col.markdown(f"""
        <div class="metric-card" style="border-color:{color};text-align:left;min-height:160px">
            <div style="font-size:1.8rem">{icon}</div>
            <div style="font-weight:700;color:#f8fafc;margin:0.4rem 0 0.3rem">{title}</div>
            <div style="font-size:0.78rem;color:#a7b3c7;line-height:1.6">{desc}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(SEC_TITLE("Strategic Recommendations"), unsafe_allow_html=True)

    recs = [
        ("", "Diversifikasi Pasar",
         f"{top_country['country']} mendominasi {top_country['revenue_pct']:.0f}% revenue  risiko konsentrasi tinggi. "
         "Target ekspansi ke EIRE, Germany, France, Netherlands yang sudah memiliki customer base kecil."),
        ("", "Inventory Prioritisation",
         f"Top produk ({top_prod['product_description'][:40]}) menghasilkan {top_prod['revenue']:,.0f}  "
         "pastikan stok tidak pernah habis. Terapkan ABC inventory management."),
        ("", "Customer Retention",
         f"{len(rfm[rfm['segment'] == 'At Risk']):,} pelanggan At Risk yang sudah pernah belanja banyak. "
         "Win-back campaign dengan personalised discount sebelum mereka masuk 'Lost'."),
        ("", "Forecast-Driven Planning",
         f"Model Optuna ExtraTrees R2={scores['R2 Test']:.4f} cukup untuk demand planning. "
         "Gunakan forecast monthly sebagai input budget procurement Q1 2012."),
        ("", "Q4 Readiness",
         "Seasonality Q4 sangat kuat setiap tahun. Mulai persiapan stok, campaign, dan logistik "
         "dari AgustusSeptember untuk memaksimalkan OctoberNovember peak."),
        ("", "Refund Reduction",
         f"Refund rate avg {compare['refund_rate'].mean():.1f}%  target < 5%. "
         "Review product description accuracy, packaging, dan customer expectation management."),
    ]
    for icon, title, desc in recs:
        st.markdown(f"""
        <div style="display:flex;gap:1rem;padding:0.8rem;border-bottom:1px solid #243149;align-items:flex-start">
            <span style="font-size:1.5rem;flex-shrink:0">{icon}</span>
            <div>
                <div style="font-weight:600;color:#f8fafc;margin-bottom:0.2rem">{title}</div>
                <div style="font-size:0.82rem;color:#a7b3c7;line-height:1.6">{desc}</div>
            </div>
        </div>""", unsafe_allow_html=True)

