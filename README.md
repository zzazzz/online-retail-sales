# Retail Analytics Dashboard
### Online Retail II — Product Revenue Forecasting, Optuna ExtraTrees & RFM Segmentation

Dashboard Streamlit end-to-end berbasis notebook **Retail Analytics HM Sampoerna** yang mencakup product-based data cleansing, EDA, model selection, Optuna hyperparameter tuning, daily revenue forecasting, refund monitoring, dan RFM customer segmentation.

---

## Daftar Isi

- [Gambaran Umum](#gambaran-umum)
- [Fitur Dashboard](#fitur-dashboard)
- [Struktur Proyek](#struktur-proyek)
- [Persyaratan](#persyaratan)
- [Instalasi & Menjalankan](#instalasi--menjalankan)
- [Format Data](#format-data)
- [Alur Pipeline](#alur-pipeline)
- [Halaman Dashboard](#halaman-dashboard)
- [Konfigurasi Model](#konfigurasi-model)
- [Caching](#caching)
- [Referensi Notebook](#referensi-notebook)

---

## Gambaran Umum

Dashboard ini mengonversi notebook analitik retail menjadi aplikasi interaktif berbasis **Streamlit** dengan dark-mode UI. Seluruh logika mengikuti notebook referensi secara ketat:

- **Tidak ada percentile outlier filtering** — transaksi produk fisik bernilai besar tetap dipertahankan.
- **Product-based cleansing** — hanya baris non-produk/admin (POSTAGE, MANUAL, FEE, dll.) yang dipisahkan.
- **Target forecasting** = daily product-sales revenue (bukan qty, bukan total termasuk admin lines).
- **RFM** dihitung dari product-sales revenue saja, bebas dari bias postage/fee/adjustment.

---

## Fitur Dashboard

| Halaman | Deskripsi |
|---|---|
| **Overview** | Executive KPI, monthly revenue + 3M MA, RFM donut, forecast tail, top countries |
| **Data Overview** | Raw dataset summary, product vs non-product split, cleaning result, top 20 SKU |
| **Product EDA** | Top N by revenue/qty, ABC Pareto, choropleth map, revenue heatmap, day-of-week pattern |
| **Model Selection** | Perbandingan 6 tree models (ExtraTrees, RF, XGBoost, LightGBM, CatBoost, HistGB) |
| **Optuna Tuning** | 50–120 trial TPE, target transform selection, feature importance, convergence plot |
| **Revenue Forecast** | Daily forecast + 95% CI, monthly summary, forced-zero hari tutup |
| **RFM Segmentation** | 8 segment RFM, scatter F×M dan R×M, top customer table |
| **Refund Analysis** | Gross vs refund overtime, refund rate trend, cancelled order count |
| **Business Insights** | Executive KPI, modelling decisions, 6 strategic recommendations |

---

## Struktur Proyek

```
project/
├── app.py                        # Entry point Streamlit
├── retail_transaction_data.csv   # Dataset utama (tidak di-commit)
├── requirements.txt
│
├── pages/
│   ├── p01_data_overview.py
│   ├── p02_product_eda.py
│   ├── p03_model_selection.py
│   ├── p04_optuna.py
│   ├── p05_forecast.py
│   ├── p06_rfm.py
│   ├── p07_refund.py
│   └── p08_insights.py
│
└── utils/
    ├── charts.py      # dark_fig(), kpi(), INFO_BOX(), WARN_BOX(), SEC_TITLE()
    ├── constants.py   # NON_PRODUCT_CODES, UK_HOLIDAYS, SEG_COLORS, FEATURE_COLS
    ├── data.py        # load_raw(), clean_retail_data_product_revenue(), build_daily_series()
    └── models.py      # tree_model_selection(), optuna_tune_extratrees(), build_forecast(), compute_rfm()
```

---

## Persyaratan

Python **3.10+** direkomendasikan.

```
streamlit>=1.35.0
plotly>=5.22.0
pandas>=2.0.0
numpy>=1.26.0
scikit-learn>=1.4.0
xgboost>=2.0.0
lightgbm>=4.0.0
catboost
optuna>=3.5.0
openpyxl>=3.1.0
```

> **Catatan:** `catboost` tidak tercantum di `requirements.txt` bawaan tapi diimpor oleh `models.py`. Tambahkan secara manual atau install terpisah.

---

## Instalasi & Menjalankan

```bash
# 1. Clone / salin proyek
git clone <repo-url>
cd retail-analytics-dashboard

# 2. Buat virtual environment (opsional tapi disarankan)
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
pip install catboost              # jika belum ada

# 4. Taruh dataset di root folder
cp /path/to/data.csv retail_transaction_data.csv

# 5. Jalankan dashboard
streamlit run app.py
```

Dashboard akan terbuka di `http://localhost:8501`.

Alternatifnya, upload CSV langsung lewat **sidebar → Dataset → Upload CSV alternatif** tanpa perlu menaruh file di folder proyek.

---

## Format Data

File CSV harus memiliki kolom berikut (nama kolom sudah sesuai Online Retail II UCI dataset):

| Kolom | Tipe | Keterangan |
|---|---|---|
| `order_id` | string | ID invoice; prefix `C` = cancellation |
| `product_id` | string | Stock code / SKU |
| `product_description` | string | Nama produk |
| `quantity` | int | Jumlah unit (negatif untuk cancellation) |
| `order_date` | datetime | Tanggal & waktu transaksi |
| `unit_price` | float | Harga per unit (GBP) |
| `customer_id` | string | ID pelanggan (boleh kosong) |
| `country` | string | Negara pelanggan |

Dataset referensi: [UCI Online Retail II](https://archive.ics.uci.edu/ml/datasets/Online+Retail+II) — periode **Desember 2009 – Desember 2011**.

---

## Alur Pipeline

```
CSV Upload
    │
    ▼
load_raw()                   ← parsing datetime, string casting
    │
    ▼
clean_retail_data_product_revenue()
    │  - Pisahkan cancellations (prefix C)
    │  - Filter quantity > 0 & unit_price > 0
    │  - Exclude NON_PRODUCT_CODES (POST, DOT, M, D, AMAZONFEE, dll.)
    │
    ▼
compute_aggregations()       ← monthly, compare (refund), product_after, country, cust
    │
    ▼
build_daily_series()         ← daily revenue + forced-zero flag + calendar features
    │
    ├──► tree_model_selection()     ← 6 model comparison, 80/20 split
    │
    ├──► optuna_tune_extratrees()   ← TPE search, target transform, best params
    │
    ├──► build_forecast()           ← recursive daily forecast + 95% CI
    │
    └──► compute_rfm()              ← RFM scoring qcut, 8 segment label
```

---

## Halaman Dashboard

### 1 — Overview
Halaman utama eksekutif. Menampilkan 6 headline KPI (Gross Revenue, Net Revenue, Unique SKUs, Customers, Model R², Best Transform), monthly revenue chart, RFM donut, forecast tail, dan top-10 countries.

### 2 — Data Overview & Cleaning
Menjelaskan keputusan cleansing: raw row count, date range, split produk vs non-produk, dan hasil setelah cleaning. Terdapat chart revenue share (pie) dan monthly revenue + orders (dual subplot).

### 3 — Product EDA
Tab berisi:
- **Top Products** — slider top N, bar revenue & qty, ABC Classification card, catalogue searchable
- **Revenue vs Volume** — bubble chart (warna = avg price)
- **Geographic** — choropleth + bar top countries (toggle exclude UK)
- **Seasonality** — heatmap year × month, avg revenue by day of week

### 4 — Model Selection
Benchmark 6 tree-based model (ExtraTrees, RandomForest, XGBoost, LightGBM, CatBoost, HistGradientBoosting) pada train/test 80/20. Tampil bar R² Test, tabel metrik lengkap, plot actual vs predicted + residuals.

### 5 — Optuna Tuning
ExtraTrees di-tuning Optuna TPE secara simultan untuk hyperparameter **dan** target transform (raw / log1p / sqrt / cuberoot / fourthroot). Tab: actual vs predicted, feature importance, convergence plot, best params, transform comparison.

### 6 — Revenue Forecast
Forecast harian hingga `horizon` hari ke depan (default 180 hari). Hari Sabtu, UK bank holidays 2012, dan observed store closures di-force ke nol. 95% CI = 1.96 × residual std test set.

### 7 — RFM Segmentation
Skor R, F, M skala 1–4 dengan `pd.qcut`. Delapan segmen: Champions, Loyal Customers, Potential Loyalists, Promising, New Customers, At Risk, Lost Customers, Others. Tab: segment bar + donut, scatter F×M dan R×M, top customer table.

### 8 — Refund Analysis
Triple subplot: Gross Revenue vs Refund, Refund Rate %, Cancelled Order Count — semua per bulan.

### 9 — Business Insights
Executive KPI, 6 kartu modelling decisions, dan 6 strategic recommendations berbasis output model dan segmentasi.

---

## Konfigurasi Model

Semua parameter model dapat disesuaikan lewat sidebar:

| Pengaturan | Default | Keterangan |
|---|---|---|
| **Optuna Trials** | 120 | Jumlah trial TPE search. Turunkan ke 10–30 untuk preview cepat |
| **Forecast Horizon** | 180 hari | Rentang forecast ke depan (30–365 hari) |

Fitur yang digunakan model (semua forecast-safe, tidak ada same-day leakage):

```
is_forced_zero, dow, dow_sin, is_q4, dayofyear, doy_sin, doy_cos,
week, week_sin, week_cos, month, time_idx,
days_since_last_closed, days_to_next_closed,
is_monday, is_tuesday, is_wednesday, is_thursday, is_friday, is_sunday
```

---

## Caching

Dashboard menggunakan dua layer caching:

- **`@st.cache_data(persist="disk")`** — data loading, cleaning, dan agregasi di-cache ke disk Streamlit agar reload halaman tetap cepat.
- **Pickle cache (`/.dashboard_cache/`)** — hasil model selection dan Optuna di-cache berdasarkan SHA-256 hash dari dataset + versi cache (`v3`). Model tidak dilatih ulang selama data tidak berubah.

Untuk memaksa retraining, hapus folder `.dashboard_cache/` atau ubah `CACHE_VERSION` di `models.py`.

---

## Referensi Notebook

Dashboard ini dibangun mengikuti alur analitik dari notebook:

> **Retail Analytics HM Sampoerna — ProductRevenue Detailed Optuna RFM**

Pendekatan utama yang dipertahankan dari notebook:
- Tidak memakai percentile outlier filtering pada produk fisik
- Target transformasi dipilih oleh Optuna (bukan ditentukan manual)
- RFM scoring menggunakan `qcut` untuk menghindari duplicate bin pada distribusi imbalanced
- Semua metrik dievaluasi di skala revenue asli setelah inverse transform

---

*Built with Streamlit · Plotly · scikit-learn · Optuna · LightGBM · XGBoost · CatBoost*
