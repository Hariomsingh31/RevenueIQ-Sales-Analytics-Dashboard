# 📊 RevenueIQ — ML-Powered Sales Analytics Platform

> An end-to-end data analytics + machine learning project: a retail sales pipeline that goes from raw CSV to cleaned data, engineered features, three trained ML models, and a live interactive dashboard — with revenue forecasting, customer/product segmentation, and anomaly detection built in.

[![Live Demo](https://img.shields.io/badge/demo-live-success?style=for-the-badge)](https://sales-analysis-dashboard-vdmh.onrender.com/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)

**[🔗 Live Demo](https://sales-analysis-dashboard-vdmh.onrender.com/)**

---

## 📌 Overview

RevenueIQ started as a BI dashboard and evolved into a full ML data analyst project. It now combines:

- A proper **data pipeline**: raw → validated → cleaned → feature-engineered
- **Three trained ML models**, each solving a distinct business question
- A **Jupyter notebook** documenting the full EDA-to-modeling workflow
- A **Streamlit app** that loads the trained models and serves live, interactive predictions
- **Unit tests** covering the data pipeline, feature engineering, and all three models

| Business Question | ML Technique | Where |
|---|---|---|
| *"What will revenue look like next month?"* | Gradient Boosting Regressor (lag + rolling + calendar features) | `src/models/forecast_model.py` |
| *"Which products/regions are our highest value?"* | K-Means clustering (RFM-style features, auto-K via silhouette score) | `src/models/segmentation_model.py` |
| *"Which orders look suspicious or unusual?"* | Isolation Forest (category-relative price/quantity behavior) | `src/models/anomaly_model.py` |

---

## 🗂️ Project Structure

```
revenueiq/
├── app/
│   ├── app.py                    # Streamlit app (BI dashboard + ML tabs)
│   └── utils.py                  # Shared UI/data helpers
├── data/
│   ├── raw/                      # Raw input CSVs (gitignored by default)
│   └── processed/                # Cleaned, ML-ready data
├── models/                       # Trained model artifacts (.joblib)
├── notebooks/
│   └── 01_eda_and_modeling.ipynb # Full EDA + model development walkthrough
├── reports/
│   └── figures/                  # Exported charts/figures
├── src/
│   ├── data/
│   │   ├── generate_sample_data.py  # Synthetic data generator (for demos/dev)
│   │   └── make_dataset.py          # Schema validation + cleaning pipeline
│   ├── features/
│   │   └── build_features.py        # Feature engineering (lags, rolling, RFM, z-scores)
│   ├── models/
│   │   ├── forecast_model.py        # SalesForecaster (Gradient Boosting)
│   │   ├── segmentation_model.py    # SegmentationModel (K-Means)
│   │   └── anomaly_model.py         # AnomalyDetector (Isolation Forest)
│   └── visualization/               # (reserved for shared plotting utilities)
├── tests/
│   ├── test_make_dataset.py
│   ├── test_build_features.py
│   └── test_models.py
├── requirements.txt               # Runtime dependencies (app + src)
├── requirements-dev.txt           # + notebook/testing dependencies
└── README.md
```

This structure separates **data**, **features**, **models**, **app**, and **tests** — the same layout used in production ML codebases — so each piece can be understood, tested, and reused independently.

---

## ✨ Features

### 📊 BI Dashboard
- KPI summary (revenue, units, avg order value, total orders)
- Time-series trends (Day/Week/Month granularity)
- Region-wise and category-wise breakdowns with interactive Plotly charts
- Top-product ranking with a live "Top N" slider
- Payment method breakdown
- Multi-dimension filtering (Category, Region, Payment Method, Date Range) synced across all tabs
- CSV export on every view

### 🤖 Machine Learning
- **🔮 Forecast** — Trains a Gradient Boosting Regressor on the filtered data on demand, shows out-of-sample cross-validation metrics (MAE, MAPE, R²), plots historical vs. forecasted revenue, and surfaces feature importances
- **🧩 Segmentation** — Clusters products or regions into value tiers (Low → Top Performer) using RFM-style features; automatically selects the optimal number of clusters via silhouette score, or lets you set it manually
- **🚨 Anomaly Detection** — Flags orders that look unusual relative to their category's typical price/quantity behavior, with an adjustable sensitivity slider and a flagged-orders export

All three models train **live, in-browser**, on whatever data is currently filtered — not just on a static pre-trained snapshot — so the ML adapts to the slice of data you're looking at.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- pip

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/revenueiq.git
cd revenueiq
pip install -r requirements.txt
```

For notebook development and running tests, install the dev extras instead:

```bash
pip install -r requirements-dev.txt
```

### 2. Generate sample data (optional)

If you don't have your own sales CSV yet, generate a realistic synthetic dataset:

```bash
python src/data/generate_sample_data.py --rows 6000 --out data/raw/sales_data.csv
```

### 3. Run the data pipeline

```bash
python src/data/make_dataset.py --input data/raw/sales_data.csv --output data/processed/sales_clean.csv
```

### 4. Train the models (optional — the app also trains on the fly)

```bash
python src/models/forecast_model.py --input data/processed/sales_clean.csv --model-out models/forecast_model.joblib
python src/models/segmentation_model.py --input data/processed/sales_clean.csv --level product --model-out models/segmentation_model.joblib
python src/models/anomaly_model.py --input data/processed/sales_clean.csv --model-out models/anomaly_model.joblib
```

### 5. Launch the app

```bash
streamlit run app/app.py
```

The app opens at `http://localhost:8501`. Either upload your own CSV or check **"Use bundled sample dataset"** in the sidebar to explore immediately.

### 6. Explore the notebook

```bash
jupyter notebook notebooks/01_eda_and_modeling.ipynb
```

Walks through EDA, feature engineering, and all three models with full output (charts, metrics, tables) already rendered.

### 7. Run the tests

```bash
pytest tests/ -v
```

27 tests covering schema validation, data cleaning, feature engineering, and all three ML models.

---

## 📊 Data Format

Your CSV should contain the following columns:

| Column | Type | Description |
|---|---|---|
| `Order ID` | string | Unique identifier for each sales order |
| `Date` | date (`YYYY-MM-DD`) | Date of the sales transaction |
| `Category` | string | Broad product category (e.g., Electronics, Home Appliances, Clothing, Books, Beauty Products, Sports) |
| `Product Name` | string | Specific name or model of the product sold |
| `Quantity` | integer | Number of units sold in the transaction |
| `Unit Price` | float | Price of one unit of the product |
| `Total Price` | float | Total revenue for the transaction (`Quantity × Unit Price`) |
| `Region` | string | Geographic region of the transaction (e.g., North America, Europe, Asia) |
| `Payment Method` | string | Method used for payment (e.g., Credit Card, PayPal, Debit Card) |

> 💡 If `Total Price` is missing, it's auto-computed from `Quantity × Unit Price`. If `Date` is unparseable, the app still works — only forecasting and trend charts are disabled.

> ⚠️ **"Missing required columns" error showing ALL columns as missing?** This almost always means a delimiter mismatch — some regional Excel exports use `;` instead of `,` as the separator. The app auto-detects and handles this, along with UTF-8 BOM markers and stray whitespace in headers, so it should resolve itself automatically. If it persists, open the file in a text editor and confirm the header row is comma-separated.

---

## 🧠 ML Methodology

### Forecasting
Revenue forecasting is framed as **supervised regression**, not classical time-series modeling. Daily (or weekly/monthly) revenue is engineered into lag features (1/7/14/30 periods back), rolling mean/std features (7/14/30-day windows), and calendar features (day-of-week, month, weekend flag). A Gradient Boosting Regressor is trained on this tabular representation. Multi-step forecasts are generated **recursively**: predict one step ahead, append it to history, recompute features, repeat. Performance is validated with `TimeSeriesSplit` to avoid look-ahead bias — no shuffling, no future data leaking into training folds.

### Segmentation
Products (or regions) are described with RFM-style features — Recency (days since last order), Frequency (orders per active day), and Monetary value (total/average revenue) — then standardized and clustered with K-Means. The number of clusters is selected automatically by maximizing silhouette score across a candidate range, rather than being hardcoded. Cluster labels are re-ranked by mean revenue so "Top Performer" always means the highest-value cluster, regardless of K-Means' arbitrary internal cluster numbering.

### Anomaly Detection
Each order is scored by an Isolation Forest using quantity, price, total value, and a **category-relative price z-score** (how unusual is this price *for this category specifically*, not globally). This catches both global outliers (huge bulk orders) and contextual outliers (a luxury-category item priced like a budget item, or vice versa) that a flat threshold would miss.

---

## 🛠️ Tech Stack

- **App / Hosting:** [Streamlit](https://streamlit.io/) on [Render](https://render.com/)
- **Data Manipulation:** [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Machine Learning:** [scikit-learn](https://scikit-learn.org/) — `GradientBoostingRegressor`, `KMeans`, `IsolationForest`
- **Visualization:** [Plotly Express](https://plotly.com/python/plotly-express/) (app), [Matplotlib](https://matplotlib.org/) / [Seaborn](https://seaborn.pydata.org/) (notebook)
- **Testing:** [pytest](https://pytest.org/)
- **Model Persistence:** [joblib](https://joblib.readthedocs.io/)

---

## 🔮 Future Enhancements

- [ ] Swap recursive forecasting for a proper multi-horizon model (e.g., direct multi-step or LightGBM with quantile loss for confidence intervals)
- [ ] Add a model registry / versioning step instead of overwriting `models/*.joblib`
- [ ] Customer-level segmentation if/when customer ID becomes available in the data
- [ ] CI pipeline (GitHub Actions) running `pytest` on every push
- [ ] Dockerfile for fully reproducible deployment
- [ ] Auto-generated PDF/PPT summary report export

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🙋 Author

**Hariom**
📍 Delhi, India

If you found this project useful, consider giving it a ⭐ on GitHub!
